//! Decoder forward pass, prompt construction, and the top-level `decode` entry point.

use super::cross_attn_capture::{CrossAttnCapture, is_alignment_layer};
use super::kv_cache::LayerKVCache;
use super::sampler::{decode_greedy, decode_sample};
use super::sdpa::{
    CachedSdpaConfig, SdpaScratch, scaled_dot_product_cached, scaled_dot_product_flat,
};

use crate::beam_search::decode_beam;
use crate::decode_utils::{DecodeArgs, DecodeConstraints};
use crate::hallucination::is_likely_hallucination;
use crate::linear;
use crate::model::ModelData;
use crate::tensor::Tensor;
use crate::tokenizer::{self, SpecialTokens};
use crate::types::Task;

/// Maximum number of tokens the decoder will generate per segment.
pub(crate) const MAX_DECODE_LENGTH: usize = 224;

/// Token id of the first language token (Whisper multilingual vocab).
const LANG_TOKEN_START: u32 = 50259;
/// One-past the last language token (99 languages total).
const LANG_TOKEN_END: u32 = 50358;

/// The decoder's tied token-embedding table.
///
/// Quantized checkpoints keep this matrix in its native block layout (it is by
/// far the largest tensor in the model — 20 M elements even for `tiny`), so the
/// embedding lookup has to be able to gather a row from either representation.
/// Requiring an f32 tensor here made every quantized whisper.cpp checkpoint
/// fail with `Missing tensor: decoder.token_embedding.weight`.
pub(crate) enum TokenEmbedding<'a> {
    /// Plain f32 table, shape `[n_state, n_vocab]` in GGML order.
    Float(&'a Tensor),
    /// Block-quantized table, shape `[n_state, n_vocab]` in GGML order.
    Quantized(&'a crate::quantize::QuantizedTensor),
}

impl TokenEmbedding<'_> {
    /// Number of vocabulary rows in the table.
    fn n_vocab(&self) -> usize {
        match self {
            Self::Float(t) => t.shape.get(1).copied().unwrap_or(0),
            Self::Quantized(q) => q.shape.get(1).copied().unwrap_or(0),
        }
    }

    /// Copy the embedding row of token `idx` into `out` (length `n_state`).
    ///
    /// Does nothing when `idx` is out of range, leaving `out` untouched.
    fn write_row(&self, idx: usize, out: &mut [f32]) {
        if idx >= self.n_vocab() {
            return;
        }
        match self {
            Self::Float(t) => {
                let n_state = out.len();
                let start = idx * n_state;
                if let Some(row) = t.data.get(start..start + n_state) {
                    out.copy_from_slice(row);
                }
            }
            Self::Quantized(q) => {
                let row = q.row_bytes(idx);
                let deq = crate::quantize::dequantize(row, out.len(), q.qtype);
                out.copy_from_slice(&deq);
            }
        }
    }
}

/// Read-only context passed to every `forward()` call.
/// Groups all model/shape parameters so call-sites stay readable.
pub(crate) struct ForwardCtx<'a> {
    /// Cross-attention key projections for each decoder layer (shape `[n_head, enc_len, head_dim]`).
    pub(crate) cross_k: &'a [Vec<f32>],
    /// Cross-attention value projections for each decoder layer (shape `[n_head, enc_len, head_dim]`).
    pub(crate) cross_v: &'a [Vec<f32>],
    /// Encoder output sequence length.
    pub(crate) enc_len: usize,
    /// Token embedding weight matrix (f32 or block-quantized).
    pub(crate) tok_emb: TokenEmbedding<'a>,
    /// Positional embedding matrix.
    pub(crate) pos_emb: &'a Tensor,
    /// Reference to all model weights.
    pub(crate) model: &'a ModelData,
    /// Total hidden state dimension (`n_head * head_dim`).
    pub(crate) n_state: usize,
    /// Number of decoder transformer layers.
    pub(crate) n_layer: usize,
    /// Number of attention heads.
    pub(crate) n_head: usize,
    /// Dimension of each attention head.
    pub(crate) head_dim: usize,
}

/// Result from the decoder containing token IDs and optionally the detected language.
pub struct DecodeResult {
    /// Decoded output token IDs (excluding prompt tokens).
    pub tokens: Vec<u32>,
    /// Per-token log-probability (same length as `tokens`).
    pub token_probs: Vec<f32>,
    /// Detected language code (e.g. `"en"`, `"ja"`), or `None` if language was
    /// explicitly specified via options.
    pub detected_language: Option<String>,
    /// Head- and layer-averaged cross-attention matrix, flat `[tokens.len() * enc_len]`
    /// row-major.  `None` unless
    /// [`TranscribeOptions::word_timestamps`](crate::TranscribeOptions::word_timestamps)
    /// was enabled and `beam_width == 1`.
    pub cross_attention: Option<Vec<f32>>,
    /// Encoder frame count (== `n_frames` for DTW).  `0` when `cross_attention` is `None`.
    pub enc_len: usize,
    /// Probability of the `<|nospeech|>` token at the first decoded position
    /// (softmax over the prefill logits, before any suppression).  In `[0, 1]`.
    pub no_speech_prob: f32,
}

/// Run the Whisper text decoder with KV cache for efficient autoregressive decoding.
///
/// Returns a `DecodeResult` containing the output tokens and optionally the
/// auto-detected language.
pub fn decode(
    encoder_output: &Tensor,
    model: &ModelData,
    opts: &crate::TranscribeOptions<'_>,
) -> Result<DecodeResult, String> {
    let hp = &model.hparams;
    let n_state = hp.n_text_state;
    let n_layer = hp.n_text_layer;
    let n_head = hp.n_text_head;
    let head_dim = n_state / n_head;
    let enc_len = encoder_output.shape[0];

    let special = SpecialTokens::new(hp.n_vocab);

    // -- Precompute cross-attention K,V for every layer (done once) --
    let mut cross_k: Vec<Vec<f32>> = Vec::with_capacity(n_layer);
    let mut cross_v: Vec<Vec<f32>> = Vec::with_capacity(n_layer);
    for layer in 0..n_layer {
        let pfx = format!("decoder.blocks.{layer}");
        let ck_name = format!("{pfx}.cross_attn.key.weight");
        let ck = linear::linear_auto(
            encoder_output,
            model.try_get(&ck_name),
            model.get_quantized(&ck_name),
            None,
        )?;
        let cv_name = format!("{pfx}.cross_attn.value.weight");
        let cv = linear::linear_auto(
            encoder_output,
            model.try_get(&cv_name),
            model.get_quantized(&cv_name),
            Some(model.get(&format!("{pfx}.cross_attn.value.bias"))?),
        )?;
        cross_k.push(to_head_first(&ck.data, n_head, enc_len, head_dim));
        cross_v.push(to_head_first(&cv.data, n_head, enc_len, head_dim));
    }

    let tok_emb_name = "decoder.token_embedding.weight";
    let tok_emb = match model.try_get(tok_emb_name) {
        Some(t) => TokenEmbedding::Float(t),
        None => TokenEmbedding::Quantized(
            model
                .get_quantized(tok_emb_name)
                .ok_or_else(|| format!("Missing tensor: {tok_emb_name}"))?,
        ),
    };

    let ctx = ForwardCtx {
        cross_k: &cross_k,
        cross_v: &cross_v,
        enc_len,
        tok_emb,
        pos_emb: model.get("decoder.positional_embedding")?,
        model,
        n_state,
        n_layer,
        n_head,
        head_dim,
    };

    // Detect language automatically if none is specified.
    let (lang_token, detected_language) = match opts.language {
        Some(lang) => (special.language_token(lang), None),
        None => {
            let token = detect_language(&ctx, &special)?;
            let lang_code = language_code_from_token(token);
            (token, Some(lang_code))
        }
    };

    let mut all_initial = match opts.initial_prompt {
        Some(text) => encode_prompt_text(text, &model.vocab),
        None => Vec::new(),
    };
    if let Some(prev) = opts.previous_tokens {
        // Prepend previous tokens before any initial prompt tokens
        let mut combined = prev.to_vec();
        combined.append(&mut all_initial);
        all_initial = combined;
    }
    let task_token = match opts.task {
        Task::Transcribe => special.transcribe,
        Task::Translate => special.translate,
    };
    let prompt = build_prompt(
        &special,
        lang_token,
        task_token,
        opts.timestamps,
        &all_initial,
    );
    let kv_capacity = prompt.len() + MAX_DECODE_LENGTH + 4;

    // Blank token for suppress_blank: greedy longest-match of " " against vocab.
    let blank_token: Option<u32> = encode_prompt_text(" ", &model.vocab).first().copied();

    let constraints = DecodeConstraints {
        suppress: opts.suppress_tokens.unwrap_or(&[]),
        no_repeat_ngram_size: opts.no_repeat_ngram_size,
        timestamp_rules: opts.timestamps,
        suppress_blank: opts.suppress_blank,
        blank_token,
    };
    let args = DecodeArgs {
        kv_capacity,
        special: &special,
        eot_threshold: special.eot,
        constraints: &constraints,
        dtype: opts.kv_cache_dtype,
    };

    // Whether to capture cross-attention for word timestamps (greedy/sample only).
    let capturing = opts.word_timestamps && opts.beam_width <= 1;

    // Accumulates attention rows from the accepted decode attempt.
    // One row per accepted output token, each of length `enc_len`.
    let mut last_attn: Vec<f32> = if capturing {
        Vec::with_capacity(MAX_DECODE_LENGTH * enc_len)
    } else {
        Vec::new()
    };

    // Dispatch closure: selects the sampler by temperature; t > 0 uses sample,
    // t == 0 uses beam or greedy. The strict > guard is load-bearing — decode_sample
    // divides by temperature and must never be called with t == 0.
    // Returns (tokens, token_probs, no_speech_prob).
    // Also clears and refills `last_attn` on each call (for retry correctness).
    let (tokens, token_probs, no_speech_prob) = {
        let mut decode_at = |t: f32| -> Result<(Vec<u32>, Vec<f32>, f32), String> {
            last_attn.clear();
            let attn_opt: Option<&mut Vec<f32>> = if capturing {
                Some(&mut last_attn)
            } else {
                None
            };
            if t > 0.0 {
                let mut o = opts.clone();
                o.temperature = t;
                decode_sample(&prompt, &ctx, &args, &o, attn_opt)
            } else if opts.beam_width > 1 {
                decode_beam(&prompt, &ctx, &args, opts.beam_width)
            } else {
                decode_greedy(&prompt, &ctx, &args, attn_opt)
            }
        };

        if opts.fallback_temperatures.is_empty() {
            // Default path: single dispatch, identical to prior behaviour.
            decode_at(opts.temperature)?
        } else {
            // Temperature fallback: try each temperature, accept the first clean result.
            let mut last: Option<(Vec<u32>, Vec<f32>, f32)> = None;
            let mut accepted = false;

            for &t in opts.fallback_temperatures {
                let (toks, probs, nsp) = decode_at(t)?;

                // Silence / no-speech: accept immediately, do not retry. Retrying would
                // risk hallucinating tokens into a genuinely silent segment.
                if toks.is_empty() {
                    last = Some((toks, probs, nsp));
                    accepted = true;
                    break;
                }

                let text = tokenizer::decode(&toks, &model.vocab);
                let avg_logprob = if probs.is_empty() {
                    0.0f32
                } else {
                    probs.iter().sum::<f32>() / probs.len() as f32
                };
                let needs_fallback =
                    is_likely_hallucination(&text, opts.compression_ratio_threshold)
                        || (!probs.is_empty() && avg_logprob < opts.logprob_threshold);

                last = Some((toks, probs, nsp));
                if !needs_fallback {
                    accepted = true;
                    break;
                }
            }

            // If no attempt passed, return the last attempt rather than nothing.
            let _ = accepted; // documented: last attempt is the fallback
            last.unwrap_or_default()
        }
    };
    // `decode_at` closure dropped here — `last_attn` reborrow released.

    // Combined OpenAI silence gate: no_speech_prob > threshold AND avg_logprob < threshold.
    // Applied to the finalised attempt (not per-retry).
    let avg_logprob = if token_probs.is_empty() {
        0.0f32
    } else {
        token_probs.iter().sum::<f32>() / token_probs.len() as f32
    };
    let is_silence = !tokens.is_empty()
        && no_speech_prob > opts.no_speech_threshold
        && avg_logprob < opts.logprob_threshold;
    let (tokens, token_probs) = if is_silence {
        (Vec::new(), Vec::new())
    } else {
        (tokens, token_probs)
    };

    let cross_attention = if capturing && !tokens.is_empty() {
        Some(std::mem::take(&mut last_attn))
    } else {
        None
    };
    let result_enc_len = if cross_attention.is_some() {
        enc_len
    } else {
        0
    };

    Ok(DecodeResult {
        tokens,
        token_probs,
        detected_language,
        cross_attention,
        enc_len: result_enc_len,
        no_speech_prob,
    })
}

/// Map a language token ID back to a BCP-47 language code string.
fn language_code_from_token(token: u32) -> String {
    let languages = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
        "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
        "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
        "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
    ];
    let idx = token.saturating_sub(LANG_TOKEN_START) as usize;
    if idx < languages.len() {
        languages[idx].to_string()
    } else {
        "en".to_string()
    }
}

/// Auto-detect language by running a single forward pass on [SOT] and taking
/// the argmax over the 99 language token logits (50259..50358).
/// Always uses F32 storage — this is a single one-shot pass where memory savings don't matter.
fn detect_language(ctx: &ForwardCtx<'_>, special: &SpecialTokens) -> Result<u32, String> {
    let cap = MAX_DECODE_LENGTH + 10;
    let mut kv: Vec<LayerKVCache> = (0..ctx.n_layer)
        .map(|_| LayerKVCache::new(ctx.n_head, ctx.head_dim, cap))
        .collect();
    let logits = forward(&[special.sot], 0, &mut kv, ctx, true, None)?;
    let end = (LANG_TOKEN_END as usize).min(logits.len());
    let start = (LANG_TOKEN_START as usize).min(end);
    if start >= end {
        return Ok(special.language_token("en"));
    }
    Ok(logits[start..end]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| LANG_TOKEN_START + i as u32)
        .unwrap_or(special.language_token("en")))
}

/// Encode text into token IDs by greedy longest-match against the model vocabulary.
/// This is a simple tokenizer suitable for prompt conditioning.
pub(crate) fn encode_prompt_text(text: &str, vocab: &[crate::model::VocabEntry]) -> Vec<u32> {
    let mut tokens = Vec::new();
    let bytes = text.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        let mut best_len = 0;
        let mut best_tok = 0u32;
        for (i, entry) in vocab.iter().enumerate() {
            let entry_bytes = entry.as_bytes();
            if entry_bytes.len() > best_len && bytes[pos..].starts_with(entry_bytes) {
                best_len = entry_bytes.len();
                best_tok = i as u32;
            }
        }
        if best_len == 0 {
            pos += 1; // skip unrecognizable byte
        } else {
            tokens.push(best_tok);
            pos += best_len;
        }
    }
    tokens
}

/// Build the decoder prompt token sequence.
///
/// Prepends `sot_prev + initial_tokens` when any initial tokens are provided,
/// then appends `sot`, the language token, `transcribe`, and optionally
/// `no_timestamps`.
pub(crate) fn build_prompt(
    special: &SpecialTokens,
    lang_token: u32,
    task_token: u32,
    timestamps: bool,
    initial_tokens: &[u32],
) -> Vec<u32> {
    let mut prompt = Vec::new();
    // Initial prompt tokens go BEFORE the SOT token (matching OpenAI's behavior)
    if !initial_tokens.is_empty() {
        prompt.push(special.sot_prev);
        prompt.extend_from_slice(initial_tokens);
    }
    prompt.push(special.sot);
    prompt.push(lang_token);
    prompt.push(task_token);
    if !timestamps {
        prompt.push(special.no_timestamps);
    }
    prompt
}

/// One decoder forward pass -- updates self_kv in place, returns logits for the last token.
///
/// `capture` is an optional cross-attention accumulator.  When `Some`, this
/// function writes the head- and layer-averaged cross-attention matrix rows for
/// each selected decoder layer (see `is_alignment_layer`) via
/// `CrossAttnCapture::layer_sink` + `commit_layer`.  Callers retrieve the
/// result with `capture.finish()` after this function returns.
/// Pass `None` for all paths that do not need word timestamps — zero overhead.
pub(crate) fn forward(
    tokens: &[u32],
    start_pos: usize,
    self_kv: &mut [LayerKVCache],
    ctx: &ForwardCtx<'_>,
    causal_mask: bool,
    mut capture: Option<&mut CrossAttnCapture>,
) -> Result<Vec<f32>, String> {
    let q_len = tokens.len();
    let n_state = ctx.n_state;
    let n_head = ctx.n_head;
    let head_dim = ctx.head_dim;

    // Token + positional embedding.
    let mut x_data = vec![0.0f32; q_len * n_state];
    for (i, &tok) in tokens.iter().enumerate() {
        let idx = tok as usize;
        // GGML tok_emb shape: [n_state, n_vocab]; out-of-range ids are skipped.
        ctx.tok_emb
            .write_row(idx, &mut x_data[i * n_state..(i + 1) * n_state]);
        let pos = start_pos + i;
        // GGML pos_emb shape: [n_state, n_text_ctx] -- check against shape[1] (n_text_ctx).
        if pos < ctx.pos_emb.shape[1] {
            let pos_off = pos * n_state;
            for j in 0..n_state {
                x_data[i * n_state + j] += ctx.pos_emb.data[pos_off + j];
            }
        }
    }
    let mut x = Tensor::from_vec(x_data, &[q_len, n_state]);

    // Scratch buffers for SDPA — allocated once, grown as needed.
    // Includes score matrix and optional K/V dequantization buffers for f16 paths.
    let mut sdpa_scratch = SdpaScratch::new();

    for (layer, layer_kv) in self_kv.iter_mut().enumerate().take(ctx.n_layer) {
        let pfx = format!("decoder.blocks.{layer}");
        let md = ctx.model;

        // -- Self-attention --
        let normed = x.layer_norm(
            md.get(&format!("{pfx}.attn_ln.weight"))?,
            md.get(&format!("{pfx}.attn_ln.bias"))?,
            1e-5,
        );
        let qw_name = format!("{pfx}.attn.query.weight");
        let q = linear::linear_auto(
            &normed,
            md.try_get(&qw_name),
            md.get_quantized(&qw_name),
            Some(md.get(&format!("{pfx}.attn.query.bias"))?),
        )?;
        let kw_name = format!("{pfx}.attn.key.weight");
        let new_k = linear::linear_auto(
            &normed,
            md.try_get(&kw_name),
            md.get_quantized(&kw_name),
            None,
        )?;
        let vw_name = format!("{pfx}.attn.value.weight");
        let new_v = linear::linear_auto(
            &normed,
            md.try_get(&vw_name),
            md.get_quantized(&vw_name),
            Some(md.get(&format!("{pfx}.attn.value.bias"))?),
        )?;

        layer_kv.append(&new_k.data, &new_v.data, q_len);
        let kv_len = layer_kv.seq_len;
        let past_len = kv_len - q_len;

        let q_hf = to_head_first(&q.data, n_head, q_len, head_dim);
        let sdpa_cfg = CachedSdpaConfig {
            n_head,
            q_len,
            kv_len,
            head_dim,
            past_len,
            causal_mask,
        };
        let attn_self = scaled_dot_product_cached(&q_hf, layer_kv, &sdpa_cfg, &mut sdpa_scratch);
        let attn_self = Tensor::from_vec(attn_self, &[q_len, n_state]);
        let attn_out_name = format!("{pfx}.attn.out.weight");
        let out = linear::linear_auto(
            &attn_self,
            md.try_get(&attn_out_name),
            md.get_quantized(&attn_out_name),
            Some(md.get(&format!("{pfx}.attn.out.bias"))?),
        )?;
        x.add_inplace(&out);

        // -- Cross-attention --
        let normed = x.layer_norm(
            md.get(&format!("{pfx}.cross_attn_ln.weight"))?,
            md.get(&format!("{pfx}.cross_attn_ln.bias"))?,
            1e-5,
        );
        let cross_qw_name = format!("{pfx}.cross_attn.query.weight");
        let q = linear::linear_auto(
            &normed,
            md.try_get(&cross_qw_name),
            md.get_quantized(&cross_qw_name),
            Some(md.get(&format!("{pfx}.cross_attn.query.bias"))?),
        )?;
        let q_hf = to_head_first(&q.data, n_head, q_len, head_dim);
        // Conditionally capture cross-attention for upper-half alignment layers.
        // The borrow of `c.layer_scratch` ends when it's moved into `scaled_dot_product_flat`
        // and the call returns, so `commit_layer` can re-borrow `*c` immediately after (NLL).
        let attn_cross_data = if is_alignment_layer(layer, ctx.n_layer) {
            if let Some(c) = capture.as_mut() {
                let scratch = c.layer_sink();
                let data = scaled_dot_product_flat(
                    &q_hf,
                    &ctx.cross_k[layer],
                    &ctx.cross_v[layer],
                    n_head,
                    q_len,
                    ctx.enc_len,
                    head_dim,
                    Some(scratch),
                );
                c.commit_layer();
                data
            } else {
                scaled_dot_product_flat(
                    &q_hf,
                    &ctx.cross_k[layer],
                    &ctx.cross_v[layer],
                    n_head,
                    q_len,
                    ctx.enc_len,
                    head_dim,
                    None,
                )
            }
        } else {
            scaled_dot_product_flat(
                &q_hf,
                &ctx.cross_k[layer],
                &ctx.cross_v[layer],
                n_head,
                q_len,
                ctx.enc_len,
                head_dim,
                None,
            )
        };
        let attn_cross = Tensor::from_vec(attn_cross_data, &[q_len, n_state]);
        let cross_out_name = format!("{pfx}.cross_attn.out.weight");
        let out = linear::linear_auto(
            &attn_cross,
            md.try_get(&cross_out_name),
            md.get_quantized(&cross_out_name),
            Some(md.get(&format!("{pfx}.cross_attn.out.bias"))?),
        )?;
        x.add_inplace(&out);

        // -- Feed-forward --
        let normed = x.layer_norm(
            md.get(&format!("{pfx}.mlp_ln.weight"))?,
            md.get(&format!("{pfx}.mlp_ln.bias"))?,
            1e-5,
        );
        let mlp0_name = format!("{pfx}.mlp.0.weight");
        let mut h = linear::linear_auto(
            &normed,
            md.try_get(&mlp0_name),
            md.get_quantized(&mlp0_name),
            Some(md.get(&format!("{pfx}.mlp.0.bias"))?),
        )?;
        h.gelu_inplace();
        let mlp2_name = format!("{pfx}.mlp.2.weight");
        let ffn = linear::linear_auto(
            &h,
            md.try_get(&mlp2_name),
            md.get_quantized(&mlp2_name),
            Some(md.get(&format!("{pfx}.mlp.2.bias"))?),
        )?;
        x.add_inplace(&ffn);
    }

    // Final layer norm.
    x = x.layer_norm(
        ctx.model.get("decoder.ln.weight")?,
        ctx.model.get("decoder.ln.bias")?,
        1e-5,
    );

    // Logits: last_hidden @ tok_emb^T
    let last = Tensor::from_vec(
        x.data[(q_len - 1) * n_state..q_len * n_state].to_vec(),
        &[1, n_state],
    );
    let tok_emb_name = "decoder.token_embedding.weight";
    Ok(linear::linear_auto(
        &last,
        ctx.model.try_get(tok_emb_name),
        ctx.model.get_quantized(tok_emb_name),
        None,
    )?
    .data)
}

/// Rearrange [seq_len, n_state] -> [n_head, seq_len, head_dim]
pub(crate) fn to_head_first(
    data: &[f32],
    n_head: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_head * seq_len * head_dim];
    for s in 0..seq_len {
        for h in 0..n_head {
            let src = s * n_head * head_dim + h * head_dim;
            let dst = h * seq_len * head_dim + s * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&data[src..src + head_dim]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── encode_prompt_text tests ──────────────────────────────────────────

    #[test]
    fn test_encode_prompt_text_basic() {
        use crate::model::VocabEntry;
        let vocab = vec![
            VocabEntry::from_text("he"),
            VocabEntry::from_text("hello"),
            VocabEntry::from_text(" "),
            VocabEntry::from_text("world"),
        ];
        let tokens = encode_prompt_text("hello world", &vocab);
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_encode_prompt_text_skips_unknown_bytes() {
        use crate::model::VocabEntry;
        let vocab = vec![VocabEntry::from_text("a"), VocabEntry::from_text("b")];
        let tokens = encode_prompt_text("axb", &vocab);
        assert_eq!(tokens, vec![0, 1]);
    }

    #[test]
    fn test_encode_prompt_text_empty() {
        use crate::model::VocabEntry;
        let vocab = vec![VocabEntry::from_text("a")];
        let tokens = encode_prompt_text("", &vocab);
        assert!(tokens.is_empty());
    }

    // ── build_prompt tests ────────────────────────────────────────────────

    #[test]
    fn test_build_prompt_without_initial_tokens() {
        let special = SpecialTokens::new(51865);
        let lang = special.language_token("en");
        let prompt = build_prompt(&special, lang, special.transcribe, false, &[]);
        assert_eq!(prompt[0], special.sot);
        assert_eq!(prompt[1], lang);
        assert_eq!(prompt[2], special.transcribe);
        assert_eq!(prompt[3], special.no_timestamps);
        assert_eq!(prompt.len(), 4);
    }

    #[test]
    fn test_build_prompt_with_initial_tokens() {
        let special = SpecialTokens::new(51865);
        let lang = special.language_token("en");
        let initial = vec![10u32, 20, 30];
        let prompt = build_prompt(&special, lang, special.transcribe, false, &initial);
        assert_eq!(prompt[0], special.sot_prev);
        assert_eq!(prompt[1], 10);
        assert_eq!(prompt[2], 20);
        assert_eq!(prompt[3], 30);
        assert_eq!(prompt[4], special.sot);
        assert_eq!(prompt[5], lang);
        assert_eq!(prompt[6], special.transcribe);
        assert_eq!(prompt[7], special.no_timestamps);
        assert_eq!(prompt.len(), 8);
    }

    #[test]
    fn test_build_prompt_with_timestamps() {
        let special = SpecialTokens::new(51865);
        let lang = special.language_token("en");
        let prompt = build_prompt(&special, lang, special.transcribe, true, &[]);
        assert_eq!(prompt.len(), 3);
        assert_eq!(prompt[0], special.sot);
        assert_eq!(prompt[2], special.transcribe);
    }

    #[test]
    fn test_build_prompt_with_previous_tokens() {
        // Simulate previous_tokens being prepended to initial_tokens
        // (as done in decode() when opts.previous_tokens is Some)
        let special = SpecialTokens::new(51865);
        let lang = special.language_token("en");

        let previous = vec![10u32, 20];
        let initial = vec![30u32, 40];

        // Mimic the decode() logic: prepend previous before initial
        let mut all_initial = initial;
        let mut combined = previous;
        combined.append(&mut all_initial);
        let all_initial = combined;

        let prompt = build_prompt(&special, lang, special.transcribe, false, &all_initial);
        // Should be: [sot_prev, 10, 20, 30, 40, sot, lang, transcribe, no_timestamps]
        assert_eq!(prompt[0], special.sot_prev);
        assert_eq!(prompt[1], 10);
        assert_eq!(prompt[2], 20);
        assert_eq!(prompt[3], 30);
        assert_eq!(prompt[4], 40);
        assert_eq!(prompt[5], special.sot);
        assert_eq!(prompt[6], lang);
        assert_eq!(prompt[7], special.transcribe);
        assert_eq!(prompt[8], special.no_timestamps);
        assert_eq!(prompt.len(), 9);
    }

    #[test]
    fn test_build_prompt_previous_tokens_only() {
        // Only previous tokens, no initial prompt
        let special = SpecialTokens::new(51865);
        let lang = special.language_token("en");

        let previous = vec![50u32, 60, 70];
        // Mimic decode() logic with no initial_prompt
        let all_initial = previous;

        let prompt = build_prompt(&special, lang, special.transcribe, false, &all_initial);
        // Should be: [sot_prev, 50, 60, 70, sot, lang, transcribe, no_timestamps]
        assert_eq!(prompt[0], special.sot_prev);
        assert_eq!(prompt[1], 50);
        assert_eq!(prompt[2], 60);
        assert_eq!(prompt[3], 70);
        assert_eq!(prompt[4], special.sot);
        assert_eq!(prompt.len(), 8);
    }

    #[test]
    fn test_build_prompt_translate_emits_translate_token() {
        let special = SpecialTokens::new(51865);
        let lang = special.language_token("en");
        let prompt_t = build_prompt(&special, lang, special.transcribe, false, &[]);
        let prompt_tr = build_prompt(&special, lang, special.translate, false, &[]);
        assert_eq!(prompt_t[2], special.transcribe);
        assert_eq!(prompt_tr[2], special.translate);
        assert_ne!(special.transcribe, special.translate);
    }

    #[test]
    fn test_build_prompt_translate_token_value() {
        // Verify the well-known Whisper token IDs.
        let special = SpecialTokens::new(51865);
        assert_eq!(special.translate, 50358);
        assert_eq!(special.transcribe, 50359);
    }
}
