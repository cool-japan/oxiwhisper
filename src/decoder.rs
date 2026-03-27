//! Core Whisper text decoder: forward pass, greedy/sample decoding, language detection.

use std::sync::Arc;

use crate::beam_search::decode_beam;
use crate::decode_utils::{
    DecodeConstraints, apply_no_repeat_ngram, apply_suppress_tokens, argmax, argmax_with_log_prob,
    is_stop_token, sample_token, token_log_prob,
};
use crate::linear;
use crate::model::ModelData;
use crate::tensor::Tensor;
use crate::tokenizer::SpecialTokens;

pub(crate) const MAX_DECODE_LENGTH: usize = 224;
/// Token id of the first language token (Whisper multilingual vocab).
const LANG_TOKEN_START: u32 = 50259;
/// One-past the last language token (99 languages total).
const LANG_TOKEN_END: u32 = 50358;

/// Self-attention KV cache for one decoder layer.
/// Pre-allocated to avoid O(T^2) copies during autoregressive generation.
/// Layout: [n_head, capacity, head_dim] -- head-first, fixed capacity.
///
/// Uses `Arc<Vec<f32>>` for copy-on-write semantics: beam search clones
/// share the same backing allocation until a beam actually writes, cutting
/// per-beam clone cost from ~2.8 MB to ~64 bytes (Arc refcount bump).
#[derive(Clone)]
pub(crate) struct LayerKVCache {
    k: Arc<Vec<f32>>,
    v: Arc<Vec<f32>>,
    pub(crate) seq_len: usize,
    pub(crate) n_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) capacity: usize, // per-head stride in elements = capacity * head_dim
}

impl LayerKVCache {
    pub(crate) fn new(n_head: usize, head_dim: usize, capacity: usize) -> Self {
        Self {
            k: Arc::new(vec![0.0f32; n_head * capacity * head_dim]),
            v: Arc::new(vec![0.0f32; n_head * capacity * head_dim]),
            seq_len: 0,
            n_head,
            head_dim,
            capacity,
        }
    }

    /// Append K,V for `new_seq_len` new tokens.
    /// new_k/v: [new_seq_len, n_state] row-major  (n_state = n_head * head_dim)
    ///
    /// Uses `Arc::make_mut` for copy-on-write: if this cache is the sole owner
    /// the write happens in-place; otherwise a deep copy is triggered first.
    pub(crate) fn append(&mut self, new_k: &[f32], new_v: &[f32], new_seq_len: usize) {
        let n_state = self.n_head * self.head_dim;
        let old = self.seq_len;
        debug_assert!(old + new_seq_len <= self.capacity, "KV cache overflow");

        let k_data = Arc::make_mut(&mut self.k);
        let v_data = Arc::make_mut(&mut self.v);

        for h in 0..self.n_head {
            for s in 0..new_seq_len {
                let src = s * n_state + h * self.head_dim;
                let dst = h * self.capacity * self.head_dim + (old + s) * self.head_dim;
                k_data[dst..dst + self.head_dim].copy_from_slice(&new_k[src..src + self.head_dim]);
                v_data[dst..dst + self.head_dim].copy_from_slice(&new_v[src..src + self.head_dim]);
            }
        }
        self.seq_len += new_seq_len;
    }

    /// K slice for head h, only the populated portion: [seq_len, head_dim]
    pub(crate) fn k_head(&self, h: usize) -> &[f32] {
        let off = h * self.capacity * self.head_dim;
        &self.k[off..off + self.seq_len * self.head_dim]
    }

    /// V slice for head h, only the populated portion: [seq_len, head_dim]
    pub(crate) fn v_head(&self, h: usize) -> &[f32] {
        let off = h * self.capacity * self.head_dim;
        &self.v[off..off + self.seq_len * self.head_dim]
    }
}

/// Read-only context passed to every `forward()` call.
/// Groups all model/shape parameters so call-sites stay readable.
pub(crate) struct ForwardCtx<'a> {
    pub(crate) cross_k: &'a [Vec<f32>],
    pub(crate) cross_v: &'a [Vec<f32>],
    pub(crate) enc_len: usize,
    pub(crate) tok_emb: &'a Tensor,
    pub(crate) pos_emb: &'a Tensor,
    pub(crate) model: &'a ModelData,
    pub(crate) n_state: usize,
    pub(crate) n_layer: usize,
    pub(crate) n_head: usize,
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

    let ctx = ForwardCtx {
        cross_k: &cross_k,
        cross_v: &cross_v,
        enc_len,
        tok_emb: model.get("decoder.token_embedding.weight")?,
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
    let prompt = build_prompt(&special, lang_token, opts.timestamps, &all_initial);
    let kv_capacity = prompt.len() + MAX_DECODE_LENGTH + 4;

    let eot_threshold = special.eot;

    let constraints = DecodeConstraints {
        suppress: opts.suppress_tokens.unwrap_or(&[]),
        no_repeat_ngram_size: opts.no_repeat_ngram_size,
    };

    let (tokens, token_probs) = if opts.temperature > 0.0 {
        decode_sample(
            &prompt,
            kv_capacity,
            &ctx,
            &special,
            opts,
            eot_threshold,
            &constraints,
        )?
    } else if opts.beam_width <= 1 {
        decode_greedy(
            &prompt,
            kv_capacity,
            &ctx,
            &special,
            eot_threshold,
            &constraints,
        )?
    } else {
        decode_beam(
            &prompt,
            kv_capacity,
            &ctx,
            &special,
            opts.beam_width,
            eot_threshold,
            &constraints,
        )?
    };

    Ok(DecodeResult {
        tokens,
        token_probs,
        detected_language,
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
fn detect_language(ctx: &ForwardCtx<'_>, special: &SpecialTokens) -> Result<u32, String> {
    let cap = MAX_DECODE_LENGTH + 10;
    let mut kv: Vec<LayerKVCache> = (0..ctx.n_layer)
        .map(|_| LayerKVCache::new(ctx.n_head, ctx.head_dim, cap))
        .collect();
    let logits = forward(&[special.sot], 0, &mut kv, ctx, true)?;
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
fn encode_prompt_text(text: &str, vocab: &[crate::model::VocabEntry]) -> Vec<u32> {
    let mut tokens = Vec::new();
    let bytes = text.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        let mut best_len = 0;
        let mut best_tok = 0u32;
        for (i, entry) in vocab.iter().enumerate() {
            let entry_bytes = entry.text.as_bytes();
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

fn build_prompt(
    special: &SpecialTokens,
    lang_token: u32,
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
    prompt.push(special.transcribe);
    if !timestamps {
        prompt.push(special.no_timestamps);
    }
    prompt
}

fn decode_greedy(
    prompt: &[u32],
    kv_capacity: usize,
    ctx: &ForwardCtx<'_>,
    special: &SpecialTokens,
    eot_threshold: u32,
    constraints: &DecodeConstraints<'_>,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    let mut self_kv: Vec<LayerKVCache> = (0..ctx.n_layer)
        .map(|_| LayerKVCache::new(ctx.n_head, ctx.head_dim, kv_capacity))
        .collect();

    let mut logits = forward(prompt, 0, &mut self_kv, ctx, true)?;
    apply_suppress_tokens(&mut logits, constraints.suppress);
    let (first_token, first_lp) = argmax_with_log_prob(&logits);

    if first_token == special.no_speech {
        #[cfg(feature = "timing")]
        eprintln!("No speech detected");
        return Ok((Vec::new(), Vec::new()));
    }
    if first_token == special.eot {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut output_tokens = vec![first_token];
    let mut output_probs = vec![first_lp];
    let mut pos = prompt.len();
    let n_text_ctx = ctx.model.hparams.n_text_ctx;

    for _ in 1..MAX_DECODE_LENGTH {
        let tok = output_tokens[output_tokens.len() - 1];
        let mut logits = forward(&[tok], pos, &mut self_kv, ctx, false)?;
        apply_suppress_tokens(&mut logits, constraints.suppress);
        if constraints.no_repeat_ngram_size > 0 {
            apply_no_repeat_ngram(
                &mut logits,
                &output_tokens,
                constraints.no_repeat_ngram_size,
            );
        }
        pos += 1;
        let (next, next_lp) = argmax_with_log_prob(&logits);
        if is_stop_token(next, eot_threshold, special) {
            break;
        }
        output_tokens.push(next);
        output_probs.push(next_lp);
        if pos >= n_text_ctx {
            break;
        }
    }

    Ok((output_tokens, output_probs))
}

/// Greedy decoding with temperature sampling (temperature > 0).
fn decode_sample(
    prompt: &[u32],
    kv_capacity: usize,
    ctx: &ForwardCtx<'_>,
    special: &SpecialTokens,
    opts: &crate::TranscribeOptions<'_>,
    eot_threshold: u32,
    constraints: &DecodeConstraints<'_>,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    let mut self_kv: Vec<LayerKVCache> = (0..ctx.n_layer)
        .map(|_| LayerKVCache::new(ctx.n_head, ctx.head_dim, kv_capacity))
        .collect();

    let mut logits = forward(prompt, 0, &mut self_kv, ctx, true)?;
    apply_suppress_tokens(&mut logits, constraints.suppress);

    if argmax(&logits) == special.no_speech {
        #[cfg(feature = "timing")]
        eprintln!("No speech detected");
        return Ok((Vec::new(), Vec::new()));
    }

    let mut rng = rand::rng();
    let first_token = sample_token(&logits, opts.temperature, opts.top_k, opts.top_p, &mut rng);
    if is_stop_token(first_token, eot_threshold, special) {
        return Ok((Vec::new(), Vec::new()));
    }
    let first_lp = token_log_prob(&logits, first_token);

    let mut output_tokens = vec![first_token];
    let mut output_probs = vec![first_lp];
    let mut pos = prompt.len();
    let n_text_ctx = ctx.model.hparams.n_text_ctx;

    for _ in 1..MAX_DECODE_LENGTH {
        let tok = output_tokens[output_tokens.len() - 1];
        let mut logits = forward(&[tok], pos, &mut self_kv, ctx, false)?;
        apply_suppress_tokens(&mut logits, constraints.suppress);
        if constraints.no_repeat_ngram_size > 0 {
            apply_no_repeat_ngram(
                &mut logits,
                &output_tokens,
                constraints.no_repeat_ngram_size,
            );
        }
        pos += 1;
        let next = sample_token(&logits, opts.temperature, opts.top_k, opts.top_p, &mut rng);
        if is_stop_token(next, eot_threshold, special) {
            break;
        }
        let next_lp = token_log_prob(&logits, next);
        output_tokens.push(next);
        output_probs.push(next_lp);
        if pos >= n_text_ctx {
            break;
        }
    }

    Ok((output_tokens, output_probs))
}

/// One decoder forward pass -- updates self_kv in place, returns logits for the last token.
pub(crate) fn forward(
    tokens: &[u32],
    start_pos: usize,
    self_kv: &mut [LayerKVCache],
    ctx: &ForwardCtx<'_>,
    causal_mask: bool,
) -> Result<Vec<f32>, String> {
    let q_len = tokens.len();
    let n_state = ctx.n_state;
    let n_head = ctx.n_head;
    let head_dim = ctx.head_dim;

    // Token + positional embedding.
    let mut x_data = vec![0.0f32; q_len * n_state];
    for (i, &tok) in tokens.iter().enumerate() {
        let idx = tok as usize;
        // GGML tok_emb shape: [n_state, n_vocab] -- check against shape[1] (n_vocab).
        if idx < ctx.tok_emb.shape[1] {
            x_data[i * n_state..(i + 1) * n_state]
                .copy_from_slice(&ctx.tok_emb.data[idx * n_state..(idx + 1) * n_state]);
        }
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
        let attn_self = scaled_dot_product_cached(&q_hf, layer_kv, &sdpa_cfg);
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
        let attn_cross = scaled_dot_product_flat(
            &q_hf,
            &ctx.cross_k[layer],
            &ctx.cross_v[layer],
            n_head,
            q_len,
            ctx.enc_len,
            head_dim,
        );
        let attn_cross = Tensor::from_vec(attn_cross, &[q_len, n_state]);
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

/// Configuration for the cached scaled dot-product attention.
struct CachedSdpaConfig {
    n_head: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    past_len: usize,
    causal_mask: bool,
}

/// Scaled dot-product attention using the pre-allocated KV cache.
///
/// Optimised causal-mask path: instead of computing scores for all `kv_len`
/// positions and then masking the invalid ones to `-inf`, we only iterate
/// over the positions that the query is allowed to attend to.
fn scaled_dot_product_cached(q: &[f32], kv: &LayerKVCache, cfg: &CachedSdpaConfig) -> Vec<f32> {
    let scale = (cfg.head_dim as f32).sqrt().recip();
    let n_state = cfg.n_head * cfg.head_dim;
    let mut out = vec![0.0f32; cfg.q_len * n_state];

    if !cfg.causal_mask {
        sdpa_cached_full(q, kv, cfg, scale, n_state, &mut out);
    } else if cfg.q_len == 1 {
        let valid_len = cfg.past_len + 1;
        sdpa_cached_single(q, kv, cfg, scale, n_state, valid_len, &mut out);
    } else {
        sdpa_cached_prefill(q, kv, cfg, scale, n_state, &mut out);
    }

    out
}

/// Full attention (no causal mask) using the KV cache.
fn sdpa_cached_full(
    q: &[f32],
    kv: &LayerKVCache,
    cfg: &CachedSdpaConfig,
    scale: f32,
    n_state: usize,
    out: &mut [f32],
) {
    for h in 0..cfg.n_head {
        let q_off = h * cfg.q_len * cfg.head_dim;
        let k = kv.k_head(h);
        let v = kv.v_head(h);

        let mut scores = vec![0.0f32; cfg.q_len * cfg.kv_len];

        for i in 0..cfg.q_len {
            let q_row = &q[q_off + i * cfg.head_dim..q_off + (i + 1) * cfg.head_dim];
            for j in 0..cfg.kv_len {
                let k_row = &k[j * cfg.head_dim..(j + 1) * cfg.head_dim];
                let mut s = 0.0f32;
                for d in 0..cfg.head_dim {
                    s += q_row[d] * k_row[d];
                }
                scores[i * cfg.kv_len + j] = s * scale;
            }
        }

        softmax_rows(&mut scores, cfg.q_len, cfg.kv_len);

        for i in 0..cfg.q_len {
            let out_row =
                &mut out[i * n_state + h * cfg.head_dim..i * n_state + (h + 1) * cfg.head_dim];
            for j in 0..cfg.kv_len {
                let s = scores[i * cfg.kv_len + j];
                if s == 0.0 {
                    continue;
                }
                let v_row = &v[j * cfg.head_dim..(j + 1) * cfg.head_dim];
                for d in 0..cfg.head_dim {
                    out_row[d] += s * v_row[d];
                }
            }
        }
    }
}

/// Optimised single-query attention (q_len == 1) with causal mask.
fn sdpa_cached_single(
    q: &[f32],
    kv: &LayerKVCache,
    cfg: &CachedSdpaConfig,
    scale: f32,
    _n_state: usize,
    valid_len: usize,
    out: &mut [f32],
) {
    for h in 0..cfg.n_head {
        let q_off = h * cfg.head_dim;
        let k = kv.k_head(h);
        let v = kv.v_head(h);
        let q_row = &q[q_off..q_off + cfg.head_dim];

        let mut scores = vec![0.0f32; valid_len];
        for j in 0..valid_len {
            let k_row = &k[j * cfg.head_dim..(j + 1) * cfg.head_dim];
            let mut s = 0.0f32;
            for d in 0..cfg.head_dim {
                s += q_row[d] * k_row[d];
            }
            scores[j] = s * scale;
        }

        softmax_rows(&mut scores, 1, valid_len);

        let out_row = &mut out[h * cfg.head_dim..(h + 1) * cfg.head_dim];
        for j in 0..valid_len {
            let s = scores[j];
            if s == 0.0 {
                continue;
            }
            let v_row = &v[j * cfg.head_dim..(j + 1) * cfg.head_dim];
            for d in 0..cfg.head_dim {
                out_row[d] += s * v_row[d];
            }
        }
    }
}

/// Optimised prefill attention (q_len > 1) with causal mask.
fn sdpa_cached_prefill(
    q: &[f32],
    kv: &LayerKVCache,
    cfg: &CachedSdpaConfig,
    scale: f32,
    n_state: usize,
    out: &mut [f32],
) {
    for h in 0..cfg.n_head {
        let q_off = h * cfg.q_len * cfg.head_dim;
        let k = kv.k_head(h);
        let v = kv.v_head(h);

        for i in 0..cfg.q_len {
            let q_row = &q[q_off + i * cfg.head_dim..q_off + (i + 1) * cfg.head_dim];
            let valid_len = (cfg.past_len + i + 1).min(cfg.kv_len);

            let mut scores = vec![0.0f32; valid_len];
            for j in 0..valid_len {
                let k_row = &k[j * cfg.head_dim..(j + 1) * cfg.head_dim];
                let mut s = 0.0f32;
                for d in 0..cfg.head_dim {
                    s += q_row[d] * k_row[d];
                }
                scores[j] = s * scale;
            }

            softmax_rows(&mut scores, 1, valid_len);

            let out_row =
                &mut out[i * n_state + h * cfg.head_dim..i * n_state + (h + 1) * cfg.head_dim];
            for j in 0..valid_len {
                let s = scores[j];
                if s == 0.0 {
                    continue;
                }
                let v_row = &v[j * cfg.head_dim..(j + 1) * cfg.head_dim];
                for d in 0..cfg.head_dim {
                    out_row[d] += s * v_row[d];
                }
            }
        }
    }
}

/// Scaled dot-product attention for cross-attention (flat head-first K,V, no causal mask).
fn scaled_dot_product_flat(
    q: &[f32], // [n_head, q_len, head_dim]
    k: &[f32], // [n_head, kv_len, head_dim]  (pre-built, tightly packed)
    v: &[f32],
    n_head: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = (head_dim as f32).sqrt().recip();
    let n_state = n_head * head_dim;
    let mut out = vec![0.0f32; q_len * n_state];

    for h in 0..n_head {
        let q_off = h * q_len * head_dim;
        let k_off = h * kv_len * head_dim;
        let v_off = h * kv_len * head_dim;

        let mut scores = vec![0.0f32; q_len * kv_len];

        for i in 0..q_len {
            let q_row = &q[q_off + i * head_dim..q_off + (i + 1) * head_dim];
            for j in 0..kv_len {
                let k_row = &k[k_off + j * head_dim..k_off + (j + 1) * head_dim];
                let mut s = 0.0f32;
                for d in 0..head_dim {
                    s += q_row[d] * k_row[d];
                }
                scores[i * kv_len + j] = s * scale;
            }
        }

        softmax_rows(&mut scores, q_len, kv_len);

        for i in 0..q_len {
            let out_row = &mut out[i * n_state + h * head_dim..i * n_state + (h + 1) * head_dim];
            for j in 0..kv_len {
                let s = scores[i * kv_len + j];
                if s == 0.0 {
                    continue;
                }
                let v_row = &v[v_off + j * head_dim..v_off + (j + 1) * head_dim];
                for d in 0..head_dim {
                    out_row[d] += s * v_row[d];
                }
            }
        }
    }

    out
}

fn softmax_rows(scores: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row = &mut scores[i * cols..(i + 1) * cols];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in row.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        if sum > 0.0 {
            let inv = sum.recip();
            for x in row.iter_mut() {
                *x *= inv;
            }
        }
    }
}

/// Rearrange [seq_len, n_state] -> [n_head, seq_len, head_dim]
fn to_head_first(data: &[f32], n_head: usize, seq_len: usize, head_dim: usize) -> Vec<f32> {
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

    // ── KV cache unit tests ─────────────────────────────────────────────────

    #[test]
    fn test_kv_cache_new() {
        let cache = LayerKVCache::new(2, 4, 8);
        assert_eq!(cache.seq_len, 0);
        assert_eq!(cache.n_head, 2);
        assert_eq!(cache.head_dim, 4);
        assert_eq!(cache.capacity, 8);
        assert!(cache.k_head(0).is_empty());
        assert!(cache.k_head(1).is_empty());
        assert!(cache.v_head(0).is_empty());
        assert!(cache.v_head(1).is_empty());
    }

    #[test]
    fn test_kv_cache_append() {
        let n_head = 2;
        let head_dim = 4;
        let capacity = 8;
        let n_state = n_head * head_dim;
        let mut cache = LayerKVCache::new(n_head, head_dim, capacity);

        let new_k: Vec<f32> = (1..=((n_state * 2) as i32)).map(|x| x as f32).collect();
        let new_v: Vec<f32> = (101..=(100 + (n_state * 2) as i32))
            .map(|x| x as f32)
            .collect();
        cache.append(&new_k, &new_v, 2);

        assert_eq!(cache.seq_len, 2);

        let k0 = cache.k_head(0);
        assert_eq!(k0.len(), 2 * head_dim);
        assert_eq!(k0[0..4], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(k0[4..8], [9.0, 10.0, 11.0, 12.0]);

        let k1 = cache.k_head(1);
        assert_eq!(k1.len(), 2 * head_dim);
        assert_eq!(k1[0..4], [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(k1[4..8], [13.0, 14.0, 15.0, 16.0]);

        let v0 = cache.v_head(0);
        assert_eq!(v0[0..4], [101.0, 102.0, 103.0, 104.0]);
        assert_eq!(v0[4..8], [109.0, 110.0, 111.0, 112.0]);
    }

    #[test]
    fn test_kv_cache_cow_clone() {
        let n_head = 2;
        let head_dim = 4;
        let capacity = 8;
        let n_state = n_head * head_dim;
        let mut cache = LayerKVCache::new(n_head, head_dim, capacity);

        let new_k: Vec<f32> = (1..=(n_state as i32)).map(|x| x as f32).collect();
        let new_v: Vec<f32> = (101..=(100 + n_state as i32)).map(|x| x as f32).collect();
        cache.append(&new_k, &new_v, 1);
        assert_eq!(cache.seq_len, 1);

        let mut clone = cache.clone();
        assert_eq!(clone.seq_len, 1);

        assert!(Arc::ptr_eq(&cache.k, &clone.k));
        assert!(Arc::ptr_eq(&cache.v, &clone.v));

        assert_eq!(cache.k_head(0), clone.k_head(0));
        assert_eq!(cache.v_head(0), clone.v_head(0));

        let new_k2: Vec<f32> = (21..=(20 + n_state as i32)).map(|x| x as f32).collect();
        let new_v2: Vec<f32> = (121..=(120 + n_state as i32)).map(|x| x as f32).collect();
        clone.append(&new_k2, &new_v2, 1);

        assert_eq!(clone.seq_len, 2);
        assert_eq!(cache.seq_len, 1);

        assert!(!Arc::ptr_eq(&cache.k, &clone.k));

        assert_eq!(cache.k_head(0), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_kv_cache_append_multiple() {
        let n_head = 2;
        let head_dim = 3;
        let capacity = 16;
        let n_state = n_head * head_dim;
        let mut cache = LayerKVCache::new(n_head, head_dim, capacity);

        for t in 0..4u32 {
            let base = (t * n_state as u32 + 1) as f32;
            let new_k: Vec<f32> = (0..n_state).map(|i| base + i as f32).collect();
            let new_v: Vec<f32> = (0..n_state).map(|i| base + 100.0 + i as f32).collect();
            cache.append(&new_k, &new_v, 1);
            assert_eq!(cache.seq_len, (t + 1) as usize);
        }

        assert_eq!(cache.seq_len, 4);

        let k0 = cache.k_head(0);
        assert_eq!(k0.len(), 4 * head_dim);
        assert_eq!(k0[0..3], [1.0, 2.0, 3.0]);
        assert_eq!(k0[3..6], [7.0, 8.0, 9.0]);
        assert_eq!(k0[6..9], [13.0, 14.0, 15.0]);
        assert_eq!(k0[9..12], [19.0, 20.0, 21.0]);

        let k1 = cache.k_head(1);
        assert_eq!(k1[0..3], [4.0, 5.0, 6.0]);
    }

    // ── encode_prompt_text tests ──────────────────────────────────────────

    #[test]
    fn test_encode_prompt_text_basic() {
        use crate::model::VocabEntry;
        let vocab = vec![
            VocabEntry {
                text: "he".to_string(),
            },
            VocabEntry {
                text: "hello".to_string(),
            },
            VocabEntry {
                text: " ".to_string(),
            },
            VocabEntry {
                text: "world".to_string(),
            },
        ];
        let tokens = encode_prompt_text("hello world", &vocab);
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_encode_prompt_text_skips_unknown_bytes() {
        use crate::model::VocabEntry;
        let vocab = vec![
            VocabEntry {
                text: "a".to_string(),
            },
            VocabEntry {
                text: "b".to_string(),
            },
        ];
        let tokens = encode_prompt_text("axb", &vocab);
        assert_eq!(tokens, vec![0, 1]);
    }

    #[test]
    fn test_encode_prompt_text_empty() {
        use crate::model::VocabEntry;
        let vocab = vec![VocabEntry {
            text: "a".to_string(),
        }];
        let tokens = encode_prompt_text("", &vocab);
        assert!(tokens.is_empty());
    }

    // ── build_prompt tests ────────────────────────────────────────────────

    #[test]
    fn test_build_prompt_without_initial_tokens() {
        let special = SpecialTokens::new(51865);
        let lang = special.language_token("en");
        let prompt = build_prompt(&special, lang, false, &[]);
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
        let prompt = build_prompt(&special, lang, false, &initial);
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
        let prompt = build_prompt(&special, lang, true, &[]);
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

        let prompt = build_prompt(&special, lang, false, &all_initial);
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

        let prompt = build_prompt(&special, lang, false, &all_initial);
        // Should be: [sot_prev, 50, 60, 70, sot, lang, transcribe, no_timestamps]
        assert_eq!(prompt[0], special.sot_prev);
        assert_eq!(prompt[1], 50);
        assert_eq!(prompt[2], 60);
        assert_eq!(prompt[3], 70);
        assert_eq!(prompt[4], special.sot);
        assert_eq!(prompt.len(), 8);
    }
}
