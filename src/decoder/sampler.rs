//! Token sampling strategies for the Whisper decoder: greedy and temperature sampling.

use super::cross_attn_capture::CrossAttnCapture;
use super::forward::{ForwardCtx, MAX_DECODE_LENGTH, forward};
use super::kv_cache::LayerKVCache;

use crate::decode_utils::{
    DecodeArgs, apply_no_repeat_ngram, apply_suppress_blank, apply_suppress_tokens,
    apply_timestamp_rules, argmax_with_log_prob, is_stop_token, sample_token, token_log_prob,
    token_prob,
};

/// Greedy (argmax) decoding with a KV cache.
///
/// Runs the decoder forward pass on the prompt, then autoregressively selects
/// the highest-probability token at each step until EOT or `MAX_DECODE_LENGTH`.
///
/// `capture_output` receives the flat `[n_tokens * enc_len]` cross-attention
/// matrix (one row per accepted output token, in the order they were accepted).
/// Pass `None` unless `word_timestamps` is enabled.
pub(crate) fn decode_greedy(
    prompt: &[u32],
    ctx: &ForwardCtx<'_>,
    args: &DecodeArgs<'_>,
    mut capture_output: Option<&mut Vec<f32>>,
) -> Result<(Vec<u32>, Vec<f32>, f32), String> {
    let kv_capacity = args.kv_capacity;
    let special = args.special;
    let eot_threshold = args.eot_threshold;
    let constraints = args.constraints;
    let dtype = args.dtype;
    let enc_len = ctx.enc_len;
    let capturing = capture_output.is_some();

    let mut self_kv: Vec<LayerKVCache> = (0..ctx.n_layer)
        .map(|_| LayerKVCache::new_with_dtype(ctx.n_head, ctx.head_dim, kv_capacity, dtype))
        .collect();

    // Prefill forward pass — optionally capture cross-attention.
    let mut prefill_cap = capturing.then(|| CrossAttnCapture::new(prompt.len(), enc_len));
    let mut logits = forward(prompt, 0, &mut self_kv, ctx, true, prefill_cap.as_mut())?;

    // Capture no_speech_prob BEFORE any suppression — suppression must not zero the no_speech
    // logit before we read the probability.
    let no_speech_prob = token_prob(&logits, special.no_speech);

    apply_suppress_tokens(&mut logits, constraints.suppress);
    if constraints.suppress_blank {
        apply_suppress_blank(&mut logits, special, constraints.blank_token);
    }
    if constraints.timestamp_rules {
        apply_timestamp_rules(&mut logits, &[], special);
    }
    let (first_token, first_lp) = argmax_with_log_prob(&logits);

    if first_token == special.eot {
        return Ok((Vec::new(), Vec::new(), no_speech_prob));
    }

    // Push the LAST row of the prefill capture (it corresponds to the first generated token).
    if let (Some(cap), Some(out)) = (prefill_cap, capture_output.as_deref_mut()) {
        let attn = cap.finish();
        let q_len = prompt.len();
        let last_row_start = (q_len.saturating_sub(1)) * enc_len;
        out.extend_from_slice(&attn[last_row_start..last_row_start + enc_len]);
    }

    let mut output_tokens = vec![first_token];
    let mut output_probs = vec![first_lp];
    let mut pos = prompt.len();
    let n_text_ctx = ctx.model.hparams.n_text_ctx;

    for _ in 1..MAX_DECODE_LENGTH {
        let tok = output_tokens[output_tokens.len() - 1];
        let mut step_cap = capturing.then(|| CrossAttnCapture::new(1, enc_len));
        let mut logits = forward(&[tok], pos, &mut self_kv, ctx, false, step_cap.as_mut())?;
        apply_suppress_tokens(&mut logits, constraints.suppress);
        if constraints.no_repeat_ngram_size > 0 {
            apply_no_repeat_ngram(
                &mut logits,
                &output_tokens,
                constraints.no_repeat_ngram_size,
            );
        }
        if constraints.timestamp_rules {
            apply_timestamp_rules(&mut logits, &output_tokens, special);
        }
        pos += 1;
        let (next, next_lp) = argmax_with_log_prob(&logits);
        if is_stop_token(next, eot_threshold, special) {
            break;
        }
        // Commit cross-attention row only when the token is accepted.
        if let (Some(cap), Some(out)) = (step_cap, capture_output.as_deref_mut()) {
            let attn = cap.finish();
            out.extend_from_slice(&attn[..enc_len]);
        }
        output_tokens.push(next);
        output_probs.push(next_lp);
        if pos >= n_text_ctx {
            break;
        }
    }

    Ok((output_tokens, output_probs, no_speech_prob))
}

/// Greedy decoding with temperature sampling (temperature > 0).
///
/// Runs the decoder forward pass on the prompt, then autoregressively samples
/// tokens according to a softmax distribution scaled by `opts.temperature`,
/// optionally filtered by `top_k` and `top_p` nucleus sampling.
///
/// `capture_output` receives the flat `[n_tokens * enc_len]` cross-attention
/// matrix.  Pass `None` unless `word_timestamps` is enabled.
pub(crate) fn decode_sample(
    prompt: &[u32],
    ctx: &ForwardCtx<'_>,
    args: &DecodeArgs<'_>,
    opts: &crate::TranscribeOptions<'_>,
    mut capture_output: Option<&mut Vec<f32>>,
) -> Result<(Vec<u32>, Vec<f32>, f32), String> {
    let kv_capacity = args.kv_capacity;
    let special = args.special;
    let eot_threshold = args.eot_threshold;
    let constraints = args.constraints;
    let dtype = args.dtype;
    let enc_len = ctx.enc_len;
    let capturing = capture_output.is_some();

    let mut self_kv: Vec<LayerKVCache> = (0..ctx.n_layer)
        .map(|_| LayerKVCache::new_with_dtype(ctx.n_head, ctx.head_dim, kv_capacity, dtype))
        .collect();

    // Prefill forward pass — optionally capture cross-attention.
    let mut prefill_cap = capturing.then(|| CrossAttnCapture::new(prompt.len(), enc_len));
    let mut logits = forward(prompt, 0, &mut self_kv, ctx, true, prefill_cap.as_mut())?;

    // Capture no_speech_prob BEFORE any suppression.
    let no_speech_prob = token_prob(&logits, special.no_speech);

    apply_suppress_tokens(&mut logits, constraints.suppress);
    if constraints.suppress_blank {
        apply_suppress_blank(&mut logits, special, constraints.blank_token);
    }
    if constraints.timestamp_rules {
        apply_timestamp_rules(&mut logits, &[], special);
    }

    let mut rng = rand::rng();
    let first_token = sample_token(&logits, opts.temperature, opts.top_k, opts.top_p, &mut rng);
    if is_stop_token(first_token, eot_threshold, special) {
        return Ok((Vec::new(), Vec::new(), no_speech_prob));
    }
    let first_lp = token_log_prob(&logits, first_token);

    // Push the LAST row of the prefill capture.
    if let (Some(cap), Some(out)) = (prefill_cap, capture_output.as_deref_mut()) {
        let attn = cap.finish();
        let q_len = prompt.len();
        let last_row_start = (q_len.saturating_sub(1)) * enc_len;
        out.extend_from_slice(&attn[last_row_start..last_row_start + enc_len]);
    }

    let mut output_tokens = vec![first_token];
    let mut output_probs = vec![first_lp];
    let mut pos = prompt.len();
    let n_text_ctx = ctx.model.hparams.n_text_ctx;

    for _ in 1..MAX_DECODE_LENGTH {
        let tok = output_tokens[output_tokens.len() - 1];
        let mut step_cap = capturing.then(|| CrossAttnCapture::new(1, enc_len));
        let mut logits = forward(&[tok], pos, &mut self_kv, ctx, false, step_cap.as_mut())?;
        apply_suppress_tokens(&mut logits, constraints.suppress);
        if constraints.no_repeat_ngram_size > 0 {
            apply_no_repeat_ngram(
                &mut logits,
                &output_tokens,
                constraints.no_repeat_ngram_size,
            );
        }
        if constraints.timestamp_rules {
            apply_timestamp_rules(&mut logits, &output_tokens, special);
        }
        pos += 1;
        let next = sample_token(&logits, opts.temperature, opts.top_k, opts.top_p, &mut rng);
        if is_stop_token(next, eot_threshold, special) {
            break;
        }
        let next_lp = token_log_prob(&logits, next);
        // Commit cross-attention row only when the token is accepted.
        if let (Some(cap), Some(out)) = (step_cap, capture_output.as_deref_mut()) {
            let attn = cap.finish();
            out.extend_from_slice(&attn[..enc_len]);
        }
        output_tokens.push(next);
        output_probs.push(next_lp);
        if pos >= n_text_ctx {
            break;
        }
    }

    Ok((output_tokens, output_probs, no_speech_prob))
}
