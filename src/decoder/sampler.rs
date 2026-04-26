//! Token sampling strategies for the Whisper decoder: greedy and temperature sampling.

use super::forward::{ForwardCtx, MAX_DECODE_LENGTH, forward};
use super::kv_cache::LayerKVCache;

use crate::decode_utils::{
    DecodeArgs, apply_no_repeat_ngram, apply_suppress_tokens, argmax, argmax_with_log_prob,
    is_stop_token, sample_token, token_log_prob,
};

/// Greedy (argmax) decoding with a KV cache.
///
/// Runs the decoder forward pass on the prompt, then autoregressively selects
/// the highest-probability token at each step until EOT or `MAX_DECODE_LENGTH`.
pub(crate) fn decode_greedy(
    prompt: &[u32],
    ctx: &ForwardCtx<'_>,
    args: &DecodeArgs<'_>,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    let kv_capacity = args.kv_capacity;
    let special = args.special;
    let eot_threshold = args.eot_threshold;
    let constraints = args.constraints;
    let dtype = args.dtype;

    let mut self_kv: Vec<LayerKVCache> = (0..ctx.n_layer)
        .map(|_| LayerKVCache::new_with_dtype(ctx.n_head, ctx.head_dim, kv_capacity, dtype))
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
///
/// Runs the decoder forward pass on the prompt, then autoregressively samples
/// tokens according to a softmax distribution scaled by `opts.temperature`,
/// optionally filtered by `top_k` and `top_p` nucleus sampling.
pub(crate) fn decode_sample(
    prompt: &[u32],
    ctx: &ForwardCtx<'_>,
    args: &DecodeArgs<'_>,
    opts: &crate::TranscribeOptions<'_>,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    let kv_capacity = args.kv_capacity;
    let special = args.special;
    let eot_threshold = args.eot_threshold;
    let constraints = args.constraints;
    let dtype = args.dtype;

    let mut self_kv: Vec<LayerKVCache> = (0..ctx.n_layer)
        .map(|_| LayerKVCache::new_with_dtype(ctx.n_head, ctx.head_dim, kv_capacity, dtype))
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
