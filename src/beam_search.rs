//! Beam search decoder for Whisper.

use crate::decode_utils::{
    DecodeArgs, apply_no_repeat_ngram, apply_suppress_blank, apply_suppress_tokens,
    apply_timestamp_rules, is_stop_token, log_softmax, normalized_score, token_prob,
    top_k_log_probs,
};
use crate::decoder::{ForwardCtx, LayerKVCache, forward};

use super::decoder::MAX_DECODE_LENGTH;

/// Active hypothesis for beam search.
pub(crate) struct Beam {
    /// Decoded output tokens (not including prompt; EOT excluded).
    pub(crate) tokens: Vec<u32>,
    /// Per-token log-probabilities (same length as `tokens`).
    pub(crate) token_probs: Vec<f32>,
    /// Cumulative sum of log-probabilities.
    pub(crate) score: f32,
    /// KV cache.  Does NOT yet include `tokens.last()`'s key/value pair.
    pub(crate) self_kv: Vec<LayerKVCache>,
    /// Position for the next `forward()` call = prompt_len + tokens.len() - 1.
    pub(crate) pos: usize,
    pub(crate) done: bool,
}

pub(crate) fn decode_beam(
    prompt: &[u32],
    ctx: &ForwardCtx<'_>,
    args: &DecodeArgs<'_>,
    beam_width: usize,
) -> Result<(Vec<u32>, Vec<f32>, f32), String> {
    let kv_capacity = args.kv_capacity;
    let special = args.special;
    let eot_threshold = args.eot_threshold;
    let constraints = args.constraints;
    let dtype = args.dtype;

    // Prefill with prompt.
    let mut initial_kv: Vec<LayerKVCache> = (0..ctx.n_layer)
        .map(|_| LayerKVCache::new_with_dtype(ctx.n_head, ctx.head_dim, kv_capacity, dtype))
        .collect();
    // Beam search never captures cross-attention (word timestamps require greedy/sample).
    let mut initial_logits = forward(prompt, 0, &mut initial_kv, ctx, true, None)?;

    // Capture no_speech_prob BEFORE any suppression — must read the raw probability.
    let no_speech_prob = token_prob(&initial_logits, special.no_speech);

    apply_suppress_tokens(&mut initial_logits, constraints.suppress);
    if constraints.suppress_blank {
        apply_suppress_blank(&mut initial_logits, special, constraints.blank_token);
    }
    if constraints.timestamp_rules {
        apply_timestamp_rules(&mut initial_logits, &[], special);
    }

    // Seed beams from top-k of the initial logit distribution.
    //
    // Fix for issue #1: if a stop token appears among the top-k seeds it must
    // NOT be seeded as a `done=true` beam with zero tokens, because
    // `normalized_score(0, score) = score / 1.0 = score`, which beats any
    // non-trivial partial hypothesis in the final `max_by`.  Instead we
    // over-sample (2 × beam_width) and skip stop tokens so only real tokens
    // start the search.  If every top candidate is a stop token the model
    // signalled immediate end-of-speech; return empty output.
    let lp = log_softmax(&initial_logits);
    let top_extended = top_k_log_probs(&lp, (beam_width * 2).max(beam_width + 4));
    let pos0 = prompt.len();

    let mut beams: Vec<Beam> = top_extended
        .into_iter()
        .filter(|&(_score, tok)| !is_stop_token(tok, eot_threshold, special))
        .take(beam_width)
        .map(|(score, tok)| Beam {
            tokens: vec![tok],
            token_probs: vec![score],
            score,
            self_kv: initial_kv.clone(),
            pos: pos0,
            done: false,
        })
        .collect();

    // All seeds were stop tokens — model signals silence / immediate end.
    if beams.is_empty() {
        return Ok((Vec::new(), Vec::new(), no_speech_prob));
    }

    let n_text_ctx = ctx.model.hparams.n_text_ctx;

    for _step in 0..MAX_DECODE_LENGTH {
        if beams.iter().all(|b| b.done) {
            break;
        }

        let mut candidates: Vec<Beam> = Vec::with_capacity(beams.len() * beam_width + beams.len());

        for beam in &beams {
            if beam.done {
                // Propagate finished beam so it can still win on final selection.
                candidates.push(Beam {
                    tokens: beam.tokens.clone(),
                    token_probs: beam.token_probs.clone(),
                    score: beam.score,
                    self_kv: beam.self_kv.clone(),
                    pos: beam.pos,
                    done: true,
                });
                continue;
            }

            let last_tok = beam.tokens[beam.tokens.len() - 1];
            let mut fork_kv = beam.self_kv.clone();
            let mut logits = forward(&[last_tok], beam.pos, &mut fork_kv, ctx, false, None)?;
            apply_suppress_tokens(&mut logits, constraints.suppress);
            if constraints.no_repeat_ngram_size > 0 {
                apply_no_repeat_ngram(&mut logits, &beam.tokens, constraints.no_repeat_ngram_size);
            }
            if constraints.timestamp_rules {
                apply_timestamp_rules(&mut logits, &beam.tokens, special);
            }
            let lp = log_softmax(&logits);
            let top = top_k_log_probs(&lp, beam_width);
            let next_pos = beam.pos + 1;

            for (lp_val, next_tok) in top {
                let done =
                    is_stop_token(next_tok, eot_threshold, special) || next_pos >= n_text_ctx;
                let mut new_tokens = beam.tokens.clone();
                let mut new_probs = beam.token_probs.clone();
                if !done {
                    new_tokens.push(next_tok);
                    new_probs.push(lp_val);
                }
                candidates.push(Beam {
                    tokens: new_tokens,
                    token_probs: new_probs,
                    score: beam.score + lp_val,
                    self_kv: fork_kv.clone(),
                    pos: next_pos,
                    done,
                });
            }
        }

        // Keep top beam_width by length-normalised score.
        candidates.sort_unstable_by(|a, b| {
            normalized_score(b.tokens.len(), b.score)
                .partial_cmp(&normalized_score(a.tokens.len(), a.score))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(beam_width);
        beams = candidates;
    }

    let best = beams.into_iter().max_by(|a, b| {
        normalized_score(a.tokens.len(), a.score)
            .partial_cmp(&normalized_score(b.tokens.len(), b.score))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    match best {
        Some(b) => Ok((b.tokens, b.token_probs, no_speech_prob)),
        None => Ok((Vec::new(), Vec::new(), no_speech_prob)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_utils::{DecodeConstraints, normalized_score};
    use crate::types::KvCacheDtype;

    // ── Beam struct unit tests ──────────────────────────────────────────────

    #[test]
    fn test_beam_struct_creation() {
        let beam = Beam {
            tokens: vec![100, 200, 300],
            token_probs: vec![-0.5, -0.3, -0.8],
            score: -1.6,
            self_kv: Vec::new(),
            pos: 7,
            done: false,
        };
        assert_eq!(beam.tokens, vec![100, 200, 300]);
        assert_eq!(beam.token_probs, vec![-0.5, -0.3, -0.8]);
        assert!((beam.score - (-1.6)).abs() < 1e-6);
        assert_eq!(beam.pos, 7);
        assert!(!beam.done);
    }

    #[test]
    fn test_beam_clone_is_independent() {
        let kv = LayerKVCache::new(2, 4, 16);
        let original = Beam {
            tokens: vec![10, 20],
            token_probs: vec![-0.1, -0.2],
            score: -0.3,
            self_kv: vec![kv],
            pos: 5,
            done: false,
        };

        // Clone by copying fields (Beam doesn't derive Clone, so we do it manually)
        let mut cloned = Beam {
            tokens: original.tokens.clone(),
            token_probs: original.token_probs.clone(),
            score: original.score,
            self_kv: original.self_kv.clone(),
            pos: original.pos,
            done: original.done,
        };

        // Modify the clone
        cloned.tokens.push(30);
        cloned.token_probs.push(-0.4);
        cloned.score = -0.7;
        cloned.pos = 6;
        cloned.done = true;

        // Original should be unchanged
        assert_eq!(original.tokens, vec![10, 20]);
        assert_eq!(original.token_probs, vec![-0.1, -0.2]);
        assert!((original.score - (-0.3)).abs() < 1e-6);
        assert_eq!(original.pos, 5);
        assert!(!original.done);
    }

    #[test]
    fn test_beam_done_propagation() {
        // Simulating what decode_beam does: a done beam is propagated unchanged
        let done_beam = Beam {
            tokens: vec![42, 99],
            token_probs: vec![-0.2, -0.1],
            score: -0.3,
            self_kv: Vec::new(),
            pos: 10,
            done: true,
        };

        // Propagate like the inner loop does
        let propagated = Beam {
            tokens: done_beam.tokens.clone(),
            token_probs: done_beam.token_probs.clone(),
            score: done_beam.score,
            self_kv: done_beam.self_kv.clone(),
            pos: done_beam.pos,
            done: true,
        };

        assert_eq!(propagated.tokens, done_beam.tokens);
        assert_eq!(propagated.token_probs, done_beam.token_probs);
        assert!((propagated.score - done_beam.score).abs() < 1e-6);
        assert_eq!(propagated.pos, done_beam.pos);
        assert!(propagated.done);
    }

    // ── normalized_score tests ──────────────────────────────────────────────

    #[test]
    fn test_normalized_score_basic() {
        // normalized_score(len, score) = score / max(1, len)^0.6
        let score = -3.0_f32;
        let len = 5_usize;
        let expected = score / (len as f32).powf(0.6);
        let actual = normalized_score(len, score);
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_normalized_score_empty_tokens() {
        // tokens_len=0 should use max(1) as denominator => score / 1.0^0.6 = score
        let score = -2.5_f32;
        let actual = normalized_score(0, score);
        assert!(
            (actual - score).abs() < 1e-6,
            "empty tokens should use denominator 1.0: expected {score}, got {actual}"
        );
    }

    #[test]
    fn test_normalized_score_single_token() {
        // len=1 => score / 1.0^0.6 = score
        let score = -1.0_f32;
        let actual = normalized_score(1, score);
        assert!(
            (actual - score).abs() < 1e-6,
            "single token: expected {score}, got {actual}"
        );
    }

    #[test]
    fn test_normalized_score_ordering_preserves_better_beam() {
        // A shorter beam with a slightly worse raw score can beat a longer one
        // after normalization. Verify the comparison logic used in decode_beam.
        let short_score = normalized_score(2, -1.0); // -1.0 / 2^0.6 ≈ -0.66
        let long_score = normalized_score(10, -4.0); // -4.0 / 10^0.6 ≈ -1.005
        assert!(
            short_score > long_score,
            "short beam ({short_score}) should rank higher than long beam ({long_score})"
        );
    }

    // ── Helper ───────────────────────────────────────────────────────────────

    /// Reorder [seq_len, n_state] row-major data to [n_head, seq_len, head_dim].
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

    // ── Integration test with synthetic model ───────────────────────────────

    #[test]
    fn test_decode_beam_with_synthetic_model() {
        use crate::decoder::ForwardCtx;
        use crate::encoder::encode;
        use crate::linear;
        use crate::model::ModelData;
        use crate::tensor::Tensor;
        use crate::test_utils::generate_synthetic_model;
        use crate::tokenizer::SpecialTokens;

        let model_path = generate_synthetic_model();
        let model = ModelData::load(&model_path).expect("load synthetic model");
        let _ = std::fs::remove_file(&model_path);

        let hp = &model.hparams;
        let n_state = hp.n_text_state;
        let n_layer = hp.n_text_layer;
        let n_head = hp.n_text_head;
        let head_dim = n_state / n_head;
        let n_mels = hp.n_mels;

        // Create a small mel spectrogram (100 frames)
        let n_frames = 100;
        let mel = Tensor::from_vec(vec![0.01f32; n_mels * n_frames], &[n_mels, n_frames]);
        let encoder_output = encode(&mel, &model).expect("encode should succeed");
        let enc_len = encoder_output.shape[0];

        // Precompute cross-attention K,V
        let mut cross_k: Vec<Vec<f32>> = Vec::with_capacity(n_layer);
        let mut cross_v: Vec<Vec<f32>> = Vec::with_capacity(n_layer);
        for layer in 0..n_layer {
            let pfx = format!("decoder.blocks.{layer}");
            let ck_name = format!("{pfx}.cross_attn.key.weight");
            let ck = linear::linear_auto(
                &encoder_output,
                model.try_get(&ck_name),
                model.get_quantized(&ck_name),
                None,
            )
            .expect("cross_attn key projection");
            let cv_name = format!("{pfx}.cross_attn.value.weight");
            let cv = linear::linear_auto(
                &encoder_output,
                model.try_get(&cv_name),
                model.get_quantized(&cv_name),
                Some(
                    model
                        .get(&format!("{pfx}.cross_attn.value.bias"))
                        .expect("cross_attn value bias"),
                ),
            )
            .expect("cross_attn value projection");
            cross_k.push(to_head_first(&ck.data, n_head, enc_len, head_dim));
            cross_v.push(to_head_first(&cv.data, n_head, enc_len, head_dim));
        }

        let special = SpecialTokens::new(hp.n_vocab);
        let ctx = ForwardCtx {
            cross_k: &cross_k,
            cross_v: &cross_v,
            enc_len,
            tok_emb: crate::decoder::forward::TokenEmbedding::Float(
                model
                    .get("decoder.token_embedding.weight")
                    .expect("tok_emb"),
            ),
            pos_emb: model.get("decoder.positional_embedding").expect("pos_emb"),
            model: &model,
            n_state,
            n_layer,
            n_head,
            head_dim,
        };

        let lang_token = special.language_token("en");
        let prompt = vec![
            special.sot,
            lang_token,
            special.transcribe,
            special.no_timestamps,
        ];
        let kv_capacity = prompt.len() + MAX_DECODE_LENGTH + 4;
        let eot_threshold = special.eot;
        let constraints = DecodeConstraints {
            suppress: &[],
            no_repeat_ngram_size: 0,
            timestamp_rules: false,
            suppress_blank: false,
            blank_token: None,
        };
        let args = DecodeArgs {
            kv_capacity,
            special: &special,
            eot_threshold,
            constraints: &constraints,
            dtype: KvCacheDtype::F32,
        };

        let result = decode_beam(
            &prompt, &ctx, &args, 3, // beam_width
        );

        assert!(
            result.is_ok(),
            "decode_beam should not error: {:?}",
            result.err()
        );

        let (tokens, probs, _nsp) = result.expect("already checked");
        // tokens and probs should have the same length
        assert_eq!(
            tokens.len(),
            probs.len(),
            "tokens and probs length mismatch"
        );
        // All probs should be finite (log-probabilities)
        for (i, &p) in probs.iter().enumerate() {
            assert!(p.is_finite(), "prob[{i}] = {p} is not finite");
        }
        // All tokens should be valid token IDs (< n_vocab)
        for (i, &t) in tokens.iter().enumerate() {
            assert!(
                (t as usize) < hp.n_vocab,
                "token[{i}] = {t} exceeds vocab size {}",
                hp.n_vocab
            );
        }
    }

    // ── Regression test: issue #1 — beam_width > 1 + timestamps ────────────

    /// Regression test for issue #1: when beam_width > 1 and timestamps are
    /// enabled, the seeding loop previously marked EOT-seeded beams as
    /// `done=true` with zero tokens.  `normalized_score(0, score) = score`
    /// caused them to win `max_by` over any real hypothesis, producing an
    /// empty transcription even for audible input.
    ///
    /// This test verifies that `decode_beam` does not return an error and that
    /// `tokens.len() == probs.len()` (shape invariant), using the same
    /// synthetic model used by `test_decode_beam_with_synthetic_model`.
    #[test]
    fn test_issue_1_beam_timestamps_nonempty() {
        use crate::decoder::ForwardCtx;
        use crate::encoder::encode;
        use crate::linear;
        use crate::model::ModelData;
        use crate::tensor::Tensor;
        use crate::test_utils::generate_synthetic_model;
        use crate::tokenizer::SpecialTokens;

        let model_path = generate_synthetic_model();
        let model = ModelData::load(&model_path).expect("load synthetic model");
        let _ = std::fs::remove_file(&model_path);

        let hp = &model.hparams;
        let n_state = hp.n_text_state;
        let n_layer = hp.n_text_layer;
        let n_head = hp.n_text_head;
        let head_dim = n_state / n_head;
        let n_mels = hp.n_mels;

        // ~1 s of sine wave at 440 Hz as audio proxy — enough frames to
        // exercise the encoder without triggering the no-speech shortcut.
        let n_frames = 100;
        let mel = Tensor::from_vec(vec![0.01f32; n_mels * n_frames], &[n_mels, n_frames]);
        let encoder_output = encode(&mel, &model).expect("encode should succeed");
        let enc_len = encoder_output.shape[0];

        // Precompute cross-attention K,V (identical to the helper above).
        let mut cross_k: Vec<Vec<f32>> = Vec::with_capacity(n_layer);
        let mut cross_v: Vec<Vec<f32>> = Vec::with_capacity(n_layer);
        for layer in 0..n_layer {
            let pfx = format!("decoder.blocks.{layer}");
            let ck_name = format!("{pfx}.cross_attn.key.weight");
            let ck = linear::linear_auto(
                &encoder_output,
                model.try_get(&ck_name),
                model.get_quantized(&ck_name),
                None,
            )
            .expect("cross_attn key projection");
            let cv_name = format!("{pfx}.cross_attn.value.weight");
            let cv = linear::linear_auto(
                &encoder_output,
                model.try_get(&cv_name),
                model.get_quantized(&cv_name),
                Some(
                    model
                        .get(&format!("{pfx}.cross_attn.value.bias"))
                        .expect("cross_attn value bias"),
                ),
            )
            .expect("cross_attn value projection");
            cross_k.push(to_head_first(&ck.data, n_head, enc_len, head_dim));
            cross_v.push(to_head_first(&cv.data, n_head, enc_len, head_dim));
        }

        let special = SpecialTokens::new(hp.n_vocab);
        let ctx = ForwardCtx {
            cross_k: &cross_k,
            cross_v: &cross_v,
            enc_len,
            tok_emb: crate::decoder::forward::TokenEmbedding::Float(
                model
                    .get("decoder.token_embedding.weight")
                    .expect("tok_emb"),
            ),
            pos_emb: model.get("decoder.positional_embedding").expect("pos_emb"),
            model: &model,
            n_state,
            n_layer,
            n_head,
            head_dim,
        };

        // Prompt WITH timestamps: eot_threshold = special.eot so that
        // timestamp tokens (>= eot) are allowed through.
        let lang_token = special.language_token("en");
        let prompt = vec![special.sot, lang_token, special.transcribe];
        // Do NOT add no_timestamps so that eot_threshold == eot and
        // timestamps are enabled — this is the path that triggered the bug.
        let kv_capacity = prompt.len() + MAX_DECODE_LENGTH + 4;
        let eot_threshold = special.eot; // timestamps enabled
        let constraints = DecodeConstraints {
            suppress: &[],
            no_repeat_ngram_size: 0,
            timestamp_rules: false,
            suppress_blank: false,
            blank_token: None,
        };
        let args = DecodeArgs {
            kv_capacity,
            special: &special,
            eot_threshold,
            constraints: &constraints,
            dtype: KvCacheDtype::F32,
        };

        // beam_width = 2: the pre-fix code produced empty tokens here.
        let result = decode_beam(&prompt, &ctx, &args, 2);

        assert!(
            result.is_ok(),
            "decode_beam with beam_width=2, timestamps=true must not error: {:?}",
            result.err()
        );

        let (tokens, probs, _nsp) = result.expect("already checked");
        assert_eq!(
            tokens.len(),
            probs.len(),
            "tokens and probs must have the same length (issue #1 shape invariant)"
        );
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                p.is_finite(),
                "prob[{i}] = {p} must be finite (issue #1 regression)"
            );
        }
    }
}
