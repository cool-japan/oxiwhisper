//! Helper functions for Whisper decoder: logit manipulation, sampling, n-gram blocking.

use rand::RngExt;

/// Extra decode-time constraints passed through to greedy/beam/sample decoders.
pub(crate) struct DecodeConstraints<'a> {
    /// Token IDs whose logits are forced to negative infinity.
    pub(crate) suppress: &'a [u32],
    /// N-gram size for repetition blocking (0 = disabled).
    pub(crate) no_repeat_ngram_size: usize,
}

/// Shared arguments for greedy / beam / sample decoders.
///
/// Bundles the parameters that are identical across all three decoding
/// strategies so each decode function signature stays under the clippy
/// `too_many_arguments` limit.
pub(crate) struct DecodeArgs<'a> {
    /// Number of KV-cache slots to pre-allocate (prompt length + max decode steps + slack).
    pub(crate) kv_capacity: usize,
    /// Special token identifiers (EOT, no-speech, …).
    pub(crate) special: &'a crate::tokenizer::SpecialTokens,
    /// Highest token id that acts as a stop signal (= `special.eot` when
    /// timestamps are disabled).
    pub(crate) eot_threshold: u32,
    /// Logit constraints (suppression list, n-gram blocking size).
    pub(crate) constraints: &'a DecodeConstraints<'a>,
    /// Numeric format for the self-attention KV cache.
    pub(crate) dtype: crate::types::KvCacheDtype,
}

/// Argmax over a slice, returning the index of the largest element.
pub(crate) fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Argmax with its log-probability (numerically stable).
pub(crate) fn argmax_with_log_prob(logits: &[f32]) -> (u32, f32) {
    let idx = argmax(logits);
    let lp = token_log_prob(logits, idx);
    (idx, lp)
}

/// Compute the log-probability of a specific token given logits.
pub(crate) fn token_log_prob(logits: &[f32], token: u32) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum = max + sum.ln();
    logits
        .get(token as usize)
        .copied()
        .unwrap_or(f32::NEG_INFINITY)
        - log_sum
}

/// Numerically stable log-softmax.
pub(crate) fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum = max + sum.ln();
    logits.iter().map(|&x| x - log_sum).collect()
}

/// Returns up to `k` (log_prob, token_id) pairs sorted by log_prob descending.
pub(crate) fn top_k_log_probs(log_probs: &[f32], k: usize) -> Vec<(f32, u32)> {
    let mut pairs: Vec<(f32, u32)> = log_probs
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();
    pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(k);
    pairs
}

/// Sample a token from logits with temperature scaling, top-k and top-p filtering.
///
/// When `temperature` is close to 0 this degenerates toward argmax.
pub(crate) fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut impl rand::Rng,
) -> u32 {
    // Temperature-scaled softmax.
    let t = temperature.max(1e-6);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(f32, u32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| (((x - max) / t).exp(), i as u32))
        .collect();
    // Sort descending by probability.
    probs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k filter.
    if top_k > 0 && top_k < probs.len() {
        probs.truncate(top_k);
    }

    // Top-p (nucleus) filter.
    if top_p < 1.0 {
        let total: f32 = probs.iter().map(|&(p, _)| p).sum();
        let mut cumsum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &(p, _)) in probs.iter().enumerate() {
            cumsum += p / total;
            if cumsum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
    }

    // Sample from the remaining distribution.
    let total: f32 = probs.iter().map(|&(p, _)| p).sum();
    let mut r = rng.random::<f32>() * total;
    for &(p, tok) in &probs {
        r -= p;
        if r <= 0.0 {
            return tok;
        }
    }
    probs.last().map(|&(_, t)| t).unwrap_or(0)
}

/// Compute a length-normalised score for beam ranking.
pub(crate) fn normalized_score(tokens_len: usize, score: f32) -> f32 {
    let len = tokens_len.max(1) as f32;
    score / len.powf(0.6)
}

/// Collect all n-grams of the given size from a token sequence.
pub(crate) fn collect_ngrams(tokens: &[u32], n: usize) -> Vec<Vec<u32>> {
    if tokens.len() < n || n == 0 {
        return Vec::new();
    }
    tokens.windows(n).map(|w| w.to_vec()).collect()
}

/// Apply no-repeat-ngram blocking: if appending a token would create an
/// n-gram that already appears in the output, set its logit to NEG_INFINITY.
pub(crate) fn apply_no_repeat_ngram(logits: &mut [f32], tokens: &[u32], ngram_size: usize) {
    if ngram_size == 0 || tokens.len() + 1 < ngram_size {
        return;
    }
    // The last (ngram_size - 1) tokens form the prefix of the potential new n-gram.
    let prefix = &tokens[tokens.len().saturating_sub(ngram_size - 1)..];
    // Collect existing n-grams.
    let existing = collect_ngrams(tokens, ngram_size);
    // For each possible next token, check if prefix + token forms a repeated n-gram.
    for (tok_id, logit) in logits.iter_mut().enumerate() {
        let mut candidate: Vec<u32> = prefix.to_vec();
        candidate.push(tok_id as u32);
        if existing.contains(&candidate) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Apply token suppression by setting logits to negative infinity.
pub(crate) fn apply_suppress_tokens(logits: &mut [f32], suppress: &[u32]) {
    for &tok in suppress {
        if (tok as usize) < logits.len() {
            logits[tok as usize] = f32::NEG_INFINITY;
        }
    }
}

/// Determine if a token should stop decoding.
///
/// When timestamps are disabled (`eot_threshold == special.eot`), any token >= EOT
/// is treated as a stop token (the original behaviour).  When timestamps are enabled,
/// only the exact EOT token stops decoding -- timestamp tokens (which are >= EOT) are
/// allowed through so they appear in the output.
pub(crate) fn is_stop_token(
    token: u32,
    eot_threshold: u32,
    special: &crate::tokenizer::SpecialTokens,
) -> bool {
    if token == special.eot {
        return true;
    }
    // When timestamps are disabled, eot_threshold == special.eot and the check above
    // already catches it. The >= check below handles any other special tokens above EOT
    // (no_speech, no_timestamps, etc.) that are NOT timestamp tokens.
    if token >= special.eot && !crate::tokenizer::SpecialTokens::is_timestamp(token) {
        return true;
    }
    // When timestamps are enabled, timestamp tokens pass through.
    // When timestamps are disabled, timestamp tokens also pass through here, but they
    // should never actually be emitted by the model because no_timestamps is in the prompt.
    let _ = eot_threshold; // used conceptually; kept as parameter for future refinement
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_softmax_sums_to_one() {
        let v = [1.0f32, 2.0, 3.0];
        let lp = log_softmax(&v);
        let sum: f32 = lp.iter().map(|x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(lp[2] > lp[1] && lp[1] > lp[0]);
    }

    #[test]
    fn test_log_softmax_uniform() {
        let v = [0.0f32; 4];
        let lp = log_softmax(&v);
        let expected = -4.0f32.ln();
        for &x in &lp {
            assert!((x - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_top_k_log_probs_ordering() {
        let lp = log_softmax(&[1.0f32, 3.0, 2.0]);
        let top2 = top_k_log_probs(&lp, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].1, 1); // token 1 has highest logit
        assert_eq!(top2[1].1, 2); // token 2 is second
        assert!(top2[0].0 > top2[1].0);
    }

    #[test]
    fn test_top_k_truncation() {
        let lp = log_softmax(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let top3 = top_k_log_probs(&lp, 3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].1, 4); // highest logit at index 4
    }

    #[test]
    fn test_top_k_clamps_to_vocab_size() {
        let lp = log_softmax(&[1.0f32, 2.0]);
        let top5 = top_k_log_probs(&lp, 5); // k > vocab size
        assert_eq!(top5.len(), 2);
    }

    #[test]
    fn test_sample_token_returns_valid_index() {
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut rng = rand::rng();
        for _ in 0..50 {
            let tok = sample_token(&logits, 1.0, 0, 1.0, &mut rng);
            assert!((tok as usize) < logits.len());
        }
    }

    #[test]
    fn test_sample_token_top_k_restricts_range() {
        // With top_k=1 we always get the argmax token.
        let logits = vec![1.0f32, 2.0, 100.0, 1.0]; // token 2 dominates
        let mut rng = rand::rng();
        for _ in 0..20 {
            let tok = sample_token(&logits, 1.0, 1, 1.0, &mut rng);
            assert_eq!(tok, 2);
        }
    }

    #[test]
    fn test_sample_token_low_temperature_picks_best() {
        // Very low temperature -> near-deterministic argmax behaviour.
        let logits = vec![1.0f32, 2.0, 50.0, 1.0];
        let mut rng = rand::rng();
        for _ in 0..20 {
            let tok = sample_token(&logits, 0.01, 0, 1.0, &mut rng);
            assert_eq!(tok, 2);
        }
    }

    #[test]
    fn test_sample_token_top_p_nucleus() {
        // With top_p very small only the highest-probability token survives.
        let logits = vec![1.0f32, 2.0, 100.0, 1.0]; // token 2 ~ prob 1.0 after softmax
        let mut rng = rand::rng();
        for _ in 0..20 {
            let tok = sample_token(&logits, 1.0, 0, 0.01, &mut rng);
            assert_eq!(tok, 2);
        }
    }

    // ── apply_suppress_tokens tests ───────────────────────────────────────

    #[test]
    fn test_suppress_tokens_sets_neg_infinity() {
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        apply_suppress_tokens(&mut logits, &[1, 3]);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert_eq!(logits[4], 5.0);
    }

    #[test]
    fn test_suppress_tokens_out_of_range_ignored() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        apply_suppress_tokens(&mut logits, &[100]);
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_suppress_tokens_empty() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        apply_suppress_tokens(&mut logits, &[]);
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    // ── No-repeat-ngram tests ─────────────────────────────────────────────

    #[test]
    fn test_collect_ngrams() {
        let tokens = vec![1, 2, 3, 4, 5];
        let ngrams = collect_ngrams(&tokens, 3);
        assert_eq!(ngrams.len(), 3);
        assert_eq!(ngrams[0], vec![1, 2, 3]);
        assert_eq!(ngrams[1], vec![2, 3, 4]);
        assert_eq!(ngrams[2], vec![3, 4, 5]);
    }

    #[test]
    fn test_collect_ngrams_too_short() {
        let tokens = vec![1, 2];
        let ngrams = collect_ngrams(&tokens, 3);
        assert!(ngrams.is_empty());
    }

    #[test]
    fn test_collect_ngrams_disabled() {
        let tokens = vec![1, 2, 3];
        let ngrams = collect_ngrams(&tokens, 0);
        assert!(ngrams.is_empty());
    }

    #[test]
    fn test_no_repeat_ngram_blocks_repetition() {
        let tokens = vec![1u32, 2, 3, 1, 2];
        let mut logits = vec![0.0f32; 5];
        apply_no_repeat_ngram(&mut logits, &tokens, 3);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert_eq!(logits[0], 0.0);
        assert_eq!(logits[1], 0.0);
        assert_eq!(logits[2], 0.0);
        assert_eq!(logits[4], 0.0);
    }

    #[test]
    fn test_no_repeat_ngram_disabled() {
        let tokens = vec![1u32, 2, 3, 1, 2];
        let mut logits = vec![1.0f32; 5];
        let original = logits.clone();
        apply_no_repeat_ngram(&mut logits, &tokens, 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_no_repeat_ngram_too_short() {
        let tokens = vec![1u32];
        let mut logits = vec![1.0f32; 5];
        let original = logits.clone();
        apply_no_repeat_ngram(&mut logits, &tokens, 3);
        assert_eq!(logits, original);
    }

    // ── log_softmax additional tests ────────────────────────────────────────

    #[test]
    fn test_log_softmax_numerical_stability() {
        // Very large inputs should not produce NaN thanks to max-subtraction.
        // Use values that differ enough to be distinguishable in f32.
        let v = [1000.0_f32, 999.0, 998.0];
        let lp = log_softmax(&v);
        for (i, &x) in lp.iter().enumerate() {
            assert!(
                x.is_finite(),
                "log_softmax should be numerically stable, but index {i} = {x}"
            );
        }
        // Sum of exp(log_probs) should still be ~1.0
        let sum: f32 = lp.iter().map(|x| x.exp()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "exp of log_softmax should sum to ~1.0, got {sum}"
        );

        // Also test with extreme uniform values — all identical very large numbers
        let v_extreme = [1e38_f32; 4];
        let lp_extreme = log_softmax(&v_extreme);
        for (i, &x) in lp_extreme.iter().enumerate() {
            assert!(
                x.is_finite(),
                "log_softmax with extreme uniform inputs: index {i} = {x}"
            );
        }
    }

    #[test]
    fn test_log_softmax_uniform_outputs_log_one_over_n() {
        // Equal inputs produce equal outputs = log(1/n)
        let n = 7;
        let v = vec![5.0_f32; n];
        let lp = log_softmax(&v);
        let expected = -(n as f32).ln();
        for (i, &x) in lp.iter().enumerate() {
            assert!(
                (x - expected).abs() < 1e-5,
                "uniform input: index {i} = {x}, expected {expected}"
            );
        }
    }

    // ── top_k_log_probs additional tests ────────────────────────────────────

    #[test]
    fn test_top_k_log_probs_correct_order() {
        // Verify top-k returns highest values first, in descending order
        let lp = vec![-3.0, -1.0, -5.0, -0.5, -2.0];
        let top3 = top_k_log_probs(&lp, 3);
        assert_eq!(top3.len(), 3);
        // Highest: -0.5 (idx 3), then -1.0 (idx 1), then -2.0 (idx 4)
        assert_eq!(top3[0].1, 3);
        assert_eq!(top3[1].1, 1);
        assert_eq!(top3[2].1, 4);
        // Values must be descending
        assert!(top3[0].0 >= top3[1].0);
        assert!(top3[1].0 >= top3[2].0);
    }

    #[test]
    fn test_top_k_larger_than_vocab() {
        // k > logits.len() should return all elements
        let lp = vec![-1.0, -2.0, -3.0];
        let top10 = top_k_log_probs(&lp, 10);
        assert_eq!(top10.len(), 3, "should return all when k > len");
        // Should still be sorted descending
        assert!(top10[0].0 >= top10[1].0);
        assert!(top10[1].0 >= top10[2].0);
    }

    // ── argmax_with_log_prob tests ──────────────────────────────────────────

    #[test]
    fn test_argmax_with_log_prob_single_element() {
        let logits = vec![42.0_f32];
        let (idx, lp) = argmax_with_log_prob(&logits);
        assert_eq!(idx, 0, "single element should return index 0");
        // log_softmax of a single element is log(1.0) = 0.0
        assert!(
            lp.abs() < 1e-5,
            "single element log_prob should be ~0.0, got {lp}"
        );
    }

    #[test]
    fn test_argmax_with_log_prob_picks_max() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let (idx, lp) = argmax_with_log_prob(&logits);
        assert_eq!(idx, 1, "should pick index of max logit");
        assert!(lp < 0.0, "log_prob should be negative");
        assert!(lp.is_finite(), "log_prob should be finite");
    }

    // ── token_log_prob tests ────────────────────────────────────────────────

    #[test]
    fn test_token_log_prob_out_of_range() {
        let logits = vec![1.0, 2.0, 3.0];
        let lp = token_log_prob(&logits, 100); // token ID way beyond logits length
        assert_eq!(
            lp,
            f32::NEG_INFINITY,
            "out-of-range token should return NEG_INFINITY"
        );
    }

    #[test]
    fn test_token_log_prob_valid_range() {
        let logits = vec![1.0, 2.0, 3.0];
        let lp = token_log_prob(&logits, 2); // token with highest logit
        assert!(lp.is_finite(), "valid token log_prob should be finite");
        assert!(lp < 0.0, "log_prob should be negative (got {lp})");
        // Token 2 has highest logit, so its log_prob should be the highest
        let lp0 = token_log_prob(&logits, 0);
        let lp1 = token_log_prob(&logits, 1);
        assert!(lp > lp1, "token 2 should have higher log_prob than token 1");
        assert!(
            lp1 > lp0,
            "token 1 should have higher log_prob than token 0"
        );
    }
}
