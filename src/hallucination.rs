//! Hallucination detection via character entropy and compression ratio analysis.

use crate::tokenizer;

/// Compute character-level entropy of text as a proxy for compression ratio.
/// Low entropy indicates highly repetitive text (likely hallucination).
/// Returns bits per character.
pub(crate) fn char_entropy(text: &str) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    let mut counts = std::collections::HashMap::new();
    let total = text.chars().count() as f32;
    for ch in text.chars() {
        *counts.entry(ch).or_insert(0u32) += 1;
    }
    let mut entropy = 0.0f32;
    for &count in counts.values() {
        let p = count as f32 / total;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Check if text is likely a hallucination based on character entropy.
/// Returns true if the text has suspiciously low entropy (high repetition).
pub(crate) fn is_likely_hallucination(text: &str, threshold: f32) -> bool {
    if threshold <= 0.0 || text.len() < 10 {
        return false; // Disabled or too short to judge
    }
    let entropy = char_entropy(text);
    // Normal speech text has entropy ~3.5-4.5 bits/char.
    // Hallucinated repetitive text has entropy < 2.0 bits/char.
    // The threshold maps to: entropy < threshold_entropy
    let threshold_entropy = 8.0 / threshold; // compression_ratio 2.4 -> ~3.33 bits threshold
    entropy < threshold_entropy
}

/// Compute per-segment average log-probability from token-level probs.
///
/// Walks through `token_ids` and `token_probs` in parallel, splitting at
/// timestamp tokens to align with the segments returned by `parse_segments`.
/// Returns a vector of average log-probabilities, one per segment.
pub(crate) fn compute_segment_confidences(
    token_ids: &[u32],
    token_probs: &[f32],
    n_segments: usize,
) -> Vec<f32> {
    if token_probs.is_empty() || n_segments == 0 {
        return vec![0.0; n_segments];
    }

    let mut confidences = Vec::with_capacity(n_segments);
    let mut seg_sum = 0.0f32;
    let mut seg_count = 0usize;
    let mut in_segment = false;

    for (i, &id) in token_ids.iter().enumerate() {
        if tokenizer::SpecialTokens::is_timestamp(id) {
            if in_segment {
                // Closing timestamp -- finalize this segment's confidence
                let avg = if seg_count > 0 {
                    seg_sum / seg_count as f32
                } else {
                    0.0
                };
                confidences.push(avg);
                seg_sum = 0.0;
                seg_count = 0;
            }
            // This timestamp may open the next segment
            in_segment = true;
        } else if in_segment {
            // Text token inside a segment
            if let Some(&prob) = token_probs.get(i) {
                seg_sum += prob;
                seg_count += 1;
            }
        }
    }

    // Pad with 0.0 if we got fewer confidences than segments
    while confidences.len() < n_segments {
        confidences.push(0.0);
    }
    confidences.truncate(n_segments);
    confidences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_entropy_uniform() {
        // All same character -> entropy = 0.0
        let entropy = char_entropy("aaaaaaaaaa");
        assert!((entropy - 0.0).abs() < 1e-6, "expected ~0.0, got {entropy}");
    }

    #[test]
    fn test_char_entropy_diverse() {
        // Many different characters -> high entropy
        let text = "abcdefghijklmnopqrstuvwxyz";
        let entropy = char_entropy(text);
        // 26 unique chars -> entropy = log2(26) ~ 4.7
        let expected = (26.0f32).log2();
        assert!(
            (entropy - expected).abs() < 0.01,
            "expected ~{expected}, got {entropy}"
        );
    }

    #[test]
    fn test_char_entropy_empty() {
        assert!((char_entropy("") - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hallucination_detection_repetitive() {
        // Highly repetitive text should be flagged
        let repetitive = "the the the the the the the the the the";
        assert!(
            is_likely_hallucination(repetitive, 2.4),
            "repetitive text should be flagged as hallucination"
        );
    }

    #[test]
    fn test_hallucination_detection_normal() {
        // Normal diverse text should not be flagged
        let normal = "The quick brown fox jumped over the lazy dog near a stream";
        assert!(
            !is_likely_hallucination(normal, 2.4),
            "normal text should not be flagged as hallucination"
        );
    }

    #[test]
    fn test_compression_ratio_disabled() {
        // threshold <= 0.0 should never flag
        let repetitive = "aaaaaaaaaaaaaaaaaa";
        assert!(!is_likely_hallucination(repetitive, 0.0));
        assert!(!is_likely_hallucination(repetitive, -1.0));
    }

    #[test]
    fn test_hallucination_short_text() {
        // Text shorter than 10 chars should never be flagged
        assert!(!is_likely_hallucination("aaa", 2.4));
    }

    #[test]
    fn test_compute_segment_confidences_empty() {
        let result = compute_segment_confidences(&[], &[], 0);
        assert!(result.is_empty());

        let result = compute_segment_confidences(&[], &[], 3);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }
}
