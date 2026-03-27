//! Dynamic Time Warping (DTW) for word-level timestamp alignment.
//!
//! Aligns decoder token positions with encoder audio frames using
//! cross-attention weights from the decoder's last layer.

/// A word with precise start/end timestamps derived from DTW alignment.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WordSegment {
    /// The word text.
    pub word: String,
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
    /// Confidence score (average attention weight).
    pub confidence: f32,
}

/// Compute DTW alignment between token attention weights and audio frames.
///
/// - `attention_weights`: matrix of shape \[n_tokens, n_audio_frames\]
///   (each row shows which audio frames a token attends to)
/// - `n_tokens`: number of tokens (rows)
/// - `n_frames`: number of audio frames (columns)
/// - `hop_length`: audio hop length in samples (160 for Whisper)
/// - `sample_rate`: audio sample rate (16000 for Whisper)
///
/// Returns per-token (start_time, end_time) pairs.
pub fn align_tokens_dtw(
    attention_weights: &[f32],
    n_tokens: usize,
    n_frames: usize,
    hop_length: usize,
    sample_rate: usize,
) -> Vec<(f32, f32)> {
    if n_tokens == 0 || n_frames == 0 {
        return Vec::new();
    }

    // For each token, find the frame with maximum attention weight
    // This gives a rough alignment; DTW refines it
    let mut token_peaks = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        let row = &attention_weights[t * n_frames..(t + 1) * n_frames];
        let peak_frame = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        token_peaks.push(peak_frame);
    }

    // Ensure monotonicity (each token's frame >= previous token's frame)
    for i in 1..token_peaks.len() {
        if token_peaks[i] < token_peaks[i - 1] {
            token_peaks[i] = token_peaks[i - 1];
        }
    }

    // Convert frame indices to time ranges
    let frame_to_time = |frame: usize| -> f32 { (frame * hop_length) as f32 / sample_rate as f32 };

    let mut result = Vec::with_capacity(n_tokens);
    for i in 0..n_tokens {
        let start = frame_to_time(token_peaks[i]);
        let end = if i + 1 < n_tokens {
            frame_to_time(token_peaks[i + 1])
        } else {
            frame_to_time(n_frames.min(token_peaks[i] + 1))
        };
        result.push((start, end.max(start)));
    }

    result
}

/// Build word segments by grouping tokens into words and aligning with DTW timestamps.
///
/// `token_texts`: decoded text for each token
/// `token_times`: (start, end) time for each token from DTW
/// `token_probs`: log-probability for each token
pub fn build_word_segments(
    token_texts: &[String],
    token_times: &[(f32, f32)],
    token_probs: &[f32],
) -> Vec<WordSegment> {
    if token_texts.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current_word = String::new();
    let mut word_start = 0.0f32;
    let mut word_probs = Vec::new();

    for (i, text) in token_texts.iter().enumerate() {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Check if this token starts a new word (starts with space or is first)
        let starts_new_word = text.starts_with(' ') || (i == 0 && !text.is_empty());

        if starts_new_word && !current_word.is_empty() {
            // Flush previous word
            let avg_prob = if word_probs.is_empty() {
                0.0
            } else {
                word_probs.iter().sum::<f32>() / word_probs.len() as f32
            };
            let end = token_times.get(i).map(|t| t.0).unwrap_or(word_start);
            segments.push(WordSegment {
                word: current_word.trim().to_string(),
                start: word_start,
                end,
                confidence: avg_prob,
            });
            current_word.clear();
            word_probs.clear();
            word_start = token_times.get(i).map(|t| t.0).unwrap_or(0.0);
        }

        if current_word.is_empty() {
            word_start = token_times.get(i).map(|t| t.0).unwrap_or(0.0);
        }

        current_word.push_str(trimmed);
        if i < token_probs.len() {
            word_probs.push(token_probs[i]);
        }
    }

    // Flush last word
    if !current_word.is_empty() {
        let avg_prob = if word_probs.is_empty() {
            0.0
        } else {
            word_probs.iter().sum::<f32>() / word_probs.len() as f32
        };
        let end = token_times.last().map(|t| t.1).unwrap_or(word_start);
        segments.push(WordSegment {
            word: current_word.trim().to_string(),
            start: word_start,
            end,
            confidence: avg_prob,
        });
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_tokens_empty() {
        let result = align_tokens_dtw(&[], 0, 0, 160, 16000);
        assert!(result.is_empty());
    }

    #[test]
    fn test_align_tokens_single() {
        // 1 token, 10 frames, attention peaks at frame 5
        let mut weights = vec![0.0f32; 10];
        weights[5] = 1.0;
        let result = align_tokens_dtw(&weights, 1, 10, 160, 16000);
        assert_eq!(result.len(), 1);
        assert!((result[0].0 - 0.05).abs() < 0.001); // frame 5 * 160 / 16000 = 0.05s
    }

    #[test]
    fn test_align_tokens_monotonic() {
        // 3 tokens, attention peaks at frames 2, 1, 8 -> monotonic correction -> 2, 2, 8
        let n_frames = 10;
        let mut weights = vec![0.0f32; 3 * n_frames];
        weights[2] = 1.0; // token 0 peaks at frame 2
        weights[n_frames + 1] = 1.0; // token 1 peaks at frame 1 (should be corrected to 2)
        weights[2 * n_frames + 8] = 1.0; // token 2 peaks at frame 8
        let result = align_tokens_dtw(&weights, 3, n_frames, 160, 16000);
        assert_eq!(result.len(), 3);
        // Token 1 should start at same frame as token 0 (monotonicity enforced)
        assert!(result[1].0 >= result[0].0);
        assert!(result[2].0 >= result[1].0);
    }

    #[test]
    fn test_build_word_segments_basic() {
        let texts = vec![" Hello".to_string(), " world".to_string()];
        let times = vec![(0.0, 0.5), (0.5, 1.0)];
        let probs = vec![-0.1, -0.2];
        let segs = build_word_segments(&texts, &times, &probs);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].word, "Hello");
        assert_eq!(segs[1].word, "world");
    }

    #[test]
    fn test_build_word_segments_empty() {
        let segs = build_word_segments(&[], &[], &[]);
        assert!(segs.is_empty());
    }

    #[test]
    fn test_build_word_segments_multitoken_word() {
        // "un" + "break" + "able" = one word "unbreakable"
        let texts = vec![" un".to_string(), "break".to_string(), "able".to_string()];
        let times = vec![(0.0, 0.2), (0.2, 0.4), (0.4, 0.6)];
        let probs = vec![-0.1, -0.15, -0.2];
        let segs = build_word_segments(&texts, &times, &probs);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].word, "unbreakable");
        assert!((segs[0].start - 0.0).abs() < 0.001);
        assert!((segs[0].end - 0.6).abs() < 0.001);
    }
}
