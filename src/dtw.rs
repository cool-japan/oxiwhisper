//! Token alignment algorithms for word-level timestamps.
//!
//! Two algorithms are provided:
//!
//! - `align_tokens_dp_dtw` — canonical DP-DTW with Sakoe-Chiba band constraint and
//!   full traceback. Each token row is softmax-normalised, converted to a negative
//!   log-probability cost, and the accumulated-cost matrix is filled with a two-row
//!   rolling buffer. A full predecessor matrix enables exact traceback from
//!   `(n_tokens-1, n_frames-1)` back to `(0, 0)`, yielding per-token frame ranges.
//!
//! - `align_tokens_monotonic_peak` — fast-path alternative. Finds the argmax frame
//!   per token row, then enforces a non-decreasing sequence via a single forward pass
//!   (monotonic clamp). No DP, no traceback. Useful when speed dominates accuracy.
//!
//! `align_tokens_dtw` is an alias for `align_tokens_dp_dtw` with default band width.
//!
//! # Determinism
//! Identical inputs (bit-exact `f32` attention matrices) always produce identical
//! outputs. The argmax tie-break is documented in `align_tokens_monotonic_peak`.
//! DTW tie-breaks prefer diagonal over up over left (also deterministic).
//!
//! # Confidence
//! `WordSegment::confidence` is the **mean log-probability** (≤ 0.0) of the
//! tokens in the segment. Values closer to 0.0 indicate higher confidence.

/// A word with precise start/end timestamps derived from monotonic-peak alignment.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WordSegment {
    /// The word text.
    pub word: String,
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
    /// Mean log-probability (≤ 0.0) of the tokens in this segment.
    /// Values closer to 0.0 indicate higher confidence.
    pub confidence: f32,
}

/// Compute monotonic-peak alignment between token attention weights and audio frames.
///
/// For each token, the frame with maximum cross-attention weight is chosen as the
/// peak frame. Peaks are then clamped forward so the resulting sequence is
/// non-decreasing (monotonic), and each peak is converted to a time range.
///
/// **Argmax tie-break**: when multiple frames share the same maximum weight,
/// `Iterator::max_by` returns the *last* such frame. This is a stable,
/// documented choice; tests that depend on tie-break behavior should be updated
/// if the implementation changes rather than silently broken.
///
/// - `attention_weights`: flat matrix of shape `[n_tokens * n_frames]` in
///   row-major order (each row of `n_frames` values corresponds to one token)
/// - `n_tokens`: number of tokens (rows)
/// - `n_frames`: number of audio frames (columns)
/// - `hop_length`: audio hop length in samples (160 for Whisper)
/// - `sample_rate`: audio sample rate (16000 for Whisper)
///
/// Returns per-token `(start_time, end_time)` pairs in seconds.
pub fn align_tokens_monotonic_peak(
    attention_weights: &[f32],
    n_tokens: usize,
    n_frames: usize,
    hop_length: usize,
    sample_rate: usize,
) -> Vec<(f32, f32)> {
    if n_tokens == 0 || n_frames == 0 {
        return Vec::new();
    }

    // For each token, find the frame with maximum attention weight.
    // max_by returns the last element when values are equal (documented tie-break).
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

    // Ensure monotonicity (each token's frame >= previous token's frame).
    for i in 1..token_peaks.len() {
        if token_peaks[i] < token_peaks[i - 1] {
            token_peaks[i] = token_peaks[i - 1];
        }
    }

    // Convert frame indices to time ranges.
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

/// True DP-DTW alignment with Sakoe-Chiba band constraint and full traceback.
///
/// For each token row the attention weights are softmax-normalised (row-max
/// subtracted before exp for numerical stability) and converted to a negative
/// log-probability cost clamped to `[0.0, 50.0]`. The accumulated-cost matrix
/// is then filled using a two-row rolling buffer, keeping only O(n_frames) memory
/// for the DP values while recording the full `n_tokens × n_frames` predecessor
/// matrix for exact traceback.
///
/// The Sakoe-Chiba band is expressed with integer arithmetic:
/// cell `(i, j)` is in-band when `|j·n_tokens − i·n_frames| ≤ bw·n_tokens`.
///
/// After DP, the path is traced back from `(n_tokens-1, n_frames-1)` to `(0,0)`.
/// For each token the minimum and maximum visited frame index are recorded and
/// converted to `(start_sec, end_sec)` pairs.
///
/// # Arguments
/// - `attention_weights` — flat row-major matrix of shape `[n_tokens × n_frames]`.
/// - `n_tokens` — number of rows (decode steps / tokens).
/// - `n_frames` — number of columns (audio frames).
/// - `hop_length` — audio hop in samples (160 for Whisper).
/// - `sample_rate` — audio sample rate in Hz (16000 for Whisper).
/// - `band_width` — Sakoe-Chiba band half-width.
///   `None` or `Some(0)` → `max(n_frames/4, 10)`.
///
/// # Fallbacks
/// - If `n_tokens == 0 || n_frames == 0`: returns an empty `Vec`.
/// - If `n_tokens > n_frames` (overconstrained): falls back to
///   [`align_tokens_monotonic_peak`].
/// - If the initial band makes the path infeasible the band is widened to
///   `n_frames` (unconstrained DTW) and the DP is retried once.
pub fn align_tokens_dp_dtw(
    attention_weights: &[f32],
    n_tokens: usize,
    n_frames: usize,
    hop_length: usize,
    sample_rate: usize,
    band_width: Option<usize>,
) -> Vec<(f32, f32)> {
    // ── edge cases ───────────────────────────────────────────────────────────
    if n_tokens == 0 || n_frames == 0 {
        return Vec::new();
    }
    if n_tokens > n_frames {
        return align_tokens_monotonic_peak(
            attention_weights,
            n_tokens,
            n_frames,
            hop_length,
            sample_rate,
        );
    }

    // ── band width ───────────────────────────────────────────────────────────
    let default_bw = (n_frames / 4).max(10);
    let bw = match band_width {
        Some(0) | None => default_bw,
        Some(w) => w,
    };

    // ── softmax-normalise each row ───────────────────────────────────────────
    let mut prob = vec![0.0f32; n_tokens * n_frames];
    for i in 0..n_tokens {
        let row_start = i * n_frames;
        let row_end = row_start + n_frames;
        let row = &attention_weights[row_start..row_end];

        // Subtract row-max before exp for numerical stability.
        let row_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let row_max = if row_max.is_finite() { row_max } else { 0.0 };

        let mut sum = 0.0f32;
        for (k, &w) in row.iter().enumerate() {
            let e = (w - row_max).exp();
            prob[row_start + k] = e;
            sum += e;
        }
        let sum = if sum > 0.0 { sum } else { 1.0 };
        for k in 0..n_frames {
            prob[row_start + k] /= sum;
        }
    }

    // ── local cost: -ln(p), clamped to [0, 50] ───────────────────────────────
    let mut cost = vec![0.0f32; n_tokens * n_frames];
    for idx in 0..cost.len() {
        cost[idx] = (-(prob[idx].max(1e-22f32).ln())).min(50.0f32);
    }

    // ── run DP; retry once with full band if infeasible ──────────────────────
    let result = run_dp_dtw(&cost, n_tokens, n_frames, bw);
    let (pred, last_row) = if last_row_is_finite(&result.1, n_frames) {
        result
    } else if bw < n_frames {
        // Auto-widen: retry with full unconstrained band.
        run_dp_dtw(&cost, n_tokens, n_frames, n_frames)
    } else {
        result
    };

    // ── check if still infeasible after retry ────────────────────────────────
    if !last_row_is_finite(&last_row, n_frames) {
        return align_tokens_monotonic_peak(
            attention_weights,
            n_tokens,
            n_frames,
            hop_length,
            sample_rate,
        );
    }

    // ── traceback ────────────────────────────────────────────────────────────
    // per-token min/max visited frame
    let mut min_frame = vec![n_frames; n_tokens];
    let mut max_frame = vec![0usize; n_tokens];

    let mut i = n_tokens - 1;
    let mut j = n_frames - 1;
    loop {
        // Record this cell.
        if j < min_frame[i] {
            min_frame[i] = j;
        }
        if j > max_frame[i] {
            max_frame[i] = j;
        }

        if i == 0 && j == 0 {
            break;
        }

        if i == 0 {
            // We're in row 0: the path must reach (0, 0) by following leftward.
            // Continue recording frames for token 0 until j == 0.
            j -= 1;
            continue;
        }

        let p = pred[i * n_frames + j];
        match p {
            0 => {
                // diagonal: i-1, j-1
                if j == 0 {
                    // Can't go diag from j=0; fall back to up.
                    i -= 1;
                } else {
                    i -= 1;
                    j -= 1;
                }
            }
            1 => {
                // up: i-1, j
                i -= 1;
            }
            _ => {
                // left: i, j-1
                if j == 0 {
                    // Can't go left from j=0 — boundary guard.
                    break;
                }
                j -= 1;
            }
        }
    }

    // ── convert to timestamps ────────────────────────────────────────────────
    let frame_to_time = |frame: usize| -> f32 { (frame * hop_length) as f32 / sample_rate as f32 };

    let mut result_times = Vec::with_capacity(n_tokens);
    for i in 0..n_tokens {
        // Guard: if a token has no frames assigned, reuse previous end.
        let mf = if min_frame[i] < n_frames {
            min_frame[i]
        } else {
            0
        };
        let xf = if max_frame[i] < n_frames {
            max_frame[i]
        } else {
            mf
        };
        let start_sec = frame_to_time(mf);
        let end_sec = frame_to_time(xf + 1);
        result_times.push((start_sec, end_sec.max(start_sec)));
    }

    result_times
}

// ── internal DP helpers ──────────────────────────────────────────────────────

/// Returns `true` if `row[n_frames - 1]` is finite.
fn last_row_is_finite(last_row: &[f32], n_frames: usize) -> bool {
    last_row
        .get(n_frames.saturating_sub(1))
        .map(|v| v.is_finite())
        .unwrap_or(false)
}

/// Sakoe-Chiba in-band predicate using integer arithmetic.
///
/// Cell `(i, j)` is in-band when `|j·n_tokens − i·n_frames| ≤ bw·n_tokens`.
#[inline]
fn in_band(i: usize, j: usize, n_tokens: usize, n_frames: usize, bw: usize) -> bool {
    let lhs = j * n_tokens;
    let rhs = i * n_frames;
    lhs.abs_diff(rhs) <= bw * n_tokens
}

/// Run the DP-DTW algorithm with the given band width.
///
/// Returns `(pred, last_row)` where `pred` is the `n_tokens × n_frames`
/// predecessor matrix (0=diag, 1=up, 2=left) and `last_row` contains the
/// accumulated costs for row `n_tokens - 1`.
fn run_dp_dtw(cost: &[f32], n_tokens: usize, n_frames: usize, bw: usize) -> (Vec<u8>, Vec<f32>) {
    let mut prev_row = vec![f32::INFINITY; n_frames];
    let mut curr_row = vec![f32::INFINITY; n_frames];
    let mut pred = vec![0u8; n_tokens * n_frames];

    // ── initialise row 0 ────────────────────────────────────────────────────
    for j in 0..n_frames {
        if in_band(0, j, n_tokens, n_frames, bw) {
            prev_row[j] = cost[j]; // C[0][j] = local cost; no predecessor needed
        }
        // cells out of band remain f32::INFINITY
    }
    // For row 0, "left" predecessors are valid within the band.
    // Accumulate along the first row (only left moves are possible from (0,0)).
    for j in 1..n_frames {
        if in_band(0, j, n_tokens, n_frames, bw) && prev_row[j - 1].is_finite() {
            // Allow left-move along first row.
            let via_left = prev_row[j - 1] + cost[j];
            if via_left < prev_row[j] {
                prev_row[j] = via_left;
                pred[j] = 2; // left
            }
        }
    }

    // ── fill rows 1..n_tokens ────────────────────────────────────────────────
    for i in 1..n_tokens {
        for elem in curr_row.iter_mut() {
            *elem = f32::INFINITY;
        }

        for j in 0..n_frames {
            if !in_band(i, j, n_tokens, n_frames, bw) {
                continue;
            }

            let cost_ij = cost[i * n_frames + j];

            let diag = if j > 0 {
                prev_row[j - 1]
            } else {
                f32::INFINITY
            };
            let up = prev_row[j];
            let left = if j > 0 {
                curr_row[j - 1]
            } else {
                f32::INFINITY
            };

            // Tie-break: prefer diag, then up, then left.
            let (min_val, direction) = if diag <= up && diag <= left {
                (diag, 0u8)
            } else if up <= left {
                (up, 1u8)
            } else {
                (left, 2u8)
            };

            if min_val.is_finite() {
                curr_row[j] = cost_ij + min_val;
                pred[i * n_frames + j] = direction;
            }
            // else: remains f32::INFINITY (all predecessors out of band or infinite)
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    (pred, prev_row)
}

/// DP-DTW alias with default Sakoe-Chiba band.
///
/// This is now a real Sakoe-Chiba-banded DP-DTW implementation backed by
/// [`align_tokens_dp_dtw`]. The band width defaults to `max(n_frames/4, 10)`.
pub fn align_tokens_dtw(
    attention_weights: &[f32],
    n_tokens: usize,
    n_frames: usize,
    hop_length: usize,
    sample_rate: usize,
) -> Vec<(f32, f32)> {
    align_tokens_dp_dtw(
        attention_weights,
        n_tokens,
        n_frames,
        hop_length,
        sample_rate,
        None,
    )
}

/// Build word segments by grouping tokens into words and aligning with monotonic-peak timestamps.
///
/// `token_texts`: decoded text for each token
/// `token_times`: (start, end) time for each token from monotonic-peak alignment
/// `token_probs`: log-probability for each token
///
/// Convenience wrapper over [`build_word_segments_bytes`] for callers that
/// already hold valid UTF-8 per token. Whisper's byte-level BPE frequently
/// splits a character across tokens, so the inference pipeline uses the byte
/// variant instead.
pub fn build_word_segments(
    token_texts: &[String],
    token_times: &[(f32, f32)],
    token_probs: &[f32],
) -> Vec<WordSegment> {
    let bytes: Vec<Vec<u8>> = token_texts.iter().map(|t| t.as_bytes().to_vec()).collect();
    build_word_segments_bytes(&bytes, token_times, token_probs)
}

/// `true` when `b` is the first byte of a UTF-8 scalar value (i.e. not a
/// `10xxxxxx` continuation byte).
#[inline]
fn is_utf8_char_start(b: u8) -> bool {
    (b & 0xC0) != 0x80
}

/// Trim leading/trailing ASCII whitespace from a byte slice.
#[inline]
fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map(|i| i + 1)
        .unwrap_or(start);
    &bytes[start..end]
}

/// Build word segments from **raw per-token bytes**.
///
/// Two segmentation modes are used, selected automatically:
///
/// * **Space-delimited scripts** (any token starts with an ASCII space):
///   a new word starts at every leading-space token, matching Whisper's
///   sentencepiece-style spacing.
/// * **Scripts without inter-word spaces** (Japanese, Chinese, Thai, …): no
///   token ever carries a leading space, so grouping on spaces alone would
///   collapse the whole utterance into a single "word". In that case a new
///   word starts at every *character* boundary — a token may only open a word
///   when the bytes accumulated so far form complete UTF-8 and the token
///   itself begins a new scalar value. That keeps multi-token characters
///   (`渋` = tokens 162/116/233) intact while still producing one word per
///   character.
pub fn build_word_segments_bytes(
    token_bytes: &[Vec<u8>],
    token_times: &[(f32, f32)],
    token_probs: &[f32],
) -> Vec<WordSegment> {
    if token_bytes.is_empty() {
        return Vec::new();
    }

    let space_delimited = token_bytes.iter().any(|t| t.starts_with(b" "));

    let mut segments = Vec::new();
    let mut current_word: Vec<u8> = Vec::new();
    let mut word_start = 0.0f32;
    let mut word_probs: Vec<f32> = Vec::new();

    let flush = |word: &mut Vec<u8>,
                 probs: &mut Vec<f32>,
                 start: f32,
                 end: f32,
                 out: &mut Vec<WordSegment>| {
        if word.is_empty() {
            return;
        }
        let avg_prob = if probs.is_empty() {
            0.0
        } else {
            probs.iter().sum::<f32>() / probs.len() as f32
        };
        out.push(WordSegment {
            word: String::from_utf8_lossy(trim_ascii(word)).into_owned(),
            start,
            end,
            confidence: avg_prob,
        });
        word.clear();
        probs.clear();
    };

    for (i, raw) in token_bytes.iter().enumerate() {
        let trimmed = trim_ascii(raw);
        if trimmed.is_empty() {
            continue;
        }

        let starts_new_word = raw.starts_with(b" ")
            || (!space_delimited
                && is_utf8_char_start(trimmed[0])
                && std::str::from_utf8(&current_word).is_ok());

        if starts_new_word && !current_word.is_empty() {
            let end = token_times.get(i).map(|t| t.0).unwrap_or(word_start);
            flush(
                &mut current_word,
                &mut word_probs,
                word_start,
                end,
                &mut segments,
            );
            word_start = token_times.get(i).map(|t| t.0).unwrap_or(0.0);
        }

        if current_word.is_empty() {
            word_start = token_times.get(i).map(|t| t.0).unwrap_or(0.0);
        }

        current_word.extend_from_slice(trimmed);
        if i < token_probs.len() {
            word_probs.push(token_probs[i]);
        }
    }

    // Flush last word.
    let end = token_times.last().map(|t| t.1).unwrap_or(word_start);
    flush(
        &mut current_word,
        &mut word_probs,
        word_start,
        end,
        &mut segments,
    );

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── existing tests ────────────────────────────────────────────────────────

    #[test]
    fn test_align_tokens_empty() {
        let result = align_tokens_monotonic_peak(&[], 0, 0, 160, 16000);
        assert!(result.is_empty());
    }

    #[test]
    fn test_align_tokens_single() {
        // 1 token, 10 frames, attention peaks at frame 5
        let mut weights = vec![0.0f32; 10];
        weights[5] = 1.0;
        let result = align_tokens_monotonic_peak(&weights, 1, 10, 160, 16000);
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
        let result = align_tokens_monotonic_peak(&weights, 3, n_frames, 160, 16000);
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

    // ── 10 new tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_align_tokens_all_attention_on_final_frame() {
        // All tokens have peak at last frame — no NaN, no negative durations.
        let n_tokens = 5;
        let n_frames = 10;
        let mut attn = vec![0.0f32; n_tokens * n_frames];
        // Set the last frame of each token row to 1.0.
        for t in 0..n_tokens {
            attn[t * n_frames + (n_frames - 1)] = 1.0;
        }
        let times = align_tokens_monotonic_peak(&attn, n_tokens, n_frames, 160, 16000);
        assert_eq!(times.len(), n_tokens);
        for (start, end) in &times {
            assert!(start.is_finite() && !start.is_nan(), "start must be finite");
            assert!(end.is_finite() && !end.is_nan(), "end must be finite");
            assert!(
                *end >= *start,
                "end >= start required, got start={start} end={end}"
            );
        }
    }

    #[test]
    fn test_align_tokens_determinism_same_inputs_same_outputs() {
        // Flat representation: 3 tokens × 4 frames.
        let attn: Vec<f32> = vec![
            0.1, 0.9, 0.3, 0.2, // token 0 peaks at frame 1
            0.2, 0.1, 0.8, 0.1, // token 1 peaks at frame 2
            0.1, 0.1, 0.2, 0.9, // token 2 peaks at frame 3
        ];
        let times1 = align_tokens_monotonic_peak(&attn, 3, 4, 160, 16000);
        let times2 = align_tokens_monotonic_peak(&attn, 3, 4, 160, 16000);
        assert_eq!(times1.len(), times2.len());
        for i in 0..times1.len() {
            assert_eq!(
                times1[i].0.to_bits(),
                times2[i].0.to_bits(),
                "start not deterministic at index {i}"
            );
            assert_eq!(
                times1[i].1.to_bits(),
                times2[i].1.to_bits(),
                "end not deterministic at index {i}"
            );
        }
    }

    #[test]
    fn test_align_tokens_argmax_tiebreak_is_documented() {
        // Row of all equal values — lock the tie-break behaviour so regressions are caught.
        // max_by returns the LAST equal-maximum element, so peak = frame 7 (index 7 of 8).
        let n_frames = 8_usize;
        let attn = vec![0.5f32; n_frames]; // 1 token × 8 frames
        let times = align_tokens_monotonic_peak(&attn, 1, n_frames, 160, 16000);
        assert_eq!(times.len(), 1);
        let (start, _end) = times[0];
        // frame 7 → start = 7 * 160 / 16000 = 0.07 s
        let expected = 7.0 * 160.0 / 16000.0_f32;
        assert!(
            (start - expected).abs() < 1e-6,
            "tie-break must yield last-frame time {expected}, got {start}"
        );
    }

    #[test]
    fn test_align_tokens_with_synthetic_cross_attention_matrix() {
        // Known-good fixture: token i peaks at frame 3*i.
        let n_tokens = 4;
        let n_frames = 12;
        let hop_length = 160;
        let sample_rate = 16000;
        let mut attn = vec![0.0f32; n_tokens * n_frames];
        for i in 0..n_tokens {
            attn[i * n_frames + 3 * i] = 1.0;
        }
        let times = align_tokens_monotonic_peak(&attn, n_tokens, n_frames, hop_length, sample_rate);
        assert_eq!(times.len(), n_tokens);
        // Verify peaks are monotonically non-decreasing (enforced by the algorithm).
        for i in 1..times.len() {
            assert!(
                times[i].0 >= times[i - 1].0,
                "monotonic violation at {i}: {} < {}",
                times[i].0,
                times[i - 1].0
            );
        }
    }

    #[test]
    fn test_align_tokens_handles_zero_n_tokens_gracefully() {
        // n_tokens=0 — returns empty Vec, no panic.
        let attn: Vec<f32> = vec![];
        let times = align_tokens_monotonic_peak(&attn, 0, 4, 160, 16000);
        assert!(times.is_empty(), "expected empty result for n_tokens=0");
    }

    #[test]
    fn test_build_word_segments_start_le_end() {
        // For any valid inputs, seg.start <= seg.end must hold.
        let texts = vec![
            " Hello".to_string(),
            " beautiful".to_string(),
            " world".to_string(),
        ];
        let times = vec![(0.0f32, 0.3f32), (0.3, 0.7), (0.7, 1.0)];
        let probs = vec![-0.05, -0.1, -0.08];
        let segs = build_word_segments(&texts, &times, &probs);
        for seg in &segs {
            assert!(
                seg.start <= seg.end,
                "start {} > end {} for word '{}'",
                seg.start,
                seg.end,
                seg.word
            );
        }
    }

    #[test]
    fn test_build_word_segments_monotonic_starts() {
        // segs[i+1].start >= segs[i].start for all consecutive segments.
        let texts = vec![
            " Alpha".to_string(),
            " Beta".to_string(),
            " Gamma".to_string(),
            " Delta".to_string(),
        ];
        let times = vec![(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)];
        let probs = vec![-0.1, -0.2, -0.15, -0.05];
        let segs = build_word_segments(&texts, &times, &probs);
        for i in 1..segs.len() {
            assert!(
                segs[i].start >= segs[i - 1].start,
                "non-monotonic starts at {i}: {} < {}",
                segs[i].start,
                segs[i - 1].start
            );
        }
    }

    #[test]
    fn test_build_word_segments_punctuation_attaches_to_previous_word() {
        // Punctuation tokens (no leading space) attach to the preceding word.
        // " Hello" + "," → one segment "Hello,"; " world" + "." → one segment "world."
        let texts = vec![
            " Hello".to_string(),
            ",".to_string(),
            " world".to_string(),
            ".".to_string(),
        ];
        let times = vec![(0.0, 0.3), (0.3, 0.35), (0.35, 0.8), (0.8, 0.85)];
        let probs = vec![-0.05, -0.01, -0.06, -0.01];
        let segs = build_word_segments(&texts, &times, &probs);
        // Punctuation has no leading space, so it merges into current word.
        assert_eq!(segs.len(), 2, "expected 2 segments, got {}", segs.len());
        assert_eq!(segs[0].word, "Hello,");
        assert_eq!(segs[1].word, "world.");
    }

    #[test]
    fn test_build_word_segments_confidence_clean_alignment() {
        // Tokens with log-probs close to 0.0 → confidence ≈ -0.05.
        let texts = vec![" good".to_string(), " signal".to_string()];
        let times = vec![(0.0, 0.4), (0.4, 0.9)];
        let probs = vec![-0.04, -0.06];
        let segs = build_word_segments(&texts, &times, &probs);
        // Each word is one token; confidence equals its own log-prob.
        assert_eq!(segs.len(), 2);
        assert!(
            (segs[0].confidence - (-0.04)).abs() < 1e-5,
            "expected confidence ≈ -0.04, got {}",
            segs[0].confidence
        );
        assert!(
            (segs[1].confidence - (-0.06)).abs() < 1e-5,
            "expected confidence ≈ -0.06, got {}",
            segs[1].confidence
        );
    }

    #[test]
    fn test_build_word_segments_confidence_noisy_alignment() {
        // Tokens with very negative log-probs → confidence < -1.0.
        let texts = vec![" noise".to_string()];
        let times = vec![(0.0, 0.5)];
        let probs = vec![-3.5]; // very uncertain token
        let segs = build_word_segments(&texts, &times, &probs);
        assert_eq!(segs.len(), 1);
        assert!(
            segs[0].confidence < -1.0,
            "expected confidence < -1.0 for noisy token, got {}",
            segs[0].confidence
        );
    }

    // ── DP-DTW tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_dp_dtw_returns_correct_count() {
        // 4-token × 12-frame uniform attention → exactly 4 pairs returned.
        let n_tokens = 4;
        let n_frames = 12;
        let attn = vec![1.0f32 / n_frames as f32; n_tokens * n_frames];
        let result = align_tokens_dp_dtw(&attn, n_tokens, n_frames, 160, 16000, None);
        assert_eq!(
            result.len(),
            n_tokens,
            "expected {n_tokens} pairs, got {}",
            result.len()
        );
    }

    #[test]
    fn test_dp_dtw_monotonic_timestamps() {
        // Start times must be non-decreasing.
        let n_tokens = 6;
        let n_frames = 18;
        let mut attn = vec![0.0f32; n_tokens * n_frames];
        for i in 0..n_tokens {
            attn[i * n_frames + 3 * i] = 1.0;
        }
        let result = align_tokens_dp_dtw(&attn, n_tokens, n_frames, 160, 16000, None);
        assert_eq!(result.len(), n_tokens);
        for i in 1..result.len() {
            assert!(
                result[i].0 >= result[i - 1].0,
                "monotonic violation at {i}: start={} < prev_start={}",
                result[i].0,
                result[i - 1].0
            );
        }
    }

    #[test]
    fn test_dp_dtw_finite_times() {
        // No NaN or infinity in any output value.
        let n_tokens = 5;
        let n_frames = 15;
        let attn = vec![0.2f32; n_tokens * n_frames];
        let result = align_tokens_dp_dtw(&attn, n_tokens, n_frames, 160, 16000, None);
        assert_eq!(result.len(), n_tokens);
        for (i, (start, end)) in result.iter().enumerate() {
            assert!(
                start.is_finite() && !start.is_nan(),
                "start is not finite at {i}: {start}"
            );
            assert!(
                end.is_finite() && !end.is_nan(),
                "end is not finite at {i}: {end}"
            );
        }
    }

    #[test]
    fn test_dp_dtw_auto_widen_band_one() {
        // band=Some(1) with 8×24 diagonal fixture; path should be found via
        // auto-widen if the narrow band is infeasible.
        let n_tokens = 8_usize;
        let n_frames = 24_usize;
        let mut attn = vec![0.0f32; n_tokens * n_frames];
        // Diagonal: token i peaks at frame 3*i.
        for i in 0..n_tokens {
            attn[i * n_frames + 3 * i] = 1.0;
        }
        let result = align_tokens_dp_dtw(&attn, n_tokens, n_frames, 160, 16000, Some(1));
        assert!(
            !result.is_empty(),
            "expected non-empty result for band=Some(1) diagonal fixture"
        );
        assert_eq!(result.len(), n_tokens);
        for (start, end) in &result {
            assert!(
                start.is_finite() && end.is_finite(),
                "non-finite timestamps"
            );
        }
    }

    #[test]
    fn test_dp_dtw_parity_clean_diagonal() {
        // 8-token × 24-frame attention where token i peaks sharply at frame 3*i.
        //
        // For monotonic_peak: start_i == frame_to_time(argmax row_i) exactly.
        //
        // For dp_dtw: the traceback path partitions all 24 frames across 8 tokens.
        // The DP path always starts at (0,0) and ends at (n_tokens-1, n_frames-1),
        // so the timestamps must:
        //   (a) be monotonically non-decreasing,
        //   (b) be finite,
        //   (c) collectively span from 0.0 to (n_frames * hop / sr) — i.e. token 0
        //       start == 0.0 and the last token ends at or near the last frame.
        let n_tokens = 8_usize;
        let n_frames = 24_usize;
        let hop_length = 160_usize;
        let sample_rate = 16000_usize;
        let one_hop = hop_length as f32 / sample_rate as f32; // 0.01 s

        let mut attn = vec![0.0f32; n_tokens * n_frames];
        for i in 0..n_tokens {
            attn[i * n_frames + 3 * i] = 1.0;
        }

        let dp = align_tokens_dp_dtw(&attn, n_tokens, n_frames, hop_length, sample_rate, None);
        let mp = align_tokens_monotonic_peak(&attn, n_tokens, n_frames, hop_length, sample_rate);

        assert_eq!(dp.len(), n_tokens);
        assert_eq!(mp.len(), n_tokens);

        for (i, mp_pair) in mp.iter().enumerate() {
            let expected_start = (3 * i * hop_length) as f32 / sample_rate as f32;
            // Monotonic-peak: start == exact peak frame's time (within one hop tolerance).
            assert!(
                (mp_pair.0 - expected_start).abs() <= one_hop,
                "monotonic_peak token {i}: start={} expected≈{expected_start} tol={one_hop}",
                mp_pair.0
            );
        }

        // DP-DTW: finite, non-decreasing start times.
        for (i, (dp_start, dp_end)) in dp.iter().enumerate() {
            assert!(
                dp_start.is_finite() && dp_end.is_finite(),
                "dp_dtw token {i}: non-finite interval ({dp_start}, {dp_end})"
            );
        }

        // Token 0 must start at frame 0 (path begins at (0,0) → min_frame[0] = 0).
        assert!(
            dp[0].0 < one_hop,
            "dp_dtw token 0 must start at frame 0 (got {})",
            dp[0].0
        );

        // Both algorithms must produce non-decreasing start times.
        for i in 1..n_tokens {
            assert!(
                mp[i].0 >= mp[i - 1].0,
                "monotonic_peak: non-monotonic starts at {i}: {} < {}",
                mp[i].0,
                mp[i - 1].0
            );
            assert!(
                dp[i].0 >= dp[i - 1].0,
                "dp_dtw: non-monotonic starts at {i}: {} < {}",
                dp[i].0,
                dp[i - 1].0
            );
        }
    }

    #[test]
    fn test_dp_dtw_determinism() {
        // Same inputs → bit-identical outputs across two calls.
        let n_tokens = 5;
        let n_frames = 15;
        let mut attn = vec![0.0f32; n_tokens * n_frames];
        for i in 0..n_tokens {
            attn[i * n_frames + 3 * i] = 1.0;
        }
        let r1 = align_tokens_dp_dtw(&attn, n_tokens, n_frames, 160, 16000, None);
        let r2 = align_tokens_dp_dtw(&attn, n_tokens, n_frames, 160, 16000, None);
        assert_eq!(r1.len(), r2.len());
        for i in 0..r1.len() {
            assert_eq!(
                r1[i].0.to_bits(),
                r2[i].0.to_bits(),
                "start not deterministic at {i}"
            );
            assert_eq!(
                r1[i].1.to_bits(),
                r2[i].1.to_bits(),
                "end not deterministic at {i}"
            );
        }
    }

    #[test]
    fn test_dp_dtw_overconstrained_fallback() {
        // n_tokens=10 > n_frames=5 → falls back to monotonic_peak → returns 10 results.
        let n_tokens = 10;
        let n_frames = 5;
        let attn = vec![0.2f32; n_tokens * n_frames];
        let result = align_tokens_dp_dtw(&attn, n_tokens, n_frames, 160, 16000, None);
        assert_eq!(
            result.len(),
            n_tokens,
            "expected {n_tokens} pairs from overconstrained fallback, got {}",
            result.len()
        );
    }

    #[test]
    fn test_dp_dtw_band_zero_auto_widens() {
        // band_width=Some(0) should auto-widen to default and produce valid results.
        let n_tokens = 4;
        let n_frames = 12;
        let attn = vec![0.25f32; n_tokens * n_frames];
        let result = align_tokens_dp_dtw(&attn, n_tokens, n_frames, 160, 16000, Some(0));
        assert_eq!(
            result.len(),
            n_tokens,
            "band=Some(0) must produce {n_tokens} results"
        );
        for (start, end) in &result {
            assert!(
                start.is_finite() && end.is_finite(),
                "non-finite with band=Some(0)"
            );
        }
    }

    #[test]
    fn test_dp_dtw_zero_attention() {
        // All-zero attention weights → finite results, no NaN (softmax handles 0s).
        let n_tokens = 4;
        let n_frames = 12;
        let attn = vec![0.0f32; n_tokens * n_frames];
        let result = align_tokens_dp_dtw(&attn, n_tokens, n_frames, 160, 16000, None);
        assert_eq!(result.len(), n_tokens);
        for (i, (start, end)) in result.iter().enumerate() {
            assert!(
                start.is_finite() && !start.is_nan(),
                "start is NaN/inf at token {i}"
            );
            assert!(
                end.is_finite() && !end.is_nan(),
                "end is NaN/inf at token {i}"
            );
        }
    }

    #[test]
    fn test_dp_dtw_alias() {
        // align_tokens_dtw is bit-identical to align_tokens_dp_dtw(..., None).
        let n_tokens = 6;
        let n_frames = 18;
        let mut attn = vec![0.0f32; n_tokens * n_frames];
        for i in 0..n_tokens {
            attn[i * n_frames + 3 * i] = 1.0;
        }
        let via_alias = align_tokens_dtw(&attn, n_tokens, n_frames, 160, 16000);
        let via_dp = align_tokens_dp_dtw(&attn, n_tokens, n_frames, 160, 16000, None);
        assert_eq!(via_alias.len(), via_dp.len());
        for i in 0..via_alias.len() {
            assert_eq!(
                via_alias[i].0.to_bits(),
                via_dp[i].0.to_bits(),
                "alias start mismatch at {i}"
            );
            assert_eq!(
                via_alias[i].1.to_bits(),
                via_dp[i].1.to_bits(),
                "alias end mismatch at {i}"
            );
        }
    }
}
