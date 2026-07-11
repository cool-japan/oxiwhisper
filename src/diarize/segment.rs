// SPDX-License-Identifier: Apache-2.0
// Copyright (c) COOLJAPAN OU (Team Kitasan)

//! Uniform sub-segmentation of VAD speech regions into overlapping fixed
//! windows.
//!
//! The diarization pipeline needs fixed-length windows to feed the speaker
//! embedding model, but VAD only hands us variable-length speech regions.
//! This module tiles each region into overlapping windows of `window_s`
//! seconds advancing by `hop_s` seconds, never letting a window span a
//! silence gap between two regions.
//!
//! There is an inherent trade-off in choosing `window_s` and `hop_s`:
//!
//! * **Shorter windows / smaller hops** localize speaker-change boundaries
//!   more precisely (less "blur" at a turn-taking point), because each
//!   window covers a narrower time span. The downstream clustering stage can
//!   therefore place a boundary more accurately in time.
//! * **Longer windows** average more audio per embedding, producing a
//!   higher-SNR (more reliable) speaker embedding -- short windows are more
//!   sensitive to phoneme-level variation, background noise and channel
//!   artifacts, which makes the resulting embedding noisier and less
//!   discriminative between speakers.
//!
//! In short: shrinking the window sharpens boundary resolution at the cost
//! of embedding reliability, and growing it does the reverse. `window_s` /
//! `hop_s` in [`crate::diarize::DiarizeOptions`] let the caller tune this
//! trade-off; this module only implements the mechanical tiling once those
//! values are chosen.

use crate::vad::SpeechSegment;

/// Tile each VAD speech region into overlapping fixed-length windows.
///
/// For every region in `regions`, windows of `round(window_s * sample_rate)`
/// samples are emitted starting at the region's `start`, advancing by
/// `round(hop_s * sample_rate)` samples each step, and never extending past
/// the region's `end` (i.e. never crossing into a silence gap or a
/// neighboring region). A trailing partial window shorter than a full window
/// is kept only if its duration is at least `min_duration_s` seconds;
/// otherwise it is dropped, since its samples are already represented in the
/// overlap with the previous window. A region entirely shorter than one
/// window is emitted as a single window equal to the region, provided it
/// meets `min_duration_s`.
///
/// Returns an empty vector (never panics) when `sample_rate == 0`, when
/// `regions` is empty, or when the computed window or hop length in samples
/// is `0` (degenerate `window_s` / `hop_s`, which would otherwise not
/// advance and loop forever).
pub fn window_speech(
    regions: &[SpeechSegment],
    sample_rate: usize,
    window_s: f32,
    hop_s: f32,
    min_duration_s: f32,
) -> Vec<SpeechSegment> {
    if sample_rate == 0 || regions.is_empty() {
        return Vec::new();
    }

    let window_samples_f = (window_s * sample_rate as f32).round();
    let hop_samples_f = (hop_s * sample_rate as f32).round();
    if window_samples_f < 1.0 || hop_samples_f < 1.0 {
        return Vec::new();
    }
    let window_samples = window_samples_f as usize;
    let hop_samples = hop_samples_f as usize;
    if window_samples == 0 || hop_samples == 0 {
        return Vec::new();
    }

    let mut windows = Vec::new();
    for region in regions {
        if region.start >= region.end {
            continue;
        }

        let mut pos = region.start;
        while pos + window_samples <= region.end {
            windows.push(SpeechSegment {
                start: pos,
                end: pos + window_samples,
            });
            pos += hop_samples;
        }

        if pos < region.end {
            let tail_samples = region.end - pos;
            let tail_s = tail_samples as f32 / sample_rate as f32;
            if tail_s >= min_duration_s {
                windows.push(SpeechSegment {
                    start: pos,
                    end: region.end,
                });
            }
        }
    }

    windows
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: usize = 16_000;

    #[test]
    fn test_single_region_exact_window_count_and_bounds() {
        // 10s region, 1.5s window, 0.75s hop -> window=24000, hop=12000
        // samples. Full windows fit for pos in {0, 12000, ..., 132000} (12
        // windows), leaving a 1.0s tail (>= min_duration_s=0.3) kept as a
        // 13th window.
        let region = SpeechSegment {
            start: 0,
            end: 10 * SAMPLE_RATE,
        };
        let windows = window_speech(&[region], SAMPLE_RATE, 1.5, 0.75, 0.3);

        assert_eq!(windows.len(), 13);
        assert_eq!(
            windows[0],
            SpeechSegment {
                start: 0,
                end: 24_000
            }
        );
        assert_eq!(
            windows[12],
            SpeechSegment {
                start: 144_000,
                end: 160_000,
            }
        );
    }

    #[test]
    fn test_trailing_tail_dropped_below_min_duration_kept_above() {
        let region = SpeechSegment {
            start: 0,
            end: 10 * SAMPLE_RATE,
        };
        // Tail is 1.0s (16000 samples): 1.5s threshold drops it, 0.5s keeps it.
        let dropped = window_speech(std::slice::from_ref(&region), SAMPLE_RATE, 1.5, 0.75, 1.5);
        let kept = window_speech(&[region], SAMPLE_RATE, 1.5, 0.75, 0.5);

        assert_eq!(dropped.len(), 12);
        assert_eq!(kept.len(), 13);
        assert_eq!(
            kept.last().expect("kept has 13 windows"),
            &SpeechSegment {
                start: 144_000,
                end: 160_000,
            }
        );
    }

    #[test]
    fn test_region_shorter_than_window_yields_single_window_equal_to_region() {
        // 1.0s region is shorter than the 1.5s window but >= min_duration_s
        // (0.3s), so it must be emitted verbatim as one window.
        let region = SpeechSegment {
            start: 1_000,
            end: 1_000 + SAMPLE_RATE, // 1.0s long
        };
        let windows = window_speech(std::slice::from_ref(&region), SAMPLE_RATE, 1.5, 0.75, 0.3);

        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0], region);
    }

    #[test]
    fn test_short_region_below_min_duration_is_dropped() {
        let region = SpeechSegment {
            start: 0,
            end: SAMPLE_RATE / 10, // 0.1s, shorter than min_duration_s
        };
        let windows = window_speech(&[region], SAMPLE_RATE, 1.5, 0.75, 0.5);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_two_regions_never_cross_the_gap() {
        // Region A: [0, 2.0s), region B: [3.0s, 5.0s) -- a 1.0s silence gap.
        let region_a = SpeechSegment {
            start: 0,
            end: 2 * SAMPLE_RATE,
        };
        let region_b = SpeechSegment {
            start: 3 * SAMPLE_RATE,
            end: 5 * SAMPLE_RATE,
        };
        let windows = window_speech(&[region_a, region_b], SAMPLE_RATE, 1.5, 0.75, 0.3);

        assert!(!windows.is_empty());
        for w in &windows {
            let inside_a = w.end <= 2 * SAMPLE_RATE;
            let inside_b = w.start >= 3 * SAMPLE_RATE && w.end <= 5 * SAMPLE_RATE;
            assert!(
                inside_a || inside_b,
                "window {:?} crosses the inter-region gap",
                w
            );
        }
    }

    #[test]
    fn test_zero_sample_rate_and_empty_regions_yield_empty() {
        let region = SpeechSegment {
            start: 0,
            end: SAMPLE_RATE,
        };
        assert!(window_speech(&[region], 0, 1.5, 0.75, 0.3).is_empty());
        assert!(window_speech(&[], SAMPLE_RATE, 1.5, 0.75, 0.3).is_empty());
    }

    #[test]
    fn test_degenerate_window_or_hop_yields_empty_not_infinite_loop() {
        let region = SpeechSegment {
            start: 0,
            end: SAMPLE_RATE,
        };
        // window_s so small it rounds to 0 samples.
        assert!(
            window_speech(
                std::slice::from_ref(&region),
                SAMPLE_RATE,
                0.0000_1,
                0.75,
                0.3
            )
            .is_empty()
        );
        // hop_s so small it rounds to 0 samples.
        assert!(window_speech(&[region], SAMPLE_RATE, 1.5, 0.0000_1, 0.3).is_empty());
    }
}
