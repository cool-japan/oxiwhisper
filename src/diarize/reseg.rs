// SPDX-License-Identifier: Apache-2.0
// Copyright (c) COOLJAPAN OU (Team Kitasan)

//! Resegmentation: turn per-window cluster labels into contiguous, sorted,
//! non-overlapping speaker segments.
//!
//! The clustering stage assigns a speaker label to every analysis window, but
//! those windows **overlap** (the hop is shorter than the window), so the raw
//! labels cannot be read out directly as time spans — consecutive windows share
//! samples, and a speaker change is only ever visible as the label flipping
//! between two overlapping windows. This module walks the windows in time order
//! and reconstructs clean speaker turns:
//!
//! 1. **Run building** — adjacent windows carrying the same label extend a
//!    single run; when the label changes, the run is cut at the **midpoint of
//!    the overlap** between the current run's end and the next window's start,
//!    which places the turn boundary in the middle of the region where the two
//!    windows disagree (the best time-localised estimate available). Boundaries
//!    are clamped to stay monotonic non-decreasing, so the emitted segments are
//!    always sorted and non-overlapping.
//! 2. **Flicker removal** — runs shorter than `min_duration_s` are spurious
//!    (a single misclassified window in the middle of a longer turn). Each such
//!    run is absorbed into a temporally-adjacent run — the previous run by
//!    preference, or the next run when it is the first run — after which
//!    same-label neighbours are coalesced. No segment shorter than
//!    `min_duration_s` is emitted unless it is the only run left.
//!
//! Times are reported in **seconds** (`sample / sample_rate`); a `sample_rate`
//! of `0`, empty inputs, or a `windows`/`labels` length mismatch all yield an
//! empty result rather than panicking.

use crate::diarize::{SpeakerId, SpeakerSegment};
use crate::vad::SpeechSegment;

/// An internal contiguous run of a single speaker label, in sample indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Run {
    /// Cluster label of this run.
    label: usize,
    /// Start sample index (inclusive).
    start: usize,
    /// End sample index (exclusive).
    end: usize,
}

impl Run {
    /// Length of the run in samples (saturating, never negative).
    fn len_samples(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

/// Floor of the midpoint of `a` and `b` without risking `a + b` overflow.
///
/// Equivalent to `(a + b) / 2` for all inputs, but computed so the intermediate
/// sum can never overflow `usize`.
fn midpoint(a: usize, b: usize) -> usize {
    a / 2 + b / 2 + ((a & 1) + (b & 1)) / 2
}

/// Convert per-window cluster `labels` over overlapping `windows` into
/// contiguous, sorted, non-overlapping [`SpeakerSegment`]s in seconds.
///
/// * `windows` — analysis windows in time order (as produced by
///   [`window_speech`](crate::diarize::segment::window_speech)); consecutive
///   windows within a speech region overlap.
/// * `labels` — one cluster label per window (from
///   [`cluster_speakers`](crate::diarize::cluster::cluster_speakers)); must be
///   the same length as `windows`.
/// * `sample_rate` — sample rate used to convert sample indices to seconds.
/// * `min_duration_s` — spurious runs shorter than this are merged into an
///   adjacent run (flicker removal). A value `<= 0` disables flicker removal.
///
/// Turn boundaries are placed at the midpoint of the overlap between the run
/// being closed and the window that first carries the new label, and are
/// clamped so boundary samples are monotonic non-decreasing — guaranteeing the
/// output is sorted by `start` and free of overlaps. The reported
/// `num_speakers` at the pipeline level is derived from the distinct
/// [`SpeakerId`]s in the returned segments.
///
/// # Silence gaps across speech regions
///
/// `resegment` is given only `windows` and `labels`; it has no knowledge of the
/// VAD speech-region boundaries the windows were drawn from. When the last
/// window of one speech region and the first window of the next carry the
/// **same** cluster label, run building extends a single run across the
/// intervening silence, so the emitted [`SpeakerSegment`] spans that gap and
/// reports the speaker as active throughout it. This deliberately favours
/// contiguous speaker turns over gap-accurate coverage: the boundary between two
/// *different* speakers is still placed inside the overlap, but a same-speaker
/// silence gap is bridged rather than punched out. Consumers that need silence
/// excluded from a speaker's span should intersect the returned segments with
/// the VAD speech regions.
///
/// # Edge cases (never panics)
///
/// * `sample_rate == 0`, empty `windows`, or empty `labels` → empty result.
/// * `windows.len() != labels.len()` → empty result (the caller wired the two
///   stages inconsistently; there is no meaningful segmentation to produce).
///   A debug assertion flags this in debug builds.
/// * A single window → a single segment equal to that window.
pub fn resegment(
    windows: &[SpeechSegment],
    labels: &[usize],
    sample_rate: usize,
    min_duration_s: f32,
) -> Vec<SpeakerSegment> {
    // A `windows`/`labels` length mismatch means the caller wired the embedding
    // and clustering stages inconsistently. Flag it loudly in debug builds, then
    // delegate to `resegment_checked`, whose runtime guard enforces the
    // return-empty contract in every build (debug and release alike). The
    // assertion is skipped for degenerate inputs (zero sample rate or an empty
    // side), which already carry a well-defined empty-result contract and must
    // not be treated as a wiring bug.
    debug_assert!(
        sample_rate == 0
            || windows.is_empty()
            || labels.is_empty()
            || windows.len() == labels.len(),
        "resegment: windows and labels must be the same length"
    );
    resegment_checked(windows, labels, sample_rate, min_duration_s)
}

/// The panic-free core of [`resegment`]: it applies the same edge-case guards
/// (zero sample rate, empty inputs, `windows`/`labels` length mismatch) as a
/// plain runtime `return Vec::new()` — no debug assertion — so the length-
/// mismatch contract is unit-testable in ordinary (debug) test builds.
fn resegment_checked(
    windows: &[SpeechSegment],
    labels: &[usize],
    sample_rate: usize,
    min_duration_s: f32,
) -> Vec<SpeakerSegment> {
    if sample_rate == 0 || windows.is_empty() || labels.is_empty() {
        return Vec::new();
    }
    if windows.len() != labels.len() {
        return Vec::new();
    }

    let runs = build_runs(windows, labels);
    let runs = remove_flickers(runs, sample_rate, min_duration_s);

    // Direct sample -> second division (`sample as f32 / sample_rate as f32`)
    // is exact for representable ratios; a multiply-by-reciprocal would not be.
    let rate = sample_rate as f32;
    runs.into_iter()
        .map(|run| {
            let start = run.start as f32 / rate;
            let end = run.end.max(run.start) as f32 / rate;
            SpeakerSegment {
                speaker: SpeakerId(run.label as u32),
                start,
                end,
            }
        })
        .collect()
}

/// Walk the overlapping windows once, emitting contiguous label runs cut at
/// overlap midpoints. The result is monotonic non-decreasing in `start`/`end`
/// and contiguous (each run's `end` equals the next run's `start`) within a
/// speech region.
fn build_runs(windows: &[SpeechSegment], labels: &[usize]) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();

    // Initialise the current run from the first window.
    let mut current = Run {
        label: labels[0],
        start: windows[0].start,
        end: windows[0].end.max(windows[0].start),
    };

    for (window, &label) in windows.iter().zip(labels.iter()).skip(1) {
        if label == current.label {
            // Same speaker: extend the run to cover this window.
            current.end = current.end.max(window.end);
            continue;
        }

        // Speaker change: cut at the midpoint of the overlap between the current
        // run's end and this window's start. Clamp so the boundary stays inside
        // `[current.start, window.end]`, which keeps runs non-negative in length
        // and boundaries monotonic non-decreasing.
        let raw_boundary = midpoint(window.start, current.end);
        let hi = window.end.max(current.start);
        let boundary = raw_boundary.clamp(current.start, hi);

        runs.push(Run {
            label: current.label,
            start: current.start,
            end: boundary,
        });
        current = Run {
            label,
            start: boundary,
            end: window.end.max(boundary),
        };
    }

    runs.push(current);
    runs
}

/// Remove runs shorter than `min_duration_s` by absorbing each into a
/// temporally-adjacent run (previous by preference, otherwise the next), then
/// coalescing any same-label neighbours created by the absorption. A single
/// remaining run is always kept, even if shorter than the threshold.
fn remove_flickers(mut runs: Vec<Run>, sample_rate: usize, min_duration_s: f32) -> Vec<Run> {
    if min_duration_s <= 0.0 {
        return runs;
    }
    // `min_duration_s` is finite and non-negative here; round to whole samples.
    let min_samples = (min_duration_s * sample_rate as f32).round().max(0.0) as usize;
    if min_samples == 0 {
        return runs;
    }

    loop {
        if runs.len() <= 1 {
            break;
        }

        // Pick the shortest sub-threshold run; ties break to the smallest index.
        let mut victim: Option<usize> = None;
        let mut victim_len = usize::MAX;
        for (i, run) in runs.iter().enumerate() {
            let len = run.len_samples();
            if len < min_samples && len < victim_len {
                victim_len = len;
                victim = Some(i);
            }
        }

        let Some(vi) = victim else {
            break;
        };

        if vi > 0 {
            // Absorb into the previous run: it swallows the flicker's span.
            let flicker_end = runs[vi].end;
            runs[vi - 1].end = runs[vi - 1].end.max(flicker_end);
        } else {
            // No previous run — absorb into the next run instead.
            let flicker_start = runs[vi].start;
            runs[vi + 1].start = runs[vi + 1].start.min(flicker_start);
        }
        runs.remove(vi);
        coalesce_same_label(&mut runs);
    }

    runs
}

/// Merge adjacent runs that share a label, keeping the span union. Runs built by
/// [`build_runs`] never have same-label neighbours, but flicker absorption can
/// create them (e.g. `[A, B, A]` with a short `B`).
fn coalesce_same_label(runs: &mut Vec<Run>) {
    let mut i = 0;
    while i + 1 < runs.len() {
        if runs[i].label == runs[i + 1].label {
            runs[i].start = runs[i].start.min(runs[i + 1].start);
            runs[i].end = runs[i].end.max(runs[i + 1].end);
            runs.remove(i + 1);
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: usize = 16_000;

    /// Build a window from second offsets, at `SR`.
    fn win(start_s: f32, end_s: f32) -> SpeechSegment {
        SpeechSegment {
            start: (start_s * SR as f32) as usize,
            end: (end_s * SR as f32) as usize,
        }
    }

    #[test]
    fn test_midpoint_is_exact_floor() {
        assert_eq!(midpoint(0, 0), 0);
        assert_eq!(midpoint(16_000, 24_000), 20_000);
        assert_eq!(midpoint(8_000, 16_000), 12_000);
        // Odd sum floors.
        assert_eq!(midpoint(3, 5), 4);
        assert_eq!(midpoint(2, 3), 2);
        // No overflow even near usize::MAX.
        assert_eq!(midpoint(usize::MAX, usize::MAX), usize::MAX);
    }

    // (a) Uniform labels collapse to a single segment spanning all windows.
    #[test]
    fn test_uniform_labels_single_segment() {
        // window = 1.0s, hop = 0.5s; three overlapping windows, all speaker 0.
        let windows = vec![win(0.0, 1.0), win(0.5, 1.5), win(1.0, 2.0)];
        let labels = vec![0, 0, 0];
        let segs = resegment(&windows, &labels, SR, 0.1);

        assert_eq!(segs.len(), 1, "uniform labels must yield one segment");
        assert_eq!(segs[0].speaker, SpeakerId(0));
        assert!((segs[0].start - 0.0).abs() < 1e-6);
        assert!(
            (segs[0].end - 2.0).abs() < 1e-6,
            "segment must span to the last window end (2.0s), got {}",
            segs[0].end
        );
    }

    // (b) A split at overlapping windows yields exactly two segments, with the
    // boundary at the expected overlap midpoint.
    #[test]
    fn test_split_labels_boundary_at_overlap_midpoint() {
        // window = 1.0s, hop = 0.5s.
        // w0=[0,1.0] w1=[0.5,1.5] w2=[1.0,2.0] w3=[1.5,2.5]; labels 0,0,1,1.
        // Run 0 covers w0,w1 -> end = 1.5s (24000). The label flips at w2 whose
        // start is 1.0s (16000); boundary = midpoint(16000, 24000) = 20000 =
        // 1.25s. Run 1 then extends through w3 to 2.5s.
        let windows = vec![win(0.0, 1.0), win(0.5, 1.5), win(1.0, 2.0), win(1.5, 2.5)];
        let labels = vec![0, 0, 1, 1];
        let segs = resegment(&windows, &labels, SR, 0.1);

        assert_eq!(
            segs.len(),
            2,
            "split labels must yield exactly two segments"
        );
        assert_eq!(segs[0].speaker, SpeakerId(0));
        assert_eq!(segs[1].speaker, SpeakerId(1));
        // Boundary time: tight tolerance around the exact 1.25s midpoint.
        assert!(
            (segs[0].end - 1.25).abs() < 1e-4,
            "expected boundary at 1.25s, got {}",
            segs[0].end
        );
        assert!(
            (segs[1].start - 1.25).abs() < 1e-4,
            "second segment must start exactly at the boundary, got {}",
            segs[1].start
        );
        assert!((segs[0].start - 0.0).abs() < 1e-6);
        assert!((segs[1].end - 2.5).abs() < 1e-6);
        // Sorted and non-overlapping.
        assert!(segs[0].end <= segs[1].start + 1e-6);
    }

    // (c) A single-window flicker shorter than min_duration is absorbed into the
    // previous run, and same-label neighbours coalesce -> one segment here, and
    // never a segment shorter than min_duration_s.
    #[test]
    fn test_flicker_absorbed_into_previous_run() {
        // window = 1.0s, hop = 0.5s.
        // w0=[0,1.0]L0 w1=[0.5,1.5]L1 w2=[1.0,2.0]L0.
        // Runs: [0,0.75]L0, [0.75,1.25]L1 (0.5s), [1.25,2.0]L0.
        // min_duration = 0.6s absorbs the 0.5s L1 flicker into the previous L0
        // run; the two L0 runs then coalesce into a single [0,2.0] segment.
        let windows = vec![win(0.0, 1.0), win(0.5, 1.5), win(1.0, 2.0)];
        let labels = vec![0, 1, 0];
        let min_duration_s = 0.6;
        let segs = resegment(&windows, &labels, SR, min_duration_s);

        assert_eq!(segs.len(), 1, "flicker + coalesce must leave one segment");
        assert_eq!(segs[0].speaker, SpeakerId(0));
        assert!((segs[0].start - 0.0).abs() < 1e-6);
        assert!((segs[0].end - 2.0).abs() < 1e-6);
        for s in &segs {
            assert!(
                s.duration() >= min_duration_s - 1e-6,
                "no emitted segment may be shorter than min_duration_s"
            );
        }
    }

    // (c') A first-run flicker has no previous run, so it is absorbed into the
    // NEXT run (the "if none, the next" branch).
    #[test]
    fn test_first_run_flicker_absorbed_into_next_run() {
        // w0=[0,1.0]L1 w1=[0.5,1.5]L0 w2=[1.0,2.0]L0.
        // Runs: [0,0.75]L1 (0.75s), [0.75,2.0]L0 (1.25s).
        // min_duration = 1.0s: the leading 0.75s L1 run is a flicker with no
        // previous run, so it merges forward into the L0 run.
        let windows = vec![win(0.0, 1.0), win(0.5, 1.5), win(1.0, 2.0)];
        let labels = vec![1, 0, 0];
        let min_duration_s = 1.0;
        let segs = resegment(&windows, &labels, SR, min_duration_s);

        assert_eq!(segs.len(), 1, "leading flicker must merge forward");
        assert_eq!(
            segs[0].speaker,
            SpeakerId(0),
            "kept label is the next run's"
        );
        assert!((segs[0].start - 0.0).abs() < 1e-6);
        assert!((segs[0].end - 2.0).abs() < 1e-6);
    }

    // (d) Sample -> seconds conversion is exact for representable values.
    #[test]
    fn test_sample_to_seconds_conversion_exactness() {
        // Single window [8000, 24000) at 16 kHz -> [0.5s, 1.5s), both exactly
        // representable in f32.
        let windows = vec![SpeechSegment {
            start: 8_000,
            end: 24_000,
        }];
        let labels = vec![2];
        let segs = resegment(&windows, &labels, SR, 0.0);

        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].speaker, SpeakerId(2));
        assert_eq!(segs[0].start, 0.5, "8000 / 16000 must be exactly 0.5");
        assert_eq!(segs[0].end, 1.5, "24000 / 16000 must be exactly 1.5");
    }

    // (e) Degenerate inputs yield an empty result, never a panic.
    #[test]
    fn test_empty_and_zero_sample_rate_yield_empty() {
        let windows = vec![win(0.0, 1.0)];
        let labels = vec![0];
        assert!(resegment(&windows, &labels, 0, 0.5).is_empty(), "sr == 0");
        assert!(resegment(&[], &labels, SR, 0.5).is_empty(), "empty windows");
        assert!(resegment(&windows, &[], SR, 0.5).is_empty(), "empty labels");
    }

    #[test]
    fn test_length_mismatch_yields_empty() {
        // The public `resegment` intentionally debug_asserts on a length
        // mismatch, so exercise the runtime return-empty guard directly through
        // `resegment_checked` (no debug assertion). This asserts the guard in
        // ordinary debug builds — the default test/CI profile — not only in the
        // otherwise-untested release path.
        let windows = vec![win(0.0, 1.0), win(0.5, 1.5)];
        let labels = vec![0]; // one short: a genuine length mismatch
        assert!(
            resegment_checked(&windows, &labels, SR, 0.1).is_empty(),
            "length mismatch must yield empty"
        );
        // Control: a matched-length pair over the same windows does segment, so
        // the emptiness above is attributable to the mismatch guard alone rather
        // than to some unrelated degenerate input.
        let matched = resegment_checked(&windows, &[0, 1], SR, 0.1);
        assert_eq!(matched.len(), 2, "matched-length control must segment");
    }

    #[test]
    fn test_single_window_single_segment() {
        let windows = vec![win(1.0, 2.5)];
        let labels = vec![3];
        let segs = resegment(&windows, &labels, SR, 0.5);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].speaker, SpeakerId(3));
        assert!((segs[0].start - 1.0).abs() < 1e-6);
        assert!((segs[0].end - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_multi_flip_is_sorted_and_non_overlapping() {
        // Alternating labels across many overlapping windows must still produce
        // strictly sorted, non-overlapping, contiguous segments.
        let windows: Vec<SpeechSegment> = (0..8)
            .map(|i| {
                let start = i as f32 * 0.5;
                win(start, start + 1.0)
            })
            .collect();
        let labels = vec![0, 1, 0, 1, 0, 1, 0, 1];
        // Small min_duration so nothing is merged away; we only check ordering.
        let segs = resegment(&windows, &labels, SR, 0.0);

        assert!(!segs.is_empty());
        for pair in segs.windows(2) {
            assert!(
                pair[0].start <= pair[1].start,
                "segments must be sorted by start"
            );
            assert!(
                pair[0].end <= pair[1].start + 1e-6,
                "segments must not overlap: {:?} then {:?}",
                pair[0],
                pair[1]
            );
        }
        // Coverage: first starts at 0, last ends at the final window end (4.5s).
        assert!((segs[0].start - 0.0).abs() < 1e-6);
        assert!(
            (segs.last().expect("non-empty").end - 4.5).abs() < 1e-6,
            "coverage must reach the last window end"
        );
    }
}

// ---------------------------------------------------------------------------
// End-to-end pipeline tests (VAD -> window -> embed -> cluster -> resegment).
//
// These exercise the FULL diarization wiring deterministically, without leaning
// on Whisper embedding quality: a planted embedder makes a real per-window
// speaker decision from measured band energy. Real speaker-accuracy / DER
// validation is intentionally deferred to Batch F (F2) -- the synthetic test
// model and the speaker-invariant Whisper-encoder baseline cannot honestly
// validate accuracy, so no accuracy/DER number is asserted here.
// ---------------------------------------------------------------------------
#[cfg(all(test, feature = "diarization"))]
mod pipeline_tests {
    use crate::WhisperModel;
    use crate::diarize::embed::SpeakerEmbedder;
    use crate::diarize::{ClusteringMethod, DiarizeOptions};
    use crate::types::OxiWhisperError;
    use std::f64::consts::PI;

    const SR: usize = crate::mel::WHISPER_SAMPLE_RATE;

    /// Delete a temp model file when the test scope ends.
    struct TempFileCleanup(std::path::PathBuf);
    impl Drop for TempFileCleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Coherent (Goertzel-style) energy of `audio` at `freq`, in f64.
    fn band_energy(audio: &[f32], sample_rate: usize, freq: f64) -> f64 {
        let w = 2.0 * PI * freq / sample_rate as f64;
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (i, &s) in audio.iter().enumerate() {
            let phase = w * i as f64;
            re += s as f64 * phase.cos();
            im += s as f64 * phase.sin();
        }
        (re * re + im * im).sqrt()
    }

    /// A PLANTED, test-only speaker embedder: it measures which of two known
    /// planted tone frequencies dominates the window and returns a near-one-hot
    /// vector for that speaker. This is a real per-window decision from the
    /// audio -- it does not fabricate labels.
    struct PlantedBandEmbedder {
        freqs: [f64; 2],
        floor: f32,
    }

    impl SpeakerEmbedder for PlantedBandEmbedder {
        fn dim(&self) -> usize {
            2
        }

        fn embed(&self, audio: &[f32], sample_rate: usize) -> Result<Vec<f32>, OxiWhisperError> {
            if audio.is_empty() {
                return Err(OxiWhisperError::InferenceFailed(
                    "planted embedder: empty window".into(),
                ));
            }
            let e0 = band_energy(audio, sample_rate, self.freqs[0]);
            let e1 = band_energy(audio, sample_rate, self.freqs[1]);
            // Hard decision, encoded near-one-hot with a small shared floor.
            let mut v = vec![self.floor; 2];
            if e0 >= e1 {
                v[0] = 1.0;
            } else {
                v[1] = 1.0;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            Ok(v.into_iter().map(|x| x / norm).collect())
        }
    }

    /// Deterministic two-speaker timeline: `f_low` for the first half of the
    /// samples, `f_high` for the second, continuous (no silence) so VAD yields
    /// a single region.
    fn two_speaker_signal(total: usize, mid: usize, f_low: f64, f_high: f64) -> Vec<f32> {
        (0..total)
            .map(|i| {
                let f = if i < mid { f_low } else { f_high };
                (2.0 * PI * f * i as f64 / SR as f64).sin() as f32 * 0.5
            })
            .collect()
    }

    #[test]
    fn test_full_pipeline_recovers_two_speakers_with_planted_embedder() {
        // 8 s timeline, speaker change at the 4 s midpoint.
        let total = 8 * SR;
        let mid = 4 * SR;
        let audio = two_speaker_signal(total, mid, 200.0, 2000.0);

        let model_path = crate::test_utils::generate_synthetic_model();
        let _cleanup = TempFileCleanup(model_path.clone());
        let model = WhisperModel::from_file(&model_path).expect("load synthetic model");

        let opts = DiarizeOptions {
            num_speakers: None, // auto-detect: the pipeline must recover 2
            clustering: ClusteringMethod::Ahc { threshold: 0.5 },
            ..DiarizeOptions::default()
        };
        let embedder = PlantedBandEmbedder {
            freqs: [200.0, 2000.0],
            floor: 0.05,
        };

        let result = model
            .diarize_with_embedder(&audio, &opts, &embedder)
            .expect("diarization pipeline must succeed");

        // The full VAD->window->embed->cluster->resegment pipeline recovers 2.
        assert_eq!(
            result.num_speakers, 2,
            "planted 2-speaker signal must recover two speakers, got {result:?}"
        );
        let segs = &result.segments;
        assert!(
            segs.len() >= 2,
            "expected at least two segments, got {segs:?}"
        );

        // Sorted, non-overlapping, non-negative.
        for pair in segs.windows(2) {
            assert!(pair[0].start <= pair[1].start, "segments must be sorted");
            assert!(
                pair[0].end <= pair[1].start + 1e-6,
                "segments must not overlap: {:?} then {:?}",
                pair[0],
                pair[1]
            );
        }
        for s in segs {
            assert!(s.duration() >= 0.0, "duration must be non-negative");
        }

        // Coverage: the single speech region [0, 8.0s) is fully spanned.
        assert!(
            segs[0].start.abs() < 1e-3,
            "coverage must start at 0, got {}",
            segs[0].start
        );
        let last_end = segs.last().expect("non-empty").end;
        assert!(
            (last_end - 8.0).abs() < 1e-3,
            "coverage must reach 8.0s, got {last_end}"
        );

        // Exactly one speaker change, near the 4 s midpoint. Tolerance is driven
        // by the window/hop geometry (one hop = 0.75 s); the true error here is
        // ~0.125 s (overlap-midpoint of the disagreeing windows).
        let mut change_time = None;
        for pair in segs.windows(2) {
            if pair[0].speaker != pair[1].speaker {
                assert!(change_time.is_none(), "expected a single speaker change");
                change_time = Some(pair[0].end);
            }
        }
        let change_time = change_time.expect("there must be a speaker change");
        assert!(
            (change_time - 4.0).abs() <= opts.hop_s,
            "speaker change {change_time} must be within one hop of the 4.0s midpoint"
        );
    }

    #[test]
    fn test_real_diarize_baseline_smoke_structural_only() {
        // SMOKE test of the real WhisperModel::diarize convenience path on the
        // synthetic model. We assert ONLY structural validity: the built-in
        // Whisper-encoder baseline is speaker-invariant and the synthetic model
        // cannot honestly validate accuracy, so NO accuracy/DER is asserted
        // here. That validation is deferred to Batch F (F2).
        let total = 4 * SR; // 4 s continuous tone -> a single VAD region
        let audio: Vec<f32> = (0..total)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / SR as f64).sin() as f32 * 0.5)
            .collect();

        let model_path = crate::test_utils::generate_synthetic_model();
        let _cleanup = TempFileCleanup(model_path.clone());
        let model = WhisperModel::from_file(&model_path).expect("load synthetic model");

        let opts = DiarizeOptions::default();
        let result = model
            .diarize(&audio, &opts)
            .expect("baseline diarize must return Ok on a valid tone");

        // Structural validity only.
        for pair in result.segments.windows(2) {
            assert!(pair[0].start <= pair[1].start, "segments must be sorted");
            assert!(
                pair[0].end <= pair[1].start + 1e-6,
                "segments must not overlap"
            );
        }
        for s in &result.segments {
            assert!(s.duration() >= 0.0, "duration must be non-negative");
        }
        assert!(
            result.num_speakers >= opts.min_speakers && result.num_speakers <= opts.max_speakers,
            "num_speakers {} must be within [{}, {}]",
            result.num_speakers,
            opts.min_speakers,
            opts.max_speakers
        );
    }
}
