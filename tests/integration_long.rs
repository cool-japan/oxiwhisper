#![cfg(feature = "test-utils")]

mod common;

use common::{TranscribeOptions, shared_model, sine_with_gaps};
use oxiwhisper::vad::VadConfig;

#[test]
fn test_long_with_vad_60s_completes() {
    // 5 segments: 8s speech + 4s gap each = 60s total (well above the 30s
    // chunking threshold, so this exercises the multi-encoder-pass /
    // multi-chunk long-form path). Measured runtime: ~20s on the synthetic
    // test model.
    let audio = sine_with_gaps(8.0, 4.0, 5);
    let opts = TranscribeOptions {
        timestamps: true,
        ..TranscribeOptions::default()
    };
    let result = shared_model()
        .transcribe_long_with_vad(&audio, &opts, &VadConfig::default())
        .expect("transcribe_long_with_vad must succeed");

    // The synthetic model's output text is not semantically meaningful (see
    // tests/integration_synthetic.rs), so assert structural invariants of the
    // returned transcript instead of its content: every reported segment
    // timestamp must be finite, non-negative, internally ordered (end >= start),
    // and segments must appear in non-decreasing start order across the whole
    // multi-chunk run. Overall text length must also stay within a sane bound.
    assert!(
        result.text.len() <= 10_000,
        "text unexpectedly long: {} bytes",
        result.text.len()
    );
    let segs = &result.segments;
    // Non-vacuity guard: the designed synthetic model decodes paired timestamp
    // tokens on every 30 s chunk, so a multi-chunk VAD run must yield segments.
    // Without this guard the invariant loop below runs zero iterations.
    assert!(
        !segs.is_empty(),
        "multi-chunk VAD transcription must produce segments"
    );
    for (i, seg) in segs.iter().enumerate() {
        assert!(
            seg.start.is_finite(),
            "segment {i} start is not finite: {}",
            seg.start
        );
        assert!(
            seg.end.is_finite(),
            "segment {i} end is not finite: {}",
            seg.end
        );
        assert!(
            seg.start >= 0.0,
            "segment {i} start is negative: {}",
            seg.start
        );
        assert!(
            seg.end >= seg.start,
            "segment {i} end ({}) precedes start ({})",
            seg.end,
            seg.start
        );
        if i > 0 {
            assert!(
                seg.start >= segs[i - 1].start,
                "segment starts not monotonic at {i}: {} < {}",
                seg.start,
                segs[i - 1].start
            );
        }
    }
}

#[test]
fn test_long_short_audio_falls_through() {
    // 9s total (under 30s chunk threshold) — goes via transcribe_segmented shortcut
    let audio = sine_with_gaps(2.0, 1.0, 3);
    let opts = TranscribeOptions::default();
    let result = shared_model()
        .transcribe_long_with_vad(&audio, &opts, &VadConfig::default())
        .expect("short-audio long transcribe");
    // Default opts have `timestamps == false`, so the segment list must be empty
    // (segments are only parsed when timestamps are requested) while the plain
    // text is still populated by the designed synthetic model. This pins the
    // short-audio shortcut's real contract instead of discarding the result.
    assert!(
        result.segments.is_empty(),
        "timestamps=false must yield no segments, got {}",
        result.segments.len()
    );
    assert!(
        !result.text.trim().is_empty(),
        "short-audio VAD transcription must still produce non-empty text"
    );
}

#[test]
fn test_long_segmented_monotonic_timestamps() {
    // 15s total
    let audio = sine_with_gaps(3.0, 2.0, 3);
    let opts = TranscribeOptions {
        timestamps: true,
        ..TranscribeOptions::default()
    };
    let result = shared_model()
        .transcribe_long_with_vad(&audio, &opts, &VadConfig::default())
        .expect("long with vad");
    let segs = &result.segments;
    // Non-vacuity guard (this test was the audit's canonical vacuous case: the
    // loop below executed zero iterations on an always-empty segment list).
    assert!(
        !segs.is_empty(),
        "timestamped VAD transcription must produce segments"
    );
    for i in 0..segs.len() {
        assert!(
            segs[i].start.is_finite() && segs[i].end.is_finite(),
            "segment {i} times must be finite: {:?}",
            segs[i]
        );
        assert!(
            segs[i].end >= segs[i].start,
            "segment {i} end ({}) precedes start ({})",
            segs[i].end,
            segs[i].start
        );
        assert!(
            !segs[i].text.trim().is_empty(),
            "segment {i} text must not be empty: {:?}",
            segs[i]
        );
        if i > 0 {
            assert!(
                segs[i].start >= segs[i - 1].start,
                "segment starts not monotonic at {i}: {} < {}",
                segs[i].start,
                segs[i - 1].start
            );
        }
    }
}
