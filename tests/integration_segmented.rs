#![cfg(feature = "test-utils")]

mod common;

use common::{TranscribeOptions, shared_model, silence};

#[test]
fn test_transcribe_and_segmented_text_consistent() {
    let audio = silence(1.0);
    let opts = TranscribeOptions::default();
    let plain_text = shared_model()
        .transcribe(&audio, &opts)
        .expect("transcribe");
    let segmented = shared_model()
        .transcribe_segmented(&audio, &opts)
        .expect("transcribe_segmented");
    // The synthetic model is designed to decode a fixed, non-empty transcript
    // (see src/test_utils.rs), so this consistency check is not vacuous: both
    // APIs must actually produce text, and it must be the same text.
    assert!(
        !plain_text.trim().is_empty(),
        "transcribe must produce non-empty text on the designed synthetic model"
    );
    // Both APIs must return the same text (modulo leading/trailing whitespace)
    assert_eq!(
        plain_text.trim(),
        segmented.text.trim(),
        "transcribe and transcribe_segmented must agree on text"
    );
}

#[test]
fn test_segmented_returns_valid_segment_list() {
    let audio = silence(2.0);
    let opts = TranscribeOptions {
        timestamps: true,
        ..TranscribeOptions::default()
    };
    let result = shared_model()
        .transcribe_segmented(&audio, &opts)
        .expect("transcribe_segmented with timestamps");

    // Non-vacuity guard: with timestamps enabled the designed synthetic model
    // emits paired timestamp tokens bracketing real text, so parse_segments must
    // yield a non-empty segment list. Without this guard the per-segment loop
    // below would silently pass on an empty vector (the original defect).
    assert!(
        !result.segments.is_empty(),
        "timestamps=true must produce at least one segment on the designed model"
    );

    // Each segment must carry non-empty text and a valid, finite time span,
    // and the segments must appear in non-decreasing start order.
    let mut prev_start = f32::NEG_INFINITY;
    for seg in &result.segments {
        assert!(
            !seg.text.trim().is_empty(),
            "segment text must not be empty: {seg:?}"
        );
        assert!(
            seg.start.is_finite() && seg.end.is_finite(),
            "segment times must be finite: {seg:?}"
        );
        assert!(
            seg.start <= seg.end,
            "segment start ({}) > end ({})",
            seg.start,
            seg.end
        );
        assert!(
            seg.start >= prev_start,
            "segment starts must be non-decreasing: {} < {prev_start}",
            seg.start
        );
        prev_start = seg.start;
    }
}

#[test]
fn test_segmented_no_timestamps_returns_empty_segments() {
    let audio = silence(1.0);
    let opts = TranscribeOptions {
        timestamps: false,
        ..TranscribeOptions::default()
    };
    let result = shared_model()
        .transcribe_segmented(&audio, &opts)
        .expect("transcribe_segmented without timestamps");
    assert!(
        result.segments.is_empty(),
        "segments should be empty when timestamps=false"
    );
}
