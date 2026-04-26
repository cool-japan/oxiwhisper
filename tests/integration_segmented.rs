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
    // Each segment must have start <= end
    for seg in &result.segments {
        assert!(
            seg.start <= seg.end,
            "segment start ({}) > end ({})",
            seg.start,
            seg.end
        );
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
