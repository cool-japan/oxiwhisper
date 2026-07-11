#![cfg(feature = "test-utils")]

mod common;

use common::{TranscribeOptions, shared_model, silence};

#[test]
fn test_stream_push_audio_and_finish() {
    // Push 2 seconds of silence in 1-second chunks (well under the 30s chunk
    // threshold), then finish.
    //
    // `finish()`'s "Empty audio" error path only fires when `audio_buf` is
    // *empty* at flush time (see `StreamTranscriber::finish` in
    // `src/stream.rs`) — here 32000 samples are still buffered, so `finish()`
    // hands them straight to `transcribe_segmented` instead. The designed
    // synthetic model (see `src/test_utils.rs`) ignores audio content
    // entirely and decodes a fixed "hello"/"world" token chain regardless of
    // input, so it produces a real `Ok` result even from silence (confirmed
    // by `test_transcribe_silence_succeeds` in `integration_synthetic.rs`).
    // This exercises the real streaming flush/decode path end-to-end.
    let model = shared_model();
    let opts = TranscribeOptions {
        timestamps: true,
        ..TranscribeOptions::default()
    };
    let mut stream = model.stream(opts);
    let chunk = silence(1.0);
    stream.push_audio(&chunk);
    stream.push_audio(&chunk);
    let result = stream
        .finish()
        .expect("finish() must succeed: the designed synthetic model decodes a transcript even from silence");

    assert!(
        !result.text.trim().is_empty(),
        "finish() must produce non-empty text on the designed synthetic model"
    );
    assert!(
        !result.segments.is_empty(),
        "finish() with timestamps=true must produce at least one segment"
    );

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
fn test_stream_push_returns_buffered_samples() {
    let model = shared_model();
    let mut stream = model.stream(TranscribeOptions::default());
    assert_eq!(stream.buffered_samples(), 0);
    let chunk = silence(1.0);
    stream.push_audio(&chunk);
    assert_eq!(stream.buffered_samples(), 16000);
}

#[test]
fn test_stream_finish_on_empty_does_not_panic() {
    let model = shared_model();
    let stream = model.stream(TranscribeOptions::default());
    // finish() on an empty stream returns Err — that's expected, not a panic
    let _ = stream.finish();
}

#[test]
fn test_stream_processed_samples_tracks_state() {
    let model = shared_model();
    let mut stream = model.stream(TranscribeOptions::default());
    assert_eq!(stream.processed_samples(), 0);
    // With less than 30s no chunk is processed, so processed_samples stays 0
    stream.push_audio(&silence(1.0));
    // next_segment won't do anything without a full 30s chunk
    let _ = stream.next_segment();
    assert_eq!(stream.processed_samples(), 0);
}
