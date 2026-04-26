#![cfg(feature = "test-utils")]

mod common;

use common::{TranscribeOptions, shared_model, silence};

#[test]
fn test_stream_push_audio_and_finish() {
    // Push 2 seconds of silence in 1-second chunks, then finish
    let model = shared_model();
    let mut stream = model.stream(TranscribeOptions::default());
    let chunk = silence(1.0);
    stream.push_audio(&chunk);
    stream.push_audio(&chunk);
    let result = stream.finish();
    // finish() returns Err("Empty audio") for silence with synthetic model — that's fine
    let _ = result;
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
