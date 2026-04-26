#![cfg(feature = "test-utils")]

mod common;

use common::{TranscribeOptions, shared_model, silence, synthetic_sine};

#[test]
fn test_batch_three_clips_returns_three_results() {
    let owned: Vec<Vec<f32>> = vec![silence(1.0), synthetic_sine(1.0), silence(0.5)];
    let clips: Vec<&[f32]> = owned.iter().map(|v| v.as_slice()).collect();
    let results = shared_model().transcribe_batch(&clips, &TranscribeOptions::default());
    assert_eq!(results.len(), 3, "must return one result per clip");
}

#[test]
fn test_batch_all_results_ok_for_valid_input() {
    let owned: Vec<Vec<f32>> = vec![silence(1.0), silence(1.0)];
    let clips: Vec<&[f32]> = owned.iter().map(|v| v.as_slice()).collect();
    let results = shared_model().transcribe_batch(&clips, &TranscribeOptions::default());
    for (i, r) in results.iter().enumerate() {
        assert!(r.is_ok(), "clip {i} should succeed: {:?}", r.as_ref().err());
    }
}

#[test]
fn test_batch_empty_input_returns_empty() {
    let clips: Vec<&[f32]> = vec![];
    let results = shared_model().transcribe_batch(&clips, &TranscribeOptions::default());
    assert!(results.is_empty(), "empty input must produce empty output");
}
