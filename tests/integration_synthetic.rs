#![cfg(feature = "test-utils")]

mod common;

use common::{TranscribeOptions, shared_model, silence, synthetic_sine};

#[test]
fn test_model_info_has_expected_shape() {
    let info = shared_model().info();
    assert!(info.n_vocab > 0, "n_vocab must be positive");
    assert!(info.n_mels > 0, "n_mels must be positive");
    assert!(info.d_model > 0, "d_model must be positive");
}

#[test]
fn test_transcribe_sine_wave_succeeds() {
    let audio = synthetic_sine(1.0);
    let result = shared_model()
        .transcribe(&audio, &TranscribeOptions::default())
        .expect("transcribe sine wave");
    // Don't assert text content — synthetic model output is not meaningful
    assert!(result.len() <= 10000, "text too long");
}

#[test]
fn test_transcribe_silence_succeeds() {
    let audio = silence(1.0);
    let _text = shared_model()
        .transcribe(&audio, &TranscribeOptions::default())
        .expect("transcribe silence");
}

#[test]
fn test_transcribe_with_initial_prompt_does_not_crash() {
    let audio = silence(1.0);
    let opts = TranscribeOptions {
        initial_prompt: Some("test prompt"),
        ..TranscribeOptions::default()
    };
    // Any result (Ok or Err) is acceptable — just no panic
    let _ = shared_model().transcribe(&audio, &opts);
}
