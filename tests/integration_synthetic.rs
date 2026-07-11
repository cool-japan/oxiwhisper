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
    // The designed synthetic model decodes a fixed non-empty transcript, so the
    // result must be non-empty (and still bounded).
    assert!(!result.trim().is_empty(), "transcript must not be empty");
    assert!(result.len() <= 10000, "text too long");
}

#[test]
fn test_transcribe_silence_succeeds() {
    let audio = silence(1.0);
    let text = shared_model()
        .transcribe(&audio, &TranscribeOptions::default())
        .expect("transcribe silence");
    // The model ignores audio content (cross-attention is neutralised in the
    // synthetic weights), so even silence decodes the designed transcript and
    // the no-speech gate does not fire — the text must be non-empty.
    assert!(
        !text.trim().is_empty(),
        "designed model must decode non-empty text even on silence"
    );
}

#[test]
fn test_transcribe_with_initial_prompt_does_not_crash() {
    let audio = silence(1.0);
    let opts = TranscribeOptions {
        initial_prompt: Some("test prompt"),
        ..TranscribeOptions::default()
    };
    // An initial prompt must not derail decoding: the call succeeds and still
    // yields the designed non-empty transcript.
    let text = shared_model()
        .transcribe(&audio, &opts)
        .expect("transcribe with initial prompt must succeed");
    assert!(
        !text.trim().is_empty(),
        "transcription with an initial prompt must still produce text"
    );
}
