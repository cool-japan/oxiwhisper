//! Shared test fixtures for oxiwhisper integration tests.
//!
//! The synthetic model is initialized lazily via `OnceLock` — once per test
//! binary process, not once per test function. This keeps test runs fast
//! while still being correct.

// Each integration test binary only uses a subset of these helpers.
// Suppress dead_code and unused_imports lints that would fire per-binary —
// the full set of exports is only consumed across multiple test binaries.
#![allow(dead_code, unused_imports)]

use std::path::PathBuf;
use std::sync::OnceLock;

pub use oxiwhisper::{TranscribeOptions, WhisperModel};

#[cfg(feature = "test-utils")]
static MODEL_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Return a reference to the path of a lazily-initialized synthetic WhisperModel.
///
/// The model binary is written to a temp file on the first call and reused
/// on subsequent calls within the same test binary process.
#[cfg(feature = "test-utils")]
pub fn synthetic_model_path() -> &'static PathBuf {
    MODEL_PATH.get_or_init(oxiwhisper::test_utils::generate_synthetic_model)
}

/// Return a reference to a loaded synthetic WhisperModel.
/// The model is loaded once and kept alive for the entire test binary process.
#[cfg(feature = "test-utils")]
pub fn shared_model() -> &'static WhisperModel {
    static MODEL: OnceLock<WhisperModel> = OnceLock::new();
    MODEL.get_or_init(|| {
        let path = synthetic_model_path();
        WhisperModel::from_file(path).expect("load synthetic WhisperModel")
    })
}

/// Generate `duration_secs` seconds of 440 Hz sine wave at 16 kHz mono f32.
pub fn synthetic_sine(duration_secs: f32) -> Vec<f32> {
    let sample_rate = 16000u32;
    let n = (sample_rate as f32 * duration_secs) as usize;
    let freq = 440.0f32;
    (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin() * 0.5)
        .collect()
}

/// Generate `duration_secs` seconds of silence (all zeros).
pub fn silence(duration_secs: f32) -> Vec<f32> {
    let n = (16000.0 * duration_secs) as usize;
    vec![0.0f32; n]
}

/// Generate audio with alternating sine (speech-like) and silence segments.
///
/// Returns `n_segments` sine segments of `speech_secs` each, separated by
/// `gap_secs` of silence.
pub fn sine_with_gaps(speech_secs: f32, gap_secs: f32, n_segments: usize) -> Vec<f32> {
    let speech_samples = (16000.0 * speech_secs) as usize;
    let gap_samples = (16000.0 * gap_secs) as usize;
    let freq = 440.0f32;
    let sample_rate = 16000.0f32;

    let mut out = Vec::with_capacity(n_segments * (speech_samples + gap_samples));
    for seg_idx in 0..n_segments {
        let base = seg_idx * speech_samples;
        for i in 0..speech_samples {
            out.push(
                (2.0 * std::f32::consts::PI * freq * (base + i) as f32 / sample_rate).sin() * 0.5,
            );
        }
        out.extend(std::iter::repeat_n(0.0f32, gap_samples));
    }
    out
}
