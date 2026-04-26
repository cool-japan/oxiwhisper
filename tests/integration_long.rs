#![cfg(feature = "test-utils")]

mod common;

use common::{TranscribeOptions, shared_model, sine_with_gaps};
use oxiwhisper::vad::VadConfig;

#[test]
#[ignore] // slow: ~60s of audio requires multiple encoder passes
fn test_long_with_vad_60s_completes() {
    // 5 segments: 8s speech + 4s gap each = 60s total
    let audio = sine_with_gaps(8.0, 4.0, 5);
    let opts = TranscribeOptions::default();
    let result = shared_model()
        .transcribe_long_with_vad(&audio, &opts, &VadConfig::default())
        .expect("transcribe_long_with_vad must succeed");
    let _ = result;
}

#[test]
fn test_long_short_audio_falls_through() {
    // 9s total (under 30s chunk threshold) — goes via transcribe_segmented shortcut
    let audio = sine_with_gaps(2.0, 1.0, 3);
    let opts = TranscribeOptions::default();
    let result = shared_model()
        .transcribe_long_with_vad(&audio, &opts, &VadConfig::default())
        .expect("short-audio long transcribe");
    let _ = result;
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
    for i in 1..segs.len() {
        assert!(
            segs[i].start >= segs[i - 1].start,
            "segment starts not monotonic at {i}: {} < {}",
            segs[i].start,
            segs[i - 1].start
        );
    }
}
