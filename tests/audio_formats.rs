//! Integration tests for audio format expansion (F1).
//! Each test is gated by the relevant feature flag.

#[cfg(feature = "audio-flac")]
mod flac_tests {
    use std::path::Path;

    fn fixture_path(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn test_load_flac_fixture() {
        let path = fixture_path("sample_440hz.flac");
        if !path.exists() {
            eprintln!("Skipping: fixture not found at {path:?}");
            return;
        }
        let samples = oxiwhisper::audio::load_audio(&path).expect("load FLAC should succeed");
        assert!(!samples.is_empty(), "FLAC samples must not be empty");
        // 1 second at 16 kHz = 16000 samples; allow +-10% for resampling
        assert!(
            samples.len() > 14000 && samples.len() < 18000,
            "expected ~16000 samples, got {}",
            samples.len()
        );
    }

    #[test]
    fn test_flac_samples_are_finite() {
        let path = fixture_path("sample_440hz.flac");
        if !path.exists() {
            return;
        }
        let samples = oxiwhisper::audio::load_audio(&path).expect("load FLAC");
        assert!(
            samples.iter().all(|s| s.is_finite()),
            "all samples must be finite"
        );
    }
}

#[cfg(feature = "audio-ogg")]
mod ogg_tests {
    use std::path::Path;

    fn fixture_path(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn test_load_ogg_vorbis_fixture() {
        let path = fixture_path("sample_440hz.ogg");
        if !path.exists() {
            eprintln!("Skipping: fixture not found");
            return;
        }
        let samples = oxiwhisper::audio::load_audio(&path).expect("load OGG");
        assert!(!samples.is_empty(), "OGG samples must not be empty");
        assert!(
            samples.iter().all(|s| s.is_finite()),
            "all OGG samples must be finite"
        );
    }
}

#[cfg(feature = "audio-mp3")]
mod mp3_tests {
    use std::path::Path;

    fn fixture_path(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn test_load_mp3_fixture() {
        let path = fixture_path("sample_440hz.mp3");
        if !path.exists() {
            eprintln!("Skipping: fixture not found");
            return;
        }
        let samples = oxiwhisper::audio::load_audio(&path).expect("load MP3");
        assert!(!samples.is_empty(), "MP3 samples must not be empty");
        assert!(
            samples.iter().all(|s| s.is_finite()),
            "all MP3 samples must be finite"
        );
    }
}

#[cfg(feature = "audio-opus")]
mod opus_tests {
    use std::path::Path;

    fn fixture_path(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn test_load_opus_fixture() {
        let path = fixture_path("sample_440hz.opus");
        if !path.exists() {
            eprintln!("Skipping: fixture not found");
            return;
        }
        let samples = oxiwhisper::audio::load_audio(&path).expect("load Opus");
        assert!(!samples.is_empty(), "Opus samples must not be empty");
        assert!(
            samples.iter().all(|s| s.is_finite()),
            "all Opus samples must be finite"
        );
    }
}

#[test]
fn test_load_audio_wav_still_works() {
    // WAV must always work (no feature gate)
    let dir = std::env::temp_dir();
    let path = dir.join("test_wav_still_works_audio_formats.wav");

    let data_size: u32 = 2; // 1 sample x 2 bytes (16-bit)
    let chunk_size: u32 = 36 + data_size;
    let mut data = Vec::new();
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&chunk_size.to_le_bytes());
    data.extend_from_slice(b"WAVE");
    data.extend_from_slice(b"fmt ");
    data.extend_from_slice(&16u32.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes()); // PCM
    data.extend_from_slice(&1u16.to_le_bytes()); // mono
    data.extend_from_slice(&16000u32.to_le_bytes()); // sample rate
    data.extend_from_slice(&32000u32.to_le_bytes()); // byte rate
    data.extend_from_slice(&2u16.to_le_bytes()); // block align
    data.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    data.extend_from_slice(b"data");
    data.extend_from_slice(&data_size.to_le_bytes());
    data.extend_from_slice(&0i16.to_le_bytes()); // 1 zero sample

    std::fs::write(&path, &data).expect("write temp WAV");
    let result = oxiwhisper::audio::load_audio(&path).expect("load WAV via load_audio");
    let _ = std::fs::remove_file(&path);
    // 1-sample WAV at 16 kHz — no resampling needed, should produce exactly 1 sample
    assert_eq!(result.len(), 1, "1-sample 16kHz WAV should yield 1 sample");
}

#[test]
fn test_unknown_format_returns_error() {
    let dir = std::env::temp_dir();
    let path = dir.join("test_unknown_format_audio_formats.xyz");
    // Write a PNG header — not an audio format we support
    std::fs::write(&path, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").expect("write PNG header");
    let result = oxiwhisper::audio::load_audio(&path);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err(), "unknown format must return error");
}

#[test]
fn test_empty_file_returns_error() {
    let dir = std::env::temp_dir();
    let path = dir.join("test_empty_audio_formats.dat");
    std::fs::write(&path, b"").expect("write empty file");
    let result = oxiwhisper::audio::load_audio(&path);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err(), "empty file must return error");
}
