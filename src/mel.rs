use std::f32::consts::PI;
use std::sync::OnceLock;

/// Audio sample rate assumed by Whisper (16 kHz).
pub const WHISPER_SAMPLE_RATE: usize = 16000;
/// FFT window size used for the mel spectrogram (400 samples = 25 ms at 16 kHz).
pub const WHISPER_N_FFT: usize = 400;
/// Hop length between successive STFT frames (160 samples = 10 ms at 16 kHz).
pub const WHISPER_HOP_LENGTH: usize = 160;
/// Number of mel filterbank channels in the Whisper mel spectrogram.
pub const WHISPER_N_MELS: usize = 80;
/// Maximum audio chunk length processed by Whisper in one pass (30 seconds).
pub const WHISPER_CHUNK_LENGTH: usize = 30; // seconds

/// Pre-computed Hann window of length WHISPER_N_FFT (400).
fn hann_window() -> &'static [f32] {
    static HANN: OnceLock<Vec<f32>> = OnceLock::new();
    HANN.get_or_init(|| {
        (0..WHISPER_N_FFT)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / WHISPER_N_FFT as f32).cos()))
            .collect()
    })
}

/// Compute log-mel spectrogram from 16kHz mono f32 audio.
/// Only processes actual audio frames (not always 30 seconds).
/// Output shape: [n_mels, n_frames_actual]  where n_frames_actual <= 3000
pub fn log_mel_spectrogram(audio: &[f32], mel_filters: &[f32]) -> Vec<f32> {
    let max_samples = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_LENGTH;
    let max_frames = max_samples / WHISPER_HOP_LENGTH; // 3000

    // Number of frames needed for actual audio (+ 1 frame margin for windowing)
    let n_frames = (audio.len().div_ceil(WHISPER_HOP_LENGTH) + 1).min(max_frames);

    let padded_len = WHISPER_N_FFT.next_power_of_two(); // 512

    let hann = hann_window();

    let n_bins = WHISPER_N_FFT / 2 + 1;
    let mut magnitudes = vec![0.0f32; n_frames * n_bins];

    // Pre-allocate real-valued windowed buffer, reused per frame
    let mut windowed_real = vec![0.0f32; padded_len];

    for frame_idx in 0..n_frames {
        let start = frame_idx * WHISPER_HOP_LENGTH;

        // Zero-fill and apply Hann window
        windowed_real.fill(0.0);
        for i in 0..WHISPER_N_FFT {
            let s = if start + i < audio.len() {
                audio[start + i]
            } else {
                0.0
            };
            windowed_real[i] = s * hann[i];
        }

        // Use oxifft::rfft directly for real-to-complex FFT
        let spectrum = oxifft::rfft::<f32>(&windowed_real);

        for bin in 0..n_bins {
            magnitudes[frame_idx * n_bins + bin] =
                spectrum[bin].re * spectrum[bin].re + spectrum[bin].im * spectrum[bin].im;
        }
    }

    assert_eq!(
        mel_filters.len(),
        WHISPER_N_MELS * n_bins,
        "mel_filters size mismatch"
    );

    let mut mel_spec = vec![0.0f32; WHISPER_N_MELS * n_frames];

    // Loop order: mel x frame x bin -> good cache behaviour for mel_filters (row-major)
    for mel_idx in 0..WHISPER_N_MELS {
        let filt = &mel_filters[mel_idx * n_bins..(mel_idx + 1) * n_bins];
        for frame_idx in 0..n_frames {
            let mag = &magnitudes[frame_idx * n_bins..(frame_idx + 1) * n_bins];
            let mut sum = 0.0f32;
            for b in 0..n_bins {
                sum += filt[b] * mag[b];
            }
            mel_spec[mel_idx * n_frames + frame_idx] = sum;
        }
    }

    // Log transform
    let max_val = mel_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_max = max_val.max(1e-10).log10();

    for v in mel_spec.iter_mut() {
        *v = v.max(1e-10).log10();
        *v = v.max(log_max - 8.0);
        *v = (*v + 4.0) / 4.0;
    }

    mel_spec
}

/// Returns the number of mel filterbank channels (`WHISPER_N_MELS = 80`).
pub fn n_mels() -> usize {
    WHISPER_N_MELS
}

/// Compute the number of mel spectrogram frames for a given number of audio samples.
///
/// The result is clamped to at most 3000 frames (= 30 s at 10 ms hop length).
pub fn n_frames_for_samples(n_samples: usize) -> usize {
    (n_samples.div_ceil(WHISPER_HOP_LENGTH) + 1)
        .min(WHISPER_SAMPLE_RATE * WHISPER_CHUNK_LENGTH / WHISPER_HOP_LENGTH)
}

/// Returns the audio context length in encoder frames (1500 = 30 s at 20 ms stride).
pub fn n_audio_ctx() -> usize {
    1500
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a sine wave at the given frequency and sample rate.
    fn sine_wave(freq_hz: f32, sample_rate: usize, n_samples: usize) -> Vec<f32> {
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * PI * freq_hz * t).sin()
            })
            .collect()
    }

    /// Create a simple mel filter bank of shape [n_mels, n_bins] with triangular-ish patterns.
    /// This is not a real mel filter bank but is sufficient for testing shapes and NaN checks.
    fn fake_mel_filters(n_mels: usize, n_bins: usize) -> Vec<f32> {
        let mut filters = vec![0.0f32; n_mels * n_bins];
        for mel in 0..n_mels {
            // Each mel filter has a small triangular response centered at a bin
            let center = (mel + 1) * n_bins / (n_mels + 2);
            let width = n_bins / (n_mels + 2);
            let lo = center.saturating_sub(width);
            let hi = (center + width).min(n_bins - 1);
            for bin in lo..=hi {
                let dist = if bin <= center {
                    (bin - lo) as f32 / (center - lo).max(1) as f32
                } else {
                    (hi - bin) as f32 / (hi - center).max(1) as f32
                };
                filters[mel * n_bins + bin] = dist.max(0.0);
            }
        }
        filters
    }

    #[test]
    fn test_mel_spectrogram_shape() {
        // 1 second of 440Hz sine at 16kHz
        let audio = sine_wave(440.0, WHISPER_SAMPLE_RATE, WHISPER_SAMPLE_RATE);
        let n_bins = WHISPER_N_FFT / 2 + 1; // 201
        let mel_filters = fake_mel_filters(WHISPER_N_MELS, n_bins);
        assert_eq!(mel_filters.len(), WHISPER_N_MELS * n_bins);

        let result = log_mel_spectrogram(&audio, &mel_filters);

        let expected_frames = n_frames_for_samples(audio.len());
        assert_eq!(
            result.len(),
            WHISPER_N_MELS * expected_frames,
            "output length should be n_mels * n_frames"
        );
    }

    #[test]
    fn test_mel_no_nan() {
        let audio = sine_wave(1000.0, WHISPER_SAMPLE_RATE, WHISPER_SAMPLE_RATE);
        let n_bins = WHISPER_N_FFT / 2 + 1;
        let mel_filters = fake_mel_filters(WHISPER_N_MELS, n_bins);

        let result = log_mel_spectrogram(&audio, &mel_filters);

        for (i, val) in result.iter().enumerate() {
            assert!(
                val.is_finite(),
                "NaN or Inf found at index {i}, value={val}"
            );
        }
    }

    #[test]
    fn test_mel_short_audio() {
        // Very short audio: 160 samples (one hop length)
        let audio = sine_wave(440.0, WHISPER_SAMPLE_RATE, WHISPER_HOP_LENGTH);
        let n_bins = WHISPER_N_FFT / 2 + 1;
        let mel_filters = fake_mel_filters(WHISPER_N_MELS, n_bins);

        let result = log_mel_spectrogram(&audio, &mel_filters);

        let expected_frames = n_frames_for_samples(audio.len());
        assert_eq!(result.len(), WHISPER_N_MELS * expected_frames);
        assert!(expected_frames >= 1, "should have at least 1 frame");

        // No NaN/Inf
        for (i, val) in result.iter().enumerate() {
            assert!(val.is_finite(), "NaN or Inf at index {i}");
        }
    }

    #[test]
    fn test_mel_silence() {
        // All-zero audio should produce valid (not NaN) output
        let audio = vec![0.0f32; WHISPER_SAMPLE_RATE];
        let n_bins = WHISPER_N_FFT / 2 + 1;
        let mel_filters = fake_mel_filters(WHISPER_N_MELS, n_bins);

        let result = log_mel_spectrogram(&audio, &mel_filters);

        for (i, val) in result.iter().enumerate() {
            assert!(val.is_finite(), "NaN or Inf at index {i} for silence input");
        }
    }

    #[test]
    fn test_n_frames_for_samples() {
        // 16000 samples (1 second) -> div_ceil(16000, 160) + 1 = 100 + 1 = 101
        let frames = n_frames_for_samples(WHISPER_SAMPLE_RATE);
        assert_eq!(frames, 101);

        // 30 seconds -> div_ceil(480000, 160) + 1 = 3000 + 1 = 3001, clamped to 3000
        let max_samples = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_LENGTH;
        let frames_max = n_frames_for_samples(max_samples);
        assert_eq!(frames_max, 3000);

        // 0 samples -> div_ceil(0, 160) + 1 = 0 + 1 = 1
        let frames_zero = n_frames_for_samples(0);
        assert_eq!(frames_zero, 1);
    }
}
