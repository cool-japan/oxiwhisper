use crate::types::OxiWhisperError;
use std::f32::consts::PI;
use std::sync::OnceLock;

/// Audio sample rate assumed by Whisper (16 kHz).
pub const WHISPER_SAMPLE_RATE: usize = 16000;
/// FFT window size used for the mel spectrogram (400 samples = 25 ms at 16 kHz).
pub const WHISPER_N_FFT: usize = 400;
/// Hop length between successive STFT frames (160 samples = 10 ms at 16 kHz).
pub const WHISPER_HOP_LENGTH: usize = 160;
/// Number of mel filterbank channels of the classic Whisper models.
///
/// This is the value for `tiny` … `large-v2`; `large-v3` uses 128. Code that
/// has access to a model must derive the channel count from
/// `Hparams::n_mels` (or, as [`log_mel_spectrogram`] does, from the shape of
/// the filter bank) instead of assuming this constant.
pub const WHISPER_N_MELS: usize = 80;
/// Maximum audio chunk length processed by Whisper in one pass (30 seconds).
pub const WHISPER_CHUNK_LENGTH: usize = 30; // seconds
/// Number of mel frames in one full Whisper window (30 s / 10 ms = 3000).
pub const WHISPER_N_FRAMES: usize = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_LENGTH / WHISPER_HOP_LENGTH;

/// Pre-computed Hann window of length WHISPER_N_FFT (400).
fn hann_window() -> &'static [f32] {
    static HANN: OnceLock<Vec<f32>> = OnceLock::new();
    HANN.get_or_init(|| {
        (0..WHISPER_N_FFT)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / WHISPER_N_FFT as f32).cos()))
            .collect()
    })
}

/// Number of mel channels implied by a flat `[n_mels, n_fft/2+1]` filter bank.
///
/// # Errors
///
/// Returns an error when the filter bank is empty or its length is not a
/// multiple of `WHISPER_N_FFT / 2 + 1` — i.e. it cannot describe a whole number
/// of mel channels.
pub fn n_mels_from_filters(mel_filters: &[f32]) -> Result<usize, OxiWhisperError> {
    let n_bins = WHISPER_N_FFT / 2 + 1;
    if mel_filters.is_empty() || !mel_filters.len().is_multiple_of(n_bins) {
        return Err(OxiWhisperError::InvalidModel(format!(
            "mel filter bank has {} coefficients, which is not a positive multiple of \
             the {n_bins} FFT bins produced by an {WHISPER_N_FFT}-point real FFT",
            mel_filters.len()
        )));
    }
    Ok(mel_filters.len() / n_bins)
}

/// Fetch `audio[idx]` with `center=True` reflect padding.
///
/// `idx` is an index into the *padded* signal, whose first
/// `WHISPER_N_FFT / 2` samples are the reflection of the signal start. Indices
/// past the end of the audio read as silence, matching a caller that has
/// zero-padded its input to the full 30 s window.
#[inline]
fn padded_sample(audio: &[f32], idx: usize) -> f32 {
    const PAD: usize = WHISPER_N_FFT / 2; // 200
    if idx < PAD {
        // Reflect around sample 0 (torch.stft's default "reflect" mode):
        // padded[PAD - k] == audio[k].
        let mirrored = PAD - idx;
        audio.get(mirrored).copied().unwrap_or(0.0)
    } else {
        audio.get(idx - PAD).copied().unwrap_or(0.0)
    }
}

/// Compute the log-mel spectrogram of 16 kHz mono f32 audio.
///
/// Mirrors OpenAI's `whisper.audio.log_mel_spectrogram`:
///
/// * `center=True` reflect padding of `n_fft/2 = 200` samples before the first
///   frame, so frame `i` is centred on sample `i * 160`;
/// * a **400-point** real FFT over the Hann-windowed frame — the window is *not*
///   zero-padded to 512. Padding to the next power of two changes the bin
///   spacing from 40 Hz to 31.25 Hz while the filter bank still assumes 40 Hz,
///   which warped every frequency by a factor 1.28 (a 1 kHz tone landed in the
///   1254 Hz band);
/// * output padded to the fixed 3000-frame Whisper window, using the same value
///   a zero-padded tail would have produced.
///
/// Output shape: `[n_mels, 3000]`, row-major, where `n_mels` is derived from the
/// filter bank (80 for `tiny`…`large-v2`, 128 for `large-v3`).
///
/// # Errors
///
/// Returns [`OxiWhisperError::InvalidModel`] when `mel_filters` does not
/// describe a whole number of mel channels over `WHISPER_N_FFT / 2 + 1` bins.
pub fn log_mel_spectrogram(
    audio: &[f32],
    mel_filters: &[f32],
) -> Result<Vec<f32>, OxiWhisperError> {
    let n_mels = n_mels_from_filters(mel_filters)?;

    // Compute every frame whose 400-sample window still overlaps real audio —
    // frame `i` spans `audio[i*160 - 200 .. i*160 + 200]`, so the last such
    // frame is `ceil((len + 200) / 160) - 1`. Everything after that is pure
    // silence and can be filled analytically instead of transformed.
    let n_frames = (audio.len() + WHISPER_N_FFT / 2)
        .div_ceil(WHISPER_HOP_LENGTH)
        .clamp(1, WHISPER_N_FRAMES);
    let mut spec = compute_log_mel(audio, mel_filters, n_mels, n_frames)?;

    if n_frames == WHISPER_N_FRAMES {
        return Ok(spec);
    }

    // Pad (or trim) to the fixed 3000-frame encoder window. The pad value is
    // exactly what silence produces, so the result is identical to zero-padding
    // the audio to 30 s before computing the spectrogram: a zero-energy bin is
    // clamped to `max(log10(1e-10), log_max - 8)` and then mapped by the final
    // affine step `(v + 4) / 4`. Recover `log_max` from the normalised maximum.
    let max_spec = spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_max = max_spec * 4.0 - 4.0;
    let floor = ((log_max - 8.0).max(-10.0) + 4.0) / 4.0;
    let mut padded = vec![floor; n_mels * WHISPER_N_FRAMES];
    let copy = n_frames.min(WHISPER_N_FRAMES);
    for m in 0..n_mels {
        padded[m * WHISPER_N_FRAMES..m * WHISPER_N_FRAMES + copy]
            .copy_from_slice(&spec[m * n_frames..m * n_frames + copy]);
    }
    spec = padded;
    Ok(spec)
}

/// Compute the log-mel spectrogram **without** padding to the 3000-frame window.
///
/// Same front-end as [`log_mel_spectrogram`] but the frame count follows the
/// input length (`audio.len() / 160`, capped at 3000). Intended for analysis
/// paths — embedding extraction, VAD, diarization — where running the encoder
/// over 30 s of mostly-silence would be pure overhead. Transcription must use
/// [`log_mel_spectrogram`]: Whisper's encoder was trained on the full window.
///
/// Output shape: `[n_mels, n_frames]`, row-major.
///
/// # Errors
///
/// Same as [`log_mel_spectrogram`].
pub fn log_mel_spectrogram_unpadded(
    audio: &[f32],
    mel_filters: &[f32],
) -> Result<Vec<f32>, OxiWhisperError> {
    let n_mels = n_mels_from_filters(mel_filters)?;
    let n_frames = n_frames_for_samples(audio.len());
    compute_log_mel(audio, mel_filters, n_mels, n_frames)
}

/// Core STFT → mel → log pipeline for an explicit frame count.
fn compute_log_mel(
    audio: &[f32],
    mel_filters: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> Result<Vec<f32>, OxiWhisperError> {
    let n_bins = WHISPER_N_FFT / 2 + 1;

    let hann = hann_window();
    let mut magnitudes = vec![0.0f32; n_frames * n_bins];

    // Real-valued windowed buffer, reused per frame. Exactly WHISPER_N_FFT long:
    // `oxifft::rfft` handles non-power-of-two transform sizes, so there is no
    // need (and no excuse) to zero-pad to 512.
    let mut windowed_real = vec![0.0f32; WHISPER_N_FFT];

    for frame_idx in 0..n_frames {
        let start = frame_idx * WHISPER_HOP_LENGTH;

        for (i, slot) in windowed_real.iter_mut().enumerate() {
            *slot = padded_sample(audio, start + i) * hann[i];
        }

        let spectrum = oxifft::rfft::<f32>(&windowed_real);

        for bin in 0..n_bins {
            magnitudes[frame_idx * n_bins + bin] =
                spectrum[bin].re * spectrum[bin].re + spectrum[bin].im * spectrum[bin].im;
        }
    }

    let mut mel_spec = vec![0.0f32; n_mels * n_frames];

    // Loop order: mel x frame x bin -> good cache behaviour for mel_filters (row-major)
    for mel_idx in 0..n_mels {
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

    Ok(mel_spec)
}

/// Returns the number of mel filterbank channels (`WHISPER_N_MELS = 80`).
///
/// Only valid for the classic 80-channel models; prefer
/// [`n_mels_from_filters`] or `Hparams::n_mels` when a model is available.
pub fn n_mels() -> usize {
    WHISPER_N_MELS
}

/// Compute the number of mel spectrogram frames for a given number of audio samples.
///
/// This is the *unpadded* frame count produced by
/// [`log_mel_spectrogram_unpadded`] — `n_samples / 160`, clamped to at least one
/// frame and at most 3000 (= 30 s at a 10 ms hop). It matches
/// `torch.stft(center=True)` followed by dropping the final frame, which is what
/// OpenAI's front-end does.
pub fn n_frames_for_samples(n_samples: usize) -> usize {
    (n_samples / WHISPER_HOP_LENGTH).clamp(1, WHISPER_N_FRAMES)
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
    fn test_mel_spectrogram_is_padded_to_full_window() {
        // 1 second of 440Hz sine at 16kHz
        let audio = sine_wave(440.0, WHISPER_SAMPLE_RATE, WHISPER_SAMPLE_RATE);
        let n_bins = WHISPER_N_FFT / 2 + 1; // 201
        let mel_filters = fake_mel_filters(WHISPER_N_MELS, n_bins);
        assert_eq!(mel_filters.len(), WHISPER_N_MELS * n_bins);

        let result = log_mel_spectrogram(&audio, &mel_filters).expect("mel should succeed");

        assert_eq!(
            result.len(),
            WHISPER_N_MELS * WHISPER_N_FRAMES,
            "output must always cover the full 30 s encoder window"
        );
    }

    #[test]
    fn test_mel_spectrogram_unpadded_shape() {
        let audio = sine_wave(440.0, WHISPER_SAMPLE_RATE, WHISPER_SAMPLE_RATE);
        let n_bins = WHISPER_N_FFT / 2 + 1;
        let mel_filters = fake_mel_filters(WHISPER_N_MELS, n_bins);

        let result =
            log_mel_spectrogram_unpadded(&audio, &mel_filters).expect("mel should succeed");

        let expected_frames = n_frames_for_samples(audio.len());
        assert_eq!(result.len(), WHISPER_N_MELS * expected_frames);
    }

    #[test]
    fn test_mel_padding_matches_zero_padded_audio() {
        // Padding the mel output must be numerically identical to zero-padding
        // the audio to the full 30 s window and computing the spectrogram.
        let n_bins = WHISPER_N_FFT / 2 + 1;
        let mel_filters = crate::mel_filters::generate_mel_filters();
        assert_eq!(mel_filters.len(), WHISPER_N_MELS * n_bins);

        let audio = sine_wave(1000.0, WHISPER_SAMPLE_RATE, WHISPER_SAMPLE_RATE);
        let mut padded_audio = audio.clone();
        padded_audio.resize(WHISPER_SAMPLE_RATE * WHISPER_CHUNK_LENGTH, 0.0);

        let from_short = log_mel_spectrogram(&audio, &mel_filters).expect("short");
        let from_padded = log_mel_spectrogram(&padded_audio, &mel_filters).expect("padded");

        assert_eq!(from_short.len(), from_padded.len());
        for (i, (&a, &b)) in from_short.iter().zip(from_padded.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "mel[{i}] differs: padded-mel={a} padded-audio={b}"
            );
        }
    }

    #[test]
    fn test_mel_1khz_tone_lands_in_correct_band() {
        // Regression for the 512-point zero-padded FFT: bin spacing became
        // 31.25 Hz while the filter bank assumes 40 Hz, warping every frequency
        // by 1.28x (a 1 kHz tone peaked in the 1254 Hz band).
        //
        // With the correct 400-point transform, the HTK mel bank used by
        // `generate_mel_filters` puts band 28 at ~1026.9 Hz and band 27 at
        // ~971.7 Hz, so a 1 kHz tone must peak in one of those.
        let n_bins = WHISPER_N_FFT / 2 + 1;
        let mel_filters = crate::mel_filters::generate_mel_filters();
        let audio = sine_wave(1000.0, WHISPER_SAMPLE_RATE, WHISPER_SAMPLE_RATE / 2);

        let spec = log_mel_spectrogram_unpadded(&audio, &mel_filters).expect("mel should succeed");
        let n_frames = spec.len() / WHISPER_N_MELS;
        assert!(n_frames > 10);

        // Average each band over the steady-state middle of the signal.
        let lo = n_frames / 4;
        let hi = n_frames * 3 / 4;
        let mut best_band = 0usize;
        let mut best_energy = f32::NEG_INFINITY;
        for m in 0..WHISPER_N_MELS {
            let row = &spec[m * n_frames..(m + 1) * n_frames];
            let energy: f32 = row[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
            if energy > best_energy {
                best_energy = energy;
                best_band = m;
            }
        }

        // Centre frequency of the peak band, from the same construction the
        // filter generator uses (HTK mel scale, 0..8000 Hz, 80 bands).
        let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f64| 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);
        let mel_max = hz_to_mel(8000.0);
        let centre_hz = mel_to_hz(mel_max * (best_band + 1) as f64 / (WHISPER_N_MELS + 1) as f64);

        assert!(
            (26..=30).contains(&best_band),
            "1 kHz tone peaked in band {best_band} (~{centre_hz:.0} Hz); expected the band \
             centred near 1026 Hz (band 28) within one band. n_bins={n_bins}"
        );
    }

    #[test]
    fn test_mel_no_nan() {
        let audio = sine_wave(1000.0, WHISPER_SAMPLE_RATE, WHISPER_SAMPLE_RATE);
        let n_bins = WHISPER_N_FFT / 2 + 1;
        let mel_filters = fake_mel_filters(WHISPER_N_MELS, n_bins);

        let result = log_mel_spectrogram(&audio, &mel_filters).expect("mel should succeed");

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

        let result =
            log_mel_spectrogram_unpadded(&audio, &mel_filters).expect("mel should succeed");

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

        let result = log_mel_spectrogram(&audio, &mel_filters).expect("mel should succeed");

        for (i, val) in result.iter().enumerate() {
            assert!(val.is_finite(), "NaN or Inf at index {i} for silence input");
        }
    }

    #[test]
    fn test_mel_rejects_misshaped_filter_bank() {
        // Previously an `assert_eq!` — reachable from the public `transcribe()`
        // with any model that is not 80-mel (large-v3 has 128).
        let audio = vec![0.0f32; WHISPER_SAMPLE_RATE];
        let err = log_mel_spectrogram(&audio, &[0.0f32; 137])
            .expect_err("a filter bank that is not a multiple of 201 must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("201"),
            "error should name the bin count: {msg}"
        );
    }

    #[test]
    fn test_mel_accepts_128_mel_filter_bank() {
        // large-v3 uses 128 mel channels — this must work, not panic.
        let n_bins = WHISPER_N_FFT / 2 + 1;
        let mel_filters = fake_mel_filters(128, n_bins);
        let audio = sine_wave(440.0, WHISPER_SAMPLE_RATE, WHISPER_SAMPLE_RATE / 2);
        let result = log_mel_spectrogram(&audio, &mel_filters).expect("128-mel must be supported");
        assert_eq!(result.len(), 128 * WHISPER_N_FRAMES);
    }

    #[test]
    fn test_n_mels_from_filters() {
        let n_bins = WHISPER_N_FFT / 2 + 1;
        assert_eq!(
            n_mels_from_filters(&vec![0.0f32; 80 * n_bins]).expect("80 mels"),
            80
        );
        assert_eq!(
            n_mels_from_filters(&vec![0.0f32; 128 * n_bins]).expect("128 mels"),
            128
        );
        assert!(n_mels_from_filters(&[]).is_err());
        assert!(n_mels_from_filters(&[0.0f32; 7]).is_err());
    }

    #[test]
    fn test_n_frames_for_samples() {
        // 16000 samples (1 second) -> 16000 / 160 = 100 frames, matching
        // torch.stft(center=True) with the trailing frame dropped.
        let frames = n_frames_for_samples(WHISPER_SAMPLE_RATE);
        assert_eq!(frames, 100);

        // 30 seconds -> 480000 / 160 = 3000
        let max_samples = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_LENGTH;
        let frames_max = n_frames_for_samples(max_samples);
        assert_eq!(frames_max, WHISPER_N_FRAMES);

        // Longer than 30 s is clamped
        assert_eq!(n_frames_for_samples(max_samples * 2), WHISPER_N_FRAMES);

        // 0 samples -> at least one frame
        assert_eq!(n_frames_for_samples(0), 1);
    }

    #[test]
    fn test_reflect_padding_is_applied() {
        // padded_sample must mirror around sample 0 for the first 200 samples.
        let audio: Vec<f32> = (0..10).map(|i| i as f32).collect();
        const PAD: usize = WHISPER_N_FFT / 2;
        assert_eq!(padded_sample(&audio, PAD), 0.0);
        assert_eq!(padded_sample(&audio, PAD + 3), 3.0);
        assert_eq!(padded_sample(&audio, PAD - 3), 3.0);
        // Beyond the end reads as silence.
        assert_eq!(padded_sample(&audio, PAD + 50), 0.0);
    }
}
