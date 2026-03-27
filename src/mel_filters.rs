//! Programmatic generation of the Whisper mel filter bank.
//!
//! ONNX models do not embed mel filters like GGML models do, so this module
//! generates the standard 80-band mel filter bank used by OpenAI Whisper.

use crate::mel::{WHISPER_N_FFT, WHISPER_N_MELS, WHISPER_SAMPLE_RATE};

/// Total number of f32 elements in the flat mel filter bank array.
/// Shape: [WHISPER_N_MELS, WHISPER_N_FFT / 2 + 1] = [80, 201].
pub const WHISPER_MEL_FILTER_SIZE: usize = WHISPER_N_MELS * (WHISPER_N_FFT / 2 + 1);

/// Convert a frequency in Hz to the mel scale (HTK formula).
#[inline]
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert a mel-scale value back to Hz.
#[inline]
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Generate the standard Whisper mel filter bank programmatically.
///
/// Returns a flat `Vec<f32>` of shape `[n_mels, n_bins]` in row-major order,
/// where `n_mels = 80` and `n_bins = n_fft / 2 + 1 = 201`.
///
/// The filter bank uses Slaney normalization: each triangular filter is
/// normalized by `2.0 / (upper_hz - lower_hz)` so that the area under
/// the triangle is 1.
pub fn generate_mel_filters() -> Vec<f32> {
    let n_mels = WHISPER_N_MELS;
    let n_fft = WHISPER_N_FFT;
    let sample_rate = WHISPER_SAMPLE_RATE;
    let n_bins = n_fft / 2 + 1; // 201

    let f_min: f64 = 0.0;
    let f_max: f64 = (sample_rate / 2) as f64; // 8000.0 Hz (Nyquist)

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // n_mels + 2 evenly spaced points in mel scale
    let n_points = n_mels + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * (i as f64) / ((n_points - 1) as f64))
        .collect();

    // Convert mel points back to Hz
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices (fractional, not floored)
    let fft_bins: Vec<f64> = hz_points
        .iter()
        .map(|&f| (n_fft as f64 + 1.0) * f / (sample_rate as f64))
        .collect();

    let mut filters = vec![0.0f32; n_mels * n_bins];

    for m in 0..n_mels {
        let left = fft_bins[m];
        let center = fft_bins[m + 1];
        let right = fft_bins[m + 2];

        // Slaney normalization factor: 2 / (upper_hz - lower_hz)
        let enorm = 2.0 / (hz_points[m + 2] - hz_points[m]);

        for k in 0..n_bins {
            let bin = k as f64;
            let weight = if bin >= left && bin < center {
                // Rising slope: left edge to center
                if (center - left).abs() < f64::EPSILON {
                    0.0
                } else {
                    (bin - left) / (center - left)
                }
            } else if bin >= center && bin <= right {
                // Falling slope: center to right edge
                if (right - center).abs() < f64::EPSILON {
                    0.0
                } else {
                    (right - bin) / (right - center)
                }
            } else {
                0.0
            };

            filters[m * n_bins + k] = (weight * enorm) as f32;
        }
    }

    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_length() {
        let filters = generate_mel_filters();
        assert_eq!(filters.len(), 80 * 201);
        assert_eq!(filters.len(), WHISPER_MEL_FILTER_SIZE);
    }

    #[test]
    fn test_all_non_negative() {
        let filters = generate_mel_filters();
        for (i, &v) in filters.iter().enumerate() {
            assert!(v >= 0.0, "Negative value {v} at index {i}",);
        }
    }

    #[test]
    fn test_each_band_has_nonzero() {
        let filters = generate_mel_filters();
        let n_bins = WHISPER_N_FFT / 2 + 1;
        for m in 0..WHISPER_N_MELS {
            let row = &filters[m * n_bins..(m + 1) * n_bins];
            let has_nonzero = row.iter().any(|&v| v > 0.0);
            assert!(has_nonzero, "Mel band {m} has no non-zero values",);
        }
    }

    #[test]
    fn test_no_nan_or_inf() {
        let filters = generate_mel_filters();
        for (i, &v) in filters.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite value {v} at index {i}",);
        }
    }

    #[test]
    fn test_mel_hz_roundtrip() {
        // Verify the mel<->Hz conversions are consistent
        let test_freqs = [0.0, 100.0, 700.0, 1000.0, 4000.0, 8000.0];
        for &f in &test_freqs {
            let mel = hz_to_mel(f);
            let recovered = mel_to_hz(mel);
            let diff = (recovered - f).abs();
            assert!(
                diff < 1e-6,
                "Roundtrip failed for {f} Hz: got {recovered} (diff={diff})",
            );
        }
    }

    #[test]
    fn test_filter_shape_properties() {
        let filters = generate_mel_filters();
        let n_bins = WHISPER_N_FFT / 2 + 1;

        // Lower mel bands should have peak energy at lower frequency bins
        // and higher mel bands at higher bins. Verify monotonicity of
        // the center-of-mass across bands.
        let mut prev_centroid = 0.0_f64;
        for m in 0..WHISPER_N_MELS {
            let row = &filters[m * n_bins..(m + 1) * n_bins];
            let total: f64 = row.iter().map(|&v| v as f64).sum();
            if total < 1e-12 {
                continue;
            }
            let centroid: f64 = row
                .iter()
                .enumerate()
                .map(|(k, &v)| k as f64 * v as f64)
                .sum::<f64>()
                / total;
            assert!(
                centroid >= prev_centroid,
                "Centroid for mel band {m} ({centroid}) is less than previous ({prev_centroid})",
            );
            prev_centroid = centroid;
        }
    }
}
