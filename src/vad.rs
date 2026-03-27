// SPDX-License-Identifier: Apache-2.0
// Copyright (c) COOLJAPAN OU (Team Kitasan)

//! Voice Activity Detection (VAD) module.
//!
//! Detects speech vs silence regions in audio so that silent chunks can be
//! skipped before running the expensive encoder.  The algorithm is based on
//! frame-level RMS energy with configurable thresholds and minimum duration
//! filters.

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for voice activity detection.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// RMS energy threshold for speech detection (0.0–1.0).
    /// Frames with RMS below this are considered silence.
    /// Default: 0.01
    pub energy_threshold: f32,
    /// Minimum speech segment duration in milliseconds.
    /// Segments shorter than this are discarded.
    /// Default: 250
    pub min_speech_ms: usize,
    /// Minimum silence duration in milliseconds to split segments.
    /// Silence shorter than this is absorbed into the speech segment.
    /// Default: 300
    pub min_silence_ms: usize,
    /// Frame size in milliseconds for energy computation.
    /// Default: 30
    pub frame_size_ms: usize,
    /// Enable adaptive threshold estimation based on noise floor.
    /// When true, `energy_threshold` is ignored and the threshold is computed
    /// as `noise_floor * noise_margin`.
    /// Default: false
    pub adaptive_threshold: bool,
    /// Multiplier for the noise floor when `adaptive_threshold` is true.
    /// Higher values → less sensitive (fewer false positives).
    /// Default: 3.0
    pub noise_margin: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            energy_threshold: 0.01,
            min_speech_ms: 250,
            min_silence_ms: 300,
            frame_size_ms: 30,
            adaptive_threshold: false,
            noise_margin: 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Speech segment
// ---------------------------------------------------------------------------

/// A detected speech segment with start and end sample indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechSegment {
    /// Start sample index (inclusive).
    pub start: usize,
    /// End sample index (exclusive).
    pub end: usize,
}

impl SpeechSegment {
    /// Duration of the segment in milliseconds.
    pub fn duration_ms(&self, sample_rate: usize) -> usize {
        if sample_rate == 0 {
            return 0;
        }
        let samples = self.end.saturating_sub(self.start);
        samples * 1000 / sample_rate
    }

    /// Duration of the segment in seconds.
    pub fn duration_seconds(&self, sample_rate: usize) -> f32 {
        if sample_rate == 0 {
            return 0.0;
        }
        let samples = self.end.saturating_sub(self.start);
        samples as f32 / sample_rate as f32
    }
}

// ---------------------------------------------------------------------------
// Core helpers
// ---------------------------------------------------------------------------

/// Compute the RMS energy of a slice of samples.
fn rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Estimate the noise floor from audio by computing RMS of each frame
/// and taking the value at the 10th percentile (quietest 10% of frames).
fn estimate_noise_floor(audio: &[f32], frame_size: usize) -> f32 {
    if audio.is_empty() || frame_size == 0 {
        return 0.0;
    }

    let n_frames = audio.len() / frame_size;
    if n_frames == 0 {
        return 0.0;
    }

    let mut frame_energies: Vec<f32> = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let frame = &audio[i * frame_size..(i + 1) * frame_size];
        let rms = (frame.iter().map(|&s| s * s).sum::<f32>() / frame_size as f32).sqrt();
        frame_energies.push(rms);
    }

    frame_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // 10th percentile
    let idx = (n_frames as f32 * 0.1) as usize;
    let idx = idx.min(n_frames.saturating_sub(1));
    frame_energies[idx]
}

/// Convert a duration in milliseconds to a number of frames.
fn ms_to_frames(ms: usize, frame_size_ms: usize) -> usize {
    if frame_size_ms == 0 {
        return 0;
    }
    // Round up so that the minimum constraint is respected.
    ms.div_ceil(frame_size_ms)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Detect speech segments in audio.
///
/// Returns a list of [`SpeechSegment`] sample ranges containing speech.
/// Audio should be 16 kHz mono f32 PCM (but the `sample_rate` parameter
/// allows flexibility).
pub fn detect_speech(audio: &[f32], sample_rate: usize, config: &VadConfig) -> Vec<SpeechSegment> {
    if audio.is_empty() || sample_rate == 0 || config.frame_size_ms == 0 {
        return Vec::new();
    }

    let frame_size = sample_rate * config.frame_size_ms / 1000;
    if frame_size == 0 {
        return Vec::new();
    }

    // ------------------------------------------------------------------
    // Step 1 – determine effective threshold
    // ------------------------------------------------------------------
    let effective_threshold = if config.adaptive_threshold {
        let noise = estimate_noise_floor(audio, frame_size);
        (noise * config.noise_margin).max(1e-6) // avoid zero threshold
    } else {
        config.energy_threshold
    };

    // ------------------------------------------------------------------
    // Step 2 – compute per-frame RMS energy
    // ------------------------------------------------------------------
    let num_frames = audio.len().div_ceil(frame_size);
    let mut is_speech: Vec<bool> = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let start = i * frame_size;
        let end = (start + frame_size).min(audio.len());
        let energy = rms_energy(&audio[start..end]);
        is_speech.push(energy >= effective_threshold);
    }

    // ------------------------------------------------------------------
    // Step 3 – collect raw speech runs (frame indices)
    // ------------------------------------------------------------------
    let mut raw_runs: Vec<(usize, usize)> = Vec::new(); // (start_frame, end_frame) exclusive
    let mut run_start: Option<usize> = None;

    for (i, &speech) in is_speech.iter().enumerate() {
        match (speech, run_start) {
            (true, None) => {
                run_start = Some(i);
            }
            (false, Some(s)) => {
                raw_runs.push((s, i));
                run_start = None;
            }
            _ => {}
        }
    }
    // Close any trailing run.
    if let Some(s) = run_start {
        raw_runs.push((s, num_frames));
    }

    // ------------------------------------------------------------------
    // Step 4 – discard runs shorter than min_speech_ms
    // ------------------------------------------------------------------
    let min_speech_frames = ms_to_frames(config.min_speech_ms, config.frame_size_ms);
    let filtered: Vec<(usize, usize)> = raw_runs
        .into_iter()
        .filter(|(s, e)| e - s >= min_speech_frames)
        .collect();

    if filtered.is_empty() {
        return Vec::new();
    }

    // ------------------------------------------------------------------
    // Step 5 – merge segments separated by silence shorter than min_silence_ms
    // ------------------------------------------------------------------
    let min_silence_frames = ms_to_frames(config.min_silence_ms, config.frame_size_ms);
    let mut merged: Vec<(usize, usize)> = Vec::new();
    merged.push(filtered[0]);

    for &(s, e) in &filtered[1..] {
        let last = merged
            .last_mut()
            .expect("merged is non-empty after initial push");
        let gap = s.saturating_sub(last.1);
        if gap < min_silence_frames {
            // Absorb the gap – extend the previous segment.
            last.1 = e;
        } else {
            merged.push((s, e));
        }
    }

    // ------------------------------------------------------------------
    // Step 6 – convert frame indices to sample indices
    // ------------------------------------------------------------------
    merged
        .into_iter()
        .map(|(s_frame, e_frame)| {
            let start_sample = s_frame * frame_size;
            let end_sample = (e_frame * frame_size).min(audio.len());
            SpeechSegment {
                start: start_sample,
                end: end_sample,
            }
        })
        .collect()
}

/// Extract speech-only audio by concatenating detected speech segments.
///
/// Adds a configurable `padding_samples` around each segment (clamped to
/// the audio boundaries).
pub fn extract_speech(
    audio: &[f32],
    segments: &[SpeechSegment],
    padding_samples: usize,
) -> Vec<f32> {
    let mut out = Vec::new();
    for seg in segments {
        let start = seg.start.saturating_sub(padding_samples);
        let end = (seg.end + padding_samples).min(audio.len());
        if start < end {
            out.extend_from_slice(&audio[start..end]);
        }
    }
    out
}

/// Returns `true` if the audio appears to contain speech based on overall
/// RMS energy.
pub fn has_speech(audio: &[f32], threshold: f32) -> bool {
    rms_energy(audio) >= threshold
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: usize = 16_000;

    /// Generate a sine tone at `freq` Hz for `duration_ms` milliseconds.
    fn tone(freq: f32, duration_ms: usize, amplitude: f32) -> Vec<f32> {
        let num_samples = SR * duration_ms / 1000;
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / SR as f32;
                amplitude * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    /// Generate silence for `duration_ms` milliseconds.
    fn silence(duration_ms: usize) -> Vec<f32> {
        vec![0.0f32; SR * duration_ms / 1000]
    }

    // --- basic ---

    #[test]
    fn silence_only_returns_empty() {
        let audio = silence(2000);
        let config = VadConfig::default();
        let segments = detect_speech(&audio, SR, &config);
        assert!(segments.is_empty());
    }

    #[test]
    fn constant_tone_returns_single_segment() {
        let audio = tone(440.0, 2000, 0.5);
        let config = VadConfig::default();
        let segments = detect_speech(&audio, SR, &config);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 0);
        // End should cover the entire audio (up to frame-boundary rounding).
        assert!(segments[0].end >= audio.len() - SR * config.frame_size_ms / 1000);
    }

    #[test]
    fn alternating_speech_silence_segments() {
        // 500ms tone, 500ms silence, 500ms tone  (silence > min_silence_ms=300)
        let mut audio = tone(440.0, 500, 0.5);
        audio.extend(silence(500));
        audio.extend(tone(440.0, 500, 0.5));

        let config = VadConfig::default();
        let segments = detect_speech(&audio, SR, &config);
        assert_eq!(
            segments.len(),
            2,
            "expected two speech segments, got {segments:?}"
        );
    }

    #[test]
    fn short_speech_is_filtered() {
        // 100ms tone (shorter than default min_speech_ms=250)
        let audio = tone(440.0, 100, 0.5);
        let config = VadConfig::default();
        let segments = detect_speech(&audio, SR, &config);
        assert!(segments.is_empty(), "short speech should be discarded");
    }

    #[test]
    fn short_silence_gaps_are_merged() {
        // Two speech bursts separated by 100ms silence (< min_silence_ms=300).
        let mut audio = tone(440.0, 500, 0.5);
        audio.extend(silence(100));
        audio.extend(tone(440.0, 500, 0.5));

        let config = VadConfig::default();
        let segments = detect_speech(&audio, SR, &config);
        assert_eq!(
            segments.len(),
            1,
            "short silence gap should be merged into one segment"
        );
    }

    // --- extract_speech ---

    #[test]
    fn extract_speech_concatenates_segments() {
        let mut audio = tone(440.0, 500, 0.5);
        audio.extend(silence(500));
        audio.extend(tone(880.0, 500, 0.5));

        let config = VadConfig::default();
        let segments = detect_speech(&audio, SR, &config);
        let extracted = extract_speech(&audio, &segments, 0);

        // Extracted length should be roughly 1000ms worth of samples.
        let expected_samples = SR; // 16000 samples for 1s
        let tolerance = SR * config.frame_size_ms / 1000; // one frame
        let diff = if extracted.len() > expected_samples {
            extracted.len() - expected_samples
        } else {
            expected_samples - extracted.len()
        };
        assert!(
            diff <= tolerance,
            "extracted length {} far from expected {expected_samples}",
            extracted.len()
        );
    }

    #[test]
    fn extract_speech_with_padding() {
        let segments = vec![SpeechSegment {
            start: 100,
            end: 200,
        }];
        let audio = vec![0.0f32; 300];
        let extracted = extract_speech(&audio, &segments, 10);
        // Should include samples 90..210
        assert_eq!(extracted.len(), 120);
    }

    #[test]
    fn extract_speech_padding_clamped_to_bounds() {
        let segments = vec![SpeechSegment { start: 5, end: 10 }];
        let audio = vec![1.0f32; 12];
        let extracted = extract_speech(&audio, &segments, 100);
        // Clamped to 0..12
        assert_eq!(extracted.len(), 12);
    }

    // --- has_speech ---

    #[test]
    fn has_speech_on_silence_returns_false() {
        let audio = silence(500);
        assert!(!has_speech(&audio, 0.01));
    }

    #[test]
    fn has_speech_on_tone_returns_true() {
        let audio = tone(440.0, 500, 0.5);
        assert!(has_speech(&audio, 0.01));
    }

    // --- edge cases ---

    #[test]
    fn empty_audio() {
        let config = VadConfig::default();
        let segments = detect_speech(&[], SR, &config);
        assert!(segments.is_empty());
        assert!(!has_speech(&[], 0.01));

        let extracted = extract_speech(&[], &[], 0);
        assert!(extracted.is_empty());
    }

    #[test]
    fn very_short_audio() {
        // Single sample – shorter than one frame.
        let audio = [0.5f32];
        let config = VadConfig::default();
        let segments = detect_speech(&audio, SR, &config);
        // Too short to meet min_speech_ms, so should be empty.
        assert!(segments.is_empty());
    }

    #[test]
    fn zero_sample_rate() {
        let audio = tone(440.0, 500, 0.5);
        let config = VadConfig::default();
        let segments = detect_speech(&audio, 0, &config);
        assert!(segments.is_empty());
    }

    #[test]
    fn segment_duration_methods() {
        let seg = SpeechSegment {
            start: 0,
            end: 16_000,
        };
        assert_eq!(seg.duration_ms(16_000), 1000);
        assert!((seg.duration_seconds(16_000) - 1.0).abs() < 1e-6);

        // Zero sample rate should not panic.
        assert_eq!(seg.duration_ms(0), 0);
        assert!((seg.duration_seconds(0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn custom_config_thresholds() {
        // With a very high threshold nothing should be detected.
        let audio = tone(440.0, 1000, 0.05);
        let config = VadConfig {
            energy_threshold: 0.9,
            ..VadConfig::default()
        };
        let segments = detect_speech(&audio, SR, &config);
        assert!(segments.is_empty());
    }

    #[test]
    fn multiple_segments_ordering() {
        // tone 500ms, silence 500ms, tone 500ms, silence 500ms, tone 500ms
        let mut audio = Vec::new();
        for i in 0..5 {
            if i % 2 == 0 {
                audio.extend(tone(440.0, 500, 0.5));
            } else {
                audio.extend(silence(500));
            }
        }
        let config = VadConfig::default();
        let segments = detect_speech(&audio, SR, &config);
        assert_eq!(segments.len(), 3);
        // Segments should be in order.
        for pair in segments.windows(2) {
            assert!(pair[0].end <= pair[1].start);
        }
    }

    // --- noise floor estimation ---

    #[test]
    fn test_estimate_noise_floor_silence() {
        // All zeros → noise floor should be 0.
        let audio = silence(2000);
        let frame_size = SR * 30 / 1000; // 30ms frames
        let floor = estimate_noise_floor(&audio, frame_size);
        assert!(
            floor.abs() < 1e-9,
            "noise floor for silence should be ~0, got {floor}"
        );
    }

    #[test]
    fn test_estimate_noise_floor_constant() {
        // Constant amplitude signal → all frames have the same RMS → floor = that RMS.
        let amplitude = 0.3;
        let audio = vec![amplitude; SR * 2]; // 2 seconds of constant signal
        let frame_size = SR * 30 / 1000;
        let floor = estimate_noise_floor(&audio, frame_size);
        assert!(
            (floor - amplitude).abs() < 1e-5,
            "noise floor for constant signal should be {amplitude}, got {floor}"
        );
    }

    #[test]
    fn test_estimate_noise_floor_with_noise() {
        // Mix of quiet background (low amplitude) and loud bursts.
        // The 10th percentile should be near the quiet level.
        let quiet_amplitude = 0.01;
        let loud_amplitude = 0.5;
        // 80% quiet, 20% loud — 10th percentile should be in the quiet range
        let mut audio = Vec::new();
        // 1600ms quiet
        audio.extend(std::iter::repeat_n(quiet_amplitude, SR * 1600 / 1000));
        // 400ms loud
        audio.extend(tone(440.0, 400, loud_amplitude));

        let frame_size = SR * 30 / 1000;
        let floor = estimate_noise_floor(&audio, frame_size);
        // Floor should be near the quiet amplitude
        assert!(
            floor < quiet_amplitude * 2.0,
            "noise floor should be near quiet level (~{quiet_amplitude}), got {floor}"
        );
    }

    #[test]
    fn test_adaptive_threshold_detects_speech() {
        // Speech (loud tone) over a noise floor (quiet constant signal).
        // Adaptive threshold should detect the loud part as speech.
        let mut audio = Vec::new();
        // 1 second of quiet noise
        let noise_level = 0.005;
        audio.extend(vec![noise_level; SR]);
        // 1 second of loud speech-like tone
        audio.extend(tone(440.0, 1000, 0.5));
        // 1 second of quiet noise again
        audio.extend(vec![noise_level; SR]);

        let config = VadConfig {
            adaptive_threshold: true,
            noise_margin: 3.0,
            min_speech_ms: 250,
            min_silence_ms: 300,
            frame_size_ms: 30,
            energy_threshold: 0.01, // ignored when adaptive
        };
        let segments = detect_speech(&audio, SR, &config);
        assert!(
            !segments.is_empty(),
            "adaptive threshold should detect the loud tone as speech"
        );
        // The speech should be roughly in the middle second
        let seg = &segments[0];
        let start_sec = seg.start as f32 / SR as f32;
        let end_sec = seg.end as f32 / SR as f32;
        assert!(
            start_sec < 1.5,
            "speech start should be near 1s, got {start_sec}"
        );
        assert!(
            end_sec > 1.5,
            "speech end should be past 1.5s, got {end_sec}"
        );
    }

    #[test]
    fn test_adaptive_threshold_rejects_noise() {
        // Only quiet noise — adaptive threshold should reject everything.
        let noise_level = 0.005;
        let audio = vec![noise_level; SR * 3]; // 3 seconds of constant quiet noise

        let config = VadConfig {
            adaptive_threshold: true,
            noise_margin: 3.0,
            min_speech_ms: 250,
            min_silence_ms: 300,
            frame_size_ms: 30,
            energy_threshold: 0.01,
        };
        let segments = detect_speech(&audio, SR, &config);
        assert!(
            segments.is_empty(),
            "adaptive threshold should reject constant noise, got {} segments",
            segments.len()
        );
    }

    #[test]
    fn test_adaptive_vs_fixed() {
        // Create audio where adaptive and fixed thresholds give different results.
        // Background noise at 0.005 amplitude (constant DC).
        // Noise floor RMS ≈ 0.005, so adaptive threshold = 0.005 * 3.0 = 0.015.
        // A sine tone with amplitude 0.02 has RMS ≈ 0.0141, which is:
        //   - above fixed threshold of 0.005 → detected
        //   - below adaptive threshold of 0.015 → not detected
        let noise_level = 0.005;
        let tone_amplitude = 0.02; // RMS ≈ 0.0141

        let mut audio = Vec::new();
        // 1 second noise
        audio.extend(vec![noise_level; SR]);
        // 500ms quiet tone (above fixed but below adaptive)
        audio.extend(tone(440.0, 500, tone_amplitude));
        // 1 second noise
        audio.extend(vec![noise_level; SR]);

        let fixed_config = VadConfig {
            energy_threshold: 0.005, // low enough to catch the tone
            min_speech_ms: 100,
            min_silence_ms: 100,
            frame_size_ms: 30,
            adaptive_threshold: false,
            noise_margin: 3.0,
        };
        let adaptive_config = VadConfig {
            adaptive_threshold: true,
            ..fixed_config.clone()
        };

        let fixed_segments = detect_speech(&audio, SR, &fixed_config);
        let adaptive_segments = detect_speech(&audio, SR, &adaptive_config);

        // Fixed (threshold=0.005) should detect the tone (RMS ~0.014 > 0.005).
        // Adaptive (threshold=0.005*3.0=0.015) should reject it (RMS ~0.014 < 0.015).
        assert!(
            !fixed_segments.is_empty(),
            "fixed threshold should detect the tone"
        );
        assert!(
            adaptive_segments.is_empty(),
            "adaptive threshold should reject the borderline tone, got {} segments",
            adaptive_segments.len()
        );
    }
}
