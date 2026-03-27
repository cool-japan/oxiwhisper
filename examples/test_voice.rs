//! End-to-end voice transcription test for oxiwhisper.
//!
//! Usage:
//!   cargo run --example test_voice -- <model_path> [audio_path.wav]
//!
//! If no audio_path is given, synthetic test signals are generated and transcribed.
//! If an audio_path is given, it is read as 16-bit PCM WAV and transcribed.

use std::path::{Path, PathBuf};
use std::time::Instant;

use oxiwhisper::{OxiWhisperError, TranscribeOptions, WhisperModel};

const SAMPLE_RATE: usize = 16_000;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    model_path: PathBuf,
    audio_path: Option<PathBuf>,
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = PathBuf::from(
        args.next()
            .ok_or("Usage: test_voice <model_path> [audio_path.wav]")?,
    );
    let audio_path = args.next().map(PathBuf::from);
    Ok(Args {
        model_path,
        audio_path,
    })
}

// ---------------------------------------------------------------------------
// Synthetic signal generators (16 kHz mono f32)
// ---------------------------------------------------------------------------

/// Generate `duration_secs` of silence.
fn generate_silence(duration_secs: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * duration_secs) as usize;
    vec![0.0_f32; n]
}

/// Generate a pure sine tone at `freq_hz` for `duration_secs`.
fn generate_sine(freq_hz: f32, duration_secs: f32, amplitude: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * duration_secs) as usize;
    let two_pi = 2.0 * std::f32::consts::PI;
    (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            amplitude * (two_pi * freq_hz * t).sin()
        })
        .collect()
}

/// Generate white noise using a simple linear congruential generator.
/// No external RNG dependency needed for test signals.
fn generate_white_noise(duration_secs: f32, amplitude: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * duration_secs) as usize;
    let mut state: u64 = 0xDEAD_BEEF_CAFE_1234;
    (0..n)
        .map(|_| {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // map to [-1, 1]
            let uniform = (state as f32) / (u64::MAX as f32) * 2.0 - 1.0;
            amplitude * uniform
        })
        .collect()
}

/// Generate a linear chirp sweeping from `f0` to `f1` over `duration_secs`.
fn generate_chirp(f0: f32, f1: f32, duration_secs: f32, amplitude: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * duration_secs) as usize;
    let two_pi = 2.0 * std::f32::consts::PI;
    let rate = (f1 - f0) / duration_secs;
    (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            let phase = two_pi * (f0 * t + 0.5 * rate * t * t);
            amplitude * phase.sin()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Minimal WAV reader (16-bit PCM only)
// ---------------------------------------------------------------------------

fn read_wav_pcm16(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;

    if data.len() < 44 {
        return Err("WAV file too small for a valid header".into());
    }

    // RIFF header
    let riff = &data[0..4];
    if riff != b"RIFF" {
        return Err(format!("Not a RIFF file (got {:?})", riff).into());
    }
    let wave = &data[8..12];
    if wave != b"WAVE" {
        return Err(format!("Not a WAVE file (got {:?})", wave).into());
    }

    // Walk sub-chunks to find "fmt " and "data"
    let mut pos = 12_usize;
    let mut fmt_found = false;
    let mut audio_format: u16 = 0;
    let mut num_channels: u16 = 0;
    let mut sample_rate_wav: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut pcm_bytes: &[u8] = &[];

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;
        let chunk_data_start = pos + 8;
        let chunk_data_end = (chunk_data_start + chunk_size).min(data.len());

        if chunk_id == b"fmt " {
            if chunk_size < 16 {
                return Err("fmt chunk too small".into());
            }
            let c = &data[chunk_data_start..chunk_data_end];
            audio_format = u16::from_le_bytes([c[0], c[1]]);
            num_channels = u16::from_le_bytes([c[2], c[3]]);
            sample_rate_wav = u32::from_le_bytes([c[4], c[5], c[6], c[7]]);
            // skip byte_rate (4 bytes) and block_align (2 bytes)
            bits_per_sample = u16::from_le_bytes([c[14], c[15]]);
            fmt_found = true;
        } else if chunk_id == b"data" {
            pcm_bytes = &data[chunk_data_start..chunk_data_end];
        }

        // Advance to next chunk (chunks are word-aligned)
        pos = chunk_data_start + ((chunk_size + 1) & !1);
    }

    if !fmt_found {
        return Err("No fmt chunk found in WAV".into());
    }
    if audio_format != 1 {
        return Err(
            format!("Unsupported audio format {audio_format} (only PCM=1 supported)").into(),
        );
    }
    if bits_per_sample != 16 {
        return Err(format!(
            "Unsupported bits per sample {bits_per_sample} (only 16-bit supported)"
        )
        .into());
    }
    if pcm_bytes.is_empty() {
        return Err("No data chunk found in WAV".into());
    }

    eprintln!(
        "WAV: {num_channels}ch, {sample_rate_wav} Hz, {bits_per_sample}-bit, {} samples/channel",
        pcm_bytes.len() / (2 * num_channels as usize)
    );

    // Decode 16-bit PCM samples, mix to mono, and resample to 16 kHz if needed.
    let bytes_per_frame = 2 * num_channels as usize;
    let num_frames = pcm_bytes.len() / bytes_per_frame;
    let mut mono = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let mut sum = 0.0_f32;
        for ch in 0..num_channels as usize {
            let offset = frame_idx * bytes_per_frame + ch * 2;
            if offset + 1 < pcm_bytes.len() {
                let sample = i16::from_le_bytes([pcm_bytes[offset], pcm_bytes[offset + 1]]);
                sum += sample as f32 / 32768.0;
            }
        }
        mono.push(sum / num_channels as f32);
    }

    // Simple linear-interpolation resampler if sample rate differs from 16 kHz
    if sample_rate_wav != SAMPLE_RATE as u32 {
        eprintln!(
            "Resampling from {sample_rate_wav} Hz to {SAMPLE_RATE} Hz (linear interpolation)"
        );
        let ratio = sample_rate_wav as f64 / SAMPLE_RATE as f64;
        let out_len = (mono.len() as f64 / ratio) as usize;
        let mut resampled = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_pos = i as f64 * ratio;
            let idx = src_pos as usize;
            let frac = (src_pos - idx as f64) as f32;
            let s0 = mono.get(idx).copied().unwrap_or(0.0);
            let s1 = mono.get(idx + 1).copied().unwrap_or(s0);
            resampled.push(s0 + frac * (s1 - s0));
        }
        Ok(resampled)
    } else {
        Ok(mono)
    }
}

// ---------------------------------------------------------------------------
// Transcription helpers
// ---------------------------------------------------------------------------

struct TranscribeResult {
    config_name: String,
    text: String,
    elapsed_ms: u128,
}

fn run_transcription(
    model: &WhisperModel,
    audio: &[f32],
    opts: &TranscribeOptions<'_>,
    config_name: &str,
) -> Result<TranscribeResult, OxiWhisperError> {
    let start = Instant::now();
    let text = model.transcribe(audio, opts)?;
    let elapsed_ms = start.elapsed().as_millis();
    Ok(TranscribeResult {
        config_name: config_name.to_string(),
        text,
        elapsed_ms,
    })
}

fn print_result(result: &TranscribeResult) {
    println!("  [{:>6} ms] {}", result.elapsed_ms, result.config_name);
    let display_text = if result.text.is_empty() {
        "(empty)"
    } else {
        &result.text
    };
    println!("             => {display_text}");
}

// ---------------------------------------------------------------------------
// Configuration presets
// ---------------------------------------------------------------------------

fn make_configs() -> Vec<(String, TranscribeOptions<'static>)> {
    vec![
        ("greedy (default)".to_string(), TranscribeOptions::default()),
        (
            "beam_search (width=3)".to_string(),
            TranscribeOptions {
                beam_width: 3,
                ..TranscribeOptions::default()
            },
        ),
        (
            "temperature (t=0.5, top_k=10)".to_string(),
            TranscribeOptions {
                temperature: 0.5,
                top_k: 10,
                ..TranscribeOptions::default()
            },
        ),
        (
            "language=en".to_string(),
            TranscribeOptions {
                language: Some("en"),
                ..TranscribeOptions::default()
            },
        ),
    ]
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;

    println!("=== OxiWhisper End-to-End Voice Transcription Test ===\n");

    // Load model
    println!("Loading model from: {}", args.model_path.display());
    let load_start = Instant::now();
    let model = WhisperModel::from_file(&args.model_path)?;
    let load_ms = load_start.elapsed().as_millis();
    println!("Model loaded in {load_ms} ms\n");

    let configs = make_configs();

    match args.audio_path {
        Some(ref wav_path) => {
            // ---- WAV file mode ----
            println!("Reading WAV: {}\n", wav_path.display());
            let audio = read_wav_pcm16(wav_path)?;
            let duration_secs = audio.len() as f32 / SAMPLE_RATE as f32;
            println!(
                "Audio: {:.2} s ({} samples at {SAMPLE_RATE} Hz)\n",
                duration_secs,
                audio.len()
            );

            println!("Transcription results:");
            for (name, opts) in &configs {
                match run_transcription(&model, &audio, opts, name) {
                    Ok(result) => print_result(&result),
                    Err(e) => println!("  [  ERROR] {name}\n             => {e}"),
                }
            }
        }
        None => {
            // ---- Synthetic signal mode ----
            let signals: Vec<(&str, Vec<f32>)> = vec![
                ("Silence (2s)", generate_silence(2.0)),
                ("440 Hz sine (2s)", generate_sine(440.0, 2.0, 0.8)),
                ("White noise (2s)", generate_white_noise(2.0, 0.3)),
                (
                    "Chirp 200-4000 Hz (3s)",
                    generate_chirp(200.0, 4000.0, 3.0, 0.6),
                ),
            ];

            for (signal_name, audio) in &signals {
                let duration_secs = audio.len() as f32 / SAMPLE_RATE as f32;
                println!(
                    "--- Signal: {signal_name} ({:.2} s, {} samples) ---",
                    duration_secs,
                    audio.len()
                );

                for (config_name, opts) in &configs {
                    match run_transcription(&model, audio, opts, config_name) {
                        Ok(result) => print_result(&result),
                        Err(e) => println!("  [  ERROR] {config_name}\n             => {e}"),
                    }
                }
                println!();
            }
        }
    }

    println!("=== Done ===");
    Ok(())
}
