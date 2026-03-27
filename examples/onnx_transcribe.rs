//! End-to-end example: load an ONNX Whisper model and transcribe audio.
//!
//! Usage:
//!   cargo run --example onnx_transcribe --features onnx -- <model.onnx> [audio.wav] [--vocab=path] [--decoder=path]
//!
//! If no audio file is given a 3-second 440 Hz sine wave is synthesised.

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use oxiwhisper::onnx_loader::OnnxModelConfig;
use oxiwhisper::{TranscribeOptions, WhisperModel};

const SAMPLE_RATE: u32 = 16_000;

// ── CLI parsing ────────────────────────────────────────────────────────

struct Args {
    model_path: PathBuf,
    audio_path: Option<PathBuf>,
    vocab_path: Option<PathBuf>,
    decoder_path: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = env::args().skip(1).collect();

    let mut model_path: Option<PathBuf> = None;
    let mut audio_path: Option<PathBuf> = None;
    let mut vocab_path: Option<PathBuf> = None;
    let mut decoder_path: Option<PathBuf> = None;

    for arg in &raw {
        if let Some(v) = arg.strip_prefix("--vocab=") {
            vocab_path = Some(PathBuf::from(v));
        } else if let Some(v) = arg.strip_prefix("--decoder=") {
            decoder_path = Some(PathBuf::from(v));
        } else if model_path.is_none() {
            model_path = Some(PathBuf::from(arg));
        } else if audio_path.is_none() {
            audio_path = Some(PathBuf::from(arg));
        } else {
            return Err(format!("unexpected argument: {arg}"));
        }
    }

    let model_path = model_path.ok_or_else(|| {
        "usage: onnx_transcribe <model.onnx> [audio.wav] [--vocab=path] [--decoder=path]"
            .to_string()
    })?;

    Ok(Args {
        model_path,
        audio_path,
        vocab_path,
        decoder_path,
    })
}

// ── Minimal WAV reader (16-bit PCM, mono, 16 kHz) ─────────────────────

fn read_wav_pcm16(path: &PathBuf) -> Result<Vec<f32>, String> {
    let mut file =
        File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;

    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    // Validate RIFF/WAVE header
    if buf.len() < 44 {
        return Err("WAV file too short for a valid header".into());
    }
    if &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        return Err("not a valid RIFF/WAVE file".into());
    }

    // Walk sub-chunks to find "fmt " and "data"
    let mut pos: usize = 12;
    let mut bits_per_sample: u16 = 0;
    let mut num_channels: u16 = 0;
    let mut sample_rate_hz: u32 = 0;
    let mut audio_format: u16 = 0;
    let mut data_start: usize = 0;
    let mut data_len: usize = 0;

    while pos + 8 <= buf.len() {
        let chunk_id = &buf[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]]) as usize;

        if chunk_id == b"fmt " {
            if chunk_size < 16 || pos + 8 + 16 > buf.len() {
                return Err("fmt chunk too small".into());
            }
            let base = pos + 8;
            audio_format = u16::from_le_bytes([buf[base], buf[base + 1]]);
            num_channels = u16::from_le_bytes([buf[base + 2], buf[base + 3]]);
            sample_rate_hz =
                u32::from_le_bytes([buf[base + 4], buf[base + 5], buf[base + 6], buf[base + 7]]);
            bits_per_sample = u16::from_le_bytes([buf[base + 14], buf[base + 15]]);
        } else if chunk_id == b"data" {
            data_start = pos + 8;
            data_len = chunk_size;
        }

        // Advance to next chunk (chunks are word-aligned)
        let advance = 8 + chunk_size + (chunk_size & 1);
        pos += advance;
    }

    if data_start == 0 || data_len == 0 {
        return Err("no data chunk found in WAV".into());
    }
    if audio_format != 1 {
        return Err(format!(
            "unsupported WAV format {audio_format} (only PCM/1 supported)"
        ));
    }
    if bits_per_sample != 16 {
        return Err(format!(
            "unsupported bits-per-sample {bits_per_sample} (only 16 supported)"
        ));
    }

    eprintln!(
        "WAV: {num_channels}ch, {sample_rate_hz} Hz, {bits_per_sample}-bit, {data_len} bytes of audio"
    );

    // Convert interleaved i16 samples to mono f32 in [-1, 1]
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let frame_size = bytes_per_sample * num_channels as usize;
    let num_frames = data_len / frame_size;

    let mut samples = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let offset = data_start + i * frame_size;
        if offset + 1 >= buf.len() {
            break;
        }
        // Take only the first channel
        let raw = i16::from_le_bytes([buf[offset], buf[offset + 1]]);
        samples.push(raw as f32 / 32768.0);
    }

    // Resample to 16 kHz if needed (simple linear interpolation)
    if sample_rate_hz != SAMPLE_RATE {
        let ratio = sample_rate_hz as f64 / SAMPLE_RATE as f64;
        let new_len = (samples.len() as f64 / ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src = i as f64 * ratio;
            let idx = src as usize;
            let frac = (src - idx as f64) as f32;
            let a = samples.get(idx).copied().unwrap_or(0.0);
            let b = samples.get(idx + 1).copied().unwrap_or(a);
            resampled.push(a + (b - a) * frac);
        }
        return Ok(resampled);
    }

    Ok(samples)
}

// ── Sine-wave synthesiser ──────────────────────────────────────────────

fn generate_sine_wave(duration_secs: f32, frequency_hz: f32) -> Vec<f32> {
    let num_samples = (SAMPLE_RATE as f32 * duration_secs) as usize;
    let two_pi = 2.0 * std::f32::consts::PI;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            (two_pi * frequency_hz * t).sin() * 0.5
        })
        .collect()
}

// ── Main ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;

    // Build ONNX config
    let config = OnnxModelConfig {
        model_path: args.model_path.clone(),
        decoder_path: args.decoder_path,
        vocab_path: args.vocab_path,
    };

    eprintln!("Loading ONNX model from {} ...", args.model_path.display());
    let t0 = Instant::now();
    let model = WhisperModel::from_onnx(&config)?;
    let load_elapsed = t0.elapsed();
    eprintln!("Model loaded in {load_elapsed:.2?}");

    // Prepare audio
    let audio = match &args.audio_path {
        Some(path) => {
            eprintln!("Reading audio from {} ...", path.display());
            read_wav_pcm16(path)?
        }
        None => {
            eprintln!("No audio file given -- synthesising a 3-second 440 Hz sine wave");
            generate_sine_wave(3.0, 440.0)
        }
    };
    let duration_secs = audio.len() as f32 / SAMPLE_RATE as f32;
    eprintln!(
        "Audio: {} samples ({duration_secs:.2} s at {SAMPLE_RATE} Hz)",
        audio.len()
    );

    // Transcribe
    let opts = TranscribeOptions::default();

    eprintln!("Transcribing ...");
    let t1 = Instant::now();
    let text = model.transcribe(&audio, &opts)?;
    let transcribe_elapsed = t1.elapsed();

    // Results
    println!("{text}");

    eprintln!("---");
    eprintln!("Transcription time : {transcribe_elapsed:.2?}");
    eprintln!("Audio duration     : {duration_secs:.2} s");
    let rtf = transcribe_elapsed.as_secs_f64() / duration_secs as f64;
    eprintln!("Real-time factor   : {rtf:.3}x");
    eprintln!("Total (load+infer) : {:.2?}", t0.elapsed());

    Ok(())
}
