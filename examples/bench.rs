//! Benchmark example for oxiwhisper.
//!
//! Usage:
//!   cargo run --example bench --release -- <model.bin> [audio.wav]
//!
//! If no WAV file is provided, a synthetic 5-second 440 Hz sine wave is used.
//!
//! Reports per-phase timing (mel, encoder, decoder) and RTF (Real-Time Factor).
//! RTF < 1.0 means faster-than-realtime processing.

use std::env;
use std::fs::File;
use std::io::{self, Read as _, Seek as _, SeekFrom};
use std::path::Path;

use oxiwhisper::{OxiWhisperError, TranscribeOptions, TranscribeTiming, WhisperModel};

const SAMPLE_RATE: u32 = 16_000;
const NUM_ITERATIONS: usize = 3;

// ---------------------------------------------------------------------------
// Minimal WAV parser (16-bit PCM only)
// ---------------------------------------------------------------------------

/// Read a WAV file and return 16 kHz mono f32 PCM samples.
///
/// Assumes:
/// - RIFF/WAVE container
/// - fmt chunk describes 16-bit PCM (format tag 1)
/// - Only the first channel is kept when stereo
/// - No sample-rate conversion is performed (the file MUST already be 16 kHz)
fn read_wav(path: &Path) -> Result<Vec<f32>, OxiWhisperError> {
    let mut f = File::open(path)?;

    // --- RIFF header ---
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)?;
    if &buf4 != b"RIFF" {
        return Err(OxiWhisperError::InvalidModel("Not a RIFF file".into()));
    }
    // Skip file size
    f.read_exact(&mut buf4)?;
    f.read_exact(&mut buf4)?;
    if &buf4 != b"WAVE" {
        return Err(OxiWhisperError::InvalidModel("Not a WAVE file".into()));
    }

    // --- Walk chunks to find fmt and data ---
    let mut num_channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut data_bytes: Vec<u8> = Vec::new();

    loop {
        let mut chunk_id = [0u8; 4];
        if f.read_exact(&mut chunk_id).is_err() {
            break;
        }
        let mut size_buf = [0u8; 4];
        f.read_exact(&mut size_buf)?;
        let chunk_size = u32::from_le_bytes(size_buf) as u64;

        match &chunk_id {
            b"fmt " => {
                let mut fmt = vec![0u8; chunk_size as usize];
                f.read_exact(&mut fmt)?;
                if fmt.len() < 16 {
                    return Err(OxiWhisperError::InvalidModel("fmt chunk too short".into()));
                }
                let audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
                if audio_format != 1 {
                    return Err(OxiWhisperError::InvalidModel(format!(
                        "Unsupported audio format tag {audio_format} (only PCM/1 is supported)"
                    )));
                }
                num_channels = u16::from_le_bytes([fmt[2], fmt[3]]);
                sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
                bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
            }
            b"data" => {
                data_bytes = vec![0u8; chunk_size as usize];
                f.read_exact(&mut data_bytes)?;
            }
            _ => {
                // Skip unknown chunk (pad to even boundary)
                let skip = (chunk_size + 1) & !1;
                f.seek(SeekFrom::Current(skip as i64))?;
            }
        }
    }

    if num_channels == 0 || data_bytes.is_empty() {
        return Err(OxiWhisperError::InvalidModel(
            "WAV file missing fmt or data chunk".into(),
        ));
    }

    if bits_per_sample != 16 {
        return Err(OxiWhisperError::InvalidModel(format!(
            "Only 16-bit PCM is supported, got {bits_per_sample}-bit"
        )));
    }

    if sample_rate != SAMPLE_RATE {
        eprintln!(
            "WARNING: WAV sample rate is {sample_rate} Hz but oxiwhisper expects {SAMPLE_RATE} Hz. \
             Results may be incorrect."
        );
    }

    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let block_align = num_channels as usize * bytes_per_sample;
    let num_frames = data_bytes.len() / block_align;

    // Convert to mono f32, taking only the first channel
    let mut samples = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let offset = i * block_align;
        if offset + 1 >= data_bytes.len() {
            break;
        }
        let raw = i16::from_le_bytes([data_bytes[offset], data_bytes[offset + 1]]);
        samples.push(raw as f32 / 32768.0);
    }

    Ok(samples)
}

// ---------------------------------------------------------------------------
// Synthetic audio generation
// ---------------------------------------------------------------------------

/// Generate a sine wave at the given frequency and duration (seconds).
fn generate_sine(frequency_hz: f32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (SAMPLE_RATE as f32 * duration_secs) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    let angular_freq = 2.0 * std::f32::consts::PI * frequency_hz / SAMPLE_RATE as f32;
    for i in 0..num_samples {
        samples.push((angular_freq * i as f32).sin() * 0.5);
    }
    samples
}

// ---------------------------------------------------------------------------
// Timing display helpers
// ---------------------------------------------------------------------------

/// Format a duration as milliseconds with appropriate precision.
fn fmt_ms(d: std::time::Duration) -> String {
    let ms = d.as_secs_f64() * 1000.0;
    if ms >= 1000.0 {
        format!("{ms:.0}ms")
    } else if ms >= 10.0 {
        format!("{ms:.1}ms")
    } else {
        format!("{ms:.2}ms")
    }
}

/// Single run result for aggregation.
struct RunResult {
    timing: TranscribeTiming,
    text: String,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <model.bin> [audio.wav]", args[0]);
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "Missing model path argument").into(),
        );
    }

    let model_path = Path::new(&args[1]);

    // --- Load audio ---
    let (audio, audio_label) = if args.len() >= 3 {
        let wav_path = Path::new(&args[2]);
        eprintln!("Loading WAV file: {}", wav_path.display());
        let samples = read_wav(wav_path)?;
        let label = format!("{} ({} samples)", wav_path.display(), samples.len());
        (samples, label)
    } else {
        let duration_secs = 5.0_f32;
        let freq = 440.0_f32;
        eprintln!(
            "No WAV file provided. Generating synthetic {freq} Hz sine wave ({duration_secs}s)."
        );
        let samples = generate_sine(freq, duration_secs);
        let label = format!("synthetic {freq} Hz sine, {duration_secs}s");
        (samples, label)
    };

    let audio_duration_secs = audio.len() as f64 / SAMPLE_RATE as f64;

    // --- Load model ---
    eprintln!("Loading model: {}", model_path.display());
    let t_load_start = std::time::Instant::now();
    let model = WhisperModel::from_file(model_path)?;
    let t_load = t_load_start.elapsed();
    eprintln!("Model loaded in {:.3}s", t_load.as_secs_f64());
    eprintln!();

    // --- Run benchmark iterations ---
    let opts = TranscribeOptions::default();

    let mut runs: Vec<RunResult> = Vec::with_capacity(NUM_ITERATIONS);

    for i in 0..NUM_ITERATIONS {
        eprintln!("  Run {} / {} ...", i + 1, NUM_ITERATIONS);
        let (text, timing) = model.transcribe_timed(&audio, &opts)?;
        runs.push(RunResult { timing, text });
    }

    // --- Print report ---
    println!();
    println!("oxiwhisper benchmark");
    println!("====================");
    println!(
        "Model: {}",
        model_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| model_path.display().to_string())
    );
    println!("Audio: {audio_duration_secs:.1}s ({audio_label})");
    println!("Model load: {}", fmt_ms(t_load));
    println!();

    let mut rtfs: Vec<f64> = Vec::with_capacity(NUM_ITERATIONS);

    for (i, run) in runs.iter().enumerate() {
        let t = &run.timing;
        let rtf = if audio_duration_secs > 0.0 {
            t.total.as_secs_f64() / audio_duration_secs
        } else {
            0.0
        };
        rtfs.push(rtf);

        println!(
            "Run {}: mel {} | encoder {} | decoder {} | total {} | RTF {:.3}",
            i + 1,
            fmt_ms(t.mel),
            fmt_ms(t.encoder),
            fmt_ms(t.decoder),
            fmt_ms(t.total),
            rtf,
        );
    }

    println!();

    // --- Summary statistics ---
    let min_rtf = rtfs.iter().copied().fold(f64::INFINITY, f64::min);
    let max_rtf = rtfs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let avg_rtf = rtfs.iter().sum::<f64>() / rtfs.len() as f64;

    let speedup = if min_rtf > f64::EPSILON {
        1.0 / min_rtf
    } else {
        f64::INFINITY
    };

    println!("Best RTF: {min_rtf:.3} ({speedup:.0}x faster than realtime)");
    println!("Avg  RTF: {avg_rtf:.3}");
    println!("Max  RTF: {max_rtf:.3}");

    if avg_rtf < 0.3 {
        println!("Target:   PASS (avg RTF {avg_rtf:.3} < 0.3)");
    } else {
        println!("Target:   MISS (avg RTF {avg_rtf:.3} >= 0.3)");
    }

    println!();

    // Print transcription to stdout so it can be piped / captured separately
    if let Some(last) = runs.last() {
        println!("Transcription: {}", last.text);
    }

    Ok(())
}
