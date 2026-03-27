//! Batch transcription example.
//!
//! Demonstrates transcribing multiple audio clips in a single call
//! using `transcribe_batch()`.
//!
//! Usage:
//!   cargo run --example batch_transcribe -- <model.bin> [audio1.wav] [audio2.wav] ...

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <model.bin> [audio1.wav] [audio2.wav] ...",
            args[0]
        );
        std::process::exit(1);
    }

    let model_path = Path::new(&args[1]);
    let model = match oxiwhisper::WhisperModel::from_file(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            std::process::exit(1);
        }
    };

    println!("Model loaded: {:?}", model.info());

    // Collect audio clips
    let mut clips: Vec<Vec<f32>> = Vec::new();

    if args.len() >= 3 {
        // Load from files
        for path in &args[2..] {
            match oxiwhisper::audio::load_wav(Path::new(path)) {
                Ok(audio) => {
                    println!(
                        "Loaded {path}: {} samples ({:.1}s)",
                        audio.len(),
                        audio.len() as f32 / 16000.0
                    );
                    clips.push(audio);
                }
                Err(e) => {
                    eprintln!("Failed to load {path}: {e}");
                }
            }
        }
    } else {
        // Generate 3 synthetic clips with different frequencies
        let sample_rate = 16000;
        for (i, freq) in [440.0f32, 880.0, 220.0].iter().enumerate() {
            let duration = 2.0 + i as f32; // 2s, 3s, 4s
            let n = (sample_rate as f32 * duration) as usize;
            let clip: Vec<f32> = (0..n)
                .map(|s| {
                    let t = s as f32 / sample_rate as f32;
                    (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3
                })
                .collect();
            println!(
                "Generated clip {}: {n} samples ({duration:.0}s, {freq}Hz)",
                i + 1
            );
            clips.push(clip);
        }
    }

    if clips.is_empty() {
        eprintln!("No audio clips to process");
        std::process::exit(1);
    }

    let opts = oxiwhisper::TranscribeOptions {
        timestamps: true,
        ..Default::default()
    };

    println!("\n--- Batch transcription ({} clips) ---", clips.len());
    let clip_refs: Vec<&[f32]> = clips.iter().map(|c| c.as_slice()).collect();
    let start = std::time::Instant::now();
    let results = model.transcribe_batch(&clip_refs, &opts);
    let elapsed = start.elapsed();

    for (i, result) in results.iter().enumerate() {
        println!("\nClip {}:", i + 1);
        match result {
            Ok(r) => {
                println!("  Text: {}", r.text);
                println!("  Segments: {}", r.segments.len());
                for seg in &r.segments {
                    println!(
                        "    [{:.2}s - {:.2}s] {} (conf: {:.3})",
                        seg.start, seg.end, seg.text, seg.confidence
                    );
                }
                if let Some(lang) = &r.language {
                    println!("  Language: {lang}");
                }
            }
            Err(e) => println!("  Error: {e}"),
        }
    }

    let total_audio: f32 = clips.iter().map(|c| c.len() as f32 / 16000.0).sum();
    println!(
        "\nProcessed {:.1}s of audio in {:.2}s (RTF: {:.2})",
        total_audio,
        elapsed.as_secs_f32(),
        elapsed.as_secs_f32() / total_audio
    );
}
