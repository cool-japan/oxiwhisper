//! Streaming transcription example.
//!
//! Demonstrates real-time-style transcription using `StreamTranscriber`.
//! Audio is pushed in 1-second chunks to simulate a live audio stream.
//!
//! Usage:
//!   cargo run --example streaming -- <model.bin> [audio.wav]

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.bin> [audio.wav]", args[0]);
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

    // Get audio: from file or generate synthetic
    let audio = if args.len() >= 3 {
        match oxiwhisper::audio::load_wav(Path::new(&args[2])) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("Failed to load audio: {e}");
                std::process::exit(1);
            }
        }
    } else {
        // Generate 5 seconds of 440Hz sine wave at 16kHz
        let sample_rate = 16000;
        let duration_secs = 5.0f32;
        let n_samples = (sample_rate as f32 * duration_secs) as usize;
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3
            })
            .collect()
    };

    println!(
        "Audio: {} samples ({:.1}s at 16kHz)",
        audio.len(),
        audio.len() as f32 / 16000.0
    );

    let opts = oxiwhisper::TranscribeOptions {
        timestamps: true,
        ..Default::default()
    };

    let mut stream = model.stream(opts);
    let chunk_size = 16000; // 1 second chunks

    println!("\n--- Streaming transcription ---");
    for (i, chunk) in audio.chunks(chunk_size).enumerate() {
        let elapsed = (i + 1) as f32;
        println!("[{elapsed:.0}s] Pushing {} samples...", chunk.len());
        stream.push_audio(chunk);

        while let Some(result) = stream.next_segment() {
            match result {
                Ok(seg) => println!(
                    "  Segment [{:.2}s - {:.2}s]: {}",
                    seg.start, seg.end, seg.text
                ),
                Err(e) => eprintln!("  Error: {e}"),
            }
        }
    }

    println!("\n--- Finishing ---");
    match stream.finish() {
        Ok(result) => {
            println!("Full text: {}", result.text);
            println!("Segments: {}", result.segments.len());
            if let Some(lang) = &result.language {
                println!("Language: {lang}");
            }
        }
        Err(e) => eprintln!("Error: {e}"),
    }
}
