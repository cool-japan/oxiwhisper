//! Simple file transcription CLI.
//!
//! Usage:
//!   cargo run --example transcribe -- <model.bin> <audio.wav> [--srt|--vtt|--timestamps]
//!
//! Options:
//!   --srt          Output in SRT subtitle format
//!   --vtt          Output in WebVTT subtitle format
//!   --timestamps   Show segment timestamps
//!   (default)      Plain text output

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <model.bin> <audio.wav> [--srt|--vtt|--timestamps]",
            args[0]
        );
        std::process::exit(1);
    }

    let model_path = Path::new(&args[1]);
    let audio_path = Path::new(&args[2]);
    let format = args.get(3).map(|s| s.as_str()).unwrap_or("--text");

    let model = match oxiwhisper::WhisperModel::from_file(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model: {e}");
            std::process::exit(1);
        }
    };

    let audio = match oxiwhisper::audio::load_wav(audio_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error loading audio: {e}");
            std::process::exit(1);
        }
    };

    let duration = audio.len() as f32 / 16000.0;
    eprintln!("Model: {:?}", model.info());
    eprintln!("Audio: {:.1}s ({} samples)", duration, audio.len());

    let opts = oxiwhisper::TranscribeOptions {
        timestamps: format != "--text",
        ..Default::default()
    };

    let start = std::time::Instant::now();

    match format {
        "--srt" => match model.transcribe_to_srt(&audio, &opts) {
            Ok(srt) => print!("{srt}"),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },
        "--vtt" => match model.transcribe_to_vtt(&audio, &opts) {
            Ok(vtt) => print!("{vtt}"),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },
        "--timestamps" => match model.transcribe_segmented(&audio, &opts) {
            Ok(result) => {
                for seg in &result.segments {
                    println!(
                        "[{:.2}s - {:.2}s] {} (conf: {:.3})",
                        seg.start, seg.end, seg.text, seg.confidence
                    );
                }
                if let Some(lang) = &result.language {
                    eprintln!("Language: {lang}");
                }
            }
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },
        _ => match model.transcribe(&audio, &opts) {
            Ok(text) => println!("{text}"),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },
    }

    let elapsed = start.elapsed();
    eprintln!(
        "Transcribed in {:.2}s (RTF: {:.2})",
        elapsed.as_secs_f32(),
        elapsed.as_secs_f32() / duration
    );
}
