//! Speaker-attributed transcription CLI.
//!
//! Loads a Whisper model and a 16 kHz mono audio file, runs the diarization
//! pipeline (VAD -> embedding -> clustering -> resegmentation), fuses the
//! result with word-level ASR output, and prints either a speaker-labeled
//! transcript or NIST RTTM.
//!
//! Usage:
//!
//! ```text
//! cargo run --example diarize --features diarization -- <model.bin> <audio.wav> [OPTIONS]
//!
//! Options:
//!   --speakers N                Exact number of speakers (otherwise estimated
//!                                automatically within [1, 10]).
//!   --clustering ahc|spectral   Clustering method (default: ahc, default
//!                                cosine-distance threshold of 0.5).
//!   --embedder whisper|ecapa:<path>
//!                                Speaker-embedding backend (default: whisper).
//!                                `ecapa:<path>` loads an externally-trained
//!                                ECAPA-TDNN / x-vector ONNX model and is only
//!                                available when this crate is built with the
//!                                `onnx` feature.
//!   --rttm                      Print NIST RTTM instead of a speaker-labeled
//!                                transcript.
//! ```
//!
//! # Accuracy caveat (read this before trusting any output)
//!
//! The `whisper` embedder (the default) is a **low-accuracy convenience
//! baseline**: it mean-pools features from the Whisper encoder, which is
//! trained to be largely speaker-**invariant** (it encodes phonetic content
//! for ASR, not speaker identity). It exists for a no-extra-model demo and
//! structural testing only -- **never** present its output as production
//! diarization. For real speaker accuracy, pass `--embedder ecapa:<path>`
//! pointing at an externally-trained ECAPA-TDNN / x-vector model exported to
//! ONNX; oxiwhisper does not ship one. See the "Speaker Diarization" section
//! of the crate README for the full set of constraints.

use std::path::{Path, PathBuf};

#[cfg(feature = "onnx")]
use oxiwhisper::EcapaOnnx;
use oxiwhisper::{
    ClusteringMethod, DiarizeOptions, SpeakerEmbedder, TranscribeOptions, WhisperEncoderEmbedder,
    WhisperModel, labeled_transcript_timed, write_rttm,
};

/// Which speaker-embedding backend to use, as selected by `--embedder`.
enum EmbedderChoice {
    /// Built-in Whisper-encoder baseline (low accuracy, no external model).
    Whisper,
    /// Externally-trained ECAPA-TDNN / x-vector ONNX model at this path.
    Ecapa(PathBuf),
}

/// Parsed command-line arguments.
struct CliArgs {
    model_path: PathBuf,
    audio_path: PathBuf,
    num_speakers: Option<usize>,
    clustering: ClusteringMethod,
    embedder: EmbedderChoice,
    rttm: bool,
}

fn usage(prog: &str) -> String {
    format!(
        "Usage: {prog} <model.bin> <audio.wav> [--speakers N] \
         [--clustering ahc|spectral] [--embedder whisper|ecapa:<path>] [--rttm]"
    )
}

fn parse_args() -> Result<CliArgs, String> {
    let raw: Vec<String> = std::env::args().collect();
    let prog = raw
        .first()
        .cloned()
        .unwrap_or_else(|| "diarize".to_string());

    let mut positional: Vec<String> = Vec::new();
    let mut num_speakers: Option<usize> = None;
    let mut clustering = ClusteringMethod::default();
    let mut embedder = EmbedderChoice::Whisper;
    let mut rttm = false;

    let mut iter = raw.into_iter().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--speakers" => {
                let val = iter
                    .next()
                    .ok_or_else(|| "--speakers requires a value".to_string())?;
                let n: usize = val
                    .parse()
                    .map_err(|_| format!("invalid --speakers value: '{val}'"))?;
                num_speakers = Some(n);
            }
            "--clustering" => {
                let val = iter
                    .next()
                    .ok_or_else(|| "--clustering requires a value".to_string())?;
                clustering = match val.as_str() {
                    "ahc" => ClusteringMethod::default(),
                    "spectral" => ClusteringMethod::Spectral,
                    other => {
                        return Err(format!(
                            "unknown --clustering value '{other}' (expected ahc|spectral)"
                        ));
                    }
                };
            }
            "--embedder" => {
                let val = iter
                    .next()
                    .ok_or_else(|| "--embedder requires a value".to_string())?;
                embedder = if val == "whisper" {
                    EmbedderChoice::Whisper
                } else if let Some(path) = val.strip_prefix("ecapa:") {
                    if path.is_empty() {
                        return Err("--embedder ecapa:<path> requires a non-empty path".to_string());
                    }
                    EmbedderChoice::Ecapa(PathBuf::from(path))
                } else {
                    return Err(format!(
                        "unknown --embedder value '{val}' (expected whisper|ecapa:<path>)"
                    ));
                };
            }
            "--rttm" => rttm = true,
            other if other.starts_with("--") => {
                return Err(format!("unknown option: {other}\n{}", usage(&prog)));
            }
            other => positional.push(other.to_string()),
        }
    }

    if positional.len() != 2 {
        return Err(usage(&prog));
    }

    Ok(CliArgs {
        model_path: PathBuf::from(&positional[0]),
        audio_path: PathBuf::from(&positional[1]),
        num_speakers,
        clustering,
        embedder,
        rttm,
    })
}

/// Run diarization (and, unless `rttm`, ASR word fusion) with `embedder` and
/// print the result.
fn run_diarization(
    model: &WhisperModel,
    audio: &[f32],
    t_opts: &TranscribeOptions<'_>,
    d_opts: &DiarizeOptions,
    embedder: &dyn SpeakerEmbedder,
    rttm: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if rttm {
        let result = model.diarize_with_embedder(audio, d_opts, embedder)?;
        eprintln!(
            "Detected {} speaker(s) across {} segment(s)",
            result.num_speakers,
            result.segments.len()
        );
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        write_rttm(&result, "audio", &mut handle)?;
    } else {
        let transcript =
            model.transcribe_with_speakers_using_embedder(audio, t_opts, d_opts, embedder)?;
        eprintln!(
            "Detected {} speaker(s) across {} turn(s)",
            transcript.num_speakers,
            transcript.turns.len()
        );
        println!("{}", labeled_transcript_timed(&transcript));
    }
    Ok(())
}

/// Load the ECAPA ONNX embedder and run diarization with it.
///
/// Only compiled when the `onnx` feature is enabled -- see the `not(onnx)`
/// twin below for the honest error printed otherwise.
#[cfg(feature = "onnx")]
fn run_with_ecapa(
    model: &WhisperModel,
    audio: &[f32],
    t_opts: &TranscribeOptions<'_>,
    d_opts: &DiarizeOptions,
    onnx_path: &Path,
    rttm: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "Loading ECAPA speaker-embedding model from {} ...",
        onnx_path.display()
    );
    let embedder = EcapaOnnx::from_path(onnx_path)?;
    run_diarization(model, audio, t_opts, d_opts, &embedder, rttm)
}

/// Honest failure path when `--embedder ecapa:<path>` is requested but this
/// binary was not built with the `onnx` feature.
#[cfg(not(feature = "onnx"))]
fn run_with_ecapa(
    _model: &WhisperModel,
    _audio: &[f32],
    _t_opts: &TranscribeOptions<'_>,
    _d_opts: &DiarizeOptions,
    onnx_path: &Path,
    _rttm: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    Err(format!(
        "--embedder ecapa:{} requires this crate to be built with the `onnx` feature \
         (ECAPA-TDNN models are loaded via oxionnx). Rebuild with: \
         cargo run --example diarize --features diarization,onnx -- ...",
        onnx_path.display()
    )
    .into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;

    eprintln!("Loading model from {} ...", args.model_path.display());
    let model = WhisperModel::from_file(&args.model_path)?;

    eprintln!("Loading audio from {} ...", args.audio_path.display());
    let audio = oxiwhisper::audio::load_wav(&args.audio_path)?;
    eprintln!(
        "Audio: {:.1}s ({} samples @ 16 kHz)",
        audio.len() as f32 / 16_000.0,
        audio.len()
    );

    let mut d_opts = DiarizeOptions {
        clustering: args.clustering,
        ..DiarizeOptions::default()
    };
    if let Some(n) = args.num_speakers {
        // Widen the search range so an explicit --speakers above the default
        // upper bound of 10 does not trip DiarizeOptions::validate().
        d_opts.max_speakers = d_opts.max_speakers.max(n);
        d_opts.min_speakers = d_opts.min_speakers.min(n);
        d_opts.num_speakers = Some(n);
    }

    let t_opts = TranscribeOptions::default();

    match &args.embedder {
        EmbedderChoice::Whisper => {
            eprintln!(
                "Using the built-in Whisper-encoder baseline embedder -- LOW ACCURACY, \
                 not suitable for production diarization (Whisper's encoder is trained to \
                 be speaker-invariant). Pass --embedder ecapa:<path> for real speaker \
                 discrimination."
            );
            let embedder = WhisperEncoderEmbedder::new(&model);
            run_diarization(&model, &audio, &t_opts, &d_opts, &embedder, args.rttm)
        }
        EmbedderChoice::Ecapa(path) => {
            run_with_ecapa(&model, &audio, &t_opts, &d_opts, path, args.rttm)
        }
    }
}
