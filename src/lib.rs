//! # OxiWhisper
//!
//! Pure Rust Whisper speech-to-text inference engine with zero C/C++ dependencies.
//!
//! OxiWhisper loads GGML-format Whisper models and transcribes audio to text,
//! supporting quantized inference (Q4_0, Q5_0, Q8_0), streaming, beam search,
//! word-level timestamps, and SIMD-accelerated kernels (AVX2, NEON, WASM simd128).
//!
//! ## Quick Start
//!
//! ```no_run
//! use oxiwhisper::{TranscribeOptions, WhisperModel};
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = WhisperModel::from_file(Path::new("ggml-tiny.bin"))?;
//! let audio = oxiwhisper::audio::load_wav(Path::new("audio.wav"))?;
//! let text = model.transcribe(&audio, &TranscribeOptions::default())?;
//! println!("{text}");
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

/// Multi-head attention primitives shared by encoder and decoder.
pub mod attention;
/// Audio I/O helpers (WAV loading, PCM resampling).
pub mod audio;
/// Beam search decoder for multi-hypothesis Whisper decoding.
pub mod beam_search;
/// Token-level decoding utilities (argmax, sampling, ngram suppression).
pub mod decode_utils;
/// Core Whisper text decoder (forward pass, KV cache, sampling).
pub mod decoder;
/// Speaker diarization ("who spoke when"): segmentation, embedding, clustering
/// (D2-D4) and resegmentation plus the end-to-end pipeline (D5) are implemented.
#[cfg(feature = "diarization")]
pub mod diarize;
/// Dynamic Time Warping utilities for word-level timestamp alignment.
pub mod dtw;
/// Whisper audio encoder (CNN + Transformer).
pub mod encoder;
/// FFT utilities backed by OxiFFT (used by the mel spectrogram pipeline).
pub mod fft;
pub(crate) mod gguf;
/// Hallucination detection heuristics for Whisper output segments.
pub mod hallucination;
/// Linear (dense) layer kernels with optional quantized weight support.
pub mod linear;
/// Log-mel spectrogram computation from 16 kHz PCM audio.
pub mod mel;
/// Pre-computed Whisper mel filterbank coefficients.
pub mod mel_filters;
/// GGML model loader and weight storage types.
pub mod model;
#[cfg(feature = "onnx")]
pub mod onnx_loader;
/// Quantization types and GGML Q4_0/Q5_0/Q8_0 dequantization kernels.
pub mod quantize;
/// Streaming transcription that accumulates audio in 30-second chunks.
pub mod stream;
/// SRT and WebVTT subtitle formatting from timed segments.
pub mod subtitle;
/// Minimal f32 tensor type used throughout the inference pipeline.
pub mod tensor;
/// Synthetic model generators for integration tests.
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
/// Thread count management and parallel iteration helpers.
pub mod threading;
/// BPE token-ID to text decoding and segment parsing for Whisper.
pub mod tokenizer;
/// Public types, error enum, and option validation for oxiwhisper.
pub mod types;
/// Voice activity detection (energy-based silence segmentation).
pub mod vad;
/// Word-level timestamp alignment via cross-attention DTW.
pub mod word_timestamps;

mod whisper_model;

#[cfg(feature = "diarization")]
pub use diarize::attribute::{SpeakerTranscript, SpeakerTurn};
#[cfg(all(feature = "diarization", feature = "onnx"))]
pub use diarize::embed::EcapaOnnx;
#[cfg(feature = "diarization")]
pub use diarize::embed::{SpeakerEmbedder, WhisperEncoderEmbedder};
#[cfg(feature = "diarization")]
pub use diarize::format::{labeled_transcript, labeled_transcript_timed, rttm_string, write_rttm};
#[cfg(feature = "diarization")]
pub use diarize::metrics::{DerOptions, DerReport, RttmSegment, der, hungarian, jer, parse_rttm};
#[cfg(feature = "diarization")]
pub use diarize::{ClusteringMethod, DiarizeOptions, DiarizeResult, SpeakerId, SpeakerSegment};
pub use dtw::WordSegment;
pub use types::*;
pub use word_timestamps::WordTimedTranscript;

use std::sync::Arc;

/// Main entry point for Whisper speech-to-text inference.
///
/// Wraps a loaded GGML model and exposes transcription methods. Internally
/// shares the model weights via `Arc` so cloning is cheap and the model is
/// usable from multiple threads.
pub struct WhisperModel {
    model_data: Arc<model::ModelData>,
}
