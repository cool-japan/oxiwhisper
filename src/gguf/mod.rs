//! GGUF format parser for oxiwhisper.
//!
//! Supports GGUF v3 files with the following quantization types:
//! F32 (0), F16 (1), Q4_0 (2), Q5_0 (6), Q8_0 (8).
//!
//! Unsupported quantization types (K-quants, I-quants, etc.) are rejected
//! with [`crate::OxiWhisperError::InvalidModel`].

pub(crate) mod parse;
pub(crate) mod spec;
pub(crate) mod whisper;
