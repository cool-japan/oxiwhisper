//! Public types, error handling, and validation for oxiwhisper.

/// Top-level error type for oxiwhisper.
#[derive(Debug)]
pub enum OxiWhisperError {
    Io(std::io::Error),
    InvalidModel(String),
    InferenceFailed(String),
    ShapeMismatch(String),
    ConfigError(String),
    AudioFormatError(String),
}

impl std::fmt::Display for OxiWhisperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::InvalidModel(msg) => write!(f, "Invalid model: {msg}"),
            Self::InferenceFailed(msg) => write!(f, "Inference failed: {msg}"),
            Self::ShapeMismatch(msg) => write!(f, "Shape mismatch: {msg}"),
            Self::ConfigError(msg) => write!(f, "Config error: {msg}"),
            Self::AudioFormatError(msg) => write!(f, "Audio format error: {msg}"),
        }
    }
}

impl std::error::Error for OxiWhisperError {}

impl From<std::io::Error> for OxiWhisperError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Decoding options passed to [`WhisperModel::transcribe`](crate::WhisperModel::transcribe).
#[derive(Debug, Clone)]
pub struct TranscribeOptions<'a> {
    /// BCP-47 language code (e.g. `"en"`, `"ja"`).
    /// `None` enables automatic language detection from the first audio frame.
    pub language: Option<&'a str>,
    /// Beam width for beam search. `1` (default) uses greedy or temperature sampling.
    /// Ignored when `temperature > 0.0`.
    pub beam_width: usize,
    /// Sampling temperature. `0.0` (default) uses deterministic argmax / beam search.
    /// Values > 0 enable random sampling scaled by `1/temperature`.
    pub temperature: f32,
    /// Top-k filter for sampling. `0` disables it (all tokens eligible).
    pub top_k: usize,
    /// Nucleus (top-p) filter for sampling. `1.0` disables it.
    pub top_p: f32,
    /// When `true`, the decoder emits timestamp tokens bracketing text segments,
    /// enabling word-level alignment via [`WhisperModel::transcribe_segmented`](crate::WhisperModel::transcribe_segmented).
    pub timestamps: bool,
    /// Initial prompt text to condition the decoder output. Useful for
    /// providing domain-specific vocabulary or formatting hints.
    /// Tokens are looked up in the model vocabulary and prepended to the decoder prompt.
    pub initial_prompt: Option<&'a str>,
    /// Token IDs to suppress during decoding. Suppressed tokens have their
    /// logits set to negative infinity before sampling/argmax.
    pub suppress_tokens: Option<&'a [u32]>,
    /// Size of n-grams to prevent from repeating (0 = disabled, default).
    /// Setting to 3 prevents any 3-gram from appearing more than once,
    /// which helps avoid "the the the..." hallucination patterns.
    pub no_repeat_ngram_size: usize,
    /// Compression ratio threshold for hallucination detection (default 2.4).
    /// Segments with character-level entropy below this threshold are flagged
    /// as likely hallucinations. Set to 0.0 to disable.
    pub compression_ratio_threshold: f32,
    /// Token IDs from a previous transcription segment, used as context for
    /// cross-chunk coherence in long audio. The decoder prepends these before
    /// the SOT token to condition output on prior context.
    pub previous_tokens: Option<&'a [u32]>,
}

impl Default for TranscribeOptions<'static> {
    fn default() -> Self {
        Self {
            language: None,
            beam_width: 1,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            timestamps: false,
            initial_prompt: None,
            suppress_tokens: None,
            no_repeat_ngram_size: 0,
            compression_ratio_threshold: 2.4,
            previous_tokens: None,
        }
    }
}

/// A transcribed segment with timing information.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Transcribed text for this segment.
    pub text: String,
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
    /// Average log-probability for this segment (higher is more confident).
    /// `0.0` when no probability information is available.
    pub confidence: f32,
    /// Whether this segment is suspected to be a hallucination,
    /// based on compression ratio / repetition analysis.
    pub is_hallucination: bool,
}

/// Full transcription result with segments and metadata.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct TranscribeResult {
    /// Full transcribed text (timestamps stripped).
    pub text: String,
    /// Segments with timing (empty if `timestamps` was `false`).
    pub segments: Vec<Segment>,
    /// Detected language code (if auto-detected).
    pub language: Option<String>,
}

/// Per-phase timing breakdown from a transcription call.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct TranscribeTiming {
    /// Time spent computing the mel spectrogram.
    pub mel: std::time::Duration,
    /// Time spent in the encoder.
    pub encoder: std::time::Duration,
    /// Time spent in the decoder (including tokenization).
    pub decoder: std::time::Duration,
    /// Total wall-clock time for the transcription.
    pub total: std::time::Duration,
}

/// Information about the loaded Whisper model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Vocabulary size
    pub n_vocab: usize,
    /// Number of audio encoder layers
    pub n_audio_layers: usize,
    /// Number of text decoder layers
    pub n_text_layers: usize,
    /// Model dimension (d_model / hidden state size)
    pub d_model: usize,
    /// Number of mel frequency bands
    pub n_mels: usize,
    /// Number of audio attention heads
    pub n_audio_heads: usize,
    /// Number of text attention heads
    pub n_text_heads: usize,
    /// Maximum audio context length
    pub n_audio_ctx: usize,
    /// Maximum text context length
    pub n_text_ctx: usize,
}

/// Statistics about the loaded model.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelStats {
    /// Total number of parameters (f32 equivalent).
    pub total_params: usize,
    /// Number of parameters stored in quantized format.
    pub quantized_params: usize,
    /// Number of parameters stored as f32.
    pub float32_params: usize,
    /// Estimated memory usage in bytes.
    pub estimated_memory_bytes: usize,
}

/// Pre-allocated buffers for repeated inference calls.
///
/// Create once via [`WhisperModel::create_buffer`](crate::WhisperModel::create_buffer)
/// and pass to [`WhisperModel::transcribe_with_buffer`](crate::WhisperModel::transcribe_with_buffer)
/// to avoid repeated allocations.
pub struct InferenceBuffer {
    /// Scratch space for mel spectrogram computation.
    pub(crate) mel_buf: Vec<f32>,
}

/// Serialize a `TranscribeResult` to a pretty-printed JSON string.
///
/// Requires the `serde` feature.
#[cfg(feature = "serde")]
pub fn to_json(result: &TranscribeResult) -> Result<String, OxiWhisperError> {
    serde_json::to_string_pretty(result)
        .map_err(|e| OxiWhisperError::InferenceFailed(format!("JSON serialization failed: {e}")))
}

/// Validate transcription options before running inference.
pub(crate) fn validate_options(opts: &TranscribeOptions<'_>) -> Result<(), OxiWhisperError> {
    if opts.beam_width == 0 {
        return Err(OxiWhisperError::ConfigError(
            "beam_width must be >= 1".into(),
        ));
    }
    if opts.temperature < 0.0 {
        return Err(OxiWhisperError::ConfigError(
            "temperature must be >= 0.0".into(),
        ));
    }
    if opts.top_p <= 0.0 || opts.top_p > 1.0 {
        return Err(OxiWhisperError::ConfigError(
            "top_p must be in (0.0, 1.0]".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_options_with_previous_tokens() {
        let tokens = [100u32, 200, 300];
        let opts = TranscribeOptions {
            previous_tokens: Some(&tokens),
            ..TranscribeOptions::default()
        };
        assert_eq!(opts.previous_tokens, Some(&tokens[..]));
        assert_eq!(opts.previous_tokens.map_or(0, |t| t.len()), 3);
    }

    #[test]
    fn test_options_previous_tokens_default_none() {
        let opts = TranscribeOptions::default();
        assert!(opts.previous_tokens.is_none());
    }

    #[test]
    fn test_options_clone_with_previous_tokens() {
        let tokens = [42u32, 43];
        let opts = TranscribeOptions {
            previous_tokens: Some(&tokens),
            ..TranscribeOptions::default()
        };
        let cloned = opts.clone();
        assert_eq!(cloned.previous_tokens, Some(&tokens[..]));
    }
}
