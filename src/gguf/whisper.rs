//! Whisper-model-specific logic for mapping GGUF KV metadata to `ModelData`.

use std::collections::HashMap;

use crate::OxiWhisperError;
use crate::model::{Hparams, VocabEntry};
use crate::tensor::Tensor;

use super::spec::{GgufValue, GgufValueType};

// ── Key resolution ────────────────────────────────────────────────────────────

/// Try candidate keys in order; return a reference to the first found value.
///
/// This lets the parser accept both `whisper.encoder.context_length`-style
/// keys (from standard GGUF converters) and any alternative spellings that
/// might appear in community-converted files.
pub(crate) fn resolve_key<'a>(
    kv: &'a HashMap<String, GgufValue>,
    candidates: &[&str],
) -> Result<&'a GgufValue, OxiWhisperError> {
    for &key in candidates {
        if let Some(v) = kv.get(key) {
            return Ok(v);
        }
    }
    Err(OxiWhisperError::InvalidModel(format!(
        "Required GGUF key not found; tried: {:?}",
        candidates
    )))
}

/// Extract a u32 hyperparameter from the KV map, trying candidate keys.
fn req_u32(kv: &HashMap<String, GgufValue>, candidates: &[&str]) -> Result<u32, OxiWhisperError> {
    let v = resolve_key(kv, candidates)?;
    v.as_u32().ok_or_else(|| {
        OxiWhisperError::InvalidModel(format!(
            "Expected integer value for key(s) {:?}, got {:?}",
            candidates, v
        ))
    })
}

// ── Hyperparameter builder ────────────────────────────────────────────────────

/// Build [`Hparams`] from GGUF KV metadata.
///
/// Uses the canonical key names from the GGUF spec (ggml/docs/gguf.md) for
/// Whisper models.  A short candidate list is checked in priority order so
/// that minor naming variations are tolerated.
pub(crate) fn build_hparams(kv: &HashMap<String, GgufValue>) -> Result<Hparams, OxiWhisperError> {
    // n_vocab — number of vocabulary tokens
    let n_vocab = req_u32(
        kv,
        &[
            "whisper.vocab_size",
            "tokenizer.ggml.tokens_count",
            "general.vocab_size",
        ],
    )? as usize;

    // Audio encoder context length (n_audio_ctx)
    let n_audio_ctx = req_u32(
        kv,
        &[
            "whisper.encoder.context_length",
            "whisper.audio.encoder.context_length",
            "whisper.n_audio_ctx",
        ],
    )? as usize;

    // Audio encoder embedding / model dimension (n_audio_state)
    let n_audio_state = req_u32(
        kv,
        &[
            "whisper.encoder.embedding_length",
            "whisper.audio.encoder.embedding_length",
            "whisper.n_audio_state",
        ],
    )? as usize;

    // Audio encoder attention heads (n_audio_head)
    let n_audio_head = req_u32(
        kv,
        &[
            "whisper.encoder.attention.head_count",
            "whisper.audio.encoder.attention.head_count",
            "whisper.n_audio_head",
        ],
    )? as usize;

    // Audio encoder transformer block count (n_audio_layer)
    let n_audio_layer = req_u32(
        kv,
        &[
            "whisper.encoder.block_count",
            "whisper.audio.encoder.block_count",
            "whisper.n_audio_layer",
        ],
    )? as usize;

    // Text decoder context length (n_text_ctx)
    let n_text_ctx = req_u32(
        kv,
        &[
            "whisper.decoder.context_length",
            "whisper.text.decoder.context_length",
            "whisper.n_text_ctx",
        ],
    )? as usize;

    // Text decoder embedding / model dimension (n_text_state)
    let n_text_state = req_u32(
        kv,
        &[
            "whisper.decoder.embedding_length",
            "whisper.text.decoder.embedding_length",
            "whisper.n_text_state",
        ],
    )? as usize;

    // Text decoder attention heads (n_text_head)
    let n_text_head = req_u32(
        kv,
        &[
            "whisper.decoder.attention.head_count",
            "whisper.text.decoder.attention.head_count",
            "whisper.n_text_head",
        ],
    )? as usize;

    // Text decoder transformer block count (n_text_layer)
    let n_text_layer = req_u32(
        kv,
        &[
            "whisper.decoder.block_count",
            "whisper.text.decoder.block_count",
            "whisper.n_text_layer",
        ],
    )? as usize;

    // Number of mel filter banks (n_mels)
    let n_mels = req_u32(
        kv,
        &[
            "whisper.encoder.mels_count",
            "whisper.audio.encoder.mels_count",
            "whisper.n_mels",
        ],
    )? as usize;

    // File type / quantization type (ftype) — optional, default 0 = F32
    let ftype = kv
        .get("general.file_type")
        .and_then(|v| v.as_u32())
        .map(|v| v as i32)
        .unwrap_or(0);

    Ok(Hparams {
        n_vocab,
        n_audio_ctx,
        n_audio_state,
        n_audio_head,
        n_audio_layer,
        n_text_ctx,
        n_text_state,
        n_text_head,
        n_text_layer,
        n_mels,
        ftype,
    })
}

// ── Vocabulary builder ────────────────────────────────────────────────────────

/// Resolve vocabulary tokens from GGUF KV metadata.
///
/// The canonical key is `tokenizer.ggml.tokens` which stores an array of
/// strings (one per vocabulary entry).  If the model has fewer tokens stored
/// than `n_vocab`, synthetic `<|i|>` entries are appended to pad the list.
pub(crate) fn build_vocab(
    kv: &HashMap<String, GgufValue>,
    n_vocab: usize,
) -> Result<Vec<VocabEntry>, OxiWhisperError> {
    // Standard tokenizer token list key
    let tokens_value = kv
        .get("tokenizer.ggml.tokens")
        .or_else(|| kv.get("whisper.tokenizer.tokens"));

    let mut vocab: Vec<VocabEntry> = if let Some(val) = tokens_value {
        let arr = val.as_array().ok_or_else(|| {
            OxiWhisperError::InvalidModel("tokenizer.ggml.tokens is not an array".to_string())
        })?;
        // Validate that the array element type is STRING (or at least coercible).
        if arr.elem_type() != GgufValueType::String && arr.elem_type() != GgufValueType::U8 {
            // Non-string token arrays are unexpected; proceed with best-effort coercion.
        }
        // Store the raw bytes of each token. GGUF strings are UTF-8 by spec, so
        // `as_bytes` is byte-exact here; the decoder joins bytes and converts
        // once, exactly as for the legacy GGML container.
        arr.values
            .iter()
            .map(|v| VocabEntry::from_text(v.as_str().unwrap_or("")))
            .collect()
    } else {
        Vec::new()
    };

    // Pad with synthetic entries if the array is shorter than n_vocab.
    while vocab.len() < n_vocab {
        vocab.push(VocabEntry::from_text(format!("<|{}|>", vocab.len())));
    }

    // Truncate if somehow longer.
    vocab.truncate(n_vocab);

    Ok(vocab)
}

// ── Mel filter resolver ───────────────────────────────────────────────────────

/// Resolve the mel filter bank using a three-tier strategy:
///
/// 1. A tensor named `"mel_filters"` already loaded in `tensors`.
/// 2. A KV entry `"whisper.mel_filters"` containing an f32 array.
/// 3. Programmatic generation via [`crate::mel_filters::generate_mel_filters`]
///    (works only for the standard 80-mel configuration).
///
/// # Errors
///
/// Returns [`OxiWhisperError::InvalidModel`] when the file carries no filter
/// bank and `n_mels` is not 80. Earlier versions silently substituted an
/// **all-zero** filter bank in that case, which made every non-80-mel model
/// (notably `large-v3`, 128 mels) transcribe pure silence while reporting
/// success.
pub(crate) fn resolve_mel_filters(
    kv: &HashMap<String, GgufValue>,
    tensors: &HashMap<String, Tensor>,
    n_mels: usize,
) -> Result<Vec<f32>, OxiWhisperError> {
    // Tier 1: dedicated tensor
    if let Some(tensor) = tensors.get("mel_filters") {
        return Ok(tensor.data.clone());
    }

    // Tier 2: KV array (accept any numeric value convertible via as_f64)
    if let Some(arr) = kv.get("whisper.mel_filters").and_then(|v| v.as_array()) {
        let filters: Vec<f32> = arr
            .values
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        if !filters.is_empty() {
            return Ok(filters);
        }
    }

    // Tier 3: programmatic fallback (standard 80-mel filter bank)
    if n_mels == crate::mel::WHISPER_N_MELS {
        return Ok(crate::mel_filters::generate_mel_filters());
    }

    Err(OxiWhisperError::InvalidModel(format!(
        "GGUF model declares {n_mels} mel channels but carries no mel filter bank \
         (neither a `mel_filters` tensor nor a `whisper.mel_filters` KV array), and \
         oxiwhisper can only generate the standard {}-channel bank",
        crate::mel::WHISPER_N_MELS
    )))
}
