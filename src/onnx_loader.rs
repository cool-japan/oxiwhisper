//! ONNX model loader for Whisper.
//!
//! Loads ONNX Whisper models using `oxionnx` and converts them into
//! oxiwhisper's internal [`ModelData`] format. Supports both HuggingFace
//! Optimum naming conventions and OpenAI/GGML naming conventions, with
//! automatic detection.
//!
//! Gated behind the `onnx` feature flag.

#![cfg(feature = "onnx")]

use crate::mel_filters::generate_mel_filters;
use crate::model::{Hparams, ModelData, VocabEntry};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Configuration for loading an ONNX Whisper model.
pub struct OnnxModelConfig {
    /// Path to the ONNX model file (single file or encoder).
    pub model_path: PathBuf,
    /// Optional path to decoder ONNX file (for split models).
    pub decoder_path: Option<PathBuf>,
    /// Optional path to vocabulary JSON file (tokenizer.json or vocab.json).
    /// If `None`, a built-in minimal Whisper vocabulary is generated.
    pub vocab_path: Option<PathBuf>,
}

/// Load an ONNX Whisper model and convert it to [`ModelData`].
pub fn load_onnx(config: &OnnxModelConfig) -> Result<ModelData, String> {
    // 1. Load all ONNX weights into a flat HashMap<String, oxionnx::Tensor>.
    let mut onnx_weights = load_onnx_weights(&config.model_path)?;

    if let Some(decoder_path) = &config.decoder_path {
        let decoder_weights = load_onnx_weights(decoder_path)?;
        onnx_weights.extend(decoder_weights);
    }

    // 2. Detect naming convention.
    let naming = detect_naming(&onnx_weights);

    // 3. Build the name-mapping table and convert tensors.
    let tensors = convert_all_tensors(&onnx_weights, &naming)?;

    // 4. Infer hyperparameters from tensor shapes.
    let hparams = infer_hparams(&tensors)?;

    // 5. Generate mel filters programmatically.
    let mel_filters = generate_mel_filters();

    // 6. Load or generate vocabulary.
    let vocab = load_vocab(config.vocab_path.as_deref(), &hparams)?;

    Ok(ModelData {
        hparams,
        mel_filters,
        vocab,
        tensors,
        quantized_tensors: std::collections::HashMap::new(),
    })
}

// ---------------------------------------------------------------------------
// Naming convention detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NamingConvention {
    /// HuggingFace Optimum style: `model.encoder.layers.0.self_attn.q_proj.weight`
    HuggingFace,
    /// OpenAI / GGML style: `encoder.blocks.0.attn.query.weight`
    OpenAI,
}

fn detect_naming(weights: &HashMap<String, oxionnx::Tensor>) -> NamingConvention {
    for name in weights.keys() {
        if name.starts_with("model.encoder.") || name.starts_with("model.decoder.") {
            return NamingConvention::HuggingFace;
        }
        if name.starts_with("encoder.blocks.") || name.starts_with("decoder.blocks.") {
            return NamingConvention::OpenAI;
        }
    }
    // Default to HuggingFace if we cannot tell — the mapping will just skip
    // names that don't match.
    NamingConvention::HuggingFace
}

// ---------------------------------------------------------------------------
// ONNX weight loading
// ---------------------------------------------------------------------------

fn load_onnx_weights(path: &Path) -> Result<HashMap<String, oxionnx::Tensor>, String> {
    let session = oxionnx::Session::from_file(path)
        .map_err(|e| format!("Failed to load ONNX model '{}': {e}", path.display()))?;
    Ok(session.weights().clone())
}

// ---------------------------------------------------------------------------
// Tensor conversion
// ---------------------------------------------------------------------------

/// Convert all ONNX tensors to GGML-convention tensors.
fn convert_all_tensors(
    onnx_weights: &HashMap<String, oxionnx::Tensor>,
    naming: &NamingConvention,
) -> Result<HashMap<String, Tensor>, String> {
    let mut tensors = HashMap::new();

    for (onnx_name, onnx_tensor) in onnx_weights {
        let ggml_name = match naming {
            NamingConvention::HuggingFace => match map_hf_name(onnx_name) {
                Some(n) => n,
                None => continue, // Skip tensors we don't need (e.g. optimizer state).
            },
            NamingConvention::OpenAI => onnx_name.clone(),
        };

        let tensor = convert_single_tensor(&ggml_name, onnx_tensor)?;
        tensors.insert(ggml_name, tensor);
    }

    Ok(tensors)
}

/// Convert one ONNX tensor to the internal format, applying necessary
/// transpositions depending on the tensor's role.
fn convert_single_tensor(ggml_name: &str, onnx_tensor: &oxionnx::Tensor) -> Result<Tensor, String> {
    let data = &onnx_tensor.data;
    let shape = &onnx_tensor.shape;

    let kind = classify_tensor(ggml_name);

    match kind {
        TensorKind::Linear => transpose_2d(data, shape),
        TensorKind::Conv1d => transpose_conv1d(data, shape),
        TensorKind::TokenEmbedding => transpose_2d(data, shape),
        TensorKind::DecoderPositionalEmbedding => transpose_2d(data, shape),
        TensorKind::EncoderPositionalEmbedding => {
            // Same layout in both ONNX and GGML: [n_audio_ctx, n_audio_state]
            Ok(Tensor::from_vec(data.clone(), shape))
        }
        TensorKind::Bias | TensorKind::LayerNorm | TensorKind::Other => {
            // 1-D tensors or layer-norm weights/biases — no transposition needed.
            Ok(Tensor::from_vec(data.clone(), shape))
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor classification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorKind {
    Linear,
    Conv1d,
    TokenEmbedding,
    EncoderPositionalEmbedding,
    DecoderPositionalEmbedding,
    LayerNorm,
    Bias,
    Other,
}

fn classify_tensor(name: &str) -> TensorKind {
    // Conv layers
    if name == "encoder.conv1.weight" || name == "encoder.conv2.weight" {
        return TensorKind::Conv1d;
    }

    // Embeddings
    if name == "decoder.token_embedding.weight" {
        return TensorKind::TokenEmbedding;
    }
    if name == "encoder.positional_embedding" {
        return TensorKind::EncoderPositionalEmbedding;
    }
    if name == "decoder.positional_embedding" {
        return TensorKind::DecoderPositionalEmbedding;
    }

    // Biases (1-D)
    if name.ends_with(".bias") {
        // Layer-norm biases and attention/mlp biases are all 1-D in GGML.
        return TensorKind::Bias;
    }

    // Layer-norm weights (1-D)
    if name.contains("_ln.weight")
        || name == "encoder.ln_post.weight"
        || name == "decoder.ln.weight"
    {
        return TensorKind::LayerNorm;
    }

    // Remaining .weight tensors in blocks are linear (2-D).
    if name.ends_with(".weight") {
        return TensorKind::Linear;
    }

    TensorKind::Other
}

// ---------------------------------------------------------------------------
// Transposition helpers
// ---------------------------------------------------------------------------

/// Transpose a 2-D tensor from [rows, cols] (ONNX: [out, in]) to [cols, rows]
/// (GGML: [in, out]).
fn transpose_2d(data: &[f32], shape: &[usize]) -> Result<Tensor, String> {
    if shape.len() != 2 {
        return Err(format!(
            "transpose_2d: expected 2 dims, got {} (shape: {:?})",
            shape.len(),
            shape
        ));
    }
    let rows = shape[0];
    let cols = shape[1];
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    Ok(Tensor::from_vec(out, &[cols, rows]))
}

/// Transpose conv1d weights from ONNX [out_ch, in_ch, kernel_size] to
/// GGML [kernel_size, in_ch, out_ch].
fn transpose_conv1d(data: &[f32], shape: &[usize]) -> Result<Tensor, String> {
    if shape.len() != 3 {
        return Err(format!(
            "transpose_conv1d: expected 3 dims, got {} (shape: {:?})",
            shape.len(),
            shape
        ));
    }
    let out_ch = shape[0];
    let in_ch = shape[1];
    let kernel = shape[2];
    let total = out_ch * in_ch * kernel;
    let mut out = vec![0.0f32; total];

    // ONNX layout [o][i][k] -> GGML layout [k][i][o]
    for o in 0..out_ch {
        for i in 0..in_ch {
            for k in 0..kernel {
                let src_idx = o * in_ch * kernel + i * kernel + k;
                let dst_idx = k * in_ch * out_ch + i * out_ch + o;
                out[dst_idx] = data[src_idx];
            }
        }
    }

    Ok(Tensor::from_vec(out, &[kernel, in_ch, out_ch]))
}

// ---------------------------------------------------------------------------
// HuggingFace name mapping
// ---------------------------------------------------------------------------

/// Map a HuggingFace Optimum tensor name to the GGML-convention name.
/// Returns `None` for tensors that are not needed (optimizer state, etc.).
fn map_hf_name(hf_name: &str) -> Option<String> {
    // Strip "model." prefix if present.
    let name = hf_name.strip_prefix("model.").unwrap_or(hf_name);

    // --- Encoder top-level ---
    if name == "encoder.conv1.weight" {
        return Some("encoder.conv1.weight".into());
    }
    if name == "encoder.conv1.bias" {
        return Some("encoder.conv1.bias".into());
    }
    if name == "encoder.conv2.weight" {
        return Some("encoder.conv2.weight".into());
    }
    if name == "encoder.conv2.bias" {
        return Some("encoder.conv2.bias".into());
    }
    if name == "encoder.embed_positions.weight" {
        return Some("encoder.positional_embedding".into());
    }
    if name == "encoder.layer_norm.weight" {
        return Some("encoder.ln_post.weight".into());
    }
    if name == "encoder.layer_norm.bias" {
        return Some("encoder.ln_post.bias".into());
    }

    // --- Decoder top-level ---
    if name == "decoder.embed_tokens.weight" {
        return Some("decoder.token_embedding.weight".into());
    }
    if name == "decoder.embed_positions.weight" {
        return Some("decoder.positional_embedding".into());
    }
    if name == "decoder.layer_norm.weight" {
        return Some("decoder.ln.weight".into());
    }
    if name == "decoder.layer_norm.bias" {
        return Some("decoder.ln.bias".into());
    }

    // --- Encoder layers ---
    if let Some(rest) = name.strip_prefix("encoder.layers.") {
        return map_hf_encoder_layer(rest);
    }

    // --- Decoder layers ---
    if let Some(rest) = name.strip_prefix("decoder.layers.") {
        return map_hf_decoder_layer(rest);
    }

    // Unknown tensor — skip it.
    None
}

/// Map `"<layer_idx>.<suffix>"` for an encoder layer.
fn map_hf_encoder_layer(rest: &str) -> Option<String> {
    let (layer_idx, suffix) = split_layer_suffix(rest)?;

    let ggml_suffix = match suffix {
        // Self-attention
        "self_attn.q_proj.weight" => "attn.query.weight",
        "self_attn.q_proj.bias" => "attn.query.bias",
        "self_attn.k_proj.weight" => "attn.key.weight",
        "self_attn.k_proj.bias" => "attn.key.bias",
        "self_attn.v_proj.weight" => "attn.value.weight",
        "self_attn.v_proj.bias" => "attn.value.bias",
        "self_attn.out_proj.weight" => "attn.out.weight",
        "self_attn.out_proj.bias" => "attn.out.bias",
        "self_attn_layer_norm.weight" => "attn_ln.weight",
        "self_attn_layer_norm.bias" => "attn_ln.bias",
        // MLP
        "fc1.weight" => "mlp.0.weight",
        "fc1.bias" => "mlp.0.bias",
        "fc2.weight" => "mlp.2.weight",
        "fc2.bias" => "mlp.2.bias",
        "final_layer_norm.weight" => "mlp_ln.weight",
        "final_layer_norm.bias" => "mlp_ln.bias",
        _ => return None,
    };

    Some(format!("encoder.blocks.{layer_idx}.{ggml_suffix}"))
}

/// Map `"<layer_idx>.<suffix>"` for a decoder layer.
fn map_hf_decoder_layer(rest: &str) -> Option<String> {
    let (layer_idx, suffix) = split_layer_suffix(rest)?;

    let ggml_suffix = match suffix {
        // Self-attention
        "self_attn.q_proj.weight" => "attn.query.weight",
        "self_attn.q_proj.bias" => "attn.query.bias",
        "self_attn.k_proj.weight" => "attn.key.weight",
        "self_attn.k_proj.bias" => "attn.key.bias",
        "self_attn.v_proj.weight" => "attn.value.weight",
        "self_attn.v_proj.bias" => "attn.value.bias",
        "self_attn.out_proj.weight" => "attn.out.weight",
        "self_attn.out_proj.bias" => "attn.out.bias",
        "self_attn_layer_norm.weight" => "attn_ln.weight",
        "self_attn_layer_norm.bias" => "attn_ln.bias",
        // Cross-attention
        "encoder_attn.q_proj.weight" => "cross_attn.query.weight",
        "encoder_attn.q_proj.bias" => "cross_attn.query.bias",
        "encoder_attn.k_proj.weight" => "cross_attn.key.weight",
        "encoder_attn.k_proj.bias" => "cross_attn.key.bias",
        "encoder_attn.v_proj.weight" => "cross_attn.value.weight",
        "encoder_attn.v_proj.bias" => "cross_attn.value.bias",
        "encoder_attn.out_proj.weight" => "cross_attn.out.weight",
        "encoder_attn.out_proj.bias" => "cross_attn.out.bias",
        "encoder_attn_layer_norm.weight" => "cross_attn_ln.weight",
        "encoder_attn_layer_norm.bias" => "cross_attn_ln.bias",
        // MLP
        "fc1.weight" => "mlp.0.weight",
        "fc1.bias" => "mlp.0.bias",
        "fc2.weight" => "mlp.2.weight",
        "fc2.bias" => "mlp.2.bias",
        "final_layer_norm.weight" => "mlp_ln.weight",
        "final_layer_norm.bias" => "mlp_ln.bias",
        _ => return None,
    };

    Some(format!("decoder.blocks.{layer_idx}.{ggml_suffix}"))
}

/// Split `"3.self_attn.q_proj.weight"` into `("3", "self_attn.q_proj.weight")`.
fn split_layer_suffix(s: &str) -> Option<(&str, &str)> {
    let dot = s.find('.')?;
    let idx_str = &s[..dot];
    // Verify it's a valid integer.
    if idx_str.parse::<usize>().is_err() {
        return None;
    }
    Some((idx_str, &s[dot + 1..]))
}

// ---------------------------------------------------------------------------
// Hyperparameter inference
// ---------------------------------------------------------------------------

fn infer_hparams(tensors: &HashMap<String, Tensor>) -> Result<Hparams, String> {
    // n_audio_state from encoder.conv1.bias (shape [n_audio_state])
    let n_audio_state = get_tensor_dim(tensors, "encoder.conv1.bias", 0)?;

    // n_mels from encoder.conv1.weight — GGML shape is [kernel, in_ch, out_ch]
    // in_ch = n_mels
    let conv1_weight = tensors
        .get("encoder.conv1.weight")
        .ok_or_else(|| "Missing encoder.conv1.weight".to_string())?;
    let n_mels = if conv1_weight.shape.len() == 3 {
        conv1_weight.shape[1] // [kernel, in_ch, out_ch]
    } else {
        return Err(format!(
            "encoder.conv1.weight: expected 3 dims, got {:?}",
            conv1_weight.shape
        ));
    };

    // n_audio_ctx from encoder.positional_embedding (shape [n_audio_ctx, n_audio_state])
    let n_audio_ctx = get_tensor_dim(tensors, "encoder.positional_embedding", 0)?;

    // Count encoder layers by finding the highest block index.
    let n_audio_layer = count_blocks(tensors, "encoder.blocks.")?;

    // n_audio_head: infer from attention — each head has dim = n_audio_state / n_heads.
    // We use a known heuristic based on model size.
    let n_audio_head = infer_head_count(n_audio_state)?;

    // n_text_state from decoder.ln.weight (shape [n_text_state])
    let n_text_state = get_tensor_dim(tensors, "decoder.ln.weight", 0)?;

    // n_text_ctx from decoder.positional_embedding — GGML shape is [n_text_state, n_text_ctx]
    let n_text_ctx = get_tensor_dim(tensors, "decoder.positional_embedding", 1)?;

    // n_vocab from decoder.token_embedding.weight — GGML shape is [n_text_state, n_vocab]
    let n_vocab = get_tensor_dim(tensors, "decoder.token_embedding.weight", 1)?;

    let n_text_layer = count_blocks(tensors, "decoder.blocks.")?;
    let n_text_head = infer_head_count(n_text_state)?;

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
        ftype: 0, // f32 — ONNX models are loaded in f32.
    })
}

fn get_tensor_dim(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    dim: usize,
) -> Result<usize, String> {
    let t = tensors
        .get(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    t.shape
        .get(dim)
        .copied()
        .ok_or_else(|| format!("Tensor '{name}' has no dim {dim} (shape: {:?})", t.shape))
}

/// Count the number of transformer blocks by scanning tensor names.
fn count_blocks(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<usize, String> {
    let mut max_idx: Option<usize> = None;
    for name in tensors.keys() {
        if let Some(rest) = name.strip_prefix(prefix)
            && let Some(dot_pos) = rest.find('.')
            && let Ok(idx) = rest[..dot_pos].parse::<usize>()
        {
            max_idx = Some(match max_idx {
                Some(prev) => prev.max(idx),
                None => idx,
            });
        }
    }
    max_idx
        .map(|m| m + 1) // 0-indexed, so count = max + 1
        .ok_or_else(|| format!("No blocks found with prefix '{prefix}'"))
}

/// Infer the number of attention heads from the model dimension.
///
/// Whisper uses a fixed head dimension of 64 across all model sizes:
///   tiny:   384 / 64 =  6 heads
///   base:   512 / 64 =  8 heads
///   small:  768 / 64 = 12 heads
///   medium: 1024 / 64 = 16 heads
///   large:  1280 / 64 = 20 heads
fn infer_head_count(n_state: usize) -> Result<usize, String> {
    const HEAD_DIM: usize = 64;
    if n_state == 0 || !n_state.is_multiple_of(HEAD_DIM) {
        return Err(format!(
            "Cannot infer head count: n_state={n_state} is not a multiple of {HEAD_DIM}"
        ));
    }
    Ok(n_state / HEAD_DIM)
}

// ---------------------------------------------------------------------------
// Vocabulary loading
// ---------------------------------------------------------------------------

fn load_vocab(vocab_path: Option<&Path>, hparams: &Hparams) -> Result<Vec<VocabEntry>, String> {
    match vocab_path {
        Some(path) => {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("Cannot read vocab file '{}': {e}", path.display()))?;
            let trimmed = content.trim();
            if trimmed.starts_with('{') {
                parse_tokenizer_json(trimmed, hparams.n_vocab)
            } else {
                parse_line_vocab(trimmed, hparams.n_vocab)
            }
        }
        None => Ok(generate_minimal_vocab(hparams.n_vocab)),
    }
}

/// Parse a simple line-delimited vocabulary file (one token per line).
fn parse_line_vocab(content: &str, n_vocab: usize) -> Result<Vec<VocabEntry>, String> {
    let lines: Vec<&str> = content.lines().collect();
    let mut vocab = Vec::with_capacity(n_vocab);
    for line in &lines {
        vocab.push(VocabEntry {
            text: line.to_string(),
        });
    }
    // Pad if the file has fewer entries than n_vocab.
    while vocab.len() < n_vocab {
        let idx = vocab.len();
        vocab.push(VocabEntry {
            text: format!("<|extra_{idx}|>"),
        });
    }
    Ok(vocab)
}

/// Minimal JSON parser for HuggingFace tokenizer.json format.
///
/// Extracts the `"vocab"` object (mapping token_string -> id) from the
/// `"model"` section. This avoids pulling in a full JSON dependency.
fn parse_tokenizer_json(content: &str, n_vocab: usize) -> Result<Vec<VocabEntry>, String> {
    // Find the "vocab" section inside "model".
    // Strategy: locate `"vocab"` key followed by `{`, then parse key-value pairs.
    let vocab_start = find_vocab_object(content)?;

    let mut entries: Vec<(usize, String)> = Vec::new();
    let mut pos = vocab_start;
    let bytes = content.as_bytes();
    let len = bytes.len();

    loop {
        // Skip whitespace.
        pos = skip_whitespace(bytes, pos, len);
        if pos >= len {
            break;
        }

        // Check for closing brace.
        if bytes[pos] == b'}' {
            break;
        }

        // Skip comma between entries.
        if bytes[pos] == b',' {
            pos += 1;
            pos = skip_whitespace(bytes, pos, len);
        }

        // Parse key (string).
        if bytes.get(pos).copied() != Some(b'"') {
            break;
        }
        let (key, next_pos) = parse_json_string(bytes, pos)?;
        pos = next_pos;

        // Skip colon.
        pos = skip_whitespace(bytes, pos, len);
        if bytes.get(pos).copied() != Some(b':') {
            return Err("Expected ':' in vocab JSON".into());
        }
        pos += 1;
        pos = skip_whitespace(bytes, pos, len);

        // Parse value (integer).
        let (id, next_pos) = parse_json_int(bytes, pos)?;
        pos = next_pos;

        if id >= 0 {
            entries.push((id as usize, key));
        }
    }

    // Build ordered vocab vector.
    let actual_size = entries
        .iter()
        .map(|(id, _)| *id + 1)
        .max()
        .unwrap_or(0)
        .max(n_vocab);
    let mut vocab = Vec::with_capacity(actual_size);
    for _ in 0..actual_size {
        vocab.push(VocabEntry {
            text: String::new(),
        });
    }
    for (id, text) in entries {
        if id < vocab.len() {
            vocab[id].text = text;
        }
    }

    // Fill blanks.
    for (i, entry) in vocab.iter_mut().enumerate() {
        if entry.text.is_empty() {
            entry.text = format!("<|extra_{i}|>");
        }
    }

    Ok(vocab)
}

/// Find the byte offset of the opening `{` of the `"vocab"` object.
fn find_vocab_object(content: &str) -> Result<usize, String> {
    // Look for "vocab" followed (after optional whitespace/colon) by `{`.
    // We search for all occurrences and pick the one that's inside "model".
    let bytes = content.as_bytes();
    let len = bytes.len();

    // First try to find "model" section, then "vocab" inside it.
    let model_pos = find_key_in_json(bytes, 0, len, "model");
    let search_start = model_pos.unwrap_or(0);

    let vocab_pos = find_key_in_json(bytes, search_start, len, "vocab")
        .ok_or_else(|| "Cannot find \"vocab\" key in tokenizer JSON".to_string())?;

    // `vocab_pos` points after the closing quote of "vocab". Skip `:` and whitespace.
    let mut pos = skip_whitespace(bytes, vocab_pos, len);
    if bytes.get(pos).copied() != Some(b':') {
        return Err("Expected ':' after \"vocab\" key".into());
    }
    pos += 1;
    pos = skip_whitespace(bytes, pos, len);
    if bytes.get(pos).copied() != Some(b'{') {
        return Err("Expected '{' after \"vocab\":".into());
    }
    // Return position just after the opening brace.
    Ok(pos + 1)
}

/// Find a JSON key (a quoted string) and return the byte position right after
/// the closing quote.
fn find_key_in_json(bytes: &[u8], start: usize, end: usize, key: &str) -> Option<usize> {
    let needle = format!("\"{key}\"");
    let needle_bytes = needle.as_bytes();
    let needle_len = needle_bytes.len();
    let mut pos = start;
    while pos + needle_len <= end {
        if &bytes[pos..pos + needle_len] == needle_bytes {
            return Some(pos + needle_len);
        }
        pos += 1;
    }
    None
}

fn skip_whitespace(bytes: &[u8], mut pos: usize, len: usize) -> usize {
    while pos < len
        && (bytes[pos] == b' ' || bytes[pos] == b'\n' || bytes[pos] == b'\r' || bytes[pos] == b'\t')
    {
        pos += 1;
    }
    pos
}

/// Parse a JSON string starting at `pos` (which must be `"`).
/// Returns the unescaped string and the position after the closing `"`.
fn parse_json_string(bytes: &[u8], pos: usize) -> Result<(String, usize), String> {
    if bytes.get(pos).copied() != Some(b'"') {
        return Err(format!("Expected '\"' at position {pos}"));
    }
    let mut i = pos + 1;
    let len = bytes.len();
    let mut result = Vec::new();
    while i < len {
        match bytes[i] {
            b'"' => {
                let s = String::from_utf8(result)
                    .map_err(|e| format!("Invalid UTF-8 in JSON string at {pos}: {e}"))?;
                return Ok((s, i + 1));
            }
            b'\\' => {
                i += 1;
                if i >= len {
                    return Err("Unexpected end of string after backslash".into());
                }
                match bytes[i] {
                    b'"' => result.push(b'"'),
                    b'\\' => result.push(b'\\'),
                    b'/' => result.push(b'/'),
                    b'n' => result.push(b'\n'),
                    b'r' => result.push(b'\r'),
                    b't' => result.push(b'\t'),
                    b'u' => {
                        // Parse 4-hex-digit unicode escape.
                        if i + 4 >= len {
                            return Err("Incomplete \\u escape".into());
                        }
                        let hex_str = std::str::from_utf8(&bytes[i + 1..i + 5])
                            .map_err(|_| "Invalid \\u escape".to_string())?;
                        let code_point = u32::from_str_radix(hex_str, 16)
                            .map_err(|_| format!("Invalid \\u escape: {hex_str}"))?;

                        match code_point {
                            0xD800..=0xDBFF => {
                                // High surrogate: must be followed by \uXXXX low surrogate.
                                // Positions: i points at 'u', so:
                                //   i+1..=i+4 = high hex digits (already parsed)
                                //   i+5 = '\'
                                //   i+6 = 'u'
                                //   i+7..=i+10 = low hex digits
                                let has_backslash = bytes.get(i + 5) == Some(&b'\\');
                                let has_u = bytes.get(i + 6) == Some(&b'u');
                                let low_hex = bytes.get(i + 7..=i + 10);
                                if !has_backslash || !has_u {
                                    return Err(format!(
                                        "High surrogate U+{code_point:04X} not followed by low surrogate"
                                    ));
                                }
                                let low_bytes = low_hex.ok_or_else(|| {
                                    format!(
                                        "High surrogate U+{code_point:04X} not followed by low surrogate"
                                    )
                                })?;
                                let low_hex_str = std::str::from_utf8(low_bytes)
                                    .map_err(|_| {
                                        format!(
                                            "High surrogate U+{code_point:04X} not followed by low surrogate"
                                        )
                                    })?;
                                let low = u32::from_str_radix(low_hex_str, 16).map_err(|_| {
                                    format!(
                                        "High surrogate U+{code_point:04X} not followed by low surrogate"
                                    )
                                })?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err(format!(
                                        "High surrogate U+{code_point:04X} not followed by low surrogate"
                                    ));
                                }
                                // Combine into supplementary code point.
                                let cp =
                                    0x10000u32 + (code_point - 0xD800) * 0x400 + (low - 0xDC00);
                                let ch = char::from_u32(cp).ok_or_else(|| {
                                    format!("Invalid supplementary scalar U+{cp:05X}")
                                })?;
                                let mut buf = [0u8; 4];
                                let encoded = ch.encode_utf8(&mut buf);
                                result.extend_from_slice(encoded.as_bytes());
                                // Advance past: 4 high hex + '\' + 'u' + 4 low hex = 10 extra
                                // The outer loop adds +1, giving 11 total from 'u'.
                                i += 10;
                            }
                            0xDC00..=0xDFFF => {
                                return Err(format!(
                                    "Unexpected lone low surrogate U+{code_point:04X}"
                                ));
                            }
                            _ => {
                                // Ordinary BMP code point.
                                let ch = char::from_u32(code_point).ok_or_else(|| {
                                    format!("Invalid Unicode scalar U+{code_point:04X}")
                                })?;
                                let mut buf = [0u8; 4];
                                let encoded = ch.encode_utf8(&mut buf);
                                result.extend_from_slice(encoded.as_bytes());
                                i += 4;
                            }
                        }
                    }
                    other => {
                        result.push(b'\\');
                        result.push(other);
                    }
                }
            }
            other => result.push(other),
        }
        i += 1;
    }
    Err(format!("Unterminated JSON string starting at {pos}"))
}

/// Parse a JSON integer starting at `pos`.
/// Returns the integer and the position after the last digit.
fn parse_json_int(bytes: &[u8], pos: usize) -> Result<(i64, usize), String> {
    let len = bytes.len();
    let mut i = pos;
    let negative = if i < len && bytes[i] == b'-' {
        i += 1;
        true
    } else {
        false
    };
    let start = i;
    while i < len && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        return Err(format!("Expected integer at position {pos}"));
    }
    let num_str = std::str::from_utf8(&bytes[start..i])
        .map_err(|_| format!("Invalid integer at position {pos}"))?;
    let val: i64 = num_str
        .parse()
        .map_err(|e| format!("Cannot parse integer '{num_str}': {e}"))?;
    Ok((if negative { -val } else { val }, i))
}

// ---------------------------------------------------------------------------
// Minimal vocabulary generation
// ---------------------------------------------------------------------------

/// Generate a minimal Whisper vocabulary with byte tokens (0-255) and
/// standard special tokens.
fn generate_minimal_vocab(n_vocab: usize) -> Vec<VocabEntry> {
    let mut vocab = Vec::with_capacity(n_vocab);

    // Indices 0..255: byte tokens.
    for i in 0..256usize.min(n_vocab) {
        let text = if (32..127).contains(&i) {
            // Printable ASCII.
            String::from(i as u8 as char)
        } else {
            format!("<|byte_{i}|>")
        };
        vocab.push(VocabEntry { text });
    }

    // Fill 256..n_vocab with placeholders, then overwrite special tokens.
    while vocab.len() < n_vocab {
        let idx = vocab.len();
        vocab.push(VocabEntry {
            text: format!("<|extra_{idx}|>"),
        });
    }

    // Standard Whisper special tokens.
    let special_tokens: &[(usize, &str)] = &[
        (50256, "<|endoftext|>"),
        (50257, "<|startoftranscript|>"),
        (50258, "<|en|>"),
        (50259, "<|zh|>"),
        (50260, "<|de|>"),
        (50261, "<|es|>"),
        (50262, "<|ru|>"),
        (50263, "<|ko|>"),
        (50264, "<|fr|>"),
        (50265, "<|ja|>"),
        (50266, "<|pt|>"),
        (50267, "<|tr|>"),
        (50268, "<|pl|>"),
        (50269, "<|ca|>"),
        (50270, "<|nl|>"),
        (50271, "<|ar|>"),
        (50272, "<|sv|>"),
        (50273, "<|it|>"),
        (50274, "<|id|>"),
        (50275, "<|hi|>"),
        (50276, "<|fi|>"),
        (50277, "<|vi|>"),
        (50278, "<|he|>"),
        (50279, "<|uk|>"),
        (50280, "<|el|>"),
        (50281, "<|ms|>"),
        (50282, "<|cs|>"),
        (50283, "<|ro|>"),
        (50284, "<|da|>"),
        (50285, "<|hu|>"),
        (50286, "<|ta|>"),
        (50287, "<|no|>"),
        (50288, "<|th|>"),
        (50289, "<|ur|>"),
        (50290, "<|hr|>"),
        (50291, "<|bg|>"),
        (50292, "<|lt|>"),
        (50293, "<|la|>"),
        (50294, "<|mi|>"),
        (50295, "<|ml|>"),
        (50296, "<|cy|>"),
        (50297, "<|sk|>"),
        (50298, "<|te|>"),
        (50299, "<|fa|>"),
        (50300, "<|lv|>"),
        (50301, "<|bn|>"),
        (50302, "<|sr|>"),
        (50303, "<|az|>"),
        (50304, "<|sl|>"),
        (50305, "<|kn|>"),
        (50306, "<|et|>"),
        (50307, "<|mk|>"),
        (50308, "<|br|>"),
        (50309, "<|eu|>"),
        (50310, "<|is|>"),
        (50311, "<|hy|>"),
        (50312, "<|ne|>"),
        (50313, "<|mn|>"),
        (50314, "<|bs|>"),
        (50315, "<|kk|>"),
        (50316, "<|sq|>"),
        (50317, "<|sw|>"),
        (50318, "<|gl|>"),
        (50319, "<|mr|>"),
        (50320, "<|pa|>"),
        (50321, "<|si|>"),
        (50322, "<|km|>"),
        (50323, "<|sn|>"),
        (50324, "<|yo|>"),
        (50325, "<|so|>"),
        (50326, "<|af|>"),
        (50327, "<|oc|>"),
        (50328, "<|ka|>"),
        (50329, "<|be|>"),
        (50330, "<|tg|>"),
        (50331, "<|sd|>"),
        (50332, "<|gu|>"),
        (50333, "<|am|>"),
        (50334, "<|yi|>"),
        (50335, "<|lo|>"),
        (50336, "<|uz|>"),
        (50337, "<|fo|>"),
        (50338, "<|ht|>"),
        (50339, "<|ps|>"),
        (50340, "<|tk|>"),
        (50341, "<|nn|>"),
        (50342, "<|mt|>"),
        (50343, "<|sa|>"),
        (50344, "<|lb|>"),
        (50345, "<|my|>"),
        (50346, "<|bo|>"),
        (50347, "<|tl|>"),
        (50348, "<|mg|>"),
        (50349, "<|as|>"),
        (50350, "<|tt|>"),
        (50351, "<|haw|>"),
        (50352, "<|ln|>"),
        (50353, "<|ha|>"),
        (50354, "<|ba|>"),
        (50355, "<|jw|>"),
        (50356, "<|su|>"),
        (50357, "<|yue|>"),
        // Task/control tokens
        (50358, "<|translate|>"),
        (50359, "<|transcribe|>"),
        (50360, "<|startoflm|>"),
        (50361, "<|startofprev|>"),
        (50362, "<|nospeech|>"),
        (50363, "<|notimestamps|>"),
    ];

    for &(idx, text) in special_tokens {
        if idx < vocab.len() {
            vocab[idx].text = text.to_string();
        }
    }

    vocab
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_2d() {
        // [2, 3] -> [3, 2]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = transpose_2d(&data, &[2, 3]).expect("transpose failed");
        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_conv1d() {
        // [out_ch=2, in_ch=1, kernel=3] -> [kernel=3, in_ch=1, out_ch=2]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = transpose_conv1d(&data, &[2, 1, 3]).expect("transpose failed");
        assert_eq!(result.shape, vec![3, 1, 2]);
        // src [o=0][i=0]: [1,2,3], [o=1][i=0]: [4,5,6]
        // dst [k=0][i=0]: [1,4], [k=1][i=0]: [2,5], [k=2][i=0]: [3,6]
        assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_classify_tensor() {
        assert_eq!(classify_tensor("encoder.conv1.weight"), TensorKind::Conv1d);
        assert_eq!(classify_tensor("encoder.conv2.weight"), TensorKind::Conv1d);
        assert_eq!(
            classify_tensor("decoder.token_embedding.weight"),
            TensorKind::TokenEmbedding
        );
        assert_eq!(
            classify_tensor("encoder.positional_embedding"),
            TensorKind::EncoderPositionalEmbedding
        );
        assert_eq!(
            classify_tensor("decoder.positional_embedding"),
            TensorKind::DecoderPositionalEmbedding
        );
        assert_eq!(
            classify_tensor("encoder.blocks.0.attn.query.bias"),
            TensorKind::Bias
        );
        assert_eq!(
            classify_tensor("encoder.blocks.0.attn_ln.weight"),
            TensorKind::LayerNorm
        );
        assert_eq!(
            classify_tensor("encoder.ln_post.weight"),
            TensorKind::LayerNorm
        );
        assert_eq!(classify_tensor("decoder.ln.weight"), TensorKind::LayerNorm);
        assert_eq!(
            classify_tensor("encoder.blocks.0.attn.query.weight"),
            TensorKind::Linear
        );
        assert_eq!(
            classify_tensor("encoder.blocks.0.mlp.0.weight"),
            TensorKind::Linear
        );
    }

    #[test]
    fn test_map_hf_encoder_names() {
        assert_eq!(
            map_hf_name("model.encoder.conv1.weight"),
            Some("encoder.conv1.weight".into())
        );
        assert_eq!(
            map_hf_name("model.encoder.embed_positions.weight"),
            Some("encoder.positional_embedding".into())
        );
        assert_eq!(
            map_hf_name("model.encoder.layer_norm.weight"),
            Some("encoder.ln_post.weight".into())
        );
        assert_eq!(
            map_hf_name("model.encoder.layers.0.self_attn.q_proj.weight"),
            Some("encoder.blocks.0.attn.query.weight".into())
        );
        assert_eq!(
            map_hf_name("model.encoder.layers.3.self_attn.k_proj.weight"),
            Some("encoder.blocks.3.attn.key.weight".into())
        );
        assert_eq!(
            map_hf_name("model.encoder.layers.0.fc1.weight"),
            Some("encoder.blocks.0.mlp.0.weight".into())
        );
        assert_eq!(
            map_hf_name("model.encoder.layers.0.fc2.weight"),
            Some("encoder.blocks.0.mlp.2.weight".into())
        );
        assert_eq!(
            map_hf_name("model.encoder.layers.0.self_attn_layer_norm.weight"),
            Some("encoder.blocks.0.attn_ln.weight".into())
        );
        assert_eq!(
            map_hf_name("model.encoder.layers.0.final_layer_norm.weight"),
            Some("encoder.blocks.0.mlp_ln.weight".into())
        );
    }

    #[test]
    fn test_map_hf_decoder_names() {
        assert_eq!(
            map_hf_name("model.decoder.embed_tokens.weight"),
            Some("decoder.token_embedding.weight".into())
        );
        assert_eq!(
            map_hf_name("model.decoder.embed_positions.weight"),
            Some("decoder.positional_embedding".into())
        );
        assert_eq!(
            map_hf_name("model.decoder.layer_norm.weight"),
            Some("decoder.ln.weight".into())
        );
        assert_eq!(
            map_hf_name("model.decoder.layers.0.self_attn.q_proj.weight"),
            Some("decoder.blocks.0.attn.query.weight".into())
        );
        assert_eq!(
            map_hf_name("model.decoder.layers.0.encoder_attn.q_proj.weight"),
            Some("decoder.blocks.0.cross_attn.query.weight".into())
        );
        assert_eq!(
            map_hf_name("model.decoder.layers.0.encoder_attn_layer_norm.weight"),
            Some("decoder.blocks.0.cross_attn_ln.weight".into())
        );
        assert_eq!(
            map_hf_name("model.decoder.layers.2.fc1.bias"),
            Some("decoder.blocks.2.mlp.0.bias".into())
        );
    }

    #[test]
    fn test_map_hf_unknown_returns_none() {
        assert_eq!(map_hf_name("some.random.tensor"), None);
        assert_eq!(map_hf_name("model.encoder.layers.0.unknown_thing"), None);
    }

    #[test]
    fn test_infer_head_count() {
        assert_eq!(infer_head_count(384).expect("tiny"), 6);
        assert_eq!(infer_head_count(512).expect("base"), 8);
        assert_eq!(infer_head_count(768).expect("small"), 12);
        assert_eq!(infer_head_count(1024).expect("medium"), 16);
        assert_eq!(infer_head_count(1280).expect("large"), 20);
        assert!(infer_head_count(100).is_err());
    }

    #[test]
    fn test_generate_minimal_vocab() {
        let vocab = generate_minimal_vocab(51364);
        assert_eq!(vocab.len(), 51364);
        // Check a few byte tokens.
        assert_eq!(vocab[33].text, "!");
        assert_eq!(vocab[65].text, "A");
        // Check special tokens.
        assert_eq!(vocab[50256].text, "<|endoftext|>");
        assert_eq!(vocab[50257].text, "<|startoftranscript|>");
        assert_eq!(vocab[50363].text, "<|notimestamps|>");
        // Non-printable byte tokens.
        assert_eq!(vocab[0].text, "<|byte_0|>");
    }

    #[test]
    fn test_parse_line_vocab() {
        let content = "hello\nworld\nfoo";
        let vocab = parse_line_vocab(content, 5).expect("parse failed");
        assert_eq!(vocab.len(), 5);
        assert_eq!(vocab[0].text, "hello");
        assert_eq!(vocab[1].text, "world");
        assert_eq!(vocab[2].text, "foo");
        assert!(vocab[3].text.starts_with("<|extra_"));
    }

    #[test]
    fn test_parse_tokenizer_json() {
        let json = r#"{"model": {"vocab": {"hello": 0, "world": 1, "foo": 2}}}"#;
        let vocab = parse_tokenizer_json(json, 5).expect("parse failed");
        assert!(vocab.len() >= 5);
        assert_eq!(vocab[0].text, "hello");
        assert_eq!(vocab[1].text, "world");
        assert_eq!(vocab[2].text, "foo");
    }

    #[test]
    fn test_count_blocks() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "encoder.blocks.0.attn.query.weight".to_string(),
            Tensor::from_vec(vec![1.0], &[1]),
        );
        tensors.insert(
            "encoder.blocks.1.attn.query.weight".to_string(),
            Tensor::from_vec(vec![1.0], &[1]),
        );
        tensors.insert(
            "encoder.blocks.3.attn.query.weight".to_string(),
            Tensor::from_vec(vec![1.0], &[1]),
        );
        // Max index is 3, so count = 4.
        assert_eq!(count_blocks(&tensors, "encoder.blocks.").expect("count"), 4);
    }

    #[test]
    fn test_json_string_escapes() {
        let bytes = br#""hello\nworld""#;
        let (s, _) = parse_json_string(bytes, 0).expect("parse");
        assert_eq!(s, "hello\nworld");

        let bytes = br#""a\\b""#;
        let (s, _) = parse_json_string(bytes, 0).expect("parse");
        assert_eq!(s, "a\\b");

        let bytes = br#""\u0041""#;
        let (s, _) = parse_json_string(bytes, 0).expect("parse");
        assert_eq!(s, "A");
    }

    #[test]
    fn test_split_layer_suffix() {
        assert_eq!(
            split_layer_suffix("3.self_attn.q_proj.weight"),
            Some(("3", "self_attn.q_proj.weight"))
        );
        assert_eq!(
            split_layer_suffix("12.fc1.weight"),
            Some(("12", "fc1.weight"))
        );
        assert_eq!(split_layer_suffix("abc.fc1.weight"), None);
    }

    #[test]
    fn test_parse_tokenizer_json_handles_unicode_escapes() {
        // \uXXXX escape sequences in vocab text should be decoded properly.
        // parse_json_string does handle \uXXXX via char::from_u32 + encode_utf8.
        // The JSON key "café" (café) should decode to the UTF-8 string "café".
        let json = r#"{"model": {"vocab": {"café": 0, "naïve": 1}}}"#;
        let vocab = parse_tokenizer_json(json, 3).expect("parse failed");
        assert!(vocab.len() >= 2, "vocab should have at least 2 entries");
        assert_eq!(vocab[0].text, "café", "\\u00e9 must decode to é");
        assert_eq!(vocab[1].text, "naïve", "\\u00ef must decode to ï");
    }

    #[test]
    fn test_parse_tokenizer_json_rejects_unpaired_surrogate() {
        // Lone high surrogate \uD800 is not a valid Unicode scalar value.
        // After BUG1 fix: parse_json_string must return Err for lone surrogates.
        let json = r#"{"model": {"vocab": {"\uD800bad": 0, "ok": 1}}}"#;
        let result = parse_tokenizer_json(json, 3);
        assert!(result.is_err(), "lone high surrogate must be rejected");
    }

    #[test]
    fn test_parse_tokenizer_json_handles_supplementary_plane() {
        // Literal UTF-8 supplementary characters in JSON keys pass through correctly.
        let bytes_smiley = "\"😀\"".as_bytes();
        let (s, _) = parse_json_string(bytes_smiley, 0).expect("literal 😀 should parse");
        assert_eq!(s, "😀");

        let bytes_math = "\"𝐀\"".as_bytes();
        let (s, _) = parse_json_string(bytes_math, 0).expect("literal 𝐀 should parse");
        assert_eq!(s, "𝐀");

        let bytes_cjk = "\"𠀀\"".as_bytes();
        let (s, _) = parse_json_string(bytes_cjk, 0).expect("literal 𠀀 should parse");
        assert_eq!(s, "𠀀");

        // Surrogate-pair \u escape form: 😀 → 😀 (U+1F600)
        // We must build the raw bytes explicitly because br#"..."# would just be UTF-8.
        let bytes_escape = b"\"\\uD83D\\uDE00\"";
        let (s, _) =
            parse_json_string(bytes_escape, 0).expect("surrogate pair \\uD83D\\uDE00 should parse");
        assert_eq!(s, "😀", "\\uD83D\\uDE00 must decode to 😀");

        // Surrogate pair: 𝄞 → 𝄞 (U+1D11E, MUSICAL SYMBOL G CLEF)
        let bytes_music = b"\"\\uD834\\uDD1E\"";
        let (s, _) =
            parse_json_string(bytes_music, 0).expect("surrogate pair \\uD834\\uDD1E should parse");
        assert_eq!(s, "\u{1D11E}", "\\uD834\\uDD1E must decode to U+1D11E");

        // Surrogate pair: 𠀀 → 𠀀 (U+20000, CJK Extension B)
        let bytes_cjk_escape = b"\"\\uD840\\uDC00\"";
        let (s, _) = parse_json_string(bytes_cjk_escape, 0)
            .expect("surrogate pair \\uD840\\uDC00 should parse");
        assert_eq!(s, "\u{20000}", "\\uD840\\uDC00 must decode to U+20000");
    }

    #[test]
    fn test_parse_tokenizer_json_rejects_lone_high_surrogate() {
        // A bare \uD800 (without a following low surrogate) must be rejected.
        let bytes = br#""\uD800bad""#;
        let result = parse_json_string(bytes, 0);
        assert!(
            result.is_err(),
            "lone high surrogate \\uD800 must be rejected"
        );
    }

    #[test]
    fn test_parse_tokenizer_json_rejects_lone_low_surrogate() {
        // A lone low surrogate \uDC00 must be rejected.
        let bytes = br#""\uDC00""#;
        let result = parse_json_string(bytes, 0);
        assert!(
            result.is_err(),
            "lone low surrogate \\uDC00 must be rejected"
        );
    }

    #[test]
    fn test_parse_tokenizer_json_rejects_high_surrogate_followed_by_non_low() {
        // High surrogate followed by 'A' instead of \uXXXX — must be rejected.
        let bytes = br#""\uD83DA""#;
        let result = parse_json_string(bytes, 0);
        assert!(
            result.is_err(),
            "high surrogate followed by 'A' must be rejected"
        );

        // High surrogate followed by arbitrary text (no second \u) — must be rejected.
        let bytes2 = br#""\uD83Dabc""#;
        let result2 = parse_json_string(bytes2, 0);
        assert!(
            result2.is_err(),
            "high surrogate not followed by \\u must be rejected"
        );
    }

    #[test]
    fn test_parse_tokenizer_json_rejects_truncated_after_high_surrogate() {
        // High surrogate with nothing following it (string ends) — must be rejected.
        let bytes = br#""\uD83D""#;
        let result = parse_json_string(bytes, 0);
        assert!(result.is_err(), "truncated high surrogate must be rejected");
    }
}
