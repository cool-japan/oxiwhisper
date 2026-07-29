//! GGUF v3 binary parser.
//!
//! Reads GGUF files from any `Read + Seek` source. The magic bytes have
//! already been consumed by `model::load_from_reader` before this is called.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use half::f16;

use crate::OxiWhisperError;
use crate::model::ModelData;

use super::spec::{GgmlType, GgufArray, GgufValue, GgufValueType, TensorInfo, align_offset};
use super::whisper::{build_hparams, build_vocab, resolve_mel_filters};

// ── Type aliases ─────────────────────────────────────────────────────────────

/// Return type for `load_tensors`: plain f32 tensors + quantized tensors.
type TensorMaps = (
    HashMap<String, crate::tensor::Tensor>,
    HashMap<String, crate::quantize::QuantizedTensor>,
);

// ── GGUF v3 constants ─────────────────────────────────────────────────────────

const GGUF_VERSION_SUPPORTED: u32 = 3;
/// Guard against memory bombs from malformed files.
const MAX_SAFE_COUNT: u64 = 1_000_000;
/// Default tensor-data alignment when `general.alignment` KV is absent.
const DEFAULT_ALIGNMENT: u64 = 32;

// ── Public entry point ────────────────────────────────────────────────────────

/// Load a GGUF v3 whisper model from `reader`.
///
/// The caller must have already read and verified the 4-byte GGUF magic
/// `[0x47, 0x47, 0x55, 0x46]` and positioned the stream immediately after it.
pub(crate) fn load_gguf_from_reader<R: Read + Seek>(
    reader: &mut R,
) -> Result<ModelData, OxiWhisperError> {
    // ── Header ────────────────────────────────────────────────────────────────
    let version = read_u32_le(reader)?;
    if version != GGUF_VERSION_SUPPORTED {
        return Err(OxiWhisperError::InvalidModel(format!(
            "Unsupported GGUF version {version}; only v3 is supported"
        )));
    }

    let tensor_count = read_u64_le(reader)?;
    let metadata_kv_count = read_u64_le(reader)?;

    if tensor_count > MAX_SAFE_COUNT {
        return Err(OxiWhisperError::InvalidModel(format!(
            "Implausible tensor count: {tensor_count}"
        )));
    }
    if metadata_kv_count > MAX_SAFE_COUNT {
        return Err(OxiWhisperError::InvalidModel(format!(
            "Implausible metadata KV count: {metadata_kv_count}"
        )));
    }

    // ── Metadata KV pairs ─────────────────────────────────────────────────────
    let kv = read_kv_map(reader, metadata_kv_count)?;

    // ── Tensor infos ──────────────────────────────────────────────────────────
    let tensor_infos = read_tensor_infos(reader, tensor_count)?;

    // ── Alignment & data-section base ─────────────────────────────────────────
    let alignment = kv
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_ALIGNMENT);

    // Align stream position to the data section.
    let current_pos = reader.stream_position()?;
    let data_base = align_offset(current_pos, alignment);
    reader.seek(SeekFrom::Start(data_base))?;

    // ── Build ModelData ───────────────────────────────────────────────────────
    let hparams = build_hparams(&kv)?;
    let vocab = build_vocab(&kv, hparams.n_vocab)?;

    // Load tensors
    let (tensors, quantized_tensors) = load_tensors(reader, &tensor_infos, data_base)?;

    // Resolve mel filters: tensor > KV > generated fallback
    let mel_filters = resolve_mel_filters(&kv, &tensors, hparams.n_mels)?;

    Ok(ModelData {
        hparams,
        mel_filters,
        vocab,
        tensors,
        quantized_tensors,
    })
}

// ── Tensor loading ────────────────────────────────────────────────────────────

/// Reject a tensor whose declared data range does not fit within the file.
///
/// This is the allocation guard that turns a file-controlled length into a
/// bounded allocation: since `need` bytes must physically exist between
/// `abs_offset` and `stream_len`, any buffer we subsequently allocate is at
/// most the size of the input, never an attacker-chosen astronomical value.
fn ensure_tensor_data_fits(
    name: &str,
    abs_offset: u64,
    need: u64,
    stream_len: u64,
) -> Result<(), OxiWhisperError> {
    let end = abs_offset.checked_add(need).ok_or_else(|| {
        OxiWhisperError::InvalidModel(format!("Tensor '{name}' data range overflows u64"))
    })?;
    if end > stream_len {
        return Err(OxiWhisperError::InvalidModel(format!(
            "Tensor '{name}' data extends beyond end of file \
             (needs {need} bytes at offset {abs_offset}, file is {stream_len} bytes)"
        )));
    }
    Ok(())
}

fn load_tensors<R: Read + Seek>(
    reader: &mut R,
    infos: &[TensorInfo],
    data_base: u64,
) -> Result<TensorMaps, OxiWhisperError> {
    use crate::quantize::{QuantType, QuantizedTensor};
    use crate::tensor::Tensor;

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut quantized_tensors: HashMap<String, QuantizedTensor> = HashMap::new();

    // Total length of the input stream. Every tensor allocation below is bounded
    // against `stream_len` so a file-controlled dimension/length can never drive
    // an unbounded (OOM) allocation: we refuse to allocate a buffer whose backing
    // bytes do not actually exist within the file.
    let stream_len = reader.seek(SeekFrom::End(0)).map_err(OxiWhisperError::Io)?;

    for info in infos {
        // Element count is computed with overflow checking; adversarial dims
        // whose product exceeds `u64` (or `usize` on this platform) are rejected
        // rather than silently wrapping or panicking.
        let n_elements_u64 = info.n_elements().ok_or_else(|| {
            OxiWhisperError::InvalidModel(format!(
                "Tensor '{}' element count overflows u64",
                info.name
            ))
        })?;
        let n_elements = usize::try_from(n_elements_u64).map_err(|_| {
            OxiWhisperError::InvalidModel(format!(
                "Tensor '{}' element count {n_elements_u64} does not fit in usize",
                info.name
            ))
        })?;
        // GGUF dims are innermost-first; convert to usize for shape
        let shape: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        let is_large_2d = shape.len() == 2 && n_elements > 1024;

        // Seek to the tensor's data.
        let abs_offset = data_base.checked_add(info.offset).ok_or_else(|| {
            OxiWhisperError::InvalidModel(format!("Tensor '{}' offset overflow", info.name))
        })?;
        reader
            .seek(SeekFrom::Start(abs_offset))
            .map_err(OxiWhisperError::Io)?;

        match info.ggml_type {
            GgmlType::F32 => {
                let need = n_elements_u64.checked_mul(4).ok_or_else(|| {
                    OxiWhisperError::InvalidModel(format!(
                        "Tensor '{}' byte size overflows u64",
                        info.name
                    ))
                })?;
                ensure_tensor_data_fits(&info.name, abs_offset, need, stream_len)?;
                let mut data = vec![0.0f32; n_elements];
                let byte_slice = unsafe {
                    std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n_elements * 4)
                };
                reader.read_exact(byte_slice).map_err(OxiWhisperError::Io)?;
                tensors.insert(info.name.clone(), Tensor::from_vec(data, &shape));
            }

            GgmlType::F16 => {
                let need = n_elements_u64.checked_mul(2).ok_or_else(|| {
                    OxiWhisperError::InvalidModel(format!(
                        "Tensor '{}' byte size overflows u64",
                        info.name
                    ))
                })?;
                ensure_tensor_data_fits(&info.name, abs_offset, need, stream_len)?;
                let mut raw = vec![0u16; n_elements];
                let byte_slice = unsafe {
                    std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u8, n_elements * 2)
                };
                reader.read_exact(byte_slice).map_err(OxiWhisperError::Io)?;
                let data: Vec<f32> = raw
                    .iter()
                    .map(|&bits| f16::from_bits(bits).to_f32())
                    .collect();
                tensors.insert(info.name.clone(), Tensor::from_vec(data, &shape));
            }

            other => {
                // Everything that is not F32/F16 is a block-quantized type. The
                // `ggml_type` discriminants are shared between GGUF and the
                // legacy GGML container, so one table serves both loaders.
                let qtype = QuantType::from_ggml_type(other as u32).ok_or_else(|| {
                    OxiWhisperError::InvalidModel(format!(
                        "Unsupported GGUF dtype {other:?} for tensor '{}'; \
                         supported: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0",
                        info.name
                    ))
                })?;
                let block_size = qtype.block_size();
                if !n_elements.is_multiple_of(block_size) {
                    return Err(OxiWhisperError::InvalidModel(format!(
                        "Tensor '{}' has {n_elements} elements, not a multiple of the {} \
                         block size {block_size}",
                        info.name,
                        qtype.name()
                    )));
                }
                let n_blocks = n_elements / block_size;
                let n_bytes = n_blocks.checked_mul(qtype.block_bytes()).ok_or_else(|| {
                    OxiWhisperError::InvalidModel(format!(
                        "Tensor '{}' byte size overflows usize",
                        info.name
                    ))
                })?;
                ensure_tensor_data_fits(&info.name, abs_offset, n_bytes as u64, stream_len)?;
                let mut raw = vec![0u8; n_bytes];
                reader.read_exact(&mut raw).map_err(OxiWhisperError::Io)?;

                if is_large_2d {
                    quantized_tensors
                        .insert(info.name.clone(), QuantizedTensor { raw, shape, qtype });
                } else {
                    let data = crate::quantize::dequantize(&raw, n_elements, qtype);
                    tensors.insert(info.name.clone(), Tensor::from_vec(data, &shape));
                }
            }
        }
    }

    Ok((tensors, quantized_tensors))
}

// ── KV metadata reader ────────────────────────────────────────────────────────

fn read_kv_map<R: Read>(
    reader: &mut R,
    count: u64,
) -> Result<HashMap<String, GgufValue>, OxiWhisperError> {
    let mut map = HashMap::with_capacity(count.min(256) as usize);
    for _ in 0..count {
        let key = read_string(reader)?;
        let vtype_raw = read_u32_le(reader)?;
        let vtype = GgufValueType::from_u32(vtype_raw)?;
        let value = read_gguf_value(reader, vtype)?;
        map.insert(key, value);
    }
    Ok(map)
}

fn read_gguf_value<R: Read>(
    reader: &mut R,
    vtype: GgufValueType,
) -> Result<GgufValue, OxiWhisperError> {
    match vtype {
        GgufValueType::U8 => Ok(GgufValue::U8(read_u8(reader)?)),
        GgufValueType::I8 => Ok(GgufValue::I8(read_u8(reader)? as i8)),
        GgufValueType::U16 => Ok(GgufValue::U16(read_u16_le(reader)?)),
        GgufValueType::I16 => Ok(GgufValue::I16(read_u16_le(reader)? as i16)),
        GgufValueType::U32 => Ok(GgufValue::U32(read_u32_le(reader)?)),
        GgufValueType::I32 => Ok(GgufValue::I32(read_u32_le(reader)? as i32)),
        GgufValueType::F32 => Ok(GgufValue::F32(read_f32_le(reader)?)),
        GgufValueType::Bool => Ok(GgufValue::Bool(read_u8(reader)? != 0)),
        GgufValueType::String => Ok(GgufValue::String(read_string(reader)?)),
        GgufValueType::Array => {
            let elem_type_raw = read_u32_le(reader)?;
            let elem_type = GgufValueType::from_u32(elem_type_raw)?;
            let elem_count = read_u64_le(reader)?;
            if elem_count > MAX_SAFE_COUNT {
                return Err(OxiWhisperError::InvalidModel(format!(
                    "Implausible array element count: {elem_count}"
                )));
            }
            let mut values = Vec::with_capacity(elem_count.min(4096) as usize);
            for _ in 0..elem_count {
                values.push(read_gguf_value(reader, elem_type)?);
            }
            Ok(GgufValue::Array(GgufArray {
                value_type: elem_type,
                values,
            }))
        }
        GgufValueType::U64 => Ok(GgufValue::U64(read_u64_le(reader)?)),
        GgufValueType::I64 => Ok(GgufValue::I64(read_u64_le(reader)? as i64)),
        GgufValueType::F64 => {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf).map_err(OxiWhisperError::Io)?;
            Ok(GgufValue::F64(f64::from_le_bytes(buf)))
        }
    }
}

// ── Tensor info reader ────────────────────────────────────────────────────────

fn read_tensor_infos<R: Read>(
    reader: &mut R,
    count: u64,
) -> Result<Vec<TensorInfo>, OxiWhisperError> {
    let mut infos = Vec::with_capacity(count.min(4096) as usize);
    for _ in 0..count {
        let name = read_string(reader)?;
        let n_dims = read_u32_le(reader)?;
        if n_dims > 4 {
            return Err(OxiWhisperError::InvalidModel(format!(
                "Tensor '{}' has {n_dims} dimensions; max supported is 4",
                name
            )));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64_le(reader)?);
        }
        let type_raw = read_u32_le(reader)?;
        let ggml_type = GgmlType::from_u32(type_raw).map_err(|_| {
            OxiWhisperError::InvalidModel(format!(
                "Unsupported GGUF dtype {} for tensor '{}'",
                type_raw, name
            ))
        })?;
        let offset = read_u64_le(reader)?;
        infos.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
        });
    }
    Ok(infos)
}

// ── Low-level I/O helpers ─────────────────────────────────────────────────────

pub(crate) fn read_u8<R: Read>(r: &mut R) -> Result<u8, OxiWhisperError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(OxiWhisperError::Io)?;
    Ok(buf[0])
}

pub(crate) fn read_u16_le<R: Read>(r: &mut R) -> Result<u16, OxiWhisperError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(OxiWhisperError::Io)?;
    Ok(u16::from_le_bytes(buf))
}

pub(crate) fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, OxiWhisperError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(OxiWhisperError::Io)?;
    Ok(u32::from_le_bytes(buf))
}

pub(crate) fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, OxiWhisperError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(OxiWhisperError::Io)?;
    Ok(u64::from_le_bytes(buf))
}

pub(crate) fn read_f32_le<R: Read>(r: &mut R) -> Result<f32, OxiWhisperError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(OxiWhisperError::Io)?;
    Ok(f32::from_le_bytes(buf))
}

/// Read a GGUF string: u64 length followed by UTF-8 bytes (no NUL terminator).
pub(crate) fn read_string<R: Read>(r: &mut R) -> Result<String, OxiWhisperError> {
    let len = read_u64_le(r)?;
    if len > MAX_SAFE_COUNT {
        return Err(OxiWhisperError::InvalidModel(format!(
            "Implausible string length: {len}"
        )));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).map_err(OxiWhisperError::Io)?;
    String::from_utf8(buf)
        .map_err(|e| OxiWhisperError::InvalidModel(format!("GGUF string is not valid UTF-8: {e}")))
}

// Property-based adversarial hardening tests live in a child file to keep this
// module well under the 2000-line ceiling.
#[cfg(test)]
#[path = "parse_proptest.rs"]
mod parse_proptest;
