//! Pure types for the GGUF binary format — no I/O.

use crate::OxiWhisperError;

/// GGUF metadata value type discriminant.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GgufValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl GgufValueType {
    pub(crate) fn from_u32(v: u32) -> Result<Self, OxiWhisperError> {
        match v {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            _ => Err(OxiWhisperError::InvalidModel(format!(
                "Unknown GGUF value type: {v}"
            ))),
        }
    }
}

/// A single GGUF metadata value.
#[derive(Debug, Clone)]
pub(crate) enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(GgufArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    /// Try to extract a u32 value (also accepts U64 that fits in u32, I32 ≥ 0).
    pub(crate) fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) if *v >= 0 => Some(*v as u32),
            Self::U64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            Self::U8(v) => Some(*v as u32),
            Self::U16(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Try to extract a u64 value.
    pub(crate) fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::U32(v) => Some(*v as u64),
            Self::I32(v) if *v >= 0 => Some(*v as u64),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            Self::U8(v) => Some(*v as u64),
            Self::U16(v) => Some(*v as u64),
            Self::I16(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    /// Try to coerce any scalar GgufValue to f64 for numeric comparisons.
    pub(crate) fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(v) => Some(*v),
            Self::F32(v) => Some(*v as f64),
            Self::I8(v) => Some(*v as f64),
            Self::I16(v) => Some(*v as f64),
            Self::I32(v) => Some(*v as f64),
            Self::I64(v) => Some(*v as f64),
            Self::U8(v) => Some(*v as f64),
            Self::U16(v) => Some(*v as f64),
            Self::U32(v) => Some(*v as f64),
            Self::U64(v) => Some(*v as f64),
            Self::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Try to extract a string reference.
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to extract a reference to an array.
    pub(crate) fn as_array(&self) -> Option<&GgufArray> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// A GGUF array value: a homogeneous sequence of [`GgufValue`]s.
#[derive(Debug, Clone)]
pub(crate) struct GgufArray {
    pub(crate) value_type: GgufValueType,
    pub(crate) values: Vec<GgufValue>,
}

impl GgufArray {
    /// Return the declared element type of this array.
    pub(crate) fn elem_type(&self) -> GgufValueType {
        self.value_type
    }
}

/// ggml tensor type IDs.
///
/// These match the `ggml_type` enum in the C implementation, and the **same**
/// discriminants are used by the legacy GGML whisper container — an earlier
/// note here claimed otherwise (`Q8_0 = 3` for legacy files), which is what led
/// the GGML loader to mis-map Q8_0 onto Q4_1's slot and reject Q5_1 and Q8_0
/// outright. `crate::quantize::QuantType::from_ggml_type` is the single shared
/// table for both loaders.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4 and 5 are reserved / deprecated
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    // 10-15: K-quants — unsupported, rejected with InvalidModel
}

impl GgmlType {
    pub(crate) fn from_u32(v: u32) -> Result<Self, OxiWhisperError> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            _ => Err(OxiWhisperError::InvalidModel(format!(
                "Unsupported GGUF tensor dtype: {v}"
            ))),
        }
    }
}

/// Align `offset` up to the next multiple of `alignment`.
///
/// Returns `offset` unchanged if it is already aligned or `alignment` is zero.
pub(crate) fn align_offset(offset: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return offset;
    }
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        // `alignment - remainder` cannot underflow (remainder < alignment).
        // Saturate on the outer add so an adversarial `general.alignment`
        // near `u64::MAX` cannot panic on overflow; a saturated base simply
        // seeks past EOF and is rejected downstream when tensor data is read.
        offset.saturating_add(alignment - remainder)
    }
}

/// Metadata about a single GGUF tensor (names, shape, type, offset in the
/// tensor-data section).
#[derive(Debug, Clone)]
pub(crate) struct TensorInfo {
    pub(crate) name: String,
    /// Logical shape — dimensions in GGUF order (often innermost-first).
    pub(crate) dims: Vec<u64>,
    pub(crate) ggml_type: GgmlType,
    /// Byte offset relative to the tensor-data section start (after alignment).
    pub(crate) offset: u64,
}

impl TensorInfo {
    /// Total number of logical elements (product of all dimensions).
    ///
    /// Returns `None` if the product overflows `u64`. A plain `.product()`
    /// would silently wrap in release builds (yielding a bogus, often small,
    /// element count) and *panic* in overflow-checked builds — both are
    /// unacceptable for an attacker-controlled tensor descriptor, so the
    /// multiplication is performed with `checked_mul`.
    pub(crate) fn n_elements(&self) -> Option<u64> {
        self.dims
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_alignment_math() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(31, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
        assert_eq!(align_offset(0, 64), 0);
        assert_eq!(align_offset(64, 64), 64);
        assert_eq!(align_offset(65, 64), 128);
    }

    #[test]
    fn test_gguf_value_type_round_trip() {
        for i in 0u32..=12 {
            let vt = GgufValueType::from_u32(i).expect("should parse");
            assert_eq!(vt as u32, i);
        }
        assert!(GgufValueType::from_u32(13).is_err());
    }

    #[test]
    fn test_ggml_type_supported() {
        assert!(GgmlType::from_u32(0).is_ok()); // F32
        assert!(GgmlType::from_u32(1).is_ok()); // F16
        assert!(GgmlType::from_u32(2).is_ok()); // Q4_0
        assert!(GgmlType::from_u32(3).is_ok()); // Q4_1
        assert!(GgmlType::from_u32(6).is_ok()); // Q5_0
        assert!(GgmlType::from_u32(7).is_ok()); // Q5_1
        assert!(GgmlType::from_u32(8).is_ok()); // Q8_0
        assert!(GgmlType::from_u32(9).is_ok()); // Q8_1
    }

    #[test]
    fn test_ggml_type_unsupported() {
        // K-quants and others are unsupported
        assert!(GgmlType::from_u32(10).is_err());
        assert!(GgmlType::from_u32(15).is_err());
        assert!(GgmlType::from_u32(28).is_err());
    }
}
