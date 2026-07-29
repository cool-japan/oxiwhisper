//! Core quantization types, block-size constants, and the QuantizedTensor struct.

/// Type of quantization.
///
/// The discriminants deliberately mirror the `ggml_type` enum used by both the
/// legacy GGML whisper format and GGUF (see [`QuantType::from_ggml_type`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// GGML Q4_0 (`ggml_type` 2): 4-bit symmetric, 32 values per block.
    Q4_0,
    /// GGML Q4_1 (`ggml_type` 3): 4-bit affine (scale + min), 32 values per block.
    Q4_1,
    /// GGML Q5_0 (`ggml_type` 6): 5-bit symmetric, 32 values per block.
    Q5_0,
    /// GGML Q5_1 (`ggml_type` 7): 5-bit affine (scale + min), 32 values per block.
    Q5_1,
    /// GGML Q8_0 (`ggml_type` 8): 8-bit symmetric, 32 values per block.
    Q8_0,
}

impl QuantType {
    /// Map a raw `ggml_type` discriminant to a [`QuantType`].
    ///
    /// Returns `None` for non-quantized types (`0` = F32, `1` = F16) and for
    /// quantization schemes this crate does not implement (Q8_1, the K-quants,
    /// the IQ family, …). Callers must surface a clear "unsupported" error
    /// rather than guessing a layout.
    ///
    /// The mapping is identical for the legacy GGML whisper container and for
    /// GGUF — both store the same `ggml_type` discriminants.
    pub fn from_ggml_type(ggml_type: u32) -> Option<Self> {
        match ggml_type {
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            _ => None,
        }
    }

    /// Human-readable GGML name (`"Q4_0"`, `"Q5_1"`, …).
    pub fn name(self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
        }
    }

    /// Number of logical f32 values encoded by one block (always 32).
    pub fn block_size(self) -> usize {
        match self {
            Self::Q4_0 => Q4_0_BLOCK_SIZE,
            Self::Q4_1 => Q4_1_BLOCK_SIZE,
            Self::Q5_0 => Q5_0_BLOCK_SIZE,
            Self::Q5_1 => Q5_1_BLOCK_SIZE,
            Self::Q8_0 => Q8_0_BLOCK_SIZE,
        }
    }

    /// On-disk byte size of one block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => Q4_0_BLOCK_BYTES,
            Self::Q4_1 => Q4_1_BLOCK_BYTES,
            Self::Q5_0 => Q5_0_BLOCK_BYTES,
            Self::Q5_1 => Q5_1_BLOCK_BYTES,
            Self::Q8_0 => Q8_0_BLOCK_BYTES,
        }
    }
}

/// A tensor stored in its original GGML quantized format.
/// Avoids the eager dequantization to f32, saving 2-8x memory.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Raw quantized bytes (block-structured).
    pub raw: Vec<u8>,
    /// Logical shape [rows, cols] (the full f32 shape).
    pub shape: Vec<usize>,
    /// Quantization type.
    pub qtype: QuantType,
}

impl QuantizedTensor {
    /// Number of logical elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Block size for this quantization type.
    pub fn block_size(&self) -> usize {
        self.qtype.block_size()
    }

    /// Bytes per block for this quantization type.
    pub fn block_bytes(&self) -> usize {
        self.qtype.block_bytes()
    }

    /// Number of elements per physical row in the quantized data.
    ///
    /// For a 2D weight with shape `[in_f, out_f]`, the physical memory layout is
    /// `out_f` rows of `in_f` elements each (same as the GEMV layout in `linear()`).
    /// So the row length in elements is `shape[0]` (the first / "inner" dimension).
    pub fn row_elements(&self) -> usize {
        if self.shape.len() >= 2 {
            self.shape[0]
        } else {
            self.numel()
        }
    }

    /// Compute the raw byte offset for a given row index.
    pub fn row_byte_offset(&self, row: usize) -> usize {
        let elems = self.row_elements();
        let blocks_per_row = elems / self.block_size();
        row * blocks_per_row * self.block_bytes()
    }

    /// Get the raw bytes for a single row.
    pub fn row_bytes(&self, row: usize) -> &[u8] {
        let elems = self.row_elements();
        let blocks_per_row = elems / self.block_size();
        let row_byte_len = blocks_per_row * self.block_bytes();
        let offset = self.row_byte_offset(row);
        &self.raw[offset..offset + row_byte_len]
    }
}

/// GGML Q4_0 block size: 32 values per block
pub const Q4_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q4_0 block: 2 (f16 scale) + 16 (nibbles) = 18
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// GGML Q4_1 block size: 32 values per block
pub const Q4_1_BLOCK_SIZE: usize = 32;
/// Bytes per Q4_1 block: 2 (f16 scale) + 2 (f16 min) + 16 (nibbles) = 20
pub const Q4_1_BLOCK_BYTES: usize = 20;

/// GGML Q5_0 block size: 32 values per block
pub const Q5_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q5_0 block: 2 (f16 scale) + 4 (high-bit mask u32) + 16 (nibbles) = 22
pub const Q5_0_BLOCK_BYTES: usize = 22;

/// GGML Q5_1 block size: 32 values per block
pub const Q5_1_BLOCK_SIZE: usize = 32;
/// Bytes per Q5_1 block: 2 (f16 scale) + 2 (f16 min) + 4 (high-bit mask u32) + 16 (nibbles) = 24
pub const Q5_1_BLOCK_BYTES: usize = 24;

/// GGML Q8_0 block size: 32 values per block
pub const Q8_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q8_0 block: 2 (f16 scale) + 32 (i8 values) = 34
pub const Q8_0_BLOCK_BYTES: usize = 34;
