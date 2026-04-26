//! Core quantization types, block-size constants, and the QuantizedTensor struct.

/// Type of quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// GGML Q4_0: 4-bit quantized, 32 values per block.
    Q4_0,
    /// GGML Q5_0: 5-bit quantized, 32 values per block.
    Q5_0,
    /// GGML Q8_0: 8-bit quantized, 32 values per block.
    Q8_0,
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
        match self.qtype {
            QuantType::Q4_0 => Q4_0_BLOCK_SIZE,
            QuantType::Q5_0 => Q5_0_BLOCK_SIZE,
            QuantType::Q8_0 => Q8_0_BLOCK_SIZE,
        }
    }

    /// Bytes per block for this quantization type.
    pub fn block_bytes(&self) -> usize {
        match self.qtype {
            QuantType::Q4_0 => Q4_0_BLOCK_BYTES,
            QuantType::Q5_0 => Q5_0_BLOCK_BYTES,
            QuantType::Q8_0 => Q8_0_BLOCK_BYTES,
        }
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

/// GGML Q5_0 block size: 32 values per block
pub const Q5_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q5_0 block: 2 (f16 scale) + 4 (high-bit mask u32) + 16 (nibbles) = 22
pub const Q5_0_BLOCK_BYTES: usize = 22;

/// GGML Q8_0 block size: 32 values per block
pub const Q8_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q8_0 block: 2 (f16 scale) + 32 (i8 values) = 34
pub const Q8_0_BLOCK_BYTES: usize = 34;
