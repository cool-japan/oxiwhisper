//! GGML quantization support: Q4_0, Q5_0, and Q8_0 block quantization and dequantization.

use half::f16;

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

/// Dequantize a single Q4_0 block (18 bytes) into 32 f32 values.
///
/// Format: [f16 scale (2 bytes)] [16 bytes of nibbles]
/// Each byte contains two 4-bit unsigned integers.
/// Values are offset: float_val = (nibble - 8) * scale
pub fn dequantize_q4_0_block(block: &[u8], output: &mut [f32]) {
    debug_assert!(block.len() >= Q4_0_BLOCK_BYTES);
    debug_assert!(output.len() >= Q4_0_BLOCK_SIZE);

    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

    for i in 0..16 {
        let byte = block[2 + i];
        let lo = (byte & 0x0F) as i32 - 8;
        let hi = ((byte >> 4) & 0x0F) as i32 - 8;
        output[i] = lo as f32 * scale;
        output[i + 16] = hi as f32 * scale;
    }
}

/// Dequantize a single Q5_0 block (22 bytes) into 32 f32 values.
///
/// Format: [f16 scale (2 bytes)] [u32 high-bit mask (4 bytes)] [16 bytes of low nibbles]
/// Each value is a 5-bit unsigned integer (0..31) centered at 16:
/// `value = ((low_nibble | (high_bit << 4)) - 16) * scale`
pub fn dequantize_q5_0_block(block: &[u8], output: &mut [f32]) {
    debug_assert!(block.len() >= Q5_0_BLOCK_BYTES);
    debug_assert!(output.len() >= Q5_0_BLOCK_SIZE);

    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

    for i in 0..16 {
        let byte = block[6 + i];
        let lo_nibble = (byte & 0x0F) as i32;
        let hi_nibble = ((byte >> 4) & 0x0F) as i32;

        // Low nibble -> element i, high nibble -> element i+16
        let hi_bit_lo = ((qh >> i) & 1) as i32;
        let hi_bit_hi = ((qh >> (i + 16)) & 1) as i32;

        output[i] = ((lo_nibble | (hi_bit_lo << 4)) - 16) as f32 * scale;
        output[i + 16] = ((hi_nibble | (hi_bit_hi << 4)) - 16) as f32 * scale;
    }
}

/// Dequantize a single Q8_0 block (34 bytes) into 32 f32 values.
///
/// Format: [f16 scale (2 bytes)] [32 i8 values]
/// float_val = i8_val * scale
pub fn dequantize_q8_0_block(block: &[u8], output: &mut [f32]) {
    debug_assert!(block.len() >= Q8_0_BLOCK_BYTES);
    debug_assert!(output.len() >= Q8_0_BLOCK_SIZE);

    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

    for i in 0..32 {
        output[i] = (block[2 + i] as i8) as f32 * scale;
    }
}

/// Dequantize an entire Q4_0 tensor into f32.
pub fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / Q4_0_BLOCK_SIZE;
    let mut output = vec![0.0f32; n_elements];
    let mut block_buf = [0.0f32; Q4_0_BLOCK_SIZE];

    for b in 0..n_blocks {
        let block_start = b * Q4_0_BLOCK_BYTES;
        dequantize_q4_0_block(&data[block_start..], &mut block_buf);
        output[b * Q4_0_BLOCK_SIZE..(b + 1) * Q4_0_BLOCK_SIZE].copy_from_slice(&block_buf);
    }
    output
}

/// Dequantize an entire Q5_0 tensor into f32.
pub fn dequantize_q5_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / Q5_0_BLOCK_SIZE;
    let mut output = vec![0.0f32; n_elements];
    let mut block_buf = [0.0f32; Q5_0_BLOCK_SIZE];

    for b in 0..n_blocks {
        let block_start = b * Q5_0_BLOCK_BYTES;
        dequantize_q5_0_block(&data[block_start..], &mut block_buf);
        output[b * Q5_0_BLOCK_SIZE..(b + 1) * Q5_0_BLOCK_SIZE].copy_from_slice(&block_buf);
    }
    output
}

/// Dequantize an entire Q8_0 tensor into f32.
pub fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / Q8_0_BLOCK_SIZE;
    let mut output = vec![0.0f32; n_elements];
    let mut block_buf = [0.0f32; Q8_0_BLOCK_SIZE];

    for b in 0..n_blocks {
        let block_start = b * Q8_0_BLOCK_BYTES;
        dequantize_q8_0_block(&data[block_start..], &mut block_buf);
        output[b * Q8_0_BLOCK_SIZE..(b + 1) * Q8_0_BLOCK_SIZE].copy_from_slice(&block_buf);
    }
    output
}

// ---------------------------------------------------------------------------
// Quantization: f32 -> Q8_0 / Q4_0
// ---------------------------------------------------------------------------

/// Statistics returned by batch quantization helpers.
#[derive(Debug, Clone)]
pub struct QuantizeStats {
    /// Number of tensors that were quantized.
    pub tensors_quantized: usize,
    /// Number of tensors kept as f32.
    pub tensors_kept_f32: usize,
    /// Total size of original f32 data in bytes.
    pub original_size_bytes: u64,
    /// Total size of quantized data in bytes.
    pub quantized_size_bytes: u64,
}

/// Quantize a single block of 32 f32 values into Q8_0 format (34 bytes).
///
/// Layout: `[f16 scale LE (2 bytes)] [32 × i8 quantized values]`
///
/// # Panics
///
/// Debug-asserts that `input.len() >= 32` and `output.len() >= 34`.
pub fn quantize_block_q8_0(input: &[f32], output: &mut [u8]) {
    debug_assert!(input.len() >= Q8_0_BLOCK_SIZE);
    debug_assert!(output.len() >= Q8_0_BLOCK_BYTES);

    // Find absolute max
    let mut amax: f32 = 0.0;
    for &v in &input[..Q8_0_BLOCK_SIZE] {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }

    let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

    // Store scale as f16 little-endian
    let scale_f16 = f16::from_f32(scale);
    let le = scale_f16.to_le_bytes();
    output[0] = le[0];
    output[1] = le[1];

    // Quantize each value: round(value / scale), clamp to [-128, 127]
    for i in 0..Q8_0_BLOCK_SIZE {
        let q = (input[i] * inv_scale).round();
        let q_clamped = q.clamp(-128.0, 127.0) as i8;
        output[2 + i] = q_clamped as u8;
    }
}

/// Quantize a single block of 32 f32 values into Q4_0 format (18 bytes).
///
/// Layout: `[f16 scale LE (2 bytes)] [16 bytes of nibble pairs]`
///
/// The nibble packing matches the dequantization layout:
/// - `byte[i] = nibble_for_element[i] | (nibble_for_element[i+16] << 4)`
/// - where `nibble = clamp(round(value / scale) + 8, 0, 15)`
///
/// # Panics
///
/// Debug-asserts that `input.len() >= 32` and `output.len() >= 18`.
pub fn quantize_block_q4_0(input: &[f32], output: &mut [u8]) {
    debug_assert!(input.len() >= Q4_0_BLOCK_SIZE);
    debug_assert!(output.len() >= Q4_0_BLOCK_BYTES);

    // Find absolute max
    let mut amax: f32 = 0.0;
    for &v in &input[..Q4_0_BLOCK_SIZE] {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }

    // Scale so that the range [-8, 7] maps to [-amax, amax*(7/8)].
    // Using amax/8.0 ensures negative extreme maps to nibble 0 (-8*scale = -amax).
    let scale = if amax == 0.0 { 0.0 } else { amax / 8.0 };
    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

    // Store scale as f16 little-endian
    let scale_f16 = f16::from_f32(scale);
    let le = scale_f16.to_le_bytes();
    output[0] = le[0];
    output[1] = le[1];

    // Quantize: nibble = clamp(round(value / scale) + 8, 0, 15)
    // Pack: byte[i] = nibble[i] | (nibble[i+16] << 4)
    for i in 0..16 {
        let q_lo = (input[i] * inv_scale).round() + 8.0;
        let lo = q_lo.clamp(0.0, 15.0) as u8;

        let q_hi = (input[i + 16] * inv_scale).round() + 8.0;
        let hi = q_hi.clamp(0.0, 15.0) as u8;

        output[2 + i] = lo | (hi << 4);
    }
}

/// Quantize a single block of 32 f32 values into Q5_0 format (22 bytes).
///
/// Layout: `[f16 scale LE (2 bytes)] [u32 high-bit mask (4 bytes)] [16 bytes of nibble pairs]`
///
/// The 5-bit quantized value is `q = clamp(round(value / scale) + 16, 0, 31)`.
/// The low 4 bits go into nibble bytes (same packing as Q4_0), and bit 4 goes
/// into the high-bit mask.
///
/// # Panics
///
/// Debug-asserts that `input.len() >= 32` and `output.len() >= 22`.
pub fn quantize_block_q5_0(input: &[f32], output: &mut [u8]) {
    debug_assert!(input.len() >= Q5_0_BLOCK_SIZE);
    debug_assert!(output.len() >= Q5_0_BLOCK_BYTES);

    // Find absolute max
    let mut amax: f32 = 0.0;
    for &v in &input[..Q5_0_BLOCK_SIZE] {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }

    // Scale so that the range [0, 31] centered at 16 maps to [-amax, amax*(15/16)].
    // Using amax/15.0 ensures max positive maps to nibble 31 (15*scale = amax).
    let scale = if amax == 0.0 { 0.0 } else { amax / 15.0 };
    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

    // Store scale as f16 little-endian
    let scale_f16 = f16::from_f32(scale);
    let le = scale_f16.to_le_bytes();
    output[0] = le[0];
    output[1] = le[1];

    // Quantize: q = clamp(round(value / scale) + 16, 0, 31)
    // Pack: low 4 bits into nibble bytes, bit 4 into high-bit mask
    let mut qh: u32 = 0;
    for i in 0..16 {
        let q_lo = (input[i] * inv_scale).round() + 16.0;
        let lo_q = q_lo.clamp(0.0, 31.0) as u32;

        let q_hi = (input[i + 16] * inv_scale).round() + 16.0;
        let hi_q = q_hi.clamp(0.0, 31.0) as u32;

        // Low 4 bits go into nibble bytes
        let lo_nibble = (lo_q & 0x0F) as u8;
        let hi_nibble = (hi_q & 0x0F) as u8;
        output[6 + i] = lo_nibble | (hi_nibble << 4);

        // Bit 4 goes into high-bit mask
        qh |= ((lo_q >> 4) & 1) << i;
        qh |= ((hi_q >> 4) & 1) << (i + 16);
    }

    // Store high-bit mask as u32 little-endian
    let qh_bytes = qh.to_le_bytes();
    output[2] = qh_bytes[0];
    output[3] = qh_bytes[1];
    output[4] = qh_bytes[2];
    output[5] = qh_bytes[3];
}

/// Quantize an entire f32 slice into Q5_0 format.
///
/// `data.len()` must be divisible by 32.
///
/// # Errors
///
/// Returns `Err` if the input length is not a multiple of `Q5_0_BLOCK_SIZE`.
pub fn quantize_to_q5_0(data: &[f32]) -> Result<Vec<u8>, String> {
    if !data.len().is_multiple_of(Q5_0_BLOCK_SIZE) {
        return Err(format!(
            "quantize_to_q5_0: input length {} is not a multiple of {}",
            data.len(),
            Q5_0_BLOCK_SIZE
        ));
    }
    let n_blocks = data.len() / Q5_0_BLOCK_SIZE;
    let mut output = vec![0u8; n_blocks * Q5_0_BLOCK_BYTES];

    for b in 0..n_blocks {
        let in_start = b * Q5_0_BLOCK_SIZE;
        let out_start = b * Q5_0_BLOCK_BYTES;
        quantize_block_q5_0(
            &data[in_start..in_start + Q5_0_BLOCK_SIZE],
            &mut output[out_start..out_start + Q5_0_BLOCK_BYTES],
        );
    }
    Ok(output)
}

/// Quantize an entire f32 slice into Q8_0 format.
///
/// `data.len()` must be divisible by 32.
///
/// # Errors
///
/// Returns `Err` if the input length is not a multiple of `Q8_0_BLOCK_SIZE`.
pub fn quantize_to_q8_0(data: &[f32]) -> Result<Vec<u8>, String> {
    if !data.len().is_multiple_of(Q8_0_BLOCK_SIZE) {
        return Err(format!(
            "quantize_to_q8_0: input length {} is not a multiple of {}",
            data.len(),
            Q8_0_BLOCK_SIZE
        ));
    }
    let n_blocks = data.len() / Q8_0_BLOCK_SIZE;
    let mut output = vec![0u8; n_blocks * Q8_0_BLOCK_BYTES];

    for b in 0..n_blocks {
        let in_start = b * Q8_0_BLOCK_SIZE;
        let out_start = b * Q8_0_BLOCK_BYTES;
        quantize_block_q8_0(
            &data[in_start..in_start + Q8_0_BLOCK_SIZE],
            &mut output[out_start..out_start + Q8_0_BLOCK_BYTES],
        );
    }
    Ok(output)
}

/// Quantize an entire f32 slice into Q4_0 format.
///
/// `data.len()` must be divisible by 32.
///
/// # Errors
///
/// Returns `Err` if the input length is not a multiple of `Q4_0_BLOCK_SIZE`.
pub fn quantize_to_q4_0(data: &[f32]) -> Result<Vec<u8>, String> {
    if !data.len().is_multiple_of(Q4_0_BLOCK_SIZE) {
        return Err(format!(
            "quantize_to_q4_0: input length {} is not a multiple of {}",
            data.len(),
            Q4_0_BLOCK_SIZE
        ));
    }
    let n_blocks = data.len() / Q4_0_BLOCK_SIZE;
    let mut output = vec![0u8; n_blocks * Q4_0_BLOCK_BYTES];

    for b in 0..n_blocks {
        let in_start = b * Q4_0_BLOCK_SIZE;
        let out_start = b * Q4_0_BLOCK_BYTES;
        quantize_block_q4_0(
            &data[in_start..in_start + Q4_0_BLOCK_SIZE],
            &mut output[out_start..out_start + Q4_0_BLOCK_BYTES],
        );
    }
    Ok(output)
}

/// Quantize an f32 tensor and wrap it into a [`QuantizedTensor`].
///
/// # Errors
///
/// Returns `Err` if the total number of elements (product of `shape`) does not
/// match `data.len()`, or if the length is not block-aligned.
pub fn quantize_tensor(
    data: &[f32],
    shape: &[usize],
    qtype: QuantType,
) -> Result<QuantizedTensor, String> {
    let numel: usize = shape.iter().product();
    if numel != data.len() {
        return Err(format!(
            "quantize_tensor: shape product {} != data length {}",
            numel,
            data.len()
        ));
    }

    let raw = match qtype {
        QuantType::Q4_0 => quantize_to_q4_0(data)?,
        QuantType::Q5_0 => quantize_to_q5_0(data)?,
        QuantType::Q8_0 => quantize_to_q8_0(data)?,
    };

    Ok(QuantizedTensor {
        raw,
        shape: shape.to_vec(),
        qtype,
    })
}

/// Compute dot product between an f32 vector and a Q4_0 quantized vector.
/// This avoids full dequantization for GEMV efficiency.
pub fn dot_q4_0(input: &[f32], quantized: &[u8], n: usize) -> f32 {
    let n_blocks = n / Q4_0_BLOCK_SIZE;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block_start = b * Q4_0_BLOCK_BYTES;
        let scale =
            f16::from_le_bytes([quantized[block_start], quantized[block_start + 1]]).to_f32();
        let input_slice = &input[b * Q4_0_BLOCK_SIZE..];

        let mut block_sum = 0.0f32;
        for i in 0..16 {
            let byte = quantized[block_start + 2 + i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            block_sum += input_slice[i] * lo as f32;
            block_sum += input_slice[i + 16] * hi as f32;
        }
        sum += block_sum * scale;
    }
    sum
}

/// Compute dot product between an f32 vector and a Q5_0 quantized vector.
/// This avoids full dequantization for GEMV efficiency.
pub fn dot_q5_0(input: &[f32], quantized: &[u8], n: usize) -> f32 {
    let n_blocks = n / Q5_0_BLOCK_SIZE;
    let mut total = 0.0f32;

    for blk in 0..n_blocks {
        let a_off = blk * Q5_0_BLOCK_BYTES;
        let b_off = blk * Q5_0_BLOCK_SIZE;
        let scale = f16::from_le_bytes([quantized[a_off], quantized[a_off + 1]]).to_f32();
        let qh = u32::from_le_bytes([
            quantized[a_off + 2],
            quantized[a_off + 3],
            quantized[a_off + 4],
            quantized[a_off + 5],
        ]);

        let mut sum = 0.0f32;
        for i in 0..16 {
            let byte = quantized[a_off + 6 + i];
            let lo = (byte & 0x0F) as i32;
            let hi = ((byte >> 4) & 0x0F) as i32;
            let hi_bit_lo = ((qh >> i) & 1) as i32;
            let hi_bit_hi = ((qh >> (i + 16)) & 1) as i32;

            let v_lo = ((lo | (hi_bit_lo << 4)) - 16) as f32;
            let v_hi = ((hi | (hi_bit_hi << 4)) - 16) as f32;
            sum += v_lo * input[b_off + i] + v_hi * input[b_off + i + 16];
        }
        total += scale * sum;
    }
    total
}

/// Compute dot product between an f32 vector and a Q8_0 quantized vector.
pub fn dot_q8_0(input: &[f32], quantized: &[u8], n: usize) -> f32 {
    let n_blocks = n / Q8_0_BLOCK_SIZE;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block_start = b * Q8_0_BLOCK_BYTES;
        let scale =
            f16::from_le_bytes([quantized[block_start], quantized[block_start + 1]]).to_f32();
        let input_slice = &input[b * Q8_0_BLOCK_SIZE..];

        let mut block_sum = 0.0f32;
        for i in 0..32 {
            block_sum += input_slice[i] * (quantized[block_start + 2 + i] as i8) as f32;
        }
        sum += block_sum * scale;
    }
    sum
}

// ---------------------------------------------------------------------------
// SIMD-accelerated quantized dot products
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod simd_q_x86 {
    use super::*;

    /// AVX2 + FMA accelerated Q4_0 dot product.
    ///
    /// Processes each 32-element block by extracting low and high nibbles from
    /// the 16 packed bytes, sign-extending via `_mm256_cvtepi8_epi32`, subtracting
    /// the offset of 8, then FMA-accumulating against the f32 input vector.
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available on the current CPU.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_q4_0_avx2(input: &[f32], quantized: &[u8], n: usize) -> f32 {
        use std::arch::x86_64::*;

        unsafe {
            let blocks = n / Q4_0_BLOCK_SIZE;
            let offset_8 = _mm256_set1_epi32(8);
            let mask_0f = _mm_set1_epi8(0x0F_u8 as i8);
            let mut total = _mm256_setzero_ps();

            for blk in 0..blocks {
                let a_off = blk * Q4_0_BLOCK_BYTES;
                let scale = f16::from_le_bytes([quantized[a_off], quantized[a_off + 1]]).to_f32();
                let b_off = blk * Q4_0_BLOCK_SIZE;
                let scale_v = _mm256_set1_ps(scale);

                // Load 16 bytes of nibble data
                let raw = _mm_loadu_si128(quantized[a_off + 2..].as_ptr() as *const __m128i);

                // Extract low nibbles (elements 0..15) and high nibbles (elements 16..31)
                let lo_nibbles = _mm_and_si128(raw, mask_0f);
                let hi_nibbles = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);

                // Process low nibbles (elements 0..15): 2 groups of 8
                for chunk in 0..2 {
                    let nibble_chunk = if chunk == 0 {
                        lo_nibbles
                    } else {
                        _mm_srli_si128(lo_nibbles, 8)
                    };
                    let i32_vals = _mm256_cvtepu8_epi32(nibble_chunk);
                    let centered = _mm256_sub_epi32(i32_vals, offset_8);
                    let f32_vals = _mm256_cvtepi32_ps(centered);
                    let scaled = _mm256_mul_ps(f32_vals, scale_v);
                    let b_vals = _mm256_loadu_ps(input.as_ptr().add(b_off + chunk * 8));
                    total = _mm256_fmadd_ps(scaled, b_vals, total);
                }

                // Process high nibbles (elements 16..31): 2 groups of 8
                for chunk in 0..2 {
                    let nibble_chunk = if chunk == 0 {
                        hi_nibbles
                    } else {
                        _mm_srli_si128(hi_nibbles, 8)
                    };
                    let i32_vals = _mm256_cvtepu8_epi32(nibble_chunk);
                    let centered = _mm256_sub_epi32(i32_vals, offset_8);
                    let f32_vals = _mm256_cvtepi32_ps(centered);
                    let scaled = _mm256_mul_ps(f32_vals, scale_v);
                    let b_vals = _mm256_loadu_ps(input.as_ptr().add(b_off + 16 + chunk * 8));
                    total = _mm256_fmadd_ps(scaled, b_vals, total);
                }
            }

            // Horizontal sum of 8 f32 lanes
            let hi = _mm256_extractf128_ps(total, 1);
            let lo = _mm256_castps256_ps128(total);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
        }
    }

    /// AVX2 + FMA accelerated Q5_0 dot product.
    ///
    /// Similar to Q4_0 but additionally extracts the 5th bit from a 32-bit
    /// high-bit mask and ORs it with the low nibble before centering at 16.
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available on the current CPU.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_q5_0_avx2(input: &[f32], quantized: &[u8], n: usize) -> f32 {
        use std::arch::x86_64::*;

        unsafe {
            let blocks = n / Q5_0_BLOCK_SIZE;
            let offset_16 = _mm256_set1_epi32(16);
            let mask_0f = _mm_set1_epi8(0x0F_u8 as i8);
            let mut total = _mm256_setzero_ps();

            for blk in 0..blocks {
                let a_off = blk * Q5_0_BLOCK_BYTES;
                let scale = f16::from_le_bytes([quantized[a_off], quantized[a_off + 1]]).to_f32();
                let qh = u32::from_le_bytes([
                    quantized[a_off + 2],
                    quantized[a_off + 3],
                    quantized[a_off + 4],
                    quantized[a_off + 5],
                ]);
                let b_off = blk * Q5_0_BLOCK_SIZE;
                let scale_v = _mm256_set1_ps(scale);

                // Load 16 bytes of nibble data
                let raw = _mm_loadu_si128(quantized[a_off + 6..].as_ptr() as *const __m128i);

                // Extract low nibbles (elements 0..15) and high nibbles (elements 16..31)
                let lo_nibbles = _mm_and_si128(raw, mask_0f);
                let hi_nibbles = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);

                // Process low nibbles (elements 0..15): 2 groups of 8
                for chunk in 0..2 {
                    let nibble_chunk = if chunk == 0 {
                        lo_nibbles
                    } else {
                        _mm_srli_si128(lo_nibbles, 8)
                    };
                    let i32_vals = _mm256_cvtepu8_epi32(nibble_chunk);

                    // Extract high bits for this chunk of 8 elements
                    let base_bit = chunk * 8;
                    let hi_bits = _mm256_set_epi32(
                        ((qh >> (base_bit + 7)) & 1) as i32,
                        ((qh >> (base_bit + 6)) & 1) as i32,
                        ((qh >> (base_bit + 5)) & 1) as i32,
                        ((qh >> (base_bit + 4)) & 1) as i32,
                        ((qh >> (base_bit + 3)) & 1) as i32,
                        ((qh >> (base_bit + 2)) & 1) as i32,
                        ((qh >> (base_bit + 1)) & 1) as i32,
                        ((qh >> base_bit) & 1) as i32,
                    );
                    let hi_shifted = _mm256_slli_epi32(hi_bits, 4);
                    let combined = _mm256_or_si256(i32_vals, hi_shifted);
                    let centered = _mm256_sub_epi32(combined, offset_16);
                    let f32_vals = _mm256_cvtepi32_ps(centered);
                    let scaled = _mm256_mul_ps(f32_vals, scale_v);
                    let b_vals = _mm256_loadu_ps(input.as_ptr().add(b_off + chunk * 8));
                    total = _mm256_fmadd_ps(scaled, b_vals, total);
                }

                // Process high nibbles (elements 16..31): 2 groups of 8
                for chunk in 0..2 {
                    let nibble_chunk = if chunk == 0 {
                        hi_nibbles
                    } else {
                        _mm_srli_si128(hi_nibbles, 8)
                    };
                    let i32_vals = _mm256_cvtepu8_epi32(nibble_chunk);

                    // Extract high bits for this chunk of 8 elements (bits 16..31 of qh)
                    let base_bit = 16 + chunk * 8;
                    let hi_bits = _mm256_set_epi32(
                        ((qh >> (base_bit + 7)) & 1) as i32,
                        ((qh >> (base_bit + 6)) & 1) as i32,
                        ((qh >> (base_bit + 5)) & 1) as i32,
                        ((qh >> (base_bit + 4)) & 1) as i32,
                        ((qh >> (base_bit + 3)) & 1) as i32,
                        ((qh >> (base_bit + 2)) & 1) as i32,
                        ((qh >> (base_bit + 1)) & 1) as i32,
                        ((qh >> base_bit) & 1) as i32,
                    );
                    let hi_shifted = _mm256_slli_epi32(hi_bits, 4);
                    let combined = _mm256_or_si256(i32_vals, hi_shifted);
                    let centered = _mm256_sub_epi32(combined, offset_16);
                    let f32_vals = _mm256_cvtepi32_ps(centered);
                    let scaled = _mm256_mul_ps(f32_vals, scale_v);
                    let b_vals = _mm256_loadu_ps(input.as_ptr().add(b_off + 16 + chunk * 8));
                    total = _mm256_fmadd_ps(scaled, b_vals, total);
                }
            }

            // Horizontal sum of 8 f32 lanes
            let hi = _mm256_extractf128_ps(total, 1);
            let lo = _mm256_castps256_ps128(total);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
        }
    }

    /// AVX2 + FMA accelerated Q8_0 dot product.
    ///
    /// Processes each 32-element block by loading 32 i8 quantized values,
    /// converting to f32 in chunks of 8 via `_mm256_cvtepi8_epi32`, then
    /// FMA-accumulating against the corresponding f32 input values.
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available on the current CPU.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_q8_0_avx2(input: &[f32], quantized: &[u8], n: usize) -> f32 {
        use std::arch::x86_64::*;

        unsafe {
            let blocks = n / Q8_0_BLOCK_SIZE;
            let mut total = 0.0f32;

            for blk in 0..blocks {
                let q_off = blk * Q8_0_BLOCK_BYTES;
                let scale = f16::from_le_bytes([quantized[q_off], quantized[q_off + 1]]).to_f32();
                let b_off = blk * Q8_0_BLOCK_SIZE;

                // Load 32 i8 quantized values as a single 256-bit register
                let qi = _mm256_loadu_si256(quantized[q_off + 2..].as_ptr() as *const __m256i);

                let mut acc = _mm256_setzero_ps();

                // Bytes 0..7: extend i8 -> i32 -> f32, FMA with input
                let lo128 = _mm256_castsi256_si128(qi);
                let i32_0 = _mm256_cvtepi8_epi32(lo128);
                let f32_0 = _mm256_cvtepi32_ps(i32_0);
                let b_0 = _mm256_loadu_ps(input.as_ptr().add(b_off));
                acc = _mm256_fmadd_ps(f32_0, b_0, acc);

                // Bytes 8..15
                let shifted_8 = _mm_srli_si128(lo128, 8);
                let i32_1 = _mm256_cvtepi8_epi32(shifted_8);
                let f32_1 = _mm256_cvtepi32_ps(i32_1);
                let b_1 = _mm256_loadu_ps(input.as_ptr().add(b_off + 8));
                acc = _mm256_fmadd_ps(f32_1, b_1, acc);

                // Bytes 16..23
                let hi128 = _mm256_extracti128_si256(qi, 1);
                let i32_2 = _mm256_cvtepi8_epi32(hi128);
                let f32_2 = _mm256_cvtepi32_ps(i32_2);
                let b_2 = _mm256_loadu_ps(input.as_ptr().add(b_off + 16));
                acc = _mm256_fmadd_ps(f32_2, b_2, acc);

                // Bytes 24..31
                let shifted_24 = _mm_srli_si128(hi128, 8);
                let i32_3 = _mm256_cvtepi8_epi32(shifted_24);
                let f32_3 = _mm256_cvtepi32_ps(i32_3);
                let b_3 = _mm256_loadu_ps(input.as_ptr().add(b_off + 24));
                acc = _mm256_fmadd_ps(f32_3, b_3, acc);

                // Horizontal sum of 8 f32 lanes
                let hi_lane = _mm256_extractf128_ps(acc, 1);
                let lo_lane = _mm256_castps256_ps128(acc);
                let sum128 = _mm_add_ps(lo_lane, hi_lane);
                let shuf = _mm_movehdup_ps(sum128);
                let sums = _mm_add_ps(sum128, shuf);
                let shuf2 = _mm_movehl_ps(sums, sums);
                let s = _mm_add_ss(sums, shuf2);
                total += scale * _mm_cvtss_f32(s);
            }

            total
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod simd_q_neon {
    use super::*;

    /// NEON-accelerated Q4_0 dot product.
    ///
    /// Extracts nibbles to a temporary f32 array per block, then uses NEON
    /// `vfmaq_f32` for the accumulation against the f32 input vector.
    ///
    /// NEON is always available on aarch64 so no runtime detection is needed.
    pub fn dot_q4_0_neon(input: &[f32], quantized: &[u8], n: usize) -> f32 {
        use std::arch::aarch64::*;

        let blocks = n / Q4_0_BLOCK_SIZE;
        let mut total = 0.0f32;

        for blk in 0..blocks {
            let a_off = blk * Q4_0_BLOCK_BYTES;
            let scale = f16::from_le_bytes([quantized[a_off], quantized[a_off + 1]]).to_f32();
            let b_off = blk * Q4_0_BLOCK_SIZE;

            // Extract all 32 dequantized values into a temp array
            let mut vals = [0.0f32; 32];
            for i in 0..16 {
                let byte = quantized[a_off + 2 + i];
                vals[i] = ((byte & 0x0F) as i32 - 8) as f32 * scale;
                vals[i + 16] = (((byte >> 4) & 0x0F) as i32 - 8) as f32 * scale;
            }

            unsafe {
                let mut acc = vdupq_n_f32(0.0);
                // Process 32 values in groups of 4 (8 iterations)
                for chunk in 0..8 {
                    let va = vld1q_f32(vals.as_ptr().add(chunk * 4));
                    let vb = vld1q_f32(input.as_ptr().add(b_off + chunk * 4));
                    acc = vfmaq_f32(acc, va, vb);
                }
                total += vaddvq_f32(acc);
            }
        }
        total
    }

    /// NEON-accelerated Q5_0 dot product.
    ///
    /// Extracts nibbles and high-bit mask to a temporary f32 array per block,
    /// then uses NEON `vfmaq_f32` for the accumulation.
    ///
    /// NEON is always available on aarch64 so no runtime detection is needed.
    pub fn dot_q5_0_neon(input: &[f32], quantized: &[u8], n: usize) -> f32 {
        use std::arch::aarch64::*;

        let blocks = n / Q5_0_BLOCK_SIZE;
        let mut total = 0.0f32;

        for blk in 0..blocks {
            let a_off = blk * Q5_0_BLOCK_BYTES;
            let scale = f16::from_le_bytes([quantized[a_off], quantized[a_off + 1]]).to_f32();
            let qh = u32::from_le_bytes([
                quantized[a_off + 2],
                quantized[a_off + 3],
                quantized[a_off + 4],
                quantized[a_off + 5],
            ]);
            let b_off = blk * Q5_0_BLOCK_SIZE;

            // Extract all 32 dequantized values into a temp array
            let mut vals = [0.0f32; 32];
            for i in 0..16 {
                let byte = quantized[a_off + 6 + i];
                let lo = (byte & 0x0F) as i32;
                let hi = ((byte >> 4) & 0x0F) as i32;
                let hi_bit_lo = ((qh >> i) & 1) as i32;
                let hi_bit_hi = ((qh >> (i + 16)) & 1) as i32;
                vals[i] = ((lo | (hi_bit_lo << 4)) - 16) as f32 * scale;
                vals[i + 16] = ((hi | (hi_bit_hi << 4)) - 16) as f32 * scale;
            }

            unsafe {
                let mut acc = vdupq_n_f32(0.0);
                // Process 32 values in groups of 4 (8 iterations)
                for chunk in 0..8 {
                    let va = vld1q_f32(vals.as_ptr().add(chunk * 4));
                    let vb = vld1q_f32(input.as_ptr().add(b_off + chunk * 4));
                    acc = vfmaq_f32(acc, va, vb);
                }
                total += vaddvq_f32(acc);
            }
        }
        total
    }

    /// NEON-accelerated Q8_0 dot product.
    ///
    /// Processes each 32-element block by loading 4 i8 values at a time,
    /// converting to f32, and accumulating via `vfmaq_f32`.
    ///
    /// NEON is always available on aarch64 so no runtime detection is needed.
    pub fn dot_q8_0_neon(input: &[f32], quantized: &[u8], n: usize) -> f32 {
        use std::arch::aarch64::*;

        let blocks = n / Q8_0_BLOCK_SIZE;
        let mut total = 0.0f32;

        for blk in 0..blocks {
            let q_off = blk * Q8_0_BLOCK_BYTES;
            let scale = f16::from_le_bytes([quantized[q_off], quantized[q_off + 1]]).to_f32();
            let b_off = blk * Q8_0_BLOCK_SIZE;

            unsafe {
                let mut acc = vdupq_n_f32(0.0);
                // Process 4 values at a time (8 iterations for 32 values)
                for chunk in 0..8 {
                    let base = chunk * 4;
                    let q0 = quantized[q_off + 2 + base] as i8 as f32;
                    let q1 = quantized[q_off + 2 + base + 1] as i8 as f32;
                    let q2 = quantized[q_off + 2 + base + 2] as i8 as f32;
                    let q3 = quantized[q_off + 2 + base + 3] as i8 as f32;
                    let qi = [q0, q1, q2, q3];
                    let vq = vld1q_f32(qi.as_ptr());
                    let vb = vld1q_f32(input.as_ptr().add(b_off + base));
                    acc = vfmaq_f32(acc, vq, vb);
                }
                total += scale * vaddvq_f32(acc);
            }
        }
        total
    }
}

/// SIMD-accelerated Q4_0 dot product when available, falling back to scalar.
///
/// On x86_64 with AVX2+FMA, uses 256-bit SIMD to process each 32-element block.
/// On aarch64, uses NEON intrinsics.
/// Otherwise falls back to the scalar `dot_q4_0`.
pub fn dot_q4_0_fast(input: &[f32], quantized: &[u8], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detection passed — AVX2 + FMA are available.
            return unsafe { simd_q_x86::dot_q4_0_avx2(input, quantized, n) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return simd_q_neon::dot_q4_0_neon(input, quantized, n);
    }
    #[allow(unreachable_code)]
    dot_q4_0(input, quantized, n)
}

/// SIMD-accelerated Q5_0 dot product when available, falling back to scalar.
///
/// On x86_64 with AVX2+FMA, uses 256-bit SIMD to process each 32-element block.
/// On aarch64, uses NEON intrinsics.
/// Otherwise falls back to the scalar `dot_q5_0`.
pub fn dot_q5_0_fast(input: &[f32], quantized: &[u8], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detection passed — AVX2 + FMA are available.
            return unsafe { simd_q_x86::dot_q5_0_avx2(input, quantized, n) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return simd_q_neon::dot_q5_0_neon(input, quantized, n);
    }
    #[allow(unreachable_code)]
    dot_q5_0(input, quantized, n)
}

/// SIMD-accelerated Q8_0 dot product when available, falling back to scalar.
///
/// On x86_64 with AVX2+FMA, uses 256-bit SIMD to process each 32-element block.
/// On aarch64, uses NEON intrinsics.
/// Otherwise falls back to the scalar `dot_q8_0`.
pub fn dot_q8_0_fast(input: &[f32], quantized: &[u8], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detection passed — AVX2 + FMA are available.
            return unsafe { simd_q_x86::dot_q8_0_avx2(input, quantized, n) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return simd_q_neon::dot_q8_0_neon(input, quantized, n);
    }
    #[allow(unreachable_code)]
    dot_q8_0(input, quantized, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a Q4_0 block from a scale and 32 integer values in [-8, 7].
    fn make_q4_0_block(scale: f32, values: &[i32; 32]) -> [u8; Q4_0_BLOCK_BYTES] {
        let mut block = [0u8; Q4_0_BLOCK_BYTES];
        let scale_f16 = f16::from_f32(scale);
        let le = scale_f16.to_le_bytes();
        block[0] = le[0];
        block[1] = le[1];
        // Q4_0 layout: output[i] = lo nibble for i in 0..16, output[i+16] = hi nibble
        for i in 0..16 {
            let lo = (values[i] + 8) as u8; // offset back to unsigned
            let hi = (values[i + 16] + 8) as u8;
            block[2 + i] = lo | (hi << 4);
        }
        block
    }

    /// Helper: build a Q8_0 block from a scale and 32 i8 values.
    fn make_q8_0_block(scale: f32, values: &[i8; 32]) -> [u8; Q8_0_BLOCK_BYTES] {
        let mut block = [0u8; Q8_0_BLOCK_BYTES];
        let scale_f16 = f16::from_f32(scale);
        let le = scale_f16.to_le_bytes();
        block[0] = le[0];
        block[1] = le[1];
        for i in 0..32 {
            block[2 + i] = values[i] as u8;
        }
        block
    }

    #[test]
    fn test_q4_0_roundtrip() {
        // Create known values: alternating pattern
        let mut values = [0i32; 32];
        for (i, val) in values.iter_mut().enumerate() {
            *val = (i as i32 % 8) - 4; // range -4..3
        }
        let scale = 2.0f32;
        let block = make_q4_0_block(scale, &values);

        let mut output = [0.0f32; Q4_0_BLOCK_SIZE];
        dequantize_q4_0_block(&block, &mut output);

        // f16 round-trip for scale: scale should be exact for 2.0
        let actual_scale = f16::from_f32(scale).to_f32();
        for i in 0..32 {
            let expected = values[i] as f32 * actual_scale;
            assert!(
                (output[i] - expected).abs() < 1e-4,
                "Q4_0 mismatch at {i}: expected {expected}, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn test_q8_0_roundtrip() {
        let mut values = [0i8; 32];
        for (i, val) in values.iter_mut().enumerate() {
            *val = (i as i8) - 16; // range -16..15
        }
        let scale = 0.5f32;
        let block = make_q8_0_block(scale, &values);

        let mut output = [0.0f32; Q8_0_BLOCK_SIZE];
        dequantize_q8_0_block(&block, &mut output);

        let actual_scale = f16::from_f32(scale).to_f32();
        for i in 0..32 {
            let expected = values[i] as f32 * actual_scale;
            assert!(
                (output[i] - expected).abs() < 1e-4,
                "Q8_0 mismatch at {i}: expected {expected}, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn test_dot_q4_0() {
        let mut values = [0i32; 32];
        for (i, val) in values.iter_mut().enumerate() {
            *val = (i as i32 % 7) - 3;
        }
        let scale = 1.5f32;
        let block = make_q4_0_block(scale, &values);

        // Dequantize to get reference f32 values
        let mut deq = [0.0f32; Q4_0_BLOCK_SIZE];
        dequantize_q4_0_block(&block, &mut deq);

        // Input vector
        let mut input = [0.0f32; 32];
        for (i, val) in input.iter_mut().enumerate() {
            *val = (i as f32 + 1.0) * 0.1;
        }

        let expected: f32 = input.iter().zip(deq.iter()).map(|(a, b)| a * b).sum();
        let actual = dot_q4_0(&input, &block, 32);

        assert!(
            (actual - expected).abs() < 1e-2,
            "dot_q4_0: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_dot_q8_0() {
        let mut values = [0i8; 32];
        for (i, val) in values.iter_mut().enumerate() {
            *val = (i as i8) - 16;
        }
        let scale = 0.25f32;
        let block = make_q8_0_block(scale, &values);

        let mut deq = [0.0f32; Q8_0_BLOCK_SIZE];
        dequantize_q8_0_block(&block, &mut deq);

        let mut input = [0.0f32; 32];
        for (i, val) in input.iter_mut().enumerate() {
            *val = (i as f32 + 1.0) * 0.1;
        }

        let expected: f32 = input.iter().zip(deq.iter()).map(|(a, b)| a * b).sum();
        let actual = dot_q8_0(&input, &block, 32);

        assert!(
            (actual - expected).abs() < 1e-2,
            "dot_q8_0: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_q4_0_zero_scale() {
        let values = [0i32; 32];
        let block = make_q4_0_block(0.0, &values);

        let mut output = [0.0f32; Q4_0_BLOCK_SIZE];
        dequantize_q4_0_block(&block, &mut output);

        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.abs() < 1e-10,
                "Q4_0 zero scale: expected ~0 at {i}, got {v}"
            );
        }
    }

    #[test]
    fn test_q8_0_symmetric() {
        // Verify that positive and negative values dequantize correctly
        let mut values = [0i8; 32];
        for i in 0..16 {
            values[i] = (i as i8) + 1; // positive: 1..16
            values[i + 16] = -((i as i8) + 1); // negative: -1..-16
        }
        let scale = 1.0f32;
        let block = make_q8_0_block(scale, &values);

        let mut output = [0.0f32; Q8_0_BLOCK_SIZE];
        dequantize_q8_0_block(&block, &mut output);

        let actual_scale = f16::from_f32(scale).to_f32();
        for i in 0..16 {
            let expected_pos = values[i] as f32 * actual_scale;
            let expected_neg = values[i + 16] as f32 * actual_scale;
            assert!(
                (output[i] - expected_pos).abs() < 1e-4,
                "Q8_0 positive mismatch at {i}: expected {expected_pos}, got {}",
                output[i]
            );
            assert!(
                (output[i + 16] - expected_neg).abs() < 1e-4,
                "Q8_0 negative mismatch at {}: expected {expected_neg}, got {}",
                i + 16,
                output[i + 16]
            );
            // Symmetry check: |pos| == |neg|
            assert!(
                (output[i].abs() - output[i + 16].abs()).abs() < 1e-4,
                "Q8_0 symmetry broken at {i}"
            );
        }
    }

    #[test]
    fn test_dequantize_q4_0_multi_block() {
        // Test the full-tensor dequantize with 2 blocks (64 elements)
        let mut values1 = [0i32; 32];
        let mut values2 = [0i32; 32];
        for i in 0..32 {
            values1[i] = (i as i32 % 5) - 2;
            values2[i] = -((i as i32 % 5) - 2);
        }
        let block1 = make_q4_0_block(1.0, &values1);
        let block2 = make_q4_0_block(2.0, &values2);

        let mut raw = Vec::with_capacity(Q4_0_BLOCK_BYTES * 2);
        raw.extend_from_slice(&block1);
        raw.extend_from_slice(&block2);

        let output = dequantize_q4_0(&raw, 64);
        assert_eq!(output.len(), 64);

        // Verify first block
        let mut expected1 = [0.0f32; 32];
        dequantize_q4_0_block(&block1, &mut expected1);
        for i in 0..32 {
            assert!(
                (output[i] - expected1[i]).abs() < 1e-4,
                "Multi-block Q4_0 mismatch at {i}"
            );
        }

        // Verify second block
        let mut expected2 = [0.0f32; 32];
        dequantize_q4_0_block(&block2, &mut expected2);
        for i in 0..32 {
            assert!(
                (output[32 + i] - expected2[i]).abs() < 1e-4,
                "Multi-block Q4_0 mismatch at {}",
                32 + i
            );
        }
    }

    #[test]
    fn test_dequantize_q8_0_multi_block() {
        let mut values1 = [0i8; 32];
        let mut values2 = [0i8; 32];
        for i in 0..32 {
            values1[i] = (i as i8) - 16;
            values2[i] = 16 - (i as i8);
        }
        let block1 = make_q8_0_block(0.5, &values1);
        let block2 = make_q8_0_block(1.5, &values2);

        let mut raw = Vec::with_capacity(Q8_0_BLOCK_BYTES * 2);
        raw.extend_from_slice(&block1);
        raw.extend_from_slice(&block2);

        let output = dequantize_q8_0(&raw, 64);
        assert_eq!(output.len(), 64);

        let mut expected1 = [0.0f32; 32];
        dequantize_q8_0_block(&block1, &mut expected1);
        for i in 0..32 {
            assert!(
                (output[i] - expected1[i]).abs() < 1e-4,
                "Multi-block Q8_0 mismatch at {i}"
            );
        }

        let mut expected2 = [0.0f32; 32];
        dequantize_q8_0_block(&block2, &mut expected2);
        for i in 0..32 {
            assert!(
                (output[32 + i] - expected2[i]).abs() < 1e-4,
                "Multi-block Q8_0 mismatch at {}",
                32 + i
            );
        }
    }

    #[test]
    fn test_quantized_tensor_creation() {
        let qt = QuantizedTensor {
            raw: vec![0u8; Q8_0_BLOCK_BYTES * 4], // 4 blocks = 128 elements
            shape: vec![4, 32],                   // 4 rows of 32 elements
            qtype: QuantType::Q8_0,
        };
        assert_eq!(qt.numel(), 128);
        assert_eq!(qt.block_size(), Q8_0_BLOCK_SIZE);
        assert_eq!(qt.block_bytes(), Q8_0_BLOCK_BYTES);
        assert_eq!(qt.qtype, QuantType::Q8_0);
    }

    #[test]
    fn test_quantized_tensor_q4_0_creation() {
        let qt = QuantizedTensor {
            raw: vec![0u8; Q4_0_BLOCK_BYTES * 4],
            shape: vec![4, 32],
            qtype: QuantType::Q4_0,
        };
        assert_eq!(qt.numel(), 128);
        assert_eq!(qt.block_size(), Q4_0_BLOCK_SIZE);
        assert_eq!(qt.block_bytes(), Q4_0_BLOCK_BYTES);
        assert_eq!(qt.qtype, QuantType::Q4_0);
    }

    #[test]
    fn test_quantized_tensor_row_bytes() {
        // Shape [in_f=64, out_f=3]: 3 physical rows, each with 64 elements (2 blocks)
        let blocks_per_row = 2; // 64 / 32 = 2
        let n_rows = 3;
        let total_blocks = n_rows * blocks_per_row;
        let mut raw = vec![0u8; total_blocks * Q8_0_BLOCK_BYTES];
        // Mark each block with a unique byte so we can verify row extraction
        for b in 0..total_blocks {
            raw[b * Q8_0_BLOCK_BYTES] = b as u8;
        }
        let qt = QuantizedTensor {
            raw,
            shape: vec![64, 3], // in_f=64, out_f=3
            qtype: QuantType::Q8_0,
        };

        assert_eq!(qt.row_elements(), 64);

        // Row 0 should start at block 0
        let row0 = qt.row_bytes(0);
        assert_eq!(row0.len(), blocks_per_row * Q8_0_BLOCK_BYTES);
        assert_eq!(row0[0], 0);

        // Row 1 should start at block 2
        let row1 = qt.row_bytes(1);
        assert_eq!(row1[0], 2);

        // Row 2 should start at block 4
        let row2 = qt.row_bytes(2);
        assert_eq!(row2[0], 4);
    }

    // -----------------------------------------------------------------------
    // Quantization (f32 -> Qx_0) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantize_q8_0_roundtrip() {
        // Create a block of known f32 values
        let mut input = [0.0f32; 32];
        for (i, val) in input.iter_mut().enumerate() {
            *val = (i as f32 - 16.0) * 0.3;
        }

        // Quantize
        let quantized = quantize_to_q8_0(&input).expect("quantize_to_q8_0 failed");
        assert_eq!(quantized.len(), Q8_0_BLOCK_BYTES);

        // Dequantize
        let mut output = [0.0f32; 32];
        dequantize_q8_0_block(&quantized, &mut output);

        // The scale is amax/127. Round-trip error per element should be at most ~scale.
        let amax = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        for i in 0..32 {
            let err = (output[i] - input[i]).abs();
            assert!(
                err <= scale + 1e-4,
                "Q8_0 roundtrip: element {i}: input={}, output={}, err={err}, max_allowed={scale}",
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_quantize_q4_0_roundtrip() {
        let mut input = [0.0f32; 32];
        for (i, val) in input.iter_mut().enumerate() {
            *val = (i as f32 - 16.0) * 0.5;
        }

        let quantized = quantize_to_q4_0(&input).expect("quantize_to_q4_0 failed");
        assert_eq!(quantized.len(), Q4_0_BLOCK_BYTES);

        let mut output = [0.0f32; 32];
        dequantize_q4_0_block(&quantized, &mut output);

        // Q4_0 has much larger quantization error (4-bit).
        // scale = amax / 8.0, each step is `scale`, max error ~ scale/2 + f16 rounding.
        let amax = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = amax / 8.0;
        let tolerance = scale + 0.1; // generous for 4-bit
        for i in 0..32 {
            let err = (output[i] - input[i]).abs();
            assert!(
                err <= tolerance,
                "Q4_0 roundtrip: element {i}: input={}, output={}, err={err}, tol={tolerance}",
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_quantize_q8_0_zeros() {
        let input = [0.0f32; 32];
        let quantized = quantize_to_q8_0(&input).expect("quantize zeros failed");
        let mut output = [0.0f32; 32];
        dequantize_q8_0_block(&quantized, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.abs() < 1e-10,
                "Q8_0 zero roundtrip: expected 0 at {i}, got {v}"
            );
        }
    }

    #[test]
    fn test_quantize_q4_0_zeros() {
        let input = [0.0f32; 32];
        let quantized = quantize_to_q4_0(&input).expect("quantize zeros failed");
        let mut output = [0.0f32; 32];
        dequantize_q4_0_block(&quantized, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.abs() < 1e-10,
                "Q4_0 zero roundtrip: expected 0 at {i}, got {v}"
            );
        }
    }

    #[test]
    fn test_quantize_q8_0_large_values() {
        let mut input = [0.0f32; 32];
        for (i, val) in input.iter_mut().enumerate() {
            *val = if i % 2 == 0 { 1000.0 } else { -1000.0 };
        }
        let quantized = quantize_to_q8_0(&input).expect("quantize large values failed");
        let mut output = [0.0f32; 32];
        dequantize_q8_0_block(&quantized, &mut output);

        let scale = 1000.0 / 127.0;
        for i in 0..32 {
            let err = (output[i] - input[i]).abs();
            assert!(
                err <= scale + 0.5,
                "Q8_0 large: element {i}: input={}, output={}, err={err}",
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_quantize_q4_0_large_values() {
        let mut input = [0.0f32; 32];
        for (i, val) in input.iter_mut().enumerate() {
            *val = if i % 2 == 0 { 1000.0 } else { -1000.0 };
        }
        let quantized = quantize_to_q4_0(&input).expect("quantize large values failed");
        let mut output = [0.0f32; 32];
        dequantize_q4_0_block(&quantized, &mut output);

        let scale = 1000.0 / 8.0;
        for i in 0..32 {
            let err = (output[i] - input[i]).abs();
            assert!(
                err <= scale + 1.0,
                "Q4_0 large: element {i}: input={}, output={}, err={err}",
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_quantize_q8_0_alignment_error() {
        let input = [0.0f32; 33]; // not multiple of 32
        let result = quantize_to_q8_0(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_q4_0_alignment_error() {
        let input = [0.0f32; 31];
        let result = quantize_to_q4_0(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_q8_0_multi_block() {
        // 64 elements = 2 blocks
        let mut input = vec![0.0f32; 64];
        for (i, val) in input.iter_mut().enumerate().take(64) {
            *val = (i as f32 - 32.0) * 0.1;
        }
        let quantized = quantize_to_q8_0(&input).expect("multi-block quantize failed");
        assert_eq!(quantized.len(), 2 * Q8_0_BLOCK_BYTES);

        let output = dequantize_q8_0(&quantized, 64);
        assert_eq!(output.len(), 64);

        let amax = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        for i in 0..64 {
            let err = (output[i] - input[i]).abs();
            assert!(
                err <= scale + 1e-3,
                "Q8_0 multi-block: element {i}: err={err}"
            );
        }
    }

    #[test]
    fn test_quantize_q4_0_multi_block() {
        let mut input = vec![0.0f32; 64];
        for (i, val) in input.iter_mut().enumerate().take(64) {
            *val = (i as f32 - 32.0) * 0.2;
        }
        let quantized = quantize_to_q4_0(&input).expect("multi-block quantize failed");
        assert_eq!(quantized.len(), 2 * Q4_0_BLOCK_BYTES);

        let output = dequantize_q4_0(&quantized, 64);
        assert_eq!(output.len(), 64);

        // Each block has its own scale, so check per-block tolerances
        for block_idx in 0..2 {
            let start = block_idx * 32;
            let block_slice = &input[start..start + 32];
            let block_amax = block_slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let block_scale = block_amax / 8.0;
            let tol = block_scale + 0.2;
            for i in 0..32 {
                let idx = start + i;
                let err = (output[idx] - input[idx]).abs();
                assert!(
                    err <= tol,
                    "Q4_0 multi-block: element {idx}: err={err}, tol={tol}"
                );
            }
        }
    }

    #[test]
    fn test_quantize_tensor_q8_0() {
        let input = vec![0.5f32; 64];
        let qt = quantize_tensor(&input, &[2, 32], QuantType::Q8_0)
            .expect("quantize_tensor Q8_0 failed");
        assert_eq!(qt.numel(), 64);
        assert_eq!(qt.qtype, QuantType::Q8_0);
        assert_eq!(qt.shape, vec![2, 32]);
        assert_eq!(qt.raw.len(), 2 * Q8_0_BLOCK_BYTES);
    }

    #[test]
    fn test_quantize_tensor_q4_0() {
        let input = vec![0.5f32; 64];
        let qt = quantize_tensor(&input, &[2, 32], QuantType::Q4_0)
            .expect("quantize_tensor Q4_0 failed");
        assert_eq!(qt.numel(), 64);
        assert_eq!(qt.qtype, QuantType::Q4_0);
        assert_eq!(qt.raw.len(), 2 * Q4_0_BLOCK_BYTES);
    }

    #[test]
    fn test_quantize_tensor_shape_mismatch() {
        let input = vec![0.5f32; 64];
        let result = quantize_tensor(&input, &[3, 32], QuantType::Q8_0);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_q8_0_preserves_sign() {
        // Ensure negative values survive the round-trip with correct sign
        let mut input = [0.0f32; 32];
        for i in 0..16 {
            input[i] = (i as f32 + 1.0) * 2.0;
            input[i + 16] = -((i as f32 + 1.0) * 2.0);
        }
        let quantized = quantize_to_q8_0(&input).expect("quantize failed");
        let mut output = [0.0f32; 32];
        dequantize_q8_0_block(&quantized, &mut output);

        for i in 0..16 {
            assert!(
                output[i] > 0.0,
                "Q8_0: expected positive at {i}, got {}",
                output[i]
            );
            assert!(
                output[i + 16] < 0.0,
                "Q8_0: expected negative at {}, got {}",
                i + 16,
                output[i + 16]
            );
            // Magnitude should be approximately symmetric
            let diff = (output[i].abs() - output[i + 16].abs()).abs();
            let scale = 32.0 / 127.0; // amax=32, scale=32/127
            assert!(
                diff <= scale + 0.1,
                "Q8_0: asymmetry at {i}: |{}| vs |{}|, diff={diff}",
                output[i],
                output[i + 16]
            );
        }
    }

    #[test]
    fn test_quantized_tensor_row_byte_offset() {
        // Shape [in_f=64, out_f=3]: 3 physical rows, each 64 elements (2 blocks of Q4_0)
        let qt = QuantizedTensor {
            raw: vec![0u8; Q4_0_BLOCK_BYTES * 6], // 6 blocks: 3 rows * 2 blocks/row
            shape: vec![64, 3],
            qtype: QuantType::Q4_0,
        };
        assert_eq!(qt.row_byte_offset(0), 0);
        assert_eq!(qt.row_byte_offset(1), 2 * Q4_0_BLOCK_BYTES);
        assert_eq!(qt.row_byte_offset(2), 4 * Q4_0_BLOCK_BYTES);
    }

    // -----------------------------------------------------------------------
    // SIMD Q8_0 dot product tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_q8_0_simd_matches_scalar() {
        // Quantize known data, verify SIMD dot matches scalar dot.
        // Use multiple blocks (2 blocks = 64 elements) for good coverage.
        let mut values1 = [0i8; 32];
        let mut values2 = [0i8; 32];
        for i in 0..32 {
            values1[i] = (i as i8) - 16;
            values2[i] = 16 - (i as i8);
        }
        let block1 = make_q8_0_block(0.25, &values1);
        let block2 = make_q8_0_block(1.5, &values2);

        let mut quantized = Vec::with_capacity(Q8_0_BLOCK_BYTES * 2);
        quantized.extend_from_slice(&block1);
        quantized.extend_from_slice(&block2);

        let mut input = vec![0.0f32; 64];
        for (i, val) in input.iter_mut().enumerate().take(64) {
            *val = (i as f32 + 1.0) * 0.05;
        }

        let scalar_result = dot_q8_0(&input, &quantized, 64);
        let fast_result = dot_q8_0_fast(&input, &quantized, 64);

        assert!(
            (fast_result - scalar_result).abs() < 1e-2,
            "SIMD dot_q8_0 mismatch: scalar={scalar_result}, fast={fast_result}"
        );
    }

    #[test]
    fn test_dot_q8_0_simd_zero_input() {
        // All zeros should produce zero result.
        let values = [0i8; 32];
        let block = make_q8_0_block(1.0, &values);
        let input = [0.0f32; 32];

        let result = dot_q8_0_fast(&input, &block, 32);
        assert!(
            result.abs() < 1e-10,
            "SIMD dot_q8_0 with zero input: expected ~0, got {result}"
        );

        // Also test with zero quantized data but non-zero input
        let zero_scale_block = make_q8_0_block(0.0, &[1i8; 32]);
        let nonzero_input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let result2 = dot_q8_0_fast(&nonzero_input, &zero_scale_block, 32);
        assert!(
            result2.abs() < 1e-10,
            "SIMD dot_q8_0 with zero scale: expected ~0, got {result2}"
        );
    }

    // -----------------------------------------------------------------------
    // SIMD Q4_0 dot product tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_q4_0_simd_matches_scalar() {
        // Quantize known data, verify SIMD dot matches scalar dot.
        // Use multiple blocks (2 blocks = 64 elements) for good coverage.
        let mut values1 = [0i32; 32];
        let mut values2 = [0i32; 32];
        for i in 0..32 {
            values1[i] = (i as i32 % 7) - 3;
            values2[i] = -((i as i32 % 5) - 2);
        }
        let block1 = make_q4_0_block(1.5, &values1);
        let block2 = make_q4_0_block(0.75, &values2);

        let mut quantized = Vec::with_capacity(Q4_0_BLOCK_BYTES * 2);
        quantized.extend_from_slice(&block1);
        quantized.extend_from_slice(&block2);

        let mut input = vec![0.0f32; 64];
        for (i, val) in input.iter_mut().enumerate().take(64) {
            *val = (i as f32 + 1.0) * 0.05;
        }

        let scalar_result = dot_q4_0(&input, &quantized, 64);
        let fast_result = dot_q4_0_fast(&input, &quantized, 64);

        assert!(
            (fast_result - scalar_result).abs() < 1e-2,
            "SIMD dot_q4_0 mismatch: scalar={scalar_result}, fast={fast_result}"
        );
    }

    #[test]
    fn test_dot_q4_0_simd_zero() {
        // All-zero quantized values should produce zero result.
        let values = [0i32; 32];
        let block = make_q4_0_block(1.0, &values);
        let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();

        let result = dot_q4_0_fast(&input, &block, 32);
        assert!(
            result.abs() < 1e-4,
            "SIMD dot_q4_0 with zero values: expected ~0, got {result}"
        );

        // Also test with zero scale
        let nonzero_values = [3i32; 32];
        let zero_scale_block = make_q4_0_block(0.0, &nonzero_values);
        let result2 = dot_q4_0_fast(&input, &zero_scale_block, 32);
        assert!(
            result2.abs() < 1e-4,
            "SIMD dot_q4_0 with zero scale: expected ~0, got {result2}"
        );

        // Zero input, non-zero quantized
        let zero_input = [0.0f32; 32];
        let block2 = make_q4_0_block(2.0, &[5i32; 32]);
        let result3 = dot_q4_0_fast(&zero_input, &block2, 32);
        assert!(
            result3.abs() < 1e-4,
            "SIMD dot_q4_0 with zero input: expected ~0, got {result3}"
        );
    }

    // -----------------------------------------------------------------------
    // Q5_0 tests
    // -----------------------------------------------------------------------

    /// Helper: build a Q5_0 block from a scale and 32 integer values in [-16, 15].
    /// Each value is stored as a 5-bit unsigned int (q + 16) with low 4 bits in
    /// nibble bytes and bit 4 in the high-bit mask.
    fn make_q5_0_block(scale: f32, values: &[i32; 32]) -> [u8; Q5_0_BLOCK_BYTES] {
        let mut block = [0u8; Q5_0_BLOCK_BYTES];
        let scale_f16 = f16::from_f32(scale);
        let le = scale_f16.to_le_bytes();
        block[0] = le[0];
        block[1] = le[1];

        let mut qh: u32 = 0;
        for i in 0..16 {
            let q_lo = (values[i] + 16) as u32; // 5-bit unsigned
            let q_hi = (values[i + 16] + 16) as u32;

            let lo_nibble = (q_lo & 0x0F) as u8;
            let hi_nibble = (q_hi & 0x0F) as u8;
            block[6 + i] = lo_nibble | (hi_nibble << 4);

            qh |= ((q_lo >> 4) & 1) << i;
            qh |= ((q_hi >> 4) & 1) << (i + 16);
        }

        let qh_bytes = qh.to_le_bytes();
        block[2] = qh_bytes[0];
        block[3] = qh_bytes[1];
        block[4] = qh_bytes[2];
        block[5] = qh_bytes[3];
        block
    }

    #[test]
    fn test_dequantize_q5_0_block() {
        // Create known values: range -10..5 repeated
        let mut values = [0i32; 32];
        for (i, val) in values.iter_mut().enumerate() {
            *val = (i as i32 % 16) - 10; // range -10..5
        }
        let scale = 2.0f32;
        let block = make_q5_0_block(scale, &values);

        let mut output = [0.0f32; Q5_0_BLOCK_SIZE];
        dequantize_q5_0_block(&block, &mut output);

        let actual_scale = f16::from_f32(scale).to_f32();
        for (i, (&val, &out)) in values.iter().zip(output.iter()).enumerate() {
            let expected = val as f32 * actual_scale;
            assert!(
                (out - expected).abs() < 1e-4,
                "Q5_0 mismatch at {i}: expected {expected}, got {out}",
            );
        }
    }

    #[test]
    fn test_dot_q5_0_basic() {
        let mut values = [0i32; 32];
        for (i, val) in values.iter_mut().enumerate() {
            *val = (i as i32 % 11) - 5; // range -5..5
        }
        let scale = 1.5f32;
        let block = make_q5_0_block(scale, &values);

        // Dequantize to get reference f32 values
        let mut deq = [0.0f32; Q5_0_BLOCK_SIZE];
        dequantize_q5_0_block(&block, &mut deq);

        // Input vector
        let mut input = [0.0f32; 32];
        for (i, val) in input.iter_mut().enumerate() {
            *val = (i as f32 + 1.0) * 0.1;
        }

        let expected: f32 = input.iter().zip(deq.iter()).map(|(a, b)| a * b).sum();
        let actual = dot_q5_0(&input, &block, 32);

        assert!(
            (actual - expected).abs() < 1e-2,
            "dot_q5_0: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_dot_q5_0_simd_matches_scalar() {
        // Use 2 blocks = 64 elements for coverage
        let mut values1 = [0i32; 32];
        let mut values2 = [0i32; 32];
        for (i, (v1, v2)) in values1.iter_mut().zip(values2.iter_mut()).enumerate() {
            *v1 = (i as i32 % 11) - 5;
            *v2 = -((i as i32 % 9) - 4);
        }
        let block1 = make_q5_0_block(1.5, &values1);
        let block2 = make_q5_0_block(0.75, &values2);

        let mut quantized = Vec::with_capacity(Q5_0_BLOCK_BYTES * 2);
        quantized.extend_from_slice(&block1);
        quantized.extend_from_slice(&block2);

        let mut input = vec![0.0f32; 64];
        for (i, val) in input.iter_mut().enumerate().take(64) {
            *val = (i as f32 + 1.0) * 0.05;
        }

        let scalar_result = dot_q5_0(&input, &quantized, 64);
        let fast_result = dot_q5_0_fast(&input, &quantized, 64);

        assert!(
            (fast_result - scalar_result).abs() < 1e-2,
            "SIMD dot_q5_0 mismatch: scalar={scalar_result}, fast={fast_result}"
        );
    }

    #[test]
    fn test_quantize_q5_0_roundtrip() {
        let mut input = [0.0f32; 32];
        for (i, val) in input.iter_mut().enumerate() {
            *val = (i as f32 - 16.0) * 0.4;
        }

        let quantized = quantize_to_q5_0(&input).expect("quantize_to_q5_0 failed");
        assert_eq!(quantized.len(), Q5_0_BLOCK_BYTES);

        let mut output = [0.0f32; 32];
        dequantize_q5_0_block(&quantized, &mut output);

        // Q5_0 has 5-bit quantization (32 levels).
        // scale = amax / 15.0, each step is `scale`, max error ~ scale/2 + f16 rounding.
        let amax = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = amax / 15.0;
        let tolerance = scale + 0.1;
        for (i, (&out, &inp)) in output.iter().zip(input.iter()).enumerate() {
            let err = (out - inp).abs();
            assert!(
                err <= tolerance,
                "Q5_0 roundtrip: element {i}: input={inp}, output={out}, err={err}, tol={tolerance}",
            );
        }
    }

    #[test]
    fn test_quantize_q5_0_zero() {
        let input = [0.0f32; 32];
        let quantized = quantize_to_q5_0(&input).expect("quantize zeros failed");
        let mut output = [0.0f32; 32];
        dequantize_q5_0_block(&quantized, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.abs() < 1e-10,
                "Q5_0 zero roundtrip: expected 0 at {i}, got {v}"
            );
        }
    }

    #[test]
    fn test_quantize_q5_0_alignment_error() {
        let input = [0.0f32; 33]; // not multiple of 32
        let result = quantize_to_q5_0(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_q5_0_multi_block() {
        let mut input = vec![0.0f32; 64];
        for (i, val) in input.iter_mut().enumerate().take(64) {
            *val = (i as f32 - 32.0) * 0.15;
        }
        let quantized = quantize_to_q5_0(&input).expect("multi-block quantize failed");
        assert_eq!(quantized.len(), 2 * Q5_0_BLOCK_BYTES);

        let output = dequantize_q5_0(&quantized, 64);
        assert_eq!(output.len(), 64);

        for block_idx in 0..2 {
            let start = block_idx * 32;
            let block_slice = &input[start..start + 32];
            let block_amax = block_slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let block_scale = block_amax / 15.0;
            let tol = block_scale + 0.15;
            for i in 0..32 {
                let idx = start + i;
                let err = (output[idx] - input[idx]).abs();
                assert!(
                    err <= tol,
                    "Q5_0 multi-block: element {idx}: err={err}, tol={tol}"
                );
            }
        }
    }

    #[test]
    fn test_quantize_tensor_q5_0() {
        let input = vec![0.5f32; 64];
        let qt = quantize_tensor(&input, &[2, 32], QuantType::Q5_0)
            .expect("quantize_tensor Q5_0 failed");
        assert_eq!(qt.numel(), 64);
        assert_eq!(qt.qtype, QuantType::Q5_0);
        assert_eq!(qt.raw.len(), 2 * Q5_0_BLOCK_BYTES);
    }

    #[test]
    fn test_quantized_tensor_q5_0_creation() {
        let qt = QuantizedTensor {
            raw: vec![0u8; Q5_0_BLOCK_BYTES * 4],
            shape: vec![4, 32],
            qtype: QuantType::Q5_0,
        };
        assert_eq!(qt.numel(), 128);
        assert_eq!(qt.block_size(), Q5_0_BLOCK_SIZE);
        assert_eq!(qt.block_bytes(), Q5_0_BLOCK_BYTES);
        assert_eq!(qt.qtype, QuantType::Q5_0);
    }

    #[test]
    fn test_q5_0_high_bit_values() {
        // Test values that require the 5th bit (>= 16 unsigned, i.e. >= 0 signed)
        let mut values = [0i32; 32];
        for (i, val) in values.iter_mut().enumerate() {
            // Values 0..15 which map to unsigned 16..31 (all need high bit set)
            *val = i as i32 % 16;
        }
        let scale = 1.0f32;
        let block = make_q5_0_block(scale, &values);

        let mut output = [0.0f32; Q5_0_BLOCK_SIZE];
        dequantize_q5_0_block(&block, &mut output);

        let actual_scale = f16::from_f32(scale).to_f32();
        for (i, (&val, &out)) in values.iter().zip(output.iter()).enumerate() {
            let expected = val as f32 * actual_scale;
            assert!(
                (out - expected).abs() < 1e-4,
                "Q5_0 high-bit test at {i}: expected {expected}, got {out}",
            );
        }
    }
}
