//! Quantization: f32 -> Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 block and full-tensor
//! quantization.

use half::f16;

use super::types::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Q4_1_BLOCK_BYTES, Q4_1_BLOCK_SIZE, Q5_0_BLOCK_BYTES,
    Q5_0_BLOCK_SIZE, Q5_1_BLOCK_BYTES, Q5_1_BLOCK_SIZE, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE,
    QuantType, QuantizedTensor,
};

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

/// Quantize a single block of 32 f32 values into Q4_1 format (20 bytes).
///
/// Layout: `[f16 scale d LE][f16 min m LE][16 bytes of nibble pairs]`.
///
/// Q4_1 is an *affine* scheme: the block minimum is stored explicitly and the
/// nibble is unsigned, `q = clamp(round((value - m) / d), 0, 15)`, so that
/// `value ≈ q * d + m`.
///
/// # Panics
///
/// Debug-asserts that `input.len() >= 32` and `output.len() >= 20`.
pub fn quantize_block_q4_1(input: &[f32], output: &mut [u8]) {
    debug_assert!(input.len() >= Q4_1_BLOCK_SIZE);
    debug_assert!(output.len() >= Q4_1_BLOCK_BYTES);

    let block = &input[..Q4_1_BLOCK_SIZE];
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in block {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let scale = (max - min) / 15.0;
    // Round-trip the affine parameters through f16 so the quantized values are
    // chosen against the coefficients the dequantizer will actually see.
    let scale = f16::from_f32(scale).to_f32();
    let min = f16::from_f32(min).to_f32();
    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

    output[0..2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
    output[2..4].copy_from_slice(&f16::from_f32(min).to_le_bytes());

    for i in 0..16 {
        let lo = (((block[i] - min) * inv_scale).round()).clamp(0.0, 15.0) as u8;
        let hi = (((block[i + 16] - min) * inv_scale).round()).clamp(0.0, 15.0) as u8;
        output[4 + i] = lo | (hi << 4);
    }
}

/// Quantize a single block of 32 f32 values into Q5_1 format (24 bytes).
///
/// Layout: `[f16 scale d LE][f16 min m LE][u32 high-bit mask LE][16 nibble pairs]`.
///
/// Affine 5-bit scheme: `q = clamp(round((value - m) / d), 0, 31)` with the low
/// four bits packed like Q4_0 and bit 4 stored in the per-block mask.
///
/// # Panics
///
/// Debug-asserts that `input.len() >= 32` and `output.len() >= 24`.
pub fn quantize_block_q5_1(input: &[f32], output: &mut [u8]) {
    debug_assert!(input.len() >= Q5_1_BLOCK_SIZE);
    debug_assert!(output.len() >= Q5_1_BLOCK_BYTES);

    let block = &input[..Q5_1_BLOCK_SIZE];
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in block {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let scale = (max - min) / 31.0;
    let scale = f16::from_f32(scale).to_f32();
    let min = f16::from_f32(min).to_f32();
    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

    output[0..2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
    output[2..4].copy_from_slice(&f16::from_f32(min).to_le_bytes());

    let mut qh: u32 = 0;
    for i in 0..16 {
        let lo_q = (((block[i] - min) * inv_scale).round()).clamp(0.0, 31.0) as u32;
        let hi_q = (((block[i + 16] - min) * inv_scale).round()).clamp(0.0, 31.0) as u32;

        output[8 + i] = (lo_q & 0x0F) as u8 | (((hi_q & 0x0F) as u8) << 4);
        qh |= ((lo_q >> 4) & 1) << i;
        qh |= ((hi_q >> 4) & 1) << (i + 16);
    }
    output[4..8].copy_from_slice(&qh.to_le_bytes());
}

/// Quantize an entire f32 slice into Q4_1 format.
///
/// `data.len()` must be divisible by 32.
///
/// # Errors
///
/// Returns `Err` if the input length is not a multiple of `Q4_1_BLOCK_SIZE`.
pub fn quantize_to_q4_1(data: &[f32]) -> Result<Vec<u8>, String> {
    if !data.len().is_multiple_of(Q4_1_BLOCK_SIZE) {
        return Err(format!(
            "quantize_to_q4_1: input length {} is not a multiple of {}",
            data.len(),
            Q4_1_BLOCK_SIZE
        ));
    }
    let n_blocks = data.len() / Q4_1_BLOCK_SIZE;
    let mut output = vec![0u8; n_blocks * Q4_1_BLOCK_BYTES];

    for b in 0..n_blocks {
        let in_start = b * Q4_1_BLOCK_SIZE;
        let out_start = b * Q4_1_BLOCK_BYTES;
        quantize_block_q4_1(
            &data[in_start..in_start + Q4_1_BLOCK_SIZE],
            &mut output[out_start..out_start + Q4_1_BLOCK_BYTES],
        );
    }
    Ok(output)
}

/// Quantize an entire f32 slice into Q5_1 format.
///
/// `data.len()` must be divisible by 32.
///
/// # Errors
///
/// Returns `Err` if the input length is not a multiple of `Q5_1_BLOCK_SIZE`.
pub fn quantize_to_q5_1(data: &[f32]) -> Result<Vec<u8>, String> {
    if !data.len().is_multiple_of(Q5_1_BLOCK_SIZE) {
        return Err(format!(
            "quantize_to_q5_1: input length {} is not a multiple of {}",
            data.len(),
            Q5_1_BLOCK_SIZE
        ));
    }
    let n_blocks = data.len() / Q5_1_BLOCK_SIZE;
    let mut output = vec![0u8; n_blocks * Q5_1_BLOCK_BYTES];

    for b in 0..n_blocks {
        let in_start = b * Q5_1_BLOCK_SIZE;
        let out_start = b * Q5_1_BLOCK_BYTES;
        quantize_block_q5_1(
            &data[in_start..in_start + Q5_1_BLOCK_SIZE],
            &mut output[out_start..out_start + Q5_1_BLOCK_BYTES],
        );
    }
    Ok(output)
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
        QuantType::Q4_1 => quantize_to_q4_1(data)?,
        QuantType::Q5_0 => quantize_to_q5_0(data)?,
        QuantType::Q5_1 => quantize_to_q5_1(data)?,
        QuantType::Q8_0 => quantize_to_q8_0(data)?,
    };

    Ok(QuantizedTensor {
        raw,
        shape: shape.to_vec(),
        qtype,
    })
}
