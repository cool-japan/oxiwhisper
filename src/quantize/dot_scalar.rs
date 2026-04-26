//! Scalar (non-SIMD) quantized dot product implementations.

use half::f16;

use super::types::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_SIZE,
};

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
