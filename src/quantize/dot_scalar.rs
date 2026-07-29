//! Scalar (non-SIMD) quantized dot product implementations.

use half::f16;

use super::types::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Q4_1_BLOCK_BYTES, Q4_1_BLOCK_SIZE, Q5_0_BLOCK_BYTES,
    Q5_0_BLOCK_SIZE, Q5_1_BLOCK_BYTES, Q5_1_BLOCK_SIZE, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE,
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

/// Compute dot product between an f32 vector and a Q4_1 quantized vector.
///
/// Q4_1 is affine (`value = q * d + m`), so the per-block contribution is
/// `d * Σ q_i·x_i + m * Σ x_i` — both sums are accumulated in one pass.
pub fn dot_q4_1(input: &[f32], quantized: &[u8], n: usize) -> f32 {
    let n_blocks = n / Q4_1_BLOCK_SIZE;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block_start = b * Q4_1_BLOCK_BYTES;
        let scale =
            f16::from_le_bytes([quantized[block_start], quantized[block_start + 1]]).to_f32();
        let min =
            f16::from_le_bytes([quantized[block_start + 2], quantized[block_start + 3]]).to_f32();
        let input_slice = &input[b * Q4_1_BLOCK_SIZE..];

        let mut q_sum = 0.0f32;
        let mut x_sum = 0.0f32;
        for i in 0..16 {
            let byte = quantized[block_start + 4 + i];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            let x_lo = input_slice[i];
            let x_hi = input_slice[i + 16];
            q_sum += x_lo * lo + x_hi * hi;
            x_sum += x_lo + x_hi;
        }
        sum += q_sum * scale + x_sum * min;
    }
    sum
}

/// Compute dot product between an f32 vector and a Q5_1 quantized vector.
///
/// Q5_1 is affine (`value = q * d + m`) with the 5th bit of every value packed
/// into a per-block `u32` mask.
pub fn dot_q5_1(input: &[f32], quantized: &[u8], n: usize) -> f32 {
    let n_blocks = n / Q5_1_BLOCK_SIZE;
    let mut total = 0.0f32;

    for blk in 0..n_blocks {
        let a_off = blk * Q5_1_BLOCK_BYTES;
        let b_off = blk * Q5_1_BLOCK_SIZE;
        let scale = f16::from_le_bytes([quantized[a_off], quantized[a_off + 1]]).to_f32();
        let min = f16::from_le_bytes([quantized[a_off + 2], quantized[a_off + 3]]).to_f32();
        let qh = u32::from_le_bytes([
            quantized[a_off + 4],
            quantized[a_off + 5],
            quantized[a_off + 6],
            quantized[a_off + 7],
        ]);

        let mut q_sum = 0.0f32;
        let mut x_sum = 0.0f32;
        for i in 0..16 {
            let byte = quantized[a_off + 8 + i];
            let lo = (byte & 0x0F) as u32;
            let hi = ((byte >> 4) & 0x0F) as u32;
            let hi_bit_lo = (qh >> i) & 1;
            let hi_bit_hi = (qh >> (i + 16)) & 1;

            let v_lo = (lo | (hi_bit_lo << 4)) as f32;
            let v_hi = (hi | (hi_bit_hi << 4)) as f32;
            let x_lo = input[b_off + i];
            let x_hi = input[b_off + i + 16];
            q_sum += v_lo * x_lo + v_hi * x_hi;
            x_sum += x_lo + x_hi;
        }
        total += scale * q_sum + min * x_sum;
    }
    total
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
