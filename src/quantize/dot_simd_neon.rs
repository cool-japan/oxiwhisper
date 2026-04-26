// This file only compiles on aarch64 targets.
// The parent mod.rs wraps the `mod dot_simd_neon;` declaration with
// `#[cfg(target_arch = "aarch64")]`.

// ---------------------------------------------------------------------------
// SIMD-accelerated quantized dot products (NEON, aarch64)
// ---------------------------------------------------------------------------

use half::f16;

use super::types::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_SIZE,
};

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
