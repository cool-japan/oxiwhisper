// This file only compiles on x86_64 targets.
// The parent mod.rs wraps the `mod dot_simd_x86;` declaration with
// `#[cfg(target_arch = "x86_64")]`.

// ---------------------------------------------------------------------------
// SIMD-accelerated quantized dot products (AVX2 + FMA, x86_64)
// ---------------------------------------------------------------------------

use half::f16;

use super::types::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_SIZE,
};

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
