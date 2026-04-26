//! Dequantization: Q4_0, Q5_0, Q8_0 block and full-tensor dequantization.

use half::f16;

use super::types::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_SIZE,
};

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
