//! Tests for the quantize module.

use half::f16;

use super::dequant::{
    dequantize_q4_0, dequantize_q4_0_block, dequantize_q4_1_block, dequantize_q5_0,
    dequantize_q5_0_block, dequantize_q5_1_block, dequantize_q8_0, dequantize_q8_0_block,
};
use super::dot_dispatch::{
    dot_q4_0_fast, dot_q4_1_fast, dot_q5_0_fast, dot_q5_1_fast, dot_q8_0_fast,
};
use super::dot_scalar::{dot_q4_0, dot_q4_1, dot_q5_0, dot_q5_1, dot_q8_0};
use super::quant::{quantize_tensor, quantize_to_q4_0, quantize_to_q5_0, quantize_to_q8_0};
use super::types::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_SIZE, QuantType, QuantizedTensor,
};

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
    let qt =
        quantize_tensor(&input, &[2, 32], QuantType::Q8_0).expect("quantize_tensor Q8_0 failed");
    assert_eq!(qt.numel(), 64);
    assert_eq!(qt.qtype, QuantType::Q8_0);
    assert_eq!(qt.shape, vec![2, 32]);
    assert_eq!(qt.raw.len(), 2 * Q8_0_BLOCK_BYTES);
}

#[test]
fn test_quantize_tensor_q4_0() {
    let input = vec![0.5f32; 64];
    let qt =
        quantize_tensor(&input, &[2, 32], QuantType::Q4_0).expect("quantize_tensor Q4_0 failed");
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
    let qt =
        quantize_tensor(&input, &[2, 32], QuantType::Q5_0).expect("quantize_tensor Q5_0 failed");
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

// ── Q4_1 / Q5_1 (affine schemes) ──────────────────────────────────────────────

/// Build a Q4_1 block (20 bytes) from `d`, `m` and 32 unsigned nibbles.
fn make_q4_1_block(d: f32, m: f32, values: &[u8; 32]) -> [u8; super::types::Q4_1_BLOCK_BYTES] {
    let mut block = [0u8; super::types::Q4_1_BLOCK_BYTES];
    block[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
    block[2..4].copy_from_slice(&f16::from_f32(m).to_le_bytes());
    for i in 0..16 {
        block[4 + i] = (values[i] & 0x0F) | ((values[i + 16] & 0x0F) << 4);
    }
    block
}

/// Build a Q5_1 block (24 bytes) from `d`, `m` and 32 unsigned 5-bit values.
fn make_q5_1_block(d: f32, m: f32, values: &[u32; 32]) -> [u8; super::types::Q5_1_BLOCK_BYTES] {
    let mut block = [0u8; super::types::Q5_1_BLOCK_BYTES];
    block[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
    block[2..4].copy_from_slice(&f16::from_f32(m).to_le_bytes());
    let mut qh: u32 = 0;
    for i in 0..16 {
        let lo = values[i];
        let hi = values[i + 16];
        block[8 + i] = (lo & 0x0F) as u8 | (((hi & 0x0F) as u8) << 4);
        qh |= ((lo >> 4) & 1) << i;
        qh |= ((hi >> 4) & 1) << (i + 16);
    }
    block[4..8].copy_from_slice(&qh.to_le_bytes());
    block
}

#[test]
fn test_ggml_type_table_matches_ggml_conventions() {
    // Regression: the legacy GGML loader used to map dtype 3 to Q8_0 (3 is
    // Q4_1) and rejected 7/8 outright, so `ggml-*-q8_0.bin` and
    // `ggml-*-q5_1.bin` could not be loaded at all.
    assert_eq!(QuantType::from_ggml_type(2), Some(QuantType::Q4_0));
    assert_eq!(QuantType::from_ggml_type(3), Some(QuantType::Q4_1));
    assert_eq!(QuantType::from_ggml_type(6), Some(QuantType::Q5_0));
    assert_eq!(QuantType::from_ggml_type(7), Some(QuantType::Q5_1));
    assert_eq!(QuantType::from_ggml_type(8), Some(QuantType::Q8_0));
    // Non-quantized and unimplemented schemes must not be guessed at.
    for code in [0u32, 1, 4, 5, 9, 10, 11, 12, 13, 14, 15] {
        assert_eq!(
            QuantType::from_ggml_type(code),
            None,
            "ggml_type {code} must not map to a quantization kernel"
        );
    }
}

#[test]
fn test_block_geometry_for_every_quant_type() {
    let expected = [
        (QuantType::Q4_0, 32usize, 18usize),
        (QuantType::Q4_1, 32, 20),
        (QuantType::Q5_0, 32, 22),
        (QuantType::Q5_1, 32, 24),
        (QuantType::Q8_0, 32, 34),
    ];
    for (qt, size, bytes) in expected {
        assert_eq!(qt.block_size(), size, "{} block size", qt.name());
        assert_eq!(qt.block_bytes(), bytes, "{} block bytes", qt.name());
    }
}

#[test]
fn test_dequantize_q4_1_block_affine() {
    // value = q * d + m
    let d = 0.25f32;
    let m = -1.5f32;
    let mut values = [0u8; 32];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i % 16) as u8;
    }
    let block = make_q4_1_block(d, m, &values);
    let mut out = [0.0f32; 32];
    dequantize_q4_1_block(&block, &mut out);
    for i in 0..32 {
        let expected = values[i] as f32 * d + m;
        assert!(
            (out[i] - expected).abs() < 1e-5,
            "element {i}: got {}, expected {expected}",
            out[i]
        );
    }
}

#[test]
fn test_dequantize_q5_1_block_uses_high_bit() {
    let d = 0.125f32;
    let m = 2.0f32;
    let mut values = [0u32; 32];
    for (i, v) in values.iter_mut().enumerate() {
        // Sweep the whole 0..=31 range so the packed 5th bit matters.
        *v = (i as u32) % 32;
    }
    let block = make_q5_1_block(d, m, &values);
    let mut out = [0.0f32; 32];
    dequantize_q5_1_block(&block, &mut out);
    for i in 0..32 {
        let expected = values[i] as f32 * d + m;
        assert!(
            (out[i] - expected).abs() < 1e-5,
            "element {i}: got {}, expected {expected}",
            out[i]
        );
    }
}

#[test]
fn test_dot_q4_1_matches_dequantized_dot() {
    let d = 0.0625f32;
    let m = -0.5f32;
    let mut values = [0u8; 32];
    for (i, v) in values.iter_mut().enumerate() {
        *v = ((i * 7) % 16) as u8;
    }
    let block = make_q4_1_block(d, m, &values);
    let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();

    let mut deq = [0.0f32; 32];
    dequantize_q4_1_block(&block, &mut deq);
    let reference: f32 = input.iter().zip(deq.iter()).map(|(a, b)| a * b).sum();

    let got = dot_q4_1(&input, &block, 32);
    assert!(
        (got - reference).abs() < 1e-4,
        "dot_q4_1 = {got}, dequantized reference = {reference}"
    );
    assert!((dot_q4_1_fast(&input, &block, 32) - reference).abs() < 1e-4);
}

#[test]
fn test_dot_q5_1_matches_dequantized_dot() {
    let d = 0.03125f32;
    let m = 1.25f32;
    let mut values = [0u32; 32];
    for (i, v) in values.iter_mut().enumerate() {
        *v = ((i * 11) % 32) as u32;
    }
    let block = make_q5_1_block(d, m, &values);
    let input: Vec<f32> = (0..32).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();

    let mut deq = [0.0f32; 32];
    dequantize_q5_1_block(&block, &mut deq);
    let reference: f32 = input.iter().zip(deq.iter()).map(|(a, b)| a * b).sum();

    let got = dot_q5_1(&input, &block, 32);
    assert!(
        (got - reference).abs() < 1e-4,
        "dot_q5_1 = {got}, dequantized reference = {reference}"
    );
    assert!((dot_q5_1_fast(&input, &block, 32) - reference).abs() < 1e-4);
}

#[test]
fn test_quantize_dequantize_round_trip_affine_types() {
    // Affine schemes carry an explicit block minimum, so they represent an
    // asymmetric range far better than the symmetric Q4_0/Q5_0.
    let data: Vec<f32> = (0..128).map(|i| 3.0 + (i as f32) * 0.02).collect();

    for (qt, tol) in [(QuantType::Q4_1, 0.05f32), (QuantType::Q5_1, 0.02)] {
        let tensor = quantize_tensor(&data, &[128], qt).expect("quantize");
        assert_eq!(tensor.qtype, qt);
        assert_eq!(tensor.raw.len(), 4 * qt.block_bytes());

        let back = super::dequant::dequantize(&tensor.raw, data.len(), qt);
        assert_eq!(back.len(), data.len());
        for (i, (&orig, &rt)) in data.iter().zip(back.iter()).enumerate() {
            assert!(
                (orig - rt).abs() < tol,
                "{} element {i}: {orig} -> {rt}",
                qt.name()
            );
        }
    }
}

#[test]
fn test_quantize_to_affine_rejects_unaligned_length() {
    assert!(super::quant::quantize_to_q4_1(&[0.0f32; 31]).is_err());
    assert!(super::quant::quantize_to_q5_1(&[0.0f32; 33]).is_err());
}

#[test]
fn test_quantized_tensor_row_access_for_affine_types() {
    // Two rows of 32 elements each; row 1 must start exactly one block in.
    let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5 - 8.0).collect();
    for qt in [QuantType::Q4_1, QuantType::Q5_1] {
        let t = quantize_tensor(&data, &[32, 2], qt).expect("quantize");
        assert_eq!(t.row_byte_offset(0), 0);
        assert_eq!(t.row_byte_offset(1), qt.block_bytes());
        assert_eq!(t.row_bytes(1).len(), qt.block_bytes());
    }
}
