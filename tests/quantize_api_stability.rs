// Smoke test: verify every public quantize symbol is accessible after the split.
// If any re-export is missing, this file won't compile.

use oxiwhisper::quantize::{
    Q4_0_BLOCK_BYTES,
    // Block-size constants
    Q4_0_BLOCK_SIZE,
    Q5_0_BLOCK_BYTES,
    Q5_0_BLOCK_SIZE,
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_SIZE,
    // Types
    QuantType,
    QuantizeStats,
    QuantizedTensor,
    // Dequantize tensor functions
    dequantize_q4_0,
    // Dequantize block functions
    dequantize_q4_0_block,
    dequantize_q5_0,
    dequantize_q5_0_block,
    dequantize_q8_0,
    dequantize_q8_0_block,
    // Scalar dot products
    dot_q4_0,
    // SIMD dispatch dot products
    dot_q4_0_fast,
    dot_q5_0,
    dot_q5_0_fast,
    dot_q8_0,
    dot_q8_0_fast,
    // Quantize block functions
    quantize_block_q4_0,
    quantize_block_q5_0,
    quantize_block_q8_0,
    quantize_tensor,
    // Quantize tensor functions
    quantize_to_q4_0,
    quantize_to_q5_0,
    quantize_to_q8_0,
};

#[test]
fn public_quantize_symbols_accessible() {
    // Verify QuantType variants are accessible.
    let _q4: QuantType = QuantType::Q4_0;
    let _q5: QuantType = QuantType::Q5_0;
    let _q8: QuantType = QuantType::Q8_0;

    // Verify QuantizedTensor can be constructed.
    let _qt = QuantizedTensor {
        raw: vec![0u8; Q8_0_BLOCK_BYTES],
        shape: vec![32],
        qtype: QuantType::Q8_0,
    };
    let _: usize = _qt.numel();
    let _: usize = _qt.block_size();
    let _: usize = _qt.block_bytes();
    let _: usize = _qt.row_elements();

    // Verify QuantizeStats can be constructed.
    let _stats = QuantizeStats {
        tensors_quantized: 0,
        tensors_kept_f32: 0,
        original_size_bytes: 0,
        quantized_size_bytes: 0,
    };

    // Verify constants have expected values.
    assert_eq!(Q4_0_BLOCK_SIZE, 32);
    assert_eq!(Q4_0_BLOCK_BYTES, 18);
    assert_eq!(Q5_0_BLOCK_SIZE, 32);
    assert_eq!(Q5_0_BLOCK_BYTES, 22);
    assert_eq!(Q8_0_BLOCK_SIZE, 32);
    assert_eq!(Q8_0_BLOCK_BYTES, 34);

    // Exercise dequantize block functions with minimal data.
    let q4_block = vec![0u8; Q4_0_BLOCK_BYTES];
    let mut out4 = vec![0.0f32; Q4_0_BLOCK_SIZE];
    dequantize_q4_0_block(&q4_block, &mut out4);

    let q5_block = vec![0u8; Q5_0_BLOCK_BYTES];
    let mut out5 = vec![0.0f32; Q5_0_BLOCK_SIZE];
    dequantize_q5_0_block(&q5_block, &mut out5);

    let q8_block = vec![0u8; Q8_0_BLOCK_BYTES];
    let mut out8 = vec![0.0f32; Q8_0_BLOCK_SIZE];
    dequantize_q8_0_block(&q8_block, &mut out8);

    // Exercise dequantize tensor functions.
    let _v4 = dequantize_q4_0(&q4_block, Q4_0_BLOCK_SIZE);
    let _v5 = dequantize_q5_0(&q5_block, Q5_0_BLOCK_SIZE);
    let _v8 = dequantize_q8_0(&q8_block, Q8_0_BLOCK_SIZE);

    // Exercise quantize block functions.
    let input32 = vec![0.0f32; 32];
    let mut qb4 = vec![0u8; Q4_0_BLOCK_BYTES];
    quantize_block_q4_0(&input32, &mut qb4);
    let mut qb5 = vec![0u8; Q5_0_BLOCK_BYTES];
    quantize_block_q5_0(&input32, &mut qb5);
    let mut qb8 = vec![0u8; Q8_0_BLOCK_BYTES];
    quantize_block_q8_0(&input32, &mut qb8);

    // Exercise quantize tensor functions.
    let _r4 = quantize_to_q4_0(&input32).expect("quantize_to_q4_0");
    let _r5 = quantize_to_q5_0(&input32).expect("quantize_to_q5_0");
    let _r8 = quantize_to_q8_0(&input32).expect("quantize_to_q8_0");
    let _qt2 = quantize_tensor(&input32, &[32], QuantType::Q8_0).expect("quantize_tensor");

    // Exercise scalar dot products.
    let _d4 = dot_q4_0(&input32, &qb4, Q4_0_BLOCK_SIZE);
    let _d5 = dot_q5_0(&input32, &qb5, Q5_0_BLOCK_SIZE);
    let _d8 = dot_q8_0(&input32, &qb8, Q8_0_BLOCK_SIZE);

    // Exercise SIMD dispatch dot products.
    let _df4 = dot_q4_0_fast(&input32, &qb4, Q4_0_BLOCK_SIZE);
    let _df5 = dot_q5_0_fast(&input32, &qb5, Q5_0_BLOCK_SIZE);
    let _df8 = dot_q8_0_fast(&input32, &qb8, Q8_0_BLOCK_SIZE);
}
