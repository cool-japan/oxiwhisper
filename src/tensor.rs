/// Minimal tensor type for Whisper inference.
/// Supports 1D-4D tensors with f32 storage.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Flat f32 element buffer in row-major (C-contiguous) order.
    pub data: Vec<f32>,
    /// Dimension sizes, e.g. `[seq_len, n_state]` for a 2-D tensor.
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a zero-filled tensor with the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape: shape.to_vec(),
        }
    }

    /// Create a tensor from an existing data buffer and shape.
    ///
    /// Panics in debug builds if `data.len() != shape.iter().product()`.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "data len {} != shape product {:?}",
            data.len(),
            shape
        );
        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    /// Returns the total number of elements (product of all dimension sizes).
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Returns the size of the last dimension.
    fn last_dim(&self) -> usize {
        self.shape[self.shape.len() - 1]
    }

    /// Returns the number of dimensions (rank) of this tensor.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Reshape to a new shape. Total elements must match.
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        debug_assert_eq!(
            self.numel(),
            new_shape.iter().product::<usize>(),
            "reshape: {} vs {:?}",
            self.numel(),
            new_shape
        );
        Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
        }
    }

    /// Zero-copy reshape: changes only the shape metadata without cloning data.
    pub fn reshape_inplace(&mut self, new_shape: &[usize]) {
        debug_assert_eq!(
            self.numel(),
            new_shape.iter().product::<usize>(),
            "reshape_inplace: {} vs {:?}",
            self.numel(),
            new_shape
        );
        self.shape = new_shape.to_vec();
    }

    /// Transpose a 2D matrix
    pub fn transpose_2d(&self) -> Self {
        assert_eq!(self.ndim(), 2);
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = self.data[r * cols + c];
            }
        }
        Self {
            data: out,
            shape: vec![cols, rows],
        }
    }

    /// 2D matrix multiply: (M, K) x (K, N) -> (M, N)
    pub fn matmul(&self, other: &Tensor) -> Self {
        assert_eq!(self.ndim(), 2);
        assert_eq!(other.ndim(), 2);
        let m = self.shape[0];
        let k = self.shape[1];
        assert_eq!(other.shape[0], k, "matmul: inner dims must match");
        let n = other.shape[1];

        let mut out = vec![0.0f32; m * n];

        // Simple tiled matmul for cache friendliness
        const TILE: usize = 32;
        for i0 in (0..m).step_by(TILE) {
            for j0 in (0..n).step_by(TILE) {
                for k0 in (0..k).step_by(TILE) {
                    let i_end = (i0 + TILE).min(m);
                    let j_end = (j0 + TILE).min(n);
                    let k_end = (k0 + TILE).min(k);
                    for i in i0..i_end {
                        for kk in k0..k_end {
                            let a_val = self.data[i * k + kk];
                            for j in j0..j_end {
                                out[i * n + j] += a_val * other.data[kk * n + j];
                            }
                        }
                    }
                }
            }
        }

        Self {
            data: out,
            shape: vec![m, n],
        }
    }

    /// Batched matmul: (..., M, K) x (..., K, N) -> (..., M, N)
    /// Both tensors must have the same batch dims.
    pub fn batched_matmul(&self, other: &Tensor) -> Self {
        assert!(self.ndim() >= 2);
        assert!(other.ndim() >= 2);

        let m = self.shape[self.ndim() - 2];
        let k = self.shape[self.ndim() - 1];
        let n = other.shape[other.ndim() - 1];
        assert_eq!(other.shape[other.ndim() - 2], k);

        let batch_size: usize = self.shape[..self.ndim() - 2].iter().product();
        let other_batch: usize = other.shape[..other.ndim() - 2].iter().product();
        assert_eq!(batch_size, other_batch);

        let a_stride = m * k;
        let b_stride = k * n;
        let c_stride = m * n;
        let mut out = vec![0.0f32; batch_size * c_stride];

        for b in 0..batch_size {
            let a_off = b * a_stride;
            let b_off = b * b_stride;
            let c_off = b * c_stride;
            for i in 0..m {
                for kk in 0..k {
                    let a_val = self.data[a_off + i * k + kk];
                    for j in 0..n {
                        out[c_off + i * n + j] += a_val * other.data[b_off + kk * n + j];
                    }
                }
            }
        }

        let mut shape = self.shape[..self.ndim() - 2].to_vec();
        shape.push(m);
        shape.push(n);
        Self { data: out, shape }
    }

    /// Element-wise add
    pub fn add(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape, "add: shapes must match");
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            return self.add_simd128(other);
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            let data: Vec<f32> = self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect();
            Self {
                data,
                shape: self.shape.clone(),
            }
        }
    }

    /// Add a bias vector to each row of a 2D tensor
    pub fn add_bias(&self, bias: &Tensor) -> Self {
        assert_eq!(self.ndim(), 2);
        assert_eq!(bias.ndim(), 1);
        assert_eq!(self.shape[1], bias.shape[0]);
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            return self.add_bias_simd128(bias);
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            let cols = self.shape[1];
            let mut data = self.data.clone();
            for row in data.chunks_mut(cols) {
                for (v, b) in row.iter_mut().zip(&bias.data) {
                    *v += b;
                }
            }
            Self {
                data,
                shape: self.shape.clone(),
            }
        }
    }

    /// Softmax along the last dimension
    pub fn softmax(&self) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            return self.softmax_simd128();
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            let last_dim = self.last_dim();
            let mut data = self.data.clone();

            for row in data.chunks_mut(last_dim) {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }

            Self {
                data,
                shape: self.shape.clone(),
            }
        }
    }

    /// GELU activation
    pub fn gelu(&self) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            return self.gelu_simd128();
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            let data: Vec<f32> = self
                .data
                .iter()
                .map(|&x| {
                    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    let c = 0.797_884_6_f32; // sqrt(2/pi)
                    let inner = c * (x + 0.044715 * x * x * x);
                    0.5 * x * (1.0 + inner.tanh())
                })
                .collect();
            Self {
                data,
                shape: self.shape.clone(),
            }
        }
    }

    /// Layer normalization along the last dimension
    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Self {
        let last_dim = self.last_dim();
        assert_eq!(weight.numel(), last_dim);
        assert_eq!(bias.numel(), last_dim);

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            return self.layer_norm_simd128(weight, bias, eps);
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            let mut data = self.data.clone();

            for row in data.chunks_mut(last_dim) {
                let mean: f32 = row.iter().sum::<f32>() / last_dim as f32;
                let var: f32 =
                    row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / last_dim as f32;
                let std = (var + eps).sqrt();

                for (i, v) in row.iter_mut().enumerate() {
                    *v = (*v - mean) / std * weight.data[i] + bias.data[i];
                }
            }

            Self {
                data,
                shape: self.shape.clone(),
            }
        }
    }

    // -----------------------------------------------------------------
    // In-place operations (avoid allocations for repeated inference)
    // -----------------------------------------------------------------

    /// GELU activation applied in-place.
    pub fn gelu_inplace(&mut self) {
        let c = 0.797_884_6_f32; // sqrt(2/pi)
        for v in &mut self.data {
            let x = *v;
            let inner = c * (x + 0.044715 * x * x * x);
            *v = 0.5 * x * (1.0 + inner.tanh());
        }
    }

    /// Softmax along the last dimension, applied in-place.
    pub fn softmax_inplace(&mut self) {
        let last_dim = self.last_dim();
        for row in self.data.chunks_mut(last_dim) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }

    /// Layer normalization applied in-place.
    pub fn layer_norm_inplace(&mut self, gamma: &Tensor, beta: &Tensor, eps: f32) {
        let last_dim = self.last_dim();
        debug_assert_eq!(gamma.numel(), last_dim);
        debug_assert_eq!(beta.numel(), last_dim);

        for row in self.data.chunks_mut(last_dim) {
            let mean: f32 = row.iter().sum::<f32>() / last_dim as f32;
            let var: f32 =
                row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / last_dim as f32;
            let std = (var + eps).sqrt();

            for (i, v) in row.iter_mut().enumerate() {
                *v = (*v - mean) / std * gamma.data[i] + beta.data[i];
            }
        }
    }

    /// Element-wise addition in-place: `self += other`.
    pub fn add_inplace(&mut self, other: &Tensor) {
        debug_assert_eq!(self.shape, other.shape, "add_inplace: shapes must match");
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
    }

    /// Element-wise addition of a 1D bias in-place (broadcasts along last dim).
    pub fn add_bias_inplace(&mut self, bias: &Tensor) {
        let last_dim = self.last_dim();
        debug_assert_eq!(
            bias.numel(),
            last_dim,
            "add_bias_inplace: bias length must match last dim"
        );
        for row in self.data.chunks_mut(last_dim) {
            for (v, b) in row.iter_mut().zip(&bias.data) {
                *v += b;
            }
        }
    }

    /// Scale all elements
    pub fn scale(&self, factor: f32) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| x * factor).collect();
        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Apply causal mask: set upper triangle to -inf
    /// Input shape: (..., seq_len, seq_len)
    pub fn causal_mask(&self) -> Self {
        let seq_len = self.last_dim();
        assert_eq!(self.shape[self.ndim() - 2], seq_len);

        let mut data = self.data.clone();
        let mat_size = seq_len * seq_len;

        for mat in data.chunks_mut(mat_size) {
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    mat[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }

        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Get a contiguous slice of the last dimension at given indices
    /// For a [B, T, D] tensor, get row [b, t, :] as a 1D tensor
    pub fn row(&self, indices: &[usize]) -> Self {
        assert_eq!(indices.len(), self.ndim() - 1);
        let last_dim = self.last_dim();
        let mut offset = 0;
        let mut stride = self.numel();
        for (i, &idx) in indices.iter().enumerate() {
            stride /= self.shape[i];
            offset += idx * stride;
        }
        let data = self.data[offset..offset + last_dim].to_vec();
        Self {
            data,
            shape: vec![last_dim],
        }
    }

    // -----------------------------------------------------------------
    // WASM simd128 accelerated implementations
    // -----------------------------------------------------------------

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn gelu_simd128(&self) -> Self {
        use std::arch::wasm32::*;

        let len = self.data.len();
        let mut out = vec![0.0f32; len];

        let c_vec = f32x4_splat(0.797_884_6_f32); // sqrt(2/pi)
        let half = f32x4_splat(0.5);
        let one = f32x4_splat(1.0);
        let coeff = f32x4_splat(0.044715);

        let chunks = len / 4;
        let remainder = len % 4;

        for i in 0..chunks {
            let offset = i * 4;
            let x = unsafe { v128_load(self.data[offset..].as_ptr() as *const v128) };

            // x^3
            let x2 = f32x4_mul(x, x);
            let x3 = f32x4_mul(x2, x);

            // inner = c * (x + 0.044715 * x^3)
            let scaled_x3 = f32x4_mul(coeff, x3);
            let sum = f32x4_add(x, scaled_x3);
            let inner = f32x4_mul(c_vec, sum);

            // tanh via element extraction (WASM SIMD has no native tanh)
            let inner_arr: [f32; 4] = [
                f32x4_extract_lane::<0>(inner),
                f32x4_extract_lane::<1>(inner),
                f32x4_extract_lane::<2>(inner),
                f32x4_extract_lane::<3>(inner),
            ];
            let tanh_vec = f32x4(
                inner_arr[0].tanh(),
                inner_arr[1].tanh(),
                inner_arr[2].tanh(),
                inner_arr[3].tanh(),
            );

            // 0.5 * x * (1 + tanh(inner))
            let one_plus_tanh = f32x4_add(one, tanh_vec);
            let result = f32x4_mul(half, f32x4_mul(x, one_plus_tanh));

            unsafe { v128_store(out[offset..].as_mut_ptr() as *mut v128, result) };
        }

        // Scalar remainder
        let scalar_start = chunks * 4;
        for i in 0..remainder {
            let x = self.data[scalar_start + i];
            let c = 0.797_884_6_f32;
            let inner = c * (x + 0.044715 * x * x * x);
            out[scalar_start + i] = 0.5 * x * (1.0 + inner.tanh());
        }

        Self {
            data: out,
            shape: self.shape.clone(),
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn softmax_simd128(&self) -> Self {
        use std::arch::wasm32::*;

        let last_dim = self.last_dim();
        let mut data = self.data.clone();
        let chunks = last_dim / 4;
        let remainder = last_dim % 4;

        for row in data.chunks_mut(last_dim) {
            // SIMD max-finding
            let mut max_vec = f32x4_splat(f32::NEG_INFINITY);
            for i in 0..chunks {
                let offset = i * 4;
                let v = unsafe { v128_load(row[offset..].as_ptr() as *const v128) };
                max_vec = f32x4_max(max_vec, v);
            }
            let mut max = f32x4_extract_lane::<0>(max_vec)
                .max(f32x4_extract_lane::<1>(max_vec))
                .max(f32x4_extract_lane::<2>(max_vec))
                .max(f32x4_extract_lane::<3>(max_vec));
            let scalar_start = chunks * 4;
            for i in 0..remainder {
                max = max.max(row[scalar_start + i]);
            }

            // Subtract max and exp (scalar — no WASM SIMD exp intrinsic)
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }

            // SIMD division by sum
            let inv_sum = f32x4_splat(1.0 / sum);
            for i in 0..chunks {
                let offset = i * 4;
                let v = unsafe { v128_load(row[offset..].as_ptr() as *const v128) };
                let result = f32x4_mul(v, inv_sum);
                unsafe { v128_store(row[offset..].as_mut_ptr() as *mut v128, result) };
            }
            let inv_sum_scalar = 1.0 / sum;
            for i in 0..remainder {
                row[scalar_start + i] *= inv_sum_scalar;
            }
        }

        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn layer_norm_simd128(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Self {
        use std::arch::wasm32::*;

        let last_dim = self.last_dim();
        let mut data = self.data.clone();
        let chunks = last_dim / 4;
        let remainder = last_dim % 4;

        for row in data.chunks_mut(last_dim) {
            // SIMD sum for mean
            let mut sum_vec = f32x4_splat(0.0);
            for i in 0..chunks {
                let offset = i * 4;
                let v = unsafe { v128_load(row[offset..].as_ptr() as *const v128) };
                sum_vec = f32x4_add(sum_vec, v);
            }
            let mut sum = f32x4_extract_lane::<0>(sum_vec)
                + f32x4_extract_lane::<1>(sum_vec)
                + f32x4_extract_lane::<2>(sum_vec)
                + f32x4_extract_lane::<3>(sum_vec);
            let scalar_start = chunks * 4;
            for i in 0..remainder {
                sum += row[scalar_start + i];
            }
            let mean = sum / last_dim as f32;

            // SIMD variance
            let mean_vec = f32x4_splat(mean);
            let mut var_vec = f32x4_splat(0.0);
            for i in 0..chunks {
                let offset = i * 4;
                let v = unsafe { v128_load(row[offset..].as_ptr() as *const v128) };
                let diff = f32x4_sub(v, mean_vec);
                var_vec = f32x4_add(var_vec, f32x4_mul(diff, diff));
            }
            let mut var = f32x4_extract_lane::<0>(var_vec)
                + f32x4_extract_lane::<1>(var_vec)
                + f32x4_extract_lane::<2>(var_vec)
                + f32x4_extract_lane::<3>(var_vec);
            for i in 0..remainder {
                let diff = row[scalar_start + i] - mean;
                var += diff * diff;
            }
            let std = (var / last_dim as f32 + eps).sqrt();

            // Normalize: (x - mean) / std * weight + bias
            let inv_std = f32x4_splat(1.0 / std);
            for i in 0..chunks {
                let offset = i * 4;
                let v = unsafe { v128_load(row[offset..].as_ptr() as *const v128) };
                let w = unsafe { v128_load(weight.data[offset..].as_ptr() as *const v128) };
                let b = unsafe { v128_load(bias.data[offset..].as_ptr() as *const v128) };
                let normed = f32x4_mul(f32x4_sub(v, mean_vec), inv_std);
                let result = f32x4_add(f32x4_mul(normed, w), b);
                unsafe { v128_store(row[offset..].as_mut_ptr() as *mut v128, result) };
            }
            for i in 0..remainder {
                let idx = scalar_start + i;
                row[idx] = (row[idx] - mean) / std * weight.data[idx] + bias.data[idx];
            }
        }

        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn add_simd128(&self, other: &Tensor) -> Self {
        use std::arch::wasm32::*;

        let len = self.data.len();
        let mut out = vec![0.0f32; len];
        let chunks = len / 4;
        let remainder = len % 4;

        for i in 0..chunks {
            let offset = i * 4;
            let a = unsafe { v128_load(self.data[offset..].as_ptr() as *const v128) };
            let b = unsafe { v128_load(other.data[offset..].as_ptr() as *const v128) };
            let result = f32x4_add(a, b);
            unsafe { v128_store(out[offset..].as_mut_ptr() as *mut v128, result) };
        }

        let scalar_start = chunks * 4;
        for i in 0..remainder {
            out[scalar_start + i] = self.data[scalar_start + i] + other.data[scalar_start + i];
        }

        Self {
            data: out,
            shape: self.shape.clone(),
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn add_bias_simd128(&self, bias: &Tensor) -> Self {
        use std::arch::wasm32::*;

        let cols = self.shape[1];
        let mut data = self.data.clone();
        let chunks = cols / 4;
        let remainder = cols % 4;

        for row in data.chunks_mut(cols) {
            for i in 0..chunks {
                let offset = i * 4;
                let v = unsafe { v128_load(row[offset..].as_ptr() as *const v128) };
                let b = unsafe { v128_load(bias.data[offset..].as_ptr() as *const v128) };
                let result = f32x4_add(v, b);
                unsafe { v128_store(row[offset..].as_mut_ptr() as *mut v128, result) };
            }
            let scalar_start = chunks * 4;
            for i in 0..remainder {
                row[scalar_start + i] += bias.data[scalar_start + i];
            }
        }

        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Concatenate along the second-to-last dimension (for growing decoder sequence)
    pub fn cat_seq(&self, other: &Tensor) -> Self {
        assert!(self.ndim() >= 2);
        assert_eq!(self.ndim(), other.ndim());
        let last = self.last_dim();
        assert_eq!(last, other.shape[other.shape.len() - 1]);

        // Batch dims must match
        for i in 0..self.ndim() - 2 {
            assert_eq!(self.shape[i], other.shape[i]);
        }

        let seq_a = self.shape[self.ndim() - 2];
        let seq_b = other.shape[other.ndim() - 2];
        let seq_out = seq_a + seq_b;

        let batch_size: usize = self.shape[..self.ndim() - 2].iter().product();
        let a_seq_stride = seq_a * last;
        let b_seq_stride = seq_b * last;
        let out_seq_stride = seq_out * last;

        let mut data = vec![0.0f32; batch_size * out_seq_stride];

        for b in 0..batch_size {
            let a_off = b * a_seq_stride;
            let b_off = b * b_seq_stride;
            let out_off = b * out_seq_stride;

            data[out_off..out_off + a_seq_stride]
                .copy_from_slice(&self.data[a_off..a_off + a_seq_stride]);
            data[out_off + a_seq_stride..out_off + out_seq_stride]
                .copy_from_slice(&other.data[b_off..b_off + b_seq_stride]);
        }

        let mut shape = self.shape.clone();
        shape[self.ndim() - 2] = seq_out;
        Self { data, shape }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        // [[1, 2], [3, 4]] x [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert!((c.data[0] - 19.0).abs() < 1e-5);
        assert!((c.data[1] - 22.0).abs() < 1e-5);
        assert!((c.data[2] - 43.0).abs() < 1e-5);
        assert!((c.data[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let s = t.softmax();
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(s.data[2] > s.data[1]);
        assert!(s.data[1] > s.data[0]);
    }

    #[test]
    fn test_layer_norm() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let w = Tensor::from_vec(vec![1.0, 1.0], &[2]);
        let b = Tensor::from_vec(vec![0.0, 0.0], &[2]);
        let ln = t.layer_norm(&w, &b, 1e-5);
        // Each row should be normalized to mean=0, var=1
        let row1_mean = (ln.data[0] + ln.data[1]) / 2.0;
        assert!(row1_mean.abs() < 1e-4);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tr = t.transpose_2d();
        assert_eq!(tr.shape, vec![3, 2]);
        assert_eq!(tr.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_gelu() {
        let t = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]);
        let g = t.gelu();
        assert!((g.data[0] - 0.0).abs() < 1e-4);
        assert!((g.data[1] - 0.8413).abs() < 0.01);
        assert!(g.data[2] < 0.0);
    }

    // ---------------------------------------------------------------
    // In-place operation tests
    // ---------------------------------------------------------------

    #[test]
    fn test_gelu_inplace_matches_gelu() {
        let t = Tensor::from_vec(vec![0.0, 1.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let expected = t.gelu();
        let mut inplace = t.clone();
        inplace.gelu_inplace();
        for (a, b) in expected.data.iter().zip(&inplace.data) {
            assert!((a - b).abs() < 1e-7, "gelu mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_softmax_inplace_matches_softmax() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0], &[2, 3]);
        let expected = t.softmax();
        let mut inplace = t.clone();
        inplace.softmax_inplace();
        for (a, b) in expected.data.iter().zip(&inplace.data) {
            assert!((a - b).abs() < 1e-7, "softmax mismatch: {a} vs {b}");
        }
        // Each row should still sum to 1
        let row0_sum: f32 = inplace.data[..3].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm_inplace_matches_layer_norm() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let w = Tensor::from_vec(vec![1.0, 0.5, 2.0], &[3]);
        let b = Tensor::from_vec(vec![0.1, -0.1, 0.0], &[3]);
        let expected = t.layer_norm(&w, &b, 1e-5);
        let mut inplace = t.clone();
        inplace.layer_norm_inplace(&w, &b, 1e-5);
        for (a, b) in expected.data.iter().zip(&inplace.data) {
            assert!((a - b).abs() < 1e-5, "layer_norm mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_add_inplace_matches_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let expected = a.add(&b);
        let mut inplace = a.clone();
        inplace.add_inplace(&b);
        assert_eq!(expected.data, inplace.data);
    }

    #[test]
    fn test_add_bias_inplace_matches_add_bias() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let bias = Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]);
        let expected = t.add_bias(&bias);
        let mut inplace = t.clone();
        inplace.add_bias_inplace(&bias);
        for (a, b) in expected.data.iter().zip(&inplace.data) {
            assert!((a - b).abs() < 1e-7, "add_bias mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_add_bias_inplace_nd() {
        // add_bias_inplace works on tensors with >2 dims by broadcasting along last dim
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
        let bias = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]);
        let mut inplace = t.clone();
        inplace.add_bias_inplace(&bias);
        assert!((inplace.data[0] - 11.0).abs() < 1e-7);
        assert!((inplace.data[1] - 22.0).abs() < 1e-7);
        assert!((inplace.data[2] - 33.0).abs() < 1e-7);
        assert!((inplace.data[3] - 14.0).abs() < 1e-7);
    }

    #[test]
    fn test_reshape_inplace_zero_copy() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let data_ptr = t.data.as_ptr();
        let mut t = t;
        t.reshape_inplace(&[3, 2]);
        // Shape changed
        assert_eq!(t.shape, vec![3, 2]);
        // Data pointer unchanged (zero-copy)
        assert_eq!(t.data.as_ptr(), data_ptr);
        // Data unchanged
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
