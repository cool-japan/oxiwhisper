//! Cross-attention probability accumulator for word-level timestamp alignment.

/// Select which decoder layers contribute to the cross-attention capture.
///
/// Uses the upper half of decoder layers — a proven metadata-free heuristic
/// for models without explicit alignment-head metadata. Upper-middle layers
/// carry the clearest monotonic text↔audio alignment signal; early layers
/// attend more diffusely.
#[inline]
pub(crate) fn is_alignment_layer(layer: usize, n_layer: usize) -> bool {
    layer >= n_layer / 2
}

/// Accumulates head- and layer-averaged cross-attention rows during a single
/// `forward()` call.
///
/// Each `forward()` call covers `q_len` query positions (1 for incremental
/// decoding, `prompt_len` for the prefill). The caller:
///
/// 1. Creates a `CrossAttnCapture::new(q_len, enc_len)` before `forward()`.
/// 2. For each selected decoder layer, calls `layer_sink()` to get the scratch
///    slice, passes it to `scaled_dot_product_flat` as the `capture` argument,
///    then calls `commit_layer()`.
/// 3. After `forward()` returns, calls `finish()` to get the `[q_len * enc_len]`
///    head- and layer-averaged attention matrix.
pub(crate) struct CrossAttnCapture {
    /// Running sum over selected layers' head-averaged attention (shape `[q_len * enc_len]`).
    accum: Vec<f32>,
    /// Reusable per-layer scratch buffer (shape `[q_len * enc_len]`).
    layer_scratch: Vec<f32>,
    /// Number of layers committed so far this forward pass.
    n_contrib: usize,
    /// Number of query positions for this forward pass.
    pub(crate) q_len: usize,
    /// Encoder frame count (== n_frames for DTW).
    pub(crate) enc_len: usize,
}

impl CrossAttnCapture {
    /// Allocate accumulators for one forward pass with `q_len` query positions
    /// and `enc_len` encoder frames.
    pub(crate) fn new(q_len: usize, enc_len: usize) -> Self {
        let size = q_len * enc_len;
        Self {
            accum: vec![0.0f32; size],
            layer_scratch: vec![0.0f32; size],
            n_contrib: 0,
            q_len,
            enc_len,
        }
    }

    /// Zero the per-layer scratch and return a mutable slice to write into.
    ///
    /// Pass the returned slice to `scaled_dot_product_flat` as the `capture`
    /// argument.  After that call completes, call `commit_layer` to fold the
    /// result into the running accumulator.
    ///
    /// The borrow of `self.layer_scratch` ends when the returned `&mut [f32]` is
    /// dropped (moved into `scaled_dot_product_flat`), at which point `self` is
    /// free to be borrowed again for `commit_layer`.
    pub(crate) fn layer_sink(&mut self) -> &mut [f32] {
        // Verify the scratch buffer matches the declared shape (debug builds only).
        debug_assert_eq!(self.layer_scratch.len(), self.q_len * self.enc_len);
        for v in self.layer_scratch.iter_mut() {
            *v = 0.0;
        }
        &mut self.layer_scratch
    }

    /// Add the contents of the layer scratch into the running accumulator and
    /// increment the contribution count.
    ///
    /// Must be called after `scaled_dot_product_flat` has written the head-mean
    /// attention for one layer into the scratch returned by `layer_sink`.
    pub(crate) fn commit_layer(&mut self) {
        for (a, &s) in self.accum.iter_mut().zip(self.layer_scratch.iter()) {
            *a += s;
        }
        self.n_contrib += 1;
    }

    /// Return the `[q_len * enc_len]` head- and layer-averaged attention matrix.
    ///
    /// Divides the accumulator by the number of committed layers.  Returns a
    /// zero-filled vector if no layers were committed (degenerate model).
    pub(crate) fn finish(self) -> Vec<f32> {
        debug_assert_eq!(self.accum.len(), self.q_len * self.enc_len);
        let mut result = self.accum;
        if self.n_contrib > 0 {
            let inv = (self.n_contrib as f32).recip();
            for v in result.iter_mut() {
                *v *= inv;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_alignment_layer_upper_half() {
        // n_layer=4: layers 0,1 are lower half; 2,3 are upper half
        assert!(!is_alignment_layer(0, 4));
        assert!(!is_alignment_layer(1, 4));
        assert!(is_alignment_layer(2, 4));
        assert!(is_alignment_layer(3, 4));
    }

    #[test]
    fn test_is_alignment_layer_single_layer() {
        // n_layer=1, n_layer/2=0 → layer 0 >= 0 → true
        assert!(is_alignment_layer(0, 1));
    }

    #[test]
    fn test_new_initialises_to_zero() {
        let cap = CrossAttnCapture::new(2, 3);
        assert_eq!(cap.enc_len, 3);
        assert_eq!(cap.q_len, 2);
        let result = cap.finish();
        assert_eq!(result.len(), 6);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_single_layer_commit_equals_input() {
        let mut cap = CrossAttnCapture::new(1, 4);
        let scratch = cap.layer_sink();
        scratch.copy_from_slice(&[0.1, 0.4, 0.3, 0.2]);
        cap.commit_layer();
        let result = cap.finish();
        assert_eq!(result.len(), 4);
        for (a, &e) in result.iter().zip([0.1f32, 0.4, 0.3, 0.2].iter()) {
            assert!((a - e).abs() < 1e-6, "{a} != {e}");
        }
    }

    #[test]
    fn test_two_layer_commit_averages() {
        let mut cap = CrossAttnCapture::new(1, 2);
        // Layer 0: [0.8, 0.2]
        let s = cap.layer_sink();
        s.copy_from_slice(&[0.8, 0.2]);
        cap.commit_layer();
        // Layer 1: [0.4, 0.6]
        let s = cap.layer_sink();
        s.copy_from_slice(&[0.4, 0.6]);
        cap.commit_layer();
        let result = cap.finish();
        assert!((result[0] - 0.6).abs() < 1e-6, "mean[0] = {}", result[0]);
        assert!((result[1] - 0.4).abs() < 1e-6, "mean[1] = {}", result[1]);
    }

    #[test]
    fn test_no_commit_returns_zeros() {
        let cap = CrossAttnCapture::new(3, 5);
        let result = cap.finish();
        assert!(result.iter().all(|&v| v == 0.0));
    }
}
