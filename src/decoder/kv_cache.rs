//! KV cache storage types for the Whisper self-attention decoder.
//!
//! Provides `KvStorage` (backing store variant) and `LayerKVCache` (per-layer
//! head-major key/value cache with Arc-based copy-on-write semantics).

use std::sync::Arc;

use crate::types::KvCacheDtype;

/// Internal storage variant for KV cache entries.
///
/// Either stores data as full-precision f32 (using `Arc<Vec<f32>>` for
/// copy-on-write semantics) or as half-precision f16 (using `Arc<Vec<half::f16>>`).
/// The half-precision path converts on read via `as_f32_slice`, writing into a
/// caller-supplied scratch buffer to avoid repeated heap allocation.
#[derive(Clone)]
pub(crate) enum KvStorage {
    /// Full-precision f32 backing store.
    F32(Arc<Vec<f32>>),
    /// Half-precision f16 backing store.
    F16(Arc<Vec<half::f16>>),
}

impl KvStorage {
    pub(crate) fn new_f32(size: usize) -> Self {
        Self::F32(Arc::new(vec![0.0f32; size]))
    }

    pub(crate) fn new_f16(size: usize) -> Self {
        Self::F16(Arc::new(vec![half::f16::ZERO; size]))
    }

    /// Append f32 data, writing head-major slices at `dst_offset` within each head's block.
    ///
    /// This is the low-level write used by `LayerKVCache::append`: data is already
    /// reordered into head-major layout by the caller before this call.
    pub(crate) fn write_slice_f32(&mut self, dst_offset: usize, src: &[f32]) {
        match self {
            Self::F32(arc) => {
                let vec = Arc::make_mut(arc);
                vec[dst_offset..dst_offset + src.len()].copy_from_slice(src);
            }
            Self::F16(arc) => {
                let vec = Arc::make_mut(arc);
                for (dst, &val) in vec[dst_offset..dst_offset + src.len()]
                    .iter_mut()
                    .zip(src.iter())
                {
                    *dst = half::f16::from_f32(val);
                }
            }
        }
    }

    /// Write head-major slices with pre-scaling: multiplies each value by `scale`
    /// before converting to f16. Only used for K in KvHalf mode to avoid overflow
    /// when computing QK^T attention scores.
    pub(crate) fn write_slice_f32_prescaled(&mut self, dst_offset: usize, src: &[f32], scale: f32) {
        match self {
            Self::F32(arc) => {
                let vec = Arc::make_mut(arc);
                for (dst, &val) in vec[dst_offset..dst_offset + src.len()]
                    .iter_mut()
                    .zip(src.iter())
                {
                    *dst = val * scale;
                }
            }
            Self::F16(arc) => {
                let vec = Arc::make_mut(arc);
                for (dst, &val) in vec[dst_offset..dst_offset + src.len()]
                    .iter_mut()
                    .zip(src.iter())
                {
                    *dst = half::f16::from_f32(val * scale);
                }
            }
        }
    }

    /// Get a f32 slice for a head-sized region, dequantizing f16 into the scratch buffer if needed.
    /// Returns a reference to either the original data or the scratch buffer.
    ///
    /// Used in tests; may also be useful for diagnostic/introspection tooling.
    #[allow(dead_code)]
    pub(crate) fn as_f32_slice_at<'a>(
        &'a self,
        offset: usize,
        len: usize,
        scratch: &'a mut Vec<f32>,
    ) -> &'a [f32] {
        match self {
            Self::F32(arc) => &arc[offset..offset + len],
            Self::F16(arc) => {
                scratch.clear();
                scratch.extend(arc[offset..offset + len].iter().map(|x| x.to_f32()));
                scratch.as_slice()
            }
        }
    }

    /// Returns true if both storages point to the same underlying Arc allocation (same pointer).
    /// Used in tests to verify copy-on-write behaviour.
    #[allow(dead_code)]
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::F32(a), Self::F32(b)) => Arc::ptr_eq(a, b),
            (Self::F16(a), Self::F16(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// Self-attention KV cache for one decoder layer.
/// Pre-allocated to avoid O(T^2) copies during autoregressive generation.
/// Layout: [n_head, capacity, head_dim] -- head-first, fixed capacity.
///
/// Uses `Arc<Vec<_>>` for copy-on-write semantics: beam search clones
/// share the same backing allocation until a beam actually writes, cutting
/// per-beam clone cost from ~2.8 MB to ~64 bytes (Arc refcount bump).
///
/// K and/or V can optionally be stored as f16 (see `KvCacheDtype`).
/// When K is stored as f16 it is pre-scaled by `1/sqrt(head_dim)` so that
/// attention scores stay within f16's representable range; callers must pass
/// `alpha = 1.0` (instead of the usual scale) to the QK^T sgemm.
#[derive(Clone)]
pub(crate) struct LayerKVCache {
    pub(crate) k: KvStorage,
    pub(crate) v: KvStorage,
    pub(crate) seq_len: usize,
    pub(crate) n_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) capacity: usize, // per-head stride in elements = capacity * head_dim
    /// True when K values are pre-scaled by `1/sqrt(head_dim)`.
    /// The QK^T sgemm must then use `alpha = 1.0` instead of `scale`.
    pub(crate) k_prescaled: bool,
    /// Storage precision for this cache (kept for introspection / testing).
    #[allow(dead_code)]
    pub(crate) dtype: KvCacheDtype,
}

impl LayerKVCache {
    /// Create a new KV cache with full f32 precision (default, lossless).
    pub(crate) fn new(n_head: usize, head_dim: usize, capacity: usize) -> Self {
        Self::new_with_dtype(n_head, head_dim, capacity, KvCacheDtype::F32)
    }

    /// Create a new KV cache with the specified storage precision.
    pub(crate) fn new_with_dtype(
        n_head: usize,
        head_dim: usize,
        capacity: usize,
        dtype: KvCacheDtype,
    ) -> Self {
        let total = n_head * capacity * head_dim;
        let (k, v, k_prescaled) = match dtype {
            KvCacheDtype::F32 => (KvStorage::new_f32(total), KvStorage::new_f32(total), false),
            KvCacheDtype::VHalf => (KvStorage::new_f32(total), KvStorage::new_f16(total), false),
            KvCacheDtype::KvHalf => (
                KvStorage::new_f16(total),
                KvStorage::new_f16(total),
                true, // K is pre-scaled when stored as f16 to avoid overflow
            ),
        };
        Self {
            k,
            v,
            seq_len: 0,
            n_head,
            head_dim,
            capacity,
            k_prescaled,
            dtype,
        }
    }

    /// Append K,V for `new_seq_len` new tokens.
    /// new_k/v: [new_seq_len, n_state] row-major  (n_state = n_head * head_dim)
    ///
    /// Performs the [seq, n_state] -> [n_head, capacity, head_dim] head-major
    /// transpose in a temporary buffer before writing to `KvStorage`, so the
    /// storage type (f32 or f16) only needs a flat write interface.
    pub(crate) fn append(&mut self, new_k: &[f32], new_v: &[f32], new_seq_len: usize) {
        let n_state = self.n_head * self.head_dim;
        let old = self.seq_len;
        debug_assert!(old + new_seq_len <= self.capacity, "KV cache overflow");

        // Pre-scale factor for K when using KvHalf to keep values in f16 range.
        let k_scale = if self.k_prescaled {
            (self.head_dim as f32).sqrt().recip()
        } else {
            1.0f32
        };

        // Reorder [new_seq_len, n_state] -> head-major, writing one head-block at a time.
        let mut tmp = vec![0.0f32; new_seq_len * self.head_dim];
        for h in 0..self.n_head {
            // Build tmp: head h's slice from the row-major new_k / new_v.
            for s in 0..new_seq_len {
                let src = s * n_state + h * self.head_dim;
                let dst = s * self.head_dim;
                tmp[dst..dst + self.head_dim].copy_from_slice(&new_k[src..src + self.head_dim]);
            }
            let dst_off = h * self.capacity * self.head_dim + old * self.head_dim;
            if self.k_prescaled {
                self.k.write_slice_f32_prescaled(dst_off, &tmp, k_scale);
            } else {
                self.k.write_slice_f32(dst_off, &tmp);
            }

            // V is never pre-scaled.
            for s in 0..new_seq_len {
                let src = s * n_state + h * self.head_dim;
                let dst = s * self.head_dim;
                tmp[dst..dst + self.head_dim].copy_from_slice(&new_v[src..src + self.head_dim]);
            }
            self.v.write_slice_f32(dst_off, &tmp);
        }
        self.seq_len += new_seq_len;
    }

    /// K slice for head h, only the populated portion: [seq_len, head_dim]
    /// Only available for F32 / VHalf storage (K is f32 in both cases).
    /// Used in tests.
    #[allow(dead_code)]
    pub(crate) fn k_head(&self, h: usize) -> &[f32] {
        let off = h * self.capacity * self.head_dim;
        let len = self.seq_len * self.head_dim;
        match &self.k {
            KvStorage::F32(arc) => &arc[off..off + len],
            KvStorage::F16(_) => {
                // KvHalf callers should use materialize_k_head instead.
                panic!("k_head() called on f16 K storage — use materialize_k_head()");
            }
        }
    }

    /// V slice for head h, only the populated portion: [seq_len, head_dim]
    /// Only available for F32 storage (K+V both f32).
    /// Used in tests.
    #[allow(dead_code)]
    pub(crate) fn v_head(&self, h: usize) -> &[f32] {
        let off = h * self.capacity * self.head_dim;
        let len = self.seq_len * self.head_dim;
        match &self.v {
            KvStorage::F32(arc) => &arc[off..off + len],
            KvStorage::F16(_) => {
                // VHalf / KvHalf callers should use materialize_v_head instead.
                panic!("v_head() called on f16 V storage — use materialize_v_head()");
            }
        }
    }

    /// Materialize K for head `h` as f32 into `scratch`, dequantizing if stored as f16.
    pub(crate) fn materialize_k_head(&self, h: usize, scratch: &mut Vec<f32>) {
        let off = h * self.capacity * self.head_dim;
        let len = self.seq_len * self.head_dim;
        match &self.k {
            KvStorage::F32(arc) => {
                scratch.clear();
                scratch.extend_from_slice(&arc[off..off + len]);
            }
            KvStorage::F16(arc) => {
                scratch.clear();
                scratch.extend(arc[off..off + len].iter().map(|x| x.to_f32()));
            }
        }
    }

    /// Materialize V for head `h` as f32 into `scratch`, dequantizing if stored as f16.
    pub(crate) fn materialize_v_head(&self, h: usize, scratch: &mut Vec<f32>) {
        let off = h * self.capacity * self.head_dim;
        let len = self.seq_len * self.head_dim;
        match &self.v {
            KvStorage::F32(arc) => {
                scratch.clear();
                scratch.extend_from_slice(&arc[off..off + len]);
            }
            KvStorage::F16(arc) => {
                scratch.clear();
                scratch.extend(arc[off..off + len].iter().map(|x| x.to_f32()));
            }
        }
    }

    /// Returns true if K is stored as f16 (requires `materialize_k_head`).
    /// Used in tests.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn k_is_f16(&self) -> bool {
        matches!(self.k, KvStorage::F16(_))
    }

    /// Returns true if V is stored as f16 (requires `materialize_v_head`).
    /// Used in tests.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn v_is_f16(&self) -> bool {
        matches!(self.v, KvStorage::F16(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_new() {
        let cache = LayerKVCache::new(2, 4, 8);
        assert_eq!(cache.seq_len, 0);
        assert_eq!(cache.n_head, 2);
        assert_eq!(cache.head_dim, 4);
        assert_eq!(cache.capacity, 8);
        assert!(cache.k_head(0).is_empty());
        assert!(cache.k_head(1).is_empty());
        assert!(cache.v_head(0).is_empty());
        assert!(cache.v_head(1).is_empty());
    }

    #[test]
    fn test_kv_cache_append() {
        let n_head = 2;
        let head_dim = 4;
        let capacity = 8;
        let n_state = n_head * head_dim;
        let mut cache = LayerKVCache::new(n_head, head_dim, capacity);

        let new_k: Vec<f32> = (1..=((n_state * 2) as i32)).map(|x| x as f32).collect();
        let new_v: Vec<f32> = (101..=(100 + (n_state * 2) as i32))
            .map(|x| x as f32)
            .collect();
        cache.append(&new_k, &new_v, 2);

        assert_eq!(cache.seq_len, 2);

        let k0 = cache.k_head(0);
        assert_eq!(k0.len(), 2 * head_dim);
        assert_eq!(k0[0..4], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(k0[4..8], [9.0, 10.0, 11.0, 12.0]);

        let k1 = cache.k_head(1);
        assert_eq!(k1.len(), 2 * head_dim);
        assert_eq!(k1[0..4], [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(k1[4..8], [13.0, 14.0, 15.0, 16.0]);

        let v0 = cache.v_head(0);
        assert_eq!(v0[0..4], [101.0, 102.0, 103.0, 104.0]);
        assert_eq!(v0[4..8], [109.0, 110.0, 111.0, 112.0]);
    }

    #[test]
    fn test_kv_cache_cow_clone() {
        let n_head = 2;
        let head_dim = 4;
        let capacity = 8;
        let n_state = n_head * head_dim;
        let mut cache = LayerKVCache::new(n_head, head_dim, capacity);

        let new_k: Vec<f32> = (1..=(n_state as i32)).map(|x| x as f32).collect();
        let new_v: Vec<f32> = (101..=(100 + n_state as i32)).map(|x| x as f32).collect();
        cache.append(&new_k, &new_v, 1);
        assert_eq!(cache.seq_len, 1);

        let mut clone = cache.clone();
        assert_eq!(clone.seq_len, 1);

        // After clone, both cache and clone share the same Arc allocations (COW).
        assert!(cache.k.ptr_eq(&clone.k));
        assert!(cache.v.ptr_eq(&clone.v));

        assert_eq!(cache.k_head(0), clone.k_head(0));
        assert_eq!(cache.v_head(0), clone.v_head(0));

        let new_k2: Vec<f32> = (21..=(20 + n_state as i32)).map(|x| x as f32).collect();
        let new_v2: Vec<f32> = (121..=(120 + n_state as i32)).map(|x| x as f32).collect();
        clone.append(&new_k2, &new_v2, 1);

        assert_eq!(clone.seq_len, 2);
        assert_eq!(cache.seq_len, 1);

        // After clone writes, it gets its own Arc allocation (COW triggered).
        assert!(!cache.k.ptr_eq(&clone.k));

        assert_eq!(cache.k_head(0), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_kv_cache_append_multiple() {
        let n_head = 2;
        let head_dim = 3;
        let capacity = 16;
        let n_state = n_head * head_dim;
        let mut cache = LayerKVCache::new(n_head, head_dim, capacity);

        for t in 0..4u32 {
            let base = (t * n_state as u32 + 1) as f32;
            let new_k: Vec<f32> = (0..n_state).map(|i| base + i as f32).collect();
            let new_v: Vec<f32> = (0..n_state).map(|i| base + 100.0 + i as f32).collect();
            cache.append(&new_k, &new_v, 1);
            assert_eq!(cache.seq_len, (t + 1) as usize);
        }

        assert_eq!(cache.seq_len, 4);

        let k0 = cache.k_head(0);
        assert_eq!(k0.len(), 4 * head_dim);
        assert_eq!(k0[0..3], [1.0, 2.0, 3.0]);
        assert_eq!(k0[3..6], [7.0, 8.0, 9.0]);
        assert_eq!(k0[6..9], [13.0, 14.0, 15.0]);
        assert_eq!(k0[9..12], [19.0, 20.0, 21.0]);

        let k1 = cache.k_head(1);
        assert_eq!(k1[0..3], [4.0, 5.0, 6.0]);
    }

    // ── KvStorage unit tests ─────────────────────────────────────────────────

    #[test]
    fn test_kv_storage_f32_passthrough() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let storage = KvStorage::F32(Arc::new(data.clone()));
        let mut scratch = Vec::new();
        let recovered = storage.as_f32_slice_at(0, data.len(), &mut scratch);
        assert_eq!(recovered, data.as_slice(), "f32 storage must be lossless");
    }

    #[test]
    fn test_kv_storage_f16_v_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let mut storage = KvStorage::new_f16(data.len());
        storage.write_slice_f32(0, &data);
        let mut scratch = Vec::new();
        let recovered = storage.as_f32_slice_at(0, data.len(), &mut scratch);
        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let err = (orig - rec).abs();
            let tol = half::f16::EPSILON.to_f32() * orig.abs() + 1e-3;
            assert!(
                err < tol,
                "f16 roundtrip error at {i}: orig={orig}, rec={rec}, err={err}, tol={tol}"
            );
        }
    }

    #[test]
    fn test_kv_storage_arc_clone_independent() {
        let data = vec![1.0f32, 2.0, 3.0];
        let mut storage = KvStorage::F32(Arc::new(data.clone()));
        let cloned = storage.clone();
        // Writing to storage triggers COW; clone should be unaffected.
        storage.write_slice_f32(0, &[10.0, 20.0, 30.0]);
        let mut scratch = Vec::new();
        let clone_data = cloned.as_f32_slice_at(0, 3, &mut scratch);
        assert_eq!(
            clone_data,
            &[1.0f32, 2.0, 3.0],
            "clone must be independent after write"
        );
    }

    #[test]
    fn test_kv_storage_ptr_eq() {
        let arc = Arc::new(vec![0.0f32; 8]);
        let s1 = KvStorage::F32(Arc::clone(&arc));
        let s2 = KvStorage::F32(Arc::clone(&arc));
        assert!(s1.ptr_eq(&s2), "same Arc should be ptr_eq");
        let s3 = KvStorage::F32(Arc::new(vec![0.0f32; 8]));
        assert!(!s1.ptr_eq(&s3), "different Arc should not be ptr_eq");
    }

    #[test]
    fn test_layer_kv_cache_f32_dtype() {
        let cache = LayerKVCache::new_with_dtype(2, 4, 8, KvCacheDtype::F32);
        assert_eq!(cache.dtype, KvCacheDtype::F32);
        assert!(!cache.k_prescaled);
        assert!(!cache.k_is_f16());
        assert!(!cache.v_is_f16());
    }

    #[test]
    fn test_layer_kv_cache_vhalf_dtype() {
        let cache = LayerKVCache::new_with_dtype(2, 4, 8, KvCacheDtype::VHalf);
        assert_eq!(cache.dtype, KvCacheDtype::VHalf);
        assert!(!cache.k_prescaled);
        assert!(!cache.k_is_f16(), "VHalf: K should be f32");
        assert!(cache.v_is_f16(), "VHalf: V should be f16");
    }

    #[test]
    fn test_layer_kv_cache_kvhalf_dtype() {
        let cache = LayerKVCache::new_with_dtype(2, 4, 8, KvCacheDtype::KvHalf);
        assert_eq!(cache.dtype, KvCacheDtype::KvHalf);
        assert!(cache.k_prescaled, "KvHalf: K should be pre-scaled");
        assert!(cache.k_is_f16(), "KvHalf: K should be f16");
        assert!(cache.v_is_f16(), "KvHalf: V should be f16");
    }

    #[test]
    fn test_layer_kv_cache_vhalf_roundtrip() {
        let n_head = 2;
        let head_dim = 4;
        let capacity = 8;
        let n_state = n_head * head_dim;
        let mut cache =
            LayerKVCache::new_with_dtype(n_head, head_dim, capacity, KvCacheDtype::VHalf);

        let new_k: Vec<f32> = (1..=((n_state * 2) as i32)).map(|x| x as f32).collect();
        let new_v: Vec<f32> = (101..=(100 + (n_state * 2) as i32))
            .map(|x| x as f32)
            .collect();
        cache.append(&new_k, &new_v, 2);
        assert_eq!(cache.seq_len, 2);

        // K should be exact (f32 storage)
        let k0 = cache.k_head(0);
        assert_eq!(k0.len(), 2 * head_dim);
        assert_eq!(k0[0..4], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(k0[4..8], [9.0, 10.0, 11.0, 12.0]);

        // V is f16 — materialize and check with tolerance
        let mut v_scratch = Vec::new();
        cache.materialize_v_head(0, &mut v_scratch);
        assert_eq!(v_scratch.len(), 2 * head_dim);
        for (orig, rec) in [101.0f32, 102.0, 103.0, 104.0, 109.0, 110.0, 111.0, 112.0]
            .iter()
            .zip(v_scratch.iter())
        {
            let err = (orig - rec).abs();
            assert!(err < 0.5, "VHalf V roundtrip error: orig={orig}, rec={rec}");
        }
    }
}
