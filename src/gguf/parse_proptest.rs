//! Adversarial, property-based hardening tests for the GGUF binary parser.
//!
//! These treat the GGUF byte stream as fully attacker-controlled. The invariant
//! under test is: **the loader must never panic, never hang, and never allocate
//! unboundedly on malformed input — it must return `Err`** (or, in the rare case
//! that random bytes happen to form a valid header, `Ok`, which is still not a
//! panic).
//!
//! Two flavours of fuzzing are used:
//! 1. Fully random byte buffers (the loader must not panic on anything).
//! 2. Structure-aware corruption: a *valid* GGUF body is built, then exactly one
//!    field is replaced with an adversarial value (`u64::MAX` counts, string
//!    lengths past EOF, huge `n_dims`, overflowing `dims`, offsets past EOF,
//!    zero / non-power-of-two alignment, unknown dtypes, bad version).
//!
//! The `parse_proptest` module is a child of `crate::gguf::parse`, so it can
//! reach the crate-private `load_gguf_from_reader` entry point and the private
//! `MAX_SAFE_COUNT` guard directly — the reason these tests are in-crate rather
//! than in `tests/`.

use std::io::Cursor;

use proptest::prelude::*;

use super::{MAX_SAFE_COUNT, load_gguf_from_reader};
use crate::gguf::spec::align_offset;

// ── GGUF value-type / dtype discriminants used by the builders ─────────────────

const VT_U32: u32 = 4;
const VT_U64: u32 = 10;
const GGML_F32: u32 = 0;

/// Number of required Whisper hyperparameter KV pairs written by
/// [`write_required_hparams`].
const REQUIRED_KV_COUNT: u64 = 10;

/// GGML dtype discriminants that the spec recognises; anything else must be
/// rejected by `read_tensor_infos`.
const VALID_DTYPES: [u32; 8] = [0, 1, 2, 3, 6, 7, 8, 9];

// ── Low-level byte writers ─────────────────────────────────────────────────────

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Write a GGUF string: `u64` byte length followed by the UTF-8 bytes.
fn put_gguf_string(buf: &mut Vec<u8>, s: &str) {
    put_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

/// Write a `u32`-typed KV pair.
fn kv_u32(buf: &mut Vec<u8>, key: &str, v: u32) {
    put_gguf_string(buf, key);
    put_u32(buf, VT_U32);
    put_u32(buf, v);
}

/// Write a `u64`-typed KV pair.
fn kv_u64(buf: &mut Vec<u8>, key: &str, v: u64) {
    put_gguf_string(buf, key);
    put_u32(buf, VT_U64);
    put_u64(buf, v);
}

/// Emit the ten required Whisper hyperparameter KV pairs. `n_mels = 80` lets the
/// mel-filter resolver fall back to programmatic generation, so a body with
/// zero tensors loads successfully.
fn write_required_hparams(buf: &mut Vec<u8>) {
    kv_u32(buf, "whisper.vocab_size", 8);
    kv_u32(buf, "whisper.encoder.context_length", 4);
    kv_u32(buf, "whisper.encoder.embedding_length", 4);
    kv_u32(buf, "whisper.encoder.attention.head_count", 2);
    kv_u32(buf, "whisper.encoder.block_count", 1);
    kv_u32(buf, "whisper.decoder.context_length", 4);
    kv_u32(buf, "whisper.decoder.embedding_length", 4);
    kv_u32(buf, "whisper.decoder.attention.head_count", 2);
    kv_u32(buf, "whisper.decoder.block_count", 1);
    kv_u32(buf, "whisper.encoder.mels_count", 80);
}

/// Write one tensor-info record. `n_dims_field` and `dtype` are written verbatim
/// so callers can inject out-of-range values independently of `dims`.
fn write_tensor_info(
    buf: &mut Vec<u8>,
    name: &str,
    n_dims_field: u32,
    dims: &[u64],
    dtype: u32,
    offset: u64,
) {
    put_gguf_string(buf, name);
    put_u32(buf, n_dims_field);
    for &d in dims {
        put_u64(buf, d);
    }
    put_u32(buf, dtype);
    put_u64(buf, offset);
}

// ── Body builders ──────────────────────────────────────────────────────────────

/// A minimal, genuinely valid GGUF body (no magic prefix — the dispatcher in
/// `model.rs` consumes the magic, and `load_gguf_from_reader` begins at the
/// version field). Zero tensors, ten hyperparameters, default alignment.
fn valid_body() -> Vec<u8> {
    let mut buf = Vec::new();
    put_u32(&mut buf, 3); // version
    put_u64(&mut buf, 0); // tensor_count
    put_u64(&mut buf, REQUIRED_KV_COUNT); // metadata_kv_count
    write_required_hparams(&mut buf);
    buf
}

/// Header with caller-chosen `version`, `tensor_count`, and `metadata_kv_count`
/// count fields, followed by the real ten hyperparameter KV pairs and no tensor
/// infos. Used to exercise the header-level guards.
fn body_with_counts(version: u32, tensor_count: u64, kv_count: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    put_u32(&mut buf, version);
    put_u64(&mut buf, tensor_count);
    put_u64(&mut buf, kv_count);
    write_required_hparams(&mut buf);
    buf
}

/// Header declaring exactly one KV pair whose key length lies about the number
/// of bytes that follow (only a single byte is actually present).
fn body_bad_string_len(declared_len: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    put_u32(&mut buf, 3); // version
    put_u64(&mut buf, 0); // tensor_count
    put_u64(&mut buf, 1); // metadata_kv_count = 1
    put_u64(&mut buf, declared_len); // lying key length
    buf.push(b'x'); // ...but only one real byte
    buf
}

/// Zero-tensor body carrying a `general.alignment` KV set to `alignment`.
/// No tensor data is emitted, so even an absurd alignment only forces a seek.
fn body_zero_tensors_with_alignment(alignment: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    put_u32(&mut buf, 3); // version
    put_u64(&mut buf, 0); // tensor_count
    put_u64(&mut buf, REQUIRED_KV_COUNT + 1); // + general.alignment
    write_required_hparams(&mut buf);
    kv_u64(&mut buf, "general.alignment", alignment);
    buf
}

/// A body with exactly one tensor. Alignment is the default 32, and the tensor
/// data section is padded to the aligned base then filled with `data_len` zero
/// bytes — mirroring what `load_gguf_from_reader` computes for `data_base`.
#[allow(clippy::too_many_arguments)]
fn body_one_tensor(
    tensor_count_field: u64,
    name: &str,
    n_dims_field: u32,
    dims: &[u64],
    dtype: u32,
    offset: u64,
    data_len: usize,
) -> Vec<u8> {
    let mut buf = Vec::new();
    put_u32(&mut buf, 3); // version
    put_u64(&mut buf, tensor_count_field);
    put_u64(&mut buf, REQUIRED_KV_COUNT);
    write_required_hparams(&mut buf);
    write_tensor_info(&mut buf, name, n_dims_field, dims, dtype, offset);

    // Pad to the aligned data base (default alignment = 32), then append data.
    let base = align_offset(buf.len() as u64, 32);
    while (buf.len() as u64) < base {
        buf.push(0);
    }
    buf.extend(std::iter::repeat_n(0u8, data_len));
    buf
}

// ── Test harness helper ────────────────────────────────────────────────────────

/// Feed `body` to the loader. Returns `true` if the loader returned (either
/// `Ok` or `Err`); a panic would unwind and fail the enclosing proptest.
fn load(body: Vec<u8>) -> Result<(), ()> {
    let mut cursor = Cursor::new(body);
    match load_gguf_from_reader(&mut cursor) {
        Ok(_) => Ok(()),
        Err(_) => Err(()),
    }
}

fn is_err(body: Vec<u8>) -> bool {
    load(body).is_err()
}

// ── Sanity: the baseline really loads ──────────────────────────────────────────

#[test]
fn valid_baseline_loads_ok() {
    let mut cursor = Cursor::new(valid_body());
    let result = load_gguf_from_reader(&mut cursor);
    assert!(
        result.is_ok(),
        "the synthetic valid baseline must load so that corruption tests are \
         meaningful; got: {result:?}"
    );
}

// ── Regression tests for the defects this task uncovered ───────────────────────

/// Regression for the `TensorInfo::n_elements` overflow defect: dims whose
/// product exceeds `u64` previously panicked (overflow-checked builds) or
/// silently wrapped (release). Minimised failing case.
#[test]
fn regression_dims_product_overflow_is_err_not_panic() {
    // u64::MAX * 2 overflows u64.
    let body = body_one_tensor(1, "t", 2, &[u64::MAX, 2], GGML_F32, 0, 0);
    assert!(
        is_err(body),
        "a tensor whose dim product overflows u64 must be rejected, not panic"
    );
}

/// Regression for the unbounded `vec![0.0f32; n_elements]` allocation: dims that
/// fit in `u64` but describe ~40 GB of F32 data must be rejected against the
/// (tiny) file length instead of being allocated. Minimised failing case.
#[test]
fn regression_huge_dims_allocation_is_bounded() {
    // 100_000 * 100_000 = 1e10 elements → 4e10 bytes of F32 if allocated.
    let body = body_one_tensor(1, "t", 2, &[100_000, 100_000], GGML_F32, 0, 0);
    assert!(
        is_err(body),
        "a tensor claiming ~40 GB of data in a tiny file must be rejected \
         before allocation (OOM guard)"
    );
}

// ── Property: fully random bytes never panic ───────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Arbitrary byte buffers fed straight to the loader must never panic. A
    /// valid header arising by chance is astronomically unlikely; either result
    /// is acceptable as long as control returns normally.
    #[test]
    fn random_bytes_never_panic(data in proptest::collection::vec(any::<u8>(), 0..512)) {
        let mut cursor = Cursor::new(data);
        let _ = load_gguf_from_reader(&mut cursor);
    }
}

// ── Structure-aware corruption properties ──────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// An implausible `tensor_count` (past the `MAX_SAFE_COUNT` guard) must be
    /// rejected without allocating a `Vec` sized by the file.
    #[test]
    fn absurd_tensor_count_is_err(tc in (MAX_SAFE_COUNT + 1)..=u64::MAX) {
        prop_assert!(is_err(body_with_counts(3, tc, REQUIRED_KV_COUNT)));
    }

    /// An implausible `metadata_kv_count` must likewise be rejected.
    #[test]
    fn absurd_kv_count_is_err(kvc in (MAX_SAFE_COUNT + 1)..=u64::MAX) {
        prop_assert!(is_err(body_with_counts(3, 0, kvc)));
    }

    /// Any version other than 3 must be rejected.
    #[test]
    fn wrong_version_is_err(ver in any::<u32>().prop_filter("v3 is valid", |v| *v != 3)) {
        prop_assert!(is_err(body_with_counts(ver, 0, REQUIRED_KV_COUNT)));
    }

    /// A string length that overruns the remaining buffer (or trips the
    /// `MAX_SAFE_COUNT` guard) must yield `Err`, never a multi-gigabyte alloc.
    #[test]
    fn string_len_past_eof_is_err(declared in 2u64..=u64::MAX) {
        prop_assert!(is_err(body_bad_string_len(declared)));
    }

    /// `n_dims` beyond the supported maximum of 4 must be rejected before any
    /// per-dimension reads.
    #[test]
    fn huge_n_dims_is_err(n_dims in 5u32..=u32::MAX) {
        // No actual dims are written; the guard fires on the count alone.
        prop_assert!(is_err(body_one_tensor(1, "t", n_dims, &[], GGML_F32, 0, 0)));
    }

    /// An unknown dtype discriminant must be rejected by `read_tensor_infos`.
    #[test]
    fn unknown_dtype_is_err(
        dtype in any::<u32>().prop_filter(
            "exclude the 8 valid dtypes",
            |d| !VALID_DTYPES.contains(d),
        ),
    ) {
        prop_assert!(is_err(body_one_tensor(1, "t", 1, &[4], dtype, 0, 0)));
    }

    /// Two large dimensions whose product overflows `u64` must be rejected,
    /// exercising the checked `n_elements` path across a wide value range.
    #[test]
    fn overflowing_dims_is_err(
        a in (1u64 << 33)..=u64::MAX,
        b in (1u64 << 33)..=u64::MAX,
    ) {
        // a, b >= 2^33 ⇒ a*b >= 2^66 > u64::MAX ⇒ checked_mul returns None.
        prop_assert!(is_err(body_one_tensor(1, "t", 2, &[a, b], GGML_F32, 0, 0)));
    }

    /// A single dimension large enough to describe more bytes than the file
    /// contains must be rejected by the allocation guard (bounded, no OOM).
    #[test]
    fn huge_single_dim_is_bounded(n in 1_000_000u64..=u64::MAX) {
        prop_assert!(is_err(body_one_tensor(1, "t", 1, &[n], GGML_F32, 0, 0)));
    }

    /// A tensor offset that points past EOF must be rejected before the read.
    #[test]
    fn offset_past_eof_is_err(offset in (1u64 << 20)..=u64::MAX) {
        // Small, valid dims (4 F32 elements); only the offset is adversarial.
        prop_assert!(is_err(body_one_tensor(1, "t", 1, &[4], GGML_F32, offset, 0)));
    }

    /// Arbitrary `general.alignment` values — including 0, non-powers-of-two,
    /// and values near `u64::MAX` — must never panic (with zero tensors the
    /// loader merely seeks; the result is `Ok`).
    #[test]
    fn arbitrary_alignment_never_panics(alignment in any::<u64>()) {
        let mut cursor = Cursor::new(body_zero_tensors_with_alignment(alignment));
        let _ = load_gguf_from_reader(&mut cursor);
    }
}
