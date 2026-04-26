//! Thread-count management and parallel iteration shim.
//!
//! Provides a uniform `par_iter_heads` helper that iterates over a slice of
//! per-head scratch buffers in parallel when the `parallel` feature is enabled,
//! or sequentially on single-threaded / WASM builds.
//!
//! The `set_thread_count` function configures the global rayon thread pool.
//! It must be called at most once, before any parallel operation.  On
//! non-parallel builds it is a no-op and always returns `Ok(())`.

/// Set the global rayon thread pool size.
///
/// Effective only when the `parallel` feature is enabled.  Must be called once,
/// before any parallel operation, typically at application startup.
///
/// On non-parallel builds this is a no-op that always returns `Ok(())`.
///
/// # Errors
///
/// Returns `Err` if the rayon global thread pool has already been initialized
/// (i.e. this function was called more than once, or rayon was already used).
#[cfg(feature = "parallel")]
pub fn set_thread_count(n: usize) -> Result<(), String> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| format!("Failed to set thread count: {e}"))
}

#[cfg(not(feature = "parallel"))]
/// Set the global thread count (no-op on non-parallel builds; always returns `Ok(())`).
pub fn set_thread_count(_n: usize) -> Result<(), String> {
    Ok(())
}
