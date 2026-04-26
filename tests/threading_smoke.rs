//! Smoke tests: verify the threading module compiles and behaves correctly
//! under all feature combinations.
//!
//! Run with: cargo nextest run --test threading_smoke
//!           cargo nextest run --test threading_smoke --features parallel
//! Both should pass with identical results.

/// `set_thread_count` must always return `Ok(())` (or a meaningful `Err` on
/// the second call with the `parallel` feature, which the rayon pool rejects).
/// Without the `parallel` feature it is always a no-op.
#[test]
fn test_set_thread_count_no_op() {
    // With `parallel` feature this initialises the global pool the first time;
    // further calls within the same process may return `Err` (rayon contract).
    // We accept both `Ok` and `Err` here — the important thing is it doesn't panic.
    let _result = oxiwhisper::threading::set_thread_count(2);
}

/// Verify `par_for_each` semantics by checking the threading behaviour
/// through the public-facing API.  Since `par_for_each` is `pub(crate)` we
/// test it indirectly: `set_thread_count` only exists on the public API and
/// serves as a smoke-check that the module compiled correctly.
#[test]
fn test_threading_module_accessible() {
    // If this compiles and runs, the `pub mod threading` declaration is correct.
    let result = oxiwhisper::threading::set_thread_count(1);
    // Without `parallel` feature this is always Ok.
    // With `parallel` feature this may be Err if the pool was already init'd.
    // Either way it must not panic.
    let _ = result;
}
