// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Parallel execution abstraction.
//!
//! When the `parallel` feature is enabled, uses rayon for parallel iteration.
//! Otherwise falls back to sequential iteration. This module provides a single
//! abstraction (`parallel_map`) so callers don't need `#[cfg]` blocks.

use crate::error::Result;

/// Worker-pool width the `parallel` feature will actually use — 1 when
/// the feature is off. Perf-dispatch signal only (e.g. the dedup
/// dispatch trades a parallel sort pass against sequential root-learn
/// rows); NEVER branch output-affecting logic on it.
#[cfg(feature = "parallel")]
pub fn effective_threads() -> usize {
    rayon::current_num_threads().max(1)
}
#[cfg(not(feature = "parallel"))]
pub fn effective_threads() -> usize {
    1
}

/// Map `f` over `0..n`, collecting results in index order.
///
/// Uses `rayon::par_iter` when the `parallel` feature is enabled,
/// otherwise uses sequential `(0..n).map(f).collect()`.
#[cfg(feature = "parallel")]
pub fn parallel_map<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    use rayon::prelude::*;
    (0..n).into_par_iter().map(f).collect()
}

/// Map `f` over `0..n`, collecting results in index order (sequential fallback).
#[cfg(not(feature = "parallel"))]
pub fn parallel_map<T, F>(n: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T,
{
    (0..n).map(f).collect()
}

/// Map `f` over `0..n`, collecting results in index order. Falls back to
/// serial execution when `n < min_parallel_n`, even with `parallel` enabled.
///
/// W44-89: small-image guard helper. Initially added to gate the AC-group
/// parallel xform fan-out on small inputs (W44-88 had reported a
/// `terminal.png` 1.75 MP regression at e3). Paired interleaved A/B
/// benchmark (`benchmarks/parallel_xform_guard_AB_2026-05-19.tsv`) showed
/// the regression is NOT reproducible on stable measurements — parallel
/// xform reliably beats serial at 35 AC groups (1.72× speedup, 15.8 ms
/// saved). Helper kept for future use if a real small-input regression
/// appears in another hot loop; not currently wired into any call site.
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn parallel_map_min<T, F>(n: usize, min_parallel_n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    if n < min_parallel_n {
        return (0..n).map(f).collect();
    }
    use rayon::prelude::*;
    (0..n).into_par_iter().map(f).collect()
}

/// Sequential fallback for [`parallel_map_min`].
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn parallel_map_min<T, F>(n: usize, _min_parallel_n: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T,
{
    (0..n).map(f).collect()
}

/// Map `f` over `0..n`, collecting results in index order, but force rayon
/// to keep at least `min_chunk` consecutive items per task. Useful when
/// per-item work is small (~1-50 μs) so the rayon scheduling overhead
/// (~1-3 μs per task) becomes a significant fraction of total work — by
/// keeping `min_chunk` items batched per task, the per-task overhead is
/// amortized over more useful work.
///
/// W44-phase3-B3-RAYON-1: introduced for a B2-ranked perf hypothesis — batch
/// the DC tree per-property fan-out in
/// [`crate::vardct::dc_tree_learn::find_best_split_variable`] /
/// `_incremental` to amortize rayon scheduling overhead. The hypothesis was
/// FALSIFIED by measurement (4-cell × 7-iter paired bench
/// `benchmarks/w44_phase3_b3_rayon_1_dc_tree_batch_2026-05-23.tsv`): at 8
/// threads, 14 fine-grained tasks already saturate the worker pool in two
/// waves (~7 per wave), while 2 chunked tasks of 7 leave 6 workers idle —
/// the parallelism loss (8-12 pp) overwhelms the per-task overhead recovery
/// (~3 pp). All 4 test cells regressed 4-11 % wall at 8 threads. The helper
/// is kept as future scaffolding (matching the W44-89 `parallel_map_min`
/// pattern — helper added, never wired, retained for future investigations
/// where the trade-off goes the other way; see the Section F row
/// "W44-phase3-B3-RAYON-1 RULED OUT" in `docs/LIBJXL_DIVERGENCES.md`).
///
/// `min_chunk = 0` and `min_chunk = 1` behave the same as `parallel_map`
/// (rayon's default is min length 1).
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn parallel_map_chunked<T, F>(n: usize, min_chunk: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    use rayon::prelude::*;
    let min_chunk = min_chunk.max(1);
    (0..n)
        .into_par_iter()
        .with_min_len(min_chunk)
        .map(f)
        .collect()
}

/// Sequential fallback for [`parallel_map_chunked`].
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn parallel_map_chunked<T, F>(n: usize, _min_chunk: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T,
{
    (0..n).map(f).collect()
}

/// Map `f` over `0..n` where `f` returns `Result<T>`, collecting results in index order.
///
/// Returns the first error encountered, or all results.
#[cfg(feature = "parallel")]
pub fn parallel_map_result<T, F>(n: usize, f: F) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(usize) -> Result<T> + Send + Sync,
{
    use rayon::prelude::*;
    (0..n).into_par_iter().map(f).collect()
}

/// Map `f` over `0..n` where `f` returns `Result<T>` (sequential fallback).
#[cfg(not(feature = "parallel"))]
pub fn parallel_map_result<T, F>(n: usize, f: F) -> Result<Vec<T>>
where
    F: Fn(usize) -> Result<T>,
{
    (0..n).map(f).collect()
}

/// Run two closures in parallel, returning both results. Falls through to
/// sequential execution when the `parallel` feature is disabled.
#[cfg(feature = "parallel")]
pub fn parallel_join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    rayon::join(a, b)
}

/// Sequential fallback for [`parallel_join`].
#[cfg(not(feature = "parallel"))]
pub fn parallel_join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA,
    B: FnOnce() -> RB,
{
    (a(), b())
}
