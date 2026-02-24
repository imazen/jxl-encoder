// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Parallel execution abstraction.
//!
//! When the `parallel` feature is enabled, uses rayon for parallel iteration.
//! Otherwise falls back to sequential iteration. This module provides a single
//! abstraction (`parallel_map`) so callers don't need `#[cfg]` blocks.

use crate::error::Result;

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

/// Fold over `0..n` with thread-local accumulators, then reduce.
///
/// Each thread gets its own accumulator (from `init()`), processes its share of
/// indices through `f(idx, &mut accumulator)`, then accumulators are merged via
/// `reduce(a, b) -> a`. This avoids per-element allocations while preserving
/// parallelism.
#[cfg(feature = "parallel")]
pub fn parallel_fold<T, Init, F, R>(n: usize, init: Init, f: F, reduce: R) -> T
where
    T: Send,
    Init: Fn() -> T + Send + Sync,
    F: Fn(usize, &mut T) + Send + Sync,
    R: Fn(T, T) -> T + Send + Sync,
{
    use rayon::prelude::*;
    (0..n)
        .into_par_iter()
        .fold(
            || init(),
            |mut acc, i| {
                f(i, &mut acc);
                acc
            },
        )
        .reduce(init, reduce)
}

/// Fold over `0..n` (sequential fallback).
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)] // Will be used by upcoming re-tokenization
pub fn parallel_fold<T, Init, F, R>(n: usize, init: Init, f: F, _reduce: R) -> T
where
    Init: Fn() -> T,
    F: Fn(usize, &mut T),
{
    let mut acc = init();
    for i in 0..n {
        f(i, &mut acc);
    }
    acc
}
