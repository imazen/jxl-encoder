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
