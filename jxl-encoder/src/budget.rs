// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Allocation budget tracker for the encoder.
//!
//! [`MemoryBudget`] is a single-encode lifetime cap on the cumulative
//! peak bytes the encoder has reserved against a configured ceiling.
//! Hot allocation sites reserve their expected size *before* allocating
//! and hold an RAII [`BudgetGuard`]; when the guard drops the memory is
//! released back to the budget. The budget is opt-in: callers that don't
//! configure one pay essentially nothing — a single cold predicted-not-
//! taken branch per allocation site.
//!
//! The cap comes from [`Limits::max_memory_bytes`] when the caller sets
//! one, or from the path-aware default [`Limits::default_max_memory_bytes`]
//! otherwise — [`Limits::DEFAULT_MAX_MEMORY_BYTES`] (4 GiB) on the lossy
//! path, [`Limits::DEFAULT_MAX_MEMORY_BYTES_LOSSLESS`] (8 GiB) on the
//! lossless path.
//!
//! ## Coverage
//!
//! The tracker is best-effort: it does not intercept every heap
//! allocation the encoder performs (small per-block scratch, internal
//! hashmaps, the entropy-coder bit buffer, etc., are unaccounted). It
//! DOES cover the large dimension-driven buffers — XYB planes, padded
//! scratch, group buffers, modular channel buffers — which dominate the
//! encoder's working set and are the attacker-controlled axis for DoS
//! by oversized input.
//!
//! ## Zero overhead when unbounded
//!
//! Threading happens as `Option<&MemoryBudget>`. The compiler eliminates
//! the `None` branch at every call site; the `Some` branch is a cold
//! relaxed atomic CAS. There is no allocation, no virtual dispatch, and
//! no extra fields on the encoder structs paid by callers who haven't
//! configured a cap.
//!
//! [`Limits::max_memory_bytes`]: crate::api::Limits::max_memory_bytes
//! [`Limits::DEFAULT_MAX_MEMORY_BYTES`]: crate::api::Limits::DEFAULT_MAX_MEMORY_BYTES

use crate::error::{Error, Result};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// Tracks cumulative reserved bytes against a hard cap.
///
/// Construct one of these at the start of an encode (or use
/// [`MemoryBudget::unbounded`] to disable). Pass the resulting `Arc` —
/// or, on hot paths, an `Option<&MemoryBudget>` — through to allocation
/// sites. Each site calls [`MemoryBudget::reserve`] to obtain a
/// [`BudgetGuard`] held alongside the buffer; when the guard drops the
/// reservation is released, freeing peak budget for sibling allocations.
#[derive(Debug)]
pub(crate) struct MemoryBudget {
    cap: u64,
    used: AtomicU64,
    peak: AtomicU64,
    /// Runtime allocation policy for the dimension-driven buffers this budget
    /// guards: `false` uses `vec![v; n]` (one `calloc`, faster), `true` uses
    /// `try_reserve_exact` (graceful OOM on untrusted sizes). See
    /// [`try_alloc_zeroed_permanent`] and [`crate::api::Limits::fallible_alloc`].
    fallible: bool,
}

impl MemoryBudget {
    /// Construct a budget with `cap` bytes available (infallible-alloc policy).
    ///
    /// Test-only convenience: production code constructs budgets via
    /// [`Self::with_alloc_policy`] so the runtime fallible-alloc policy
    /// (`Limits::fallible_alloc`) is always threaded explicitly.
    #[cfg(test)]
    pub fn new(cap: u64) -> Arc<Self> {
        Self::with_alloc_policy(cap, false)
    }

    /// Construct a budget with `cap` bytes available and an explicit
    /// fallible-allocation policy (`true` = `try_reserve`, `false` = `vec!`).
    pub fn with_alloc_policy(cap: u64, fallible: bool) -> Arc<Self> {
        Arc::new(Self {
            cap,
            used: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            fallible,
        })
    }

    /// Whether dimension-driven buffers should be allocated fallibly
    /// (`try_reserve`) rather than via the faster infallible `vec!` path.
    pub fn is_fallible(&self) -> bool {
        self.fallible
    }

    /// Construct a budget that never denies a reservation.
    ///
    /// Equivalent to `MemoryBudget::new(u64::MAX)` but more legible at
    /// call sites. Note: this still creates an `Arc` and tracks usage —
    /// for true zero-overhead, thread `Option<&MemoryBudget>` and pass
    /// `None`.
    #[cfg(test)]
    pub fn unbounded() -> Arc<Self> {
        Self::new(u64::MAX)
    }

    /// The cap configured for this budget.
    #[allow(dead_code)] // exposed for diagnostics / future stats wiring
    pub fn cap(&self) -> u64 {
        self.cap
    }

    /// Currently-reserved bytes.
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// Highest cumulative usage seen during this budget's lifetime.
    #[allow(dead_code)] // exposed for diagnostics / future stats wiring
    pub fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    /// Attempt to reserve `bytes` against the cap.
    ///
    /// On success returns a [`BudgetGuard`] whose `Drop` releases the
    /// bytes. Hold the guard for as long as the underlying buffer is
    /// alive; usually that means storing it next to (or inside) the
    /// `Vec` or letting it bind to a function-scope `let _g = ...;`.
    ///
    /// Returns [`Error::AllocationLimit`] if the reservation would push
    /// usage past `cap`, or [`Error::DimensionOverflow`]-flavored
    /// `Error::InvalidInput` if `current + bytes` overflows `u64` (which
    /// only happens with deliberately malformed input).
    pub fn reserve(self: &Arc<Self>, bytes: u64) -> Result<BudgetGuard> {
        self.try_reserve_raw(bytes)?;
        Ok(BudgetGuard {
            bytes,
            budget: Some(Arc::clone(self)),
        })
    }

    /// Reserve via an `Option<&MemoryBudget>`. Returns an inert guard
    /// when `budget` is `None`, side-stepping the `Arc` clone.
    ///
    /// Prefer this at hot call sites that already thread the budget as
    /// an `Option`. The `None` branch is unconditional and predicts
    /// not-taken in the unbounded case.
    pub fn reserve_opt(budget: Option<&Arc<MemoryBudget>>, bytes: u64) -> Result<BudgetGuard> {
        match budget {
            Some(b) => b.reserve(bytes),
            None => Ok(BudgetGuard {
                bytes: 0,
                budget: None,
            }),
        }
    }

    /// Reserve `bytes` permanently — no `BudgetGuard`, no release.
    ///
    /// Use this for buffers that live for the full encode (XYB planes,
    /// padded scratch, group buffers). The encoder's `MemoryBudget` is
    /// dropped at end-of-encode regardless, so "leaking" the
    /// reservation is structurally fine and avoids threading guards
    /// through every consumer.
    ///
    /// For sibling scratch allocations whose peaks don't overlap,
    /// prefer [`MemoryBudget::reserve`] + RAII so the encoder's high-
    /// water mark stays accurate.
    pub fn reserve_permanent(&self, bytes: u64) -> Result<()> {
        self.try_reserve_raw(bytes)
    }

    /// `reserve_permanent` via an `Option<&MemoryBudget>`. No-op when
    /// `budget` is `None`.
    pub fn reserve_permanent_opt(budget: Option<&Arc<MemoryBudget>>, bytes: u64) -> Result<()> {
        match budget {
            Some(b) => b.reserve_permanent(bytes),
            None => Ok(()),
        }
    }

    fn try_reserve_raw(&self, bytes: u64) -> Result<()> {
        // CAS loop: load current, check fit, store new. Bounded — under
        // contention each retry loads a higher `used` than the previous,
        // so we either succeed or hit the cap and bail.
        let mut current = self.used.load(Ordering::Relaxed);
        loop {
            let new_used = current.checked_add(bytes).ok_or(Error::AllocationLimit {
                requested: bytes,
                used: current,
                cap: self.cap,
            })?;
            if new_used > self.cap {
                return Err(Error::AllocationLimit {
                    requested: bytes,
                    used: current,
                    cap: self.cap,
                });
            }
            match self.used.compare_exchange_weak(
                current,
                new_used,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Bump the high-water mark. We don't need this to
                    // be perfectly synchronized with `used`; a relaxed
                    // best-effort fetch_max is fine for diagnostics.
                    self.peak.fetch_max(new_used, Ordering::Relaxed);
                    return Ok(());
                }
                Err(actual) => current = actual,
            }
        }
    }

    fn release(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        // Saturating subtract — mismatched reserve/release pairs (a bug)
        // won't underflow but will result in the budget over-reporting
        // available space.
        self.used
            .fetch_sub(bytes.min(self.used()), Ordering::Relaxed);
    }
}

/// RAII guard for an outstanding budget reservation.
///
/// Drop releases the bytes back to the budget. Move the guard around
/// freely — only the final drop is observable. Forgetting (`mem::forget`)
/// the guard leaks the reservation, which is benign: the encode is
/// single-shot, so the budget is dropped wholesale on encode exit.
///
/// Most encoder buffers live for the entire encode and use
/// [`MemoryBudget::reserve_permanent`] (no guard). Use this RAII path
/// for sibling scratch allocations whose peaks don't overlap, where
/// returning bytes between phases keeps the high-water mark accurate
/// and lets a tighter cap survive the encode.
#[must_use = "BudgetGuard releases its reservation on drop; binding to `_` defeats the budget"]
#[derive(Debug)]
#[allow(dead_code)] // Currently the encoder uses reserve_permanent throughout;
// the RAII guard infrastructure is exercised by unit tests and reserved
// for future scratch-allocation tracking (e.g., per-tile butteraugli
// reconstruction buffers).
pub(crate) struct BudgetGuard {
    bytes: u64,
    budget: Option<Arc<MemoryBudget>>,
}

#[allow(dead_code)] // see BudgetGuard above
impl BudgetGuard {
    /// An inert guard that owns no reservation. Useful as a default
    /// when a function returns `Vec + Guard` and the caller hasn't
    /// configured a budget yet.
    pub fn inert() -> Self {
        Self {
            bytes: 0,
            budget: None,
        }
    }

    /// Bytes this guard will release on drop.
    #[cfg(test)]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for BudgetGuard {
    fn drop(&mut self) {
        if let Some(b) = self.budget.as_ref() {
            b.release(self.bytes);
        }
    }
}

/// Compute `len * size_of::<T>()` as u64, returning an
/// [`Error::AllocationLimit`] (with `cap=0` to flag overflow) if the
/// multiplication would not fit, or if the byte count exceeds what can
/// be addressed in `usize` on this platform (for example
/// `len = usize::MAX, size = 4` on a 32-bit host: the multiply doesn't
/// overflow `u64` but the allocation cannot be made).
fn elem_bytes<T>(len: usize) -> Result<u64> {
    let bytes = (len as u64)
        .checked_mul(core::mem::size_of::<T>() as u64)
        .ok_or(Error::AllocationLimit {
            requested: u64::MAX,
            used: 0,
            cap: 0,
        })?;
    if bytes > usize::MAX as u64 {
        return Err(Error::AllocationLimit {
            requested: bytes,
            used: 0,
            cap: 0,
        });
    }
    Ok(bytes)
}

/// Reserve and allocate a `Vec<T: Default + Clone>` of `len` elements.
///
/// Returns the vec paired with a [`BudgetGuard`]; both must outlive the
/// allocation as a unit. Pass `budget = None` for zero-overhead.
#[allow(dead_code)] // RAII helper — see BudgetGuard.
pub(crate) fn try_alloc_vec<T: Default + Clone>(
    budget: Option<&Arc<MemoryBudget>>,
    len: usize,
) -> Result<(Vec<T>, BudgetGuard)> {
    let bytes = elem_bytes::<T>(len)?;
    let guard = MemoryBudget::reserve_opt(budget, bytes)?;
    Ok((vec![T::default(); len], guard))
}

/// Reserve and allocate a zero-filled `Vec<f32>` of `len` elements.
///
/// Specialization of [`try_alloc_vec`] for the common XYB plane case;
/// LLVM lowers `vec![0.0f32; n]` to a single `calloc`, which is faster
/// than the generic `T::default()` path.
#[allow(dead_code)] // RAII helper — see BudgetGuard.
pub(crate) fn try_alloc_vec_f32(
    budget: Option<&Arc<MemoryBudget>>,
    len: usize,
) -> Result<(Vec<f32>, BudgetGuard)> {
    let bytes = elem_bytes::<f32>(len)?;
    let guard = MemoryBudget::reserve_opt(budget, bytes)?;
    Ok((vec![0.0f32; len], guard))
}

/// Reserve and produce a `jxl_simd::vec_f32_dirty` of `len` elements.
///
/// The dirty variant skips the zero-fill — safe for buffers that the
/// caller is about to fully overwrite. Callers should still document
/// the overwrite invariant at the call site.
#[allow(dead_code)] // RAII helper — see BudgetGuard.
pub(crate) fn try_alloc_vec_f32_dirty(
    budget: Option<&Arc<MemoryBudget>>,
    len: usize,
) -> Result<(Vec<f32>, BudgetGuard)> {
    let bytes = elem_bytes::<f32>(len)?;
    let guard = MemoryBudget::reserve_opt(budget, bytes)?;
    Ok((jxl_simd::vec_f32_dirty(len), guard))
}

/// Reserve permanently and allocate a zero-filled `Vec<f32>`. Suitable
/// for whole-encode buffers (XYB planes, padded scratch) where there's
/// no point pretending the bytes will be released before the budget
/// itself is dropped.
#[allow(dead_code)] // Currently the encoder uses _dirty_permanent for all XYB
// allocations; this is here for future zero-init plane sites.
pub(crate) fn try_alloc_vec_f32_permanent(
    budget: Option<&Arc<MemoryBudget>>,
    len: usize,
) -> Result<Vec<f32>> {
    let bytes = elem_bytes::<f32>(len)?;
    MemoryBudget::reserve_permanent_opt(budget, bytes)?;
    Ok(vec![0.0f32; len])
}

/// Reserve permanently and allocate a `vec_f32_dirty`.
pub(crate) fn try_alloc_vec_f32_dirty_permanent(
    budget: Option<&Arc<MemoryBudget>>,
    len: usize,
) -> Result<Vec<f32>> {
    let bytes = elem_bytes::<f32>(len)?;
    MemoryBudget::reserve_permanent_opt(budget, bytes)?;
    Ok(jxl_simd::vec_f32_dirty(len))
}

/// Create an empty `Vec<T>` with reserved capacity for `cap` elements, honoring
/// a runtime fallible-allocation policy: `Vec::with_capacity` (infallible,
/// fast) when `fallible` is false, `try_reserve` (returns
/// [`Error::OutOfMemory`] instead of aborting) when true. Capacity sizing only
/// — reserves nothing against a [`MemoryBudget`]; pair with an explicit
/// `reserve_permanent_opt` for the budget accounting.
pub(crate) fn vec_with_capacity_fallible<T>(fallible: bool, cap: usize) -> Result<Vec<T>> {
    if fallible {
        let mut v: Vec<T> = Vec::new();
        v.try_reserve(cap)?;
        Ok(v)
    } else {
        Ok(Vec::with_capacity(cap))
    }
}

/// Reserve `len * size_of::<T>()` permanently against `budget`, then allocate a
/// zeroed `Vec<T>`, honoring the budget's **runtime** fallible-allocation
/// policy ([`MemoryBudget::is_fallible`]):
/// - infallible (default): `vec![T::default(); len]` — LLVM lowers the zeroed
///   case to a single `calloc`, the faster path for trusted sizes;
/// - fallible: `try_reserve_exact` + `resize` — returns
///   [`Error::OutOfMemory`] instead of aborting if the (untrusted-derived)
///   size cannot be allocated.
///
/// `None` budget reserves nothing and uses the infallible path. Used for the
/// JPEG-transcode coefficient buffers and the modular channel buffers, both
/// sized from untrusted dimensions.
pub(crate) fn try_alloc_zeroed_permanent<T: Copy + Default>(
    budget: Option<&Arc<MemoryBudget>>,
    len: usize,
) -> Result<Vec<T>> {
    let bytes = elem_bytes::<T>(len)?;
    MemoryBudget::reserve_permanent_opt(budget, bytes)?;
    if budget.is_some_and(|b| b.is_fallible()) {
        let mut v: Vec<T> = Vec::new();
        v.try_reserve_exact(len)?;
        v.resize(len, T::default());
        Ok(v)
    } else {
        Ok(vec![T::default(); len])
    }
}

/// Convenience macro: `try_alloc_plane_f32!(budget, w, h)` reserves and
/// allocates a `width * height` `f32` plane, propagating allocation
/// errors with `?`.
///
/// Expands to a `(Vec<f32>, BudgetGuard)` tuple. Pair the guard with
/// the vec for the lifetime of the allocation.
#[macro_export]
#[doc(hidden)]
macro_rules! try_alloc_plane_f32 {
    ($budget:expr, $width:expr, $height:expr) => {{
        let __w: usize = $width;
        let __h: usize = $height;
        let __n = __w
            .checked_mul(__h)
            .ok_or_else(|| $crate::error::Error::DimensionOverflow {
                width: __w,
                height: __h,
                channels: 1,
            })?;
        $crate::budget::try_alloc_vec_f32($budget, __n)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn under_cap_succeeds() {
        let b = MemoryBudget::new(1024);
        let _g = b.reserve(512).unwrap();
        assert_eq!(b.used(), 512);
    }

    #[test]
    fn over_cap_fails_with_allocation_limit() {
        let b = MemoryBudget::new(1024);
        let err = b.reserve(2048).unwrap_err();
        match err {
            Error::AllocationLimit {
                requested,
                used,
                cap,
            } => {
                assert_eq!(requested, 2048);
                assert_eq!(used, 0);
                assert_eq!(cap, 1024);
            }
            other => panic!("expected AllocationLimit, got {other:?}"),
        }
        assert_eq!(b.used(), 0);
    }

    #[test]
    fn drop_releases_bytes() {
        let b = MemoryBudget::new(1024);
        {
            let _g = b.reserve(500).unwrap();
            assert_eq!(b.used(), 500);
        }
        assert_eq!(b.used(), 0);
    }

    #[test]
    fn nested_guards_release_in_order() {
        let b = MemoryBudget::new(1024);
        let g1 = b.reserve(400).unwrap();
        let g2 = b.reserve(400).unwrap();
        assert_eq!(b.used(), 800);
        drop(g2);
        assert_eq!(b.used(), 400);
        drop(g1);
        assert_eq!(b.used(), 0);
    }

    #[test]
    fn peak_records_high_water_mark() {
        let b = MemoryBudget::new(2048);
        {
            let _g1 = b.reserve(800).unwrap();
            let _g2 = b.reserve(700).unwrap();
        }
        assert_eq!(b.used(), 0);
        assert_eq!(b.peak(), 1500);
        let _g3 = b.reserve(900).unwrap();
        assert_eq!(b.peak(), 1500); // 900 < 1500
    }

    #[test]
    fn cumulative_with_dropped_guards() {
        let b = MemoryBudget::new(1024);
        {
            let _g = b.reserve(800).unwrap();
        }
        // After drop, the next reservation should fit.
        let _g = b.reserve(800).unwrap();
        assert_eq!(b.used(), 800);
    }

    #[test]
    fn unbounded_never_denies() {
        let b = MemoryBudget::unbounded();
        let _g1 = b.reserve(u64::MAX / 2).unwrap();
        let _g2 = b.reserve(u64::MAX / 2).unwrap();
        // Pushing past u64::MAX overflows, surfaced as AllocationLimit.
        assert!(b.reserve(2).is_err());
    }

    #[test]
    fn reserve_opt_none_is_inert() {
        let g = MemoryBudget::reserve_opt(None, 1 << 60).unwrap();
        assert_eq!(g.bytes(), 0);
    }

    #[test]
    fn reserve_opt_some_tracks() {
        let b = MemoryBudget::new(1024);
        let g = MemoryBudget::reserve_opt(Some(&b), 256).unwrap();
        assert_eq!(b.used(), 256);
        assert_eq!(g.bytes(), 256);
        drop(g);
        assert_eq!(b.used(), 0);
    }

    #[test]
    fn try_alloc_vec_f32_succeeds() {
        let b = MemoryBudget::new(4096);
        let (v, _g) = try_alloc_vec_f32(Some(&b), 256).unwrap();
        assert_eq!(v.len(), 256);
        assert_eq!(b.used(), 256 * 4);
    }

    #[test]
    fn try_alloc_vec_f32_over_cap() {
        let b = MemoryBudget::new(64);
        assert!(try_alloc_vec_f32(Some(&b), 256).is_err());
        assert_eq!(b.used(), 0);
    }

    #[test]
    fn try_alloc_vec_f32_unbounded() {
        let (v, g) = try_alloc_vec_f32(None, 256).unwrap();
        assert_eq!(v.len(), 256);
        assert_eq!(g.bytes(), 0);
    }

    #[test]
    fn try_alloc_plane_macro() {
        fn run() -> Result<()> {
            let b = MemoryBudget::new(1024 * 1024);
            let (v, _g) = try_alloc_plane_f32!(Some(&b), 100usize, 100usize)?;
            assert_eq!(v.len(), 10000);
            Ok(())
        }
        run().unwrap();
    }

    #[test]
    fn elem_bytes_overflow_caught() {
        assert!(elem_bytes::<f32>(usize::MAX).is_err());
    }
}
