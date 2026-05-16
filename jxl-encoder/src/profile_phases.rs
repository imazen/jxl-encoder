// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later.

//! Per-phase wall-clock instrumentation for the lossless modular encode
//! pipeline (issue #23 — e3->e7 cliff investigation).
//!
//! Zero cost when the `profile-phases` feature is disabled: every public
//! macro / accessor expands to a no-op or returns an empty snapshot.
//!
//! When the feature is enabled, [`profile_time!`] records nanoseconds
//! spent in each named phase into a process-global accumulator (mutex-
//! guarded BTreeMap). The harness reads the accumulator with
//! [`take_snapshot`] between encodes.
//!
//! Phases are addressed by a `&'static str` to keep the macro hot path
//! free of allocations.
//!
//! Why a global mutex instead of thread-local: the lossless encoder
//! parallelizes per-group ANS via rayon. Thread-local accumulators on
//! worker threads are invisible to the main thread that prints the
//! snapshot. The mutex is only contended at phase boundaries (drop time
//! for the RAII guard) and adds <100ns per record on a hot machine,
//! which is acceptable for diagnostic builds.

#[cfg(feature = "profile-phases")]
mod inner {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::time::Instant;

    use once_cell::sync::Lazy;

    static ACC: Lazy<Mutex<BTreeMap<&'static str, u128>>> = Lazy::new(|| Mutex::new(BTreeMap::new()));

    /// Add a duration in nanoseconds to the accumulator for the given phase.
    pub fn record(phase: &'static str, ns: u128) {
        if let Ok(mut m) = ACC.lock() {
            *m.entry(phase).or_insert(0) += ns;
        }
    }

    /// Drain the accumulator and return phase -> total nanoseconds.
    pub fn take_snapshot() -> Vec<(&'static str, u128)> {
        if let Ok(mut m) = ACC.lock() {
            let out: Vec<_> = m.iter().map(|(k, v)| (*k, *v)).collect();
            m.clear();
            out
        } else {
            Vec::new()
        }
    }

    /// Reset the accumulator without returning the snapshot.
    pub fn reset() {
        if let Ok(mut m) = ACC.lock() {
            m.clear();
        }
    }

    /// RAII guard that records elapsed time on drop.
    pub struct PhaseGuard {
        start: Instant,
        phase: &'static str,
    }

    impl PhaseGuard {
        pub fn new(phase: &'static str) -> Self {
            Self {
                start: Instant::now(),
                phase,
            }
        }
    }

    impl Drop for PhaseGuard {
        fn drop(&mut self) {
            let ns = self.start.elapsed().as_nanos();
            record(self.phase, ns);
        }
    }
}

#[cfg(not(feature = "profile-phases"))]
mod inner {
    pub fn record(_phase: &'static str, _ns: u128) {}
    pub fn take_snapshot() -> alloc::vec::Vec<(&'static str, u128)> {
        alloc::vec::Vec::new()
    }
    pub fn reset() {}

    /// Stub guard. Drops do nothing; no `Instant` field exists in `no_std`.
    pub struct PhaseGuard;

    impl PhaseGuard {
        #[inline(always)]
        pub fn new(_phase: &'static str) -> Self {
            PhaseGuard
        }
    }
}

pub use inner::{PhaseGuard, record, reset, take_snapshot};

/// Time the supplied expression block and record under the given phase.
///
/// Compiles to a plain block when `profile-phases` is disabled.
#[macro_export]
macro_rules! profile_time {
    ($phase:literal, $body:block) => {{
        let _g = $crate::profile_phases::PhaseGuard::new($phase);
        $body
    }};
}
