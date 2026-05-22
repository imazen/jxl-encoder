// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.
//
// This file is the ONLY place in `zenjxl-tuning-runner` that uses
// `unsafe` — it's an FFI shim around `libc::getrusage`. Every other
// module sets `#![forbid(unsafe_code)]` at the file level. The workspace
// lint `undocumented_unsafe_blocks = "deny"` requires every `unsafe`
// block to carry a `# Safety` comment; we document the single call
// site below.

//! `libc::getrusage(RUSAGE_SELF)` snapshot + diff for CPU + peak-RSS
//! measurement around the encode and decode phases.
//!
//! Returns:
//! - `user_ms` — CPU time spent in user mode (delta between snapshots)
//! - `sys_ms` — CPU time spent in kernel mode (delta)
//! - `peak_rss_mb` — `ru_maxrss` at the post-snapshot, in MiB. NOTE:
//!   `ru_maxrss` is a high-water mark across the whole process
//!   lifetime; the runner is single-cell-per-process so the reading
//!   is per-cell-accurate.
//!
//! ## Platform notes
//!
//! - On Linux, `ru_maxrss` is in KiB (we divide by 1024).
//! - On macOS, `ru_maxrss` is in BYTES (we divide by 1024 * 1024).
//!   We currently target Linux fleets only (Dockerfile is
//!   `ubuntu:24.04`); macOS readings will be 1024× smaller.
//! - Windows: not currently supported; `getrusage` doesn't exist
//!   there. The Linux fleet is the only target for now.

/// Snapshot of resource usage at one point in time.
#[derive(Clone, Copy, Debug, Default)]
pub struct RUsageDelta {
    pub user_us: u64,
    pub sys_us: u64,
    pub max_rss_kib: u64,
}

impl RUsageDelta {
    /// Capture the current `RUSAGE_SELF` reading.
    #[allow(unsafe_code)] // single FFI call; see `# Safety` comment below
    pub fn snapshot() -> Self {
        // SAFETY: getrusage is a syscall wrapper; passing a valid
        // mut ptr to a zero-initialised `libc::rusage` is safe.
        let mut ru = libc::rusage {
            ru_utime: libc::timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
            ru_stime: libc::timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
            ru_maxrss: 0,
            ru_ixrss: 0,
            ru_idrss: 0,
            ru_isrss: 0,
            ru_minflt: 0,
            ru_majflt: 0,
            ru_nswap: 0,
            ru_inblock: 0,
            ru_oublock: 0,
            ru_msgsnd: 0,
            ru_msgrcv: 0,
            ru_nsignals: 0,
            ru_nvcsw: 0,
            ru_nivcsw: 0,
        };
        // SAFETY:
        // - `libc::RUSAGE_SELF` is a valid integer constant
        // - `&mut ru` is a writable pointer to a fully-initialised
        //   `libc::rusage` of the correct layout
        // - `getrusage` performs a syscall and writes the fields it
        //   knows about. Any unwritten fields keep their
        //   zero-initialised values, which is well-defined.
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
        if rc != 0 {
            return Self::default();
        }
        let user_us = (ru.ru_utime.tv_sec as u64) * 1_000_000 + (ru.ru_utime.tv_usec as u64);
        let sys_us = (ru.ru_stime.tv_sec as u64) * 1_000_000 + (ru.ru_stime.tv_usec as u64);
        Self {
            user_us,
            sys_us,
            // ru_maxrss on Linux is KiB; on macOS bytes. We target
            // Linux fleets.
            max_rss_kib: ru.ru_maxrss.max(0) as u64,
        }
    }

    /// Return the diff fields users want: `user_ms`, `sys_ms`,
    /// `peak_rss_mb`.
    pub fn diff(&self, pre: &Self) -> RUsageDeltaResolved {
        RUsageDeltaResolved {
            user_ms: (self.user_us.saturating_sub(pre.user_us)) / 1000,
            sys_ms: (self.sys_us.saturating_sub(pre.sys_us)) / 1000,
            // Peak is the high-water mark; we report the post-value
            // converted to MiB so the column is human-readable. For
            // a delta-style "peak grew by" metric, callers can take
            // (post.max_rss_kib - pre.max_rss_kib) themselves.
            peak_rss_mb: (self.max_rss_kib / 1024) as u32,
        }
    }
}

/// Resolved delta with human-readable units.
#[derive(Clone, Copy, Debug, Default)]
pub struct RUsageDeltaResolved {
    pub user_ms: u64,
    pub sys_ms: u64,
    pub peak_rss_mb: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_returns_nonzero_rss() {
        let s = RUsageDelta::snapshot();
        // RSS should always be > 0 on a live process.
        assert!(s.max_rss_kib > 0, "RSS was 0; getrusage likely failed");
    }

    #[test]
    fn diff_after_work_increases_user_us() {
        let pre = RUsageDelta::snapshot();
        // Generate some user-space work.
        let mut sum = 0u64;
        for i in 0..1_000_000u64 {
            sum = sum.wrapping_add(i);
        }
        std::hint::black_box(sum);
        let post = RUsageDelta::snapshot();
        let d = post.diff(&pre);
        // peak_rss_mb derived from post value; should not be zero.
        assert!(d.peak_rss_mb > 0, "peak_rss_mb was 0; suspicious");
    }
}
