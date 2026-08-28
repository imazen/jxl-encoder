//! Minimal library memory probe for `scripts/mem_peak_calibrate.py`.
//!
//! Loads a PNG, then measures the encoder's MARGINAL peak working set —
//! the `VmHWM` high-water delta across the `encode()` call only, so the
//! binary's static footprint and the input-buffer load (both present
//! before the encode) cancel out. That delta is what
//! `estimate_peak_memory_bytes` should predict (the encoder's own
//! allocations on top of the caller-provided pixels), unlike the CLI
//! whole-process RSS which is inflated by a ~126 MB binary/decode floor.
//!
//! Usage: mem_probe <png> <lossy|lossless> <effort> <distance> <8|16> [rgb|rgba] [threads] [tree]
//! Env (lossless): MEM_PROBE_CROP=WxH, MEM_PROBE_PATCHES=0|1, MEM_PROBE_GROUP_SHIFT=0..=3
//! Prints: `delta_kb=<n> peak_kb=<n> wall_ms=<f> user_ms=<f> sys_ms=<f> bytes=<n> \
//!          threads=<n> est_min_kb=<n> est_typ_kb=<n> est_max_kb=<n> est_time_ms=<f> \
//!          tree=<s> live_pre_kb=<n> peak_live_kb=<n> marginal_live_kb=<n> allocs=<n>`
//! Time is isolated to the `encode()` call (wall via `Instant`, user/sys via
//! `/proc/self/stat`), so the PNG-load and process startup don't count.
//!
//! The optional 7th arg is the encoder thread count for the vCPU resource
//! sweep (default 1). `with_threads(n)` installs a dedicated n-thread rayon
//! pool for the parallel stages (lossless tree-learning, EPF search), so
//! peak working set grows with n (per-worker SplitWorkspace) and wall time
//! drops — exactly the axis `estimate_encode` does NOT yet model. The
//! `est_*` columns are `heuristics::estimate_encode`'s prediction (its
//! thread-independent typical/min/max peak + single-thread time_ms) emitted
//! in the SAME row so prediction-vs-measurement is one join-free record.
//!
//! The optional 6th arg selects the channel layout. `rgba` builds a
//! 4-channel buffer whose alpha plane is the source's GREEN channel — a
//! deterministic, high-entropy (≈ worst-case) alpha, since the calibration
//! corpus is all-opaque. That measures the conservative extra working set
//! the encoder spends on an alpha extra-channel (modular alpha alongside
//! VarDCT, or the 4th channel in lossless), which is what a memory cap
//! should budget for.
//!
//! The optional 8th arg pins the lossless tree mode (imazen/jxl-encoder#96):
//! `auto` (default, the production `SectionedTrees::Auto` policy),
//! `global` (`Off`), `sectioned` (`On`), `hybrid` (`Hybrid`). Ignored for
//! lossy. The sectioned mode's peak is a different working set (one
//! group's tree-learn set per in-flight worker on top of the image copies
//! and the patches-detection planes, instead of the whole-image gather),
//! which the estimator's sectioned arm is calibrated against — pin the
//! mode explicitly when calibrating so the `Auto` thread policy cannot
//! silently swap the path under the measurement.
//!
//! `MEM_PROBE_CROP=WxH` crops the top-left `W`×`H` region of the source
//! before materializing the buffer, so the size axis of a calibration sweep
//! (tiny → large) runs on REAL content rather than synthetic fixtures, and
//! without resampling (which would smooth away the high-frequency detail the
//! working set scales with).
//!
//! ## Allocator-agnostic peak: `peak_live_kb`
//!
//! `delta_kb` / `peak_kb` are `VmHWM` (Linux-only; they read 0 elsewhere)
//! and fold in whatever the platform allocator declined to return to the
//! OS, so they move with the allocator. The counting global allocator below
//! reports the high-water mark of LIVE allocated bytes across the encode
//! (`peak_live_kb`, absolute — the caller's pixel buffer is live and
//! counted, matching the `input` term of `estimate_encode`) plus the live
//! bytes just before the encode (`live_pre_kb`, ≈ the input buffer) and
//! their difference (`marginal_live_kb`, the encoder's own working set).
//! `allocs` is the allocation count during the encode. These are the
//! numbers the estimator's measured-cell tests pin (same definition as the
//! `peak_live` column of the 2026-08-13 4K cells, which came from zenjxl's
//! `mem_probe_encode` counting allocator).

use std::fs;
use std::time::Instant;

/// Counting allocator: allocation COUNT and the high-water mark of LIVE
/// bytes. Racy under contention by at most one concurrent delta, which is
/// irrelevant at the magnitudes reported.
mod counting_alloc {
    use core::alloc::{GlobalAlloc, Layout};
    use core::sync::atomic::{AtomicUsize, Ordering};

    pub static COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static LIVE: AtomicUsize = AtomicUsize::new(0);
    pub static PEAK_LIVE: AtomicUsize = AtomicUsize::new(0);

    pub struct Counting;

    impl Counting {
        fn record_alloc(size: usize) {
            COUNT.fetch_add(1, Ordering::Relaxed);
            let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
            PEAK_LIVE.fetch_max(live, Ordering::Relaxed);
        }
    }

    // SAFETY: every method forwards to `std::alloc::System` with the
    // caller's own arguments and only adds atomic bookkeeping around the
    // call, so the `GlobalAlloc` contract is exactly `System`'s.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: forwarded verbatim to the system allocator under the
            // caller's `GlobalAlloc::alloc` contract (non-zero-size layout).
            let p = unsafe { std::alloc::System.alloc(layout) };
            if !p.is_null() {
                Self::record_alloc(layout.size());
            }
            p
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // SAFETY: same contract as `alloc`, forwarded verbatim.
            let p = unsafe { std::alloc::System.alloc_zeroed(layout) };
            if !p.is_null() {
                Self::record_alloc(layout.size());
            }
            p
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            // SAFETY: `ptr`/`layout` were produced by this allocator's
            // `alloc`/`realloc`, which forward to `System` — the caller's
            // `dealloc` contract carries over unchanged.
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // SAFETY: `ptr` came from `System` via this allocator; the
            // caller's `realloc` contract (matching layout, non-zero
            // `new_size`) is forwarded unchanged.
            let p = unsafe { std::alloc::System.realloc(ptr, layout, new_size) };
            if !p.is_null() {
                // Between the two sizes the allocator may hold both — model
                // the worst case so growth-by-realloc shows in the peak.
                Self::record_alloc(new_size);
                LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            }
            p
        }
    }
}

#[global_allocator]
static GLOBAL: counting_alloc::Counting = counting_alloc::Counting;

/// (utime, stime) of this process in clock ticks, from /proc/self/stat.
/// Fields after the last ')': state ppid ... utime(idx 11) stime(idx 12).
fn cpu_ticks() -> (u64, u64) {
    let s = fs::read_to_string("/proc/self/stat").unwrap_or_default();
    if let Some(p) = s.rfind(')') {
        let f: Vec<&str> = s[p + 1..].split_whitespace().collect();
        if f.len() > 12 {
            return (f[11].parse().unwrap_or(0), f[12].parse().unwrap_or(0));
        }
    }
    (0, 0)
}
// Linux USER_HZ = 100 (10 ms ticks).
const TICK_MS: f64 = 10.0;

fn vmhwm_kb() -> u64 {
    let s = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
        }
    }
    0
}

fn main() {
    use core::sync::atomic::Ordering;

    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!(
            "usage: mem_probe <png> <lossy|lossless> <effort> <distance> <8|16> \
             [rgb|rgba] [threads] [auto|global|sectioned|hybrid]"
        );
        std::process::exit(2);
    }
    let (path, mode, effort, distance, depth) = (
        &a[1],
        &a[2],
        a[3].parse::<u8>().unwrap(),
        a[4].parse::<f32>().unwrap(),
        a[5].parse::<u8>().unwrap(),
    );
    let alpha = a.get(6).map(String::as_str).unwrap_or("rgb");
    let threads: usize = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(1);
    let tree = a.get(8).map(String::as_str).unwrap_or("auto");

    use jxl_encoder::api::SectionedTrees;
    use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};
    let sectioned = match tree {
        "auto" => SectionedTrees::Auto,
        "global" => SectionedTrees::Off,
        "sectioned" => SectionedTrees::On,
        "hybrid" => SectionedTrees::Hybrid,
        other => {
            eprintln!("unknown tree mode {other:?} (auto|global|sectioned|hybrid)");
            std::process::exit(2);
        }
    };

    let mut img = image::open(path).expect("open png");
    if let Ok(spec) = std::env::var("MEM_PROBE_CROP") {
        let (cw, ch) = spec
            .split_once('x')
            .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
            .expect("MEM_PROBE_CROP=WxH");
        assert!(
            cw <= img.width() && ch <= img.height(),
            "crop {cw}x{ch} exceeds source {}x{}",
            img.width(),
            img.height()
        );
        img = img.crop_imm(0, 0, cw, ch);
    }
    let (w, h) = (img.width(), img.height());

    // Materialize the caller-provided pixel buffer BEFORE the baseline so it
    // is part of the load floor, not the measured encode delta. For `rgba`
    // the alpha plane is the green channel (deterministic high-entropy alpha).
    let (pixels, layout): (Vec<u8>, PixelLayout) = match (depth, alpha) {
        (16, "rgba") => {
            let buf = img.to_rgba16();
            let raw = buf.as_raw(); // RGBA interleaved
            let mut bytes = Vec::with_capacity(raw.len() * 2);
            for px in raw.as_chunks::<4>().0 {
                let g = px[1];
                bytes.extend_from_slice(&px[0].to_ne_bytes());
                bytes.extend_from_slice(&px[1].to_ne_bytes());
                bytes.extend_from_slice(&px[2].to_ne_bytes());
                bytes.extend_from_slice(&g.to_ne_bytes()); // alpha := green
            }
            (bytes, PixelLayout::Rgba16)
        }
        (16, _) => {
            let buf = img.to_rgb16();
            let mut bytes = Vec::with_capacity(buf.as_raw().len() * 2);
            for &v in buf.as_raw() {
                bytes.extend_from_slice(&v.to_ne_bytes());
            }
            (bytes, PixelLayout::Rgb16)
        }
        (_, "rgba") => {
            let mut buf = img.to_rgba8().into_raw();
            for px in buf.as_chunks_mut::<4>().0 {
                px[3] = px[1]; // alpha := green
            }
            (buf, PixelLayout::Rgba8)
        }
        _ => (img.to_rgb8().into_raw(), PixelLayout::Rgb8),
    };
    // The decoded source is dead once the buffer is materialized; drop it so
    // `live_pre_kb` is (essentially) the caller's pixel buffer alone.
    drop(img);

    // Model prediction for the same encode (thread-independent — calibrated
    // at threads=1). input_bpp/has_alpha derive from the materialized layout.
    let is_lossless = mode == "lossless";
    let input_bpp: u8 = match (depth, alpha == "rgba") {
        (16, true) => 8,
        (16, _) => 6,
        (_, true) => 4,
        _ => 3,
    };
    let est = jxl_encoder::heuristics::estimate_encode(
        w,
        h,
        input_bpp,
        alpha == "rgba",
        is_lossless,
        effort,
    );

    let baseline = vmhwm_kb();
    let live_pre = counting_alloc::LIVE.load(Ordering::Relaxed);
    counting_alloc::PEAK_LIVE.store(live_pre, Ordering::Relaxed);
    counting_alloc::COUNT.store(0, Ordering::Relaxed);
    let (cu0, cs0) = cpu_ticks();
    let t0 = Instant::now();
    // `with_threads(n)`: n>=1 installs a dedicated n-thread rayon pool for
    // the parallel stages (the vCPU axis); n=1 forces sequential.
    // Optional lossless knobs for attribution / tuning runs (#96):
    // `MEM_PROBE_PATCHES=0|1` overrides the effort default for the
    // lossless patches pre-pass; `MEM_PROBE_GROUP_SHIFT=0..=3` sets the
    // modular group dimension {128, 256, 512, 1024}.
    let patches_override: Option<bool> = std::env::var("MEM_PROBE_PATCHES").ok().map(|v| v != "0");
    let group_shift: Option<u8> = std::env::var("MEM_PROBE_GROUP_SHIFT")
        .ok()
        .map(|v| v.parse::<u8>().expect("MEM_PROBE_GROUP_SHIFT=0..=3"));
    let encoded = if is_lossless {
        let mut cfg = LosslessConfig::new()
            .with_effort(effort)
            .with_threads(threads)
            .with_sectioned_trees(sectioned);
        if let Some(p) = patches_override {
            cfg = cfg.with_patches(p);
        }
        if let Some(g) = group_shift {
            cfg = cfg.with_modular_group_size(Some(g));
        }
        cfg.encode_request(w, h, layout).encode(&pixels)
    } else {
        LossyConfig::new(distance)
            .with_effort(effort)
            .with_threads(threads)
            .encode_request(w, h, layout)
            .encode(&pixels)
    };
    let wall = t0.elapsed();
    let (cu1, cs1) = cpu_ticks();
    let peak = vmhwm_kb();
    let peak_live = counting_alloc::PEAK_LIVE.load(Ordering::Relaxed);
    let allocs = counting_alloc::COUNT.load(Ordering::Relaxed);
    let len = match encoded {
        Ok(d) => d.len(),
        Err(e) => {
            eprintln!("encode failed: {e}");
            0
        }
    };
    let (est_min, est_typ, est_max, est_t) = est
        .map(|e| {
            (
                e.peak_memory_bytes_min / 1024,
                e.peak_memory_bytes / 1024,
                e.peak_memory_bytes_max / 1024,
                e.time_ms,
            )
        })
        .unwrap_or((0, 0, 0, 0.0));
    println!(
        "delta_kb={} peak_kb={} wall_ms={:.1} user_ms={:.1} sys_ms={:.1} bytes={} \
         threads={} est_min_kb={} est_typ_kb={} est_max_kb={} est_time_ms={:.1} \
         tree={} live_pre_kb={} peak_live_kb={} marginal_live_kb={} allocs={}",
        peak.saturating_sub(baseline),
        peak,
        wall.as_secs_f64() * 1000.0,
        (cu1 - cu0) as f64 * TICK_MS,
        (cs1 - cs0) as f64 * TICK_MS,
        len,
        threads,
        est_min,
        est_typ,
        est_max,
        est_t,
        tree,
        live_pre / 1024,
        peak_live / 1024,
        peak_live.saturating_sub(live_pre) / 1024,
        allocs,
    );
}
