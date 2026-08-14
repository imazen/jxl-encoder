// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Encode resource estimation (peak memory / time / output size).
//!
//! Mirrors the zen-ecosystem per-codec estimation pattern
//! (cf. `zenwebp::heuristics` — same [`EncodeEstimate`] shape with
//! min / typical / max peak memory). The two callers care about
//! different points of the distribution:
//!
//! - [`EncodeEstimate::peak_memory_bytes_max`] — a conservative upper
//!   bound for "will it fit / how big a cap" decisions. Sized to cover
//!   the worst content in the calibration corpus plus margin; should not
//!   under-report.
//! - [`EncodeEstimate::peak_memory_bytes`] — the *typical* (≈ p50) peak
//!   for capacity planning / scheduling concurrent encodes.
//! - [`EncodeEstimate::peak_memory_bytes_min`] — best case (simple
//!   content).
//!
//! ## Model
//!
//! `peak = input_buffer + fixed_overhead + bytes_per_pixel(path, effort) · pixels`
//!
//! `bytes_per_pixel` is the encoder's *marginal* working set per pixel
//! (the f32 XYB/recon planes, transform coefficients, EPF sharpness
//! search, entropy state — everything the encoder allocates on top of the
//! caller-supplied input buffer). Unlike a smooth quality dial, it has
//! **step jumps** at the effort thresholds where the heavy machinery
//! turns on:
//!   - lossy: two bands — e ≤ 6 base (≈ 52 B/px measured flat 12→108 MP;
//!     β 80 keeps the 1024²-bulge envelope) and e ≥ 7 (≈ up to 122 B/px
//!     measured at d6.0; β 135). The buttloop (e ≥ 8) adds ~0 memory over
//!     e7 at equal quality, so it shares the e7 band.
//!   - lossless: four bands — e ≤ 5, e6 (partial search), e7–e9 full MA
//!     tree-learning, e ≥ 10; plus a +230 B/px alpha term (the 4th
//!     channel through tree-learning). Lossless is ~size-independent
//!     (1 MP ≈ 12 MP ≈ measured constants at every size).
//!
//! ## Calibration
//!
//! Constants are the measured marginal working set (mem_probe `VmHWM`
//! delta around `encode()`, which excludes the binary floor and the
//! caller's input buffer), recalibrated 2026-06-23 from a full SIZE sweep
//! per the no-extrapolate-memory discipline, then re-anchored 2026-08-01
//! at REAL LARGE SIZES (12/20/27/48/75/108 MP zensysbench corpus photos,
//! d1.75 + d6.0, threads 1–16, `benchmarks/
//! jxl_encode_mem_threads_postfix_2026-08-01.tsv` — measured AFTER the
//! per-tile AC-strategy quadratic-peak fix; the matching pre-fix grid in
//! `jxl_encode_mem_threads_2026-08-01.tsv` documents the removed
//! px²/262144 term). The 2026-08-01 pass raised the lossy e ≥ 7 band
//! (q30-regime, never swept before), the lossless base and tree bands,
//! and re-confirmed the lossy 2.5 MB/thread term (unclamped — see
//! [`estimate_encode_threaded`]). The 2026-06-23 recalibration replaced the
//! 2026-06-14 constants, which were fit at **1024² only** — a single size
//! conflates the fixed overhead α into the per-pixel β and inflates the
//! apparent B/px, making the TYP behave like a MAX (≈ 1.5–4× over the real
//! marginal, worst at small sizes where α dominates and at e ≥ 8 where the
//! 1024²-anchored buttloop B/px was a 1024²-artifact).
//!
//! Sweep: sizes {256, 512, 1024, 2048} × {lossy, lossless} × effort
//! {1,4,5,6,7,8,9} × content {photo, screenshot} × threads {1, 8, 16},
//! lossy at q50 (worst-case: the e7 quant + buttloop working set is heavier
//! at low quality) and q90. Per (mode, effort-band, content, thread) we fit
//! `marginal = α + β·pixels` across sizes, then choose (α, β) as the smallest
//! linear upper bound that clears the MAX measured cell at every size with a
//! **≥ 10 % safety margin** (verified — min margin 1.10 over all measured
//! cells, 1.26–1.28 at the 12 MP asymptote). Provenance:
//! `benchmarks/jxl_encode_mem_2026-06-23.tsv` + `scripts/mem_peak_fit.py`
//! (the prior `mem_peak_quick_2026-06-14.tsv` is superseded for the per-pixel
//! values; its rgb-vs-rgba alpha term is retained — not re-measured here).
//!
//! Findings that shaped the model:
//!   - The lossy working set is **not monotone in B/px**: it bulges at
//!     1024² (≈ 122 B/px at q50) then drops to ≈ 65 B/px at 2048² and
//!     ≈ 75 B/px at 12 MP. The 1024² point is the binding constraint, so β
//!     is set to clear it (≈ 80) with a larger α (50 MB), which over-predicts
//!     the 2048²/12 MP asymptote by ≈ 1.5× — the safe direction.
//!   - **The buttloop (e ≥ 8) adds essentially ZERO memory over e7 at the
//!     same quality** (lossy e7 q50 and e8/e9 q50 measure byte-for-byte the
//!     same working set across all sizes). The prior 300 B/px buttloop band
//!     was a 1024²-α-artifact; the real value tracks the base (β ≈ 80–90).
//!   - lossless is ~size-independent in B/px (1 MP ≈ 12 MP): base (e ≤ 5)
//!     ≈ 72 B/px, e6 ≈ 215 B/px (the prior 140 UNDER-predicted — e6 is a
//!     much heavier partial-search than assumed; β set to 235 for headroom),
//!     e7–e9 tree-learning ≈ 360–425 B/px (e9 is the band's top; β 465 keeps
//!     ≥ 10 % margin). e10 (620) is unswept here and retained from
//!     2026-06-14, as is the +230 B/px lossless alpha term.
//!   - Content spread (photo vs screenshot) and thread count (1/8/16) barely
//!     move the working set: lossy +~95 KB/thread (NOT 2.5 MB), lossless
//!     thread-invariant; the envelope already folds these in. So the per-
//!     pixel β is content- and thread-independent and the per-thread term is
//!     additive (`mem_bytes_per_thread`, kept conservative).
//!   - est/max ratio (heaptrack `peak heap` requested vs probe `VmHWM`
//!     marginal, 3000²+ cells): requested-heap / working is 1.02–1.25, so
//!     `MULT_MAX = 1.8` covers both the content tail AND the requested-heap-
//!     vs-RSS gap. Bit depth barely moves it (f32 internals dominate), so
//!     only the input buffer carries the depth, not `bytes_per_pixel`.

/// Resource estimate for an encode operation. `#[non_exhaustive]` so
/// fields can be added without a breaking change.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct EncodeEstimate {
    /// Best-case peak memory in bytes (simple / low-entropy content).
    pub peak_memory_bytes_min: u64,
    /// Typical (≈ p50) peak memory in bytes for natural content.
    pub peak_memory_bytes: u64,
    /// Conservative upper-bound peak memory in bytes (worst content +
    /// margin). Use this to size a [`crate::api::Limits`] cap.
    pub peak_memory_bytes_max: u64,
    /// Rough single-thread encode time in milliseconds, effort-aware.
    /// Calibrated from measured wall/user time (`mem_probe`, threads=1,
    /// 2026-06-14) — effort dominates: lossless e9 is ~300× e1 per pixel.
    /// Divide by thread count for an approximate wall-latency estimate.
    pub time_ms: f32,
    /// Rough estimated output size in bytes.
    pub output_bytes: u64,
}

// ── Calibrated constants (2026-06-23, full size-sweep marginal working set) ──

/// Encoder fixed overhead (size-independent): plans, tables, thread pool,
/// and — for lossy — the size-sublinear butteraugli/EPF precompute that
/// makes the marginal bulge at 1024². The lossy α is set to 50 MB so the
/// `α + β·pixels` line clears the 1024² bulge (≈ 122 MB worst case) with a
/// ≥ 10 % safety margin at β ≈ 80 (the alternative, a smaller α + steeper β,
/// over-predicts the 2048²/12 MP asymptote much more). Lossless has no such
/// bulge, so its α is the small true fixed cost (16 MB covers the worst
/// lossless band's intercept with margin).
const LOSSY_FIXED_OVERHEAD: u64 = 50 << 20;
const LOSSLESS_FIXED_OVERHEAD: u64 = 16 << 20;

/// Typical encoder marginal working set, bytes/pixel (size-sweep envelope).
/// Lossy base covers e ≤ 6; e ≥ 7 is its own band (2026-08-01 re-measure).
///
/// 2026-08-01 large-size recalibration (`benchmarks/
/// jxl_encode_mem_threads_postfix_2026-08-01.tsv`, real 12–108 MP corpus
/// photos, d1.75 + d6.0, AFTER the per-tile AC-strategy quadratic fix):
///   - e5 marginal is a dead-flat ≈ 52 B/px from 12 MP through 108 MP
///     (625 MiB @12 MP → 5.6 GiB @108 MP, zero intercept) and
///     distance-invariant — the base β=80 (which the 1024²-bulge envelope
///     from 2026-06-23 still requires: α 50 MB + β·1 MP ≥ 122 MB) covers
///     it with ≥ 1.5× headroom at large sizes. Unchanged.
///   - e7 measured UP TO ≈ 122 B/px at d6.0 on the 12 MP corpus photo
///     (1.42 GiB marginal; the 20/48 MP photos measured ≈ 59 — strong
///     content spread), well above the old shared base 80. The 2026-06-23
///     sweep only measured q50/q90 (d4/d1) — the q30 regime was never
///     swept, which is exactly where the zensysbench fleet ran. New e≥7
///     band β=135 = envelope of the 122 max cell + ≥ 10 % margin.
///   - e ≥ 8 (buttloop) keeps tracking e7 (2026-06-23 finding: buttloop
///     adds ~0 working set at equal quality), so one band covers e ≥ 7.
const LOSSY_BPP_BASE: f64 = 80.0;
const LOSSY_BPP_E7PLUS: f64 = 135.0;
/// Lossless ramps in four bands (not a clean step): e ≤ 5 base
/// (no tree learning), e6 (heavier partial-search), e7–e9 full MA
/// tree-learning, e ≥ 10 (`fine_grained_step`/multi-seed). Unlike lossy,
/// the lossless working set is ~size-independent (1 MP ≈ 12 MP measured), so
/// these B/px hold at the large sizes where memory matters. The e10 value is
/// retained from the 2026-06-14 grid (e ≥ 10 not re-swept since).
///
/// 2026-08-01 re-measure on the 12 MP zensysbench corpus photo
/// (`benchmarks/jxl_encode_mem_threads_postfix_2026-08-01.tsv`): e5
/// marginal 995 MiB (≈ 83 B/px — the prior 76 under-covered by 7 %;
/// raised to 92, +12 % margin) and e7 marginal 5.88 GiB at threads=1
/// (≈ 490 B/px — the prior 465 under-covered by 4 %; raised to 540,
/// +11 % margin — content spread vs the 2026-06-23 photo's 4.19 GiB is
/// 1.4×, and β is the envelope-of-max per the calibration discipline).
/// Threads=8 measured BELOW threads=1 (4.59 GiB) — lossless stays
/// thread-invariant in the model (γ = 0, anchored at the t=1 max).
const LOSSLESS_BPP_BASE: f64 = 92.0;
const LOSSLESS_BPP_E6: f64 = 235.0;
const LOSSLESS_BPP_TREE: f64 = 540.0;
const LOSSLESS_BPP_E10: f64 = 620.0;

/// Extra marginal working set for a lossless encode with an alpha
/// extra-channel (the 4th channel goes through tree-learning). Measured
/// +226…255 B/px, flat across effort (2026-06-14, rgb-vs-rgba sweep). The
/// lossy path's alpha (modular extra-channel alongside VarDCT) adds only
/// +2…5 B/px — within noise — so there is no lossy alpha term; the alpha
/// input buffer's extra byte/px is already counted via `input_bpp`.
const LOSSLESS_BPP_ALPHA: f64 = 230.0;

/// Content-spread multipliers around the typical (median) estimate.
/// `max` 1.8 is kept as the MAX-tier multiplier: the 2026-06-23 size sweep
/// confirmed it covers BOTH the modest content/thread spread (photo vs
/// screenshot, 1/8/16 threads, all within ~1.1× of each other once the band
/// β is the worst-case envelope) AND the requested-heap-vs-RSS gap measured
/// by heaptrack (`peak heap` / `VmHWM` working = 1.02–1.25 on 3000²+ cells)
/// — so `working · 1.8` ≥ the allocator's requested-heap peak with headroom.
/// `min` 0.85 is intentionally NOT the measured floor; since β is now the
/// worst-case envelope (which over-predicts the large-size asymptote), the
/// MIN can even sit slightly above a very-large-size measurement — that is
/// harmless (MIN is informational best-case, never a cap callers OOM on;
/// only TYP/MAX under-prediction would cause OOM, and the envelope prevents
/// that at every measured size).
const MULT_MIN: f64 = 0.85;
const MULT_MAX: f64 = 1.8;

/// Single-thread encode microseconds/pixel per effort, measured 2026-06-14
/// (`mem_probe` wall/user, threads=1, 512²+1024², median over 3 classes;
/// `benchmarks/jxl_encode_time_2026-06-14.tsv`). Encode time is dominated
/// by EFFORT, not path-flat throughput — lossless spans 0.04 → 12.8 µs/px
/// (e1 → e9, ~300×), lossy 0.05 → 1.4. The old flat 3/6 MP/s model was
/// ~8× high at low effort and ~38× low at e9. Anchors are interpolated
/// linearly between measured efforts and clamped to [1, 9] (e10+ unmeasured
/// — the e9 value is a lower bound there). `(effort, us_per_px)`.
const LOSSY_TIME_ANCHORS: [(f64, f64); 5] = [
    (1.0, 0.053),
    (3.0, 0.058),
    (5.0, 0.21),
    (7.0, 0.343),
    (9.0, 1.435),
];
const LOSSLESS_TIME_ANCHORS: [(f64, f64); 5] = [
    (1.0, 0.043),
    (3.0, 0.081),
    (5.0, 0.587),
    (7.0, 2.542),
    (9.0, 12.827),
];

fn encode_us_per_px(is_lossless: bool, effort: u8) -> f64 {
    let anchors = if is_lossless {
        &LOSSLESS_TIME_ANCHORS
    } else {
        &LOSSY_TIME_ANCHORS
    };
    let e = (effort as f64).clamp(anchors[0].0, anchors[4].0);
    for w in anchors.windows(2) {
        let (e0, u0) = w[0];
        let (e1, u1) = w[1];
        if e <= e1 {
            return u0 + (u1 - u0) * (e - e0) / (e1 - e0);
        }
    }
    anchors[4].1
}

/// Typical marginal-working-set bytes/pixel for the given path + effort
/// (the size-sweep envelope β; the size-dependent fixed term is
/// `*_FIXED_OVERHEAD`, added once in [`estimate_encode`]). Lossy e ≥ 7 is
/// one band: e7 carries the heavier DCT64/EPF-era working set (measured
/// up to ≈ 122 B/px at d6.0, 2026-08-01) and the buttloop (e ≥ 8) adds
/// ~0 on top of e7 at equal quality (2026-06-23 finding; its butteraugli
/// precompute is folded into the lossy α, not a per-pixel surcharge).
fn bytes_per_pixel(is_lossless: bool, effort: u8) -> f64 {
    if is_lossless {
        if effort >= 10 {
            LOSSLESS_BPP_E10
        } else if effort >= 7 {
            LOSSLESS_BPP_TREE
        } else if effort == 6 {
            LOSSLESS_BPP_E6
        } else {
            LOSSLESS_BPP_BASE
        }
    } else if effort >= 7 {
        LOSSY_BPP_E7PLUS
    } else {
        LOSSY_BPP_BASE
    }
}

/// Estimate peak memory / time / output for an encode.
///
/// * `width`, `height` — image dimensions in pixels.
/// * `input_bpp` — bytes per pixel of the caller's input buffer (e.g. 3
///   for RGB8, 4 for RGBA8, 6 for RGB16, 12 for f32 RGB). The input buffer
///   is live during encode, so it is part of the peak; the encoder's own
///   per-pixel working set is on top and is `bpp`-independent (f32
///   internals dominate).
/// * `has_alpha` — whether the layout carries an alpha extra-channel. For
///   lossless this adds a substantial per-pixel term (the 4th channel goes
///   through tree-learning, +230 B/px); for lossy it is negligible.
/// * `is_lossless`, `effort` — select the calibration stratum.
///
/// Returns `None` only on dimension overflow.
#[must_use]
pub fn estimate_encode(
    width: u32,
    height: u32,
    input_bpp: u8,
    has_alpha: bool,
    is_lossless: bool,
    effort: u8,
) -> Option<EncodeEstimate> {
    let pixels = (width as u64).checked_mul(height as u64)?;
    let input = pixels.checked_mul(input_bpp as u64)?;
    let fixed = if is_lossless {
        LOSSLESS_FIXED_OVERHEAD
    } else {
        LOSSY_FIXED_OVERHEAD
    };
    let mut bpp = bytes_per_pixel(is_lossless, effort);
    if has_alpha && is_lossless {
        bpp += LOSSLESS_BPP_ALPHA;
    }
    let working = (pixels as f64 * bpp) as u64;
    let typical = fixed.checked_add(input)?.checked_add(working)?;

    // Multipliers apply to the content-dependent working set, not the
    // deterministic fixed + input terms.
    let base = fixed + input;
    let min = base + (working as f64 * MULT_MIN) as u64;
    let max = base + (working as f64 * MULT_MAX) as u64;

    let time_ms = (pixels as f64 * encode_us_per_px(is_lossless, effort) / 1_000.0) as f32;
    // Coarse output estimate: lossless ~0.5× input; lossy scales loosely.
    let output_bytes = if is_lossless {
        input / 2
    } else {
        (input as f64 * 0.08) as u64
    };

    Some(EncodeEstimate {
        peak_memory_bytes_min: min,
        peak_memory_bytes: typical,
        peak_memory_bytes_max: max,
        time_ms,
        output_bytes,
    })
}

/// How an encode scales across CPU cores (measured, single-photo sparse fit,
/// `benchmarks/vcpu_resource_sweep_2026-06-20.tsv`). `estimate_encode`'s
/// `time_ms` is a single-thread figure; use [`estimate_encode_threaded`] (or
/// [`ThreadingInfo::speedup`]) to fold in the available cores — wall time does
/// NOT scale as `1/cores`: speedup saturates per-codec and per-effort.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ThreadingInfo {
    /// Whether the encode uses more than one core at all.
    pub parallel: bool,
    /// Threads beyond this yield no further speedup (group/tile/block count
    /// caps it). 1 = serial.
    pub max_useful_threads: u32,
    /// Amdahl parallel fraction `p` fitted from measurement; peak speedup is
    /// `1/(1-p)`. 0 = serial.
    pub parallel_fraction: f32,
    /// Extra peak working-set per added worker thread, bytes (the γ term;
    /// peak-RSS basis — lossless heap is thread-invariant, lossy carries
    /// per-worker buttloop/EPF scratch).
    pub mem_bytes_per_thread: u64,
}

impl ThreadingInfo {
    /// Threads that actually do work given `cores` available (clamped to
    /// `max_useful_threads`).
    #[must_use]
    pub fn effective_threads(&self, cores: usize) -> u64 {
        (cores.max(1) as u64).min(self.max_useful_threads.max(1) as u64)
    }
    /// Achieved wall-time speedup at `cores` (Amdahl, clamped). 1.0 = serial.
    #[must_use]
    pub fn speedup(&self, cores: usize) -> f32 {
        let n = self.effective_threads(cores);
        if !self.parallel || n <= 1 {
            return 1.0;
        }
        let p = self.parallel_fraction as f64;
        (1.0 / ((1.0 - p) + p / n as f64)) as f32
    }
}

/// Threading characterisation for a jxl encode. lossless: rayon group/tree
/// parallel, peak speedup ~3.5×, heap thread-invariant (2026-08-01: 12 MP
/// e7 t=8 measured BELOW t=1 — 4.59 vs 5.88 GiB). lossy: buttloop/EPF
/// parallel, peak ~1.9×, ~2.5 MB/thread scratch (2026-08-01 re-measure,
/// `benchmarks/jxl_encode_mem_threads_postfix_2026-08-01.tsv`: 12 MP e7
/// d6.0 rises 1423.5 → 1461.8 MiB across t=1→16 ≈ 2.5 MB/thread exactly;
/// e5 and 20 MP+ are thread-flat — the constant is the worst measured
/// slope).
#[must_use]
pub fn encode_threading_info(is_lossless: bool, _effort: u8) -> ThreadingInfo {
    if is_lossless {
        ThreadingInfo {
            parallel: true,
            max_useful_threads: 16,
            parallel_fraction: 0.72,
            mem_bytes_per_thread: 0,
        }
    } else {
        ThreadingInfo {
            parallel: true,
            max_useful_threads: 8,
            parallel_fraction: 0.55,
            mem_bytes_per_thread: 2_500_000,
        }
    }
}

/// [`estimate_encode`] adjusted for `cores` available CPU cores: `time_ms` is
/// divided by the measured (saturating) speedup and the peak terms gain the
/// per-thread working-set. Pair with [`encode_threading_info`] to inspect the
/// scaling. Returns `None` only on dimension overflow.
///
/// The memory term uses the RAW `cores` count, NOT the speedup-clamped
/// [`ThreadingInfo::effective_threads`]: `max_useful_threads` is where
/// wall-time speedup saturates, but a wider rayon pool still materializes
/// per-worker scratch (12 MP e7 measured a continued +2.5 MB/thread rise
/// through t=16 > `max_useful` 8, 2026-08-01) — clamping would flatten
/// the estimate exactly where a budget-driven thread walk-down needs it
/// to keep growing.
#[must_use]
pub fn estimate_encode_threaded(
    width: u32,
    height: u32,
    input_bpp: u8,
    has_alpha: bool,
    is_lossless: bool,
    effort: u8,
    cores: usize,
) -> Option<EncodeEstimate> {
    let mut e = estimate_encode(width, height, input_bpp, has_alpha, is_lossless, effort)?;
    let ti = encode_threading_info(is_lossless, effort);
    e.time_ms = (e.time_ms as f64 / ti.speedup(cores) as f64) as f32;
    let extra = ti
        .mem_bytes_per_thread
        .saturating_mul((cores.max(1) as u64).saturating_sub(1));
    e.peak_memory_bytes_min = e.peak_memory_bytes_min.saturating_add(extra);
    e.peak_memory_bytes = e.peak_memory_bytes.saturating_add(extra);
    e.peak_memory_bytes_max = e.peak_memory_bytes_max.saturating_add(extra);
    Some(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 12 MP estimates must SAFELY cover the measured marginal working set
    /// (mem_probe `VmHWM` delta, 3000×4000, 2026-06-23 re-measure): lossy e9
    /// ≈ 857 MB, lossless e7 ≈ 4186 MB. The 2026-06-14 anchors this test used
    /// (3468 MB lossy / 5072 MB lossless) were ~4× / ~1.2× over the real
    /// marginal — the old model over-predicted to match them. The binding
    /// safety contract is `measured + input ≤ MAX` (callers size a cap from
    /// `peak_memory_bytes_max`); TYP must be a *tight* upper bound (not 2×
    /// over), not strictly bracket — MIN (0.85·working) can sit above a
    /// large-size measurement once β is the worst-case envelope, which is
    /// harmless (MIN is informational best-case).
    #[test]
    fn estimate_safely_covers_measured_12mp() {
        let px12 = (3000, 4000);
        let input12 = 3000u64 * 4000 * 3;
        // lossy e9: measured 857 MB working + ~34 MB input.
        let lossy = estimate_encode(px12.0, px12.1, 3, false, false, 9).unwrap();
        let measured_lossy = (857u64 << 20) + input12;
        assert!(
            measured_lossy <= lossy.peak_memory_bytes_max,
            "lossy e9 12MP measured {measured_lossy} exceeds MAX {} — UNSAFE (would OOM a \
             cap sized from the estimate)",
            lossy.peak_memory_bytes_max
        );
        // TYP must be a tight upper bound: ≥ measured but < 2× measured (the
        // old model was ~4× over here — that over-prediction is what this
        // recalibration fixes).
        assert!(
            lossy.peak_memory_bytes >= measured_lossy
                && lossy.peak_memory_bytes < measured_lossy * 2,
            "lossy e9 12MP TYP {} not a tight cover of measured {measured_lossy} \
             (want [meas, 2·meas))",
            lossy.peak_memory_bytes
        );
        // lossless e7: measured ~4186 MB working + ~34 MB input.
        let ll = estimate_encode(px12.0, px12.1, 3, false, true, 7).unwrap();
        let measured_ll = (4186u64 << 20) + input12;
        assert!(
            measured_ll <= ll.peak_memory_bytes_max,
            "lossless e7 12MP measured {measured_ll} exceeds MAX {} — UNSAFE",
            ll.peak_memory_bytes_max
        );
        assert!(
            ll.peak_memory_bytes >= measured_ll && ll.peak_memory_bytes < measured_ll * 2,
            "lossless e7 12MP TYP {} not a tight cover of measured {measured_ll}",
            ll.peak_memory_bytes
        );
    }

    /// Lossless is a much heavier memory regime than lossy at the same
    /// effort; the effort steps (lossless e6, lossless e7, lossless e10)
    /// must show up. NOTE (2026-06-23 re-measure): the lossy buttloop (e ≥ 8)
    /// does NOT create a memory step — it measures the same working set as e7
    /// at the same quality — so e9 is only modestly above e7 (the band's
    /// extra-headroom bump), not 2× as the prior 300-B/px buttloop const
    /// implied. The real steps are all on the lossless side.
    #[test]
    fn effort_steps_and_path_ordering() {
        let (w, h) = (4096, 4096);
        let p = |ll, e| {
            estimate_encode(w, h, 3, false, ll, e)
                .unwrap()
                .peak_memory_bytes
        };
        let lossy7 = p(false, 7);
        let lossy9 = p(false, 9);
        let (ll5, ll6, ll7, ll10) = (p(true, 5), p(true, 6), p(true, 7), p(true, 10));
        // The buttloop adds little memory: e9 ≥ e7 but well under 2× (it is
        // NOT a step). Asserting the absence of a phantom step guards against
        // re-inflating the buttloop band.
        assert!(
            lossy9 >= lossy7,
            "lossy e9 (buttloop band) at least e7 base"
        );
        assert!(
            lossy9 < lossy7 * 2,
            "lossy buttloop must NOT be a 2× memory step (measured ≈ e7 working set)"
        );
        // Lossless ramps e5 < e6 < e7 (e6 ≈ 215 B/px partial-search,
        // e7 ≈ 430 full tree-learning) and steps again at e10
        // (fine_grained_step, ~620 B/px).
        assert!(ll6 > ll5, "lossless e6 above e5 base");
        assert!(ll7 > ll6, "lossless full tree-learning step at e7");
        assert!(ll10 > ll7, "lossless e10 step above the e7-e9 band");
        assert!(ll7 > lossy9, "lossless e7 heavier than lossy e9");
    }

    /// Encode time is effort-aware (the flat-throughput model was wrong):
    /// higher effort → strictly more time, lossless e9 ≫ e1, and lossless
    /// is slower than lossy at high effort.
    #[test]
    fn encode_time_effort_aware() {
        let (w, h) = (2048, 2048);
        let t = |ll, e| estimate_encode(w, h, 3, false, ll, e).unwrap().time_ms;
        assert!(
            t(false, 9) > t(false, 5) && t(false, 5) > t(false, 1),
            "lossy time ↑ with effort"
        );
        assert!(
            t(true, 9) > t(true, 7) && t(true, 7) > t(true, 3),
            "lossless time ↑ with effort"
        );
        // lossless e9 measured ~300× e1 per pixel — assert a big gap.
        assert!(
            t(true, 9) > t(true, 1) * 100.0,
            "lossless e9 ≫ e1, got {}",
            t(true, 9) / t(true, 1)
        );
        assert!(t(true, 9) > t(false, 9), "lossless e9 slower than lossy e9");
        // effort below/above the measured range clamps (no panic, monotone).
        assert_eq!(
            t(false, 1),
            estimate_encode(w, h, 3, false, false, 0).unwrap().time_ms
        );
    }

    /// 2026-08-01 large-size anchors (12–108 MP zensysbench corpus,
    /// `benchmarks/jxl_encode_mem_threads_postfix_2026-08-01.tsv`,
    /// measured AFTER the per-tile AC-strategy quadratic fix): the TYP
    /// estimate must COVER every measured cell — under-prediction is what
    /// admitted the 108 MP encodes that kernel-OOM-killed 32 GiB fleet
    /// boxes — without inflating past the envelope's purpose (e5 is
    /// content-tight, ≤ 2×; e7's band is an envelope of a 2.1× content
    /// spread, so its ceiling is looser at 3× on this content-light
    /// photo).
    /// The LOSSLESS bands had no measured-cell coverage — only lossy did, via
    /// the 108 MP test below — and the lossless e7-e9 band was found
    /// under-predicting at 4K on 2026-08-13.
    ///
    /// Cells are the ALLOCATOR-AGNOSTIC peak (high-water mark of live
    /// allocated bytes, from the counting global allocator in zenjxl's
    /// examples/mem_probe_encode), NOT peak RSS. RSS is the wrong thing to
    /// pin an estimate against: it folds in whatever the platform allocator
    /// declined to return, so it swung 4161-4522 MB run-to-run on the same
    /// binary and input here, while `peak_live` held at 3141 MB across both
    /// runs AND both content classes. Pinning to RSS would make this test
    /// flaky and would encode one allocator's retention policy as a codec
    /// constant.
    ///
    /// Measured 3840x2160 (8.29 MP), RGB8, threads=1, worst case over
    /// {photo, screen}, at jxl-encoder 3e237d11:
    ///   lossless e7  peak_live  873 MB (RSS 1433)  e9  peak_live 1104 MB (RSS 1785)
    /// The peak is the gather/dedup working set at its minimal exact layout
    /// (per-site attribution in benchmarks/jxl_alloc_sites_4k_2026-08-13.md):
    /// property columns freed-before-dedup (b22d122e), width-halved
    /// (PropColumn i16, 8b9b6121), masked to the configured set (21684778);
    /// dedup keys packed at the rounded width (d1074adc) then partitioned by
    /// the two lead bytes (5119668d); both whole-image copies freed before
    /// tree learning (3e237d11).
    ///   lossy    e3  peak_live  412 MB      lossy    e9  peak_live  517 MB
    ///
    /// Keeping pre-fix numbers here would leave the gate looser than the
    /// encoder now warrants, so it would stop catching a regression that gave
    /// the reduction back. Re-measure and re-pin whenever an encode-path
    /// buffer changes.
    /// Provenance: benchmarks/jxl_ceiling_peaklive_4k_2026-08-13.tsv.meta +
    /// benchmarks/jxl_alloc_sites_4k_2026-08-13.md.meta.
    ///
    /// The MAX tier is separately required to clear the measured peak RSS, so
    /// a caller sizing a hard cap from `peak_memory_bytes_max` still survives
    /// an allocator that retains aggressively.
    #[test]
    fn estimate_covers_measured_4k_cells_2026_08_13() {
        const MB: u64 = 1024 * 1024;
        // (w, h, is_lossless, effort, measured peak_live, measured peak RSS)
        let cells: &[(u32, u32, bool, u8, u64, u64)] = &[
            (3840, 2160, true, 7, 873 * MB, 1433 * MB),
            (3840, 2160, true, 9, 1104 * MB, 1785 * MB),
            (3840, 2160, false, 3, 412 * MB, 429 * MB),
            (3840, 2160, false, 9, 517 * MB, 697 * MB),
        ];
        for &(w, h, lossless, effort, live, rss) in cells {
            let e = estimate_encode(w, h, 3, false, lossless, effort).unwrap();
            assert!(
                e.peak_memory_bytes >= live,
                "lossless={lossless} e{effort}: TYP {} under measured peak_live {live}",
                e.peak_memory_bytes
            );
            assert!(
                e.peak_memory_bytes_max >= rss,
                "lossless={lossless} e{effort}: MAX {} under measured peak RSS {rss}",
                e.peak_memory_bytes_max
            );
        }
    }

    #[test]
    fn estimate_covers_measured_large_sizes_2026_08() {
        // 108 MP lossy e5 t=1: measured marginal 5_602_944 KiB + 324 MB input.
        let m_e5 = 5_602_944u64 * 1024 + 12000 * 9000 * 3;
        let e5 = estimate_encode(12000, 9000, 3, false, false, 5).unwrap();
        assert!(
            e5.peak_memory_bytes >= m_e5,
            "e5 108MP TYP {} under measured {m_e5}",
            e5.peak_memory_bytes
        );
        assert!(
            e5.peak_memory_bytes < m_e5 * 2,
            "e5 108MP TYP {} not tight (≥2× measured {m_e5})",
            e5.peak_memory_bytes
        );
        // 108 MP lossy e7 t=1 d6.0: measured marginal 6_342_420 KiB.
        let m_e7 = 6_342_420u64 * 1024 + 12000 * 9000 * 3;
        let e7 = estimate_encode(12000, 9000, 3, false, false, 7).unwrap();
        assert!(
            e7.peak_memory_bytes >= m_e7,
            "e7 108MP TYP {} under measured {m_e7}",
            e7.peak_memory_bytes
        );
        assert!(
            e7.peak_memory_bytes < m_e7 * 3,
            "e7 108MP TYP {} runaway (≥3× measured {m_e7})",
            e7.peak_memory_bytes
        );
        // 12 MP lossless e7 t=1: measured marginal 5_879_984 KiB (the
        // 2026-08-01 corpus photo, 1.4× the 2026-06-23 one — β re-anchored).
        let m_ll = 5_879_984u64 * 1024 + 4000 * 3000 * 3;
        let ll = estimate_encode(4000, 3000, 3, false, true, 7).unwrap();
        assert!(
            ll.peak_memory_bytes >= m_ll,
            "lossless e7 12MP TYP {} under measured {m_ll}",
            ll.peak_memory_bytes
        );
        assert!(
            ll.peak_memory_bytes < m_ll * 2,
            "lossless e7 12MP TYP {} not tight",
            ll.peak_memory_bytes
        );
        // Threaded: the lossy per-thread term is measured (2.5 MB/thread,
        // 12 MP e7 t=1→16) and unclamped past max_useful_threads.
        let t1 = estimate_encode_threaded(4000, 3000, 3, false, false, 7, 1)
            .unwrap()
            .peak_memory_bytes;
        let t16 = estimate_encode_threaded(4000, 3000, 3, false, false, 7, 16)
            .unwrap()
            .peak_memory_bytes;
        assert_eq!(t16 - t1, 2_500_000 * 15, "unclamped per-thread term");
    }

    /// Alpha adds a substantial lossless per-pixel term (~+230 B/px, the
    /// 4th channel through tree-learning) but is negligible for lossy.
    #[test]
    fn alpha_term_lossless_only() {
        let (w, h) = (4096, 4096);
        let px = (w as u64) * (h as u64);
        // lossless: rgba working set ~+230 B/px over rgb (input_bpp held
        // equal to isolate the working-set term, not the input buffer).
        let ll_rgb = estimate_encode(w, h, 3, false, true, 7)
            .unwrap()
            .peak_memory_bytes;
        let ll_rgba = estimate_encode(w, h, 3, true, true, 7)
            .unwrap()
            .peak_memory_bytes;
        let delta = ll_rgba - ll_rgb;
        assert!(
            delta >= px * 200 && delta <= px * 260,
            "lossless alpha term {delta} not ~230 B/px of {px}px"
        );
        // lossy: alpha is folded into nothing (only the input byte counts,
        // held equal here), so the working set is unchanged.
        let lossy_rgb = estimate_encode(w, h, 3, false, false, 9)
            .unwrap()
            .peak_memory_bytes;
        let lossy_rgba = estimate_encode(w, h, 3, true, false, 9)
            .unwrap()
            .peak_memory_bytes;
        assert_eq!(
            lossy_rgb, lossy_rgba,
            "lossy alpha must not change working set"
        );
    }
}
