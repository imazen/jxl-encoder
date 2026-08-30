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
/// (no tree learning), e6 (heavier partial-search), e7–e10 full MA
/// tree-learning (new e10 = e9 single-seed since the 2026-08-29 ladder
/// shift), e ≥ 11 (multi-seed / config-trial tiers — measured at their
/// pre-shift "e10" label below). Unlike lossy, the lossless working set is
/// ~size-independent (1 MP ≈ 12 MP measured), so these B/px hold at the
/// large sizes where memory matters.
///
/// 2026-08-28 RECALIBRATION (imazen/jxl-encoder#96 follow-up,
/// `benchmarks/jxl_lossless_band_2026-08-28.{tsv,meta}`): the thirteen
/// August memory reductions (dedup partitioning, probe-tree predictor
/// pruning, exact gather-time bucketization, wire-stream token reuse, the
/// patches-scan fix, …) left the 2026-08-01 anchors (`base 92 / e6 235 /
/// tree 540 / e10 620` B/px, from a 490 B/px 12 MP e7 cell) 3–5× stale-
/// high, so `Auto`'s memory-pressure gate and `estimate_encode` over-
/// predicted the global path by that factor. Re-measured with the
/// allocator-agnostic `peak_live` probe, threads=1, three content classes
/// (imazen-26 1403 photo 64² → 12 MP crops, gb82-sc imac_dark 5.6 MP,
/// qoi reddit.com 1313×4096 / ×8008), marginal B/px:
///
/// | effort | photo 1 MP | photo 4 MP | photo 8.3 MP | photo 12 MP | imac | reddit |
/// |---|---|---|---|---|---|---|
/// | e5 | 73.1 | — | — | 70.3 | 66.7 | 66.7 |
/// | e6 | 79.4 | — | — | 72.3 | 62.2 | 67.8 |
/// | e7 | 105.5 | 98.1 | 96.4 | 94.6 | 85.6 | 86.6 / 85.6 |
/// | e8 | 114.1 | — | — | 100.2 | 87.3 | 87.9 |
/// | e9 | 141.3 | 121.2 | 127.0 | 129.5 | 137.7 | 129.7 / 130.2 |
/// | e10 | 236.0 | 129.1 | — | — | — | — |
///
/// The bands are the per-effort envelopes with ≥ 10 % margin (e6 keeps a
/// small step over e5 — measured 79 vs 73 at 1 MP; e7 and e8 share one;
/// e9 is its own). The intercept
/// is effort-dependent (256² cells: e7 20.5 MB, e8 25.9 MB, e9 67.3 MB,
/// pre-shift-e10 67.3 MB; 64² e7 6.0 MB) — see [`lossless_fixed_overhead`].
/// The multi-seed learn (pre-shift e10 label = today's e11) carries a
/// ~150 MB size-independent term on top of a ≈ 91 B/px slope (1 → 4 MP
/// fit), modelled as the 160 MiB intercept + the e9 slope. Lossless stays
/// thread-invariant in the model (γ = 0): the 2026-08-27 sweep measured
/// threads ≥ 4 at or below t=1.
const LOSSLESS_BPP_BASE: f64 = 92.0;
const LOSSLESS_BPP_E6: f64 = 100.0;
const LOSSLESS_BPP_TREE: f64 = 128.0;
const LOSSLESS_BPP_E9: f64 = 160.0;
const LOSSLESS_BPP_E10: f64 = 160.0;

/// Lossless size-independent term by effort band (2026-08-28 grid, see
/// the band note above): the tree learner's fixed working set grows with
/// the effort's params (e9's split workspace + tensors ≈ 64 MB even on a
/// 256² image; e10's multi-seed ≈ 150 MB).
#[must_use]
fn lossless_fixed_overhead(effort: u8) -> u64 {
    // 2026-08-29 ladder shift (issue #45): multi-seed tree learning (the
    // ~150 MB size-independent term measured on the pre-shift e10) now
    // starts at e11; new e10 is single-seed and shares the e9 band. The
    // e11 TectonicPlate config trial runs its trials sequentially, so its
    // peak stays one trial's working set — the multi-seed intercept
    // remains the envelope.
    if effort >= 11 {
        160 << 20
    } else if effort >= 9 {
        64 << 20
    } else if effort >= 7 {
        24 << 20
    } else {
        LOSSLESS_FIXED_OVERHEAD
    }
}

/// Extra marginal working set for a lossless encode with an alpha
/// extra-channel (the 4th channel goes through tree-learning). Re-measured
/// 2026-08-28 (rgba − rgb marginal, alpha := the source's green plane,
/// photo 1403): e7 +37.3 B/px at 1 MP / +33.5 at 8.3 MP, e9 +62.8 / +27.1
/// — envelope +62.8, set with margin (was 230 from the 2026-06-14 sweep,
/// pre-reductions). The lossy path's alpha (modular extra-channel
/// alongside VarDCT) adds only +2…5 B/px — within noise — so there is no
/// lossy alpha term; the alpha input buffer's extra byte/px is already
/// counted via `input_bpp`.
const LOSSLESS_BPP_ALPHA: f64 = 72.0;

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
        // Post-2026-08-29-shift: the multi-seed slope band starts at e11
        // (new e10 is single-seed, e9-equivalent). The two constants are
        // currently equal (160 B/px), so this boundary only matters if
        // they diverge on a future re-measure.
        if effort >= 11 {
            LOSSLESS_BPP_E10
        } else if effort >= 9 {
            LOSSLESS_BPP_E9
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
///   lossless this adds a per-pixel term (the 4th channel goes through
///   tree-learning, +72 B/px band); for lossy it is negligible.
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
        lossless_fixed_overhead(effort)
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

// ── Sectioned local-tree lossless mode (imazen/jxl-encoder#96) ─────────────

/// Effort band the sectioned estimate is calibrated for: the tree-learning
/// efforts 7–9 (the sectioned path only exists where tree learning runs,
/// and e ≥ 10 was not swept — callers keep the whole-image band there).
#[must_use]
pub(crate) fn sectioned_estimate_available(effort: u8) -> bool {
    (7..=9).contains(&effort)
}

/// Sectioned-mode fixed overhead by effort: the effort's tree-learn
/// scaffolding that is NOT proportional to the group's pixel count
/// (histogram tables, property/predictor bookkeeping, the split-search
/// workspaces). Measured 64×64 real-photo crop, threads=1 (2026-08-27,
/// `benchmarks/jxl_sectioned_mem_2026-08-27.tsv`): e7 5.9 MiB, e9 25.3 MiB
/// peak_live. Set with margin; e8 shares the e9 value (unswept, e9 is the
/// band's top). It sits OUTSIDE the phase max below because it is live
/// across both phases, and it is what keeps the sub-group-sized cells
/// (64², 256²) covered.
const SECTIONED_FIXED_E7: u64 = 8 << 20;
const SECTIONED_FIXED_E9: u64 = 32 << 20;

// ── The phase decomposition (re-derived 2026-08-30, issue #99 item 3) ──
//
// The sectioned peak is the MAXIMUM of two phases that do not overlap, not
// their sum:
//
//   1. the PRE-TREE floor — the `select_best_rct` trial wave (and, at one
//      worker, the patches detector's internals) over the `ModularImage`;
//   2. the TREE-LEARN phase — the resident `ModularImage` plus one group's
//      learn working set per in-flight worker.
//
// The thread-dense grid this is fitted on
// (`benchmarks/jxl_sectioned_thread_dense_2026-08-30.tsv`, 192 cells × 3
// repeats, threads {1,2,3,4,6,8,12}) shows the max shape directly: photo
// 12 MP measures 48.0 B/px marginal at t=2 AND at t=12, at BOTH e7 and e9
// — the twelve per-group learn sets are entirely UNDER the floor there, so
// they cost nothing. The same cells at 1024² (16 groups) are the opposite:
// the learn phase out-peaks the floor from t=2 and the estimate must track
// it all the way to 443 B/px at t=12.
//
// The model this replaced ADDED the two, which is why it over-predicted
// palette content by 2.8× at 12 workers (imac_dark e9 t12: TYP 847.9 MB vs
// 305.9 MB measured) — and, less visibly, UNDER-predicted 12 of the 192
// cells (all of them 1–9 groups, worst photo 256² e9 t1 at 0.54×: TYP
// 37.0 MB for a measured 68.6 MB peak). Under-prediction is the unsafe
// direction: admission sizes a cap from the estimate.

/// The i32 `ModularImage` plane, per colour channel — live through BOTH
/// phases, so it is the floor of each arm. `channel.rs` stores i32
/// samples, hence exactly 4 B/px/channel; the measured floors below are
/// integer multiples of it, which is what identifies each phase's
/// plane count.
const SECTIONED_RESIDENT_BPP_PER_CHANNEL: f64 = 4.0;

/// Pre-tree floor at ≥ 2 workers, per colour channel: the
/// `select_best_rct` trial wave. `RCT_TRIAL_WAVE = 2` in-flight trials +
/// the running best + the image = 4 i32 planes per channel = **16.0
/// B/px/channel**, which is what the grid measures to three digits —
/// 48.0 B/px on rgb at 2048²/3840×2160/4000×3000/reddit and 64.0 B/px on
/// rgba 3840×2160, at every thread count from 2 to 12 and at both
/// efforts. 22.0 carries a 1.375× margin over that for the meta channels
/// and the patches dictionary palette content adds on top (imac_dark
/// measures 48.2).
const SECTIONED_WAVE_BPP_PER_CHANNEL: f64 = 22.0;

/// Pre-tree floor at ONE worker, ABOVE the resident image: the patches
/// detector's internals. Single-worker pools price RCT candidates with
/// the streaming evaluator (`estimate_cost_rct_streaming`, 2026-08-30) so
/// the wave above does not exist there, and the peak instant moves into
/// detection. Measured marginal floors at t=1: photo 24.0–29.2 B/px,
/// reddit 37.2–38.1, imac_dark 38.6 — i.e. 12.0–26.6 B/px above the
/// 12 B/px resident. 38.0 covers the worst measured content at 1.43×;
/// the t=1 arm is the ADMISSION floor (`min` over 1 and 2 workers), so
/// its margin buys rejection-safety on content that was never swept.
const SECTIONED_DETECT_BPP_THREADS1: f64 = 38.0;

/// Per-worker term: one in-flight group's tree-learn working set. Fitted
/// as the smallest constant that covers every cell of the thread-dense
/// grid given the rest of the model (`scripts/mem_sectioned_model_fit.py`
/// prints the binding cell): e7 needs 12.10 MiB (photo 1024² t8), e9
/// needs 33.32 MiB (photo 1024² t2). The chosen values carry 1.49× / 1.38×
/// over those — deliberately more than the 1.16× worst repeat spread the
/// grid measures on tree-learn-bound cells, since the constant is
/// content-blind and photo is the most expensive content swept.
///
/// This is ~30 % ABOVE the old additive term (12/36 MiB) even though the
/// model as a whole is far tighter: the old term only had to explain the
/// growth left over after the floor was added underneath it, whereas here
/// the learn arm must carry the whole tree-learn phase on its own.
const SECTIONED_PER_THREAD_E7: u64 = 18 << 20;
const SECTIONED_PER_THREAD_E9: u64 = 46 << 20;

/// Alpha per-worker factor (rgba − rgb, 2026-08-27 + 2026-08-30 rgba
/// cells): the 4-channel group learn grows ×1.23 (e7) / ×1.44 (e9) at
/// 1024², t=1→8. The per-PIXEL arms need no separate alpha term any more
/// — both are per-channel constants, and the measured rgba floors are
/// exactly 4/3 of the rgb ones (64.0 vs 48.0 B/px at t ≥ 2).
const SECTIONED_PER_THREAD_ALPHA_NUM: u64 = 3;
const SECTIONED_PER_THREAD_ALPHA_DEN: u64 = 2;

/// Modular group dimension the in-flight clamp assumes: the
/// `modular_group_size_shift` default (shift 1 → `128 << 1`). A larger
/// group makes each learn set bigger and the group count smaller; the
/// knob is `pub(crate)` plumbing with no public setter, and neither this
/// model nor the one it replaced takes it as an input — flagged here so a
/// future public knob does not silently invalidate the calibration.
const SECTIONED_GROUP_DIM: u64 = 256;

/// The learn phase cannot hold more group sets than the image has groups.
/// The slack is what a single-group image still gains from the
/// `parallel-tree-learning` fork engine: photo 256² (one group) measures
/// 68.6 MB at t=1 and 102.3 MB from t=2 onward — flat through t=12 — i.e.
/// exactly one extra set, then nothing. Two is carried rather than one
/// because it also levels the fitted per-worker requirement across the
/// 1/2/4/9-group ladder (39.19 MiB at slack 1, driven by the 512² t12
/// cell, versus 33.32 at slack 2), which says the fork engine is worth
/// closer to two sets than one; the difference only ever RAISES the
/// estimate, and only below 2 MP where the clamp can bind at all.
const SECTIONED_INFLIGHT_SLACK: u64 = 2;

/// Measured slope of the FLAT region — the per-worker pool bookkeeping
/// that survives when neither phase grows. Photo 12 MP e7 t2→t12
/// 597864 → 597937 KiB, imac_dark e7 281061 → 281134, reddit 523754 →
/// 523797, photo 2048² e7 209013 → 209056: **+7.3 KiB per worker** on all
/// four, independent of size, content and effort. 16 KiB carries 2.2×.
///
/// It is small, but it is not decoration: it is what keeps the estimate
/// STRICTLY monotone in the thread count across a plateau, which the
/// pre-flight's thread walk-down and
/// `api_tests::encode_preflight_sectioned` rely on. A `max()` of two
/// terms is only non-decreasing; this makes it increasing without
/// inventing memory that was not measured.
const SECTIONED_POOL_BYTES_PER_THREAD: u64 = 16 << 10;

/// Peak-memory estimate for a lossless encode that runs the SECTIONED
/// local-tree mode ([`crate::api::SectionedTrees`] `On`, or `Auto` where
/// its gate engages — imazen/jxl-encoder#96) at `cores` worker threads.
///
/// ```text
/// peak = input + fixed(effort)
///      + max( pre_tree_floor(cores, channels) · pixels,
///             resident(channels) · pixels
///                 + per_worker(effort, channels) · in_flight(cores, pixels) )
///      + pool · (cores − 1)
/// ```
///
/// The `max` is the whole point: the pre-tree floor and the per-group
/// tree learns are consecutive phases, not concurrent ones (see the
/// constants above for the measurement that shows it). `in_flight` is
/// `min(cores, groups + slack)` — a worker cannot hold a group set for a
/// group that does not exist.
///
/// All terms are measured on real content (photo + two screenshot classes,
/// 64² → 12 MP, threads 1/2/3/4/6/8/12, e7/e9, rgb + rgba;
/// `benchmarks/jxl_sectioned_thread_dense_2026-08-30.{tsv,meta}`, fitted
/// and re-verified by `scripts/mem_sectioned_model_fit.py`). Only valid
/// where [`sectioned_estimate_available`] holds; `time_ms` /
/// `output_bytes` are the whole-image figures (the mode is measured
/// byte-neutral at the median and faster, so they remain upper bounds).
///
/// **Residual error is one-sided.** Over the 192-cell grid TYP covers
/// every cell (worst 1.10× at 512² e9 t12) and over-predicts by at most
/// 2.28× at ≥ 2 MP (imac_dark e9 t12, whose palette groups learn far
/// cheaper trees than the photo content the content-blind per-worker term
/// must cover) and 2.26× at 2048² e9 t1 (the conservative one-worker
/// admission floor). Below 2 MP the over-prediction reaches 8.6× on
/// trivially-compressible screen crops, where the absolute figure is tens
/// of MB. Over-prediction costs capacity; under-prediction would let
/// admission size a cap the encode then exceeds, so the constants are set
/// on the worst content swept, not the median.
///
/// Returns `None` only on dimension overflow.
#[must_use]
pub(crate) fn estimate_encode_sectioned(
    width: u32,
    height: u32,
    input_bpp: u8,
    has_alpha: bool,
    effort: u8,
    cores: usize,
) -> Option<EncodeEstimate> {
    debug_assert!(sectioned_estimate_available(effort));
    let base = estimate_encode(width, height, input_bpp, has_alpha, true, effort)?;
    let pixels = (width as u64).checked_mul(height as u64)?;
    let input = pixels.checked_mul(input_bpp as u64)?;
    let cores = cores.max(1) as u64;
    // Modular channel count: colour + alpha. NOT `input_bpp` — a 16-bit
    // rgb buffer is 6 B/px of input but still three i32 planes.
    let channels = if has_alpha { 4.0 } else { 3.0 };
    let (fixed, mut per_thread) = if effort >= 8 {
        (SECTIONED_FIXED_E9, SECTIONED_PER_THREAD_E9)
    } else {
        (SECTIONED_FIXED_E7, SECTIONED_PER_THREAD_E7)
    };
    if has_alpha {
        per_thread = per_thread * SECTIONED_PER_THREAD_ALPHA_NUM / SECTIONED_PER_THREAD_ALPHA_DEN;
    }

    // Phase 1 — pre-tree floor. At one worker the RCT candidates are
    // priced by the streaming evaluator (no trial clones) and the peak
    // instant is inside patches detection; from two workers the trial
    // wave is what sets it.
    let floor_bpp = if cores > 1 {
        SECTIONED_WAVE_BPP_PER_CHANNEL * channels
    } else {
        SECTIONED_RESIDENT_BPP_PER_CHANNEL * channels + SECTIONED_DETECT_BPP_THREADS1
    };
    let floor = (pixels as f64 * floor_bpp) as u64;

    // Phase 2 — per-group tree learn, one set per in-flight worker over
    // the resident image.
    let groups = (width as u64)
        .div_ceil(SECTIONED_GROUP_DIM)
        .checked_mul((height as u64).div_ceil(SECTIONED_GROUP_DIM))?;
    let in_flight = cores.min(groups.saturating_add(SECTIONED_INFLIGHT_SLACK));
    let learn = ((pixels as f64 * SECTIONED_RESIDENT_BPP_PER_CHANNEL * channels) as u64)
        .checked_add(per_thread.checked_mul(in_flight)?)?;

    let working = fixed
        .checked_add(floor.max(learn))?
        .checked_add(SECTIONED_POOL_BYTES_PER_THREAD.checked_mul(cores - 1)?)?;
    let typical = input.checked_add(working)?;
    let min = input + (working as f64 * MULT_MIN) as u64;
    let max = input.checked_add((working as f64 * MULT_MAX) as u64)?;
    Some(EncodeEstimate {
        peak_memory_bytes_min: min,
        peak_memory_bytes: typical,
        peak_memory_bytes_max: max,
        time_ms: base.time_ms,
        output_bytes: base.output_bytes,
    })
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
        // lossless e7: measured 1_108_269 KiB working + ~34 MB input
        // (2026-08-28 re-measure of the same photo; the 2026-06-23 figure
        // was 4186 MiB, before the August reductions).
        let ll = estimate_encode(px12.0, px12.1, 3, false, true, 7).unwrap();
        let measured_ll = 1_108_269u64 * 1024 + input12;
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
        let (ll5, ll6, ll7, ll11) = (p(true, 5), p(true, 6), p(true, 7), p(true, 11));
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
        // Lossless ramps e5 < e6 < e7 (2026-08-28 bands: e6 100 B/px
        // partial-search, e7 128 full tree-learning) and steps again at
        // e11 (multi-seed post-2026-08-29-shift: 160 B/px + a 160 MiB
        // intercept).
        assert!(ll6 > ll5, "lossless e6 above e5 base");
        assert!(ll7 > ll6, "lossless full tree-learning step at e7");
        assert!(ll11 > ll7, "lossless e11 step above the e7-e10 band");
        // NOTE: the pre-2026-08-28 `ll7 > lossy9` ordering held only
        // because the lossless band was 4-5× stale-high; the lossy e ≥ 7
        // band (135 B/px, a d6.0 envelope) now sits above the re-anchored
        // lossless e7 band (128 B/px) in the MODEL even though the
        // measured 12 MP peaks still order lossless e7 (1.14 GB) above
        // lossy e9 (0.94 GB) — see `estimate_safely_covers_measured_12mp`.
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
    ///   lossy    e3  peak_live  318 MB (RSS 429)  e9  peak_live  434 MB (RSS 714)
    /// (lossy re-measured at 4b464975 after the patches-DFS/linear-lifetime
    /// fixes; worst over {photo, screen}, loop-free probe build)
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
            (3840, 2160, false, 3, 318 * MB, 429 * MB),
            (3840, 2160, false, 9, 434 * MB, 714 * MB),
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
        // 12 MP lossless e7 t=1: measured marginal 1_108_269 KiB (the
        // 2026-08-01 corpus photo re-measured 2026-08-28 after the August
        // reductions — the 2026-08-01 figure was 5_879_984 KiB; β
        // re-anchored, `benchmarks/jxl_lossless_band_2026-08-28.tsv`).
        let m_ll = 1_108_269u64 * 1024 + 4000 * 3000 * 3;
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

    /// Whole-image (global-tree) lossless band, 2026-08-28 recalibration
    /// grid (`benchmarks/jxl_lossless_band_2026-08-28.{tsv,meta}`,
    /// threads=1, allocator-agnostic peak_live incl. the input buffer):
    /// TYP covers every cell and is a tight cover (< 2.5×) at ≥ 1 MP.
    #[test]
    fn lossless_estimate_covers_measured_cells_2026_08_28() {
        const KB: u64 = 1024;
        // (w, h, has_alpha, effort, measured peak_live KiB)
        let cells: &[(u32, u32, bool, u8, u64)] = &[
            // photo 1403 crops
            (64, 64, false, 7, 6010),
            (256, 256, false, 7, 20745),
            (1024, 1024, false, 7, 111109),
            (2048, 2048, false, 7, 413912),
            (3840, 2160, false, 7, 804904),
            (4000, 3000, false, 7, 1143426),
            (1024, 1024, false, 5, 77966),
            (4000, 3000, false, 5, 859031),
            (1024, 1024, false, 6, 84396),
            (4000, 3000, false, 6, 882621),
            (256, 256, false, 8, 26103),
            (1024, 1024, false, 8, 119959),
            (4000, 3000, false, 8, 1209079),
            (256, 256, false, 9, 67316),
            (1024, 1024, false, 9, 147813),
            (2048, 2048, false, 9, 508750),
            (3840, 2160, false, 9, 1053230),
            (4000, 3000, false, 9, 1553020),
            // Measured 2026-08-28 at the pre-shift "e10" label (2-seed
            // multi-seed tree learn) — that behaviour lives at e11 since
            // the 2026-08-29 ladder shift (issue #45), so the cells pin
            // the e11 band. Post-shift e10 is single-seed and covered by
            // the e9 rows above.
            (256, 256, false, 11, 67333),
            (1024, 1024, false, 11, 244742),
            (2048, 2048, false, 11, 541108),
            // rgba (alpha := green)
            (1024, 1024, true, 7, 150367),
            (1024, 1024, true, 9, 213110),
            (3840, 2160, true, 7, 1084576),
            (3840, 2160, true, 9, 1281103),
            // gb82-sc imac_dark (palette + patches screenshot).
            // e6/e7/e8 re-pinned 2026-08-30 (patches-phase lifetime fix,
            // jxl_sectioned_patches_lifetime_2026-08-30.tsv: detection
            // runs before the ModularImage build): the light-effort
            // global peaks carried part of the detection working set —
            // e6 was 357844, e7 486110, e8 495594 (e7 includes ~1 MiB of
            // pre-existing drift: 484850 measured at 29be5e32 both
            // before and after the fix). e5/e9 unchanged, verified.
            (2940, 1912, false, 5, 382546),
            (2940, 1912, false, 6, 353001),
            (2940, 1912, false, 7, 484850),
            (2940, 1912, false, 8, 492569),
            (2940, 1912, false, 9, 772396),
            // qoi reddit.com screenshot. Same 2026-08-30 re-pin: e5 was
            // 715897, e6 727010, e7 909647 (904584 already at 29be5e32
            // pre-fix — drift), e8 933119, 4096-crop e7 470541; e9 cells
            // unchanged, verified.
            (1313, 4096, false, 7, 467733),
            (1313, 8008, false, 5, 715316),
            (1313, 8008, false, 6, 723394),
            (1313, 8008, false, 7, 904584),
            (1313, 8008, false, 8, 927365),
            (1313, 4096, false, 9, 696734),
            (1313, 8008, false, 9, 1367236),
        ];
        for &(w, h, alpha, effort, live_kb) in cells {
            let bpp = if alpha { 4 } else { 3 };
            let e = estimate_encode(w, h, bpp, alpha, true, effort).unwrap();
            let live = live_kb * KB;
            assert!(
                e.peak_memory_bytes >= live,
                "{w}x{h} alpha={alpha} e{effort}: TYP {} under measured peak_live {live}",
                e.peak_memory_bytes
            );
            if (w as u64) * (h as u64) >= 1_000_000 {
                assert!(
                    e.peak_memory_bytes < live * 5 / 2,
                    "{w}x{h} alpha={alpha} e{effort}: TYP {} not a tight cover of {live} (≥ 2.5×)",
                    e.peak_memory_bytes
                );
            }
        }
    }

    /// Alpha adds a lossless per-pixel term (+72 B/px band over the
    /// measured +27…63, the 4th channel through tree-learning) but is
    /// negligible for lossy.
    #[test]
    fn alpha_term_lossless_only() {
        let (w, h) = (4096, 4096);
        let px = (w as u64) * (h as u64);
        // lossless: rgba working set +72 B/px over rgb (input_bpp held
        // equal to isolate the working-set term, not the input buffer).
        let ll_rgb = estimate_encode(w, h, 3, false, true, 7)
            .unwrap()
            .peak_memory_bytes;
        let ll_rgba = estimate_encode(w, h, 3, true, true, 7)
            .unwrap()
            .peak_memory_bytes;
        let delta = ll_rgba - ll_rgb;
        assert!(
            delta >= px * 63 && delta <= px * 80,
            "lossless alpha term {delta} not ~72 B/px of {px}px"
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

    /// imazen/jxl-encoder#96 sectioned-mode cells — THREAD-DENSE grid,
    /// 2026-08-30, macOS/M-series laptop (12 cores), release build.
    /// Provenance: `benchmarks/jxl_sectioned_thread_dense_2026-08-30.tsv`
    /// and its `.meta` (`scripts/mem_sectioned_threads_sweep.sh`, REPEATS=3;
    /// probe `jxl-encoder-cli/examples/mem_probe`, allocator-agnostic
    /// `peak_live` = high-water of LIVE bytes across `encode()`, input
    /// buffer included). Real content only: imazen-26 png-v3 photo 1403
    /// (4000×3000) and its top-left crops; qoi-benchmark
    /// `screenshot_web/reddit.com.png` (1313×8008) and crops; gb82-sc
    /// `imac_dark.png` (2940×1912). `SectionedTrees::On`. rgba rows take
    /// alpha := the source's green plane (worst-case entropy).
    ///
    /// Each row is the MAXIMUM over three repeats AND over every content
    /// class measured at that geometry: the tree-learn-bound cells vary up
    /// to 1.16× run to run with worker scheduling (reddit 1313×4096 e9 t12
    /// measured 330729 / 345335 / 384245 KiB in three consecutive runs), so
    /// a single sample under-states what the estimate must cover. Rows
    /// marked as such carry a higher figure from the 2026-08-27/30 sweeps.
    ///
    /// Contract, in order of importance:
    ///
    /// 1. **TYP covers every cell.** Under-prediction is a correctness bug:
    ///    admission (and the thread walk-down) size a cap from TYP, so an
    ///    encode that is let in then exceeds it. The additive model this
    ///    grid replaced under-predicted 12 of these cells — every one of
    ///    them 1–9 groups, worst 0.54× at photo 256² e9 t1 — which is why
    ///    the small-crop ladder (64² / 256² / 512×256 / 512² / 768²) is
    ///    pinned here and not just the large sizes.
    /// 2. **TYP stays a tight cover (< 2.5×) at ≥ 2 MP.** Over-prediction
    ///    only wastes capacity, but 2.5× of a 12 MP lossless working set is
    ///    hundreds of MB of admission headroom nobody can use.
    ///
    /// Re-measure and re-pin whenever the sectioned writer, the patches
    /// detector, the RCT trial wave or the modular image lifetime changes —
    /// and re-run `scripts/mem_sectioned_model_fit.py` over the new grid,
    /// which names the cell that binds each per-worker constant.
    #[test]
    fn sectioned_estimate_covers_measured_cells_2026_08_30() {
        const KB: u64 = 1024;
        // (w, h, has_alpha, effort, threads, measured peak_live KiB)
        let sectioned_cells: &[(u32, u32, bool, u8, usize, u64)] = &[
            // 64x64 — 0.00 MP, 1 group
            (64, 64, false, 7, 1, 6008),
            (64, 64, false, 7, 2, 7720),
            (64, 64, false, 7, 4, 7735),
            (64, 64, false, 7, 8, 7763),
            (64, 64, false, 7, 12, 7793),
            (64, 64, false, 9, 1, 25865),
            (64, 64, false, 9, 2, 30329),
            (64, 64, false, 9, 4, 30344),
            (64, 64, false, 9, 8, 30372),
            (64, 64, false, 9, 12, 30402),
            // 256x256 — 0.07 MP, 1 group
            (256, 256, false, 7, 1, 20742),
            (256, 256, false, 7, 2, 27487),
            (256, 256, false, 7, 4, 27501),
            (256, 256, false, 7, 8, 27530),
            (256, 256, false, 7, 12, 27556),
            (256, 256, false, 9, 1, 66944),
            (256, 256, false, 9, 2, 99860),
            (256, 256, false, 9, 4, 99874),
            (256, 256, false, 9, 8, 99903),
            (256, 256, false, 9, 12, 99933),
            // 512x256 — 0.13 MP, 2 groups
            (512, 256, false, 7, 1, 16389),
            (512, 256, false, 7, 2, 28942),
            (512, 256, false, 7, 4, 36230),
            (512, 256, false, 7, 8, 36318),
            (512, 256, false, 7, 12, 36288),
            (512, 256, false, 9, 1, 58320),
            (512, 256, false, 9, 2, 90003),
            (512, 256, false, 9, 4, 122058),
            (512, 256, false, 9, 8, 138132),
            (512, 256, false, 9, 12, 138051),
            // 512x512 — 0.26 MP, 4 groups
            (512, 512, false, 7, 1, 18310),
            (512, 512, false, 7, 2, 29642),
            (512, 512, false, 7, 4, 51740),
            (512, 512, false, 7, 8, 70655),
            (512, 512, false, 7, 12, 71496),
            (512, 512, false, 9, 1, 60240),
            (512, 512, false, 9, 2, 92692),
            (512, 512, false, 9, 4, 149777),
            (512, 512, false, 9, 8, 219704),
            (512, 512, false, 9, 12, 237413),
            // 768x768 — 0.59 MP, 9 groups
            (768, 768, false, 7, 1, 24616),
            (768, 768, false, 7, 2, 34308),
            (768, 768, false, 7, 4, 61389),
            (768, 768, false, 7, 8, 101840),
            (768, 768, false, 7, 12, 119279),
            (768, 768, false, 9, 1, 65042),
            (768, 768, false, 9, 2, 105877),
            (768, 768, false, 9, 4, 169615),
            (768, 768, false, 9, 8, 294780),
            (768, 768, false, 9, 12, 381569),
            // 1024x1024 — 1.05 MP, 16 groups
            (1024, 1024, false, 7, 1, 32276),
            (1024, 1024, false, 7, 2, 52293),
            (1024, 1024, false, 7, 3, 55629),
            (1024, 1024, false, 7, 4, 69321),
            (1024, 1024, false, 7, 6, 90449),
            (1024, 1024, false, 7, 8, 122777),
            (1024, 1024, false, 7, 12, 150954), // 2026-08-27/30 sweep: higher than this grid's own run
            (1024, 1024, false, 9, 1, 73994),
            (1024, 1024, false, 9, 2, 116383),
            (1024, 1024, false, 9, 3, 148259),
            (1024, 1024, false, 9, 4, 174776),
            (1024, 1024, false, 9, 6, 245328),
            (1024, 1024, false, 9, 8, 315152),
            (1024, 1024, false, 9, 12, 446851),
            // 1024x1024 rgba — 1.05 MP, 16 groups
            (1024, 1024, true, 7, 1, 43710),
            (1024, 1024, true, 7, 2, 69701),
            (1024, 1024, true, 7, 4, 91507),
            (1024, 1024, true, 7, 8, 159346),
            (1024, 1024, true, 7, 12, 199213),
            (1024, 1024, true, 9, 1, 100734),
            (1024, 1024, true, 9, 2, 140789),
            (1024, 1024, true, 9, 4, 221599),
            (1024, 1024, true, 9, 8, 404080), // 2026-08-27/30 sweep: higher than this grid's own run
            (1024, 1024, true, 9, 12, 589534),
            // 2048x2048 — 4.19 MP, 64 groups
            (2048, 2048, false, 7, 1, 110653),
            (2048, 2048, false, 7, 2, 209013),
            (2048, 2048, false, 7, 3, 209020),
            (2048, 2048, false, 7, 4, 209027),
            (2048, 2048, false, 7, 6, 209041),
            (2048, 2048, false, 7, 8, 209056),
            (2048, 2048, false, 7, 12, 232414),
            (2048, 2048, false, 9, 1, 110653),
            (2048, 2048, false, 9, 2, 209013),
            (2048, 2048, false, 9, 3, 209020),
            (2048, 2048, false, 9, 4, 209027),
            (2048, 2048, false, 9, 6, 256888),
            (2048, 2048, false, 9, 8, 300787),
            (2048, 2048, false, 9, 12, 417589),
            // 1313x4096 — 5.38 MP, 96 groups
            (1313, 4096, false, 7, 1, 211121),
            (1313, 4096, false, 7, 2, 267934),
            (1313, 4096, false, 7, 3, 267941),
            (1313, 4096, false, 7, 4, 267948),
            (1313, 4096, false, 7, 6, 267963),
            (1313, 4096, false, 7, 8, 267977),
            (1313, 4096, false, 7, 12, 268007),
            (1313, 4096, false, 9, 1, 211121),
            (1313, 4096, false, 9, 2, 267934),
            (1313, 4096, false, 9, 3, 267941),
            (1313, 4096, false, 9, 4, 267948),
            (1313, 4096, false, 9, 6, 267963),
            (1313, 4096, false, 9, 8, 304247),
            (1313, 4096, false, 9, 12, 384245),
            // 1313x4096 rgba — 5.38 MP, 96 groups
            (1313, 4096, true, 7, 8, 357262), // 2026-08-27/30 sweep: higher than this grid's own run
            // 2940x1912 — 5.62 MP, 96 groups
            (2940, 1912, false, 7, 1, 228308),
            (2940, 1912, false, 7, 2, 281061),
            (2940, 1912, false, 7, 3, 281068),
            (2940, 1912, false, 7, 4, 281076),
            (2940, 1912, false, 7, 6, 281090),
            (2940, 1912, false, 7, 8, 281104),
            (2940, 1912, false, 7, 12, 281134),
            (2940, 1912, false, 9, 1, 228308),
            (2940, 1912, false, 9, 2, 281061),
            (2940, 1912, false, 9, 3, 281068),
            (2940, 1912, false, 9, 4, 281076),
            (2940, 1912, false, 9, 6, 281090),
            (2940, 1912, false, 9, 8, 281104),
            (2940, 1912, false, 9, 12, 298692),
            // 3840x2160 — 8.29 MP, 135 groups
            (3840, 2160, false, 7, 1, 260680),
            (3840, 2160, false, 7, 2, 413301),
            (3840, 2160, false, 7, 3, 413308),
            (3840, 2160, false, 7, 4, 413315),
            (3840, 2160, false, 7, 6, 413329),
            (3840, 2160, false, 7, 8, 413344),
            (3840, 2160, false, 7, 12, 413373),
            (3840, 2160, false, 9, 1, 260680),
            (3840, 2160, false, 9, 2, 413301),
            (3840, 2160, false, 9, 3, 413308),
            (3840, 2160, false, 9, 4, 413315),
            (3840, 2160, false, 9, 6, 413329),
            (3840, 2160, false, 9, 8, 413344),
            (3840, 2160, false, 9, 12, 481735),
            // 3840x2160 rgba — 8.29 MP, 135 groups
            (3840, 2160, true, 7, 1, 291703),
            (3840, 2160, true, 7, 2, 551001),
            (3840, 2160, true, 7, 4, 551015),
            (3840, 2160, true, 7, 8, 551044),
            (3840, 2160, true, 7, 12, 551074),
            (3840, 2160, true, 9, 1, 291703),
            (3840, 2160, true, 9, 2, 551001),
            (3840, 2160, true, 9, 4, 551015),
            (3840, 2160, true, 9, 8, 551044),
            (3840, 2160, true, 9, 12, 650862),
            // 1313x8008 — 10.51 MP, 192 groups
            (1313, 8008, false, 7, 1, 422495),
            (1313, 8008, false, 7, 2, 523754),
            (1313, 8008, false, 7, 3, 523761),
            (1313, 8008, false, 7, 4, 523768),
            (1313, 8008, false, 7, 6, 523782),
            (1313, 8008, false, 7, 8, 523797),
            (1313, 8008, false, 7, 12, 523826),
            (1313, 8008, false, 9, 1, 422495),
            (1313, 8008, false, 9, 2, 523754),
            (1313, 8008, false, 9, 3, 523761),
            (1313, 8008, false, 9, 4, 523768),
            (1313, 8008, false, 9, 6, 523782),
            (1313, 8008, false, 9, 8, 523797),
            (1313, 8008, false, 9, 12, 523826),
            // 4000x3000 — 12.00 MP, 192 groups
            (4000, 3000, false, 7, 1, 376747),
            (4000, 3000, false, 7, 2, 597864),
            (4000, 3000, false, 7, 3, 597872),
            (4000, 3000, false, 7, 4, 597879),
            (4000, 3000, false, 7, 6, 597893),
            (4000, 3000, false, 7, 8, 597907),
            (4000, 3000, false, 7, 12, 597937),
            (4000, 3000, false, 9, 1, 376747),
            (4000, 3000, false, 9, 2, 597864),
            (4000, 3000, false, 9, 3, 597872),
            (4000, 3000, false, 9, 4, 597879),
            (4000, 3000, false, 9, 6, 597893),
            (4000, 3000, false, 9, 8, 597907),
            (4000, 3000, false, 9, 12, 597937),
        ];
        assert_eq!(
            sectioned_cells.len(),
            169,
            "grid cell count changed — re-read the note above"
        );
        for &(w, h, alpha, effort, threads, live_kb) in sectioned_cells {
            let bpp = if alpha { 4 } else { 3 };
            let e = estimate_encode_sectioned(w, h, bpp, alpha, effort, threads).unwrap();
            let live = live_kb * KB;
            assert!(
                e.peak_memory_bytes >= live,
                "{w}x{h} alpha={alpha} e{effort} t{threads}: TYP {} under measured peak_live {live} \
                 — UNDER-PREDICTION is an admission-safety bug, not a tightness one",
                e.peak_memory_bytes
            );
            if (w as u64) * (h as u64) >= 2_000_000 {
                assert!(
                    e.peak_memory_bytes < live * 5 / 2,
                    "{w}x{h} alpha={alpha} e{effort} t{threads}: TYP {} not a tight cover of {live} (>= 2.5x)",
                    e.peak_memory_bytes
                );
            }
        }
        // The 2026-08-27 global-fallback peaks of the same imac_dark cells
        // (486110 / 772396 / 772476 KiB) are what the meta-channel arm
        // removed; MAX still covers the e7 t=1 and e9 t=12 figures so a
        // regression to the fallback stays inside the admitted envelope.
        // (The e9 t=1 fallback peak, 772396 KiB, is deliberately NOT pinned:
        // that path no longer runs for this content and the runtime
        // `MemoryBudget` bounds it if it ever did.)
        let former_fallback_cells: &[(u32, u32, u8, usize, u64)] =
            &[(2940, 1912, 7, 1, 486110), (2940, 1912, 9, 12, 772476)];
        for &(w, h, effort, threads, live_kb) in former_fallback_cells {
            let e = estimate_encode_sectioned(w, h, 3, false, effort, threads).unwrap();
            assert!(
                e.peak_memory_bytes_max >= live_kb * KB,
                "{w}x{h} e{effort} t{threads} (former global fallback): MAX {} under measured {}",
                e.peak_memory_bytes_max,
                live_kb * KB
            );
        }
    }

    /// The sectioned arm is what makes large lossless encodes admissible:
    /// it sits well below the whole-image band at 4K / 12 MP / 21 MP e7,
    /// grows STRICTLY with the pool width, and is only offered in the
    /// calibrated tree-learning band.
    ///
    /// Strict monotonicity is a CONTRACT, not an accident of the shape: the
    /// pre-flight's thread walk-down steps down one worker at a time and
    /// stops the moment the estimate fails to fall
    /// (`api::encode_preflight_with_sectioned`), and
    /// `api_tests::encode_preflight_sectioned` pins that behaviour. Since
    /// 2026-08-30 the model is a `max()` of two phases, which on its own is
    /// only NON-DECREASING — the flat region is real (photo 12 MP measures
    /// 48.0 B/px at every thread count from 2 to 12). What restores
    /// strictness is `SECTIONED_POOL_BYTES_PER_THREAD`, the measured
    /// +7.3 KiB/worker slope of that flat region, rather than a fudge.
    #[test]
    fn sectioned_estimate_shape() {
        for &(w, h) in &[(3840u32, 2160u32), (4000, 3000), (4096, 5120)] {
            for e in 7..=9 {
                let whole = estimate_encode(w, h, 3, false, true, e)
                    .unwrap()
                    .peak_memory_bytes;
                let sect = estimate_encode_sectioned(w, h, 3, false, e, 1)
                    .unwrap()
                    .peak_memory_bytes;
                // Measured ratio ≈ 0.44–0.54 (4K / 12 MP, e7 / e9) on the
                // 2026-08-28 bands; the old `< whole / 3` held only against
                // the stale-high whole-image band.
                assert!(
                    sect * 3 < whole * 2,
                    "{w}x{h} e{e}: sectioned {sect} vs whole {whole}"
                );
            }
        }

        // STRICT monotonicity in the pool width, at every size class and
        // both efforts — including the sizes where one phase or the other
        // dominates the whole sweep (64² is a single group; 12 MP sits on
        // the pre-tree floor at every thread count it was measured at).
        for &(w, h) in &[
            (64u32, 64u32),
            (256, 256),
            (768, 768),
            (1024, 1024),
            (2048, 2048),
            (4000, 3000),
        ] {
            for e in 7..=9 {
                for alpha in [false, true] {
                    let at = |t| {
                        estimate_encode_sectioned(w, h, if alpha { 4 } else { 3 }, alpha, e, t)
                            .unwrap()
                            .peak_memory_bytes
                    };
                    for t in 1..24 {
                        assert!(
                            at(t) < at(t + 1),
                            "{w}x{h} alpha={alpha} e{e}: t{t} {} not below t{} {} — the thread \
                             walk-down needs a STRICTLY falling estimate",
                            at(t),
                            t + 1,
                            at(t + 1)
                        );
                    }
                }
            }
        }

        let at = |t| {
            estimate_encode_sectioned(2048, 2048, 3, false, 9, t)
                .unwrap()
                .peak_memory_bytes
        };
        assert_eq!(
            at(0),
            at(1),
            "0 = ambient is estimated at the 1-thread floor by the caller"
        );
        // Past the crossover — where the per-group learn sets out-peak the
        // pre-tree floor — the per-worker term is unclamped and exact.
        // 2048² is 64 groups, so the in-flight clamp cannot bind below 24.
        assert_eq!(
            at(24) - at(12),
            (SECTIONED_PER_THREAD_E9 + SECTIONED_POOL_BYTES_PER_THREAD) * 12,
            "unclamped per-worker term above the crossover"
        );
        // Below the crossover the estimate rides the pre-tree floor, so it
        // grows by the pool term ALONE. This is the property the additive
        // model could not express, and the reason it over-predicted
        // palette content by 2.8× at 12 workers.
        assert_eq!(
            at(3) - at(2),
            SECTIONED_POOL_BYTES_PER_THREAD,
            "on the pre-tree floor only the pool term grows"
        );
        // The one-worker arm is a different phase (streaming RCT + patches
        // detection, no trial wave), and is strictly below two workers —
        // this is what the pre-flight's min-over-{1,2} admission floor
        // resolves to.
        assert!(at(1) < at(2), "2048² e9: monotone from t=1");

        // In-flight clamp: a worker cannot hold a group set for a group
        // that does not exist. 256² is ONE group, so from `groups + slack`
        // workers on, only the pool term grows — measured directly (photo
        // 256² e9 is 99860 KiB at t=2 and 99933 at t=12).
        let one_group = |t| {
            estimate_encode_sectioned(256, 256, 3, false, 9, t)
                .unwrap()
                .peak_memory_bytes
        };
        assert_eq!(
            one_group(12) - one_group(3),
            SECTIONED_POOL_BYTES_PER_THREAD * 9,
            "single-group image: clamped at groups + slack in-flight sets"
        );
        assert_eq!(
            one_group(3) - one_group(2),
            SECTIONED_PER_THREAD_E9 + SECTIONED_POOL_BYTES_PER_THREAD,
            "the clamp binds at groups + slack = 3, not before"
        );

        // Alpha is a per-CHANNEL scaling of both per-pixel arms (the
        // measured rgba floors are exactly 4/3 of the rgb ones) plus the
        // ×1.5 per-worker factor. `input_bpp` is held equal at 4 so the
        // input term cancels and only the model's channel count moves.
        let rgb = |t| {
            estimate_encode_sectioned(2048, 2048, 4, false, 9, t)
                .unwrap()
                .peak_memory_bytes
        };
        let rgba = |t| {
            estimate_encode_sectioned(2048, 2048, 4, true, 9, t)
                .unwrap()
                .peak_memory_bytes
        };
        let px = 2048u64 * 2048;
        // The one-worker floor is `resident·channels + detect`: only the
        // resident image scales with the channel count (the patches
        // detector works on the colour planes), so alpha adds exactly one
        // more i32 plane there — matching the measured +2.8 B/px at
        // 3840×2160 t=1 (rgb 29.2 → rgba 32.0).
        assert_eq!(
            rgba(1) - rgb(1),
            (px as f64 * SECTIONED_RESIDENT_BPP_PER_CHANNEL) as u64,
            "one-worker floor gains exactly one resident plane"
        );
        assert_eq!(
            rgba(2) - rgb(2),
            (px as f64 * SECTIONED_WAVE_BPP_PER_CHANNEL) as u64,
            "the RCT trial wave is one more channel's worth of planes"
        );
        // At 5 workers both sides are still on their pre-tree floors at
        // 4 MP, so the per-worker alpha factor is checked where the learn
        // arm is what dominates: 1024², 12 workers.
        let rgb_small = |t| {
            estimate_encode_sectioned(1024, 1024, 4, false, 9, t)
                .unwrap()
                .peak_memory_bytes
        };
        let rgba_small = |t| {
            estimate_encode_sectioned(1024, 1024, 4, true, 9, t)
                .unwrap()
                .peak_memory_bytes
        };
        assert_eq!(
            (rgba_small(12) - rgba_small(1)) - (rgb_small(12) - rgb_small(1)),
            SECTIONED_PER_THREAD_E9 / 2 * 12 - SECTIONED_PER_THREAD_E9 / 2,
            "alpha per-worker factor"
        );

        assert!(!sectioned_estimate_available(6) && !sectioned_estimate_available(10));
        assert!((7..=9).all(sectioned_estimate_available));
    }
}
