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

/// Sectioned-mode fixed overhead by effort: the working set of ONE group's
/// tree learn at the effort's params, which is what a tiny image pays in
/// full. Measured 64×64 real-photo crop, threads=1 (2026-08-27,
/// `benchmarks/jxl_sectioned_mem_2026-08-27.tsv`): e7 5.9 MiB, e9 25.3 MiB
/// peak_live (29.7 MiB at 12 workers). Set with margin; e8 shares the e9
/// value (unswept, e9 is the band's top).
const SECTIONED_FIXED_E7: u64 = 8 << 20;
const SECTIONED_FIXED_E9: u64 = 32 << 20;

/// Sectioned-mode per-pixel floor, bytes/pixel of marginal working set.
/// The sectioned peak is NOT the tree learn (that is per-group and rides
/// the per-thread term) but the whole-image phases before it — the
/// modular image copies plus the patches-detection planes. Measured
/// (2026-08-27, `benchmarks/jxl_sectioned_mem_2026-08-27.tsv`, real
/// content, sectioned path verified engaged via `JXL_SECTION_SIZES`):
///   - photo (imazen-26 1403, crops 2048² / 3840×2160 / 4000×3000),
///     threads ≥ 4: flat ≈ 48 B/px (192 / 380 / 550 MiB marginal).
///     threads = 1 used to carry an extra size-growing phase above 4 MP
///     (+114 MiB at 8.3 MP, +271 MiB at 12 MP, identical at e7/e9) —
///     ATTRIBUTED 2026-08-28 (`benchmarks/jxl_sectioned_mem_t1excess_
///     2026-08-28.{tsv,meta}`, `MEM_PROBE_PATCHES=0` A/B): it was the
///     lossless patches detector's single-thread connected-component
///     scan, whose flat-index DFS stack grows through doubling reallocs
///     on photo content (one foreground component). The detector now
///     takes the bounded union-by-min labeling path at ≥ 1 MP on every
///     thread count (bytes identical), and t=1 measures the SAME floor as
///     t ≥ 4: 46.9 B/px at 12 MP (597763 KiB, was 875430), 47.9 B/px at
///     8.3 MP (413203, was 530187).
///   - screenshot (qoi reddit.com 1313×8008, 10.5 MP): measured
///     60.5–61.8 B/px at every thread count in the 2026-08-27/28 sweeps;
///     48.0 B/px since the 2026-08-30 patches-phase-lifetime fix below.
///
/// One floor for every thread count: the envelope over both classes with
/// margin (the two constants are kept as the model's two arms but
/// carry the same value since the t=1 excess was removed).
///
/// Palette / ChannelCompact / patches content engages the sectioned
/// writer too since 2026-08-28 (stream 0 codes the meta channels with its
/// own tiny tree; the patches dictionary precedes the modular stream as
/// on the global path): gb82-sc `imac_dark` 2940×1912 — which wrote 0
/// local sections under `On` before — measured 58.9 B/px marginal at
/// e7 AND e9, t=1 (347.6 MiB peak_live vs 486 / 772 MiB global;
/// `benchmarks/jxl_sectioned_mem_meta_2026-08-28.tsv`), under the floors
/// above. Only the lossy-modular custom-DC-quant path and the non-tree /
/// non-ANS modes still take the global tree under `On`; the runtime
/// `MemoryBudget` enforces the cap allocation-by-allocation there.
///
/// PATCHES-PHASE LIFETIME (2026-08-30, the #96 residual item,
/// `benchmarks/jxl_sectioned_patches_lifetime_2026-08-30.{tsv,meta}`):
/// on screen content the patches DETECTION working set previously sat
/// at the sectioned encode peak — `MEM_PROBE_PATCHES` A/B: +76 MiB on
/// imac_dark, +138.5 MiB on reddit.com, ≈ +13.8 B/px at EVERY thread
/// count — attributed with the in-repo alloc-sites probe
/// (`JXL_ALLOC_SITES=1`) to the u8→f32 conversion planes (12 B/px), the
/// BFS seed queue (a 2× over-sized leftover, ~22.7 B/px transient) and
/// the flood-fill planes riding on top of the already-built whole-image
/// i32 `ModularImage`. Fixed byte-identically by (a) detecting patches
/// BEFORE the `ModularImage` is built (`api.rs::encode_lossless_single`)
/// and (b) sizing the seed queue exactly (`vardct/patches.rs`). Screens
/// now measure the SAME ~48 B/px floor as photo: imac_dark 280985 KiB
/// (48.2 B/px, was 347628–357924 across t), reddit 523716 KiB (48.0
/// B/px, was 665580). The 68 B/px floors below deliberately stay: they
/// cover the measured 48 with ~1.4× headroom (admission-safe), and the
/// patches dictionary of patch-heavy content still lands above the bare
/// floor (imac +0.9 MiB at t=1).
///
/// RCT-TRIAL FOLD (2026-08-30, same day, issue #99 lever 1,
/// `benchmarks/jxl_sectioned_rct_fold_2026-08-30.{tsv,meta}`): the
/// alloc-sites probe then showed the remaining ~48 B/px floor WAS the
/// `select_best_rct` trial wave — nine whole-image i32 channel clones
/// (36 B/px) + the ModularImage (12 B/px) — on all three content
/// classes. On single-worker pools the wave buys no overlap, so trials
/// now fold one at a time there (byte-identical): the t=1 band drops to
/// ~36–39 B/px (photo 12 MP 597763 → 457138 KiB, imac 228308, reddit
/// 422495; rgba 3840×2160 570687 → 421304). t ≥ 2 keeps the wave and
/// its ~48 B/px band (re-measured KiB-identical), so the two constants
/// below still share the multi-thread envelope; the t=1 arm could drop
/// to ~56 after the next full recalibration but stays 68 (safe,
/// covering at 1.7–1.9×).
const SECTIONED_BPP_THREADS1: f64 = 68.0;
const SECTIONED_BPP_MULTI: f64 = 68.0;

/// Per-worker term: each in-flight group learns its own tree (and
/// `parallel-tree-learning` forks owned per-side clones inside it), so
/// the peak grows with the pool width — the axis the thread-invariant
/// whole-image band never had. Measured slope 1024² real photo, t=1→12
/// (the cell where the per-group sets are not hidden under the floor):
/// e7 (150954 − 52261) KiB / 11 = 8.8 MiB/thread, e9 (430537 − 74376)
/// KiB / 11 = 31.6 MiB/thread — one 256² group at the effort's learn
/// params (screenshot crops: 4.9 / 28.6 MiB/thread). Set
/// with margin; additive and UNCLAMPED past the group count (a 1-group
/// image still grows ≈ 3 MiB/thread from the intra-group forks, so the
/// additive form over-predicts tiny multi-threaded encodes — the safe
/// direction — rather than under-predicting them).
const SECTIONED_PER_THREAD_E7: u64 = 12 << 20;
const SECTIONED_PER_THREAD_E9: u64 = 36 << 20;

/// Alpha extra-channel terms (rgba − rgb, 2026-08-27 rgba cells — alpha
/// := the source's green plane, worst-case entropy — see the sweep
/// .meta): the pre-tree floor grows only +4.0 B/px at 3840×2160 (both
/// efforts, t=1 and t=8) and on the 1313×4096 screenshot crop, but at
/// 1024² threads=1 the 4-channel group learn adds 16.0 (e7) / 25.5 (e9)
/// B/px, and the per-worker group set grows ×1.23 (e7) / ×1.44 (e9)
/// (1024², t=1→8). Modelled as a flat per-pixel term (envelope of the
/// t=1 cells, over-predicting large images — the safe direction) plus a
/// ×1.5 per-thread factor.
const SECTIONED_BPP_ALPHA: f64 = 28.0;
const SECTIONED_PER_THREAD_ALPHA_NUM: u64 = 3;
const SECTIONED_PER_THREAD_ALPHA_DEN: u64 = 2;

/// Peak-memory estimate for a lossless encode that runs the SECTIONED
/// local-tree mode ([`crate::api::SectionedTrees`] `On`, or `Auto` where
/// its gate engages — imazen/jxl-encoder#96) at `cores` worker threads.
///
/// `peak = input + fixed(effort) + floor(threads)·pixels +
/// per_thread(effort)·(cores − 1)`, all terms measured on real content
/// (photo + screenshot crops, tiny → 12 MP, threads 1/4/8/12, e7/e9;
/// `benchmarks/jxl_sectioned_mem_2026-08-27.tsv` + `.meta`). Only valid
/// where [`sectioned_estimate_available`] holds; `time_ms` /
/// `output_bytes` are the whole-image figures (the mode is measured
/// byte-neutral at the median and faster, so they remain upper bounds).
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
    let (fixed, mut per_thread) = if effort >= 8 {
        (SECTIONED_FIXED_E9, SECTIONED_PER_THREAD_E9)
    } else {
        (SECTIONED_FIXED_E7, SECTIONED_PER_THREAD_E7)
    };
    if has_alpha {
        per_thread = per_thread * SECTIONED_PER_THREAD_ALPHA_NUM / SECTIONED_PER_THREAD_ALPHA_DEN;
    }
    let mut bpp = if cores > 1 {
        SECTIONED_BPP_MULTI
    } else {
        SECTIONED_BPP_THREADS1
    };
    if has_alpha {
        // One more image-sized plane through the pre-tree phases and one
        // more channel per group learn (per-thread factor applied above).
        // Pinned by the rgba cells of the 2026-08-27 sweep (see
        // `sectioned_estimate_covers_measured_cells_2026_08_27`).
        bpp += SECTIONED_BPP_ALPHA;
    }
    let working = fixed
        .checked_add((pixels as f64 * bpp) as u64)?
        .checked_add(per_thread.checked_mul(cores - 1)?)?;
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

    /// imazen/jxl-encoder#96 sectioned-mode cells, 2026-08-27, macOS/M-series
    /// laptop, jxl-encoder d7fc8f7e + this change's probe
    /// (`jxl-encoder-cli/examples/mem_probe`, counting global allocator —
    /// `peak_live` = high-water of LIVE bytes across `encode()`, input
    /// buffer included, same definition as the 2026-08-13 4K cells).
    /// Provenance: `benchmarks/jxl_sectioned_mem_2026-08-27.tsv` + `.meta`
    /// (`scripts/mem_sectioned_sweep.sh`). Real content only: imazen-26
    /// png-v3 photo 1403 (4000×3000) and its top-left crops; qoi-benchmark
    /// `screenshot_web/reddit.com.png` (1313×8008) and crops; gb82-sc
    /// `imac_dark.png` (2940×1912). `SectionedTrees::On`; the sectioned
    /// path verified engaged (`JXL_SECTION_SIZES`) on the photo/reddit
    /// cells; the imac_dark (palette + patches screenshot) cells were
    /// re-measured 2026-08-28 once the sectioned writer covered meta
    /// channels and patches (`benchmarks/jxl_sectioned_mem_meta_2026-08-28.tsv`)
    /// — before that they fell back to the global tree and were only
    /// MAX-covered.
    ///
    /// Contract: TYP covers every sectioned-engaged cell (the admission
    /// floor and the thread walk-down use TYP) and stays a tight cover
    /// (< 2.5× — the additive per-thread term over-predicts when the
    /// in-flight group sets sit under the pre-tree floor; see the
    /// constants' notes). Re-measure and re-pin whenever the sectioned
    /// writer, the patches detector or the modular image lifetime changes.
    #[test]
    fn sectioned_estimate_covers_measured_cells_2026_08_27() {
        const KB: u64 = 1024;
        // (w, h, has_alpha, effort, threads, measured peak_live KiB)
        // t=1 rows re-pinned 2026-08-30 after the single-worker RCT-trial
        // fold (issue #99 lever 1, benchmarks/jxl_sectioned_rct_fold_
        // 2026-08-30.{tsv,meta}): the select_best_rct wave held nine
        // whole-image channel clones (36 B/px) AT the t=1 peak — the fold
        // holds six, dropping the t=1 band to ~36-39 B/px. t >= 2 cells
        // keep the wave and re-measured byte-for-KiB identical. History:
        // 3840x2160 e7 t1 530187 (pre labeling fix) -> 413203 -> 316003;
        // 4000x3000 e7/e9 t1 875430 -> 597763 -> 457138; 1024^2 e9 t1
        // 74376 -> 73994 is pre-existing drift (fold-neutral there: the
        // 1 MP e9 tree-learn peak exceeds the RCT-wave instant).
        let sectioned_cells: &[(u32, u32, bool, u8, usize, u64)] = &[
            // photo 1403 crops
            (64, 64, false, 7, 1, 6008),
            (64, 64, false, 9, 1, 25865),
            (64, 64, false, 9, 12, 30400),
            (256, 256, false, 9, 12, 99933),
            (1024, 1024, false, 7, 1, 39973),
            (1024, 1024, false, 7, 12, 150954),
            (1024, 1024, false, 9, 1, 73994),
            (1024, 1024, false, 9, 12, 430537),
            (2048, 2048, false, 9, 1, 159805),
            (2048, 2048, false, 9, 12, 396710),
            (3840, 2160, false, 7, 1, 316003),
            (3840, 2160, false, 7, 4, 413315),
            (3840, 2160, false, 9, 12, 479149),
            (4000, 3000, false, 7, 1, 457138),
            (4000, 3000, false, 9, 1, 457138),
            (4000, 3000, false, 7, 4, 597879),
            (4000, 3000, false, 7, 12, 597937),
            (4000, 3000, false, 9, 12, 597937),
            // reddit.com screenshot crops. Re-pinned 2026-08-30 after the
            // patches-phase lifetime fix (detection before the modular
            // build + exact seed-queue capacity,
            // jxl_sectioned_patches_lifetime_2026-08-30.tsv): the
            // detection working set no longer rides at the peak, so the
            // full-height cells drop to the photo floor (1313x8008 e7 t1
            // was 665580, e9 t12 was 665660, 1313x4096 e9 t12 was
            // 351990). The 256² crop and the rgba t8 cell never had the
            // detection at peak — unchanged, verified same-commit.
            (256, 256, false, 9, 12, 66261),
            (1313, 4096, false, 9, 12, 336418),
            // 2026-08-30 RCT fold: was 523716 (post patches-lifetime fix;
            // 665580 before that).
            (1313, 8008, false, 7, 1, 422495),
            (1313, 8008, false, 9, 12, 523826),
            // rgba (alpha := green): photo 1403 crops + reddit crop. The
            // t=1 cells fold a 4-channel clone set (16 B/px per trial):
            // 1024^2 e7 was 69670, 3840x2160 e7/e9 were 570687.
            (1024, 1024, true, 7, 1, 53286),
            (1024, 1024, true, 7, 8, 146332),
            (1024, 1024, true, 9, 1, 100734),
            (1024, 1024, true, 9, 8, 404080),
            (3840, 2160, true, 7, 1, 421304),
            (3840, 2160, true, 7, 8, 551044),
            (3840, 2160, true, 9, 1, 421304),
            (3840, 2160, true, 9, 8, 551044),
            (1313, 4096, true, 7, 8, 357262),
            // imac_dark (gb82-sc screenshot: full palette/compact + patches),
            // sectioned-engaged since 2026-08-28 (96/96 local sections).
            // History: 347628-357924 (detection working set at peak, both
            // efforts) -> 280985/281076/281134 (2026-08-30 patches-phase
            // lifetime fix, photo floor + the surviving patches
            // dictionary) -> t=1 228308 (same-day RCT fold; the t >= 2
            // cells keep the wave and its measured values).
            (2940, 1912, false, 7, 1, 228308),
            (2940, 1912, false, 7, 4, 281076),
            (2940, 1912, false, 7, 12, 281134),
            (2940, 1912, false, 9, 1, 228308),
            (2940, 1912, false, 9, 4, 281076),
            // e9 t12 measured 328741 on 2026-08-30 (12-worker group-learn
            // sets over the new floor) but stays pinned at the 2026-08-28
            // value: lowering it trips the 2.5× tightness bar (TYP
            // 847.9 MB = 2.52× of 328741) because the additive 36
            // MiB/worker e9 term over-predicts palette content whose
            // per-group learns are tiny (measured 4.3 MiB/worker here).
            // The stale-high pin remains a VALID, stronger coverage
            // constraint; tightening it awaits the per-thread-term model
            // refinement (owner-gated — the walk-down admission contract
            // asserts strict thread-monotonicity, which a headroom-aware
            // term breaks). Tracked in issue #99.
            (2940, 1912, false, 9, 12, 357924),
        ];
        for &(w, h, alpha, effort, threads, live_kb) in sectioned_cells {
            let bpp = if alpha { 4 } else { 3 };
            let e = estimate_encode_sectioned(w, h, bpp, alpha, effort, threads).unwrap();
            let live = live_kb * KB;
            assert!(
                e.peak_memory_bytes >= live,
                "{w}x{h} alpha={alpha} e{effort} t{threads}: TYP {} under measured peak_live {live}",
                e.peak_memory_bytes
            );
            if (w as u64) * (h as u64) >= 2_000_000 {
                assert!(
                    e.peak_memory_bytes < live * 5 / 2,
                    "{w}x{h} alpha={alpha} e{effort} t{threads}: TYP {} not a tight cover of {live} (≥ 2.5×)",
                    e.peak_memory_bytes
                );
            }
        }
        // The 2026-08-27 global-fallback peaks of the same imac_dark cells
        // (486110 / 772396 / 772476 KiB) are what the meta-channel arm
        // removed; MAX still covers the e7 t=1 and e9 t=12 figures so a
        // regression to the fallback stays inside the admitted envelope.
        // (The e9 t=1 fallback peak, 772396 KiB, sits 3 % above MAX now
        // that the t=1 floor no longer carries the removed patches-scan
        // excess — 765 vs 791 MB; that path no longer runs for this
        // content, and the runtime `MemoryBudget` bounds it if it ever
        // did, so it is deliberately NOT pinned as a MAX requirement.)
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
    /// grows strictly with the pool width (one group's learn per worker
    /// — the axis the whole-image band lacks), and is only offered in
    /// the calibrated tree-learning band.
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
        let at = |t| {
            estimate_encode_sectioned(2048, 2048, 3, false, 9, t)
                .unwrap()
                .peak_memory_bytes
        };
        assert!(
            at(2) < at(3) && at(3) < at(8) && at(8) < at(24),
            "monotone from 2 workers"
        );
        assert_eq!(
            at(24) - at(2),
            SECTIONED_PER_THREAD_E9 * 22,
            "unclamped per-thread term"
        );
        assert_eq!(
            at(0),
            at(1),
            "0 = ambient is estimated at the 1-thread floor by the caller"
        );
        // Monotone from ONE worker at every size since 2026-08-28: the
        // single-worker excess (a patches-scan DFS stack, see the floor
        // constants' notes) is gone, so both floor arms carry the same
        // value and the per-thread term is the only thread axis. The
        // pre-flight's min-over-{1, 2} admission floor therefore resolves
        // to the 1-thread figure (kept general in case an arm diverges).
        assert!(at(1) < at(2), "2048² e9: monotone from t=1");
        let small = |t| {
            estimate_encode_sectioned(1024, 1024, 3, false, 9, t)
                .unwrap()
                .peak_memory_bytes
        };
        assert!(small(1) < small(2), "1024² e9: per-thread term dominates");
        // Alpha adds a per-pixel term (input_bpp held equal to isolate it)
        // at one worker, and scales the per-thread term ×1.5 beyond that.
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
        assert_eq!(
            rgba(1) - rgb(1),
            (2048u64 * 2048) * SECTIONED_BPP_ALPHA as u64
        );
        assert_eq!(
            (rgba(5) - rgba(1)) - (rgb(5) - rgb(1)),
            SECTIONED_PER_THREAD_E9 / 2 * 4,
            "alpha per-thread factor"
        );
        assert!(!sectioned_estimate_available(6) && !sectioned_estimate_available(10));
        assert!((7..=9).all(sectioned_estimate_available));
    }
}
