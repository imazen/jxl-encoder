// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Butteraugli quantization loop for iterative quality refinement.
//!
//! Iteratively refines per-block quant_field by measuring perceptual distance
//! (butteraugli) between the original and reconstructed image.
//!
//! Matches libjxl's FindBestQuantization (enc_adaptive_quantization.cc:929-1115):
//! - Works in float quant field domain (values ~0.3-1.5), NOT integer (1-255)
//! - Recomputes global_scale each iteration via SetQuantField (median/MAD)
//! - Returns final DistanceParams for use in CfL pass 2 and encoding

use core::sync::atomic::{AtomicI32, Ordering};

use super::ac_strategy::AcStrategyMap;
use super::adaptive_quant::quantize_quant_field;
use super::chroma_from_luma::CflMap;
use super::common::*;
use super::encoder::VarDctEncoder;
use super::frame::DistanceParams;
use crate::debug_rect;
use crate::error::Result;

/// libjxl's hardcoded `kInitMul` (`enc_adaptive_quantization.cc:1042`)
/// that pulls the post-`kOriginalComparisonRound` quant field back toward
/// the initial AC heuristic field. Single-seed encodes use only this
/// value (bit-identical to libjxl).
pub(crate) const LIBJXL_INIT_MUL: f64 = 0.6;

// ===== Distance-aware buttloop tuning scaffolding (W38-2 #3.1).
//
// Infrastructure ported from GPU `d75bf7c` (memory
// `buttloop_rd_gap_2026-05-14.md`) for hot-swap A/B sweeps on the
// per-iter `(cur_pow, max_increase)` knobs. **Production defaults
// match libjxl at every regime** (`cur_pow=0.2`, `max_increase=100.0`
// ≈ "no cap"); the GPU's tuned LOW-regime values (`cur_pow=0.5`,
// `max_increase=1.3`) regressed RD-pareto on CPU at LOW (bfly +4-13 %
// on screenshots, +1-8 % on photos at d<2.0) when tested A/B — see
// `benchmarks/buttloop_distance_split_port_2026-05-18.tsv`. The GPU's
// tuning was calibrated against its own e7 baseline (≈9 % smaller
// bytes than cjxl e7); CPU's baseline differs and the same factor
// over-shrinks the quant field.
//
// The atomic overrides below let sweep harnesses search for a
// CPU-specific LOW value (or any other tuning) without rebuilds.
// Production code never sets them — see `resolved_cur_pow` /
// `resolved_max_increase` helpers.
//
// Hash-lock invariants at default e7 are preserved because the
// buttloop is gated at effort >= 8 (`speed_tier <= kKitten`).

/// Sweep override for `cur_pow` at low distances (`target_distance <
/// [`DEFAULT_DISTANCE_SPLIT`]`). Stored as `value × 1000` (so 500 = 0.5).
/// `i32::MIN` means "not overridden — use [`DEFAULT_CUR_POW_LOW`]".
pub static CUR_POW_X1000_LOW: AtomicI32 = AtomicI32::new(i32::MIN);

/// Sweep override for `cur_pow` at high distances (`target_distance >=
/// [`DEFAULT_DISTANCE_SPLIT`]`). `i32::MIN` means "use
/// [`DEFAULT_CUR_POW_HIGH`]".
pub static CUR_POW_X1000_HIGH: AtomicI32 = AtomicI32::new(i32::MIN);

/// Sweep override for `max_increase` (per-iter bad-block bump cap) at
/// low distances. Stored as `value × 1000`. `i32::MIN` means "use
/// [`DEFAULT_MAX_INCREASE_LOW`]".
pub static MAX_INCREASE_X1000_LOW: AtomicI32 = AtomicI32::new(i32::MIN);

/// Sweep override for `max_increase` at high distances. `i32::MIN` means
/// "use [`DEFAULT_MAX_INCREASE_HIGH`]".
pub static MAX_INCREASE_X1000_HIGH: AtomicI32 = AtomicI32::new(i32::MIN);

/// Sweep override for `max_increase` (per-iter bad-block bump cap) at
/// HIGH distances **on screenshot-class content only**
/// (`median(mask1x1) > [`SCREENSHOT_MEDIAN_THRESHOLD`]`).
///
/// Stored as `value × 1000`. `i32::MIN` means "use
/// [`DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT`]".
///
/// **Rationale**: W38-2 RD-curve audit WF3 found that e8/e9 buttloop
/// over-compresses screenshots at d≥2.0 (bfly +9-19 %, ssim2 -2 to -5
/// vs cjxl). The libjxl defaults (no cap) let the bad-block bump
/// double-or-worse on a single bad iter when text/UI blocks register
/// high tile_dist — recoverable on photos (next iter can rebalance),
/// catastrophic on screenshots (text re-bumps round-trip the next iter
/// too). A cap at HIGH for the screenshot class only is the W39-1
/// commit's documented "real WF3 fix".
///
/// Photo-class HIGH regime continues to read [`MAX_INCREASE_X1000_HIGH`]
/// → [`DEFAULT_MAX_INCREASE_HIGH`] (libjxl defaults, no cap). LOW
/// regime is unchanged from W39-1 (literal GPU LOW values regressed
/// CPU; HIGH is the only place this fix lives).
pub static MAX_INCREASE_X1000_HIGH_SCREENSHOT: AtomicI32 = AtomicI32::new(i32::MIN);

/// Sweep override for the threshold between LOW and HIGH regimes. The
/// per-iter loop picks LOW when `target_distance < threshold`, else HIGH.
/// Defaults to `2000` (= 2.0) — see [`DEFAULT_DISTANCE_SPLIT`].
///
/// Unlike the other overrides this slot is initialised to its default
/// value (NOT `i32::MIN`) so that `resolved_*` helpers always see a
/// valid split even when production runs without any harness present.
pub static DISTANCE_SPLIT_X1000: AtomicI32 = AtomicI32::new(2000);

/// Helper: read an `_X1000` override; return `default` when unset.
fn read_override_x1000(slot: &AtomicI32, default: f64) -> f64 {
    let v = slot.load(Ordering::Relaxed);
    if v == i32::MIN {
        default
    } else {
        v as f64 / 1000.0
    }
}

/// Production default for `cur_pow` in the LOW regime
/// (`target_distance < DEFAULT_DISTANCE_SPLIT`). Matches libjxl's
/// default — **the literal GPU port (`0.5`) was tested A/B on CPU
/// and over-reclaims, costing 1-13 % butteraugli at d<2.0** (see
/// `benchmarks/buttloop_distance_split_port_*.{tsv,meta}`). The
/// scaffolding stays so sweep harnesses can find a CPU-specific
/// LOW value via `CUR_POW_X1000_LOW`, but the default is the
/// libjxl-faithful value until that sweep lands.
///
/// GPU equivalent: `DEFAULT_CUR_POW_LOW = 0.5` in
/// `jxl-encoder-gpu/src/forks/butteraugli_loop.rs`. The GPU's
/// `0.5` tuning was calibrated to its own baseline (≈9 % smaller
/// bytes at e7 than cjxl); CPU's e7 baseline differs and the same
/// value is too aggressive here.
pub const DEFAULT_CUR_POW_LOW: f64 = 0.2;

/// Production default for `cur_pow` in the HIGH regime
/// (`target_distance >= DEFAULT_DISTANCE_SPLIT`). Matches libjxl's
/// default (`enc_adaptive_quantization.cc:1106`) — no change from
/// pre-port CPU behaviour.
pub const DEFAULT_CUR_POW_HIGH: f64 = 0.2;

/// Production default for `max_increase` (per-iter bad-block bump cap)
/// in the LOW regime. Matches libjxl's implicit "no cap" — set to
/// `100.0` (effectively infinite). See `DEFAULT_CUR_POW_LOW` for the
/// rationale on why the literal GPU port (`1.3`) is not the default.
pub const DEFAULT_MAX_INCREASE_LOW: f64 = 100.0;

/// Production default for `max_increase` in the HIGH regime. Matches
/// libjxl's implicit "no cap" — set to `100.0` (effectively infinite).
pub const DEFAULT_MAX_INCREASE_HIGH: f64 = 100.0;

/// Production default for `max_increase` in the HIGH regime on
/// **screenshot-class** content (`median(mask1x1) >
/// [`SCREENSHOT_MEDIAN_THRESHOLD`]`).
///
/// Default `100.0` (≈ "no cap" / libjxl-faithful). The W38-2 audit
/// recommended capping this to fix WF3 (e8/e9 over-compresses
/// screenshots at d≥2.0); the sweep harness
/// `examples/buttloop_screenshot_cap_sweep.rs` searches
/// `{1.3, 1.5, 1.8, 2.0, ∞}` for the winning value. After analysis the
/// const flips to the chosen value (default-on).
///
/// Photo-class HIGH regime continues to use
/// [`DEFAULT_MAX_INCREASE_HIGH`] (no cap), so this lever is invisible
/// on non-screenshot inputs.
pub const DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT: f64 = 100.0;

/// `median(mask1x1)` threshold above which the HIGH-regime buttloop
/// reads the screenshot cap ([`MAX_INCREASE_X1000_HIGH_SCREENSHOT`] /
/// [`DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT`]) instead of the photo
/// default ([`MAX_INCREASE_X1000_HIGH`] /
/// [`DEFAULT_MAX_INCREASE_HIGH`]).
///
/// `95.0` matches the existing screenshot discriminator used by
/// `entropy_mul` content-aware dispatch
/// (`encoder::CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD`) and
/// `splines::looks_like_screenshot` so we don't introduce a third
/// boundary value. The classifier is computed once per encode on the
/// pre-gaborish XYB Y plane (same as the existing
/// `compute_mask1x1` call in [`super::encoder`]).
pub const SCREENSHOT_MEDIAN_THRESHOLD: f32 = 95.0;

/// Default split point between LOW and HIGH regimes.
/// `target_distance >= DEFAULT_DISTANCE_SPLIT` triggers the HIGH regime.
pub const DEFAULT_DISTANCE_SPLIT: f64 = 2.0;

/// W44-105: production default scale factor applied to the buttloop's initial
/// `quant_field_float` on screenshot-class content at `target_distance >= 2.0`.
///
/// **Why this matters**: butteraugli is too lenient on text-heavy screenshot
/// reconstructions — our iter-0 reconstruction reports a butteraugli score
/// well below the target distance, so the loop spends iter 0/1 *reducing*
/// quality (`cur_pow=0.2` path), starving text blocks of the AC precision
/// they need for sharp glyph rendering. cjxl avoids this because its
/// internal `RoundtripImage` reports a much higher iter-0 score (47.7 for
/// the W44-103 terminal e8 d=4 wedge cell, vs ours 2.07), which triggers
/// the `bad-block` bump path and pushes text qac up to 97+.
///
/// The fix: scale the initial quant_field_float up by this factor before
/// the buttloop starts. The loop then runs its normal `cur_pow=0.2` backoff
/// but settles at a higher equilibrium that preserves text precision.
///
/// **Default = 4.0**: empirically chosen from the W44-105 scale sweep on
/// terminal e8 d=4 (SCALE=4 gives +3.42 SSIM2 / +31% bytes vs SCALE=1, with
/// final bytes still 34% smaller than cjxl). Higher values (SCALE=6..10)
/// give bigger SSIM2 wins but pay more bytes — at SCALE=10 we still ship
/// 14% fewer bytes than cjxl with matching SSIM2 (87.5 vs 87.6).
///
/// Gated on:
///   - `is_screenshot` (median(mask1x1) > [`SCREENSHOT_MEDIAN_THRESHOLD`])
///   - `target_distance >= [`BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE`]`
///   - `butteraugli_iters > 0` (only applies when buttloop runs)
///
/// Photo-class content is unaffected (scale stays at 1.0 → byte-identical
/// pre-W44-105 behaviour).
pub const DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE: f32 = 4.0;

/// W44-107: minimum `target_distance` at which the W44-105
/// [`DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE`] gate fires on
/// screenshot-class content.
///
/// **Why this is 3.5** (raised from W44-105's 2.0): the W44-106 full
/// ledger refresh found that the W44-105 seed-scale fix caused one
/// FIXED→OPEN regression on `codec_wiki.png e8 d=3` (bytes +3.3%,
/// bfly +25.7%, ssim2 -0.30). At d=3, codec_wiki's mid-tier wiki
/// content (text + diagrams + photo crops) responds poorly to the 4×
/// seed-scale bump — cjxl appears to engage a different threshold in
/// this regime that we don't yet match, producing a non-monotonic
/// bfly profile (d=2.5: +1.3%, d=3.0: +25.7%, d=4.0: +5.4%).
///
/// Tightening the lower gate from `d >= 2.0` to `d >= 3.5` excludes
/// the d=3 regression cell while preserving the W44-105 wins at d=4+
/// (the largest cluster: terminal d=4 SSIM2 +3.28, terminal d=5 +3.31,
/// codec_wiki d=4 +1.61, codec_wiki d=5 +2.12). Wins at d=2/d=2.5 are
/// sacrificed (terminal d=4-cell SSIM2 win is the largest reported in
/// W44-105 and is preserved). The W44-106 ledger entries for the
/// sacrificed cells either remain FIXED (cjxl-comparable) or shift
/// from "beats cjxl" to "matches cjxl pre-W44-105" — neither flips
/// FIXED→OPEN per the W44-106 baseline data.
///
/// **Followups**: a per-image discriminator (e.g. zenanalyze
/// `palette_log2_size` / `flat_color_block_ratio`) could re-engage
/// the gate at d=2..3.5 for terminal/imac_g3-class content while
/// keeping codec_wiki excluded. Tracked in `Investigation Notes` as
/// the W44-108 follow-on.
pub const BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE: f32 = 3.5;

/// W44-108: lower distance bound for the sub-discriminator that recovers
/// the 8 W44-105 wins W44-107 sacrificed at d=2..3 on terminal-class
/// content. The gate fires in `[BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE,
/// BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE)` only when the image's
/// `m3_colourfulness` proxy is below
/// [`BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX`] — separating
/// terminal/imac_g3/imac_dark (M3 ≈ 14..21, monochrome / low-colour
/// screenshots) from codec_wiki (M3 ≈ 146, richly-coloured wiki page
/// with photos). Below this d the buttloop's HIGH-regime tuning does
/// not engage strongly enough to recover the W44-105 wins; above this
/// d the full W44-107 gate (no m3 sub-check) already fires.
pub const BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE: f32 = 2.0;

/// W44-108: upper bound on `ZenanalyzeProxies::m3_colourfulness` for the
/// sub-discriminator that admits the d=2..3 fire-band. The probe
/// (`examples/w44_108_proxy_probe.rs`) measured:
///
/// - terminal: M3 = 13.85
/// - imac_g3: M3 = 14.29
/// - imac_dark: M3 = 20.96
/// - **codec_wiki: M3 = 145.73** (rich colour content, the W44-107 regression target)
///
/// The 30.0 threshold separates with ~6× margin both sides — robust to
/// natural variance in M3 across similar-class images. Photos in the
/// validation set range M3 ≈ 32..99 — outside the sub-band by design
/// (photos also fail the `is_screenshot` mask>95 gate, so this is
/// belt-and-braces).
pub const BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX: f32 = 30.0;

/// W44-109: maximum effort at which the screenshot-class adaptive-quant
/// pre-scale fires. Mirrors the W44-105 buttloop seed-scale mechanism
/// but at adaptive_quant time, before the buttloop runs (the buttloop
/// is itself effort-gated at `butteraugli_iters > 0`, which means
/// `effort >= 8`).
///
/// **Why this is 7** (i.e. fires at e ∈ [5, 7]): the W44-106 ledger
/// refresh showed that terminal e5/e6/e7 d=4 retains SSIM2 -4.6 to -5.4
/// vs cjxl even AFTER W44-105 landed — the same root cause (butteraugli
/// reports a too-low intermediate score on text-class content, leading
/// the adaptive-quant pipeline to under-quantize text blocks) but
/// without the buttloop's iter-1+ correction path to recover. At e<5
/// the encoder uses a flat quant field
/// (`profile.use_adaptive_quant = false`) so the pre-scale has no
/// useful target. At e>=8 the W44-105 buttloop path takes over.
///
/// The fix mirrors W44-105 exactly: scale `quant_field_float` by 4×
/// when the same gate predicate fires (is_screenshot AND
/// (d >= W44-107 distance OR (m3 < W44-108 m3 AND d >= W44-108 d))),
/// but at adaptive_quant time so it actually engages at e5/e6/e7. The
/// e>=8 path is unchanged (W44-109 skips when `butteraugli_iters > 0`
/// to avoid double-applying the scale).
pub const ADAPTIVE_QUANT_QF_SEED_SCALE_MAX_EFFORT: u8 = 7;

/// W44-120: minimum `target_distance` at which the W44-117 EPF sharpness
/// seed compute fires (on top of the W44-118 `is_screenshot` gate).
///
/// **Why this is 1.0**: the W44-119 ledger refresh
/// (`benchmarks/cjxl_parity_ledger_2026-05-20_w44_119.{tsv,meta}`)
/// surfaced a NEW regression introduced by W44-117 that wasn't visible
/// in the 44-cell W44-117 acceptance bench: terminal e8/e9 d=0.8 SSIM2
/// dropped from -0.73 → -2.60 (-1.87 vs the pre-W44-117 baseline) on a
/// near-FIXED cell. At very low distance the buttloop's target
/// butteraugli is already very low and the legacy uniform-4 sharpness
/// path was a close-enough match to the production sharpness map; the
/// W44-117 seed's iter-0-fitted sharpness map over-protects edges that
/// the encoder should be quantizing more freely.
///
/// The W44-120 bisection
/// (`benchmarks/w44_120_distance_bisect_2026-05-20.{tsv,meta}`) swept
/// thresholds 0.8 / 1.0 / 1.2 / 1.5 on terminal × e8/e9 × d ∈ {0.5,
/// 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0}. Threshold 1.0:
/// - terminal e8/e9 d=0.8 SSIM2 regression closed (-1.87 → 0.000 — F=A
///   when gate triggers, byte-identical to pre-W44-117 uniform-4).
/// - terminal e8/e9 d=4 W44-117 wins preserved (still +0.90 vs
///   pre-W44-117 baseline).
/// - terminal e8/e9 d=1.0..1.4 wins preserved (W44-117 still fires).
///
/// Higher thresholds (1.2, 1.5) give back wins above the threshold;
/// lower thresholds (0.8) preserve the regression. 1.0 is the
/// pareto-optimal cutoff: every cell below 1.0 ALREADY had the
/// uniform-4 path producing close-to-optimal recon for buttloop, so
/// the seed compute is pure overhead with no upside.
///
/// **Photos are unaffected**: the W44-118 `is_screenshot` gate already
/// excludes them from the W44-117 mechanism; this distance gate
/// composes underneath and is moot for photos.
///
/// Sweep override: `JXL_W44_120_EPF_SEED_MIN_DISTANCE=<f32>` lets a
/// harness search for per-corpus tuning without rebuilds. Production
/// default is `1.0`. No production runtime cost.
pub const W44_120_EPF_SEED_MIN_DISTANCE: f32 = 1.0;

/// W44-140 EPF sharpness seed distance fade upper bound.
///
/// W44-119 ledger refresh post-W44-118 surfaced a residual SSIM2
/// oscillation cluster on terminal e8/e9 d=1.0-1.6 that W44-120
/// documented as out-of-scope for pure threshold tightening: the
/// W44-117 seed produces SSIM2 wins at d=1.0/d=1.4 (+0.529/+0.685)
/// but regressions at d=1.2/d=1.5 (-0.726/-0.959) — no single
/// `EPF_SEED_MIN_DISTANCE` threshold value closes the regressions
/// without sacrificing the adjacent wins.
///
/// W44-140 adds a distance-aware linear blend between the W44-117
/// sharpness map and uniform-4 in the band
/// `[W44_120_EPF_SEED_MIN_DISTANCE, W44_140_EPF_SEED_FADE_MAX]`:
///
/// ```text
///   weight = clamp((target_distance - min_distance) /
///                  (fade_max - min_distance), 0.0, 1.0)
///   blended[i] = round(weight * w44_117_seed[i] + (1 - weight) * 4)
/// ```
///
/// At `target_distance >= fade_max` weight = 1 → byte-identical to
/// pre-W44-140 main (full W44-117 seed). At `target_distance ==
/// min_distance` weight = 0 → uniform-4 (= legacy seed). Linear
/// interpolation in between.
///
/// W44-140 bisection on terminal e8/e9 × d ∈ {0.8, 1.0, 1.2, 1.4,
/// 1.5, 1.6, 2.0..=5.0} (20 cells) measured 3 candidate fade
/// thresholds {1.5, 2.0, 3.0}:
///
/// | candidate | cluster net ΔSSIM2 vs A_legacy | d=1.4 ΔSSIM2 | d=1.2 ΔSSIM2 |
/// |---|---|---|---|
/// | (pre-W44-140 main) | -0.186 (= 2× sum across e8+e9) | +0.685 | -0.726 |
/// | fade_max = 1.5 (SHIP) | **+1.124** | **+1.014** | +0.129 |
/// | fade_max = 2.0 | +0.282 | -0.197 | +0.186 |
/// | fade_max = 3.0 | +0.614 | +0.907 | 0.000 |
///
/// `fade_max = 1.5` is pareto-optimal: closes d=1.2 (-0.726 → +0.129),
/// boosts d=1.4 (+0.685 → +1.014; the partial blend lands the seed in
/// a sweet spot vs full W44-117), preserves d=1.0 win (0% blend → full
/// uniform-4 → byte-identical to A_legacy = preserves the W44-117 win
/// vs A_legacy of 0 SSIM2). All d >= 1.5 cells byte-identical to
/// pre-W44-140 main (weight = 1.0 → full W44-117 seed unchanged).
///
/// Protection cells:
/// - terminal e8/e9 × d ∈ {2.0, 3.0, 4.0, 5.0}: byte-identical
/// - codec_wiki e8 d=3 (W44-107 protected): byte-identical (weight = 1.0)
/// - photo 1418519 / 1025469 (W44-118 gate): byte-identical (W44-117 doesn't fire)
///
/// Sweep override: `JXL_W44_140_EPF_SEED_FADE_MAX=<f32>` lets a
/// harness search for per-corpus tuning without rebuilds. Production
/// default is `1.5`. Setting to a value `<= W44_120_EPF_SEED_MIN_DISTANCE`
/// disables the blend (full W44-117 seed at every d >= min_distance).
///
/// No production runtime cost when fade_max is `1.5` and target distance
/// is outside `[1.0, 1.5)` (the blend branch isn't taken).
pub const W44_140_EPF_SEED_FADE_MAX: f32 = 1.5;

/// W44-142 (2026-05-20): minimum `m3_colourfulness` zenanalyze proxy at
/// which the W44-117/140 EPF sharpness seed mechanism is suppressed on
/// screenshot-class content in the low-distance band
/// `[W44_120_EPF_SEED_MIN_DISTANCE, W44_142_EPF_SEED_SUPPRESS_MAX_DISTANCE)`.
///
/// **Context**: the W44-141 cjxl-parity ledger refresh
/// (`benchmarks/cjxl_parity_ledger_2026-05-20_w44_141.{tsv,meta}`) on
/// W44-140 main (`b8333091`) surfaced a NEW regression cluster on
/// codec_wiki e8/e9 d=1.2/1.6/1.8 (SSIM2 deltas −0.60 to −0.72 vs
/// W44-134 baseline). codec_wiki is mixed-content (text + diagrams +
/// photo crops) — the W44-140 fade's partial blend over-corrects on
/// codec_wiki at d=1.2 specifically (where the buttloop has fewer
/// iterations to settle vs e9 at the same distance). Terminal (pure
/// text) at the same distance band wants the full W44-117 + W44-140
/// fade unchanged (W44-140 closes the terminal d=1.2 oscillation
/// +0.855 SSIM2, boosts terminal d=1.4 +1.014 above pre-W44-140 main).
///
/// **The split signal** mirrors the W44-124 (`bc9f71eb`) auto-discriminator
/// exactly:
///
/// | image       | m3      | edge_density | W44-142 suppress fires? |
/// |---          |---      |---           |---                      |
/// | codec_wiki  | 145.73  | 0.0396       | **YES** (WANT)          |
/// | terminal    |  13.85  | 0.0874       | no (m3 gate)            |
/// | imac_g3     |  14.29  | 0.1227       | no (m3 gate)            |
/// | imac_dark   |  20.96  | 0.1438       | no (m3 gate)            |
/// | windows95   |  27.19  | 0.3165       | no (m3 + ed)            |
/// | imessage    |  67.65  | 0.0533       | no (ed gate)            |
/// | windows     |  20.04  | 0.1201       | no                      |
/// | graph       |  11.75  | 0.0698       | no                      |
/// | 1189261     |  98.84  | 0.4895       | no (ed gate)            |
/// | 1418519     |  36.84  | 0.1637       | no (m3 + ed)            |
/// | 1420710     |  32.93  | 0.9298       | no                      |
/// | 1531677     |  12.30  | 0.8766       | no                      |
///
/// Both gates load-bearing:
/// - m3 alone rejects terminal/imac_g3/imac_dark/windows95/windows/graph.
/// - ed alone is needed to reject imessage (m3=67.65 passes m3 alone).
/// - Both needed to reject 1189261 (m3=98.84 passes m3, ed=0.4895 fails ed).
///
/// The thresholds are intentionally identical to W44-124's
/// `W44_124_DCT32_KEEP_M3_MIN` and `W44_124_DCT32_KEEP_EDGE_DENSITY_MAX`:
/// both gates address the same codec_wiki-vs-other-screenshots
/// discrimination problem with the same predicate. The 60.0/0.05 split
/// has been validated against the full gb82-sc + CID22 corpus by W44-124.
///
/// Active when:
///   - `target_distance < W44_142_EPF_SEED_SUPPRESS_MAX_DISTANCE` (= 1.5)
///   - `zenanalyze_proxies` is `Some` (8-bit sRGB layouts only)
///   - proxies show m3 ≥ `W44_142_EPF_SEED_SUPPRESS_M3_MIN`
///   - proxies show edge_density < `W44_142_EPF_SEED_SUPPRESS_EDGE_DENSITY_MAX`
///
/// When the gate fires, the W44-117 seed compute is skipped (= legacy
/// uniform-4 seed) AND the W44-140 fade block is skipped (no per-block
/// blend). Equivalent to forcing `EpfSharpnessSeed::LegacyUniform4` for
/// the duration of the call, but only on codec_wiki-class screenshots
/// at low distance. Caller-provided `Some(true)` / `Some(false)` on the
/// `high_d_photo_hint` API still works (unaffected — this is an
/// independent sub-gate inside the EPF seed admission).
///
/// **Why d<1.5 (the W44-140 fade-band edge)**: the W44-142 bisect
/// (`benchmarks/w44_142_max_distance_bisect_2026-05-20.{tsv,meta}`)
/// swept thresholds {1.5, 1.7, 2.0} on 24 cells (16 codec_wiki +
/// 8 terminal protection). Only the d=1.5 threshold limits the gate
/// to cells INSIDE the W44-140 fade band — at d>=1.5 the W44-140 fade
/// is already weight=1.0 (no blend, full W44-117 seed) and the
/// W44-141 cluster regressions at d=1.6/1.8 are NOT caused by W44-140
/// (bytes are byte-identical to pre-W44-140 main at those distances;
/// the W44-141 attribution conflated W44-135's dct32_keep distance
/// gate change with W44-140). Suppressing W44-117 at d=1.6/1.8 only
/// makes some e9 cells better at the cost of e8 cells WORSE
/// (W44-142 e8 d=1.6 SSIM2 -1.02 vs W134; e8 d=1.8 SSIM2 -0.58)
/// — see DO-NOT list below.
///
/// The conservative d<1.5 cutoff:
/// - Closes the only true W44-140-attributable cluster cell (codec_wiki
///   e9 d=1.2 SSIM2 -0.599 vs W134 → -0.012 vs W134, gate met)
/// - Leaves d=1.6/1.8 byte-identical to current main (no new
///   regressions, residual W44-141 cluster remains, but is structurally
///   a W44-135 follow-on, not a W44-140 issue)
/// - Conservative: minimizes scope of behaviour change, follows the
///   "narrow first" pattern from W44-91/96/124.
///
/// **Why d>=1.0 (no explicit lower bound)**: at d=1.0 the W44-140 fade
/// already yields weight=0 → uniform-4, so this gate is a no-op at
/// d=1.0. The W44-120 `min_distance=1.0` gate already prevents W44-117
/// firing at d<1.0. The implicit lower bound is therefore the W44-120
/// gate, not a new threshold.
///
/// Sweep override: `JXL_W44_142_SUPPRESS_DISABLE=1` opts the gate out
/// without a rebuild (sweep harness use).
/// `JXL_W44_142_SUPPRESS_MAX_DISTANCE=<f32>` overrides the upper
/// distance cap for harness search.
pub const W44_142_EPF_SEED_SUPPRESS_M3_MIN: f32 = 60.0;

/// W44-142: maximum `edge_density` zenanalyze proxy at which the W44-142
/// EPF seed suppression sub-gate fires. Belt-and-suspenders with
/// [`W44_142_EPF_SEED_SUPPRESS_M3_MIN`].
///
/// Identical to `W44_124_DCT32_KEEP_EDGE_DENSITY_MAX` (= 0.05) because
/// the same codec_wiki-vs-other-screen discrimination problem applies.
/// All CID22 photos have edge_density ≥ 0.16 (textured high-frequency
/// content), so the gate cannot false-fire on photo content even if a
/// future photo's m3 spiked above 60 (1189261 at m3=98.84 / ed=0.4895
/// is the closest such case — rejected by ed).
///
/// See [`W44_142_EPF_SEED_SUPPRESS_M3_MIN`] for the full rationale and
/// the gb82-sc + CID22 separation table.
pub const W44_142_EPF_SEED_SUPPRESS_EDGE_DENSITY_MAX: f32 = 0.05;

/// W44-142: upper distance bound (exclusive) for the W44-117/140 EPF
/// seed suppression sub-gate on codec_wiki-class content.
///
/// **Why 1.5 (= W44-140 fade-band upper edge)**: the W44-142 bisect
/// (`benchmarks/w44_142_max_distance_bisect_2026-05-20.{tsv,meta}`)
/// measured 4 variants on the W44-141 regression cluster:
///
/// | variant | codec_wiki e9 d=1.2 vs W134 | e9 d=1.6 vs W134 | e8 d=1.6 vs W134 | new regressions |
/// |---|---|---|---|---|
/// | A (current main, no W44-142) | -0.599 | -0.615 | -0.615 | (baseline) |
/// | **B (d<1.5, SHIP)** | **-0.012** ✓ | -0.615 (no fire) | -0.615 (no fire) | 0 |
/// | C (d<1.7) | -0.012 ✓ | -0.037 ✓ | **-1.023** (regress!) | 1 |
/// | D (d<2.0) | -0.012 ✓ | -0.037 ✓ | -1.023 (regress!) | 2 (also e8 d=1.8 -0.58) |
///
/// `B (d<1.5)` is pareto-optimal — closes the only W44-140-attributable
/// regression cell (codec_wiki e9 d=1.2 was -0.60 vs W134; W44-142
/// brings it to -0.01) without introducing new regressions elsewhere.
///
/// Variants C and D appear to close more cells but split asymmetrically
/// on effort: at e9 (4 buttloop iters) uniform-4 lets the buttloop
/// settle to a better state; at e8 (2 iters) uniform-4 starves the
/// buttloop, regressing by ~-0.4 to -1.0 SSIM2 on the same cell. The
/// e8/e9 asymmetry is intrinsic to the W44-117 / W44-140 mechanism
/// at d in [1.5, 2.0) on codec_wiki; pure threshold tightening can't
/// resolve it (the same shape that drove the W44-140 fade design).
///
/// **What's NOT closed by W44-142 ship**: codec_wiki d=1.6/1.8
/// regressions (-0.62 / -0.72 SSIM2 vs W134) persist. Per W44-142
/// investigation, these are NOT caused by W44-140 (bytes at d>=1.5
/// are byte-identical pre/post-W44-140 main); they are caused by
/// W44-135's dct32_keep distance gate change (commit `2b9c98d0`)
/// reverting the W44-124 lift on codec_wiki at d<2.0. The W44-141
/// memo conflated W44-135's effect with W44-140's. Fixing d=1.6/1.8
/// requires revisiting W44-135's distance gate (separate chunk,
/// W44-143 follow-on candidate).
///
/// Caller override: `LossyConfig::with_high_d_photo_hint(Some(true))`
/// or `Some(false)` is not directly tied to this gate (the
/// suppression is internal to the EPF seed admission, gated separately
/// from the high-d photo lift table). Sweep override:
/// `JXL_W44_142_SUPPRESS_DISABLE=1` opts the entire suppression out;
/// `JXL_W44_142_SUPPRESS_MAX_DISTANCE=<f32>` overrides this cap for
/// harness search.
pub const W44_142_EPF_SEED_SUPPRESS_MAX_DISTANCE: f32 = 1.5;

/// W44-109: scale factor applied to `quant_field_float` at adaptive-quant
/// time on screenshot-class content at low effort. **Effort-dependent
/// magnitude** (e5/e6 vs e7): without the buttloop the scale propagates
/// straight to the shipped quant_field with no settling correction, so
/// the W44-105 magnitude of 4× would inflate bytes ~78% on e5/e6 d=4.
///
/// Empirical sweep on terminal.png e5/e6/e7 d=4 (atomic env override
/// `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE`):
///
/// | scale | e5 bytes Δ | e5 SSIM2 Δ | e6 bytes Δ | e6 SSIM2 Δ | e7 bytes Δ | e7 SSIM2 Δ |
/// |-------|------------|------------|------------|------------|------------|------------|
/// | 1.0   | -3.81%     | -5.38      | -2.50%     | -5.29      | -7.07%     | -4.62      |
/// | 1.5   | +16.7%     | -3.16      | —          | —          | —          | —          |
/// | 2.0   | +32.8%     | -1.93      | +32.4%     | -1.60      | +12.3%     | -2.51      |
/// | 2.5   | +46.6%     | -1.41      | +45.0%     | -1.03      | +20.8%     | -2.43      |
/// | 3.0   | +58.0%     | -0.54      | —          | —          | +29.2%     | -1.94      |
/// | 4.0   | +77.9%     | +0.21      | +76.1%     | +0.55      | +42.2%     | -1.58      |
///
/// At e5/e6 the scale chases the W44-105 8/e9 SSIM2 wins (+3 to +4 SSIM2)
/// at a steep bytes cost; we cap at 2.0 to keep bytes within ~30% of
/// pre-W44-109 baseline. At e7 the baseline ssim2 is already higher
/// (78.4 vs 83.0 vs e5) so a larger scale (3.0) is needed to reach the
/// same +2.5 SSIM2 improvement — and pays a smaller relative bytes
/// cost (+29%) because e7's per-pixel work is finer-grained.
///
/// The W44-105 buttloop default of 4.0 is preserved for e>=8 and
/// shared via [`DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE`].
pub const DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6: f32 = 2.0;

/// W44-109: scale factor at e7 (the highest pre-buttloop effort) where
/// per-pixel quality is already much closer to e8 (baseline SSIM2 83.0
/// vs e5's 78.4). A larger scale is needed to reach the same +2.5
/// SSIM2 improvement, and pays a smaller relative bytes cost.
pub const DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7: f32 = 3.0;

/// W44-109: compute the screenshot-class qf seed scale to apply at
/// adaptive_quant time (low-effort path), or `1.0` to leave qf
/// untouched.
///
/// Mirrors the W44-105/107/108 buttloop gate predicate exactly, with
/// two additional guards:
///
/// 1. `effort <= [`ADAPTIVE_QUANT_QF_SEED_SCALE_MAX_EFFORT`]` — at
///    e>=8 the W44-105 buttloop path applies its own seed scale and
///    we must not double-apply (would multiply 4× × 4× = 16×).
/// 2. `butteraugli_iters == 0` — belt-and-braces double-check; the
///    W44-105 path fires iff `butteraugli_iters > 0`. This is true at
///    e>=8 by `profile.butteraugli_iters` (see effort.rs:956) but
///    callers can also pin `LossyConfig::with_butteraugli_iters(0)` at
///    high effort, in which case BOTH fixes are off — correct per the
///    "buttloop unavailable, no scale" semantics.
///
/// Returns the screenshot seed scale
/// ([`DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE`]) when the
/// gate fires, else `1.0`. Atomic env override
/// `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE` allows harness sweeps without
/// rebuild (no production effect when unset).
#[cfg_attr(not(feature = "std"), allow(unused_variables))]
pub(crate) fn resolved_adaptive_quant_qf_seed_scale(
    effort: u8,
    butteraugli_iters: u32,
    is_screenshot: bool,
    target_distance: f32,
    m3_colourfulness: Option<f32>,
) -> f32 {
    resolved_adaptive_quant_qf_seed_scale_with_policy(
        effort,
        butteraugli_iters,
        is_screenshot,
        target_distance,
        m3_colourfulness,
        crate::api::AdaptiveQuantQfSeedPolicy::AutoScalePerEffort,
    )
}

/// W44-129 Chunk C variant of [`resolved_adaptive_quant_qf_seed_scale`]
/// that consults the resolved [`crate::api::AdaptiveQuantQfSeedPolicy`]
/// enum from `ResolvedImprovements`.
///
/// The legacy unpolicy'd entry point delegates to this with
/// `AutoScalePerEffort` to preserve the pre-Chunk-C behaviour for any
/// remaining callers (tests / examples) that don't carry a resolved
/// policy.
#[cfg_attr(not(feature = "std"), allow(unused_variables))]
pub(crate) fn resolved_adaptive_quant_qf_seed_scale_with_policy(
    effort: u8,
    butteraugli_iters: u32,
    is_screenshot: bool,
    target_distance: f32,
    m3_colourfulness: Option<f32>,
    policy: crate::api::AdaptiveQuantQfSeedPolicy,
) -> f32 {
    // Off policy short-circuits before the gate evaluation: Libjxl
    // strategy never pre-scales.
    if matches!(policy, crate::api::AdaptiveQuantQfSeedPolicy::Off) {
        return 1.0;
    }
    if effort > ADAPTIVE_QUANT_QF_SEED_SCALE_MAX_EFFORT {
        return 1.0;
    }
    if butteraugli_iters > 0 {
        // W44-105 buttloop path will apply the scale — don't double-apply.
        return 1.0;
    }
    if !is_screenshot {
        return 1.0;
    }
    let w44_108_low_colour =
        m3_colourfulness.is_some_and(|m3| m3 < BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX);
    let gate_fires = target_distance >= BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE
        || (w44_108_low_colour && target_distance >= BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE);
    if !gate_fires {
        return 1.0;
    }
    // Effort-dependent magnitude: e5/e6 cap at 2.0 to bound bytes
    // regression; e7 lifts to 3.0 to clear the +2.5 SSIM2 gate (baseline
    // ssim2 at e7 is much closer to cjxl, so it needs more boost to
    // overshoot the buttloop-measurement gap by the same margin).
    //
    // Policy translation:
    //   * `AutoScalePerEffort` (default) → the production 2.0/3.0 split
    //   * `AutoScaleCustom { e5_e6, e7 }` → caller-supplied per-effort
    //     scales (replaces the 2.0/3.0 defaults but keeps the same gate
    //     predicate above).
    //   * `Off` → handled above (early return 1.0).
    let base_scale = match policy {
        crate::api::AdaptiveQuantQfSeedPolicy::AutoScalePerEffort => {
            if effort >= 7 {
                DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7
            } else {
                DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6
            }
        }
        crate::api::AdaptiveQuantQfSeedPolicy::AutoScaleCustom { e5_e6, e7 } => {
            if effort >= 7 {
                e7
            } else {
                e5_e6
            }
        }
        crate::api::AdaptiveQuantQfSeedPolicy::Off => unreachable!(),
    };
    // W44-132 Chunk F: env-var `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE` is
    // now consumed inside `EncoderStrategy::resolve`'s env-var fallback
    // layer (api.rs::apply_env_var_fallbacks) — when the policy is at
    // its `Default::default()` (== `AutoScalePerEffort`), the env-var
    // value promotes it to `AutoScaleCustom { e5_e6: env, e7: env }`
    // BEFORE this function sees it. The single env value replaces both
    // per-effort defaults. Explicit caller settings via
    // `EncoderStrategy::Custom` or `StrategyOverrides` win over the
    // env-var; the legacy "env-var always overrides" semantic ended
    // with Chunk F. See `apply_env_var_fallbacks` for the rationale.
    base_scale
}

/// W44-145: lower edge of the per-block mask1x1 band where the W44-109
/// adaptive-quant qf scale ramps from baseline (1.0) to the full
/// per-effort scale ([`DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6`]
/// or [`DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7`]).
///
/// **Why this exists**: W44-144 Phase 1 dump confirmed W44-109 raises
/// qac UNIFORMLY by 2-3× on terminal d=4 e5/e6/e7 — every block now
/// runs at the lifted scale. cjxl's buttloop on the same content
/// produces a BIMODAL qac pattern (text-glyph blocks at qac ~80-97,
/// blank background blocks at qac ~7). The blanket scale costs +30%
/// bytes by over-quantizing blank regions where cjxl spends ~1× qac.
///
/// `mask1x1` in `compute_mask1x1` is `1 / (log1p(diff) + 0.01)` — HIGH
/// values mark uniform / flat regions (blank), LOW values mark sharp
/// edges (text glyphs). On terminal blocks:
/// - blank background: mask1x1 saturates at ~100 (`log1p(0) = 0` → 1/0.01 = 100)
/// - text glyphs: mask1x1 drops to 20-60 (`log1p(0.5..1.5) = 0.4..0.9`)
/// - mixed edges: mask1x1 in between
///
/// W44-145 routes the W44-109 scale through a per-block lookup:
/// - per-block mask1x1 mean ≥ [`W44_145_PER_BLOCK_MASK_HIGH`] (= 99.5): scale = 1.0
///   (blank — no scaling, keeps qac at baseline ~7-8)
/// - per-block mask1x1 mean ≤ [`W44_145_PER_BLOCK_MASK_LOW`] (= 70.0): scale = full W44-109
///   (text — gets the full 2×/3× lift, matching cjxl's bimodal text-qac)
/// - linear interpolation in between
///
/// The thresholds sit inside the screenshot-class mask range (median > 95
/// from `SCREENSHOT_MEDIAN_THRESHOLD`): blank screenshot blocks saturate
/// above 99.5, text/glyph blocks fall below 70. Synthetic flat
/// fixtures (hash-locks) stay at mask ≈ 100 (smooth gradient → no Laplacian
/// activity → 1/0.01), so the per-block scale collapses to 1.0 everywhere
/// and the gate's outer is_screenshot predicate is also false on those
/// fixtures (W44-109 gate predicates are inherited unchanged).
pub const W44_145_PER_BLOCK_MASK_LOW: f32 = 70.0;

/// W44-145: upper edge of the per-block mask1x1 band. See
/// [`W44_145_PER_BLOCK_MASK_LOW`] for full context.
pub const W44_145_PER_BLOCK_MASK_HIGH: f32 = 99.5;

/// W44-145: linearly interpolate the W44-109 per-effort `full_scale` based
/// on a single 8×8 block's mean `mask1x1`. Returns the per-block scale
/// to apply to `quant_field_float` for that block.
///
/// `block_mask_mean = mean(mask1x1[y*W+x] for (x,y) in block_pixels)`.
/// `full_scale` is the W44-109 per-effort scale (2.0 at e5/e6, 3.0 at e7).
///
/// Mapping:
/// - `block_mask_mean >= W44_145_PER_BLOCK_MASK_HIGH` → returns 1.0 (no scaling)
/// - `block_mask_mean <= W44_145_PER_BLOCK_MASK_LOW` → returns `full_scale`
/// - between thresholds: linear blend
///
/// Hot path: this runs once per 8×8 block (≈xsize_blocks × ysize_blocks
/// iterations), which is `nblocks` = padded_width × padded_height / 64.
/// A 1646×1062 terminal image has ~27k blocks. Pure arithmetic; no
/// allocation. The mean lookup table is computed once per encode via
/// [`per_block_mask1x1_mean`].
///
/// **Currently unused in production** (W44-145 HONEST-STOP — see
/// `vardct/encoder.rs` qf_pre_scale apply site). Retained for the
/// reproducer + potential future e8+ application where cjxl actually
/// has bimodal qac structure (terminal e8/e9 post-buttloop).
#[inline]
#[allow(dead_code)]
pub(crate) fn w44_145_per_block_qf_scale(block_mask_mean: f32, full_scale: f32) -> f32 {
    if full_scale == 1.0 {
        return 1.0;
    }
    if block_mask_mean >= W44_145_PER_BLOCK_MASK_HIGH {
        return 1.0;
    }
    if block_mask_mean <= W44_145_PER_BLOCK_MASK_LOW {
        return full_scale;
    }
    // Linear blend: scale = 1 + t * (full_scale - 1) where t is the
    // distance from HIGH (blank) toward LOW (text). HIGH → t=0 (scale=1),
    // LOW → t=1 (scale=full).
    let t = (W44_145_PER_BLOCK_MASK_HIGH - block_mask_mean)
        / (W44_145_PER_BLOCK_MASK_HIGH - W44_145_PER_BLOCK_MASK_LOW);
    1.0 + t * (full_scale - 1.0)
}

/// W44-145: compute per-8×8-block mean of a per-pixel `mask1x1` plane
/// of dimensions `padded_width × padded_height` where
/// `padded_width = xsize_blocks * 8` and `padded_height = ysize_blocks * 8`.
///
/// Returns a `Vec<f32>` of length `xsize_blocks * ysize_blocks` in
/// row-major block order (by * xsize_blocks + bx). Each entry is the
/// mean of the 64 mask1x1 pixels covering that block.
///
/// **No size budget reservation**: callers already account a `nblocks`
/// f32 buffer when they own `quant_field_float`; the returned vector
/// is the same size and lives for the duration of the qf-pre-scale
/// pass (dropped after the multiply loop).
///
/// **Currently unused in production** (W44-145 HONEST-STOP — see
/// [`w44_145_per_block_qf_scale`]). Retained for the reproducer +
/// potential future e8+ application.
#[allow(dead_code)]
pub(crate) fn per_block_mask1x1_mean(
    mask1x1: &[f32],
    padded_width: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> alloc::vec::Vec<f32> {
    let nblocks = xsize_blocks * ysize_blocks;
    let mut means = alloc::vec::Vec::with_capacity(nblocks);
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let mut sum: f32 = 0.0;
            for dy in 0..8 {
                let y = by * 8 + dy;
                let row_off = y * padded_width;
                for dx in 0..8 {
                    let x = bx * 8 + dx;
                    sum += mask1x1[row_off + x];
                }
            }
            means.push(sum * (1.0 / 64.0));
        }
    }
    means
}

/// Resolve `cur_pow` for the current iter + `target_distance`, honouring
/// any sweep overrides set in `CUR_POW_X1000_{LOW,HIGH}`.
///
/// Returns 0.0 for `iter >= 2` regardless of override (only iter < 2
/// has a good-block reclamation regime; later iters only bump bad
/// blocks — same as libjxl `enc_adaptive_quantization.cc:1106`).
pub(crate) fn resolved_cur_pow(iter: usize, target_distance: f64) -> f64 {
    if iter >= 2 {
        return 0.0;
    }
    let split = read_override_x1000(&DISTANCE_SPLIT_X1000, DEFAULT_DISTANCE_SPLIT);
    if target_distance < split {
        read_override_x1000(&CUR_POW_X1000_LOW, DEFAULT_CUR_POW_LOW)
    } else {
        read_override_x1000(&CUR_POW_X1000_HIGH, DEFAULT_CUR_POW_HIGH)
    }
}

/// Resolve `max_increase` (per-iter bad-block bump cap) for the current
/// `target_distance`, honouring sweep overrides. Content-class-blind —
/// equivalent to [`resolved_max_increase_with_class`] called with
/// `is_screenshot = false`. Retained as a stable shim for harness /
/// test code that doesn't have a content class to pass.
///
/// Production callers should prefer [`resolved_max_increase_with_class`]
/// directly so the W39-2 WF3 fix can engage on screenshot-class input.
#[allow(dead_code)]
pub(crate) fn resolved_max_increase(target_distance: f64) -> f64 {
    resolved_max_increase_with_class(target_distance, false)
}

/// Resolve `max_increase` for the current `target_distance` AND content
/// class, honouring sweep overrides.
///
/// Only the HIGH regime (`target_distance >= DEFAULT_DISTANCE_SPLIT`) +
/// `is_screenshot = true` path reads the screenshot-specific slot
/// ([`MAX_INCREASE_X1000_HIGH_SCREENSHOT`] →
/// [`DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT`]). Every other combination
/// reproduces [`resolved_max_increase`] exactly so the public defaults
/// are byte-identical on photo-class content.
///
/// This is the "real WF3 fix" the W39-1 commit documented (W39-1
/// scaffolding shipped the atomic infrastructure; this helper wires
/// content-class dispatch on top).
pub(crate) fn resolved_max_increase_with_class(target_distance: f64, is_screenshot: bool) -> f64 {
    let split = read_override_x1000(&DISTANCE_SPLIT_X1000, DEFAULT_DISTANCE_SPLIT);
    if target_distance < split {
        // LOW regime: no per-class split (W39-1 confirmed the literal
        // GPU LOW tuning regresses CPU; not the place for the WF3 fix).
        read_override_x1000(&MAX_INCREASE_X1000_LOW, DEFAULT_MAX_INCREASE_LOW)
    } else if is_screenshot {
        // HIGH regime + screenshot class → consult the screenshot slot
        // first; fall back to the photo HIGH default when neither is
        // overridden so the public defaults stay byte-identical.
        let screenshot = read_override_x1000(
            &MAX_INCREASE_X1000_HIGH_SCREENSHOT,
            DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT,
        );
        // If the screenshot slot is at its "no cap" default and the
        // shared HIGH slot is overridden (sweep harness), honour the
        // harness so old sweeps keep working unchanged. The "min" picks
        // the more-restrictive cap when both are present.
        let shared = read_override_x1000(&MAX_INCREASE_X1000_HIGH, DEFAULT_MAX_INCREASE_HIGH);
        screenshot.min(shared)
    } else {
        // HIGH regime + photo class: libjxl default (no cap), honouring
        // any harness override on the shared HIGH slot.
        read_override_x1000(&MAX_INCREASE_X1000_HIGH, DEFAULT_MAX_INCREASE_HIGH)
    }
}

/// Seed values for the multi-seed butteraugli sweep (RFC#45 pick #1
/// chunk 3). Each seed runs the full quantization loop with a different
/// `kInitMul` (the constant that biases iter=1 toward the initial field
/// vs the per-iteration update). Different basins of the optimization
/// surface converge to different (qf, scale) pairs at the same butteraugli
/// target — we pick the seed with the largest mean(quant_field_float)
/// (proxy for smallest encoded bytes) that meets the butteraugli bound.
///
/// **Index 0 is ALWAYS the libjxl default** so the picker can never
/// regress below the single-seed baseline.
///
/// - `seeds = 1` ⇒ `[0.6]` — bit-identical to libjxl `FindBestQuantization`.
/// - `seeds = 2` ⇒ `[0.6, 0.4]` — adds a "trust the per-iter update more"
///   basin (smaller pullback toward initial → larger qf perturbation).
/// - `seeds = 3` ⇒ `[0.6, 0.4, 0.8]` — adds a "trust the initial more"
///   basin (more conservative; smaller qf perturbation, often hits
///   target with finer quant on noisy inputs).
/// - `seeds = 4` ⇒ `[0.6, 0.4, 0.8, 0.5]` — fills the gap near the
///   default with a fourth basin.
///
/// Capped at the length returned here (4); requesting more silently
/// saturates. The values are chosen empirically to span the
/// `kInitMul ∈ [0, 1]` interpolation interval without clustering near
/// the endpoints (where the loop degenerates to pure-update or
/// pure-pullback dynamics).
pub(crate) fn init_mul_seeds(seeds: u8) -> &'static [f64] {
    const ALL: [f64; 4] = [LIBJXL_INIT_MUL, 0.4, 0.8, 0.5];
    let n = (seeds.max(1) as usize).min(ALL.len());
    &ALL[..n]
}

/// Outcome of one butteraugli-loop seed used by the multi-seed picker
/// in [`VarDctEncoder::butteraugli_refine_quant_field`].
#[derive(Clone)]
struct SeedOutcome {
    /// Final `DistanceParams` after the loop's terminal SetQuantField.
    params: DistanceParams,
    /// `u8` quant_field after the loop's terminal SetQuantField (length
    /// `xsize_blocks * ysize_blocks`).
    quant_field: alloc::vec::Vec<u8>,
    /// Float quant_field at loop exit (length matches `quant_field`).
    quant_field_float: alloc::vec::Vec<f32>,
    /// Butteraugli score from the compare-only last iteration (`f64::INFINITY`
    /// if the reference compare failed at any point).
    final_score: f64,
    /// Mean of the final float quant_field — the smallest-bytes proxy
    /// (larger = coarser quantization = fewer non-zero coefficients).
    mean_qf: f64,
    /// `k_init_mul` value used for this seed (for debug logging).
    /// Only read when the `debug-rect` feature is enabled.
    #[cfg_attr(not(feature = "debug-rect"), allow(dead_code))]
    k_init_mul: f64,
}

impl VarDctEncoder {
    /// Butteraugli quantization loop: iteratively refines per-block quant_field
    /// by measuring perceptual distance (butteraugli) between the original image
    /// and the reconstruction from quantized coefficients.
    ///
    /// **Float-domain operation** (matching libjxl FindBestQuantization):
    /// The quant field is maintained in float domain (~0.3-1.5 range). Each
    /// iteration recomputes global_scale from the float field's median/MAD
    /// (matching libjxl's SetQuantField), then converts to u8 for quantization.
    ///
    /// Algorithm:
    /// For each iteration:
    ///   1. SetQuantField: recompute global_scale from float field, convert to u8
    ///   2. transform_and_quantize with current quant_field and new params
    ///   3. reconstruct XYB → apply gab → EPF → XYB-to-linear
    ///   4. butteraugli(original_linear, reconstructed_linear) → per-block distmap
    ///   5. Adjust float quant_field based on tile distances
    ///   6. Enforce deviation bounds from initial field
    ///
    /// AC strategy is FIXED throughout — only quant_field changes.
    /// Returns the final DistanceParams (with recomputed global_scale).
    #[cfg(feature = "butteraugli-loop")]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn butteraugli_refine_quant_field(
        &self,
        linear_rgb: &[f32],
        width: usize,
        height: usize,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        padded_width: usize,
        padded_height: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        initial_params: &DistanceParams,
        quant_field: &mut [u8],
        quant_field_float: &mut [f32],
        initial_quant_field_float: &[f32],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        patches_data: Option<&super::patches::PatchesData>,
        splines_data: Option<&super::splines::SplinesData>,
        // W39-2 (WF3 fix): content-class hint computed once at the
        // call site from `median(mask1x1) > SCREENSHOT_MEDIAN_THRESHOLD`.
        // `true` switches the HIGH-regime `max_increase` resolution to
        // [`resolved_max_increase_with_class`]'s screenshot slot. `false`
        // (default for photo / unknown / mask-not-computed) is
        // byte-identical to pre-W39-2 behaviour.
        is_screenshot: bool,
        // W44-117: optional precomputed mask1x1 (padded, length
        // `padded_width * padded_height`). When provided AND
        // `epf_iters > 0` AND `epf_dynamic_sharpness`, we run
        // `compute_epf_sharpness` once before the loop on the initial
        // quant_dc/quant_ac, then use the resulting per-block sharpness
        // map for every buttloop iter's `apply_epf` (instead of the
        // uniform `vec![4u8; num_blocks]` seed). This closes the
        // ~0.05-0.17 max-abs linear-RGB residual identified by W44-116
        // between the buttloop's INTERNAL reconstruction (which jxl-rs
        // sees) and the shipped bitstream's production-sharpness
        // reconstruction. When `None` (e.g. animation frames, ssim2
        // path), falls back to the legacy uniform-4 seed (byte-identical
        // to pre-W44-117 behaviour).
        mask1x1: Option<&[f32]>,
    ) -> Result<DistanceParams> {
        use crate::budget::MemoryBudget;

        // EX-J11 chunk 2: HDR-aware loss dispatch.
        //
        // Default path (`HdrLoss::Butteraugli`) is byte-identical to
        // every release prior to EX-J11 — the existing butteraugli
        // reference setup + per-iter compare below runs unchanged.
        //
        // `HdrLoss::Vdp2` (opt-in via [`crate::LossyConfig::with_hdr_loss`])
        // skips the butteraugli reference precompute and routes each
        // per-iter compare through [`super::hdr_vdp2_lite::compare_vdp2_planar`]
        // — a calibrated subset of HDR-VDP-2 that adapts to the encode's
        // `intensity_target`. See the module docs in
        // [`super::hdr_vdp2_lite`] for the deviations from the full paper
        // (cortex channels, chromatic sensitivity, masking model — all
        // chunk-3 follow-ons).
        if let Err(e) = super::hdr_metrics::validate_loss(self.hdr_loss) {
            return Err(crate::error::Error::NotImplemented(alloc::format!(
                "HDR loss dispatch: {e} (selected: {})",
                self.hdr_loss.as_str()
            )));
        }
        // EX-J11 chunk 4: belt-and-braces resolve of `HdrLoss::Auto`.
        // The public LossyConfig pipeline calls
        // `LossyConfig::resolve_hdr_loss(...)` before assigning
        // `enc.hdr_loss`, so by the time we reach this loop `Auto`
        // has normally been replaced by `Butteraugli` or `Vdp2`.
        // Direct construction of `VarDctEncoder` (e.g. from tests or
        // internal callers) may still leave `Auto` here — resolve with
        // `None` (no transfer-function hint available at this layer)
        // so we land on the SDR-safe `Butteraugli` path.
        let resolved_loss = self.hdr_loss.resolve(None);
        let use_vdp2 = matches!(resolved_loss, super::hdr_metrics::HdrLoss::Vdp2);

        let budget = self.budget.as_ref();
        let target_distance = self.distance;
        let num_blocks = xsize_blocks * ysize_blocks;
        let padded_pixels = padded_width * padded_height;

        // W39-2 diagnostic — gated by env var so it's free in normal
        // runs. Set `JXL_BUTTLOOP_W39_DEBUG=1` to see the screenshot
        // classification + cap-resolution per encode.
        #[cfg(feature = "std")]
        if std::env::var("JXL_BUTTLOOP_W39_DEBUG").is_ok() {
            let cap = resolved_max_increase_with_class(target_distance as f64, is_screenshot);
            eprintln!(
                "[W39-2 buttloop] dist={:.3} is_screenshot={} resolved_max_increase={:.3} iters={}",
                target_distance, is_screenshot, cap, self.butteraugli_iters,
            );
        }

        // Precompute the perceptual reference from the original image ONCE.
        // Deinterleave to planar so both metric paths consume the same layout.
        //
        // For `HdrLoss::Butteraugli` we additionally build a `ButteraugliReference`
        // (the cached separated-frequencies + masking precompute). For
        // `HdrLoss::Vdp2` we skip that precompute — VDP2-lite has no separable
        // per-image cache; it walks both planes per-iter (the pyramid construction
        // dominates and is only sub-linear in the image size).
        //
        // The planar `ref_r/g/b` planes are kept alive for the full loop
        // duration so the VDP2 path can re-use them across iterations. Budget
        // is reserved permanently in the VDP2 branch (vs the butteraugli branch
        // where the planes are released after the reference precompute takes
        // ownership of an internal copy).
        let n = width * height;
        let mut ref_r = vec![0.0f32; n];
        let mut ref_g = vec![0.0f32; n];
        let mut ref_b = vec![0.0f32; n];
        for i in 0..n {
            ref_r[i] = linear_rgb[i * 3];
            ref_g[i] = linear_rgb[i * 3 + 1];
            ref_b[i] = linear_rgb[i * 3 + 2];
        }
        // intensity_target the VDP2-lite path uses to map linear-RGB [0,1]
        // onto absolute display luminance in cd/m². Pulled from the
        // VarDctEncoder field that the public LossyConfig::with_intensity_target
        // setter populates. SDR encodes default to 255.0, matching the
        // existing initialiser in vardct/encoder.rs:549.
        let vdp2_intensity_target = self.intensity_target;

        let reference: Option<butteraugli::ButteraugliReference> = if use_vdp2 {
            // VDP2 path: hold onto the planar refs permanently (one
            // n*4*3 reservation) and skip the butteraugli precompute.
            MemoryBudget::reserve_permanent_opt(budget, (n as u64).saturating_mul(4 * 3))?;
            None
        } else {
            // Butteraugli path: transient n*4*3 reservation released as
            // soon as the reference precompute owns its internal copy.
            let _g = MemoryBudget::reserve_opt(budget, (n as u64).saturating_mul(4 * 3))?;
            let butteraugli_params = butteraugli::ButteraugliParams::new()
                .with_intensity_target(80.0)
                .with_compute_diffmap(true);
            let r = match butteraugli::ButteraugliReference::new_linear_planar(
                &ref_r,
                &ref_g,
                &ref_b,
                width,
                height,
                width,
                butteraugli_params,
            ) {
                Ok(r) => r,
                Err(_) => return Ok(initial_params.clone()),
            };
            Some(r)
        };

        // Compute deviation bounds from the FLOAT initial field (libjxl lines 968-976).
        // These prevent the quant field from diverging too far from the initial field.
        let initial_qf_min = initial_quant_field_float
            .iter()
            .copied()
            .reduce(f32::min)
            .unwrap_or(0.01)
            .max(1e-6);
        let initial_qf_max = initial_quant_field_float
            .iter()
            .copied()
            .reduce(f32::max)
            .unwrap_or(1.0);
        let initial_qf_ratio = initial_qf_max / initial_qf_min;
        let qf_max_deviation_low = (250.0f32 / initial_qf_ratio).sqrt();
        let asymmetry = 2.0f32.min(qf_max_deviation_low);
        let qf_lower = initial_qf_min / (asymmetry * qf_max_deviation_low);
        let qf_higher = initial_qf_max * (qf_max_deviation_low / asymmetry);

        // W44-105 production fix: on screenshot-class content at d>=2 + e>=8 buttloop,
        // scale up the initial quant_field_float by a content-aware factor BEFORE the
        // loop runs. Rationale: butteraugli's metric is too lenient on text-heavy
        // screenshot reconstructions — our iter-0 reconstruction reports score ≈ 2
        // vs target=4 (lots of headroom), so the loop reduces precision globally
        // (`cur_pow=0.2` path on "good" blocks), starving text blocks of qac. cjxl
        // produces internal buttloop score 47.7 on the same initial qf (per
        // libjxl debug patch verified W44-105), then refines text qac to 97+ via
        // `bad block` bumps. Our reconstruction doesn't trigger the bad-block path
        // because butteraugli reports our text as fine.
        //
        // The fix: start the buttloop from a coarser qf field on
        // screenshot-class content. The loop then naturally refines DOWN via
        // `cur_pow=0.2` but settles at a higher equilibrium that preserves text
        // qac. Verified A/B on terminal e8 d=4 (W44-103 wedge cell):
        //   scale=1.0 → bytes=28088 SSIM2=81.84 bfly=2.80 (default — wedge)
        //   scale=4.0 → bytes=36788 SSIM2=85.26 bfly=1.76 (+31% bytes, +3.42 SSIM2)
        //   scale=6.0 → bytes=41187 SSIM2=86.70 bfly=1.84 (+47% bytes, +4.86 SSIM2)
        //   cjxl     → bytes=56066 SSIM2=87.58 bfly=2.52 (reference)
        //
        // Even at SCALE=10 we ship FEWER bytes than cjxl (48k vs 56k) with
        // matching SSIM2 (87.5 vs 87.6) — the scale-up doesn't cause runaway
        // bytes, it just gives the buttloop a wider exploration window.
        //
        // Gated on:
        //   - `is_screenshot` (median(mask1x1) > SCREENSHOT_MEDIAN_THRESHOLD=95)
        //   - target_distance >= BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE (= 3.5
        //     after W44-107; was 2.0 in W44-105 — the W44-106 ledger refresh
        //     showed the d=2..3 regime regressed codec_wiki e8 d=3 from
        //     FIXED to OPEN. Tightening the gate to d>=3.5 preserves the
        //     d=4+ wins (the largest cluster) and reverts the codec_wiki
        //     d=3 cell to the pre-W44-105 baseline; the sacrificed d=2/2.5
        //     wins remain within FIXED status per W44-106 paired data)
        //   - butteraugli_iters > 0 (only applies when buttloop runs at all)
        //
        // W44-108 sub-discriminator (admits the d in [2.0, 3.5) band the
        // W44-107 tightening sacrificed): when the image's
        // `m3_colourfulness` is below
        // `BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX` (= 30.0), fire the
        // 4× seed scale at d >= `BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE`
        // (= 2.0) instead of waiting for d >= 3.5. The probe
        // (`examples/w44_108_proxy_probe.rs`) showed terminal (M3=14),
        // imac_g3 (M3=14), imac_dark (M3=21) cleanly separated from
        // codec_wiki (M3=146); the 30.0 threshold has ~5× margin both
        // sides. Photos already fail `is_screenshot` (mask<<95), so this
        // sub-gate cannot fire on them. Reads
        // `self.zenanalyze_proxies` — `None` (defaults / non-sRGB-u8
        // input layouts) falls back to the W44-107 gate
        // (byte-identical to that path).
        //
        // The atomic override below lets sweep harnesses tune the scale per-class
        // without rebuilds; production defaults to 4.0 (SCALE=4 from the sweep).
        // W44-129 Chunk C: read the resolved `buttloop_qf_seed` enum from
        // `ResolvedImprovements` (populated by Chunk B
        // `LossyConfig::resolve_improvements`).
        //
        // Policy translation:
        //   * `AutoScale4` (default) → existing W44-105/107/108 auto gate
        //     with scale `DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE` = 4.0
        //   * `AutoScale(s)` → same gate predicate but caller picks the scale
        //   * `ForceScale(s)` → no gate; always scale by `s` (sweep harness)
        //   * `Off` → never scale; `scale == 1.0` (Libjxl strategy)
        //
        // W44-130 Chunk D: `resolved_improvements` is now always
        // populated (default `AutoScale4` for direct
        // `VarDctEncoder::new` test callers) — bit-identical to
        // pre-Chunk-D.
        let buttloop_qf_seed_policy = self.resolved_improvements.buttloop_qf_seed;
        let w44_108_low_colour = self
            .zenanalyze_proxies
            .is_some_and(|p| p.m3_colourfulness < BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX);
        let auto_gate_fires = is_screenshot
            && (target_distance >= BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE
                || (w44_108_low_colour
                    && target_distance >= BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE));
        let buttloop_qf_seed_scale = match buttloop_qf_seed_policy {
            crate::api::ButtloopQfSeedPolicy::AutoScale4 => {
                if auto_gate_fires {
                    DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE
                } else {
                    1.0
                }
            }
            crate::api::ButtloopQfSeedPolicy::AutoScale(s) => {
                if auto_gate_fires {
                    s
                } else {
                    1.0
                }
            }
            crate::api::ButtloopQfSeedPolicy::ForceScale(s) => s,
            crate::api::ButtloopQfSeedPolicy::Off => 1.0,
        };

        // W44-132 Chunk F: env-var `JXL_BUTTLOOP_INITIAL_QF_SCALE` is
        // now consumed inside `EncoderStrategy::resolve`'s env-var
        // fallback layer (api.rs::apply_env_var_fallbacks) — when the
        // policy is at its `Default::default()` (== `AutoScale4`), the
        // env value promotes it to `AutoScale(env_value)` BEFORE this
        // call site sees it (still subject to the gate). Explicit
        // caller settings via `EncoderStrategy::Custom` or
        // `StrategyOverrides` win over the env-var; the legacy
        // "env-var always overrides" semantic ended with Chunk F. See
        // `apply_env_var_fallbacks` for the rationale.

        if buttloop_qf_seed_scale != 1.0 {
            for v in quant_field_float.iter_mut() {
                *v *= buttloop_qf_seed_scale;
            }
        }

        // Pre-allocate buffers reused across iterations.
        // These live for the duration of the loop — accounted permanently.
        // sharpness is u8, tile_dist is f32 of num_blocks, recon_* are f32 of padded_pixels.
        MemoryBudget::reserve_permanent_opt(
            budget,
            (num_blocks as u64)
                .saturating_add((num_blocks as u64).saturating_mul(4))
                .saturating_add((padded_pixels as u64).saturating_mul(4 * 3)),
        )?;
        let mut tile_dist = vec![0.0f32; num_blocks];
        let mut recon_r = vec![0.0f32; padded_pixels];
        let mut recon_g = vec![0.0f32; padded_pixels];
        let mut recon_b = vec![0.0f32; padded_pixels];
        let mut transform_out =
            super::transform::TransformOutput::new(xsize_blocks, ysize_blocks, budget)?;

        // W44-117: seed the buttloop's `apply_epf` sharpness map from
        // `compute_epf_sharpness` on the initial reconstruction.
        //
        // Pre-W44-117 the loop passed `vec![4u8; num_blocks]` (uniform
        // sharpness=4) to every `apply_epf` call. The shipped bitstream's
        // EPF metadata, however, comes from `compute_epf_sharpness` run
        // AFTER the buttloop on the FINAL recon — so the buttloop's
        // INTERNAL reconstruction (the linear-RGB image it measures
        // butteraugli against) diverged from jxl-rs's decoded output
        // by 0.05-0.17 max-abs on photos (W44-111 → W44-116 forensic
        // chain). The W44-105/107/108/109 qac-scale chain was a
        // partial palliative that lifted the seed quant_field so the
        // buttloop's optimistic-by-construction measurement settled on
        // a coarser equilibrium; the underlying mismatch was untouched.
        //
        // W44-116 Option B closes the gap by computing the sharpness
        // map ONCE before the loop (using the same `transform_out`
        // scratch the loop reuses), then passing it to every iter's
        // `apply_epf`. The post-loop production `compute_epf_sharpness`
        // call still runs on the final recon (so the SHIPPED bitstream
        // gets a final-iter-fitted map), but the buttloop's iters
        // converge against a map that's much closer to the production
        // map than uniform-4 was.
        //
        // Cost: one extra transform-quantize-reconstruct + one
        // `compute_epf_sharpness` call. The sharpness compute reuses
        // `base_recon` across its `candidates.len() ∈ {2, 3}`
        // candidates, so amortised cost is roughly equivalent to one
        // extra buttloop iter. At e9 (4 iters) that's +25%; the
        // measured impact is gated by the cell budget gate (≤ +20%
        // accept).
        //
        // Gated on:
        //   - `initial_params.epf_iters > 0` — no point computing a
        //     sharpness map if EPF won't run anyway. When EPF is off
        //     the sharpness vector is never read by `apply_epf`.
        //   - `self.profile.epf_dynamic_sharpness` — at low effort
        //     levels the production path also skips the per-block
        //     search and writes a uniform default sharpness map; seed
        //     consistently with that path (uniform-4 → `apply_epf`
        //     uses sharpness=4 → matches production exactly).
        //   - `mask1x1.is_some()` — `compute_epf_sharpness` needs a
        //     per-pixel mask. Callers that didn't precompute mask1x1
        //     (animation frames, ssim2 path) fall back to uniform-4
        //     (byte-identical to pre-W44-117).
        //
        // Fallback path is the legacy uniform-4: identical behaviour
        // to every pre-W44-117 release.
        //
        // W44-129 Chunk C: read the resolved `buttloop_epf_sharpness_seed`
        // enum from `ResolvedImprovements` (populated by Chunk B
        // `LossyConfig::resolve_improvements`).
        //
        // Policy translation:
        //   * `AutoW44_117 { min_distance }` (default) → existing W44-117
        //     seed compute, gated on `target_distance >= min_distance`.
        //   * `LegacyUniform4` → force pre-W44-117 uniform-4 seed (Libjxl
        //     strategy). Equivalent to `JXL_W44_117_DISABLE=1`.
        //   * `PerIterRecompute` → `#[doc(hidden)]` reserved-for-future;
        //     currently behaves identically to `AutoW44_117`.
        //
        // W44-130 Chunk D: `resolved_improvements` is now always
        // populated (default `AutoW44_117 { min_distance: 1.0 }` for
        // direct `VarDctEncoder::new` test callers) — bit-identical to
        // pre-Chunk-D.
        //
        // W44-132 Chunk F: env-vars `JXL_W44_117_DISABLE=1` and
        // `JXL_W44_120_EPF_SEED_MIN_DISTANCE=<f32>` are now consumed
        // inside `EncoderStrategy::resolve`'s env-var fallback layer
        // (api.rs::apply_env_var_fallbacks) — when the policy is at
        // its `Default::default()` (== `AutoW44_117 { min_distance:
        // 1.0 }`), the env vars promote it to `LegacyUniform4` or
        // `AutoW44_117 { min_distance: env }` BEFORE this call site
        // sees it. Explicit caller settings via `EncoderStrategy::
        // Custom` or `StrategyOverrides` win over the env-var; the
        // legacy "env-var always overrides" semantic ended with
        // Chunk F. See `apply_env_var_fallbacks` for the rationale.
        let epf_seed_policy = self.resolved_improvements.buttloop_epf_sharpness_seed;
        let w44_117_force_off = matches!(
            epf_seed_policy,
            crate::api::EpfSharpnessSeed::LegacyUniform4
        );

        // W44-120 distance gate: the W44-117 seed compute only fires at
        // `target_distance >= min_distance`. Default `1.0` (W44-120
        // bisect pick) closes the terminal e8/e9 d=0.8 SSIM2 -1.87
        // regression vs pre-W44-117. The threshold is read from the
        // resolved `EpfSharpnessSeed::AutoW44_117 { min_distance }`
        // variant — production default is 1.0 (Chunk A enum Default
        // impl) matching the const `W44_120_EPF_SEED_MIN_DISTANCE`.
        // For `LegacyUniform4` / `PerIterRecompute` variants the
        // distance gate logic is moot (the outer `w44_117_force_off`
        // already forces the legacy path under `LegacyUniform4`).
        let w44_120_min_distance = match epf_seed_policy {
            crate::api::EpfSharpnessSeed::AutoW44_117 { min_distance } => min_distance,
            _ => W44_120_EPF_SEED_MIN_DISTANCE,
        };
        let w44_120_distance_gate_passes = target_distance >= w44_120_min_distance;

        // W44-142 (2026-05-20): codec_wiki-class suppression sub-gate.
        //
        // Reads `self.zenanalyze_proxies` and the env-var opt-out. If the
        // proxies show a codec_wiki-class image (m3 ≥ 60 AND ed < 0.05)
        // AND target_distance is below the suppression cap (< 2.0), the
        // W44-117 sharpness map compute is skipped (= legacy uniform-4
        // seed) AND the W44-140 fade block is skipped. Closes the
        // codec_wiki d=1.2/1.6/1.8 SSIM2 regression cluster that the
        // W44-141 ledger refresh surfaced as a follow-on to W44-140.
        //
        // The sub-gate composes with the W44-118 `is_screenshot`
        // mask>95 gate (already passes for codec_wiki) and the W44-120
        // distance gate (`target_distance >= 1.0`); both are
        // load-bearing for the EPF seed mechanism in general.
        //
        // Photos cannot fire this gate: all CID22 photos have edge_density
        // ≥ 0.16 (textured), so even the colourful 1189261 (m3=98.84) is
        // correctly rejected by the ed sub-gate. Terminal-class screens
        // (text-only, low colourfulness) are rejected by the m3 sub-gate.
        //
        // Env override: `JXL_W44_142_SUPPRESS_DISABLE=1` opts out without
        // a rebuild (sweep harness use). When unset / `0` the gate fires
        // per the constants. Production default is `0` (enabled).
        #[cfg(feature = "std")]
        let env_w44_142_disable = std::env::var("JXL_W44_142_SUPPRESS_DISABLE")
            .ok()
            .as_deref()
            == Some("1");
        #[cfg(not(feature = "std"))]
        let env_w44_142_disable = false;
        #[cfg(feature = "std")]
        let env_w44_142_max_distance = std::env::var("JXL_W44_142_SUPPRESS_MAX_DISTANCE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok());
        #[cfg(not(feature = "std"))]
        let env_w44_142_max_distance: Option<f32> = None;
        let w44_142_max_distance =
            env_w44_142_max_distance.unwrap_or(W44_142_EPF_SEED_SUPPRESS_MAX_DISTANCE);
        let w44_142_suppress_fires = !env_w44_142_disable
            && target_distance < w44_142_max_distance
            && self.zenanalyze_proxies.is_some_and(|p| {
                p.m3_colourfulness >= W44_142_EPF_SEED_SUPPRESS_M3_MIN
                    && p.edge_density < W44_142_EPF_SEED_SUPPRESS_EDGE_DENSITY_MAX
            });

        let sharpness: Vec<u8> = if !w44_117_force_off
            && !w44_142_suppress_fires
            && w44_120_distance_gate_passes
            && initial_params.epf_iters > 0
            && self.profile.epf_dynamic_sharpness
            && let Some(m1x1) = mask1x1
            && m1x1.len() == padded_pixels
        {
            // Quantize the (post-W44-105-scale) quant_field_float to u8
            // using initial_params.inv_scale, matching the inner-seed
            // loop's iter-0 quantization step. The inner loop will
            // re-quantize per-iter via its own `current_params.inv_scale`;
            // this one-shot quantization is just to feed the seed
            // transform.
            let qf_vec = quantize_quant_field(quant_field_float, initial_params.inv_scale);

            // One-shot transform + quantize into `transform_out`. The
            // inner-seed loop will overwrite `transform_out` on its
            // first iter (via the same `transform_and_quantize_into`
            // call), so the scratch is correctly reused.
            //
            // We pass a mutable copy of `qf_vec` because
            // `transform_and_quantize_into` takes `&mut [u8]`; it
            // doesn't mutate the field for our shipped paths, but the
            // signature requires it.
            let mut qf_seed = qf_vec;
            self.transform_and_quantize_into(
                xyb_x,
                xyb_y,
                xyb_b,
                padded_width,
                xsize_blocks,
                ysize_blocks,
                initial_params,
                &mut qf_seed,
                cfl_map,
                ac_strategy,
                &mut transform_out,
            );

            // Call `compute_epf_sharpness` with the same `original_xyb`
            // (xyb_x/y/b is post-patches-subtraction xyb), quant_dc /
            // quant_ac from the one-shot transform, the seed
            // quant_field_u8, the precomputed mask, the initial
            // DistanceParams, the CFL map, the AC strategy map, and
            // `enable_gaborish`. Identical signature to the post-loop
            // call site in `encoder.rs::encode_inner`.
            super::epf::compute_epf_sharpness(
                [xyb_x, xyb_y, xyb_b],
                &transform_out.quant_dc,
                &transform_out.quant_ac,
                &qf_seed,
                m1x1,
                initial_params,
                cfl_map,
                ac_strategy,
                self.enable_gaborish,
                xsize_blocks,
                ysize_blocks,
                budget,
            )?
        } else {
            // Legacy / fallback path: uniform sharpness=4 seed.
            // Byte-identical to every pre-W44-117 release.
            vec![4u8; num_blocks]
        };

        // W44-140: distance-aware linear blend between the W44-117
        // sharpness map and uniform-4 within the band
        // `[w44_120_min_distance, w44_140_fade_max]`. Closes the residual
        // terminal e8/e9 d=1.0-1.6 SSIM2 oscillations W44-120 documented
        // as out-of-scope for pure threshold tightening — net +1.124
        // SSIM2 across the cluster vs the pre-W44-140 always-100%-W44-117
        // path, byte-identical to pre-W44-140 main at all distances
        // outside the blend band.
        //
        // Active when (in addition to the W44-117 gate predicate above):
        //   - `target_distance < w44_140_fade_max` (above fade_max the
        //     weight clamps to 1 → no-op).
        //   - `w44_140_fade_max > w44_120_min_distance` (degenerate
        //     fade_max <= min_distance disables the blend — caller can
        //     set `JXL_W44_140_EPF_SEED_FADE_MAX=1.0` to opt out without
        //     a rebuild).
        //
        // Production default: `W44_140_EPF_SEED_FADE_MAX` = 1.5. Sweep
        // override: `JXL_W44_140_EPF_SEED_FADE_MAX=<f32>`.
        //
        // Blend formula (per-block):
        //   weight = clamp((target_distance - min_distance) /
        //                  (fade_max - min_distance), 0.0, 1.0)
        //   blended[i] = round(weight * sharpness[i] + (1 - weight) * 4)
        //
        // At `target_distance >= fade_max` weight = 1 → byte-identical
        // to pre-W44-140 main. At `target_distance == min_distance`
        // weight = 0 → uniform-4 (= LegacyUniform4 effective in this
        // band). Linear interp between.
        let mut sharpness = sharpness;
        if !w44_117_force_off
            && !w44_142_suppress_fires
            && w44_120_distance_gate_passes
            && initial_params.epf_iters > 0
            && self.profile.epf_dynamic_sharpness
        {
            #[cfg(feature = "std")]
            let env_fade_max = std::env::var("JXL_W44_140_EPF_SEED_FADE_MAX")
                .ok()
                .and_then(|s| s.parse::<f32>().ok());
            #[cfg(not(feature = "std"))]
            let env_fade_max: Option<f32> = None;
            let fade_max = env_fade_max.unwrap_or(W44_140_EPF_SEED_FADE_MAX);
            if fade_max > w44_120_min_distance && target_distance < fade_max {
                let span = fade_max - w44_120_min_distance;
                let weight = ((target_distance - w44_120_min_distance) / span).clamp(0.0, 1.0);
                if weight < 1.0 {
                    let inv_w = 1.0 - weight;
                    for v in sharpness.iter_mut() {
                        let blended = weight * (*v as f32) + inv_w * 4.0;
                        *v = blended.round().clamp(0.0, 7.0) as u8;
                    }
                }
            }
        }

        // Saturate at consumption to bound worst-case CPU even when the
        // caller skipped LossyConfig::validate (which would have rejected
        // values > MAX_QUANT_LOOP_ITERS with IterCountOutOfRange). Each
        // iteration runs a full butteraugli pipeline; capping prevents
        // a malicious or buggy caller from DoS-ing the encoder.
        let iters = (self.butteraugli_iters.min(crate::api::MAX_QUANT_LOOP_ITERS)) as usize;
        // RFC#45 chunk 1 + chunk 2: e10/e11/e12 push butteraugli_iters to
        // 8/16/32 via the effort table (see effort.rs). The saturating
        // `.min()` above already bounds the loop; this debug-assert documents
        // the structural invariant so future effort levels can't sneak past
        // `MAX_QUANT_LOOP_ITERS` (= 32 after chunk 2) and underflow the
        // compare-only exit (`if iter == iters { break }` below).
        debug_assert!(
            iters <= crate::api::MAX_QUANT_LOOP_ITERS as usize,
            "butteraugli loop iters={} exceeds MAX_QUANT_LOOP_ITERS={} \
             (effort table must saturate at the cap)",
            iters,
            crate::api::MAX_QUANT_LOOP_ITERS,
        );

        // RFC#45 pick #1 chunk 3 — multi-seed butteraugli sweep.
        //
        // At e ≤ 9 the profile sets `lossy_search_seeds = 1` and the seed
        // table is `[LIBJXL_INIT_MUL]` (= 0.6) — bit-identical to the
        // single-seed libjxl `FindBestQuantization`. At e10/e11 we fan
        // out 2/4 different `kInitMul` values, run the full loop on a
        // clone of (`quant_field`, `quant_field_float`) per seed, then
        // pick the seed with the largest mean(`quant_field_float`)
        // (proxy for smallest encoded bytes — coarser quant produces
        // fewer non-zero AC coefficients and thus shorter Huffman/ANS
        // streams) whose final butteraugli score does not exceed
        // `K_BUTTERAUGLI_ACCEPT_FACTOR * target_distance`. If no seed
        // meets that bound (rare; usually means target_distance is so
        // small the loop didn't converge on any seed), the seed with
        // the smallest final score wins instead — the worst-case for
        // multi-seed is the same `final_score` as single-seed because
        // `init_mul_seeds[0]` is always `LIBJXL_INIT_MUL`.
        let seeds = init_mul_seeds(self.profile.lossy_search_seeds);
        const K_BUTTERAUGLI_ACCEPT_FACTOR: f64 = 1.05;

        // Snapshot the caller's starting buffers so each seed starts
        // from the SAME initial state (the caller hands us the post-AQ
        // float field; without snapshotting, seed N+1 would start from
        // seed N's post-loop field and the sweep would degenerate).
        let initial_qf_u8_snapshot = quant_field.to_vec();
        let initial_qf_float_snapshot = quant_field_float.to_vec();

        let mut outcomes: alloc::vec::Vec<SeedOutcome> =
            alloc::vec::Vec::with_capacity(seeds.len());

        // W44-118 Mode D probe: per-iter sharpness recompute (Option A
        // from W44-116 fix-shape menu). Env-gated; production default
        // (unset) preserves W44-117 Option B one-shot seed behaviour.
        // Cost: extra compute_epf_sharpness call per buttloop iter.
        #[cfg(feature = "std")]
        let w44_118_per_iter_sharpness =
            std::env::var("JXL_W44_118_PER_ITER_SHARPNESS").is_ok_and(|v| v == "1");
        #[cfg(not(feature = "std"))]
        let w44_118_per_iter_sharpness = false;

        // Mutable sharpness so the per-iter recompute (Mode D) can
        // overwrite. Default path never mutates it (Mode B reuses the
        // W44-117 one-shot seed verbatim).
        let mut sharpness_mut = sharpness;

        for &k_init_mul in seeds {
            // Restore starting state for this seed (skipped on seed 0 because
            // quant_field/quant_field_float already hold it, but cheap enough
            // to always do for clarity).
            quant_field.copy_from_slice(&initial_qf_u8_snapshot);
            quant_field_float.copy_from_slice(&initial_qf_float_snapshot);

            let outcome = self.butteraugli_refine_quant_field_inner_seed(
                xyb_x,
                xyb_y,
                xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                initial_params,
                quant_field,
                quant_field_float,
                initial_quant_field_float,
                cfl_map,
                ac_strategy,
                patches_data,
                splines_data,
                reference.as_ref(),
                &ref_r,
                &ref_g,
                &ref_b,
                width,
                height,
                use_vdp2,
                vdp2_intensity_target,
                qf_lower,
                qf_higher,
                &mut sharpness_mut,
                &mut tile_dist,
                &mut recon_r,
                &mut recon_g,
                &mut recon_b,
                &mut transform_out,
                iters,
                k_init_mul,
                is_screenshot,
                w44_118_per_iter_sharpness,
                mask1x1,
            )?;
            outcomes.push(outcome);
        }

        // Pick the winner. Selection rule:
        //   1. Prefer seeds with final_score <= K_BUTTERAUGLI_ACCEPT_FACTOR * target.
        //   2. Among those, pick the largest mean_qf (proxy for smallest bytes).
        //   3. If none meet bound, pick the smallest final_score (degenerates
        //      to single-seed worst-case because seed 0 = LIBJXL_INIT_MUL).
        let accept_bound = K_BUTTERAUGLI_ACCEPT_FACTOR * target_distance as f64;
        let winner_idx = {
            let qualifying: alloc::vec::Vec<usize> = (0..outcomes.len())
                .filter(|&i| outcomes[i].final_score <= accept_bound)
                .collect();
            if !qualifying.is_empty() {
                qualifying
                    .into_iter()
                    .max_by(|&a, &b| outcomes[a].mean_qf.total_cmp(&outcomes[b].mean_qf))
                    .expect("non-empty by filter")
            } else {
                (0..outcomes.len())
                    .min_by(|&a, &b| outcomes[a].final_score.total_cmp(&outcomes[b].final_score))
                    .unwrap_or(0)
            }
        };

        // Emit a one-line debug summary of all seeds so post-hoc analysis
        // can spot when the picker is consistently choosing non-default
        // seeds (signal that the libjxl default `kInitMul=0.6` is
        // sub-optimal on this image / distance combination). The
        // `summary` is only formatted when the `debug-rect` feature
        // is enabled — without it the macro `if false {}` gate drops
        // the whole arm.
        #[cfg(feature = "debug-rect")]
        let summary: alloc::string::String = {
            use alloc::string::String;
            let mut s = String::new();
            for (i, o) in outcomes.iter().enumerate() {
                if i > 0 {
                    s.push(' ');
                }
                let marker = if i == winner_idx { "*" } else { "" };
                s.push_str(&alloc::format!(
                    "[{marker}{i} k={:.2} bfly={:.3} qf_mean={:.4}]",
                    o.k_init_mul,
                    o.final_score,
                    o.mean_qf
                ));
            }
            s
        };
        #[cfg(not(feature = "debug-rect"))]
        let summary = "";
        debug_rect!(
            "bfly/seeds",
            0,
            0,
            width,
            height,
            "n={} winner={} accept<={:.3} {}",
            outcomes.len(),
            winner_idx,
            accept_bound,
            summary,
        );

        // Promote winner into the caller's mutable buffers.
        let winner = outcomes.swap_remove(winner_idx);
        quant_field.copy_from_slice(&winner.quant_field);
        quant_field_float.copy_from_slice(&winner.quant_field_float);
        Ok(winner.params)
    }

    /// Inner per-seed body of the butteraugli quantization loop.
    /// Runs the full `iters + 1` iteration sequence on the supplied
    /// `quant_field_float` (and the matching u8 `quant_field`) and
    /// returns the resulting [`SeedOutcome`].
    ///
    /// `k_init_mul` selects the basin: it scales the iter-1 pullback
    /// toward `initial_quant_field_float` (libjxl uses 0.6;
    /// [`init_mul_seeds`] returns other values for the multi-seed
    /// sweep at e10/e11). Buffers (`tile_dist`, `recon_*`,
    /// `transform_out`) are re-used between seeds — the caller is
    /// responsible for resetting `quant_field`/`quant_field_float`
    /// between calls (each seed starts from the same initial state).
    #[cfg(feature = "butteraugli-loop")]
    #[allow(clippy::too_many_arguments)]
    fn butteraugli_refine_quant_field_inner_seed(
        &self,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        padded_width: usize,
        padded_height: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        initial_params: &DistanceParams,
        quant_field: &mut [u8],
        quant_field_float: &mut [f32],
        initial_quant_field_float: &[f32],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        patches_data: Option<&super::patches::PatchesData>,
        splines_data: Option<&super::splines::SplinesData>,
        // `Some(reference)` for the butteraugli path (chunk-1 default).
        // `None` for the VDP2-lite path (chunk-2 opt-in via
        // `HdrLoss::Vdp2`) — the metric reads `ref_r/g/b` directly.
        reference: Option<&butteraugli::ButteraugliReference>,
        // Planar linear-RGB reference planes (always populated by the
        // top-level call). Sized `width × height` with stride = width.
        // Owned by the caller for the duration of the seed loop so we
        // can re-use them across iterations without re-deinterleaving.
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        // Logical image extent. Distinct from `padded_width`/`padded_height`
        // (which describe the reconstruction buffer's row stride).
        width: usize,
        height: usize,
        // EX-J11 chunk 2: select the perceptual metric.
        use_vdp2: bool,
        // VDP2-lite display-luminance target in cd/m². Unused on the
        // butteraugli path (which hardcodes `intensity_target = 80`).
        vdp2_intensity_target: f32,
        qf_lower: f32,
        qf_higher: f32,
        sharpness: &mut [u8],
        tile_dist: &mut [f32],
        recon_r: &mut [f32],
        recon_g: &mut [f32],
        recon_b: &mut [f32],
        transform_out: &mut super::transform::TransformOutput,
        iters: usize,
        k_init_mul: f64,
        // W39-2 (WF3 fix): plumbed from
        // [`Self::butteraugli_refine_quant_field`] so the inner loop's
        // per-iter `resolved_max_increase_with_class` can pick the
        // screenshot HIGH cap when content was classified as
        // screenshot. `false` (default for photo / unknown) is
        // byte-identical to pre-W39-2 behaviour.
        is_screenshot: bool,
        // W44-118 Mode D bisection: when true AND mask1x1 is Some,
        // recompute sharpness per-iter using the current transform_out
        // (post-iter-quantize) instead of reusing the W44-117 one-shot
        // seed. Env-gated; production default false.
        w44_118_per_iter_sharpness: bool,
        mask1x1: Option<&[f32]>,
    ) -> Result<SeedOutcome> {
        use super::epf;
        use super::reconstruct::{gab_smooth, reconstruct_xyb, xyb_to_linear_rgb_planar};

        let target_distance = self.distance;
        let num_blocks = xsize_blocks * ysize_blocks;
        let padded_pixels = padded_width * padded_height;
        debug_assert_eq!(ref_r.len(), width * height);
        debug_assert_eq!(ref_g.len(), width * height);
        debug_assert_eq!(ref_b.len(), width * height);
        debug_assert!(use_vdp2 == reference.is_none());
        debug_assert_eq!(padded_pixels, recon_r.len());
        let mut current_params;
        // Score from the final compare-only iteration (i == iters).
        // `INFINITY` until first compare succeeds — propagates to picker
        // selection: any seed that failed every compare is unselectable
        // unless every seed failed.
        let mut last_score: f64 = f64::INFINITY;

        // Loop runs iters+1 times (matching libjxl: last iteration is compare-only).
        // i=0..iters-1: SetQuantField + roundtrip + compare + adjust
        // i=iters: SetQuantField + roundtrip + compare + break
        for iter in 0..iters + 1 {
            // Step 1: SetQuantField — recompute global_scale from float field,
            // then convert float → u8.
            // (libjxl: quantizer.SetQuantField(initial_quant_dc, quant_field, &raw_quant_field))
            current_params =
                DistanceParams::compute_from_quant_field(target_distance, quant_field_float);
            // Preserve chromacity adjustments and EPF from initial params
            current_params.x_qm_scale = initial_params.x_qm_scale;
            current_params.b_qm_scale = initial_params.b_qm_scale;
            current_params.epf_iters = initial_params.epf_iters;

            // Convert float → u8 with current params' inv_scale
            // (libjxl: SetQuantFieldRect: ClampVal(row_qf[x] * inv_global_scale_ + 0.5f, 1, 255))
            let qf_vec = quantize_quant_field(quant_field_float, current_params.inv_scale);
            quant_field.copy_from_slice(&qf_vec);

            // Step 2: Transform and quantize with current params.
            // `transform_out` is `&mut TransformOutput` from our caller;
            // the helper wants `&mut TransformOutput` too, so reborrow.
            self.transform_and_quantize_into(
                xyb_x,
                xyb_y,
                xyb_b,
                padded_width,
                xsize_blocks,
                ysize_blocks,
                &current_params,
                quant_field,
                cfl_map,
                ac_strategy,
                &mut *transform_out,
            );

            // W44-118 Mode D: per-iter sharpness recompute. Replaces
            // the W44-117 one-shot seed (which froze sharpness from
            // iter-0 quantization, drifting from the post-buttloop
            // production sharpness map at high d). When enabled, the
            // sharpness used for THIS iter's apply_epf matches what
            // compute_epf_sharpness would produce on the SAME
            // quant_dc/quant_ac the iter just produced — much closer
            // to what the post-loop production compute_epf_sharpness
            // would produce on the converged state.
            //
            // Cost: extra compute_epf_sharpness call per iter (~10-30%
            // buttloop slowdown). Production default off; this is a
            // bisection switch only.
            if w44_118_per_iter_sharpness
                && current_params.epf_iters > 0
                && self.profile.epf_dynamic_sharpness
                && let Some(m1x1) = mask1x1
                && m1x1.len() == padded_width * padded_height
                && let Ok(new_sharpness) = super::epf::compute_epf_sharpness(
                    [xyb_x, xyb_y, xyb_b],
                    &transform_out.quant_dc,
                    &transform_out.quant_ac,
                    quant_field,
                    m1x1,
                    &current_params,
                    cfl_map,
                    ac_strategy,
                    self.enable_gaborish,
                    xsize_blocks,
                    ysize_blocks,
                    self.budget.as_ref(),
                )
            {
                // Overwrite sharpness with the per-iter computed map.
                sharpness.copy_from_slice(&new_sharpness);
            }

            // Step 3: Reconstruct XYB from quantized coefficients
            let mut planes = reconstruct_xyb(
                &transform_out.quant_dc,
                &transform_out.quant_ac,
                &current_params,
                quant_field,
                cfl_map,
                ac_strategy,
                xsize_blocks,
                ysize_blocks,
            );

            // W44-116: per-step XYB capture (FINAL iter only). The hook is
            // populated incrementally as each step runs; only built into a
            // StepXyb struct & stored once the full pipeline finishes.
            #[cfg(feature = "__internal_recon_hook")]
            let capture_steps = iter == iters && recon_hook::steps_capture_enabled();
            #[cfg(not(feature = "__internal_recon_hook"))]
            let capture_steps = false;

            #[cfg(feature = "__internal_recon_hook")]
            let mut step_after_recon: Option<recon_hook::Xyb> = None;
            #[cfg(feature = "__internal_recon_hook")]
            let mut step_after_gab: Option<recon_hook::Xyb> = None;
            #[cfg(feature = "__internal_recon_hook")]
            let mut step_after_epf: Option<recon_hook::Xyb> = None;
            #[cfg(feature = "__internal_recon_hook")]
            let mut step_after_patches: Option<recon_hook::Xyb> = None;
            #[cfg(feature = "__internal_recon_hook")]
            let mut step_after_splines: Option<recon_hook::Xyb> = None;

            #[cfg(feature = "__internal_recon_hook")]
            if capture_steps {
                step_after_recon = Some(recon_hook::Xyb {
                    x: planes[0].clone(),
                    y: planes[1].clone(),
                    b: planes[2].clone(),
                });
            }

            if self.enable_gaborish {
                gab_smooth(&mut planes, padded_width, padded_height);
            }
            #[cfg(feature = "__internal_recon_hook")]
            if capture_steps && self.enable_gaborish {
                step_after_gab = Some(recon_hook::Xyb {
                    x: planes[0].clone(),
                    y: planes[1].clone(),
                    b: planes[2].clone(),
                });
            }

            if current_params.epf_iters > 0 {
                epf::apply_epf(
                    &mut planes,
                    quant_field,
                    sharpness,
                    current_params.scale,
                    current_params.epf_iters,
                    xsize_blocks,
                    ysize_blocks,
                    padded_width,
                    padded_height,
                    self.budget.as_ref(),
                )?;
            }
            #[cfg(feature = "__internal_recon_hook")]
            if capture_steps && current_params.epf_iters > 0 {
                step_after_epf = Some(recon_hook::Xyb {
                    x: planes[0].clone(),
                    y: planes[1].clone(),
                    b: planes[2].clone(),
                });
            }

            if let Some(pd) = patches_data {
                super::patches::add_patches(&mut planes, padded_width, pd);
            }
            #[cfg(feature = "__internal_recon_hook")]
            if capture_steps && patches_data.is_some() {
                step_after_patches = Some(recon_hook::Xyb {
                    x: planes[0].clone(),
                    y: planes[1].clone(),
                    b: planes[2].clone(),
                });
            }

            if let Some(sd) = splines_data {
                super::splines::add_splines(&mut planes, padded_width, width, height, sd);
            }
            #[cfg(feature = "__internal_recon_hook")]
            if capture_steps && splines_data.is_some() {
                step_after_splines = Some(recon_hook::Xyb {
                    x: planes[0].clone(),
                    y: planes[1].clone(),
                    b: planes[2].clone(),
                });
            }

            #[cfg(feature = "__internal_recon_hook")]
            if capture_steps {
                recon_hook::store_steps(recon_hook::StepXyb {
                    padded_width,
                    padded_height,
                    width,
                    height,
                    after_recon_xyb: step_after_recon.expect("after_recon_xyb always populated"),
                    after_gab: step_after_gab,
                    after_epf: step_after_epf,
                    after_patches: step_after_patches,
                    after_splines: step_after_splines,
                });
            }

            // Step 4: Convert reconstructed XYB to planar linear RGB
            xyb_to_linear_rgb_planar(
                &planes[0],
                &planes[1],
                &planes[2],
                recon_r,
                recon_g,
                recon_b,
                padded_pixels,
            );

            // Debug hook (Layer-1 invariant for the quality-drift investigation):
            // capture the buttloop's INTERNAL reconstruction at the FINAL iter,
            // cropped to (width, height) — this is the linear-RGB image the loop
            // measures butteraugli against. The drift hypothesis is that this
            // diverges from what the user-facing decoder produces from the SHIPPED
            // bitstream (jxl-rs / jxl-oxide). Comparing the two pinpoints the bug.
            // See memory/quality_drift_investigation_2026-05-15.md.
            #[cfg(feature = "__internal_recon_hook")]
            if iter == iters && recon_hook::capture_enabled() {
                let mut cropped_r = alloc::vec![0.0f32; width * height];
                let mut cropped_g = alloc::vec![0.0f32; width * height];
                let mut cropped_b = alloc::vec![0.0f32; width * height];
                for y in 0..height {
                    let dst = y * width;
                    let src = y * padded_width;
                    cropped_r[dst..dst + width].copy_from_slice(&recon_r[src..src + width]);
                    cropped_g[dst..dst + width].copy_from_slice(&recon_g[src..src + width]);
                    cropped_b[dst..dst + width].copy_from_slice(&recon_b[src..src + width]);
                }
                // Snapshot per-block strategy + per-tile CfL for chunk-2's
                // diff-map correlation. These are cheap (a few bytes per block,
                // 2 i8 per tile) and only allocated when capture is enabled.
                let nblocks = xsize_blocks * ysize_blocks;
                let mut raw_strategy_v = alloc::vec![0u8; nblocks];
                let mut is_first_block = alloc::vec![false; nblocks];
                for by in 0..ysize_blocks {
                    for bx in 0..xsize_blocks {
                        let idx = by * xsize_blocks + bx;
                        raw_strategy_v[idx] = ac_strategy.raw_strategy(bx, by);
                        is_first_block[idx] = ac_strategy.is_first(bx, by);
                    }
                }
                recon_hook::store(recon_hook::InternalRecon {
                    width,
                    height,
                    r: cropped_r,
                    g: cropped_g,
                    b: cropped_b,
                    iter,
                    iters,
                    xsize_blocks,
                    ysize_blocks,
                    raw_strategy: raw_strategy_v,
                    is_first_block,
                    quant_field_u8: quant_field.to_vec(),
                    xsize_tiles: cfl_map.xsize_tiles,
                    ysize_tiles: cfl_map.ysize_tiles,
                    cfl_ytox: cfl_map.ytox.clone(),
                    cfl_ytob: cfl_map.ytob.clone(),
                    // W44-112: capture final-iter `current_params` so the
                    // discriminator can compare them against production's
                    // `params` (`ProductionQf`). Per code inspection these
                    // should be byte-identical when `final_params` is recomputed
                    // from the same `quant_field_float` post-loop — surface them
                    // anyway so a mismatch is a loud regression.
                    final_global_scale: current_params.global_scale,
                    final_scale: current_params.scale,
                    final_inv_scale: current_params.inv_scale,
                });
            }

            // Step 5: Perceptual comparison.
            //
            // Dispatches on `use_vdp2`:
            //  - false (default): butteraugli `compare_linear_planar` against
            //    the precomputed reference (chunk-1 byte-identical path).
            //  - true (`HdrLoss::Vdp2` opt-in): VDP2-lite, walks ref + rec
            //    planar planes through the multi-scale CSF pyramid in
            //    [`super::hdr_vdp2_lite::compare_vdp2_planar`].
            //
            // Both metrics return `(score: f64, diffmap: Vec<f32>)` sized to
            // the logical `width × height` extent. On compare failure (rare —
            // typically NaN inputs the reconstruction shouldn't produce) we
            // bail out with the previous iter's score so the picker prefers
            // any seed that converged.
            let (iter_score, diffmap_vec): (f64, alloc::vec::Vec<f32>) = if use_vdp2 {
                match super::hdr_vdp2_lite::compare_vdp2_planar(
                    ref_r,
                    ref_g,
                    ref_b,
                    recon_r,
                    recon_g,
                    recon_b,
                    width,
                    height,
                    padded_width,
                    vdp2_intensity_target,
                ) {
                    Ok(r) => (r.score, r.diffmap),
                    Err(_) => {
                        let mean_qf = mean_qf_float(quant_field_float);
                        return Ok(SeedOutcome {
                            params: current_params,
                            quant_field: quant_field.to_vec(),
                            quant_field_float: quant_field_float.to_vec(),
                            final_score: last_score,
                            mean_qf,
                            k_init_mul,
                        });
                    }
                }
            } else {
                let bref = reference.expect(
                    "non-VDP2 path must carry a butteraugli reference (top-level invariant)",
                );
                let result =
                    match bref.compare_linear_planar(recon_r, recon_g, recon_b, padded_width) {
                        Ok(r) => r,
                        Err(_) => {
                            let mean_qf = mean_qf_float(quant_field_float);
                            return Ok(SeedOutcome {
                                params: current_params,
                                quant_field: quant_field.to_vec(),
                                quant_field_float: quant_field_float.to_vec(),
                                final_score: last_score,
                                mean_qf,
                                k_init_mul,
                            });
                        }
                    };
                let dm = match result.diffmap {
                    Some(dm) => dm,
                    None => {
                        let mean_qf = mean_qf_float(quant_field_float);
                        return Ok(SeedOutcome {
                            params: current_params,
                            quant_field: quant_field.to_vec(),
                            quant_field_float: quant_field_float.to_vec(),
                            final_score: last_score,
                            mean_qf,
                            k_init_mul,
                        });
                    }
                };
                // ImgVec is contiguous when produced by the butteraugli
                // crate (stride == width — confirmed in lib.rs:510). Take
                // ownership of the backing Vec to match the VDP2 branch's
                // owned-Vec return type without re-allocating.
                (result.score, dm.into_buf())
            };

            // Record metric score for the picker (rewritten every iter;
            // the value at loop exit is what the picker compares against the
            // target). Stored before the iter==iters early-break below so the
            // compare-only last iteration is included.
            last_score = iter_score;

            // Step 6: Compute per-block tile distance (16th-power norm, matching libjxl TileDistMap)
            const K_TILE_NORM: f32 = 1.2;
            let diffmap_buf: &[f32] = &diffmap_vec;
            tile_dist.fill(0.0);
            for by in 0..ysize_blocks {
                for bx in 0..xsize_blocks {
                    if !ac_strategy.is_first(bx, by) {
                        continue;
                    }
                    let covered_x = ac_strategy.covered_blocks_x(bx, by);
                    let covered_y = ac_strategy.covered_blocks_y(bx, by);
                    let px_start_x = bx * BLOCK_DIM;
                    let px_start_y = by * BLOCK_DIM;
                    let px_end_x = ((bx + covered_x) * BLOCK_DIM).min(width);
                    let px_end_y = ((by + covered_y) * BLOCK_DIM).min(height);
                    if px_start_x >= width || px_start_y >= height {
                        continue;
                    }
                    let mut dist_norm = 0.0f64;
                    let mut pixels = 0.0f64;
                    for py in px_start_y..px_end_y {
                        for px in px_start_x..px_end_x {
                            let v = diffmap_buf[py * width + px] as f64;
                            let v2 = v * v;
                            let v4 = v2 * v2;
                            let v8 = v4 * v4;
                            let v16 = v8 * v8;
                            dist_norm += v16;
                            pixels += 1.0;
                        }
                    }
                    if pixels == 0.0 {
                        pixels = 1.0;
                    }
                    let td = K_TILE_NORM * (dist_norm / pixels).sqrt().sqrt().sqrt().sqrt() as f32;
                    for sy in 0..covered_y {
                        for sx in 0..covered_x {
                            tile_dist[(by + sy) * xsize_blocks + (bx + sx)] = td;
                        }
                    }
                }
            }

            // Log per-iteration summary
            {
                let qf_min = quant_field_float
                    .iter()
                    .copied()
                    .reduce(f32::min)
                    .unwrap_or(0.0);
                let qf_max = quant_field_float
                    .iter()
                    .copied()
                    .reduce(f32::max)
                    .unwrap_or(0.0);
                let qf_sum: f64 = quant_field_float.iter().map(|&v| v as f64).sum();
                let qf_avg = qf_sum / quant_field_float.len() as f64;
                let td_max = tile_dist.iter().copied().reduce(f32::max).unwrap_or(0.0);
                let bad_blocks = tile_dist.iter().filter(|&&d| d > target_distance).count();
                debug_rect!(
                    "bfly/iter",
                    0,
                    0,
                    width,
                    height,
                    "iter={}/{} score={:.3} target={:.3} gs={} qf_avg={:.4} qf=[{:.4};{:.4}] td_max={:.2} bad={}",
                    iter,
                    iters,
                    iter_score,
                    target_distance,
                    current_params.global_scale,
                    qf_avg,
                    qf_min,
                    qf_max,
                    td_max,
                    bad_blocks
                );
            }

            // Last iteration is compare-only (libjxl: if (i == iters) break;)
            if iter == iters {
                break;
            }

            // Step 7: kOriginalComparisonRound = 1: constrain toward initial BEFORE adjustment.
            // Prevents oscillation by keeping qf from diverging too far from initial.
            // (libjxl enc_adaptive_quantization.cc:1039-1057)
            //
            // `k_init_mul` is the seed parameter — libjxl hardcodes 0.6 here;
            // RFC#45 chunk 3 sweeps multiple values at e10/e11 and picks the
            // best per-image outcome. See `init_mul_seeds()` and the
            // `lossy_search_seeds` field on [`EffortProfile`].
            const K_ORIGINAL_COMPARISON_ROUND: usize = 1;
            if iter == K_ORIGINAL_COMPARISON_ROUND {
                let k_one_minus_init_mul = 1.0 - k_init_mul;
                for bi in 0..num_blocks {
                    let init_qf = initial_quant_field_float[bi] as f64;
                    let cur_qf = quant_field_float[bi] as f64;
                    let clamp_val = k_one_minus_init_mul * cur_qf + k_init_mul * init_qf;
                    if cur_qf < clamp_val {
                        let mut v = clamp_val as f32;
                        if v > qf_higher {
                            v = qf_higher;
                        }
                        if v < qf_lower {
                            v = qf_lower;
                        }
                        quant_field_float[bi] = v;
                    }
                }
            }

            // Step 8: Adjust float quant_field based on tile distances.
            // (libjxl enc_adaptive_quantization.cc:1059-1110)
            //
            // Distance-aware tuning scaffolding (W38-2 #3.1; ported
            // from GPU `d75bf7c` as infrastructure-only).
            //
            // **Production defaults match libjxl at both regimes**
            // (`cur_pow=0.2`, `max_increase=100.0` ≈ "no cap"). The
            // literal GPU LOW-regime tuning (cur_pow=0.5,
            // max_increase=1.3) regressed bfly +4-13 % on CPU and was
            // not adopted as default — see
            // `benchmarks/buttloop_distance_split_port_2026-05-18.tsv`.
            //
            // The atomic overrides
            // (`CUR_POW_X1000_{LOW,HIGH}` / `MAX_INCREASE_X1000_{LOW,HIGH}`
            // / `DISTANCE_SPLIT_X1000`) let sweep harnesses search for
            // a CPU-specific LOW value that survives RD-pareto without
            // rebuilds; production code never sets them.
            //
            // `cur_pow` is 0.0 for `iter >= 2` regardless of regime
            // (only iter < 2 reduces quality of good blocks; later
            // iters only bump bad blocks — same as libjxl
            // `enc_adaptive_quantization.cc:1106`).
            let cur_pow: f64 = resolved_cur_pow(iter, target_distance as f64);
            // W39-2 (WF3 fix): HIGH-regime + screenshot-class content
            // can cap `max_increase` at a lower value than the libjxl
            // default. `is_screenshot` was classified once at the
            // call site (`encoder::encode` → `median(mask1x1) >
            // SCREENSHOT_MEDIAN_THRESHOLD`); the resolver gates the
            // cap on the HIGH branch only — LOW + photo HIGH continue
            // to read the legacy slots and stay byte-identical.
            let max_increase: f64 =
                resolved_max_increase_with_class(target_distance as f64, is_screenshot);

            // InvGlobalScale and Scale from current iteration's params
            // (these change per iteration as global_scale is recomputed)
            let inv_global_scale = current_params.inv_scale; // = 65536 / global_scale
            let quantizer_scale = current_params.scale; // = global_scale / 65536

            if cur_pow == 0.0 {
                // Only adjust bad blocks (diff > 1.0)
                // (libjxl enc_adaptive_quantization.cc:1066-1086)
                for bi in 0..num_blocks {
                    // butteraugli's ButteraugliReference is finite by
                    // construction on any finite XYB input — non-finite
                    // here is always an upstream bug. The 270-encode
                    // trigger-fixture sweep + the math (XYB transform is
                    // total on ℝ via cbrt + bias) prove these never fire
                    // on legitimate input.
                    assert!(
                        tile_dist[bi].is_finite(),
                        "butteraugli loop: non-finite tile_dist[{bi}] = {} \
                         (upstream butteraugli should never produce non-finite)",
                        tile_dist[bi]
                    );
                    assert!(
                        quant_field_float[bi].is_finite(),
                        "butteraugli loop: non-finite quant_field_float[{bi}] = {} \
                         (clamps should keep this finite every iter)",
                        quant_field_float[bi]
                    );
                    let diff_raw = tile_dist[bi] / target_distance;
                    // W38-2 #3.1: cap the per-iter bump (no-op in HIGH
                    // regime where max_increase = 100.0).
                    let diff = diff_raw.min(max_increase as f32);
                    if diff > 1.0 {
                        let old = quant_field_float[bi];
                        quant_field_float[bi] = old * diff;
                        // Minimum step check: if rounding to integer quant produces
                        // the same value, bump by one quantizer step
                        // (libjxl: if (fi == pi) row_q[x] = old + quantizer.Scale())
                        let qf_old = (old * inv_global_scale + 0.5).floor() as i32;
                        let qf_new =
                            (quant_field_float[bi] * inv_global_scale + 0.5).floor() as i32;
                        if qf_old == qf_new {
                            quant_field_float[bi] = old + quantizer_scale;
                        }
                    }
                    quant_field_float[bi] = quant_field_float[bi].clamp(qf_lower, qf_higher);
                }
            } else {
                // Adjust both directions (libjxl enc_adaptive_quantization.cc:1087-1110)
                for bi in 0..num_blocks {
                    assert!(
                        tile_dist[bi].is_finite(),
                        "butteraugli loop: non-finite tile_dist[{bi}] = {}",
                        tile_dist[bi]
                    );
                    assert!(
                        quant_field_float[bi].is_finite(),
                        "butteraugli loop: non-finite quant_field_float[{bi}] = {}",
                        quant_field_float[bi]
                    );
                    let diff_raw = tile_dist[bi] / target_distance;
                    // W38-2 #3.1: cap the per-iter bump for bad blocks
                    // (no-op in HIGH regime where max_increase = 100.0,
                    // no-op for good blocks where diff <= 1.0 anyway).
                    let diff = diff_raw.min(max_increase as f32);
                    if diff <= 1.0 {
                        // Good quality: reduce precision to save bits.
                        // `diff` must be finite — NaN here indicates a real bug
                        // (target_distance == 0, or polluted reconstruction from
                        // a previous butteraugli iteration). Surface loudly rather
                        // than silently coercing to 0 via .max() (IEEE-754 ordered
                        // max returns the non-NaN operand, and 0.0.powf(x) = 0.0
                        // is finite, so the downstream assert can't catch it).
                        assert!(
                            diff.is_finite(),
                            "butteraugli loop: non-finite diff = {diff} \
                             (tile_dist={}, target_distance={target_distance})",
                            tile_dist[bi]
                        );
                        // Negative diff would produce NaN through powf for
                        // non-integer cur_pow — guard via max(0).
                        let safe_diff = diff.max(0.0) as f64;
                        let factor = safe_diff.powf(cur_pow) as f32;
                        assert!(
                            factor.is_finite(),
                            "butteraugli loop: non-finite powf factor diff={diff} pow={cur_pow}"
                        );
                        quant_field_float[bi] *= factor;
                    } else {
                        // Bad quality: increase precision
                        let old = quant_field_float[bi];
                        quant_field_float[bi] = old * diff;
                        // Minimum step check
                        let qf_old = (old * inv_global_scale + 0.5).floor() as i32;
                        let qf_new =
                            (quant_field_float[bi] * inv_global_scale + 0.5).floor() as i32;
                        if qf_old == qf_new {
                            quant_field_float[bi] = old + quantizer_scale;
                        }
                    }
                    quant_field_float[bi] = quant_field_float[bi].clamp(qf_lower, qf_higher);
                }
            }
        }

        // Final SetQuantField: compute definitive params from final float field
        // (libjxl enc_adaptive_quantization.cc:1112-1113)
        let mut final_params =
            DistanceParams::compute_from_quant_field(target_distance, quant_field_float);
        final_params.x_qm_scale = initial_params.x_qm_scale;
        final_params.b_qm_scale = initial_params.b_qm_scale;
        final_params.epf_iters = initial_params.epf_iters;

        // Convert final float → u8 with definitive params
        let qf_vec = quantize_quant_field(quant_field_float, final_params.inv_scale);
        quant_field.copy_from_slice(&qf_vec);

        let mean_qf = mean_qf_float(quant_field_float);
        Ok(SeedOutcome {
            params: final_params,
            quant_field: quant_field.to_vec(),
            quant_field_float: quant_field_float.to_vec(),
            final_score: last_score,
            mean_qf,
            k_init_mul,
        })
    }
}

/// Mean of the float quant_field — the picker's smallest-bytes proxy
/// in [`VarDctEncoder::butteraugli_refine_quant_field`]. Larger mean
/// means coarser per-block quantization, which empirically correlates
/// with smaller encoded bytes on photographic content (fewer non-zero
/// AC coefficients → shorter Huffman/ANS streams). Computed in `f64`
/// to avoid catastrophic cancellation on large block counts.
#[cfg(feature = "butteraugli-loop")]
fn mean_qf_float(quant_field_float: &[f32]) -> f64 {
    if quant_field_float.is_empty() {
        return 0.0;
    }
    let sum: f64 = quant_field_float.iter().map(|&v| v as f64).sum();
    sum / quant_field_float.len() as f64
}

/// Debug hook for capturing the buttloop's internal reconstruction at the
/// final iteration. Off by default; gated by `feature = "__internal_recon_hook"`.
///
/// The hook is single-threaded by design (a global `Mutex<Option<...>>`) — it's
/// only meant for the Layer-1 drift-investigation test, which runs one encode
/// at a time. Concurrent encodes with capture enabled will race and one will
/// overwrite the other's recon.
///
/// The recon stored here is exactly what the buttloop measures butteraugli
/// against on its last iteration: planar linear RGB, cropped to (width, height),
/// AFTER reconstruct_xyb → gab_smooth → EPF → add_patches → add_splines →
/// xyb_to_linear_rgb_planar. If this diverges from what the user-facing decoder
/// produces from the shipped bitstream, the buttloop is targeting an image the
/// decoder never delivers — that's the drift root cause.
#[cfg(feature = "__internal_recon_hook")]
pub mod recon_hook {
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    /// Captured internal reconstruction: the exact planar linear RGB image the
    /// buttloop compared against the original on its final iteration.
    ///
    /// `r`, `g`, `b` are each `width * height` f32 in linear RGB (NOT sRGB).
    /// Values are NOT clamped to [0, 1] — the encoder operates on linear-light
    /// floats and may produce values slightly outside that range near saturation.
    #[derive(Clone)]
    pub struct InternalRecon {
        pub width: usize,
        pub height: usize,
        pub r: Vec<f32>,
        pub g: Vec<f32>,
        pub b: Vec<f32>,
        pub iter: usize,
        pub iters: usize,
        // Per-block strategy info for chunk-2 diff-map correlation.
        // Length: xsize_blocks * ysize_blocks, row-major.
        pub xsize_blocks: usize,
        pub ysize_blocks: usize,
        pub raw_strategy: Vec<u8>,
        pub is_first_block: Vec<bool>,
        pub quant_field_u8: Vec<u8>,
        // Per-tile CfL state used by the buttloop's reconstruction.
        // Length: xsize_tiles * ysize_tiles, row-major.
        pub xsize_tiles: usize,
        pub ysize_tiles: usize,
        pub cfl_ytox: Vec<i8>,
        pub cfl_ytob: Vec<i8>,
        // W44-112 Layer-1.5: `DistanceParams` the buttloop's FINAL iter used.
        // Compared against `ProductionQf` (captured after the post-buttloop
        // `transform_and_quantize_with_source`) to discriminate candidates
        // 1 (params drift) and 2 (sequential vs parallel).
        pub final_global_scale: i32,
        pub final_scale: f32,
        pub final_inv_scale: f32,
    }

    static CAPTURE_ENABLED: AtomicBool = AtomicBool::new(false);
    static LAST_RECON: Mutex<Option<InternalRecon>> = Mutex::new(None);

    /// Enable or disable capture. Defaults to disabled — even with the feature
    /// compiled in, no recon is captured unless this is set to `true`.
    pub fn set_capture_enabled(enabled: bool) {
        CAPTURE_ENABLED.store(enabled, Ordering::SeqCst);
    }

    /// Returns the current capture-enabled state. Called by the buttloop on
    /// every final iteration; cheap relaxed load.
    pub fn capture_enabled() -> bool {
        CAPTURE_ENABLED.load(Ordering::Relaxed)
    }

    /// Store the recon from the buttloop's final iteration. Overwrites any
    /// prior recon — pair with `take_last` to drain between encodes.
    pub fn store(recon: InternalRecon) {
        let mut guard = LAST_RECON.lock().expect("recon_hook mutex poisoned");
        *guard = Some(recon);
    }

    /// Take (consume) the last captured recon, leaving `None` behind.
    /// Returns `None` if no encode has captured a recon since the last take
    /// (or since process start).
    pub fn take_last() -> Option<InternalRecon> {
        let mut guard = LAST_RECON.lock().expect("recon_hook mutex poisoned");
        guard.take()
    }

    // ═════════════════════════════════════════════════════════════════════════
    // W44-112: post-PRODUCTION quant_field capture
    //
    // The InternalRecon above captures the buttloop's FINAL-iter quant_field
    // (post-internal-AdjustQuantBlockAC). After the buttloop returns, the
    // production path runs `transform_and_quantize_with_source` again, which
    // applies AdjustQuantBlockAC a SECOND time via parallel groups — possibly
    // producing different `quant_adjustments` if `params` drifted, or if the
    // parallel-vs-sequential pattern compounds differently.
    //
    // W44-112's Layer-1.5 test compares INTERNAL quant_field vs PRODUCTION
    // quant_field per-block to discriminate W44-111 candidates 1 (params
    // drift), 2 (sequential vs parallel), and 3 (recon vs decoder).
    // ═════════════════════════════════════════════════════════════════════════

    /// Captured production-path quant_field (post-`transform_and_quantize_with_source`).
    /// Length = `xsize_blocks * ysize_blocks`, row-major.
    #[derive(Clone)]
    pub struct ProductionQf {
        pub xsize_blocks: usize,
        pub ysize_blocks: usize,
        pub quant_field_u8: Vec<u8>,
        /// `DistanceParams.global_scale` used by production transform.
        pub global_scale: i32,
        /// `DistanceParams.scale` used by production transform.
        pub scale: f32,
        /// `DistanceParams.inv_scale` used by production transform.
        pub inv_scale: f32,
    }

    static PROD_QF_CAPTURE_ENABLED: AtomicBool = AtomicBool::new(false);
    static LAST_PROD_QF: Mutex<Option<ProductionQf>> = Mutex::new(None);

    /// Enable or disable production-quant-field capture. Independent from
    /// `set_capture_enabled` (the recon hook) so callers can pay only the
    /// cost of the slot they need.
    pub fn set_production_qf_capture_enabled(enabled: bool) {
        PROD_QF_CAPTURE_ENABLED.store(enabled, Ordering::SeqCst);
    }

    /// Cheap relaxed read for the production-side capture point.
    pub fn production_qf_capture_enabled() -> bool {
        PROD_QF_CAPTURE_ENABLED.load(Ordering::Relaxed)
    }

    /// Store the production-path quant_field. Overwrites any prior capture.
    pub fn store_production_qf(qf: ProductionQf) {
        let mut guard = LAST_PROD_QF
            .lock()
            .expect("production_qf hook mutex poisoned");
        *guard = Some(qf);
    }

    /// Take (consume) the last captured production quant_field.
    pub fn take_last_production_qf() -> Option<ProductionQf> {
        let mut guard = LAST_PROD_QF
            .lock()
            .expect("production_qf hook mutex poisoned");
        guard.take()
    }

    // ═════════════════════════════════════════════════════════════════════════
    // W44-116: per-step XYB capture
    //
    // The W44-113 audit ranked 4 candidates for the remaining R/G linear-RGB
    // residual between buttloop's internal recon and jxl-rs decoded output.
    // W44-114 ruled out AFV IDCT bit-parity. W44-115 (per-strategy IDCT
    // precision audit) — assume the IDCTs themselves are at parity. The
    // remaining candidates collapse to "which STEP of the buttloop's
    // recon-pipeline (reconstruct_xyb → gab_smooth → apply_epf → add_patches
    // → add_splines → xyb_to_linear_rgb_planar) is the divergent one?".
    //
    // The trick to identify the divergent step without needing a jxl-rs-side
    // intermediate dump: snapshot the XYB planes AFTER each step, then
    // convert each snapshot to linear-RGB via the SAME xyb_to_linear_rgb_planar
    // (parity-guaranteed). Each snapshot answers "what would the linear-RGB
    // output be if I stopped after step N?". The decoder always runs the
    // full pipeline, so:
    //
    //   step_div(N) = max_abs(linear_rgb_after_step_N, jxl_rs_linear_rgb)
    //
    // If the encoder pipeline is at parity with the decoder, step_div(N)
    // monotonically DECREASES as N advances through the pipeline (each
    // missing step is an error vs the decoder). If some step INCREASES
    // step_div, that step is the divergent one (it's pushing our pipeline
    // AWAY from the decoder's).
    //
    // For a divergent ordering, the FIRST step whose addition fails to
    // monotonically decrease step_div is the bug.
    // ═════════════════════════════════════════════════════════════════════════

    /// XYB planes captured AFTER each reconstruction step. All planes are
    /// `padded_width * padded_height` (not cropped). Use the existing
    /// `xyb_to_linear_rgb_planar` to convert to linear-RGB for comparison.
    ///
    /// Steps applied (in order):
    ///   1. `after_recon_xyb` — output of `reconstruct_xyb` (dequant + CfL +
    ///      LFFromDC + IDCT for all blocks). Always present.
    ///   2. `after_gab` — after `gab_smooth` if `enable_gaborish`; else None.
    ///   3. `after_epf` — after `apply_epf` if `epf_iters > 0`; else None.
    ///   4. `after_patches` — after `add_patches` if `patches_data` is Some;
    ///      else None.
    ///   5. `after_splines` — after `add_splines` if `splines_data` is Some;
    ///      else None.
    ///
    /// Each `Option<Xyb>` is `Some` iff that step was actually applied. The
    /// final `after_splines` (or the last `Some` step in the chain) should
    /// match the existing `InternalRecon::{r,g,b}` after xyb→RGB conversion.
    #[derive(Clone, Default)]
    pub struct Xyb {
        pub x: Vec<f32>,
        pub y: Vec<f32>,
        pub b: Vec<f32>,
    }

    /// W44-116 per-step XYB snapshots. See module docs above.
    #[derive(Clone)]
    pub struct StepXyb {
        pub padded_width: usize,
        pub padded_height: usize,
        pub width: usize,
        pub height: usize,
        /// Always present (first step always runs).
        pub after_recon_xyb: Xyb,
        pub after_gab: Option<Xyb>,
        pub after_epf: Option<Xyb>,
        pub after_patches: Option<Xyb>,
        pub after_splines: Option<Xyb>,
    }

    static STEPS_CAPTURE_ENABLED: AtomicBool = AtomicBool::new(false);
    static LAST_STEPS: Mutex<Option<StepXyb>> = Mutex::new(None);

    /// Enable/disable per-step XYB capture. Independent of the linear-RGB
    /// `set_capture_enabled` hook so callers can pay only the cost they need.
    /// Cost when enabled: 5 × 3 × padded_pixels f32 clones at the FINAL iter
    /// (negligible vs the buttloop itself which clones planes anyway).
    pub fn set_steps_capture_enabled(enabled: bool) {
        STEPS_CAPTURE_ENABLED.store(enabled, Ordering::SeqCst);
    }

    /// Cheap relaxed read for the per-step capture point.
    pub fn steps_capture_enabled() -> bool {
        STEPS_CAPTURE_ENABLED.load(Ordering::Relaxed)
    }

    /// Store the per-step XYB snapshot. Overwrites any prior capture.
    pub fn store_steps(steps: StepXyb) {
        let mut guard = LAST_STEPS.lock().expect("steps hook mutex poisoned");
        *guard = Some(steps);
    }

    /// Take (consume) the last per-step XYB snapshot.
    pub fn take_last_steps() -> Option<StepXyb> {
        let mut guard = LAST_STEPS.lock().expect("steps hook mutex poisoned");
        guard.take()
    }
}

// ===== Distance-aware buttloop tuning unit tests (W38-2 #3.1) =====
//
// These tests share global atomic state with sweep harnesses. Run
// serially (`cargo test --lib -- --test-threads=1` if interleaved
// flakes appear). They mirror the GPU encoder's
// `forks/butteraugli_loop.rs::resolved_*` tests for parity.
#[cfg(test)]
mod tuning_tests {
    use super::*;

    fn reset_overrides() {
        CUR_POW_X1000_LOW.store(i32::MIN, Ordering::Relaxed);
        CUR_POW_X1000_HIGH.store(i32::MIN, Ordering::Relaxed);
        MAX_INCREASE_X1000_LOW.store(i32::MIN, Ordering::Relaxed);
        MAX_INCREASE_X1000_HIGH.store(i32::MIN, Ordering::Relaxed);
        MAX_INCREASE_X1000_HIGH_SCREENSHOT.store(i32::MIN, Ordering::Relaxed);
        DISTANCE_SPLIT_X1000.store(2000, Ordering::Relaxed);
    }

    #[test]
    fn resolved_cur_pow_uses_low_default_below_split() {
        reset_overrides();
        // d=1.0 < 2.0 → LOW regime.
        let v = resolved_cur_pow(0, 1.0);
        assert!(
            (v - DEFAULT_CUR_POW_LOW).abs() < 1e-9,
            "expected DEFAULT_CUR_POW_LOW={DEFAULT_CUR_POW_LOW}, got {v}"
        );
        // iter=1 also LOW regime.
        let v1 = resolved_cur_pow(1, 1.5);
        assert!((v1 - DEFAULT_CUR_POW_LOW).abs() < 1e-9);
    }

    #[test]
    fn resolved_cur_pow_uses_high_default_at_or_above_split() {
        reset_overrides();
        // d=2.0 >= 2.0 → HIGH regime.
        let v = resolved_cur_pow(0, 2.0);
        assert!(
            (v - DEFAULT_CUR_POW_HIGH).abs() < 1e-9,
            "expected DEFAULT_CUR_POW_HIGH={DEFAULT_CUR_POW_HIGH}, got {v}"
        );
        // d=3.0 — RD-pareto target; HIGH.
        let v3 = resolved_cur_pow(0, 3.0);
        assert!((v3 - DEFAULT_CUR_POW_HIGH).abs() < 1e-9);
    }

    #[test]
    fn resolved_cur_pow_zero_at_late_iterations() {
        reset_overrides();
        // iter >= 2 → 0.0 regardless of regime.
        assert_eq!(resolved_cur_pow(2, 1.0), 0.0);
        assert_eq!(resolved_cur_pow(3, 3.0), 0.0);
        assert_eq!(resolved_cur_pow(99, 5.0), 0.0);
    }

    #[test]
    fn resolved_max_increase_picks_per_regime_default() {
        reset_overrides();
        let v_low = resolved_max_increase(1.0);
        assert!((v_low - DEFAULT_MAX_INCREASE_LOW).abs() < 1e-9);
        let v_high = resolved_max_increase(3.0);
        assert!((v_high - DEFAULT_MAX_INCREASE_HIGH).abs() < 1e-9);
        // Edge: exactly at split → HIGH.
        let v_split = resolved_max_increase(2.0);
        assert!((v_split - DEFAULT_MAX_INCREASE_HIGH).abs() < 1e-9);
    }

    #[test]
    fn override_round_trip_x1000() {
        reset_overrides();
        // Confirm the X1000 encoding round-trips through resolve helpers.
        CUR_POW_X1000_HIGH.store(350, Ordering::Relaxed); // 0.350
        let v = resolved_cur_pow(0, 3.0);
        assert!((v - 0.35).abs() < 1e-9, "got {v}");
        MAX_INCREASE_X1000_LOW.store(1500, Ordering::Relaxed); // 1.500
        let m = resolved_max_increase(1.0);
        assert!((m - 1.5).abs() < 1e-9, "got {m}");
        reset_overrides();
    }

    /// Production defaults must match libjxl
    /// `enc_adaptive_quantization.cc:1106` at every regime — both LOW
    /// and HIGH ship libjxl-faithful values until A/B sweeps find
    /// CPU-specific tuning that survives RD-pareto.
    ///
    /// The atomic-override scaffolding is intentional (sweep harnesses
    /// can override LOW), but production CPU encodes are byte-identical
    /// to pre-port behaviour.
    #[test]
    fn production_defaults_are_libjxl_faithful() {
        // libjxl `kPow = {0.2, 0.2, 0, 0, ...}` (one entry per iter).
        assert_eq!(DEFAULT_CUR_POW_LOW, 0.2);
        assert_eq!(DEFAULT_CUR_POW_HIGH, 0.2);
        // libjxl applies no cap to `diff = tile_dist / target_distance`.
        // Encode as 100.0 ("effectively infinite" — block diffs of
        // 100× would already saturate at qf_higher).
        assert_eq!(DEFAULT_MAX_INCREASE_LOW, 100.0);
        assert_eq!(DEFAULT_MAX_INCREASE_HIGH, 100.0);
        assert_eq!(DEFAULT_DISTANCE_SPLIT, 2.0);
    }

    /// W44-105: production seed-scale default for screenshot-class
    /// content at d>=2 in the buttloop. Empirically chosen from a
    /// scale sweep on the terminal e8 d=4 wedge cell (SCALE=4 hits
    /// +3.42 SSIM2 / +31% bytes vs baseline; higher values give bigger
    /// SSIM2 wins but cost more bytes; at SCALE=10 we still ship 14%
    /// fewer bytes than cjxl with matching SSIM2). 4.0 is the
    /// conservative balance — passes the +3 SSIM2 acceptance gate
    /// while staying ~30% smaller than cjxl on the wedge cell.
    ///
    /// Photo-class content is byte-identical to pre-W44-105 (scale=1.0).
    #[test]
    fn w44_105_default_screenshot_seed_scale_is_4() {
        assert_eq!(DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE, 4.0);
        // The constant must be positive and finite — used as a multiplier
        // on the float quant field. Negative or NaN would silently corrupt
        // the loop's starting state.
        assert!(DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE.is_finite());
        assert!(DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE > 0.0);
    }

    /// W44-107: the lower-distance gate on the W44-105 seed-scale fix
    /// is `BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE = 3.5`. Raised from the
    /// original W44-105 value of 2.0 after the W44-106 ledger refresh
    /// flagged a FIXED→OPEN regression on `codec_wiki.png e8 d=3`.
    ///
    /// The constant MUST be a positive finite f32 — it is compared
    /// directly to `target_distance` (also f32 — `VarDctEncoder.distance`)
    /// so NaN / negative would silently disable the gate. 3.5 sits
    /// between d=3 (the regression cell) and d=4 (the largest W44-105
    /// win cluster), giving codec_wiki d=3 the pre-W44-105 byte-identical
    /// behaviour while preserving every d=4+ win.
    #[test]
    fn w44_107_seed_scale_min_distance_is_3p5() {
        assert_eq!(BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE, 3.5);
        assert!(BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE.is_finite());
        assert!(BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE > 0.0);
        // Must sit strictly between d=3 (the codec_wiki regression cell)
        // and d=4 (the largest W44-105 win cluster) so the gate boundary
        // can never re-engage on d=3 due to a floating-point coercion.
        assert!(BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE > 3.0);
        assert!(BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE <= 4.0);
    }

    /// W44-108: the sub-discriminator lower bound (admit d=2..3.5 fire
    /// for low-colour screenshots) MUST be 2.0 — recovers the 8 W44-105
    /// wins W44-107 sacrificed (terminal d=2/2.5/3, imac_g3 d=3,
    /// terminal e9 d=2.5, codec_wiki d=2/2.5 stay rejected via the
    /// `m3` gate).
    #[test]
    fn w44_108_seed_scale_sub_min_distance_is_2p0() {
        assert_eq!(BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE, 2.0);
        assert!(BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE.is_finite());
        assert!(BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE > 0.0);
        // Must be STRICTLY LESS than the W44-107 upper gate so the
        // sub-band is non-empty. Equality would collapse into the
        // W44-107-only gate.
        assert!(BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE < BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE);
    }

    /// W44-108: the colour separator MUST be 30.0 — splits codec_wiki
    /// (M3 ≈ 146, the W44-107 regression target) from terminal (M3 ≈ 14),
    /// imac_g3 (M3 ≈ 14), imac_dark (M3 ≈ 21) with ~5× margin both
    /// sides per `examples/w44_108_proxy_probe.rs`.
    #[test]
    fn w44_108_seed_scale_low_colour_m3_max_is_30() {
        assert_eq!(BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX, 30.0);
        assert!(BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX.is_finite());
        assert!(BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX > 0.0);
        // Margin floor: must be > imac_dark's measured M3 = 21 by at
        // least 5× upper-bound headroom for natural variance, and must
        // be < codec_wiki's 146 by at least 4× margin.
        assert!(BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX > 21.0);
        assert!(BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX < 100.0);
    }

    #[test]
    fn distance_split_override_shifts_regime() {
        reset_overrides();
        // Lower the split to 1.0 — then d=1.5 is HIGH.
        DISTANCE_SPLIT_X1000.store(1000, Ordering::Relaxed);
        let v = resolved_cur_pow(0, 1.5);
        assert!((v - DEFAULT_CUR_POW_HIGH).abs() < 1e-9, "got {v}");
        reset_overrides();
    }

    // ===== W39-2 (WF3 fix): screenshot-class HIGH-regime cap tests =====

    /// `resolved_max_increase_with_class(d, false)` is byte-identical to
    /// `resolved_max_increase(d)` at every regime — photo class is the
    /// pre-W39-2 path.
    #[test]
    fn class_blind_resolver_byte_identical_to_legacy() {
        reset_overrides();
        for &d in &[0.5_f64, 1.0, 1.5, 1.99, 2.0, 3.0, 4.0, 5.0] {
            let legacy = resolved_max_increase(d);
            let class_blind = resolved_max_increase_with_class(d, false);
            assert!(
                (legacy - class_blind).abs() < 1e-9,
                "d={d}: legacy={legacy} class_blind={class_blind}",
            );
        }
    }

    /// Screenshot class at LOW regime uses LOW slot (the WF3 fix
    /// applies to HIGH only — LOW is W39-1 territory and the literal
    /// GPU LOW tuning regressed CPU; we don't add a screenshot LOW
    /// branch here).
    #[test]
    fn screenshot_class_low_regime_uses_low_default() {
        reset_overrides();
        let v = resolved_max_increase_with_class(1.0, true);
        assert!(
            (v - DEFAULT_MAX_INCREASE_LOW).abs() < 1e-9,
            "screenshot LOW: expected DEFAULT_MAX_INCREASE_LOW={DEFAULT_MAX_INCREASE_LOW}, got {v}"
        );
        // Edge: d just below split → LOW.
        let v2 = resolved_max_increase_with_class(1.99, true);
        assert!((v2 - DEFAULT_MAX_INCREASE_LOW).abs() < 1e-9, "got {v2}");
    }

    /// Screenshot class at HIGH regime, unmodified slot, picks the
    /// screenshot default. With both defaults at `100.0` ("no cap")
    /// the result must equal `100.0` exactly — hash-locks rely on this.
    #[test]
    fn screenshot_class_high_regime_unmodified_picks_screenshot_default() {
        reset_overrides();
        let v = resolved_max_increase_with_class(3.0, true);
        // Both DEFAULT_MAX_INCREASE_HIGH (100.0) and
        // DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT (100.0) are at "no
        // cap"; the .min() picks 100.0 deterministically.
        assert!(
            (v - DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT).abs() < 1e-9,
            "expected DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT={DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT}, got {v}"
        );
        // Edge: exactly at split → HIGH branch fires.
        let v_split = resolved_max_increase_with_class(2.0, true);
        assert!(
            (v_split - DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT).abs() < 1e-9,
            "got {v_split}"
        );
    }

    /// Sweep override on the screenshot slot is honoured only for
    /// screenshot-class HIGH lookups. Photo class (and LOW) stay on
    /// the legacy slots — critical so production photo encodes are
    /// byte-identical to pre-W39-2 when the harness is running.
    #[test]
    fn screenshot_override_only_affects_screenshot_high() {
        reset_overrides();
        // Set screenshot HIGH cap to 1.5.
        MAX_INCREASE_X1000_HIGH_SCREENSHOT.store(1500, Ordering::Relaxed);

        // Screenshot HIGH → reads override.
        let v_s = resolved_max_increase_with_class(3.0, true);
        assert!(
            (v_s - 1.5).abs() < 1e-9,
            "screenshot HIGH: expected 1.5, got {v_s}"
        );

        // Photo HIGH → reads legacy HIGH slot (default 100.0).
        let v_p = resolved_max_increase_with_class(3.0, false);
        assert!(
            (v_p - DEFAULT_MAX_INCREASE_HIGH).abs() < 1e-9,
            "photo HIGH: expected DEFAULT_MAX_INCREASE_HIGH={DEFAULT_MAX_INCREASE_HIGH}, got {v_p}"
        );

        // Screenshot LOW → reads LOW slot (default 100.0), unaffected
        // by the HIGH screenshot override.
        let v_sl = resolved_max_increase_with_class(1.0, true);
        assert!(
            (v_sl - DEFAULT_MAX_INCREASE_LOW).abs() < 1e-9,
            "screenshot LOW: expected DEFAULT_MAX_INCREASE_LOW={DEFAULT_MAX_INCREASE_LOW}, got {v_sl}"
        );

        // Photo LOW → reads LOW slot (default 100.0).
        let v_pl = resolved_max_increase_with_class(1.0, false);
        assert!(
            (v_pl - DEFAULT_MAX_INCREASE_LOW).abs() < 1e-9,
            "photo LOW: expected DEFAULT_MAX_INCREASE_LOW={DEFAULT_MAX_INCREASE_LOW}, got {v_pl}"
        );
        reset_overrides();
    }

    /// When BOTH the shared HIGH slot and the screenshot HIGH slot are
    /// overridden, the screenshot HIGH lookup must pick the
    /// *more-restrictive* (lower) cap. This lets a sweep harness pin
    /// the shared HIGH cap to 1.5 (e.g., for cross-class
    /// experimentation) without losing the screenshot-specific
    /// override of 1.3 if it's also set.
    #[test]
    fn screenshot_high_picks_min_of_shared_and_screenshot_slots() {
        reset_overrides();
        // Shared HIGH = 2.0, screenshot HIGH = 1.3 → expect 1.3.
        MAX_INCREASE_X1000_HIGH.store(2000, Ordering::Relaxed);
        MAX_INCREASE_X1000_HIGH_SCREENSHOT.store(1300, Ordering::Relaxed);
        let v = resolved_max_increase_with_class(3.0, true);
        assert!((v - 1.3).abs() < 1e-9, "expected 1.3, got {v}");

        // Reverse: shared HIGH = 1.3, screenshot HIGH = 2.0 → expect 1.3.
        MAX_INCREASE_X1000_HIGH.store(1300, Ordering::Relaxed);
        MAX_INCREASE_X1000_HIGH_SCREENSHOT.store(2000, Ordering::Relaxed);
        let v = resolved_max_increase_with_class(3.0, true);
        assert!((v - 1.3).abs() < 1e-9, "expected 1.3, got {v}");
        reset_overrides();
    }

    /// W39-2 default-off invariant: the screenshot HIGH default
    /// constant is 100.0 ("no cap"). This guards against accidentally
    /// flipping the default before the sweep has identified a winning
    /// cap value. Once the sweep is analysed and a value chosen, flip
    /// the constant AND update this test.
    #[test]
    fn screenshot_high_default_is_no_cap_until_sweep_lands() {
        // Currently 100.0 (default-off) — pending sweep results from
        // `examples/buttloop_screenshot_cap_sweep.rs`.
        assert_eq!(DEFAULT_MAX_INCREASE_HIGH_SCREENSHOT, 100.0);
        // The classifier threshold reuses the same `95.0` value as
        // splines::looks_like_screenshot / encoder.rs's
        // CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD so we don't have
        // a third boundary to maintain.
        assert_eq!(SCREENSHOT_MEDIAN_THRESHOLD, 95.0);
    }

    // W44-109: resolved_adaptive_quant_qf_seed_scale tests.

    #[test]
    fn w44_109_high_effort_returns_1_to_avoid_double_scale() {
        // e>=8 → W44-105 buttloop owns the scale. Helper returns 1.0
        // (no double-apply).
        for effort in 8..=12 {
            assert_eq!(
                resolved_adaptive_quant_qf_seed_scale(effort, 2, true, 4.0, Some(14.0)),
                1.0,
                "effort {effort} should return 1.0 (W44-105 buttloop path)"
            );
        }
    }

    #[test]
    fn w44_109_buttloop_iters_gt_0_returns_1() {
        // Even at low effort, if the caller pinned butteraugli_iters,
        // the W44-105 path will run — don't double-apply.
        assert_eq!(
            resolved_adaptive_quant_qf_seed_scale(7, 2, true, 4.0, Some(14.0)),
            1.0
        );
    }

    #[test]
    fn w44_109_non_screenshot_returns_1() {
        // Photo content (mask1x1 median ≤ 95) → is_screenshot=false →
        // helper returns 1.0. Verifies "do not regress photos" guarantee.
        assert_eq!(
            resolved_adaptive_quant_qf_seed_scale(5, 0, false, 4.0, Some(14.0)),
            1.0
        );
        assert_eq!(
            resolved_adaptive_quant_qf_seed_scale(6, 0, false, 4.0, Some(50.0)),
            1.0
        );
    }

    #[test]
    fn w44_109_low_distance_returns_1() {
        // d < SUB_MIN_DISTANCE (=2.0) → never fires regardless of m3.
        assert_eq!(
            resolved_adaptive_quant_qf_seed_scale(5, 0, true, 1.9, Some(14.0)),
            1.0
        );
        // d >= 2.0 but < 3.5 AND m3 missing → W44-108 sub-gate can't fire.
        assert_eq!(
            resolved_adaptive_quant_qf_seed_scale(5, 0, true, 2.5, None),
            1.0
        );
    }

    #[test]
    fn w44_109_high_distance_fires_on_screenshot() {
        // d >= W44-107 threshold (3.5) → fires for is_screenshot
        // regardless of m3 value. Effort-dependent magnitude:
        // e5/e6 = E5_E6 constant; e7 = E7 constant.
        for effort in 5..=6u8 {
            for dist in [3.5_f32, 4.0, 5.0, 6.0].iter() {
                let v = resolved_adaptive_quant_qf_seed_scale(effort, 0, true, *dist, None);
                assert!(
                    (v - DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6).abs() < 1e-9,
                    "effort {effort} d={dist} m3=None: expected {DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6}, got {v}"
                );
                let v = resolved_adaptive_quant_qf_seed_scale(effort, 0, true, *dist, Some(14.0));
                assert!(
                    (v - DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6).abs() < 1e-9,
                    "effort {effort} d={dist} m3=14: expected {DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6}, got {v}"
                );
            }
        }
        // e7 uses the higher constant.
        for dist in [3.5_f32, 4.0, 5.0, 6.0].iter() {
            let v = resolved_adaptive_quant_qf_seed_scale(7, 0, true, *dist, None);
            assert!(
                (v - DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7).abs() < 1e-9,
                "e7 d={dist}: expected {DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7}, got {v}"
            );
        }
    }

    #[test]
    fn w44_109_w44_108_sub_gate_fires_on_low_colour() {
        // d ∈ [2.0, 3.5) AND m3 < 30 → W44-108 sub-discriminator
        // engages (terminal/imac_g3/imac_dark-class). Magnitude follows
        // the same effort split.
        for dist in [2.0_f32, 2.5, 3.0, 3.4].iter() {
            let v = resolved_adaptive_quant_qf_seed_scale(5, 0, true, *dist, Some(14.0));
            assert!(
                (v - DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6).abs() < 1e-9,
                "e5 d={dist} m3=14: expected {DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6}, got {v}"
            );
            let v = resolved_adaptive_quant_qf_seed_scale(7, 0, true, *dist, Some(14.0));
            assert!(
                (v - DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7).abs() < 1e-9,
                "e7 d={dist} m3=14: expected {DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7}, got {v}"
            );
        }
    }

    #[test]
    fn w44_109_w44_108_sub_gate_skips_on_high_colour() {
        // codec_wiki-class (m3 = 146) at d=3.0 → sub-gate REJECTS,
        // preserves codec_wiki d=3 (W44-107/108 design intent).
        for dist in [2.0_f32, 2.5, 3.0, 3.4].iter() {
            let v = resolved_adaptive_quant_qf_seed_scale(5, 0, true, *dist, Some(146.0));
            assert_eq!(v, 1.0, "d={dist} m3=146 must NOT fire (codec_wiki class)");
        }
    }

    #[test]
    fn w44_109_default_scales_documented_magnitudes() {
        // W44-109 e5/e6 = 2.0, e7 = 3.0 — both LOWER than W44-105's 4.0
        // because there's no buttloop settling. Asserting the
        // documented values so empirical re-tuning lands a code review.
        assert_eq!(DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6, 2.0);
        assert_eq!(DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7, 3.0);
        // W44-105 (e>=8 with buttloop) stays at 4.0.
        assert_eq!(DEFAULT_BUTTLOOP_SCREENSHOT_QF_SEED_SCALE, 4.0);
    }

    #[test]
    fn w44_109_default_max_effort_is_7() {
        // Gate fires at e ∈ {5, 6, 7}; e>=8 is owned by W44-105.
        assert_eq!(ADAPTIVE_QUANT_QF_SEED_SCALE_MAX_EFFORT, 7);
    }

    // W44-145: per-block adaptive qac scaling tests.

    #[test]
    fn w44_145_per_block_scale_collapses_when_full_scale_is_1() {
        // When the outer W44-109 gate doesn't fire, full_scale == 1.0;
        // per-block scale must be exactly 1.0 regardless of mask value
        // (no per-block compute should run, but the helper still
        // short-circuits defensively).
        assert_eq!(w44_145_per_block_qf_scale(0.0, 1.0), 1.0);
        assert_eq!(w44_145_per_block_qf_scale(50.0, 1.0), 1.0);
        assert_eq!(w44_145_per_block_qf_scale(100.0, 1.0), 1.0);
    }

    #[test]
    fn w44_145_per_block_scale_high_mask_returns_1() {
        // Blank background blocks (mask saturates near 100) → return 1.0
        // (no scaling), matching cjxl's qac ≈ 7 in those regions.
        assert_eq!(w44_145_per_block_qf_scale(99.5, 2.0), 1.0);
        assert_eq!(w44_145_per_block_qf_scale(100.0, 2.0), 1.0);
        assert_eq!(w44_145_per_block_qf_scale(150.0, 3.0), 1.0);
    }

    #[test]
    fn w44_145_per_block_scale_low_mask_returns_full() {
        // Text-glyph blocks (mask ≤ LOW threshold) → return full W44-109
        // scale, matching cjxl's qac ≈ 97 in those regions.
        let e5_full = DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6;
        let e7_full = DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7;
        assert_eq!(w44_145_per_block_qf_scale(W44_145_PER_BLOCK_MASK_LOW, e5_full), e5_full);
        assert_eq!(w44_145_per_block_qf_scale(50.0, e5_full), e5_full);
        assert_eq!(w44_145_per_block_qf_scale(20.0, e7_full), e7_full);
        assert_eq!(w44_145_per_block_qf_scale(0.0, e7_full), e7_full);
    }

    #[test]
    fn w44_145_per_block_scale_interpolates_linearly() {
        // Midpoint of [70, 99.5] is 84.75; at full_scale = 2.0 returns
        // 1.0 + 0.5 * (2.0 - 1.0) = 1.5.
        let mid = (W44_145_PER_BLOCK_MASK_LOW + W44_145_PER_BLOCK_MASK_HIGH) * 0.5;
        let v = w44_145_per_block_qf_scale(mid, 2.0);
        assert!((v - 1.5).abs() < 1e-6, "midpoint scale: expected 1.5, got {v}");
        // 1/4 from HIGH: t = 0.25 → scale = 1.0 + 0.25 * (3.0 - 1.0) = 1.5
        let quarter_high = W44_145_PER_BLOCK_MASK_HIGH
            - 0.25 * (W44_145_PER_BLOCK_MASK_HIGH - W44_145_PER_BLOCK_MASK_LOW);
        let v = w44_145_per_block_qf_scale(quarter_high, 3.0);
        assert!((v - 1.5).abs() < 1e-5, "quarter-from-HIGH e7 scale: expected 1.5, got {v}");
    }

    #[test]
    fn w44_145_per_block_scale_monotonic_in_mask() {
        // As mask drops from HIGH to LOW, scale rises monotonically from
        // 1.0 to full_scale. This invariant catches sign flips or
        // accidental clamping bugs.
        let full = 2.0_f32;
        let mut prev = 1.0_f32;
        let n_steps = 32_usize;
        for i in 0..=n_steps {
            let frac = i as f32 / n_steps as f32;
            let mask = W44_145_PER_BLOCK_MASK_HIGH
                - frac * (W44_145_PER_BLOCK_MASK_HIGH - W44_145_PER_BLOCK_MASK_LOW);
            let v = w44_145_per_block_qf_scale(mask, full);
            assert!(
                v >= prev - 1e-6,
                "monotonicity broken at mask={mask}: scale {v} < prev {prev}"
            );
            prev = v;
        }
        // Endpoints: HIGH → 1.0, LOW → full
        let lo = w44_145_per_block_qf_scale(W44_145_PER_BLOCK_MASK_LOW, full);
        let hi = w44_145_per_block_qf_scale(W44_145_PER_BLOCK_MASK_HIGH, full);
        assert!((lo - full).abs() < 1e-6);
        assert!((hi - 1.0).abs() < 1e-6);
    }

    #[test]
    fn w44_145_per_block_mask_mean_uniform_field() {
        // 2x2 blocks (16x16 padded plane), mask = uniform 100.0 → every
        // block mean = 100.0.
        let xsize_blocks = 2;
        let ysize_blocks = 2;
        let padded_width = xsize_blocks * 8;
        let padded_height = ysize_blocks * 8;
        let mask = vec![100.0_f32; padded_width * padded_height];
        let means = per_block_mask1x1_mean(&mask, padded_width, xsize_blocks, ysize_blocks);
        assert_eq!(means.len(), 4);
        for m in &means {
            assert!((m - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn w44_145_per_block_mask_mean_split_field() {
        // Top-left block (8x8) all 100, top-right all 50, bottom-left
        // all 80, bottom-right all 20. Verify the row-major block order
        // and per-block reduction land the right means in the right slots.
        let xsize_blocks = 2;
        let ysize_blocks = 2;
        let padded_width = xsize_blocks * 8;
        let padded_height = ysize_blocks * 8;
        let mut mask = vec![0.0_f32; padded_width * padded_height];
        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                let val = match (by, bx) {
                    (0, 0) => 100.0,
                    (0, 1) => 50.0,
                    (1, 0) => 80.0,
                    (1, 1) => 20.0,
                    _ => unreachable!(),
                };
                for dy in 0..8 {
                    for dx in 0..8 {
                        mask[(by * 8 + dy) * padded_width + (bx * 8 + dx)] = val;
                    }
                }
            }
        }
        let means = per_block_mask1x1_mean(&mask, padded_width, xsize_blocks, ysize_blocks);
        assert!((means[0] - 100.0).abs() < 1e-6);
        assert!((means[1] - 50.0).abs() < 1e-6);
        assert!((means[2] - 80.0).abs() < 1e-6);
        assert!((means[3] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn w44_142_constants_match_w44_124_split() {
        // The W44-142 EPF seed suppression sub-gate intentionally reuses
        // the exact codec_wiki-vs-other-screens predicate as W44-124's
        // dct32_keep auto-discriminator. If the W44-124 thresholds are
        // tightened/loosened, the W44-142 thresholds should follow (and
        // vice versa). Pin both constants to the same numeric values to
        // catch silent drift in future tunings.
        //
        // (The W44-124 constants live in `vardct/encoder.rs`. If those
        // are renamed or moved, update this test together with both
        // call sites.)
        assert_eq!(
            W44_142_EPF_SEED_SUPPRESS_M3_MIN, 60.0,
            "W44-142 m3 floor must match W44-124's W44_124_DCT32_KEEP_M3_MIN (60.0); \
             both gates discriminate codec_wiki (m3=145.73) from terminal/imac_g3 \
             (m3 ≈ 14..21) and imessage (m3=67.65, blocked by ed)"
        );
        assert_eq!(
            W44_142_EPF_SEED_SUPPRESS_EDGE_DENSITY_MAX, 0.05,
            "W44-142 ed ceiling must match W44-124's W44_124_DCT32_KEEP_EDGE_DENSITY_MAX \
             (0.05); both gates exclude imessage (ed=0.0533) and CID22 photos (ed >= 0.16)"
        );
    }

    #[test]
    fn w44_142_max_distance_equals_w44_140_fade_max() {
        // W44-142 conservative cutoff: only suppress W44-117 inside the
        // W44-140 fade band [W44_120_EPF_SEED_MIN_DISTANCE,
        // W44_140_EPF_SEED_FADE_MAX) = [1.0, 1.5). Above d=1.5 the
        // W44-140 fade is already weight=1.0 (full W44-117 seed), and
        // the W44-141 cluster regressions at d=1.6/1.8 are NOT caused
        // by W44-140 (bytes byte-identical pre/post-W44-140 main at
        // d>=1.5) — they are W44-135 follow-on candidates. Suppressing
        // at d=1.6/1.8 introduces NEW e8 regressions (-1.02 SSIM2 vs
        // W134, worse than the W141 baseline of -0.62) per the
        // W44-142 bisect.
        assert_eq!(
            W44_142_EPF_SEED_SUPPRESS_MAX_DISTANCE, W44_140_EPF_SEED_FADE_MAX,
            "W44-142 cutoff must equal W44-140 fade-band upper edge to keep \
             the suppression strictly INSIDE the W44-140 fade band"
        );
        assert!(
            W44_142_EPF_SEED_SUPPRESS_MAX_DISTANCE > W44_120_EPF_SEED_MIN_DISTANCE,
            "the suppression band must have positive width — must be \
             strictly above W44_120_EPF_SEED_MIN_DISTANCE"
        );
        // Strict 1.5 pick documented in const comment.
        assert_eq!(W44_142_EPF_SEED_SUPPRESS_MAX_DISTANCE, 1.5);
    }
}
