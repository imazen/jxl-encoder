// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Always-compiled perceptual-tuning constants + the adaptive-quant
//! qf-seed pre-scale surface.
//!
//! These items were extracted from [`super::perceptual_loop`] (which is
//! gated behind the `butteraugli-loop` cargo feature) because the core
//! VarDCT encoder and the [`crate::tuning`] re-export surface reference
//! them UNCONDITIONALLY — encode-only builds (no `butteraugli-loop`) must
//! still resolve `SCREENSHOT_MEDIAN_THRESHOLD`,
//! `resolved_adaptive_quant_qf_seed_scale_with_policy`, and the W44-* qf-seed
//! tuning constants.
//!
//! Everything here is butteraugli-CRATE-free: pure `pub const` literals plus
//! four functions that depend only on [`crate::api::AdaptiveQuantQfSeedPolicy`],
//! [`crate::vardct::encoder::ZenanalyzeProxies`], the
//! [`crate::runtime_or_default!`] macro, and each other.
//!
//! [`super::perceptual_loop`] re-exports this module's symbols via
//! `pub(crate) use super::perceptual_tuning::*;` so the loop-internal code
//! (the EPF-seed blend, the per-block mask scale, `run_buttloop`) keeps
//! referencing them by bare name unchanged.

// Several of these tuning constants / the no-policy `resolved_*` wrapper are
// only consumed by feature-gated code (the `butteraugli-loop` quant-refine
// loop + the `__buttloop_overrides` A/B harness surface). In encode-only
// builds (no `butteraugli-loop`) they compile but are unused; allow that
// rather than gating individual items, since the values are load-bearing for
// the gated paths and must stay verbatim.
#![allow(dead_code)]

/// libjxl's hardcoded `kInitMul` (`enc_adaptive_quantization.cc:1042`)
/// that pulls the post-`kOriginalComparisonRound` quant field back toward
/// the initial AC heuristic field. Single-seed encodes use only this
/// value (bit-identical to libjxl).
pub(crate) const LIBJXL_INIT_MUL: f64 = 0.6;

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

/// W44-AUDIT-6 Phase 2C (2026-05-24): minimum m3_colourfulness for high-colour exclude.
pub const W44_AUDIT_6_HIGH_COLOUR_M3_MIN: f32 = 80.0;
/// W44-AUDIT-6 Phase 3: minimum fcbr (disjunct with edge_density).
pub const W44_AUDIT_6_HIGH_COLOUR_FCBR_MIN: f32 = 0.5;
/// W44-AUDIT-6 Phase 3: minimum edge_density (disjunct with fcbr).
pub const W44_AUDIT_6_HIGH_COLOUR_EDGE_DENSITY_MIN: f32 = 0.45;

/// W44-AUDIT-6 Phase 3: returns true when proxies indicate a high-colour
/// mixed-content screenshot (m3 >= 80 AND (fcbr >= 0.5 OR ed >= 0.45)).
pub(crate) fn w44_audit_6_is_high_colour_class(
    proxies: Option<&crate::vardct::encoder::ZenanalyzeProxies>,
) -> bool {
    let Some(p) = proxies else {
        return false;
    };
    if p.m3_colourfulness < W44_AUDIT_6_HIGH_COLOUR_M3_MIN {
        return false;
    }
    p.flat_color_block_ratio >= W44_AUDIT_6_HIGH_COLOUR_FCBR_MIN
        || p.edge_density >= W44_AUDIT_6_HIGH_COLOUR_EDGE_DENSITY_MIN
}

/// W44-108: upper bound on `ZenanalyzeProxies::m3_colourfulness` for the
/// sub-discriminator that admits the d=2..3 fire-band. The probe
/// (`examples/w44_108_proxy_probe.rs`) measured:
///
/// - terminal: M3 = 13.85
/// - imac_g3: M3 = 14.29
/// - imac_dark: M3 = 20.96
/// - **codec_wiki: M3 = 145.73** (rich colour content, the W44-107 regression target)
///
/// The original 30.0 threshold claimed "~6× margin both sides", but the
/// 6× was vs codec_wiki (145.7) only — the never-measured (21, 30) band
/// turned out to contain real content. GOAL_BEAT_CJXL wedge 2
/// (scoreboard 2026-06-12): imazen-26 ai-products (smooth low-colour
/// studio shots, m3 = 28.19, fcbr = 0.70 — passes every block-based
/// screenshot proxy INCLUDING the W44-164 classifier) tripped the
/// sub-gate at d ≥ 2 and took the e7 3× qf lift, producing a
/// distance-MONOTONICITY VIOLATION (d2.0 output bigger AND higher
/// quality than our own d1.9: 56.6 KB → 115 KB) and +108..116 % bytes
/// vs cjxl. 24.0 re-splits the band at the ratio midpoint of the
/// measured winner max (imac_dark 20.96) and the misfire class
/// (ai-products 28.19; windows95 27.19 also stops sub-band-firing —
/// re-bisection measured its sub-band lift as bytes-negative at flat
/// quality, see `benchmarks/qfseed_m3_rebisect_2026-06-12.{tsv,meta}`).
/// Photos in the validation set range M3 ≈ 32..99 — outside the
/// sub-band by design (photos also fail the `is_screenshot` mask
/// gate, so this is belt-and-braces).
pub const BUTTLOOP_QF_SEED_SCALE_LOW_COLOUR_M3_MAX: f32 = 24.0;

/// Task #12 (#74, 2026-07-14): minimum `mask1x1` 25th-percentile required
/// for the W44-108 low-colour SUB-band (`is_screenshot AND m3 < 24 AND
/// d ∈ [2.0, 3.5)`) to fire the qf-seed lift. The wedge-2 m3 threshold
/// (30 → 24, 2026-06-12) caught the ai-products misfires at m3 ≈ 28 but
/// LEAKS the residual pure-white-background low-colour product shots at
/// m3 < 24: `9290_...beard-oil` (m3 = 23.4, is_screenshot median = 100)
/// still tripped the sub-band and took the e7 3× lift at d = 2.0,
/// re-incurring the distance-MONOTONICITY VIOLATION (d1.99 = 62790 B →
/// d2.0 = 128100 B, +104 %; +112 % vs cjxl at ssim2 87 when the d2.0
/// target wants ssim2 76). Ground truth: cjxl does NOT over-allocate on
/// these (cjxl d2.0 = 60312 B ≈ our lift-OFF 62533 B), whereas on the
/// genuine text-class screenshots the lift is meant to match cjxl's
/// bimodal-qac over-allocation (imac_dark d2.0: cjxl = 461960 B ≫ our
/// lift-off 200197 B).
///
/// `m3`/`fcbr`/`edge`/`luma_var` CANNOT separate the misfire from the win
/// (beard-oil ≈ imac_dark on all four). `mask1x1_p25` CAN, orthogonally:
/// flat UI panels saturate `compute_mask1x1` at ~100 (`1/(log1p(0)+0.01)`),
/// so genuine screenshots have ≥ 25 % perfectly-flat blocks → high p25;
/// photographic texture keeps p25 down. Measured within the m3 < 24
/// sub-band (`benchmarks/qfseed_p25_discriminator_2026-07-14.tsv`):
/// - low-colour SHIP screenshots (gmessages/graph/gui/imac_dark/imac_g3/
///   imac_g3_strip/terminal/windows): p25 ∈ [98.9, 100.0]
/// - misfiring white-bg product photos (beard-oil/9285/9286 + baby set):
///   p25 ∈ [45, 89]
///
/// 95.0 sits in the clean gap (worst SHIP imac_dark 98.9 vs worst misfire
/// 89.0), 6 pp above the worst photo and 3.9 pp below the worst SHIP cell.
///
/// Applied to ALL LOW-COLOUR firing (`m3 < 24`) — BOTH the W44-108 sub-band
/// AND the d ≥ 3.5 main band. The original sub-band-only fix (commit
/// 3564728f) left the main band unguarded, which re-incurred the misfire at
/// d ≥ 3.5 (beard-oil d3.5 = 86826 B, +136 % vs cjxl) and created a NEW
/// monotonicity cliff (d3.0 = 44420 → d3.5 = 86826) once the sub-band was
/// fixed. Extending the exclude to the main band closes both. The extend is
/// SCOPED to low-colour: HIGH-colour main-band cells (windows95 m3 = 27.2,
/// imessage m3 = 67, codec_wiki m3 = 145) short-circuit past the p25 check
/// (`!w44_108_low_colour`), so their calibrated main-band behaviour is
/// byte-identical. Every low-colour SHIP screenshot has p25 ≥ 98.9, so those
/// are byte-identical too — only the low-colour flat-bg photos (p25 < 95)
/// lose the lift on the main band. The residual high-colour main-band
/// misfire (windows95 at d ≥ 3.5, cjxl 0.97× = no over-allocation) is a
/// SEPARATE, still-open issue (LIBJXL_DIVERGENCES "needs a new proxy").
pub const BUTTLOOP_QF_SEED_SCALE_SUB_BAND_MIN_P25: f32 = 95.0;

/// W44-176: terminal-class exclude sub-discriminator — `luma_var` lower
/// bound. Inside the W44-108 firing class (`is_screenshot AND m3 < 30`),
/// this gate excludes terminal-like images where the W44-109 lift is
/// net-negative pareto (terminal e7 d=4: bytes +28% AND SSIM2 -1.94 vs
/// cjxl; the lift buys +2.70 SSIM2 over baseline but cjxl is still 1.94
/// above the lifted result — quality NOT competitive with cjxl, bytes
/// substantially over).
///
/// **Why this exists** (W44-174 diagnosis + W44-176 probe): the W44-108
/// firing class contains a mix of pareto wins (graph d=5 +12 SSIM2,
/// imac_dark d=5 +9, imac_g3 d=5 +6, gmessages d=5 +2.4, gui d=5 +4.5)
/// and one regression (terminal d=4-5). Distance-narrow [2.0, 3.0]
/// would sacrifice EVERY win in d=4-5 to fix the one regression. A
/// per-image discriminator targeting just terminal is preferable.
///
/// **Discriminator** (W44-176 probe `examples/w44_176_terminal_proxy_probe.rs`,
/// 17-image corpus = 8 gb82-sc + 6 CID22 + 3 borderline):
/// `luma_var ∈ [1500, 2200] AND fcbr > 0.70`. Fires ONLY on terminal
/// across the entire probe corpus:
///
/// | image       | luma_var | fcbr   | in_band? |
/// |---          |---       |---     |---       |
/// | terminal    | 1706     | 0.833  | **YES**  |
/// | graph       |  415     | 0.809  | below    |
/// | imac_g3     | 5244     | 0.775  | above    |
/// | imac_dark   | 3303     | 0.728  | above    |
/// | gmessages   | 1046     | 0.899  | below    |
/// | gui         | 1051     | 0.858  | below    |
/// | windows95   | 4478     | 0.360  | above + fcbr |
/// | codec_wiki  | 1374     | 0.904  | below    |
/// | imessage    | 2774     | 0.864  | above    |
/// | windows     | 3434     | 0.769  | above    |
/// | 1418519     | 1620     | 0.098  | in band but fcbr=0.098 |
/// | 1025469     | 2468     | 0.017  | above + fcbr |
/// | 1531677     | 2068     | 0.000  | in band but fcbr=0.000 |
/// | 1189261     | 3087     | 0.003  | above + fcbr |
/// | 1420710     | 2171     | 0.000  | in band but fcbr=0.000 |
/// | 2389166     | 1920     | 0.134  | in band but fcbr=0.134 |
///
/// **Safety margins** (all ≥10% per task spec):
/// - `luma_var` lower bound 1500: terminal 1706 = +13.7% above. Nearest
///   in-class excluded: gui 1051 (29.9% below). Nearest in-band photo:
///   1418519 luma_var=1620 (excluded by fcbr=0.098 ≪ 0.70).
/// - `luma_var` upper bound 2200: terminal 1706 = -22.4% below. Nearest
///   in-class excluded: imac_dark 3303 (+50% above). Nearest in-band
///   photo: 1420710 luma_var=2171 (excluded by fcbr=0.000 ≪ 0.70).
/// - `fcbr > 0.70`: terminal 0.833 = +19.0% above. Nearest in-band
///   excluded by fcbr: 2389166 photo at 0.134 (-80.9% below). All
///   photos excluded by fcbr (max photo fcbr in corpus = 0.134).
///
/// All margins exceed the 10% acceptance threshold.
pub const W44_176_TERMINAL_CLASS_LUMA_VAR_MIN: f32 = 1500.0;

/// W44-176: terminal-class exclude — `luma_var` upper bound. See
/// [`W44_176_TERMINAL_CLASS_LUMA_VAR_MIN`] for the full discriminator
/// design + probe corpus + safety margins.
pub const W44_176_TERMINAL_CLASS_LUMA_VAR_MAX: f32 = 2200.0;

/// W44-176: terminal-class exclude — `fcbr` (flat_color_block_ratio)
/// lower bound. Excludes photos (max corpus fcbr = 0.134) and screens
/// with high chroma activity (windows95 = 0.360). Keeps terminal-class
/// pure-text screenshots in the firing-via-exclude class. See
/// [`W44_176_TERMINAL_CLASS_LUMA_VAR_MIN`] for the full discriminator
/// design + probe corpus + safety margins.
pub const W44_176_TERMINAL_CLASS_FCBR_MIN: f32 = 0.70;

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
        /* mask1x1_p25 */ None,
        crate::api::AdaptiveQuantQfSeedPolicy::AutoScalePerEffort,
        /* proxies */ None,
        /* terminal_class_exclude */ false,
        /* high_colour_class_exclude */ false,
    )
}

/// W44-176: returns `true` when the [`crate::vardct::encoder::ZenanalyzeProxies`]
/// proxies indicate a terminal-class screenshot — `luma_var` in
/// `[`W44_176_TERMINAL_CLASS_LUMA_VAR_MIN`,
/// [`W44_176_TERMINAL_CLASS_LUMA_VAR_MAX`]] AND `fcbr` ≥
/// [`W44_176_TERMINAL_CLASS_FCBR_MIN`].
///
/// Returns `false` when proxies are absent (non-sRGB-u8 layouts,
/// streaming/animation paths). The W44-176 exclude is a defence-in-depth
/// narrow — it only fires when both proxies are present AND in the
/// terminal band, otherwise the W44-108 sub-gate behaviour is
/// preserved.
#[inline]
pub(crate) fn w44_176_is_terminal_class(
    proxies: Option<&crate::vardct::encoder::ZenanalyzeProxies>,
) -> bool {
    let Some(p) = proxies else {
        return false;
    };
    p.luma_var >= W44_176_TERMINAL_CLASS_LUMA_VAR_MIN
        && p.luma_var <= W44_176_TERMINAL_CLASS_LUMA_VAR_MAX
        && p.flat_color_block_ratio >= W44_176_TERMINAL_CLASS_FCBR_MIN
}

/// W44-129 Chunk C variant of [`resolved_adaptive_quant_qf_seed_scale`]
/// that consults the resolved [`crate::api::AdaptiveQuantQfSeedPolicy`]
/// enum from `ResolvedImprovements`.
///
/// The legacy unpolicy'd entry point delegates to this with
/// `AutoScalePerEffort` to preserve the pre-Chunk-C behaviour for any
/// remaining callers (tests / examples) that don't carry a resolved
/// policy.
///
/// W44-176 (2026-05-21): if `terminal_class_exclude` is `true` AND the
/// `proxies` indicate a terminal-class screenshot (per
/// [`w44_176_is_terminal_class`]), the gate is bypassed (returns 1.0)
/// even when the W44-108 m3 sub-gate would otherwise fire. Excludes
/// terminal.png e7 d=4-5 net-negative pareto without disabling the
/// gate for graph/imac_g3/imac_dark/gmessages/gui (which buy real
/// SSIM2 with the +28-50% bytes overhead per W44-174 measurement).
#[cfg_attr(not(feature = "std"), allow(unused_variables))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolved_adaptive_quant_qf_seed_scale_with_policy(
    effort: u8,
    butteraugli_iters: u32,
    is_screenshot: bool,
    target_distance: f32,
    m3_colourfulness: Option<f32>,
    mask1x1_p25: Option<f32>,
    policy: crate::api::AdaptiveQuantQfSeedPolicy,
    proxies: Option<&crate::vardct::encoder::ZenanalyzeProxies>,
    terminal_class_exclude: bool,
    high_colour_class_exclude: bool,
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
    // W44-213: route the W44-107 min-distance threshold through the
    // tuning-override macro so sweep-runner builds can swap it at runtime.
    let min_distance = crate::runtime_or_default!(
        BUTTLOOP_QF_SEED_SCALE_MIN_DISTANCE,
        buttloop_qf_seed_scale_min_distance,
    );
    // Task #12 (#74): LOW-COLOUR screenshots (m3 < 24) require a saturated
    // `mask1x1_p25` (≥ 25 % perfectly-flat blocks — genuine flat-UI content)
    // on BOTH the W44-108 sub-band AND the d ≥ 3.5 main band. Pure-white-
    // background low-colour PRODUCT PHOTOS (beard-oil m3 = 23.4, is_screenshot
    // median = 100) leak the m3 < 24 gate but have photographic p25, so
    // without this they take the lift and lose +112 % (sub-band d ∈ [2, 3.5))
    // / +136 % (main band d ≥ 3.5) vs cjxl — a distance-monotonicity
    // violation (file GROWS with distance). The exclude is SCOPED to
    // low-colour: HIGH-colour main-band SHIP cells (windows95 m3 = 27.2,
    // imessage m3 = 67) short-circuit to `true` (untouched), and every
    // low-colour SHIP screenshot has p25 ≥ 98.9 (imac_dark worst) so all
    // stay byte-identical. `None` (the legacy wrapper / callers that don't
    // thread p25) maps to `true` → behaviour unchanged there. See
    // [`BUTTLOOP_QF_SEED_SCALE_SUB_BAND_MIN_P25`].
    let low_colour_p25_ok = !w44_108_low_colour
        || mask1x1_p25.is_none_or(|p25| p25 >= BUTTLOOP_QF_SEED_SCALE_SUB_BAND_MIN_P25);
    let gate_fires = low_colour_p25_ok
        && (target_distance >= min_distance
            || (w44_108_low_colour && target_distance >= BUTTLOOP_QF_SEED_SCALE_SUB_MIN_DISTANCE));
    if !gate_fires {
        return 1.0;
    }
    // W44-176: terminal-class exclude — suppress the W44-109 lift on
    // terminal-class screenshots where the lift over-allocates bytes
    // without catching cjxl's SSIM2 (terminal e7 d=4: lift buys +2.70
    // SSIM2 from 81.40→84.10 but cjxl is at 87.62 — STILL 3.52 above
    // the lifted result, AND bytes are +28% vs cjxl). graph/imac_g3/
    // imac_dark/gmessages/gui benefit from the lift (it carries them
    // ABOVE cjxl SSIM2 — true wins) and are preserved.
    //
    // Env hook for A/B: `JXL_W44_176_DISABLE=1` forces the exclude OFF.
    let exclude_env =
        std::env::var_os("JXL_W44_176_DISABLE").is_some_and(|v| v != "0" && !v.is_empty());
    if terminal_class_exclude && !exclude_env && w44_176_is_terminal_class(proxies) {
        return 1.0;
    }
    // W44-AUDIT-6 Phase 1+3: high-colour-class exclude
    let audit_6_env =
        std::env::var_os("JXL_W44_AUDIT_6_DISABLE").is_some_and(|v| v != "0" && !v.is_empty());
    if high_colour_class_exclude && !audit_6_env && w44_audit_6_is_high_colour_class(proxies) {
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
            // W44-213: tuning-override-aware per-effort scale lookups.
            if effort >= 7 {
                crate::runtime_or_default!(
                    DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7,
                    adaptive_quant_screenshot_qf_seed_scale_e7,
                )
            } else {
                crate::runtime_or_default!(
                    DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6,
                    adaptive_quant_screenshot_qf_seed_scale_e5_e6,
                )
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
