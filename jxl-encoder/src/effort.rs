// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Centralized effort-derived encoder decisions.
//!
//! Every effort-gated decision in the encoder reads from an [`EffortProfile`]
//! instead of checking `if effort >= N` inline. Construct once from
//! `(effort, mode)`, then pass to all subsystems.

use crate::api::EncoderMode;
use crate::entropy_coding::ans::ANSHistogramStrategy;
use crate::entropy_coding::lz77::Lz77Method;

/// Per-strategy raw entropy multipliers for the AC strategy cost model.
///
/// These control the relative preference for each transform type in AC strategy
/// selection. Higher values penalize a strategy (making it less likely to be chosen);
/// lower values favor it. The 8x8-class values are normalized by DCT8's value before
/// use, so DCT8 always evaluates at 1.0. Larger transforms use raw values directly.
///
/// Default values match libjxl `enc_ac_strategy.cc:584` (`kTransforms8x8[i].entropy_mul`).
/// Experimental values from libjxl PR #4506 (Jon Sneyers, VarDCT cost tuning).
///
/// `#[non_exhaustive]` so future libjxl-side strategy additions can land
/// without a breaking change. Construct via [`Self::reference`] or
/// [`Self::experimental`] and mutate fields as needed.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EntropyMulTable {
    /// DCT8 base value. All 8x8-class transforms are normalized by this.
    /// Reference: 0.8 (libjxl `enc_ac_strategy.cc:357`, `kTransforms8x8[0].entropy_mul`).
    pub dct8: f32,

    /// DCT4x4 (four 4x4 sub-blocks per 8x8 block).
    /// Reference: 1.08. Experimental: 0.88 (PR #4506, ~19% reduction).
    /// Lowering favors DCT4x4 for textured/detailed regions (screenshots, text).
    pub dct4x4: f32,

    /// DCT4x8 / DCT8x4 (half-block transforms for edges/detail).
    /// Reference: 0.859316 (libjxl `enc_ac_strategy.cc`).
    pub dct4x8: f32,

    /// Identity (pixel copy, no transform).
    /// Reference: 1.0428. Experimental: 0.88 (PR #4506, ~16% reduction).
    /// Lowering favors identity blocks for flat/noisy regions.
    pub identity: f32,

    /// DCT2x2 (2x2 Hadamard-like transform).
    /// Reference: 0.95 (libjxl `enc_ac_strategy.cc`).
    pub dct2x2: f32,

    /// AFV (Adaptive Frequency Variable, corner DCT).
    /// Reference: 0.818. Experimental: 0.75 (PR #4506, ~8% reduction).
    /// Lowering favors AFV for edge blocks with mixed content.
    pub afv: f32,

    /// DCT16x8 / DCT8x16 (larger transforms use raw values, not normalized by DCT8).
    /// Reference: 1.21 (libjxl `enc_ac_strategy.cc`).
    pub dct16x8: f32,

    /// DCT16x16.
    /// Reference: 1.34 (libjxl `enc_ac_strategy.cc`).
    pub dct16x16: f32,

    /// DCT16x32 / DCT32x16.
    /// Reference: 1.49 (libjxl `enc_ac_strategy.cc`).
    pub dct16x32: f32,

    /// DCT32x32.
    /// Reference: 1.48 (libjxl `enc_ac_strategy.cc`).
    pub dct32x32: f32,

    /// DCT64x32 / DCT32x64.
    /// Reference: 2.25 (libjxl `enc_ac_strategy.cc`).
    pub dct64x32: f32,

    /// DCT64x64.
    /// Reference: 2.25 (libjxl `enc_ac_strategy.cc`).
    pub dct64x64: f32,
}

impl EntropyMulTable {
    /// Default values matching libjxl `enc_ac_strategy.cc:584`.
    pub fn reference() -> Self {
        Self {
            dct8: 0.8,
            dct4x4: 1.08,
            dct4x8: 0.859_316_37,
            identity: 1.0428,
            dct2x2: 0.95,
            afv: 0.817_794_9,
            dct16x8: 1.21,
            dct16x16: 1.34,
            dct16x32: 1.49,
            dct32x32: 1.48,
            dct64x32: 2.25,
            dct64x64: 2.25,
        }
    }

    /// Experimental values from libjxl PR #4506 (Jon Sneyers, VarDCT cost tuning).
    ///
    /// Changes vs reference:
    /// - dct4x4: 1.08 → 0.88 (~19% reduction) — favor detail-preserving 4x4 sub-blocks
    /// - identity: 1.0428 → 0.88 (~16% reduction) — favor pixel-copy for flat regions
    /// - afv: 0.818 → 0.75 (~8% reduction) — favor corner DCT for edge blocks
    pub fn experimental() -> Self {
        Self {
            dct4x4: 0.88,
            identity: 0.88,
            afv: 0.75,
            ..Self::reference()
        }
    }

    /// Screen-content-tuned values that **lift** entropy_mul on the 8x8-class
    /// transforms that tend to over-pick on flat / glyph / UI content,
    /// suppressing block-strategy churn around sharp edges.
    ///
    /// Mirrors the lifted values bisected on the GPU encoder (sibling
    /// `jxl-encoder-gpu/src/lossy_encoder.rs:1535-1539` per the
    /// 2026-05-15 dropped-optimizations log, item #3) for screenshot /
    /// terminal / UI content. The discriminator is
    /// `median(mask1x1) > 95` (high mask values → uniform / flat regions
    /// → screen-content). On photo content the GPU encoder leaves
    /// these at libjxl reference values; on screenshots it lifts them
    /// to suppress IDENTITY/DCT2x2/AFV over-selection that produces
    /// visible artifacts around sharp text glyph edges.
    ///
    /// Changes vs reference (all lifts, never reductions):
    /// - identity: 1.0428 → 1.85 (~77% lift, the dominant wedge)
    /// - dct2x2: 0.95 → 1.15 (~21% lift, mirrors GPU path-flip threshold)
    /// - afv: 0.818 → 0.95 (~16% lift, suppresses corner-DCT churn)
    /// - dct4x8: 0.859316 → 0.98 (~14% lift, suppresses path-flip below 0.95)
    /// - dct4x4: 1.08 (unchanged — bisected to libjxl reference)
    /// - all larger transforms (dct16x8 … dct64x64): unchanged
    ///
    /// Currently used only when [`crate::api::LossyConfig::with_content_aware_entropy_mul`]
    /// is opted in AND the per-image content discriminator routes
    /// the encode into the screenshot class. Default-off until a
    /// wider sweep (chunk-2) validates a default-on flip.
    pub fn screenshot_suppressed() -> Self {
        Self {
            identity: 1.85,
            dct2x2: 1.15,
            afv: 0.95,
            dct4x8: 0.98,
            ..Self::reference()
        }
    }

    /// Smooth-photo-tuned values that **lower** entropy_mul on the large
    /// (16×16 and 32×32) DCT transforms at high distance (d ≥ 4) on smooth
    /// photo content. Makes large transforms relatively cheaper than DCT8 →
    /// reduces DCT8 over-selection on flat regions → reduces the
    /// AdjustQuantBlockAC D-heuristic firing rate that drives the F-D
    /// residual-photo byte gap vs cjxl at d ≥ 4 (W44-27 audit, W44-28
    /// bisection).
    ///
    /// **Source**: W44-28 sweep top winner (dct16=1.27, dct32=1.20) closed
    /// -7.65 % bytes on the F-D residual cells but PATH-FLIPPED imac_g3
    /// e=7 d=4 (+36 % butteraugli) — too aggressive for global use. The
    /// content-aware gate restricts the lowering to smooth-photo content
    /// only, where the path flip cannot occur (imac_g3 has high mask1x1
    /// median, so the smooth-content discriminator suppresses the swap).
    ///
    /// W44-95 attempted to widen `dct32` further to 1.27 (variant W) or
    /// 1.20 (variant Z) — both closed 5-6 of 13 OPEN F-D cells on the
    /// W44-94 narrow population (1420710, 1531677, 1189261, 1418519)
    /// with worst SSIM2 -0.17 to -0.18, but a wider 3-photo spot-check
    /// (2389166 mask=46.24, 3637739 mask=47.80, 1044329 mask=48.03 —
    /// other mask<50 photos in CID22 where the W44-29 gate also fires)
    /// found:
    /// - 2-3 FIXED → OPEN flips on 3637739 e5/e7
    /// - -0.35 to -0.82 SSIM2 drops on 2389166 e5/e6 (exceeds ≤0.3 budget)
    ///
    /// Honest-stopped at the current W44-29 values pending a per-image
    /// discriminator (W44-96 candidate: 2389166 differs from 1420710 in
    /// zenanalyze features even though both are mask < 50).
    ///
    /// Changes vs reference (all reductions, never lifts):
    /// - dct16x16: 1.34 → 1.27 (~5.2 % cheaper)
    /// - dct32x32: 1.48 → 1.34 (~9.5 % cheaper)
    /// - dct16x32 / dct32x16: 1.49 → 1.35 (~9.4 % cheaper, scaled with
    ///   dct32x32 by the libjxl 1.49/1.48 ratio)
    /// - all other transforms: unchanged
    ///
    /// **Gate**: only applied when (a) `distance >= HIGH_D_PHOTO_MIN_DISTANCE`
    /// (3.0, W44-78) AND (b) `median(mask1x1) < HIGH_D_PHOTO_SMOOTH_THRESHOLD`
    /// (50.0, smooth-content discriminator — high mask1x1 medians indicate
    /// screenshot/text content where the W22-1 `screenshot_suppressed`
    /// lift fires instead). Caller can override via
    /// [`crate::api::LossyConfig::with_high_d_photo_hint`].
    pub fn high_d_photo_smooth_suppressed() -> Self {
        Self {
            dct16x16: 1.27,
            dct32x32: 1.34,
            // Scale dct16x32 with dct32x32 by the libjxl 1.49/1.48 ratio
            // (mirrors the W44-28 sweep harness `build_table` helper).
            dct16x32: 1.34 * (1.49 / 1.48),
            ..Self::reference()
        }
    }

    /// W44-96 **variant Z** lift table for the *high-edge, low-flat-color*
    /// sub-class of [`high_d_photo_smooth_suppressed`].
    ///
    /// Built from the W44-95 honest-stop measurement: variant Z
    /// (dct32x32=1.20 instead of 1.34) closes 5-6 of 13 OPEN F-D cells on
    /// {1420710, 1531677} at d ∈ {5, 6} but regresses {2389166, 1044329}
    /// SSIM2 by -0.30 to -0.82 (exceeds the ≤0.3 budget). The W44-96
    /// proxy probe identified a clean per-image discriminator:
    /// `edge_density > 0.7 AND flat_color_block_ratio < 0.01` admits
    /// {1420710, 1531677} and rejects {2389166, 1044329, 7062219}
    /// cleanly across every measured proxy value (see
    /// `benchmarks/w44_96_*.tsv` for the per-image probe).
    ///
    /// Changes vs [`high_d_photo_smooth_suppressed`] (all reductions):
    /// - dct32x32: 1.34 → 1.22 (~9.0 % cheaper) — **W44-154 micro-raise**
    ///   from W44-148's 1.24 after W44-153 full-ledger refresh found 6
    ///   cells flipped FIXED→OPEN at the 1.24 boundary (1420710 e5 d=6;
    ///   1531677 e5/e6 d=6 + e7/e8/e9 d=5), all SSIM2 wins but with
    ///   bytes crossing the +3% wedge. W44-154 micro-bisected {1.22,
    ///   1.23} between W44-148's 1.24 and pre-W44-148 1.20; 1.22 closed
    ///   5 of 6 newly-flipped cells while preserving 100% of the W44-148
    ///   wins on 1418519 d=5 (gate doesn't fire there) and 100% of the
    ///   W44-152 codec_wiki d=3 collateral wins. C (1.23) only closed
    ///   3 of 6. B (1.22) ships. The original W44-148 raise from 1.20
    ///   (W44-96 measurement) was driven by broader d=5/6 measurement on
    ///   the W44-147 audit's photo deficit cluster — 1.20 over-fired
    ///   DCT32X32 by SSIM2 -0.14 to -0.49 on 1420710/1531677 d=5/6 vs
    ///   cjxl. 1.22 is the Pareto-optimal middle: PROTECT_1420710_D5
    ///   SSIM2 +0.099 mean (NET WIN), worst-cell SSIM2 -0.072 (within
    ///   the 0.30 budget), zero CONTROL_NOGATE violations, 1 PROTECT
    ///   cell (1420710 e7 d=5) crosses the +3% bytes wedge but its bfly
    ///   stays at +16.81% identical to W44-148 — documented pareto trade
    ///   matching the W44-153 / W44-148 pattern.
    /// - dct16x32 / dct32x16: scaled by 1.49/1.48 (= 1.228)
    /// - dct16x16: unchanged at 1.27
    ///
    /// **Gate**: only applied when ALL hold (per
    /// `vardct::encoder::compute_ac_strategy`):
    ///   1. The existing W44-29 lift fires (`w44_29_lower=true`).
    ///   2. `distance >= W44_96_VARIANT_Z_MIN_DISTANCE` (4.5 — covers
    ///      the W44-95 measured wins at d ∈ {5, 6} and excludes
    ///      d=4 / d=3 cells where the variant Z gain is marginal).
    ///   3. `mask1x1_median < HIGH_D_PHOTO_SMOOTH_THRESHOLD` (50 — keeps
    ///      the variant Z lift strictly within the W44-29 sub-band, NOT
    ///      the W44-91 colourful-textured-photo path).
    ///   4. `ZenanalyzeProxies.edge_density >= W44_96_EDGE_DENSITY_MIN`
    ///      (0.7).
    ///   5. `ZenanalyzeProxies.flat_color_block_ratio < W44_96_FCBR_MAX`
    ///      (0.01).
    ///
    /// Of the 5 CID22 photos that currently fire W44-29 (1420710, 1531677,
    /// 2389166, 1044329, 7062219), only {1420710, 1531677} pass this
    /// discriminator; the other 3 stay on the default
    /// [`high_d_photo_smooth_suppressed`] table.
    ///
    /// **Bench**: `benchmarks/w44_96_zenanalyze_dct32_discriminator_2026-05-19.{tsv,meta}`
    /// (original 1.20 measurement); `benchmarks/w44_148_variant_z_dct32_bisect_2026-05-21.{tsv,meta}`
    /// (raised to 1.24 after broader-corpus measurement);
    /// `benchmarks/w44_154_dct32x32_micro_bisect_2026-05-21.{tsv,meta}`
    /// (micro-raised to 1.22 after W44-153 ledger refresh).
    pub fn high_d_photo_smooth_suppressed_z() -> Self {
        Self {
            dct16x16: 1.27,
            dct32x32: 1.22,
            // Scale dct16x32 with dct32x32 by the libjxl 1.49/1.48 ratio
            // (mirrors the W44-28 sweep harness `build_table` helper).
            dct16x32: 1.22 * (1.49 / 1.48),
            ..Self::reference()
        }
    }

    /// W44-98 **variant Z' (Z-high-colour)** — lifts `dct16x32`
    /// independently of `dct32x32` within the W44-96 variant Z gate,
    /// for the *high-colourfulness* sub-class of variant Z.
    ///
    /// **Source**: W44-97 per-strategy AC tokenization dump on the 7
    /// OPEN cells remaining post-W44-96 identified DCT32X16 as the
    /// universal #1 overspender (+10017 Y_delta total, max +2425 on
    /// 1531677 e6 d=5) and DCT16X32 as #2 (+2465). Both share the
    /// `dct16x32` slot in [`EntropyMulTable`] (see
    /// `ac_strategy.rs:713`). Lifting `dct16x32` makes BOTH
    /// rectangular 32-class transforms more expensive relative to
    /// `dct32x32` (square merge) and `dct16x16` (smaller square),
    /// pushing strategy selection toward the cjxl-matching picks.
    ///
    /// The W44-98 A/B sweep (`benchmarks/w44_98_dct16x32_lift_z_bisect_2026-05-19.tsv`,
    /// 4 lift values × 29 cells, paired interleaved A/B) found:
    /// - **1420710** (m3_colourfulness=32.93): tolerates `dct16x32`
    ///   up to **1.30** with SSIM2 deltas in +0.03 to +0.07 (gains),
    ///   closes 3/3 OPEN cells (e5 d5/d6, e7 d5) with byte deltas
    ///   -1.25 to -1.73 pp vs the default variant Z table.
    /// - **1531677** (m3_colourfulness=12.30): regresses SSIM2 by
    ///   -0.34 to -0.93 under ANY `dct16x32` ≥ 1.30 (exceeds the
    ///   ≤0.30 SSIM2 budget). Stays on the default variant Z table.
    ///
    /// This table is the **high-colourfulness** sub-variant: same as
    /// `high_d_photo_smooth_suppressed_z` but with `dct16x32 = 1.30`
    /// instead of `1.208`. Routed in
    /// `vardct::encoder::compute_ac_strategy` by an additional
    /// `m3_colourfulness >= W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN`
    /// (25.0) gate INSIDE the variant Z dispatch — when both
    /// W44-96 fires AND the m3 sub-gate fires, swap to this table
    /// instead of the default variant Z.
    ///
    /// **Gate** (ALL must hold, on top of every W44-96 gate condition):
    ///   * `ZenanalyzeProxies.m3_colourfulness >= W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN`
    ///     (25.0 — sits between 1531677's 12.30 and 1420710's 32.93
    ///     with 1.3× margin on each side).
    ///
    /// Of the 2 CID22 photos that pass the W44-96 variant Z gate
    /// (1420710, 1531677), only **1420710** passes this additional
    /// m3 gate; 1531677 stays on
    /// [`high_d_photo_smooth_suppressed_z`].
    ///
    /// **Bench**: `benchmarks/w44_98_dct16x32_lift_z_bisect_2026-05-19.{tsv,meta}`.
    pub fn high_d_photo_smooth_suppressed_z_high_colour() -> Self {
        Self {
            dct16x16: 1.27,
            // W44-148 raised from 1.20 to 1.24 in parallel with variant Z;
            // **W44-154 micro-raises to 1.22** after the W44-153 ledger
            // refresh found 6 pareto FIXED→OPEN flips at the W44-148
            // boundary. See [`high_d_photo_smooth_suppressed_z`] for the
            // full W44-154 rationale. The HC table mirrors the variant Z
            // dct32x32 value to keep the cost-model gap to DCT32X16 and
            // DCT16X32 consistent across HC and the default Z.
            dct32x32: 1.22,
            // dct16x32 LIFTED to 1.30, breaking the libjxl 1.49/1.48
            // ratio. Makes DCT32X16 / DCT16X32 strictly more expensive
            // than DCT32X32 (square merge wins more often). The W44-148
            // raise touched `dct32x32` only — `dct16x32` stays at 1.30
            // because the W44-98 high-colour bisect was independent of
            // `dct32x32` (it measured the lift ratio dct16x32/dct32x32,
            // not the absolute values). W44-154's micro-step from 1.24
            // to 1.22 leaves the relative gap to dct32x32 even larger:
            // 1.30/1.22 = 1.066 (vs the W44-148-era 1.30/1.24 = 1.048
            // and the W44-98 original 1.30/1.20 = 1.083), so DCT16X32
            // remains strictly more expensive than DCT32X32.
            dct16x32: 1.30,
            ..Self::reference()
        }
    }

    /// W44-99 **variant Z'' (Z-low-colour)** — lifts `dct16x32` modestly
    /// (1.208 → 1.22) within the W44-96 variant Z gate, for the
    /// *low-colourfulness* sub-class of variant Z. Mirror of
    /// [`high_d_photo_smooth_suppressed_z_high_colour`] for images that
    /// fail the W44-98 m3 ≥ 25 escalation gate.
    ///
    /// **Source**: W44-97 per-strategy AC tokenization dump on the 7
    /// OPEN cells remaining post-W44-96 identified DCT32X16 as the
    /// universal #1 overspender (+10017 Y_delta total). W44-98 closed
    /// 3 cells on 1420710 (m3=32.93) by lifting `dct16x32` to 1.30 but
    /// W44-98 measured that 1531677 (m3=12.30) regresses SSIM2 by -0.34
    /// to -0.93 under `dct16x32` ≥ 1.30. The W44-99 A/B sweep
    /// (`benchmarks/w44_99_1531677_d5_attack_2026-05-19.tsv`, 5 lift
    /// values × 29 cells) found a smaller lift (1.22, +1.0% over 1.208)
    /// preserves SSIM2 (worst -0.0100 on the W44-99 target cells; many
    /// gains) while closing 3 of 4 1531677 OPEN cells (e6 d=5, e8 d=5,
    /// e9 d=5; e5 d=5 stays OPEN at +3.09% bytes vs +3.55% baseline,
    /// just over the +3.0% threshold).
    ///
    /// **Why a smaller lift works on low-colour**: low-m3 photos have
    /// less colour variance per block, so DCT32X16 → DCT32X32 strategy
    /// re-selection produces less Y-channel ringing. The 1420710 (high
    /// m3) photo HAS strong colour variance, which tolerates the
    /// stronger 1.30 lift; 1531677 (low m3) does not. Two different
    /// optimal points on the same shared-slot knob.
    ///
    /// Variant ZD bisect data (`benchmarks/w44_98_dct16x32_lift_z_bisect_2026-05-19.tsv`)
    /// already measured `dct16x32=1.25` on 1531677 — closes only 2 of 4
    /// OPEN cells. The W44-99 bisect added 1.22 / 1.27 / 1.28 and found
    /// 1.22 strictly dominates 1.25 (more closes, lower SSIM2 cost) on
    /// this image. 1.27 closes the e8/e9 cells more aggressively but
    /// regresses e5/e6 (non-monotonic at e<8 because no butteraugli loop
    /// can recover SSIM2 there) and exceeds the ≤0.3 SSIM2 budget on
    /// e5/e6.
    ///
    /// Changes vs [`high_d_photo_smooth_suppressed_z`] (single lift):
    /// - dct16x32: kept at 1.23 (W44-100 micro-bisect value, originally
    ///   1.22 in W44-99). After W44-148 raised variant Z's `dct32x32` to
    ///   1.24, variant Z's auto-scaled `dct16x32 = 1.24 * 1.49/1.48 ≈
    ///   1.248` now exceeds LC's 1.23 (the W44-99/100 "LC dct16x32 lifted
    ///   ABOVE Z" semantic is INVERTED by W44-148, but LC's value at 1.23
    ///   measured Pareto-positive in the W44-148 bisect — DEFICIT_LC
    ///   SSIM2 +0.257 mean at the new (1.24, 1.23) configuration).
    /// - all other transforms: unchanged from variant Z
    ///
    /// **Gate** (ALL must hold, on top of every W44-96 gate condition):
    ///   * `ZenanalyzeProxies.m3_colourfulness <
    ///     W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN`
    ///     (< 25.0 — the inverse of the W44-98 escalation gate; mutually
    ///     exclusive with [`high_d_photo_smooth_suppressed_z_high_colour`]).
    ///
    /// Of the 2 CID22 photos that pass the W44-96 variant Z gate
    /// (1420710 m3=32.93, 1531677 m3=12.30), only **1531677** passes
    /// this m3 < 25 gate; 1420710 stays on the high_colour Z' table.
    ///
    /// **Bench**: `benchmarks/w44_99_1531677_d5_attack_2026-05-19.{tsv,meta}`.
    pub fn high_d_photo_smooth_suppressed_z_low_colour() -> Self {
        Self {
            dct16x16: 1.27,
            // W44-148 raised from 1.20 to 1.24 in parallel with variant Z;
            // **W44-154 micro-raises to 1.22** in parallel with the
            // variant Z update. The W44-148 bisect measured DEFICIT_LC
            // (1531677 d=5/6 × e7/e8) SSIM2 +0.167 to +0.439 per cell at
            // 1.24 vs 1.20. The W44-154 micro-bisect (against the new
            // dct16x32=1.23 LC table) found 1.22 closes 4 of 5 1531677
            // OPEN cells from W44-153 (the only 1531677 cell that stays
            // OPEN under B is e5 d=6 which is not LC-class — it's plain
            // variant Z because d=6 fires the same auto-scaled dct16x32
            // path). LC's dct16x32 stays at 1.23.
            dct32x32: 1.22,
            // dct16x32 LIFTED to 1.23 (+0.8% above the post-W44-154
            // variant Z dct16x32 = 1.22 * 1.49/1.48 ≈ 1.228, narrowly
            // above the auto-scaled value but ~5.7% below high_colour
            // Z''s 1.30). W44-100 micro-bisect of {1.22, 1.23, 1.24,
            // 1.25} found 1.23 strictly dominates the alternatives on
            // the last remaining OPEN cell (1531677 e5 d=5). After
            // W44-148 raised dct32x32 to 1.24, Z's auto-scaled
            // dct16x32 = 1.248 > LC's 1.23. After W44-154 (this commit)
            // dropped dct32x32 to 1.22, Z's auto-scaled dct16x32 ≈ 1.228
            // is now narrowly BELOW LC's 1.23 again — the W44-99/100
            // "LC dct16x32 lifted ABOVE Z" semantic is re-established,
            // mirroring the original W44-99 design intent. The cost
            // model is non-monotonic in this region (W44-100 finding);
            // re-bisecting LC's dct16x32 against the new dct32x32=1.22
            // baseline is deferred to W44-155+.
            dct16x32: 1.23,
            ..Self::reference()
        }
    }

    /// W44-156 **variant Z (d-high)** — distance-band sub-variant of the
    /// default variant Z table for `target_distance > W44_156_VARIANT_Z_D_HIGH_THRESHOLD`
    /// (5.5 by default). Mirror of
    /// [`high_d_photo_smooth_suppressed_z_high_colour`] (W44-98) and
    /// [`high_d_photo_smooth_suppressed_z_low_colour`] (W44-99/100) sub-discriminator
    /// pattern, but split on the DISTANCE axis instead of the
    /// `m3_colourfulness` axis.
    ///
    /// **Source**: W44-155 (`6739107c`) per-strategy AC tokenization dump on
    /// 1420710 e5 d=5 vs d=6 showed:
    /// - Our model over-consolidates into DCT32X32 (78.6% of first-blocks at
    ///   d=5 vs cjxl's 51.8%, 76.2% at d=6 vs cjxl's 57.2%).
    /// - We don't shed small blocks at the d=5→d=6 transition (DCT8: cjxl
    ///   39→16, ours 10→2).
    /// - Per-region qac is AT PARITY — pure strategy selection issue.
    /// - Variant Z dct32x32 = 1.22 (W44-154) is right for d=4/d=5 but too
    ///   aggressive at d=6 (it MORE strongly suppresses small blocks, the
    ///   OPPOSITE of what cjxl does at d=6).
    ///
    /// At d > 5.5 we want a WEAKER dct32x32 lift (1.20, pre-W44-148
    /// baseline) to let DCT32X32 win less often, giving more room for
    /// DCT32X16 / DCT16X32 / DCT16X16 / DCT8 picks — closer to cjxl's
    /// strategy distribution at high d.
    ///
    /// **Changes vs [`high_d_photo_smooth_suppressed_z`]**:
    /// - `dct32x32`: 1.22 → **1.20** (weaker DCT32X32 lift at high d)
    /// - `dct16x32`: auto-scaled from `dct32x32` via the libjxl 1.49/1.48
    ///   ratio (so 1.20 * 1.49/1.48 ≈ 1.208)
    /// - all other transforms: unchanged from variant Z
    ///
    /// **Gate** (ALL must hold, on top of every W44-96 gate condition):
    ///   * `target_distance > W44_156_VARIANT_Z_D_HIGH_THRESHOLD` (5.5)
    ///   * Sub-variant Z' (high-colour) and Z'' (low-colour) gates are
    ///     checked FIRST — the d-high split applies only to the PLAIN
    ///     variant Z dispatch (when neither HC nor LC fires).
    ///
    /// **Why this is a distance split, not an m3 split**: the W44-155
    /// diagnosis identified the cell's failure as a d=5→d=6 strategy-shift
    /// problem, not a colourfulness problem. The 1420710 image has m3=32.93
    /// (HIGH colour, fires W44-98 Z'), so this d-high split applies to
    /// variant Z' (HC) too — see
    /// [`high_d_photo_smooth_suppressed_z_high_colour_d_high`] for the HC
    /// d-high mirror. The plain Z d-high table here covers any
    /// hypothetical future image that fires plain variant Z (today, none
    /// of the gated CID22 photos do — 1420710 fires HC and 1531677 fires
    /// LC, but Z' / Z'' d-high mirror the same dct32x32 = 1.20 logic).
    ///
    /// **Bench**: `benchmarks/w44_156_distance_aware_variant_z_2026-05-21.{tsv,meta}`.
    pub fn high_d_photo_smooth_suppressed_z_d_high() -> Self {
        Self {
            dct16x16: 1.27,
            // Pre-W44-148 baseline: weaker DCT32X32 lift at high d
            // (per W44-155 diagnosis — cjxl keeps DCT32X32 flat from d=5 to
            // d=6 and sheds small blocks instead, requiring LESS dct32x32
            // suppression at d=6 than at d=5).
            dct32x32: 1.20,
            // Auto-scaled by the libjxl 1.49/1.48 ratio (matches W44-148-era
            // variant Z scaling).
            dct16x32: 1.20 * (1.49 / 1.48),
            ..Self::reference()
        }
    }

    /// W44-156 **variant Z' (high-colour, d-high)** — distance-band
    /// sub-variant of [`high_d_photo_smooth_suppressed_z_high_colour`] for
    /// `target_distance > W44_156_VARIANT_Z_D_HIGH_THRESHOLD` (5.5).
    ///
    /// **Source**: same W44-155 diagnosis as
    /// [`high_d_photo_smooth_suppressed_z_d_high`]. The HC d-high table
    /// applies when both the W44-98 HC gate fires AND `target_distance >
    /// 5.5`. 1420710 fires HC (m3=32.93) and is the W44-156 target image
    /// (e5 d=6 is the cell to close).
    ///
    /// **Changes vs [`high_d_photo_smooth_suppressed_z_high_colour`]**:
    /// - `dct32x32`: 1.22 → **1.20** (weaker DCT32X32 lift at high d)
    /// - `dct16x32`: unchanged at 1.30 (W44-98 independent lift, mirrors
    ///   the relationship Z' has to Z — the d-high split affects only the
    ///   dct32x32 axis).
    /// - all other transforms: unchanged from Z'
    pub fn high_d_photo_smooth_suppressed_z_high_colour_d_high() -> Self {
        Self {
            dct16x16: 1.27,
            dct32x32: 1.20,
            // dct16x32 stays at 1.30 (W44-98 independent lift, mirrors
            // the parent HC's W44-148-era relationship: dct16x32 ratio
            // to dct32x32 = 1.30/1.20 = 1.083, even larger gap than
            // W44-154-era 1.30/1.22 = 1.066, keeping DCT16X32 strictly
            // more expensive than DCT32X32).
            dct16x32: 1.30,
            ..Self::reference()
        }
    }

    /// W44-156 **variant Z'' (low-colour, d-high)** — distance-band
    /// sub-variant of [`high_d_photo_smooth_suppressed_z_low_colour`] for
    /// `target_distance > W44_156_VARIANT_Z_D_HIGH_THRESHOLD` (5.5).
    ///
    /// **Source**: same W44-155 diagnosis as
    /// [`high_d_photo_smooth_suppressed_z_d_high`]. The LC d-high table
    /// applies when both the W44-99 LC gate fires AND `target_distance >
    /// 5.5`. 1531677 fires LC (m3=12.30); its d=6 cluster sits at
    /// SSIM2 -0.247 post-W44-154 — the d-high split protects this cluster
    /// from over-rotation.
    ///
    /// **Changes vs [`high_d_photo_smooth_suppressed_z_low_colour`]**:
    /// - `dct32x32`: 1.22 → **1.20** (weaker DCT32X32 lift at high d)
    /// - `dct16x32`: unchanged at 1.23 (W44-100 micro-bisect value,
    ///   mirrors the relationship LC has to Z — the d-high split affects
    ///   only the dct32x32 axis).
    /// - all other transforms: unchanged from LC
    pub fn high_d_photo_smooth_suppressed_z_low_colour_d_high() -> Self {
        Self {
            dct16x16: 1.27,
            dct32x32: 1.20,
            // dct16x32 stays at 1.23 (W44-100 micro-bisect). With
            // dct32x32 = 1.20 (this table) the relationship LC vs Z
            // becomes: LC dct16x32 = 1.23 > Z d_high auto-scaled
            // ≈ 1.208, restoring the W44-99/100 "LC ABOVE Z" semantic
            // at high d (mirroring the post-W44-154 state at d<=5.5).
            dct16x32: 1.23,
            ..Self::reference()
        }
    }
}

/// All effort-derived encoder decisions, centralized.
///
/// Replaces scattered `if effort >= N` checks throughout the codebase.
/// Construct once from (effort, mode, encoding path), pass to all subsystems.
///
/// **Field categories**:
/// - **Effort-derived**: changes value across effort levels (e.g., `nb_rcts_to_try`,
///   `tree_max_buckets`, `butteraugli_iters`).
/// - **Tuning constants**: same value at every effort in the reference profile,
///   mode-dependent in experimental (e.g., `k_favor_2x2`, `k_info_loss_mul_base`,
///   `entropy_mul_table`, `k8x8` etc.). The picker can dial these independently
///   of effort.
///
/// `#[non_exhaustive]` so we can grow the field set as the picker discovers new
/// useful knobs without breaking external `EffortProfile { ... }` constructions.
/// Construct via [`Self::lossy`] or [`Self::lossless`] and mutate fields as needed.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EffortProfile {
    /// The raw effort level (1–12).
    ///
    /// e10/e11/e12 extends libjxl's kTortoise=9 ceiling: those levels reuse
    /// the e9 code paths but get extended search budgets on the knobs that
    /// scale (`butteraugli_iters`, `tree_learn_seeds`, `lossy_search_seeds`).
    /// e12 doubles `butteraugli_iters` from 16 → 32 (RFC#45 chunk 2) and
    /// requires the `ITER_MAX = 32` cap bump in `validation.rs`.
    pub effort: u8,

    // ─── Feature flags ───────────────────────────────────────────────────
    /// Use ANS entropy coding instead of Huffman.
    pub use_ans: bool,
    /// Use two-pass mode with optimized entropy codes.
    pub optimize_codes: bool,
    /// Use custom coefficient ordering (AC scan order from statistics).
    pub custom_orders: bool,
    /// Enable gaborish inverse pre-filter.
    pub gaborish: bool,
    /// Enable pixel-domain loss in AC strategy selection.
    pub pixel_domain_loss: bool,
    /// Enable error diffusion in AC quantization.
    pub error_diffusion: bool,
    /// Enable patches/dictionary detection.
    pub patches: bool,
    /// Enable content-adaptive MA tree learning (modular path).
    pub tree_learning: bool,
    /// Enable LZ77 backward references in entropy coding.
    pub lz77: bool,
    /// LZ77 method when lz77 is enabled.
    pub lz77_method: Lz77Method,
    /// Number of butteraugli quantization loop iterations.
    pub butteraugli_iters: u32,

    // ─── AC strategy search ──────────────────────────────────────────────
    /// Enable adaptive AC strategy selection (multi-block transforms).
    pub ac_strategy_enabled: bool,
    /// Try DCT16x16/DCT16x8/DCT8x16 transforms (multi-block 16x16 merges).
    pub try_dct16: bool,
    /// Try DCT32x32/DCT32x16/DCT16x32 transforms.
    pub try_dct32: bool,
    /// Try DCT64x64/DCT64x32/DCT32x64 transforms.
    pub try_dct64: bool,
    /// Try DCT4x8/DCT8x4/DCT4x4/AFV transforms (effort >= 6 in libjxl).
    pub try_dct4x8_afv: bool,
    /// Enable non-aligned evaluation pass (odd-aligned 16x16 regions).
    pub non_aligned_eval: bool,
    /// Step size for fine-grained AC strategy search on 32x32+ blocks.
    /// `1` evaluates every position (effort 10+, extends past libjxl
    /// kGlacier); `2` every other position (default for effort 1..=9,
    /// matches libjxl `enc_ac_strategy.cc:1046` for `speed_tier >=
    /// kTortoise`). W38-2 wedge #1.1: we previously used step=1 at e9
    /// — 4× more search work than libjxl's reference but consistently
    /// found worse partitions in the wedge audit.
    pub fine_grained_step: u8,

    // ─── VarDCT pipeline options ──────────────────────────────────────────
    /// Apply pixel-level chromacity adjustments (effort >= 7 in libjxl).
    pub chromacity_adjustment: bool,
    /// Use pair-merge clustering for VarDCT entropy codes (effort >= 9 in libjxl).
    /// When false, uses fast k-means-only clustering.
    pub enhanced_clustering_vardct: bool,
    /// Optimize per-histogram HybridUint configs for VarDCT entropy codes.
    /// libjxl uses uint_method=kNone (no optimization, default {4,2,0}) at effort < 9.
    /// The fast optimization picks non-default configs whose signaling overhead
    /// exceeds their coding benefit on VarDCT token distributions.
    pub optimize_uint_configs_vardct: bool,
    /// ANS histogram normalization strategy for VarDCT entropy codes.
    ///
    /// libjxl `enc_ans_params.h:60-75` (HistogramParams ctor):
    /// `tier >= kSquirrel` (effort <= 7 in our scheme) → `ANSHistogramStrategy::kApproximate`;
    /// otherwise → `kPrecise` (the struct default).
    ///
    /// `Precise` tries all 12 shift values to find the lowest-cost ANS distribution
    /// header; `Approximate` tries every other shift (7 values, step 2). For most
    /// distributions the best shift sits on an even index, so Approximate finds a
    /// header within a few bits of Precise while spending less encoder CPU AND
    /// shipping smaller headers on average (W44-42 codec_wiki wedge: precise's
    /// extra search overhead on shift-odd candidates often picks a tighter data
    /// fit at the cost of a larger header — Approximate's coarser grid lands on
    /// the better total-bytes point on this kind of content).
    pub ans_histogram_strategy_vardct: ANSHistogramStrategy,
    /// Compute per-block dynamic EPF sharpness (effort >= 6 in libjxl).
    pub epf_dynamic_sharpness: bool,
    /// Recompute CfL map after initial quantization for better estimates (effort >= 7 in libjxl).
    pub cfl_two_pass: bool,
    /// Use Newton's method (perceptual cost model) for CfL fitting (effort >= 7 in libjxl).
    /// When false, uses fast least-squares fitting (quadratic cost, single-pass).
    pub cfl_newton: bool,
    /// Newton finite-difference epsilon for CfL fitting (default
    /// `false`-path). Default 1.0; libjxl-parity uses 100.0. See
    /// `cfl_newton_libjxl_parity` for the strategy gate.
    pub cfl_newton_eps: f32,
    /// Maximum Newton iterations for CfL fitting (default
    /// `false`-path). Default 10; libjxl-parity uses 20.
    pub cfl_newton_max_iters: usize,
    /// **W44-184**: when `true`, the CfL Newton call sites pass
    /// `libjxl_parity = true` into [`jxl_simd::cfl_find_best_multiplier_newton`],
    /// which ignores `cfl_newton_eps` / `cfl_newton_max_iters` and runs
    /// the libjxl-bit-exact Newton (eps=100, max_iters=20, start `x=0`,
    /// no LS fallback — mirrors libjxl `enc_chroma_from_luma.cc:152-167`).
    /// When `false` (default), the existing eps/iters parameters drive
    /// the loop with `ls_x` warm-start and LS fallback (the W44-183-shipped
    /// behaviour that the downstream cost model is calibrated against).
    ///
    /// Set to `true` by [`crate::effort::EffortProfile::apply_section_c_cfl_newton_libjxl_parity`]
    /// when [`crate::api::ResolvedImprovements::cfl_newton_libjxl_parity`]
    /// is `true` — i.e. only under [`crate::api::EncoderStrategy::Libjxl`].
    /// W44-183 (`286bf5f1`) honest-stop demonstrated that flipping this
    /// at the default path regresses 25/27 photo cells by 0.25-13.02
    /// SSIM2 + 7.82% mean bytes — only safe under the all-libjxl-parity
    /// strategy preset.
    pub cfl_newton_libjxl_parity: bool,

    /// **W44-AUDIT-5 Phase 2 (Mode C)**: when `true`, the CfL Newton call
    /// sites pass `libjxl_math_with_ls_warm_start = true` into
    /// [`jxl_simd::cfl_find_best_multiplier_newton`], which uses libjxl's
    /// Newton math (eps=100, max_iters=20) BUT starts from `ls_x`
    /// (least-squares warm-start) with the existing LS fallback on
    /// non-convergence. `cfl_newton_libjxl_parity == true` takes priority
    /// over this flag inside the SIMD kernel (the two are mutually-exclusive
    /// even though encoded as two booleans).
    ///
    /// **Why this exists**: the W44-AUDIT-5 Phase 1 diagnosis
    /// (`memory/w44_audit_5_phase1_structural_gap_diagnosis_2026-05-24.md`)
    /// identified that on `codec_wiki.png e7 d=4` the `EncoderStrategy::Libjxl`
    /// path is -5.51 SSIM2 vs cjxl despite +12% bytes overhead. The
    /// SSIM2 deficit on screenshots traces to the LS-only refinement
    /// `cfl_newton_libjxl_parity = false` produces (it picks a different
    /// chroma multiplier than libjxl's Newton on high-detail screenshot
    /// content). However the full libjxl-parity flip
    /// (`cfl_newton_libjxl_parity = true`) regressed 25/27 W44-183 photo
    /// cells by 0.25-13.02 SSIM2 — too damaging to ship on Zenjxl.
    ///
    /// Mode C is the in-between: libjxl Newton math (recovers screenshot
    /// SSIM2) but with the LS warm-start (preserves the W44-29..W44-172
    /// cost-model calibration tuned against LS solutions on photos).
    ///
    /// Default `false` everywhere (opt-in only). Libjxl strategy keeps
    /// `libjxl_parity = true` (bit-exact, mutually-exclusive); Zenjxl /
    /// Aggressive / LeanFaster ALSO keep this `false` after the
    /// W44-AUDIT-5 Phase 2 HONEST-STOP — the 3-mode bisect on
    /// codec_wiki e7 d=4 + 2 photo cells measured Mode C byte-identical
    /// to Mode A (the pre-Phase-2 Zenjxl LS-only refinement), because
    /// the i8 CfL multipliers round identically when both Newton paths
    /// start from `ls_x` warm-start on these inputs. The codec_wiki
    /// SSIM2 deficit (-5.51 Mode B vs cjxl) is NOT closed by Mode C on
    /// Zenjxl — the deficit lives on Mode B (Libjxl strategy) which
    /// MUST keep bit-exact parity per the byte-lock invariant. Mode C
    /// ships as opt-in API surface for callers who want to A/B it via
    /// env `JXL_W44_AUDIT_5_FORCE_LS_WARM_START={0,1}` or by setting
    /// the field on `EncoderImprovementsCustom`.
    pub cfl_newton_libjxl_math_with_ls_warm_start: bool,

    /// **W44-AUDIT-5 Phase 3**: when `true`, the encoder routes CfL Pass-1
    /// (and Pass-2 if it fires) through the libjxl-bit-exact `x=0` start
    /// path for **screenshot-class images only** — i.e. when the
    /// per-image [`crate::vardct::encoder::ZenanalyzeProxies`] satisfy
    /// `m3_colourfulness >= W44_AUDIT_6_HIGH_COLOUR_M3_MIN` (= 80.0).
    /// Photo cells and non-sRGB-u8 layouts (where proxies are absent)
    /// stay on the LS warm-start / LS-only path.
    ///
    /// **Why this exists**: the W44-AUDIT-5 Phase 1 + Phase 2 chain
    /// established that:
    /// - The +12pp Libjxl-strategy bytes overhead on `codec_wiki.png e7 d=4`
    ///   AND its -5.51 SSIM2 deficit vs cjxl are caused by the CfL
    ///   warm-start choice (`x=0` start vs `ls_x` warm-start), not the
    ///   Newton math itself.
    /// - Mode C (libjxl-math + ls_x warm-start) was byte-identical to
    ///   Mode A (LS-only refinement, the Zenjxl baseline) on screenshots:
    ///   both refinement paths land at the same `i8` multiplier when
    ///   started from `ls_x`. The deficit lives on the START position.
    ///
    /// Phase 3 is the **route-by-content-class** mechanism: use the
    /// `m3 >= 80` zenanalyze discriminator from W44-AUDIT-6 Phase 1 to
    /// admit only mixed-content screenshots (codec_wiki, etc.) to the
    /// `x=0` start path. Photos keep `ls_x` warm-start so the
    /// W44-29..W44-172 calibration is preserved.
    ///
    /// **Mutual exclusion**: when `cfl_newton_libjxl_parity == true`
    /// (Libjxl strategy), the `x=0` path fires for EVERY tile, so this
    /// field is moot. When `cfl_newton_libjxl_math_with_ls_warm_start ==
    /// true` (Mode C opt-in), the LS warm-start takes priority — Mode C
    /// callers want the warm-start universally.
    ///
    /// **Composition**: Phase 3 only fires when
    /// `cfl_newton_libjxl_parity == false` AND
    /// `cfl_newton_libjxl_math_with_ls_warm_start == false` AND
    /// `cfl_pass1_screenshot_x0_start == true` AND the per-image proxies
    /// match the high-colour-class predicate. The route flip is
    /// equivalent to flipping `libjxl_parity = true` for the single
    /// `compute_cfl_map` / `refine_cfl_map` call only.
    ///
    /// Default `true` on Zenjxl / Aggressive (production-shipped after
    /// the Phase 3 bisect + 36-cell regression validation); `false` on
    /// Libjxl (irrelevant — `libjxl_parity` already on) and LeanFaster
    /// (drops per-image content gates per the standing pattern).
    ///
    /// Env hook: `JXL_W44_AUDIT_5_P3_DISABLE=1` forces OFF at every
    /// dispatch site.
    pub cfl_pass1_screenshot_x0_start: bool,

    /// **W44-197 Candidate B**: enable CfL Pass-2 with LS-only solver at
    /// effort ∈ {5, 6} (matches libjxl `fast=true` dispatch at
    /// `speed_tier >= kWombat`). When `true` AND effort is 5 or 6, the
    /// encoder fires `refine_cfl_map(..., use_newton=false, ...)` AT
    /// THOSE EFFORTS in addition to the existing `cfl_two_pass: effort >=
    /// 7` Newton path.
    ///
    /// Default `false` — Zenjxl / Aggressive / LeanFaster keep the
    /// no-Pass-2-at-e=5/6 baseline that W44-29..W44-172 cost-model
    /// calibration was tuned against (W44-102 measured that adding FULL
    /// Newton Pass-2 at e=5/6 regressed 2 cells beyond the -0.3 SSIM2
    /// budget — the same calibration concern applies to LS-only Pass-2,
    /// possibly with smaller magnitude). Set to `true` by
    /// [`Self::apply_section_c_cfl_newton_libjxl_parity`] when
    /// [`crate::api::ResolvedImprovements::cfl_pass2_ls_at_low_effort`]
    /// is `true` — i.e. only under [`crate::api::EncoderStrategy::Libjxl`].
    ///
    /// W44-189 D12 audit identified this as the MED-HIGH-EV unsalvaged
    /// CfL Pass-2 item (W44-102 measured FULL Newton; LS-only at e=5/6
    /// was NEVER measured). Mirrors libjxl `enc_heuristics.cc:1190-1194`
    /// where the dispatch shape is `if e>=5 { ComputeTile(..., fast = e<=6, ...) }`.
    pub cfl_pass2_ls_at_low_effort: bool,

    /// **W44-AUDIT-9 / SA-G Fix C** (2026-05-25): when `true`, the
    /// encoder substitutes a zero-filled [`crate::vardct::chroma_from_luma::CflMap::zeros`]
    /// for the `cfl_map` argument to `compute_ac_strategy_for_tiles`,
    /// mirroring libjxl `enc_ac_strategy.cc` at `speed_tier > kSquirrel`.
    /// The actual emitted `cfl_map` (Pass-1 / Pass-2 Newton-derived)
    /// is NOT touched — the bitstream cmap stays libjxl-parity per
    /// `cfl_newton_libjxl_parity`. Only the SEARCH consumption is
    /// zeroed, not the EMIT.
    ///
    /// **Why this exists**: SA-G report (`7d383785`) measured that on
    /// `clic_22ea12 e9 d=4 --strategy libjxl` our SIMD Newton
    /// converges to different chroma multipliers than libjxl's scalar
    /// `FindBestMultiplier` on smooth tiles. The wrong cmap_x inflates
    /// the EstimateEntropy decorrelation cost for DCT8 candidates,
    /// flipping the AC strategy search to pick partials. Zeroing the
    /// search-side cmap brings partial first-blocks 2,241 → 2,495
    /// (vs cjxl 2,499 = +0.16% parity) and bytes -0.6%. Fix C is the
    /// independent workaround above the Newton kernel (Fix B);
    /// composes — if Fix B closes the cmap divergence Fix C becomes a
    /// no-op.
    ///
    /// Set to `true` by
    /// [`Self::apply_section_c_cfl_newton_libjxl_parity`] when
    /// [`crate::api::ResolvedImprovements::cfl_zero_for_search`] is
    /// `true` — i.e. only under [`crate::api::EncoderStrategy::Libjxl`]
    /// at the default. Opt-in callers can flip the field on
    /// `EncoderImprovementsCustom` for Zenjxl/Aggressive/LeanFaster
    /// A/B testing.
    pub cfl_zero_for_search: bool,

    // ─── Quantization ────────────────────────────────────────────────────
    /// Use adaptive (content-dependent) quant field via InitialQuantField.
    /// When false (effort < 5), uses flat quant field = 0.79/distance.
    /// Matches libjxl enc_heuristics.cc:1097-1128.
    pub use_adaptive_quant: bool,
    /// Enable per-block AdjustQuantBlockAC (effort >= 5 in libjxl).
    pub adjust_quant_ac: bool,
    /// Numerator for the effort-fixed q parameter used in global_scale computation.
    /// libjxl: 0.39 at effort >= 5, 0.79 at effort < 5.
    /// global_scale = 65536 * (initial_q_numerator / distance) / 5.0
    pub initial_q_numerator: f32,
    /// Fixed quantization thresholds applied per-coefficient on the Y channel
    /// when [`Self::adjust_quant_ac`] is `false`.
    ///
    /// Pipeline stage: VarDCT post-DCT quantization (`vardct/transform.rs`).
    /// The four entries gate progressively higher coefficient bands; values
    /// below the threshold round to zero.
    /// From libjxl `enc_group.cc:358` (`kThresholdMul` constants for low-effort path).
    /// Lowering the entries preserves more high-frequency Y detail at the cost
    /// of bitrate; raising flattens texture. Override when an asset class needs
    /// different texture-vs-bitrate balance than the libjxl defaults give.
    pub fixed_thresholds_y: [f32; 4],
    /// Initial quantization thresholds used when [`Self::adjust_quant_ac`] is
    /// `true` (effort >= 5). Per-block adjustment iterates from these.
    /// From libjxl `enc_group.cc:390`.
    /// Pipeline stage: VarDCT post-DCT quantization, prior to the
    /// `AdjustQuantBlockAC` per-block tweak. Useful as a starting point for
    /// pickers exploring the threshold-vs-rate frontier per content class.
    pub adjust_thresholds: [f32; 4],

    /// **W44-AUDIT-8 Phase 5**: DC precision bit-shift in the bitstream
    /// `extra_dc_precision` field. `1 << extra_dc_precision` scales the
    /// DC quantization inv_factor on the encoder side; the decoder applies
    /// the symmetric `mul = 1.0 / (1 << extra_dc_precision)` dequant
    /// (jxl-rs `frame/modular/mod.rs:1135`, zenjxl-decoder mirror).
    ///
    /// libjxl `enc_cache.cc:232-234`: `nl_dc = (speed_tier < kFalcon)`
    /// → TRUE at effort ≤ 7 → `enc_modular.cc:1580` sets
    /// `extra_dc_precision = 1` and `mul = 2`. At effort ≥ 8 the
    /// butteraugli loop owns DC quantisation refinement and libjxl drops
    /// back to `extra_dc_precision = 0` (1× precision).
    ///
    /// We mirror that gate **on every strategy** (Libjxl + Zenjxl +
    /// Aggressive + LeanFaster) because the W44-AUDIT-8 Phase 4 DC dump
    /// confirmed cjxl emits the 2× DC precision unconditionally at
    /// effort ≤ 7. The bitstream `extra_dc_precision` field is part of
    /// the strict cjxl byte-parity invariant on Libjxl strategy.
    ///
    /// Default `0` keeps every direct field literal (test fixtures,
    /// `lossy_minimum_init()`, etc.) at the pre-Phase-5 baseline.
    /// [`Self::lossy_reference`] / [`Self::lossy_experimental`] set
    /// `1` at effort ≤ 7, `0` at effort ≥ 8.
    pub extra_dc_precision: u8,

    /// **W44-AUDIT-8 Phase 6**: when `true`, applies libjxl's
    /// `QuantizeWP` shape to DC values during the post-transform pass:
    ///
    /// 1. WP-prediction-relative residual coding (`Predictor::Weighted`
    ///    over already-quantized DC).
    /// 2. 0.62 deadzone (residuals with `|svalue| < 0.62` → 0).
    /// 3. Snap-to-even multiple for residuals with `|residual| > 2`.
    ///
    /// Mirrors libjxl `enc_modular.cc::QuantizeWP` (lines 1542-1559),
    /// active in the `nl_dc` branch (lines 1640-1674). The libjxl
    /// `nl_dc = speed_tier < kFalcon` condition fires at effort ≤ 7
    /// (paired with `extra_dc_precision = 1` from Phase 5).
    ///
    /// Applies to every strategy (Libjxl + Zenjxl + Aggressive +
    /// LeanFaster) because cjxl emits this gate unconditionally at
    /// effort ≤ 7 alongside the extra_dc_precision flip; the
    /// QuantizeWP-shape output is part of strict cjxl byte-parity.
    ///
    /// Default `false` keeps every direct field literal (test
    /// fixtures, lossless path) on the pre-Phase-6 plain-round
    /// quantization. [`Self::lossy_reference`] /
    /// [`Self::lossy_experimental`] set `true` at effort ≤ 7,
    /// `false` at effort ≥ 8 (mirroring the existing
    /// `extra_dc_precision` gate).
    pub use_libjxl_wp_dc_quant: bool,

    // ─── Cost model constants ────────────────────────────────────────────
    // All five `k_*` constants below feed `vardct/ac_strategy_search.rs`
    // (the per-8×8 cost evaluator that picks DCT8 vs DCT4x4 vs IDENTITY vs
    // larger merges). Default values come from libjxl's reference encoder
    // and are *the same at every effort level* — they describe the cost
    // model itself, not the search depth. The picker / sweep harness uses
    // them to retune the model per content class without touching effort.
    /// kFavor2X2AtHighQuality weight (-0.4 in libjxl,
    /// `enc_ac_strategy.cc::kFavor2X2AtHighQuality`).
    /// Applied as `k_favor_2x2 * ((5-distance)/5)^2` to IDENTITY/DCT2X2
    /// entropy at distance < 5. More-negative values aggressively favor
    /// pixel-copy / 2×2 blocks at low distances; useful for screenshots /
    /// pixel art where the default photo-tuned bias under-uses IDENTITY.
    pub k_favor_2x2: f32,
    /// Base penalty added to every non-DCT8 strategy's cost
    /// (libjxl `kAvoidEntropyOfTransforms = 0.5`,
    /// `enc_ac_strategy.cc::EvalAcStrategy`). Higher values discourage the
    /// AC strategy search from leaving DCT8; lower values let it spread to
    /// IDENTITY / DCT4x4 / DCT16x16 more freely.
    pub k_avoid_transforms_base: f32,
    /// Base multiplier on the IDCT-domain (pixel-domain) error term in
    /// `EstimateEntropy` (libjxl 1.2, `enc_ac_strategy.cc`).
    /// PR #4506 raised this to 1.3 for the experimental profile — heavier
    /// weight on visible artifacts vs coefficient-domain entropy.
    pub k_info_loss_mul_base: f32,
    /// Base multiplier on the zero-coefficient cost term (libjxl 9.309,
    /// `enc_ac_strategy.cc`). Increasing rewards strategies that leave
    /// many coefficients exactly zero (boosts large-DCT use on smooth
    /// regions). Lowering lets non-zero residuals stay cheaper.
    pub k_zeros_mul_base: f32,
    /// Base delta added inside the cost-model interpolation (libjxl 10.833,
    /// `enc_ac_strategy.cc`). Acts as an "exchange rate" between rate
    /// (entropy proxy) and distortion (info-loss term); rarely retuned
    /// outside picker/sweep work.
    pub k_cost_delta_base: f32,
    /// Quantization-cost constant used when materializing the initial
    /// quant field (libjxl 0.765, `enc_adaptive_quantization.cc`). Read by
    /// `vardct/precomputed.rs` and `vardct/encoder.rs`. Lower values
    /// produce a coarser initial field (less rate, more distortion);
    /// higher refines.
    pub k_ac_quant: f32,

    // ─── Coefficient-domain multiplier constants ─────────────────────────
    // Each tuple is `(mul1, mul2, base)` for the EstimateEntropy /
    // info-loss formula in `vardct/ac_strategy_search.rs`. `mul1` weights
    // the negative log-rate term, `mul2` weights the AC magnitude term,
    // and `base` is added unconditionally. Defaults come from libjxl's
    // `enc_ac_strategy.cc`. Mode-/effort-independent in both reference
    // and experimental — cost-model knobs the picker can dial.
    /// DCT8x8 coefficient-domain multiplier `(mul1, mul2, base)`.
    /// Note: stored values include libjxl's 0.75 factor on `mul1`/`mul2`
    /// (applied at `enc_ac_strategy.cc:790` for 8×8-class transforms).
    pub k8x8: (f32, f32, f32),
    /// DCT16x8 / DCT8x16 coefficient-domain multiplier `(mul1, mul2, base)`.
    /// Larger transforms skip the 0.75 factor and use the libjxl raw values.
    pub k16x8: (f32, f32, f32),
    /// DCT16x16 coefficient-domain multiplier `(mul1, mul2, base)`.
    pub k16x16: (f32, f32, f32),
    /// DCT4x8 / DCT8x4 coefficient-domain multiplier `(mul1, mul2, base)`.
    /// 4×N strategies share the 0.75 factor with 8×8.
    pub k4x8: (f32, f32, f32),
    /// DCT4x4 coefficient-domain multiplier `(mul1, mul2, base)`.
    /// 4×4 strategies share the 0.75 factor with 8×8.
    pub k4x4: (f32, f32, f32),

    // ─── Entropy multiplier table ──────────────────────────────────────────
    /// Per-strategy entropy multipliers for AC strategy cost model.
    /// Controls relative preference for each transform type.
    pub entropy_mul_table: EntropyMulTable,

    // ─── Patch encoding ────────────────────────────────────────────────────
    /// Use tree learning for patch reference frame encoding.
    /// When true AND ref frame is large enough (>= 128×128), enables adaptive
    /// prediction in the modular encoder for patch ref frames.
    /// Reference: false (libjxl uses simple Gradient predictor).
    /// Experimental: true at effort >= 7 (PR #4533 style improvement).
    pub patch_ref_tree_learning: bool,

    // ─── RCT selection ───────────────────────────────────────────────────
    /// Number of Reversible Color Transform variants to evaluate before
    /// committing to one (0 = skip search, use `RctType::GBR_SUBGR`
    /// unconditionally — the calibrated fallback since `99162a2a`; note
    /// the default search often picks GBR_SUBGR too, so `0` can be
    /// byte-identical to the default on some content — see
    /// jxl-encoder#67 / W44-137 before using `0` as an A/B signal).
    ///
    /// Pipeline stage: modular pre-transform, before predictor + tree
    /// learning (`modular/encode.rs::select_best_rct`,
    /// `modular/frame.rs::select_best_rct_at`). Each candidate runs a
    /// cost estimate; the cheapest wins.
    /// Effort interaction: 0 at e<5, 4 at e5, 5 at e6, 7 at e7, 9 at e8,
    /// 19 at e9+ (libjxl `kSquirrel`/`kKitten`/`kTortoise` schedule).
    /// Override when a specific content class (e.g., film stills) has a
    /// known-best RCT and the search is wasted compute, or when sweeping
    /// to discover content-specific defaults.
    pub nb_rcts_to_try: u8,

    /// Caller-supplied RCT colorspace override. When `Some(rct)`,
    /// `select_best_rct(_at)` skips the search and applies the given
    /// RCT directly. Mirrors libjxl's `cparams.colorspace`. Default
    /// `None` (use the per-effort `nb_rcts_to_try` search).
    pub forced_rct: Option<crate::modular::rct::RctType>,

    // ─── WP parameter search ───────────────────────────────────────────────
    /// Number of weighted-predictor parameter sets to try when tuning the
    /// modular WP per channel (0 = use the libjxl default parameters
    /// without searching).
    ///
    /// Pipeline stage: modular predictor selection
    /// (`modular/predictor.rs::find_best_wp_params`, called from
    /// `modular/section.rs`, `modular/frame.rs`, `modular/encode.rs`).
    /// Effort interaction: 0 at e<8, 2 at e8, 5 at e9+. The search is
    /// expensive (each candidate runs a cost estimate over all WP-eligible
    /// channels), which is why libjxl gates it behind `kKitten`/`kTortoise`.
    /// Override to force the search on at lower effort (e.g., when a picker
    /// wants e6-quality bytes with WP-fitted parameters), or off at e9 for
    /// faster sweeps.
    pub wp_num_param_sets: u8,

    // ─── Tree learning parameters ────────────────────────────────────────
    // Read by `modular/tree_learn.rs::TreeLearningParams::from_profile`.
    // These describe the *shape* of the MA tree — wider trees split on
    // more properties / finer buckets, deeper trees use lower thresholds,
    // and the sampling caps trade tree-learning compute for accuracy.
    /// Number of MA-tree decision properties to evaluate per split.
    /// Capped to the order length defined in `modular/tree_learn.rs`
    /// (15 without `group_id`, 16 with).
    /// Effort interaction: 3 at e<=4, 4 at e5, 5 at e6, 7 at e7, 10 at e8,
    /// 16 at e9+. More properties = better trees but quadratic cost in
    /// `LearnTree`. Override to retune the speed/quality knee per content.
    pub tree_num_properties: u8,
    /// Maximum number of quantization buckets per property when building
    /// the histogram for tree splits. Matches libjxl
    /// `enc_modular.cc:556-590` `max_property_values` per speed tier.
    /// Effort interaction: 32 at e<=4, 48 at e5, 64 at e6, 96 at e7,
    /// 128 at e8, 256 at e9+. Higher = finer thresholds at higher learning
    /// cost. Override when a corpus benefits from coarser/finer splits
    /// than the libjxl tier table predicts.
    pub tree_max_buckets: u16,
    /// Base entropy-cost threshold a candidate split must beat to be
    /// accepted (libjxl `75 + 14 * speed_tier` in
    /// `enc_modular.cc::LearnTreeHeuristics`).
    /// Effort interaction: 173 at e<=1 (speed_tier=9), 117 at e5 (5),
    /// 75 at e9+ (1). Lower threshold = more splits = larger tree. Override
    /// to bias the tree shallower (cheaper decode) or deeper (better fit).
    pub tree_threshold_base: f32,
    /// Hard cap on samples drawn for tree learning when set; `0` defers
    /// to [`Self::tree_sample_fraction`].
    /// Read by `modular/tree_learn.rs::sample_count_for_profile`.
    /// Effort interaction: 65,000 at e<=4 (cheap, fixed budget), 0 at e>=5
    /// (let the fraction-based path scale with image size). Override to
    /// fix the tree-learning compute regardless of input pixels.
    pub tree_max_samples_fixed: u32,
    /// Fraction of total pixels to sample for tree learning when
    /// [`Self::tree_max_samples_fixed`] is `0`. Floor of 65,536 samples.
    /// Read by `modular/tree_learn.rs::sample_count_for_profile`.
    /// Effort interaction: 0.15 at e<=4, 0.25 at e5, 0.35 at e6, 0.5 at e7,
    /// 0.55 at e8, 0.65 at e9+ (libjxl PR #4236). Higher fractions improve
    /// tree fit (especially on large images) at proportional cost. Override
    /// to densify sampling on large images at moderate effort, or thin
    /// sampling for fast sweeps at high effort.
    pub tree_sample_fraction: f32,
    /// Use the streaming two-hash cuckoo dedup (libjxl `AddSample` parity,
    /// `enc_ma.cc:602-655`) instead of the default packed-key sort during
    /// tree-sample deduplication.
    ///
    /// Default `false` at every effort. The streaming path **regresses**
    /// end-to-end wall-clock by +3 % to +8 % at e7 on CLIC photos because
    /// `pack_sample_key` random-accesses parallel SoA arrays per sample
    /// (no cache locality) and the sort path exploits spatial coherence
    /// the hash path cannot. Retained as an opt-in for experimentation
    /// toward issue #41 Phase 2 — integrating dedup into the gather pass
    /// itself, where libjxl gets its actual win because keys land once
    /// during ingest.
    pub use_streaming_dedup: bool,
    /// Integrate the two-hash cuckoo dedup into the gather loop itself
    /// (libjxl `AddSample` parity, `enc_ma.cc:711`). This is Phase 2 of
    /// issue #41 — see
    /// [`crate::modular::tree_learn::TreeLearningParams::gather_dedup`].
    ///
    /// Default `false` at every effort: output is **not** byte-identical
    /// to the sort path because gather-time dedup hashes on raw i32
    /// property values rather than post-quantization bucket indices, so
    /// the unique set is a strict superset (the post-`pre_quantize` sort
    /// pass would have collapsed bucket-equivalent rows that gather-time
    /// kept separate). Callers opt in via the `__expert` lossless
    /// override; sweep harnesses re-bake hash-locks when they do.
    pub gather_dedup: bool,
    /// Phase 3 of issue #41 — switch gather-time dedup to the
    /// inline-fingerprint cuckoo table
    /// (`crate::modular::inline_dedup_table::InlineDedupTable`) instead of
    /// Phase 2's [`Self::gather_dedup`] SoA-indexed table.
    ///
    /// Only meaningful when [`Self::gather_dedup`] is also `true`: the
    /// switch happens inside `gather_samples_strided_with_dedup`, where
    /// Phase 2 builds a `GatherDedupTable` and Phase 3 builds an
    /// `InlineDedupTable` instead. Both produce the same unique-set
    /// semantics (strict superset of the post-sort merge); the
    /// post-`pre_quantize` sort pass collapses the difference downstream.
    /// Hash-locks therefore stay the same as Phase 2's locked variant —
    /// the post-sort arbiter remains the final byte-determinant.
    ///
    /// Default `false`. Callers opt in via the `__expert` lossless
    /// override (`LosslessInternalParams::gather_dedup_phase3`).
    ///
    /// The microbench (`benches/dedup_samples_strategies.rs`,
    /// `benchmarks/inline_dedup_microbench_2026-05-17.txt`) shows
    /// +36 %-53 % gather-throughput vs Phase 1 on high-duplication
    /// streams; real-photo gather payoff depends on spatial duplication
    /// ratios and is decided by Chunk 2's end-to-end A/B at e7 / 1.05 MP.
    pub gather_dedup_phase3: bool,

    // ─── Parallel tree-learning tuning ────────────────────────────────────
    // Read by `modular/tree_learn.rs` (gated on
    // `feature = "parallel-tree-learning"`). These control the rayon
    // fan-out shape in the divide-and-conquer subtree builder. The
    // original constants (depth=4, floor=16384, root_threshold=8192) were
    // tuned on e7 trees (~2,425 nodes on a 1024² photo). At e8/e9 the
    // tree is +30%/+118% larger and the per-fork work is heavier, so
    // deeper fanout + lower floor saturate more cores. See chunk-2 of
    // `lossless_e8_e9_cliff_2026-05-16.md`.
    /// Maximum depth of parallel recursion in the borrowed-view subtree
    /// builder. `2^depth` is the upper bound on parallel leaf tasks.
    /// Effort interaction: 4 at effort ≤ 7 (16 leaf tasks), 5 at effort ≥ 8
    /// (32 leaf tasks). Picker may override; raising costs nothing at
    /// small inputs because the floor terminates fanout early.
    pub tree_parallel_max_depth: u32,
    /// Minimum subtree size below which further parallel fork is skipped
    /// and the iterative sequential builder runs instead. Below this
    /// rayon::join + workspace setup exceeds the parallel savings.
    /// Effort interaction: 16384 at effort ≤ 7, 8192 at effort ≥ 8.
    pub tree_parallel_floor: usize,
    /// Minimum total sample count required before attempting the parallel
    /// root split. Below this the sequential loop is faster.
    /// Effort interaction: 8192 at effort ≤ 7, 4096 at effort ≥ 8.
    pub tree_parallel_root_threshold: usize,
    /// Small-image fallback for the parallel-tree-learning path.
    ///
    /// When `true`, the tree-learner bypasses the thread-local
    /// [`SplitWorkspace`] cache (allocating a fresh workspace per
    /// `find_best_split` call instead of routing through the
    /// `RefCell::borrow_mut` indirection). On small inputs the cache
    /// pays its own per-call cost without meaningful amortisation,
    /// matching the +0.85% small-mean regression documented in the
    /// `cb5e202` commit body.
    ///
    /// The parallel root split + recursive borrowed-view fan-out
    /// remain ENABLED in this fallback regime — they are still the
    /// largest single wall-clock win at 8 threads and stripping them
    /// out costs more (~50-80% slowdown on small images observed
    /// during dev of this fix) than the thread-local cache they sit
    /// inside.
    ///
    /// The companion small-image regression from the borrowed-view
    /// zero-clone fork (`fe2d3a27`, +6.2% small-mean) is NOT addressed
    /// by this flag — fixing it requires resurrecting the deleted
    /// owned-clone path (`split_tree_samples_owned` / `split_pq_owned`
    /// / `build_subtree_recursive_parallel`), which is tracked as a
    /// separate follow-up. See
    /// `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/
    ///  rejected_optimizations_conditional_value_2026-05-17.md` #9.
    ///
    /// Bitstream-equivalent: tree topology depends on the samples,
    /// not the workspace identity, so hash-locks stay byte-identical.
    ///
    /// Default `false`. Set automatically to `true` by
    /// [`Self::adapt_small_image_fallback`] when
    /// `pixels < SMALL_IMAGE_PIXEL_THRESHOLD` (1 MP).
    pub tree_parallel_small_image_fallback: bool,

    /// Number of random-seeded tree-learning runs to perform per encode,
    /// keeping the tree whose tokens have the lowest entropy cost.
    ///
    /// libjxl's `FindBestSplit` is greedy ID3 — locally optimal at each
    /// split but sensitive to which pixels were sampled. Running gather→
    /// tree with multiple stride offsets (or RNG seeds in the future)
    /// and picking the cheapest-encoding tree closes part of that gap.
    ///
    /// Effort interaction (set by `Self::tree_learn_seeds_for`):
    /// - effort ≤ 9: `1` (single run — libjxl-equivalent, byte-identical
    ///   to the pre-RFC#45-chunk-2 baseline)
    /// - effort = 10: `2`
    /// - effort = 11: `8` (RFC#45 chunk 5 — expanded from 4 so that
    ///   chunk-3-only perturbations (seeds 0..3) and chunk-4
    ///   perturbations (seeds 4..7) can both contribute candidate trees
    ///   instead of overwriting each other inside a fixed 4-seed budget)
    ///
    /// Bitstream-valid: each seed produces a normal, spec-valid tree;
    /// the picker just chooses among them. Bytes change only when
    /// `seeds > 1` (e10/e11) — no hash-lock churn at e ≤ 9.
    ///
    /// `0` is treated as `1` (defensive). Read by
    /// `modular/tree_learn.rs::select_best_tree_multi_seed`.
    pub tree_learn_seeds: u8,

    /// Number of butteraugli quantization-loop seeds to run in parallel,
    /// then pick the smallest-bytes result among those that meet the
    /// target butteraugli (RFC#45 pick #1 chunk 3 — lossy analog of
    /// [`Self::tree_learn_seeds`]).
    ///
    /// libjxl's `FindBestQuantization` uses a single hard-coded
    /// `kInitMul = 0.6` (`enc_adaptive_quantization.cc:1042`) which
    /// pulls the post-iteration-1 quant field back toward the initial
    /// AC heuristic field. That single starting point is locally
    /// optimal but the optimization surface has multiple basins —
    /// different `kInitMul` values converge to different (qf, scale)
    /// pairs with measurably different output bytes at the same
    /// butteraugli target.
    ///
    /// At `seeds > 1` we run the loop N times with the seed values
    /// from [`crate::vardct::butteraugli_loop::init_mul_seeds`] (the
    /// libjxl default `0.6` is always included as the first seed so
    /// the worst case is no-regression). The picker keeps the seed
    /// with the largest mean(quant_field_float) (proxy for smallest
    /// encoded bytes — coarser quant → fewer non-zero coefficients)
    /// whose final butteraugli score does not exceed
    /// `1.05 * target_distance`. If no seed meets that bound, the
    /// seed with the smallest final butteraugli score wins.
    ///
    /// Effort interaction (set by `Self::lossy_search_seeds_for`):
    /// - effort ≤ 9: `1` (libjxl-equivalent, bit-identical to pre-RFC#45-chunk-3)
    /// - effort = 10: `2`
    /// - effort = 11: `4`
    ///
    /// Bitstream-valid: each seed produces a normal, spec-valid encode;
    /// the picker just chooses among them. Bytes change only when
    /// `seeds > 1` (e10/e11) — no hash-lock churn at e ≤ 9.
    ///
    /// `0` is treated as `1` (defensive). Read by
    /// `vardct/butteraugli_loop.rs::butteraugli_refine_quant_field`.
    pub lossy_search_seeds: u8,

    /// Use **Lloyd-Max iterative clustering** for MA-tree property
    /// bucket boundaries on the three residual-energy proxy properties
    /// (4 = `|N|`, 5 = `|W|`, 15 = `wp_max_error`).
    ///
    /// Spec-legal reinterpretation of EX-J5 (CALIC energy-quantized
    /// context, Golchin & Paliwal 1998). The original proposal adds a
    /// 17th MA-tree property index for an energy bin — JXL hard-codes
    /// `kNumNonrefProperties = 16`, so any new property index would be
    /// interpreted as a (nonexistent) reference-channel property by
    /// decoders. Refining bucket boundaries of the existing energy
    /// proxies preserves the spec, changes only encoder-side candidate
    /// splitvals, and captures the same "give the tree learner better
    /// energy-aware thresholds" intent.
    ///
    /// Read by [`crate::modular::tree_learn::TreeLearningParams::lloyd_max_buckets`]
    /// and applied inside `pre_quantize` only when the property is one
    /// of the energy-correlated three. Other 13 MA-tree properties keep
    /// the cheap sort-quantile path.
    ///
    /// Default `false` at every effort: Lloyd-Max changes encoder
    /// output bytes (different candidate splitvals → different chosen
    /// tree splits), so `true` requires re-baking
    /// `hash_lock_expected.txt`. Sweep harnesses opt in via the
    /// `__expert` lossless override
    /// ([`LosslessInternalParams::lloyd_max_buckets`]).
    pub lloyd_max_buckets: bool,
}

impl EffortProfile {
    /// Create an effort profile for lossy (VarDCT) encoding.
    ///
    /// Accepts effort in `1..=12`. e10/e11/e12 are our extensions beyond
    /// libjxl's kTortoise=9 ceiling: longer search budgets (more butteraugli
    /// iters at e10/e11/e12 — 8/16/32 respectively — and multi-seed tree
    /// learning at e10+ in a follow-on chunk). The bitstream remains 100%
    /// spec-valid — only encoder search effort changes.
    pub fn lossy(effort: u8, mode: EncoderMode) -> Self {
        let effort = effort.clamp(1, 12);
        match mode {
            EncoderMode::Reference => Self::lossy_reference(effort),
            EncoderMode::Experimental => Self::lossy_experimental(effort),
        }
    }

    /// Create an effort profile for lossless (modular) encoding.
    ///
    /// Accepts effort in `1..=12`. e10/e11/e12 reserve future multi-seed
    /// tree learning (chunk 2 of RFC#45 pick #1). Today they fall through
    /// to the e9 (kTortoise) lossless code paths; only the multi-seed
    /// knobs `tree_learn_seeds` consume the extra budget on lossless
    /// (lossy adds `butteraugli_iters` and `lossy_search_seeds`).
    pub fn lossless(effort: u8, mode: EncoderMode) -> Self {
        let effort = effort.clamp(1, 12);
        match mode {
            EncoderMode::Reference => Self::lossless_reference(effort),
            EncoderMode::Experimental => Self::lossless_experimental(effort),
        }
    }

    fn lossy_reference(effort: u8) -> Self {
        let speed_tier = 10u8.saturating_sub(effort);

        Self {
            effort,

            // ── Feature flags ──
            use_ans: effort >= 3,
            optimize_codes: effort >= 3,
            custom_orders: effort >= 4,
            gaborish: effort >= 5,
            pixel_domain_loss: effort >= 5,
            error_diffusion: false, // libjxl accepts param but never uses it
            patches: effort >= 7,
            tree_learning: effort >= 7,
            // libjxl does NOT use LZ77 for VarDCT DC or AC at effort < 9.
            // DC: ForModular() → lz77_method = kNone (modular_mode=false).
            // AC: HistogramParams(kSquirrel, num_ctx) → lz77_method = kNone
            //     (enc_frame.cc overrides since tier > kTortoise).
            // Only kTortoise (effort 9+) enables LZ77 for VarDCT streams.
            lz77: effort >= 9,
            // **Lz77Method::Optimal at e9+ is deliberate** (issue #29).
            // libjxl uses Lz77Method::Rle for ALL VarDCT encodes regardless
            // of tier; we use Optimal because v07 RD analysis shows ~5×
            // size regression on synthetic gradients with RLE
            // (498B → 2,417B on 1024×1024 gradients), bit-identical
            // quality, while photographic content (~98% of inputs) is
            // byte-identical RLE-vs-Optimal.
            //
            // Caveat: Optimal trips a latent bug in jxl-rs's VarDCT AC
            // decoder path (libjxl/jxl-rs#765, our tracker #29). Affected
            // pipelines: anything that round-trips through zenjxl-decoder
            // (which forks jxl-rs unchanged). djxl + jxl-oxide decode
            // these bitstreams cleanly. DO NOT flip the default to RLE
            // to "fix" the decoder — that'd silently degrade gradient
            // encodes 5×. Wait for the upstream jxl-rs fix.
            lz77_method: match effort {
                0..=8 => Lz77Method::Rle,
                _ => Lz77Method::Optimal,
            },
            butteraugli_iters: match effort {
                // libjxl runs FindBestQuantization unconditionally for lossy
                // encoding. Gated at speed_tier <= kKitten (effort >= 8) in libjxl
                // (enc_adaptive_quantization.cc:1282). kDefaultButteraugliIters=2,
                // kMaxButteraugliIters=4 for kTortoise (effort 9+).
                //
                // RFC#45 chunk 1 + chunk 2: e10/e11/e12 extend the budget past
                // libjxl's cap on a power-of-two ladder (8 → 16 → 32). The loop
                // structurally bounds itself at `MAX_QUANT_LOOP_ITERS = ITER_MAX`
                // (see validation.rs:152 — bumped to 32 in chunk 2 to admit e12),
                // so `_ => 32` is the natural saturation point.
                //
                // Per W21-2 chunk-1 acceptance, e11's 16 iters saturate the
                // single-axis loop on most photo cells; e12 doubling to 32 is
                // motivated by (a) per-image variance — a minority of cells
                // continue to refine past iter 16 in the existing convergence
                // log, and (b) keeping the per-effort doubling pattern so the
                // e12 admit gate has a concrete differentiator vs e11 (RFC#45
                // chunk-2 acceptance bench: `benchmarks/effort_12_admit_2026-05-18.tsv`).
                0..=7 => 0,
                8 => 2,
                9 => 4,
                10 => 8,
                11 => 16,
                _ => 32,
            },

            // ── AC strategy search ──
            ac_strategy_enabled: effort >= 5,
            try_dct16: effort >= 5,
            try_dct32: effort >= 5,
            // libjxl gates DCT64 evaluation in
            // `FindBestFirstLevelDivisionForSquare(8, ...)` only on
            // `cparams.decoding_speed_tier < 4` (default 0; see
            // `libjxl/lib/jxl/enc_ac_strategy.cc:948`), NOT on encoding
            // effort. We use `effort >= 7` instead — see W44-93 honest-
            // stop note in CLAUDE.md "Investigation Notes" for the
            // measured photo SSIM2 collateral that widening to e5 caused
            // (the same pattern W44-38 saw at e6). The single-image
            // smart-dispatch in `adapt_to_image_lossy_with_smoothness`
            // already widens to e5 for classified-smooth photos.
            try_dct64: effort >= 7,
            try_dct4x8_afv: effort >= 6,
            non_aligned_eval: effort >= 6,
            // libjxl gates step=1 at `speed_tier < kTortoise` (effort >= 10 on our
            // scale); at speed_tier >= kTortoise (effort 1..=9 on our scale, which
            // maps to libjxl kTortoise/kKitten/.../kLightning) it uses step=2.
            // See libjxl `enc_ac_strategy.cc:1046`:
            //   `size_t step = cparams.speed_tier >= SpeedTier::kTortoise ? 2 : 1;`
            // W38-2 wedge #1.1 found we had this inverted at e9 — we were doing
            // 4× more 32×32 search work than libjxl at the same effort and finding
            // worse partitions (the cost model favours the libjxl-spaced grid).
            // Keep the finer step=1 only at our e10+ where we explicitly extend
            // libjxl past kGlacier.
            fine_grained_step: if effort >= 10 { 1 } else { 2 },

            // ── VarDCT pipeline ──
            chromacity_adjustment: effort >= 7,
            enhanced_clustering_vardct: effort >= 9,
            optimize_uint_configs_vardct: effort >= 9,
            // libjxl `enc_ans_params.h:72-74`: `tier >= kSquirrel`
            // (cjxl effort <= 7) → Approximate; tier < kSquirrel
            // (cjxl effort >= 8) → kPrecise (struct default).
            // Approximate trades a few bits of data-fit for smaller
            // headers — wins on diverse-context streams (W44-43).
            ans_histogram_strategy_vardct: if effort >= 8 {
                ANSHistogramStrategy::Precise
            } else {
                ANSHistogramStrategy::Approximate
            },
            epf_dynamic_sharpness: effort >= 6,
            // W44-102 INVESTIGATED + RULED OUT (2026-05-19):
            // libjxl `enc_heuristics.cc:1190` gates CFL Pass-2 at
            // `speed_tier <= kHare` (effort >= 5), and our `cfl_newton`
            // gate at effort >= 7 already matches the fast/Newton split.
            // The W44-101 audit predicted widening would close bfly wedges
            // at e6 (1420710 d=5, 1025469 d=4, codec_wiki d=0.2,
            // 1418519 d=6) and add 0.3-1.0% byte savings. A 143-cell
            // paired A/B (benchmarks/w44_102_cfl_two_pass_e5_2026-05-19.tsv)
            // measured: zero meaningful bfly improvement on all 4 named
            // wedges (|Δbfly| < 0.04% on each), bytes near-parity overall
            // (±0.1%), and 2 cells exceeding the -0.3 SSIM2 regression gate
            // (1418519 e6 d=1 Δssim2=-0.524; 2389166 e5 d=2 Δssim2=-0.388).
            // The e6→e7 12.8pp byte jump in the W44-100 ledger is driven
            // by `patches` + `tree_learning` + `try_dct64` + clustering
            // activation at e7, not by CFL Pass-2. Gate retained at
            // effort >= 7. Do NOT re-investigate widening this gate.
            cfl_two_pass: effort >= 7,
            cfl_newton: effort >= 7,
            cfl_newton_eps: jxl_simd::NEWTON_EPS_DEFAULT,
            cfl_newton_max_iters: jxl_simd::NEWTON_MAX_ITERS_DEFAULT,
            // W44-184: default `false` — only flipped by
            // `apply_section_c_cfl_newton_libjxl_parity` when
            // `EncoderStrategy::Libjxl` is selected.
            cfl_newton_libjxl_parity: false,
            // W44-AUDIT-5 Phase 2 (Mode C): default `false` — opt-in
            // only after the HONEST-STOP (Mode C measured byte-identical
            // to Mode A on the 3-cell bisect). Field flips ON only when
            // caller explicitly opts in via env hook
            // `JXL_W44_AUDIT_5_FORCE_LS_WARM_START=1` or by setting it
            // on `EncoderImprovementsCustom`. Libjxl strategy keeps it
            // `false` too (parity takes priority in the SIMD kernel).
            cfl_newton_libjxl_math_with_ls_warm_start: false,
            // W44-AUDIT-5 Phase 3: default `false` here in the base
            // `EffortProfile::new`. Flipped to `true` by
            // `apply_section_c_cfl_newton_libjxl_parity` when the
            // resolved Zenjxl/Aggressive bundle carries the field on
            // (after the Phase 3 bisect + 36-cell regression validation
            // pass). Stays `false` under Libjxl (irrelevant — parity
            // already on) and LeanFaster (drops per-image gates).
            cfl_pass1_screenshot_x0_start: false,
            // W44-197: default `false` — only flipped by
            // `apply_section_c_cfl_newton_libjxl_parity` when
            // `EncoderStrategy::Libjxl` is selected. See field doc on
            // `EffortProfile::cfl_pass2_ls_at_low_effort`.
            cfl_pass2_ls_at_low_effort: false,
            // W44-AUDIT-9 / SA-G Fix C: default `false` — only flipped
            // by `apply_section_c_cfl_newton_libjxl_parity` when
            // `EncoderStrategy::Libjxl` is selected (mirrors libjxl
            // `enc_ac_strategy.cc` speed_tier > kSquirrel). Zenjxl /
            // Aggressive / LeanFaster keep this `false` to preserve
            // the W44-29..W44-172 cost-model calibration (default-flip
            // discussion deferred to a follow-on chunk).
            cfl_zero_for_search: false,

            // ── Quantization ──
            use_adaptive_quant: effort >= 5,
            adjust_quant_ac: effort >= 5,
            initial_q_numerator: if effort >= 5 { 0.39 } else { 0.79 },
            fixed_thresholds_y: [0.56, 0.62, 0.62, 0.62],
            adjust_thresholds: [0.58, 0.64, 0.64, 0.64],
            // W44-AUDIT-8 Phase 5: mirror libjxl `nl_dc = speed_tier <
            // kFalcon` (effort ≤ 7 → 2× DC precision; effort ≥ 8 → 1×,
            // butteraugli loop owns DC refinement). Applies to every
            // strategy because cjxl emits this gate unconditionally;
            // the bitstream field is part of strict cjxl byte-parity.
            extra_dc_precision: if effort >= 8 { 0 } else { 1 },
            // W44-AUDIT-8 Phase 6 (HONEST-STOP on default-flip): mirror
            // libjxl `QuantizeWP` shape (WP-relative residual + 0.62
            // deadzone + snap-to-even). Bisect on clic_22ea12 e7 d=4
            // showed Phase 6 cuts bytes by 25% vs Phase 5 (still beats
            // cjxl on SSIM2 by +0.24 vs cjxl) — BUT default-flip breaks
            // the `test_optimize_codes_roundtrip_small` invariant
            // because the static-codes path emits DC tokens via
            // `clamped_gradient` predictor whose residual statistics
            // diverge from the QuantizeWP-shaped quant_dc distribution
            // (static path inflates 16×16 red square from 97 → 1098
            // bytes; decoded pixels diverge by 3e-4 vs static/dynamic).
            //
            // Shipped as OPT-IN only — callers can set
            // `use_libjxl_wp_dc_quant = true` via direct EffortProfile
            // field mutation (no public API surface in this chunk).
            // Future Phase 7 candidates: thread WP predictor through
            // the static-codes DC path, OR per-effort/per-distance
            // dispatch that flips ON only when the gradient-predictor
            // divergence is empirically acceptable.
            use_libjxl_wp_dc_quant: false,

            // ── Cost model constants (from libjxl) ──
            k_favor_2x2: -0.4,
            k_avoid_transforms_base: 0.5,
            k_info_loss_mul_base: 1.2,
            k_zeros_mul_base: 9.308_906,
            k_cost_delta_base: 10.833_273,
            k_ac_quant: 0.765,

            // ── Coefficient-domain multipliers ──
            // Note: k8x8 mul1 has 0.75 factor applied (libjxl enc_ac_strategy.cc:790)
            k8x8: (-0.55 * 0.75, 1.073_575_8 * 0.75, 1.4),
            k16x8: (-0.55, 0.901_958_8, 1.6),
            k16x16: (-0.65, 0.88, 1.8),
            k4x8: (-0.50 * 0.75, 0.88, 1.3),
            k4x4: (-0.45 * 0.75, 0.85, 1.2),

            // ── Entropy multiplier table ──
            entropy_mul_table: EntropyMulTable::reference(),

            // ── Patch encoding ──
            patch_ref_tree_learning: false,

            // ── RCT selection ──
            nb_rcts_to_try: match effort {
                0..=4 => 0,
                5 => 4,
                6 => 5,
                7 => 7,
                8 => 9,
                _ => 19,
            },
            forced_rct: None,

            // ── WP parameter search ──
            wp_num_param_sets: match effort {
                0..=7 => 0,
                8 => 2,
                _ => 5,
            },

            // ── Tree learning ──
            tree_num_properties: Self::tree_num_properties_for(effort),
            tree_max_buckets: Self::tree_max_buckets_for(effort),
            tree_threshold_base: 75.0 + 14.0 * speed_tier as f32,
            tree_max_samples_fixed: if effort <= 4 { 65_000 } else { 0 },
            // Effort-scaled nb_repeats matching libjxl PR #4236
            tree_sample_fraction: Self::tree_sample_fraction_for(effort),
            // Default OFF: streaming dedup regresses end-to-end wall-clock
            // on real photos (issue #41) in our post-gather pipeline.
            use_streaming_dedup: false,
            // Default OFF: gather-time dedup ships bytes that don't match
            // the sort-path hash-locks (raw vs bucket-quantized property
            // hashing — see TreeLearningParams::gather_dedup). Opt-in
            // via the __expert lossless override when sweep harnesses
            // are ready to re-bake hash_lock sidecars.
            gather_dedup: false,
            // Default OFF: Phase 3 inline-fingerprint dedup is opt-in
            // (the post-sort arbiter keeps hash-locks stable, but the
            // gather-time table layout switch is still a perf-only
            // override decided by Chunk 2's end-to-end A/B).
            gather_dedup_phase3: false,

            // Parallel-tree-learning fanout (only used on the lossless
            // path, but set on the lossy profile too for shape parity).
            tree_parallel_max_depth: Self::tree_parallel_max_depth_for(effort),
            tree_parallel_floor: Self::tree_parallel_floor_for(effort),
            tree_parallel_root_threshold: Self::tree_parallel_root_threshold_for(effort),
            // Default false; adapt_to_image() flips this on for <1 MP inputs.
            tree_parallel_small_image_fallback: false,

            // RFC#45 chunk 2: 1 at e ≤ 9 (libjxl-equivalent, byte-identical),
            // 2 at e10, 4 at e11.
            tree_learn_seeds: Self::tree_learn_seeds_for(effort),

            // RFC#45 chunk 3 (lossy multi-seed butteraugli sweep): 1 at e ≤ 9
            // (libjxl-equivalent, bit-identical), 2 at e10, 4 at e11. The
            // butteraugli loop is no-op below e8 (butteraugli_iters = 0) so
            // this field only takes effect at e10/e11.
            lossy_search_seeds: Self::lossy_search_seeds_for(effort),

            // EX-J5 reinterpretation: default OFF on lossy too (lossless
            // is where MA-tree property pre-quantization runs, so the
            // lossy default has no runtime effect — but keep shape parity
            // with the lossless profile struct so both initialisers stay
            // exhaustive and the field is never accidentally left undefined).
            lloyd_max_buckets: false,
        }
    }

    fn lossless_reference(effort: u8) -> Self {
        let speed_tier = 10u8.saturating_sub(effort);

        Self {
            effort,

            // ── Feature flags ──
            use_ans: effort >= 3,
            optimize_codes: effort >= 2,
            custom_orders: effort >= 3,
            gaborish: false,          // N/A for lossless
            pixel_domain_loss: false, // N/A for lossless
            error_diffusion: false,   // N/A for lossless
            patches: effort >= 5,
            tree_learning: effort >= 7,
            lz77: effort >= 7,
            lz77_method: match effort {
                0..=7 => Lz77Method::Rle,
                8 => Lz77Method::Greedy,
                _ => Lz77Method::Optimal,
            },
            butteraugli_iters: 0, // N/A for lossless

            // ── AC strategy (N/A for lossless) ──
            ac_strategy_enabled: false,
            try_dct16: false,
            try_dct32: false,
            try_dct64: false,
            try_dct4x8_afv: false,
            non_aligned_eval: false,
            fine_grained_step: 2,

            // ── VarDCT pipeline (N/A for lossless) ──
            chromacity_adjustment: false,
            enhanced_clustering_vardct: false,
            optimize_uint_configs_vardct: false, // N/A for lossless
            ans_histogram_strategy_vardct: ANSHistogramStrategy::Precise, // N/A for lossless
            epf_dynamic_sharpness: false,
            cfl_two_pass: false,
            cfl_newton: false,
            cfl_newton_eps: jxl_simd::NEWTON_EPS_DEFAULT,
            cfl_newton_max_iters: jxl_simd::NEWTON_MAX_ITERS_DEFAULT,
            // W44-184: default `false` (Lossless mode never runs Newton
            // anyway since `cfl_newton: false`).
            cfl_newton_libjxl_parity: false,
            // W44-AUDIT-5 Phase 2 (Mode C): default `false` for lossless
            // — Newton never fires (cfl_newton: false) so this is moot
            // on this path. Mirrors `cfl_newton_libjxl_parity`'s lossless
            // default for the same reason.
            cfl_newton_libjxl_math_with_ls_warm_start: false,
            // W44-AUDIT-5 Phase 3: lossless never runs Newton (`cfl_newton:
            // false`) so this is moot on this path. Mirrors the
            // `cfl_newton_libjxl_parity` lossless default for the same
            // reason.
            cfl_pass1_screenshot_x0_start: false,
            // W44-197: default `false` — Lossless mode never runs Pass-2
            // anyway since `cfl_two_pass: false`.
            cfl_pass2_ls_at_low_effort: false,
            // W44-AUDIT-9 / SA-G Fix C: default `false` — lossless path
            // never runs the AC strategy search (`ac_strategy_enabled =
            // false` on the lossless modular path), so this is moot here.
            cfl_zero_for_search: false,

            // ── Quantization (N/A for lossless) ──
            use_adaptive_quant: false,
            adjust_quant_ac: false,
            initial_q_numerator: 0.39,
            fixed_thresholds_y: [0.56, 0.62, 0.62, 0.62],
            adjust_thresholds: [0.58, 0.64, 0.64, 0.64],
            // W44-AUDIT-8 Phase 5: lossless path doesn't quantize DC the
            // same way (modular RCT, not VarDCT DC channel), so the field
            // is moot here. Kept at `0` for the lossless default.
            extra_dc_precision: 0,
            // W44-AUDIT-8 Phase 6: lossless path doesn't run the VarDCT
            // DC quantization at all, so the QuantizeWP shape is N/A.
            use_libjxl_wp_dc_quant: false,

            // ── Cost model constants (used for tree learning cost estimates) ──
            k_favor_2x2: -0.4,
            k_avoid_transforms_base: 0.5,
            k_info_loss_mul_base: 1.2,
            k_zeros_mul_base: 9.308_906,
            k_cost_delta_base: 10.833_273,
            k_ac_quant: 0.765,

            // ── Coefficient-domain multipliers (N/A for lossless) ──
            k8x8: (-0.55 * 0.75, 1.073_575_8 * 0.75, 1.4),
            k16x8: (-0.55, 0.901_958_8, 1.6),
            k16x16: (-0.65, 0.88, 1.8),
            k4x8: (-0.50 * 0.75, 0.88, 1.3),
            k4x4: (-0.45 * 0.75, 0.85, 1.2),

            // ── Entropy multiplier table (N/A for lossless, but struct requires it) ──
            entropy_mul_table: EntropyMulTable::reference(),

            // ── Patch encoding ──
            patch_ref_tree_learning: false,

            // ── RCT selection ──
            nb_rcts_to_try: match effort {
                0..=4 => 0,
                5 => 4,
                6 => 5,
                7 => 7,
                8 => 9,
                _ => 19,
            },
            forced_rct: None,

            // ── WP parameter search ──
            wp_num_param_sets: match effort {
                0..=7 => 0,
                8 => 2,
                _ => 5,
            },

            // ── Tree learning ──
            tree_num_properties: Self::tree_num_properties_for(effort),
            tree_max_buckets: Self::tree_max_buckets_for(effort),
            tree_threshold_base: 75.0 + 14.0 * speed_tier as f32,
            tree_max_samples_fixed: if effort <= 4 { 65_000 } else { 0 },
            // Effort-scaled nb_repeats matching libjxl PR #4236
            tree_sample_fraction: Self::tree_sample_fraction_for(effort),
            // Default OFF: streaming dedup regresses end-to-end wall-clock
            // on real photos (issue #41) in our post-gather pipeline.
            use_streaming_dedup: false,
            // Default OFF: gather-time dedup ships bytes that don't match
            // the sort-path hash-locks (raw vs bucket-quantized property
            // hashing — see TreeLearningParams::gather_dedup). Opt-in
            // via the __expert lossless override when sweep harnesses
            // are ready to re-bake hash_lock sidecars.
            gather_dedup: false,
            // Default OFF: Phase 3 inline-fingerprint dedup is opt-in
            // (the post-sort arbiter keeps hash-locks stable, but the
            // gather-time table layout switch is still a perf-only
            // override decided by Chunk 2's end-to-end A/B).
            gather_dedup_phase3: false,

            // Parallel-tree-learning fanout. e8/e9 trees are larger and
            // the per-leaf work is heavier — deeper fanout + lower floor
            // saturate more cores. See chunk-2 of
            // `lossless_e8_e9_cliff_2026-05-16.md`.
            tree_parallel_max_depth: Self::tree_parallel_max_depth_for(effort),
            tree_parallel_floor: Self::tree_parallel_floor_for(effort),
            tree_parallel_root_threshold: Self::tree_parallel_root_threshold_for(effort),
            // Default false; adapt_to_image() flips this on for <1 MP inputs.
            tree_parallel_small_image_fallback: false,

            // RFC#45 chunk 2: 1 at e ≤ 9 (libjxl-equivalent, byte-identical),
            // 2 at e10, 4 at e11.
            tree_learn_seeds: Self::tree_learn_seeds_for(effort),

            // Lossless never runs the butteraugli loop — keep at 1 so the
            // shared `EffortProfile` struct stays well-formed without
            // implying a phantom lossy sweep on lossless encodes.
            lossy_search_seeds: 1,

            // EX-J5 reinterpretation (CALIC energy-quantized context via
            // Lloyd-Max bucket boundaries on properties 4, 5, 15).
            // Default OFF — flipping to true changes encoder output bytes
            // and requires re-baking `hash_lock_expected.txt`. Sweep
            // harnesses opt in via the `__expert` lossless override
            // (LosslessInternalParams::lloyd_max_buckets).
            lloyd_max_buckets: false,
        }
    }

    /// Experimental lossy profile with tuning from libjxl PRs and our own improvements.
    ///
    /// Divergences from reference (documented per-field):
    /// - `k_info_loss_mul_base`: 1.2 → 1.3 (PR #4506, +8% pixel-domain loss weight)
    /// - `entropy_mul_table`: PR #4506 values (favor DCT4x4, Identity, AFV)
    /// - `enhanced_clustering_vardct`: enabled at effort >= 7 (was e9+)
    /// - `patch_ref_tree_learning`: true at effort >= 7 (tree learning for patch ref frames)
    fn lossy_experimental(effort: u8) -> Self {
        let mut p = Self::lossy_reference(effort);

        // PR #4506 (Jon Sneyers): +8% weight on pixel-domain loss improves visual quality
        // on detailed content. The info_loss_mul scales the IDCT-domain error term in
        // EstimateEntropy, making the cost model more sensitive to visible artifacts.
        // Reference: 1.2 (libjxl enc_ac_strategy.cc). Experimental: 1.3.
        p.k_info_loss_mul_base = 1.3;

        // PR #4506 entropy multiplier rebalancing: favor small/detail-preserving transforms.
        p.entropy_mul_table = EntropyMulTable::experimental();

        // Pair-merge histogram clustering helps VarDCT at effort 7+ (not just e9+).
        // The ANS header cost savings from merging similar distributions outweigh the
        // slight data cost increase from sharing code tables across contexts.
        if effort >= 7 {
            p.enhanced_clustering_vardct = true;
        }

        // Tree learning for patch reference frames: adapts prediction to packed glyphs
        // instead of using fixed Gradient predictor. Significant on large ref frames
        // (screenshots with many unique patterns). Gated at effort >= 7.
        if effort >= 7 {
            p.patch_ref_tree_learning = true;
        }

        p
    }

    fn lossless_experimental(effort: u8) -> Self {
        Self::lossless_reference(effort)
    }

    fn tree_num_properties_for(effort: u8) -> u8 {
        match effort {
            0..=4 => 3,
            5 => 4,
            6 => 5,
            7 => 7,
            8 => 10,
            // 16 = all properties including group_id.
            // Non-squeeze array has 15 elements, so .min(15) caps correctly.
            // Squeeze array has 16 elements (group_id always included).
            _ => 16,
        }
    }

    /// Effort-scaled pixel sampling fraction for tree learning (libjxl PR #4236).
    fn tree_sample_fraction_for(effort: u8) -> f32 {
        match effort {
            0..=4 => 0.15,
            5 => 0.25,
            6 => 0.35,
            7 => 0.5,
            8 => 0.55,
            _ => 0.65,
        }
    }

    fn tree_max_buckets_for(effort: u8) -> u16 {
        // Matches libjxl enc_modular.cc:556-590 max_property_values by speed_tier.
        match effort {
            0..=4 => 32, // <=Cheetah
            5 => 48,     // Hare
            6 => 64,     // Wombat
            7 => 96,     // Squirrel
            8 => 128,    // Kitten
            _ => 256,    // Tortoise
        }
    }

    /// Parallel-tree-learning fanout depth by effort.
    ///
    /// e8/e9 trees are 30-118% larger than e7 (`lossless_e8_e9_cliff_2026-05-16.md`),
    /// and each per-leaf subtree-build call is heavier. Doubling the leaf-task
    /// budget at high effort lets rayon saturate idle workers; at low effort
    /// the floor terminates fanout early so the extra budget is harmless.
    fn tree_parallel_max_depth_for(effort: u8) -> u32 {
        if effort >= 8 { 5 } else { 4 }
    }

    /// Subtree-size floor below which parallel fork is skipped.
    fn tree_parallel_floor_for(effort: u8) -> usize {
        if effort >= 8 { 8_192 } else { 16_384 }
    }

    /// Total-sample threshold to attempt the parallel root split.
    fn tree_parallel_root_threshold_for(effort: u8) -> usize {
        if effort >= 8 { 4_096 } else { 8_192 }
    }

    /// Number of multi-seed tree-learning runs by effort (RFC#45 pick #1
    /// chunk 2, extended by chunks 5 + 6). e ≤ 9 keeps the single-pass
    /// libjxl behaviour (byte-identical hash-locks); e10/e11 fan out
    /// 2 / 16 seeded runs and pick the cheapest-encoding tree.
    ///
    /// Chunk 6 raised e11 from 8 → 16 seeds by adding two new variance
    /// dimensions on top of chunks 3-5. The 16-seed layout is:
    /// - seeds 0..=3 chunk-3-only perturbations (split_threshold
    ///   jitter + property-order rotation + stride; chunk-4/5/6 helpers
    ///   hold to canonical no-op).
    /// - seeds 4..=7 chunk-4 dimensions on top of chunk-3
    ///   (sample-fraction override + predictor-evaluation-order
    ///   shuffle).
    /// - seeds 8..=11 chunk-6 split-bucket-count override
    ///   (`max_property_values` ∈ {64, 128, 192, canonical 256},
    ///   chunk-4 helpers held to no-op).
    /// - seeds 12..=15 chunk-6 properties-slice truncation (truncate
    ///   to {8, 10, 12, canonical 14+} leading properties, chunk-4 +
    ///   chunk-6-bucket helpers held to no-op).
    ///
    /// Chunk 5 raised e11 from 4 → 8 seeds so that chunk-3-only
    /// perturbations and chunk-4 dimensions both contribute candidates.
    /// Honest W8-3-r2 benching showed chunk 4 regressed vs chunk 3 at
    /// e11 (+0.39% bytes) because a fixed 4-seed budget cycled through
    /// *different* 4 trees rather than *more*; the 8-seed split fixed
    /// that. The chunk-5 → chunk-6 doubling extends the same pattern:
    /// each new variance dimension gets its own 4-seed slot rather than
    /// being recombined with the others inside a fixed budget.
    fn tree_learn_seeds_for(effort: u8) -> u8 {
        match effort {
            0..=9 => 1,
            10 => 2,
            _ => 16,
        }
    }

    /// Number of butteraugli-loop seeds to run by effort (RFC#45 pick #1
    /// chunk 3). e ≤ 9 keeps the single-seed libjxl behaviour
    /// (bit-identical hash-locks); e10 fans out 2 seeds, e11 fans out 4,
    /// and the picker keeps the smallest-bytes seed that meets target
    /// butteraugli. See [`EffortProfile::lossy_search_seeds`] for the
    /// selection rule and seed values.
    fn lossy_search_seeds_for(effort: u8) -> u8 {
        match effort {
            0..=9 => 1,
            10 => 2,
            _ => 4,
        }
    }

    /// Default for [`crate::api::LossyConfig::auto_splines`] when the
    /// caller hasn't explicitly opted in/out via
    /// [`crate::api::LossyConfig::with_auto_splines`].
    ///
    /// **Currently returns `false` at every effort level.** The
    /// chunk-3 ridge-following detector (commit `f76cbe6`) is guarded
    /// by a trial-encode + measured-energy cost gate
    /// ([`crate::vardct::splines::spline_passes_trial_encode_gate`])
    /// and near-coincident-candidate dedup. Chunk 4
    /// (`benchmarks/auto_splines_bench_2026-05-17_chunk4.tsv`)
    /// recalibrated the `BYTES_PER_ENERGY_UNIT_AT_D1` constant from a
    /// stale `50.0` anchor down to `0.20`, cutting the gate's
    /// false-positive admit count from ~20 splines per screenshot down
    /// to 6-33 on long bright ridges where the bbox-area-linear energy
    /// proxy still over-claims. Chunk 5 layered a content discriminator
    /// ([`crate::vardct::splines::looks_like_screenshot`], threshold
    /// shared with the GPU encoder's W7-3 AFV cost-grid gate at
    /// `median(mask1x1) > 95.0`) that skips the detector entirely on
    /// screenshot-class content. Chunk 6
    /// (`benchmarks/auto_splines_bench_2026-05-17_chunk6_fp.tsv`)
    /// added a bbox-span gate
    /// ([`crate::vardct::splines::detect_params::MIN_BBOX_SPAN_OF_IMAGE_LONG_DIM`])
    /// inside `spline_passes_trial_encode_gate`: any candidate whose
    /// `max(bbox_width, bbox_height)` doesn't span the image's long
    /// dimension is rejected. This closes the only observed real-photo
    /// false-positive cluster — 4 of 42 CID22-512 photos that regressed
    /// by +0.05% to +1.19% on opt-in auto_splines because the
    /// trial-encode L2-energy proxy couldn't tell a true thin feature
    /// from a sub-image ridge segment riding through textured photo
    /// content. With chunks 5+6 active every image in
    /// `benchmarks/auto_splines_bench_2026-05-17_chunk5.tsv` AND in
    /// the chunk-6 42-image CID22 sweep goes byte-identical.
    ///
    /// Default-on at e7+ remains rejected. The discriminator + chunk-6
    /// span gate are now so effective at filtering candidates that
    /// there are no observed wins to flip on. Photos go byte-identical
    /// under the combined chunk-4 cost gate + chunk-6 span gate.
    /// Screenshots go byte-identical under chunk 5 (discriminator skips).
    /// The chunk-3 synthetic wins (-2 to -3% on multi-line power-line
    /// images at e7/e8) are also gone because the synthetics have flat
    /// 80-grey backgrounds that the discriminator correctly classifies
    /// as screenshot-class — they were a calibration artefact, not a
    /// real-world content win. Flipping `auto_splines_default(_) = true`
    /// at any effort would add compute cost (one
    /// [`crate::vardct::adaptive_quant::compute_mask1x1`] pass per
    /// encode) for zero observable benefit on any tested image.
    ///
    /// The flag remains opt-in for callers who hand-tune for full-image-
    /// spanning thin features (power lines crossing a noisy sky edge to
    /// edge, hair strands spanning a photo background) where the
    /// discriminator does NOT fire AND the span gate admits the
    /// candidate AND the cost gate admits the candidate.
    ///
    /// Chunk 7
    /// (`benchmarks/auto_splines_bench_2026-05-17_chunk7.tsv`) re-asked
    /// the default-on question on 18 cells at d=1.0/e8: 5 power-line
    /// synthetics with photo-realistic backgrounds that bypass the
    /// chunk-5 discriminator + 10 CID22-512 photos (including all 4
    /// original chunk-6 FPs) + 3 CLIC2025-1024 photos. All 13 photos
    /// went byte-identical (chunk-6 closure holds), but 3 of 5 wire
    /// synthetics REGRESS by +3.1% / +4.3% / +5.5% at e8 — the
    /// trial-encode L2-energy proxy predicts a saving but the
    /// butteraugli loop re-quantizes the post-splines XYB and emits
    /// a strictly worse encode. The chunk-6 bbox-span gate rejects
    /// the other 2 (long_dim ≥ 2048, polyline tracer caps at ~1042 px).
    /// Default-on at e8+ would therefore net +0 photos and +3-5%
    /// wire regressions; the proxy is structurally mis-calibrated at
    /// the buttloop's effort range. A future flip needs either a
    /// buttloop-aware cost proxy or an effort-axis split.
    ///
    /// libjxl ships its own `enc_splines.cc:104-107` `FindSplines` as
    /// a stub at `speed_tier <= kSquirrel` (effort >= 7); the real
    /// detector landing on a future libjxl release would let us
    /// revisit this default with a properly compared algorithm.
    pub fn auto_splines_default(_effort: u8) -> bool {
        false
    }

    /// Smart per-image fanout adapter (opt-in via
    /// [`crate::api::LosslessConfig::with_smart_fanout`]).
    ///
    /// Re-tunes the three `tree_parallel_*` fields based on the input
    /// image's pixel count, not just effort. Per the
    /// `smart_fanout_sweep_2026-05-17` (8-image × 3-effort × 6-cell)
    /// investigation, depth=6 + floor=4096 wins or ties the
    /// effort-only defaults on every (image, effort) cell measured,
    /// EXCEPT large + e9 where the per-leaf subtree-build is enormous
    /// (~21 s sequential) and the current depth=5 is already optimal.
    ///
    /// Rule (post-sweep):
    /// - `pixels >= 4_000_000` and `effort >= 9`: keep effort default
    ///   (depth=5, floor=8192) — large e9 ceiling is the per-leaf
    ///   subtree, not parallel granularity.
    /// - otherwise: bump to depth=6, floor=4096, root_threshold=4096.
    ///
    /// Parallelism does not change the bitstream — the tree topology
    /// is determined by the samples, not the build order — so
    /// hash_lock sidecars stay byte-identical. This is purely a
    /// wall-clock knob.
    ///
    /// Investigation memory file:
    /// `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/
    ///  zenanalyze_tree_size_correlation_2026-05-17.md`.
    pub fn adapt_to_image(&mut self, pixels: u64) {
        let effort = self.effort;
        let large = pixels >= 4_000_000;
        if large && effort >= 9 {
            // Keep effort-only default (already tuned for the huge-tree case).
            return;
        }
        self.tree_parallel_max_depth = 6;
        self.tree_parallel_floor = 4_096;
        self.tree_parallel_root_threshold = 4_096;
    }

    /// Pixel-count + effort gate for the small-image parallel-tree-
    /// learning fallback. Always-on (NOT opt-in) — addresses the
    /// +0.85% small-image mean wall-clock regression documented in
    /// commit `cb5e202` (thread-local [`SplitWorkspace`] cache).
    ///
    /// When `pixels < SMALL_IMAGE_PIXEL_THRESHOLD` (1 MP) AND
    /// `effort <= 7`, flips `tree_parallel_small_image_fallback` to
    /// `true`. That causes
    /// [`crate::modular::tree_learn::compute_best_tree`] to allocate a
    /// fresh [`SplitWorkspace`] per `find_best_split` call instead of
    /// routing through the thread-local cache. The cache pays its own
    /// `RefCell::borrow_mut` indirection cost without amortising on
    /// small inputs at low effort (the workspace allocates once per
    /// encode anyway, and the tree is small enough that the cache hit
    /// rate doesn't matter).
    ///
    /// At effort >= 8 the tree grows enough that the per-call
    /// `SplitWorkspace::new` cost dominates the cache's `borrow_mut`
    /// indirection (paired bench at 0.26 MP × e9 measured the no-cache
    /// variant 7.45% SLOWER than the cached variant — exceeds the
    /// audit's small-image regression by an order of magnitude). The
    /// gate excludes e8+ to avoid that regression.
    ///
    /// The parallel root split + recursive borrowed-view fan-out
    /// remain ENABLED in this fallback regime — they are still the
    /// largest single wall-clock win at 8 threads, even on 0.26 MP.
    ///
    /// Bitstream-equivalent: tree topology depends only on the samples,
    /// not the workspace identity. Hash-locks stay byte-identical.
    ///
    /// Threshold rationale: per the
    /// `rejected_optimizations_conditional_value_2026-05-17.md` audit
    /// (item #10), the cache regression pivot is between 0.26 MP
    /// (small, +0.85% slower with cache) and 1.05 MP (medium, -2.6%
    /// faster with cache), measured at e7. The size gate is 1 MP and
    /// the effort gate is e7 (the audit's measurement effort).
    pub fn adapt_small_image_fallback(&mut self, pixels: u64) {
        if pixels < SMALL_IMAGE_PIXEL_THRESHOLD && self.effort <= 7 {
            self.tree_parallel_small_image_fallback = true;
        }
    }

    /// Pixel-count + effort gate for the `tree_max_buckets` dispatch
    /// (audit item #3, conditional-value catalog
    /// `rejected_optimizations_conditional_value_2026-05-17.md`).
    /// Always-on (NOT opt-in) — bytes change at large+e9 only, where
    /// the dispatch saves wall-clock at near-zero byte cost.
    ///
    /// When `pixels >= LARGE_IMAGE_PIXEL_THRESHOLD` (4 MP) AND
    /// `effort >= 9`, drops `tree_max_buckets` from the effort default
    /// (256 at e9) to [`LARGE_E9_TREE_MAX_BUCKETS`] (192).
    ///
    /// **Pareto evidence**
    /// (commit `4572790` Pareto sweep, `benchmarks/tree_max_buckets_pareto_2026-05-17.tsv`,
    /// 5 samples × 3 profile images × 6 bucket values @ effort 9,
    /// `RAYON_NUM_THREADS=8`, release build with `parallel-tree-learning`):
    ///
    /// | buckets | small_0.26MP   | medium_1.05MP  | large_4.19MP        |
    /// |---------|----------------|----------------|---------------------|
    /// | **192** | +1.79% / -4.9% | +0.10% / +1.9% | **+0.09% / -12.1%** |
    /// | 256     | baseline       | baseline       | baseline            |
    ///
    /// 192 is the only candidate where bytes stay in the noise floor on
    /// large (+0.09%) AND wall-clock wins are real (-12.1%). The
    /// per-property bucket-sweep in `find_best_split` scales roughly
    /// linearly with the bucket cap; the savings amortise only when
    /// the tree is deep enough — i.e. at the largest tier (≥4 MP at
    /// the most-expensive effort).
    ///
    /// Hash-locks: change at large+e9 cells (+0.09% byte cost is
    /// intentional). All other (size, effort) cells stay byte-identical
    /// because the dispatch does not fire.
    ///
    /// Threshold rationale: the same `pixels >= 4_000_000` boundary as
    /// [`Self::adapt_to_image`] (smart-fanout's large carve-out) and
    /// the audit's "≥3 MP" guidance. Effort gate is `>= 9` because the
    /// Pareto sweep was run only at e9 — at e7/e8 the win was not
    /// measured (`tree_max_buckets_for` returns 96/128 at e7/e8, so
    /// 192 would be an INCREASE, never measured).
    pub fn adapt_tree_max_buckets_for_image(&mut self, pixels: u64) {
        if pixels >= LARGE_IMAGE_PIXEL_THRESHOLD && self.effort >= 9 {
            self.tree_max_buckets = LARGE_E9_TREE_MAX_BUCKETS;
        }
    }

    /// Pixel-count + distance gate for the lossy VarDCT
    /// `try_dct64` (chunk 1) and `try_dct32` (chunk 2a / issue #43)
    /// evaluations. Always-on (NOT opt-in) — purely a wall-clock win
    /// on the small + low-distance cell.
    ///
    /// **Chunk 1** (try_dct64): when `pixels < LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD`
    /// (500_000) AND `distance < LOSSY_LOW_DISTANCE_THRESHOLD` (2.0),
    /// drops `try_dct64` from the effort default (`true` at effort ≥ 7) to
    /// `false`.
    ///
    /// **Chunk 2a** (try_dct32): independently, when
    /// `pixels < LOSSY_TINY_IMAGE_PIXEL_THRESHOLD` (100_000) AND
    /// `distance < LOSSY_VERY_LOW_DISTANCE_THRESHOLD` (0.5) AND
    /// `effort >= 7`, drops `try_dct32` (default `true` at effort >= 5)
    /// to `false`. The chunk 2a cell is a strict subset of the chunk 1
    /// cell, so on chunk-2a-firing inputs both flips happen together.
    ///
    /// Skips the entire
    /// [`crate::vardct::ac_strategy_search::find_best_64x64_transform`]
    /// pipeline (DCT64x64, 2×DCT64x32, 2×DCT32x64 candidates plus their
    /// 4×`find_best_32x32_transform` reuse path).
    ///
    /// **Rationale**: DCT64-class transforms cover 64×64 pixels. On a
    /// small image at low distance the cost-model entropy_mul
    /// (`2.25` for DCT64x64/DCT64x32 in pixel-domain mode) heavily
    /// penalises the 4096-coefficient block. On 512×512 (8×8 of 64×64
    /// tiles) at `d ≤ 1.0` they are essentially never picked —
    /// the per-tile cost gate in `find_best_64x64_transform` falls
    /// through to four recursive `find_best_32x32_transform` calls.
    /// The wasted work is the upfront DCT64x64 + 4 DCT64x32 +
    /// 4 DCT32x64 entropy estimates per 64×64 tile.
    ///
    /// **Hash-locks**: byte-identical at the gated cells (the skipped
    /// strategies were not winning at those sizes anyway — verified by
    /// the per-effort hash_lock sidecars at 13×17 / 32×32 / 48×48,
    /// none of which can evaluate a 64×64 block to begin with).
    ///
    /// **Threshold rationale**:
    /// - `pixels < 500_000`: covers the bench harness's `small_0.26MP`
    ///   cell (512×512 = 262_144 px). At ≥ 1 MP the corpus_regression
    ///   bench shows DCT64 starts winning on smooth regions, so the
    ///   gate stops short of medium.
    /// - `distance < 2.0`: matches the conservative gate documented in
    ///   `dropped_optimizations_for_parity_2026-05-15.md` (item #1
    ///   neighbourhood — DCT64 is "gated to d≥3.0" in the cost model
    ///   notes, and at d ∈ [2.0, 3.0] some images do pick DCT64).
    ///
    /// **Effort gate**: only applies when `try_dct64` is already on
    /// (`effort ≥ 7`). At effort < 7 this is a no-op.
    ///
    /// Bench provenance: paired A/B in
    /// `jxl-encoder/examples/vardct_ac_dispatch_paired_ab.rs`, results
    /// in `benchmarks/vardct_ac_dispatch_paired_2026-05-17.tsv`.
    ///
    /// **W44-35 smooth-photo escape hatch**: the W44-34 root-cause
    /// investigation traced 5 OPEN cells on `1418519.png` (CID22-512
    /// validation, 262_144 px) at e6/e7 × d ∈ {1.0, 1.2, 1.6} to this
    /// gate firing on a smooth photo where DCT64 actually WINS
    /// (-5 to -7 % bytes vs the gated default). The single-image
    /// calibration of the original gate (7256805 at small_0.26MP)
    /// missed the smooth-photo class entirely. The
    /// [`Self::adapt_to_image_lossy_with_smoothness`] variant takes a
    /// `smooth_photo_hint` boolean that, when `true`, suppresses the
    /// `try_dct64 -> false` flip even on the gated cell.
    pub fn adapt_to_image_lossy(&mut self, pixels: u64, distance: f32) {
        self.adapt_to_image_lossy_with_smoothness(pixels, distance, false);
    }

    /// Variant of [`Self::adapt_to_image_lossy`] that takes an explicit
    /// smooth-photo admission hint (W44-35).
    ///
    /// When `smooth_photo_hint == true` AND the would-otherwise-gated
    /// cell holds (small image + low distance), the behaviour swaps:
    /// instead of dropping `try_dct64 -> false`, the gate forces
    /// `try_dct64 -> true` so the encoder evaluates DCT64-class
    /// transforms on the smooth photo and the cost model picks the
    /// right partition. The W44-34 root-cause forensics found this
    /// admits the 5 OPEN ledger cells on `1418519.png` for -6.07 %
    /// paired total at e6/e7 × d ∈ {1.0, 1.2, 1.6}. The forced-true
    /// is gated on `ac_strategy_enabled` (effort >= 5) — at effort < 5
    /// the encoder skips AC strategy search entirely so DCT64 has no
    /// way to fire even if requested.
    ///
    /// When `smooth_photo_hint == false` (the default callers should
    /// use unless they've explicitly classified the image), behaviour is
    /// byte-identical to [`Self::adapt_to_image_lossy`].
    ///
    /// **Caller responsibility**: only set `true` when the input is
    /// classified as a smooth photo (low edge density, low HF energy,
    /// low solid-color block ratio). See
    /// `jxl-encoder/examples/dct64_smart_dispatch_calibrate.rs` for the
    /// W44-35 discriminator calibration and proxy implementation in
    /// `crate::api::detect_smooth_photo_for_dct64_from_layout`.
    pub fn adapt_to_image_lossy_with_smoothness(
        &mut self,
        pixels: u64,
        distance: f32,
        smooth_photo_hint: bool,
    ) {
        // Chunk 2a (issue #43): drop `try_dct32` on tiny + very-low-d
        // cells at effort >= 7. Independent of (and orthogonal to) the
        // chunk 1 `try_dct64` gate below — chunk 2a's tiny+very-low-d
        // cell is a strict subset of the chunk 1 small+low-d cell, so
        // when chunk 2a fires chunk 1's `try_dct64 = false` flip also
        // fires (only the chunk 2a `try_dct32 = false` is additional).
        //
        // Rule:
        //   if pixels < LOSSY_TINY_IMAGE_PIXEL_THRESHOLD (100_000)
        //      AND distance < LOSSY_VERY_LOW_DISTANCE_THRESHOLD (0.5)
        //      AND effort >= 7
        //      AND self.try_dct32 was true:
        //         self.try_dct32 = false
        //
        // The `effort >= 7` gate matches the chunk 1 / try_dct64 effort
        // gate. At effort < 7 the chunk 1 gate is a no-op (try_dct64
        // already off at effort < 7); for chunk 2a try_dct32 is on at
        // effort >= 5, so we explicitly cap at effort >= 7 to mirror
        // the chunk 1 conservatism and keep the dispatch family
        // calibrated against a single effort-band.
        //
        // Smoothness hint is NOT consulted for the try_dct32 gate —
        // chunk 1's smooth-photo escape hatch (W44-35) applies only
        // to try_dct64 because DCT64 is the strategy that wins on
        // smooth photos; DCT32 trade-off is independent.
        if pixels < LOSSY_TINY_IMAGE_PIXEL_THRESHOLD
            && distance < LOSSY_VERY_LOW_DISTANCE_THRESHOLD
            && self.effort >= 7
            && self.try_dct32
        {
            self.try_dct32 = false;
        }

        // Only consider the (chunk 1 / try_dct64) gate when the cell
        // holds (small + low-d).
        if pixels >= LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD || distance >= LOSSY_LOW_DISTANCE_THRESHOLD {
            return;
        }
        if smooth_photo_hint {
            // W44-35 admission: force-enable try_dct64 on the gated
            // cell (only when AC strategy search runs, i.e. effort >= 5
            // per `EffortProfile::lossy` defaults). At effort < 5 the
            // entire AC strategy search is skipped and DCT64 cannot
            // fire — admitting would be a no-op + sweep noise.
            if self.ac_strategy_enabled {
                self.try_dct64 = true;
            }
        } else if self.try_dct64 {
            // Original (pre-W44-35) gate: drop try_dct64 to false on
            // small + low-d cells. Preserves the byte-identical
            // behaviour documented in `vardct_ac_dispatch_paired_2026-05-17`.
            self.try_dct64 = false;
        }
    }

    /// Apply the Section A effort-gate divergences ([`crate::api::EffortGate`])
    /// to the three lossy-only profile fields (`cfl_two_pass`,
    /// `try_dct64`, `epf_dynamic_sharpness`) per
    /// [`docs/LIBJXL_DIVERGENCES.md`](https://github.com/imazen/jxl-encoder/blob/main/docs/LIBJXL_DIVERGENCES.md)
    /// Section A.
    ///
    /// **W44-133 Chunk G** — final chunk of the EncoderStrategy API
    /// consolidation. When [`crate::api::EncoderStrategy::Libjxl`] is
    /// selected, every Section A row flips to the libjxl threshold
    /// listed in the divergence table. The flip happens AFTER
    /// `lossy_reference` constructs the profile but BEFORE the encoder
    /// reads the fields — so legacy callers that bypass
    /// [`crate::api::EncoderStrategy`] (raw `EffortProfile::lossy(...)`
    /// in tests, examples, harnesses) are byte-identical.
    ///
    /// Per-site `(ours_min_effort, libjxl_min_effort)` pairs
    /// (mirroring the `effort.rs::lossy_reference` constants and the
    /// libjxl source-tree lookups documented in `EffortGate::evaluate`):
    /// - `cfl_two_pass`: `(7, 5)`
    /// - `try_dct64`: `(7, 0)` — libjxl has no effort gate
    ///   (`enc_ac_strategy.cc:948` uses `decoding_speed_tier < 4`)
    /// - `epf_dynamic_sharpness`: `(6, 0)` — libjxl has no effort gate
    ///
    /// **Important — `EffortGate::Ours` is a NO-OP**: when the resolved
    /// field equals the default [`crate::api::EffortGate::Ours`], the
    /// profile field is LEFT UNTOUCHED. This preserves any prior
    /// `adapt_to_image_lossy_with_smoothness` / explicit
    /// `with_internal_params` / `apply_faster_decoding` adjustments to
    /// the field — `Ours` means "do nothing", NOT "re-evaluate from
    /// `lossy_reference`'s threshold". Only the explicit `Libjxl` /
    /// `Off` / `AtLeast(n)` variants actually rewrite the field.
    ///
    /// Only relevant for lossy profiles. The Lossless path always
    /// returns these fields as `false` (see `lossless_reference`) so
    /// this method is structurally a no-op for lossless encodes — the
    /// `EffortGate::Libjxl` flip on `try_dct64` would technically
    /// re-evaluate to `true`, but lossless callers never construct a
    /// `ResolvedImprovements` with non-default Section A picks today.
    pub(crate) fn apply_section_a_effort_gates(
        &mut self,
        resolved: &crate::api::ResolvedImprovements,
    ) {
        use crate::api::EffortGate;
        let effort = self.effort;
        // cfl_two_pass: we e7+, libjxl e5+
        if !matches!(resolved.cfl_two_pass_min_effort, EffortGate::Ours) {
            self.cfl_two_pass = resolved
                .cfl_two_pass_min_effort
                .evaluate(effort, /*ours=*/ 7, /*libjxl=*/ 5);
        }
        // try_dct64: we e7+, libjxl has no effort gate
        if !matches!(resolved.try_dct64_min_effort, EffortGate::Ours) {
            self.try_dct64 = resolved
                .try_dct64_min_effort
                .evaluate(effort, /*ours=*/ 7, /*libjxl=*/ 0);
        }
        // epf_dynamic_sharpness: we e6+, libjxl has no effort gate
        if !matches!(resolved.epf_dynamic_sharpness_min_effort, EffortGate::Ours) {
            self.epf_dynamic_sharpness = resolved
                .epf_dynamic_sharpness_min_effort
                .evaluate(effort, /*ours=*/ 6, /*libjxl=*/ 0);
        }
    }

    /// Apply the W44-184 Section C CfL Newton libjxl-parity flip.
    ///
    /// When [`crate::api::ResolvedImprovements::cfl_newton_libjxl_parity`]
    /// is `true` (set only by [`crate::api::EncoderStrategy::Libjxl`]),
    /// flips [`Self::cfl_newton_libjxl_parity`] to `true` so the
    /// downstream Newton call sites pass `libjxl_parity = true` into
    /// [`jxl_simd::cfl_find_best_multiplier_newton`].
    ///
    /// The Section C divergence is documented as INTENTIONAL at the
    /// default path (W44-183 measurement: 25/27 photo cells regress
    /// SSIM2 by 0.25-13.02 + 7.82% mean bytes). Under
    /// `EncoderStrategy::Libjxl` the downstream cost model is also
    /// flipped to libjxl-parity (Section A effort gates widen, Section
    /// B content-aware gates disable, Section D KNOWN-BUGs re-enable),
    /// so the +7.82% regression observed at the default-path port does
    /// NOT apply on that strategy — the entire pipeline is uniformly
    /// libjxl-parity.
    ///
    /// Called from
    /// [`crate::api::LossyConfig::effective_profile_for_image_with_smoothness`]
    /// alongside `apply_section_a_effort_gates`. Default (`false`)
    /// preserves pre-W44-184 byte-identical output on Zenjxl / LeanFaster
    /// / Aggressive strategies (their `ResolvedImprovements` carries
    /// `cfl_newton_libjxl_parity: false`).
    pub(crate) fn apply_section_c_cfl_newton_libjxl_parity(
        &mut self,
        resolved: &crate::api::ResolvedImprovements,
    ) {
        // W44-184: NO-OP semantic when the resolved field is `false`
        // (mirrors `EffortGate::Ours` in `apply_section_a_effort_gates`):
        // preserve any prior adapter-set value. Today only the strict
        // `EncoderStrategy::Libjxl` constructor sets this `true`; future
        // per-image discriminators could set it differently and we don't
        // want to clobber them.
        if resolved.cfl_newton_libjxl_parity {
            self.cfl_newton_libjxl_parity = true;
        }
        // W44-AUDIT-5 Phase 2 (Mode C): same NO-OP-when-false semantic.
        // Set by Zenjxl / Aggressive presets after the AUDIT-5 default-flip.
        // Mutually-exclusive with `cfl_newton_libjxl_parity` inside the
        // SIMD kernel — Libjxl strategy keeps the bit-exact path
        // (`libjxl_parity = true`, `libjxl_math_with_ls_warm_start = false`);
        // Zenjxl / Aggressive ship Mode C (`libjxl_parity = false`,
        // `libjxl_math_with_ls_warm_start = true`).
        if resolved.cfl_newton_libjxl_math_with_ls_warm_start {
            self.cfl_newton_libjxl_math_with_ls_warm_start = true;
        }
        // W44-AUDIT-5 Phase 3: same NO-OP-when-false semantic. Set by
        // Zenjxl / Aggressive presets after the Phase 3 bisect +
        // regression validation. Composes with the per-image
        // `m3 >= 80` discriminator at the CfL Pass-1 / Pass-2 dispatch
        // sites — the route flip only fires when both this field AND
        // the per-image proxies indicate a high-colour-class screenshot.
        if resolved.cfl_pass1_screenshot_x0_start {
            self.cfl_pass1_screenshot_x0_start = true;
        }
        // W44-197: same NO-OP semantic. Only `EncoderStrategy::Libjxl`
        // sets `cfl_pass2_ls_at_low_effort = true` in its
        // `ResolvedImprovements`. When `true`, the encoder fires
        // `refine_cfl_map` with `use_newton=false` at effort ∈ {5, 6}
        // in addition to the existing `cfl_two_pass: effort >= 7`
        // Newton path. See field docstring on
        // `EffortProfile::cfl_pass2_ls_at_low_effort` + W44-189 D12
        // audit memo.
        if resolved.cfl_pass2_ls_at_low_effort {
            self.cfl_pass2_ls_at_low_effort = true;
        }
        // W44-AUDIT-9 / SA-G Fix C: same NO-OP-when-false semantic.
        // Set by `EncoderStrategy::Libjxl` to mirror libjxl
        // `enc_ac_strategy.cc` speed_tier > kSquirrel behaviour (force
        // cmap=zeros during AC strategy SEARCH only). Zenjxl /
        // Aggressive / LeanFaster keep `false` to preserve their
        // W44-29..W44-172 cost-model calibration baseline. See field
        // docstring on `EffortProfile::cfl_zero_for_search`.
        if resolved.cfl_zero_for_search {
            self.cfl_zero_for_search = true;
        }
    }

    /// Content-class-aware per-image adapter (RFC #45 pick #4 chunk 1).
    ///
    /// Specializes encoder defaults based on a coarse content class
    /// (`Photo` vs `Screenshot` vs `Other` vs `Unknown`). Designed to be
    /// called *after* [`Self::adapt_to_image_lossy`] /
    /// [`Self::adapt_to_image`] so the size-dependent gates run first.
    ///
    /// **Current dispatch rule (chunk 1)** — `Screenshot`-class content
    /// at lossy effort 5 / 6 with `pixels >= CONTENT_CLASS_MIN_PIXELS`
    /// (256 × 256 = 65 536) and `distance > 0.0` flips
    /// `self.patches = true`. The libjxl default keeps patches off until
    /// effort 7 for VarDCT; on screenshots the per-corpus measured
    /// savings (≈ 37 % on GB82-SC at e7) justify enabling them one or two
    /// effort levels earlier. Photos and unknown-class inputs are
    /// untouched, so hash-locks on the standard fixtures stay byte-
    /// identical (those fixtures are all sub-256² synthetic test images,
    /// well below the `CONTENT_CLASS_MIN_PIXELS` gate).
    ///
    /// **Current dispatch rule (issue #43 chunk 2c)** — `Screenshot`-class
    /// content at lossy effort 5 with `pixels >= CONTENT_CLASS_MIN_PIXELS`
    /// and `distance` inside the measured win band
    /// [[`AFV_SCREENSHOT_LIFT_MIN_DISTANCE`] (1.0),
    /// [`AFV_SCREENSHOT_LIFT_MAX_DISTANCE`] (2.0)] (inclusive) flips
    /// `self.try_dct4x8_afv = true`, enabling the DCT4X8 / DCT8X4 /
    /// DCT4X4 / AFV0-3 evaluation block in AC strategy search one effort
    /// level below its libjxl-parity entry point (`effort >= 6`,
    /// e6/Wombat). The chunk's original premise ("AFV is opt-in,
    /// auto-enable at e >= 6") was stale — the block is already
    /// default-on at e >= 6 for every strategy, so e5 is the only
    /// residual dispatch surface. Note this changes only **which images**
    /// evaluate the 8×8-class block (per W44-60: AFV evaluation policy
    /// itself stays at libjxl parity), and it lifts the whole
    /// `try_dct4x8_afv` block — AFV is not separable from DCT4X8 / DCT8X4
    /// / DCT4X4 without restructuring the search. Env hook
    /// `JXL_DISPATCH_AFV_SCREENSHOT_DISABLE=1` suppresses the lift
    /// (diagnostic A/B + emergency rollback; no-op in `no_std` builds).
    ///
    /// All other content classes / effort levels are no-ops; the dispatch
    /// surface is extensible and future chunks can add more rules
    /// (per-class `tree_max_buckets`, etc.) without breaking callers.
    ///
    /// **Spec-compliance**: every dispatched change leaves the bitstream
    /// 100 % spec-valid (patches and all 8×8-class transforms are normal
    /// encoder features, libjxl decoder reads them natively).
    ///
    /// **Effort gate rationale**: the patches dispatch fires at e ∈ {5, 6}
    /// because (a) e7 already has patches on by default and (b) e ≤ 4
    /// disables most VarDCT machinery that patches piggybacks on (AC
    /// strategy search). The AFV/8×8-class dispatch fires at e == 5 only
    /// because `try_dct4x8_afv` is already `true` at e >= 6 and the same
    /// e ≤ 4 machinery argument applies below. The pixel gate excludes
    /// synthetic fixtures (the largest hash-lock fixture is 48 × 48 =
    /// 2 304 px, three orders of magnitude below 65 536).
    pub fn adapt_to_image_content(
        &mut self,
        pixels: u64,
        distance: f32,
        content_class: ImageContentClass,
    ) {
        if pixels < CONTENT_CLASS_MIN_PIXELS {
            return;
        }
        if content_class == ImageContentClass::Screenshot
            && distance > 0.0
            && (self.effort == 5 || self.effort == 6)
            && !self.patches
        {
            self.patches = true;
        }
        // Issue #43 chunk 2c: extend the e6+ default-on 8×8-class
        // transform set (DCT4X8/DCT8X4/DCT4X4/AFV0-3) down to e5 on
        // Screenshot-class content, distance-banded to the measured
        // win region (see Section B row in docs/LIBJXL_DIVERGENCES.md
        // + benchmarks/dispatch_2c_afv_screenshot_2026-06-10.{tsv,meta}).
        // In-band, 14/14 gb82-sc cells win bytes (mean -1.21 %) at
        // better mean butteraugli; out-of-band (d = 0.5 / 4.0) the
        // block trades bytes FOR quality instead (terminal e5 d=4
        // +2.45 % bytes for -0.22 butteraugli), so the band keeps the
        // chunk's bytes-win contract. W44-135 distance-band precedent.
        if content_class == ImageContentClass::Screenshot
            && (AFV_SCREENSHOT_LIFT_MIN_DISTANCE..=AFV_SCREENSHOT_LIFT_MAX_DISTANCE)
                .contains(&distance)
            && self.effort == 5
            && !self.try_dct4x8_afv
            && !afv_screenshot_lift_disabled_by_env()
        {
            self.try_dct4x8_afv = true;
        }
    }

    /// Bias the profile toward bitstreams that decode faster, at the cost
    /// of compression. Mirrors libjxl `cparams.decoding_speed_tier` /
    /// `cjxl --faster_decoding 0..4`. Applied on top of the effort-derived
    /// profile and any `__expert` overrides — call last.
    ///
    /// Per-tier effects (additive — tier N applies the changes for tiers
    /// 1..=N):
    ///
    /// - `0`: no-op (default).
    /// - `1`: disable LZ77 backward references.
    ///   - VarDCT: AC stream tokens no longer rate-search LZ77 (libjxl
    ///     `enc_ans.cc:1372` flips `lz77_method = kNone` for VarDCT at
    ///     `decoding_speed_tier >= 1`).
    ///   - Modular: residual streams skip LZ77 (libjxl `enc_modular.cc`
    ///     `cparams_.decoding_speed_tier >= 1` clamps the histogram-pass
    ///     LZ77 method).
    ///   - Modular DC stream switches to the fixed `kGradientFixedDC` tree
    ///     (libjxl `enc_modular.cc:1600`) — handled by [`Self::tree_learning`]
    ///     being false on the DC sub-stream below.
    /// - `2`: tier 1 plus drop enhanced (pair-merge) histogram clustering
    ///   for VarDCT. libjxl caps modular `max_histograms = 12` and forces
    ///   `modular_group_size_shift = 0` at this tier; the group-size
    ///   override is applied by the per-config getter
    ///   ([`crate::api::LosslessConfig::effective_modular_group_size_shift`]),
    ///   not on this profile.
    /// - `3`: tier 2 plus drop custom coefficient orders. Decoders skip the
    ///   per-block permutation lookup and use the fixed natural order
    ///   (libjxl `enc_modular.cc:533` raises the tree-split threshold by
    ///   `+10 * decoding_speed_tier` — captured here by lowering tree
    ///   shape parameters).
    /// - `4`: tier 3 plus simpler context tree + no patches/tree-learning
    ///   pass on the modular path. libjxl also disables gaborish
    ///   (`enc_frame.cc:280`), DCT32X32 (`enc_ac_strategy.cc:936`), and
    ///   the `decoding_speed_tier_max_limit < 4` AC merges; mirrored here
    ///   by flipping `gaborish` / `try_dct32` / `try_dct64`.
    ///
    /// Bitstream remains 100 % spec-valid at every tier — these are encoder
    /// choices the libjxl decoder reads natively.
    pub fn apply_faster_decoding(&mut self, tier: u8) {
        if tier == 0 {
            return;
        }
        // Tier 1: disable LZ77.
        if tier >= 1 {
            self.lz77 = false;
        }
        // Tier 2: + disable enhanced (pair-merge) clustering for VarDCT.
        if tier >= 2 {
            self.enhanced_clustering_vardct = false;
        }
        // Tier 3: + drop custom coefficient orders, raise tree-split
        // threshold (libjxl enc_modular.cc:533 `+10 * speed_tier`).
        if tier >= 3 {
            self.custom_orders = false;
            // Mirror libjxl `splitting_heuristics_node_threshold +=
            // 10 * decoding_speed_tier` — at tier 3 that's +30 over the
            // effort-derived base, biasing the tree shallower.
            self.tree_threshold_base += 10.0 * tier as f32;
        }
        // Tier 4: + no MA tree learning, no patches; force-disable the
        // libjxl-gated VarDCT features (gaborish, DCT32X32, DCT64).
        if tier >= 4 {
            self.tree_learning = false;
            self.patches = false;
            self.gaborish = false;
            self.try_dct32 = false;
            self.try_dct64 = false;
            // Tighter MA-tree shape on the modular side (libjxl
            // enc_modular.cc:506-513 `nb_repeats = 0` is the strongest
            // signal — captured by zeroing tree_sample_fraction so the
            // sampler returns the 65k floor and the tree learner sees
            // minimal data).
            self.tree_sample_fraction = 0.0;
        }
    }
}

/// Issue #43 chunk 2c: inclusive distance band for the Screenshot-class
/// `try_dct4x8_afv` lift at effort 5. The 7-image × 4-distance gb82-sc
/// production-context A/B (`benchmarks/dispatch_2c_afv_screenshot_2026-06-10.tsv`)
/// wins bytes on 14/14 cells inside d ∈ [1.0, 2.0] (mean -1.21 %) but is
/// mixed-to-regressive at d = 0.5 (windows95 +0.59 %) and d = 4.0
/// (terminal +2.45 % — the block buys butteraugli instead of bytes
/// there). Band = the measured win region, nothing wider (W44-135
/// precedent: ship the measured band, not the extrapolated one).
pub const AFV_SCREENSHOT_LIFT_MIN_DISTANCE: f32 = 1.0;

/// Issue #43 chunk 2c: upper edge (inclusive) of the distance band for
/// the Screenshot-class `try_dct4x8_afv` lift at effort 5. See
/// [`AFV_SCREENSHOT_LIFT_MIN_DISTANCE`].
pub const AFV_SCREENSHOT_LIFT_MAX_DISTANCE: f32 = 2.0;

/// Issue #43 chunk 2c env hook: `JXL_DISPATCH_AFV_SCREENSHOT_DISABLE=1`
/// suppresses the Screenshot-class `try_dct4x8_afv` lift at effort 5 in
/// [`EffortProfile::adapt_to_image_content`] (diagnostic A/B + emergency
/// rollback). Mirrors the `#[cfg(feature = "std")]` guard pattern from
/// `gate_registry::apply_w44_120_min_distance_env_fallback` — always
/// `false` (lift active) when env vars are unreadable in `no_std`.
#[must_use]
fn afv_screenshot_lift_disabled_by_env() -> bool {
    #[cfg(feature = "std")]
    {
        afv_screenshot_lift_disable_value(
            std::env::var("JXL_DISPATCH_AFV_SCREENSHOT_DISABLE")
                .ok()
                .as_deref(),
        )
    }
    #[cfg(not(feature = "std"))]
    {
        false
    }
}

/// Pure predicate behind [`afv_screenshot_lift_disabled_by_env`], split
/// out so the value-parsing contract (`"1"` disables; anything else —
/// including empty / `"0"` / unset — keeps the lift active) is unit-
/// testable without process-env mutation (which requires `unsafe` in
/// edition 2024; the lib crate forbids unsafe). The env-mutation path is
/// covered by the integration test in
/// `tests/it/dispatch_2c_afv_screenshot.rs`.
#[cfg(any(feature = "std", test))]
#[must_use]
fn afv_screenshot_lift_disable_value(v: Option<&str>) -> bool {
    v == Some("1")
}

/// Coarse content class used by [`EffortProfile::adapt_to_image_content`].
///
/// Computed externally (typically via the optional `zenanalyze` integration
/// in [`crate::api`]); the [`EffortProfile`] surface intentionally
/// does not depend on the feature-extraction crate. Callers that don't have
/// classification available should pass [`Self::Unknown`] — every dispatch
/// rule treats it as "no change".
///
/// **Stability**: the variant set is `#[non_exhaustive]`; future chunks may
/// add classes (e.g., `Illustration`, `Document`, `LineArt`) without a
/// breaking change. Match arms must use `_` for the catch-all.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageContentClass {
    /// No classification available (default). Every dispatch rule treats
    /// this as "leave profile alone".
    #[default]
    Unknown,
    /// Natural photograph — high `edge_density`, low
    /// `flat_color_block_ratio`, non-zero `skin_tone_fraction` on portraits.
    Photo,
    /// Screen content — UI / document / terminal capture. High
    /// `flat_color_block_ratio` and `uniformity`, low `chroma_complexity`.
    /// Drives `patches` enablement at lower effort levels.
    Screenshot,
    /// Other / mixed content that fits none of the above buckets cleanly.
    /// No dispatch rules fire on this class today.
    Other,
}

/// Pixel-count threshold below which the parallel-tree-learning path
/// bypasses the thread-local [`SplitWorkspace`] cache (per-call
/// `SplitWorkspace::new` instead). The parallel root split + recursive
/// fan-out remain enabled — only the workspace allocation strategy
/// changes. See [`EffortProfile::adapt_small_image_fallback`].
pub const SMALL_IMAGE_PIXEL_THRESHOLD: u64 = 1_000_000;

/// Pixel-count threshold at or above which the `tree_max_buckets`
/// dispatch fires (at effort >= 9). See
/// [`EffortProfile::adapt_tree_max_buckets_for_image`].
pub const LARGE_IMAGE_PIXEL_THRESHOLD: u64 = 4_000_000;

/// `tree_max_buckets` value at large+e9 cells. Replaces the e9 default
/// of 256. See [`EffortProfile::adapt_tree_max_buckets_for_image`].
pub const LARGE_E9_TREE_MAX_BUCKETS: u16 = 192;

/// Pixel-count threshold below which the lossy VarDCT
/// `adapt_to_image_lossy` adapter disables the DCT64 strategy class
/// at low distance. See [`EffortProfile::adapt_to_image_lossy`].
pub const LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD: u64 = 500_000;

/// Distance below which the lossy VarDCT `adapt_to_image_lossy`
/// adapter disables the DCT64 strategy class on small images.
/// See [`EffortProfile::adapt_to_image_lossy`].
pub const LOSSY_LOW_DISTANCE_THRESHOLD: f32 = 2.0;

/// Pixel-count threshold below which the lossy VarDCT
/// `adapt_to_image_lossy` adapter (chunk 2a / issue #43) disables the
/// DCT32 strategy class at very low distance + effort >= 7.
///
/// Tiny + very-low-d is a strict subset of the chunk 1 small + low-d
/// cell. DCT32 evaluation runs four 32×32 candidates per 32×32 tile
/// (DCT32X32 + DCT32X16 + DCT16X32 + 4×`find_best_32x32_transform`
/// reuse path) and at very low distance + tiny image they essentially
/// never win — the cost-model entropy_mul (1.48 base, 1024 coefficients)
/// heavily penalises the 32×32 block on tiny images.
///
/// See [`EffortProfile::adapt_to_image_lossy_with_smoothness`].
pub const LOSSY_TINY_IMAGE_PIXEL_THRESHOLD: u64 = 100_000;

/// Distance below which the lossy VarDCT `adapt_to_image_lossy`
/// adapter (chunk 2a / issue #43) disables the DCT32 strategy class
/// on tiny images at effort >= 7.
///
/// Strict subset of [`LOSSY_LOW_DISTANCE_THRESHOLD`] (= 2.0). At
/// very low distance the cost-model favours smaller blocks (DCT8 /
/// DCT4X8 / DCT4X4) that capture fine detail at the bitrates achieved
/// at d < 0.5.
///
/// See [`EffortProfile::adapt_to_image_lossy_with_smoothness`].
pub const LOSSY_VERY_LOW_DISTANCE_THRESHOLD: f32 = 0.5;

/// Minimum pixel count for content-class dispatch to consider firing.
/// Below this the classifier is unreliable (synthetic / thumbnail content)
/// and the per-fixture hash-locks are well below the threshold.
/// See [`EffortProfile::adapt_to_image_content`].
pub const CONTENT_CLASS_MIN_PIXELS: u64 = 65_536;

// ─────────────────────────────────────────────────────────────────────────
// Public expert surface — segmented Lossy / Lossless internal-param structs
// ─────────────────────────────────────────────────────────────────────────
//
// `LossyInternalParams` and `LosslessInternalParams` are the public picker /
// sweep escape hatch (gated behind `__expert`). They split the internal
// [`EffortProfile`] into two type-disjoint surfaces — one per encode mode —
// so callers cannot accidentally hand the lossy encoder a knob that only
// affects modular output, and vice-versa. The type system enforces
// mode-correctness instead of relying on documentation.
//
// Each `Some(_)` field overrides the corresponding `EffortProfile` field
// the lossy / lossless code path actually reads. Fields left at `None` keep
// the (effort, mode)-derived default. This matches the segmented
// `InternalParams` pattern used by zenavif / zenwebp / zenravif.

/// Picker / sweep override knobs for the **lossy (VarDCT)** encode path.
///
/// Apply via [`crate::api::LossyConfig::with_internal_params`]. Fields are
/// optional: `Some(value)` overrides the corresponding effort-derived
/// default; `None` keeps the default. `#[non_exhaustive]` so additional
/// knobs can land additively without a breaking change.
///
/// The fields here are the lossy-side knobs that flow through `profile.X`
/// at lossy encode time (verified against `vardct/encoder.rs`,
/// `vardct/ac_strategy_search.rs`, `vardct/transform.rs`,
/// `vardct/precomputed.rs`, and `vardct/bitstream.rs`). Modular-only knobs
/// (RCT search, WP parameter scan, tree-learning shape) live on
/// [`LosslessInternalParams`] — VarDCT's DC frame uses a fixed Gradient
/// predictor, so those knobs do not affect lossy bytes.
#[cfg(feature = "__expert")]
#[non_exhaustive]
#[derive(Default, Clone, Debug)]
pub struct LossyInternalParams {
    /// Try DCT16x16 / DCT16x8 / DCT8x16 transforms in AC strategy search.
    /// Default at effort 7: `true`. Disabling forces no 16×16-class merges.
    pub try_dct16: Option<bool>,

    /// Try DCT32x32 / DCT32x16 / DCT16x32 transforms.
    /// Default at effort 7: `true`. Disabling forces no 32×32-class merges.
    pub try_dct32: Option<bool>,

    /// Try DCT64x64 / DCT64x32 / DCT32x64 transforms.
    /// Default at effort 7: `true`. Disabling forces no 64×64-class merges.
    pub try_dct64: Option<bool>,

    /// Try DCT4x8 / DCT8x4 / DCT4x4 / AFV transforms.
    /// Default at effort 6+: `true`. Disabling forces 8×8-or-larger only.
    pub try_dct4x8_afv: Option<bool>,

    /// Step size for fine-grained AC strategy search on 32×32+ blocks.
    /// `1` evaluates every position (effort 10+, extends past libjxl
    /// kGlacier), `2` every other (default, matches libjxl
    /// `enc_ac_strategy.cc:1046` for `speed_tier >= kTortoise`).
    pub fine_grained_step: Option<u8>,

    /// Base multiplier on the IDCT-domain (pixel-domain) error term in
    /// `EstimateEntropy`. Reference: 1.2 (libjxl). Experimental: 1.3
    /// (PR #4506). Higher values weight visible artifacts more heavily
    /// vs coefficient-domain entropy.
    pub k_info_loss_mul_base: Option<f32>,

    /// Per-strategy entropy multipliers for AC strategy cost model.
    /// Controls relative preference for each transform type.
    pub entropy_mul_table: Option<EntropyMulTable>,

    /// Recompute CfL map after initial quantization for better estimates.
    /// Default at effort 7+: `true`.
    pub cfl_two_pass: Option<bool>,

    /// Apply pixel-level chromacity adjustments. Default at effort 7+:
    /// `true`. Disabling skips per-pixel chromacity nudges.
    pub chromacity_adjustment: Option<bool>,

    /// Use tree learning for patch reference frame encoding instead of the
    /// fixed Gradient predictor. Reference: `false`. Experimental at
    /// effort 7+: `true`. Significant on screenshots / packed glyph patches.
    pub patch_ref_tree_learning: Option<bool>,

    /// Enable non-aligned evaluation pass (odd-aligned 16×16 regions) in
    /// AC strategy search. Default at effort 6+: `true`. Disabling halves
    /// the search depth.
    pub non_aligned_eval: Option<bool>,

    /// Use pair-merge clustering for VarDCT entropy codes. Reference at
    /// effort 9+: `true`; experimental at effort 7+: `true`. When `false`,
    /// uses fast k-means-only clustering (cheaper, slightly larger codes).
    pub enhanced_clustering_vardct: Option<bool>,

    /// ANS histogram normalization strategy for VarDCT entropy codes.
    /// Default mirrors libjxl `enc_ans_params.h:60-75`: `Approximate` at
    /// effort <= 7 (libjxl `tier >= kSquirrel`), `Precise` at effort >= 8.
    /// `Fast` is exposed for sweeps; users should rarely override the
    /// effort-derived default.
    pub ans_histogram_strategy_vardct: Option<ANSHistogramStrategy>,

    /// Quantization-cost constant used when materializing the initial
    /// quant field (libjxl 0.765, `enc_adaptive_quantization.cc`). Lower
    /// values produce a coarser initial field (less rate, more distortion);
    /// higher values refine.
    pub k_ac_quant: Option<f32>,

    /// Override the number of butteraugli-loop seeds (RFC#45 pick #1
    /// chunk 3). See [`EffortProfile::lossy_search_seeds`] for the
    /// per-effort defaults and the seed-selection rule. Setting this to
    /// `Some(1)` reverts to libjxl's single-seed loop even at e10/e11.
    pub lossy_search_seeds: Option<u8>,
}

/// Picker / sweep override knobs for the **lossless (modular)** encode path.
///
/// Apply via [`crate::api::LosslessConfig::with_internal_params`]. Fields
/// are optional: `Some(value)` overrides the corresponding effort-derived
/// default; `None` keeps the default. `#[non_exhaustive]` so additional
/// knobs can land additively without a breaking change.
///
/// The fields here are the modular-path knobs that flow through `profile.X`
/// in `modular/encode.rs`, `modular/frame.rs`, `modular/section.rs`,
/// `modular/predictor.rs`, and `modular/tree_learn.rs`. AC-strategy and
/// CfL knobs live on [`LossyInternalParams`].
#[cfg(feature = "__expert")]
#[non_exhaustive]
#[derive(Default, Clone, Debug)]
pub struct LosslessInternalParams {
    /// Number of Reversible Color Transform variants to evaluate before
    /// committing (0 = skip search, use `RctType::GBR_SUBGR`
    /// unconditionally — NOT YCoCg; calibrated fallback since `99162a2a`).
    /// Because the default search often picks GBR_SUBGR as the winner,
    /// `Some(0)` can be byte-identical to the default on some content —
    /// use `Some(1)` (identity-RCT-only) as an override-propagation test
    /// signal instead (jxl-encoder#67, W44-137).
    /// Effort interaction: 0 at e<5, 4 at e5, 5 at e6, 7 at e7, 9 at e8,
    /// 19 at e9+ (libjxl `kSquirrel`/`kKitten`/`kTortoise` schedule).
    pub nb_rcts_to_try: Option<u8>,

    /// Force a specific RCT colorspace; when `Some(rct)`,
    /// `select_best_rct(_at)` skips the search entirely.
    /// Mirrors libjxl's `cparams.colorspace`. `None` keeps the
    /// per-effort search behaviour.
    pub forced_rct: Option<crate::modular::rct::RctType>,

    /// Number of weighted-predictor parameter sets to try per WP-eligible
    /// channel (0 = use libjxl defaults without searching).
    /// Effort interaction: 0 at e<8, 2 at e8, 5 at e9+.
    pub wp_num_param_sets: Option<u8>,

    /// Maximum quantization buckets per property when building the
    /// histogram for tree splits.
    /// Effort interaction: 32 at e<=4, 48 at e5, 64 at e6, 96 at e7,
    /// 128 at e8, 256 at e9+. Higher = finer thresholds at higher cost.
    pub tree_max_buckets: Option<u16>,

    /// Number of MA-tree decision properties to evaluate per split.
    /// Effort interaction: 3 at e<=4, 4 at e5, 5 at e6, 7 at e7, 10 at e8,
    /// 16 at e9+.
    pub tree_num_properties: Option<u8>,

    /// Base entropy-cost threshold a candidate split must beat to be
    /// accepted (libjxl `75 + 14 * speed_tier`). Lower = more splits =
    /// larger tree.
    pub tree_threshold_base: Option<f32>,

    /// Fraction of total pixels to sample for tree learning (when
    /// `tree_max_samples_fixed` is `0`). Floor of 65,536 samples.
    /// Effort interaction: 0.15 at e<=4 ramping to 0.65 at e9+
    /// (libjxl PR #4236).
    pub tree_sample_fraction: Option<f32>,

    /// Hard cap on samples drawn for tree learning when set; `0` defers
    /// to [`Self::tree_sample_fraction`].
    /// Effort interaction: 65,000 at e<=4, 0 at e>=5.
    pub tree_max_samples_fixed: Option<u32>,

    /// Switch the tree-sample dedup backend.
    ///
    /// `Some(true)` enables the streaming two-hash cuckoo path
    /// (`dedup_samples_streaming`, libjxl `AddSample` parity). `Some(false)`
    /// keeps the default packed-key sort path
    /// (`dedup_samples_packed_sort`). `None` leaves the effort profile
    /// default (always `false` today; see [`EffortProfile::use_streaming_dedup`]).
    ///
    /// The streaming path **regresses** wall-clock by +3 % to +8 % at e7
    /// on real CLIC photos (issue #41 measurement, 2026-05-16). Retained
    /// for experimentation toward issue #41 Phase 2 (gather-integrated
    /// dedup); not recommended for production sweeps.
    pub use_streaming_dedup: Option<bool>,

    /// Enable libjxl-parity gather-time dedup (Phase 2 of issue #41).
    ///
    /// `Some(true)` runs each gathered sample through a two-hash cuckoo
    /// table inside `gather_channel_samples`, merging duplicates *during*
    /// the gather pass. The post-gather `dedup_samples_packed_sort` then
    /// operates on a much smaller surviving set. `Some(false)` keeps the
    /// existing post-pass dedup-only flow. `None` leaves the
    /// effort-profile default (always `false` today; see
    /// [`EffortProfile::gather_dedup`]).
    ///
    /// **Bytes are not byte-identical to the sort-only path.** Gather-time
    /// dedup hashes on raw i32 property values (pre-quantization runs
    /// later), so the surviving unique set is a strict superset of the
    /// bucket-equivalence set the sort path collapses to. Hash-locks must
    /// be re-baked when sweep harnesses enable this.
    pub gather_dedup: Option<bool>,

    /// Phase 3 of issue #41 — when [`Self::gather_dedup`] is `Some(true)`,
    /// route the gather-time dedup through
    /// `crate::modular::inline_dedup_table::InlineDedupTable` instead of
    /// Phase 2's [`crate::modular::tree_learn::GatherDedupTable`].
    ///
    /// `Some(true)` enables the inline-fingerprint cuckoo table; `Some(false)`
    /// stays on the Phase 2 (SoA-indexed) table; `None` leaves the
    /// effort-profile default (always `false` today; see
    /// [`EffortProfile::gather_dedup_phase3`]).
    ///
    /// Has no effect unless [`Self::gather_dedup`] also routes traffic into
    /// the gather-time dedup path; gather-time dedup is a prerequisite.
    ///
    /// Hash-locks behave identically to Phase 2 (the post-`pre_quantize`
    /// sort path remains the byte-determining arbiter), so flipping this
    /// switch on top of an already-enabled `gather_dedup` does NOT require
    /// re-baking hash_lock sidecars — but it DOES change end-to-end
    /// wall-clock, which is the only reason to use it.
    pub gather_dedup_phase3: Option<bool>,

    /// Maximum depth of parallel recursion in the tree learner
    /// (`tree_learn.rs` `build_subtree_recursive_parallel_borrowed`).
    /// `2^depth` is the upper bound on parallel leaf tasks.
    /// Default schedule: 4 at effort ≤ 7 (16 leaf tasks), 5 at effort ≥ 8
    /// (32 leaf tasks — deeper e8/e9 trees benefit from finer-grained fanout).
    pub tree_parallel_max_depth: Option<u32>,

    /// Minimum subtree size below which recursive parallel fork is skipped
    /// (`tree_learn.rs` `PARALLEL_RECURSION_FLOOR`). Below this sample
    /// count rayon task overhead exceeds the parallel savings.
    /// Default schedule: 16384 at effort ≤ 7, 8192 at effort ≥ 8.
    pub tree_parallel_floor: Option<usize>,

    /// Minimum total sample count to even attempt the parallel root split
    /// (`tree_learn.rs` `PARALLEL_THRESHOLD`). Below this the sequential
    /// loop is faster overall.
    /// Default schedule: 8192 at effort ≤ 7, 4096 at effort ≥ 8.
    pub tree_parallel_root_threshold: Option<usize>,

    /// Override the small-image parallel-tree-learning fallback
    /// (see [`EffortProfile::tree_parallel_small_image_fallback`]).
    ///
    /// `Some(true)`: force the sequential fallback regardless of image
    /// size. `Some(false)`: force the parallel + thread-local-cache path
    /// regardless of image size (the pre-audit default behaviour).
    /// `None`: keep the always-on auto-gate that flips this to `true`
    /// for inputs smaller than [`SMALL_IMAGE_PIXEL_THRESHOLD`] (1 MP).
    ///
    /// Intended for sweep harnesses A/B-ing the gate; production
    /// callers should leave this `None`.
    pub tree_parallel_small_image_fallback: Option<bool>,

    /// Override the number of multi-seed tree-learning runs
    /// (see [`EffortProfile::tree_learn_seeds`]).
    ///
    /// `Some(1)` forces single-pass tree learning (libjxl-equivalent,
    /// byte-identical to the pre-RFC#45-chunk-2 default at any effort).
    /// `Some(N)` with `N >= 2` runs gather→tree `N` times with different
    /// stride offsets and keeps the tree whose tokens have the lowest
    /// entropy cost. `None` keeps the effort-derived default (1 at
    /// e ≤ 9, 2 at e10, 8 at e11 — RFC#45 chunk 5 expanded e11 from 4
    /// so chunk-3 perturbations and chunk-4 dimensions each get
    /// dedicated seed slots).
    ///
    /// Output is bitstream-valid for any `N`. Sweep harnesses re-baking
    /// hash_lock sidecars should be aware that `N >= 2` *can* change the
    /// chosen tree per (image, distance) cell.
    pub tree_learn_seeds: Option<u8>,

    /// EX-J5 reinterpretation — use **Lloyd-Max iterative clustering**
    /// for MA-tree bucket boundaries on the three residual-energy proxy
    /// properties (4 = `|N|`, 5 = `|W|`, 15 = `wp_max_error`).
    ///
    /// `Some(true)` opts in to Lloyd-Max bucket boundaries; `Some(false)`
    /// forces the sort-quantile default; `None` keeps the effort-profile
    /// default (always `false` today; see
    /// [`EffortProfile::lloyd_max_buckets`]).
    ///
    /// **Bytes change** when this flag flips from `false` → `true`
    /// (different candidate splitvals → different chosen tree splits),
    /// so sweep harnesses must re-bake `hash_lock_expected.txt` and
    /// re-validate decoder roundtrip with jxl-rs, jxl-oxide, and djxl.
    /// Bitstream remains 100 % spec-legal — the JXL property set is
    /// untouched; only the candidate splitval shortlist the tree learner
    /// chooses from is refined.
    pub lloyd_max_buckets: Option<bool>,
}

#[cfg(feature = "__expert")]
impl LossyInternalParams {
    /// Apply each `Some(_)` field on top of `profile`.
    pub(crate) fn apply_to(self, profile: &mut EffortProfile) {
        let LossyInternalParams {
            try_dct16,
            try_dct32,
            try_dct64,
            try_dct4x8_afv,
            fine_grained_step,
            k_info_loss_mul_base,
            entropy_mul_table,
            cfl_two_pass,
            chromacity_adjustment,
            patch_ref_tree_learning,
            non_aligned_eval,
            enhanced_clustering_vardct,
            ans_histogram_strategy_vardct,
            k_ac_quant,
            lossy_search_seeds,
        } = self;
        if let Some(v) = try_dct16 {
            profile.try_dct16 = v;
        }
        if let Some(v) = try_dct32 {
            profile.try_dct32 = v;
        }
        if let Some(v) = try_dct64 {
            profile.try_dct64 = v;
        }
        if let Some(v) = try_dct4x8_afv {
            profile.try_dct4x8_afv = v;
        }
        if let Some(v) = fine_grained_step {
            profile.fine_grained_step = v;
        }
        if let Some(v) = k_info_loss_mul_base {
            profile.k_info_loss_mul_base = v;
        }
        if let Some(v) = entropy_mul_table {
            profile.entropy_mul_table = v;
        }
        if let Some(v) = cfl_two_pass {
            profile.cfl_two_pass = v;
        }
        if let Some(v) = chromacity_adjustment {
            profile.chromacity_adjustment = v;
        }
        if let Some(v) = patch_ref_tree_learning {
            profile.patch_ref_tree_learning = v;
        }
        if let Some(v) = non_aligned_eval {
            profile.non_aligned_eval = v;
        }
        if let Some(v) = enhanced_clustering_vardct {
            profile.enhanced_clustering_vardct = v;
        }
        if let Some(v) = ans_histogram_strategy_vardct {
            profile.ans_histogram_strategy_vardct = v;
        }
        if let Some(v) = k_ac_quant {
            profile.k_ac_quant = v;
        }
        if let Some(v) = lossy_search_seeds {
            profile.lossy_search_seeds = v;
        }
    }
}

#[cfg(feature = "__expert")]
impl LosslessInternalParams {
    /// Apply each `Some(_)` field on top of `profile`.
    pub(crate) fn apply_to(self, profile: &mut EffortProfile) {
        let LosslessInternalParams {
            nb_rcts_to_try,
            forced_rct,
            wp_num_param_sets,
            tree_max_buckets,
            tree_num_properties,
            tree_threshold_base,
            tree_sample_fraction,
            tree_max_samples_fixed,
            use_streaming_dedup,
            gather_dedup,
            gather_dedup_phase3,
            tree_parallel_max_depth,
            tree_parallel_floor,
            tree_parallel_root_threshold,
            tree_parallel_small_image_fallback,
            tree_learn_seeds,
            lloyd_max_buckets,
        } = self;
        if let Some(v) = nb_rcts_to_try {
            profile.nb_rcts_to_try = v;
        }
        if forced_rct.is_some() {
            profile.forced_rct = forced_rct;
        }
        if let Some(v) = wp_num_param_sets {
            profile.wp_num_param_sets = v;
        }
        if let Some(v) = tree_max_buckets {
            profile.tree_max_buckets = v;
        }
        if let Some(v) = tree_num_properties {
            profile.tree_num_properties = v;
        }
        if let Some(v) = tree_threshold_base {
            profile.tree_threshold_base = v;
        }
        if let Some(v) = tree_sample_fraction {
            profile.tree_sample_fraction = v;
        }
        if let Some(v) = tree_max_samples_fixed {
            profile.tree_max_samples_fixed = v;
        }
        if let Some(v) = use_streaming_dedup {
            profile.use_streaming_dedup = v;
        }
        if let Some(v) = gather_dedup {
            profile.gather_dedup = v;
        }
        if let Some(v) = gather_dedup_phase3 {
            profile.gather_dedup_phase3 = v;
        }
        if let Some(v) = tree_parallel_max_depth {
            profile.tree_parallel_max_depth = v;
        }
        if let Some(v) = tree_parallel_floor {
            profile.tree_parallel_floor = v;
        }
        if let Some(v) = tree_parallel_root_threshold {
            profile.tree_parallel_root_threshold = v;
        }
        if let Some(v) = tree_parallel_small_image_fallback {
            profile.tree_parallel_small_image_fallback = v;
        }
        if let Some(v) = tree_learn_seeds {
            profile.tree_learn_seeds = v;
        }
        if let Some(v) = lloyd_max_buckets {
            profile.lloyd_max_buckets = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lossy_reference_e7() {
        let p = EffortProfile::lossy(7, EncoderMode::Reference);
        assert_eq!(p.effort, 7);
        assert!(p.use_ans);
        assert!(p.optimize_codes);
        assert!(p.custom_orders);
        assert!(p.gaborish);
        assert!(p.pixel_domain_loss);
        assert!(!p.error_diffusion);
        assert!(p.patches);
        assert!(!p.lz77); // libjxl only enables LZ77 for VarDCT at e9+ (kTortoise)
        assert_eq!(p.butteraugli_iters, 0); // libjxl gates at speed_tier <= kKitten (e8+)
        assert!(p.ac_strategy_enabled);
        assert!(p.try_dct32);
        assert!(p.try_dct64);
        assert!(p.try_dct4x8_afv); // e6+
        assert!(p.non_aligned_eval);
        assert_eq!(p.fine_grained_step, 2);
        assert!(p.chromacity_adjustment); // e7+
        assert!(!p.enhanced_clustering_vardct); // e9+
        assert!(!p.optimize_uint_configs_vardct); // e9+ (libjxl kNone at e<9)
        assert!(p.epf_dynamic_sharpness); // e6+
        assert!(p.cfl_two_pass); // e7+
        assert!(p.cfl_newton); // e7+ with pass 2
        assert!(p.use_adaptive_quant);
        assert!(p.adjust_quant_ac);
        assert_eq!(p.initial_q_numerator, 0.39);
        assert_eq!(p.k_favor_2x2, -0.4);
        assert_eq!(p.k_ac_quant, 0.765);
        assert_eq!(p.nb_rcts_to_try, 7);
        assert_eq!(p.wp_num_param_sets, 0); // e8+
        assert_eq!(p.tree_num_properties, 7);
        assert_eq!(p.tree_max_buckets, 96);
    }

    #[test]
    fn test_lossy_reference_e5() {
        let p = EffortProfile::lossy(5, EncoderMode::Reference);
        assert_eq!(p.effort, 5);
        assert!(p.use_ans);
        assert!(p.gaborish);
        assert!(p.pixel_domain_loss);
        assert!(!p.error_diffusion); // e7+
        assert!(!p.patches); // e7+
        assert!(!p.lz77); // e9+ for VarDCT
        assert!(p.ac_strategy_enabled);
        assert!(p.try_dct32);
        assert!(!p.try_dct64); // e7+
        assert!(!p.try_dct4x8_afv); // e6+
        assert!(!p.non_aligned_eval); // e6+
        assert!(!p.chromacity_adjustment); // e7+
        assert!(!p.enhanced_clustering_vardct); // e9+
        assert!(!p.optimize_uint_configs_vardct); // e9+
        assert!(!p.epf_dynamic_sharpness); // e6+
        assert!(!p.cfl_two_pass); // e7+
        assert!(!p.cfl_newton); // e7+
        assert!(p.use_adaptive_quant);
        assert!(p.adjust_quant_ac);
        assert_eq!(p.initial_q_numerator, 0.39);
        assert_eq!(p.butteraugli_iters, 0); // libjxl gates at speed_tier <= kKitten (e8+)
        assert_eq!(p.nb_rcts_to_try, 4);
        assert_eq!(p.wp_num_param_sets, 0); // e8+
    }

    #[test]
    fn test_lossy_reference_e9() {
        let p = EffortProfile::lossy(9, EncoderMode::Reference);
        assert!(p.lz77); // VarDCT LZ77 enabled at e9+ (kTortoise)
        assert_eq!(p.lz77_method, Lz77Method::Optimal);
        assert_eq!(p.butteraugli_iters, 4);
        // W38-2 wedge #1.1: e9 matches libjxl kTortoise (step=2). step=1 only
        // at our e10+ extension. See `enc_ac_strategy.cc:1046`.
        assert_eq!(p.fine_grained_step, 2);
        assert!(p.enhanced_clustering_vardct); // e9+
        assert!(p.optimize_uint_configs_vardct); // e9+
        assert_eq!(p.nb_rcts_to_try, 19);
        assert_eq!(p.wp_num_param_sets, 5); // e9+
        assert_eq!(p.tree_num_properties, 16);
        assert_eq!(p.tree_max_buckets, 256);
    }

    #[test]
    fn test_lossy_reference_e8() {
        let p = EffortProfile::lossy(8, EncoderMode::Reference);
        assert!(!p.lz77); // libjxl only enables LZ77 for VarDCT at e9+
        assert_eq!(p.lz77_method, Lz77Method::Rle);
        assert_eq!(p.butteraugli_iters, 2);
        assert_eq!(p.fine_grained_step, 2);
        assert!(!p.enhanced_clustering_vardct); // e9+
        assert!(!p.optimize_uint_configs_vardct); // e9+
        assert_eq!(p.wp_num_param_sets, 2); // e8
    }

    #[test]
    fn test_fine_grained_step_libjxl_parity() {
        // W38-2 wedge #1.1: libjxl uses step=2 at speed_tier >= kTortoise
        // (= effort 1..=9 on our scale), step=1 only at speed_tier < kTortoise
        // (kGlacier/kTectonicPlate = our e10+ extension). See
        // `enc_ac_strategy.cc:1046`.
        for effort in 1..=9 {
            let p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert_eq!(
                p.fine_grained_step, 2,
                "e{effort}: libjxl kTortoise+ uses step=2"
            );
        }
        for effort in 10..=12 {
            let p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert_eq!(
                p.fine_grained_step, 1,
                "e{effort}: extended past libjxl kGlacier uses finer step=1"
            );
        }
    }

    #[test]
    fn test_lossy_reference_e3() {
        let p = EffortProfile::lossy(3, EncoderMode::Reference);
        assert!(p.use_ans);
        assert!(p.optimize_codes);
        assert!(!p.gaborish);
        assert!(!p.ac_strategy_enabled);
        assert!(!p.use_adaptive_quant);
        assert!(!p.adjust_quant_ac);
        assert_eq!(p.initial_q_numerator, 0.79);
    }

    #[test]
    fn test_lossless_reference_e7() {
        let p = EffortProfile::lossless(7, EncoderMode::Reference);
        assert!(p.use_ans);
        assert!(p.tree_learning);
        assert!(p.lz77);
        assert_eq!(p.lz77_method, Lz77Method::Rle);
        assert!(p.patches);
        assert!(!p.gaborish); // N/A
        assert!(!p.pixel_domain_loss); // N/A
        assert!(!p.ac_strategy_enabled); // N/A
    }

    #[test]
    fn test_lossless_reference_e4() {
        let p = EffortProfile::lossless(4, EncoderMode::Reference);
        assert!(p.use_ans);
        assert!(!p.tree_learning); // e7+
        assert!(!p.lz77); // e7+
        assert!(!p.patches); // e5+
    }

    #[test]
    fn test_effort_clamp() {
        let p = EffortProfile::lossy(0, EncoderMode::Reference);
        assert_eq!(p.effort, 1);
        // RFC#45 chunk 1: clamp bumped 10 → 11 to admit e10/e11.
        // RFC#45 chunk 2: clamp bumped 11 → 12 to admit e12 (32 iters).
        let p = EffortProfile::lossy(99, EncoderMode::Reference);
        assert_eq!(p.effort, 12);
    }

    #[test]
    fn test_lossy_search_seeds_e10_e11_extended() {
        // RFC#45 chunk 3: multi-seed butteraugli sweep at e10/e11.
        // e ≤ 9 keeps the libjxl single-seed behaviour (bit-identical
        // hash-locks); e10/e11/e12 fan out 2/4/4 seeds (the seed table
        // saturates at 4 — see `init_mul_seeds()`; e12 inherits e11's
        // 4-seed fan-out because chunk 2 differentiates e12 via the
        // `butteraugli_iters` axis, not the seed-count axis) and pick
        // smallest bytes.
        for effort in 1..=9 {
            let p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert_eq!(
                p.lossy_search_seeds, 1,
                "e{effort}: single seed (libjxl-equivalent)"
            );
        }
        let p10 = EffortProfile::lossy(10, EncoderMode::Reference);
        let p11 = EffortProfile::lossy(11, EncoderMode::Reference);
        let p12 = EffortProfile::lossy(12, EncoderMode::Reference);
        assert_eq!(p10.lossy_search_seeds, 2, "e10 = 2× seeds");
        assert_eq!(p11.lossy_search_seeds, 4, "e11 = 4× seeds");
        assert_eq!(
            p12.lossy_search_seeds, 4,
            "e12 = 4× seeds (caps at init_mul_seeds table length)"
        );

        // Lossless never runs the buttloop; field must stay at 1 so a
        // future lossless caller that accidentally checks it doesn't
        // launch a phantom sweep.
        for effort in 1..=12 {
            let p = EffortProfile::lossless(effort, EncoderMode::Reference);
            assert_eq!(
                p.lossy_search_seeds, 1,
                "lossless e{effort}: never fans out"
            );
        }

        // Experimental inherits the reference value.
        let pe11 = EffortProfile::lossy(11, EncoderMode::Experimental);
        let pe12 = EffortProfile::lossy(12, EncoderMode::Experimental);
        assert_eq!(pe11.lossy_search_seeds, 4);
        assert_eq!(pe12.lossy_search_seeds, 4);
    }

    #[test]
    #[cfg(feature = "butteraugli-loop")]
    fn test_init_mul_seeds_invariants() {
        use crate::vardct::butteraugli_loop::{LIBJXL_INIT_MUL, init_mul_seeds};
        // Index 0 must ALWAYS be the libjxl default so multi-seed can
        // never regress below single-seed worst-case.
        for seeds in [1, 2, 3, 4, 5, 10, 99, 255_u8] {
            let table = init_mul_seeds(seeds);
            assert!(!table.is_empty(), "seeds={seeds}: table empty");
            assert!(
                (table[0] - LIBJXL_INIT_MUL).abs() < f64::EPSILON,
                "seeds={seeds}: index 0 ({}) must equal LIBJXL_INIT_MUL ({LIBJXL_INIT_MUL})",
                table[0]
            );
            // Saturation cap: each seed is unique, no NaN/inf, bounded.
            for (i, &v) in table.iter().enumerate() {
                assert!(v.is_finite(), "seeds={seeds}[{i}]: non-finite {v}");
                assert!(
                    (0.0..=1.0).contains(&v),
                    "seeds={seeds}[{i}]: {v} outside [0, 1]"
                );
            }
        }
        // `0` defensively bumps to `1` (same single-seed behaviour).
        assert_eq!(init_mul_seeds(0).len(), 1);
        assert_eq!(init_mul_seeds(1).len(), 1);
        assert_eq!(init_mul_seeds(2).len(), 2);
        assert_eq!(init_mul_seeds(3).len(), 3);
        assert_eq!(init_mul_seeds(4).len(), 4);
        // Saturate at table length so requesting more is safe.
        assert_eq!(init_mul_seeds(255).len(), 4);
    }

    #[test]
    fn test_butteraugli_iters_e10_e11_extended() {
        // RFC#45 chunk 1 + chunk 2: longer butteraugli search budgets at
        // e10/e11/e12 on a power-of-two ladder (8 → 16 → 32). e12 is the
        // new chunk-2 tier and requires the `ITER_MAX = 32` cap (was 16).
        // e9 = libjxl kTortoise max (4 iters).
        let p9 = EffortProfile::lossy(9, EncoderMode::Reference);
        let p10 = EffortProfile::lossy(10, EncoderMode::Reference);
        let p11 = EffortProfile::lossy(11, EncoderMode::Reference);
        let p12 = EffortProfile::lossy(12, EncoderMode::Reference);
        assert_eq!(p9.butteraugli_iters, 4, "e9 = libjxl kTortoise default");
        assert_eq!(p10.butteraugli_iters, 8, "e10 = 2× e9 budget");
        assert_eq!(p11.butteraugli_iters, 16, "e11 = 4× e9 budget");
        assert_eq!(
            p12.butteraugli_iters, 32,
            "e12 = 8× e9, saturated at MAX_QUANT_LOOP_ITERS = 32"
        );
        // Sanity: stays at saturation cap even if effort overshoots.
        // (The lossy() clamp pins at 12; verify the table never returns
        // anything above the loop's structural cap.)
        assert!(
            p12.butteraugli_iters <= crate::api::MAX_QUANT_LOOP_ITERS,
            "butteraugli_iters must not exceed MAX_QUANT_LOOP_ITERS"
        );
        // The cap itself should be 32 in chunk 2 (was 16 in chunk 1).
        assert_eq!(
            crate::api::MAX_QUANT_LOOP_ITERS,
            32,
            "RFC#45 chunk 2 bumps MAX_QUANT_LOOP_ITERS from 16 to 32 to admit e12"
        );
    }

    #[test]
    fn test_experimental_diverges_from_reference() {
        // Experimental should share effort/feature-flag structure with reference
        for effort in 1..=12 {
            let r = EffortProfile::lossy(effort, EncoderMode::Reference);
            let e = EffortProfile::lossy(effort, EncoderMode::Experimental);
            assert_eq!(r.effort, e.effort);
            assert_eq!(r.use_ans, e.use_ans);
            assert_eq!(r.k_favor_2x2, e.k_favor_2x2);
            assert_eq!(r.butteraugli_iters, e.butteraugli_iters);
            assert_eq!(r.nb_rcts_to_try, e.nb_rcts_to_try);
        }

        // Verify specific divergences at effort 7
        let r = EffortProfile::lossy(7, EncoderMode::Reference);
        let e = EffortProfile::lossy(7, EncoderMode::Experimental);

        // k_info_loss_mul_base: 1.2 → 1.3 (PR #4506)
        assert_eq!(r.k_info_loss_mul_base, 1.2);
        assert_eq!(e.k_info_loss_mul_base, 1.3);

        // entropy_mul_table: PR #4506 rebalancing
        assert_eq!(r.entropy_mul_table.dct4x4, 1.08);
        assert_eq!(e.entropy_mul_table.dct4x4, 0.88);
        assert_eq!(r.entropy_mul_table.identity, 1.0428);
        assert_eq!(e.entropy_mul_table.identity, 0.88);
        assert_eq!(r.entropy_mul_table.afv, 0.817_794_9);
        assert_eq!(e.entropy_mul_table.afv, 0.75);
        // Unchanged values should match
        assert_eq!(r.entropy_mul_table.dct8, e.entropy_mul_table.dct8);
        assert_eq!(r.entropy_mul_table.dct16x8, e.entropy_mul_table.dct16x8);
        assert_eq!(r.entropy_mul_table.dct32x32, e.entropy_mul_table.dct32x32);

        // enhanced_clustering_vardct: e9+ → e7+ in experimental
        assert!(!r.enhanced_clustering_vardct); // reference e7: off
        assert!(e.enhanced_clustering_vardct); // experimental e7: on

        // patch_ref_tree_learning: false → true at e7+
        assert!(!r.patch_ref_tree_learning);
        assert!(e.patch_ref_tree_learning);

        // At effort 5, experimental should NOT enable the e7+ features
        let e5 = EffortProfile::lossy(5, EncoderMode::Experimental);
        assert!(!e5.enhanced_clustering_vardct);
        assert!(!e5.patch_ref_tree_learning);
        // But should still have the entropy_mul and info_loss_mul changes
        assert_eq!(e5.k_info_loss_mul_base, 1.3);
        assert_eq!(e5.entropy_mul_table.dct4x4, 0.88);
    }

    #[test]
    fn test_entropy_mul_table_reference_values() {
        // Verify all reference values match libjxl enc_ac_strategy.cc:584
        let t = EntropyMulTable::reference();
        assert_eq!(t.dct8, 0.8);
        assert_eq!(t.dct4x4, 1.08);
        assert_eq!(t.dct4x8, 0.859_316_37);
        assert_eq!(t.identity, 1.0428);
        assert_eq!(t.dct2x2, 0.95);
        assert_eq!(t.afv, 0.817_794_9);
        assert_eq!(t.dct16x8, 1.21);
        assert_eq!(t.dct16x16, 1.34);
        assert_eq!(t.dct16x32, 1.49);
        assert_eq!(t.dct32x32, 1.48);
        assert_eq!(t.dct64x32, 2.25);
        assert_eq!(t.dct64x64, 2.25);
    }

    #[test]
    fn test_entropy_mul_table_experimental_values() {
        // Verify PR #4506 changes and that unchanged values are preserved
        let t = EntropyMulTable::experimental();
        let r = EntropyMulTable::reference();

        // Changed values (PR #4506)
        assert_eq!(t.dct4x4, 0.88); // was 1.08
        assert_eq!(t.identity, 0.88); // was 1.0428
        assert_eq!(t.afv, 0.75); // was 0.818

        // Unchanged values
        assert_eq!(t.dct8, r.dct8);
        assert_eq!(t.dct4x8, r.dct4x8);
        assert_eq!(t.dct2x2, r.dct2x2);
        assert_eq!(t.dct16x8, r.dct16x8);
        assert_eq!(t.dct16x16, r.dct16x16);
        assert_eq!(t.dct16x32, r.dct16x32);
        assert_eq!(t.dct32x32, r.dct32x32);
        assert_eq!(t.dct64x32, r.dct64x32);
        assert_eq!(t.dct64x64, r.dct64x64);
    }

    #[test]
    fn test_entropy_mul_table_screenshot_suppressed_values() {
        // Verify the screen-content discriminator-fed table lifts the
        // four 8x8-class transforms most prone to over-pick on UI/glyph
        // content and leaves every other field bit-identical to
        // `reference()`. Used by `LossyConfig::with_content_aware_entropy_mul`.
        let t = EntropyMulTable::screenshot_suppressed();
        let r = EntropyMulTable::reference();

        // Lifted values (screenshot-suppressed direction).
        assert_eq!(t.identity, 1.85); // was 1.0428 — the dominant wedge
        assert_eq!(t.dct2x2, 1.15); // was 0.95
        assert_eq!(t.afv, 0.95); // was 0.817_794_9
        assert_eq!(t.dct4x8, 0.98); // was 0.859_316_37

        // Every other field MUST match reference (gate must not perturb
        // the libjxl-faithful behaviour on transforms the wedge doesn't
        // hit).
        assert_eq!(t.dct8, r.dct8);
        assert_eq!(t.dct4x4, r.dct4x4);
        assert_eq!(t.dct16x8, r.dct16x8);
        assert_eq!(t.dct16x16, r.dct16x16);
        assert_eq!(t.dct16x32, r.dct16x32);
        assert_eq!(t.dct32x32, r.dct32x32);
        assert_eq!(t.dct64x32, r.dct64x32);
        assert_eq!(t.dct64x64, r.dct64x64);

        // All lifts are strict increases over reference (sanity check
        // that we never accidentally REDUCE a value into the
        // `experimental()` favouring direction).
        assert!(t.identity > r.identity);
        assert!(t.dct2x2 > r.dct2x2);
        assert!(t.afv > r.afv);
        assert!(t.dct4x8 > r.dct4x8);
    }

    #[test]
    fn test_entropy_mul_table_high_d_photo_smooth_suppressed_values() {
        // Verify the W44-29 high-d photo smooth table lowers the large
        // (16x16, 32x32, 16x32/32x16) transforms per the W44-28 sweep
        // top-5 (dct16=1.27, dct32=1.34 — the largest reduction that
        // does NOT trigger the imac_g3 path flip when content-gated)
        // and leaves every other field bit-identical to `reference()`.
        let t = EntropyMulTable::high_d_photo_smooth_suppressed();
        let r = EntropyMulTable::reference();

        // Lowered values (favor large-transform direction).
        assert_eq!(t.dct16x16, 1.27); // was 1.34 (~5.2% cheaper)
        assert_eq!(t.dct32x32, 1.34); // was 1.48 (~9.5% cheaper)
        // dct16x32 scaled with dct32x32 by the libjxl 1.49/1.48 ratio.
        let expected_dct16x32 = 1.34 * (1.49 / 1.48);
        assert!((t.dct16x32 - expected_dct16x32).abs() < 1e-6);

        // Every other field MUST match reference.
        assert_eq!(t.dct8, r.dct8);
        assert_eq!(t.dct4x4, r.dct4x4);
        assert_eq!(t.dct4x8, r.dct4x8);
        assert_eq!(t.identity, r.identity);
        assert_eq!(t.dct2x2, r.dct2x2);
        assert_eq!(t.afv, r.afv);
        assert_eq!(t.dct16x8, r.dct16x8);
        assert_eq!(t.dct64x32, r.dct64x32);
        assert_eq!(t.dct64x64, r.dct64x64);

        // All changes are strict reductions (the W44-29 direction is
        // "make large transforms cheaper", opposite of the W22-1
        // screenshot lift).
        assert!(t.dct16x16 < r.dct16x16);
        assert!(t.dct32x32 < r.dct32x32);
        assert!(t.dct16x32 < r.dct16x32);
    }

    #[test]
    fn test_entropy_mul_table_high_d_photo_smooth_suppressed_z_values() {
        // W44-154 variant Z: dct32x32 = 1.22 (micro-raised from W44-148's
        // 1.24 after W44-153 ledger refresh found 6 pareto FIXED→OPEN
        // flips at 1.24). 1.22 closes 5 of 6 of those cells while
        // preserving 100% of W44-148/152 wins on 1418519 d=5 (gate
        // doesn't fire there) and 100% of codec_wiki d=3 collateral
        // wins. W44-148 originally raised from W44-96's 1.20 after
        // broader d=5/6 measurement showed 1.20 over-fires DCT32X32.
        // Every other field except dct16x32 must match the default
        // suppressed table (dct16x16 unchanged).
        let t = EntropyMulTable::high_d_photo_smooth_suppressed_z();
        let d = EntropyMulTable::high_d_photo_smooth_suppressed();
        let r = EntropyMulTable::reference();

        // Variant Z reductions:
        assert_eq!(t.dct16x16, d.dct16x16); // unchanged at 1.27
        assert_eq!(t.dct32x32, 1.22); // W44-154: 1.24 → 1.22 in variant Z
        let expected_dct16x32 = 1.22 * (1.49 / 1.48);
        assert!((t.dct16x32 - expected_dct16x32).abs() < 1e-6);

        // Strict-lower than default suppressed on the large-DCT axis.
        assert!(t.dct32x32 < d.dct32x32);
        assert!(t.dct16x32 < d.dct16x32);
        // Lower than reference too.
        assert!(t.dct32x32 < r.dct32x32);
        assert!(t.dct16x32 < r.dct16x32);

        // Every OTHER field MUST match reference (same shape as the
        // default suppressed table — variant Z only widens dct32x32).
        assert_eq!(t.dct8, r.dct8);
        assert_eq!(t.dct4x4, r.dct4x4);
        assert_eq!(t.dct4x8, r.dct4x8);
        assert_eq!(t.identity, r.identity);
        assert_eq!(t.dct2x2, r.dct2x2);
        assert_eq!(t.afv, r.afv);
        assert_eq!(t.dct16x8, r.dct16x8);
        assert_eq!(t.dct64x32, r.dct64x32);
        assert_eq!(t.dct64x64, r.dct64x64);
    }

    #[test]
    fn test_entropy_mul_table_high_d_photo_smooth_suppressed_z_high_colour_values() {
        // W44-154 variant Z' (high-colour): dct32x32 = 1.22 (micro-raised
        // from W44-148's 1.24 in parallel with variant Z), dct16x32
        // unchanged at 1.30 (independent W44-98 lift, not scaled with
        // dct32x32). Used to make DCT32X16 / DCT16X32 strictly more
        // expensive than DCT32X32 in the high-colourfulness sub-class of
        // variant Z (currently {1420710}).
        let t = EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour();
        let z = EntropyMulTable::high_d_photo_smooth_suppressed_z();
        let r = EntropyMulTable::reference();

        // Same dct16x16 / dct32x32 as variant Z (only dct16x32 differs).
        assert_eq!(t.dct16x16, z.dct16x16);
        assert_eq!(t.dct32x32, z.dct32x32);
        assert_eq!(t.dct32x32, 1.22); // W44-154: 1.24 → 1.22

        // dct16x32 = 1.30 (LIFTED above variant Z's 1.208 — breaks ratio).
        assert_eq!(t.dct16x32, 1.30);
        // Strict-higher than variant Z on dct16x32 (the lift direction).
        assert!(t.dct16x32 > z.dct16x32);
        // But still below the libjxl reference (1.49) — strict reduction.
        assert!(t.dct16x32 < r.dct16x32);

        // Every OTHER field MUST match reference (same shape as the
        // variant Z table — Z' only lifts dct16x32 vs Z).
        assert_eq!(t.dct8, r.dct8);
        assert_eq!(t.dct4x4, r.dct4x4);
        assert_eq!(t.dct4x8, r.dct4x8);
        assert_eq!(t.identity, r.identity);
        assert_eq!(t.dct2x2, r.dct2x2);
        assert_eq!(t.afv, r.afv);
        assert_eq!(t.dct16x8, r.dct16x8);
        assert_eq!(t.dct64x32, r.dct64x32);
        assert_eq!(t.dct64x64, r.dct64x64);
    }

    #[test]
    fn test_entropy_mul_table_high_d_photo_smooth_suppressed_z_low_colour_values() {
        // W44-154 variant Z'' (low-colour): dct32x32 = 1.22 (micro-raised
        // from W44-148's 1.24 in parallel with variant Z) AND dct16x32
        // unchanged at 1.23 (W44-100 micro-bisect value, kept across
        // W44-148 and W44-154 raises). Used for 1531677-class images
        // (m3_colourfulness < 25) that can't tolerate the stronger 1.30
        // lift (SSIM2 regression -0.34 to -0.93 per W44-98) but benefit
        // from a moderate lift.
        let t = EntropyMulTable::high_d_photo_smooth_suppressed_z_low_colour();
        let z = EntropyMulTable::high_d_photo_smooth_suppressed_z();
        let hc = EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour();
        let r = EntropyMulTable::reference();

        // Same dct16x16 / dct32x32 as variant Z (only dct16x32 differs).
        assert_eq!(t.dct16x16, z.dct16x16);
        assert_eq!(t.dct32x32, z.dct32x32);
        assert_eq!(t.dct32x32, 1.22); // W44-154: 1.24 → 1.22

        // dct16x32 = 1.23 (W44-100 micro-bisect against the original
        // dct32x32=1.20 baseline). The relationship to Z's auto-scaled
        // dct16x32 has flipped twice:
        //   * pre-W44-148 (dct32x32=1.20): Z's dct16x32 = 1.208 < LC's
        //     1.23 → "LC ABOVE Z" (W44-99/100 design intent).
        //   * W44-148 (dct32x32=1.24): Z's dct16x32 ≈ 1.248 > LC's 1.23
        //     → "LC BELOW Z" (inversion, but still Pareto-positive per
        //     W44-148 bisect).
        //   * W44-154 (dct32x32=1.22, this commit): Z's dct16x32 ≈ 1.228
        //     < LC's 1.23 again → "LC ABOVE Z" semantic restored,
        //     mirroring the original W44-99/100 design intent.
        // The W44-99/100 design intent (DCT16X32 still cheaper than
        // DCT32X32 in LC: 1.23 > 1.22) is also preserved.
        assert_eq!(t.dct16x32, 1.23);
        // After W44-154: Z's dct16x32 ≈ 1.228 < LC's 1.23 (re-inverted).
        // The W44-99/100 "LC dct16x32 lifted ABOVE Z" semantic is
        // restored. Strict-higher than Z on the dct16x32 axis.
        assert!(t.dct16x32 > z.dct16x32);
        // Strict-LOWER than high_colour Z' on dct16x32 (this is the
        // milder lift for the low-colour sub-class).
        assert!(t.dct16x32 < hc.dct16x32);
        // Still below the libjxl reference (1.49) — strict reduction.
        assert!(t.dct16x32 < r.dct16x32);

        // Every OTHER field MUST match reference (same shape as the
        // variant Z table — Z'' only lifts dct16x32 vs Z).
        assert_eq!(t.dct8, r.dct8);
        assert_eq!(t.dct4x4, r.dct4x4);
        assert_eq!(t.dct4x8, r.dct4x8);
        assert_eq!(t.identity, r.identity);
        assert_eq!(t.dct2x2, r.dct2x2);
        assert_eq!(t.afv, r.afv);
        assert_eq!(t.dct16x8, r.dct16x8);
        assert_eq!(t.dct64x32, r.dct64x32);
        assert_eq!(t.dct64x64, r.dct64x64);
    }

    #[test]
    fn test_entropy_mul_table_high_d_photo_smooth_suppressed_z_d_high_values() {
        // W44-156 variant Z (d-high): dct32x32 = 1.20 (weaker than the
        // post-W44-154 1.22 in variant Z), used when
        // target_distance > W44_156_VARIANT_Z_D_HIGH_THRESHOLD (5.5).
        // W44-155 per-strategy dump on 1420710 e5 d=6 showed cjxl sheds
        // small blocks at d=5→d=6 transition; the W44-154 1.22 lift is
        // TOO aggressive at d=6 (forces more DCT32X32 consolidation
        // instead of letting smaller blocks win). The pre-W44-148 1.20
        // baseline matches cjxl's strategy distribution better at high d.
        let t = EntropyMulTable::high_d_photo_smooth_suppressed_z_d_high();
        let z = EntropyMulTable::high_d_photo_smooth_suppressed_z();
        let d = EntropyMulTable::high_d_photo_smooth_suppressed();
        let r = EntropyMulTable::reference();

        // d-high variant: dct32x32 strictly LOWER than variant Z (the
        // weaker-lift direction).
        assert_eq!(t.dct16x16, z.dct16x16); // unchanged at 1.27
        assert_eq!(t.dct32x32, 1.20); // W44-156: 1.20 at d > 5.5
        assert!(t.dct32x32 < z.dct32x32); // strict-lower than variant Z
        let expected_dct16x32 = 1.20 * (1.49 / 1.48);
        assert!((t.dct16x32 - expected_dct16x32).abs() < 1e-6);

        // Still strict-lower than default suppressed (variant Z direction).
        assert!(t.dct32x32 < d.dct32x32);
        assert!(t.dct16x32 < d.dct16x32);
        // Lower than reference too.
        assert!(t.dct32x32 < r.dct32x32);
        assert!(t.dct16x32 < r.dct16x32);

        // Every OTHER field MUST match reference (same shape as the
        // default suppressed table — Z d-high only re-tunes dct32x32).
        assert_eq!(t.dct8, r.dct8);
        assert_eq!(t.dct4x4, r.dct4x4);
        assert_eq!(t.dct4x8, r.dct4x8);
        assert_eq!(t.identity, r.identity);
        assert_eq!(t.dct2x2, r.dct2x2);
        assert_eq!(t.afv, r.afv);
        assert_eq!(t.dct16x8, r.dct16x8);
        assert_eq!(t.dct64x32, r.dct64x32);
        assert_eq!(t.dct64x64, r.dct64x64);
    }

    #[test]
    fn test_entropy_mul_table_high_d_photo_smooth_suppressed_z_high_colour_d_high_values() {
        // W44-156 variant Z' (high-colour, d-high): dct32x32 = 1.20
        // (weaker than W44-154 HC's 1.22), dct16x32 unchanged at 1.30
        // (W44-98 independent lift). Mirrors plain Z d-high but keeps
        // the W44-98 dct16x32 = 1.30 lift for the high-colourfulness
        // sub-class. Applies when both W44-98 HC gate fires AND
        // target_distance > 5.5.
        let t = EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour_d_high();
        let hc = EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour();
        let z_d_high = EntropyMulTable::high_d_photo_smooth_suppressed_z_d_high();
        let r = EntropyMulTable::reference();

        // Same dct16x16 / dct32x32 as plain Z d-high (only dct16x32 differs).
        assert_eq!(t.dct16x16, z_d_high.dct16x16);
        assert_eq!(t.dct32x32, z_d_high.dct32x32);
        assert_eq!(t.dct32x32, 1.20); // W44-156: 1.20 at d > 5.5

        // dct32x32 strictly LOWER than HC (the weaker-lift direction).
        assert!(t.dct32x32 < hc.dct32x32);

        // dct16x32 = 1.30 (mirrors HC — W44-98 independent lift).
        assert_eq!(t.dct16x32, 1.30);
        assert_eq!(t.dct16x32, hc.dct16x32);
        // Strict-higher than plain Z d-high on dct16x32 (the HC lift).
        assert!(t.dct16x32 > z_d_high.dct16x32);
        // Still below the libjxl reference (1.49) — strict reduction.
        assert!(t.dct16x32 < r.dct16x32);

        // Every OTHER field MUST match reference.
        assert_eq!(t.dct8, r.dct8);
        assert_eq!(t.dct4x4, r.dct4x4);
        assert_eq!(t.dct4x8, r.dct4x8);
        assert_eq!(t.identity, r.identity);
        assert_eq!(t.dct2x2, r.dct2x2);
        assert_eq!(t.afv, r.afv);
        assert_eq!(t.dct16x8, r.dct16x8);
        assert_eq!(t.dct64x32, r.dct64x32);
        assert_eq!(t.dct64x64, r.dct64x64);
    }

    #[test]
    fn test_entropy_mul_table_high_d_photo_smooth_suppressed_z_low_colour_d_high_values() {
        // W44-156 variant Z'' (low-colour, d-high): dct32x32 = 1.20
        // (weaker than W44-154 LC's 1.22), dct16x32 unchanged at 1.23
        // (W44-100 micro-bisect value). Mirrors plain Z d-high but keeps
        // the W44-99/100 dct16x32 = 1.23 lift for the low-colourfulness
        // sub-class. Applies when both W44-99 LC gate fires AND
        // target_distance > 5.5.
        let t = EntropyMulTable::high_d_photo_smooth_suppressed_z_low_colour_d_high();
        let lc = EntropyMulTable::high_d_photo_smooth_suppressed_z_low_colour();
        let z_d_high = EntropyMulTable::high_d_photo_smooth_suppressed_z_d_high();
        let hc_d_high = EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour_d_high();
        let r = EntropyMulTable::reference();

        // Same dct16x16 / dct32x32 as plain Z d-high.
        assert_eq!(t.dct16x16, z_d_high.dct16x16);
        assert_eq!(t.dct32x32, z_d_high.dct32x32);
        assert_eq!(t.dct32x32, 1.20); // W44-156: 1.20 at d > 5.5

        // dct32x32 strictly LOWER than LC (the weaker-lift direction).
        assert!(t.dct32x32 < lc.dct32x32);

        // dct16x32 = 1.23 (mirrors LC — W44-100 micro-bisect).
        assert_eq!(t.dct16x32, 1.23);
        assert_eq!(t.dct16x32, lc.dct16x32);
        // Strict-higher than plain Z d-high on dct16x32 (LC above Z
        // semantic at d-high, mirroring the d <= 5.5 relationship).
        assert!(t.dct16x32 > z_d_high.dct16x32);
        // Strict-LOWER than HC d-high on dct16x32 (LC milder lift than HC).
        assert!(t.dct16x32 < hc_d_high.dct16x32);
        // Still below the libjxl reference (1.49) — strict reduction.
        assert!(t.dct16x32 < r.dct16x32);

        // Every OTHER field MUST match reference.
        assert_eq!(t.dct8, r.dct8);
        assert_eq!(t.dct4x4, r.dct4x4);
        assert_eq!(t.dct4x8, r.dct4x8);
        assert_eq!(t.identity, r.identity);
        assert_eq!(t.dct2x2, r.dct2x2);
        assert_eq!(t.afv, r.afv);
        assert_eq!(t.dct16x8, r.dct16x8);
        assert_eq!(t.dct64x32, r.dct64x32);
        assert_eq!(t.dct64x64, r.dct64x64);
    }

    #[test]
    fn test_lossless_experimental_matches_reference() {
        // Lossless experimental is currently identical to reference
        for effort in 1..=12 {
            let r = EffortProfile::lossless(effort, EncoderMode::Reference);
            let e = EffortProfile::lossless(effort, EncoderMode::Experimental);
            assert_eq!(r.effort, e.effort);
            assert_eq!(r.use_ans, e.use_ans);
            assert_eq!(r.tree_learning, e.tree_learning);
            assert_eq!(r.lz77, e.lz77);
        }
    }

    #[test]
    fn test_tree_threshold_base_formula() {
        // speed_tier = 10 - effort
        // threshold = 75 + 14 * speed_tier
        let p = EffortProfile::lossy(7, EncoderMode::Reference);
        assert_eq!(p.tree_threshold_base, 75.0 + 14.0 * 3.0); // speed_tier=3
        let p = EffortProfile::lossy(9, EncoderMode::Reference);
        assert_eq!(p.tree_threshold_base, 75.0 + 14.0 * 1.0); // speed_tier=1
        let p = EffortProfile::lossy(5, EncoderMode::Reference);
        assert_eq!(p.tree_threshold_base, 75.0 + 14.0 * 5.0); // speed_tier=5
    }

    /// chunk-2 (`lossless_e8_e9_cliff_2026-05-16.md`): effort-tune the rayon
    /// fanout shape for the parallel tree learner. At e ≤ 7 the schedule
    /// matches the pre-chunk-2 hardcoded constants exactly so the e7 hash
    /// lock and bytes are byte-identical. At e ≥ 8 the deeper trees +
    /// heavier per-leaf work benefit from a deeper fanout + lower floor.
    #[test]
    fn test_tree_parallel_schedule_lossless() {
        for effort in [1u8, 5, 7] {
            let p = EffortProfile::lossless(effort, EncoderMode::Reference);
            assert_eq!(p.tree_parallel_max_depth, 4, "e{}", effort);
            assert_eq!(p.tree_parallel_floor, 16_384, "e{}", effort);
            assert_eq!(p.tree_parallel_root_threshold, 8_192, "e{}", effort);
        }
        for effort in [8u8, 9, 10] {
            let p = EffortProfile::lossless(effort, EncoderMode::Reference);
            assert_eq!(p.tree_parallel_max_depth, 5, "e{}", effort);
            assert_eq!(p.tree_parallel_floor, 8_192, "e{}", effort);
            assert_eq!(p.tree_parallel_root_threshold, 4_096, "e{}", effort);
        }
    }

    #[test]
    fn test_tree_parallel_schedule_lossy_matches_lossless() {
        // Lossy and lossless both surface the parallel-tree-learning knobs
        // (lossy uses tree learning for patch reference frames). The defaults
        // must match so a picker sees one canonical schedule per effort.
        for effort in 1u8..=12 {
            let l = EffortProfile::lossless(effort, EncoderMode::Reference);
            let v = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert_eq!(l.tree_parallel_max_depth, v.tree_parallel_max_depth);
            assert_eq!(l.tree_parallel_floor, v.tree_parallel_floor);
            assert_eq!(
                l.tree_parallel_root_threshold,
                v.tree_parallel_root_threshold
            );
        }
    }

    /// `adapt_to_image` is the per-image smart-fanout rule shipped with the
    /// `smart_fanout_sweep_2026-05-17` chunk-1 investigation. For all
    /// `(effort, pixels)` combos EXCEPT large+e9 (where the per-leaf
    /// subtree-build ceiling dominates) it bumps the schedule to
    /// depth=6 / floor=4096 / root_threshold=4096. Large+e9 keeps the
    /// effort-only schedule.
    #[test]
    fn test_adapt_to_image_smart_fanout_rule() {
        // Small / medium / large @ e7: rule should kick in for all.
        for &pixels in &[262_144u64, 1_048_576, 4_194_304] {
            let mut p = EffortProfile::lossless(7, EncoderMode::Reference);
            p.adapt_to_image(pixels);
            assert_eq!(p.tree_parallel_max_depth, 6, "e7 pixels={pixels}");
            assert_eq!(p.tree_parallel_floor, 4_096, "e7 pixels={pixels}");
            assert_eq!(p.tree_parallel_root_threshold, 4_096, "e7 pixels={pixels}");
        }
        // e8: same as e7 (rule applies to all sizes).
        for &pixels in &[262_144u64, 1_048_576, 4_194_304] {
            let mut p = EffortProfile::lossless(8, EncoderMode::Reference);
            p.adapt_to_image(pixels);
            assert_eq!(p.tree_parallel_max_depth, 6, "e8 pixels={pixels}");
            assert_eq!(p.tree_parallel_floor, 4_096, "e8 pixels={pixels}");
        }
        // e9 large: keep effort-only (depth=5 / floor=8192).
        let mut p = EffortProfile::lossless(9, EncoderMode::Reference);
        p.adapt_to_image(8_000_000);
        assert_eq!(p.tree_parallel_max_depth, 5, "e9 large");
        assert_eq!(p.tree_parallel_floor, 8_192, "e9 large");
        // e9 medium: rule still kicks in.
        let mut p = EffortProfile::lossless(9, EncoderMode::Reference);
        p.adapt_to_image(1_048_576);
        assert_eq!(p.tree_parallel_max_depth, 6, "e9 medium");
        assert_eq!(p.tree_parallel_floor, 4_096, "e9 medium");
    }

    /// `adapt_small_image_fallback` is the always-on per-image gate (NOT
    /// opt-in) that flips `tree_parallel_small_image_fallback` to `true`
    /// for inputs below 1 MP AT EFFORT <= 7. Fixes the cache regression
    /// from `cb5e202` (+0.85% mean) at e7 small without triggering the
    /// inverse regression at e8/e9 where the tree is large enough that
    /// per-call `SplitWorkspace::new` dominates the cache's `borrow_mut`
    /// indirection.
    #[test]
    fn test_adapt_small_image_fallback_threshold() {
        // Default profile starts with fallback OFF.
        for effort in 1u8..=12 {
            let p = EffortProfile::lossless(effort, EncoderMode::Reference);
            assert!(
                !p.tree_parallel_small_image_fallback,
                "default profile must not pre-set fallback (effort={effort})"
            );
            let pl = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(
                !pl.tree_parallel_small_image_fallback,
                "lossy default profile must not pre-set fallback (effort={effort})"
            );
        }

        // Below size threshold AND effort <= 7: gate flips ON.
        for &pixels in &[1u64, 1_024, 262_144, 524_288, 999_999] {
            let mut p = EffortProfile::lossless(7, EncoderMode::Reference);
            p.adapt_small_image_fallback(pixels);
            assert!(
                p.tree_parallel_small_image_fallback,
                "fallback must be ON for pixels={pixels} (< {SMALL_IMAGE_PIXEL_THRESHOLD}) at e7"
            );
        }

        // At/above size threshold: gate stays OFF (regardless of effort).
        for &pixels in &[SMALL_IMAGE_PIXEL_THRESHOLD, 1_048_576, 4_194_304, 8_000_000] {
            for effort in 1u8..=12 {
                let mut p = EffortProfile::lossless(effort, EncoderMode::Reference);
                p.adapt_small_image_fallback(pixels);
                assert!(
                    !p.tree_parallel_small_image_fallback,
                    "fallback must be OFF for pixels={pixels} \
                     (>= {SMALL_IMAGE_PIXEL_THRESHOLD}) at effort={effort}"
                );
            }
        }

        // At small size: gate applies ONLY at effort <= 7. At e8+ the cache
        // dominates per-call alloc and disabling it regresses by 7%+ (audit
        // bench evidence — see effort.rs:adapt_small_image_fallback docs).
        for effort in 1u8..=7 {
            let mut p = EffortProfile::lossless(effort, EncoderMode::Reference);
            p.adapt_small_image_fallback(262_144);
            assert!(
                p.tree_parallel_small_image_fallback,
                "fallback must be ON at effort={effort} for 0.26 MP"
            );
        }
        for effort in 8u8..=10 {
            let mut p = EffortProfile::lossless(effort, EncoderMode::Reference);
            p.adapt_small_image_fallback(262_144);
            assert!(
                !p.tree_parallel_small_image_fallback,
                "fallback must be OFF at effort={effort} for 0.26 MP \
                 (cache helps at high effort — per-call alloc dominates)"
            );
        }
    }

    /// `adapt_tree_max_buckets_for_image` is the always-on per-image
    /// dispatch (audit item #3) that drops `tree_max_buckets` from 256
    /// to [`LARGE_E9_TREE_MAX_BUCKETS`] (192) on large+e9 cells only.
    /// Verifies the gate boundaries on both sides (pixels, effort) and
    /// confirms the rule never fires at e7/e8 or below 4 MP.
    #[test]
    fn test_adapt_tree_max_buckets_for_image_threshold() {
        // Pre-dispatch baseline values (matches tree_max_buckets_for).
        let baseline = |effort: u8| -> u16 {
            match effort {
                0..=4 => 32,
                5 => 48,
                6 => 64,
                7 => 96,
                8 => 128,
                _ => 256,
            }
        };

        // 1. e9 large (>= 4 MP): rule fires, buckets drop to 192.
        for &pixels in &[
            LARGE_IMAGE_PIXEL_THRESHOLD,
            4_194_304,
            8_000_000,
            16_777_216,
        ] {
            let mut p = EffortProfile::lossless(9, EncoderMode::Reference);
            assert_eq!(p.tree_max_buckets, 256, "e9 baseline buckets=256");
            p.adapt_tree_max_buckets_for_image(pixels);
            assert_eq!(
                p.tree_max_buckets, LARGE_E9_TREE_MAX_BUCKETS,
                "e9 pixels={pixels}: must drop to 192"
            );
        }
        // e10 large: same dispatch fires.
        let mut p = EffortProfile::lossless(10, EncoderMode::Reference);
        p.adapt_tree_max_buckets_for_image(8_000_000);
        assert_eq!(p.tree_max_buckets, LARGE_E9_TREE_MAX_BUCKETS, "e10 large");

        // 2. e9 below threshold (< 4 MP): rule does NOT fire, buckets stay 256.
        for &pixels in &[1u64, 1_024, 262_144, 1_048_576, 3_999_999] {
            let mut p = EffortProfile::lossless(9, EncoderMode::Reference);
            p.adapt_tree_max_buckets_for_image(pixels);
            assert_eq!(
                p.tree_max_buckets, 256,
                "e9 pixels={pixels} (< {LARGE_IMAGE_PIXEL_THRESHOLD}): must stay 256"
            );
        }

        // 3. e7/e8 large: rule does NOT fire (effort gate), keep effort default.
        for effort in 1u8..=8 {
            for &pixels in &[
                LARGE_IMAGE_PIXEL_THRESHOLD,
                4_194_304,
                8_000_000,
                16_777_216,
            ] {
                let mut p = EffortProfile::lossless(effort, EncoderMode::Reference);
                let want = baseline(effort);
                p.adapt_tree_max_buckets_for_image(pixels);
                assert_eq!(
                    p.tree_max_buckets, want,
                    "effort={effort} pixels={pixels}: \
                     must stay at baseline {want} (effort < 9)"
                );
            }
        }

        // 4. Cross-product spot check: all (effort, pixels) cells outside
        //    the (effort>=9 AND pixels>=4MP) box leave the profile unchanged.
        for effort in 1u8..=12 {
            for &pixels in &[262_144u64, 1_048_576, 3_999_999] {
                let mut p = EffortProfile::lossless(effort, EncoderMode::Reference);
                let want = baseline(effort);
                p.adapt_tree_max_buckets_for_image(pixels);
                assert_eq!(
                    p.tree_max_buckets, want,
                    "effort={effort} pixels={pixels}: no dispatch fire"
                );
            }
        }
    }

    /// Lossy profile must also honour the dispatch (lossy patches /
    /// reference frames go through tree learning too — the constants
    /// must stay consistent so a single canonical schedule applies).
    #[test]
    fn test_adapt_tree_max_buckets_lossy_profile_parity() {
        for effort in 9u8..=10 {
            let mut pl = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert_eq!(pl.tree_max_buckets, 256);
            pl.adapt_tree_max_buckets_for_image(8_000_000);
            assert_eq!(
                pl.tree_max_buckets, LARGE_E9_TREE_MAX_BUCKETS,
                "lossy e{effort} large: dispatch must apply"
            );
        }
    }

    /// Chunk 1 VarDCT AC strategy dispatch: `adapt_to_image_lossy`
    /// must flip `try_dct64` to `false` only on the
    /// (`pixels < LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD`,
    ///  `distance < LOSSY_LOW_DISTANCE_THRESHOLD`) cell, and only when
    /// effort already had `try_dct64 = true` (effort >= 7).
    #[test]
    fn test_adapt_to_image_lossy_dct64_gate() {
        // 1. Small + low-d at e7+: dispatch fires.
        for effort in 7u8..=10 {
            for &pixels in &[1u64, 1024, 262_144, 499_999] {
                for &distance in &[0.1_f32, 0.5, 1.0, 1.5, 1.999] {
                    let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
                    assert!(p.try_dct64, "baseline e{effort}: try_dct64 must be true");
                    p.adapt_to_image_lossy(pixels, distance);
                    assert!(
                        !p.try_dct64,
                        "e{effort} pixels={pixels} d={distance}: \
                         try_dct64 must drop to false"
                    );
                }
            }
        }

        // 2. Above pixel threshold: no fire.
        for &pixels in &[
            LOSSY_SMALL_IMAGE_PIXEL_THRESHOLD,
            1_048_576,
            4_194_304,
            16_777_216,
        ] {
            for &distance in &[0.5_f32, 1.0, 1.5] {
                let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
                p.adapt_to_image_lossy(pixels, distance);
                assert!(
                    p.try_dct64,
                    "pixels={pixels} d={distance}: must stay true (pixel gate)"
                );
            }
        }

        // 3. At or above distance threshold: no fire.
        for &distance in &[LOSSY_LOW_DISTANCE_THRESHOLD, 2.5_f32, 3.0, 5.0, 10.0] {
            for &pixels in &[1u64, 262_144, 499_999] {
                let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
                p.adapt_to_image_lossy(pixels, distance);
                assert!(
                    p.try_dct64,
                    "pixels={pixels} d={distance}: must stay true (distance gate)"
                );
            }
        }

        // 4. Effort < 7: baseline try_dct64 already false — adapter
        //    must not flip it to true and must not panic.
        for effort in 1u8..=6 {
            let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(
                !p.try_dct64,
                "baseline e{effort}: try_dct64 should be false"
            );
            p.adapt_to_image_lossy(262_144, 1.0);
            assert!(
                !p.try_dct64,
                "e{effort}: adapter must not flip false → true"
            );
        }

        // 5. Lossy "experimental" mode also covered (try_dct64
        //    follows the same effort schedule).
        let mut p = EffortProfile::lossy(7, EncoderMode::Experimental);
        p.adapt_to_image_lossy(262_144, 1.0);
        assert!(
            !p.try_dct64,
            "lossy experimental e7 small + low-d: adapter still fires"
        );
    }

    /// Chunk 2a (issue #43) VarDCT AC strategy dispatch:
    /// `adapt_to_image_lossy` must flip `try_dct32` to `false` only on
    /// the (`pixels < LOSSY_TINY_IMAGE_PIXEL_THRESHOLD`,
    ///  `distance < LOSSY_VERY_LOW_DISTANCE_THRESHOLD`, `effort >= 7`)
    /// cell, and only when `try_dct32` was already true (effort >= 5).
    /// Composes orthogonally with the chunk 1 try_dct64 gate.
    #[test]
    fn test_adapt_to_image_lossy_dct32_gate_chunk2a() {
        // 1. Tiny + very-low-d at e7+: dispatch fires (try_dct32 →
        //    false). Note: chunk 1's try_dct64 also flips on the same
        //    cell (chunk 2a cell ⊂ chunk 1 cell).
        for effort in 7u8..=10 {
            for &pixels in &[1u64, 1024, 65_535, 99_999] {
                for &distance in &[0.01_f32, 0.1, 0.25, 0.4, 0.499] {
                    let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
                    assert!(p.try_dct32, "baseline e{effort}: try_dct32 must be true");
                    p.adapt_to_image_lossy(pixels, distance);
                    assert!(
                        !p.try_dct32,
                        "chunk 2a: e{effort} pixels={pixels} d={distance}: \
                         try_dct32 must drop to false"
                    );
                    // Chunk 1 still fires on the same cell.
                    assert!(
                        !p.try_dct64,
                        "chunk 2a + chunk 1 compose: e{effort} pixels={pixels} d={distance}: \
                         try_dct64 must also drop to false (chunk 2a cell ⊂ chunk 1 cell)"
                    );
                }
            }
        }

        // 2. At or above tiny pixel threshold: no chunk 2a fire
        //    (try_dct32 stays true). Some cells still fire chunk 1
        //    (e.g. 100_000..500_000 px) but try_dct32 untouched.
        for &pixels in &[
            LOSSY_TINY_IMAGE_PIXEL_THRESHOLD,
            262_144,
            499_999,
            1_048_576,
        ] {
            for &distance in &[0.1_f32, 0.25, 0.4] {
                let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
                p.adapt_to_image_lossy(pixels, distance);
                assert!(
                    p.try_dct32,
                    "chunk 2a: pixels={pixels} d={distance}: \
                     try_dct32 must stay true (pixel gate)"
                );
            }
        }

        // 3. At or above very-low-distance threshold: no chunk 2a fire.
        for &distance in &[
            LOSSY_VERY_LOW_DISTANCE_THRESHOLD,
            0.7_f32,
            1.0,
            1.5,
            2.0,
            5.0,
        ] {
            for &pixels in &[1u64, 50_000, 99_999] {
                let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
                p.adapt_to_image_lossy(pixels, distance);
                assert!(
                    p.try_dct32,
                    "chunk 2a: pixels={pixels} d={distance}: \
                     try_dct32 must stay true (distance gate)"
                );
            }
        }

        // 4. Effort < 7: chunk 2a gate does NOT fire even on the
        //    tiny + very-low-d cell. try_dct32 stays at its effort
        //    default (true at effort >= 5, false at effort < 5).
        for effort in 5u8..=6 {
            let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(
                p.try_dct32,
                "baseline e{effort}: try_dct32 must be true (effort >= 5 default)"
            );
            p.adapt_to_image_lossy(50_000, 0.25);
            assert!(
                p.try_dct32,
                "chunk 2a effort gate: e{effort} (< 7): try_dct32 must stay true"
            );
        }
        for effort in 1u8..=4 {
            let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(
                !p.try_dct32,
                "baseline e{effort}: try_dct32 must be false (effort < 5)"
            );
            p.adapt_to_image_lossy(50_000, 0.25);
            assert!(
                !p.try_dct32,
                "chunk 2a: e{effort} (< 5): adapter must not flip false → true"
            );
        }

        // 5. Smoothness hint must NOT affect chunk 2a (only chunk 1
        //    consults the hint). try_dct32 always drops on the chunk
        //    2a cell at effort >= 7 regardless of smooth_photo_hint.
        for smooth_hint in [false, true] {
            let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
            p.adapt_to_image_lossy_with_smoothness(50_000, 0.25, smooth_hint);
            assert!(
                !p.try_dct32,
                "chunk 2a is hint-agnostic: smooth_hint={smooth_hint}: \
                 try_dct32 must drop to false"
            );
        }

        // 6. Experimental mode covered (try_dct32 follows same effort
        //    schedule as Reference).
        let mut p = EffortProfile::lossy(7, EncoderMode::Experimental);
        p.adapt_to_image_lossy(50_000, 0.25);
        assert!(
            !p.try_dct32,
            "lossy experimental e7 tiny + very-low-d: chunk 2a still fires"
        );
    }

    /// W44-35: `adapt_to_image_lossy_with_smoothness(.., true)` must
    /// suppress the `try_dct64 -> false` flip on the gated cell so the
    /// encoder evaluates DCT64 transforms on smooth photos that benefit.
    /// `with_smoothness(.., false)` must be byte-identical to
    /// `adapt_to_image_lossy`.
    #[test]
    fn test_adapt_to_image_lossy_with_smoothness_w44_35() {
        // 1. Smooth-photo hint TRUE on gated cell: dispatch suppressed.
        for effort in 7u8..=10 {
            for &pixels in &[1u64, 1024, 262_144, 499_999] {
                for &distance in &[0.1_f32, 0.5, 1.0, 1.5, 1.999] {
                    let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
                    p.adapt_to_image_lossy_with_smoothness(pixels, distance, true);
                    assert!(
                        p.try_dct64,
                        "smooth_photo=true e{effort} pixels={pixels} d={distance}: \
                         try_dct64 must stay true (W44-35 admission)"
                    );
                }
            }
        }

        // 2. Smooth-photo hint FALSE on gated cell: byte-identical to
        //    `adapt_to_image_lossy` (try_dct64 drops to false).
        for effort in 7u8..=10 {
            let mut p_with = EffortProfile::lossy(effort, EncoderMode::Reference);
            p_with.adapt_to_image_lossy_with_smoothness(262_144, 1.0, false);
            let mut p_without = EffortProfile::lossy(effort, EncoderMode::Reference);
            p_without.adapt_to_image_lossy(262_144, 1.0);
            assert_eq!(
                p_with.try_dct64, p_without.try_dct64,
                "e{effort}: smooth_photo=false must match adapt_to_image_lossy()"
            );
            assert!(
                !p_with.try_dct64,
                "e{effort} smooth_photo=false: try_dct64 must drop"
            );
        }

        // 3. Above pixel threshold: smoothness hint is irrelevant.
        let mut p_smooth = EffortProfile::lossy(7, EncoderMode::Reference);
        p_smooth.adapt_to_image_lossy_with_smoothness(1_048_576, 1.0, true);
        assert!(
            p_smooth.try_dct64,
            "medium image: try_dct64 stays true regardless of hint"
        );

        // 4. At/above distance threshold: smoothness hint is irrelevant.
        let mut p_smooth = EffortProfile::lossy(7, EncoderMode::Reference);
        p_smooth.adapt_to_image_lossy_with_smoothness(262_144, 2.0, true);
        assert!(
            p_smooth.try_dct64,
            "d=2.0: try_dct64 stays true (gate not firing) regardless of hint"
        );

        // 5. Effort 5..=6 (ac_strategy_enabled but try_dct64 default
        //    false): smooth_photo=true admits DCT64 via the W44-35
        //    force-enable. Closes the 1418519 e6 cells.
        for effort in 5u8..=6 {
            let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(!p.try_dct64, "e{effort} baseline: try_dct64 must be false");
            assert!(
                p.ac_strategy_enabled,
                "e{effort} baseline: ac_strategy_enabled must be true"
            );
            p.adapt_to_image_lossy_with_smoothness(262_144, 1.0, true);
            assert!(
                p.try_dct64,
                "e{effort} smooth_photo=true: try_dct64 flipped to true (W44-35)"
            );
        }

        // 6. Effort < 5: AC strategy search disabled; smooth_photo hint
        //    is a no-op (admitting DCT64 wouldn't fire anyway).
        for effort in 1u8..=4 {
            let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(
                !p.ac_strategy_enabled,
                "e{effort} baseline: ac_strategy_enabled must be false"
            );
            p.adapt_to_image_lossy_with_smoothness(262_144, 1.0, true);
            assert!(
                !p.try_dct64,
                "e{effort} smooth_photo=true: try_dct64 stays false (ac search off)"
            );
        }
    }

    /// RFC #45 pick #4 chunk 1 — `adapt_to_image_content` must flip
    /// `patches = true` on Screenshot-class content at e ∈ {5, 6} with
    /// `pixels >= CONTENT_CLASS_MIN_PIXELS`. All other (class, effort,
    /// pixels, distance) tuples must be no-ops.
    #[test]
    fn test_adapt_to_image_content_screenshot_enables_patches_at_e5_e6() {
        // 1. Screenshot at e5/e6, above pixel + distance threshold: fires.
        for effort in [5u8, 6] {
            for &pixels in &[CONTENT_CLASS_MIN_PIXELS, 262_144, 1_048_576, 4_194_304] {
                for &distance in &[0.5_f32, 1.0, 2.0, 5.0] {
                    let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
                    assert!(
                        !p.patches,
                        "baseline e{effort}: patches must be false (gate is e>=7)"
                    );
                    p.adapt_to_image_content(pixels, distance, ImageContentClass::Screenshot);
                    assert!(
                        p.patches,
                        "e{effort} pixels={pixels} d={distance} Screenshot: \
                         patches must flip to true"
                    );
                }
            }
        }

        // 2. Other content classes: no fire at e5/e6.
        for class in [
            ImageContentClass::Unknown,
            ImageContentClass::Photo,
            ImageContentClass::Other,
        ] {
            for effort in [5u8, 6] {
                let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
                p.adapt_to_image_content(262_144, 1.0, class);
                assert!(
                    !p.patches,
                    "e{effort} class={class:?}: patches must stay false"
                );
            }
        }

        // 3. Below pixel threshold: no fire even on Screenshot.
        for &pixels in &[0u64, 1, 1024, 65_535] {
            let mut p = EffortProfile::lossy(5, EncoderMode::Reference);
            p.adapt_to_image_content(pixels, 1.0, ImageContentClass::Screenshot);
            assert!(
                !p.patches,
                "pixels={pixels} Screenshot: pixel gate must hold"
            );
        }

        // 4. distance == 0.0: no fire (lossless-equivalent reserved path).
        let mut p = EffortProfile::lossy(5, EncoderMode::Reference);
        p.adapt_to_image_content(262_144, 0.0, ImageContentClass::Screenshot);
        assert!(!p.patches, "distance=0.0 Screenshot: must stay false");

        // 5. Effort 7+ (patches already on) — adapter is a no-op flag-wise.
        for effort in 7u8..=10 {
            let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(p.patches, "baseline e{effort}: patches must be true");
            p.adapt_to_image_content(262_144, 1.0, ImageContentClass::Screenshot);
            assert!(p.patches, "e{effort} Screenshot: patches must remain true");
        }

        // 6. Effort < 5: adapter must NOT enable patches (libjxl path
        //    needs AC strategy search which is off at e<5).
        for effort in 1u8..=4 {
            let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(!p.patches, "baseline e{effort}: patches must be false");
            p.adapt_to_image_content(262_144, 1.0, ImageContentClass::Screenshot);
            assert!(
                !p.patches,
                "e{effort} Screenshot: must respect effort floor"
            );
        }

        // 7. Default ImageContentClass is Unknown.
        let default_class: ImageContentClass = Default::default();
        assert_eq!(default_class, ImageContentClass::Unknown);
    }

    /// Issue #43 chunk 2c — `adapt_to_image_content` must flip
    /// `try_dct4x8_afv = true` on Screenshot-class content at e == 5 with
    /// `pixels >= CONTENT_CLASS_MIN_PIXELS` and `distance` inside the
    /// measured win band [1.0, 2.0] (inclusive both ends). All other
    /// (class, effort, pixels, distance) tuples must leave the
    /// effort-derived value untouched. Composes with the chunk-1 patches
    /// rule (both fire on in-band e5 Screenshot input).
    ///
    /// The env-disable hook (`JXL_DISPATCH_AFV_SCREENSHOT_DISABLE=1`) is
    /// covered by `tests/it/dispatch_2c_afv_screenshot.rs` (env mutation
    /// requires `unsafe` in edition 2024 — not allowed in the lib crate).
    #[test]
    fn test_adapt_to_image_content_screenshot_afv_lift_e5_chunk2c() {
        // 1. Screenshot at e5, above pixel floor, distance in band:
        //    fires (and the patches rule fires alongside).
        for &pixels in &[CONTENT_CLASS_MIN_PIXELS, 262_144, 1_048_576, 4_194_304] {
            for &distance in &[
                AFV_SCREENSHOT_LIFT_MIN_DISTANCE,
                1.5_f32,
                AFV_SCREENSHOT_LIFT_MAX_DISTANCE,
            ] {
                let mut p = EffortProfile::lossy(5, EncoderMode::Reference);
                assert!(
                    !p.try_dct4x8_afv,
                    "baseline e5: try_dct4x8_afv must be false (gate is e>=6)"
                );
                p.adapt_to_image_content(pixels, distance, ImageContentClass::Screenshot);
                assert!(
                    p.try_dct4x8_afv,
                    "e5 pixels={pixels} d={distance} Screenshot: \
                     try_dct4x8_afv must flip to true"
                );
                assert!(p.patches, "chunk-1 patches rule must still fire");
            }
        }

        // 1b. Out-of-band distances: no fire (the band IS the measured
        //     win region — d=0.5 and d=4.0 measured mixed-to-regressive
        //     bytes on gb82-sc; see the 2026-06-10 bench).
        for &distance in &[0.0_f32, 0.5, 0.99, 2.01, 4.0, 5.0] {
            let mut p = EffortProfile::lossy(5, EncoderMode::Reference);
            p.adapt_to_image_content(262_144, distance, ImageContentClass::Screenshot);
            assert!(
                !p.try_dct4x8_afv,
                "e5 d={distance} Screenshot: out-of-band must not fire"
            );
        }

        // 2. e6+: the effort-derived default is already true; the adapter
        //    must leave it true (the `!self.try_dct4x8_afv` guard makes
        //    the rule a structural no-op — this is also why the original
        //    chunk-2c spec "auto-enable at e>=6" had no effect to add).
        for effort in 6u8..=10 {
            let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(
                p.try_dct4x8_afv,
                "baseline e{effort}: try_dct4x8_afv must be true (e>=6 default)"
            );
            p.adapt_to_image_content(262_144, 1.0, ImageContentClass::Screenshot);
            assert!(
                p.try_dct4x8_afv,
                "e{effort} Screenshot: try_dct4x8_afv must remain true"
            );
        }

        // 3. Effort < 5: must NOT enable (AC strategy machinery limited).
        for effort in 1u8..=4 {
            let mut p = EffortProfile::lossy(effort, EncoderMode::Reference);
            assert!(!p.try_dct4x8_afv, "baseline e{effort}");
            p.adapt_to_image_content(262_144, 1.0, ImageContentClass::Screenshot);
            assert!(
                !p.try_dct4x8_afv,
                "e{effort} Screenshot: must respect the e==5 scope"
            );
        }

        // 4. Other content classes at e5: no fire.
        for class in [
            ImageContentClass::Unknown,
            ImageContentClass::Photo,
            ImageContentClass::Other,
        ] {
            let mut p = EffortProfile::lossy(5, EncoderMode::Reference);
            p.adapt_to_image_content(262_144, 1.0, class);
            assert!(
                !p.try_dct4x8_afv,
                "e5 class={class:?}: try_dct4x8_afv must stay false"
            );
        }

        // 5. Below pixel threshold: no fire even on Screenshot (this is
        //    the hash-lock fixture guard — largest fixture is 48x48).
        for &pixels in &[0u64, 1, 2_304, 65_535] {
            let mut p = EffortProfile::lossy(5, EncoderMode::Reference);
            p.adapt_to_image_content(pixels, 1.0, ImageContentClass::Screenshot);
            assert!(!p.try_dct4x8_afv, "pixels={pixels}: pixel gate must hold");
        }

        // 6. distance == 0.0: no fire.
        let mut p = EffortProfile::lossy(5, EncoderMode::Reference);
        p.adapt_to_image_content(262_144, 0.0, ImageContentClass::Screenshot);
        assert!(!p.try_dct4x8_afv, "distance=0.0: must stay false");
    }

    /// Issue #43 chunk 2c — pure env-value contract for the disable hook:
    /// only the exact string `"1"` disables; unset / empty / `"0"` /
    /// anything else keeps the lift active.
    #[test]
    fn test_afv_screenshot_lift_disable_value_chunk2c() {
        assert!(afv_screenshot_lift_disable_value(Some("1")));
        assert!(!afv_screenshot_lift_disable_value(None));
        assert!(!afv_screenshot_lift_disable_value(Some("")));
        assert!(!afv_screenshot_lift_disable_value(Some("0")));
        assert!(!afv_screenshot_lift_disable_value(Some("true")));
        assert!(!afv_screenshot_lift_disable_value(Some("11")));
    }

    /// W44-133 Chunk G: `EffortProfile::apply_section_a_effort_gates`
    /// must flip the 3 Section A fields to the libjxl threshold when
    /// `EncoderStrategy::Libjxl` is selected. `EffortGate::Ours` (the
    /// default) must preserve the pre-Chunk-G byte values AND must NOT
    /// re-evaluate fields that earlier per-image adapters set (e.g.
    /// `adapt_to_image_lossy_with_smoothness` flipping `try_dct64` to
    /// `false` on small + low-d cells).
    #[test]
    fn test_apply_section_a_effort_gates_ours_preserves_default() {
        // Default (Ours) at e7: all 3 gates fire (matching lossy_reference)
        let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
        let resolved = crate::api::ResolvedImprovements::default();
        let pre_cfl = p.cfl_two_pass;
        let pre_dct64 = p.try_dct64;
        let pre_epf = p.epf_dynamic_sharpness;
        p.apply_section_a_effort_gates(&resolved);
        assert_eq!(p.cfl_two_pass, pre_cfl);
        assert_eq!(p.try_dct64, pre_dct64);
        assert_eq!(p.epf_dynamic_sharpness, pre_epf);
        assert!(p.cfl_two_pass); // e7 with Ours
        assert!(p.try_dct64); // e7 with Ours
        assert!(p.epf_dynamic_sharpness); // e7 with Ours
    }

    /// Verifies the NO-OP semantic for `EffortGate::Ours`: if a prior
    /// adapter (smart-dispatch / __expert override) flipped a Section A
    /// field, `Ours` must NOT clobber it back to the lossy_reference
    /// threshold value.
    #[test]
    fn test_apply_section_a_effort_gates_ours_does_not_clobber_smart_dispatch() {
        // Simulate W44-34/35 smart-dispatch: small + low-d image at
        // e7 drops try_dct64 → false. With strategy = Ours (default),
        // `apply_section_a_effort_gates` must leave try_dct64 at false.
        let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
        p.adapt_to_image_lossy(256 * 256, 1.0);
        assert!(!p.try_dct64, "smart-dispatch: try_dct64 dropped");

        let ours = crate::api::ResolvedImprovements::default(); // Ours on all 3
        p.apply_section_a_effort_gates(&ours);
        assert!(
            !p.try_dct64,
            "Ours strategy must preserve smart-dispatch's try_dct64=false"
        );
    }

    #[test]
    fn test_apply_section_a_effort_gates_libjxl_widens() {
        // At e5: Ours gates all 3 to FALSE; Libjxl widens:
        //  - cfl_two_pass: libjxl >= 5 → true at e5
        //  - try_dct64: libjxl no effort gate → true at e5
        //  - epf_dynamic_sharpness: libjxl no effort gate → true at e5
        let mut p = EffortProfile::lossy(5, EncoderMode::Reference);
        assert!(!p.cfl_two_pass); // ours: e5 < 7
        assert!(!p.try_dct64); // ours: e5 < 7
        assert!(!p.epf_dynamic_sharpness); // ours: e5 < 6

        let libjxl = crate::api::ResolvedImprovements {
            cfl_two_pass_min_effort: crate::api::EffortGate::Libjxl,
            try_dct64_min_effort: crate::api::EffortGate::Libjxl,
            epf_dynamic_sharpness_min_effort: crate::api::EffortGate::Libjxl,
            ..Default::default()
        };
        p.apply_section_a_effort_gates(&libjxl);
        assert!(p.cfl_two_pass, "Libjxl: cfl_two_pass fires at e5");
        assert!(
            p.try_dct64,
            "Libjxl: try_dct64 fires at e5 (no effort gate)"
        );
        assert!(
            p.epf_dynamic_sharpness,
            "Libjxl: epf_dynamic_sharpness fires at e5 (no effort gate)"
        );
    }

    #[test]
    fn test_apply_section_a_effort_gates_off_always_fires() {
        // `Off` semantics: always evaluate (true)
        let mut p = EffortProfile::lossy(1, EncoderMode::Reference);
        let off = crate::api::ResolvedImprovements {
            cfl_two_pass_min_effort: crate::api::EffortGate::Off,
            try_dct64_min_effort: crate::api::EffortGate::Off,
            epf_dynamic_sharpness_min_effort: crate::api::EffortGate::Off,
            ..Default::default()
        };
        p.apply_section_a_effort_gates(&off);
        assert!(p.cfl_two_pass);
        assert!(p.try_dct64);
        assert!(p.epf_dynamic_sharpness);
    }

    #[test]
    fn test_apply_section_a_effort_gates_at_least_custom() {
        // `AtLeast(n)` semantics: custom threshold
        let mut p = EffortProfile::lossy(4, EncoderMode::Reference);
        let custom = crate::api::ResolvedImprovements {
            cfl_two_pass_min_effort: crate::api::EffortGate::AtLeast(4),
            try_dct64_min_effort: crate::api::EffortGate::AtLeast(5),
            epf_dynamic_sharpness_min_effort: crate::api::EffortGate::AtLeast(3),
            ..Default::default()
        };
        p.apply_section_a_effort_gates(&custom);
        assert!(p.cfl_two_pass); // e4 >= 4
        assert!(!p.try_dct64); // e4 < 5
        assert!(p.epf_dynamic_sharpness); // e4 >= 3
    }

    /// W44-184: `apply_section_c_cfl_newton_libjxl_parity` is a no-op
    /// on the `cfl_newton_libjxl_parity` flag when the resolved field
    /// is `false` (preserves the W44-183-shipped default-path behaviour
    /// byte-identically). Zenjxl / LeanFaster / Aggressive /
    /// Custom-with-default-flag all produce `cfl_newton_libjxl_parity =
    /// false` here. W44-AUDIT-5 Phase 2 (Mode C) DOES flip the
    /// `cfl_newton_libjxl_math_with_ls_warm_start` bit on default
    /// (because Zenjxl/Aggressive default-flipped that field to `true`),
    /// so this test focuses on the `libjxl_parity` field which stays
    /// false on the default path.
    #[test]
    fn test_apply_section_c_cfl_newton_libjxl_parity_default_is_noop() {
        let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
        assert!(!p.cfl_newton_libjxl_parity);
        let resolved = crate::api::ResolvedImprovements::default();
        assert!(!resolved.cfl_newton_libjxl_parity);
        p.apply_section_c_cfl_newton_libjxl_parity(&resolved);
        assert!(
            !p.cfl_newton_libjxl_parity,
            "default resolved.cfl_newton_libjxl_parity must NOT flip the profile bit"
        );
    }

    /// W44-AUDIT-5 Phase 2 (Mode C): HONEST-STOP — default Zenjxl
    /// resolved keeps Mode C = `false` (opt-in only). The 3-mode bisect
    /// measured Mode C byte-identical to Mode A on codec_wiki e7 d=4 +
    /// 2 photos, so the default-flip was reverted. The bit remains
    /// reachable via env hook `JXL_W44_AUDIT_5_FORCE_LS_WARM_START=1`
    /// for A/B debugging on any strategy.
    #[test]
    fn test_apply_section_c_mode_c_default_off_zenjxl_path() {
        let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
        assert!(!p.cfl_newton_libjxl_math_with_ls_warm_start);
        let resolved = crate::api::ResolvedImprovements::default();
        assert!(
            !resolved.cfl_newton_libjxl_math_with_ls_warm_start,
            "Zenjxl-default resolved must keep Mode C = false (W44-AUDIT-5 Phase 2 HONEST-STOP)"
        );
        p.apply_section_c_cfl_newton_libjxl_parity(&resolved);
        assert!(
            !p.cfl_newton_libjxl_math_with_ls_warm_start,
            "default resolved must NOT flip the profile's Mode C bit (opt-in only)"
        );
    }

    /// W44-AUDIT-5 Phase 2 (Mode C): when the caller explicitly sets
    /// the field on `EncoderImprovementsCustom`, the
    /// `apply_section_c_cfl_newton_libjxl_parity` adapter flips the
    /// profile bit. Mirrors the `_libjxl_flips` test for `cfl_newton_libjxl_parity`.
    #[test]
    fn test_apply_section_c_mode_c_explicit_opt_in_flips() {
        let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
        assert!(!p.cfl_newton_libjxl_math_with_ls_warm_start);
        let resolved = crate::api::ResolvedImprovements {
            cfl_newton_libjxl_math_with_ls_warm_start: true,
            ..Default::default()
        };
        p.apply_section_c_cfl_newton_libjxl_parity(&resolved);
        assert!(
            p.cfl_newton_libjxl_math_with_ls_warm_start,
            "explicit Mode C opt-in must flip the profile bit"
        );
    }

    /// W44-184: `apply_section_c_cfl_newton_libjxl_parity` flips the
    /// profile bit when the resolved field is `true` (set by
    /// `EncoderStrategy::Libjxl` ONLY).
    #[test]
    fn test_apply_section_c_cfl_newton_libjxl_parity_libjxl_flips() {
        let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
        assert!(!p.cfl_newton_libjxl_parity);
        let resolved = crate::api::ResolvedImprovements {
            cfl_newton_libjxl_parity: true,
            ..Default::default()
        };
        p.apply_section_c_cfl_newton_libjxl_parity(&resolved);
        assert!(
            p.cfl_newton_libjxl_parity,
            "resolved.cfl_newton_libjxl_parity = true must flip the profile bit"
        );
    }

    /// W44-184: when the profile bit is ALREADY `true` (set by some prior
    /// adapter — not currently possible, but forward-compat), a default
    /// resolved value must NOT clobber it back to `false`. Mirrors the
    /// `EffortGate::Ours` NO-OP semantic for Section A.
    #[test]
    fn test_apply_section_c_cfl_newton_libjxl_parity_default_preserves_prior_true() {
        let mut p = EffortProfile::lossy(7, EncoderMode::Reference);
        p.cfl_newton_libjxl_parity = true; // simulate a prior adapter flip
        let resolved = crate::api::ResolvedImprovements::default();
        p.apply_section_c_cfl_newton_libjxl_parity(&resolved);
        assert!(
            p.cfl_newton_libjxl_parity,
            "default resolved field must NOT clobber a prior-flipped profile bit"
        );
    }

    #[test]
    fn test_effort_gate_evaluate() {
        use crate::api::EffortGate;
        // Ours: effort >= ours_min
        assert!(EffortGate::Ours.evaluate(7, 7, 5));
        assert!(!EffortGate::Ours.evaluate(6, 7, 5));
        // Libjxl: effort >= libjxl_min
        assert!(EffortGate::Libjxl.evaluate(5, 7, 5));
        assert!(!EffortGate::Libjxl.evaluate(4, 7, 5));
        // Off: always true
        assert!(EffortGate::Off.evaluate(1, 7, 5));
        assert!(EffortGate::Off.evaluate(0, 7, 5));
        // AtLeast: explicit threshold
        assert!(EffortGate::AtLeast(3).evaluate(3, 7, 5));
        assert!(!EffortGate::AtLeast(3).evaluate(2, 7, 5));
    }
}
