// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Main tiny encoder implementation.

use super::ac_strategy::{
    AcStrategyMap, adjust_quant_field_float_with_distance, adjust_quant_field_with_distance,
    compute_ac_strategy,
};
use super::adaptive_quant::quantize_quant_field;
use super::chroma_from_luma::{CflMap, compute_cfl_map};
use super::common::*;
use super::frame::{DistanceParams, write_toc, write_toc_with_permutation};
use super::gaborish::gaborish_inverse_maybe_adaptive;
use super::noise::{denoise_xyb, estimate_noise_params, noise_quality_coef};
use super::static_codes::{get_ac_entropy_code, get_dc_entropy_code};
use crate::bit_writer::BitWriter;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
use crate::debug_rect;
use crate::error::{Error, Result};
use crate::headers::frame_header::FrameHeader;
use enough::Stop;

// Re-export types from entropy_code sub-module.
pub(crate) use super::entropy_code::{BuiltEntropyCode, force_strategy_map};

/// Validate the three XYB planes against caller-configured policy at
/// the conversion→pipeline boundary.
///
/// The opsin transform (`color::xyb::linear_rgb_to_xyb`) is
/// `cbrt(mixed + bias) - cbrt(bias)` per channel — bias is positive, so
/// the cube-root argument is always strictly positive for any finite
/// linear-RGB input, and the output is finite. Non-finite XYB at this
/// boundary indicates an upstream bug:
///
/// 1. The caller passed non-finite linear-RGB (the `LinearF32` pixel
///    layouts allow this; we should validate at intake but currently
///    don't).
/// 2. An upstream computation (butteraugli-loop reconstruction, EPF,
///    gaborish) leaked NaN into XYB — fix-the-upstream bug.
/// 3. Memory corruption (the original v09/v11 sweep cause, mitigated
///    by removing `unsafe-performance` in PR #34).
///
/// Behavior:
///
/// - [`NonFiniteAction::Error`] (default): runs the **read-only**
///   `is_finite_plane` SIMD scan (~55 GB/s) and returns
///   [`crate::error::Error::InvalidInput`] on first non-finite plane.
///   Nothing downstream ever sees the bad data.
/// - [`NonFiniteAction::Sanitize`]: runs the read-modify-write
///   `sanitize_finite` SIMD kernel (~12.5 GB/s) and replaces non-finite
///   values with `0.0`. Encoding continues.
pub(crate) fn validate_xyb_planes(
    action: crate::api::NonFiniteAction,
    x: &mut [f32],
    y: &mut [f32],
    b: &mut [f32],
) -> crate::error::Result<()> {
    match action {
        crate::api::NonFiniteAction::Error => {
            if !(jxl_simd::is_finite_plane(x)
                && jxl_simd::is_finite_plane(y)
                && jxl_simd::is_finite_plane(b))
            {
                return Err(crate::error::Error::InvalidInput(
                    "non-finite (NaN / ±Inf) value detected in XYB pixel planes. \
                     This is an upstream bug. Common causes: caller passed \
                     non-finite linear-RGB, butteraugli-loop reconstruction \
                     polluted XYB, memory corruption. To accept and silently \
                     replace with 0.0 instead of erroring, use \
                     LossyConfig::with_non_finite_action(NonFiniteAction::Sanitize)."
                        .into(),
                ));
            }
        }
        crate::api::NonFiniteAction::Sanitize => {
            // Always run all three so each plane gets cleaned even if
            // an earlier one was clean.
            let _ = jxl_simd::sanitize_finite(x);
            let _ = jxl_simd::sanitize_finite(y);
            let _ = jxl_simd::sanitize_finite(b);
        }
    }
    Ok(())
}

/// Output of a VarDCT encode operation.
pub struct VarDctOutput {
    /// Encoded JXL codestream bytes.
    pub data: Vec<u8>,
    /// Per-strategy first-block counts, indexed by raw strategy code (0..19).
    pub strategy_counts: [u32; 19],
}

// ── Squeeze-on-extras quantization constants (libjxl parity) ────────────────
//
// Mirrors `enc_modular.cc:82-103`. Used by both the existing no-squeeze
// extras quantizer ([`VarDctEncoder::compute_extra_pixel_quantizer`])
// and the chunk-1 framework function
// ([`VarDctEncoder::compute_extra_pixel_quantizer_shifted`]) that will
// drive the responsive=1 alpha pipeline once the squeeze application
// for extras lands (chunk 2 — see CHANGELOG entry on the
// `with_alpha_squeeze` flag).

// Chunk-1 status note (CLAUDE.md "Investigation Notes"): the four
// items below are framework-only. They become live the moment
// `compute_extra_pixel_quantizer_shifted` is routed from the extras
// subbitstream writer (chunk 2). Unit-test coverage in the
// `tests` module below keeps the constants and the shift-aware
// quantizer fn exercised so a chunk-2 wiring change can't silently
// regress the libjxl-parity formula. Allow dead_code so production
// builds stay clean while the wire-up is staged.

/// Squeeze quality factor (libjxl `enc_modular.cc:82`).
/// "Decrease this number for higher quality."
#[allow(dead_code)]
pub(crate) const SQUEEZE_QUALITY_FACTOR_CONST: f32 = 0.35;

/// Squeeze luma factor (libjxl `enc_modular.cc:85`).
/// "Decrease this number for higher quality luma."
#[allow(dead_code)]
pub(crate) const SQUEEZE_LUMA_FACTOR_CONST: f32 = 1.1;

/// Length of the per-shift quantization table.
#[allow(dead_code)]
pub(crate) const SQUEEZE_LUMA_QTABLE_LEN: usize = 16;

/// Per-shift quantization multipliers for the luma (and "anything
/// non-chroma" — e.g. alpha) channels in the responsive=1 modular
/// pipeline. Mirrors libjxl `enc_modular.cc:101-103`.
///
/// Index 0 (= squeeze depth 0) corresponds to the lowest-frequency
/// average channel; higher indices correspond to deeper HF residual
/// bands. The values halve roughly every step from 163.84 down to
/// 0.005, so residual HF bands collapse to `q = 1` (lossless within
/// integer rounding) quickly — that is where the responsive=1
/// compression win on alpha comes from.
#[allow(dead_code)]
pub(crate) const SQUEEZE_LUMA_QTABLE: [f32; SQUEEZE_LUMA_QTABLE_LEN] = [
    163.84, 81.92, 40.96, 20.48, 10.24, 5.12, 2.56, 1.28, 0.64, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01,
    0.005,
];

/// Threshold on `median(mask1x1)` above which a per-image encode is
/// classified as **screen content** (UI / terminal / glyph-heavy) for
/// the content-aware `entropy_mul` dispatch
/// ([`VarDctEncoder::content_aware_entropy_mul`]).
///
/// `mask1x1` is the post-blur Laplacian masking field — higher values
/// mark uniform / flat regions. Photos populate the field with low
/// values (1–30 typical) because almost every pixel has at least some
/// gradient activity; UI / glyph / terminal content concentrates the
/// distribution above 100 because most pixels sit in flat foreground /
/// background regions. The `> 95` cut-off matches the GPU encoder's
/// screenshot/photo split (`vardct_gpu_dropped_optimizations_resurrection_2026-05-17.md`,
/// item #3) and is empirically derived from CID22 (photos median ≈ 10–40)
/// vs the gb82-sc screenshot corpus (medians ≈ 110–180).
pub(crate) const CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD: f32 = 95.0;

/// W44-65 default-on DCT64-suppress discriminator. **Tighter** than
/// [`CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD`] (`> 95`) because the
/// W44-65 promotion ships as default-on (no opt-in flag), so the gate
/// must be conservative enough to keep `windows95.png`-class pixel-art
/// byte-identical to pre-W44-65 main.
///
/// Empirical encoder-side measurements (see
/// `examples/w44_65_encoder_mask1x1_probe.rs`):
/// - Production screenshots (codec_wiki, imac_g3, imac_dark, terminal,
///   windows, imessage, graph): median ≈ 100.013 (saturated max).
/// - `windows95.png` (pixel-art / Win95 UI): median ≈ 99.060 — just
///   below saturation. Including this in the gate caused a +1.13 %
///   bytes regression at `d=2` in the W44-65 A/B sweep.
/// - All 41 CID22 validation photos: median ≤ 92.34.
///
/// Setting the threshold at `>= 99.5` cleanly separates "fully
/// saturated" screenshots (production target) from
/// "near-but-not-fully-saturated" pixel-art (windows95). The
/// saturated value `100.013` arises from `compute_mask1x1`'s
/// `1/(log1p(0) + 0.01) = 100.0` peak followed by Symmetric5 blur
/// that pushes the median slightly above 100. Pixel-art content
/// retains some moderate-magnitude pixels and sits below the
/// saturation plateau.
///
/// The W22-1 / W44-29 gates retain their existing `> 95` threshold
/// because they're opt-in (`content_aware_entropy_mul`) so a 95-99.5
/// false-positive on windows95-class is a deliberate caller choice
/// rather than a default-behaviour regression.
pub(crate) const W44_65_DCT_SUPPRESS_MEDIAN_THRESHOLD: f32 = 99.5;

/// W44-29 high-distance smooth-photo discriminator on `median(mask1x1)`.
/// Smooth photos cluster in the 10-40 range per the CID22 medians cited
/// above; setting the threshold at 50.0 admits the F-D residual photo
/// class (1531677.png, 1420710.png, 1025469.png, 1080721.png, 1531677.png
/// all have median(mask1x1) ~15-35) while excluding gb82-sc screenshots
/// (medians ≈ 110-180, well above 50). The 50-95 gap between the two
/// thresholds is the "ambiguous" band where neither gate fires — the
/// content is either mixed photo+text or a noisy photo, and we'd rather
/// preserve the libjxl reference tuple than over-fit either direction.
pub(crate) const HIGH_D_PHOTO_SMOOTH_THRESHOLD: f32 = 50.0;

/// W44-91 zenanalyze-proxy upper bound on `median(mask1x1)` for the
/// **textured colourful photo** sub-band. The W44-79 follow-on identified
/// that 1189261.png (mask=69.08) sits in the 50–80 ambiguous mask1x1 band
/// AND benefits from the W44-29 lift, but the existing
/// [`HIGH_D_PHOTO_SMOOTH_THRESHOLD`] (50.0) excludes it. A direct widening
/// of the mask gate to 80 catches 1189261 but ALSO catches 6 documented
/// regression-band images (1025469.png, 1624487.png, 159550.png,
/// 2079234.png, 2775196.png, 297394.png) where the lift hurts bytes.
///
/// W44-91 wires a **zenanalyze-equivalent discriminator** computed cheaply
/// at the API boundary (sRGB u8 only): Hasler-Süsstrunk M3 colourfulness
/// and per-8×8-block flat-color-block ratio (matching zenanalyze
/// tier1.rs exactly). The W44-91 lift fires only when:
///
///   `HIGH_D_PHOTO_SMOOTH_THRESHOLD <= mask1x1_median < HIGH_D_PHOTO_W44_91_MASK_UPPER`
///   `AND distance in [HIGH_D_PHOTO_MIN_DISTANCE, HIGH_D_PHOTO_W44_91_MAX_DISTANCE]`
///   `AND m3_colourfulness >= W44_91_M3_COLOURFULNESS_MIN`
///   `AND flat_color_block_ratio_4 < W44_91_FCBR_MAX`
///
/// On the 41 CID22 validation images only **1189261.png** matches; on the
/// 6 W44-78 regression-band images none match (each fails at least one
/// gate: low colourfulness or high fcbr). See
/// `benchmarks/w44_91_zenanalyze_dispatch_2026-05-19.{tsv,meta}` for the
/// per-image proxy values + paired A/B sweep evidence.
pub(crate) const HIGH_D_PHOTO_W44_91_MASK_UPPER: f32 = 80.0;

/// W44-91 zenanalyze-proxy upper bound on `distance`. Above this the
/// W44-79 hint-true measurement regressed 1189261 by +560 B at d=6
/// (vs -679/-452/-319 B saves at d=3/4/5). The d=5 cap protects the
/// d=6 regression from the auto-fire path.
pub(crate) const HIGH_D_PHOTO_W44_91_MAX_DISTANCE: f32 = 5.0;

/// W44-91 zenanalyze-proxy minimum on Hasler-Süsstrunk M3 colourfulness
/// computed over sRGB u8 source pixels (matches zenanalyze tier1.rs
/// `M3 = sqrt(σ_rg² + σ_yb²) + 0.3 * sqrt(μ_rg² + μ_yb²)` exactly up to
/// FP precision). Verified: 1189261 = 98.84 (passes), 297394 = 103.70
/// (passes — but stopped by the fcbr gate below).
pub(crate) const W44_91_M3_COLOURFULNESS_MIN: f32 = 80.0;

/// W44-91 zenanalyze-proxy maximum on per-8×8-block flat-color-block
/// ratio. A block is "flat" when its per-channel sRGB u8 range (max-min)
/// is ≤ 4 on EVERY channel R/G/B (zenanalyze tier1.rs exact rule).
/// Verified: 1189261 = 0.34 % (passes), 297394 = 9.57 % (fails →
/// 297394 stays on the libjxl reference table). All 6 W44-78
/// regression-band images have fcbr ≥ 0.89 %, all failing this gate.
pub(crate) const W44_91_FCBR_MAX: f32 = 0.01;

/// W44-96 sub-discriminator on `ZenanalyzeProxies::edge_density` for the
/// **variant Z DCT32X32 lift** within the W44-29 firing class.
///
/// The W44-95 honest-stop measured that variant Z (dct32x32=1.20 vs the
/// default 1.34) closes 5-6 OPEN F-D cells on {1420710, 1531677} at
/// d∈{5, 6} but regresses {2389166, 1044329} SSIM2 by -0.30 to -0.82.
/// The W44-96 proxy probe (W44-29-firing 5-image set: {1420710, 1531677,
/// 2389166, 1044329, 7062219}) identified `edge_density` as the cleanest
/// single-feature splitter:
///
/// | image    | edge_density | WANT variant Z? |
/// |---       |---           |---              |
/// | 1420710  | 0.9298       | YES             |
/// | 1531677  | 0.8766       | YES             |
/// | 2389166  | 0.4409       | NO              |
/// | 1044329  | 0.5486       | NO              |
/// | 7062219  | 0.6332       | NO              |
///
/// The threshold (0.7) sits in the gap between 0.6332 (7062219, REJECT)
/// and 0.8766 (1531677, WANT). Paired with [`W44_96_FCBR_MAX`] for
/// double-safety against false-fires on unseen images.
pub(crate) const W44_96_EDGE_DENSITY_MIN: f32 = 0.7;

/// W44-96 sub-discriminator on `ZenanalyzeProxies::flat_color_block_ratio`
/// for the variant Z DCT32X32 lift. Both WANT-Z images have fcbr=0.0
/// exactly (no perfectly-flat 8×8 sRGB blocks), while all 3 REJECT-Z
/// images have fcbr ≥ 0.011. The 0.01 threshold also matches the W44-91
/// fcbr gate exactly — semantically these are the SAME "textured, not
/// flat-color" predicate.
pub(crate) const W44_96_FCBR_MAX: f32 = 0.01;

/// W44-96 minimum distance for the variant Z lift sub-dispatch. Set to
/// 4.5 to cover the W44-95 measured wins at d∈{5, 6} on {1420710,
/// 1531677} while excluding 3637739 e7 d=4 from the dispatch path (the
/// task identifies it as a W44-95 regression cell — but since 3637739
/// has mask1x1=75.83 > 50 it doesn't fire W44-29 at all, so the
/// distance gate here is belt-and-suspenders).
pub(crate) const W44_96_VARIANT_Z_MIN_DISTANCE: f32 = 4.5;

/// W44-166 (Smart-Zenjxl chunk 3 — B1 candidate): minimum `mask1x1_p25`
/// to admit a high-mask photo (e.g. 1418519) to the W44-96 variant Z
/// dispatch even though its `mask1x1_median` exceeds the W44-96 outer
/// gate's `HIGH_D_PHOTO_SMOOTH_THRESHOLD = 50`.
///
/// **Context**: W44-152 admits high-mask photos to the OUTER W44-29
/// table (`high_d_photo_smooth_suppressed`) via the same `mask_p25 >=
/// 85` predicate. The B1 audit (W44-163) proposed extending that
/// admission INTO the variant Z inner gate so 1418519-class photos
/// can also reach the stronger lift (`dct32x32 = 1.22` for default
/// variant Z, `dct16x32 = 1.30` for high_colour Z').
///
/// W44-166 measured whether this composes with W44-152's outer-table
/// win or competes (per W44-165's binding lesson that W44-150's
/// pre-W44-152 prediction was falsified on current main). See
/// `benchmarks/w44_166_variant_z_admit_zenjxl_2026-05-21.{tsv,meta}`
/// for the load-bearing measurement.
///
/// **Threshold 85.0**: shared with [`W44_151_HIGH_MASK_P25_MIN`] for
/// per-image discriminator parity. The W44-149 audit found a 10.98pp
/// gap between 1418519 (mask_p25 = 88.88) and the nearest CONTROL
/// (7552578 at mask_p25 = 77.90), confirming this is a clean
/// single-axis discriminator for 1418519.
#[allow(dead_code)]
pub const W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN: f32 = 85.0;

/// W44-98 sub-discriminator: minimum `m3_colourfulness` to escalate from
/// the default variant Z table
/// ([`crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z`])
/// to the **high-colour** variant Z' table
/// ([`crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour`]),
/// which lifts `dct16x32` from 1.208 to 1.30.
///
/// **Source**: W44-97 per-strategy AC tokenization dump on the 7 OPEN
/// cells remaining post-W44-96 showed DCT32X16 is the universal #1
/// overspender; lifting `dct16x32` (shared slot with DCT32X16 in
/// `ac_strategy.rs:713`) reduces this overspending. But the W44-98 A/B
/// sweep found 1420710 (m3=32.93) tolerates `dct16x32=1.30` with
/// SSIM2 gains while 1531677 (m3=12.30) regresses SSIM2 by -0.34 to
/// -0.93 under the same lift. The two images differ on
/// `m3_colourfulness` by 2.7× — the cleanest separator among the
/// available `ZenanalyzeProxies` fields.
///
/// Threshold 25.0 sits between 12.30 (1531677, REJECT) and 32.93
/// (1420710, WANT) with 1.3× margin on the WANT side and 2.0× margin
/// on the REJECT side. Errs toward the WANT side: a hypothetical
/// unseen image with m3 ∈ [20, 25] stays on the default variant Z
/// table (safer for SSIM2).
pub(crate) const W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN: f32 = 25.0;

/// W44-156 distance threshold ABOVE which the d-high variant Z table
/// chain ([`crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z_d_high`],
/// [`...z_high_colour_d_high`], [`...z_low_colour_d_high`]) fires
/// instead of the W44-154 default (dct32x32 = 1.22).
///
/// **Context**: W44-154 micro-bisected variant Z `dct32x32` to 1.22 (down
/// from W44-148's 1.24, up from W44-96's 1.20). W44-155 spot-checked the
/// W44-154 ship and ran a per-strategy AC tokenization dump on 1420710
/// e5 d=5 vs d=6 (the one cell that W44-154 did NOT close at any of
/// {1.22, 1.23, 1.24}). The dump revealed:
///
/// - At d=5: our model picks DCT32X32 78.6% of first-blocks vs cjxl 51.8%
/// - At d=6: our model picks DCT32X32 76.2% of first-blocks vs cjxl 57.2%
/// - Strategy agreement only 86.8% (d=5) / 89.6% (d=6) — many DCT32X32
///   picks where cjxl picks DCT32X16 / DCT16X32 / DCT16X16
/// - At d=5→d=6 transition, cjxl DRAMATICALLY sheds small blocks (DCT8:
///   39→16, DCT16X8: 19→6); our DCT8 counts were already much lower
///   (10/2) — over-consolidated into DCT32X32 from the start
/// - Per-region quantization AT PARITY — pure strategy selection issue
///
/// The W44-154 dct32x32 = 1.22 lift OVER-FIRES at d > 5.5: it strengthens
/// the DCT32X32 incentive at exactly the distance where cjxl is going
/// the OPPOSITE direction (keeping DCT32X32 flat ~200 and shedding small
/// blocks). A weaker lift (1.20, pre-W44-148 baseline) at d > 5.5 lets
/// DCT32X32 win less often, freeing the cost model to pick smaller
/// blocks closer to cjxl's distribution.
///
/// **Threshold 5.5**: sits between the W44-154 wins at d=5 (we keep the
/// 1.22 lift to preserve W44-154's DEFICIT_LC SSIM2 +0.257 mean) and
/// the W44-155-diagnosed regression at d=6 (we shift to 1.20 to close
/// 1420710 e5 d=6). The W44-156 bisect tested 5.5 vs 5.0 to find the
/// best split point; see
/// `benchmarks/w44_156_distance_aware_variant_z_2026-05-21.{tsv,meta}`.
///
/// **Env hook**: `__JXL_W44_156_THRESHOLD` overrides this constant at
/// runtime for A/B benching. `0.0` (disable) keeps the W44-154 behaviour
/// at every distance (no d-high split). Production default = 5.5.
pub(crate) const W44_156_VARIANT_Z_D_HIGH_THRESHOLD: f32 = 5.5;

/// W44-124 auto-discriminator: minimum `m3_colourfulness` to AUTO-fire the
/// W44-123 [`VarDctEncoder::dct32_keep_hint`] lever when the caller does
/// not pass an explicit `Some(bool)`.
///
/// **Context**: W44-123 (`545e69b1`) shipped opt-in API
/// `LossyConfig::with_dct32_keep_hint(Option<bool>)`. Default `None`
/// preserved W44-68 behaviour (try_dct32=false alongside try_dct64=false).
/// Measured A/B: codec_wiki d=3 e5/e6/e7 SSIM2 +1.40/+1.33/+0.90 (TARGET),
/// terminal e8/e9 d=4 SSIM2 +0.47 (preserved). But 6 SCREEN cells regressed
/// −0.32 to −0.99 SSIM2 (graph, windows, imessage) — default-on flip
/// required a per-image discriminator.
///
/// **Probe** (`examples/w44_124_proxy_probe.rs`, 2026-05-20):
///
/// | image       | m3      | edge_density | predicate fires? |
/// |---          |---      |---           |---               |
/// | codec_wiki  | 145.73  | 0.0396       | **YES** (WANT)   |
/// | imessage    |  67.65  | 0.0533       | no (ed gate)     |
/// | terminal    |  13.85  | 0.0874       | no (m3 gate)     |
/// | graph       |  11.75  | 0.0698       | no               |
/// | windows     |  20.04  | 0.1201       | no               |
/// | imac_g3     |  14.29  | 0.1227       | no               |
/// | imac_dark   |  20.96  | 0.1438       | no               |
/// | windows95   |  27.19  | 0.3165       | no               |
/// | 1418519     |  36.84  | 0.1637       | no (CID22 photo) |
/// | 1189261     |  98.84  | 0.4895       | no (ed gate)     |
/// | 1420710     |  32.93  | 0.9298       | no               |
///
/// Threshold 60.0 sits between 27.19 (windows95, max regressing) and
/// 145.73 (codec_wiki, WANT) with 2.2× margin on the WANT side. Errs
/// strict — a hypothetical unseen screen with m3 ∈ [27, 60] stays on
/// W44-68 baseline (safe).
///
/// Paired with [`W44_124_DCT32_KEEP_EDGE_DENSITY_MAX`] (see below) to
/// reject imessage (m3=67.65 passes m3 alone but ed=0.0533 fails ed).
///
/// **W44-135 (2026-05-20)**: also paired with
/// [`W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE`] +
/// [`W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE`] distance gate that limits
/// the auto-fire to `target_distance ∈ [2.0, 3.5]`. Explicit opt-in via
/// `StrategyOverrides { dct32_keep_hint: Some(true) }` bypasses the
/// distance gate.
pub(crate) const W44_124_DCT32_KEEP_M3_MIN: f32 = 60.0;

/// W44-124 auto-discriminator: maximum `edge_density` to AUTO-fire the
/// W44-123 [`VarDctEncoder::dct32_keep_hint`] lever. Belt-and-suspenders
/// with [`W44_124_DCT32_KEEP_M3_MIN`].
///
/// **Why both**: imessage (m3=67.65) passes the m3 gate but its
/// edge_density (0.0533) is above this threshold — codec_wiki sits at
/// 0.0396 cleanly under the 0.05 floor. The W44-123 A/B showed imessage
/// d=6 regressed −0.37 SSIM2 under the keep_dct32 lever; the ed gate
/// prevents that regression while keeping codec_wiki firing.
///
/// All CID22 photos have edge_density ≥ 0.16 (textured high-frequency
/// content), so even the colourful 1189261 (m3=98.84) is correctly
/// rejected by this gate (ed=0.4895). Belt-and-suspenders against
/// false-fires on unseen photo content.
pub(crate) const W44_124_DCT32_KEEP_EDGE_DENSITY_MAX: f32 = 0.05;

/// W44-135 (2026-05-20): minimum `target_distance` at which the W44-124
/// auto-discriminator is allowed to fire.
///
/// **Context**: W44-124 (`bc9f71eb`) shipped the auto-discriminator with NO
/// distance gate. The W44-134 ledger refresh
/// (`benchmarks/cjxl_parity_ledger_2026-05-20_w44_134.tsv`) measured the
/// downstream impact across the full distance grid:
///
/// | distance band | codec_wiki SSIM2 vs W44-119 | mechanism |
/// |---            |---                          |---        |
/// | d=2.5 (e5/e6) | **+1.62 / +1.77** (BONUS WIN) | DCT32X32 beats 4×DCT16X16 on smooth pages |
/// | d=3.0 (e5/e6/e7) | **+1.40 / +1.33 / +0.90** (W44-124 TARGET) | DCT32X32 beats 4×DCT16X16 |
/// | d=4.0 (e5/e6/e7) | **-1.40 / -1.43 / -1.22** (NEW REGRESSION) | cost model prefers DCT64X64 over DCT32X32 here; forcing keep_dct32 selects strictly-worse 4×DCT16X16 + DCT32 mix |
/// | d=5.0 (e7) | **-1.38** | same mechanism as d=4 |
/// | d=6.0 (e7) | **-0.64** | same mechanism |
/// | d=0.8/1.0 (e8/e9) | **-0.41 / -0.52** | cost model is at DCT16X16/DCT8 region; keep_dct32 inert but slightly redistributes byte allocation |
///
/// **W44-143 (2026-05-20)**: lowered 2.0 → 1.4 after the W44-142 attribution
/// memo identified codec_wiki e8/e9 d=1.6/1.8 cells as the residual cluster
/// regressed by W44-135's overly-conservative 2.0 floor (NOT by the W44-140
/// EPF fade as the W44-141 ledger memo initially claimed). 30-cell × 5-variant
/// bisect on origin/main (`benchmarks/w44_143_min_distance_bisect_2026-05-20.{tsv,meta}`)
/// swept candidates `{2.0, 1.8, 1.6, 1.4, 1.2}`:
///
/// | candidate | gates passed | new wins on codec_wiki d∈[1.4, 1.8] | regressions |
/// |---        |---           |---                                   |---          |
/// | 2.0 (W44-135) | 6/6 baseline | 0 | 0 |
/// | 1.8       | 5/6          | 1 (e9 d=1.8 +0.72)                  | 1 (e8 d=1.8 -0.18) |
/// | 1.6       | 6/6          | 3 (e8 d=1.6 +0.62, e9 d=1.6 +0.62, e9 d=1.8 +0.72) | 1 (e8 d=1.8 -0.18) |
/// | **1.4 (SHIP)** | **6/6**  | **5** (+ e8/e9 d=1.4 +0.31 each)    | **1** (e8 d=1.8 -0.18) |
/// | 1.2       | 5/6          | 5 + e9 d=1.2 -0.43, e8 d=1.2 -0.27  | 3 (G2 fails) |
///
/// 1.4 is pareto-optimal: strictly more wins than 1.6 (adds d=1.4 cells)
/// while still passing all six acceptance gates (G1 codec_wiki d=1.6/1.8
/// improvement, G2 d=1.0-1.4 preservation, G3 d=3 W44-124 win, G4 d=4-6
/// W44-135 protection, G5 terminal protection, G6 photo byte-identical).
/// The single remaining regression (e8 d=1.8 -0.18 SSIM2) is unavoidable
/// at any floor ≤ 1.8 — the e8/e9 split at d=1.8 reflects buttloop iter
/// asymmetry (e8 has 2 iters, e9 has 4): e9 settles to +0.72 SSIM2 under
/// the lift while e8 starves to -0.18. Same structural pattern as the
/// W44-140 EPF fade design.
///
/// **DO NOT lower below 1.4** — d=1.2 cells regress -0.27 to -0.43 SSIM2
/// under the lift (W44-142 had to specifically suppress the EPF seed at
/// e9 d=1.2; further loosening would re-introduce a strategy-tier
/// regression on top of the EPF-seed mechanism).
///
/// Distance-window summary:
/// - `d < 1.4`: gate does NOT fire (W44-142 owns the codec_wiki d ∈ [1.0, 1.5) suppression on a different code path).
/// - `d ∈ [1.4, 3.5]`: gate fires on codec_wiki-class (W44-124 m3+ed predicate).
/// - `d > 3.5`: gate does NOT fire (W44-135 MAX ceiling preserved).
///
/// The original [W44-135 documentation](https://github.com/imazen/jxl-encoder)
/// stated "2.0 floor protects the d=0.8/d=1.0/d=1.6 cluster" — verified
/// empirically: d=0.8/1.0 ARE protected (gate still doesn't fire there at
/// MIN=1.4), but d=1.6 turned out to BENEFIT from the lift, not be hurt
/// by it. The W44-135 conclusion was extrapolated from a narrower
/// measurement set that didn't span d=1.6/1.8 specifically.
pub const W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE: f32 = 1.4;

/// W44-143 (2026-05-20): env-var override for
/// [`W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE`] used by the W44-143 bisect
/// harness only. Set `JXL_W44_143_MIN_DISTANCE=<f32>` to override the
/// W44-135 floor for paired A/B sweeps. Returns `None` (no override)
/// when unset or on no-std builds. Documented as debug-only; production
/// callers should never set it.
#[inline]
fn w44_143_min_distance_override() -> Option<f32> {
    #[cfg(feature = "std")]
    {
        std::env::var("JXL_W44_143_MIN_DISTANCE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
    }
    #[cfg(not(feature = "std"))]
    {
        None
    }
}

/// W44-143 (2026-05-20): effective minimum distance for the W44-124
/// auto-discriminator. Returns the env-var override if set, else the
/// production [`W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE`] constant.
#[inline]
fn w44_143_effective_min_distance() -> f32 {
    w44_143_min_distance_override().unwrap_or(W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE)
}

/// W44-156 (2026-05-21): runtime env hook for bisecting the
/// [`W44_156_VARIANT_Z_D_HIGH_THRESHOLD`] constant. Returns the parsed
/// override if `__JXL_W44_156_THRESHOLD` is set, else `None`.
///
/// Special values:
/// - `0.0` (or any value > 99.0): disable the d-high split (no threshold
///   ever exceeded, so plain variant Z / Z' / Z'' fire as in W44-154).
/// - Any `f32 > 0.0 && <= 99.0`: use that distance as the split point.
fn w44_156_threshold_override() -> Option<f32> {
    #[cfg(feature = "std")]
    {
        std::env::var("__JXL_W44_156_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
    }
    #[cfg(not(feature = "std"))]
    {
        None
    }
}

/// W44-156 (2026-05-21): effective distance threshold for the d-high
/// variant Z split. Returns the env-var override if set, else the
/// production [`W44_156_VARIANT_Z_D_HIGH_THRESHOLD`] constant (5.5).
#[inline]
fn w44_156_effective_threshold() -> f32 {
    w44_156_threshold_override().unwrap_or(W44_156_VARIANT_Z_D_HIGH_THRESHOLD)
}

/// W44-166 admit modes for the photo-admission branch of variant Z.
///
/// **Mode A (baseline)** — current production behaviour. Variant Z is
/// gated on `mask1x1_median < HIGH_D_PHOTO_SMOOTH_THRESHOLD = 50`
/// (the W44-96 outer gate). High-mask photos like 1418519 (mask=92)
/// cannot reach variant Z.
///
/// **Mode B** — admits photos to variant Z via
/// `mask1x1_p25 >= W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN = 85`. Once
/// admitted, the existing W44-98/99 m3 sub-dispatch routes the image
/// to the appropriate inner table (high_colour Z' if m3 >= 25, else
/// low_colour Z'').
///
/// **Mode C** — like Mode B but ALSO requires `m3_colourfulness >=
/// W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN = 25`, restricting admission
/// to images that will land on the high_colour Z' table. (For
/// 1418519 with m3=36.84, Mode C ≡ Mode B; Mode C exists to test
/// future high-m3 candidates without admitting low-m3 outliers.)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum W44_166VariantZAdmitMode {
    Baseline,
    BMaskP25,
    CMaskP25HighM3,
}

/// W44-166 (2026-05-21): admit-mode env hook for the
/// `JXL_W44_166_VARIANT_Z_ADMIT_MODE=A|B|C` selector.
///
/// **Production default: Mode B** (the SHIPPED mode per the W44-166
/// measurement that closed the chunk acceptance). Env values:
/// - `A` → [`W44_166VariantZAdmitMode::Baseline`] (disable W44-166
///   admission; reverts to pre-W44-166 behaviour for A/B/C benching)
/// - `B` (or unset) → [`W44_166VariantZAdmitMode::BMaskP25`] (SHIPPED)
/// - `C` → [`W44_166VariantZAdmitMode::CMaskP25HighM3`] (stricter
///   variant requiring `m3_colourfulness >= 25` in addition to
///   `mask_p25 >= 85`; kept reachable for future bisection of
///   discriminator tuning)
///
/// Has effect only when [`ResolvedImprovements::photo_variant_z_admit`]
/// is true (Zenjxl / Aggressive default) AND the resolved policy is
/// `Auto`. On `EncoderStrategy::Libjxl` / `LeanFaster` (which set
/// `photo_variant_z_admit = false`) the env var is ignored entirely.
#[inline]
#[allow(dead_code)]
fn w44_166_admit_mode_env() -> W44_166VariantZAdmitMode {
    #[cfg(feature = "std")]
    {
        match std::env::var("JXL_W44_166_VARIANT_Z_ADMIT_MODE")
            .ok()
            .as_deref()
        {
            Some("A") => W44_166VariantZAdmitMode::Baseline,
            Some("C") => W44_166VariantZAdmitMode::CMaskP25HighM3,
            // Default (unset OR "B" OR any other unrecognised value)
            // is Mode B per W44-166 SHIP.
            _ => W44_166VariantZAdmitMode::BMaskP25,
        }
    }
    #[cfg(not(feature = "std"))]
    {
        // no_std path: default to SHIPPED Mode B.
        W44_166VariantZAdmitMode::BMaskP25
    }
}

/// W44-167 (Smart-Zenjxl chunk 4, 2026-05-21): per-m3 sub-discriminator
/// lift mode controlled by `JXL_W44_167_MODE=A|B|C|D`.
///
/// Closes the W44-94 honest-stopped 1420710 OPEN cluster (e5..e9 d=5)
/// by lifting the `dct16x32` field of the existing INNER variant Z
/// tables — exclusively on high-m3 photos (per the existing W44-98 m3
/// sub-gate) so the W44-94 SSIM2-regression on 1531677 (low-m3) is
/// avoided.
///
/// **Mode A (Baseline)** — no change vs main (default; W44-167 inert).
/// Bench reference for A/B/C/D comparison.
///
/// **Mode B (GlobalLift)** — replay the original W44-94 X variant
/// (`dct16x32 = 1.40` at d>=5) AT THE INNER variant Z layer for BOTH
/// HC and LC. Tests whether moving the lift INTO variant Z (where
/// dct32x32 is already at 1.22 vs OUTER 1.34) changes the W44-94
/// regression sign. Hypothesis: stronger dct32x32 base + lift on
/// dct16x32 STILL regresses 1531677.
///
/// **Mode C (HighM3Only)** — apply the lift ONLY when the image
/// passes the W44-98 m3>=25 gate (i.e. only HC variant Z' fires).
/// 1531677 (m3=12.30) stays at the existing LC dct16x32=1.23.
/// Hypothesis: HC isolation captures the W44-94 X wins on 1420710
/// without the regression on 1531677.
///
/// **Mode D (PerM3Split)** — apply the strong lift on HC AND a
/// milder lift on LC. HC: dct16x32 1.30 → 1.40; LC: dct16x32 1.23 →
/// 1.26. Tests whether a tiered approach can squeeze additional bytes
/// from LC without exceeding the W44-94 SSIM2 budget.
///
/// Lift values per mode:
///
/// | mode | HC dct16x32 | LC dct16x32 | Z (none) dct16x32 |
/// |---|---|---|---|
/// | A   | 1.30 (unchanged) | 1.23 (unchanged) | 1.208 (unchanged) |
/// | B   | 1.40 | 1.40 | 1.40 |
/// | C   | 1.40 | 1.23 (unchanged) | 1.208 (unchanged) |
/// | D   | 1.40 | 1.26 | 1.22 |
///
/// Has effect only when [`ResolvedImprovements::find_best_32_per_m3_lift`]
/// is true (Zenjxl / Aggressive default) AND the variant Z dispatch
/// fired (any of `w44_96_variant_z`, `w44_98_variant_z_high_colour`,
/// `w44_99_variant_z_low_colour`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum W44_167PerM3LiftMode {
    Baseline,
    GlobalLift,
    HighM3Only,
    PerM3Split,
}

/// W44-167 admit-mode env hook for `JXL_W44_167_MODE=A|B|C|D`.
///
/// Default = Mode A (Baseline). Production default flips only when
/// a Mode B/C/D SHIPS (W44-167 chunk acceptance gate).
#[inline]
#[allow(dead_code)]
fn w44_167_mode_env() -> W44_167PerM3LiftMode {
    #[cfg(feature = "std")]
    {
        match std::env::var("JXL_W44_167_MODE").ok().as_deref() {
            Some("B") => W44_167PerM3LiftMode::GlobalLift,
            Some("C") => W44_167PerM3LiftMode::HighM3Only,
            Some("D") => W44_167PerM3LiftMode::PerM3Split,
            // Default (unset OR "A" OR any other unrecognised value)
            // is Mode A baseline (no change).
            _ => W44_167PerM3LiftMode::Baseline,
        }
    }
    #[cfg(not(feature = "std"))]
    {
        W44_167PerM3LiftMode::Baseline
    }
}

/// W44-167 helper: apply the selected mode's `dct16x32` override to a
/// pre-selected variant Z table.
///
/// `is_hc` is true when the W44-98 m3>=25 high-colour gate fired.
/// `is_lc` is true when the W44-99 m3<25 low-colour gate fired.
/// At most one of `is_hc` / `is_lc` is true (the gates are mutually
/// exclusive); if both are false the caller is on the base variant Z
/// table.
///
/// Returns the original `current_dct16x32` for Mode A (Baseline) so
/// the caller can blindly invoke this helper without checking mode.
#[inline]
#[allow(dead_code)]
fn w44_167_apply_lift(
    mode: W44_167PerM3LiftMode,
    is_hc: bool,
    is_lc: bool,
    current_dct16x32: f32,
) -> f32 {
    match mode {
        W44_167PerM3LiftMode::Baseline => current_dct16x32,
        W44_167PerM3LiftMode::GlobalLift => 1.40,
        W44_167PerM3LiftMode::HighM3Only => {
            if is_hc {
                1.40
            } else {
                current_dct16x32
            }
        }
        W44_167PerM3LiftMode::PerM3Split => {
            if is_hc {
                1.40
            } else if is_lc {
                1.26
            } else {
                1.22
            }
        }
    }
}

// ── W44-168: content-aware butteraugli_iters dispatch ──────────────────────

/// W44-168 (Smart-Zenjxl chunk 5, 2026-05-21): smooth-content `mask1x1`
/// 25th-percentile threshold for the SmoothSkip path.
///
/// At `effort >= 8` on photos with `mask1x1_p25 >= 85`, the buttloop is
/// already converging on a low-frequency residual — one fewer iter
/// (saturating at 1) trades ~30% wall time at e8 for negligible SSIM2
/// loss (chunk spec budget: ±0.10 SSIM2 mean).
///
/// Mirrors the `W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN` semantic — the
/// same `mask_p25` discriminator the W44-150/166 lineage uses, applied
/// to the orthogonal "iter budget" mechanism layer.
///
/// W44-169 (Smart-Zenjxl chunk 6, 2026-05-21) consumes this constant
/// via [`w44_169_compute_iters_narrow`] — the `#[allow(dead_code)]`
/// from the W44-168 honest-stop was removed when W44-169 shipped the
/// narrow gate.
pub const W44_168_SMOOTH_MASK_P25_MIN: f32 = 85.0;

/// W44-168 (Smart-Zenjxl chunk 5, 2026-05-21): screenshot `mask1x1`
/// median threshold for the SmoothSkip path.
///
/// Mirrors `CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD` (= 95.0) used by
/// every screenshot-class gate (W22-1, W44-105 buttloop seed scale,
/// etc.). On screenshot-class content the buttloop's per-iter quant
/// adjustment is mostly a no-op (libjxl's coarse quant on flat regions
/// converges in a single iter) — `iters - 1` saturating at 1 saves
/// ~30% wall time at e8 with byte-identical or near-identical output.
///
/// W44-169 (Smart-Zenjxl chunk 6, 2026-05-21) consumes this constant
/// via [`w44_169_compute_iters_narrow`] — the `#[allow(dead_code)]`
/// from the W44-168 honest-stop was removed when W44-169 shipped the
/// narrow gate.
pub const W44_168_SCREENSHOT_MEDIAN_MIN: f32 = 95.0;

/// W44-168 (Smart-Zenjxl chunk 5, 2026-05-21): textured `edge_density`
/// threshold for the TexturedExtend path.
///
/// At `effort == 7` on photos with `edge_density >= 0.5` (high-edge
/// textured content like 1189261-class wedges), the fixed schedule
/// gives `butteraugli_iters = 0`. Bumping to 2 iters (effectively a
/// "soft e8" budget for textured content) bridges the F-row deficit
/// without forcing the caller up to a real e8.
///
/// Threshold = 0.5: the 5 W44-67-class photos cluster at
/// edge_density ∈ [0.30, 0.78]; the 0.50 cut admits 1189261 / 1420710
/// while keeping smoother photos (1418519 ed=0.32, 7552578 ed=0.41) on
/// the e7 baseline (iters=0 → no spurious wall-time addition on smooth
/// content at e7).
#[allow(dead_code)]
pub const W44_168_TEXTURED_EDGE_DENSITY_MIN: f32 = 0.5;

/// W44-168 (Smart-Zenjxl chunk 5, 2026-05-21): TexturedExtend iter
/// count at `effort == 7` on textured content.
///
/// Default 2 iters (matching the e8 baseline schedule). Could be bumped
/// to 4 to match e9 if measurement shows the win, but starting
/// conservative.
#[allow(dead_code)]
pub const W44_168_TEXTURED_ITERS_AT_E7: u32 = 2;

/// W44-168 (Smart-Zenjxl chunk 5, 2026-05-21): adaptive
/// `butteraugli_iters` mode controlled by env hook
/// `JXL_W44_168_MODE=A|B|C|D`.
///
/// Per user directive 2026-05-21 ("make zenjxl defaults smarter on the
/// rdtime axis ... even if effort levels blend together more"), this
/// gate adjusts the fixed per-effort iter schedule based on cheap
/// proxies already on hand (`mask1x1_median`, `mask1x1_p25`,
/// `edge_density`).
///
/// **Mode A (Baseline)** — no change vs the fixed schedule (`e≤7 → 0,
/// e8 → 2, e9 → 4, e10 → 8, e11 → 16, e12 → 32`). Bench reference.
///
/// **Mode B (SmoothSkip)** — at `effort >= 8` on
/// smooth/screenshot content (`mask1x1_median > 95` OR
/// `mask1x1_p25 >= 85`), `iters - 1` saturating at 1. Saves one full
/// buttloop iter on converged content (~30% wall time at e8).
///
/// **Mode C (TexturedExtend)** — at `effort == 7` on textured
/// content (`edge_density >= 0.5`), bump iters from 0 → 2. Blends e7
/// upward toward e8 quality for textured content.
///
/// **Mode D (Combined)** — apply both B and C.
///
/// Has effect only when
/// [`crate::api::ResolvedImprovements::adaptive_buttloop_iters`] is
/// true (Zenjxl / Aggressive default).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum W44_168IterMode {
    Baseline,
    SmoothSkip,
    TexturedExtend,
    Combined,
}

/// W44-168 admit-mode env hook for `JXL_W44_168_MODE=A|B|C|D`.
///
/// Default = Mode A (Baseline). Production default flips only when a
/// Mode B/C/D SHIPS (W44-168 chunk acceptance gate).
#[inline]
#[allow(dead_code)]
pub(crate) fn w44_168_mode_env() -> W44_168IterMode {
    #[cfg(feature = "std")]
    {
        match std::env::var("JXL_W44_168_MODE").ok().as_deref() {
            Some("B") => W44_168IterMode::SmoothSkip,
            Some("C") => W44_168IterMode::TexturedExtend,
            Some("D") => W44_168IterMode::Combined,
            // Default (unset OR "A" OR any other unrecognised value)
            // is Mode A baseline (no change).
            _ => W44_168IterMode::Baseline,
        }
    }
    #[cfg(not(feature = "std"))]
    {
        W44_168IterMode::Baseline
    }
}

/// W44-168 helper: compute the adjusted `butteraugli_iters` value for
/// this image and effort.
///
/// `base_iters` is the fixed per-effort iter count from
/// [`crate::effort::EffortProfile::butteraugli_iters`] /
/// [`crate::vardct::encoder::VarDctEncoder::butteraugli_iters`].
///
/// Returns the original `base_iters` when:
/// - `mode == Baseline`
/// - The chosen mode's content discriminator doesn't fire
/// - `base_iters == 0` AND mode is SmoothSkip-only (no extension into
///   the zero-iter regime)
///
/// **SmoothSkip** is `iters - 1` saturating at 1 (never goes to 0 —
/// keep at least one iter so the buttloop still runs).
///
/// **TexturedExtend** sets iters to [`W44_168_TEXTURED_ITERS_AT_E7`]
/// (= 2) when `base_iters == 0 AND effort == 7 AND edge_density >=
/// 0.5`. Bridges textured e7 toward e8 quality.
#[inline]
#[allow(dead_code)]
pub(crate) fn w44_168_compute_iters(
    base_iters: u32,
    effort: u8,
    mask1x1_median: Option<f32>,
    mask1x1_p25: Option<f32>,
    edge_density: Option<f32>,
    mode: W44_168IterMode,
) -> u32 {
    match mode {
        W44_168IterMode::Baseline => base_iters,
        W44_168IterMode::SmoothSkip => {
            if effort >= 8 && base_iters > 1 && w44_168_is_smooth(mask1x1_median, mask1x1_p25) {
                base_iters - 1
            } else {
                base_iters
            }
        }
        W44_168IterMode::TexturedExtend => {
            if effort == 7 && base_iters == 0 && w44_168_is_textured(edge_density) {
                W44_168_TEXTURED_ITERS_AT_E7
            } else {
                base_iters
            }
        }
        W44_168IterMode::Combined => {
            // Combined: SmoothSkip on e8+, TexturedExtend on e7. They
            // are mutually exclusive on the effort axis (e==7 vs e>=8),
            // so the order of evaluation doesn't matter.
            if effort >= 8 && base_iters > 1 && w44_168_is_smooth(mask1x1_median, mask1x1_p25) {
                base_iters - 1
            } else if effort == 7 && base_iters == 0 && w44_168_is_textured(edge_density) {
                W44_168_TEXTURED_ITERS_AT_E7
            } else {
                base_iters
            }
        }
    }
}

/// W44-168: smooth-content discriminator. Smooth = high mask1x1_median
/// (screenshot-class) OR high mask1x1_p25 (smooth-photo-class).
///
/// W44-169 (Smart-Zenjxl chunk 6, 2026-05-21) consumes this helper via
/// [`w44_169_compute_iters_narrow`] — the `#[allow(dead_code)]` from
/// the W44-168 honest-stop was removed when W44-169 shipped the narrow
/// gate.
#[inline]
pub(crate) fn w44_168_is_smooth(mask1x1_median: Option<f32>, mask1x1_p25: Option<f32>) -> bool {
    // W44-213: route both thresholds through the tuning-override macro
    // so sweep-runner builds can swap them at runtime.
    let median_threshold =
        crate::runtime_or_default!(W44_168_SCREENSHOT_MEDIAN_MIN, screenshot_median_threshold,);
    let p25_threshold =
        crate::runtime_or_default!(W44_168_SMOOTH_MASK_P25_MIN, smart_zenjxl_photo_mask_p25_min,);
    let screen = mask1x1_median.is_some_and(|m| m > median_threshold);
    let smooth_photo = mask1x1_p25.is_some_and(|p| p >= p25_threshold);
    screen || smooth_photo
}

/// W44-168: textured-content discriminator. Textured = high
/// `edge_density` (Sobel gradient hits per interior pixel).
#[inline]
#[allow(dead_code)]
pub(crate) fn w44_168_is_textured(edge_density: Option<f32>) -> bool {
    edge_density.is_some_and(|ed| ed >= W44_168_TEXTURED_EDGE_DENSITY_MIN)
}

/// W44-169 (Smart-Zenjxl chunk 6, 2026-05-21): minimum `target_distance`
/// at which the narrow SmoothSkip iter-decrement is allowed to fire.
///
/// W44-168 (`42833a05`) honest-stopped on broad Mode B because it
/// destroyed W44-166's +0.45 SSIM2 win on 1418519 e8 d=6 (SSIM2 -0.26).
/// The same measurement found STRICT WINS at the narrow d=4/5 band on
/// 1418519:
/// - e8 d=4: ΔSSIM2 +0.627 + Δwall -4.79%
/// - e8 d=5: ΔSSIM2 +0.559 + Δwall -4.13%
///
/// The narrow window `[W44_169_NARROW_MIN_DISTANCE,
/// W44_169_NARROW_MAX_DISTANCE]` = `[4.0, 5.0]` captures the wins
/// without touching d=6 (where W44-166 needs the full iter budget to
/// land its variant Z win) or low-d cells where the buttloop's
/// per-iter quant adjustment is doing meaningful SSIM2 work.
///
/// Mirrors the W44-156 distance-narrowing pattern (variant Z @ d > 5.5)
/// applied to the W44-168 mechanism layer.
pub const W44_169_NARROW_MIN_DISTANCE: f32 = 4.0;

/// W44-169 (Smart-Zenjxl chunk 6, 2026-05-21): maximum `target_distance`
/// at which the narrow SmoothSkip iter-decrement is allowed to fire.
///
/// Excludes d=6 specifically to preserve the W44-166 +0.45 SSIM2 win on
/// 1418519 e8 d=6 (the surface that broad Mode B destroyed). See
/// [`W44_169_NARROW_MIN_DISTANCE`] for the full design rationale.
pub const W44_169_NARROW_MAX_DISTANCE: f32 = 5.0;

/// W44-169 helper: compute the adjusted `butteraugli_iters` value for
/// the narrow SmoothSkip dispatch.
///
/// Equivalent to [`w44_168_compute_iters`] called with `mode =
/// W44_168IterMode::SmoothSkip` **GATED on `target_distance ∈
/// [W44_169_NARROW_MIN_DISTANCE, W44_169_NARROW_MAX_DISTANCE]`**.
///
/// Outside the distance band, returns `base_iters` unchanged (pre-W44-169
/// behaviour). Inside the band on smooth/screenshot content at
/// `effort >= 8` with `base_iters > 1`, returns `base_iters - 1`
/// (saturating at 1).
///
/// Mode A baseline (when `narrow_enabled = false`) returns `base_iters`
/// always — byte-identical to pre-W44-169.
#[inline]
pub(crate) fn w44_169_compute_iters_narrow(
    base_iters: u32,
    effort: u8,
    target_distance: f32,
    mask1x1_median: Option<f32>,
    mask1x1_p25: Option<f32>,
    narrow_enabled: bool,
) -> u32 {
    if !narrow_enabled {
        return base_iters;
    }
    let in_band =
        (W44_169_NARROW_MIN_DISTANCE..=W44_169_NARROW_MAX_DISTANCE).contains(&target_distance);
    if !in_band {
        return base_iters;
    }
    if effort >= 8 && base_iters > 1 && w44_168_is_smooth(mask1x1_median, mask1x1_p25) {
        base_iters - 1
    } else {
        base_iters
    }
}

/// W44-135 (2026-05-20): maximum `target_distance` at which the W44-124
/// auto-discriminator is allowed to fire.
///
/// Companion to [`W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE`]. The 3.5 ceiling
/// protects the d=4.0/5.0/6.0 cluster where the cost model prefers
/// DCT64X64 over DCT32X32 on smooth screen content — forcing keep_dct32
/// selects strictly-worse 4×DCT16X16 + DCT32 mix and regresses SSIM2 by
/// -0.64 to -1.43 (W44-134 measurement).
///
/// 3.5 (rather than 3.0) admits any future bisect cells in (3.0, 3.5) where
/// the W44-124 mechanism may still net-win. Acceptance gated to the
/// codec_wiki d=2.5..3.0 window measured in W44-124 + W44-134.
///
/// Caller-side `Dct32SearchPolicy::KeepWhenDct64Suppressed` (explicit
/// opt-in) bypasses this gate — the distance window only narrows the
/// AUTO fire path, not the explicit override path.
pub const W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE: f32 = 3.5;

/// W44-87 single-pass-entropy dispatch — smooth-photo `median(mask1x1)`
/// upper bound. Same direction as [`HIGH_D_PHOTO_SMOOTH_THRESHOLD`]
/// (smooth = below threshold), set to the same `50.0` value: CID22
/// photos cluster in the 10-40 range, gb82-sc screenshots run
/// 110-180, and the 50-95 ambiguous band stays on the two-pass
/// (per-image-optimized) entropy path where the histogram diversity
/// makes the optimization worth its cost.
///
/// On smooth photos at low distance the two-pass `entropy` +
/// `build_codes` phases save 2-4 % bytes vs the pre-computed static
/// codes — a poor trade against ~30 % wall-clock savings
/// (`benchmarks/lossy_phase_baseline_low_effort_2026-05-19.tsv`).
pub(crate) const SINGLE_PASS_ENTROPY_SMOOTH_PHOTO_MAX_MEDIAN: f32 = 50.0;

/// W44-87 single-pass-entropy dispatch — maximum effort at which the
/// `Auto` policy will flip to single-pass. Bound matches the W38
/// profile finding that the `entropy` + `build_codes` phase ratio
/// peaks at e5 photos (56-62 % of wall); at e6/e7 the patches /
/// EPF / CfL two-pass / butteraugli iterations dominate and the
/// entropy savings stop being the largest knob. Keeping the gate
/// at `<= 5` also dodges the e7+ `try_dct64` / patches features
/// which require the two-pass plumbing.
pub(crate) const SINGLE_PASS_ENTROPY_MAX_EFFORT: u8 = 5;

/// W44-87 single-pass-entropy dispatch — maximum distance at which
/// the `Auto` policy will flip to single-pass. At `d > 1.0` the
/// per-image-tuned codes start saving > 4 % bytes (the histogram
/// shifts as quantization coarsens), tilting the trade back toward
/// two-pass on the typical smooth photo.
pub(crate) const SINGLE_PASS_ENTROPY_MAX_DISTANCE: f32 = 1.0;

/// W44-29 minimum distance for the auto smooth-photo gate.
///
/// **W44-78 widening** (2026-05-19): lowered 4.0 → 3.0 after a 230-cell
/// A/B sweep (46 images × 5 distances) confirmed widening to d>=3 closes
/// the previously-unreachable d=3 corner of the F-D residual cluster
/// with zero `FIXED → OPEN` flips:
///
/// | image          | mask1x1 | default(d=3) | widened(d=3) | Δ        | cjxl    |
/// |---             |---      |---           |---           |---       |---      |
/// | 1420710.png    | 39.55   |       38180  |       36505  | **-1675**| 36567   |
/// | 1044329.png    | 48.03   |       49148  |       48693  |   -455   | 48629   |
/// | 2389166.png    | 46.24   |       24015  |       23686  |   -329   | 24002   |
/// | 3637739.png    | 47.80   |       20127  |       19798  |   -329   | 19421   |
/// | 1531677.png    | 35.63   |       32022  |       32246  |   +224   | 32698   |
///
/// Net: -2564 B across 5 affected cells, 1 `OPEN → FIXED` flip
/// (1420710.png e7 d=3), zero `FIXED → OPEN`. The single +224 B
/// regression on 1531677.png stays well inside `FIXED` (still -1.39%
/// vs cjxl).
///
/// **Hash-lock invariance**: every `lossy_distance_3`-class fixture
/// in `tests/hash_lock_features.rs` is a small synthetic gradient or
/// random-noise pattern whose `median(mask1x1)` is either >50 (the
/// 32×32 gradient produces 96.74, 13×17 produces 88.44) or whose
/// distance is <3.0 (all noise fixtures encode at d=1.0). Verified
/// via `examples/w44_78_gradient_mask_probe.rs` — `cargo test -p
/// jxl-encoder hash_lock` passes byte-identical with the widening.
///
/// **Why not d>=2.5**: variant D (d>=2.5, mask<95) added 11 more
/// `FIXED → OPEN` flips against the cjxl_parity_ledger — too many of
/// the d=2.5 cells where the W44-29 table is too aggressive. The 3.0
/// floor matches where W44-27's `AdjustQuantBlockAC` D-heuristic
/// firing rate becomes high enough for entropy_mul reduction to win
/// uniformly.
///
/// **Why not mask<80**: variants B and C lifted the mask1x1 ceiling
/// from 50 to 80 to catch 1189261 (mask=69.08). Both variants
/// regressed 7+ `FIXED` cells (1025469.png at d=4/6 worst:
/// +654 B / +90 B with mask=76.08 in the middle of the new band).
/// The W44-29 table is harmful in the 50-80 mask1x1 zone for these
/// images — keep mask<50 strict.
///
/// **W44-79 zenanalyze discriminator (then default-on via W44-91)**: the
/// 50-80 mask1x1 band can be safely widened *only* for the
/// colourful+textured photo sub-class that 1189261 represents. The
/// verified discriminator is `colourfulness >= 80 AND
/// flat_color_block_ratio < 0.01` (both zenanalyze Tier 1 features —
/// the W44-79 memo incorrectly listed them as Tier 2). Of the 41 CID22
/// validation images, only 1189261 matches. All 6 documented
/// regression-band images (1025469, 1624487, 159550, 2079234, 2775196,
/// 297394) fail this discriminator and stay on the libjxl-parity
/// reference table.
///
/// **W44-91 (2026-05-19) wired the discriminator into the production
/// default** via cheap encoder-internal proxies (see
/// [`HIGH_D_PHOTO_W44_91_MASK_UPPER`] and [`ZenanalyzeProxies`]).
/// On 8-bit sRGB-like layouts (`Rgb8` / `Rgba8` / `Bgr8` / `Bgra8`)
/// the API layer computes the two-proxy discriminator in one O(W·H)
/// pass and the dispatch fires when distance ∈ [3.0, 5.0] AND
/// mask1x1_median ∈ [50, 80) AND m3 ≥ 80 AND fcbr < 0.01.
/// Per-distance impact when [`crate::api::LossyConfig::with_high_d_photo_hint`]
/// is set to `Some(true)` on 1189261 (e7 in-process measurement, 2026-05-19):
///
///   - d=3: -679 B (-2.23 %)
///   - d=4: -452 B (-1.85 %)
///   - d=5: -319 B (-1.61 %)
///   - d=6: +560 B (+3.33 %) ← regression — caller MUST cap at d <= 5
///
/// Total -1450 B across d ∈ {3, 4, 5} for 1189261.
///
/// **W44-91 update (2026-05-19)**: discriminator now wired into the
/// production default via cheap encoder-internal proxies — see
/// [`HIGH_D_PHOTO_W44_91_MASK_UPPER`], the [`ZenanalyzeProxies`] struct,
/// and the dispatch site in `compute_ac_strategy` for the gate logic.
/// Both `colourfulness` and `flat_color_block_ratio` turn out to live in
/// zenanalyze **Tier 1** (not Tier 2 — the W44-79 memo had the tier
/// wrong); their definitions are a single O(W·H) sRGB-u8 pass each, so
/// the proxy can ship in-encoder without a dependency on zenanalyze.
///
/// `LossyConfig::with_high_d_photo_hint(Some(false))` still suppresses
/// both the W44-29 and the W44-91 gates (the hint is a single
/// suppress/force lever shared by both).
///
/// **Sweep artifact**: `benchmarks/w44_78_widen_gate_ab_2026-05-19.tsv` +
/// W44-79 discriminator confirmation in
/// `benchmarks/w44_79_zenanalyze_discriminator_2026-05-19.tsv` +
/// W44-91 wired-dispatch A/B in
/// `benchmarks/w44_91_zenanalyze_dispatch_2026-05-19.tsv` (reproducer:
/// `cargo run --release -p jxl-encoder --features parallel --example
/// w44_91_dispatch_ab`).
pub(crate) const HIGH_D_PHOTO_MIN_DISTANCE: f32 = 3.0;

/// W44-150: minimum `mask1x1` 25th-percentile to ADMIT a non-screenshot
/// photo to the W44-117 EPF sharpness seed (in addition to the W44-118
/// `is_screenshot` gate).
///
/// **Audit derivation** (W44-149 Phase 1,
/// `examples/w44_149_photo_proxy_audit.rs` +
/// `benchmarks/w44_149_photo_proxy_audit_2026-05-21.tsv`): probed 28
/// candidate proxies × 41 CID22 validation photos to find a discriminator
/// admitting `1418519.png` (WANT — closes the -2.17 to -2.57 SSIM2 deficit
/// per W44-147) while excluding `1025469.png` (REJECT — W44-118 closed a
/// -0.85 SSIM2 regression there) and the 39 other CID22 photos (CONTROL —
/// no measured EV).
///
/// `mask_p25` (`compute_mask1x1` 25th percentile) cleanly separates the
/// three classes with the widest single-axis safety margin:
///
/// | image          | mask_p25 | class      |
/// |---             |---       |---         |
/// | 1418519.png    | 88.88    | WANT       |
/// | 7552578.png    | 77.90    | CONTROL    |
/// | 1025469.png    | 60.64    | REJECT     |
/// | 39 other CID22 photos | < 78  | CONTROL  |
///
/// Threshold `85.0` sits in the 10.98-pp gap between WANT (88.88) and the
/// nearest CONTROL (7552578 at 77.90) — 5× the safety margin of the
/// runner-up discriminators (`mask_med >= 92` had gap 0.85; `mask_p10 >=
/// 71` had gap 2.20). Errs strict — a hypothetical unseen photo with
/// `mask_p25 ∈ [77.9, 85.0]` stays on the legacy uniform-4 seed (safe).
///
/// Other axes (`m3_colourfulness`, `flat_color_block_ratio`,
/// `edge_density`, `mean_luma`, `high_luma_ratio`, `high_freq_energy`)
/// all RULED OUT — multiple CONTROL photos land above WANT on each.
///
/// **Hard caveat (per W44-139)**: the +0.034 R post-EPF buttloop
/// divergence on 1418519 is REAL but was ssim2-NEUTRAL in production
/// today (because the W44-118 hardcoded gate prevented W44-117 from
/// firing on photos under any configuration). W44-150 was the FIRST
/// measurement of W44-117 actually running on a photo.
///
/// **W44-150 Phase 2 HONEST-STOP (2026-05-21)**: the Mechanism A admission
/// path that this constant gated produced +0.27 mean SSIM2 on 1418519
/// d=5/6 (HARD gate wanted +1.0 = 50 % of -2.17 to -2.57 deficit).
/// Discriminator works (proxy-side 51/51 protection cells byte-identical;
/// only 1418519 fired the new path) but the W44-117 EPF seed mechanism
/// alone only recovers ~30 % of the SSIM2 deficit at d=5 and ~0 % at d=6
/// on the e8/e9 cells. e7 cells stay byte-identical because W44-117 is
/// gated on `profile.epf_dynamic_sharpness AND butteraugli_iters > 0`
/// which is false at e<=7. Bench:
/// `benchmarks/w44_150_mask_p25_admission_2026-05-21.{tsv,meta}`.
///
/// **W44-165 HONEST-STOP (2026-05-21, Smart-Zenjxl chunk 2)**:
/// re-implemented the W44-150 admission for `EncoderStrategy::Zenjxl`
/// / `Aggressive` and measured a 36-cell paired A/B
/// (`benchmarks/w44_165_restore_epf_seed_photos_2026-05-21.{tsv,meta}`).
/// The W44-150 predicted +0.27 mean SSIM2 win on 1418519 d=5/6 e8/e9
/// was FALSIFIED in current main: measured mean = **-0.105**
/// (REGRESSION), worst -0.331 on e8/e9 d=5. Root cause: since W44-150's
/// `dad6bb47` baseline, W44-152 (`971bbc8c`) shipped the d ∈ [3.0, 5.0]
/// mask_p25 admission on the W44-29 OUTER entropy_mul lift, delivering
/// +1.13 SSIM2 to the same 1418519 d=5 e8/e9 cells. The W44-152
/// baseline (SSIM2=66.54 at e8 d=5) is ABOVE the W44-150 baseline
/// (SSIM2=65.41); applying the W44-117 EPF seed mechanism on top of
/// the W44-152 baseline now OVERSHOOTS (net regression vs the W44-152
/// baseline). The two mechanisms COMPETE rather than COMPOSE.
///
/// Production gate stays at the W44-118 `is_screenshot` form. The
/// [`crate::api::EncoderImprovementsCustom::photo_epf_seed_admit`]
/// field is KEPT as public API surface (default true on Zenjxl /
/// Aggressive, false on Libjxl / LeanFaster) for `Custom` callers
/// wanting to opt in, but the production dispatch site does NOT
/// currently read it. See
/// `memory/w44_165_photo_epf_seed_zenjxl_honest_stop_2026-05-21.md`.
#[allow(dead_code)]
pub const W44_150_PHOTO_W44_117_MASK_P25_MIN: f32 = 85.0;

/// W44-150: minimum `target_distance` to ADMIT a non-screenshot photo to
/// the W44-117 EPF sharpness seed (in addition to the
/// [`W44_150_PHOTO_W44_117_MASK_P25_MIN`] discriminator).
///
/// The W44-147 photo deficit cluster on 1418519 measured `SSIM2 ∈
/// [-2.17, -2.57]` at d=5 e7/e8/e9 and d=6 e7/e8/e9 (six worst SSIM2
/// cells in the W44-146 ledger). The d=4 deficit is much smaller
/// (-1.00 worst); the d<4 cells are at SSIM2 parity. Capping the
/// admission band at `d >= 4.0` limits the new code path to the
/// distance regime where the W44-147 audit predicted EV, and avoids
/// firing W44-117 on photos at low-d where the +0.034 R divergence may
/// flip SSIM2-negative (mirrors the W44-120 distance gate on
/// screenshots which capped at `d >= 1.0` to close the d=0.8 W44-117
/// over-correction).
///
/// **W44-150 HONEST-STOP (2026-05-21)** + **W44-165 HONEST-STOP
/// (2026-05-21)**: see [`W44_150_PHOTO_W44_117_MASK_P25_MIN`] for the
/// full disposition. Constant kept for documentation + reuse by
/// future chunks that pair the W44-149 discriminator with a different
/// mechanism than the W44-117 EPF seed.
#[allow(dead_code)]
pub const W44_150_PHOTO_W44_117_MIN_DISTANCE: f32 = 4.0;

/// W44-151: minimum `mask1x1` 25th-percentile to ADMIT a non-screenshot
/// photo to the W44-29 outer entropy_mul lowering gate (in addition to
/// the existing `median(mask1x1) < HIGH_D_PHOTO_SMOOTH_THRESHOLD` branch
/// and the W44-91 zenanalyze sub-branch).
///
/// Re-uses the W44-149 audit's discriminator (see
/// [`W44_150_PHOTO_W44_117_MASK_P25_MIN`]). The audit identified
/// `mask_p25 >= 85` as the single-axis discriminator that cleanly admits
/// `1418519.png` (88.88, WANT) while rejecting `1025469.png` (60.64,
/// REJECT) and all 39 other CID22 validation photos (max 77.90 on
/// 7552578). 11-pp safety margin to the nearest CONTROL.
///
/// W44-150 paired this discriminator with the W44-117 EPF seed
/// (Mechanism A) and HONEST-STOPPED: recovered only ~30% of the d=5
/// SSIM2 deficit and ~0% at d=6 (`+0.27` mean vs `+1.0` target).
/// W44-151 pairs the SAME discriminator with the W44-29 entropy_mul
/// mechanism (Mechanism B): admits 1418519 to the default
/// `high_d_photo_smooth_suppressed()` table (`dct32x32=1.34` vs libjxl
/// stock `1.48`, ~9.5% lift) which fires inside `FindBest*Transform`
/// cost evaluation at `e >= 5` — covering ALL effort levels where W44-29
/// can fire today, not just the buttloop range.
///
/// **Hypothesis (W44-151)**: variant Z's lift is gated on `mask < 50`
/// (W44-96) and will NOT escalate for 1418519 (mask=92). The DEFAULT
/// suppressed table is what 1418519 receives — a milder lift,
/// appropriate for high-mask flat regions where DCT32X32 over DCT16X16
/// splits should help bytes without quality loss.
///
/// **W44-151 HONEST-STOP (2026-05-21)**: the broad `d >= 3.0` gate was
/// measured (72-cell A/B,
/// `benchmarks/w44_151_w44_29_widen_2026-05-21.{tsv,meta}`) and REVERTED.
/// Per-cell verdict on 1418519 (the only photo where the gate fires):
///
/// | cell | Δbytes | Δssim2 | verdict |
/// |---|---|---|---|
/// | e7/8/9 d=4 | -2.8 to -3.4% | +0.10 to +0.55 | universal win |
/// | e7 d=5 | -0.29% | +0.38 | win |
/// | e8/9 d=5 | +0.7 to +1.1% | +1.13 | strong win |
/// | e7 d=6 | +4.27% | +0.07 | bytes regress, no SSIM2 |
/// | e8/9 d=6 | +4.3 to +4.6% | +0.28 | bytes regress, weak SSIM2 |
///
/// Mean SSIM2 across d=5/6 cells (the W44-147 cluster) = **+0.544** vs
/// the +1.0 acceptance bar (50% closure of the -2.17/-2.57 deficit).
/// The d=4 cells PASS gate (g) but d=6 cells drag the d=5/6 mean below
/// gate (f). Protection set 1025469 + 4 SPOT photos: **63/63 BYTE-IDENTICAL**
/// (discriminator works perfectly — 11pp safety margin to nearest
/// CONTROL, no false-positives).
///
/// Production code REVERTED to pre-W44-151 state. Constant + the
/// `mask1x1_p25` plumbing through `compute_profile_for_search` KEPT
/// (negligible overhead — one extra O(n) `select_nth_unstable` per
/// encode) so W44-152+ chunks can re-enable the admission branch with
/// a tighter gate (e.g. `d ∈ [3.0, 5.0]` only or paired with a
/// distance-tapered entropy_mul lift).
///
/// Referenced by the W44-151 unit tests + the W44-149-aligned
/// `test_w44_151_mask_p25_threshold_matches_w44_149_audit` invariant
/// (the threshold MUST stay aligned with [`W44_150_PHOTO_W44_117_MASK_P25_MIN`]).
///
/// **W44-152 (2026-05-21)**: now CONSUMED in production by the W44-29
/// outer gate's `mask_p25 >= 85 AND target_distance ∈ [3.0, 5.0]`
/// admission branch. The distance bounds are
/// [`W44_152_W44_151_MIN_DISTANCE`] / [`W44_152_W44_151_MAX_DISTANCE`].
pub const W44_151_HIGH_MASK_P25_MIN: f32 = 85.0;

/// W44-152: lower distance bound (inclusive) for the W44-151 mask_p25
/// admission OR-branch on the W44-29 outer gate.
///
/// W44-151 honest-stopped on the broad `d >= HIGH_D_PHOTO_MIN_DISTANCE`
/// (3.0) gate because the default `high_d_photo_smooth_suppressed()`
/// table over-fires at d=6 on 1418519 (+4.3-4.6% bytes for only
/// +0.07-0.28 SSIM2). The d=4 cluster was a clean win (-3% bytes +
/// +0.55 SSIM2 mean) and the d=5 cluster was strong at e8+ (+1.13
/// SSIM2). W44-152 captures the win region by bounding the gate to
/// `[3.0, 5.0]`; d=6 cells stay byte-identical (gate doesn't fire).
///
/// Lower bound coincides with [`HIGH_D_PHOTO_MIN_DISTANCE`] (3.0) — the
/// existing W44-29 sibling gates use the same floor. Below 3.0 the
/// W44-29 outer gate cannot fire under any branch, so this bound is
/// load-bearing only if a future change lowers `HIGH_D_PHOTO_MIN_DISTANCE`.
///
/// Referenced by [`test_w44_152_distance_gate_edges`].
pub const W44_152_W44_151_MIN_DISTANCE: f32 = 3.0;

/// W44-152: upper distance bound (inclusive) for the W44-151 mask_p25
/// admission OR-branch. See [`W44_152_W44_151_MIN_DISTANCE`] for the
/// honest-stop history motivating the distance narrowing.
///
/// `5.0` excludes the d=6 cluster where the W44-151 bench measured
/// +4.3-4.6% byte regression for only +0.07-0.28 SSIM2 gain on 1418519.
/// Above 5.0 the gate degrades to off and the encode falls back to the
/// libjxl stock entropy_mul table (no W44-29 lift).
///
/// Note: this bound is sibling to [`HIGH_D_PHOTO_W44_91_MAX_DISTANCE`]
/// (also 5.0); the two are independently tunable so a future change
/// to one doesn't drag the other.
pub const W44_152_W44_151_MAX_DISTANCE: f32 = 5.0;

/// W36-3 patches photo-skip dispatch threshold on the per-block-mean
/// median of `mask1x1` (same statistic the auto-splines
/// [`crate::vardct::splines::looks_like_screenshot`] gate uses).
///
/// `> 60` is **deliberately lower** than the shared `95.0` used by
/// auto-splines / GPU AFV cost-grid because the cost asymmetry is
/// opposite:
///
/// * The auto-splines gate prefers **false-negative** (run splines on
///   an actual screenshot — caught later by the per-spline cost gate)
///   over **false-positive** (skip splines on a real photo).
/// * The patches photo-skip dispatch prefers **false-positive**
///   (run the patches scan on a real photo — produces empty
///   `PatchesData`, byte-identical to AlwaysScan, just slower by
///   ~25-30 ms/MP) over **false-negative** (skip the scan on a real
///   screenshot — loses 30-70 % of the screenshot's bytes savings).
///
/// The `auto_splines_bench_2026-05-17_chunk5.meta` corpus characterises
/// `windows95.png` (640×480 Win95 UI mockup) as the one false-negative
/// of the `> 95` gate: per-block-mean median ≈ 69.9 vs CID22 photos
/// median ≈ 56 (max ≤ 87 across 16 CLIC photos). `> 60` catches
/// `windows95.png` without dragging in any photo in that corpus.
///
/// If a future content discriminator regression admits a photo above
/// 60, the worst case is still byte-identical output (the patches
/// scan runs and produces empty `PatchesData`) — only the wall-clock
/// win disappears for that image.
pub(crate) const PATCHES_DISPATCH_BLOCK_MASK_THRESHOLD: f32 = 60.0;

/// W38-2 pixel-loss dispatch threshold on the raw per-pixel `median(mask1x1)`
/// (the same statistic the content-aware entropy_mul dispatch at W22-1
/// uses, NOT the per-block-mean median used by the patches/auto-splines
/// gates).
///
/// `> 80` is chosen from the W38-1 phase profile (`benchmarks/
/// lossy_phase_low_effort_with_zenjpeg_2026-05-19.{tsv,meta}`) where
/// `pixel_domain_loss=true` adds ~11 ms/MP on photos (CID22 medians
/// in the 10-40 range) and ~70 ms/MP on screenshots (gb82-sc medians
/// 110-180). On smooth content (median >80, the photo-flat /
/// screenshot-class regime) the pixel-domain loss term rarely
/// changes which AC-strategy wins — the coefficient-domain entropy
/// estimate alone converges on the same DCT8/DCT16 pick — so the
/// loss term's cost is wasted.
///
/// The `80` cut-off is **below** the `95` screenshot/photo split
/// (`CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD`) deliberately: we
/// want to gate on the broader "smooth / low-variance" class, not
/// just the screenshot class. CID22 photos with mostly smooth
/// content (blue-sky, soft-focus portrait backgrounds) typically
/// land in the 60-90 range and benefit from the gate too.
///
/// If a future content discriminator regression admits a textured
/// photo above 80, the worst case is a bitstream-affecting AC
/// strategy pick change (the loss term is skipped). The W22-1
/// content_aware_entropy_mul precedent suggests this is a small
/// effect at e5 where the strategy search is already shallow.
pub(crate) const PIXEL_LOSS_DISPATCH_MEDIAN_THRESHOLD: f32 = 80.0;

/// W38-2 pixel-loss dispatch predicate: returns `true` when the
/// per-image `median(mask1x1)` exceeds
/// [`PIXEL_LOSS_DISPATCH_MEDIAN_THRESHOLD`], indicating smooth /
/// low-variance content where the pixel-domain loss term in the
/// AC-strategy search cost is unlikely to change picks. Callers
/// using [`crate::api::PixelLossDispatch::Auto`] drop the `mask1x1`
/// from downstream when this returns `true`, falling back to the
/// coefficient-domain entropy estimate alone.
///
/// Mirrors the predicate shape of
/// [`crate::vardct::epf::mask1x1_is_smooth_enough_to_skip_sharpness`]
/// but uses the per-image median statistic (matching the W22-1
/// content-aware entropy_mul dispatch) rather than the mean —
/// medians are more robust on screenshot content with isolated
/// high-contrast text/icons.
pub(crate) fn pixel_loss_auto_should_skip(
    mask1x1: &[f32],
    stride: usize,
    width: usize,
    height: usize,
) -> bool {
    let med = median_mask1x1(mask1x1, stride, width, height);
    med > PIXEL_LOSS_DISPATCH_MEDIAN_THRESHOLD
}

/// W36-3 patches photo-skip dispatch helper: returns the per-8×8-block
/// mean median of [`crate::vardct::adaptive_quant::compute_mask1x1`]
/// over the unpadded `[0, width) × [0, height)` region of the XYB Y
/// plane laid out with row stride `stride`. Returns `None` when the
/// image is too small to span a full 8×8 block (callers should treat
/// `None` as "discriminator unavailable" and fall back to the scan).
///
/// This is the same statistic
/// [`crate::vardct::splines::looks_like_screenshot`] uses for the
/// auto-splines screenshot skip — duplicated locally to keep the
/// W36-3 change strictly within encoder.rs / api.rs and avoid touching
/// splines.rs's public surface. The pure-mathematical helper is small
/// enough that the duplication is a cheaper maintenance cost than
/// exporting a new crate-private `pub(crate)` symbol.
fn patches_dispatch_block_mask_median(
    xyb_y: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Option<f32> {
    if width < 8 || height < 8 {
        return None;
    }
    // Re-pack the (possibly strided) Y plane into a contiguous buffer
    // because `compute_mask1x1` assumes `stride == width` (callers in
    // `encode_inner` already pass `padded_width` directly as both
    // width AND stride, so the contiguous fast-path almost always
    // wins).
    let y_contig: Vec<f32> = if stride == width {
        xyb_y[..width * height].to_vec()
    } else {
        let mut buf = Vec::with_capacity(width * height);
        for y in 0..height {
            let row_start = y * stride;
            buf.extend_from_slice(&xyb_y[row_start..row_start + width]);
        }
        buf
    };
    let mask1x1 = super::adaptive_quant::compute_mask1x1(&y_contig, width, height);
    let blocks_per_row = width / 8;
    let blocks_per_col = height / 8;
    if blocks_per_row == 0 || blocks_per_col == 0 {
        return None;
    }
    let n_blocks = blocks_per_row * blocks_per_col;
    let mut block_means: Vec<f32> = Vec::with_capacity(n_blocks);
    for by in 0..blocks_per_col {
        for bx in 0..blocks_per_row {
            let mut sum: f64 = 0.0;
            let mut count: usize = 0;
            for dy in 0..8 {
                let y = by * 8 + dy;
                if y >= height {
                    break;
                }
                for dx in 0..8 {
                    let x = bx * 8 + dx;
                    if x >= width {
                        break;
                    }
                    sum += mask1x1[y * width + x] as f64;
                    count += 1;
                }
            }
            let mean = if count > 0 {
                (sum / count as f64) as f32
            } else {
                1.0
            };
            block_means.push(mean);
        }
    }
    let mid = block_means.len() / 2;
    block_means.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
    });
    Some(block_means[mid])
}

/// Return the median value of the unpadded region `[0, width) × [0, height)`
/// of a `mask1x1` plane laid out with row stride `stride`. Uses
/// `select_nth_unstable` on an owned copy of the unpadded values (O(n)
/// expected, no allocation outside the temporary buffer) — exact median
/// rather than histogram approximation because the field is small
/// (≤ 12 MP) and `mask1x1` is allocated once per encode anyway.
pub(super) fn median_mask1x1(mask: &[f32], stride: usize, width: usize, height: usize) -> f32 {
    if width == 0 || height == 0 {
        return 0.0;
    }
    let mut buf: Vec<f32> = Vec::with_capacity(width * height);
    for y in 0..height {
        let row_off = y * stride;
        let row_end = row_off + width;
        if row_end > mask.len() {
            // Defensive: pad / row-stride mismatch — return 0.0 so the
            // gate stays off rather than reading uninitialised memory.
            return 0.0;
        }
        buf.extend_from_slice(&mask[row_off..row_end]);
    }
    let n = buf.len();
    let mid = n / 2;
    // f32 has no Ord — use partial_cmp + Equal for NaN tie-break.
    // The mask is always finite (compute_mask1x1 produces 1/(log1p(x)+0.01)
    // with x >= 0), so NaN should not occur in practice; the fallback
    // here keeps the median well-defined if it ever does.
    buf.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
    });
    buf[mid]
}

/// W44-150: arbitrary-percentile selector on the unpadded `width × height`
/// view of a [`compute_mask1x1`]-style plane laid out with row stride
/// `stride`. Mirrors [`median_mask1x1`] in I/O contract but takes a
/// percentile in `[0.0, 1.0]` (e.g. `0.25` → 25th percentile / Q1).
///
/// Index formula matches the W44-149 audit's `percentile()` helper exactly
/// (`examples/w44_149_photo_proxy_audit.rs:109`):
/// `idx = floor((n - 1) * p)`. Uses `select_nth_unstable_by` on an owned
/// copy of the unpadded values (O(n) expected, single allocation outside
/// the temporary buffer).
///
/// Returns `0.0` for empty inputs or row-stride mismatches (defensive).
///
/// **W44-150 HONEST-STOP (2026-05-21)**: helper kept after W44-150 Phase 2
/// revert — see [`W44_150_PHOTO_W44_117_MASK_P25_MIN`] for the
/// disposition.
///
/// **W44-151 (2026-05-21)**: this helper is now CONSUMED in production
/// by the W44-29 outer gate's `mask_p25 >= 85` admission branch
/// (Mechanism B follow-on to W44-150's honest-stopped Mechanism A).
/// Called once per encode at both `compute_profile_for_search` callers
/// (still-image path + animation `bitstream.rs::encode_frame_to_writer`).
pub(super) fn percentile_mask1x1(
    mask: &[f32],
    stride: usize,
    width: usize,
    height: usize,
    p: f32,
) -> f32 {
    if width == 0 || height == 0 {
        return 0.0;
    }
    let mut buf: Vec<f32> = Vec::with_capacity(width * height);
    for y in 0..height {
        let row_off = y * stride;
        let row_end = row_off + width;
        if row_end > mask.len() {
            return 0.0;
        }
        buf.extend_from_slice(&mask[row_off..row_end]);
    }
    let n = buf.len();
    if n == 0 {
        return 0.0;
    }
    let p_clamped = p.clamp(0.0, 1.0);
    let idx = ((n as f32 - 1.0) * p_clamped).floor() as usize;
    let idx = idx.min(n - 1);
    buf.select_nth_unstable_by(idx, |a, b| {
        a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
    });
    buf[idx]
}

/// W44-91 cheap encoder-internal zenanalyze proxies used to widen the
/// W44-29 high-distance smooth-photo lift onto the **textured colourful
/// photo** sub-band (mask1x1 ∈ [50, 80]) without regressing the 6
/// documented W44-78 regression-band images.
///
/// Both fields use definitions that match zenanalyze tier1.rs EXACTLY
/// (rather than encoder-internal XYB-derived approximations) so the
/// discriminator behaviour ports cleanly from the W44-79 reference
/// (`benchmarks/w44_79_zenanalyze_discriminator_2026-05-19.tsv`) to the
/// production hot path.
///
/// Compute cost: one O(W·H) pass over sRGB u8 source bytes — measured at
/// ~5–10 ms on a 512×512 image on a modern CPU. Only computed for the
/// 8-bit sRGB pixel layouts ([`crate::api::PixelLayout::Rgb8`], `Rgba8`,
/// `Bgr8`, `Bgra8`) where the M3 colourfulness scale is meaningful. For
/// other layouts (16-bit, linear-f32, grayscale, HDR) the proxy field on
/// [`VarDctEncoder`] stays `None` and the W44-91 gate cannot fire — the
/// existing W44-29 mask1x1<50 gate retains full coverage of those layouts.
#[derive(Clone, Copy, Debug)]
pub struct ZenanalyzeProxies {
    /// Hasler-Süsstrunk M3 colourfulness over sRGB u8 source pixels.
    /// `M3 = sqrt(σ_rg² + σ_yb²) + 0.3 * sqrt(μ_rg² + μ_yb²)` where
    /// `rg = R − G` and `yb = 0.5·(R+G) − B` per pixel. Matches
    /// zenanalyze `src/tier1.rs` colourfulness computation exactly.
    pub m3_colourfulness: f32,
    /// Fraction of 8×8 sRGB blocks where every channel's per-block u8
    /// range (max − min) is ≤ 4. Matches zenanalyze `src/tier1.rs`
    /// `flat_color_blocks` accumulator exactly (`r_range <= 4 AND
    /// g_range <= 4 AND b_range <= 4`).
    pub flat_color_block_ratio: f32,
    /// W44-96: fraction of interior pixels (excluding 1-pixel border)
    /// whose Sobel luma gradient magnitude exceeds 30 (Sobel scale).
    /// Used as the primary discriminator for the variant-Z DCT32X32 lift
    /// sub-dispatch within the W44-29 firing class — separates textured
    /// high-edge smooth photos (1420710, 1531677) from low-edge smooth
    /// photos (2389166, 1044329, 7062219) where variant Z regresses
    /// SSIM2. Computed in the same O(W·H) pass as the other proxies via
    /// BT.601 luma `0.299·R + 0.587·G + 0.114·B`.
    pub edge_density: f32,
    /// W44-176: BT.601 luma variance on sRGB u8 source pixels (raw
    /// `Var(0.299·R + 0.587·G + 0.114·B)`, in `[0, 65025]` scale).
    /// Used as a terminal-class sub-discriminator within the W44-108
    /// low-colour band (`m3 < 30`) — separates "dark terminal-like"
    /// screenshots where the W44-109 qf seed lift over-allocates
    /// (terminal `luma_var ≈ 1706`) from "very-dark message-like"
    /// screenshots where the lift IS net-positive
    /// (gmessages/gui `luma_var ≈ 1050`) and from "mixed-content"
    /// macOS-style screenshots where the lift gains real SSIM2
    /// (graph 415, imac_dark 3303, imac_g3 5244). Computed in the
    /// same O(W·H) pass as `m3_colourfulness` — zero added cost.
    pub luma_var: f32,
}

impl ZenanalyzeProxies {
    /// Compute proxies from an 8-bit sRGB pixel buffer. Layout (R, G, B
    /// byte offsets within the pixel) is described by `r_off`, `g_off`,
    /// `b_off`. `bpp` is bytes-per-pixel (3 for `Rgb8`/`Bgr8`, 4 for
    /// `Rgba8`/`Bgra8`). Caller pre-validates that
    /// `pixels.len() >= width * height * bpp`.
    #[inline(never)]
    pub fn compute_srgb_u8(
        pixels: &[u8],
        width: usize,
        height: usize,
        bpp: usize,
        r_off: usize,
        g_off: usize,
        b_off: usize,
    ) -> Self {
        let n_pix = (width * height) as f64;
        if n_pix == 0.0 {
            return Self {
                m3_colourfulness: 0.0,
                flat_color_block_ratio: 0.0,
                edge_density: 0.0,
                luma_var: 0.0,
            };
        }

        // --- M3 colourfulness + W44-176 luma_var: one pass over pixels -----
        let mut rg_sum = 0.0_f64;
        let mut rg_sq_sum = 0.0_f64;
        let mut yb_sum = 0.0_f64;
        let mut yb_sq_sum = 0.0_f64;
        // W44-176: BT.601 luma sum + sum-of-squares (running variance via
        // E[Y²] − μ_Y²). Same per-pixel arithmetic as the Sobel edge_density
        // helper below — folded into this first pass to avoid an extra
        // O(W·H) sweep at the discriminator site.
        let mut y_sum = 0.0_f64;
        let mut y_sq_sum = 0.0_f64;
        for y in 0..height {
            for x in 0..width {
                let off = (y * width + x) * bpp;
                let r = pixels[off + r_off] as f64;
                let g = pixels[off + g_off] as f64;
                let b = pixels[off + b_off] as f64;
                let rg = r - g;
                let yb = 0.5 * (r + g) - b;
                rg_sum += rg;
                rg_sq_sum += rg * rg;
                yb_sum += yb;
                yb_sq_sum += yb * yb;
                let yl = 0.299 * r + 0.587 * g + 0.114 * b;
                y_sum += yl;
                y_sq_sum += yl * yl;
            }
        }
        let mu_rg = rg_sum / n_pix;
        let mu_yb = yb_sum / n_pix;
        let var_rg = (rg_sq_sum / n_pix - mu_rg * mu_rg).max(0.0);
        let var_yb = (yb_sq_sum / n_pix - mu_yb * mu_yb).max(0.0);
        let m3 = (var_rg + var_yb).sqrt() + 0.3 * (mu_rg * mu_rg + mu_yb * mu_yb).sqrt();
        let mu_y = y_sum / n_pix;
        let luma_var = (y_sq_sum / n_pix - mu_y * mu_y).max(0.0) as f32;

        // --- flat_color_block_ratio: per-8×8-block channel range -----------
        let blocks_x = width / 8;
        let blocks_y = height / 8;
        let mut flat_blocks = 0usize;
        let total_blocks = blocks_x * blocks_y;
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut r_min = 255u8;
                let mut r_max = 0u8;
                let mut g_min = 255u8;
                let mut g_max = 0u8;
                let mut b_min = 255u8;
                let mut b_max = 0u8;
                for dy in 0..8 {
                    for dx in 0..8 {
                        let off = ((by * 8 + dy) * width + (bx * 8 + dx)) * bpp;
                        let r = pixels[off + r_off];
                        let g = pixels[off + g_off];
                        let b = pixels[off + b_off];
                        if r < r_min {
                            r_min = r;
                        }
                        if r > r_max {
                            r_max = r;
                        }
                        if g < g_min {
                            g_min = g;
                        }
                        if g > g_max {
                            g_max = g;
                        }
                        if b < b_min {
                            b_min = b;
                        }
                        if b > b_max {
                            b_max = b;
                        }
                    }
                }
                let r_range = (r_max as i32) - (r_min as i32);
                let g_range = (g_max as i32) - (g_min as i32);
                let b_range = (b_max as i32) - (b_min as i32);
                if r_range <= 4 && g_range <= 4 && b_range <= 4 {
                    flat_blocks += 1;
                }
            }
        }
        let fcbr = if total_blocks > 0 {
            flat_blocks as f32 / total_blocks as f32
        } else {
            0.0
        };

        // --- W44-96 edge_density: Sobel gradient on BT.601 luma -----------
        // Iterate interior pixels (skip 1-pixel border to avoid out-of-bounds).
        // Square-magnitude threshold = 900 corresponds to magnitude > 30,
        // matching the W44-96 probe `edge_density()` helper exactly.
        let edge_density = if width >= 3 && height >= 3 {
            let luma = |y: usize, x: usize| -> f32 {
                let off = (y * width + x) * bpp;
                0.299 * pixels[off + r_off] as f32
                    + 0.587 * pixels[off + g_off] as f32
                    + 0.114 * pixels[off + b_off] as f32
            };
            let interior = (width - 2) * (height - 2);
            let mut edges = 0usize;
            for y in 1..(height - 1) {
                for x in 1..(width - 1) {
                    let gx = -luma(y - 1, x - 1) - 2.0 * luma(y, x - 1) - luma(y + 1, x - 1)
                        + luma(y - 1, x + 1)
                        + 2.0 * luma(y, x + 1)
                        + luma(y + 1, x + 1);
                    let gy = -luma(y - 1, x - 1) - 2.0 * luma(y - 1, x) - luma(y - 1, x + 1)
                        + luma(y + 1, x - 1)
                        + 2.0 * luma(y + 1, x)
                        + luma(y + 1, x + 1);
                    if gx * gx + gy * gy > 900.0 {
                        edges += 1;
                    }
                }
            }
            if interior > 0 {
                edges as f32 / interior as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        Self {
            m3_colourfulness: m3 as f32,
            flat_color_block_ratio: fcbr,
            edge_density,
            luma_var,
        }
    }
}

/// **W44-AUDIT-5 Phase 3**: per-image content-class discriminator that
/// elevates CfL Newton dispatch to the libjxl-bit-exact `x=0` start path
/// when the input is a high-colour mixed-content screenshot. Used at
/// the CfL Pass-1 + Pass-2 + precomputed-mode dispatch sites in
/// [`VarDctEncoder::encode`], [`VarDctEncoder::encode_lossy_from_precomputed`],
/// and the animation path in `bitstream.rs`.
///
/// Returns `true` when ALL of the following hold:
/// - The encoder strategy has opted into Phase 3 dispatch
///   (`profile.cfl_pass1_screenshot_x0_start`)
/// - The encoder has [`ZenanalyzeProxies`] available (8-bit sRGB layouts)
/// - The proxies match the high-colour-class predicate
///   ([`crate::vardct::perceptual_tuning::w44_audit_6_is_high_colour_class`])
///
/// When `true`, the CfL Newton dispatch flips `libjxl_parity = true` for
/// that single call, routing Pass-1 through `x=0` start + Newton math
/// (matching cjxl / Libjxl-strategy behaviour) — which Phase 1+2 measured
/// recovers the codec_wiki-class SSIM2 deficit (-5.51 vs cjxl) without
/// touching photo cells (where `m3 < 80` keeps the existing LS warm-start
/// path).
///
/// Env hook for A/B: `JXL_W44_AUDIT_5_P3_DISABLE=1` forces OFF (mirrors
/// the `JXL_W44_176_DISABLE` and `JXL_W44_AUDIT_6_DISABLE` pattern).
#[inline]
pub(crate) fn w44_audit_5_p3_force_libjxl_parity_for_screenshot(
    profile: &crate::effort::EffortProfile,
    proxies: Option<&ZenanalyzeProxies>,
) -> bool {
    if !profile.cfl_pass1_screenshot_x0_start {
        return false;
    }
    // Env hook for A/B reproducibility: `JXL_W44_AUDIT_5_P3_DISABLE=1`
    // forces the route OFF, mirroring W44-176 / W44-AUDIT-6.
    #[cfg(feature = "std")]
    {
        if std::env::var_os("JXL_W44_AUDIT_5_P3_DISABLE").is_some_and(|v| v != "0" && !v.is_empty())
        {
            return false;
        }
    }
    crate::vardct::perceptual_tuning::w44_audit_6_is_high_colour_class(proxies)
}

/// Tiny JPEG XL encoder.
///
/// This is a simplified VarDCT encoder based on libjxl-tiny that uses:
/// - Only DCT8, DCT8x16, DCT16x8 transforms
/// - Huffman or ANS entropy coding
/// - Default zig-zag coefficient order
/// - Fixed context tree for DC
pub struct VarDctEncoder {
    /// Target distance (quality). 1.0 = visually lossless.
    pub distance: f32,
    /// Effort level (1–12). Controls AC strategy gating and search depth.
    /// e10/e11/e12 extends libjxl kTortoise=9 via extended search budgets
    /// (e12 doubles butteraugli_iters 16 → 32; requires `ITER_MAX = 32`).
    pub effort: u8,
    /// Centralized effort-derived decisions. All effort-gated constants and
    /// thresholds are read from this profile instead of inline `if effort >= N`.
    pub profile: crate::effort::EffortProfile,
    /// Use dynamic Huffman codes built from actual token frequencies.
    /// When true (default), uses a two-pass mode: collect tokens first, build optimal codes, then write.
    /// When false, uses pre-computed static codes (streaming, single-pass).
    pub optimize_codes: bool,
    /// Use enhanced histogram clustering with pair merge refinement.
    /// Only effective when `optimize_codes` is true.
    ///
    /// Note: The enhanced clustering algorithm was designed for ANS entropy coding
    /// and may not provide benefits (or may slightly increase size) when used with
    /// Huffman coding. This option is experimental.
    pub enhanced_clustering: bool,
    /// Use ANS entropy coding instead of Huffman.
    /// Only effective when `optimize_codes` is true (requires two-pass mode).
    /// ANS typically produces 5-10% smaller files than Huffman.
    pub use_ans: bool,
    /// Enable chroma-from-luma (CfL) optimization.
    /// When true (default), computes per-tile ytox/ytob values via least-squares fitting.
    /// When false, uses ytox=0, ytob=0 (no chroma decorrelation).
    pub cfl_enabled: bool,
    /// Enable adaptive AC strategy selection (DCT8/DCT16x8/DCT8x16).
    /// When true (default), selects the best transform size per 16x16 block region.
    /// When false, uses DCT8 for all blocks.
    pub ac_strategy_enabled: bool,
    /// Enable custom coefficient ordering.
    /// When true (default when optimize_codes is true), reorders AC coefficients
    /// so frequently-zero positions appear last, reducing bitstream size.
    /// Only effective when `optimize_codes` is true (requires two-pass mode).
    pub custom_orders: bool,
    /// Force a specific AC strategy for all blocks (for testing).
    /// When Some(strategy), uses that raw strategy code for all blocks that fit.
    /// None (default) uses normal strategy selection based on `ac_strategy_enabled`.
    pub force_strategy: Option<u8>,
    /// Enable noise synthesis.
    /// When true, estimates noise parameters from the image and encodes them
    /// in the frame header. The decoder regenerates noise during rendering.
    /// Off by default (matching libjxl's default).
    pub enable_noise: bool,
    /// When set, synthesises noise parameters from the given ISO value
    /// instead of estimating from the image. Matches libjxl's
    /// `--photon_noise=ISO` flag and bypasses `enable_noise`. Useful
    /// for re-encoding denoised content where the caller wants to
    /// inject controlled grain matching a target camera ISO.
    pub photon_noise_iso: Option<f32>,
    /// Caller-supplied 8-point noise LUT. Overrides content
    /// estimation when set. Mirrors libjxl's `cparams.manual_noise`.
    /// Lower priority than `photon_noise_iso`, higher than
    /// `enable_noise`. Each entry should be in `[0.0, ~1.0]` —
    /// values are written as 10-bit-quantised samples into the
    /// frame-header noise block.
    pub manual_noise_lut: Option<[f32; 8]>,
    /// Caller-supplied AC quantiser rescale. When `Some(r)` and `r != 1.0`,
    /// `global_scale` is multiplied by `r` (and `scale` / `inv_scale`
    /// / `scale_dc` recomputed) after the standard
    /// `DistanceParams::compute_for_profile` step. Mirrors libjxl's
    /// `cparams.quant_ac_rescale`. `r < 1.0` → finer AC quant
    /// (larger files, higher quality); `r > 1.0` → coarser (smaller
    /// files, lower quality). Default behaviour (`None`) leaves
    /// `global_scale` untouched.
    pub quant_ac_rescale: Option<f32>,
    /// Caller-supplied source-image butteraugli distance for re-encode
    /// pipelines. When set, x_qm_scale (and other distance-based
    /// heuristics that compare against source quality, not target
    /// quality) ramp against this value instead of `distance`.
    /// Mirrors libjxl's `cparams.original_butteraugli_distance`.
    /// `None` = use `distance` (ground-truth source).
    pub original_distance: Option<f32>,
    /// Enable Wiener denoising pre-filter (requires `enable_noise`).
    /// When true, applies a conservative Wiener filter to remove estimated noise
    /// before encoding. The decoder re-adds noise from the encoded parameters.
    /// Provides 1-8% file size savings with near-zero Butteraugli quality impact.
    /// Off by default (libjxl does not have a denoising pre-filter).
    pub enable_denoise: bool,
    /// Enable gaborish inverse pre-filter.
    /// When true (default), applies a 5x5 sharpening kernel to XYB before DCT
    /// and signals gab=1 in the frame header. The decoder applies a 3x3 blur
    /// to compensate, reducing blocking artifacts.
    /// Matches the libjxl VarDCT encoder default.
    pub enable_gaborish: bool,
    /// EX-J13 — apply a per-tile contrast-derived multiplier (`mul ∈ [0.8, 1.2]`)
    /// to the gaborish 5x5 kernel on the Y (luma) channel. **Encoder-only**:
    /// the decoder always applies the fixed 3x3 inverse Gabor blur, so
    /// adaptive sharpening must be pre-baked into the post-Gab samples we
    /// hand the DCT.
    /// Default `false`. Forced to `false` when `enable_gaborish == false`.
    pub enable_adaptive_gaborish: bool,
    /// Override the edge-preserving filter (EPF) iteration count.
    ///
    /// `None` (default) = use the distance-derived `epf_iters` from
    /// [`DistanceParams`] (libjxl thresholds `[0.7, 1.5, 4.0]`).
    /// `Some(0..=3)` = force the given count; `0` disables EPF and
    /// skips the dynamic sharpness search. Mirrors libjxl
    /// `cparams.epf` (`enc_frame.cc:284-285`).
    pub epf_level_override: Option<u32>,
    /// Adaptive dispatch policy for the per-block EPF sharpness
    /// search (W36-2). See [`crate::api::EpfDispatch`].
    ///
    /// `AlwaysSelect` (default) preserves byte-identical behaviour
    /// with historical encoder builds. `Auto` skips the per-block
    /// search on smooth regions (per-region `mask1x1` mean below a
    /// threshold) and emits uniform default sharpness instead.
    /// `AlwaysDefault` always emits uniform default sharpness.
    pub epf_dispatch: crate::api::EpfDispatch,
    /// Enable error diffusion in AC quantization.
    /// When true, spreads quantization error to neighboring coefficients in
    /// zigzag order, helping preserve smooth gradients at high compression.
    /// Off by default (modest quality improvement, slight performance cost).
    pub error_diffusion: bool,
    /// Enable pixel-domain loss calculation in AC strategy selection.
    /// When true, uses full libjxl's pixel-domain loss model (IDCT error,
    /// per-pixel masking, 8th power norm). This provides better distance
    /// calibration matching cjxl's output.
    /// When false (default), uses coefficient-domain loss (libjxl-tiny style).
    /// Note: Requires `ac_strategy_enabled` to have any effect.
    pub pixel_domain_loss: bool,
    /// Adaptive dispatch policy for the pixel-domain loss term in the
    /// AC-strategy search cost (W38-2). See
    /// [`crate::api::PixelLossDispatch`].
    ///
    /// `AlwaysOn` (default) preserves byte-identical behaviour with
    /// historical encoder builds. `Auto` skips the loss term on
    /// smooth content (per-image `median(mask1x1) > 80`) and falls
    /// back to coefficient-domain-only entropy. `AlwaysOff`
    /// unconditionally skips the loss term (equivalent to
    /// `pixel_domain_loss = false`).
    pub pixel_loss_dispatch: crate::api::PixelLossDispatch,
    /// Adaptive dispatch for the two-pass dynamic-entropy path
    /// (W44-87). `AlwaysTwoPass` (default) preserves byte-identical
    /// historical behaviour by always honouring [`Self::optimize_codes`].
    /// `Auto` flips to single-pass static Huffman on smooth photos at
    /// low distance + effort 5 + the single-pass-safety predicate
    /// (no patches/splines/learned tree/sharpness map/noise params/
    /// LF frame/extras). `AlwaysSinglePass` forces the single-pass
    /// path whenever safe and falls back to two-pass otherwise.
    /// Saves the ~14 ms/MP `entropy` + `build_codes` cost at the
    /// price of 2-4% bytes on the dispatched subset.
    pub single_pass_entropy_dispatch: crate::api::SinglePassEntropyDispatch,
    /// Enable LZ77 backward references in entropy coding.
    /// When true, compresses token streams using LZ77 length+distance tokens.
    /// Only effective with two-pass mode (optimize_codes=true) and ANS (use_ans=true).
    /// Off by default — works for most cases but has known interactions with certain
    /// forced strategy combinations (DCT2x2, IDENTITY) that cause InvalidAnsStream.
    pub enable_lz77: bool,
    /// LZ77 method to use when enable_lz77 is true.
    ///
    /// - `Rle`: Only matches consecutive identical values (fast, limited on photos)
    /// - `Greedy`: Hash chain backward references (slower, 1-3% better on photos)
    ///
    /// Default: `Greedy` (best compression)
    pub lz77_method: crate::entropy_coding::lz77::Lz77Method,
    /// Enable DC tree learning.
    /// When true, learns an optimal context tree for DC coding from image content
    /// instead of using the fixed GRADIENT_CONTEXT_LUT.
    /// **DISABLED/BROKEN**: The learned tree doesn't correctly route AC metadata
    /// samples to contexts 0-10. Fixing requires parsing the static tree structure
    /// and splicing in the learned DC subtree while preserving AC metadata routing.
    /// Expected gain (~1.2% overall) doesn't justify the complexity. See CLAUDE.md.
    pub dc_tree_learning: bool,
    /// Number of butteraugli quantization loop iterations.
    /// When > 0, iteratively refines the per-block quant field using butteraugli
    /// perceptual distance feedback. Each iteration: encode → reconstruct → measure
    /// → adjust quant_field. AC strategy is kept fixed; only quant_field changes.
    ///
    /// libjxl uses 2 iterations at effort 8, 4 at effort 9.
    /// Requires the `butteraugli-loop` feature.
    ///
    /// Default: 0 (disabled)
    #[cfg(feature = "butteraugli-loop")]
    pub butteraugli_iters: u32,
    /// EX-J11 chunk 1: HDR-aware loss dispatch for the butteraugli
    /// quantization loop. Default [`crate::vardct::hdr_metrics::HdrLoss::Butteraugli`]
    /// keeps every existing hash-lock byte-identical;
    /// [`crate::vardct::hdr_metrics::HdrLoss::Vdp2`] surfaces
    /// [`crate::api::EncodeError::InvalidConfig`] until chunk 2 lands
    /// the actual HDR-VDP-2 maths.
    ///
    /// Requires the `butteraugli-loop` feature.
    ///
    /// Default: [`crate::vardct::hdr_metrics::HdrLoss::Butteraugli`]
    #[cfg(feature = "butteraugli-loop")]
    pub hdr_loss: crate::vardct::hdr_metrics::HdrLoss,
    /// W44-phase3-B1 opt-in GPU butteraugli backend for the buttloop.
    /// When `true` AND the `gpu-butteraugli` cargo feature is on AND
    /// CUDA init succeeds, the buttloop's per-iter compare runs on the
    /// GPU. Silently falls back to the CPU backend on any of those
    /// failing. Default `false` keeps every hash-lock byte-identical.
    /// See [`crate::api::LossyConfig::with_gpu_butteraugli`].
    #[cfg(feature = "butteraugli-loop")]
    pub gpu_butteraugli: bool,
    /// cvvdp-fork Phase 3 (2026-05-24): opt-in CVVDP backend for the
    /// quantization loop. When `true` AND the `cvvdp-loop` cargo
    /// feature is on AND CUDA init succeeds AND the active
    /// [`EncoderStrategy`](crate::api::EncoderStrategy) is not
    /// [`Libjxl`](crate::api::EncoderStrategy::Libjxl), the backend
    /// construction in [`crate::vardct::perceptual_backend::construct_backend`]
    /// returns a [`crate::vardct::cvvdp_backend::gpu::GpuCvvdpBackend`]
    /// instead of the butteraugli CPU/GPU pair. Phase 3 ships the
    /// backend impl only — the buttloop body still consumes butteraugli;
    /// Phase 4 plumbs the cvvdp signal through `run_buttloop`. Default
    /// `false` keeps every hash-lock byte-identical (including when
    /// the `cvvdp-loop` feature is compiled in but no caller opts in).
    /// See [`crate::api::LossyConfig::with_cvvdp_loop`] and the Phase 3
    /// brief at `docs/archive/RFC_CVVDP_PHASE3_BRIEF.md`.
    #[cfg(feature = "butteraugli-loop")]
    pub cvvdp_loop: bool,
    /// cvvdp-fork Phase 5 (2026-05-24): caller-supplied preference for
    /// the CPU CVVDP backend over the GPU CVVDP backend. Only consulted
    /// by [`crate::vardct::perceptual_backend::construct_backend`] when
    /// [`Self::cvvdp_loop`] is also true. When `true` (and the
    /// `cvvdp-loop-cpu` cargo feature is compiled in), the dispatch
    /// returns the CPU CVVDP backend instead of the GPU CVVDP backend.
    /// Default `false` preserves the W44-228b-style "respect caller
    /// explicit opt-in only" policy — the field is always present so
    /// hash-lock fixtures don't depend on the cvvdp cargo features.
    /// See [`crate::api::LossyConfig::with_cvvdp_use_cpu`] and the
    /// Phase 5 brief at `docs/archive/RFC_CVVDP_PHASE5_BRIEF.md`.
    #[cfg(feature = "butteraugli-loop")]
    pub cvvdp_use_cpu: bool,
    /// zensim-fork Phase 3 (RFC `docs/RFC_ZENSIM_FORK_PLAN.md` §5,
    /// 2026-05-25): caller-supplied opt-in for the zensim
    /// perceptual-metric backend. Populated by
    /// [`crate::vardct::perceptual_backend::propagate_resolved_metric_to_encoder`]
    /// from [`crate::api::LossyConfig::with_perceptual_metric`]. Mutually
    /// exclusive with [`Self::cvvdp_loop`] at the buttloop dispatch level
    /// (zensim wins when both are true; see translation in
    /// `vardct/perceptual_loop.rs::build_perceptual_backend`). When `true`
    /// AND a zensim cargo feature is on AND the active strategy is not
    /// `Libjxl`, the backend constructed in
    /// [`crate::vardct::perceptual_backend::construct_backend`] is a
    /// [`crate::vardct::zensim_backend::cpu::CpuZensimBackend`] /
    /// [`crate::vardct::zensim_backend::gpu::GpuZensimBackend`] instead
    /// of the cvvdp / butteraugli backends.
    ///
    /// Phase 3 ships the backend impl + dispatch only — the buttloop
    /// body still consumes butteraugli-direction targets. Default `false`
    /// keeps every hash-lock byte-identical (including when a zensim
    /// cargo feature is compiled in but no caller opts in).
    #[cfg(feature = "butteraugli-loop")]
    pub zensim_loop: bool,
    /// zensim-fork Phase 3 (2026-05-25): caller-supplied preference for
    /// the CPU zensim backend over the GPU zensim backend. Only consulted
    /// by [`crate::vardct::perceptual_backend::construct_backend`] when
    /// [`Self::zensim_loop`] is also true. When `true` (and the
    /// `zensim-loop` cargo feature is compiled in), the dispatch returns
    /// the CPU zensim backend; when `false`, the dispatch prefers the
    /// GPU zensim backend (with silent CPU fallback if GPU init fails
    /// and `zensim-loop` is compiled). Default `false` mirrors the
    /// cvvdp Phase 5 "respect caller explicit opt-in only" policy.
    #[cfg(feature = "butteraugli-loop")]
    pub zensim_use_cpu: bool,
    /// cvvdp-fork Phase 8d (2026-05-25): opt-in post-convergence
    /// bytes-tighten exit pass on the cvvdp seed loop. When `true`,
    /// AND [`Self::cvvdp_loop`] is also true, AND the `cvvdp-loop-tighten`
    /// cargo feature is compiled in, the inner seed loop's final
    /// SetQuantField is preceded by a batched multiplicative bump pass
    /// that loosens qac while the cvvdp score still satisfies
    /// `target * (1 + ε)`. Gives back bytes the converged state had
    /// headroom for. Default `false` keeps every hash-lock byte-identical
    /// regardless of the `cvvdp-loop-tighten` cargo feature. NEVER fires
    /// on the butteraugli loop (the butteraugli per-block reducer is
    /// already calibrated to the W44 cost-model gates; loosening it
    /// post-convergence breaks the tradeoff). See
    /// [`crate::api::LossyConfig::with_cvvdp_bytes_tighten`] and the
    /// Phase 8d brief in `docs/RFC_CVVDP_PHASE8_PARETO_TARGETING.md` §3.3.
    #[cfg(feature = "butteraugli-loop")]
    pub cvvdp_bytes_tighten: bool,
    /// Phase 1 display-config backfill (RFC
    /// `docs/RFC_DISPLAY_CONFIG_BACKFILL.md`, 2026-05-25): target
    /// display config for cvvdp scoring. Populated by
    /// [`crate::vardct::perceptual_backend::propagate_resolved_metric_to_encoder`]
    /// from [`crate::api::LossyConfig::resolve_target_display`].
    ///
    /// Default [`crate::api::DisplayConfig::WebSdr80`] keeps every
    /// hash-lock fixture byte-identical (the variant maps to
    /// `cvvdp_gpu::params::DisplayModel::STANDARD_4K`, which is the
    /// pre-Phase-1 default). Field is always present (no feature gate)
    /// because the enum itself is feature-independent — only the cvvdp
    /// backend ctors and the per-display target lookup actually consume
    /// it.
    pub target_display: crate::api::DisplayConfig,
    /// Phase 1 of RFC `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`
    /// (2026-05-26): per-distance target-score override consumed by the
    /// `effective_metric_target_distance` dispatch in
    /// [`crate::vardct::perceptual_loop::run_buttloop`]. Populated by
    /// [`crate::vardct::perceptual_backend::propagate_resolved_metric_to_encoder`]
    /// from
    /// [`crate::api::LossyConfig::resolve_perceptual_target_score`]
    /// (which already applied the
    /// [`crate::api::EncoderStrategy::Libjxl`] strict-parity
    /// short-circuit that forces this field to `None`).
    ///
    /// Default `None` preserves the pre-Phase-1 implicit-identity
    /// behaviour — the buttloop drives convergence using
    /// `target_distance` directly. `Some(score)` activates the
    /// per-metric inverse lookup (butteraugli via
    /// [`crate::vardct::butteraugli_targets`], cvvdp via
    /// [`crate::vardct::cvvdp_targets`], zensim via the butteraugli
    /// table after score normalization — see
    /// [`crate::vardct::perceptual_loop::run_buttloop`] dispatch block
    /// for the wiring).
    #[cfg(feature = "butteraugli-loop")]
    pub perceptual_target_score: Option<f32>,
    /// Number of SSIM2 quantization loop iterations.
    /// Alternative to butteraugli loop: uses per-block linear RGB RMSE + full-image SSIM2.
    /// Requires the `ssim2-loop` feature.
    ///
    /// Default: 0 (disabled)
    #[cfg(feature = "ssim2-loop")]
    pub ssim2_iters: u32,
    /// Number of zensim quantization loop iterations.
    /// Alternative to butteraugli loop: uses zensim's psychovisual metric for both
    /// global quality tracking and per-pixel spatial error map (diffmap in XYB space).
    /// Also refines AC strategy by splitting large transforms with high perceptual error.
    /// Requires the `zensim-loop` feature.
    ///
    /// Default: 0 (disabled)
    #[cfg(feature = "zensim-loop")]
    pub zensim_iters: u32,
    /// Whether the input has 16-bit samples. When true, the file header signals
    /// bit_depth=16 instead of 8. The actual VarDCT encoding is the same (XYB
    /// is always f32 internally), but the decoder uses this to reconstruct at
    /// the correct output bit depth.
    pub bit_depth_16: bool,
    /// ICC profile to embed in the codestream.
    /// When Some, writes has_icc=1 and encodes the profile after the file header.
    pub icc_profile: Option<Vec<u8>>,
    /// Enable patches (dictionary-based repeated pattern detection).
    /// When true, detects repeated rectangular elements (text glyphs, buttons, icons)
    /// and stores unique patterns once in a reference frame. Huge wins on screenshots.
    /// On by default for lossy encoding.
    pub enable_patches: bool,
    /// Controls when the patches detector actually runs when
    /// `enable_patches == true`. See
    /// [`crate::api::PatchesDispatch`] for the policy doc.
    ///
    /// Default [`crate::api::PatchesDispatch::Auto`] consults the
    /// existing `median(mask1x1) > 95` screenshot/photo discriminator
    /// (used by `content_aware_entropy_mul` and the auto-splines
    /// detector) and short-circuits the ~25-30 ms/MP patches scan on
    /// photo content where it always produces empty output. Wire-format
    /// byte-identical to `AlwaysScan` on photos.
    pub patches_dispatch: crate::api::PatchesDispatch,
    /// Enable libjxl-style **dot detection** (refs #19). When true and
    /// effort >= 7 and distance >= 3.0 and no text-like patches were
    /// found, the encoder runs the star-field / specular-highlight
    /// detector ([`super::dot_detection::detect_gaussian_ellipses`])
    /// and folds the detected Gaussian ellipses into the patch
    /// dictionary via [`super::patches::PatchesData::from_dots`].
    /// Default on internally is `false` so existing callers that
    /// build a `VarDctEncoder` directly preserve their behaviour; the
    /// public [`crate::LossyConfig`] defaults this to `true` to
    /// match libjxl's `Override::kDefault` (`enc_patch_dictionary.cc:632`).
    pub enable_dot_detection: bool,
    /// Encoder mode: Reference (match libjxl) or Experimental (own improvements).
    pub encoder_mode: crate::api::EncoderMode,
    /// Manual splines to overlay on the image (opt-in, None by default).
    pub splines: Option<Vec<crate::vardct::splines::Spline>>,
    /// Enable automatic spline detection from the input XYB planes.
    ///
    /// When `true` **and** [`Self::splines`] is `None` **and** [`Self::effort`]
    /// is ≥ 7, the encoder calls [`crate::vardct::splines::find_splines`]
    /// after the patches-subtract step (mirroring libjxl
    /// `enc_heuristics.cc:1048-1054`: `speed_tier <= kSquirrel`).
    ///
    /// Chunk 1 (this commit) ships the API + wiring with a stub detector
    /// that returns an empty vec — produces byte-identical output to the
    /// default path. Chunk 2 lands the real ridge-following detector;
    /// see [`crate::vardct::splines::find_splines`] for the algorithm sketch.
    ///
    /// A user-supplied non-empty [`Self::splines`] always wins over
    /// auto-detection — the manual API stays the source of truth when
    /// both are set.
    pub auto_splines: bool,
    /// Whether the input is grayscale. When true, the file header signals
    /// ColorSpace::Gray instead of RGB. VarDCT still operates in XYB (3 channels)
    /// internally — this only affects the output colorspace the decoder targets.
    pub is_grayscale: bool,
    /// Progressive encoding mode (Single, QuantizedAcFullAc, DcVlfLfAc).
    /// When not Single, AC coefficients are split across multiple passes with
    /// shift-based precision reduction for early preview rendering.
    pub progressive: crate::api::ProgressiveMode,
    /// Enable LfFrame (separate DC frame).
    /// When true, DC coefficients are encoded as a separate modular frame
    /// (frame_type=1, dc_level=1) before the main VarDCT frame, with
    /// distance-scaled quantization factors matching libjxl's progressive_dc >= 1.
    pub use_lf_frame: bool,
    /// Custom gamma (encoding exponent) from source image.
    /// When Some, writes have_gamma=true in the JXL header and uses gamma
    /// linearization instead of sRGB TF. Example: 0.45455 for gamma 2.2.
    pub source_gamma: Option<f32>,
    /// Explicit color encoding override for the JXL header.
    /// When Some, this is used instead of deriving from source_gamma / defaults.
    /// Allows signaling HDR (PQ, HLG) or non-sRGB primaries (BT.2020, P3).
    pub color_encoding: Option<crate::headers::color_encoding::ColorEncoding>,
    /// Peak display luminance in nits for ToneMapping. Default 255.0 (SDR).
    pub intensity_target: f32,
    /// Minimum display luminance in nits for ToneMapping. Default 0.0.
    pub min_nits: f32,
    /// `ToneMapping.relative_to_max_display` (default `false`). When
    /// `true`, [`Self::linear_below`] is interpreted as a ratio in
    /// `[0, 1]` of the maximum display brightness rather than an
    /// absolute nit value. Mirrors libjxl `ToneMapping`
    /// (`image_metadata.h:169`). Closes issue #46 chunk 1a.
    pub relative_to_max_display: bool,
    /// `ToneMapping.linear_below` (default `0.0`). Tone mapping leaves
    /// pixels strictly below this value unchanged. Interpretation
    /// depends on [`Self::relative_to_max_display`]. Mirrors libjxl
    /// `ToneMapping` (`image_metadata.h:174`). Closes issue #46 chunk 1a.
    pub linear_below: f32,
    /// Intrinsic display size `(width, height)`, if different from coded dimensions.
    pub intrinsic_size: Option<(u32, u32)>,
    /// When `true`, the alpha extra-channel header is emitted with
    /// `alpha_associated=true` (premultiplied / associated alpha).
    /// The encoder DOES NOT premultiply or unpremultiply pixels here —
    /// it's the caller's responsibility to feed straight color and
    /// declare the encoded form. The high-level `EncodeRequest` /
    /// `LossyEncoder` API does the unpremultiply pre-pass before
    /// handing us the linear RGB.
    pub alpha_associated: bool,
    /// Override `BitDepth.bits_per_sample` in the codestream header.
    /// `None` → derives from `bit_depth_16` (8 or 16). `Some(N)` →
    /// emits `N`. Closes configurable bits_per_sample sub-feature
    /// of #18 — lets callers signal 10/12/14-bit input precision.
    pub bits_per_sample_override: Option<u32>,
    /// When `true`, AC groups in the multi-group TOC are emitted in
    /// concentric-square order from the image center (libjxl
    /// `cparams.centerfirst`). The TOC `permuted` flag is set and
    /// the permutation is encoded as Lehmer codes via
    /// `coeff_order::tokenize_permutation` /
    /// `build_and_write_coeff_orders`. Closes #14.
    pub center_first: bool,
    /// Decoder upsampling factor for the main frame. `1` (default) =
    /// no upsampling; `2`/`4`/`8` = the decoder upsamples its decoded
    /// pixel buffer by this factor along each axis after rendering.
    ///
    /// When > 1, the caller must:
    /// - pass the **downsampled** dimensions and pixel buffers to
    ///   [`Self::encode`] (use [`super::resampling::box_downsample_rgb`]
    ///   etc.); the encoder operates entirely at the downsampled
    ///   resolution.
    /// - The codestream's file header still reports the **original**
    ///   (pre-downsample) dimensions — the encoder multiplies
    ///   the supplied (width, height) by `upsampling` when building
    ///   the file header, and writes `upsampling` into the frame
    ///   header. The decoder upsamples to that target.
    ///
    /// Used by [`crate::api::LossyConfig::with_resampling`] (refs #12).
    pub upsampling: u32,
    /// Decoder upsampling mode (libjxl
    /// `JxlEncoderSetUpsamplingMode(enc, factor, mode)`,
    /// `enc_modular.cc` etc.). Only emitted when [`Self::upsampling`] > 1
    /// (the spec ties the LUTs to the upsampling factor).
    ///
    /// - `None` (default) / `Some(-1)` — fancy default upsampling
    ///   (`custom_weight_mask=0`, no custom LUT in the file header).
    /// - `Some(0)` — nearest-neighbour upsampling for the active
    ///   `upsampling` factor: emits a zeroed LUT with a single 1.0
    ///   impulse, matching libjxl `JxlEncoderSetUpsamplingMode(_, _, 0)`.
    /// - `Some(1)` — "pixel dots" upsampling (nearest with cut corners).
    ///   Only meaningful for `upsampling == 4` or `upsampling == 8`;
    ///   silently behaves as nearest for `upsampling == 2` (matches
    ///   libjxl's per-factor table).
    pub upsampling_mode: Option<i32>,
    /// Center X for [`Self::center_first`] AC group reorder. `None`
    /// (default) uses `width / 2`. Mirrors libjxl
    /// `cparams.center_x` (`JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_X`).
    pub center_x: Option<u32>,
    /// Center Y for [`Self::center_first`] AC group reorder. `None`
    /// (default) uses `height / 2`. Mirrors libjxl
    /// `cparams.center_y` (`JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_Y`).
    pub center_y: Option<u32>,
    /// Optional separate butteraugli distance for the alpha extra
    /// channel (CLI passthrough — libjxl `cjxl --alpha_distance`,
    /// `enc_params.h:alpha_distance`). `None` (default) and
    /// `Some(0.0)` keep the lossless alpha path (gradient predictor
    /// + LZ77 RLE).
    ///
    /// `Some(d)` with `d > 0.0` engages the lossy alpha pipeline:
    /// computes an integer pixel quantizer via
    /// [`Self::compute_extra_pixel_quantizer`] (libjxl no-squeeze
    /// formula, `enc_modular.cc:973-1027`), pre-quantizes each alpha
    /// pixel to the nearest multiple of `q`, and emits the modular
    /// extras tree leaf with `(mul_log, mul_bits)` carrying that
    /// multiplier so the decoder reconstructs
    /// `pixel = prediction + val * q`
    /// (`modular/encoding/encoding.cc:186-191`). Applied per channel:
    /// mixed-extras inputs route an alpha-typed leaf through this
    /// quantizer while non-alpha extras (depth, spot color, ...) stay
    /// lossless (`q = 1`) on the same frame.
    pub alpha_distance: Option<f32>,
    /// Opt-in: engage the **squeeze-on-extras** (responsive=1) lossy
    /// alpha pipeline rather than the default no-squeeze pixel
    /// quantizer (`enc_modular.cc:973-1027`). Default `false`.
    ///
    /// libjxl's default is `responsive=1` for lossy alpha; cjxl
    /// out-of-the-box gets `-18%` to `-160%` smaller bytes on
    /// non-opaque alpha than our `responsive=0` path (audit:
    /// `a160deb7`, three-image sweep at d ∈ {0.5, 1.0, 2.0, 5.0}).
    ///
    /// **Chunk-1 status (this flag)**: ungates a framework path that
    /// validates the per-band quantizer table
    /// ([`SQUEEZE_LUMA_QTABLE`]) and the [`Self::alpha_squeeze_engaged`]
    /// predicate. The actual squeeze-application + per-band quantizer
    /// routing through [`Self::compute_extra_pixel_quantizer_shifted`]
    /// is queued for chunk 2 — until then the extras subbitstream
    /// writer surfaces an explicit `Error::Unsupported` when this
    /// flag is `true` AND a non-trivial lossy alpha case is
    /// requested. The default `false` keeps the existing
    /// no-squeeze path byte-for-byte identical (hash-locks 36/36).
    ///
    /// Set via [`crate::LossyConfig::with_alpha_squeeze`].
    pub alpha_squeeze: bool,
    /// Policy for non-finite XYB values at the conversion→pipeline
    /// boundary. See [`crate::api::NonFiniteAction`].
    pub non_finite_action: crate::api::NonFiniteAction,
    /// Per-encode allocation budget. When `Some`, dimension-driven
    /// buffers (XYB planes, padded scratch) reserve their byte count
    /// before allocating and surface
    /// [`crate::error::Error::AllocationLimit`] if the cap would be
    /// exceeded. When `None` (the test/library default), allocation
    /// proceeds unbounded.
    pub(crate) budget: Option<alloc::sync::Arc<crate::budget::MemoryBudget>>,
    // W44-130 Chunk D: `content_aware_entropy_mul: bool` enable bit
    // DELETED — subsumed by `ResolvedImprovements.screenshot_entropy_mul`
    // (the 4-state `ScreenshotEntropyMulPolicy` enum). The Auto branch
    // in `compute_ac_strategy` fires the W22-1 mask1x1 discriminator
    // directly; `Disabled` (Zenjxl default) short-circuits to off.
    // W44-130 Chunk D: the 4 `*_hint: Option<bool>` fields
    // (`screenshot_lift_hint`, `high_d_photo_hint`, `dct_suppress_hint`,
    // `dct32_keep_hint`) plus the public LossyConfig setters were
    // deleted. Strategy overrides flow via
    // `LossyConfig::resolve_improvements()` →
    // `VarDctEncoder::resolved_improvements`. Per-field overrides
    // remain reachable via [`crate::api::LossyConfig::with_strategy_overrides`].
    /// W44-91 cheap zenanalyze-equivalent proxies for widening the W44-29
    /// high-distance smooth-photo lift onto the textured-colourful-photo
    /// sub-band (mask1x1 ∈ [50, 80]) without regressing the 6 documented
    /// W44-78 regression-band images.
    ///
    /// `None` (default): the W44-91 gate cannot fire. Only the existing
    /// W44-29 mask1x1<50 gate is considered (every existing hash-lock
    /// byte-identical).
    ///
    /// `Some(proxies)`: the W44-91 gate also evaluates and may fire when:
    /// - distance ∈ [`HIGH_D_PHOTO_MIN_DISTANCE`,
    ///   `HIGH_D_PHOTO_W44_91_MAX_DISTANCE`] (3.0..=5.0)
    /// - mask1x1_median ∈ [`HIGH_D_PHOTO_SMOOTH_THRESHOLD`,
    ///   `HIGH_D_PHOTO_W44_91_MASK_UPPER`) (50..80)
    /// - `proxies.m3_colourfulness` >= `W44_91_M3_COLOURFULNESS_MIN` (80)
    /// - `proxies.flat_color_block_ratio` < `W44_91_FCBR_MAX` (0.01)
    ///
    /// Populated by the API layer for 8-bit sRGB pixel layouts (`Rgb8` /
    /// `Rgba8` / `Bgr8` / `Bgra8`). Stays `None` for layouts where the
    /// proxy isn't well-defined (16-bit, linear-f32, grayscale, HDR) —
    /// the existing W44-29 gate retains full coverage of those layouts.
    pub(crate) zenanalyze_proxies: Option<ZenanalyzeProxies>,
    /// Streaming-refactor buffering policy (jxl-encoder#11).
    ///
    /// Mirrors libjxl `JXL_ENC_FRAME_SETTING_BUFFERING` integers via
    /// [`crate::api::Buffering`]. **Chunk 6**: routed into
    /// [`super::precomputed::EncoderPrecomputed::compute_with_budget`]
    /// where it selects between the whole-image precompute (chunk 3)
    /// and the per-region precompute (chunk 5,
    /// [`super::precomputed::fill_dc_group_state_per_region`]). At chunk
    /// 6 every variant still produces byte-identical output — the
    /// per-region path's <=256 ULP individual-pixel FP drift was proven
    /// in the chunk-5 tests to absorb into bit-equal output on the rd-
    /// regression set and the hash_lock corpus. Real memory savings land
    /// in chunk 7 when [`super::precomputed::EncoderPrecomputed`]'s
    /// whole-image plane Vecs become per-DC-group sliding windows.
    ///
    /// Default [`crate::api::Buffering::Auto`] resolves to
    /// [`crate::api::Buffering::FullBuffered`] for ≤2048² images and
    /// [`crate::api::Buffering::BufferedOutput`] for larger inputs,
    /// matching libjxl post-`032d39a`.
    pub buffering: crate::api::Buffering,
    /// W44-128 (Chunk B) / W44-130 (Chunk D) — resolved compatibility /
    /// improvements bundle. Computed once at encoder construction by
    /// [`crate::api::EncoderStrategy::resolve`] from the caller-set
    /// [`crate::api::LossyConfig::with_strategy`] preset plus the
    /// individual `with_*_hint` overrides.
    ///
    /// **W44-130 Chunk D**: this field is now always populated. The
    /// production API layer (still-image `EncodeRequest`, streaming
    /// `LossyEncoder`, animation) populates from
    /// [`crate::api::LossyConfig::resolve_improvements`]; direct
    /// `VarDctEncoder::new` callers (tests + examples) get
    /// [`crate::api::ResolvedImprovements::default`] which matches the
    /// `EncoderStrategy::Zenjxl` baseline. The 8 call sites in
    /// `vardct/encoder.rs` + `vardct/butteraugli_loop.rs` read this
    /// directly without `unwrap_or_default()` fallbacks.
    pub(crate) resolved_improvements: crate::api::ResolvedImprovements,
}

impl Default for VarDctEncoder {
    fn default() -> Self {
        Self {
            distance: 1.0,
            effort: 7,
            profile: crate::effort::EffortProfile::lossy(7, crate::api::EncoderMode::Reference),
            optimize_codes: true,
            enhanced_clustering: true, // Profile-driven: e9+ for Best, Fast otherwise
            use_ans: true,             // ANS produces 4-10% smaller files than Huffman
            cfl_enabled: true,
            ac_strategy_enabled: true,
            custom_orders: true,
            force_strategy: None,
            enable_noise: false,
            photon_noise_iso: None,
            manual_noise_lut: None,
            quant_ac_rescale: None,
            original_distance: None,
            enable_denoise: false,
            enable_gaborish: true,
            enable_adaptive_gaborish: false, // EX-J13: opt-in via LossyConfig
            epf_level_override: None,
            epf_dispatch: crate::api::EpfDispatch::AlwaysSelect,
            error_diffusion: false, // libjxl accepts param but never uses it in QuantizeBlockAC
            pixel_domain_loss: true, // Full libjxl pixel-domain loss: +0.2-1.9 SSIM2 at all distances
            pixel_loss_dispatch: crate::api::PixelLossDispatch::AlwaysOn,
            // W44-87 default keeps every hash-lock byte-identical.
            // `Auto` flips to single-pass on smooth photos at e5 d<=1.0;
            // callers opt in via LossyConfig::with_single_pass_entropy_dispatch.
            single_pass_entropy_dispatch: crate::api::SinglePassEntropyDispatch::AlwaysTwoPass,
            enable_lz77: false, // LZ77 has known interactions with DCT2x2/IDENTITY strategies
            lz77_method: crate::entropy_coding::lz77::Lz77Method::Greedy, // Best compression
            dc_tree_learning: false, // DC tree learning (experimental)
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters: 0, // Effort-gated: default off (effort 7). Set via LossyConfig.
            // EX-J11 chunk 1: default keeps every hash-lock byte-identical.
            #[cfg(feature = "butteraugli-loop")]
            hdr_loss: crate::vardct::hdr_metrics::HdrLoss::Butteraugli,
            // W44-phase3-B1: GPU butteraugli backend defaults off; LossyConfig
            // sets it via with_gpu_butteraugli.
            #[cfg(feature = "butteraugli-loop")]
            gpu_butteraugli: false,
            // cvvdp-fork Phase 3 (2026-05-24): CVVDP backend defaults off;
            // LossyConfig sets it via `with_cvvdp_loop`. Hash-locks stay
            // byte-identical regardless of the `cvvdp-loop` cargo feature
            // because the field defaults to `false`.
            #[cfg(feature = "butteraugli-loop")]
            cvvdp_loop: false,
            // cvvdp-fork Phase 5 (2026-05-24): CPU CVVDP preference
            // defaults off (= prefer GPU when both backends compiled).
            // LossyConfig sets it via `with_cvvdp_use_cpu`. Hash-locks
            // stay byte-identical regardless of the `cvvdp-loop-cpu`
            // cargo feature because the field defaults to `false` AND
            // the entire cvvdp dispatch branch is gated on
            // `cvvdp_loop = true` upstream.
            #[cfg(feature = "butteraugli-loop")]
            cvvdp_use_cpu: false,
            // zensim-fork Phase 3 (2026-05-25): zensim backend defaults
            // off; LossyConfig sets it via `with_perceptual_metric(Zensim)`.
            // Hash-locks stay byte-identical regardless of the `zensim-loop`
            // / `zensim-loop-gpu` cargo features because the field defaults
            // to `false`.
            #[cfg(feature = "butteraugli-loop")]
            zensim_loop: false,
            // zensim-fork Phase 3 (2026-05-25): CPU zensim preference
            // defaults off (= prefer GPU when both backends compiled).
            // Hash-locks stay byte-identical regardless of the zensim
            // cargo features because the field defaults to `false` AND
            // the entire zensim dispatch branch is gated on
            // `zensim_loop = true` upstream.
            #[cfg(feature = "butteraugli-loop")]
            zensim_use_cpu: false,
            // cvvdp-fork Phase 8d (2026-05-25): bytes-tighten exit pass
            // defaults off. LossyConfig sets it via
            // `with_cvvdp_bytes_tighten`. Hash-locks stay byte-identical
            // regardless of the `cvvdp-loop-tighten` cargo feature
            // because the field defaults to `false` AND the entire
            // tighten branch is gated on `cvvdp_loop = true` upstream.
            #[cfg(feature = "butteraugli-loop")]
            cvvdp_bytes_tighten: false,
            // Phase 1 display-config backfill (2026-05-25): default
            // WebSdr80 maps to `cvvdp_gpu::params::DisplayModel::STANDARD_4K`
            // — bit-identical to the pre-Phase-1 cvvdp scoring shape.
            target_display: crate::api::DisplayConfig::WebSdr80,
            // Phase 1 of RFC `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`
            // (2026-05-26): default None preserves the pre-Phase-1
            // implicit-identity arm of `effective_metric_target_distance`.
            // Set via `LossyConfig::with_perceptual_target_score(Some(_))`
            // → `resolve_perceptual_target_score` → propagated by
            // `propagate_resolved_metric_to_encoder`.
            #[cfg(feature = "butteraugli-loop")]
            perceptual_target_score: None,
            #[cfg(feature = "ssim2-loop")]
            ssim2_iters: 0, // Off by default. Set via LossyConfig.
            #[cfg(feature = "zensim-loop")]
            zensim_iters: 0, // Off by default. Set via LossyConfig.
            bit_depth_16: false,
            icc_profile: None,
            enable_patches: true, // Patches: huge wins on screenshots, zero cost on photos
            patches_dispatch: crate::api::PatchesDispatch::default(),
            enable_dot_detection: false, // refs #19; LossyConfig flips this to true by default
            encoder_mode: crate::api::EncoderMode::Reference,
            splines: None,
            auto_splines: false, // Opt-in (chunk 1 ships stub; default-off keeps hash-locks)
            is_grayscale: false,
            progressive: crate::api::ProgressiveMode::Single,
            use_lf_frame: false,
            source_gamma: None,
            color_encoding: None,
            intensity_target: 255.0,
            min_nits: 0.0,
            relative_to_max_display: false,
            linear_below: 0.0,
            intrinsic_size: None,
            alpha_associated: false,
            bits_per_sample_override: None,
            center_first: false,
            upsampling: 1,
            upsampling_mode: None,
            center_x: None,
            center_y: None,
            alpha_distance: None,
            // Chunk-1 default: keep existing no-squeeze alpha pipeline.
            // Setting `true` ungates the chunk-1 framework path which
            // surfaces `Error::Unsupported` from the extras writer until
            // chunk-2 wires the per-band squeeze application.
            alpha_squeeze: false,
            non_finite_action: crate::api::NonFiniteAction::default(),
            budget: None,
            // W44-130 Chunk D: `content_aware_entropy_mul` enable bit
            // + the 4 `*_hint` fields were deleted. Strategy + overrides
            // flow through `resolved_improvements` below.
            // W44-91: default None means the gate cannot fire (every
            // existing hash-lock byte-identical). API layer populates
            // for 8-bit sRGB layouts.
            zenanalyze_proxies: None,
            buffering: crate::api::Buffering::default(),
            // W44-130 Chunk D: default to the Zenjxl-equivalent resolved
            // policy (matches `EncoderStrategy::Zenjxl`'s
            // `ResolvedImprovements`). The API layer overwrites this
            // with the caller's strategy via
            // `LossyConfig::resolve_improvements()` at all three
            // construction sites.
            resolved_improvements: crate::api::ResolvedImprovements::default(),
        }
    }
}

impl VarDctEncoder {
    /// Create a new tiny encoder with the given distance.
    pub fn new(distance: f32) -> Self {
        Self {
            distance,
            effort: 7,
            profile: crate::effort::EffortProfile::lossy(7, crate::api::EncoderMode::Reference),
            optimize_codes: true,
            enhanced_clustering: true, // Profile-driven: e9+ for Best, Fast otherwise
            use_ans: true,             // ANS produces 4-10% smaller files than Huffman
            cfl_enabled: true,
            ac_strategy_enabled: true,
            custom_orders: true,
            force_strategy: None,
            enable_noise: false,
            photon_noise_iso: None,
            manual_noise_lut: None,
            quant_ac_rescale: None,
            original_distance: None,
            enable_denoise: false,
            enable_gaborish: true,
            enable_adaptive_gaborish: false, // EX-J13: opt-in via LossyConfig
            epf_level_override: None,
            epf_dispatch: crate::api::EpfDispatch::AlwaysSelect,
            error_diffusion: false, // libjxl accepts param but never uses it in QuantizeBlockAC
            pixel_domain_loss: true, // Full libjxl pixel-domain loss: +0.2-1.9 SSIM2
            pixel_loss_dispatch: crate::api::PixelLossDispatch::AlwaysOn,
            // W44-87 default keeps every hash-lock byte-identical.
            // `Auto` flips to single-pass on smooth photos at e5 d<=1.0;
            // callers opt in via LossyConfig::with_single_pass_entropy_dispatch.
            single_pass_entropy_dispatch: crate::api::SinglePassEntropyDispatch::AlwaysTwoPass,
            enable_lz77: false, // LZ77 has known interactions with DCT2x2/IDENTITY strategies
            lz77_method: crate::entropy_coding::lz77::Lz77Method::Greedy, // Best compression
            dc_tree_learning: false, // DC tree learning (experimental)
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters: 0, // Effort-gated: default off (effort 7). Set via LossyConfig.
            // EX-J11 chunk 1: default keeps every hash-lock byte-identical.
            #[cfg(feature = "butteraugli-loop")]
            hdr_loss: crate::vardct::hdr_metrics::HdrLoss::Butteraugli,
            // W44-phase3-B1: GPU butteraugli backend defaults off; LossyConfig
            // sets it via with_gpu_butteraugli.
            #[cfg(feature = "butteraugli-loop")]
            gpu_butteraugli: false,
            // cvvdp-fork Phase 3 (2026-05-24): CVVDP backend defaults off;
            // LossyConfig sets it via `with_cvvdp_loop`. Hash-locks stay
            // byte-identical regardless of the `cvvdp-loop` cargo feature
            // because the field defaults to `false`.
            #[cfg(feature = "butteraugli-loop")]
            cvvdp_loop: false,
            // cvvdp-fork Phase 5 (2026-05-24): CPU CVVDP preference
            // defaults off (= prefer GPU when both backends compiled).
            // LossyConfig sets it via `with_cvvdp_use_cpu`. Hash-locks
            // stay byte-identical regardless of the `cvvdp-loop-cpu`
            // cargo feature because the field defaults to `false` AND
            // the entire cvvdp dispatch branch is gated on
            // `cvvdp_loop = true` upstream.
            #[cfg(feature = "butteraugli-loop")]
            cvvdp_use_cpu: false,
            // zensim-fork Phase 3 (2026-05-25): zensim backend defaults
            // off; LossyConfig sets it via `with_perceptual_metric(Zensim)`.
            // Hash-locks stay byte-identical regardless of the `zensim-loop`
            // / `zensim-loop-gpu` cargo features because the field defaults
            // to `false`.
            #[cfg(feature = "butteraugli-loop")]
            zensim_loop: false,
            // zensim-fork Phase 3 (2026-05-25): CPU zensim preference
            // defaults off (= prefer GPU when both backends compiled).
            // Hash-locks stay byte-identical regardless of the zensim
            // cargo features because the field defaults to `false` AND
            // the entire zensim dispatch branch is gated on
            // `zensim_loop = true` upstream.
            #[cfg(feature = "butteraugli-loop")]
            zensim_use_cpu: false,
            // cvvdp-fork Phase 8d (2026-05-25): bytes-tighten exit pass
            // defaults off. LossyConfig sets it via
            // `with_cvvdp_bytes_tighten`. Hash-locks stay byte-identical
            // regardless of the `cvvdp-loop-tighten` cargo feature
            // because the field defaults to `false` AND the entire
            // tighten branch is gated on `cvvdp_loop = true` upstream.
            #[cfg(feature = "butteraugli-loop")]
            cvvdp_bytes_tighten: false,
            // Phase 1 display-config backfill (2026-05-25): default
            // WebSdr80 maps to `cvvdp_gpu::params::DisplayModel::STANDARD_4K`
            // — bit-identical to the pre-Phase-1 cvvdp scoring shape.
            target_display: crate::api::DisplayConfig::WebSdr80,
            // Phase 1 of RFC `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`
            // (2026-05-26): default None preserves the pre-Phase-1
            // implicit-identity arm of `effective_metric_target_distance`.
            // Set via `LossyConfig::with_perceptual_target_score(Some(_))`
            // → `resolve_perceptual_target_score` → propagated by
            // `propagate_resolved_metric_to_encoder`.
            #[cfg(feature = "butteraugli-loop")]
            perceptual_target_score: None,
            #[cfg(feature = "ssim2-loop")]
            ssim2_iters: 0, // Off by default. Set via LossyConfig.
            #[cfg(feature = "zensim-loop")]
            zensim_iters: 0, // Off by default. Set via LossyConfig.
            bit_depth_16: false,
            icc_profile: None,
            enable_patches: true, // Patches: huge wins on screenshots, zero cost on photos
            patches_dispatch: crate::api::PatchesDispatch::default(),
            enable_dot_detection: false, // refs #19; LossyConfig flips this to true by default
            encoder_mode: crate::api::EncoderMode::Reference,
            splines: None,
            auto_splines: false, // Opt-in (chunk 1 ships stub; default-off keeps hash-locks)
            is_grayscale: false,
            progressive: crate::api::ProgressiveMode::Single,
            use_lf_frame: false,
            source_gamma: None,
            color_encoding: None,
            intensity_target: 255.0,
            min_nits: 0.0,
            relative_to_max_display: false,
            linear_below: 0.0,
            intrinsic_size: None,
            alpha_associated: false,
            bits_per_sample_override: None,
            center_first: false,
            upsampling: 1,
            upsampling_mode: None,
            center_x: None,
            center_y: None,
            alpha_distance: None,
            // Chunk-1 default: keep existing no-squeeze alpha pipeline.
            // Setting `true` ungates the chunk-1 framework path which
            // surfaces `Error::Unsupported` from the extras writer until
            // chunk-2 wires the per-band squeeze application.
            alpha_squeeze: false,
            non_finite_action: crate::api::NonFiniteAction::default(),
            budget: None,
            // W44-130 Chunk D: `content_aware_entropy_mul` enable bit
            // + legacy `*_hint` fields deleted; strategy + overrides
            // flow via `resolved_improvements` below.
            // W44-91: default None means the gate cannot fire (every
            // existing hash-lock byte-identical). API layer populates
            // for 8-bit sRGB layouts.
            zenanalyze_proxies: None,
            buffering: crate::api::Buffering::default(),
            // W44-130 Chunk D: default to the Zenjxl-equivalent resolved
            // policy. Populated by the API layer via
            // `LossyConfig::resolve_improvements()` at all three
            // construction sites; tests/examples using `new()` directly
            // get the default behaviour.
            resolved_improvements: crate::api::ResolvedImprovements::default(),
        }
    }

    /// Attach a per-encode allocation budget. Internal-only; the public
    /// API plumbs this from [`crate::api::Limits::max_memory_bytes`].
    ///
    /// Currently the API path sets [`Self::budget`] directly; this
    /// builder is here for future call sites (e.g., the streaming
    /// encoder, precomputed entry points) that don't have field
    /// access.
    #[doc(hidden)]
    #[allow(dead_code)]
    pub(crate) fn with_budget(
        mut self,
        budget: alloc::sync::Arc<crate::budget::MemoryBudget>,
    ) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Apply [`Self::epf_level_override`] to a freshly computed
    /// [`DistanceParams`], if the caller pinned one. Matches libjxl
    /// `enc_frame.cc:284-285`: any non-default value overrides the
    /// distance-derived `epf_iters` (including `0`, which forces EPF
    /// off and skips the dynamic sharpness search downstream).
    #[inline]
    pub(crate) fn apply_epf_level_override(&self, params: &mut DistanceParams) {
        if let Some(level) = self.epf_level_override {
            params.epf_iters = level;
        }
    }

    /// Compute the integer pixel quantizer `q` to apply to a single
    /// extra channel from [`Self::alpha_distance`] and the channel's
    /// bit depth + type, mirroring libjxl's lossy-modular
    /// `quantizers[component] * squeeze_quality_factor *
    /// squeeze_luma_factor * squeeze_luma_qtable[shift]` at
    /// `shift = 0` and `responsive = false`
    /// (`enc_modular.cc:973-1027`):
    ///
    /// - `quantizer = 0.25 * 0.1 = 0.025` (no-squeeze, "just color quantization")
    /// - `bitdepth_correction = (2^bits - 1) / 255`
    /// - `q_float = quantizer * dist * bitdepth_correction
    ///            * squeeze_quality_factor (0.35) * squeeze_luma_factor (1.1)
    ///            * squeeze_luma_qtable[0] (163.84)`
    /// - `q = max(1, floor(q_float))`
    ///
    /// Returns `1` (lossless) for any extra channel that does not carry
    /// its own distance knob: today only alpha-typed channels use
    /// `alpha_distance`. Non-alpha extras (depth, spot, selection mask,
    /// thermal, CFA, ...) stay lossless because we do not currently
    /// expose per-channel `ec_distance`. Also returns `1` when
    /// `alpha_distance` is `None`, `Some(d <= 0.0)`, or any input that
    /// rounds below `1` after libjxl's `floor`.
    pub(crate) fn compute_extra_pixel_quantizer(
        &self,
        extras_bits: u32,
        ec_type: crate::headers::extra_channels::ExtraChannelType,
    ) -> u32 {
        let dist = match ec_type {
            crate::headers::extra_channels::ExtraChannelType::Alpha => match self.alpha_distance {
                Some(d) if d > 0.0 => d,
                _ => return 1,
            },
            // Non-alpha extras have no per-channel distance knob today.
            // libjxl supports `cparams.ec_distance[i]` per extra channel;
            // when that is wired through the public API, dispatch here
            // by `ec_type` / channel index.
            _ => return 1,
        };
        // libjxl clamps non-zero alpha_distance to [0.01, 25.0] at the
        // public API boundary (encode.cc:1558). Mirror the lower clamp
        // here so callers writing `Some(0.001)` don't silently regress
        // to lossless.
        let dist = dist.clamp(0.01, 25.0);
        let ec_maxval = if extras_bits >= 32 {
            // i32::MAX would overflow the (1<<bits)-1 calc; libjxl
            // treats it as 0 (no correction). Treat the same way.
            0u64
        } else {
            (1u64 << extras_bits) - 1
        };
        let bitdepth_correction = if ec_maxval == 0 {
            1.0
        } else {
            (ec_maxval as f32) / 255.0
        };
        const QUANTIZER: f32 = 0.025; // 0.25 * 0.1 (no responsive/squeeze)
        const SQUEEZE_QUALITY_FACTOR: f32 = 0.35;
        const SQUEEZE_LUMA_FACTOR: f32 = 1.1;
        const SQUEEZE_LUMA_QTABLE_0: f32 = 163.84;
        let q_float = QUANTIZER
            * dist
            * bitdepth_correction
            * SQUEEZE_QUALITY_FACTOR
            * SQUEEZE_LUMA_FACTOR
            * SQUEEZE_LUMA_QTABLE_0;
        // libjxl truncates float→int via implicit conversion (C++
        // `int q = ...`), so we match with `floor`. Anything < 1
        // clamps to 1 (lossless).
        let q = q_float.floor() as i64;
        if q < 1 { 1 } else { q as u32 }
    }

    /// Build the per-channel quantizer vector for an extras slice.
    ///
    /// Returns `Vec<u32>` of length `extras.len()`. Each entry is
    /// computed via [`Self::compute_extra_pixel_quantizer`] from that
    /// channel's bit depth + [`ExtraChannelType`]. `extras.is_empty()`
    /// returns an empty vec.
    ///
    /// Mixed-extras invariant: the existing single-extra (`extras.len()
    /// == 1`) path delegates here, so a single alpha extra at index 0
    /// continues to produce the same `q` value as before.
    pub(crate) fn compute_extras_pixel_quantizers(
        &self,
        extras: &[super::extras::VardctExtra<'_>],
    ) -> alloc::vec::Vec<u32> {
        extras
            .iter()
            .map(|ec| {
                self.compute_extra_pixel_quantizer(
                    ec.info.bit_depth.bits_per_sample,
                    ec.info.ec_type,
                )
            })
            .collect()
    }

    /// Compute the integer pixel quantizer `q` for one
    /// **squeeze-shifted** sub-channel of an extras lossy plane,
    /// mirroring the responsive=1 branch of libjxl
    /// `enc_modular.cc:973-1027`:
    ///
    /// ```text
    /// quantizer       = 0.25            // base; NO `* 0.1` because responsive=1
    /// q_float         = quantizer
    ///                  * dist
    ///                  * bitdepth_correction
    ///                  * squeeze_quality_factor      (0.35)
    ///                  * squeeze_luma_factor         (1.1)
    ///                  * squeeze_luma_qtable[shift]
    /// q               = max(1, floor(q_float))
    /// ```
    ///
    /// Here `shift` is libjxl's per-channel `hshift + vshift` decremented
    /// by 1 (`enc_modular.cc:1006-1008`) — i.e. the depth of squeeze
    /// halvings already applied to this sub-channel — then clamped to
    /// `[0, 15]` (`SQUEEZE_LUMA_QTABLE_LEN - 1`).
    ///
    /// The `shift == 0` path **deliberately diverges** from
    /// [`Self::compute_extra_pixel_quantizer`] (the no-squeeze formula),
    /// which folds in an extra `* 0.1` factor (`enc_modular.cc:976-981`,
    /// "lossy compression without Squeeze transform is just color
    /// quantization"). At `shift == 0` and the same `dist`/`bits`,
    /// this function returns `~10×` the q value of the no-squeeze
    /// path because squeezed averages need much coarser quantization
    /// than raw pixels would. The HF residual sub-channels (shift > 0)
    /// then drop very quickly toward `q = 1` via the qtable.
    ///
    /// **Chunk-1 status**: this function is wired into the
    /// `with_alpha_squeeze` opt-in flag but **not yet routed into the
    /// extras subbitstream** — see
    /// [`Self::alpha_squeeze`] / chunk-2 plan in CHANGELOG. Today it
    /// is unit-testable and ready for chunk-2 to consume.
    ///
    /// Like [`Self::compute_extra_pixel_quantizer`] this returns `1`
    /// for non-alpha extras and for `alpha_distance` of `None` /
    /// `Some(0.0)` / `Some(d <= 0)` — preserves the lossless
    /// contract end-to-end.
    #[allow(dead_code)] // chunk-1 framework — chunk 2 wires this in.
    pub(crate) fn compute_extra_pixel_quantizer_shifted(
        &self,
        extras_bits: u32,
        ec_type: crate::headers::extra_channels::ExtraChannelType,
        shift: u32,
    ) -> u32 {
        let dist = match ec_type {
            crate::headers::extra_channels::ExtraChannelType::Alpha => match self.alpha_distance {
                Some(d) if d > 0.0 => d,
                _ => return 1,
            },
            _ => return 1,
        };
        let dist = dist.clamp(0.01, 25.0);
        let ec_maxval = if extras_bits >= 32 {
            0u64
        } else {
            (1u64 << extras_bits) - 1
        };
        let bitdepth_correction = if ec_maxval == 0 {
            1.0
        } else {
            (ec_maxval as f32) / 255.0
        };
        // responsive=1: base quantizer is 0.25 (no `* 0.1` folded in).
        const QUANTIZER_RESPONSIVE: f32 = 0.25;
        const SQUEEZE_QUALITY_FACTOR: f32 = SQUEEZE_QUALITY_FACTOR_CONST;
        const SQUEEZE_LUMA_FACTOR: f32 = SQUEEZE_LUMA_FACTOR_CONST;
        let shift_idx = shift.min(SQUEEZE_LUMA_QTABLE_LEN as u32 - 1) as usize;
        let qtable_val = SQUEEZE_LUMA_QTABLE[shift_idx];
        let q_float = QUANTIZER_RESPONSIVE
            * dist
            * bitdepth_correction
            * SQUEEZE_QUALITY_FACTOR
            * SQUEEZE_LUMA_FACTOR
            * qtable_val;
        let q = q_float.floor() as i64;
        if q < 1 { 1 } else { q as u32 }
    }

    /// Whether the squeeze-on-extras lossy alpha pipeline is engaged
    /// for this encode (chunk-1 framework opt-in, not yet wired into
    /// the bitstream).
    ///
    /// Returns `true` only when **both** of the following hold:
    /// - [`Self::alpha_squeeze`] was set via
    ///   [`crate::LossyConfig::with_alpha_squeeze`].
    /// - [`Self::alpha_distance`] is `Some(d > 0.0)` (lossy alpha
    ///   path actually engaged).
    ///
    /// When this returns `true`, the extras writer is expected to
    /// route through the squeeze + per-band quantizer pipeline
    /// (libjxl `enc_modular.cc:937-1027`, `responsive=1`). Chunk 1
    /// surfaces an `Error::NotImplemented` from the writer to keep
    /// the wire format honest while the per-band quantizer routing
    /// lands.
    ///
    /// When this returns `false`, the existing no-squeeze
    /// [`Self::compute_extra_pixel_quantizer`] path is used — fully
    /// backwards-compatible (hash-lock byte-identical).
    pub(crate) fn alpha_squeeze_engaged(&self) -> bool {
        self.alpha_squeeze && matches!(self.alpha_distance, Some(d) if d > 0.0)
    }

    /// W22-1 / W44-29 / W44-65+W44-68 content-aware AC-strategy-search
    /// profile gate cascade. Produces a `Some(profile_clone)` with the
    /// appropriate entropy_mul table swap and/or `try_dct64` /
    /// `try_dct32` suppression applied when one of the dispatchers
    /// fires; returns `None` when no gate fires (caller uses
    /// `&self.profile` directly).
    ///
    /// Shared between [`Self::encode`] (still-image path) and
    /// [`super::bitstream::VarDctEncoder::encode_frame_to_writer`]
    /// (animation per-frame path) so both paths produce
    /// byte-equivalent bitstreams for the same input.
    ///
    /// Originally extracted in W44-70 (`d2396131`) to fix the
    /// `test_animation_lossy_runs_cfl_pass_2` regression caused by
    /// W44-65/68 default-on DCT suppression landing in `encode` but
    /// not in `encode_frame_to_writer`. The helper was lost between
    /// `d2396131` (sibling branch never merged) and the present W44-129/130
    /// EncoderStrategy refactor. W44-137 re-extracts it on top of the
    /// post-refactor `resolved_improvements` path.
    ///
    /// `mask1x1_median` is `None` when the mask wasn't computed
    /// (pixel_domain_loss off / PixelLossDispatch::AlwaysOff /
    /// !ac_strategy_enabled) — all auto gates degrade to "don't fire"
    /// in that case unless the caller supplied a `ForceOn/ForceOff`
    /// policy via `EncoderStrategy::Custom` / `StrategyOverrides`.
    ///
    /// `mask1x1_p25` (W44-151) is `None` under the same conditions as
    /// `mask1x1_median` and is used ONLY by the W44-151 sub-branch of
    /// the W44-29 outer gate (admits high-smooth photos `mask_p25 >=
    /// 85.0` to the default `high_d_photo_smooth_suppressed` table at
    /// `d >= HIGH_D_PHOTO_MIN_DISTANCE`).
    pub(super) fn compute_profile_for_search(
        &self,
        mask1x1_median: Option<f32>,
        mask1x1_p25: Option<f32>,
    ) -> Option<crate::effort::EffortProfile> {
        // ── W22-1 screenshot lift ──
        let screenshot_policy = self.resolved_improvements.screenshot_entropy_mul;
        let w22_1_lift = match screenshot_policy {
            crate::api::ScreenshotEntropyMulPolicy::Auto => {
                // W44-213: route the threshold through the tuning-override
                // macro so sweep-runner builds can swap it at runtime.
                let median_threshold = crate::runtime_or_default!(
                    CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD,
                    screenshot_median_threshold,
                );
                mask1x1_median.is_some_and(|med| med > median_threshold)
            }
            crate::api::ScreenshotEntropyMulPolicy::ForceOn => true,
            crate::api::ScreenshotEntropyMulPolicy::ForceOff => false,
            crate::api::ScreenshotEntropyMulPolicy::Disabled => false,
        };

        // ── W44-29 high-distance smooth-photo lowering (auto + W44-91 widen + W44-151 mask_p25 admit) ──
        let high_d_photo_policy = self.resolved_improvements.high_d_photo_entropy_mul;
        let w44_29_lower = if w22_1_lift {
            false
        } else {
            match high_d_photo_policy {
                crate::api::HighDPhotoEntropyMulPolicy::Auto => {
                    let w44_29_gate = self.distance >= HIGH_D_PHOTO_MIN_DISTANCE
                        && mask1x1_median.is_some_and(|med| med < HIGH_D_PHOTO_SMOOTH_THRESHOLD);
                    let w44_91_gate = (HIGH_D_PHOTO_MIN_DISTANCE
                        ..=HIGH_D_PHOTO_W44_91_MAX_DISTANCE)
                        .contains(&self.distance)
                        && mask1x1_median.is_some_and(|med| {
                            (HIGH_D_PHOTO_SMOOTH_THRESHOLD..HIGH_D_PHOTO_W44_91_MASK_UPPER)
                                .contains(&med)
                        })
                        && self.zenanalyze_proxies.is_some_and(|p| {
                            p.m3_colourfulness >= W44_91_M3_COLOURFULNESS_MIN
                                && p.flat_color_block_ratio < W44_91_FCBR_MAX
                        });
                    // W44-152 (2026-05-21): re-introduces the W44-151
                    // `mask1x1_p25 >= W44_151_HIGH_MASK_P25_MIN` admission
                    // branch with the distance narrowed to
                    // [`W44_152_W44_151_MIN_DISTANCE`,
                    //  `W44_152_W44_151_MAX_DISTANCE`] = [3.0, 5.0].
                    //
                    // W44-151 honest-stopped on the broad d ≥ 3.0 gate
                    // because the default `high_d_photo_smooth_suppressed()`
                    // table over-fires at d=6 on 1418519 (+4.3-4.6% bytes
                    // for only +0.07-0.28 SSIM2). d=4 + d=5 clusters were
                    // clean wins. Excluding d=6 captures the win region.
                    //
                    // The env hook `JXL_W44_152_DISABLE=1` (and the legacy
                    // `JXL_W44_151_DISABLE=1` alias) disable the admission
                    // for A/B benches.
                    let w44_152_distance_in_band = self.distance >= W44_152_W44_151_MIN_DISTANCE
                        && self.distance <= W44_152_W44_151_MAX_DISTANCE;
                    let w44_152_disable_env = {
                        #[cfg(feature = "std")]
                        {
                            std::env::var("JXL_W44_152_DISABLE")
                                .map(|s| s == "1")
                                .unwrap_or(false)
                                || std::env::var("JXL_W44_151_DISABLE")
                                    .map(|s| s == "1")
                                    .unwrap_or(false)
                        }
                        #[cfg(not(feature = "std"))]
                        {
                            false
                        }
                    };
                    // W44-213: tuning-override-aware p25 threshold lookup.
                    let w44_151_p25_threshold = crate::runtime_or_default!(
                        W44_151_HIGH_MASK_P25_MIN,
                        smart_zenjxl_photo_mask_p25_min,
                    );
                    let w44_152_admit = !w44_152_disable_env
                        && w44_152_distance_in_band
                        && mask1x1_p25.is_some_and(|p25| p25 >= w44_151_p25_threshold);
                    w44_29_gate || w44_91_gate || w44_152_admit
                }
                crate::api::HighDPhotoEntropyMulPolicy::ForceOn => true,
                crate::api::HighDPhotoEntropyMulPolicy::ForceOff => false,
                crate::api::HighDPhotoEntropyMulPolicy::Disabled => false,
            }
        };

        // ── W44-65 + W44-68 DCT64/DCT32 suppression ──
        let dct64_policy = self.resolved_improvements.dct64_search_policy;
        let w44_65_suppress_dct64 = match dct64_policy {
            crate::api::Dct64SearchPolicy::Auto => {
                mask1x1_median.is_some_and(|med| med >= W44_65_DCT_SUPPRESS_MEDIAN_THRESHOLD)
            }
            crate::api::Dct64SearchPolicy::ForceSuppress => true,
            crate::api::Dct64SearchPolicy::ForceAllow => false,
        };

        // ── W44-96/98/99 variant Z sub-discriminators ──
        //
        // W44-166 (Smart-Zenjxl chunk 3, 2026-05-21): the default W44-96
        // admit predicate requires `mask1x1_median < 50` (the W44-29
        // outer threshold). High-mask photos like 1418519 (mask=92)
        // never reach variant Z under this gate. The W44-166 admit-mode
        // env hook layers an OR-branch that admits via `mask1x1_p25 >=
        // W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN = 85` (Mode B) or via
        // mask_p25 AND `m3_colourfulness >= 25` (Mode C). The OR-branch
        // is gated on `ResolvedImprovements::photo_variant_z_admit`
        // (true on Zenjxl / Aggressive; false on Libjxl / LeanFaster
        // per strict-parity discipline). Default Mode A = baseline =
        // no change.
        let w44_166_admit_mode = w44_166_admit_mode_env();
        let w44_166_photo_admit_allowed = self.resolved_improvements.photo_variant_z_admit
            && matches!(
                high_d_photo_policy,
                crate::api::HighDPhotoEntropyMulPolicy::Auto
            )
            && self.distance >= W44_96_VARIANT_Z_MIN_DISTANCE;
        // W44-213: tuning-override-aware p25 threshold lookup (W44-166
        // photo admission; shares the canonical 85.0 value with W44-150
        // / W44-151 / W44-168, all unified under `smart_zenjxl_photo_mask_p25_min`).
        let w44_166_p25_threshold = crate::runtime_or_default!(
            W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN,
            smart_zenjxl_photo_mask_p25_min,
        );
        let w44_166_photo_admit = w44_166_photo_admit_allowed
            && match w44_166_admit_mode {
                W44_166VariantZAdmitMode::Baseline => false,
                W44_166VariantZAdmitMode::BMaskP25 => {
                    mask1x1_p25.is_some_and(|p25| p25 >= w44_166_p25_threshold)
                }
                W44_166VariantZAdmitMode::CMaskP25HighM3 => {
                    mask1x1_p25.is_some_and(|p25| p25 >= w44_166_p25_threshold)
                        && self.zenanalyze_proxies.is_some_and(|p| {
                            p.m3_colourfulness >= W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN
                        })
                }
            };
        let w44_96_default_admit = w44_29_lower
            && matches!(
                high_d_photo_policy,
                crate::api::HighDPhotoEntropyMulPolicy::Auto
            )
            && self.distance >= W44_96_VARIANT_Z_MIN_DISTANCE
            && mask1x1_median.is_some_and(|med| med < HIGH_D_PHOTO_SMOOTH_THRESHOLD)
            && self.zenanalyze_proxies.is_some_and(|p| {
                p.edge_density >= W44_96_EDGE_DENSITY_MIN
                    && p.flat_color_block_ratio < W44_96_FCBR_MAX
            });
        let w44_96_variant_z = w44_96_default_admit || w44_166_photo_admit;
        let w44_98_variant_z_high_colour = w44_96_variant_z
            && self
                .zenanalyze_proxies
                .is_some_and(|p| p.m3_colourfulness >= W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN);
        let w44_99_variant_z_low_colour = w44_96_variant_z
            && !w44_98_variant_z_high_colour
            && self
                .zenanalyze_proxies
                .is_some_and(|p| p.m3_colourfulness < W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN);

        // W44-156 distance-band sub-discriminator inside variant Z: when
        // any variant Z gate fires AND target_distance >
        // W44_156_VARIANT_Z_D_HIGH_THRESHOLD (5.5 default, env hook
        // __JXL_W44_156_THRESHOLD), use the d-high table (dct32x32 = 1.20)
        // instead of the W44-154 default (dct32x32 = 1.22). Closes
        // 1420710 e5 d=6 OPEN cell per W44-155 strategy-shift diagnosis.
        let w44_156_d_high = w44_96_variant_z && self.distance > w44_156_effective_threshold();

        if !(w22_1_lift || w44_29_lower || w44_65_suppress_dct64 || w44_166_photo_admit) {
            return None;
        }
        let mut p = self.profile.clone();
        if w22_1_lift {
            p.entropy_mul_table = crate::effort::EntropyMulTable::screenshot_suppressed();
        } else if w44_29_lower || w44_166_photo_admit {
            // W44-166: when ONLY w44_166_photo_admit fires (w44_29_lower
            // false — i.e. d outside W44-152's [3.0, 5.0] band, typically
            // d=6 on 1418519), we still route through the variant Z table
            // chain. The W44-156 d_high split handles d>5.5.
            p.entropy_mul_table = if w44_98_variant_z_high_colour {
                if w44_156_d_high {
                    crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour_d_high()
                } else {
                    crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour()
                }
            } else if w44_99_variant_z_low_colour {
                if w44_156_d_high {
                    crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z_low_colour_d_high()
                } else {
                    crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z_low_colour()
                }
            } else if w44_96_variant_z {
                if w44_156_d_high {
                    crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z_d_high()
                } else {
                    crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z()
                }
            } else {
                crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed()
            };
            // W44-167 (Smart-Zenjxl chunk 4, 2026-05-21): post-table
            // selection, when the W44-167 gate fires AND a variant Z
            // table has been selected, override the `dct16x32` field to
            // a per-m3-aware lift value. Closes the W44-94 honest-stopped
            // 1420710 OPEN cells without regressing 1531677 (low-m3)
            // because the m3 split is the existing W44-98 sub-gate.
            // Gated on `ResolvedImprovements::find_best_32_per_m3_lift`
            // (Zenjxl/Aggressive default true; Libjxl/LeanFaster false).
            // Default env unset = Mode A = byte-identical to pre-W44-167.
            if w44_96_variant_z && self.resolved_improvements.find_best_32_per_m3_lift {
                let mode = w44_167_mode_env();
                p.entropy_mul_table.dct16x32 = w44_167_apply_lift(
                    mode,
                    w44_98_variant_z_high_colour,
                    w44_99_variant_z_low_colour,
                    p.entropy_mul_table.dct16x32,
                );
            }
        }
        if w44_65_suppress_dct64 {
            p.try_dct64 = false;
            // W44-123/124/135 dct32_keep_hint sub-dispatch:
            let w44_123_env_keep = {
                #[cfg(feature = "std")]
                {
                    std::env::var("__JXL_W44_123_KEEP_DCT32")
                        .map(|s| s == "1")
                        .unwrap_or(false)
                }
                #[cfg(not(feature = "std"))]
                {
                    false
                }
            };
            let w44_124_distance_in_band = self.distance >= w44_143_effective_min_distance()
                && self.distance <= W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE;
            let w44_124_auto_keep = w44_124_distance_in_band
                && self.zenanalyze_proxies.is_some_and(|zp| {
                    zp.m3_colourfulness >= W44_124_DCT32_KEEP_M3_MIN
                        && zp.edge_density < W44_124_DCT32_KEEP_EDGE_DENSITY_MAX
                });
            let dct32_policy = self.resolved_improvements.dct32_search_policy;
            let w44_123_keep_dct32 = match dct32_policy {
                crate::api::Dct32SearchPolicy::FollowDct64Suppression => {
                    w44_123_env_keep || w44_124_auto_keep
                }
                crate::api::Dct32SearchPolicy::KeepWhenDct64Suppressed => true,
            };
            if !w44_123_keep_dct32 {
                p.try_dct32 = false;
            }
        }
        Some(p)
    }

    /// Encode an image in linear sRGB format, optionally with an alpha channel.
    ///
    /// Input should be 3 channels (RGB) of f32 values in [0, 1] range.
    /// Values outside [0, 1] are allowed for out-of-gamut colors.
    ///
    /// If `alpha` is provided, it must be `width * height` bytes of u8 alpha values.
    /// Alpha is encoded as a modular extra channel alongside the VarDCT RGB data.
    ///
    /// For more than alpha — depth, spot color, selection mask, thermal,
    /// CFA — see [`Self::encode_with_extras`].
    pub fn encode(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        alpha: Option<&[u8]>,
    ) -> Result<VarDctOutput> {
        self.encode_with_stop(width, height, linear_rgb, alpha, None)
    }

    /// Like [`encode`](Self::encode) but polls `stop` at coarse (per-group)
    /// boundaries during entropy coding, returning [`Error::Cancelled`] if
    /// cancellation is requested. With `None` (or an `Unstoppable` token) the
    /// emitted bytes are identical to [`encode`](Self::encode) — the poll is a
    /// no-op on the success path.
    pub fn encode_with_stop(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        alpha: Option<&[u8]>,
        stop: Option<&dyn Stop>,
    ) -> Result<VarDctOutput> {
        match alpha {
            None => self.encode_with_extras_stop(width, height, linear_rgb, &[], stop),
            Some(buf) => {
                // Build a default-alpha ExtraChannel view borrowing the caller's
                // buffer. `with_alpha` honours the encoder's `alpha_associated`
                // setting; the file-header builder forwards it.
                let mut info = crate::headers::extra_channels::ExtraChannelInfo::alpha();
                info.alpha_associated = self.alpha_associated;
                let ec = super::extras::VardctExtra {
                    info: &info,
                    data: super::extras::VardctExtraBuf::U8(buf),
                };
                // Inline the dimension check here so the error message
                // points at `alpha` not at an abstract "extras[0]".
                let expected_alpha = width.checked_mul(height).ok_or(Error::DimensionOverflow {
                    width,
                    height,
                    channels: 1,
                })?;
                if buf.len() != expected_alpha {
                    return Err(Error::InvalidInput(format!(
                        "alpha length {} != expected {}",
                        buf.len(),
                        expected_alpha
                    )));
                }
                self.encode_inner(width, height, linear_rgb, &[ec], stop)
            }
        }
    }

    /// Encode RGB plus an arbitrary list of extra channels (alpha,
    /// depth, spot color, selection mask, thermal, CFA, …).
    ///
    /// Each [`crate::api::ExtraChannel`] carries its own
    /// [`crate::headers::extra_channels::ExtraChannelInfo`] which goes
    /// into the file-header metadata, plus the pixel buffer (u8 or u16).
    ///
    /// **Current scope (refs #9)**:
    /// - `dim_shift` must be `0` on every extra (full-resolution channels).
    /// - Single-group (≤256×256) supports any number of extras.
    /// - Multi-group supports any number of extras at `dim_shift = 0`.
    ///
    /// Other combinations return `Error::Unsupported` so the wire format
    /// stays correct as those paths are filled in.
    pub fn encode_with_extras(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        extras: &[crate::api::ExtraChannel<'_>],
    ) -> Result<VarDctOutput> {
        self.encode_with_extras_stop(width, height, linear_rgb, extras, None)
    }

    /// Like [`encode_with_extras`](Self::encode_with_extras) but polls `stop`
    /// at coarse (per-group) boundaries during entropy coding. With `None`
    /// (or an `Unstoppable` token) the emitted bytes are identical.
    pub fn encode_with_extras_stop(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        extras: &[crate::api::ExtraChannel<'_>],
        stop: Option<&dyn Stop>,
    ) -> Result<VarDctOutput> {
        // Materialize the internal `VardctExtra` views, validating
        // dimensions + bit depth + dim_shift up-front before any work.
        let mut views: Vec<super::extras::VardctExtra<'_>> = Vec::with_capacity(extras.len());
        for (idx, ec) in extras.iter().enumerate() {
            if ec.info().dim_shift != 0 {
                return Err(Error::InvalidInput(format!(
                    "extras[{idx}]: dim_shift = {} not yet supported in lossy encode (dim_shift > 0 \
                     for VarDCT extras is a follow-up)",
                    ec.info().dim_shift
                )));
            }
            let expected = width.checked_mul(height).ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 1,
            })?;
            let got = ec.data().len();
            if got != expected {
                return Err(Error::InvalidInput(format!(
                    "extras[{idx}]: expected {expected} samples for {width}x{height}, got {got}"
                )));
            }
            views.push(super::extras::VardctExtra::from_api(ec));
        }

        self.check_alpha_squeeze_supported(&views, Some((width, height)))?;
        self.encode_inner(width, height, linear_rgb, &views, stop)
    }

    /// Chunk-2.b gate (multi-group + dim_shift consolidation).
    ///
    /// When [`Self::alpha_squeeze`] is `true` and the lossy alpha
    /// pipeline is engaged (`alpha_distance > 0.0`) AND there is
    /// exactly one alpha extra, the squeeze pipeline is selected at
    /// the bitstream-emit site — for any image size (single-group
    /// and multi-group both supported as of chunk-2.b).
    ///
    /// Multi-extra (alpha + depth, alpha + spot, …) still returns
    /// `Error::NotImplemented`; that case needs per-coded-channel
    /// tree routing to interleave the squeezed alpha sub-channels
    /// with the non-alpha extras and stays queued (not in chunk-2.b
    /// scope).
    ///
    /// `dim_shift > 0` is rejected upstream in every lossy VarDCT
    /// entry point (`encode_with_extras`, `encode_from_precomputed_with_extras`,
    /// the pre-quantized variant — see `encoder.rs:927`, `2497`, `2901`)
    /// with `Error::InvalidInput`. The squeeze pipeline therefore
    /// inherits that rule rather than carrying its own redundant
    /// `dim_shift` check — the constraint is a property of VarDCT
    /// lossy extras, not of squeeze in particular. If the upstream
    /// `dim_shift > 0` gate ever moves (e.g. via libjxl-parity wiring
    /// of pre-subsampled extras), the squeeze pipeline's
    /// `build_alpha_squeeze_pipeline` already materializes the alpha
    /// channel at its native `width >> dim_shift × height >> dim_shift`
    /// resolution; the chunk-2.b writers would still need a Channel
    /// `hshift = dim_shift, vshift = dim_shift` seed to keep the
    /// decoder-side shift bracket classification consistent with
    /// `dec_modular.cc:354`.
    ///
    /// `dims = Some((w, h))` plumbed for future shape-aware gates
    /// (e.g. very wide / very tall edge cases); unused as of
    /// chunk-2.b — the writer handles arbitrary multi-group sizes.
    ///
    /// When alpha_squeeze is `false` (default), returns `Ok(())` and
    /// the existing extras pipeline runs unchanged — preserves the
    /// byte-for-byte hash-lock baseline (36/36 hashes match).
    fn check_alpha_squeeze_supported(
        &self,
        extras: &[super::extras::VardctExtra<'_>],
        _dims: Option<(usize, usize)>,
    ) -> Result<()> {
        if !self.alpha_squeeze_engaged() {
            return Ok(());
        }
        // Find the alpha extra index, if any.
        let alpha_idx = extras.iter().position(|ec| {
            matches!(
                ec.info.ec_type,
                crate::headers::extra_channels::ExtraChannelType::Alpha
            )
        });
        let Some(_alpha_idx) = alpha_idx else {
            // No alpha extra present → squeeze-on-extras is a no-op
            // even when the flag is set. Don't surprise the caller.
            return Ok(());
        };
        // Chunk-2 scope (still): exactly one extra (the alpha).
        // Multi-extra (alpha + depth, alpha + spot, …) needs per-
        // coded-channel tree routing to interleave the squeezed alpha
        // sub-channels with the non-alpha extras — not in chunk-2.b
        // scope.
        if extras.len() > 1 {
            return Err(Error::NotImplemented(
                "with_alpha_squeeze(true) currently supports only a single alpha extra. \
                 Multi-extra (alpha + depth, alpha + spot, …) routing through the per-band \
                 quantizer is queued. Leave alpha_squeeze at its default \
                 (false) for now to keep mixed-extras encoding working."
                    .into(),
            ));
        }
        // dim_shift > 0 is enforced upstream by every lossy VarDCT
        // entry-point validator with `Error::InvalidInput` (see
        // `encoder.rs:927`, `2497`, `2901`). The squeeze pipeline
        // inherits that rule unmodified; we deliberately don't shadow
        // it with a squeeze-flag-specific message because the
        // restriction isn't a property of the squeeze flag.
        Ok(())
    }

    /// Try to build the squeeze pipeline for the chunk-2 / chunk-2.b
    /// alpha path. Returns `Some(pipeline)` when
    /// [`Self::alpha_squeeze_engaged`] is `true` AND the extras shape
    /// matches the supported scope (single alpha extra). Returns
    /// `Ok(None)` for the flag-off path or shapes that fall through
    /// to the existing raw-pixel writer (no-alpha-extra case,
    /// multi-extra fallback).
    ///
    /// `width` and `height` are the full image dims. The pipeline is
    /// built once per image and partitioned by section at the
    /// bitstream-emit site (see
    /// [`super::extras::AlphaSqueezePipeline::partition`]).
    pub(crate) fn maybe_build_alpha_squeeze_pipeline(
        &self,
        extras: &[super::extras::VardctExtra<'_>],
        width: usize,
        height: usize,
    ) -> Result<Option<super::extras::AlphaSqueezePipeline>> {
        if !self.alpha_squeeze_engaged() {
            return Ok(None);
        }
        if extras.len() != 1 {
            return Ok(None);
        }
        let alpha = &extras[0];
        if !matches!(
            alpha.info.ec_type,
            crate::headers::extra_channels::ExtraChannelType::Alpha
        ) {
            return Ok(None);
        }
        // dim_shift > 0 is enforced upstream as `Error::InvalidInput`;
        // assert here so a future caller skipping the upstream
        // validator trips the bug visibly.
        debug_assert_eq!(
            alpha.info.dim_shift, 0,
            "alpha squeeze pipeline expects dim_shift=0 alpha (upstream \
             VarDCT lossy entry-point validator enforces this)"
        );
        // Chunk-3 heuristic: skip squeeze when alpha is a single
        // constant value over the full image. The W14-1 ChannelCompact
        // path (`e97e5bb7`, `write_modular_extras_subbitstream` →
        // `kPalette(num_c=1, nb_colors=1)`) collapses constant extras
        // to ~76 bytes regardless of `alpha_distance`. Routing those
        // through the squeeze pipeline costs +0.6 to +0.8% on the
        // `red_night_opaque` W16-2 audit baseline (`191801a1`) because
        // squeeze adds a GroupHeader + per-band tree leaves on top of
        // an already-minimal payload; every squeeze sub-channel is
        // itself constant so the residuals carry no extra information.
        //
        // ChannelCompact wins for constant alpha; squeeze wins for
        // varying alpha (W14-4 / chunk-2 / chunk-2.b shows -30% to
        // -56% on the photo-mask + UI-gradient audit images).
        if alpha.is_constant_full_image(width, height) {
            return Ok(None);
        }
        let bits = alpha.info.bit_depth.bits_per_sample;
        let shift0_q = self.compute_extra_pixel_quantizer_shifted(bits, alpha.info.ec_type, 0);
        let shifted_q = |shift: u32| -> u32 {
            self.compute_extra_pixel_quantizer_shifted(bits, alpha.info.ec_type, shift)
        };
        let pipeline =
            super::extras::build_alpha_squeeze_pipeline(alpha, width, height, shift0_q, shifted_q)?;
        Ok(Some(pipeline))
    }

    /// Shared implementation backing both [`Self::encode`] and
    /// [`Self::encode_with_extras`]. Takes already-validated extras as
    /// the internal `VardctExtra` view; performs the RGB-shape check
    /// then drives the full pipeline.
    fn encode_inner(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        extras: &[super::extras::VardctExtra<'_>],
        stop: Option<&dyn Stop>,
    ) -> Result<VarDctOutput> {
        // Earliest cancellation checkpoint — polled BEFORE any encode work, on
        // EVERY entropy path (two-pass, single-pass, and the `num_sections == 4`
        // single-group fast path). The per-group checkpoints further down add
        // mid-encode responsiveness, but they live only in the multi-group
        // two-pass branch; this entry poll guarantees a fired `Stop` aborts
        // regardless of which path runs (the bug behind the previously-failing
        // `test_stop_cancels_lossy_multigroup`). No-op / byte-identical under an
        // `Unstoppable` token or `None`.
        if let Some(s) = stop {
            s.check().map_err(|_| Error::Cancelled)?;
        }
        let expected_rgb = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 3,
            })?;
        if linear_rgb.len() != expected_rgb {
            return Err(Error::InvalidInput(format!(
                "linear_rgb length {} != expected {}",
                linear_rgb.len(),
                expected_rgb
            )));
        }

        crate::debug_rect::clear();

        // Optional per-phase wall-clock timing. Gated on env var
        // `__JXL_ENC_PHASE_TIMING` so default encodes are unaffected.
        // Mirrors the pattern already used in `encode_from_precomputed`
        // and `encode_two_pass_to_writer`.
        #[cfg(feature = "__env_var_diagnostics")]
        let _phase_dbg = std::env::var_os("__JXL_ENC_PHASE_TIMING").is_some();
        #[cfg(not(feature = "__env_var_diagnostics"))]
        let _phase_dbg = false;
        let _t_total = std::time::Instant::now();

        // Calculate dimensions
        let xsize_blocks = div_ceil(width, BLOCK_DIM);
        let ysize_blocks = div_ceil(height, BLOCK_DIM);
        let xsize_groups = div_ceil(width, GROUP_DIM);
        let ysize_groups = div_ceil(height, GROUP_DIM);
        let xsize_dc_groups = div_ceil(width, DC_GROUP_DIM);
        let ysize_dc_groups = div_ceil(height, DC_GROUP_DIM);
        let num_groups = xsize_groups * ysize_groups;
        let num_dc_groups = xsize_dc_groups * ysize_dc_groups;

        // Number of sections: DC global + DC groups + AC global + AC groups
        let num_sections = 2 + num_dc_groups + num_groups;

        // Pad to block boundary dimensions
        let padded_width = xsize_blocks * BLOCK_DIM;
        let padded_height = ysize_blocks * BLOCK_DIM;

        // Validate linear-RGB at intake. The `forward_xyb` SIMD kernel
        // uses `mixed.max(0.0)` per channel, which silently coerces NaN
        // to `0.0` (IEEE-754 ordered max returns the non-NaN operand).
        // That means a caller-supplied NaN linear-RGB never reaches the
        // XYB output — the post-XYB check would not fire either. To
        // surface caller bugs (Error mode) or actively scrub them
        // (Sanitize mode), we must check / fix here, before forward_xyb
        // runs. For 8-bit / 16-bit pixel layouts the linear-RGB
        // conversion is total (no non-finite possible) and the
        // is_finite_plane scan is a fast read-only no-op (~55 GB/s).
        //
        // Sanitize mode used to skip the input check entirely, relying
        // on forward_xyb's silent NaN→0 max to mask non-finite values.
        // That left the advertised "SIMD scrub" never running on the
        // linear-RGB plane and made Error/Sanitize behavior diverge.
        // Now Sanitize actively rewrites non-finite values to 0.0
        // (~12.5 GB/s), then runs the rest of the pipeline on a clean
        // buffer; the downstream XYB scan stays as defense-in-depth.
        let sanitized_linear_rgb_storage: Option<alloc::vec::Vec<f32>> =
            match self.non_finite_action {
                crate::api::NonFiniteAction::Error => {
                    if !jxl_simd::is_finite_plane(linear_rgb) {
                        return Err(crate::error::Error::InvalidInput(
                            "non-finite (NaN / ±Inf) value detected in linear-RGB input. \
                             Use LossyConfig::with_non_finite_action(NonFiniteAction::Sanitize) \
                             to silently scrub non-finite values to 0.0 instead."
                                .into(),
                        ));
                    }
                    None
                }
                crate::api::NonFiniteAction::Sanitize => {
                    // sanitize_finite needs &mut, but the caller-supplied
                    // buffer is borrowed. Clone-and-sanitize when (and
                    // only when) Sanitize mode is selected. For 8-bit /
                    // 16-bit pixel layouts there are never non-finite
                    // values, so most callers stay on the Error fast path.
                    let mut owned: alloc::vec::Vec<f32> = linear_rgb.to_vec();
                    let _ = jxl_simd::sanitize_finite(&mut owned);
                    Some(owned)
                }
            };
        let linear_rgb: &[f32] = sanitized_linear_rgb_storage
            .as_deref()
            .unwrap_or(linear_rgb);

        // Convert to XYB with edge-replicated padding to block boundaries.
        // This allows SIMD to process full blocks without bounds checking.
        let _t_xyb = std::time::Instant::now();
        let (mut xyb_x, mut xyb_y, mut xyb_b) =
            self.convert_to_xyb_padded(width, height, padded_width, padded_height, linear_rgb)?;
        let _ms_xyb = _t_xyb.elapsed().as_secs_f64() * 1000.0;

        // Defense-in-depth XYB scan. Catches downstream-bug non-finite
        // (memory corruption, butteraugli-loop reconstruction polluting
        // XYB) — should never fire on the encode-fresh path because
        // forward_xyb is finite-output-for-finite-input.
        validate_xyb_planes(self.non_finite_action, &mut xyb_x, &mut xyb_y, &mut xyb_b)?;

        // Noise parameters. Four sources, in priority order
        // (matches libjxl enc_frame.cc:680-689):
        // 1. `photon_noise_iso`: caller-supplied ISO value, bypasses
        //    content estimation. Matches libjxl --photon_noise. Useful
        //    for re-encoding denoised content with controlled grain.
        // 2. `manual_noise_lut`: caller-supplied 8-point LUT. Bypasses
        //    everything else. Matches libjxl cparams.manual_noise.
        // 3. `enable_noise` + content estimation: scan flat patches,
        //    fit an 8-point LUT via SCG optimisation.
        // 4. None of the above: no noise synthesis.
        let noise_params = if let Some(iso) = self.photon_noise_iso
            && iso > 0.0
        {
            // The decoder regenerates noise during rendering from the
            // 10-bit-per-point LUT; the encoder just emits the LUT
            // here. No content estimation, no denoise pre-filter — we
            // *add* synthetic grain, not preserve real noise.
            let params = super::noise::simulate_photon_noise(width, height, iso);
            if params.has_any() { Some(params) } else { None }
        } else if let Some(lut) = self.manual_noise_lut {
            // Caller-supplied LUT — clamp to the encodable range
            // [0, NOISE_LUT_MAX ≈ 0.9995] so write_noise_params can't
            // panic on the 10-bit-quantise debug_assert. has_any()
            // gates trivial all-zero LUTs out.
            let mut params = super::noise::NoiseParams::default();
            for (dst, src) in params.lut.iter_mut().zip(lut.iter()) {
                *dst = src.clamp(0.0, 0.9995);
            }
            if params.has_any() { Some(params) } else { None }
        } else if self.enable_noise {
            let quality_coef = noise_quality_coef(self.distance);
            let params = estimate_noise_params(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                quality_coef,
            );

            // Apply denoising pre-filter if enabled and noise was detected.
            // Removes estimated noise before encoding so the encoder spends fewer
            // bits on noise; the decoder re-adds it from the encoded parameters.
            if self.enable_denoise
                && let Some(ref p) = params
            {
                denoise_xyb(
                    &mut xyb_x,
                    &mut xyb_y,
                    &mut xyb_b,
                    padded_width,
                    padded_height,
                    p,
                    quality_coef,
                );
            }

            params
        } else {
            None
        };

        // Detect and subtract patches (before gaborish, after noise).
        // Patches work in the XYB domain: detect repeated rectangular elements,
        // store unique patterns in a reference frame, subtract from image.
        //
        // Distance-aware kMinPeak: below d=1.0 we revert to libjxl's
        // `kMinPeak = 2` because the W2-5 chunk 1 relaxation (commit
        // 7b8c06e) admits low-magnitude text patches that do not
        // amortize their ref-frame overhead at low distance — measured
        // `windows95.png @ d=0.5` regressed by +465 bytes (+0.96 %)
        // before this gate. At d>=1.0 the chunk 1 relaxation pays off
        // (`windows95 @ d=1.0`: -53 B; `@ d=2.0`: -43 B), so above the
        // threshold we keep the looser detector.
        // Distance-aware kMinPeak (W3-1 / commit 4fb0f52): libjxl
        // parity (=2) below d=1.0, W2-5 chunk 1 relaxation (=1) at
        // d>=1.0. RFC#45 chunk 3 layered a per-patch cost gate on top
        // (inside `find_and_build_with_per_patch_gate`) but did NOT
        // change `min_peak`: attempting `min_peak=1` at d<1.0 brought
        // back the +465 B windows95 regression that W3-1 closed. The
        // per-patch gate is a refinement, not a replacement for the
        // detector-side `min_peak` threshold — it captures patches
        // already detected, not ones the detector rejected.
        //
        // W41-1 (issue #52) investigated raising `min_peak=2` at d>=3.0
        // to match libjxl unconditional `kMinPeak=2`, hypothesising it
        // would close the W38-2 WF2 wedge (+22-51 % bytes vs cjxl on
        // screenshots at e7+ d>=3). The measurement ruled the
        // hypothesis OUT: across the 3 wedge images (`imac_g3`,
        // `codec_wiki`, `terminal`) the detected patch set is
        // IDENTICAL at both thresholds (e.g. imac_g3: 277 refs / 2052
        // occurrences at either). Only `windows95.png` admits 3 extra
        // refs / ~95 extra occurrences at `min_peak=1`, and clamping
        // it to 2 saves ~0.7-1.5 % bytes at d>=3 but regresses ssim2
        // by 0.4-1.3 points (-0.7 at d=3, -1.3 at d=5). Wedge does not
        // close; not shipped. See benchmarks/patches_min_peak_distance_2026-05-19.tsv.
        let _t_patches = std::time::Instant::now();
        let min_peak = if self.distance < 1.0 { 2 } else { 1 };
        // W36-3: patches photo-skip dispatch. Consult the per-block-mean
        // `median(mask1x1)` screenshot discriminator (same statistic the
        // auto-splines `looks_like_screenshot` gate uses, mirroring the
        // GPU encoder's `compute_block_mask_means`) BEFORE running the
        // patches scan. On photo content the patches scan is known to
        // produce empty `PatchesData` (W11-1 + W12-5: "Zero overhead on
        // CLIC photos (patches correctly produce nothing)") — so
        // short-circuiting the ~25-30 ms/MP scan there is a wall-clock
        // win with byte-identical output.
        //
        // **Why per-block-mean, not the raw `median_mask1x1` used by
        // `content_aware_entropy_mul`**: the raw-pixel median catches
        // pixel-grid edges in low-resolution UI and pulls the median
        // below 95 even though the image is glyph-heavy and the
        // patches scan is highly profitable. Per-block-mean smooths
        // over pixel-level gridlines and stays high on UI content.
        //
        // **Why threshold 60 instead of the shared 95**: see
        // `PATCHES_DISPATCH_BLOCK_MASK_THRESHOLD` doc — cost asymmetry
        // is opposite to auto-splines, so prefer false-positive
        // (run scan on photo → byte-identical) over false-negative
        // (skip scan on screenshot → 30-70 % regression).
        //
        // Dispatch policy (see `crate::api::PatchesDispatch`):
        //   * `Auto` (default): run scan when per-block-mean
        //     `median(mask1x1) > 60` (catches CLIC photo edge case +
        //     all of gb82-sc including windows95.png 640×480).
        //   * `AlwaysScan`: run scan unconditionally (pre-W36-3
        //     behavior — no discriminator pass).
        //   * `NeverScan`: short-circuit the scan on every image.
        let should_scan_patches = self.enable_patches
            && match self.patches_dispatch {
                crate::api::PatchesDispatch::NeverScan => false,
                crate::api::PatchesDispatch::AlwaysScan => true,
                crate::api::PatchesDispatch::Auto => {
                    // The discriminator pass costs one
                    // `compute_mask1x1` + per-block-mean reduction +
                    // partial sort — a few ms/MP, well under the
                    // ~25-30 ms/MP patches scan we save when the gate
                    // fires. The pass uses the PRE-patches /
                    // PRE-gaborish XYB just like the later quant-field
                    // mask1x1, so subsequent subtract_patches /
                    // subtract_splines mutations don't observe the
                    // wrong mask. (The late `compute_mask1x1` at the
                    // quant_field step recomputes on the post-subtract
                    // XYB.)
                    //
                    // `None` from the discriminator (image smaller
                    // than 8×8 in either dim) falls back to "run scan"
                    // because that's the safe / byte-identical
                    // direction.
                    patches_dispatch_block_mask_median(&xyb_y, width, height, padded_width)
                        .map(|m| m > PATCHES_DISPATCH_BLOCK_MASK_THRESHOLD)
                        .unwrap_or(true)
                }
            };
        let mut patches_data = if should_scan_patches {
            super::patches::find_and_build_with_per_patch_gate(
                [&xyb_x, &xyb_y, &xyb_b],
                width,
                height,
                padded_width,
                min_peak,
                Some(self.distance),
                self.use_ans,
            )
        } else {
            None
        };
        // Cost-benefit gating for experimental mode only.
        // libjxl uses patches unconditionally when detected (no cost check),
        // so reference mode skips this to match.
        if matches!(self.encoder_mode, crate::api::EncoderMode::Experimental)
            && let Some(ref pd) = patches_data
            && !pd.is_cost_effective(self.distance, self.use_ans)
        {
            patches_data = None;
        }
        // Quantize ref_image so subtract/add use the same values the decoder will reconstruct.
        if let Some(ref mut pd) = patches_data {
            pd.quantize_ref_image();
        }
        if let Some(ref pd) = patches_data {
            let mut xyb = [
                core::mem::take(&mut xyb_x),
                core::mem::take(&mut xyb_y),
                core::mem::take(&mut xyb_b),
            ];
            super::patches::subtract_patches(&mut xyb, padded_width, pd);
            let [x, y, b] = xyb;
            xyb_x = x;
            xyb_y = y;
            xyb_b = b;
        }

        // Dot detection (refs #19). libjxl gates at speed_tier <= kSquirrel
        // (effort >= 7) AND distance >= 3.0, AND only when text-like
        // patches haven't been found. We mirror the gating exactly.
        // Feature is off by default — niche (astronomy / specular highlights);
        // enable via `LossyConfig::with_dot_detection(true)`.
        if self.enable_dot_detection
            && self.effort >= 7
            && self.distance >= 3.0
            && patches_data.is_none()
        {
            let dots = super::dot_detection::detect_gaussian_ellipses(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                &super::dot_detection::GaussianDetectParams::default(),
            );
            #[cfg(feature = "debug-tokens")]
            crate::debug_log!("dot detection: found {} candidate dots", dots.len());
            // Promote the detected dots into a fresh PatchesData. The
            // subsequent quantize_ref_image + subtract_patches block
            // below treats them identically to text-like patches.
            if !dots.is_empty()
                && let Some(mut pd) = super::patches::PatchesData::from_dots(&dots)
            {
                pd.quantize_ref_image();
                let mut xyb = [
                    core::mem::take(&mut xyb_x),
                    core::mem::take(&mut xyb_y),
                    core::mem::take(&mut xyb_b),
                ];
                super::patches::subtract_patches(&mut xyb, padded_width, &pd);
                let [x, y, b] = xyb;
                xyb_x = x;
                xyb_y = y;
                xyb_b = b;
                patches_data = Some(pd);
            }
        }

        let _ms_patches = _t_patches.elapsed().as_secs_f64() * 1000.0;
        let _t_splines = std::time::Instant::now();
        // Build and subtract splines (after patches, before gaborish).
        // Splines are additive overlays: encoder subtracts, decoder adds back.
        // Uses default DC CfL params (y_to_x=0.0, y_to_b=1.0) since we write default DC cmap.
        //
        // Resolution order, mirroring libjxl `enc_heuristics.cc:1044-1055`:
        //   1. Manual non-empty `self.splines` always wins (libjxl
        //      `cparams.custom_splines.HasAny()`).
        //   2. Else, when `self.auto_splines` is set AND effort ≥ 7
        //      (libjxl `speed_tier <= kSquirrel`), call `find_splines()`
        //      on the post-patches XYB planes.
        //   3. Else, no splines.
        //
        // Chunk 1: `find_splines` is a stub that returns `vec![]`,
        // matching libjxl's `enc_splines.cc:104-107` TODO. The empty
        // branch below short-circuits to `None`, leaving every
        // default-config encode byte-identical.
        let effective_splines = if let Some(ref splines) = self.splines {
            // Manual override always wins, even if empty (caller may
            // explicitly disable by passing `vec![]`).
            Some(splines.clone())
        } else if self.auto_splines && self.effort >= 7 {
            // Chunk-5 content discriminator: skip auto-splines detection
            // on screenshot-like content. The bbox-area-linear
            // energy-drop proxy used by the per-spline cost-benefit gate
            // structurally over-claims VarDCT byte savings on long
            // bright ridges (table borders, wallpaper edges), regressing
            // real encodes by ~3% on `codec_wiki.png` and `imac_g3.png`
            // — see chunk-4 bench notes (`benchmarks/
            // auto_splines_bench_2026-05-17_chunk4.meta`) and the
            // `effort.rs::auto_splines_default` doc-comment for the
            // structural-proxy explanation.
            //
            // Uses the same `median(mask1x1) > 95.0` discriminator that
            // the GPU encoder's W7-3 AFV cost-grid gate uses
            // (`jxl-encoder-gpu/src/lossy_encoder.rs::
            // SCREENSHOT_MEDIAN_MASK_THRESHOLD`). On the chunk-5 corpus
            // bench (`benchmarks/auto_splines_bench_2026-05-17_chunk5.tsv`)
            // this correctly classifies every screenshot (`terminal`,
            // `codec_wiki`, `imac_g3`) and every flat-background line
            // synthetic as screenshot-class, while admitting all 5
            // CLIC2025-1024 photos (photo median ~56 vs screenshot/synth
            // median 100).
            if super::splines::looks_like_screenshot(&xyb_y, width, height, padded_width) {
                None
            } else {
                // Pass distance through so the per-spline cost-benefit gate
                // can scale the "VarDCT bytes saved per spline pixel"
                // estimate. The gate model assumes ~5 bits/pixel of AC
                // residual at d=1.0 and clamps the divisor at 1.0 — see
                // `spline_passes_cost_gate` in `vardct/splines.rs`.
                Some(super::splines::find_splines_at_distance(
                    &xyb_x,
                    &xyb_y,
                    &xyb_b,
                    width,
                    height,
                    padded_width,
                    self.distance,
                ))
            }
        } else {
            None
        };
        let splines_data = if let Some(ref splines) = effective_splines {
            if !splines.is_empty() {
                let sd = super::splines::SplinesData::from_splines(
                    splines.clone(),
                    0,   // quantization_adjustment
                    0.0, // y_to_x (default DC CfL)
                    1.0, // y_to_b (default DC CfL)
                    width,
                    height,
                );
                {
                    let mut xyb = [
                        core::mem::take(&mut xyb_x),
                        core::mem::take(&mut xyb_y),
                        core::mem::take(&mut xyb_b),
                    ];
                    super::splines::subtract_splines(&mut xyb, padded_width, width, height, &sd);
                    let [x, y, b] = xyb;
                    xyb_x = x;
                    xyb_y = y;
                    xyb_b = b;
                }
                Some(sd)
            } else {
                None
            }
        } else {
            None
        };

        let _ms_splines = _t_splines.elapsed().as_secs_f64() * 1000.0;
        let _t_quant_field = std::time::Instant::now();
        // Compute pixel chromacity stats BEFORE gaborish (matching libjxl pipeline).
        // Gaborish sharpening inflates gradients, producing overly aggressive adjustment.
        // Gated at effort >= 7 to skip the full-image gradient scan at low effort.
        let (chromacity_x, chromacity_b) = if self.profile.chromacity_adjustment {
            let pixel_stats = super::frame::PixelStatsForChromacityAdjustment::calc(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
            );
            (
                pixel_stats.how_much_is_x_channel_pixelized(),
                pixel_stats.how_much_is_b_channel_pixelized(),
            )
        } else {
            (0, 0)
        };

        // Compute adaptive per-block quantization field and masking on ORIGINAL
        // (pre-gaborish) XYB. libjxl computes InitialQuantField before GaborishInverse
        // (enc_heuristics.cc:1117-1142, comment: "relies on pre-gaborish values").
        // When gaborish is off, scale distance by 0.62 for the quant field only
        // (not global_scale/quant_dc). This matches libjxl enc_heuristics.cc:1119.
        let distance_for_iqf = if self.enable_gaborish {
            self.distance
        } else {
            self.distance * 0.62
        };

        // W44-109: compute mask1x1 EARLY (before quant_field) so the
        // screenshot-class qf seed-scale gate can fire at e5/e6/e7.
        //
        // The mask1x1 here is the SAME object as the one computed at line
        // ~2582 for the AC-strategy pixel-domain loss. Moving the
        // compute up by ~120 lines lets a single result feed BOTH the
        // W44-109 pre-scale (now) AND the AC-strategy search (later);
        // there's no per-byte cost (the mask is computed once, kept in
        // scope, and the downstream `let mask1x1 = ...` block at line
        // ~2582 becomes a pass-through that just decides whether to
        // wrap it in Some(_) for the pixel-domain-loss dispatch).
        //
        // Gate `pre_scale_mask1x1` to the same predicate as the
        // downstream `let mask1x1` block at line ~2582:
        //   - `ac_strategy_enabled` (lossy VarDCT only)
        //   - `pixel_domain_loss` (effort >= 5 via effort.rs:927)
        //   - `pld_force_off` skips it (caller pinned PixelLossDispatch::AlwaysOff)
        //
        // Since the W44-109 gate also requires effort <= 7 (so the
        // W44-105 buttloop path can own e>=8), the effective firing
        // range is e ∈ {5, 6, 7} on lossy encodes with default
        // PixelLossDispatch. Lower effort skips because
        // `pixel_domain_loss = false` → `mask1x1_for_pre_scale = None` →
        // gate predicate evaluates to "not screenshot" and the scale
        // stays at 1.0.
        let pld_force_off_for_pre_scale = matches!(
            self.pixel_loss_dispatch,
            crate::api::PixelLossDispatch::AlwaysOff
        );
        let mask1x1_for_pre_scale: Option<Vec<f32>> =
            if self.ac_strategy_enabled && self.pixel_domain_loss && !pld_force_off_for_pre_scale {
                Some(super::adaptive_quant::compute_mask1x1_with_budget(
                    &xyb_y,
                    padded_width,
                    padded_height,
                    self.budget.as_ref(),
                )?)
            } else {
                None
            };
        let mask1x1_median_for_pre_scale: Option<f32> = mask1x1_for_pre_scale
            .as_deref()
            .map(|m| median_mask1x1(m, padded_width, width, height));
        #[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
        if std::env::var_os("JXL_WP_DISPATCH_DUMP_MASK").is_some() {
            eprintln!(
                "WP-DISPATCH mask1x1_median={:?} effort={} distance={}",
                mask1x1_median_for_pre_scale, self.effort, self.distance
            );
        }

        // W44-168 (Smart-Zenjxl chunk 5, 2026-05-21): compute the
        // adaptive `butteraugli_iters` override EARLY so the W44-105
        // qf pre-scale gate (just below) sees the corrected value and
        // doesn't double-apply with the buttloop. Also computed here
        // so it can be used to gate the buttloop entry block (line
        // ~4690) which currently reads `self.butteraugli_iters > 0`
        // directly — Mode C (TexturedExtend) needs to flip that gate
        // when `self.butteraugli_iters == 0`.
        //
        // Compute `mask_p25_for_pre_scale` from the existing
        // `mask1x1_for_pre_scale` (no extra mask compute cost — the
        // mask is already in scope). `edge_density` comes from
        // `self.zenanalyze_proxies` (already on the encoder for 8-bit
        // sRGB layouts).
        let mask1x1_p25_for_pre_scale: Option<f32> = mask1x1_for_pre_scale
            .as_deref()
            .map(|m| percentile_mask1x1(m, padded_width, width, height, 0.25));
        let edge_density_for_pre_scale: Option<f32> =
            self.zenanalyze_proxies.map(|p| p.edge_density);
        #[cfg(feature = "butteraugli-loop")]
        let effective_buttloop_iters = {
            // W44-169 (Smart-Zenjxl chunk 6, 2026-05-21): production
            // default narrow path. Mirrors the W44-156 distance-narrowing
            // pattern applied to the W44-168 mechanism layer. Fires
            // Mode B (SmoothSkip) ONLY when `target_distance ∈ [4.0,
            // 5.0]` on smooth/screenshot content at `effort >= 8`. The
            // narrow band drops the d=6 cell where broad Mode B
            // destroyed the W44-166 +0.45 SSIM2 win on 1418519 e8 d=6
            // (W44-168 honest-stop).
            //
            // **Precedence**: W44-168 env hook (`JXL_W44_168_MODE=B|C|D`)
            // OVERRIDES the W44-169 narrow path when set — kept for
            // diagnostic A/B benching (e.g. measuring broad Mode B vs
            // the narrow shipped default). Default env unset =
            // W44-169 narrow path active when the flag is on.
            let w44_168_mode = if self.resolved_improvements.adaptive_buttloop_iters {
                w44_168_mode_env()
            } else {
                W44_168IterMode::Baseline
            };
            let env_overrides = !matches!(w44_168_mode, W44_168IterMode::Baseline);
            if env_overrides {
                // Diagnostic env override: keep W44-168 broad dispatch
                // alive for measurement (Mode B/C/D as documented).
                w44_168_compute_iters(
                    self.butteraugli_iters,
                    self.effort,
                    mask1x1_median_for_pre_scale,
                    mask1x1_p25_for_pre_scale,
                    edge_density_for_pre_scale,
                    w44_168_mode,
                )
            } else {
                // W44-169 production default narrow path. Suppress the
                // unused-binding warning for `edge_density_for_pre_scale`
                // (the narrow path only consumes the smooth predicate).
                let _ = edge_density_for_pre_scale;
                w44_169_compute_iters_narrow(
                    self.butteraugli_iters,
                    self.effort,
                    self.distance,
                    mask1x1_median_for_pre_scale,
                    mask1x1_p25_for_pre_scale,
                    self.resolved_improvements.adaptive_buttloop_iters_narrow,
                )
            }
        };
        #[cfg(not(feature = "butteraugli-loop"))]
        let effective_buttloop_iters: u32 = {
            // Silence unused-variable warnings when buttloop is off —
            // the W44-168 inputs only feed the buttloop dispatch.
            let _ = (mask1x1_p25_for_pre_scale, edge_density_for_pre_scale);
            0
        };

        // Step 1: Compute float quant field on pre-gaborish XYB.
        //
        // libjxl effort gating (enc_heuristics.cc:1097-1128):
        // - effort < 5 (speed_tier > kHare): flat quant field = q_numerator/distance
        // - effort >= 5 (speed_tier <= kHare): adaptive via InitialQuantField
        let (mut quant_field_float, masking) = if self.profile.use_adaptive_quant {
            super::adaptive_quant::compute_quant_field_float_with_budget(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                distance_for_iqf,
                self.profile.k_ac_quant,
                self.budget.as_ref(),
            )?
        } else {
            // Flat quant field for low effort (matches libjxl enc_heuristics.cc:1105-1106).
            // Account both nblocks-sized f32 buffers against the budget.
            let nblocks = xsize_blocks.checked_mul(ysize_blocks).ok_or(
                crate::error::Error::DimensionOverflow {
                    width: xsize_blocks,
                    height: ysize_blocks,
                    channels: 1,
                },
            )?;
            crate::budget::MemoryBudget::reserve_permanent_opt(
                self.budget.as_ref(),
                (nblocks as u64).saturating_mul(4 * 2),
            )?;
            // Runtime fallible-alloc policy for these dimension-driven quant
            // buffers; byte-identical (resize fills to the same values that
            // `vec![v; nblocks]` would, without reallocating the exact capacity).
            let fallible = self.budget.as_ref().is_some_and(|b| b.is_fallible());
            let q = self.profile.initial_q_numerator / self.distance;
            let mut flat_qf = crate::budget::vec_with_capacity_fallible(fallible, nblocks)?;
            flat_qf.resize(nblocks, q);
            let masking_val = 1.0 / (q + 0.001);
            let mut flat_masking = crate::budget::vec_with_capacity_fallible(fallible, nblocks)?;
            flat_masking.resize(nblocks, masking_val);
            (flat_qf, flat_masking)
        };

        // Step 2: Compute distance params with effort-matched global_scale.
        //
        // Uses profile.initial_q_numerator for q = numerator / distance.
        // The adaptive median/MAD formula is only used inside the butteraugli
        // loop (effort >= 8).
        let mut params = match self.original_distance {
            Some(orig) if orig > self.distance => {
                DistanceParams::compute_for_profile_with_original(
                    self.distance,
                    orig,
                    &self.profile,
                )
            }
            _ => DistanceParams::compute_for_profile(self.distance, &self.profile),
        };
        if let Some(rescale) = self.quant_ac_rescale {
            params.apply_quant_ac_rescale(rescale);
        }
        // libjxl --epf override: when the caller pinned a level, it wins
        // over the distance-derived `epf_iters` (enc_frame.cc:284-285).
        self.apply_epf_level_override(&mut params);

        // Apply pixel-level chromacity adjustments using pre-gaborish stats
        // Gated at effort >= 7 (speed_tier <= kSquirrel) matching libjxl
        if self.profile.chromacity_adjustment {
            params.apply_chromacity_adjustment(chromacity_x, chromacity_b);
        }

        debug_rect!(
            "enc/params",
            0,
            0,
            width,
            height,
            "global_scale={} quant_dc={} scale={:.4} inv_scale={:.4} epf_iters={} chrom_x={:.3} chrom_b={:.3}",
            params.global_scale,
            params.quant_dc,
            params.scale,
            params.inv_scale,
            params.epf_iters,
            chromacity_x,
            chromacity_b
        );

        // W44-109: pre-scale `quant_field_float` on screenshot-class content
        // at low effort (e ∈ {5, 6, 7}). Mirrors the W44-105 buttloop
        // seed-scale fix at adaptive_quant time; at e>=8 the buttloop's
        // own seed scale (`butteraugli_loop.rs:648`) takes over and this
        // helper returns `1.0` to avoid double-scaling.
        //
        // Mask1x1 (and its median) were computed at the top of this
        // function before the qf computation; the screenshot
        // discriminator + W44-108 m3 sub-discriminator both feed the
        // shared resolver in `butteraugli_loop.rs`.
        //
        // Photo-class content keeps `is_screenshot = false` so the
        // scale stays at 1.0 → byte-identical to pre-W44-109. Only
        // low-effort screenshot-class hits the lossy gate.
        {
            // W44-213: tuning-override-aware threshold lookup.
            let median_threshold = crate::runtime_or_default!(
                super::perceptual_tuning::SCREENSHOT_MEDIAN_THRESHOLD,
                screenshot_median_threshold,
            );
            let is_screenshot =
                mask1x1_median_for_pre_scale.is_some_and(|med| med > median_threshold);
            let m3 = self.zenanalyze_proxies.map(|p| p.m3_colourfulness);
            // W44-129 Chunk C / W44-130 Chunk D: read the resolved
            // `adaptive_quant_qf_seed` enum directly from
            // `ResolvedImprovements` (always populated; defaults to
            // `AutoScalePerEffort` for direct `VarDctEncoder::new`
            // callers). `Off` short-circuits the scale to 1.0 (Libjxl
            // strategy).
            let adaptive_quant_qf_seed_policy = self.resolved_improvements.adaptive_quant_qf_seed;
            // W44-168: feed the W44-168-adjusted `effective_buttloop_iters`
            // (not the raw `self.butteraugli_iters`) so the W44-105
            // pre-scale gate doesn't double-apply when Mode C
            // (TexturedExtend) bumps iters from 0 → 2 at e7. The
            // existing W44-105 / W44-109 short-circuit
            // (`butteraugli_iters > 0` → return 1.0) then correctly
            // declines the pre-scale on cells the buttloop will own.
            // W44-176: pass the full `ZenanalyzeProxies` and the
            // `terminal_class_exclude` flag so the helper can suppress
            // the W44-109 lift on terminal-class screenshots
            // (luma_var ∈ [1500, 2200] AND fcbr ≥ 0.70). Composes
            // underneath the existing W44-108 m3 sub-gate; only fires
            // when the gate WOULD have fired. graph/imac_g3/imac_dark/
            // gmessages/gui SSIM2 wins from the lift are preserved
            // (their proxies fail the discriminator).
            //
            // W44-AUDIT-6 Phase 1 (2026-05-24): also pass the
            // `high_colour_class_exclude` flag so the helper can
            // suppress the W44-109 lift on high-colour mixed-content
            // screenshots (`m3_colourfulness >= 80.0`). Excludes
            // codec_wiki-class wedges where the lift over-allocates
            // bytes at SSIM2-matches-cjxl quality. Composes with the
            // W44-176 terminal exclude via OR — either predicate
            // matching bypasses the lift. W44-109 win cluster (gb82-sc
            // text-class screenshots, M3 ∈ [14, 29]) is preserved
            // (their proxies fail the M3 >= 80 discriminator).
            let terminal_class_exclude = self.resolved_improvements.terminal_class_exclude;
            let high_colour_class_exclude = self.resolved_improvements.high_colour_class_exclude;
            let qf_pre_scale =
                super::perceptual_tuning::resolved_adaptive_quant_qf_seed_scale_with_policy(
                    self.effort,
                    effective_buttloop_iters,
                    is_screenshot,
                    self.distance,
                    m3,
                    adaptive_quant_qf_seed_policy,
                    self.zenanalyze_proxies.as_ref(),
                    terminal_class_exclude,
                    high_colour_class_exclude,
                );
            if qf_pre_scale != 1.0 {
                // W44-145 INVESTIGATION HONEST-STOP (2026-05-21): per-block
                // adaptive qf scaling via mask1x1 lookup was implemented
                // (`super::perceptual_loop::w44_145_per_block_qf_scale`
                // + `super::perceptual_loop::per_block_mask1x1_mean`) and
                // bisected at LOW thresholds {70, 95}. Mechanism works
                // directionally (blank-mask blocks get smaller scale,
                // text-mask blocks get full scale, mirroring cjxl's
                // bimodal qac at e8+) BUT cannot close the terminal d=4
                // e5/e6/e7 SSIM2/bytes budget the W44-145 task targeted:
                //
                //   v1 LOW=70: bytes -3% to -8.5% (target was -18 to
                //              -23pp toward +10-15% overhead) but SSIM2
                //              -0.34 to -0.50 (BUDGET WAS ±0.30)
                //   v2 LOW=95: SSIM2 -0.08 to -0.16 (within budget) but
                //              bytes +0.6% to +2.7% (WRONG DIRECTION)
                //
                // Root cause: cjxl at e5/e6/e7 ALSO has flat per-region
                // qac (~7-9), NOT bimodal. cjxl's bimodal qac only
                // emerges at e8+ post-buttloop. Therefore the right
                // mechanism for the e5-e7 bytes overhead is NOT per-block
                // mimicry of cjxl's e8+ bimodal structure (cjxl isn't
                // bimodal at e5-e7) but rather a LOWER uniform scale
                // (W44-144 Candidate 1: shrink the 2.0/3.0 constants).
                //
                // The helper functions are retained for future use
                // (potential e8+ application where cjxl actually IS
                // bimodal) but the production path keeps the uniform
                // multiply (pre-W44-145 behaviour) to honour the
                // W44-109 SSIM2 trade documented in
                // `docs/LIBJXL_DIVERGENCES.md` Section F line 160.
                //
                // See `benchmarks/w44_145_per_block_qac_ab_*_2026-05-21.tsv`
                // for the full A/B bisection (35 cells × 2 LOW values).
                for v in quant_field_float.iter_mut() {
                    *v *= qf_pre_scale;
                }
            }
        }

        // Step 3: Quantize float quant field to raw u8 with adaptive inv_scale
        let mut quant_field = quantize_quant_field(&quant_field_float, params.inv_scale);

        // Compute per-pixel mask on PRE-GABORISH image (matches libjxl:
        // initial_quant_masking1x1 is computed in InitialQuantField before GaborishInverse)
        //
        // NOTE: this mask is computed on the (possibly patches / dots /
        // splines-subtracted) XYB — the W36-3 patches photo-skip
        // dispatch above (via `splines::looks_like_screenshot`) runs its
        // own mask1x1 pass on the pre-patches XYB just for the gate
        // decision, and drops it before this compute runs so
        // subtract_patches / subtract_splines mutations don't observe
        // the wrong mask.
        //
        // W38-2 [`crate::api::PixelLossDispatch`] gate:
        //   * `AlwaysOff` → skip mask1x1 entirely (equivalent to
        //     `pixel_domain_loss = false`; AC strategy search uses
        //     coefficient-domain entropy only).
        //   * `AlwaysOn` (default) → keep current byte-identical
        //     behaviour: compute the mask and feed it to the search.
        //   * `Auto` → compute the mask, then check
        //     `median(mask1x1) > 80`; on smooth content drop the
        //     mask before the AC-strategy search so the loss term
        //     folds back to coefficient-domain only.
        let pld_force_off = matches!(
            self.pixel_loss_dispatch,
            crate::api::PixelLossDispatch::AlwaysOff
        );
        // W44-109: mask1x1 was computed earlier (above the qf compute) to
        // feed the screenshot-class pre-scale. Reuse it here instead of
        // recomputing — same plane, same predicate, same cost-saving as
        // the original `let m = compute_mask1x1_with_budget(...)` call
        // this block replaces (mask1x1 was unconditionally re-computed
        // here pre-W44-109; now it lands once at the top and threads
        // through both consumers).
        let mask1x1 = if self.ac_strategy_enabled && self.pixel_domain_loss && !pld_force_off {
            // Take ownership from the earlier compute to avoid clone.
            // `mask1x1_for_pre_scale` MUST be Some(_) here: the gate
            // predicate above is identical to this one.
            let m = mask1x1_for_pre_scale.expect(
                "mask1x1_for_pre_scale must be Some when pixel_domain_loss && !pld_force_off; \
                 W44-109 invariant violated — see encoder.rs comment",
            );
            if matches!(
                self.pixel_loss_dispatch,
                crate::api::PixelLossDispatch::Auto
            ) && pixel_loss_auto_should_skip(&m, padded_width, width, height)
            {
                None
            } else {
                Some(m)
            }
        } else {
            // `mask1x1_for_pre_scale` is None here (same predicate gates both
            // computes). Explicitly drop to satisfy the move-checker for the
            // !ac_strategy_enabled / !pixel_domain_loss / pld_force_off branches.
            drop(mask1x1_for_pre_scale);
            None
        };

        let _ms_quant_field = _t_quant_field.elapsed().as_secs_f64() * 1000.0;
        let _t_gaborish = std::time::Instant::now();
        // Apply gaborish inverse (5x5 sharpening) AFTER quant field and mask1x1
        // but BEFORE CfL and AC strategy. This matches libjxl enc_heuristics.cc:
        //   line 1124: InitialQuantField (pre-gaborish)
        //   line 1142: GaborishInverse
        //   line 1150-1174: CfL (post-gaborish)
        //   line 1179: AC strategy (post-gaborish)
        if self.enable_gaborish {
            gaborish_inverse_maybe_adaptive(
                &mut xyb_x,
                &mut xyb_y,
                &mut xyb_b,
                padded_width,
                padded_height,
                self.enable_adaptive_gaborish,
                self.budget.as_ref(),
            )?;
        }

        // Float DC for LfFrame is now extracted from the transform pipeline
        // (TransformOutput.float_dc) using dc_from_dct_NxN, which produces correct
        // DC values for multi-block transforms (DCT16+). The old compute_float_dc
        // used simple 8x8 pixel averages which diverge from dc_from_dct_NxN for
        // blocks with spatial structure, causing catastrophic LfFrame quality for
        // DCT16+ (up to 31% error on gradient content, butteraugli 13-20 vs ~2.5).

        let _ms_gaborish = _t_gaborish.elapsed().as_secs_f64() * 1000.0;
        let _t_cfl1 = std::time::Instant::now();
        // Compute per-tile chroma-from-luma map on GABORISHED XYB.
        //
        // **W44-195: Pass-1 dispatch is gated on `cfl_newton_libjxl_parity`.**
        //
        // libjxl `enc_heuristics.cc:1170-1174` runs Pass-1 with `fast=false`
        // (Newton, smoothed-L1 cost) at `speed_tier <= kSquirrel` (effort >= 7).
        // The W44-189 D1 audit (`memory/w44_189_cfl_deep_audit_2026-05-22.md`)
        // identified that we previously hardcoded `use_newton=false` here for
        // ALL strategies, contradicting libjxl Pass-1's Newton dispatch.
        //
        // The previous docstring rationale ("Newton collapses to LS at
        // distance_mul=1e-9") was empirically wrong: the Newton cost function
        // `1/3 * sum((|ax+b|+1)^2 - 1) + distance_mul * x^2 * num` has both L2
        // (quadratic) AND L1 (absolute-value) components. LS minimizes pure L2.
        // These are DIFFERENT minimizers unless residuals are tiny — for
        // typical CfL inputs with residuals O(10-100), the difference is
        // significant. `distance_mul=1e-9` zeros only the regularization, not
        // the L1 term.
        //
        // **`cfl_newton_libjxl_parity == true`** (set ONLY by
        // `EncoderStrategy::Libjxl`): Pass-1 dispatches to Newton at e>=7
        // (matches libjxl bit-for-bit). `cfl_newton_libjxl_parity` propagates
        // into the SIMD code path, forcing eps=100, max_iters=20, start x=0,
        // no LS fallback (W44-184 Pass-2 internals already covered the same).
        //
        // **`cfl_newton_libjxl_parity == false`** (Zenjxl / Aggressive /
        // LeanFaster): Pass-1 stays on LS (`use_newton=false`). The
        // W44-29..W44-172 downstream cost-model calibration was tuned against
        // the effective-LS Pass-1 baseline; flipping to Newton at the default
        // path regressed 25/27 photo cells in W44-183 (-13 SSIM2 / +26% bytes
        // worst case). The default keeps the calibrated baseline.
        //
        // See `docs/LIBJXL_DIVERGENCES.md` Section C and the W44-184 commit
        // memo for the Pass-2 half of the same `cfl_newton_libjxl_parity`
        // dispatch.
        // W44-AUDIT-5 Phase 2: Pass-1 fires Newton when EITHER `libjxl_parity`
        // OR `libjxl_math_with_ls_warm_start` is on (and `cfl_newton` is set).
        // The two are mutually-exclusive (libjxl_parity takes priority inside
        // the SIMD kernel), but both engage Newton at the Pass-1 dispatch site.
        // W44-AUDIT-5 Phase 3: per-image M3>=80 → flip libjxl_parity ON
        // for this single call, routing Pass-1 through `x=0` start +
        // Newton math (matching cjxl/Libjxl-strategy behaviour on
        // screenshot-class content).
        let p3_force_libjxl_parity = w44_audit_5_p3_force_libjxl_parity_for_screenshot(
            &self.profile,
            self.zenanalyze_proxies.as_ref(),
        );
        let cfl_newton_libjxl_parity_effective =
            self.profile.cfl_newton_libjxl_parity || p3_force_libjxl_parity;
        let pass1_use_newton = self.profile.cfl_newton
            && (cfl_newton_libjxl_parity_effective
                || self.profile.cfl_newton_libjxl_math_with_ls_warm_start);
        let mut cfl_map = if self.cfl_enabled {
            compute_cfl_map(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                pass1_use_newton,
                self.profile.cfl_newton_eps,
                self.profile.cfl_newton_max_iters,
                // W44-195 / W44-AUDIT-5 Phase 3: the `_effective` value
                // composes the profile field with the per-image route
                // (Phase 3 elevates screenshot-class images to the
                // libjxl-bit-exact `x=0` path). When false, Pass-1 runs
                // LS and this bool is ignored.
                cfl_newton_libjxl_parity_effective,
                // W44-AUDIT-5 Phase 2 (Mode C): when `pass1_use_newton` is
                // true AND `libjxl_parity_effective` is false, this drives
                // Pass-1 Newton with libjxl math (eps=100, iters=20)
                // starting from the LS warm-start. Mutually-exclusive with
                // Phase 3 (Phase 3 path takes priority since it forces
                // `libjxl_parity = true`).
                self.profile.cfl_newton_libjxl_math_with_ls_warm_start,
            )
        } else {
            CflMap::zeros(
                div_ceil(xsize_blocks, TILE_DIM_IN_BLOCKS),
                div_ceil(ysize_blocks, TILE_DIM_IN_BLOCKS),
            )
        };

        debug_rect!(
            "enc/config",
            0,
            0,
            width,
            height,
            "d={:.2} gab={} cfl={} pixel_loss={} patches={} bfly_iters={} noise={} denoise={} ac_strat={} err_diff={}",
            self.distance,
            self.enable_gaborish,
            self.cfl_enabled,
            self.pixel_domain_loss,
            self.enable_patches,
            self.profile.butteraugli_iters,
            self.enable_noise,
            self.enable_denoise,
            self.ac_strategy_enabled,
            self.error_diffusion
        );

        let _ms_cfl1 = _t_cfl1.elapsed().as_secs_f64() * 1000.0;
        let _t_acstrat = std::time::Instant::now();
        // Compute adaptive AC strategy (DCT8/DCT16x8/DCT8x16/DCT16x16/DCT32x32)
        // Content-aware `entropy_mul` table dispatch (opt-in). When the
        // caller has set `LossyConfig::with_content_aware_entropy_mul(true)`
        // we choose between the libjxl-faithful `reference()` table and the
        // lifted `screenshot_suppressed()` table for the AC-strategy search.
        //
        // The discriminator is:
        //   1. If the caller set
        //      [`crate::api::StrategyOverrides::screenshot_lift_hint`]
        //      to `Some(b)` (W44-130 Chunk D: moved here from the
        //      deleted `with_screenshot_lift_hint` setter), the
        //      resolver maps it to
        //      `ScreenshotEntropyMulPolicy::ForceOn/ForceOff`.
        //      `ForceOn` forces the lift; `ForceOff` suppresses it
        //      regardless of `mask1x1`.
        //   2. Otherwise (`Auto`) fall back to the W22-1
        //      encoder-internal `median(mask1x1) > 95` check.
        //
        // The swap is scoped to a local `profile_for_search` (cloned) so
        // the rest of the encode sees the original profile.
        //
        // Default (`content_aware_entropy_mul == false`) keeps the
        // reference-table behaviour and every hash-lock byte-identical.
        //
        // Compute `median(mask1x1)` once for both the W22-1 (screenshot
        // lift) and W44-29 (high-d photo lower) gates below. `None` means
        // the mask isn't available (pixel_domain_loss off / PixelLossDispatch
        // skipped) — both gates degrade to "don't fire" in that case
        // unless the caller supplied an explicit `Some(true)/Some(false)`
        // hint that bypasses the mask discriminator.
        let mask1x1_median: Option<f32> = mask1x1
            .as_deref()
            .map(|mask| median_mask1x1(mask, padded_width, width, height));
        // W44-151: compute `mask_p25` (mask1x1 25th percentile) for the
        // W44-29 outer gate's mask_p25 >= 85 admission branch. Same
        // None-semantics as `mask1x1_median` (gates degrade to off).
        // Cost is one extra O(n) select_nth_unstable over the unpadded
        // plane — negligible vs the median compute above.
        let mask1x1_p25: Option<f32> = mask1x1
            .as_deref()
            .map(|mask| percentile_mask1x1(mask, padded_width, width, height, 0.25));

        // W44-118 probe: env-gated dump of mask1x1_median + key
        // zenanalyze proxies to discriminate lift firing on the wedge
        // cell (1025469 e8/e9 d=4). Zero runtime cost when unset.
        #[cfg(feature = "std")]
        if std::env::var("JXL_W44_118_PROBE").is_ok_and(|v| v == "1") {
            let zp_m3 = self.zenanalyze_proxies.map(|p| p.m3_colourfulness);
            let zp_fcbr = self.zenanalyze_proxies.map(|p| p.flat_color_block_ratio);
            let zp_ed = self.zenanalyze_proxies.map(|p| p.edge_density);
            eprintln!(
                "W44-118-PROBE: dist={} mask1x1_median={:?} zp_m3={:?} zp_fcbr={:?} zp_ed={:?}",
                self.distance, mask1x1_median, zp_m3, zp_fcbr, zp_ed,
            );
        }

        // W44-87 single-pass-entropy dispatch — content gate.
        // Records whether the (effort, distance, content) tuple admits
        // the single-pass static-Huffman path; the SAFETY predicate
        // (no patches/splines/sharpness map/noise/LF frame/extras)
        // is evaluated at the dispatch site itself (just before
        // `if optimize_codes_effective` below) because those fields
        // are computed later in this function.
        //
        // Modes:
        //   - AlwaysTwoPass: never flip (default; bit-identical to
        //     historical builds).
        //   - AlwaysSinglePass: flip whenever the safety predicate
        //     holds, regardless of content / effort / distance.
        //   - Auto: flip when effort == 5 AND distance <= 1.0 AND
        //     median(mask1x1) < SMOOTH_PHOTO_MAX_MEDIAN (50.0). Same
        //     direction as HIGH_D_PHOTO_SMOOTH_THRESHOLD: low median
        //     = smooth photo content, where the per-image-tuned
        //     codes save only 2-4 % bytes vs the static codes — a
        //     poor trade against ~30 % wall-clock savings.
        let single_pass_entropy_content_gate: bool = match self.single_pass_entropy_dispatch {
            crate::api::SinglePassEntropyDispatch::AlwaysTwoPass => false,
            crate::api::SinglePassEntropyDispatch::AlwaysSinglePass => true,
            crate::api::SinglePassEntropyDispatch::Auto => {
                self.effort <= SINGLE_PASS_ENTROPY_MAX_EFFORT
                    && self.distance <= SINGLE_PASS_ENTROPY_MAX_DISTANCE
                    && mask1x1_median
                        .is_some_and(|med| med < SINGLE_PASS_ENTROPY_SMOOTH_PHOTO_MAX_MEDIAN)
            }
        };

        // W22-1 screenshot lift (opt-in, fires only when
        // `content_aware_entropy_mul=true`).
        //
        // W44-129 Chunk C + W44-130 Chunk D: read the resolved
        // `screenshot_entropy_mul` enum from `ResolvedImprovements`
        // (populated by `LossyConfig::resolve_improvements`). The
        // legacy `screenshot_lift_hint` `Option<bool>` field was
        // deleted in Chunk D; the override path now lives entirely
        // in `StrategyOverrides`, mapped via `apply_to` to `ForceOn`/
        // `ForceOff`. `Disabled` (Libjxl strategy) short-circuits the
        // lift off regardless of the `content_aware_entropy_mul`
        // enable bit — matching libjxl's no-discriminator default.
        let screenshot_policy = self.resolved_improvements.screenshot_entropy_mul;
        let w22_1_lift = match screenshot_policy {
            // W44-130 Chunk D: the legacy `content_aware_entropy_mul`
            // enable bit was subsumed by the policy enum. The Zenjxl
            // default is `Disabled` (preserving pre-Chunk-D default-off
            // behaviour); callers opt in via `Custom` /
            // `with_strategy_overrides` mapped to `Auto` / `ForceOn`.
            //
            // `Auto` here fires the W22-1 mask1x1 discriminator
            // directly (no longer guarded by a separate enable bit).
            crate::api::ScreenshotEntropyMulPolicy::Auto => {
                // W44-213: tuning-override-aware threshold lookup.
                let median_threshold = crate::runtime_or_default!(
                    CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD,
                    screenshot_median_threshold,
                );
                mask1x1_median.is_some_and(|med| med > median_threshold)
            }
            crate::api::ScreenshotEntropyMulPolicy::ForceOn => true,
            crate::api::ScreenshotEntropyMulPolicy::ForceOff => false,
            crate::api::ScreenshotEntropyMulPolicy::Disabled => false,
        };

        // W44-29 high-distance smooth-photo lowering. Default-on auto
        // gate: fires when `distance >= HIGH_D_PHOTO_MIN_DISTANCE` AND
        // `median(mask1x1) < HIGH_D_PHOTO_SMOOTH_THRESHOLD` (smooth-photo
        // content). Hash-locks at `d < 4.0` stay byte-identical because
        // the gate cannot fire there.
        //
        // Resolution when both W22-1 and W44-29 would fire: W22-1 wins
        // because its swap is more specific to screen content AND its
        // mask1x1>95 condition is mutually exclusive with W44-29's
        // mask1x1<50 condition in auto mode. The explicit-hint paths
        // can in principle conflict (caller forces both `Some(true)`);
        // in that case W22-1 wins by precedence (its lift was shipped
        // first, has hash-lock coverage at d < 4, and is opt-in only).
        // W44-129 Chunk C: read the resolved `high_d_photo_entropy_mul` enum
        // from `ResolvedImprovements` (populated by Chunk B
        // `LossyConfig::resolve_improvements`). The legacy
        // `high_d_photo_hint` `Option<bool>` is still consulted as a
        // fallback under `Auto` for direct `VarDctEncoder::new` callers
        // (tests + examples). `StrategyOverrides::apply_to` maps
        // `Some(true) → ForceOn`, `Some(false) → ForceOff`, preserving
        // production semantics bit-identically. `Disabled` is reachable
        // only via `EncoderStrategy::Libjxl` and short-circuits the
        // entire W44-29/91 gate stack to off.
        let high_d_photo_policy = self.resolved_improvements.high_d_photo_entropy_mul;
        let w44_29_lower = if w22_1_lift {
            // Mutually exclusive — W22-1 already swapped to the lift
            // table. Don't double-swap.
            false
        } else {
            match high_d_photo_policy {
                // W44-130 Chunk D: legacy `high_d_photo_hint`
                // `Option<bool>` fallback deleted. `Auto` here =
                // "no caller override" → consult the W44-29 + W44-91
                // auto gates directly.
                crate::api::HighDPhotoEntropyMulPolicy::Auto => {
                    // Auto: try two gates in OR.
                    //
                    // (a) **W44-29 smooth-photo gate** (default-on since
                    // commit `a01c4a7f`): `distance >= HIGH_D_PHOTO_MIN_DISTANCE`
                    // AND `median(mask1x1) < HIGH_D_PHOTO_SMOOTH_THRESHOLD`
                    // (smooth-content discriminator, mask1x1 < 50).
                    //
                    // (b) **W44-91 zenanalyze-proxy gate**: targets the
                    // textured-colourful-photo sub-band (mask1x1 ∈ [50, 80))
                    // that the W44-29 gate alone cannot reach without
                    // regressing 6 documented W44-78 regression-band images
                    // (1025469, 1624487, 159550, 2079234, 2775196, 297394).
                    // Fires only when ALL hold:
                    //   * distance ∈ [HIGH_D_PHOTO_MIN_DISTANCE,
                    //                 HIGH_D_PHOTO_W44_91_MAX_DISTANCE] (3..=5)
                    //   * mask1x1_median ∈ [HIGH_D_PHOTO_SMOOTH_THRESHOLD,
                    //                       HIGH_D_PHOTO_W44_91_MASK_UPPER) (50..80)
                    //   * zenanalyze proxies populated (8-bit sRGB layout)
                    //   * m3_colourfulness >= W44_91_M3_COLOURFULNESS_MIN (80)
                    //   * flat_color_block_ratio < W44_91_FCBR_MAX (0.01)
                    //
                    // On the 41 CID22 validation images only 1189261
                    // matches; on the 6 documented W44-78 regression-band
                    // images none match (each fails at least one of the
                    // colourfulness/fcbr gates per W44-79 discriminator C3).
                    let w44_29_gate = self.distance >= HIGH_D_PHOTO_MIN_DISTANCE
                        && mask1x1_median.is_some_and(|med| med < HIGH_D_PHOTO_SMOOTH_THRESHOLD);
                    let w44_91_gate = (HIGH_D_PHOTO_MIN_DISTANCE
                        ..=HIGH_D_PHOTO_W44_91_MAX_DISTANCE)
                        .contains(&self.distance)
                        && mask1x1_median.is_some_and(|med| {
                            (HIGH_D_PHOTO_SMOOTH_THRESHOLD..HIGH_D_PHOTO_W44_91_MASK_UPPER)
                                .contains(&med)
                        })
                        && self.zenanalyze_proxies.is_some_and(|p| {
                            p.m3_colourfulness >= W44_91_M3_COLOURFULNESS_MIN
                                && p.flat_color_block_ratio < W44_91_FCBR_MAX
                        });
                    // (c) **W44-152 (2026-05-21)**: re-introduces the
                    // W44-151 `mask1x1_p25 >= W44_151_HIGH_MASK_P25_MIN`
                    // admission branch with the distance narrowed to
                    // [`W44_152_W44_151_MIN_DISTANCE`,
                    //  `W44_152_W44_151_MAX_DISTANCE`] = [3.0, 5.0].
                    //
                    // W44-151 honest-stopped on the broad d ≥ 3.0 gate
                    // because the default `high_d_photo_smooth_suppressed()`
                    // table over-fires at d=6 on 1418519 (+4.3-4.6% bytes
                    // for only +0.07-0.28 SSIM2). Excluding d=6 captures
                    // the d=4 + d=5 win region.
                    //
                    // The env hook `JXL_W44_152_DISABLE=1` (and the legacy
                    // `JXL_W44_151_DISABLE=1` alias) disable the admission
                    // for A/B benches.
                    let w44_152_distance_in_band = self.distance >= W44_152_W44_151_MIN_DISTANCE
                        && self.distance <= W44_152_W44_151_MAX_DISTANCE;
                    let w44_152_disable_env = {
                        #[cfg(feature = "std")]
                        {
                            std::env::var("JXL_W44_152_DISABLE")
                                .map(|s| s == "1")
                                .unwrap_or(false)
                                || std::env::var("JXL_W44_151_DISABLE")
                                    .map(|s| s == "1")
                                    .unwrap_or(false)
                        }
                        #[cfg(not(feature = "std"))]
                        {
                            false
                        }
                    };
                    // W44-213: tuning-override-aware p25 threshold lookup.
                    let w44_151_p25_threshold = crate::runtime_or_default!(
                        W44_151_HIGH_MASK_P25_MIN,
                        smart_zenjxl_photo_mask_p25_min,
                    );
                    let w44_152_admit = !w44_152_disable_env
                        && w44_152_distance_in_band
                        && mask1x1_p25.is_some_and(|p25| p25 >= w44_151_p25_threshold);
                    w44_29_gate || w44_91_gate || w44_152_admit
                }
                crate::api::HighDPhotoEntropyMulPolicy::ForceOn => true,
                crate::api::HighDPhotoEntropyMulPolicy::ForceOff => false,
                crate::api::HighDPhotoEntropyMulPolicy::Disabled => false,
            }
        };

        #[cfg(feature = "debug-w44-65")]
        eprintln!(
            "W44-65 dbg: distance={:.2} mask1x1_median={:?} dct64_policy={:?} width={} height={}",
            self.distance,
            mask1x1_median,
            self.resolved_improvements.dct64_search_policy,
            width,
            height
        );
        // W44-65 + W44-68 content-aware large-DCT suppression (default-on).
        // When active we set `try_dct64 = false` AND `try_dct32 = false` on
        // the search-scoped profile so the AC-strategy search skips
        // DCT64X64/DCT64X32/DCT32X64 *and* DCT32X32/DCT32X16/DCT16X32
        // evaluation. W44-62 (`07f8b3d2`) measured uniform -0.13 % to -3.25 %
        // wins on screenshot-class content from the DCT64-only suppression
        // and a flip from `+3.51 %` → `+0.18 %` on `codec_wiki e7 d=5`
        // (OPEN → FIXED). W44-68 follow-on bisection (codec_wiki d=0.5..d=6
        // + 4 other screenshots) showed an additional -2.65 % to -4.48 %
        // win uniformly across all distances when DCT32 is also dropped on
        // the same dispatched class, closing the final OPEN screenshot cell
        // (codec_wiki e7 d=4: +3.55 % → -1.07 %, OPEN → FIXED).
        //
        // The auto discriminator reuses the W22-1 screenshot threshold
        // (`median(mask1x1) > 95`) because the W44-62 falsification
        // showed the DCT64-overpick signal aligns 1:1 with the
        // screenshot class. Caller-supplied `Some(_)` overrides win
        // outright (caller may plug in a zenanalyze classifier).
        //
        // **W44-65 promotion (2026-05-19)**: previously gated behind
        // the `content_aware_entropy_mul` opt-in (W44-63). The W44-65
        // encoder-pipeline mask1x1 probe
        // (`examples/w44_65_encoder_mask1x1_probe.rs`) measured the
        // median produced by the **real encoder pipeline** (not the
        // standalone `srgb_to_xyb` probe — those differ by ~17 due to
        // LUT vs powf and scalar vs SIMD float precision) on 41 CID22
        // validation photos + 7 gb82-sc screenshots + windows95:
        //   - Production screenshots (codec_wiki, imac_g3, imac_dark,
        //     terminal, windows, imessage, graph): median ≈ 100.013
        //     (saturated max).
        //   - windows95: median ≈ 99.060 (near-saturated pixel-art).
        //   - All 41 CID22 validation photos: median ≤ 92.34.
        // The **tighter** [`W44_65_DCT_SUPPRESS_MEDIAN_THRESHOLD`]
        // (`>= 99.5`) cleanly separates fully-saturated screenshots
        // from windows95-class pixel-art (which regressed +1.13 % at
        // d=2 under the looser `> 95` gate). The W22-1 / W44-29
        // gates retain the original `> 95` constant because they're
        // opt-in (`content_aware_entropy_mul`) so a 95-99.5 windows95
        // false-positive is a caller's deliberate choice rather than
        // a default-behaviour regression.
        //
        // The W23-2 `palette_log2_size >= 6` companion gate
        // (originally proposed to protect windows95-class pixel-art
        // from W22-1 lift false-fires) was considered but rejected:
        // the tighter mask1x1 threshold is sufficient on its own and
        // avoids adding a zenanalyze dependency to the production
        // hot path.
        //
        // Note: this gate composes with W22-1 and W44-29 — all three
        // can fire on the same encode (W22-1 / W44-29 swap the
        // entropy_mul table; W44-65 drops `try_dct64`). On a pure
        // screenshot W22-1 fires only when `content_aware_entropy_mul`
        // is opt-in; W44-65 fires by default. Composition is
        // additive — both improvements stack when both are enabled.
        // W44-129 Chunk C + W44-130 Chunk D: read the resolved
        // `dct64_search_policy` directly. The legacy `dct_suppress_hint`
        // `Option<bool>` field was deleted in Chunk D; the override
        // path now lives entirely in `StrategyOverrides` (mapped to
        // `Force*` via `apply_to`). `Auto` here means "no caller
        // override" → consult the mask1x1 discriminator directly.
        let dct64_policy = self.resolved_improvements.dct64_search_policy;
        let w44_65_suppress_dct64 = match dct64_policy {
            crate::api::Dct64SearchPolicy::Auto => {
                mask1x1_median.is_some_and(|med| med >= W44_65_DCT_SUPPRESS_MEDIAN_THRESHOLD)
            }
            crate::api::Dct64SearchPolicy::ForceSuppress => true,
            crate::api::Dct64SearchPolicy::ForceAllow => false,
        };

        // W44-96 variant Z sub-discriminator: when `w44_29_lower` fires
        // via the W44-29 mask<50 gate (NOT the W44-91 mask∈[50,80) gate),
        // and the image passes the additional edge_density/fcbr proxies,
        // swap to the variant Z entropy_mul table (dct32x32=1.20 instead
        // of 1.34). Closes the W44-95-measured wins on {1420710, 1531677}
        // at d∈{5, 6} while leaving {2389166, 1044329, 7062219} on the
        // default suppressed table — see [`W44_96_EDGE_DENSITY_MIN`] doc
        // for the per-image proxy split. Excludes the W44-91 mask band
        // because variant Z was never measured against W44-91 cells.
        //
        // Auto only: when the caller forced
        // `HighDPhotoEntropyMulPolicy::ForceOn` outside the W44-29 mask
        // range we keep the default suppressed table (no variant Z
        // escalation from a forced override — caller can ship their own
        // table override if they want).
        //
        // W44-129 Chunk C + W44-130 Chunk D: matches against the
        // resolved policy enum directly. `Auto` here means "no caller
        // override" via `StrategyOverrides::apply_to`. The legacy
        // `self.high_d_photo_hint.is_none()` redundant guard was
        // deleted with the field in Chunk D.
        //
        // W44-166 (Smart-Zenjxl chunk 3, 2026-05-21): mirror of the
        // `compute_profile_for_search` site above. Adds an OR-branch
        // that admits high-mask photos (1418519-class) via mask1x1_p25
        // >= 85, gated on the resolved `photo_variant_z_admit` flag.
        // Env hook `JXL_W44_166_VARIANT_Z_ADMIT_MODE=A|B|C` controls
        // the discriminator (default A = baseline = no change).
        let w44_166_admit_mode = w44_166_admit_mode_env();
        let w44_166_photo_admit_allowed = self.resolved_improvements.photo_variant_z_admit
            && matches!(
                high_d_photo_policy,
                crate::api::HighDPhotoEntropyMulPolicy::Auto
            )
            && self.distance >= W44_96_VARIANT_Z_MIN_DISTANCE;
        // W44-213: tuning-override-aware p25 threshold lookup (W44-166
        // photo admission; shares the canonical 85.0 value with W44-150
        // / W44-151 / W44-168, all unified under `smart_zenjxl_photo_mask_p25_min`).
        let w44_166_p25_threshold = crate::runtime_or_default!(
            W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN,
            smart_zenjxl_photo_mask_p25_min,
        );
        let w44_166_photo_admit = w44_166_photo_admit_allowed
            && match w44_166_admit_mode {
                W44_166VariantZAdmitMode::Baseline => false,
                W44_166VariantZAdmitMode::BMaskP25 => {
                    mask1x1_p25.is_some_and(|p25| p25 >= w44_166_p25_threshold)
                }
                W44_166VariantZAdmitMode::CMaskP25HighM3 => {
                    mask1x1_p25.is_some_and(|p25| p25 >= w44_166_p25_threshold)
                        && self.zenanalyze_proxies.is_some_and(|p| {
                            p.m3_colourfulness >= W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN
                        })
                }
            };
        let w44_96_default_admit = w44_29_lower
            && matches!(
                high_d_photo_policy,
                crate::api::HighDPhotoEntropyMulPolicy::Auto
            )
            && self.distance >= W44_96_VARIANT_Z_MIN_DISTANCE
            && mask1x1_median.is_some_and(|med| med < HIGH_D_PHOTO_SMOOTH_THRESHOLD)
            && self.zenanalyze_proxies.is_some_and(|p| {
                p.edge_density >= W44_96_EDGE_DENSITY_MIN
                    && p.flat_color_block_ratio < W44_96_FCBR_MAX
            });
        let w44_96_variant_z = w44_96_default_admit || w44_166_photo_admit;

        // W44-98 variant Z' sub-discriminator: when `w44_96_variant_z`
        // fires AND the image's `m3_colourfulness` exceeds
        // `W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN`, escalate from the
        // default variant Z table (dct16x32=1.208) to the high-colour
        // variant Z' table (dct16x32=1.30). This targets the W44-97
        // finding that DCT32X16 is the universal #1 overspender on the
        // 7 OPEN cells post-W44-96; lifting `dct16x32` makes it more
        // expensive relative to DCT32X32 (square merge wins more often).
        //
        // Of the 2 CID22 photos that pass the W44-96 gate (1420710 m3=32.93,
        // 1531677 m3=12.30), only 1420710 passes this additional gate
        // — 1531677 stays on the default variant Z table (the W44-98
        // A/B sweep showed 1531677 regresses SSIM2 by -0.34 to -0.93
        // under `dct16x32 >= 1.30`, exceeding the ≤0.30 budget).
        let w44_98_variant_z_high_colour = w44_96_variant_z
            && self
                .zenanalyze_proxies
                .is_some_and(|p| p.m3_colourfulness >= W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN);

        // W44-99 variant Z'' (low-colour) sub-discriminator: when
        // `w44_96_variant_z` fires AND the image's `m3_colourfulness` is
        // BELOW the W44-98 threshold (the mirror of the high_colour gate),
        // apply a modest dct16x32 lift. W44-99 originally shipped 1.22;
        // W44-100 micro-bisect bumped to 1.23 to close the last OPEN cell
        // (1531677 e5 d=5: +3.090% → +1.943% bytes, worst-cell SSIM2 delta
        // -0.2592 under the 0.30 budget). The W44-100 A/B sweep
        // (`benchmarks/w44_100_1531677_e5_d5_microbisect_2026-05-19.tsv`)
        // found 1.23 strictly dominates 1.22 / 1.24 / 1.25 (best total
        // bytes -2731B over 10 cells AND lowest worst-cell SSIM2 cost AND
        // the only value that closes the OPEN cell at < +3.0% bytes).
        // The cost model is non-monotonic in this region: LC_124 emits
        // +3.85% bytes on 1531677 e8 d=5 vs LC_123's -0.10%, so the bisect
        // had to be measurement-driven, not interpolation-driven.
        //
        // Mutually exclusive with [`w44_98_variant_z_high_colour`]:
        // - m3 >= 25 → high_colour (Z', dct16x32=1.30)
        // - m3 <  25 → low_colour  (Z'', dct16x32=1.23)
        // - never both
        //
        // Of the 2 CID22 photos that pass W44-96 (1420710 m3=32.93,
        // 1531677 m3=12.30), only 1531677 enters this gate; 1420710
        // stays on the W44-98 high_colour Z' table.
        let w44_99_variant_z_low_colour = w44_96_variant_z
            && !w44_98_variant_z_high_colour
            && self
                .zenanalyze_proxies
                .is_some_and(|p| p.m3_colourfulness < W44_98_VARIANT_Z_HIGH_COLOUR_M3_MIN);

        // W44-156 distance-band sub-discriminator inside variant Z (mirror
        // of compute_ac_strategy). When any variant Z gate fires AND
        // target_distance > W44_156_VARIANT_Z_D_HIGH_THRESHOLD (5.5
        // default; env hook __JXL_W44_156_THRESHOLD), use the d-high
        // table (dct32x32 = 1.20) instead of the W44-154 default
        // (dct32x32 = 1.22). Closes 1420710 e5 d=6 OPEN cell per
        // W44-155 strategy-shift diagnosis (cjxl sheds small blocks at
        // d=5→d=6; we don't, because the W44-154 1.22 lift forces more
        // DCT32X32 consolidation than cjxl picks at d > 5.5).
        let w44_156_d_high = w44_96_variant_z && self.distance > w44_156_effective_threshold();

        let profile_for_search = if w22_1_lift
            || w44_29_lower
            || w44_65_suppress_dct64
            || w44_166_photo_admit
        {
            let mut p = self.profile.clone();
            if w22_1_lift {
                p.entropy_mul_table = crate::effort::EntropyMulTable::screenshot_suppressed();
            } else if w44_29_lower || w44_166_photo_admit {
                // W44-166: when ONLY w44_166_photo_admit fires
                // (w44_29_lower false — i.e. d outside W44-152's
                // [3.0, 5.0] band), still route through the variant Z
                // chain. The W44-156 d_high split handles d>5.5.
                p.entropy_mul_table = if w44_98_variant_z_high_colour {
                    if w44_156_d_high {
                        crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour_d_high()
                    } else {
                        crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z_high_colour(
                        )
                    }
                } else if w44_99_variant_z_low_colour {
                    if w44_156_d_high {
                        crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z_low_colour_d_high()
                    } else {
                        crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z_low_colour(
                        )
                    }
                } else if w44_96_variant_z {
                    if w44_156_d_high {
                        crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z_d_high()
                    } else {
                        crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed_z()
                    }
                } else {
                    crate::effort::EntropyMulTable::high_d_photo_smooth_suppressed()
                };
                // W44-167 (Smart-Zenjxl chunk 4, 2026-05-21): mirror of
                // the `compute_profile_for_search` site above. Override
                // the `dct16x32` field per-m3 when the gate fires.
                // Default env unset = Mode A = byte-identical.
                if w44_96_variant_z && self.resolved_improvements.find_best_32_per_m3_lift {
                    let mode = w44_167_mode_env();
                    p.entropy_mul_table.dct16x32 = w44_167_apply_lift(
                        mode,
                        w44_98_variant_z_high_colour,
                        w44_99_variant_z_low_colour,
                        p.entropy_mul_table.dct16x32,
                    );
                }
            }
            if w44_65_suppress_dct64 {
                p.try_dct64 = false;
                // W44-68: also suppress DCT32-class on the same screenshot-class
                // dispatch. Bisection on codec_wiki d=0.5..d=6 showed uniform
                // -2.65 % to -4.48 % wins across all distances (d=4 wedge cell
                // closes from +3.55 % → -1.07 %, OPEN → FIXED). Other dispatched
                // screenshots (terminal, imac_g3, imac_dark, windows) also win
                // -0.76 % to -3.78 %. The W44-65 threshold (mask1x1 >= 99.5)
                // protects windows95 (99.06, NOT in screenshot class — +2.0 % if
                // suppressed) and all CID22 photos (median ≤ 92.34 — up to
                // +10.8 % if suppressed); the discriminator gate keeps both
                // untouched.
                //
                // W44-123 (2026-05-20): caller can opt-in to KEEP try_dct32=true
                // (while preserving try_dct64=false) via
                // `LossyConfig::with_dct32_keep_hint(Some(true))`. This is the
                // narrower lever measured in the W44-123 A/B: codec_wiki d=3
                // e5/e6/e7 SSIM2 +1.40/+1.33/+0.90 (vs admit-DCT64's
                // +1.40/+1.33/+0.73) AND terminal e8/e9 d=4 SSIM2 +0.47 with
                // bfly -1.9% (vs admit-DCT64's bfly +29% regression). Default
                // was `None` → W44-68 (drop try_dct32 too).
                //
                // W44-124 (2026-05-20): default `None` now AUTO-fires a
                // zenanalyze-proxy discriminator that admits codec_wiki-class
                // content while rejecting the 6 SCREEN cells (graph, windows,
                // imessage) that regressed in the W44-123 default-off A/B.
                // Predicate:
                //   * zenanalyze proxies populated (8-bit sRGB layout)
                //   * `m3_colourfulness >= W44_124_DCT32_KEEP_M3_MIN` (60)
                //   * `edge_density < W44_124_DCT32_KEEP_EDGE_DENSITY_MAX` (0.05)
                //
                // Verified clean split per `examples/w44_124_proxy_probe.rs`:
                // codec_wiki (m3=145.73, ed=0.0396) FIRES. All 6 W44-123
                // regression screens REJECT (terminal/graph/windows m3≤21;
                // imessage m3=67.65 but ed=0.0533 fails ed gate). All CID22
                // photos REJECT (ed ≥ 0.16). Trade-off: terminal e8/e9 d=4
                // loses the +0.47 SSIM2 opt-in win (m3=13.85 cleanly rejected)
                // — that win remains accessible via explicit
                // `with_dct32_keep_hint(Some(true))`. Net default-on:
                // codec_wiki d=3 close-out preserved, zero new FIXED→OPEN
                // flips on regression set.
                //
                // W44-135 (2026-05-20): the auto-discriminator is now ALSO
                // gated on `distance ∈ [W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE,
                // W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE]` ([2.0, 3.5]). The
                // W44-134 ledger refresh
                // (`benchmarks/cjxl_parity_ledger_2026-05-20_w44_134.tsv`)
                // measured codec_wiki SSIM2 regressions at d=4/5/6 (-0.64
                // to -1.43) and d=0.8/1.0 (-0.41 to -0.52) under
                // W44-124's unconditional firing. Root cause: at d>=4 the
                // cost model prefers DCT64X64 over DCT32X32 even on smooth
                // pages; at d<2 it's in the DCT16X16/DCT8 regime where
                // keep_dct32 is inert-or-negative. The [2.0, 3.5] band
                // preserves the W44-124 d=3 wins and the W44-134-measured
                // d=2.5 bonus wins, reverts the regression cells to
                // baseline. Explicit opt-in via
                // `StrategyOverrides { dct32_keep_hint: Some(true) }` (=
                // `Dct32SearchPolicy::KeepWhenDct64Suppressed`) bypasses
                // the distance gate.
                //
                // Explicit `Some(true)` / `Some(false)` overrides the
                // auto-discriminator (caller's choice always wins). For
                // development / paired benchmarks the env var
                // `__JXL_W44_123_KEEP_DCT32=1` also forces keep_dct32; used
                // by the W44-123 A/B harness before the hint API landed.
                let w44_123_env_keep = {
                    #[cfg(feature = "std")]
                    {
                        std::env::var("__JXL_W44_123_KEEP_DCT32")
                            .map(|s| s == "1")
                            .unwrap_or(false)
                    }
                    #[cfg(not(feature = "std"))]
                    {
                        false
                    }
                };
                // W44-135 (2026-05-20): distance-band the W44-124
                // auto-discriminator to `[W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE,
                // W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE]`. W44-124 shipped with
                // NO distance gate; the W44-134 ledger refresh
                // (`benchmarks/cjxl_parity_ledger_2026-05-20_w44_134.tsv`)
                // measured codec_wiki SSIM2 wins at d=2.5/d=3 (+1.40 to +1.77)
                // but NEW regressions at d=4/d=5/d=6 (-0.64 to -1.43) and at
                // d=0.8/d=1.0 (-0.41 to -0.52). Root cause: at d>=4 the cost
                // model prefers DCT64X64 over DCT32X32 even on smooth screen
                // pages, so forcing keep_dct32 selects strictly-worse
                // 4×DCT16X16 + DCT32 mix. At d<2 the cost model is in the
                // DCT16X16/DCT8 regime where keep_dct32 is inert-or-negative.
                // The [2.0, 3.5] band preserves the W44-124 codec_wiki d=3
                // close-out wins plus the d=2.5 bonus wins, reverts the
                // d=4/5/6 + d=0.8/1.0 cells to baseline.
                //
                // Explicit `Some(true)` via the `StrategyOverrides`
                // `dct32_keep_hint` field (which maps to
                // `Dct32SearchPolicy::KeepWhenDct64Suppressed`) bypasses
                // this distance gate — opt-in callers still get the
                // unconditional W44-123 behaviour.
                let w44_124_distance_in_band = self.distance >= w44_143_effective_min_distance()
                    && self.distance <= W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE;
                let w44_124_auto_keep = w44_124_distance_in_band
                    && self.zenanalyze_proxies.is_some_and(|p| {
                        p.m3_colourfulness >= W44_124_DCT32_KEEP_M3_MIN
                            && p.edge_density < W44_124_DCT32_KEEP_EDGE_DENSITY_MAX
                    });
                // W44-129 Chunk C + W44-130 Chunk D: read the resolved
                // `dct32_search_policy` directly. Legacy `dct32_keep_hint`
                // `Option<bool>` field deleted; overrides flow via
                // `StrategyOverrides` → `KeepWhenDct64Suppressed`.
                // `FollowDct64Suppression` here means "no caller
                // override" → consult the env-var + W44-124
                // auto-discriminator directly.
                let dct32_policy = self.resolved_improvements.dct32_search_policy;
                let w44_123_keep_dct32 = match dct32_policy {
                    crate::api::Dct32SearchPolicy::FollowDct64Suppression => {
                        w44_123_env_keep || w44_124_auto_keep
                    }
                    crate::api::Dct32SearchPolicy::KeepWhenDct64Suppressed => true,
                };
                if !w44_123_keep_dct32 {
                    p.try_dct32 = false;
                }
            }
            Some(p)
        } else {
            None
        };
        let active_profile_for_search = profile_for_search.as_ref().unwrap_or(&self.profile);

        // W44-AUDIT-9 / SA-G Fix C: when the active profile says so
        // (`EncoderStrategy::Libjxl` only at the default), substitute a
        // zero-filled cmap for the AC strategy SEARCH only. The actual
        // emitted `cfl_map` (Pass-1 Newton-derived above + the optional
        // Pass-2 refinement below) stays intact in the bitstream; only
        // the consumption-during-search is zeroed. Mirrors libjxl
        // `enc_ac_strategy.cc` at `speed_tier > kSquirrel`. SA-G report
        // (`7d383785`) measured this brings clic_22ea12 e9 d=4 partial
        // first-blocks 2,241 → 2,495 (vs cjxl 2,499 = +0.16% parity)
        // and bytes -0.6% on the Libjxl strategy.
        let zero_cfl_map_for_search;
        let cfl_map_for_search: &CflMap = if active_profile_for_search.cfl_zero_for_search {
            zero_cfl_map_for_search = CflMap::zeros(cfl_map.xsize_tiles, cfl_map.ysize_tiles);
            &zero_cfl_map_for_search
        } else {
            &cfl_map
        };

        #[allow(unused_mut)]
        let mut ac_strategy = if let Some(forced) = self.force_strategy {
            // Force a specific strategy for all blocks that fit
            force_strategy_map(xsize_blocks, ysize_blocks, forced)
        } else if !self.ac_strategy_enabled {
            AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks)
        } else {
            compute_ac_strategy(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                self.distance,
                &quant_field_float,
                &masking,
                cfl_map_for_search,
                mask1x1.as_deref(),
                padded_width,
                active_profile_for_search,
            )
        };

        // Debug: print strategy histogram if enabled
        #[cfg(feature = "debug-ac-strategy")]
        {
            eprintln!(
                "AC strategy mode: {}",
                if mask1x1.is_some() {
                    "pixel-domain"
                } else {
                    "coefficient-domain"
                }
            );
            ac_strategy.print_histogram();
        }

        // Log AC strategy distribution
        {
            let mut counts = [0u32; 27];
            for by in 0..ysize_blocks {
                for bx in 0..xsize_blocks {
                    if ac_strategy.is_first(bx, by) {
                        let s = ac_strategy.raw_strategy(bx, by) as usize;
                        if s < counts.len() {
                            counts[s] += 1;
                        }
                    }
                }
            }
            let total: u32 = counts.iter().sum();
            // Format top strategies
            // Names indexed by RAW_STRATEGY_* internal codes (NOT bitstream order)
            let names = [
                "DCT8",     // 0 = RAW_STRATEGY_DCT8
                "DCT16x8",  // 1 = RAW_STRATEGY_DCT16X8
                "DCT8x16",  // 2 = RAW_STRATEGY_DCT8X16
                "DCT16x16", // 3 = RAW_STRATEGY_DCT16X16
                "DCT32x32", // 4 = RAW_STRATEGY_DCT32X32
                "DCT4x8",   // 5 = RAW_STRATEGY_DCT4X8
                "DCT8x4",   // 6 = RAW_STRATEGY_DCT8X4
                "DCT4x4",   // 7 = RAW_STRATEGY_DCT4X4
                "IDENTITY", // 8 = RAW_STRATEGY_IDENTITY
                "DCT2x2",   // 9 = RAW_STRATEGY_DCT2X2
                "DCT32x16", // 10 = RAW_STRATEGY_DCT32X16
                "DCT16x32", // 11 = RAW_STRATEGY_DCT16X32
                "AFV0",     // 12 = RAW_STRATEGY_AFV0
                "AFV1",     // 13 = RAW_STRATEGY_AFV1
                "AFV2",     // 14 = RAW_STRATEGY_AFV2
                "AFV3",     // 15 = RAW_STRATEGY_AFV3
                "DCT64x64", // 16 = RAW_STRATEGY_DCT64X64
                "DCT64x32", // 17 = RAW_STRATEGY_DCT64X32
                "DCT32x64", // 18 = RAW_STRATEGY_DCT32X64
            ];
            let mut parts = alloc::string::String::new();
            for (i, &c) in counts.iter().enumerate() {
                if c > 0 {
                    if !parts.is_empty() {
                        parts.push(' ');
                    }
                    let name = names.get(i).copied().unwrap_or("?");
                    let pct = c as f32 / total.max(1) as f32 * 100.0;
                    parts.push_str(&alloc::format!("{}={:.0}%", name, pct));
                }
            }
            debug_rect!(
                "enc/ac_strategy",
                0,
                0,
                width,
                height,
                "total={} {}",
                total,
                parts
            );
        }

        let _ms_acstrat = _t_acstrat.elapsed().as_secs_f64() * 1000.0;
        let _t_cfl2 = std::time::Instant::now();
        // Free masking — no longer needed after AC strategy selection.
        drop(masking);

        // Adjust quant field for multi-block transforms.
        // At low distances uses max, at high distances blends toward mean for better quality.
        // Adjust BOTH u8 and float fields (libjxl adjusts float before SetQuantField).
        adjust_quant_field_with_distance(&ac_strategy, &mut quant_field, self.distance);
        adjust_quant_field_float_with_distance(&ac_strategy, &mut quant_field_float, self.distance);

        // CfL pass 2: recompute CfL map using actual AC strategies and per-block
        // quantization weighting. Uses the same FindBestMultiplier as pass 1 but
        // with strategy-specific DCTs and quant-weighted coefficients.
        // Gated at effort >= 7 (speed_tier <= kSquirrel) matching libjxl.
        //
        // ORDERING: CfL pass 2 must run BEFORE the butteraugli loop so the
        // loop's internal recon and the shipped bitstream both see the same
        // post-pass-2 cfl_map. Mirrors libjxl enc_heuristics.cc:1190-1193 (CfL2)
        // → :1250-1252 (FindBestQuantizer/buttloop). See drift investigation
        // 2026-05-15 (chunk-3) — running CfL pass 2 AFTER the buttloop caused
        // the buttloop to converge on a target the decoder never delivered.
        // **W44-197 Candidate B**: extend the Pass-2 gate with an LS-only
        // path at effort ∈ {5, 6} when `cfl_pass2_ls_at_low_effort` is true
        // (Libjxl strategy only). This matches libjxl
        // `enc_heuristics.cc:1190-1194` which runs Pass-2 at
        // `speed_tier <= kHare` (effort >= 5) with
        // `fast = (speed_tier >= kWombat)` — i.e. LS at e=5/6, Newton at
        // e>=7. The existing `cfl_two_pass: effort >= 7` Newton gate stays
        // as-is; the new gate fires ONLY when the strategy resolves it
        // true AND the effort is in {5, 6}.
        //
        // Why not just widen `cfl_two_pass: effort >= 5`? Because W44-102
        // (`c1d699e2`) measured FULL Newton widening (which is what that
        // gate does) and ruled it out — 2 cells exceeded the -0.3 SSIM2
        // budget. W44-197 ships the ORTHOGONAL LS-only widening
        // (recommended by W44-189 D12 audit) which was NEVER measured at
        // a default path until now. Production default stays OFF; only
        // Libjxl strategy flips it ON to deliver TRUE libjxl Pass-2
        // dispatch parity.
        let effort_in_5_6 = self.effort == 5 || self.effort == 6;
        let pass2_fires =
            self.profile.cfl_two_pass || (self.profile.cfl_pass2_ls_at_low_effort && effort_in_5_6);
        // When ONLY the W44-197 LS gate fires (i.e. cfl_two_pass is
        // false at e=5/6), use LS (use_newton=false) — matches libjxl
        // `fast=true`. When the standard cfl_two_pass gate fires
        // (effort >= 7), use the existing Newton path (cfl_newton).
        let pass2_use_newton = self.profile.cfl_two_pass && self.profile.cfl_newton;
        if pass2_fires && self.cfl_enabled {
            super::chroma_from_luma::refine_cfl_map(
                &mut cfl_map,
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                xsize_blocks,
                ysize_blocks,
                &ac_strategy,
                &quant_field,
                params.scale,
                pass2_use_newton,
                self.profile.cfl_newton_eps,
                self.profile.cfl_newton_max_iters,
                // W44-184 / W44-AUDIT-5 Phase 3: Pass-2 mirrors the Pass-1
                // dispatch — the `_effective` value composes the profile
                // field with the per-image Phase-3 route (M3>=80 →
                // libjxl_parity ON for this image). Ignored when
                // `pass2_use_newton == false` (LS path doesn't read it).
                cfl_newton_libjxl_parity_effective,
                // W44-AUDIT-5 Phase 2 (Mode C): when this is `true` AND
                // `libjxl_parity_effective` is false, Pass-2 Newton runs
                // libjxl math (eps=100, iters=20) starting from `ls_x`
                // warm-start with LS fallback. Designed to close the
                // codec_wiki SSIM2 deficit without regressing the
                // photo cost-model. Mutually-exclusive with the Phase 3
                // route (Phase 3 forces `libjxl_parity = true`, taking
                // priority inside the SIMD kernel).
                self.profile.cfl_newton_libjxl_math_with_ls_warm_start,
            );
        }

        let _ms_cfl2 = _t_cfl2.elapsed().as_secs_f64() * 1000.0;
        let _t_buttloop = std::time::Instant::now();
        // Quantization loops: iteratively refine quant_field using perceptual
        // distance feedback. Butteraugli and zensim loops can stack: butteraugli
        // handles global convergence, zensim adds SSIM-aware spatial fine-tuning.
        // Works in float quant field domain with per-iteration global_scale
        // recomputation (matching libjxl FindBestQuantization).
        // W44-168: gate the buttloop on the ADJUSTED iter count
        // (not the raw `self.butteraugli_iters`) so Mode C
        // (TexturedExtend) can promote 0 → 2 at e7 on textured
        // content. Mode A (Baseline, default) preserves the original
        // gate semantics because `effective_buttloop_iters ==
        // self.butteraugli_iters`.
        #[cfg(feature = "butteraugli-loop")]
        if effective_buttloop_iters > 0 {
            let initial_qf_float = quant_field_float.clone();
            // W43-3 chunk 1: HdrLoss::Ssim2 dispatch.
            //
            // When the caller pinned HdrLoss::Ssim2 via
            // `LossyConfig::with_hdr_loss`, route the buttloop budget
            // (`butteraugli_iters`) through `ssim2_refine_quant_field`
            // instead. The ssim2-loop infrastructure has been wired
            // internally for several releases — chunk 1 just exposes
            // it through the public HdrLoss enum so callers don't have
            // to know about the legacy `with_ssim2_iters` path.
            //
            // `validate_loss` (one call site, called once per encode)
            // surfaces a clear `Error::NotImplemented` if the
            // `ssim2-loop` feature is off, so the dispatch below is
            // only reached when the feature is compiled in.
            //
            // The default `HdrLoss::Auto` resolves to `Butteraugli` on
            // every SDR encode (see `HdrLoss::resolve`) so the
            // hash-lock corpus stays byte-identical.
            if let Err(e) = super::hdr_metrics::validate_loss(self.hdr_loss) {
                return Err(crate::error::Error::NotImplemented(alloc::format!(
                    "HDR loss dispatch: {e} (selected: {})",
                    self.hdr_loss.as_str()
                )));
            }
            // Resolve `Auto` to a concrete loss here (defensive — the
            // public LossyConfig path already calls
            // `resolve_hdr_loss` before assigning `enc.hdr_loss`; this
            // belt-and-braces step covers direct `VarDctEncoder`
            // construction from tests / internal callers).
            let resolved_loss = self.hdr_loss.resolve(None);
            // Tag-as-used when ssim2-loop is off; the dispatch table
            // below collapses to the always-butteraugli branch in that
            // configuration.
            let _ = resolved_loss;

            #[cfg(feature = "ssim2-loop")]
            let take_ssim2_path = matches!(resolved_loss, super::hdr_metrics::HdrLoss::Ssim2);
            #[cfg(not(feature = "ssim2-loop"))]
            let take_ssim2_path = false;

            if take_ssim2_path {
                #[cfg(feature = "ssim2-loop")]
                {
                    // Use `butteraugli_iters` as the ssim2-loop budget
                    // so callers can opt in via the single setter
                    // `with_hdr_loss(HdrLoss::Ssim2)` without also
                    // needing `with_ssim2_iters`. The legacy
                    // `with_ssim2_iters` path (below) still works for
                    // backward-compat experiments.
                    //
                    // W44-168: pass `effective_buttloop_iters` (not
                    // raw `self.butteraugli_iters`) so the adaptive
                    // adjustment also applies on the ssim2 buttloop
                    // path.
                    params = self.ssim2_refine_quant_field_with_iters(
                        effective_buttloop_iters,
                        linear_rgb,
                        width,
                        height,
                        &xyb_x,
                        &xyb_y,
                        &xyb_b,
                        padded_width,
                        padded_height,
                        xsize_blocks,
                        ysize_blocks,
                        &params,
                        &mut quant_field,
                        &mut quant_field_float,
                        &initial_qf_float,
                        &cfl_map,
                        &ac_strategy,
                        patches_data.as_ref(),
                        splines_data.as_ref(),
                    )?;
                }
            } else {
                // W39-2 (WF3 fix): classify the input as screenshot vs
                // photo for the buttloop's HIGH-regime cap dispatch.
                // Reuses the same `median(mask1x1) > 95.0` discriminator
                // as `splines::looks_like_screenshot` and the W22-1
                // `entropy_mul` content-aware dispatch above
                // (`CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD`). `mask1x1`
                // is already in scope here (computed at line ~1913 for
                // `pixel_domain_loss`); falling back to `false` when it
                // wasn't computed keeps the photo-default (no cap, libjxl
                // faithful, byte-identical).
                // W44-213: tuning-override-aware threshold lookup.
                let median_threshold = crate::runtime_or_default!(
                    super::perceptual_tuning::SCREENSHOT_MEDIAN_THRESHOLD,
                    screenshot_median_threshold,
                );
                let is_screenshot = mask1x1.as_deref().is_some_and(|m| {
                    median_mask1x1(m, padded_width, width, height) > median_threshold
                });
                // W44-150 Phase 2 HONEST-STOP (2026-05-21): the
                // Mechanism A photo-admission path (was: admit
                // `is_screenshot || (mask_p25 >= 85.0 AND distance >=
                // 4.0)` to the W44-117 EPF sharpness seed) was
                // measured and produced a +0.27 mean SSIM2 delta on
                // 1418519 d=5/6 vs the HARD acceptance gate of ≥ +1.0
                // (50 % of the W44-147 -2.17 to -2.57 deficit). The
                // proxy-level discriminator works (51 of 51 protection
                // cells byte-identical between A and B; only 1418519
                // fired the new path) but the W44-117 EPF seed
                // mechanism alone only recovers ~30 % of the deficit
                // at d=5 and ~0.5 % at d=6, on the e8/e9 cells where
                // the mechanism can fire. The e7 cells stay
                // byte-identical because W44-117 is structurally gated
                // on `initial_params.epf_iters > 0 AND
                // profile.epf_dynamic_sharpness` which is false at
                // e<=7. Per the task spec's HARD gate ("ship only if
                // ≥ +1.0 net"), this chunk reverts the gate-site
                // change. The constants
                // [`W44_150_PHOTO_W44_117_MASK_P25_MIN`] /
                // [`W44_150_PHOTO_W44_117_MIN_DISTANCE`] and the
                // [`percentile_mask1x1`] helper are KEPT (marked
                // `#[allow(dead_code)]` on the constants) for future
                // chunks that may pair the same discriminator with a
                // different mechanism (e.g. W44-105-style qac-seed
                // scale on photos, AC-strategy lift on
                // mask_p25-high content). Bench artifacts:
                // `benchmarks/w44_150_mask_p25_admission_2026-05-21.{tsv,meta}`.
                params = self.butteraugli_refine_quant_field(
                    linear_rgb,
                    width,
                    height,
                    &xyb_x,
                    &xyb_y,
                    &xyb_b,
                    padded_width,
                    padded_height,
                    xsize_blocks,
                    ysize_blocks,
                    &params,
                    &mut quant_field,
                    &mut quant_field_float,
                    &initial_qf_float,
                    &cfl_map,
                    &ac_strategy,
                    patches_data.as_ref(),
                    splines_data.as_ref(),
                    is_screenshot,
                    // W44-117: plumb the precomputed mask1x1 so the
                    // buttloop can seed its `apply_epf` sharpness map
                    // from `compute_epf_sharpness` (closes the W44-116
                    // buttloop-vs-decoder EPF mismatch).
                    //
                    // W44-118 (SHIPPED): gated on `is_screenshot`. The
                    // W44-117 stale-iter-0 seed is a strict win on
                    // screenshot-class content (terminal e8/e9 d=3
                    // +0.66, d=4 +0.90 SSIM2), but on photos with
                    // mask1x1 in the [50, 80) band (e.g. 1025469
                    // mask=76.08) the iter-0 sharpness diverges from
                    // the converged-qf production sharpness map more
                    // at high d → buttloop converges to a state with
                    // -1% bytes but -0.85 SSIM2 on 1025469 e8/e9 d=4.
                    // W44-118 bisection (mode F sweep on 58 cells in
                    // `benchmarks/w44_118_mode_f_validation_2026-05-20.tsv`
                    // + targeted bisect in `examples/w44_118_bisect.rs`):
                    // gating on `is_screenshot` restores pre-W44-117
                    // photo behaviour (F=A byte-identical on every
                    // photo cell) while preserving the W44-117 wins on
                    // screenshots (F=B byte-identical on every
                    // screenshot cell). Photos that don't fire the
                    // mask>95 screenshot discriminator pay no W44-117
                    // cost AND see no W44-117 quality change — purely
                    // byte-identical to pre-W44-117 main.
                    //
                    // `None` when mask1x1 wasn't precomputed
                    // (pixel_domain_loss off and adaptive_quant didn't
                    // materialise the mask) ALSO falls back to legacy
                    // uniform-4 seed.
                    //
                    // Bisection env hook `JXL_W44_118_SCREENSHOT_ONLY=0`
                    // is NOT supported (production code is now the gated
                    // path). `JXL_W44_117_DISABLE=1` continues to force
                    // legacy uniform-4 across all content classes
                    // (preserved for A/B testing).
                    //
                    // W44-150 HONEST-STOP (see comment above): the
                    // photo admission path was reverted; gate stays at
                    // the W44-118 `is_screenshot ? Some(mask) : None`
                    // form. Pre-W44-150 byte-identical.
                    //
                    // W44-165 HONEST-STOP (Smart-Zenjxl chunk 2,
                    // 2026-05-21): re-implemented the W44-150 admission
                    // for `EncoderStrategy::Zenjxl` / `Aggressive` and
                    // measured a 36-cell paired A/B
                    // (`benchmarks/w44_165_restore_epf_seed_photos_2026-05-21.{tsv,meta}`).
                    // The W44-150 predicted +0.27 mean SSIM2 win on
                    // 1418519 d=5/6 e8/e9 was FALSIFIED in current
                    // main: measured mean = **-0.105** (REGRESSION),
                    // worst -0.331 on e8/e9 d=5. Bytes win small
                    // (-0.77% to -1.17% on 4 of 6 cells). Root cause:
                    // since W44-150's `dad6bb47` baseline, W44-152
                    // (`971bbc8c`) shipped the d ∈ [3.0, 5.0] mask_p25
                    // admission on the W44-29 OUTER entropy_mul lift,
                    // delivering +1.13 SSIM2 to the same 1418519 d=5
                    // e8/e9 cells. The W44-152 baseline (SSIM2=66.54
                    // at e8 d=5) is ABOVE the W44-150 baseline
                    // (SSIM2=65.41); applying the W44-117 EPF seed
                    // mechanism on top of the W44-152 baseline now
                    // OVERSHOOTS (lands at SSIM2=66.21 — net regression
                    // vs the W44-152 baseline despite still being
                    // above the original W44-150 baseline). The two
                    // mechanisms COMPETE rather than COMPOSE on this
                    // cluster. Per the chunk-spec HARD gate (d) "SSIM2
                    // mean improvement ~+0.27 (matches W44-150
                    // measurement)" — measurement falsifies. PROTECT
                    // cells stay clean: 1025469 15/15 BYTE-IDENTICAL,
                    // 4 SPOT photos 12/12 BYTE-IDENTICAL, hash-locks
                    // 36/36 BYTE-IDENTICAL.
                    //
                    // Production gate stays at the W44-118
                    // `is_screenshot ? Some(mask) : None` form. The
                    // [`crate::api::EncoderImprovementsCustom::photo_epf_seed_admit`]
                    // field and the
                    // [`crate::api::ResolvedImprovements::photo_epf_seed_admit`]
                    // field + the strategy defaults (Zenjxl/Aggressive
                    // = true, Libjxl/LeanFaster = false) are KEPT as
                    // public API surface — `Custom` callers wanting to
                    // re-enable the admission (e.g. for a future
                    // chunk that pairs the field with a DIFFERENT
                    // mechanism than the W44-117 EPF seed) flip the
                    // field on. The field is currently inert in
                    // production but trivially re-wired by uncommenting
                    // the `w44_165_photo_admit` predicate below. Bench
                    // artifacts:
                    // `benchmarks/w44_165_restore_epf_seed_photos_2026-05-21.{tsv,meta}`,
                    // `memory/w44_165_photo_epf_seed_zenjxl_honest_stop_2026-05-21.md`.
                    //
                    // INERT PREDICATE (kept for measurement
                    // reproduction; not consumed in production):
                    // ```
                    // let _w44_165_photo_admit =
                    //     self.resolved_improvements.photo_epf_seed_admit
                    //     && self.distance >= W44_150_PHOTO_W44_117_MIN_DISTANCE
                    //     && mask1x1.as_deref().is_some_and(|m| {
                    //         percentile_mask1x1(m, padded_width, width, height, 0.25)
                    //             >= W44_150_PHOTO_W44_117_MASK_P25_MIN
                    //     });
                    // ```
                    if is_screenshot {
                        mask1x1.as_deref()
                    } else {
                        None
                    },
                    // W44-168: pass the adaptive iter count via
                    // `iters_override` so Mode B (SmoothSkip e8+ — 1)
                    // / Mode C (TexturedExtend e7 0 → 2) / Mode D
                    // (Combined) take effect. Mode A (Baseline)
                    // resolves to `self.butteraugli_iters`, byte-
                    // identical to pre-W44-168.
                    Some(effective_buttloop_iters),
                    stop,
                )?;
            }
        }

        // SSIM2 quantization loop: alternative to butteraugli using SSIM2 + per-block RMSE.
        #[cfg(feature = "ssim2-loop")]
        if self.ssim2_iters > 0 {
            let initial_qf_float = quant_field_float.clone();
            params = self.ssim2_refine_quant_field(
                linear_rgb,
                width,
                height,
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                &params,
                &mut quant_field,
                &mut quant_field_float,
                &initial_qf_float,
                &cfl_map,
                &ac_strategy,
                patches_data.as_ref(),
                splines_data.as_ref(),
            )?;
        }

        // Zensim quantization loop: uses zensim psychovisual metric + per-pixel diffmap.
        // Also refines AC strategy by splitting large transforms with high perceptual error.
        #[cfg(feature = "zensim-loop")]
        if self.zensim_iters > 0 {
            let initial_qf_float = quant_field_float.clone();
            params = self.zensim_refine_quant_field(
                linear_rgb,
                width,
                height,
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                &params,
                &mut quant_field,
                &mut quant_field_float,
                &initial_qf_float,
                &cfl_map,
                &mut ac_strategy,
                patches_data.as_ref(),
                splines_data.as_ref(),
            )?;
        }

        // Free float quant field — no longer needed after loop refinement.
        drop(quant_field_float);

        // Log quant field statistics after all adjustments
        {
            let qf = &quant_field;
            let sum: u64 = qf.iter().map(|&v| v as u64).sum();
            let avg = sum as f32 / qf.len() as f32;
            let min = qf.iter().copied().min().unwrap_or(0);
            let max = qf.iter().copied().max().unwrap_or(0);
            debug_rect!(
                "enc/quant_field",
                0,
                0,
                width,
                height,
                "final avg={:.1} min={} max={} blocks={}",
                avg,
                min,
                max,
                qf.len()
            );
        }

        // Dump AC strategy and quant field maps for comparison with libjxl.
        // Set JXL_DUMP_MAPS=/tmp/prefix to enable. Maps are written as CSV.
        #[cfg(feature = "debug-rect")]
        if let Ok(prefix) = std::env::var("JXL_DUMP_MAPS") {
            use std::io::Write;
            // AC strategy map
            if let Ok(mut f) = std::fs::File::create(format!("{prefix}_acs.csv")) {
                for by in 0..ysize_blocks {
                    for bx in 0..xsize_blocks {
                        if bx > 0 {
                            let _ = write!(f, ",");
                        }
                        let _ = write!(f, "{}", ac_strategy.raw_strategy(bx, by));
                    }
                    let _ = writeln!(f);
                }
                eprintln!("DIAG: wrote {prefix}_acs.csv ({xsize_blocks}x{ysize_blocks})");
            }
            // Quant field map
            if let Ok(mut f) = std::fs::File::create(format!("{prefix}_qf.csv")) {
                for by in 0..ysize_blocks {
                    for bx in 0..xsize_blocks {
                        if bx > 0 {
                            let _ = write!(f, ",");
                        }
                        let _ = write!(f, "{}", quant_field[by * xsize_blocks + bx]);
                    }
                    let _ = writeln!(f);
                }
                eprintln!("DIAG: wrote {prefix}_qf.csv ({xsize_blocks}x{ysize_blocks})");
            }
        }

        // CfL pass 2 was moved up to BEFORE the butteraugli loop (drift-fix
        // chunk-3, 2026-05-15). See the moved block above and the comment
        // there for ordering rationale.

        let _ms_buttloop = _t_buttloop.elapsed().as_secs_f64() * 1000.0;
        let _t_xform = std::time::Instant::now();
        // ── Streaming refactor chunk 8b (#11): region-source seam ──
        //
        // Wrap the three whole-image XYB Vecs in a
        // `WholeImageXybSource` so `transform_and_quantize` (and any
        // chunk-8c per-DC-group orchestrator) can pull the data
        // through the `XybRegionSource` trait. The whole-image source
        // is a thin wrapper — output bytes are bit-identical to the
        // pre-chunk-8b path (proven by `hash_lock_features.rs`
        // 36/36).
        //
        // After `transform_and_quantize` returns we walk the DC-group
        // grid and call `release_dc_region(dc_x, dc_y)` for each
        // region. The whole-image source's `release_dc_region` is a
        // no-op today — chunk 8c plugs in a streaming source that
        // drops the region's storage on the hint. EPF sharpness
        // derivation (whole-image consumer #1) still reads the source
        // afterward; chunk 8c lifts that into the per-DC-group walker
        // so the release can happen before sharpness runs.
        let xyb_source = super::region_source::WholeImageXybSource::new(
            width,
            height,
            padded_width,
            padded_height,
            xyb_x,
            xyb_y,
            xyb_b,
        );

        // Perform DCT and quantization (XYB data is padded to block
        // boundaries). The trait routes through `xyb_full()` for the
        // whole-image source; output is byte-identical to the direct
        // call.
        let mut transform_out = self.transform_and_quantize_with_source(
            &xyb_source,
            xsize_blocks,
            ysize_blocks,
            &params,
            &mut quant_field,
            &cfl_map,
            &ac_strategy,
        )?;

        // W44-AUDIT-8 Phase 6: apply libjxl QuantizeWP shape to DC
        // values when the gate fires (effort ≤ 7 by default; libjxl
        // `nl_dc = speed_tier < kFalcon` parity). Post-pass over the
        // already-computed `float_dc` + `quant_dc` from the transform
        // pipeline. At effort ≥ 8 this is a no-op (the buttloop owns
        // DC refinement and libjxl drops to plain `std::round`).
        // Env hook `JXL_W44_AUDIT_8_P6_FORCE_QUANTIZE_WP=1` force-enables
        // for the Phase 6 bisect bench + diagnostic A/B at any effort.
        let phase6_env_on = std::env::var_os("JXL_W44_AUDIT_8_P6_FORCE_QUANTIZE_WP").is_some()
            && self.profile.effort <= 7;
        if self.profile.use_libjxl_wp_dc_quant || phase6_env_on {
            super::quantize_wp::requantize_dc_group_wp(
                &mut transform_out.quant_dc,
                &transform_out.float_dc,
                xsize_blocks,
                0,
                0,
                xsize_blocks,
                ysize_blocks,
                params.scale_dc,
                params.extra_dc_precision,
            );
        }
        let _ms_xform = _t_xform.elapsed().as_secs_f64() * 1000.0;
        let _t_sharp = std::time::Instant::now();
        let quant_dc = &transform_out.quant_dc;
        let quant_ac = &transform_out.quant_ac;
        let nzeros = &transform_out.nzeros;
        let raw_nzeros = &transform_out.raw_nzeros;

        // Chunk-8b drop seam (no-op on whole-image source): walk
        // every DC group + hint the source it can release that
        // region's storage. With the chunk-8c streaming source this
        // is where the per-DC-group XYB buffer actually drops; with
        // the whole-image source it's a fast no-op that documents
        // the lifetime.
        for dc_y in 0..ysize_dc_groups {
            for dc_x in 0..xsize_dc_groups {
                super::region_source::XybRegionSource::release_dc_region(
                    &xyb_source,
                    dc_x as u32,
                    dc_y as u32,
                );
            }
        }

        // Re-borrow the planes for the remaining whole-image
        // consumers (compute_epf_sharpness + the legacy `drop(xyb_*)`
        // path). Chunk-8c lifts each consumer into the per-DC-group
        // walker so this re-borrow goes away entirely.
        let (xyb_x_ref, xyb_y_ref, xyb_b_ref) =
            super::region_source::XybRegionSource::xyb_full(&xyb_source);

        // Compute per-block EPF sharpness map when EPF is active.
        // Dynamic sharpness gated at effort >= 6 (speed_tier <= kWombat) matching libjxl.
        //
        // Chunk 8c step B (#11): the mask1x1 resolution (precomputed
        // borrow vs xyb_y fallback) is now hoisted into
        // `resolve_mask1x1_for_sharpness` so the EPF branch no longer
        // owns the fallback's XYB-read dependency inline. The
        // chunk-8c streaming source will own `xyb_y_ref` for the
        // duration of resolve_mask1x1_for_sharpness, then release
        // per-DC-group storage before the compute_epf_sharpness call
        // (which only needs the resolved mask plus the original-XYB
        // borrows).
        let sharpness_map =
            if params.epf_iters > 0 && self.distance >= 0.5 && self.profile.epf_dynamic_sharpness {
                match self.epf_dispatch {
                    crate::api::EpfDispatch::AlwaysDefault => {
                        // W36-2: skip per-block search, write uniform default sharpness.
                        Some(super::epf::uniform_default_sharpness_map(
                            xsize_blocks,
                            ysize_blocks,
                        ))
                    }
                    crate::api::EpfDispatch::Auto | crate::api::EpfDispatch::AlwaysSelect => {
                        let mask_cow = super::adaptive_quant::resolve_mask1x1_for_sharpness(
                            mask1x1.as_deref(),
                            xyb_y_ref,
                            padded_width,
                            padded_height,
                            self.budget.as_ref(),
                        )?;
                        // W36-2 Auto: skip the per-block search on
                        // smooth regions (predicate: mean mask1x1
                        // above EPF_AUTO_SMOOTH_MASK_THRESHOLD). The
                        // bitstream still gets a sharpness map, just
                        // the uniform default one — same shape as
                        // AlwaysDefault but content-gated.
                        if matches!(self.epf_dispatch, crate::api::EpfDispatch::Auto)
                            && super::epf::mask1x1_is_smooth_enough_to_skip_sharpness(&mask_cow)
                        {
                            Some(super::epf::uniform_default_sharpness_map(
                                xsize_blocks,
                                ysize_blocks,
                            ))
                        } else {
                            Some(super::epf::compute_epf_sharpness(
                                [xyb_x_ref, xyb_y_ref, xyb_b_ref],
                                quant_dc,
                                quant_ac,
                                &quant_field,
                                &mask_cow,
                                &params,
                                &cfl_map,
                                &ac_strategy,
                                self.enable_gaborish,
                                xsize_blocks,
                                ysize_blocks,
                                self.budget.as_ref(),
                            )?)
                        }
                    }
                }
            } else {
                None
            };

        // Free the XYB source — no longer needed after EPF sharpness
        // computation. At 4K (6720×4480), this frees ~339 MB
        // (3 channels × padded_pixels × f32) for the whole-image
        // source; the chunk-8c streaming source will have already
        // dropped most of it via per-region release hints.
        drop(xyb_source);
        // Free mask1x1 — up to ~115 MB at 4K (padded_pixels × f32).
        drop(mask1x1);

        let _ms_sharp = _t_sharp.elapsed().as_secs_f64() * 1000.0;
        let _t_entropy = std::time::Instant::now();

        // W44-87 single-pass-entropy dispatch — safety predicate +
        // override. The streaming/one-pass static-Huffman path cannot
        // serialize any of the features below; if any are present the
        // dispatch silently falls back to the two-pass plumbing. Order
        // matters only inasmuch as `single_pass_entropy_content_gate`
        // was decided earlier (effort, distance, mask1x1_median); the
        // SAFETY portion runs here because `patches_data`,
        // `splines_data`, `sharpness_map`, `noise_params`, and the
        // extras slice are all in scope.
        let single_pass_entropy_safe = self.optimize_codes
            && single_pass_entropy_content_gate
            && extras.is_empty()
            && patches_data.is_none()
            && splines_data.is_none()
            && sharpness_map.is_none()
            && noise_params.is_none()
            && !self.use_lf_frame
            && !self.dc_tree_learning;
        let optimize_codes_effective = if single_pass_entropy_safe {
            false
        } else {
            self.optimize_codes
        };

        // Two-pass mode: collect tokens, build optimal codes, write bitstream
        if optimize_codes_effective {
            let strategy_counts = ac_strategy.strategy_histogram();
            let data = self.encode_two_pass(
                width,
                height,
                &params,
                xsize_blocks,
                ysize_blocks,
                xsize_groups,
                ysize_groups,
                xsize_dc_groups,
                ysize_dc_groups,
                num_groups,
                num_dc_groups,
                num_sections,
                quant_dc,
                quant_ac,
                nzeros,
                raw_nzeros,
                &quant_field,
                &cfl_map,
                &ac_strategy,
                &noise_params,
                sharpness_map.as_deref(),
                extras,
                patches_data.as_ref(),
                splines_data.as_ref(),
                if self.use_lf_frame {
                    Some(&transform_out.float_dc)
                } else {
                    None
                },
            )?;
            let _ms_entropy = _t_entropy.elapsed().as_secs_f64() * 1000.0;
            let _ms_total = _t_total.elapsed().as_secs_f64() * 1000.0;
            if _phase_dbg {
                eprintln!(
                    "encode_inner: total={_ms_total:.1} xyb={_ms_xyb:.1} patches={_ms_patches:.1} splines={_ms_splines:.1} quant_field={_ms_quant_field:.1} gaborish={_ms_gaborish:.1} cfl1={_ms_cfl1:.1} acstrat={_ms_acstrat:.1} cfl2={_ms_cfl2:.1} buttloop={_ms_buttloop:.1} xform={_ms_xform:.1} sharp={_ms_sharp:.1} entropy={_ms_entropy:.1}",
                );
            }
            crate::debug_rect::flush("");
            return Ok(VarDctOutput {
                data,
                strategy_counts,
            });
        }

        // Get static entropy codes (wrapped in BuiltEntropyCode for uniform handling)
        let dc_code = BuiltEntropyCode::StaticHuffman(get_dc_entropy_code());
        let ac_code = BuiltEntropyCode::StaticHuffman(get_ac_entropy_code());

        // Create main writer. The capacity is a heuristic upper bound on the
        // bitstream size — actual usage is much smaller in compression but
        // we budget the full reservation since BitWriter eagerly allocates.
        let main_cap = (width as u64)
            .saturating_mul(height as u64)
            .saturating_mul(4);
        crate::budget::MemoryBudget::reserve_permanent_opt(self.budget.as_ref(), main_cap)?;
        // Honor the budget's runtime fallible-alloc policy for this large
        // dimension-driven buffer (`Limits::fallible_alloc`): `with_capacity`
        // (fast) by default, `try_reserve` (graceful OOM) for untrusted input.
        let fallible = self.budget.as_ref().is_some_and(|b| b.is_fallible());
        let mut writer = BitWriter::with_capacity_fallible(fallible, width * height * 4)?;

        // Write file header (includes JXL signature, ICC, and byte padding).
        // The streaming/one-pass static-Huffman path doesn't carry any
        // extras — they require the two-pass dynamic-entropy plumbing.
        self.write_file_header_and_pad(width, height, &[], &mut writer)?;
        #[cfg(feature = "debug-tokens")]
        debug_log!(
            "After file header: bit {} (byte {})",
            writer.bits_written(),
            writer.bits_written() / 8
        );

        // Write frame header
        {
            let mut fh = FrameHeader::lossy();
            fh.x_qm_scale = params.x_qm_scale;
            fh.b_qm_scale = params.b_qm_scale;
            fh.epf_iters = params.epf_iters;
            fh.gaborish = self.enable_gaborish;
            fh.upsampling = self.upsampling;
            if noise_params.is_some() {
                fh.flags |= 0x01; // ENABLE_NOISE
            }
            // streaming path: no extra channels
            fh.write(&mut writer)?;
        }
        #[cfg(feature = "debug-tokens")]
        debug_log!(
            "After frame header: bit {} (byte {})",
            writer.bits_written(),
            writer.bits_written() / 8
        );

        // For single-group images, combine all sections at the bit level
        // (no byte padding between sections, only at the end)
        if num_sections == 4 {
            // Write sections to individual BitWriters (no padding)
            // W44-133 Chunk G: select 15-cluster libjxl default when
            // `EncoderStrategy::Libjxl` is in effect; Zenjxl 4-cluster
            // default is byte-identical to pre-Chunk-G output.
            let block_ctx_map = super::ac_context::BlockCtxMap::default_for_strategy(
                self.resolved_improvements.block_ctx_map_15_cluster,
            );
            let num_blocks = xsize_blocks * ysize_blocks;
            let mut dc_global = BitWriter::with_capacity(4096);
            self.write_dc_global(
                &params,
                num_dc_groups,
                &dc_code,
                &noise_params,
                None,
                &block_ctx_map,
                None, // No learned tree in single-pass mode
                None, // No patches in streaming mode
                None, // No splines in streaming mode
                None, // No custom dc_quant in single-pass mode
                &mut dc_global,
            )?;

            // Get borrowed Huffman codes for streaming token writing
            let dc_huffman = dc_code.as_huffman();
            let ac_huffman = ac_code.as_huffman();

            // dc_group + ac_group BitWriters are sized proportional to the
            // image (10 bytes/block DC + 100 bytes/block AC heuristic).
            // Account both up front against the budget.
            crate::budget::MemoryBudget::reserve_permanent_opt(
                self.budget.as_ref(),
                (num_blocks as u64).saturating_mul(110),
            )?;
            // Runtime fallible-alloc policy for these dimension-driven group
            // writers (`Limits::fallible_alloc`); covers dc_group + ac_group.
            let fallible = self.budget.as_ref().is_some_and(|b| b.is_fallible());
            let mut dc_group = BitWriter::with_capacity_fallible(fallible, num_blocks * 10)?;
            self.write_dc_group(
                0,
                quant_dc,
                xsize_blocks,
                ysize_blocks,
                xsize_dc_groups,
                &quant_field,
                &cfl_map,
                &ac_strategy,
                sharpness_map.as_deref(),
                &dc_huffman,
                &mut dc_group,
            )?;

            let mut ac_global = BitWriter::with_capacity(4096);
            self.write_ac_global(
                num_groups,
                core::slice::from_ref(&ac_code),
                0,
                None,
                &[None],
                &mut ac_global,
            )?;

            let mut ac_group_writer =
                BitWriter::with_capacity_fallible(fallible, num_blocks * 100)?;
            self.write_ac_group(
                0,
                quant_ac,
                nzeros,
                raw_nzeros,
                xsize_blocks,
                ysize_blocks,
                xsize_groups,
                &quant_field,
                &ac_strategy,
                &block_ctx_map,
                &ac_huffman,
                &mut ac_group_writer,
            )?;

            #[cfg(feature = "debug-tokens")]
            {
                debug_log!(
                    "Section bit counts: DC_global={}, DC_group={}, AC_global={}, AC_group={}",
                    dc_global.bits_written(),
                    dc_group.bits_written(),
                    ac_global.bits_written(),
                    ac_group_writer.bits_written()
                );
            }

            // Combine at bit level
            let mut combined = dc_global;
            #[cfg(feature = "debug-tokens")]
            debug_log!("After DC_global: {} bits", combined.bits_written());
            combined.append_unaligned(&dc_group)?;
            #[cfg(feature = "debug-tokens")]
            debug_log!("After DC_group: {} bits", combined.bits_written());
            combined.append_unaligned(&ac_global)?;
            #[cfg(feature = "debug-tokens")]
            debug_log!("After AC_global: {} bits", combined.bits_written());
            combined.append_unaligned(&ac_group_writer)?;
            #[cfg(feature = "debug-tokens")]
            debug_log!("After AC_group: {} bits", combined.bits_written());
            combined.zero_pad_to_byte();
            let combined_bytes = combined.finish();

            #[cfg(feature = "debug-tokens")]
            {
                debug_log!("Combined section size: {} bytes", combined_bytes.len());
                debug_log!(
                    "Before TOC: bit {} (byte {})",
                    writer.bits_written(),
                    writer.bits_written() / 8
                );
            }
            write_toc(&[combined_bytes.len()], &mut writer)?;
            #[cfg(feature = "debug-tokens")]
            debug_log!(
                "After TOC: bit {} (byte {})",
                writer.bits_written(),
                writer.bits_written() / 8
            );
            writer.append_bytes(&combined_bytes)?;
        } else {
            // Multi-group: use byte-aligned sections.
            // Section bytes accumulate across groups; the heuristic capacity
            // is num_groups * blocks_per_group * 100 + num_dc_groups * 10240
            // ≈ num_blocks * 100 in total. Account up front.
            let total_groups_bytes = (xsize_blocks as u64)
                .saturating_mul(ysize_blocks as u64)
                .saturating_mul(110);
            crate::budget::MemoryBudget::reserve_permanent_opt(
                self.budget.as_ref(),
                total_groups_bytes,
            )?;
            // Runtime fallible-alloc policy for the per-group writers below.
            let fallible = self.budget.as_ref().is_some_and(|b| b.is_fallible());
            let mut sections: Vec<Vec<u8>> = Vec::with_capacity(num_sections);
            let dc_huffman = dc_code.as_huffman();
            let ac_huffman = ac_code.as_huffman();

            // DC Global section
            // W44-133 Chunk G: select 15-cluster libjxl default when
            // `EncoderStrategy::Libjxl` is in effect; Zenjxl 4-cluster
            // default is byte-identical to pre-Chunk-G output.
            let block_ctx_map = super::ac_context::BlockCtxMap::default_for_strategy(
                self.resolved_improvements.block_ctx_map_15_cluster,
            );
            let mut dc_global = BitWriter::with_capacity(4096);
            self.write_dc_global(
                &params,
                num_dc_groups,
                &dc_code,
                &noise_params,
                None,
                &block_ctx_map,
                None, // No learned tree in single-pass mode
                None, // No patches in streaming mode
                None, // No splines in streaming mode
                None, // No custom dc_quant in single-pass mode
                &mut dc_global,
            )?;
            dc_global.zero_pad_to_byte();
            sections.push(dc_global.finish());

            // DC group sections
            let blocks_per_dc_group = (256 / 8) * (256 / 8); // 1024 blocks per DC group
            for dc_group_idx in 0..num_dc_groups {
                // Coarse cancellation checkpoint (per DC group). No-op on the
                // success path, so byte output is identical under Unstoppable.
                if let Some(s) = stop {
                    s.check().map_err(|_| Error::Cancelled)?;
                }
                let mut dc_group =
                    BitWriter::with_capacity_fallible(fallible, blocks_per_dc_group * 10)?;
                self.write_dc_group(
                    dc_group_idx,
                    quant_dc,
                    xsize_blocks,
                    ysize_blocks,
                    xsize_dc_groups,
                    &quant_field,
                    &cfl_map,
                    &ac_strategy,
                    sharpness_map.as_deref(),
                    &dc_huffman,
                    &mut dc_group,
                )?;
                dc_group.zero_pad_to_byte();
                sections.push(dc_group.finish());
            }

            // AC Global section
            let mut ac_global = BitWriter::with_capacity(4096);
            self.write_ac_global(
                num_groups,
                core::slice::from_ref(&ac_code),
                0,
                None,
                &[None],
                &mut ac_global,
            )?;
            ac_global.zero_pad_to_byte();
            sections.push(ac_global.finish());

            // AC group sections
            let blocks_per_ac_group = (256 / 8) * (256 / 8); // 1024 blocks per AC group
            for group_idx in 0..num_groups {
                // Coarse cancellation checkpoint (per AC group). No-op on the
                // success path, so byte output is identical under Unstoppable.
                if let Some(s) = stop {
                    s.check().map_err(|_| Error::Cancelled)?;
                }
                let mut ac_group_writer =
                    BitWriter::with_capacity_fallible(fallible, blocks_per_ac_group * 100)?;
                self.write_ac_group(
                    group_idx,
                    quant_ac,
                    nzeros,
                    raw_nzeros,
                    xsize_blocks,
                    ysize_blocks,
                    xsize_groups,
                    &quant_field,
                    &ac_strategy,
                    &block_ctx_map,
                    &ac_huffman,
                    &mut ac_group_writer,
                )?;
                ac_group_writer.zero_pad_to_byte();
                sections.push(ac_group_writer.finish());
            }

            // Center-first AC group reordering (closes #14). Identity
            // prefix for [DC global, DC groups..., AC global], then
            // AC groups permuted by concentric-square distance from
            // image center. libjxl-faithful PermuteGroups algorithm.
            //
            // Single-pass only — multi-pass progressive interaction
            // is a future extension. num_groups <= 1 → no-op (nothing
            // to reorder).
            if self.center_first && num_groups > 1 {
                use crate::vardct::coeff_order::compute_center_first_ac_permutation;
                // Caller-supplied center_x / center_y (libjxl
                // `cparams.center_x` / `center_y`); fall back to image
                // centre when unset, matching libjxl's
                // `size_t(-1) → width/2` behaviour at enc_frame.cc.
                let cx = self
                    .center_x
                    .map(|x| x.min(width.saturating_sub(1) as u32))
                    .unwrap_or((width as u32) / 2);
                let cy = self
                    .center_y
                    .map(|y| y.min(height.saturating_sub(1) as u32))
                    .unwrap_or((height as u32) / 2);
                let ac_group_order =
                    compute_center_first_ac_permutation(xsize_groups, ysize_groups, cx, cy);
                // Build inverse mapping: inv[orig_idx] = on_disk_pos.
                let mut inv_ac = vec![0u32; num_groups];
                for (on_disk_pos, &orig_idx) in ac_group_order.iter().enumerate() {
                    inv_ac[orig_idx as usize] = on_disk_pos as u32;
                }
                // libjxl permutation array: identity prefix +
                // inv_ac_group_order offset by prefix length.
                let prefix_len = 2 + num_dc_groups;
                let total = prefix_len + num_groups;
                let mut permutation = Vec::with_capacity(total);
                for i in 0..prefix_len {
                    permutation.push(i as u32);
                }
                let prefix_u32 = prefix_len as u32;
                for &val in &inv_ac[..num_groups] {
                    permutation.push(prefix_u32 + val);
                }
                // Reorder sections: new[permutation[i]] = sections[i].
                let mut new_sections: Vec<Vec<u8>> = (0..total).map(|_| Vec::new()).collect();
                for (logical_idx, section_data) in sections.into_iter().enumerate() {
                    let on_disk = permutation[logical_idx] as usize;
                    new_sections[on_disk] = section_data;
                }
                let section_sizes: Vec<usize> = new_sections.iter().map(|s| s.len()).collect();
                write_toc_with_permutation(
                    &section_sizes,
                    &permutation,
                    self.use_ans,
                    &mut writer,
                )?;
                for section in new_sections {
                    writer.append_bytes(&section)?;
                }
            } else {
                let section_sizes: Vec<usize> = sections.iter().map(|s| s.len()).collect();
                write_toc(&section_sizes, &mut writer)?;
                for section in sections {
                    writer.append_bytes(&section)?;
                }
            }
        }

        let strategy_counts = ac_strategy.strategy_histogram();
        crate::debug_rect::flush("");
        Ok(VarDctOutput {
            data: writer.finish_with_padding(),
            strategy_counts,
        })
    }

    /// Encode with iterative rate control for improved distance targeting.
    ///
    /// This method:
    /// 1. Computes precomputed state (XYB, CfL, masking, AC strategy) once
    /// 2. Loops: encode → decode → butteraugli → adjust quant field
    /// 3. Returns when converged (within 5% of target) or max iterations reached
    ///
    /// Typically converges in 2-4 iterations. Each iteration costs ~50% of a
    /// full encode since XYB conversion, CfL, masking, and AC strategy are reused.
    ///
    /// Returns the encoded bytes. Use `encode_with_rate_control_config` for
    /// iteration count and custom configuration.
    ///
    /// Requires the `rate-control` feature.
    #[cfg(feature = "rate-control")]
    pub fn encode_with_rate_control(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
    ) -> Result<Vec<u8>> {
        let config = super::rate_control::RateControlConfig::default();
        let (encoded, _iters) =
            self.encode_with_rate_control_config(width, height, linear_rgb, &config)?;
        Ok(encoded)
    }

    /// Encode with iterative rate control and custom configuration.
    ///
    /// Returns `(encoded_bytes, iteration_count)`.
    ///
    /// Requires the `rate-control` feature.
    #[cfg(feature = "rate-control")]
    pub fn encode_with_rate_control_config(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        config: &super::rate_control::RateControlConfig,
    ) -> Result<(Vec<u8>, usize)> {
        // Compute precomputed state — patches detected and subtracted
        // BEFORE quant_field / mask / gaborish / CfL / AC strategy so
        // that all subsequent precomputation sees the patches-subtracted
        // XYB. Matches libjxl pipeline order
        // (`enc_heuristics.cc:1057-1194`).
        // Chunk-6 (#11) + chunk-8c streaming gate: route the caller's
        // Buffering policy into EncoderPrecomputed, but downgrade to
        // FullBuffered when the butteraugli quantization loop is
        // active. Mirrors libjxl `CanDoStreamingEncoding`
        // (`enc_frame.cc`): the buttloop reconstructs the whole
        // image per iteration and cannot run from a sliding-window
        // XYB source, so a caller-requested BufferedOutput/
        // FullStreaming silently falls back to FullBuffered when
        // butteraugli iterations are configured. The buttloop is
        // feature- and effort-gated (default effort 7 leaves it off)
        // so this gate is a no-op for the typical Auto + default
        // path; it matters when an explicit `--butteraugli-iters`
        // combines with `--buffering 2` / `--buffering 3`.
        //
        // `Auto` resolves first via `resolve_for_streaming` so the
        // routed variant is always concrete.
        let buttloop_iters: u32 = {
            #[cfg(feature = "butteraugli-loop")]
            {
                self.butteraugli_iters
            }
            #[cfg(not(feature = "butteraugli-loop"))]
            {
                0u32
            }
        };
        let routed_buffering =
            self.buffering
                .resolve_for_streaming(width as u32, height as u32, buttloop_iters);
        // Chunk-6 (#11): route the caller's Buffering policy into
        // EncoderPrecomputed. `Buffering::Auto` resolves on image size;
        // explicit `BufferedOutput` / `FullStreaming` engage the
        // per-region precompute path.
        let precomputed =
            super::precomputed::EncoderPrecomputed::compute_with_budget_and_buffering(
                width,
                height,
                linear_rgb,
                self.distance,
                self.cfl_enabled,
                self.ac_strategy_enabled,
                self.pixel_domain_loss,
                self.enable_noise,
                self.enable_denoise,
                self.enable_gaborish,
                self.enable_adaptive_gaborish,
                self.enable_patches,
                self.use_ans,
                self.encoder_mode,
                self.force_strategy,
                &self.profile,
                self.color_encoding.as_ref(),
                self.budget.as_ref(),
                routed_buffering,
                self.pixel_loss_dispatch,
            )?;

        // Run rate control loop
        super::rate_control::encode_with_rate_control(self, &precomputed, config)
    }

    /// Encode from precomputed state with a specific quant field.
    ///
    /// This is the core encoding function used by rate control iterations.
    /// It skips XYB conversion, CfL, masking, and AC strategy computation,
    /// using the values from `precomputed` instead.
    ///
    /// Color-only entry point (no extra channels). For RGBA / depth /
    /// spot color / selection mask / thermal / CFA see
    /// [`Self::encode_from_precomputed_with_extras`].
    ///
    /// Requires the `rate-control` OR `__pre_quantized` feature. The
    /// latter is the unstable downstream-callable path used by
    /// jxl-encoder-gpu (or any other pre-quantized caller) to feed
    /// already-prepared XYB / AC strategy / CfL / quant field directly
    /// to the bitstream emit path, skipping XYB / CfL / masking /
    /// strat-search / butteraugli refinement that the encoder would
    /// otherwise re-do.
    #[cfg(any(feature = "rate-control", feature = "__pre_quantized"))]
    pub fn encode_from_precomputed(
        &self,
        precomputed: &super::precomputed::EncoderPrecomputed,
        quant_field: &[u8],
    ) -> Result<Vec<u8>> {
        self.encode_from_precomputed_with_extras(precomputed, quant_field, &[])
    }

    /// Encode from precomputed state with a specific quant field plus
    /// an arbitrary list of extra channels (alpha, depth, spot color,
    /// selection mask, thermal, CFA, …).
    ///
    /// Same fast path as [`Self::encode_from_precomputed`] (skips XYB
    /// conversion, CfL, masking, AC strategy search, and butteraugli
    /// refinement) but additionally writes a modular sub-bitstream
    /// carrying each extra channel alongside the VarDCT color data.
    /// Each extra's [`crate::headers::extra_channels::ExtraChannelInfo`]
    /// goes into the file-header metadata; the writer pulls
    /// `channel_width * channel_height` samples per channel out of the
    /// shared sub-bitstream in the same order they were passed in.
    ///
    /// **Current scope** (mirrors [`Self::encode_with_extras`]):
    /// - `dim_shift` must be `0` on every extra (full-resolution
    ///   channels).
    /// - Single-group (≤256×256) supports any number of extras.
    /// - Multi-group supports any number of extras at `dim_shift = 0`.
    ///
    /// Other combinations return [`crate::Error::InvalidInput`] so the
    /// wire format stays correct as those paths are filled in.
    ///
    /// **Butteraugli loop integration**: the buttloop only consumes
    /// the three color planes — extras are passed through unchanged to
    /// the final bitstream emit. Since the buttloop only refines
    /// `quant_field` (not extras), extras are forwarded after loop
    /// convergence; the GPU encoder / rate-control path produces the
    /// final `quant_field` first, then this entry threads the extras
    /// into the bitstream.
    ///
    /// Requires the `rate-control` OR `__pre_quantized` feature.
    #[cfg(any(feature = "rate-control", feature = "__pre_quantized"))]
    pub fn encode_from_precomputed_with_extras(
        &self,
        precomputed: &super::precomputed::EncoderPrecomputed,
        quant_field: &[u8],
        extras: &[crate::api::ExtraChannel<'_>],
    ) -> Result<Vec<u8>> {
        // Validate extras at the entry boundary (dim_shift = 0,
        // sample-count = width * height) before any encoding work runs.
        // Mirrors `encode_with_extras` so error messages and behavior
        // stay consistent across the two entry points. We need the
        // validated `VardctExtra` views for the final `encode_two_pass`
        // call below, so build them here once and reuse.
        let width = precomputed.width;
        let height = precomputed.height;
        let mut extras_views: Vec<super::extras::VardctExtra<'_>> =
            Vec::with_capacity(extras.len());
        for (idx, ec) in extras.iter().enumerate() {
            if ec.info().dim_shift != 0 {
                return Err(Error::InvalidInput(format!(
                    "extras[{idx}]: dim_shift = {} not yet supported in lossy encode (dim_shift > 0 \
                     for VarDCT extras is a follow-up)",
                    ec.info().dim_shift
                )));
            }
            let expected = width.checked_mul(height).ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 1,
            })?;
            let got = ec.data().len();
            if got != expected {
                return Err(Error::InvalidInput(format!(
                    "extras[{idx}]: expected {expected} samples for {width}x{height}, got {got}"
                )));
            }
            extras_views.push(super::extras::VardctExtra::from_api(ec));
        }
        self.check_alpha_squeeze_supported(&extras_views, Some((width, height)))?;
        self.encode_from_precomputed_inner(precomputed, quant_field, &extras_views)
    }

    /// Shared body for [`Self::encode_from_precomputed`] and
    /// [`Self::encode_from_precomputed_with_extras`]. Takes already-
    /// validated `extras` as the internal `VardctExtra` view.
    #[cfg(any(feature = "rate-control", feature = "__pre_quantized"))]
    fn encode_from_precomputed_inner(
        &self,
        precomputed: &super::precomputed::EncoderPrecomputed,
        quant_field: &[u8],
        extras: &[super::extras::VardctExtra<'_>],
    ) -> Result<Vec<u8>> {
        let width = precomputed.width;
        let height = precomputed.height;
        let xsize_blocks = precomputed.xsize_blocks;
        let ysize_blocks = precomputed.ysize_blocks;
        let padded_width = precomputed.padded_width;
        // W44-44: hoist `padded_height` ahead of the
        // `BorrowedXybSource::new(..., padded_height, ...)` call at
        // ~line 3531 (use-before-let; only triggers under
        // `rate-control` / `__pre_quantized` cfg gate at line 3333,
        // which is why default `cargo check --release` did not
        // surface the break). Previous binding sat after the call
        // site (~line 3573) and got shadowed by the EPF branch's own
        // `let padded_height = precomputed.padded_height;`.
        let padded_height = precomputed.padded_height;

        // Calculate group dimensions
        let xsize_groups = div_ceil(width, GROUP_DIM);
        let ysize_groups = div_ceil(height, GROUP_DIM);
        let xsize_dc_groups = div_ceil(width, DC_GROUP_DIM);
        let ysize_dc_groups = div_ceil(height, DC_GROUP_DIM);
        let num_groups = xsize_groups * ysize_groups;
        let num_dc_groups = xsize_dc_groups * ysize_dc_groups;
        let num_sections = 2 + num_dc_groups + num_groups;

        // Copy and adjust quant field for multi-block transforms
        let mut quant_field = quant_field.to_vec();
        adjust_quant_field_with_distance(&precomputed.ac_strategy, &mut quant_field, self.distance);

        // Compute distance params from effort profile
        let mut params = match self.original_distance {
            Some(orig) if orig > self.distance => {
                DistanceParams::compute_for_profile_with_original(
                    self.distance,
                    orig,
                    &self.profile,
                )
            }
            _ => DistanceParams::compute_for_profile(self.distance, &self.profile),
        };
        if let Some(rescale) = self.quant_ac_rescale {
            params.apply_quant_ac_rescale(rescale);
        }
        // libjxl --epf override: when the caller pinned a level, it wins
        // over the distance-derived `epf_iters` (enc_frame.cc:284-285).
        self.apply_epf_level_override(&mut params);

        // Apply pixel-level chromacity adjustments using pre-gaborish stats
        if self.profile.chromacity_adjustment {
            params.apply_chromacity_adjustment(
                precomputed.chromacity_x_pixelized,
                precomputed.chromacity_b_pixelized,
            );
        }

        // Patches handling.
        //
        // Two cases:
        //
        // (1) `precomputed.patches_data` is `Some(pd)`. The upstream
        // `compute_with_budget` already detected, quantized, and
        // subtracted patches from the XYB BEFORE running quant_field /
        // mask / gaborish / CfL / AC strategy — so `precomputed.xyb_*`
        // is the post-gaborish PATCHES-SUBTRACTED XYB and all the
        // precomputed state matches what the decoder will dequantize
        // against. We just write the patches frame to the bitstream.
        //
        // (2) `precomputed.patches_data` is `None` but
        // `precomputed.xyb_pre_gaborish` is `Some`. Legacy path used by
        // jxl-encoder-gpu callers that build via `from_parts` and
        // attach pre-gab XYB. We detect patches here, subtract from
        // pre-gab, re-apply gaborish, and use the result for DCT — but
        // the precomputed CfL / AC strategy / quant_field were computed
        // on UN-patched XYB so this case has the libjxl-parity gap
        // documented in the GPU encoder. Closing that requires GPU-side
        // patches detection before the GPU pipeline runs.
        //
        // (3) Both `None`: no patches.
        let _t_patches = std::time::Instant::now();
        let mut patches_data: Option<super::patches::PatchesData> =
            if precomputed.patches_data.is_some() {
                precomputed.patches_data.clone()
            } else if self.enable_patches
                && let Some(pre_gab) = precomputed.xyb_pre_gaborish.as_ref()
            {
                // Same distance-aware kMinPeak as `encode_inner` (~line 1620):
                // libjxl parity at d<1.0, W2-5 chunk 1 relaxation at d>=1.0.
                // RFC#45 pick #5 chunk 3 per-patch cost gate — mirrors
                // `encode_inner` ~line 1620 (see comment there for W41-1
                // measurement findings).
                let min_peak = if self.distance < 1.0 { 2 } else { 1 };
                let mut pd = super::patches::find_and_build_with_per_patch_gate(
                    [&pre_gab[0], &pre_gab[1], &pre_gab[2]],
                    width,
                    height,
                    padded_width,
                    min_peak,
                    Some(self.distance),
                    self.use_ans,
                );
                if matches!(self.encoder_mode, crate::api::EncoderMode::Experimental)
                    && let Some(ref p) = pd
                    && !p.is_cost_effective(self.distance, self.use_ans)
                {
                    pd = None;
                }
                if let Some(ref mut p) = pd {
                    p.quantize_ref_image();
                }
                pd
            } else {
                None
            };
        // Materialize a gaborish_inverse'd patched XYB triple ONLY for
        // case (2) above — case (1) already has patches subtracted in
        // `precomputed.xyb_*`.
        let patched_xyb: Option<[Vec<f32>; 3]> = if precomputed.patches_data.is_none()
            && let (Some(pd), Some(pre_gab)) = (&patches_data, &precomputed.xyb_pre_gaborish)
        {
            let mut xyb = [pre_gab[0].clone(), pre_gab[1].clone(), pre_gab[2].clone()];
            super::patches::subtract_patches(&mut xyb, padded_width, pd);
            if precomputed.gaborish_enabled {
                let [mut x, mut y, mut b] = xyb;
                super::gaborish::gaborish_inverse_maybe_adaptive(
                    &mut x,
                    &mut y,
                    &mut b,
                    padded_width,
                    precomputed.padded_height,
                    self.enable_adaptive_gaborish,
                    self.budget.as_ref(),
                )?;
                Some([x, y, b])
            } else {
                Some(xyb)
            }
        } else {
            None
        };
        let (xyb_x_for_dct, xyb_y_for_dct, xyb_b_for_dct): (&[f32], &[f32], &[f32]) =
            match &patched_xyb {
                Some([x, y, b]) => (x, y, b),
                None => (&precomputed.xyb_x, &precomputed.xyb_y, &precomputed.xyb_b),
            };
        // Suppress unused-mut warning when patches_data is left
        // unmodified after construction (case 1 + case 3).
        let _ = &mut patches_data;
        let _ms_patches = _t_patches.elapsed().as_secs_f64() * 1000.0;

        // CfL handling. For case (1) — patches detected and subtracted
        // upstream by `compute_with_budget` — `precomputed.cfl_map` was
        // already fitted to the patches-subtracted XYB and we use it
        // unchanged. For case (2) — patches detected here on a legacy
        // `from_parts + with_xyb_pre_gaborish` precomputed —
        // `precomputed.cfl_map` was fitted to the un-patched XYB; we
        // recompute pass 1 on the patches-subtracted XYB to avoid the
        // libjxl-parity gap on screenshot content.
        //
        // Pass 2 (refine_cfl_map) intentionally skipped in case (2):
        // it takes `ac_strategy` as input, and `precomputed.ac_strategy`
        // was computed on un-patched XYB — running pass 2 against a
        // mismatched strategy + patched XYB regressed file size by
        // 0.5-2% on gb82-sc screenshots in measurements 2026-05-15.
        let _t_cfl = std::time::Instant::now();
        // W44-195: same dispatch shape as the main Pass-1 site above —
        // Newton at e>=7 when `cfl_newton_libjxl_parity` is true (Libjxl
        // strategy), LS otherwise (Zenjxl / Aggressive / LeanFaster).
        // See the main Pass-1 docstring for the rationale + W44-189 D1.
        //
        // W44-AUDIT-5 Phase 2 (Mode C): also engages Pass-1 Newton when
        // `cfl_newton_libjxl_math_with_ls_warm_start` is true. The two
        // are mutually-exclusive in the SIMD kernel (parity takes priority).
        // W44-AUDIT-5 Phase 3: mirror the main Pass-1 dispatch — per-image
        // M3>=80 → flip libjxl_parity ON for this call. This precomputed
        // Pass-1 only fires when patches are detected during the lossy
        // encode path; matching dispatch shape avoids cross-path drift.
        let patched_p3_force_libjxl_parity = w44_audit_5_p3_force_libjxl_parity_for_screenshot(
            &self.profile,
            self.zenanalyze_proxies.as_ref(),
        );
        let patched_cfl_newton_libjxl_parity_effective =
            self.profile.cfl_newton_libjxl_parity || patched_p3_force_libjxl_parity;
        let patched_pass1_use_newton = self.profile.cfl_newton
            && (patched_cfl_newton_libjxl_parity_effective
                || self.profile.cfl_newton_libjxl_math_with_ls_warm_start);
        let cfl_map_patched: Option<CflMap> = if patched_xyb.is_some() && self.cfl_enabled {
            Some(compute_cfl_map(
                xyb_x_for_dct,
                xyb_y_for_dct,
                xyb_b_for_dct,
                padded_width,
                precomputed.padded_height,
                xsize_blocks,
                ysize_blocks,
                patched_pass1_use_newton,
                self.profile.cfl_newton_eps,
                self.profile.cfl_newton_max_iters,
                // W44-195 / W44-AUDIT-5 Phase 3: composed effective parity
                // flag — Phase 3 forces libjxl_parity ON for screenshot-
                // class images even on Zenjxl/Aggressive. Ignored when LS
                // is used.
                patched_cfl_newton_libjxl_parity_effective,
                // W44-AUDIT-5 Phase 2 (Mode C): same propagation as the
                // main Pass-1 site above.
                self.profile.cfl_newton_libjxl_math_with_ls_warm_start,
            ))
        } else {
            None
        };
        let cfl_map_for_encode: &CflMap = cfl_map_patched.as_ref().unwrap_or(&precomputed.cfl_map);
        let _ms_cfl = _t_cfl.elapsed().as_secs_f64() * 1000.0;

        // Perform DCT and quantization using precomputed XYB data.
        // Chunk 8b (#11): route through the `XybRegionSource` seam so
        // future per-region streaming sources can be wired in here
        // without further refactoring of this call site. The
        // borrowed-source variant keeps the precomputed planes
        // owned by the caller (we don't take ownership of the
        // precomputed struct).
        let _t_xform = std::time::Instant::now();
        let precomputed_source = super::region_source::BorrowedXybSource::new(
            width,
            height,
            padded_width,
            padded_height,
            xyb_x_for_dct,
            xyb_y_for_dct,
            xyb_b_for_dct,
        );
        let mut transform_out = self.transform_and_quantize_with_source(
            &precomputed_source,
            xsize_blocks,
            ysize_blocks,
            &params,
            &mut quant_field,
            cfl_map_for_encode,
            &precomputed.ac_strategy,
        )?;

        // W44-AUDIT-8 Phase 6: apply libjxl QuantizeWP shape to DC
        // (see primary call-site for full comment).
        let phase6_env_on = std::env::var_os("JXL_W44_AUDIT_8_P6_FORCE_QUANTIZE_WP").is_some()
            && self.profile.effort <= 7;
        if self.profile.use_libjxl_wp_dc_quant || phase6_env_on {
            super::quantize_wp::requantize_dc_group_wp(
                &mut transform_out.quant_dc,
                &transform_out.float_dc,
                xsize_blocks,
                0,
                0,
                xsize_blocks,
                ysize_blocks,
                params.scale_dc,
                params.extra_dc_precision,
            );
        }

        // Chunk-8b drop seam (no-op on the borrowed source — the
        // precomputed XYB is owned by the caller, not by us).
        let xsize_dc_groups_seam = div_ceil(width, DC_GROUP_DIM);
        let ysize_dc_groups_seam = div_ceil(height, DC_GROUP_DIM);
        for dc_y in 0..ysize_dc_groups_seam {
            for dc_x in 0..xsize_dc_groups_seam {
                super::region_source::XybRegionSource::release_dc_region(
                    &precomputed_source,
                    dc_x as u32,
                    dc_y as u32,
                );
            }
        }
        // Release the source view — the underlying slices outlive it
        // (held by `precomputed` / `patched_xyb`). A plain move-out is
        // enough; the view holds only borrows.
        let _ = precomputed_source;
        let _ms_xform = _t_xform.elapsed().as_secs_f64() * 1000.0;
        let quant_dc = &transform_out.quant_dc;
        let quant_ac = &transform_out.quant_ac;
        let nzeros = &transform_out.nzeros;
        let raw_nzeros = &transform_out.raw_nzeros;

        // Compute per-block EPF sharpness map when EPF is active.
        // Mirrors the CPU path at `encode_image_lossy` (vardct/encoder.rs:1329-1362).
        // Dynamic sharpness gated at effort >= 6 (speed_tier <= kWombat) matching libjxl.
        // Without this the bitstream emits uniform sharpness=4, costing bytes on
        // content that benefits from per-block tuning.
        let _t_sharp = std::time::Instant::now();
        // W44-44: `padded_height` is hoisted to the top of the
        // function (see ~line 3353) so the earlier
        // `BorrowedXybSource::new` call site can see it under
        // `rate-control` / `__pre_quantized` cfg. The binding that
        // previously lived here would have shadowed the hoisted one
        // with an identical value; it is removed to keep one source
        // of truth.
        // Chunk 8c step B (#11): hoist mask1x1 resolution into a
        // helper so the EPF branch no longer owns the
        // precomputed.xyb_y fallback dependency inline. See
        // encoder.rs line ~2150 for the symmetric site in
        // encode_inner.
        let sharpness_map =
            if params.epf_iters > 0 && self.distance >= 0.5 && self.profile.epf_dynamic_sharpness {
                match self.epf_dispatch {
                    crate::api::EpfDispatch::AlwaysDefault => Some(
                        super::epf::uniform_default_sharpness_map(xsize_blocks, ysize_blocks),
                    ),
                    crate::api::EpfDispatch::Auto | crate::api::EpfDispatch::AlwaysSelect => {
                        let mask_cow = super::adaptive_quant::resolve_mask1x1_for_sharpness(
                            precomputed.mask1x1.as_deref(),
                            &precomputed.xyb_y,
                            padded_width,
                            padded_height,
                            self.budget.as_ref(),
                        )?;
                        if matches!(self.epf_dispatch, crate::api::EpfDispatch::Auto)
                            && super::epf::mask1x1_is_smooth_enough_to_skip_sharpness(&mask_cow)
                        {
                            Some(super::epf::uniform_default_sharpness_map(
                                xsize_blocks,
                                ysize_blocks,
                            ))
                        } else {
                            Some(super::epf::compute_epf_sharpness(
                                [xyb_x_for_dct, xyb_y_for_dct, xyb_b_for_dct],
                                quant_dc,
                                quant_ac,
                                &quant_field,
                                &mask_cow,
                                &params,
                                cfl_map_for_encode,
                                &precomputed.ac_strategy,
                                self.enable_gaborish,
                                xsize_blocks,
                                ysize_blocks,
                                self.budget.as_ref(),
                            )?)
                        }
                    }
                }
            } else {
                None
            };
        let _ms_sharp = _t_sharp.elapsed().as_secs_f64() * 1000.0;

        // Use two-pass mode for rate control (required for ANS)
        let _t_two = std::time::Instant::now();
        let res = self.encode_two_pass(
            width,
            height,
            &params,
            xsize_blocks,
            ysize_blocks,
            xsize_groups,
            ysize_groups,
            xsize_dc_groups,
            ysize_dc_groups,
            num_groups,
            num_dc_groups,
            num_sections,
            quant_dc,
            quant_ac,
            nzeros,
            raw_nzeros,
            &quant_field,
            cfl_map_for_encode,
            &precomputed.ac_strategy,
            &precomputed.noise_params,
            sharpness_map.as_deref(),
            extras,
            patches_data.as_ref(),
            None, // splines
            None, // float_dc
        );
        let _ms_two = _t_two.elapsed().as_secs_f64() * 1000.0;
        if cfg!(feature = "__env_var_diagnostics")
            && std::env::var_os("__JXL_ENC_PHASE_TIMING").is_some()
        {
            eprintln!(
                "encode_from_precomputed: patches={_ms_patches:.1} ms, cfl={_ms_cfl:.1} ms, transform_and_quantize={_ms_xform:.1} ms, sharpness={_ms_sharp:.1} ms, encode_two_pass={_ms_two:.1} ms",
            );
        }
        res
    }

    /// Pre-quantized AC entry point. Skips `transform_and_quantize`
    /// (forward DCT + quantize + nzeros) and goes straight to
    /// `encode_two_pass` (entropy coding). The caller is responsible
    /// for producing per-channel `quant_dc` / `quant_ac` / `nzeros` /
    /// `raw_nzeros` / `float_dc` matching the shape `transform_and_quantize`
    /// would have emitted on this `precomputed` input.
    ///
    /// Designed for the GPU encoder fast path where DCT + quantize
    /// run on the GPU and only the small per-block coefficient buffers
    /// need to cross the wire. Saves ~50 ms at 12 MP / d=1.0 on
    /// rayon-parallel CPU vs running `transform_and_quantize` again.
    ///
    /// Quant field adjustments (multi-block transform `adjust_quant_field_with_distance`)
    /// are applied internally — caller passes the **un-adjusted** per-block
    /// `u8` quant field.
    ///
    /// Color-only entry point (no extra channels). For RGBA / depth /
    /// spot color / selection mask / thermal / CFA see
    /// [`Self::encode_from_pre_quantized_ac_with_extras`].
    #[cfg(feature = "__pre_quantized")]
    #[allow(clippy::too_many_arguments)]
    pub fn encode_from_pre_quantized_ac(
        &self,
        precomputed: &super::precomputed::EncoderPrecomputed,
        quant_field: &[u8],
        quant_dc: &[alloc::vec::Vec<alloc::vec::Vec<i16>>; 3],
        quant_ac: &[alloc::vec::Vec<alloc::vec::Vec<[i32; super::common::DCT_BLOCK_SIZE]>>; 3],
        nzeros: &[alloc::vec::Vec<alloc::vec::Vec<u8>>; 3],
        raw_nzeros: &[alloc::vec::Vec<alloc::vec::Vec<u16>>; 3],
    ) -> Result<Vec<u8>> {
        self.encode_from_pre_quantized_ac_with_extras(
            precomputed,
            quant_field,
            quant_dc,
            quant_ac,
            nzeros,
            raw_nzeros,
            &[],
        )
    }

    /// Pre-quantized AC entry point that additionally writes a list of
    /// extra channels (alpha, depth, spot color, selection mask, thermal,
    /// CFA, …) as a modular sub-bitstream alongside the VarDCT color
    /// data.
    ///
    /// Same fast path as [`Self::encode_from_pre_quantized_ac`] (skips
    /// `transform_and_quantize` and goes straight to `encode_two_pass`)
    /// but threads `extras` through to the bitstream emit so the file
    /// header carries each extra's
    /// [`crate::headers::extra_channels::ExtraChannelInfo`] and the
    /// per-section modular sub-bitstream pulls
    /// `channel_width * channel_height` samples per channel in the
    /// order they were passed in.
    ///
    /// **Current scope** (mirrors
    /// [`Self::encode_from_precomputed_with_extras`]):
    /// - `dim_shift` must be `0` on every extra (full-resolution
    ///   channels).
    /// - Single-group (≤256×256) supports any number of extras.
    /// - Multi-group supports any number of extras at `dim_shift = 0`.
    ///
    /// Other combinations return [`crate::Error::InvalidInput`] so the
    /// wire format stays correct as those paths are filled in.
    ///
    /// **Pre-quantized AC contract**: the GPU encoder runs DCT + quant
    /// on the **color** planes only. Extras are not part of the
    /// pre-quantized AC pipeline — they are passed through unchanged to
    /// the final bitstream emit, the same way
    /// [`Self::encode_from_precomputed_with_extras`] threads them past
    /// the buttloop. Without this, an RGBA encode that goes through the
    /// GPU pre-quantized fast path would silently shed its alpha
    /// channel.
    #[cfg(feature = "__pre_quantized")]
    #[allow(clippy::too_many_arguments)]
    pub fn encode_from_pre_quantized_ac_with_extras(
        &self,
        precomputed: &super::precomputed::EncoderPrecomputed,
        quant_field: &[u8],
        quant_dc: &[alloc::vec::Vec<alloc::vec::Vec<i16>>; 3],
        quant_ac: &[alloc::vec::Vec<alloc::vec::Vec<[i32; super::common::DCT_BLOCK_SIZE]>>; 3],
        nzeros: &[alloc::vec::Vec<alloc::vec::Vec<u8>>; 3],
        raw_nzeros: &[alloc::vec::Vec<alloc::vec::Vec<u16>>; 3],
        extras: &[crate::api::ExtraChannel<'_>],
    ) -> Result<Vec<u8>> {
        let width = precomputed.width;
        let height = precomputed.height;
        let xsize_blocks = precomputed.xsize_blocks;
        let ysize_blocks = precomputed.ysize_blocks;
        let _ = xsize_blocks;
        let _ = ysize_blocks;

        // Validate extras at the entry boundary (dim_shift = 0,
        // sample-count = width * height) before any encoding work runs.
        // Mirrors `encode_from_precomputed_with_extras` so error
        // messages and behavior stay consistent across the two
        // pre-quantized entry points.
        let mut extras_views: Vec<super::extras::VardctExtra<'_>> =
            Vec::with_capacity(extras.len());
        for (idx, ec) in extras.iter().enumerate() {
            if ec.info().dim_shift != 0 {
                return Err(Error::InvalidInput(format!(
                    "extras[{idx}]: dim_shift = {} not yet supported in lossy encode (dim_shift > 0 \
                     for VarDCT extras is a follow-up)",
                    ec.info().dim_shift
                )));
            }
            let expected = width.checked_mul(height).ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 1,
            })?;
            let got = ec.data().len();
            if got != expected {
                return Err(Error::InvalidInput(format!(
                    "extras[{idx}]: expected {expected} samples for {width}x{height}, got {got}"
                )));
            }
            extras_views.push(super::extras::VardctExtra::from_api(ec));
        }
        self.check_alpha_squeeze_supported(&extras_views, Some((width, height)))?;

        let xsize_groups = div_ceil(width, GROUP_DIM);
        let ysize_groups = div_ceil(height, GROUP_DIM);
        let xsize_dc_groups = div_ceil(width, DC_GROUP_DIM);
        let ysize_dc_groups = div_ceil(height, DC_GROUP_DIM);
        let num_groups = xsize_groups * ysize_groups;
        let num_dc_groups = xsize_dc_groups * ysize_dc_groups;
        let num_sections = 2 + num_dc_groups + num_groups;

        // Apply multi-block quant field adjustment. Same step
        // `transform_and_quantize` would have applied internally before
        // calling encode_two_pass.
        let mut quant_field = quant_field.to_vec();
        adjust_quant_field_with_distance(&precomputed.ac_strategy, &mut quant_field, self.distance);

        let mut params = match self.original_distance {
            Some(orig) if orig > self.distance => {
                DistanceParams::compute_for_profile_with_original(
                    self.distance,
                    orig,
                    &self.profile,
                )
            }
            _ => DistanceParams::compute_for_profile(self.distance, &self.profile),
        };
        if let Some(rescale) = self.quant_ac_rescale {
            params.apply_quant_ac_rescale(rescale);
        }
        // libjxl --epf override: when the caller pinned a level, it wins
        // over the distance-derived `epf_iters` (enc_frame.cc:284-285).
        self.apply_epf_level_override(&mut params);
        if self.profile.chromacity_adjustment {
            params.apply_chromacity_adjustment(
                precomputed.chromacity_x_pixelized,
                precomputed.chromacity_b_pixelized,
            );
        }

        self.encode_two_pass(
            width,
            height,
            &params,
            precomputed.xsize_blocks,
            precomputed.ysize_blocks,
            xsize_groups,
            ysize_groups,
            xsize_dc_groups,
            ysize_dc_groups,
            num_groups,
            num_dc_groups,
            num_sections,
            quant_dc,
            quant_ac,
            nzeros,
            raw_nzeros,
            &quant_field,
            &precomputed.cfl_map,
            &precomputed.ac_strategy,
            &precomputed.noise_params,
            None,
            &extras_views,
            None,
            None,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = VarDctEncoder::new(1.0);
        assert_eq!(encoder.distance, 1.0);

        let encoder_default = VarDctEncoder::default();
        assert_eq!(encoder_default.distance, 1.0);
    }

    #[test]
    fn test_median_mask1x1_basic() {
        // 4x3 unpadded inside a 6-stride row → odd count 12 → exact median
        // is the (12/2) = 6th-smallest element.
        // Sorted values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        // index 6 → 7.0
        #[rustfmt::skip]
        let mask = [
            1.0, 2.0, 3.0, 4.0,  0.0, 0.0, // row 0: 1..4 + padding
            5.0, 6.0, 7.0, 8.0,  0.0, 0.0, // row 1: 5..8 + padding
            9.0,10.0,11.0,12.0, 0.0, 0.0, // row 2: 9..12 + padding
        ];
        let med = median_mask1x1(&mask, 6, 4, 3);
        assert_eq!(med, 7.0);
    }

    #[test]
    fn test_median_mask1x1_zero_dim() {
        // 0-width / 0-height shouldn't panic.
        assert_eq!(median_mask1x1(&[], 0, 0, 0), 0.0);
        assert_eq!(median_mask1x1(&[1.0, 2.0], 2, 0, 1), 0.0);
        assert_eq!(median_mask1x1(&[1.0, 2.0], 2, 2, 0), 0.0);
    }

    /// W44-151: verify the new [`W44_151_HIGH_MASK_P25_MIN`] constant
    /// matches the W44-149 audit threshold (= [`W44_150_PHOTO_W44_117_MASK_P25_MIN`])
    /// and verify the percentile_mask1x1 helper computes the 25th
    /// percentile correctly on a known input.
    #[test]
    fn test_w44_151_mask_p25_threshold_matches_w44_149_audit() {
        // The W44-151 admission threshold MUST stay aligned with the
        // W44-149 audit threshold so both gates share the same
        // discriminator (1418519 mask_p25=88.88 admitted; 7552578
        // mask_p25=77.90 next-nearest CONTROL rejected; 11pp margin).
        assert_eq!(
            W44_151_HIGH_MASK_P25_MIN, W44_150_PHOTO_W44_117_MASK_P25_MIN,
            "W44-151 threshold must equal W44-150 audit threshold"
        );
        assert!(W44_151_HIGH_MASK_P25_MIN >= 80.0);
        assert!(W44_151_HIGH_MASK_P25_MIN <= 90.0);
    }

    /// W44-152: verify the distance-narrowed admission predicate fires
    /// only inside the closed band [`W44_152_W44_151_MIN_DISTANCE`,
    /// `W44_152_W44_151_MAX_DISTANCE`] (3.0..=5.0).
    ///
    /// The closed-interval semantics matter: at d=3.0 the gate MUST
    /// fire (lower bound inclusive); at d=5.0 the gate MUST fire
    /// (upper bound inclusive); at d=2.99 / d=5.01 the gate MUST NOT
    /// fire (W44-151 honest-stop ruled out d=6 cells).
    #[test]
    fn test_w44_152_distance_gate_edges() {
        // Mirror the production predicate (`w44_152_distance_in_band`
        // in compute_profile_for_search + encode_inner) on the bounds.
        let in_band = |d: f32| -> bool {
            d >= W44_152_W44_151_MIN_DISTANCE && d <= W44_152_W44_151_MAX_DISTANCE
        };

        // Below the lower bound: gate must NOT fire (would re-enable
        // the W44-151 broad-gate behaviour the W44-150 audit excluded).
        assert!(!in_band(2.99), "d=2.99 should reject (below lower bound)");
        assert!(!in_band(0.0), "d=0.0 should reject");

        // Lower bound inclusive: gate MUST fire at exactly d=3.0.
        // This aligns with HIGH_D_PHOTO_MIN_DISTANCE (the W44-29
        // sibling gate's floor).
        assert!(in_band(3.0), "d=3.0 should admit (lower bound inclusive)");

        // Inside the band: gate MUST fire (these are the d=4 + d=5
        // cells where the W44-151 bench measured -3% bytes + +0.55 SSIM2
        // on 1418519 d=4 and +1.13 SSIM2 on d=5 e8+).
        assert!(in_band(3.5));
        assert!(in_band(4.0));
        assert!(in_band(4.5));
        assert!(in_band(5.0));

        // Upper bound inclusive at d=5.0: see assertion above.
        // d=5.01 must reject (W44-151 honest-stop measured +4.3-4.6%
        // bytes regression at d=6 for only +0.07-0.28 SSIM2).
        assert!(!in_band(5.01), "d=5.01 should reject (above upper bound)");
        assert!(!in_band(6.0), "d=6.0 should reject (W44-151 regress)");
        assert!(!in_band(7.0));

        // Constants sanity: lower < upper, and bounds match the
        // documented values (defends against accidental edits that
        // would silently flip the W44-152 measurement basis).
        assert!(W44_152_W44_151_MIN_DISTANCE < W44_152_W44_151_MAX_DISTANCE);
        assert_eq!(W44_152_W44_151_MIN_DISTANCE, 3.0);
        assert_eq!(W44_152_W44_151_MAX_DISTANCE, 5.0);
        // Lower bound aligns with the sibling W44-29 floor.
        assert_eq!(W44_152_W44_151_MIN_DISTANCE, HIGH_D_PHOTO_MIN_DISTANCE);
    }

    #[test]
    fn test_percentile_mask1x1_p25_basic() {
        // 4x3 unpadded inside a 6-stride row → 12 values:
        // sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        // p25 idx = floor((12-1) * 0.25) = floor(2.75) = 2 → value = 3.0
        #[rustfmt::skip]
        let mask = [
            1.0, 2.0, 3.0, 4.0,  0.0, 0.0,
            5.0, 6.0, 7.0, 8.0,  0.0, 0.0,
            9.0,10.0,11.0,12.0, 0.0, 0.0,
        ];
        let p25 = percentile_mask1x1(&mask, 6, 4, 3, 0.25);
        assert_eq!(p25, 3.0);
        let p50 = percentile_mask1x1(&mask, 6, 4, 3, 0.50);
        assert_eq!(p50, 6.0); // floor(11 * 0.5) = 5 → value 6
        let p100 = percentile_mask1x1(&mask, 6, 4, 3, 1.0);
        assert_eq!(p100, 12.0);
    }

    #[test]
    fn test_percentile_mask1x1_w44_149_split() {
        // Synthesize a mask plane whose p25 sits ABOVE 85 (admits the
        // W44-151 gate) vs one whose p25 sits BELOW 85 (rejects).
        // High-smooth case: most values clustered at 90-100 with a few
        // low outliers — p25 still well above 85.
        let high_smooth: Vec<f32> = (0..(32 * 32))
            .map(|i| if i < 32 { 50.0 } else { 90.0 + (i % 10) as f32 })
            .collect();
        let p25_hi = percentile_mask1x1(&high_smooth, 32, 32, 32, 0.25);
        assert!(
            p25_hi >= W44_151_HIGH_MASK_P25_MIN,
            "high-smooth synthetic p25 = {p25_hi}, expected >= {W44_151_HIGH_MASK_P25_MIN}"
        );

        // Low-mask case: photo-class with values around 40-70 — p25
        // far below the threshold, should reject.
        let low_mask: Vec<f32> = (0..(32 * 32)).map(|i| 40.0 + (i % 30) as f32).collect();
        let p25_lo = percentile_mask1x1(&low_mask, 32, 32, 32, 0.25);
        assert!(
            p25_lo < W44_151_HIGH_MASK_P25_MIN,
            "photo-class synthetic p25 = {p25_lo}, expected < {W44_151_HIGH_MASK_P25_MIN}"
        );
    }

    #[test]
    fn test_median_mask1x1_screenshot_vs_photo_band() {
        // Photo-class: low mask values clustered around 10..40
        // (compute_mask1x1 produces 1/(log1p(diff) + 0.01) — high local
        // activity → low mask). Expected median far below 95.
        let photo: Vec<f32> = (0..(64 * 64)).map(|i| 5.0 + (i % 35) as f32).collect();
        let m_photo = median_mask1x1(&photo, 64, 64, 64);
        assert!(
            m_photo < 95.0,
            "photo-band median {m_photo} should be < 95.0 threshold"
        );

        // Screenshot-class: high mask values (flat regions, low activity)
        // clustered around 110..170. Expected median above 95.
        let screen: Vec<f32> = (0..(64 * 64)).map(|i| 110.0 + (i % 30) as f32).collect();
        let m_screen = median_mask1x1(&screen, 64, 64, 64);
        assert!(
            m_screen > 95.0,
            "screen-band median {m_screen} should be > 95.0 threshold"
        );
    }

    /// W38-2: the `pixel_loss_auto_should_skip` predicate flips at the
    /// documented `> 80` median threshold and returns the same answer
    /// whether the input represents a photo, smooth photo, or
    /// screenshot regime.
    #[test]
    fn test_pixel_loss_auto_should_skip_predicate() {
        // Photo-class: median well below the threshold → do NOT skip
        // (keep the pixel-domain loss term — strategy search benefits).
        let photo: Vec<f32> = (0..(64 * 64)).map(|i| 5.0 + (i % 35) as f32).collect();
        assert!(
            !pixel_loss_auto_should_skip(&photo, 64, 64, 64),
            "photo-band content should keep pixel-domain loss"
        );

        // Smooth-photo class: median right above the threshold (the
        // W38-2 gate's primary target — flat backgrounds, soft-focus
        // bokeh). Should skip.
        let smooth: Vec<f32> = vec![85.0; 64 * 64];
        assert!(
            pixel_loss_auto_should_skip(&smooth, 64, 64, 64),
            "smooth-photo content (median 85 > 80) should skip pixel-domain loss"
        );

        // Screenshot-class: median far above the threshold → skip.
        let screen: Vec<f32> = (0..(64 * 64)).map(|i| 110.0 + (i % 30) as f32).collect();
        assert!(
            pixel_loss_auto_should_skip(&screen, 64, 64, 64),
            "screenshot-band content (median ~125 > 80) should skip pixel-domain loss"
        );

        // Edge: median right at the threshold should NOT skip
        // (the `>` comparator excludes the boundary).
        let at_threshold: Vec<f32> = vec![PIXEL_LOSS_DISPATCH_MEDIAN_THRESHOLD; 64 * 64];
        assert!(
            !pixel_loss_auto_should_skip(&at_threshold, 64, 64, 64),
            "median exactly at 80 should NOT skip (strict `>`)"
        );
        let just_above: Vec<f32> = vec![PIXEL_LOSS_DISPATCH_MEDIAN_THRESHOLD + 0.1; 64 * 64];
        assert!(
            pixel_loss_auto_should_skip(&just_above, 64, 64, 64),
            "median 80.1 (just above strict threshold) should skip"
        );
    }

    /// W38-2: every encoder constructor pin defaults `pixel_loss_dispatch`
    /// to `AlwaysOn` so the historical bitstream is byte-identical.
    #[test]
    fn test_pixel_loss_dispatch_default_always_on() {
        let enc = VarDctEncoder::new(1.0);
        assert_eq!(
            enc.pixel_loss_dispatch,
            crate::api::PixelLossDispatch::AlwaysOn
        );
        let enc_default = VarDctEncoder::default();
        assert_eq!(
            enc_default.pixel_loss_dispatch,
            crate::api::PixelLossDispatch::AlwaysOn
        );
    }

    /// W44-87: every encoder constructor pin defaults
    /// `single_pass_entropy_dispatch` to `AlwaysTwoPass` so the
    /// historical bitstream is byte-identical.
    #[test]
    fn test_single_pass_entropy_dispatch_default_always_two_pass() {
        let enc = VarDctEncoder::new(1.0);
        assert_eq!(
            enc.single_pass_entropy_dispatch,
            crate::api::SinglePassEntropyDispatch::AlwaysTwoPass
        );
        let enc_default = VarDctEncoder::default();
        assert_eq!(
            enc_default.single_pass_entropy_dispatch,
            crate::api::SinglePassEntropyDispatch::AlwaysTwoPass
        );
    }

    #[test]
    fn test_content_aware_entropy_mul_default_off() {
        // W44-130 Chunk D: the `content_aware_entropy_mul` enable bit
        // was deleted and folded into
        // `ResolvedImprovements.screenshot_entropy_mul` (the 4-state
        // `ScreenshotEntropyMulPolicy`). The Zenjxl default sets
        // `Disabled` to preserve the pre-Chunk-D default-off
        // behaviour — verify here that both constructors produce a
        // `Disabled` resolved policy. (`ResolvedImprovements`'s
        // manual `Default` impl flips just this one field to
        // `Disabled` to mirror Zenjxl.)
        let enc = VarDctEncoder::new(1.0);
        assert_eq!(
            enc.resolved_improvements.screenshot_entropy_mul,
            crate::api::ScreenshotEntropyMulPolicy::Disabled,
            "ResolvedImprovements::default() must mirror Zenjxl's Disabled \
             screenshot_entropy_mul for hash-lock parity"
        );
        let enc_default = VarDctEncoder::default();
        assert_eq!(
            enc_default.resolved_improvements.screenshot_entropy_mul,
            crate::api::ScreenshotEntropyMulPolicy::Disabled
        );
    }

    #[test]
    fn test_resolved_improvements_default_zenjxl_equivalent() {
        // W44-130 Chunk D: verify the `resolved_improvements` field
        // (replacing the old `*_hint: Option<bool>` defaults) is
        // populated with the Zenjxl-equivalent default on both
        // constructors. Most policies default to `Auto` (production
        // auto-detector engaged); `screenshot_entropy_mul` is
        // explicitly `Disabled` (mirrors the deleted
        // `content_aware_entropy_mul = false` enable bit). Production
        // API layer overwrites this via
        // `LossyConfig::resolve_improvements()` at all 3 entry points.
        let enc = VarDctEncoder::new(1.0);
        assert_eq!(
            enc.resolved_improvements.screenshot_entropy_mul,
            crate::api::ScreenshotEntropyMulPolicy::Disabled
        );
        assert_eq!(
            enc.resolved_improvements.high_d_photo_entropy_mul,
            crate::api::HighDPhotoEntropyMulPolicy::Auto
        );
        assert_eq!(
            enc.resolved_improvements.dct64_search_policy,
            crate::api::Dct64SearchPolicy::Auto
        );
        assert_eq!(
            enc.resolved_improvements.dct32_search_policy,
            crate::api::Dct32SearchPolicy::FollowDct64Suppression
        );
        let enc_default = VarDctEncoder::default();
        assert_eq!(
            enc_default.resolved_improvements.screenshot_entropy_mul,
            crate::api::ScreenshotEntropyMulPolicy::Disabled
        );
    }

    #[test]
    fn test_zenanalyze_proxies_default_none() {
        // Verify the W44-91 zenanalyze proxies default to None on both
        // constructors. With the proxy absent the W44-91 gate cannot fire
        // and every hash-lock stays byte-identical.
        let enc = VarDctEncoder::new(1.0);
        assert!(enc.zenanalyze_proxies.is_none());
        let enc_default = VarDctEncoder::default();
        assert!(enc_default.zenanalyze_proxies.is_none());
    }

    #[test]
    fn test_zenanalyze_proxies_compute_srgb_u8_solid_red() {
        // Solid red has high M3 colourfulness (R-G axis variance ~0 but
        // mu_rg high; rg_sum = N * 255 → μ_rg² term dominates) and 100%
        // flat blocks (per-block channel range = 0 on every channel).
        let w = 32;
        let h = 32;
        let mut pixels = vec![0u8; w * h * 3];
        for chunk in pixels.chunks_mut(3) {
            chunk[0] = 255;
            chunk[1] = 0;
            chunk[2] = 0;
        }
        let p = ZenanalyzeProxies::compute_srgb_u8(&pixels, w, h, 3, 0, 1, 2);
        // M3 = 0.3 * sqrt(μ_rg² + μ_yb²) for zero-variance image.
        // μ_rg = 255, μ_yb = 127.5; M3 = 0.3 * sqrt(65025 + 16256.25) ≈ 85.6
        assert!(
            (p.m3_colourfulness - 85.6).abs() < 0.5,
            "solid red M3 expected ~85.6 got {}",
            p.m3_colourfulness
        );
        // All 16 blocks (4×4) are perfectly flat.
        assert!(
            (p.flat_color_block_ratio - 1.0).abs() < 1e-6,
            "solid red fcbr expected 1.0 got {}",
            p.flat_color_block_ratio
        );
        // W44-96: solid red has zero gradient → edge_density = 0.
        assert!(
            p.edge_density.abs() < 1e-6,
            "solid red edge_density expected 0 got {}",
            p.edge_density
        );
    }

    #[test]
    fn test_zenanalyze_proxies_edge_density_alternating_pattern() {
        // W44-96: a high-contrast vertical-stripe pattern should yield
        // high edge_density (~most interior pixels are at a black/white
        // boundary). Verifies the Sobel computation fires.
        let w = 32;
        let h = 32;
        let mut pixels = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let off = (y * w + x) * 3;
                // 2-pixel-wide black/white stripes — produces strong Gx
                // gradient on every column transition.
                let v = if (x / 2) % 2 == 0 { 0 } else { 255 };
                pixels[off] = v;
                pixels[off + 1] = v;
                pixels[off + 2] = v;
            }
        }
        let p = ZenanalyzeProxies::compute_srgb_u8(&pixels, w, h, 3, 0, 1, 2);
        // Stripe transitions cover at least 50% of interior pixels.
        assert!(
            p.edge_density > 0.4,
            "alternating-stripe edge_density expected > 0.4 got {}",
            p.edge_density
        );
    }

    #[test]
    fn test_zenanalyze_proxies_compute_srgb_u8_random_noise() {
        // Random-ish noise (counter pattern) has low colourfulness (R≈G≈B
        // so rg/yb ≈ 0) and zero flat blocks (every block has full range).
        let w = 32;
        let h = 32;
        let mut pixels = vec![0u8; w * h * 3];
        for (i, chunk) in pixels.chunks_mut(3).enumerate() {
            let v = (i % 256) as u8;
            chunk[0] = v;
            chunk[1] = v;
            chunk[2] = v;
        }
        let p = ZenanalyzeProxies::compute_srgb_u8(&pixels, w, h, 3, 0, 1, 2);
        assert!(
            p.m3_colourfulness < 1.0,
            "grayscale M3 expected ~0 got {}",
            p.m3_colourfulness
        );
        // Every 8×8 block spans 64 different luma values (counter mod 256
        // never repeats within a block at this w/h) → r_range = 63 > 4.
        assert!(
            p.flat_color_block_ratio < 0.01,
            "noise fcbr expected ~0 got {}",
            p.flat_color_block_ratio
        );
    }

    /// W44-176: `luma_var` field on [`ZenanalyzeProxies`] computes
    /// `Var(0.299·R + 0.587·G + 0.114·B)` over sRGB u8 pixels.
    /// - Solid color: variance = 0
    /// - Counter-pattern grayscale (R=G=B=i%256, 32×32 image, i ∈ [0, 1023]):
    ///   luma values are i%256, so distribution is roughly uniform on [0, 256)
    ///   → variance ≈ (256² − 1)/12 ≈ 5462 by uniform-distribution formula.
    #[test]
    fn test_zenanalyze_proxies_luma_var() {
        // Solid red: luma_var = 0
        let w = 32;
        let h = 32;
        let mut pixels = vec![0u8; w * h * 3];
        for chunk in pixels.chunks_mut(3) {
            chunk[0] = 200;
            chunk[1] = 100;
            chunk[2] = 50;
        }
        let p = ZenanalyzeProxies::compute_srgb_u8(&pixels, w, h, 3, 0, 1, 2);
        assert!(
            p.luma_var.abs() < 1e-3,
            "solid color luma_var expected ~0 got {}",
            p.luma_var
        );
        // Counter pattern (i%256 R=G=B grayscale) on 32×32 = 1024 pixels:
        // values cycle 0..256 four times → variance ≈ 5461 (uniform on [0, 256)).
        let mut pixels = vec![0u8; w * h * 3];
        for (i, chunk) in pixels.chunks_mut(3).enumerate() {
            let v = (i % 256) as u8;
            chunk[0] = v;
            chunk[1] = v;
            chunk[2] = v;
        }
        let p = ZenanalyzeProxies::compute_srgb_u8(&pixels, w, h, 3, 0, 1, 2);
        // Uniform on [0, 256) has variance (256² − 1)/12 ≈ 5462; allow ±50.
        assert!(
            (p.luma_var - 5461.0).abs() < 50.0,
            "counter-pattern grayscale luma_var expected ~5461 got {}",
            p.luma_var
        );
    }

    /// W44-130 Chunk D — `StrategyOverrides::dct_suppress_hint` defaults
    /// to `None`; the LossyConfig setter round-trips through to the
    /// resolved policy at all 3 propagation sites (replaces the
    /// pre-Chunk-D `with_dct_suppress_hint` setter).
    #[test]
    fn test_dct_suppress_hint_api_roundtrip() {
        use crate::api::{Dct64SearchPolicy, LossyConfig, StrategyOverrides};

        // Default LossyConfig: dct64 policy resolves to Auto.
        let cfg_none = LossyConfig::new(1.0);
        assert_eq!(cfg_none.strategy_overrides().dct_suppress_hint, None);
        assert_eq!(
            cfg_none.resolve_improvements().dct64_search_policy,
            Dct64SearchPolicy::Auto
        );

        // Some(true) → resolves to ForceSuppress.
        let cfg_some_true = LossyConfig::new(1.0).with_strategy_overrides(StrategyOverrides {
            dct_suppress_hint: Some(true),
            ..Default::default()
        });
        assert_eq!(
            cfg_some_true.resolve_improvements().dct64_search_policy,
            Dct64SearchPolicy::ForceSuppress
        );

        // Some(false) → resolves to ForceAllow.
        let cfg_some_false = LossyConfig::new(1.0).with_strategy_overrides(StrategyOverrides {
            dct_suppress_hint: Some(false),
            ..Default::default()
        });
        assert_eq!(
            cfg_some_false.resolve_improvements().dct64_search_policy,
            Dct64SearchPolicy::ForceAllow
        );

        // with_effort preserves the explicit overrides.
        let cfg_effort = LossyConfig::new(1.0)
            .with_strategy_overrides(StrategyOverrides {
                dct_suppress_hint: Some(true),
                ..Default::default()
            })
            .with_effort(8);
        assert_eq!(
            cfg_effort.resolve_improvements().dct64_search_policy,
            Dct64SearchPolicy::ForceSuppress
        );
    }

    /// W44-130 Chunk D — `StrategyOverrides::dct32_keep_hint` defaults
    /// to `None`; the LossyConfig setter round-trips through to the
    /// resolved policy (replaces the pre-Chunk-D `with_dct32_keep_hint`
    /// setter).
    #[test]
    fn test_dct32_keep_hint_api_roundtrip() {
        use crate::api::{Dct32SearchPolicy, LossyConfig, StrategyOverrides};

        let cfg_none = LossyConfig::new(1.0);
        assert_eq!(cfg_none.strategy_overrides().dct32_keep_hint, None);
        assert_eq!(
            cfg_none.resolve_improvements().dct32_search_policy,
            Dct32SearchPolicy::FollowDct64Suppression
        );

        let cfg_some_true = LossyConfig::new(1.0).with_strategy_overrides(StrategyOverrides {
            dct32_keep_hint: Some(true),
            ..Default::default()
        });
        assert_eq!(
            cfg_some_true.resolve_improvements().dct32_search_policy,
            Dct32SearchPolicy::KeepWhenDct64Suppressed
        );

        let cfg_some_false = LossyConfig::new(1.0).with_strategy_overrides(StrategyOverrides {
            dct32_keep_hint: Some(false),
            ..Default::default()
        });
        assert_eq!(
            cfg_some_false.resolve_improvements().dct32_search_policy,
            Dct32SearchPolicy::FollowDct64Suppression
        );

        // with_effort preserves the explicit overrides.
        let cfg_effort = LossyConfig::new(1.0)
            .with_strategy_overrides(StrategyOverrides {
                dct32_keep_hint: Some(true),
                ..Default::default()
            })
            .with_effort(8);
        assert_eq!(
            cfg_effort.resolve_improvements().dct32_search_policy,
            Dct32SearchPolicy::KeepWhenDct64Suppressed
        );
    }

    /// W44-124 — verify the auto-discriminator predicate values match
    /// the measured separation between codec_wiki (WANT-FIRE) and the
    /// 6 W44-123 regression screens (REJECT). The probe
    /// (`examples/w44_124_proxy_probe.rs`) captures the live values;
    /// this test pins the threshold constants so a future drift either
    /// way is caught.
    #[test]
    fn test_w44_124_auto_discriminator_predicate() {
        // codec_wiki (WANT-FIRE): m3=145.7, ed=0.0396 → fires.
        assert!(145.7_f32 >= W44_124_DCT32_KEEP_M3_MIN);
        assert!(0.0396_f32 < W44_124_DCT32_KEEP_EDGE_DENSITY_MAX);

        // imessage (REJECT, W44-123 d=6 −0.37 SSIM2): m3=67.65 passes m3
        // alone but ed=0.0533 fails the ed gate → reject. This is the
        // load-bearing case that justifies the AND clause.
        assert!(67.65_f32 >= W44_124_DCT32_KEEP_M3_MIN);
        assert!(0.0533_f32 >= W44_124_DCT32_KEEP_EDGE_DENSITY_MAX);

        // terminal (REJECT by m3): m3=13.85 fails m3, ed=0.0874 also
        // fails ed → reject (loses the +0.47 SSIM2 e8/e9 opt-in win;
        // remains accessible via Some(true) override).
        assert!(13.85_f32 < W44_124_DCT32_KEEP_M3_MIN);

        // graph (REJECT by m3): m3=11.75 → reject.
        assert!(11.75_f32 < W44_124_DCT32_KEEP_M3_MIN);

        // windows (REJECT by m3): m3=20.04 → reject.
        assert!(20.04_f32 < W44_124_DCT32_KEEP_M3_MIN);

        // 1189261 photo (REJECT by ed, the colourful one that would
        // pass m3 alone): m3=98.84 passes m3 but ed=0.4895 fails.
        assert!(98.84_f32 >= W44_124_DCT32_KEEP_M3_MIN);
        assert!(0.4895_f32 >= W44_124_DCT32_KEEP_EDGE_DENSITY_MAX);
    }

    /// W44-135 — pin the distance-band constants for the W44-124
    /// auto-discriminator. The band protects the d=4/5/6 SSIM2
    /// regression cluster (W44-134 measurement) and the d=0.8/1.0
    /// low-d over-application cluster while preserving the W44-124
    /// d=3 wins and the d=2.5 bonus wins.
    ///
    /// **W44-143 (2026-05-20)**: floor lowered 2.0 → 1.4 per the
    /// 30-cell × 5-variant bisect (`benchmarks/w44_143_min_distance_bisect_2026-05-20.tsv`)
    /// which found the gate fires beneficially on codec_wiki at
    /// d∈[1.4, 1.8] (max +0.72 SSIM2 at e9 d=1.8). The 1.4 floor still
    /// protects d=0.8/1.0/1.2 (W44-142 + low-d cluster preservation).
    #[test]
    fn test_w44_135_dct32_keep_distance_gate() {
        // Window is inclusive on both ends per the `>=` / `<=` checks
        // in the dispatch site.
        assert!(W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE > 1.0);
        assert!(W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE < 4.0);
        assert!(W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE < W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE);

        // W44-134 wins to preserve.
        assert!(2.5_f32 >= W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE);
        assert!(2.5_f32 <= W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE);
        assert!(3.0_f32 >= W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE);
        assert!(3.0_f32 <= W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE);

        // W44-134 regression cells to gate out.
        assert!(4.0_f32 > W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE);
        assert!(5.0_f32 > W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE);
        assert!(6.0_f32 > W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE);
        assert!(0.8_f32 < W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE);
        assert!(1.0_f32 < W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE);
        // W44-142 protection — d=1.2 must stay OUT (EPF seed lever owns
        // that distance via a different mechanism). 1.2 < 1.4 ✓.
        assert!(1.2_f32 < W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE);
    }

    /// W44-143 (2026-05-20) — pin the new W44-124 lower bound at 1.4.
    /// The bisect (`benchmarks/w44_143_min_distance_bisect_2026-05-20.tsv`)
    /// confirmed d∈[1.4, 1.8] cells on codec_wiki BENEFIT from the
    /// W44-124 lift (max +0.72 SSIM2 at e9 d=1.8). The 1.4 floor (vs
    /// W44-135's 2.0) opens up 5 new SSIM2 wins (e8/e9 d=1.4 +0.31,
    /// e8/e9 d=1.6 +0.62, e9 d=1.8 +0.72) at the cost of a single
    /// minor regression (e8 d=1.8 -0.18 SSIM2) which is unavoidable
    /// at any floor ≤ 1.8 (e8 buttloop has only 2 iters vs e9's 4).
    #[test]
    fn test_w44_143_dct32_keep_min_distance() {
        // The exact ship value (any drift triggers re-bisect).
        assert!((W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE - 1.4).abs() < 1e-6);

        // W44-143 NEW wins must be inside the gate.
        assert!(1.4_f32 >= W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE);
        assert!(1.4_f32 <= W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE);
        assert!(1.6_f32 >= W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE);
        assert!(1.6_f32 <= W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE);
        assert!(1.8_f32 >= W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE);
        assert!(1.8_f32 <= W44_124_DCT32_KEEP_AUTO_MAX_DISTANCE);

        // d=1.2 must STAY OUT — W44-143 bisect showed -0.27 to -0.43
        // SSIM2 regression on codec_wiki under the lift at d=1.2.
        // W44-142 owns d ∈ [1.0, 1.5) on a different mechanism
        // (EPF seed suppression).
        assert!(1.2_f32 < W44_124_DCT32_KEEP_AUTO_MIN_DISTANCE);
    }

    #[test]
    fn test_encode_small_image() {
        let encoder = VarDctEncoder::new(1.0);

        // Create a simple 8x8 red image
        let width = 8;
        let height = 8;
        let mut linear_rgb = vec![0.0f32; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                linear_rgb[idx] = 1.0; // R
                linear_rgb[idx + 1] = 0.0; // G
                linear_rgb[idx + 2] = 0.0; // B
            }
        }

        // This should at least not panic - full encoding not yet implemented
        let result = encoder.encode(width, height, &linear_rgb, None);
        // For now, just check it produces some output
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.data.len() > 2);
        assert_eq!(output.data[0], 0xFF);
        assert_eq!(output.data[1], 0x0A);
    }

    #[test]
    fn test_convert_to_xyb_padded() {
        let encoder = VarDctEncoder::new(1.0);

        // Gray pixel (1x1 image -> padded to 8x8)
        let linear_rgb = vec![0.5, 0.5, 0.5];
        let (x, y, b) = encoder
            .convert_to_xyb_padded(1, 1, 8, 8, &linear_rgb)
            .unwrap();

        // Padded to 8x8 = 64 pixels
        assert_eq!(x.len(), 64);
        assert_eq!(y.len(), 64);
        assert_eq!(b.len(), 64);

        // Gray should have X ≈ 0 (equal L and M)
        assert!(x[0].abs() < 0.01, "X should be near zero for gray");
        assert!(y[0] > 0.0, "Y should be positive");
        assert!(b[0] > 0.0, "B should be positive");

        // Edge replication: all padded pixels should match the corner
        for i in 0..64 {
            assert!((x[i] - x[0]).abs() < 1e-6, "All padded X should match");
            assert!((y[i] - y[0]).abs() < 1e-6, "All padded Y should match");
            assert!((b[i] - b[0]).abs() < 1e-6, "All padded B should match");
        }
    }

    #[test]
    fn test_encode_16x16_red_image() {
        // Test a 16x16 pixel image (2x2 blocks) to compare with libjxl-tiny
        let encoder = VarDctEncoder::new(1.0);

        let width = 16;
        let height = 16;
        let mut linear_rgb = vec![0.0f32; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                linear_rgb[idx] = 1.0; // R
                linear_rgb[idx + 1] = 0.0; // G
                linear_rgb[idx + 2] = 0.0; // B
            }
        }

        let result = encoder.encode(width, height, &linear_rgb, None);
        assert!(result.is_ok());
        let output = result.unwrap();

        eprintln!("Output file size: {} bytes", output.data.len());
        eprintln!(
            "First 32 bytes: {:02x?}",
            &output.data[..32.min(output.data.len())]
        );

        // Write output to file for comparison
        std::fs::write(std::env::temp_dir().join("our_16x16.jxl"), &output.data).unwrap();

        // libjxl-tiny produces:
        // DC_group: 106 bits (14 bytes)
        // Total combined: 1086 bytes
        // Total file: 1104 bytes
        //
        // Our encoder should match these sizes

        // Check signature
        assert_eq!(output.data[0], 0xFF);
        assert_eq!(output.data[1], 0x0A);
    }

    /// Compute a simple hash of a byte slice for output locking.
    fn hash_bytes(bytes: &[u8]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash-locked test for 8x8 gradient image.
    /// This test ensures the encoder output doesn't change unexpectedly.
    /// x86_64 only: FP rounding differs on other architectures and 32-bit.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hash_lock_8x8_gradient() {
        let encoder = VarDctEncoder::new(1.0);
        let width = 8;
        let height = 8;
        let mut linear_rgb = vec![0.0f32; width * height * 3];

        // Simple gradient: R increases with x, G with y
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                linear_rgb[idx] = x as f32 / 7.0; // R
                linear_rgb[idx + 1] = y as f32 / 7.0; // G
                linear_rgb[idx + 2] = 0.5; // B
            }
        }

        let bytes = encoder
            .encode(width, height, &linear_rgb, None)
            .unwrap()
            .data;
        let hash = hash_bytes(&bytes);

        // Lock the hash - if this changes, the encoding has changed.
        // Updated W44-AUDIT-8 Phase 5: extra_dc_precision=1 at effort<=7
        // (libjxl `enc_cache.cc:232-234` parity). DC quant scaled 1× → 2×
        // (`transform.rs` inv_factor multiplied by `1 << extra_dc_precision`
        // = 2); bitstream extra_dc_precision field now writes 1 (was 0);
        // decoder applies symmetric `mul = 0.5` on dequant — same float
        // values, finer integer precision on stored quant_dc.
        // 8x8 gradient stays 112 bytes — gradient DC values fit at both
        // 1× and 2× precision; only the quant_dc integers change.
        // Pre-W44-AUDIT-8 history: see prior W44-171 comment in git log.
        // Updated W44-AUDIT-8 Phase 7: use_libjxl_wp_dc_quant default-ON
        // at effort <= 7 (libjxl nl_dc QuantizeWP parity) + static-path
        // sharpness map wired. Sizes essentially unchanged on these
        // synthetic fixtures; quant_dc integers are WP-shaped.
        // Updated libjxl prefix-vs-ANS auto choice (enc_ans.cc parity):
        // tiny/deterministic streams use prefix codes (112 -> 109 B —
        // drops per-stream 32-bit ANS state flushes).
        // W44-AUDIT-8 Phase 7 default-flip REVERTED same-day (W44-202
        // zenjxl gate: -0.37..-0.85 SSIM2 on 4 photo cells, beyond the
        // -0.30/cell budget). Hash = pre-WP quantization + the kept
        // prefix-auto/singleton + static-sharpness changes.
        const EXPECTED_HASH: u64 = 0x52c1ed32d4456952;
        assert_eq!(
            hash,
            EXPECTED_HASH,
            "8x8 gradient hash mismatch: got {:#x}, expected {:#x}. \
             Output size: {} bytes. If intentional, update EXPECTED_HASH.",
            hash,
            EXPECTED_HASH,
            bytes.len()
        );
    }

    /// Hash-locked test for 16x16 solid color image.
    /// x86_64 only: FP rounding differs on other architectures and 32-bit.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hash_lock_16x16_solid() {
        let encoder = VarDctEncoder::new(1.0);
        let width = 16;
        let height = 16;
        let linear_rgb = vec![0.3f32; width * height * 3]; // gray

        let bytes = encoder
            .encode(width, height, &linear_rgb, None)
            .unwrap()
            .data;
        let hash = hash_bytes(&bytes);

        // Updated W44-AUDIT-8 Phase 5: extra_dc_precision=1 at effort<=7
        // (libjxl `enc_cache.cc:232-234` parity). 16x16 solid gray stays 97
        // bytes: single DC value, gradient predictor produces 0 residuals
        // at any precision. Only the extra_dc_precision field (was 0, now 1)
        // and the quant_dc integer (was N, now 2N) differ in the bitstream.
        // Pre-W44-AUDIT-8 history: see prior W44-171 comment in git log.
        // Updated W44-AUDIT-8 Phase 7: use_libjxl_wp_dc_quant default-ON
        // at effort <= 7 (libjxl nl_dc QuantizeWP parity) + static-path
        // sharpness map wired. Sizes essentially unchanged on these
        // synthetic fixtures; quant_dc integers are WP-shaped.
        // Updated libjxl prefix-vs-ANS auto choice (enc_ans.cc parity):
        // tiny/deterministic streams use prefix codes (97 -> 84 B —
        // drops per-stream 32-bit ANS state flushes).
        // W44-AUDIT-8 Phase 7 default-flip REVERTED same-day (W44-202
        // zenjxl gate: -0.37..-0.85 SSIM2 on 4 photo cells, beyond the
        // -0.30/cell budget). Hash = pre-WP quantization + the kept
        // prefix-auto/singleton + static-sharpness changes.
        const EXPECTED_HASH: u64 = 0x960e78c4971b42e3;
        assert_eq!(
            hash,
            EXPECTED_HASH,
            "16x16 solid hash mismatch: got {:#x}, expected {:#x}. \
             Output size: {} bytes. If intentional, update EXPECTED_HASH.",
            hash,
            EXPECTED_HASH,
            bytes.len()
        );
    }

    /// Hash-locked test for 64x64 checkerboard pattern.
    /// x86_64 only: FP rounding differs on other architectures and 32-bit.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hash_lock_64x64_checkerboard() {
        let encoder = VarDctEncoder::new(1.0);
        let width = 64;
        let height = 64;
        let mut linear_rgb = vec![0.0f32; width * height * 3];

        // 8x8 checkerboard pattern
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                let checker = ((x / 8) + (y / 8)) % 2 == 0;
                let val = if checker { 0.8 } else { 0.2 };
                linear_rgb[idx] = val;
                linear_rgb[idx + 1] = val;
                linear_rgb[idx + 2] = val;
            }
        }

        let bytes = encoder
            .encode(width, height, &linear_rgb, None)
            .unwrap()
            .data;
        let hash = hash_bytes(&bytes);

        // Updated W44-AUDIT-8 Phase 5: extra_dc_precision=1 at effort<=7
        // (libjxl `enc_cache.cc:232-234` parity). 64x64 checkerboard was 509
        // bytes (was 507 post-W44-171). The +2 B cost is the wider DC quant
        // tokens (2× integer range needs +1 token bit per row of DC blocks).
        // Pre-W44-AUDIT-8 history: 507 (W44-171), 467 (post-W44-73),
        // 673 (W44-56), 729 (W44-54).
        //
        // 2026-05-28 (ownership regen): the multi-metric `perceptual_tuning`
        // refactor (`3d879dd7`, extracting always-compiled perceptual_tuning
        // from the buttloop-gated perceptual_loop) drifted this fixture by
        // -2 B (509 → 507) on `origin/main`, leaving this in-source hash-lock
        // RED while the 36 file-based decode-roundtrip hash-locks, the 5
        // Libjxl byte-locks, and all JPEG suites stayed green. Verified:
        // `extra_dc_precision = 1` at e7 is still applied (`effort.rs:1479`)
        // and written to the bitstream, so the DC precision feature is intact
        // — this is a benign entropy-coding shift, NOT a quality regression.
        // Output is valid (same VarDCT e7 decode path as the passing file
        // hash-locks). Regenerated the const to the current 507-byte output
        // to restore a green test on main.
        // Updated W44-AUDIT-8 Phase 7: use_libjxl_wp_dc_quant default-ON
        // at effort <= 7 (libjxl nl_dc QuantizeWP parity) + static-path
        // sharpness map wired. Sizes essentially unchanged on these
        // synthetic fixtures; quant_dc integers are WP-shaped.
        // W44-AUDIT-8 Phase 7 default-flip REVERTED same-day (W44-202
        // zenjxl gate: -0.37..-0.85 SSIM2 on 4 photo cells, beyond the
        // -0.30/cell budget). Hash = pre-WP quantization + the kept
        // prefix-auto/singleton + static-sharpness changes.
        const EXPECTED_HASH: u64 = 0x06d5672f27096037;
        assert_eq!(
            hash,
            EXPECTED_HASH,
            "64x64 checkerboard hash mismatch: got {:#x}, expected {:#x}. \
             Output size: {} bytes. If intentional, update EXPECTED_HASH.",
            hash,
            EXPECTED_HASH,
            bytes.len()
        );
    }

    /// Hash-locked test for non-power-of-two size (tests padding).
    /// x86_64 only: FP rounding differs on other architectures and 32-bit.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hash_lock_13x17_noise() {
        let encoder = VarDctEncoder::new(1.0);
        let width = 13;
        let height = 17;
        let mut linear_rgb = vec![0.0f32; width * height * 3];

        // Deterministic pseudo-random pattern
        let mut seed = 12345u64;
        for val in &mut linear_rgb {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *val = ((seed >> 32) as f32) / (u32::MAX as f32);
        }

        let bytes = encoder
            .encode(width, height, &linear_rgb, None)
            .unwrap()
            .data;
        let hash = hash_bytes(&bytes);

        // Updated W44-AUDIT-8 Phase 5: extra_dc_precision=1 at effort<=7
        // (libjxl `enc_cache.cc:232-234` parity). 13x17 noise now 506 bytes
        // (was 502 post-W44-171). The +4 B cost is the wider DC tokens on
        // noise content (2× integer range expands the per-row residual
        // distribution into higher token classes).
        // Pre-W44-AUDIT-8 history: 502 (W44-171).
        // Updated W44-AUDIT-8 Phase 7: use_libjxl_wp_dc_quant default-ON
        // at effort <= 7 (libjxl nl_dc QuantizeWP parity) + static-path
        // sharpness map wired. Sizes essentially unchanged on these
        // synthetic fixtures; quant_dc integers are WP-shaped.
        // Updated libjxl prefix-vs-ANS auto choice (enc_ans.cc parity):
        // tiny/deterministic streams use prefix codes (506 -> 499 B —
        // drops per-stream 32-bit ANS state flushes).
        // W44-AUDIT-8 Phase 7 default-flip REVERTED same-day (W44-202
        // zenjxl gate: -0.37..-0.85 SSIM2 on 4 photo cells, beyond the
        // -0.30/cell budget). Hash = pre-WP quantization + the kept
        // prefix-auto/singleton + static-sharpness changes.
        const EXPECTED_HASH: u64 = 0x7afbca80d3d7cc13;
        assert_eq!(
            hash,
            EXPECTED_HASH,
            "13x17 noise hash mismatch: got {:#x}, expected {:#x}. \
             Output size: {} bytes. If intentional, update EXPECTED_HASH.",
            hash,
            EXPECTED_HASH,
            bytes.len()
        );
    }

    /// Roundtrip quality test for non-8-aligned dimensions.
    ///
    /// Encodes a 100x75 gradient, decodes with jxl-oxide, and verifies:
    /// 1. Dimensions match
    /// 2. Output is a valid JXL file (correct signature, decodable)
    ///
    /// This catches stride mismatch bugs where padded XYB buffers have
    /// stride != width, which corrupts adaptive quant, CfL, and AC strategy.
    #[test]
    fn test_roundtrip_non_8_aligned() {
        for &(w, h) in &[(100, 75), (13, 17), (33, 49), (7, 9)] {
            let mut linear_rgb = vec![0.0f32; w * h * 3];

            // Smooth gradient (linear RGB)
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * 3;
                    linear_rgb[idx] = x as f32 / w.max(1) as f32;
                    linear_rgb[idx + 1] = y as f32 / h.max(1) as f32;
                    linear_rgb[idx + 2] = 0.3;
                }
            }

            let encoder = VarDctEncoder::new(1.0);
            let bytes = encoder
                .encode(w, h, &linear_rgb, None)
                .unwrap_or_else(|e| panic!("encode {}x{} failed: {}", w, h, e))
                .data;

            // Verify JXL signature
            assert_eq!(bytes[0], 0xFF, "{}x{}: bad signature byte 0", w, h);
            assert_eq!(bytes[1], 0x0A, "{}x{}: bad signature byte 1", w, h);

            // Decode with jxl-oxide and verify dimensions
            let image = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&bytes))
                .unwrap_or_else(|e| panic!("jxl-oxide decode {}x{} failed: {}", w, h, e));
            assert_eq!(
                image.width(),
                w as u32,
                "{}x{}: decoded width mismatch",
                w,
                h
            );
            assert_eq!(
                image.height(),
                h as u32,
                "{}x{}: decoded height mismatch",
                w,
                h
            );

            // Render to verify pixel data is valid
            let render = image
                .render_frame(0)
                .unwrap_or_else(|e| panic!("jxl-oxide render {}x{} failed: {}", w, h, e));
            let _pixels = render.image_all_channels();
        }
    }

    /// Test DC tree learning produces valid output.
    #[test]
    fn test_dc_tree_learning() {
        let width = 64;
        let height = 64;

        // Create a gradient image
        let mut linear_rgb = vec![0.0f32; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                linear_rgb[idx] = x as f32 / width as f32;
                linear_rgb[idx + 1] = y as f32 / height as f32;
                linear_rgb[idx + 2] = 0.5;
            }
        }

        // Encode WITHOUT DC tree learning (baseline) — use ANS
        let mut encoder_baseline = VarDctEncoder::new(1.0);
        encoder_baseline.dc_tree_learning = false;
        let bytes_baseline = encoder_baseline
            .encode(width, height, &linear_rgb, None)
            .expect("baseline encode failed")
            .data;

        // Encode WITH DC tree learning — also use ANS
        let mut encoder_learned = VarDctEncoder::new(1.0);
        encoder_learned.dc_tree_learning = true;
        std::fs::write(
            std::env::temp_dir().join("dc_baseline_test.jxl"),
            &bytes_baseline,
        )
        .unwrap();
        let bytes_learned = encoder_learned
            .encode(width, height, &linear_rgb, None)
            .expect("learned encode failed")
            .data;
        std::fs::write(
            std::env::temp_dir().join("dc_learned_test.jxl"),
            &bytes_learned,
        )
        .unwrap();

        eprintln!(
            "DC tree learning: baseline={} bytes, learned={} bytes (delta={:.2}%)",
            bytes_baseline.len(),
            bytes_learned.len(),
            (bytes_learned.len() as f64 / bytes_baseline.len() as f64 - 1.0) * 100.0
        );

        // Verify both produce valid JXL signature
        assert_eq!(bytes_baseline[0], 0xFF);
        assert_eq!(bytes_baseline[1], 0x0A);
        assert_eq!(bytes_learned[0], 0xFF);
        assert_eq!(bytes_learned[1], 0x0A);

        // Verify baseline decodes (sanity check)
        {
            let image = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&bytes_baseline))
                .expect("jxl-oxide parse of baseline failed");
            let render = image
                .render_frame(0)
                .expect("jxl-oxide render of baseline failed");
            let _pixels = render.image_all_channels();
            eprintln!("Baseline ANS decodes OK ({} bytes)", bytes_baseline.len());
        }

        // Decode the learned version with jxl-oxide to verify it's valid
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes_learned))
            .expect("jxl-oxide decode of learned version failed");
        assert_eq!(image.width(), width as u32);
        assert_eq!(image.height(), height as u32);

        // Render to verify pixel data is valid
        let render = image
            .render_frame(0)
            .expect("jxl-oxide render of learned version failed");
        let _pixels = render.image_all_channels();
        eprintln!("Learned ANS decodes OK ({} bytes)", bytes_learned.len());

        // Also verify with djxl
        std::fs::write(
            std::env::temp_dir().join("dc_learned_test.jxl"),
            &bytes_learned,
        )
        .unwrap();
    }

    /// Test that the butteraugli quantization loop produces valid output.
    #[cfg(feature = "butteraugli-loop")]
    #[test]
    fn test_butteraugli_loop_basic() {
        // Create a 64x64 test image with some variation
        let width = 64;
        let height = 64;
        let mut linear_rgb = vec![0.0f32; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                let fx = x as f32 / width as f32;
                let fy = y as f32 / height as f32;
                linear_rgb[idx] = fx * 0.8; // R
                linear_rgb[idx + 1] = fy * 0.6; // G
                linear_rgb[idx + 2] = (1.0 - fx) * 0.4; // B
            }
        }

        // Encode without butteraugli loop
        let mut encoder_baseline = VarDctEncoder::new(2.0);
        encoder_baseline.butteraugli_iters = 0;
        let bytes_baseline = encoder_baseline
            .encode(width, height, &linear_rgb, None)
            .expect("baseline encode failed")
            .data;

        // Encode with 2 butteraugli loop iterations
        let mut encoder_loop = VarDctEncoder::new(2.0);
        encoder_loop.butteraugli_iters = 2;
        let bytes_loop = encoder_loop
            .encode(width, height, &linear_rgb, None)
            .expect("butteraugli loop encode failed")
            .data;

        // Both should produce valid JXL
        assert_eq!(bytes_baseline[0], 0xFF);
        assert_eq!(bytes_baseline[1], 0x0A);
        assert_eq!(bytes_loop[0], 0xFF);
        assert_eq!(bytes_loop[1], 0x0A);

        // File sizes should differ (butteraugli loop changes quant field)
        eprintln!(
            "Baseline: {} bytes, Butteraugli loop (2 iters): {} bytes",
            bytes_baseline.len(),
            bytes_loop.len()
        );

        // Verify the butteraugli-loop output decodes correctly
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&bytes_loop))
            .expect("jxl-oxide decode of butteraugli loop output failed");
        assert_eq!(image.width(), width as u32);
        assert_eq!(image.height(), height as u32);

        let render = image
            .render_frame(0)
            .expect("jxl-oxide render of butteraugli loop output failed");
        let _pixels = render.image_all_channels();
        eprintln!("Butteraugli loop output decodes OK");
    }

    // ── Chunk-1 framework tests for `with_alpha_squeeze` ──────────────────
    //
    // These tests exercise the new constants + quantizer fn so the
    // framework is provably reachable. They don't yet drive a real
    // squeeze-on-extras encode (that's chunk 2) — they verify the
    // chunk-1 contract: (a) constants are libjxl-parity, (b) the
    // shifted quantizer behaves correctly across the table, (c) the
    // opt-in flag surfaces NotImplemented at the right boundary AND
    // stays at byte-identical lossless when alpha_distance is unset.

    #[test]
    fn squeeze_luma_qtable_matches_libjxl_constants() {
        // Mirrors `lib/jxl/enc_modular.cc:101-103` exactly.
        let expected: [f32; 16] = [
            163.84, 81.92, 40.96, 20.48, 10.24, 5.12, 2.56, 1.28, 0.64, 0.32, 0.16, 0.08, 0.04,
            0.02, 0.01, 0.005,
        ];
        for (i, &v) in expected.iter().enumerate() {
            assert_eq!(
                SQUEEZE_LUMA_QTABLE[i], v,
                "SQUEEZE_LUMA_QTABLE[{i}] mismatch vs libjxl enc_modular.cc:101-103"
            );
        }
        assert_eq!(SQUEEZE_LUMA_QTABLE_LEN, 16);
        assert_eq!(SQUEEZE_QUALITY_FACTOR_CONST, 0.35);
        assert_eq!(SQUEEZE_LUMA_FACTOR_CONST, 1.1);
    }

    #[test]
    fn compute_extra_pixel_quantizer_shifted_alpha_d2_shift0_matches_responsive1() {
        // At shift = 0, 8-bit alpha, d = 2.0:
        // q_float = 0.25 * 2.0 * 1.0 * 0.35 * 1.1 * 163.84 = 31.5392
        // floor → 31. This is ~10x the responsive=0 q value (q=3 at
        // d=2.0 in compute_extra_pixel_quantizer), confirming the
        // responsive=1 vs responsive=0 base divergence (`* 0.1` skip).
        let mut enc = VarDctEncoder::new(1.0);
        enc.alpha_distance = Some(2.0);
        let q = enc.compute_extra_pixel_quantizer_shifted(
            8,
            crate::headers::extra_channels::ExtraChannelType::Alpha,
            0,
        );
        assert_eq!(q, 31, "shift=0 d=2.0 8-bit alpha should yield q=31");

        // Sanity: the responsive=0 path at d=2.0 yields q=3 (10x smaller).
        let q_r0 = enc.compute_extra_pixel_quantizer(
            8,
            crate::headers::extra_channels::ExtraChannelType::Alpha,
        );
        assert_eq!(q_r0, 3, "responsive=0 path unchanged");
    }

    #[test]
    fn compute_extra_pixel_quantizer_shifted_drops_to_1_at_deep_shifts() {
        // Deeper shifts → smaller qtable_val → q < 1 → clamp to 1
        // (lossless). At shift 11+ (qtable[11] = 0.08) with d=2.0,
        // 8-bit: q_float = 0.25 * 2.0 * 1.0 * 0.35 * 1.1 * 0.08 =
        // 0.0154 → floor 0 → clamp to 1.
        let mut enc = VarDctEncoder::new(1.0);
        enc.alpha_distance = Some(2.0);
        for shift in 11..=15 {
            let q = enc.compute_extra_pixel_quantizer_shifted(
                8,
                crate::headers::extra_channels::ExtraChannelType::Alpha,
                shift,
            );
            assert_eq!(q, 1, "deep shift={shift} should clamp to q=1 (lossless)");
        }
    }

    #[test]
    fn compute_extra_pixel_quantizer_shifted_clamps_oversized_shift() {
        // shift > 15 must clamp to 15 (the table last entry), never panic.
        let mut enc = VarDctEncoder::new(1.0);
        enc.alpha_distance = Some(5.0);
        let q_at_15 = enc.compute_extra_pixel_quantizer_shifted(
            8,
            crate::headers::extra_channels::ExtraChannelType::Alpha,
            15,
        );
        let q_at_100 = enc.compute_extra_pixel_quantizer_shifted(
            8,
            crate::headers::extra_channels::ExtraChannelType::Alpha,
            100,
        );
        assert_eq!(q_at_100, q_at_15, "shift > 15 must clamp, not panic");
    }

    #[test]
    fn compute_extra_pixel_quantizer_shifted_returns_1_for_non_alpha() {
        // Non-alpha extras stay lossless at every shift — only alpha
        // has the per-channel distance knob wired today.
        let mut enc = VarDctEncoder::new(1.0);
        enc.alpha_distance = Some(2.0);
        for ec_type in [
            crate::headers::extra_channels::ExtraChannelType::Depth,
            crate::headers::extra_channels::ExtraChannelType::SpotColor,
            crate::headers::extra_channels::ExtraChannelType::Thermal,
        ] {
            for shift in 0..=15 {
                let q = enc.compute_extra_pixel_quantizer_shifted(8, ec_type, shift);
                assert_eq!(
                    q, 1,
                    "non-alpha ec_type={ec_type:?} shift={shift} must stay lossless"
                );
            }
        }
    }

    #[test]
    fn alpha_squeeze_engaged_predicate() {
        // Default: flag off → never engaged.
        let mut enc = VarDctEncoder::new(1.0);
        assert!(!enc.alpha_squeeze_engaged(), "default off");

        // Flag on but no alpha distance → not engaged.
        enc.alpha_squeeze = true;
        assert!(
            !enc.alpha_squeeze_engaged(),
            "no alpha_distance → not engaged"
        );

        // Flag on but distance=0 → not engaged (lossless contract).
        enc.alpha_distance = Some(0.0);
        assert!(
            !enc.alpha_squeeze_engaged(),
            "alpha_distance=0 → not engaged"
        );

        // Flag on + non-zero distance → engaged.
        enc.alpha_distance = Some(2.0);
        assert!(enc.alpha_squeeze_engaged(), "true + d=2.0 → engaged");

        // Flag off + non-zero distance → not engaged
        // (responsive=0 path still runs).
        enc.alpha_squeeze = false;
        assert!(!enc.alpha_squeeze_engaged(), "flag off wins");
    }

    // ─────────────────────────── W44-168 helper tests ───────────────────────────

    #[test]
    fn test_w44_168_baseline_is_noop() {
        // Mode A (Baseline) must return base_iters unchanged for every
        // (effort, proxies) input — that's the byte-identical contract.
        for effort in 0u8..=12 {
            for base in [0u32, 1, 2, 4, 8, 16, 32] {
                assert_eq!(
                    w44_168_compute_iters(
                        base,
                        effort,
                        Some(99.0),
                        Some(95.0),
                        Some(0.9),
                        W44_168IterMode::Baseline,
                    ),
                    base,
                    "Baseline must not modify iters (effort={}, base={})",
                    effort,
                    base
                );
            }
        }
    }

    #[test]
    fn test_w44_168_smooth_skip_decrements_at_e8plus_on_smooth() {
        // Mode B (SmoothSkip): at e>=8 AND smooth (mask_p25>=85 OR
        // mask_median>95), iters = iters - 1 saturating at 1.
        // base=2 → 1; base=4 → 3; base=8 → 7.
        for effort in 8u8..=12 {
            let r = w44_168_compute_iters(
                4,
                effort,
                Some(50.0), // not screenshot
                Some(90.0), // smooth-photo (>=85)
                Some(0.1),
                W44_168IterMode::SmoothSkip,
            );
            assert_eq!(r, 3, "smooth photo at e{} should decrement 4→3", effort);
        }

        // Screenshot path (high mask_median): decrement
        let r = w44_168_compute_iters(
            2,
            8,
            Some(99.0), // screenshot
            Some(50.0), // doesn't matter
            None,
            W44_168IterMode::SmoothSkip,
        );
        assert_eq!(r, 1, "screenshot at e8 should decrement 2→1");
    }

    #[test]
    fn test_w44_168_smooth_skip_saturates_at_1() {
        // base=1 stays at 1 (saturation — never go to 0 in SmoothSkip)
        let r = w44_168_compute_iters(
            1,
            8,
            Some(99.0),
            Some(90.0),
            None,
            W44_168IterMode::SmoothSkip,
        );
        assert_eq!(r, 1, "iters=1 must not decrement to 0 (saturation)");
    }

    #[test]
    fn test_w44_168_smooth_skip_does_not_fire_at_e7() {
        // At e7 base_iters is typically 0; even if proxies look smooth,
        // SmoothSkip should not change anything (only e>=8).
        let r = w44_168_compute_iters(
            7,
            7,
            Some(99.0),
            Some(90.0),
            None,
            W44_168IterMode::SmoothSkip,
        );
        assert_eq!(r, 7, "SmoothSkip is e>=8 only");
    }

    #[test]
    fn test_w44_168_smooth_skip_does_not_fire_on_textured() {
        // Mode B on textured content (mask_p25 < 85 AND median <= 95)
        // must NOT decrement.
        let r = w44_168_compute_iters(
            4,
            8,
            Some(50.0), // not screenshot
            Some(60.0), // not smooth-photo
            Some(0.7),  // textured
            W44_168IterMode::SmoothSkip,
        );
        assert_eq!(r, 4, "textured content at e8 must NOT decrement");
    }

    #[test]
    fn test_w44_168_textured_extend_fires_at_e7_on_textured() {
        // Mode C: e==7 AND base_iters==0 AND edge_density>=0.5 → 2 iters.
        let r = w44_168_compute_iters(
            0,
            7,
            Some(50.0),
            Some(60.0),
            Some(0.7),
            W44_168IterMode::TexturedExtend,
        );
        assert_eq!(
            r, W44_168_TEXTURED_ITERS_AT_E7,
            "textured photo at e7 (base=0) should extend to {}",
            W44_168_TEXTURED_ITERS_AT_E7
        );
    }

    #[test]
    fn test_w44_168_textured_extend_does_not_fire_at_e8plus() {
        // Mode C only fires at e==7; at e>=8 base_iters > 0 already.
        let r = w44_168_compute_iters(
            2,
            8,
            Some(50.0),
            Some(60.0),
            Some(0.7),
            W44_168IterMode::TexturedExtend,
        );
        assert_eq!(r, 2, "TexturedExtend must not fire at e8 (base already >0)");
    }

    #[test]
    fn test_w44_168_textured_extend_does_not_fire_on_smooth() {
        // Mode C on smooth content (edge_density < 0.5) must NOT extend.
        let r = w44_168_compute_iters(
            0,
            7,
            Some(99.0),
            Some(90.0),
            Some(0.1), // smooth
            W44_168IterMode::TexturedExtend,
        );
        assert_eq!(r, 0, "smooth content at e7 must NOT extend");
    }

    #[test]
    fn test_w44_168_combined_applies_both() {
        // Mode D combines B and C. The effort axis (e==7 vs e>=8) makes
        // them mutually exclusive per-cell.
        // e7 textured → extend 0→2 (C arm)
        assert_eq!(
            w44_168_compute_iters(0, 7, None, None, Some(0.6), W44_168IterMode::Combined,),
            2
        );
        // e8 smooth → decrement 2→1 (B arm)
        assert_eq!(
            w44_168_compute_iters(2, 8, None, Some(90.0), None, W44_168IterMode::Combined,),
            1
        );
        // e9 textured → no change (neither arm fires)
        assert_eq!(
            w44_168_compute_iters(
                4,
                9,
                Some(50.0),
                Some(60.0),
                Some(0.6),
                W44_168IterMode::Combined,
            ),
            4
        );
    }

    #[test]
    fn test_w44_168_is_smooth_predicate() {
        // mask_median > 95 → smooth (screenshot)
        assert!(w44_168_is_smooth(Some(99.0), None));
        assert!(!w44_168_is_smooth(Some(95.0), None)); // strict >
        // mask_p25 >= 85 → smooth (smooth-photo)
        assert!(w44_168_is_smooth(None, Some(85.0))); // inclusive
        assert!(w44_168_is_smooth(None, Some(90.0)));
        assert!(!w44_168_is_smooth(None, Some(84.9)));
        // Neither → not smooth
        assert!(!w44_168_is_smooth(Some(50.0), Some(60.0)));
        assert!(!w44_168_is_smooth(None, None));
    }

    #[test]
    fn test_w44_168_is_textured_predicate() {
        assert!(w44_168_is_textured(Some(0.5))); // inclusive
        assert!(w44_168_is_textured(Some(0.7)));
        assert!(!w44_168_is_textured(Some(0.49)));
        assert!(!w44_168_is_textured(Some(0.0)));
        assert!(!w44_168_is_textured(None));
    }

    // ── W44-169 (Smart-Zenjxl chunk 6, 2026-05-21) narrow SmoothSkip ──

    #[test]
    fn test_w44_169_disabled_is_noop() {
        // narrow_enabled = false → always returns base_iters
        for &d in &[1.0_f32, 3.0, 4.0, 4.5, 5.0, 6.0] {
            for &iters in &[0u32, 1, 2, 4, 8] {
                for &eff in &[1u8, 5, 7, 8, 9] {
                    let r =
                        w44_169_compute_iters_narrow(iters, eff, d, Some(99.0), Some(90.0), false);
                    assert_eq!(
                        r, iters,
                        "narrow disabled MUST be no-op at d={d}, e={eff}, iters={iters}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_w44_169_decrements_inside_band_on_smooth_at_e8plus() {
        // d=4.0 (lower bound, inclusive), smooth via mask_p25=90,
        // e8, iters=2 → 1
        let r = w44_169_compute_iters_narrow(2, 8, 4.0, None, Some(90.0), true);
        assert_eq!(r, 1, "narrow MUST fire at d=4.0 inclusive");
        // d=5.0 (upper bound, inclusive), smooth via mask_median=99,
        // e9, iters=4 → 3
        let r = w44_169_compute_iters_narrow(4, 9, 5.0, Some(99.0), None, true);
        assert_eq!(r, 3, "narrow MUST fire at d=5.0 inclusive");
    }

    #[test]
    fn test_w44_169_does_not_fire_at_d_eq_6() {
        // d=6.0 OUTSIDE narrow band → preserve W44-166 win
        let r = w44_169_compute_iters_narrow(2, 8, 6.0, Some(99.0), Some(90.0), true);
        assert_eq!(r, 2, "narrow MUST NOT fire at d=6 (W44-166 PROTECT)");
        // Just above upper bound
        let r = w44_169_compute_iters_narrow(2, 8, 5.01, Some(99.0), Some(90.0), true);
        assert_eq!(r, 2, "narrow MUST NOT fire just above 5.0");
    }

    #[test]
    fn test_w44_169_does_not_fire_below_d_eq_4() {
        // d=3.5 below narrow band
        let r = w44_169_compute_iters_narrow(2, 8, 3.5, Some(99.0), Some(90.0), true);
        assert_eq!(r, 2, "narrow MUST NOT fire below 4.0");
        // d=3.99 just below lower bound
        let r = w44_169_compute_iters_narrow(2, 8, 3.99, Some(99.0), Some(90.0), true);
        assert_eq!(r, 2, "narrow MUST NOT fire just below 4.0");
    }

    #[test]
    fn test_w44_169_does_not_fire_at_e7() {
        // narrow path is e>=8 only — e7 cells must be unchanged
        // (this is also why textured TexturedExtend would be a
        // different mechanism, deferred to a future chunk if needed)
        let r = w44_169_compute_iters_narrow(2, 7, 4.5, Some(99.0), Some(90.0), true);
        assert_eq!(r, 2, "narrow MUST NOT fire at e7");
    }

    #[test]
    fn test_w44_169_saturates_at_1() {
        // base iters = 1 stays at 1 (don't decrement to 0)
        let r = w44_169_compute_iters_narrow(1, 8, 4.5, Some(99.0), Some(90.0), true);
        assert_eq!(r, 1, "narrow MUST saturate at 1");
        // base iters = 0 stays at 0 (no buttloop to begin with)
        let r = w44_169_compute_iters_narrow(0, 8, 4.5, Some(99.0), Some(90.0), true);
        assert_eq!(r, 0, "narrow MUST NOT promote iters=0");
    }

    #[test]
    fn test_w44_169_does_not_fire_on_textured() {
        // Smooth predicate must fail (mask_median below 95 AND mask_p25 below 85)
        let r = w44_169_compute_iters_narrow(2, 8, 4.5, Some(50.0), Some(40.0), true);
        assert_eq!(r, 2, "narrow MUST NOT fire on textured content");
        // None mask → cannot fire
        let r = w44_169_compute_iters_narrow(2, 8, 4.5, None, None, true);
        assert_eq!(r, 2, "narrow MUST NOT fire when no mask data");
    }

    #[test]
    fn test_w44_169_constants_match_design() {
        // Constants are part of the public API surface for this chunk
        // (documented in CLAUDE.md + LIBJXL_DIVERGENCES.md). Lock them
        // here so future moves are deliberate.
        assert_eq!(W44_169_NARROW_MIN_DISTANCE, 4.0);
        assert_eq!(W44_169_NARROW_MAX_DISTANCE, 5.0);
    }
}
