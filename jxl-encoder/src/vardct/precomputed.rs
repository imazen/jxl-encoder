// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Precomputed encoder state for iterative rate control.
//!
//! This module holds cached computations that don't change between rate control
//! iterations, allowing ~50% time savings per iteration.

use super::ac_strategy::AcStrategyMap;
use super::chroma_from_luma::CflMap;
use super::common::*;
use super::noise::NoiseParams;
use super::patches::PatchesData;

/// Precomputed encoder state that can be reused across rate control iterations.
///
/// These computations are independent of the quant field scaling and don't need
/// to be recomputed when adjusting quantization:
/// - XYB color conversion
/// - Gaborish pre-filter
/// - CfL map
/// - Noise params
/// - Float quant field (pre-scaling)
/// - Masking field
/// - Per-pixel mask (for pixel-domain loss)
/// - AC strategy map
pub struct EncoderPrecomputed {
    /// Original image width in pixels.
    pub width: usize,
    /// Original image height in pixels.
    pub height: usize,
    /// Number of 8x8 blocks in x direction.
    pub xsize_blocks: usize,
    /// Number of 8x8 blocks in y direction.
    pub ysize_blocks: usize,
    /// Padded width (rounded up to block boundary).
    pub padded_width: usize,
    /// Padded height (rounded up to block boundary).
    pub padded_height: usize,

    /// XYB X channel (after gaborish if enabled), padded.
    pub xyb_x: Vec<f32>,
    /// XYB Y channel (after gaborish if enabled), padded.
    pub xyb_y: Vec<f32>,
    /// XYB B channel (after gaborish if enabled), padded.
    pub xyb_b: Vec<f32>,

    /// Original linear RGB data (for butteraugli comparison).
    pub linear_rgb: Vec<f32>,

    /// Chroma-from-luma map.
    pub cfl_map: CflMap,
    /// Noise parameters (if noise synthesis enabled).
    pub noise_params: Option<NoiseParams>,
    /// Float quant field (before scaling by inv_scale).
    pub quant_field_float: Vec<f32>,
    /// Masking field for AC strategy selection.
    pub masking: Vec<f32>,
    /// Per-pixel mask for pixel-domain loss (if enabled).
    pub mask1x1: Option<Vec<f32>>,
    /// AC strategy map.
    pub ac_strategy: AcStrategyMap,

    /// Whether gaborish was applied.
    pub gaborish_enabled: bool,
    /// Distance used for initial quant field computation.
    pub base_distance: f32,
    /// X channel pixel chromacity (max gradient of pre-gaborish XYB X).
    pub chromacity_x_pixelized: u32,
    /// B channel pixel chromacity (from pre-gaborish XYB Y/B).
    pub chromacity_b_pixelized: u32,

    /// Pre-gaborish XYB planes [X, Y, B], padded.
    ///
    /// When `Some`, patches detection in `encode_from_precomputed` runs
    /// on these planes (matching libjxl's pipeline order: noise → patches
    /// → gaborish → DCT). The encoder subtracts patches from this
    /// pre-gaborish XYB, re-applies gaborish_inverse, and DCTs the
    /// result; the patches reference frame stores the original
    /// pre-gaborish patch values, which the decoder adds back to its
    /// gaborish-blurred + EPF-filtered reconstruction (decoder pipeline
    /// per libjxl/lib/jxl/dec_cache.cc:148-194).
    ///
    /// When `None`, patches are disabled in `encode_from_precomputed`
    /// — detecting on post-gaborish XYB and subtracting from
    /// post-gaborish XYB does NOT roundtrip (sharpening halos around
    /// every glyph leave catastrophic butteraugli, e.g. 0.5 → 8.3 on
    /// terminal.png at d=0.5).
    ///
    /// `compute_with_budget` populates this field before running
    /// gaborish so the rate-control path picks up screenshot wins
    /// automatically. Callers using `from_parts` (jxl-encoder-gpu's
    /// precomputed paths) should populate via
    /// [`Self::with_xyb_pre_gaborish`] before handing the struct to
    /// `encode_from_precomputed`.
    pub xyb_pre_gaborish: Option<[Vec<f32>; 3]>,

    /// Pre-detected patches dictionary, computed on the patches-subtracted
    /// pre-gaborish XYB (matches libjxl `enc_heuristics.cc:1057-1065`:
    /// patches detected after splines, before InitialQuantField → Gaborish
    /// → CfL → ACS).
    ///
    /// When `Some`, all subsequent precomputed state (`quant_field_float`,
    /// `masking`, `cfl_map`, `mask1x1`, `ac_strategy`) was fitted to the
    /// patches-subtracted XYB — `xyb_x` / `xyb_y` / `xyb_b` already have
    /// patches subtracted. `encode_from_precomputed` writes these patches
    /// into the bitstream and skips its own internal patches detection.
    ///
    /// When `None`, no patches were detected (or patches detection was
    /// disabled). `encode_from_precomputed` falls back to its
    /// `xyb_pre_gaborish` based detection for backwards compatibility
    /// with `from_parts` callers (jxl-encoder-gpu) that haven't yet
    /// migrated to host-side patches detection.
    ///
    /// `pub(crate)` because `PatchesData` itself is internal — outside
    /// callers cannot construct or inspect it.
    pub(crate) patches_data: Option<PatchesData>,
}

impl EncoderPrecomputed {
    /// Compute precomputed state from linear RGB input.
    ///
    /// This performs all computations that are independent of the final
    /// quant field scaling:
    /// - XYB conversion with edge-replicated padding
    /// - Gaborish inverse (if enabled)
    /// - Noise estimation and optional denoising (if enabled)
    /// - Float quant field and masking
    /// - CfL map
    /// - Per-pixel mask (if pixel-domain loss enabled)
    /// - AC strategy selection
    ///
    /// Public entry point. Equivalent to [`Self::compute_with_budget`] with no allocation cap.
    ///
    /// Patches detection is disabled in this entry point (no `enable_patches` parameter).
    /// Callers that want patches must use the internal `compute_with_budget` variant.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        distance: f32,
        cfl_enabled: bool,
        ac_strategy_enabled: bool,
        pixel_domain_loss: bool,
        enable_noise: bool,
        enable_denoise: bool,
        enable_gaborish: bool,
        force_strategy: Option<u8>,
        profile: &crate::effort::EffortProfile,
        color_encoding: Option<&crate::headers::color_encoding::ColorEncoding>,
    ) -> crate::error::Result<Self> {
        Self::compute_with_budget(
            width,
            height,
            linear_rgb,
            distance,
            cfl_enabled,
            ac_strategy_enabled,
            pixel_domain_loss,
            enable_noise,
            enable_denoise,
            enable_gaborish,
            /* enable_adaptive_gaborish */ false,
            /* enable_patches */ false,
            /* use_ans */ true,
            crate::api::EncoderMode::Reference,
            force_strategy,
            profile,
            color_encoding,
            None,
        )
    }

    /// Internal-only variant that accounts allocations against an optional
    /// per-encode [`MemoryBudget`].
    ///
    /// `enable_patches` enables patches detection in the precompute pipeline.
    /// When patches are detected, they are subtracted from the pre-gaborish
    /// XYB BEFORE quant_field / mask / gaborish / CfL / AC strategy are
    /// computed. This matches libjxl's pipeline order
    /// (`enc_heuristics.cc:1057-1194`) and is required for correct
    /// rate-distortion behavior on screenshot content.
    ///
    /// `use_ans` and `encoder_mode` are forwarded to the patches
    /// `is_cost_effective` gate (mirroring the logic in
    /// `vardct/encoder.rs::encode_inner`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_with_budget(
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        distance: f32,
        cfl_enabled: bool,
        ac_strategy_enabled: bool,
        pixel_domain_loss: bool,
        enable_noise: bool,
        enable_denoise: bool,
        enable_gaborish: bool,
        enable_adaptive_gaborish: bool,
        enable_patches: bool,
        use_ans: bool,
        encoder_mode: crate::api::EncoderMode,
        force_strategy: Option<u8>,
        profile: &crate::effort::EffortProfile,
        color_encoding: Option<&crate::headers::color_encoding::ColorEncoding>,
        budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    ) -> crate::error::Result<Self> {
        use super::ac_strategy::compute_ac_strategy;
        // adaptive_quant helpers are referenced through their module path below.
        use super::chroma_from_luma::compute_cfl_map;
        use super::gaborish::gaborish_inverse_maybe_adaptive;
        use super::noise::{denoise_xyb, estimate_noise_params, noise_quality_coef};

        assert_eq!(linear_rgb.len(), width * height * 3);

        // Calculate dimensions
        let xsize_blocks = div_ceil(width, BLOCK_DIM);
        let ysize_blocks = div_ceil(height, BLOCK_DIM);
        let padded_width = xsize_blocks * BLOCK_DIM;
        let padded_height = ysize_blocks * BLOCK_DIM;

        // Convert to XYB with edge-replicated padding
        let (mut xyb_x, mut xyb_y, mut xyb_b) = convert_to_xyb_padded(
            width,
            height,
            padded_width,
            padded_height,
            linear_rgb,
            color_encoding,
            budget,
        )?;

        // Estimate noise parameters (if enabled)
        let noise_params = if enable_noise {
            let quality_coef = noise_quality_coef(distance);
            let params = estimate_noise_params(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                quality_coef,
            );

            // Apply denoising pre-filter if enabled
            if enable_denoise && let Some(ref p) = params {
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

        // Snapshot pre-gaborish XYB so the rate-control path's
        // `encode_from_precomputed` can run patches detection against
        // it. Patches MUST be detected on the unsharpened XYB to
        // roundtrip correctly through the decoder's
        // `IDCT → gaborish → EPF → patches` pipeline (see field doc on
        // `xyb_pre_gaborish` and the matching block in
        // `vardct/encoder.rs::encode_from_precomputed`).
        //
        // The clone is ~3 × `padded_width × padded_height × 4` bytes
        // (~16 MB at 12 MP). Skipped when gaborish is off because
        // post-gaborish == pre-gaborish in that case (encoder uses
        // `precomputed.xyb_*` directly when this field is `None`).
        let xyb_pre_gaborish: Option<[Vec<f32>; 3]> = if enable_gaborish {
            crate::budget::MemoryBudget::reserve_permanent_opt(
                budget,
                (padded_width as u64)
                    .saturating_mul(padded_height as u64)
                    .saturating_mul(4 * 3),
            )?;
            Some([xyb_x.clone(), xyb_y.clone(), xyb_b.clone()])
        } else {
            None
        };

        // Patches detection on PRE-gaborish XYB. libjxl pipeline order
        // (enc_heuristics.cc:1057-1194):
        //   noise → splines → patches detect/subtract → InitialQuantField
        //   → GaborishInverse → CfL pass 1 → AC strategy → CfL pass 2 → DCT
        //
        // Patches MUST be detected and subtracted BEFORE quant_field /
        // mask / gaborish / CfL / AC strategy so that ALL downstream
        // computations see the patches-subtracted XYB. Without this, on
        // screenshot content (text, UI) the quant_field is fitted to
        // sharp text edges that PATCHES will absorb — producing
        // over-aggressive quantization when the actual residual is
        // smooth, and CfL is fitted to the dominant low-frequency luma
        // that patches will remove — producing chroma residuals that
        // don't match the actual encoded geometry.
        //
        // The decoder pipeline (dec_cache.cc:148-194) reverses this:
        //   IDCT → gaborish → EPF → ChannelUpsampling → add patches
        // — patches are added back AFTER gaborish, so the patches
        // reference frame must store the PRE-gaborish patch values
        // (which is what `find_and_build` extracts from the
        // pre-gaborish XYB here).
        // Distance-aware kMinPeak (W3-1 / commit 4fb0f52): libjxl
        // parity (=2) below d=1.0, W2-5 chunk 1 relaxation (=1) at
        // d>=1.0. See `vardct/encoder.rs::encode_inner` for why
        // RFC#45 chunk 3's per-patch gate does NOT lower this.
        let min_peak = if distance < 1.0 { 2 } else { 1 };
        // RFC#45 pick #5 chunk 3 per-patch cost gate — mirrors
        // `vardct/encoder.rs::encode_inner` (see comment there).
        let mut patches_data = if enable_patches {
            super::patches::find_and_build_with_per_patch_gate(
                [&xyb_x, &xyb_y, &xyb_b],
                width,
                height,
                padded_width,
                min_peak,
                Some(distance),
                use_ans,
            )
        } else {
            None
        };
        // Cost-benefit gating for experimental mode only — libjxl uses
        // patches unconditionally when detected, so reference mode
        // skips this to match (mirrors `encode_inner` line ~750).
        if matches!(encoder_mode, crate::api::EncoderMode::Experimental)
            && let Some(ref pd) = patches_data
            && !pd.is_cost_effective(distance, use_ans)
        {
            patches_data = None;
        }
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

        // Compute pixel chromacity stats AFTER patches subtract, BEFORE
        // gaborish (mirrors `encode_inner` ordering at vardct/encoder.rs
        // line ~850 — chromacity is on the patches-subtracted PRE-gab
        // XYB so the X/B channel pixelization metric reflects the
        // chroma the encoder will actually quantize). Gated at effort
        // >= 7 to skip the full-image gradient scan at low effort.
        let (chromacity_x_pixelized, chromacity_b_pixelized) = if profile.chromacity_adjustment {
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

        // Compute adaptive per-block quantization field and masking on
        // PRE-gaborish XYB. libjxl computes InitialQuantField BEFORE
        // GaborishInverse (`enc_heuristics.cc:1117-1142`, comment:
        // "relies on pre-gaborish values"). Gaborish sharpening
        // inflates gradients which inflates masking → smaller quant
        // values → finer quantization → more bits.
        //
        // When gaborish is off, scale distance by 0.62 for the quant
        // field (matches libjxl `enc_heuristics.cc:1119`).
        let distance_for_iqf = if enable_gaborish {
            distance
        } else {
            distance * 0.62
        };

        let (quant_field_float, masking) =
            super::adaptive_quant::compute_quant_field_float_with_budget(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                distance_for_iqf,
                profile.k_ac_quant,
                budget,
            )?;

        // Compute per-pixel mask for pixel-domain loss on PRE-gaborish
        // XYB (matches libjxl `InitialQuantField` which produces
        // `initial_quant_masking1x1` before `GaborishInverse`).
        let mask1x1 = if ac_strategy_enabled && pixel_domain_loss {
            Some(super::adaptive_quant::compute_mask1x1_with_budget(
                &xyb_y,
                padded_width,
                padded_height,
                budget,
            )?)
        } else {
            None
        };

        // Apply gaborish inverse (5x5 sharpening) on patches-subtracted
        // XYB AFTER quant_field / mask1x1.
        if enable_gaborish {
            gaborish_inverse_maybe_adaptive(
                &mut xyb_x,
                &mut xyb_y,
                &mut xyb_b,
                padded_width,
                padded_height,
                enable_adaptive_gaborish,
                budget,
            )?;
        }

        // Compute CfL map on POST-gaborish patches-subtracted XYB.
        let cfl_map = if cfl_enabled {
            compute_cfl_map(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                profile.cfl_newton,
                profile.cfl_newton_eps,
                profile.cfl_newton_max_iters,
            )
        } else {
            CflMap::zeros(
                div_ceil(xsize_blocks, TILE_DIM_IN_BLOCKS),
                div_ceil(ysize_blocks, TILE_DIM_IN_BLOCKS),
            )
        };

        // Compute AC strategy
        let ac_strategy = if let Some(forced) = force_strategy {
            AcStrategyMap::force_strategy(xsize_blocks, ysize_blocks, forced)
        } else if !ac_strategy_enabled {
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
                distance,
                &quant_field_float,
                &masking,
                &cfl_map,
                mask1x1.as_deref(),
                padded_width,
                profile,
            )
        };

        // CfL pass 2 refinement happens in encoder.rs after the butteraugli loop
        // produces the final quant_field. No refinement here — pass 1 values from
        // compute_cfl_map are sufficient for initial AC strategy selection.

        Ok(Self {
            width,
            height,
            xsize_blocks,
            ysize_blocks,
            padded_width,
            padded_height,
            xyb_x,
            xyb_y,
            xyb_b,
            linear_rgb: linear_rgb.to_vec(),
            cfl_map,
            noise_params,
            quant_field_float,
            masking,
            mask1x1,
            ac_strategy,
            gaborish_enabled: enable_gaborish,
            base_distance: distance,
            chromacity_x_pixelized,
            chromacity_b_pixelized,
            xyb_pre_gaborish,
            patches_data,
        })
    }

    /// Construct an `EncoderPrecomputed` from caller-supplied parts —
    /// every field is the responsibility of the caller. Intended for
    /// downstream consumers (e.g. jxl-encoder-gpu) that have already
    /// run XYB conversion / gaborish / CfL / strat-search / masking on
    /// their own pipeline (in our case, the GPU) and want to hand the
    /// result to [`super::encoder::VarDctEncoder::encode_from_precomputed`]
    /// directly, skipping the encoder's own re-computation.
    ///
    /// The caller is responsible for layout consistency:
    /// - `xyb_x` / `xyb_y` / `xyb_b` MUST each have
    ///   `padded_width * padded_height` entries, in row-major order,
    ///   with edge-replicated padding to the block boundary.
    /// - `linear_rgb` is interleaved RGB (`width * height * 3` entries),
    ///   used by the rate-control loop's butteraugli measurement; pass
    ///   an empty `Vec` if the caller will not run rate-control on top.
    /// - `cfl_map` covers the padded image at 8×8 block resolution.
    /// - `quant_field_float` and `masking` are per-8×8-block
    ///   (`xsize_blocks * ysize_blocks` entries each).
    /// - `mask1x1` is per-pixel padded
    ///   (`Some(padded_width * padded_height)` entries) when the
    ///   pixel-domain loss term is enabled, else `None`.
    /// - `ac_strategy` is sized by the caller to cover the padded grid.
    ///
    /// **No validation is performed** — wrong sizes here will panic in
    /// the encoder downstream (or worse, produce a garbled bitstream).
    /// Gated behind the `__pre_quantized` cargo feature, which is
    /// `#[doc(hidden)]` and not part of the stable API.
    #[cfg(feature = "__pre_quantized")]
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        width: usize,
        height: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        padded_width: usize,
        padded_height: usize,
        xyb_x: Vec<f32>,
        xyb_y: Vec<f32>,
        xyb_b: Vec<f32>,
        linear_rgb: Vec<f32>,
        cfl_map: CflMap,
        noise_params: Option<NoiseParams>,
        quant_field_float: Vec<f32>,
        masking: Vec<f32>,
        mask1x1: Option<Vec<f32>>,
        ac_strategy: AcStrategyMap,
        gaborish_enabled: bool,
        base_distance: f32,
        chromacity_x_pixelized: u32,
        chromacity_b_pixelized: u32,
    ) -> Self {
        Self {
            width,
            height,
            xsize_blocks,
            ysize_blocks,
            padded_width,
            padded_height,
            xyb_x,
            xyb_y,
            xyb_b,
            linear_rgb,
            cfl_map,
            noise_params,
            quant_field_float,
            masking,
            mask1x1,
            ac_strategy,
            gaborish_enabled,
            base_distance,
            chromacity_x_pixelized,
            chromacity_b_pixelized,
            xyb_pre_gaborish: None,
            patches_data: None,
        }
    }

    /// Attach the pre-gaborish XYB triple so
    /// [`super::encoder::VarDctEncoder::encode_from_precomputed`] can
    /// run patches detection against it.
    ///
    /// The three planes MUST each have `padded_width * padded_height`
    /// entries laid out the same way `xyb_x` is — they're literally the
    /// values the upstream pipeline had on hand BEFORE running the
    /// 5x5 gaborish_inverse sharpening filter. When unset, patches are
    /// silently disabled in `encode_from_precomputed` (callers
    /// constructing via `from_parts` who don't carry pre-gab XYB take
    /// the no-patches hit on screenshot content; closing the gap
    /// requires either preserving pre-gab on the GPU side or a separate
    /// host-side download just for patches detection).
    ///
    /// Building only the Y channel is not allowed — patches detection
    /// needs all three (the L1 distance metric in
    /// `find_text_like_patches` reads X and B for color similarity).
    ///
    /// Gated behind the `__pre_quantized` cargo feature for the same
    /// reason as `from_parts` — `#[doc(hidden)]` and not part of the
    /// stable API.
    #[cfg(feature = "__pre_quantized")]
    #[doc(hidden)]
    pub fn with_xyb_pre_gaborish(mut self, xyb: [Vec<f32>; 3]) -> Self {
        debug_assert_eq!(xyb[0].len(), self.padded_width * self.padded_height);
        debug_assert_eq!(xyb[1].len(), self.padded_width * self.padded_height);
        debug_assert_eq!(xyb[2].len(), self.padded_width * self.padded_height);
        self.xyb_pre_gaborish = Some(xyb);
        self
    }

    /// Attach a pre-detected, pre-quantized [`super::patches::PatchesData`]
    /// so [`super::encoder::VarDctEncoder::encode_from_precomputed`]
    /// hits the case-1 path: it writes patches into the bitstream
    /// directly and skips its in-function re-detection (which would
    /// otherwise force a CfL pass-1 recompute on a mismatched
    /// `ac_strategy`).
    ///
    /// Contract for case-1 parity (mirrors what `compute_with_budget`
    /// does on the rate-control path):
    /// 1. Detect patches on the PRE-gaborish XYB via
    ///    [`super::patches::find_and_build`].
    /// 2. `quantize_ref_image()` on the resulting `PatchesData` so the
    ///    encoder's subtract matches the decoder's add bit-for-bit.
    /// 3. `subtract_patches(...)` on BOTH the pre-gaborish XYB and the
    ///    post-gaborish XYB (gaborish is linear; subtracting the same
    ///    quantized values from both planes after a 5x5 gaborish_inverse
    ///    is equivalent to subtracting from pre-gab and re-running the
    ///    filter, but avoids the redundant convolution).
    /// 4. Recompute `cfl_map` (and ideally `quant_field` / `masking` /
    ///    `ac_strategy`) on the patches-subtracted XYB so all
    ///    downstream precomputed state matches what the bitstream
    ///    decode will see. At minimum, recompute `cfl_map` — the
    ///    encoder's case-2 fallback path does this much, so case 1
    ///    must do at least as much.
    /// 5. Hand the patches-subtracted post-gab XYB / new `cfl_map` to
    ///    [`Self::from_parts`], call this method with the
    ///    `PatchesData` from step 1, then optionally
    ///    [`Self::with_xyb_pre_gaborish`] (the encoder ignores
    ///    `xyb_pre_gaborish` once `patches_data` is set, but supplying
    ///    it keeps the precomputed self-consistent for any future
    ///    invariants).
    ///
    /// **No validation** beyond non-empty positions — wrong inputs
    /// produce a corrupt bitstream. Gated behind `__pre_quantized` for
    /// the same reason as [`Self::from_parts`].
    #[cfg(feature = "__pre_quantized")]
    #[doc(hidden)]
    pub fn with_patches_data(mut self, patches: super::patches::PatchesData) -> Self {
        debug_assert!(
            !patches.positions.is_empty(),
            "with_patches_data: PatchesData must have at least one position; \
             pass `None` (don't call this method) if no patches were detected"
        );
        self.patches_data = Some(patches);
        self
    }
}

/// Convert linear RGB to XYB color space with padding to block boundaries.
///
/// If `primaries` is non-sRGB, applies a 3x3 matrix to convert to sRGB primaries
/// before the XYB transform (the opsin matrix is defined for sRGB/BT.709).
fn convert_to_xyb_padded(
    width: usize,
    height: usize,
    padded_width: usize,
    padded_height: usize,
    linear_rgb: &[f32],
    color_encoding: Option<&crate::headers::color_encoding::ColorEncoding>,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    use super::xyb::primaries_to_srgb_matrix;
    use crate::color::xyb::linear_rgb_to_xyb;

    let primaries_matrix = color_encoding.and_then(primaries_to_srgb_matrix);

    let padded_n =
        padded_width
            .checked_mul(padded_height)
            .ok_or(crate::error::Error::DimensionOverflow {
                width: padded_width,
                height: padded_height,
                channels: 3,
            })?;
    // Three padded XYB planes are returned to caller — account permanently.
    crate::budget::MemoryBudget::reserve_permanent_opt(
        budget,
        (padded_n as u64).saturating_mul(4 * 3),
    )?;
    // Output planes are fully overwritten below: rows 0..height by the per-row
    // conversion + right-edge pad, rows height..padded_height by the bottom-pad
    // loop. Safe to dirty-initialize.
    let mut xyb_x = jxl_simd::vec_f32_dirty(padded_n);
    let mut xyb_y = jxl_simd::vec_f32_dirty(padded_n);
    let mut xyb_b = jxl_simd::vec_f32_dirty(padded_n);

    // Scratch buffers for deinterleaving + optional matrix transform. These
    // are written in full every row before being read. Transient — released
    // before this function returns.
    let _row_g =
        crate::budget::MemoryBudget::reserve_opt(budget, (width as u64).saturating_mul(4 * 3))?;
    let mut row_r = jxl_simd::vec_f32_dirty(width);
    let mut row_g = jxl_simd::vec_f32_dirty(width);
    let mut row_b = jxl_simd::vec_f32_dirty(width);

    // Convert the actual image pixels
    for y in 0..height {
        let src_row = y * width;
        for x in 0..width {
            let si = (src_row + x) * 3;
            row_r[x] = linear_rgb[si];
            row_g[x] = linear_rgb[si + 1];
            row_b[x] = linear_rgb[si + 2];
        }

        if let Some(ref m) = primaries_matrix {
            super::xyb::apply_matrix_3x3(&mut row_r, &mut row_g, &mut row_b, m);
        }

        let dst_row = y * padded_width;
        for x in 0..width {
            let (xv, yv, bv) = linear_rgb_to_xyb(row_r[x], row_g[x], row_b[x]);
            xyb_x[dst_row + x] = xv;
            xyb_y[dst_row + x] = yv;
            xyb_b[dst_row + x] = bv;
        }

        // Pad right edge with last pixel value
        if padded_width > width {
            let last_x_idx = y * padded_width + (width - 1);
            let last_x = xyb_x[last_x_idx];
            let last_y = xyb_y[last_x_idx];
            let last_b = xyb_b[last_x_idx];
            for x in width..padded_width {
                let dst_idx = y * padded_width + x;
                xyb_x[dst_idx] = last_x;
                xyb_y[dst_idx] = last_y;
                xyb_b[dst_idx] = last_b;
            }
        }
    }

    // Pad bottom rows by copying the last row
    if padded_height > height {
        let last_row_start = (height - 1) * padded_width;
        for y in height..padded_height {
            let dst_row_start = y * padded_width;
            for x in 0..padded_width {
                xyb_x[dst_row_start + x] = xyb_x[last_row_start + x];
                xyb_y[dst_row_start + x] = xyb_y[last_row_start + x];
                xyb_b[dst_row_start + x] = xyb_b[last_row_start + x];
            }
        }
    }

    Ok((xyb_x, xyb_y, xyb_b))
}
