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
    ///
    /// **Implementation (#11 chunk 2, 2026-05-18):** internally calls
    /// [`EncoderPrecomputedGlobal::compute_global_only`] then runs the
    /// per-DC-group pipeline ([`fill_dc_group_state_whole_image`]) over
    /// the whole image as a single region. Bit-identical to the prior
    /// monolithic implementation. The split exists so chunks 3+ can
    /// stream `compute_dc_group` over real DC-group-sized windows
    /// without disturbing the rate-control / butteraugli-loop
    /// consumers, which still see the same assembled
    /// [`EncoderPrecomputed`].
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
        // Step 1: compute the truly global state (XYB convert, noise,
        // patches, chromacity stats, pre-gaborish snapshot). This is
        // the part of the pipeline that fundamentally needs to see the
        // whole image — patches detection runs an 8-connected BFS that
        // can span the entire image, noise estimation aggregates
        // statistics across all pixels, and XYB conversion allocates
        // whole-image planes that later steps mutate in place.
        let mut global = EncoderPrecomputedGlobal::compute_global_only(
            width,
            height,
            linear_rgb,
            distance,
            enable_noise,
            enable_denoise,
            enable_gaborish,
            enable_patches,
            use_ans,
            encoder_mode,
            profile,
            color_encoding,
            budget,
        )?;

        // Step 2: run the per-DC-group pipeline (quant_field / mask1x1
        // / gaborish / CfL / AC strategy). Dispatch:
        //
        //   - Default (chunk 3): whole-image precompute → per-DC-group
        //     slice for CfL / AC strategy. Bit-identical to the
        //     pre-refactor monolith. This is the hash-locked path.
        //
        //   - Opt-in (chunk 5): per-region quant_field / mask1x1 /
        //     gaborish via `fill_dc_group_state_per_region`. Currently
        //     gated behind the `JXL_STREAMING_CHUNK5` env var for
        //     correctness validation; not wired to any `Buffering`
        //     variant in the default path. Chunk 6 will wire this once
        //     `encode_dc_group` (chunk 4) lets the loop driver drop
        //     per-region bitstream state.
        #[cfg(feature = "std")]
        let per_region = matches!(
            std::env::var("JXL_STREAMING_CHUNK5"),
            Ok(ref v) if v == "1"
        );
        #[cfg(not(feature = "std"))]
        let per_region = false;
        let DcGroupFill {
            quant_field_float,
            masking,
            mask1x1,
            cfl_map,
            ac_strategy,
        } = if per_region {
            fill_dc_group_state_per_region(
                &mut global,
                cfl_enabled,
                ac_strategy_enabled,
                pixel_domain_loss,
                enable_adaptive_gaborish,
                force_strategy,
                profile,
                budget,
            )?
        } else {
            fill_dc_group_state_whole_image(
                &mut global,
                cfl_enabled,
                ac_strategy_enabled,
                pixel_domain_loss,
                enable_adaptive_gaborish,
                force_strategy,
                profile,
                budget,
            )?
        };

        // Step 3: assemble into the public-shape EncoderPrecomputed
        // that downstream consumers (rate_control, encode_inner,
        // butteraugli loop) expect. Field-for-field passthrough — no
        // additional computation here.
        Ok(Self {
            width,
            height,
            xsize_blocks: global.xsize_blocks,
            ysize_blocks: global.ysize_blocks,
            padded_width: global.padded_width,
            padded_height: global.padded_height,
            xyb_x: global.xyb_x,
            xyb_y: global.xyb_y,
            xyb_b: global.xyb_b,
            linear_rgb: linear_rgb.to_vec(),
            cfl_map,
            noise_params: global.noise_params,
            quant_field_float,
            masking,
            mask1x1,
            ac_strategy,
            gaborish_enabled: enable_gaborish,
            base_distance: distance,
            chromacity_x_pixelized: global.chromacity_x_pixelized,
            chromacity_b_pixelized: global.chromacity_b_pixelized,
            xyb_pre_gaborish: global.xyb_pre_gaborish,
            patches_data: global.patches_data,
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

// =====================================================================
// #11 chunk 2 — split global vs per-DC-group precompute
// =====================================================================
//
// `EncoderPrecomputedGlobal` holds only the state that fundamentally
// needs to see the whole image. Everything that can in principle be
// processed per-DC-group (quant_field, mask1x1, gaborish, CfL,
// AC strategy) is delegated to [`fill_dc_group_state_whole_image`] in
// chunk 2 and will be split into a real per-DC-group loop in chunk 3.
//
// Why this split: chunks 3-7 of the streaming refactor (jxl-encoder#11)
// want to stream input + buffer output, processing one 2048×2048 DC
// group at a time and dropping its plane slice as soon as the per-group
// section is encoded. Today's `compute_with_budget` allocates ~24 B/pixel
// in whole-image Vecs (200 MB at 4K, 800 MB at 8K). The chunk-2 split
// is the *structural* prerequisite — it isolates the truly global state
// (chromacity stats, patches detection, noise estimation,
// xyb_pre_gaborish snapshot) into a smaller struct that streamed
// per-DC-group runs can hold in memory while iterating, without
// having to re-derive any of it per group.
//
// Bit-identical to today's pipeline: in chunk 2 the per-DC-group fill
// processes the whole image as ONE region, so every byte of the
// assembled `EncoderPrecomputed` matches what `compute_with_budget`
// produced before the refactor. The `corpus_regression` hash-lock is
// the load-bearing invariant.

/// Truly-global encoder state — the part of the precompute pipeline
/// that can't be processed per-DC-group (or that ALL DC-group passes
/// need to see).
///
/// Hand-off shape for chunk 3+: a streamed encoder loop holds a
/// `&mut EncoderPrecomputedGlobal` while iterating over DC groups,
/// calling [`fill_dc_group_state`]-style functions on each. The
/// `xyb_x` / `xyb_y` / `xyb_b` planes are mutated in place by the
/// per-DC-group pass (gaborish_inverse writes back into them).
///
/// **Hidden cross-DC-group dependencies surfaced by this split**
/// (documented here so chunk 3 doesn't accidentally regress
/// bit-identical output):
///
/// 1. **Gaborish 5×5 kernel** — `gaborish_inverse` reads a 2-pixel
///    border around each output pixel. Per-DC-group gaborish needs
///    EITHER a 2-pixel replicated border from the next DC group OR a
///    full-image gaborish pass before splitting. Chunk-3 risk.
/// 2. **mask1x1 5×5 stencil** — `compute_mask1x1` reads a 2-pixel
///    border on the PRE-gaborish XYB Y plane. Same border requirement
///    as gaborish.
/// 3. **quant_field 3×3 block neighbours** — `compute_quant_field_float`
///    reads neighbour blocks for masking. Needs 1-block border (8
///    pixels at full res) on each side of every DC group.
/// 4. **CfL tile granularity** — `compute_cfl_map` operates on
///    8-block × 8-block tiles (64×64 pixels). Tiles don't align to
///    DC groups (32-block / 256-pixel DC group). Chunk-3 risk: either
///    run CfL on multi-tile windows that fall on DC group boundaries,
///    OR accept that the boundary tile spans two DC groups and pull
///    the second DC group's data on demand.
/// 5. **AC strategy search** — `compute_ac_strategy` uses neighbour-
///    block info for AVOID_2X2 / AVOID_HF_4X4 heuristics. Needs the
///    same 1-block border as quant_field.
///
/// All five are dormant in chunk 2 because the per-DC-group fill
/// processes the whole image (no DC-group boundaries are crossed). The
/// fix-or-accept decision moves into chunk 3.
pub(crate) struct EncoderPrecomputedGlobal {
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

    /// Distance used for the initial quant field — kept on the global
    /// so per-DC-group fill (and downstream rate-control) can reproduce
    /// the same `distance_for_iqf` (gaborish-on → distance,
    /// gaborish-off → distance × 0.62).
    pub base_distance: f32,
    /// Whether gaborish was requested.
    pub gaborish_enabled: bool,

    /// Patches-subtracted PRE-gaborish XYB X. Per-DC-group fill mutates
    /// this in place when running `gaborish_inverse`. Sized
    /// `padded_width × padded_height`, row-major.
    pub xyb_x: Vec<f32>,
    /// See [`Self::xyb_x`].
    pub xyb_y: Vec<f32>,
    /// See [`Self::xyb_x`].
    pub xyb_b: Vec<f32>,

    /// Pre-gaborish XYB snapshot (only `Some` when gaborish is
    /// enabled). Same layout as `xyb_x`. Used by `encode_from_precomputed`
    /// for patches case-1 (re-detect on pre-gaborish XYB when
    /// `patches_data` is `None` but `xyb_pre_gaborish` is `Some`).
    pub xyb_pre_gaborish: Option<[Vec<f32>; 3]>,

    /// Noise parameters (full-image scan).
    pub noise_params: Option<super::noise::NoiseParams>,

    /// Patches dictionary (full-image BFS).
    pub patches_data: Option<super::patches::PatchesData>,

    /// X channel pixel chromacity (max gradient of pre-gaborish XYB X).
    pub chromacity_x_pixelized: u32,
    /// B channel pixel chromacity (from pre-gaborish XYB Y/B).
    pub chromacity_b_pixelized: u32,
}

impl EncoderPrecomputedGlobal {
    /// Run only the global part of the precompute pipeline: XYB
    /// conversion, noise estimation, optional denoising, patches
    /// detection and subtraction, chromacity stats, and the
    /// pre-gaborish XYB snapshot.
    ///
    /// Does NOT compute quant_field, mask1x1, gaborish, CfL, or
    /// AC strategy — those are the per-DC-group pass's job. Callers
    /// either run [`fill_dc_group_state_whole_image`] (chunk 2 = today's
    /// monolithic equivalent) or, in chunk 3+, iterate
    /// `fill_dc_group_state` over real DC-group-sized regions.
    ///
    /// Allocation accounting matches what `compute_with_budget` did
    /// pre-refactor: three padded XYB planes + (when gaborish) three
    /// padded pre-gaborish snapshots.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_global_only(
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        distance: f32,
        enable_noise: bool,
        enable_denoise: bool,
        enable_gaborish: bool,
        enable_patches: bool,
        use_ans: bool,
        encoder_mode: crate::api::EncoderMode,
        profile: &crate::effort::EffortProfile,
        color_encoding: Option<&crate::headers::color_encoding::ColorEncoding>,
        budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    ) -> crate::error::Result<Self> {
        use super::noise::{denoise_xyb, estimate_noise_params, noise_quality_coef};

        assert_eq!(linear_rgb.len(), width * height * 3);

        // Calculate dimensions
        let xsize_blocks = div_ceil(width, BLOCK_DIM);
        let ysize_blocks = div_ceil(height, BLOCK_DIM);
        let padded_width = xsize_blocks * BLOCK_DIM;
        let padded_height = ysize_blocks * BLOCK_DIM;

        // Convert to XYB with edge-replicated padding.
        // **Cross-DC-group consideration**: per-DC-group XYB conversion
        // is straightforward (XYB is per-pixel; padding only matters at
        // the image edge). Chunk 3 can stream this row-by-row.
        let (mut xyb_x, mut xyb_y, mut xyb_b) = convert_to_xyb_padded(
            width,
            height,
            padded_width,
            padded_height,
            linear_rgb,
            color_encoding,
            budget,
        )?;

        // Estimate noise parameters — full-image aggregation. Per-DC-group
        // streaming MUST either (a) compute per-group noise and then merge
        // the histograms, or (b) accept that noise estimation requires a
        // full-image pre-pass. libjxl picks (b) for streaming mode.
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

            // Apply denoising pre-filter if enabled.
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
        // [`EncoderPrecomputed::xyb_pre_gaborish`] and the matching
        // block in `vardct/encoder.rs::encode_from_precomputed`).
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
        //
        // **Cross-DC-group consideration**: patches detection runs an
        // 8-connected BFS that can span the whole image (a single
        // patch can occupy multiple DC groups). libjxl-style streaming
        // MUST do patches detection as a full-image pre-pass before
        // the per-DC-group loop, then apply the subtract per group as
        // each group's pixels are streamed in. This is why
        // `patches_data` lives on the GLOBAL — it's effectively
        // immutable input to every per-DC-group fill.
        //
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
        //
        // **Cross-DC-group consideration**: chromacity is a global
        // aggregation (max gradient across the whole image). Streaming
        // mode either pre-passes the full image or accepts per-DC-group
        // stats and merges them.
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

        Ok(Self {
            width,
            height,
            xsize_blocks,
            ysize_blocks,
            padded_width,
            padded_height,
            base_distance: distance,
            gaborish_enabled: enable_gaborish,
            xyb_x,
            xyb_y,
            xyb_b,
            xyb_pre_gaborish,
            noise_params,
            patches_data,
            chromacity_x_pixelized,
            chromacity_b_pixelized,
        })
    }
}

/// Per-DC-group precomputed state returned by
/// [`fill_dc_group_state_whole_image`] (chunk 2) and by the future
/// per-DC-group fill function (chunk 3).
///
/// In chunk 2 these fields cover the whole image; in chunk 3 they'll
/// cover a single DC group, and the calling loop will assemble them
/// into the whole-image Vecs on the assembled `EncoderPrecomputed`.
pub(crate) struct DcGroupFill {
    pub quant_field_float: Vec<f32>,
    pub masking: Vec<f32>,
    pub mask1x1: Option<Vec<f32>>,
    pub cfl_map: CflMap,
    pub ac_strategy: AcStrategyMap,
}

/// Run the per-DC-group precompute pipeline (quant_field / mask1x1 /
/// gaborish / CfL / AC strategy) by iterating over DC-group-sized
/// regions.
///
/// **Chunk 3 (#11)**: This is now a thin loop driver around
/// [`compute_dc_group`]. It iterates the real `DC_GROUP_DIM`-aligned
/// regions of the image, assembling per-region [`DcGroupFill`] slices
/// into the whole-image Vecs that the downstream encoder consumers
/// (`rate_control`, butteraugli loop, `encode_from_precomputed`)
/// still expect.
///
/// **Chunk 3 invariant**: bit-identical output to chunk 2 (and to the
/// pre-refactor monolith). For functions whose per-region split would
/// require explicit border-replication to stay byte-identical
/// (gaborish, mask1x1, quant_field), chunk 3 runs them ONCE on the
/// whole image and slices into per-region output. For functions that
/// are already per-tile and align cleanly with DC groups (CfL,
/// AC strategy), chunk 3 dispatches per DC group's tile range. The
/// five hidden cross-group dependencies surfaced by chunk 2 each get
/// an explicit resolution here (see per-call comments below).
///
/// **Memory profile**: Same as chunk 2. The whole-image plane Vecs
/// remain allocated end-to-end — `Buffering::BufferedOutput` is wired
/// to route through this loop, but actual memory savings (dropping
/// per-region slices) lands in chunk 5 along with per-region
/// versions of the three "whole-image precompute" steps.
///
/// **Mutates** the XYB planes on `global` in place —
/// `gaborish_inverse` rewrites them from pre-gaborish to post-gaborish.
/// After this returns, `global.xyb_x` etc. hold the post-gaborish
/// patches-subtracted XYB (which is exactly what CfL pass 2 and the
/// bitstream encoder downstream consume).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_dc_group_state_whole_image(
    global: &mut EncoderPrecomputedGlobal,
    cfl_enabled: bool,
    ac_strategy_enabled: bool,
    pixel_domain_loss: bool,
    enable_adaptive_gaborish: bool,
    force_strategy: Option<u8>,
    profile: &crate::effort::EffortProfile,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<DcGroupFill> {
    fill_dc_group_state_dispatch(
        global,
        cfl_enabled,
        ac_strategy_enabled,
        pixel_domain_loss,
        enable_adaptive_gaborish,
        force_strategy,
        profile,
        budget,
        /* per_region_precompute */ false,
    )
}

/// Per-region variant of [`fill_dc_group_state_whole_image`] — runs
/// the streaming-refactor chunk-5 per-region precomputes
/// (quant_field / mask1x1 / gaborish) instead of computing them on the
/// whole image.
///
/// **Streaming refactor chunk 5 (#11)**: opt-in entry point that
/// exercises the per-region functions through the production loop
/// driver. Currently NOT wired to any `Buffering` variant in the
/// default-features path — hash-lock byte-identity (the chunk-3
/// invariant) is enforced via the whole-image path, while this
/// per-region path may differ by a small FP drift (1-256 ULPs on
/// individual quant_field / mask1x1 / gaborish values, propagating
/// through the AC strategy / CfL search to ~0.1% bytes-level
/// divergence in the worst case).
///
/// Memory profile is identical to chunk-3 today — actual savings land
/// when chunk-4 ([encode_dc_group][`super::encoder`]) drops per-DC-group
/// section state immediately after the per-region precompute. This
/// function is the structural prereq for that wiring; tests in
/// `tests/buffering_dispatch.rs` and the chunk-5 per-region byte-
/// identity tests in `vardct::{adaptive_quant,gaborish}::tests` cover
/// the correctness contract.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn fill_dc_group_state_per_region(
    global: &mut EncoderPrecomputedGlobal,
    cfl_enabled: bool,
    ac_strategy_enabled: bool,
    pixel_domain_loss: bool,
    enable_adaptive_gaborish: bool,
    force_strategy: Option<u8>,
    profile: &crate::effort::EffortProfile,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<DcGroupFill> {
    fill_dc_group_state_dispatch(
        global,
        cfl_enabled,
        ac_strategy_enabled,
        pixel_domain_loss,
        enable_adaptive_gaborish,
        force_strategy,
        profile,
        budget,
        /* per_region_precompute */ true,
    )
}

#[allow(clippy::too_many_arguments)]
fn fill_dc_group_state_dispatch(
    global: &mut EncoderPrecomputedGlobal,
    cfl_enabled: bool,
    ac_strategy_enabled: bool,
    pixel_domain_loss: bool,
    enable_adaptive_gaborish: bool,
    force_strategy: Option<u8>,
    profile: &crate::effort::EffortProfile,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    per_region_precompute: bool,
) -> crate::error::Result<DcGroupFill> {
    use super::gaborish::{gaborish_inverse_for_region, gaborish_inverse_maybe_adaptive};

    let xsize_blocks = global.xsize_blocks;
    let ysize_blocks = global.ysize_blocks;
    let padded_width = global.padded_width;
    let padded_height = global.padded_height;
    let distance = global.base_distance;
    let enable_gaborish = global.gaborish_enabled;

    // libjxl computes InitialQuantField BEFORE GaborishInverse
    // (`enc_heuristics.cc:1117-1142`). When gaborish is off, scale
    // distance by 0.62 for the quant field
    // (matches libjxl `enc_heuristics.cc:1119`).
    let distance_for_iqf = if enable_gaborish {
        distance
    } else {
        distance * 0.62
    };

    let xsize_dc_groups_pre = div_ceil(padded_width, DC_GROUP_DIM);
    let ysize_dc_groups_pre = div_ceil(padded_height, DC_GROUP_DIM);

    // -----------------------------------------------------------------
    // quant_field / masking / mask1x1 + gaborish: whole-image precompute
    // (chunk-3 default) OR per-region assembly (chunk-5 opt-in).
    //
    // Chunk-3 path: one call each over the whole image; per-DC-group
    // iteration below slices into these whole-image buffers via
    // `compute_dc_group`. Bit-identical to the pre-refactor monolith.
    //
    // Chunk-5 path: per-region quant_field / mask1x1 / gaborish via
    // the helpers in `super::adaptive_quant` / `super::gaborish`,
    // accumulated into the same whole-image-sized output buffers. The
    // per-region functions use 1-block / 2-pixel borders from the
    // pre-gaborish XYB on `global` (gaborish-disabled) or from
    // `global.xyb_pre_gaborish` (gaborish-enabled). FP drift bounded
    // at the per-function unit tests in `vardct::{adaptive_quant,
    // gaborish}::tests::test_per_region_*`.
    let (whole_quant_field, whole_masking, whole_mask1x1) = if per_region_precompute {
        // For gaborish src reads we need a stable snapshot of
        // pre-gaborish XYB. When gaborish is enabled, that's
        // `global.xyb_pre_gaborish`. When disabled, the per-region
        // gaborish pass is a no-op and we just process from
        // `global.xyb_*` directly (which IS pre-gaborish in that
        // case).
        let mut acc_qf = vec![0.0_f32; xsize_blocks * ysize_blocks];
        let mut acc_mask = vec![0.0_f32; xsize_blocks * ysize_blocks];
        let mut acc_mask1x1 = if ac_strategy_enabled && pixel_domain_loss {
            Some(vec![0.0_f32; padded_width * padded_height])
        } else {
            None
        };

        // Snapshot pre-gaborish XYB if gaborish is enabled so we can
        // mutate global.xyb_* in place to hold the post-gaborish
        // accumulator. When gaborish is enabled, `xyb_pre_gaborish`
        // is already populated (see EncoderPrecomputedGlobal::
        // compute_global_only); we read from it as the SRC.
        for dc_y in 0..ysize_dc_groups_pre {
            for dc_x in 0..xsize_dc_groups_pre {
                let bx0 = dc_x * DC_GROUP_DIM_IN_BLOCKS;
                let by0 = dc_y * DC_GROUP_DIM_IN_BLOCKS;
                let rw_b = DC_GROUP_DIM_IN_BLOCKS.min(xsize_blocks - bx0);
                let rh_b = DC_GROUP_DIM_IN_BLOCKS.min(ysize_blocks - by0);

                // Per-region quant_field + masking. Reads PRE-gaborish
                // XYB (because quant_field is computed before gaborish
                // in libjxl). If gaborish is enabled, we read from the
                // `xyb_pre_gaborish` snapshot; otherwise from
                // `global.xyb_*` (which is pre-gaborish == post-gaborish
                // in that case).
                let (src_x, src_y, src_b) = if let Some(ref pre) = global.xyb_pre_gaborish {
                    (&pre[0][..], &pre[1][..], &pre[2][..])
                } else {
                    (&global.xyb_x[..], &global.xyb_y[..], &global.xyb_b[..])
                };

                let (region_qf, region_mask) =
                    super::adaptive_quant::compute_quant_field_float_for_region(
                        src_x,
                        src_y,
                        src_b,
                        padded_width,
                        padded_height,
                        xsize_blocks,
                        ysize_blocks,
                        bx0,
                        by0,
                        rw_b,
                        rh_b,
                        distance_for_iqf,
                        profile.k_ac_quant,
                        budget,
                    )?;
                for ry in 0..rh_b {
                    let dst_off = (by0 + ry) * xsize_blocks + bx0;
                    let src_off = ry * rw_b;
                    acc_qf[dst_off..dst_off + rw_b]
                        .copy_from_slice(&region_qf[src_off..src_off + rw_b]);
                    acc_mask[dst_off..dst_off + rw_b]
                        .copy_from_slice(&region_mask[src_off..src_off + rw_b]);
                }

                // Per-region mask1x1.
                if let Some(ref mut acc) = acc_mask1x1 {
                    let region_x0 = bx0 * 8;
                    let region_y0 = by0 * 8;
                    let region_w = rw_b * 8;
                    let region_h = rh_b * 8;
                    let region = super::adaptive_quant::compute_mask1x1_for_region(
                        src_y,
                        padded_width,
                        padded_height,
                        region_x0,
                        region_y0,
                        region_w,
                        region_h,
                        budget,
                    )?;
                    for ry in 0..region_h {
                        let dst_off = (region_y0 + ry) * padded_width + region_x0;
                        let src_off = ry * region_w;
                        acc[dst_off..dst_off + region_w]
                            .copy_from_slice(&region[src_off..src_off + region_w]);
                    }
                }
            }
        }

        // Per-region gaborish: in-place on global.xyb_* using a
        // pre-gaborish snapshot for SRC reads. Borrow checker: snapshot
        // first (or borrow `xyb_pre_gaborish` ref-only), then mutate
        // global.xyb_*.
        if enable_gaborish {
            // SAFETY-WISE — split borrows: take `xyb_pre_gaborish` by
            // reference for SRC, and mutate `global.xyb_x/y/b` for DST.
            // Both live on the same struct so we destructure into local
            // bindings to satisfy the borrow checker.
            let EncoderPrecomputedGlobal {
                xyb_x: ref mut dst_x,
                xyb_y: ref mut dst_y,
                xyb_b: ref mut dst_b,
                xyb_pre_gaborish: ref pre,
                ..
            } = *global;
            let pre = pre
                .as_ref()
                .expect("xyb_pre_gaborish must be Some when gaborish is enabled");

            for dc_y in 0..ysize_dc_groups_pre {
                for dc_x in 0..xsize_dc_groups_pre {
                    let region_x0 = dc_x * DC_GROUP_DIM;
                    let region_y0 = dc_y * DC_GROUP_DIM;
                    let region_w = DC_GROUP_DIM.min(padded_width - region_x0);
                    let region_h = DC_GROUP_DIM.min(padded_height - region_y0);

                    gaborish_inverse_for_region(
                        &pre[0],
                        &pre[1],
                        &pre[2],
                        dst_x,
                        dst_y,
                        dst_b,
                        padded_width,
                        padded_height,
                        region_x0,
                        region_y0,
                        region_w,
                        region_h,
                        enable_adaptive_gaborish,
                        budget,
                    )?;
                }
            }
        }

        (acc_qf, acc_mask, acc_mask1x1)
    } else {
        let (qf, mask) = super::adaptive_quant::compute_quant_field_float_with_budget(
            &global.xyb_x,
            &global.xyb_y,
            &global.xyb_b,
            padded_width,
            padded_height,
            xsize_blocks,
            ysize_blocks,
            distance_for_iqf,
            profile.k_ac_quant,
            budget,
        )?;

        let mask1x1 = if ac_strategy_enabled && pixel_domain_loss {
            Some(super::adaptive_quant::compute_mask1x1_with_budget(
                &global.xyb_y,
                padded_width,
                padded_height,
                budget,
            )?)
        } else {
            None
        };

        if enable_gaborish {
            gaborish_inverse_maybe_adaptive(
                &mut global.xyb_x,
                &mut global.xyb_y,
                &mut global.xyb_b,
                padded_width,
                padded_height,
                enable_adaptive_gaborish,
                budget,
            )?;
        }
        (qf, mask, mask1x1)
    };

    // -----------------------------------------------------------------
    // Per-DC-group iteration: CfL + AC strategy via the per-tile-list
    // helpers. These two ARE genuinely per-region in chunk 3 — they're
    // already structured as per-tile parallel loops, and DC groups
    // (256×256 blocks = 32×32 CfL tiles) align cleanly with tile
    // boundaries (deps #4 + #5 are resolved by this alignment, no
    // border replication needed).
    //
    // Total per-tile work is identical to a single whole-image call
    // (each tile is processed exactly once across all DC groups). The
    // per-DC-group structure exists so chunk 4 can hand each DC
    // group's section off to the bitstream encoder immediately after
    // `compute_dc_group` returns, and chunk 5 can stream the input
    // pixels for one DC group at a time instead of holding the
    // whole-image XYB.
    let xsize_tiles = div_ceil(xsize_blocks, TILE_DIM_IN_BLOCKS);
    let ysize_tiles = div_ceil(ysize_blocks, TILE_DIM_IN_BLOCKS);

    let xsize_dc_groups = div_ceil(padded_width, DC_GROUP_DIM);
    let ysize_dc_groups = div_ceil(padded_height, DC_GROUP_DIM);

    let mut cfl_map = if cfl_enabled {
        CflMap::zeros(xsize_tiles, ysize_tiles)
    } else {
        CflMap::zeros(xsize_tiles, ysize_tiles)
    };
    let mut ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);

    for dc_y in 0..ysize_dc_groups {
        for dc_x in 0..xsize_dc_groups {
            let fill = compute_dc_group(
                global,
                dc_x as u32,
                dc_y as u32,
                xsize_dc_groups as u32,
                ysize_dc_groups as u32,
                &whole_quant_field,
                &whole_masking,
                whole_mask1x1.as_deref(),
                &cfl_map,
                cfl_enabled,
                ac_strategy_enabled,
                force_strategy,
                profile,
            )?;

            // Assemble per-region CfL into the whole-image map.
            // DC groups align with TILE boundaries (256 blocks = 32
            // tiles), so the tile rectangle is well-defined.
            let tile_bx0 = (dc_x * DC_GROUP_DIM_IN_BLOCKS) / TILE_DIM_IN_BLOCKS;
            let tile_by0 = (dc_y * DC_GROUP_DIM_IN_BLOCKS) / TILE_DIM_IN_BLOCKS;
            for sub_ty in 0..fill.cfl_region_h {
                let dst_ty = tile_by0 + sub_ty;
                let dst_off = dst_ty * xsize_tiles + tile_bx0;
                let src_off = sub_ty * fill.cfl_region_w;
                cfl_map.ytox[dst_off..dst_off + fill.cfl_region_w]
                    .copy_from_slice(&fill.cfl_region_ytox[src_off..src_off + fill.cfl_region_w]);
                cfl_map.ytob[dst_off..dst_off + fill.cfl_region_w]
                    .copy_from_slice(&fill.cfl_region_ytob[src_off..src_off + fill.cfl_region_w]);
            }

            // Assemble per-region AC strategy into the whole-image map.
            let bx0 = dc_x * DC_GROUP_DIM_IN_BLOCKS;
            let by0 = dc_y * DC_GROUP_DIM_IN_BLOCKS;
            let bx1 = (bx0 + DC_GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
            let by1 = (by0 + DC_GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
            ac_strategy.copy_region_from(&fill.ac_strategy, bx0, by0, bx1, by1);
        }
    }

    // CfL pass 2 refinement happens in encoder.rs after the butteraugli loop
    // produces the final quant_field. No refinement here — pass 1 values from
    // compute_cfl_map are sufficient for initial AC strategy selection.

    Ok(DcGroupFill {
        quant_field_float: whole_quant_field,
        masking: whole_masking,
        mask1x1: whole_mask1x1,
        cfl_map,
        ac_strategy,
    })
}

/// Per-DC-group output from [`compute_dc_group`].
///
/// Holds only the per-region CfL and AC strategy slices. The
/// quant_field / mask1x1 / masking arrays are computed at whole-image
/// scope in the loop driver (see [`fill_dc_group_state_whole_image`]),
/// so they're not duplicated here.
///
/// Chunk 3 shape; chunk 5 will extend this to carry per-region
/// quant_field / mask1x1 / masking slices as well, once the
/// border-replication strategy lands.
pub(crate) struct PerDcGroupFill {
    /// CfL ytox values for the tiles inside this DC group, row-major,
    /// `cfl_region_w * cfl_region_h` entries.
    pub cfl_region_ytox: Vec<i8>,
    /// CfL ytob values for the tiles inside this DC group, row-major,
    /// `cfl_region_w * cfl_region_h` entries.
    pub cfl_region_ytob: Vec<i8>,
    /// Width of the CfL tile rectangle covered by this DC group
    /// (in tile units).
    pub cfl_region_w: usize,
    /// Height of the CfL tile rectangle covered by this DC group
    /// (in tile units).
    #[allow(dead_code)]
    pub cfl_region_h: usize,

    /// Whole-image-sized AC strategy map with only this DC group's
    /// blocks filled in (other positions hold the default DCT8 sentinel
    /// from [`AcStrategyMap::new_dct8`]). The loop driver
    /// `copy_region_from`s the DC group's block rectangle into the
    /// aggregated map. The whole-image sizing keeps the existing
    /// per-tile parallel infrastructure usable without forking
    /// [`super::ac_strategy::compute_ac_strategy`].
    pub ac_strategy: AcStrategyMap,
}

/// Compute the per-DC-group CfL + AC strategy for a single DC group at
/// `(dc_x, dc_y)` (in DC-group coordinates; a DC group is
/// `DC_GROUP_DIM` × `DC_GROUP_DIM` pixels = `DC_GROUP_DIM_IN_BLOCKS`
/// blocks per side).
///
/// **Cross-group dependency resolution (chunk 3)**:
/// - Dep #1 (gaborish 5×5): post-gaborish XYB is already on
///   `global.xyb_*` because the loop driver ran `gaborish_inverse` on
///   the whole image before iterating. Per-region call here reads
///   from those already-post-gaborish planes.
/// - Dep #2 (mask1x1 5×5): whole-image mask1x1 is passed in as
///   `whole_mask1x1`; per-region call slices it via stride.
/// - Dep #3 (quant_field 3×3 block): whole-image quant_field +
///   masking are passed in; per-region call slices them.
/// - Dep #4 (CfL tile alignment): DC groups are 32×32 blocks, CfL
///   tiles are 8×8 blocks → 4×4 tiles per DC group; aligns cleanly,
///   no border needed.
/// - Dep #5 (AC strategy 1-block border): the per-tile AC search
///   inside this DC group's tile range reads from the whole-image
///   `global.xyb_*` planes (passed by slice through
///   `compute_ac_strategy`), so neighbour-block reads at DC-group
///   edges automatically see the actual neighbour data — no border
///   replication needed for byte-identity.
///
/// **Byte-identity gate**: this function produces output that, when
/// assembled by the loop driver via `copy_region_from`, is identical
/// bit-for-bit to the chunk-2 monolithic
/// `fill_dc_group_state_whole_image` output. Verified by
/// `hash_lock_features` (36/36 byte-identical).
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_dc_group(
    global: &EncoderPrecomputedGlobal,
    dc_x: u32,
    dc_y: u32,
    dc_groups_x: u32,
    dc_groups_y: u32,
    whole_quant_field: &[f32],
    whole_masking: &[f32],
    whole_mask1x1: Option<&[f32]>,
    aggregated_cfl: &CflMap,
    cfl_enabled: bool,
    ac_strategy_enabled: bool,
    force_strategy: Option<u8>,
    profile: &crate::effort::EffortProfile,
) -> crate::error::Result<PerDcGroupFill> {
    debug_assert!(dc_x < dc_groups_x);
    debug_assert!(dc_y < dc_groups_y);

    let xsize_blocks = global.xsize_blocks;
    let ysize_blocks = global.ysize_blocks;
    let padded_width = global.padded_width;
    let padded_height = global.padded_height;
    let distance = global.base_distance;

    // DC-group block range (clamped at image right/bottom edges).
    let bx0 = (dc_x as usize) * DC_GROUP_DIM_IN_BLOCKS;
    let by0 = (dc_y as usize) * DC_GROUP_DIM_IN_BLOCKS;
    let bx1 = (bx0 + DC_GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
    let by1 = (by0 + DC_GROUP_DIM_IN_BLOCKS).min(ysize_blocks);

    // Tile range covered by this DC group. DC groups are 256 blocks =
    // 32 CfL tiles per side. Tile-aligned by construction.
    let tile_bx0 = bx0 / TILE_DIM_IN_BLOCKS;
    let tile_by0 = by0 / TILE_DIM_IN_BLOCKS;
    let tile_bx1 = div_ceil(bx1, TILE_DIM_IN_BLOCKS);
    let tile_by1 = div_ceil(by1, TILE_DIM_IN_BLOCKS);
    let cfl_region_w = tile_bx1 - tile_bx0;
    let cfl_region_h = tile_by1 - tile_by0;

    // ----- CfL: per-DC-group tile evaluation -----
    //
    // Build the tile list for this DC group's CfL tiles and run the
    // existing per-tile parallel CfL helper. Each tile's CfL search
    // reads only its own tile's XYB data (no cross-tile state in
    // `find_best_multiplier`), so the per-DC-group call is
    // byte-identical to the corresponding slice of the whole-image
    // `compute_cfl_map`. Dep #4 (CfL tile alignment) resolved by
    // construction.
    let (cfl_region_ytox, cfl_region_ytob) = if cfl_enabled {
        super::chroma_from_luma::compute_cfl_map_for_tiles(
            &global.xyb_x,
            &global.xyb_y,
            &global.xyb_b,
            padded_width,
            padded_height,
            xsize_blocks,
            ysize_blocks,
            tile_bx0,
            tile_by0,
            cfl_region_w,
            cfl_region_h,
            profile.cfl_newton,
            profile.cfl_newton_eps,
            profile.cfl_newton_max_iters,
        )
    } else {
        (
            vec![0i8; cfl_region_w * cfl_region_h],
            vec![0i8; cfl_region_w * cfl_region_h],
        )
    };

    // ----- AC strategy: per-DC-group tile evaluation -----
    //
    // The loop driver passes in `aggregated_cfl` (the whole-image-sized
    // CflMap that already holds CfL for earlier-processed DC groups
    // plus zero-defaults for later ones). We need to inject THIS DC
    // group's freshly-computed CfL values into a local CflMap before
    // running AC strategy: each per-tile AC search reads
    // `cfl_map.ytox_at(tx, ty)` for tiles inside its own DC group, and
    // those tiles MUST hold the values we just computed in `cfl_region_*`.
    //
    // Allocating a fresh whole-image-sized CflMap per DC group is
    // ~`xsize_tiles*ysize_tiles*2` bytes (≤ 32 KB at 4K, ≤ 128 KB at
    // 8K) — small relative to the per-DC-group XYB / quant_field
    // working set. Chunk 5 can replace this with an in-place update of
    // the aggregated CflMap if the savings matter.
    let ac_strategy = if let Some(forced) = force_strategy {
        AcStrategyMap::force_strategy(xsize_blocks, ysize_blocks, forced)
    } else if !ac_strategy_enabled {
        AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks)
    } else {
        let xsize_tiles = div_ceil(xsize_blocks, TILE_DIM_IN_BLOCKS);
        let mut local_cfl = aggregated_cfl.clone();
        for sub_ty in 0..cfl_region_h {
            let dst_ty = tile_by0 + sub_ty;
            let dst_off = dst_ty * xsize_tiles + tile_bx0;
            let src_off = sub_ty * cfl_region_w;
            local_cfl.ytox[dst_off..dst_off + cfl_region_w]
                .copy_from_slice(&cfl_region_ytox[src_off..src_off + cfl_region_w]);
            local_cfl.ytob[dst_off..dst_off + cfl_region_w]
                .copy_from_slice(&cfl_region_ytob[src_off..src_off + cfl_region_w]);
        }

        // Build the tile list for this DC group's AC strategy tiles.
        let mut tiles = Vec::with_capacity((tile_bx1 - tile_bx0) * (tile_by1 - tile_by0));
        for tile_by in (by0..by1).step_by(TILE_DIM_IN_BLOCKS) {
            for tile_bx in (bx0..bx1).step_by(TILE_DIM_IN_BLOCKS) {
                tiles.push((tile_bx, tile_by));
            }
        }

        super::ac_strategy::compute_ac_strategy_for_tiles(
            &global.xyb_x,
            &global.xyb_y,
            &global.xyb_b,
            padded_width,
            padded_height,
            xsize_blocks,
            ysize_blocks,
            distance,
            whole_quant_field,
            whole_masking,
            &local_cfl,
            whole_mask1x1,
            padded_width,
            profile,
            &tiles,
        )
    };

    Ok(PerDcGroupFill {
        cfl_region_ytox,
        cfl_region_ytob,
        cfl_region_w,
        cfl_region_h,
        ac_strategy,
    })
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
