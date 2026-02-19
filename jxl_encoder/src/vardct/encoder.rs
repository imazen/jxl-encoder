// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Main tiny encoder implementation.

use super::ac_strategy::{AcStrategyMap, adjust_quant_field_with_distance, compute_ac_strategy};
use super::adaptive_quant::{compute_mask1x1, compute_quant_field_float, quantize_quant_field};
use super::chroma_from_luma::{CflMap, compute_cfl_map};
use super::common::*;
use super::frame::{DistanceParams, write_toc};
use super::gaborish::gaborish_inverse;
use super::noise::{denoise_xyb, estimate_noise_params, noise_quality_coef};
use super::static_codes::{get_ac_entropy_code, get_dc_entropy_code};
use crate::bit_writer::BitWriter;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
use crate::debug_rect;
use crate::error::Result;
use crate::headers::frame_header::FrameHeader;

// Re-export types from entropy_code sub-module.
pub(crate) use super::entropy_code::{BuiltEntropyCode, force_strategy_map};

/// Output of a VarDCT encode operation.
pub struct VarDctOutput {
    /// Encoded JXL codestream bytes.
    pub data: Vec<u8>,
    /// Per-strategy first-block counts, indexed by raw strategy code (0..19).
    pub strategy_counts: [u32; 19],
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
    /// Effort level (1–10). Controls AC strategy gating and search depth.
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
    /// Encoder mode: Reference (match libjxl) or Experimental (own improvements).
    pub encoder_mode: crate::api::EncoderMode,
    /// Manual splines to overlay on the image (opt-in, None by default).
    pub splines: Option<Vec<crate::vardct::splines::Spline>>,
    /// Whether the input is grayscale. When true, the file header signals
    /// ColorSpace::Gray instead of RGB. VarDCT still operates in XYB (3 channels)
    /// internally — this only affects the output colorspace the decoder targets.
    pub is_grayscale: bool,
    /// Progressive encoding mode (Single, QuantizedAcFullAc, DcVlfLfAc).
    /// When not Single, AC coefficients are split across multiple passes with
    /// shift-based precision reduction for early preview rendering.
    pub progressive: crate::api::ProgressiveMode,
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
            enable_denoise: false,
            enable_gaborish: true,
            error_diffusion: true, // libjxl enables at speed_tier <= kSquirrel (effort 7)
            pixel_domain_loss: true, // Full libjxl pixel-domain loss: +0.2-1.9 SSIM2 at all distances
            enable_lz77: false,      // LZ77 has known interactions with DCT2x2/IDENTITY strategies
            lz77_method: crate::entropy_coding::lz77::Lz77Method::Greedy, // Best compression
            dc_tree_learning: false, // DC tree learning (experimental)
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters: 0, // Effort-gated: default off (effort 7). Set via LossyConfig.
            bit_depth_16: false,
            icc_profile: None,
            enable_patches: true, // Patches: huge wins on screenshots, zero cost on photos
            encoder_mode: crate::api::EncoderMode::Reference,
            splines: None,
            is_grayscale: false,
            progressive: crate::api::ProgressiveMode::Single,
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
            enable_denoise: false,
            enable_gaborish: true,
            error_diffusion: true, // libjxl enables at speed_tier <= kSquirrel (effort 7)
            pixel_domain_loss: true, // Full libjxl pixel-domain loss: +0.2-1.9 SSIM2
            enable_lz77: false,    // LZ77 has known interactions with DCT2x2/IDENTITY strategies
            lz77_method: crate::entropy_coding::lz77::Lz77Method::Greedy, // Best compression
            dc_tree_learning: false, // DC tree learning (experimental)
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters: 0, // Effort-gated: default off (effort 7). Set via LossyConfig.
            bit_depth_16: false,
            icc_profile: None,
            enable_patches: true, // Patches: huge wins on screenshots, zero cost on photos
            encoder_mode: crate::api::EncoderMode::Reference,
            splines: None,
            is_grayscale: false,
            progressive: crate::api::ProgressiveMode::Single,
        }
    }

    /// Encode an image in linear sRGB format, optionally with an alpha channel.
    ///
    /// Input should be 3 channels (RGB) of f32 values in [0, 1] range.
    /// Values outside [0, 1] are allowed for out-of-gamut colors.
    ///
    /// If `alpha` is provided, it must be `width * height` bytes of u8 alpha values.
    /// Alpha is encoded as a modular extra channel alongside the VarDCT RGB data.
    pub fn encode(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        alpha: Option<&[u8]>,
    ) -> Result<VarDctOutput> {
        assert_eq!(linear_rgb.len(), width * height * 3);
        if let Some(a) = alpha {
            assert_eq!(a.len(), width * height);
        }

        crate::debug_rect::clear();

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

        // Convert to XYB with edge-replicated padding to block boundaries.
        // This allows SIMD to process full blocks without bounds checking.
        let (mut xyb_x, mut xyb_y, mut xyb_b) =
            self.convert_to_xyb_padded(width, height, padded_width, padded_height, linear_rgb);

        // Estimate noise parameters (if enabled).
        // The decoder adds noise during rendering; the encoder just encodes the params.
        let noise_params = if self.enable_noise {
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
        let mut patches_data = if self.enable_patches {
            super::patches::find_and_build([&xyb_x, &xyb_y, &xyb_b], width, height, padded_width)
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

        // Build and subtract splines (after patches, before gaborish).
        // Splines are additive overlays: encoder subtracts, decoder adds back.
        // Uses default DC CfL params (y_to_x=0.0, y_to_b=1.0) since we write default DC cmap.
        let splines_data = if let Some(ref splines) = self.splines {
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

        // Compute pixel chromacity stats BEFORE gaborish (matching libjxl pipeline).
        // Gaborish sharpening inflates gradients, producing overly aggressive adjustment.
        let pixel_stats = super::frame::PixelStatsForChromacityAdjustment::calc(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            padded_width,
            padded_height,
        );
        let chromacity_x = pixel_stats.how_much_is_x_channel_pixelized();
        let chromacity_b = pixel_stats.how_much_is_b_channel_pixelized();

        // Apply gaborish inverse (5x5 sharpening) before adaptive quant.
        // The decoder will apply a 3x3 blur to compensate.
        if self.enable_gaborish {
            gaborish_inverse(
                &mut xyb_x,
                &mut xyb_y,
                &mut xyb_b,
                padded_width,
                padded_height,
            );
        }

        // Compute adaptive per-block quantization field and masking.
        // Pass padded dimensions: XYB buffers have stride=padded_width, and all
        // modulation/extraction functions index as [py * stride + px].
        // When gaborish is off, scale distance by 0.62 for the quant field only
        // (not global_scale/quant_dc). This matches libjxl enc_heuristics.cc:1119.
        let distance_for_iqf = if self.enable_gaborish {
            self.distance
        } else {
            self.distance * 0.62
        };

        // Step 1: Compute float quant field.
        //
        // libjxl effort gating (enc_heuristics.cc:1097-1128):
        // - effort < 5 (speed_tier > kHare): flat quant field = q_numerator/distance
        // - effort >= 5 (speed_tier <= kHare): adaptive via InitialQuantField
        let (quant_field_float, masking) = if self.profile.use_adaptive_quant {
            compute_quant_field_float(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                distance_for_iqf,
                self.profile.k_ac_quant,
            )
        } else {
            // Flat quant field for low effort (matches libjxl enc_heuristics.cc:1105-1106)
            let q = self.profile.initial_q_numerator / self.distance;
            let flat_qf = vec![q; xsize_blocks * ysize_blocks];
            let masking_val = 1.0 / (q + 0.001);
            let flat_masking = vec![masking_val; xsize_blocks * ysize_blocks];
            (flat_qf, flat_masking)
        };

        // Step 2: Compute distance params with effort-matched global_scale.
        //
        // Uses profile.initial_q_numerator for q = numerator / distance.
        // The adaptive median/MAD formula is only used inside the butteraugli
        // loop (effort >= 8).
        let mut params = DistanceParams::compute_for_profile(self.distance, &self.profile);

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

        // Step 3: Quantize float quant field to raw u8 with adaptive inv_scale
        let mut quant_field = quantize_quant_field(&quant_field_float, params.inv_scale);

        // Compute per-tile chroma-from-luma map
        let mut cfl_map = if self.cfl_enabled {
            compute_cfl_map(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                self.profile.cfl_newton,
            )
        } else {
            CflMap::zeros(
                div_ceil(xsize_blocks, TILE_DIM_IN_BLOCKS),
                div_ceil(ysize_blocks, TILE_DIM_IN_BLOCKS),
            )
        };

        // Compute per-pixel mask for pixel-domain loss (full libjxl cost model)
        // Only compute if AC strategy selection is enabled
        let mask1x1 = if self.ac_strategy_enabled && self.pixel_domain_loss {
            Some(compute_mask1x1(&xyb_y, padded_width, padded_height))
        } else {
            None
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
            self.butteraugli_iters,
            self.enable_noise,
            self.enable_denoise,
            self.ac_strategy_enabled,
            self.error_diffusion
        );

        // Compute adaptive AC strategy (DCT8/DCT16x8/DCT8x16/DCT16x16/DCT32x32)
        let ac_strategy = if let Some(forced) = self.force_strategy {
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
                &cfl_map,
                mask1x1.as_deref(),
                padded_width,
                &self.profile,
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
            let names = [
                "DCT8",
                "HORNETS",
                "DCT2x2",
                "DCT4x4",
                "DCT16x16",
                "DCT32x32",
                "DCT8x16",
                "DCT8x4",
                "DCT4x8",
                "AFV0",
                "AFV1",
                "AFV2",
                "AFV3",
                "DCT64x64",
                "DCT64x32",
                "DCT32x64",
                "DCT128x128",
                "DCT128x64",
                "DCT64x128",
                "DCT256x256",
                "DCT256x128",
                "DCT128x256",
                "IDENTITY",
                "DCT16x8",
                "DCT32x16",
                "DCT16x32",
                "DCT8x32",
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

        // Adjust quant field for multi-block transforms.
        // At low distances uses max, at high distances blends toward mean for better quality.
        adjust_quant_field_with_distance(&ac_strategy, &mut quant_field, self.distance);

        // Butteraugli quantization loop: iteratively refine quant_field using
        // perceptual distance feedback. AC strategy is fixed; only quant_field changes.
        #[cfg(feature = "butteraugli-loop")]
        if self.butteraugli_iters > 0 {
            let initial_quant_field = quant_field.clone();
            self.butteraugli_refine_quant_field(
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
                &initial_quant_field,
                &cfl_map,
                &ac_strategy,
                patches_data.as_ref(),
                splines_data.as_ref(),
            );
        }

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

        // CfL two-pass refinement: try ±1 for each tile's ytox/ytob after AC strategy
        // is determined, to account for quantization effects.
        // Gated at effort >= 7 (speed_tier <= kSquirrel) matching libjxl.
        if self.profile.cfl_two_pass && self.cfl_enabled {
            super::chroma_from_luma::refine_cfl_map(
                &mut cfl_map,
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                xsize_blocks,
                ysize_blocks,
            );
        }

        // Perform DCT and quantization (XYB data is padded to block boundaries)
        let transform_out = self.transform_and_quantize(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            padded_width,
            xsize_blocks,
            ysize_blocks,
            &params,
            &mut quant_field,
            &cfl_map,
            &ac_strategy,
        );
        let quant_dc = &transform_out.quant_dc;
        let quant_ac = &transform_out.quant_ac;
        let nzeros = &transform_out.nzeros;
        let raw_nzeros = &transform_out.raw_nzeros;

        // Compute per-block EPF sharpness map when EPF is active
        // Dynamic sharpness gated at effort >= 6 (speed_tier <= kWombat) matching libjxl
        let sharpness_map =
            if params.epf_iters > 0 && self.distance >= 0.5 && self.profile.epf_dynamic_sharpness {
                let mask = mask1x1.unwrap_or_else(|| {
                    super::adaptive_quant::compute_mask1x1(&xyb_y, padded_width, padded_height)
                });
                Some(super::epf::compute_epf_sharpness(
                    [&xyb_x, &xyb_y, &xyb_b],
                    quant_dc,
                    quant_ac,
                    &quant_field,
                    &mask,
                    &params,
                    &cfl_map,
                    &ac_strategy,
                    self.enable_gaborish,
                    xsize_blocks,
                    ysize_blocks,
                ))
            } else {
                None
            };

        // Two-pass mode: collect tokens, build optimal codes, write bitstream
        if self.optimize_codes {
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
                alpha,
                patches_data.as_ref(),
                splines_data.as_ref(),
            )?;
            crate::debug_rect::flush("");
            return Ok(VarDctOutput {
                data,
                strategy_counts,
            });
        }

        // Get static entropy codes (wrapped in BuiltEntropyCode for uniform handling)
        let dc_code = BuiltEntropyCode::StaticHuffman(get_dc_entropy_code());
        let ac_code = BuiltEntropyCode::StaticHuffman(get_ac_entropy_code());

        // Create main writer
        let mut writer = BitWriter::with_capacity(width * height * 4);

        // Write file header (includes JXL signature, ICC, and byte padding)
        // Streaming path does not support alpha
        self.write_file_header_and_pad(width, height, false, &mut writer)?;
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
            let block_ctx_map = super::ac_context::BlockCtxMap::default();
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
                &mut dc_global,
            )?;

            // Get borrowed Huffman codes for streaming token writing
            let dc_huffman = dc_code.as_huffman();
            let ac_huffman = ac_code.as_huffman();

            let mut dc_group = BitWriter::with_capacity(num_blocks * 10);
            self.write_dc_group(
                0,
                quant_dc,
                xsize_blocks,
                ysize_blocks,
                xsize_dc_groups,
                &quant_field,
                &cfl_map,
                &ac_strategy,
                None, // no sharpness map in single-pass mode
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

            let mut ac_group_writer = BitWriter::with_capacity(num_blocks * 100);
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
            // Multi-group: use byte-aligned sections
            let mut sections: Vec<Vec<u8>> = Vec::with_capacity(num_sections);
            let dc_huffman = dc_code.as_huffman();
            let ac_huffman = ac_code.as_huffman();

            // DC Global section
            let block_ctx_map = super::ac_context::BlockCtxMap::default();
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
                &mut dc_global,
            )?;
            dc_global.zero_pad_to_byte();
            sections.push(dc_global.finish());

            // DC group sections
            let blocks_per_dc_group = (256 / 8) * (256 / 8); // 1024 blocks per DC group
            for dc_group_idx in 0..num_dc_groups {
                let mut dc_group = BitWriter::with_capacity(blocks_per_dc_group * 10);
                self.write_dc_group(
                    dc_group_idx,
                    quant_dc,
                    xsize_blocks,
                    ysize_blocks,
                    xsize_dc_groups,
                    &quant_field,
                    &cfl_map,
                    &ac_strategy,
                    None, // no sharpness map in single-pass mode
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
                let mut ac_group_writer = BitWriter::with_capacity(blocks_per_ac_group * 100);
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

            let section_sizes: Vec<usize> = sections.iter().map(|s| s.len()).collect();
            write_toc(&section_sizes, &mut writer)?;
            for section in sections {
                writer.append_bytes(&section)?;
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
        // Compute precomputed state
        let precomputed = super::precomputed::EncoderPrecomputed::compute(
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
            self.force_strategy,
            &self.profile,
        );

        // Run rate control loop
        super::rate_control::encode_with_rate_control(self, &precomputed, config)
    }

    /// Encode from precomputed state with a specific quant field.
    ///
    /// This is the core encoding function used by rate control iterations.
    /// It skips XYB conversion, CfL, masking, and AC strategy computation,
    /// using the values from `precomputed` instead.
    ///
    /// Requires the `rate-control` feature.
    #[cfg(feature = "rate-control")]
    pub fn encode_from_precomputed(
        &self,
        precomputed: &super::precomputed::EncoderPrecomputed,
        quant_field: &[u8],
    ) -> Result<Vec<u8>> {
        let width = precomputed.width;
        let height = precomputed.height;
        let xsize_blocks = precomputed.xsize_blocks;
        let ysize_blocks = precomputed.ysize_blocks;
        let padded_width = precomputed.padded_width;

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
        let mut params = DistanceParams::compute_for_profile(self.distance, &self.profile);

        // Apply pixel-level chromacity adjustments using pre-gaborish stats
        if self.profile.chromacity_adjustment {
            params.apply_chromacity_adjustment(
                precomputed.chromacity_x_pixelized,
                precomputed.chromacity_b_pixelized,
            );
        }

        // Perform DCT and quantization using precomputed XYB data
        let transform_out = self.transform_and_quantize(
            &precomputed.xyb_x,
            &precomputed.xyb_y,
            &precomputed.xyb_b,
            padded_width,
            xsize_blocks,
            ysize_blocks,
            &params,
            &mut quant_field,
            &precomputed.cfl_map,
            &precomputed.ac_strategy,
        );
        let quant_dc = &transform_out.quant_dc;
        let quant_ac = &transform_out.quant_ac;
        let nzeros = &transform_out.nzeros;
        let raw_nzeros = &transform_out.raw_nzeros;

        // Use two-pass mode for rate control (required for ANS)
        self.encode_two_pass(
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
            &precomputed.cfl_map,
            &precomputed.ac_strategy,
            &precomputed.noise_params,
            None, // TODO: compute sharpness_map for rate control path
            None, // TODO: thread alpha through butteraugli path
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
        let (x, y, b) = encoder.convert_to_xyb_padded(1, 1, 8, 8, &linear_rgb);

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

        // Lock the hash - if this changes, the encoding has changed
        // Updated: disable CfL Newton (causes regression without pass 2)
        const EXPECTED_HASH: u64 = 0x8e3855cfba82de2b;
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

        // Updated: fix global_scale to use effort-matched fixed q (libjxl parity)
        const EXPECTED_HASH: u64 = 0x2b2dd2aaba0b53f4;
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

        // Updated: disable CfL Newton (causes regression without pass 2)
        const EXPECTED_HASH: u64 = 0x5f4f631b2291961f;
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

        // Updated: disable CfL Newton (causes regression without pass 2)
        const EXPECTED_HASH: u64 = 0xe73eeabbb5221327;
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
}
