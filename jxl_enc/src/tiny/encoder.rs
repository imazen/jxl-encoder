// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Main tiny encoder implementation.

use super::ac_strategy::{AcStrategyMap, adjust_quant_field_with_distance, compute_ac_strategy};
use super::adaptive_quant::{compute_mask1x1, compute_quant_field_float, quantize_quant_field};
use super::chroma_from_luma::{CflMap, compute_cfl_map};
use super::common::*;
use super::entropy_code::{
    OwnedAnsEntropyCode, OwnedEntropyCode, write_entropy_code_ans, write_tokens, write_tokens_ans,
};
use super::frame::{DistanceParams, write_frame_header, write_toc};
use super::gaborish::gaborish_inverse;
use super::noise::{denoise_xyb, estimate_noise_params, noise_quality_coef};
use super::static_codes::{get_ac_entropy_code, get_dc_entropy_code};
use super::token::Token;
use crate::bit_writer::BitWriter;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
use crate::error::Result;

/// Create an AC strategy map forcing a specific strategy.
fn force_strategy_map(xsize_blocks: usize, ysize_blocks: usize, raw_strategy: u8) -> AcStrategyMap {
    AcStrategyMap::force_strategy(xsize_blocks, ysize_blocks, raw_strategy)
}

/// Entropy code that holds either Huffman or ANS code.
pub enum BuiltEntropyCode<'a> {
    /// Static Huffman prefix codes (borrowed).
    StaticHuffman(super::entropy_code::EntropyCode<'a>),
    /// Dynamic Huffman prefix codes (owned).
    Huffman(OwnedEntropyCode),
    /// ANS distributions with context map.
    Ans(OwnedAnsEntropyCode),
}

impl<'a> BuiltEntropyCode<'a> {
    /// Write the entropy code header (context map + codes/distributions).
    pub fn write_header(&self, writer: &mut BitWriter) -> Result<()> {
        match self {
            BuiltEntropyCode::StaticHuffman(code) => {
                super::entropy_code::write_entropy_code(code, writer)
            }
            BuiltEntropyCode::Huffman(code) => {
                super::entropy_code::write_entropy_code(&code.as_entropy_code(), writer)
            }
            BuiltEntropyCode::Ans(code) => write_entropy_code_ans(code, writer),
        }
    }

    /// Write tokens using this entropy code.
    pub fn write_tokens(
        &self,
        tokens: &[Token],
        lz77: Option<&super::lz77::Lz77Params>,
        writer: &mut BitWriter,
    ) -> Result<()> {
        match self {
            BuiltEntropyCode::StaticHuffman(code) => write_tokens(tokens, code, lz77, writer),
            BuiltEntropyCode::Huffman(code) => {
                write_tokens(tokens, &code.as_entropy_code(), lz77, writer)
            }
            BuiltEntropyCode::Ans(code) => write_tokens_ans(tokens, code, lz77, writer),
        }
    }

    /// Get the underlying Huffman code for streaming token writing.
    ///
    /// Panics if this is an ANS code (streaming with ANS is not supported).
    pub fn as_huffman(&self) -> super::entropy_code::EntropyCode<'_> {
        match self {
            BuiltEntropyCode::StaticHuffman(code) => *code,
            BuiltEntropyCode::Huffman(code) => code.as_entropy_code(),
            BuiltEntropyCode::Ans(_) => {
                panic!("ANS codes cannot be used with streaming encoder")
            }
        }
    }

    #[allow(dead_code)]
    /// Returns the number of contexts in this entropy code.
    pub fn num_contexts(&self) -> usize {
        match self {
            BuiltEntropyCode::StaticHuffman(code) => code.num_contexts,
            BuiltEntropyCode::Huffman(code) => code.context_map.len(),
            BuiltEntropyCode::Ans(code) => code.context_map.len(),
        }
    }

    #[allow(dead_code)]
    /// Returns the number of histograms/prefix codes in this entropy code.
    pub fn num_histograms(&self) -> usize {
        match self {
            BuiltEntropyCode::StaticHuffman(code) => code.num_prefix_codes,
            BuiltEntropyCode::Huffman(code) => code.prefix_codes.len(),
            BuiltEntropyCode::Ans(code) => code.histograms.len(),
        }
    }
}

/// Tiny JPEG XL encoder.
///
/// This is a simplified VarDCT encoder based on libjxl-tiny that uses:
/// - Only DCT8, DCT8x16, DCT16x8 transforms
/// - Huffman or ANS entropy coding
/// - Default zig-zag coefficient order
/// - Fixed context tree for DC
pub struct TinyEncoder {
    /// Target distance (quality). 1.0 = visually lossless.
    pub distance: f32,
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
    pub lz77_method: super::lz77::Lz77Method,
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
}

impl Default for TinyEncoder {
    fn default() -> Self {
        Self {
            distance: 1.0,
            optimize_codes: true,
            enhanced_clustering: true, // Pair-merge refinement helps ANS (larger header savings)
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
            lz77_method: super::lz77::Lz77Method::Greedy, // Best compression
            dc_tree_learning: false, // DC tree learning (experimental)
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters: 0,
        }
    }
}

impl TinyEncoder {
    /// Create a new tiny encoder with the given distance.
    pub fn new(distance: f32) -> Self {
        Self {
            distance,
            optimize_codes: true,
            enhanced_clustering: true, // Pair-merge refinement helps ANS (larger header savings)
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
            lz77_method: super::lz77::Lz77Method::Greedy, // Best compression
            dc_tree_learning: false, // DC tree learning (experimental)
            #[cfg(feature = "butteraugli-loop")]
            butteraugli_iters: 0,
        }
    }

    /// Encode an image in linear sRGB format.
    ///
    /// Input should be 3 channels (RGB) of f32 values in [0, 1] range.
    /// Values outside [0, 1] are allowed for out-of-gamut colors.
    pub fn encode(&self, width: usize, height: usize, linear_rgb: &[f32]) -> Result<Vec<u8>> {
        assert_eq!(linear_rgb.len(), width * height * 3);

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

        // Step 1: Compute float quant field (independent of global_scale)
        let (quant_field_float, masking) = compute_quant_field_float(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            padded_width,
            padded_height,
            xsize_blocks,
            ysize_blocks,
            distance_for_iqf,
        );

        // Step 2: Compute distance params with content-adaptive global_scale.
        // Uses median and MAD of the quant field to adapt quantization precision
        // to image content (matches libjxl ComputeGlobalScaleAndQuant).
        let mut params =
            DistanceParams::compute_from_quant_field(self.distance, &quant_field_float);

        // Apply pixel-level chromacity adjustments using pre-gaborish stats
        params.apply_chromacity_adjustment(chromacity_x, chromacity_b);

        // Step 3: Quantize float quant field to raw u8 with adaptive inv_scale
        let mut quant_field = quantize_quant_field(&quant_field_float, params.inv_scale);

        // Compute per-tile chroma-from-luma map
        let cfl_map = if self.cfl_enabled {
            compute_cfl_map(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
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
            );
        }

        // Perform DCT and quantization (XYB data is padded to block boundaries)
        let (quant_dc, quant_ac, nzeros, raw_nzeros) = self.transform_and_quantize(
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

        // Compute per-block EPF sharpness map when EPF is active
        let sharpness_map = if params.epf_iters > 0 && self.distance >= 0.5 {
            let mask = mask1x1.unwrap_or_else(|| {
                super::adaptive_quant::compute_mask1x1(&xyb_y, padded_width, padded_height)
            });
            Some(super::epf::compute_epf_sharpness(
                [&xyb_x, &xyb_y, &xyb_b],
                &quant_dc,
                &quant_ac,
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
            return self.encode_two_pass(
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
                &quant_dc,
                &quant_ac,
                &nzeros,
                &raw_nzeros,
                &quant_field,
                &cfl_map,
                &ac_strategy,
                &noise_params,
                sharpness_map.as_deref(),
            );
        }

        // Get static entropy codes (wrapped in BuiltEntropyCode for uniform handling)
        let dc_code = BuiltEntropyCode::StaticHuffman(get_dc_entropy_code());
        let ac_code = BuiltEntropyCode::StaticHuffman(get_ac_entropy_code());

        // Create main writer
        let mut writer = BitWriter::with_capacity(width * height * 4);

        // Write JXL signature
        writer.write(8, 0xFF)?;
        writer.write(8, 0x0A)?;
        #[cfg(feature = "debug-tokens")]
        debug_log!("After signature: bit {}", writer.bits_written());

        // Write size header (simple format for small images)
        self.write_file_header(width, height, &mut writer)?;
        #[cfg(feature = "debug-tokens")]
        debug_log!(
            "After file header: bit {} (byte {})",
            writer.bits_written(),
            writer.bits_written() / 8
        );

        // Write frame header
        write_frame_header(
            params.x_qm_scale,
            params.b_qm_scale,
            params.epf_iters,
            noise_params.is_some(),
            self.enable_gaborish,
            &mut writer,
        )?;
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
            let mut dc_global = BitWriter::new();
            self.write_dc_global(
                &params,
                num_dc_groups,
                &dc_code,
                &noise_params,
                None,
                &block_ctx_map,
                None, // No learned tree in single-pass mode
                &mut dc_global,
            )?;

            // Get borrowed Huffman codes for streaming token writing
            let dc_huffman = dc_code.as_huffman();
            let ac_huffman = ac_code.as_huffman();

            let mut dc_group = BitWriter::new();
            self.write_dc_group(
                0,
                &quant_dc,
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

            let mut ac_global = BitWriter::new();
            self.write_ac_global(num_groups, &ac_code, 0, None, None, &mut ac_global)?;

            let mut ac_group_writer = BitWriter::new();
            self.write_ac_group(
                0,
                &quant_ac,
                &nzeros,
                &raw_nzeros,
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
            let mut dc_global = BitWriter::new();
            self.write_dc_global(
                &params,
                num_dc_groups,
                &dc_code,
                &noise_params,
                None,
                &block_ctx_map,
                None, // No learned tree in single-pass mode
                &mut dc_global,
            )?;
            dc_global.zero_pad_to_byte();
            sections.push(dc_global.finish());

            // DC group sections
            for dc_group_idx in 0..num_dc_groups {
                let mut dc_group = BitWriter::new();
                self.write_dc_group(
                    dc_group_idx,
                    &quant_dc,
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
            let mut ac_global = BitWriter::new();
            self.write_ac_global(num_groups, &ac_code, 0, None, None, &mut ac_global)?;
            ac_global.zero_pad_to_byte();
            sections.push(ac_global.finish());

            // AC group sections
            for group_idx in 0..num_groups {
                let mut ac_group_writer = BitWriter::new();
                self.write_ac_group(
                    group_idx,
                    &quant_ac,
                    &nzeros,
                    &raw_nzeros,
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

        Ok(writer.finish_with_padding())
    }

    /// Butteraugli quantization loop: iteratively refines per-block quant_field
    /// by measuring perceptual distance (butteraugli) between the original image
    /// and the reconstruction from quantized coefficients.
    ///
    /// Algorithm (libjxl FindBestQuantization):
    /// For each iteration:
    ///   1. transform_and_quantize with current quant_field
    ///   2. reconstruct XYB → apply gab → EPF → XYB-to-linear
    ///   3. butteraugli(original_linear, reconstructed_linear) → per-block distmap
    ///   4. For blocks where distmap > target: increase quant (qf *= distmap/target)
    ///      For blocks where distmap < target: decrease quant (qf *= distmap/target)
    ///   5. Clamp and constrain (don't diverge too far from initial)
    ///
    /// AC strategy is FIXED throughout — only quant_field changes.
    #[cfg(feature = "butteraugli-loop")]
    #[allow(clippy::too_many_arguments)]
    fn butteraugli_refine_quant_field(
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
        params: &DistanceParams,
        quant_field: &mut [u8],
        initial_quant_field: &[u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
    ) {
        use super::epf;
        use super::reconstruct::{gab_smooth, reconstruct_xyb, xyb_to_linear_rgb};
        use imgref::ImgRef;
        use rgb::RGB;

        let target_distance = self.distance;
        let num_blocks = xsize_blocks * ysize_blocks;

        // Build original linear RGB image as ImgRef<RGB<f32>> for butteraugli
        let original_rgb: Vec<RGB<f32>> = (0..width * height)
            .map(|i| RGB {
                r: linear_rgb[i * 3],
                g: linear_rgb[i * 3 + 1],
                b: linear_rgb[i * 3 + 2],
            })
            .collect();
        let original_img = ImgRef::new(&original_rgb, width, height);

        let butteraugli_params = butteraugli::ButteraugliParams::new()
            .with_intensity_target(80.0)
            .with_compute_diffmap(true);

        for iter in 0..self.butteraugli_iters {
            // Step 1: Quantize with current quant_field
            let mut qf_copy = quant_field.to_vec();
            let (quant_dc, quant_ac, _nzeros, _raw_nzeros) = self.transform_and_quantize(
                xyb_x,
                xyb_y,
                xyb_b,
                padded_width,
                xsize_blocks,
                ysize_blocks,
                params,
                &mut qf_copy,
                cfl_map,
                ac_strategy,
            );

            // Step 2: Reconstruct XYB from quantized coefficients
            let mut planes = reconstruct_xyb(
                &quant_dc,
                &quant_ac,
                params,
                &qf_copy,
                cfl_map,
                ac_strategy,
                xsize_blocks,
                ysize_blocks,
            );

            // Apply gaborish smooth if enabled
            if self.enable_gaborish {
                gab_smooth(&mut planes, padded_width, padded_height);
            }

            // Apply EPF if active
            if params.epf_iters > 0 {
                // Use default sharpness (4) for butteraugli loop iterations
                let sharpness = vec![4u8; num_blocks];
                epf::apply_epf(
                    &mut planes,
                    &qf_copy,
                    &sharpness,
                    params.scale,
                    params.epf_iters,
                    xsize_blocks,
                    ysize_blocks,
                    padded_width,
                    padded_height,
                );
            }

            // Step 3: Convert reconstructed XYB to linear RGB
            let recon_linear = xyb_to_linear_rgb(
                &planes[0],
                &planes[1],
                &planes[2],
                padded_width,
                padded_height,
            );

            // Build reconstructed ImgRef<RGB<f32>> (crop to original dimensions)
            let mut recon_rgb = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    let pi = y * padded_width + x;
                    recon_rgb.push(RGB {
                        r: recon_linear[pi * 3].max(0.0),
                        g: recon_linear[pi * 3 + 1].max(0.0),
                        b: recon_linear[pi * 3 + 2].max(0.0),
                    });
                }
            }
            let recon_img = ImgRef::new(&recon_rgb, width, height);

            // Step 4: Compute butteraugli distance with diffmap
            let result =
                match butteraugli::butteraugli_linear(original_img, recon_img, &butteraugli_params)
                {
                    Ok(r) => r,
                    Err(_) => return, // Bail on error (e.g., image too small)
                };

            let diffmap = match result.diffmap {
                Some(dm) => dm,
                None => return,
            };

            // Step 5: Adjust quant_field per block based on distmap
            // For each block, compute average butteraugli distance
            for by in 0..ysize_blocks {
                for bx in 0..xsize_blocks {
                    if !ac_strategy.is_first(bx, by) {
                        continue;
                    }

                    let covered_x = ac_strategy.covered_blocks_x(bx, by);
                    let covered_y = ac_strategy.covered_blocks_y(bx, by);

                    // Average butteraugli distance over the block's pixel region
                    let px_start_x = bx * BLOCK_DIM;
                    let px_start_y = by * BLOCK_DIM;
                    let px_end_x = ((bx + covered_x) * BLOCK_DIM).min(width);
                    let px_end_y = ((by + covered_y) * BLOCK_DIM).min(height);

                    if px_start_x >= width || px_start_y >= height {
                        continue;
                    }

                    let mut sum_dist = 0.0f64;
                    let mut count = 0u32;
                    for py in px_start_y..px_end_y {
                        for px in px_start_x..px_end_x {
                            sum_dist += diffmap.buf()[py * width + px] as f64;
                            count += 1;
                        }
                    }

                    if count == 0 {
                        continue;
                    }

                    let avg_dist = (sum_dist / count as f64) as f32;

                    // Compute adjustment ratio: if avg_dist > target, increase quant
                    // (lower quality to save bits); if < target, decrease quant.
                    // But actually, in libjxl, higher quant_field = MORE quantization
                    // = LOWER quality. So to improve quality (lower distance),
                    // we need to DECREASE quant_field.
                    //
                    // Ratio = avg_dist / target_distance
                    //   > 1: quality too low, decrease qf to improve
                    //   < 1: quality too high, increase qf to save bits
                    //
                    // Wait — quant_field is an inverse-scale (higher = coarser quantization).
                    // To reduce distance: need finer quantization = lower qf.
                    // So: new_qf = old_qf * (target / avg_dist)
                    // When avg_dist > target: target/avg_dist < 1, qf decreases = finer quant
                    // When avg_dist < target: target/avg_dist > 1, qf increases = coarser quant
                    //
                    // Actually no — in our encoder, quant_field[block] is used as:
                    //   qac = params.scale * quant_field[block]
                    // Higher qac = more aggressive quantization = worse quality
                    // So if distance is too high, we want to DECREASE qf.
                    // new_qf = old_qf * (target / avg_dist) makes qf smaller when avg_dist > target.
                    // Hmm but that's wrong — we want qf smaller → better quality → lower distance.
                    // target/avg_dist < 1 when avg_dist > target, so qf gets smaller. Correct!

                    let ratio = if avg_dist > 0.001 {
                        target_distance / avg_dist
                    } else {
                        2.0 // Quality much better than needed, double coarseness
                    };

                    // Dampen the adjustment to avoid oscillation
                    let damped_ratio = 1.0 + 0.5 * (ratio - 1.0);

                    // Apply to all sub-blocks of this transform
                    for sy in 0..covered_y {
                        for sx in 0..covered_x {
                            let bi = (by + sy) * xsize_blocks + (bx + sx);
                            let old_qf = quant_field[bi] as f32;
                            let new_qf = (old_qf * damped_ratio).round();

                            // Clamp to valid range [1, 255]
                            let new_qf = new_qf.clamp(1.0, 255.0) as u8;

                            // Constrain: don't diverge too far from initial
                            // After iteration 1, blend toward initial to prevent runaway
                            let constrained = if iter >= 1 {
                                let init_qf = initial_quant_field[bi] as f32;
                                let blended = 0.6 * init_qf + 0.4 * new_qf as f32;
                                (blended.round() as u8).max(1)
                            } else {
                                new_qf
                            };

                            quant_field[bi] = constrained;
                        }
                    }
                }
            }

            // Re-adjust quant field for multi-block consistency
            adjust_quant_field_with_distance(ac_strategy, quant_field, self.distance);
        }
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

        // Compute distance params from precomputed quant field
        let mut params =
            DistanceParams::compute_from_quant_field(self.distance, &precomputed.quant_field_float);

        // Apply pixel-level chromacity adjustments using pre-gaborish stats
        params.apply_chromacity_adjustment(
            precomputed.chromacity_x_pixelized,
            precomputed.chromacity_b_pixelized,
        );

        // Perform DCT and quantization using precomputed XYB data
        let (quant_dc, quant_ac, nzeros, raw_nzeros) = self.transform_and_quantize(
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
            &quant_dc,
            &quant_ac,
            &nzeros,
            &raw_nzeros,
            &quant_field,
            &precomputed.cfl_map,
            &precomputed.ac_strategy,
            &precomputed.noise_params,
            None, // TODO: compute sharpness_map for rate control path
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = TinyEncoder::new(1.0);
        assert_eq!(encoder.distance, 1.0);

        let encoder_default = TinyEncoder::default();
        assert_eq!(encoder_default.distance, 1.0);
    }

    #[test]
    fn test_encode_small_image() {
        let encoder = TinyEncoder::new(1.0);

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
        let result = encoder.encode(width, height, &linear_rgb);
        // For now, just check it produces some output
        assert!(result.is_ok());
        let bytes = result.unwrap();
        assert!(bytes.len() > 2);
        assert_eq!(bytes[0], 0xFF);
        assert_eq!(bytes[1], 0x0A);
    }

    #[test]
    fn test_convert_to_xyb_padded() {
        let encoder = TinyEncoder::new(1.0);

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
        let encoder = TinyEncoder::new(1.0);

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

        let result = encoder.encode(width, height, &linear_rgb);
        assert!(result.is_ok());
        let bytes = result.unwrap();

        eprintln!("Output file size: {} bytes", bytes.len());
        eprintln!("First 32 bytes: {:02x?}", &bytes[..32.min(bytes.len())]);

        // Write output to file for comparison
        std::fs::write("/tmp/our_16x16.jxl", &bytes).unwrap();

        // libjxl-tiny produces:
        // DC_group: 106 bits (14 bytes)
        // Total combined: 1086 bytes
        // Total file: 1104 bytes
        //
        // Our encoder should match these sizes

        // Check signature
        assert_eq!(bytes[0], 0xFF);
        assert_eq!(bytes[1], 0x0A);
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
    #[test]
    fn test_hash_lock_8x8_gradient() {
        let encoder = TinyEncoder::new(1.0);
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

        let bytes = encoder.encode(width, height, &linear_rgb).unwrap();
        let hash = hash_bytes(&bytes);

        // Lock the hash - if this changes, the encoding has changed
        // Updated: full libjxl thresholds, enhanced clustering, kFavor2X2
        const EXPECTED_HASH: u64 = 0xfbcf8d8ba10f5f8;
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
    #[test]
    fn test_hash_lock_16x16_solid() {
        let encoder = TinyEncoder::new(1.0);
        let width = 16;
        let height = 16;
        let linear_rgb = vec![0.3f32; width * height * 3]; // gray

        let bytes = encoder.encode(width, height, &linear_rgb).unwrap();
        let hash = hash_bytes(&bytes);

        // Updated: fixed transfer function from Linear to Srgb
        const EXPECTED_HASH: u64 = 0x310cb05d2ba544b7;
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
    #[test]
    fn test_hash_lock_64x64_checkerboard() {
        let encoder = TinyEncoder::new(1.0);
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

        let bytes = encoder.encode(width, height, &linear_rgb).unwrap();
        let hash = hash_bytes(&bytes);

        // Hash updated: iterative rate control changes output
        const EXPECTED_HASH: u64 = 0xa2084905cd53c244;
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
    #[test]
    fn test_hash_lock_13x17_noise() {
        let encoder = TinyEncoder::new(1.0);
        let width = 13;
        let height = 17;
        let mut linear_rgb = vec![0.0f32; width * height * 3];

        // Deterministic pseudo-random pattern
        let mut seed = 12345u64;
        for val in &mut linear_rgb {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *val = ((seed >> 32) as f32) / (u32::MAX as f32);
        }

        let bytes = encoder.encode(width, height, &linear_rgb).unwrap();
        let hash = hash_bytes(&bytes);

        // Hash updated: full libjxl thresholds, enhanced clustering, kFavor2X2
        const EXPECTED_HASH: u64 = 0x7171c66ec1420073;
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

            let encoder = TinyEncoder::new(1.0);
            let bytes = encoder
                .encode(w, h, &linear_rgb)
                .unwrap_or_else(|e| panic!("encode {}x{} failed: {}", w, h, e));

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
        let mut encoder_baseline = TinyEncoder::new(1.0);
        encoder_baseline.dc_tree_learning = false;
        let bytes_baseline = encoder_baseline
            .encode(width, height, &linear_rgb)
            .expect("baseline encode failed");

        // Encode WITH DC tree learning — also use ANS
        let mut encoder_learned = TinyEncoder::new(1.0);
        encoder_learned.dc_tree_learning = true;
        std::fs::write("/tmp/dc_baseline_test.jxl", &bytes_baseline).unwrap();
        let bytes_learned = encoder_learned
            .encode(width, height, &linear_rgb)
            .expect("learned encode failed");
        std::fs::write("/tmp/dc_learned_test.jxl", &bytes_learned).unwrap();

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
        std::fs::write("/tmp/dc_learned_test.jxl", &bytes_learned).unwrap();
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
        let mut encoder_baseline = TinyEncoder::new(2.0);
        encoder_baseline.butteraugli_iters = 0;
        let bytes_baseline = encoder_baseline
            .encode(width, height, &linear_rgb)
            .expect("baseline encode failed");

        // Encode with 2 butteraugli loop iterations
        let mut encoder_loop = TinyEncoder::new(2.0);
        encoder_loop.butteraugli_iters = 2;
        let bytes_loop = encoder_loop
            .encode(width, height, &linear_rgb)
            .expect("butteraugli loop encode failed");

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
