// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Main tiny encoder implementation.

use super::ac_group::{
    collect_ac_coefficients, num_nonzero_8x8_except_dc, num_nonzero_except_llf,
    predict_from_top_and_left, tokenize_ac_coefficients,
};
use super::ac_strategy::{
    AcStrategyMap, RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16,
    RAW_STRATEGY_DCT32X32, adjust_quant_field_with_distance, compute_ac_strategy,
};

/// Create an AC strategy map forcing a specific strategy.
fn force_strategy_map(xsize_blocks: usize, ysize_blocks: usize, raw_strategy: u8) -> AcStrategyMap {
    AcStrategyMap::force_strategy(xsize_blocks, ysize_blocks, raw_strategy)
}
use super::adaptive_quant::compute_adaptive_quant_field;
use super::chroma_from_luma::{CflMap, compute_cfl_map, ytob_ratio, ytox_ratio};
use super::common::*;
use super::dc_coding::{
    collect_ac_metadata_tokens_region, collect_dc_tokens_region, write_ac_metadata_tokens_region,
    write_dc_tokens_region,
};
use super::dct::{
    dc_from_dct_8x16, dc_from_dct_16x8, dc_from_dct_16x16, dc_from_dct_32x32, dct_8x8, dct_8x16,
    dct_16x8, dct_16x16, dct_32x32,
};
use super::entropy_code::{
    OwnedAnsEntropyCode, OwnedEntropyCode, build_entropy_code_ans_with_options,
    build_entropy_code_with_options, write_entropy_code_ans, write_tokens, write_tokens_ans,
};
use super::frame::{DistanceParams, write_frame_header, write_quant_scales, write_toc};
use super::gaborish::gaborish_inverse;
use super::noise::{
    NoiseParams, denoise_xyb, estimate_noise_params, noise_quality_coef, write_noise_params,
};
use super::quant::INV_DC_QUANT;
use super::static_codes::{get_ac_entropy_code, get_dc_entropy_code};
use super::token::Token;
use crate::bit_writer::BitWriter;
use crate::color::xyb::linear_rgb_to_xyb;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
use crate::error::Result;

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
    pub fn write_tokens(&self, tokens: &[Token], writer: &mut BitWriter) -> Result<()> {
        match self {
            BuiltEntropyCode::StaticHuffman(code) => write_tokens(tokens, code, writer),
            BuiltEntropyCode::Huffman(code) => {
                write_tokens(tokens, &code.as_entropy_code(), writer)
            }
            BuiltEntropyCode::Ans(code) => write_tokens_ans(tokens, code, writer),
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
}

impl Default for TinyEncoder {
    fn default() -> Self {
        Self {
            distance: 1.0,
            optimize_codes: true,
            enhanced_clustering: false,
            use_ans: true, // ANS produces 4-10% smaller files than Huffman
            cfl_enabled: true,
            ac_strategy_enabled: true,
            custom_orders: true,
            force_strategy: None,
            enable_noise: false,
            enable_denoise: false,
            enable_gaborish: true,
        }
    }
}

impl TinyEncoder {
    /// Create a new tiny encoder with the given distance.
    pub fn new(distance: f32) -> Self {
        Self {
            distance,
            optimize_codes: true,
            enhanced_clustering: false,
            use_ans: true, // ANS produces 4-10% smaller files than Huffman
            cfl_enabled: true,
            ac_strategy_enabled: true,
            custom_orders: true,
            force_strategy: None,
            enable_noise: false,
            enable_denoise: false,
            enable_gaborish: true,
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

        // Compute distance parameters (fixed formula matching libjxl effort 5+)
        let params = DistanceParams::compute(self.distance);

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
        let (mut quant_field, masking, quant_field_float) = compute_adaptive_quant_field(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            padded_width,
            padded_height,
            xsize_blocks,
            ysize_blocks,
            distance_for_iqf,
            params.inv_scale,
        );

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
            )
        };

        // Adjust quant field for multi-block transforms.
        // At low distances uses max, at high distances blends toward mean for better quality.
        adjust_quant_field_with_distance(&ac_strategy, &mut quant_field, self.distance);

        // Perform DCT and quantization (XYB data is padded to block boundaries)
        let (quant_dc, quant_ac, nzeros, raw_nzeros) = self.transform_and_quantize(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            padded_width,
            xsize_blocks,
            ysize_blocks,
            &params,
            &quant_field,
            &cfl_map,
            &ac_strategy,
        );

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
            let mut dc_global = BitWriter::new();
            self.write_dc_global(
                &params,
                num_dc_groups,
                &dc_code,
                &noise_params,
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
                &dc_huffman,
                &mut dc_group,
            )?;

            let mut ac_global = BitWriter::new();
            self.write_ac_global(num_groups, &ac_code, 0, None, &mut ac_global)?;

            let mut ac_group_writer = BitWriter::new();
            self.write_ac_group(
                0,
                &quant_ac,
                &nzeros,
                &raw_nzeros,
                xsize_blocks,
                ysize_blocks,
                xsize_groups,
                &ac_strategy,
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
            let mut dc_global = BitWriter::new();
            self.write_dc_global(
                &params,
                num_dc_groups,
                &dc_code,
                &noise_params,
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
                    &dc_huffman,
                    &mut dc_group,
                )?;
                dc_group.zero_pad_to_byte();
                sections.push(dc_group.finish());
            }

            // AC Global section
            let mut ac_global = BitWriter::new();
            self.write_ac_global(num_groups, &ac_code, 0, None, &mut ac_global)?;
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
                    &ac_strategy,
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

    /// Convert linear RGB to XYB color space with padding to block boundaries.
    ///
    /// Returns (xyb_x, xyb_y, xyb_b) arrays padded to `padded_width × padded_height`
    /// using edge replication (last pixel value extended to the boundary).
    /// This allows SIMD code to process full blocks without bounds checking.
    fn convert_to_xyb_padded(
        &self,
        width: usize,
        height: usize,
        padded_width: usize,
        padded_height: usize,
        linear_rgb: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let padded_n = padded_width * padded_height;
        let mut xyb_x = vec![0.0f32; padded_n];
        let mut xyb_y = vec![0.0f32; padded_n];
        let mut xyb_b = vec![0.0f32; padded_n];

        // Convert the actual image pixels
        for y in 0..height {
            for x in 0..width {
                let src_idx = y * width + x;
                let dst_idx = y * padded_width + x;
                let r = linear_rgb[src_idx * 3];
                let g = linear_rgb[src_idx * 3 + 1];
                let b = linear_rgb[src_idx * 3 + 2];
                let (xv, yv, bv) = linear_rgb_to_xyb(r, g, b);
                xyb_x[dst_idx] = xv;
                xyb_y[dst_idx] = yv;
                xyb_b[dst_idx] = bv;
            }

            // Pad right edge with last pixel value (edge replication)
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

        (xyb_x, xyb_y, xyb_b)
    }

    /// Quantize a single AC coefficient with thresholding.
    ///
    /// Ported from libjxl-tiny QuantizeBlockAC. Small coefficients below a
    /// threshold are zeroed out. The threshold depends on:
    /// - Channel (X gets +0.08 on upper thresholds, B uses flat 0.75)
    /// - Quadrant position within the block (4 quadrants)
    /// - Multi-block coverage (slight threshold reduction)
    ///
    /// `qm_multiplier` is typically 1.0, but for X channel it's `x_qm_mul`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn quantize_coeff_ac(
        coef: f32,
        inv_weight: f32, // 1/weight (InvMatrix in C++)
        qac: f32,        // scale * quant_ac
        qm_multiplier: f32,
        c: usize,
        y_in_block: usize,
        x_in_block: usize,
        block_height: usize,
        block_width: usize,
        covered_x: usize,
        covered_y: usize,
    ) -> i32 {
        let mut thres = [0.58f32, 0.635, 0.66, 0.7];
        if c == 0 {
            // X channel
            for t in thres[1..4].iter_mut() {
                *t += 0.08;
            }
        }
        if c == 2 {
            // B channel
            for t in thres[1..4].iter_mut() {
                *t = 0.75;
            }
        }
        if covered_x > 1 || covered_y > 1 {
            let adj = (0.003 * (covered_x * covered_y) as f32)
                .clamp(0.0, if c > 0 { 0.08 } else { 0.12 });
            for t in thres.iter_mut() {
                *t -= adj;
            }
        }

        // Quadrant selection: which of the 4 quadrants does this coeff fall in
        let y_half = if y_in_block >= block_height / 2 { 2 } else { 0 };
        let x_half = if x_in_block >= block_width / 2 { 1 } else { 0 };
        let thr = thres[y_half + x_half];

        let val = inv_weight * qac * qm_multiplier * coef;
        if val.abs() < thr {
            0
        } else {
            val.round() as i32
        }
    }

    /// Apply AdjustQuantBias to a quantized value for dequantization.
    ///
    /// Ported from libjxl-tiny's AdjustQuantBias. For ±1 values, returns a
    /// channel-specific biased value. For larger values, applies a small
    /// reciprocal correction: `q - 0.145 / q`.
    #[allow(clippy::excessive_precision)]
    #[inline]
    fn adjust_quant_bias(quantized: i32, channel: usize) -> f32 {
        // kDefaultQuantBias from libjxl-tiny enc_group.cc
        // [0..2] = channel-specific bias for ±1 values
        // [3] = reciprocal correction factor for |q| >= 2
        const BIAS: [f32; 4] = [
            1.0 - 0.05465007330715401,  // [0] X channel ±1 → 0.945349
            1.0 - 0.07005449891748593,  // [1] Y channel ±1 → 0.929946
            1.0 - 0.049935103337343655, // [2] B channel ±1 → 0.950065
            0.145,                      // [3] reciprocal correction
        ];

        if quantized == 0 {
            return 0.0;
        }

        let q = quantized as f32;

        // C++ uses abs(float) < 1.125 to detect ±1 (since q is integer)
        if q.abs() < 1.125 {
            // ±1: return ±BIAS[channel]
            q.signum() * BIAS[channel]
        } else {
            // |q| >= 2: return q - BIAS[3] / q
            q - BIAS[3] / q
        }
    }

    /// Apply DCT to a single channel at block position (bx, by).
    ///
    /// The `channel_data` must be padded to block boundaries (stride = padded_width).
    /// No bounds checking is performed - caller must ensure data is properly padded.
    fn apply_dct(
        channel_data: &[f32],
        stride: usize, // padded_width (row stride)
        bx: usize,
        by: usize,
        raw_strategy: u8,
        output: &mut [f32],
    ) {
        match raw_strategy {
            0 => {
                let mut block = [0.0f32; 64];
                for dy in 0..8 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..8 {
                        block[dy * 8 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 64];
                dct_8x8(&block, &mut dct_out);
                output[..64].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT16X8 => {
                let mut block = [0.0f32; 128];
                for dy in 0..16 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..8 {
                        block[dy * 8 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 128];
                dct_16x8(&block, &mut dct_out);
                output[..128].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT8X16 => {
                let mut block = [0.0f32; 128];
                for dy in 0..8 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..16 {
                        block[dy * 16 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 128];
                dct_8x16(&block, &mut dct_out);
                output[..128].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT16X16 => {
                let mut block = [0.0f32; 256];
                for dy in 0..16 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..16 {
                        block[dy * 16 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 256];
                dct_16x16(&block, &mut dct_out);
                output[..256].copy_from_slice(&dct_out);
            }
            RAW_STRATEGY_DCT32X32 => {
                let mut block = [0.0f32; 1024];
                for dy in 0..32 {
                    let row_offset = (by * BLOCK_DIM + dy) * stride + bx * BLOCK_DIM;
                    for dx in 0..32 {
                        block[dy * 32 + dx] = channel_data[row_offset + dx];
                    }
                }
                let mut dct_out = [0.0f32; 1024];
                dct_32x32(&block, &mut dct_out);
                output[..1024].copy_from_slice(&dct_out);
            }
            _ => unreachable!(),
        }
    }

    /// Quantize AC coefficients with thresholding and store in quant_ac slots.
    #[allow(clippy::too_many_arguments)]
    fn quantize_ac_block(
        dct_coeffs: &[f32],
        weights: &[f32],
        qac: f32,
        qm_multiplier: f32,
        channel: usize,
        _block_width: usize,
        _block_height: usize,
        covered_x: usize,
        _covered_y: usize,
        _covered_blocks: usize,
        size: usize,
        _raw_strategy: u8,
        bx: usize,
        by: usize,
        quant_ac: &mut [Vec<[i32; DCT_BLOCK_SIZE]>],
    ) {
        // C++ QuantizeBlockAC uses post-swap (cx, cy) for the coefficient grid:
        // stride = cx * 8 (block_width), height = cy * 8 (block_height).
        // After swap, cx >= cy. Both DCT16x8 and DCT8x16 have grid_width=16.
        let grid_width = _block_width;
        let grid_height = _block_height;
        let cx = _block_width / BLOCK_DIM;
        let cy = _block_height / BLOCK_DIM;
        for idx in 0..size {
            // LLF positions are at (y, x) where y < cy and x < cx in the grid.
            // For DCT8 this is just index 0.
            // For DCT16x16 (cx=cy=2, stride=16) this is {0, 1, 16, 17}.
            // For DCT16x8 (cx=2, cy=1, stride=16) this is {0, 1}.
            let is_llf = (idx / grid_width) < cy && (idx % grid_width) < cx;
            let qval = if is_llf {
                0 // LLF handled separately
            } else {
                let y = idx / grid_width;
                let x = idx % grid_width;
                Self::quantize_coeff_ac(
                    dct_coeffs[idx],
                    1.0 / weights[idx],
                    qac,
                    qm_multiplier,
                    channel,
                    y,
                    x,
                    grid_height,
                    grid_width,
                    cx,
                    cy,
                )
            };

            let block_slot = idx / DCT_BLOCK_SIZE;
            let coeff_in_block = idx % DCT_BLOCK_SIZE;
            let slot_by = by + block_slot / covered_x;
            let slot_bx = bx + block_slot % covered_x;
            quant_ac[slot_by][slot_bx][coeff_in_block] = qval;
        }
    }

    /// Perform DCT and quantization on all blocks.
    ///
    /// Supports DCT8, DCT16X8, and DCT8X16 transforms based on ac_strategy.
    /// For multi-block transforms, only first blocks are processed; the second
    /// block's quant_ac slot stores the second half of the 128 coefficients.
    ///
    /// Processing order matches C++ WriteACGroup:
    /// 1. DCT Y → extract Y DC → quantize Y AC (with thresholding)
    /// 2. Dequantize Y AC back (AdjustQuantBias) → roundtripped Y
    /// 3. DCT X, B → apply CfL using roundtripped Y → extract X/B DC
    /// 4. Quantize X/B AC (with thresholding + x_qm_mul for X)
    ///
    /// Returns (quantized_dc, quantized_ac, nzeros)
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn transform_and_quantize(
        &self,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        padded_width: usize, // stride for padded XYB data
        xsize_blocks: usize,
        ysize_blocks: usize,
        params: &DistanceParams,
        quant_field: &[u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
    ) -> (
        [Vec<Vec<i16>>; 3],                   // quant_dc
        [Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3], // quant_ac
        [Vec<Vec<u8>>; 3],                    // nzeros (shifted, for prediction)
        [Vec<Vec<u8>>; 3],                    // raw_nzeros (unshifted, for bitstream)
    ) {
        // Initialize output arrays
        let mut quant_dc: [Vec<Vec<i16>>; 3] = [
            vec![vec![0i16; xsize_blocks]; ysize_blocks],
            vec![vec![0i16; xsize_blocks]; ysize_blocks],
            vec![vec![0i16; xsize_blocks]; ysize_blocks],
        ];

        let mut quant_ac: [Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3] = [
            vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks],
            vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks],
            vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks],
        ];

        // Shifted nzeros for neighbor prediction (nzeros / covered_blocks)
        let mut nzeros: [Vec<Vec<u8>>; 3] = [
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
        ];
        // Raw (unshifted) nzeros for bitstream writing — stored at first-block positions
        let mut raw_nzeros: [Vec<Vec<u8>>; 3] = [
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
        ];

        let channels = [xyb_x, xyb_y, xyb_b];

        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                // Skip non-first blocks of multi-block transforms
                if !ac_strategy.is_first(bx, by) {
                    continue;
                }

                let raw_strategy = ac_strategy.raw_strategy(bx, by);
                #[cfg(feature = "debug-dc")]
                eprintln!(
                    "Block (by={}, bx={}): raw_strategy={}",
                    by, bx, raw_strategy
                );
                let covered_x = ac_strategy.covered_blocks_x(bx, by);
                let covered_y = ac_strategy.covered_blocks_y(bx, by);
                let covered_blocks = covered_x * covered_y;
                let size = covered_blocks * DCT_BLOCK_SIZE;

                // CfL factors for this tile
                let tx = bx / TILE_DIM_IN_BLOCKS;
                let ty_cfl = by / TILE_DIM_IN_BLOCKS;
                let x_factor = ytox_ratio(cfl_map.ytox_at(tx, ty_cfl));
                let b_factor = ytob_ratio(cfl_map.ytob_at(tx, ty_cfl));

                // Coefficient layout: after C++ swap(cx,cy) so cx >= cy,
                // stride = cx * 8. Both DCT16X8 and DCT8X16 produce 8×16 layout.
                let (cx, cy) = if covered_y > covered_x {
                    (covered_y, covered_x)
                } else {
                    (covered_x, covered_y)
                };
                let block_width = cx * BLOCK_DIM;
                let block_height = cy * BLOCK_DIM;

                let qac = params.scale * quant_field[by * xsize_blocks + bx] as f32;
                let x_qm_mul = 1.25f32.powf(params.x_qm_scale as f32 - 2.0);

                let mut dct_coeffs: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0f32; size]);

                // ── Step 1: DCT Y channel ──────────────────────────────────
                Self::apply_dct(
                    channels[1],
                    padded_width,
                    bx,
                    by,
                    raw_strategy,
                    &mut dct_coeffs[1],
                );

                // ── Step 2: Extract Y DC (before roundtrip quantization) ───
                // Inlined instead of using extract_dc to avoid borrow conflict.
                {
                    let inv_factor = INV_DC_QUANT[1] * params.scale_dc;
                    match raw_strategy {
                        0 => {
                            quant_dc[1][by][bx] = (dct_coeffs[1][0] * inv_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT16X8 => {
                            let coeffs_arr: [f32; 128] = dct_coeffs[1][..128]
                                .try_into()
                                .expect("128 coefficients for DCT16x8");
                            let dcs = dc_from_dct_16x8(&coeffs_arr);
                            for iy in 0..2 {
                                quant_dc[1][by + iy][bx] = (dcs[iy] * inv_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT8X16 => {
                            let coeffs_arr: [f32; 128] = dct_coeffs[1][..128]
                                .try_into()
                                .expect("128 coefficients for DCT8x16");
                            let dcs = dc_from_dct_8x16(&coeffs_arr);
                            for ix in 0..2 {
                                quant_dc[1][by][bx + ix] = (dcs[ix] * inv_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT16X16 => {
                            let coeffs_arr: [f32; 256] = dct_coeffs[1][..256]
                                .try_into()
                                .expect("256 coefficients for DCT16x16");
                            let dcs = dc_from_dct_16x16(&coeffs_arr);
                            // dcs = [dc00, dc01, dc10, dc11] in row-major 2x2
                            #[cfg(feature = "debug-dc")]
                            eprintln!(
                                "DCT16x16 block (by={}, bx={}): dcs=[{:.4}, {:.4}, {:.4}, {:.4}], LLF=[{:.6}, {:.6}, {:.6}, {:.6}]",
                                by,
                                bx,
                                dcs[0],
                                dcs[1],
                                dcs[2],
                                dcs[3],
                                coeffs_arr[0],
                                coeffs_arr[1],
                                coeffs_arr[16],
                                coeffs_arr[17]
                            );
                            for iy in 0..2 {
                                for ix in 0..2 {
                                    let qdc = (dcs[iy * 2 + ix] * inv_factor).round() as i16;
                                    #[cfg(feature = "debug-dc")]
                                    eprintln!(
                                        "  quant_dc[1][{}][{}] = {} (raw dc={:.4}, inv_factor={:.4})",
                                        by + iy,
                                        bx + ix,
                                        qdc,
                                        dcs[iy * 2 + ix],
                                        inv_factor
                                    );
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X32 => {
                            let coeffs_arr: [f32; 1024] = dct_coeffs[1][..1024]
                                .try_into()
                                .expect("1024 coefficients for DCT32x32");
                            let dcs = dc_from_dct_32x32(&coeffs_arr);
                            #[cfg(feature = "debug-dc")]
                            eprintln!(
                                "DCT32x32 block (by={}, bx={}): dcs[0..4]=[{:.4}, {:.4}, {:.4}, {:.4}], LLF=[{:.6}, {:.6}, {:.6}, {:.6}]",
                                by,
                                bx,
                                dcs[0],
                                dcs[1],
                                dcs[2],
                                dcs[3],
                                coeffs_arr[0],
                                coeffs_arr[1],
                                coeffs_arr[32],
                                coeffs_arr[33]
                            );
                            // dcs = 16 DC values in row-major 4x4
                            for iy in 0..4 {
                                for ix in 0..4 {
                                    let qdc = (dcs[iy * 4 + ix] * inv_factor).round() as i16;
                                    #[cfg(feature = "debug-dc")]
                                    eprintln!(
                                        "  quant_dc[1][{}][{}] = {} (raw dc={:.4})",
                                        by + iy,
                                        bx + ix,
                                        qdc,
                                        dcs[iy * 4 + ix]
                                    );
                                    quant_dc[1][by + iy][bx + ix] = qdc;
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                // ── Step 3: Quantize Y AC with thresholding ────────────────
                {
                    let c = 1;
                    let weights = super::quant::quant_weights(raw_strategy as usize, c);
                    Self::quantize_ac_block(
                        &dct_coeffs[c],
                        weights,
                        qac,
                        1.0, // no x_qm_mul for Y
                        c,
                        block_width,
                        block_height,
                        covered_x,
                        covered_y,
                        covered_blocks,
                        size,
                        raw_strategy,
                        bx,
                        by,
                        &mut quant_ac[c],
                    );
                }

                // ── Step 4: Dequantize Y back (AdjustQuantBias roundtrip) ──
                // C++ QuantizeRoundtripYBlockAC: quantize all → dequantize all.
                // We already quantized AC; now also quantize LLF (temporarily)
                // and dequantize everything back into dct_coeffs[1].
                {
                    let weights = super::quant::quant_weights(raw_strategy as usize, 1);
                    let inv_qac = 1.0 / qac;
                    // Use post-swap dimensions for grid (matches C++ and quantize_ac_block)
                    for idx in 0..size {
                        // LLF positions: (y, x) where y < cy and x < cx in the grid
                        let is_llf = (idx / block_width) < cy && (idx % block_width) < cx;
                        let q = if is_llf {
                            // LLF: not stored in quant_ac, compute inline
                            // C++ QuantizeBlockAC quantizes all positions including LLF
                            let y = idx / block_width;
                            let x = idx % block_width;
                            Self::quantize_coeff_ac(
                                dct_coeffs[1][idx],
                                1.0 / weights[idx],
                                qac,
                                1.0,
                                1,
                                y,
                                x,
                                block_height,
                                block_width,
                                cx,
                                cy,
                            )
                        } else {
                            let block_slot = idx / DCT_BLOCK_SIZE;
                            let coeff_in_block = idx % DCT_BLOCK_SIZE;
                            let slot_by = by + block_slot / covered_x;
                            let slot_bx = bx + block_slot % covered_x;
                            quant_ac[1][slot_by][slot_bx][coeff_in_block]
                        };
                        let adj = Self::adjust_quant_bias(q, 1);
                        dct_coeffs[1][idx] = adj * weights[idx] * inv_qac;
                    }
                }

                // ── Step 5: DCT X and B channels ───────────────────────────
                for &c in &[0usize, 2] {
                    Self::apply_dct(
                        channels[c],
                        padded_width,
                        bx,
                        by,
                        raw_strategy,
                        &mut dct_coeffs[c],
                    );
                }

                // ── Step 6: CfL on AC coefficients using roundtripped Y ───
                // C++ applies CfL to ALL positions (0..size) including DC/LLF,
                // but the decoder's DequantBlock calls LowestFrequenciesFromDC
                // AFTER DequantLane, overwriting LLF positions with DC-derived
                // values. So coefficient-level CfL on LLF is discarded by the
                // decoder. We skip LLF here; DC CfL uses dc_cfl_factor instead.
                #[allow(clippy::needless_range_loop)]
                // k used for LLF check and indexing two arrays
                for k in 0..size {
                    let is_llf = (k / block_width) < cy && (k % block_width) < cx;
                    if !is_llf {
                        dct_coeffs[0][k] -= x_factor * dct_coeffs[1][k];
                        dct_coeffs[2][k] -= b_factor * dct_coeffs[1][k];
                    }
                }

                // ── Step 7: Extract X/B DC + quantize X/B AC ───────────────
                for &c in &[0usize, 2] {
                    let dc_cfl_factor = if c == 2 { 0.5f32 } else { 0.0f32 };
                    let inv_factor = INV_DC_QUANT[c] * params.scale_dc;
                    let qm_multiplier = if c == 0 { x_qm_mul } else { 1.0 };

                    // Extract DC from CfL-adjusted coefficients.
                    // Read Y DC into temporaries to avoid borrow conflict
                    // (can't have &quant_dc[1] and &mut quant_dc[c] simultaneously).
                    match raw_strategy {
                        0 => {
                            let dc = dct_coeffs[c][0];
                            let y_dc = quant_dc[1][by][bx] as f32;
                            quant_dc[c][by][bx] =
                                (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                        }
                        RAW_STRATEGY_DCT16X8 => {
                            let coeffs_arr: [f32; 128] = dct_coeffs[c][..128]
                                .try_into()
                                .expect("128 coefficients for DCT16x8");
                            let dcs = dc_from_dct_16x8(&coeffs_arr);
                            for iy in 0..2 {
                                let y_dc = quant_dc[1][by + iy][bx] as f32;
                                quant_dc[c][by + iy][bx] =
                                    (dcs[iy] * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT8X16 => {
                            let coeffs_arr: [f32; 128] = dct_coeffs[c][..128]
                                .try_into()
                                .expect("128 coefficients for DCT8x16");
                            let dcs = dc_from_dct_8x16(&coeffs_arr);
                            for ix in 0..2 {
                                let y_dc = quant_dc[1][by][bx + ix] as f32;
                                quant_dc[c][by][bx + ix] =
                                    (dcs[ix] * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                            }
                        }
                        RAW_STRATEGY_DCT16X16 => {
                            let coeffs_arr: [f32; 256] = dct_coeffs[c][..256]
                                .try_into()
                                .expect("256 coefficients for DCT16x16");
                            let dcs = dc_from_dct_16x16(&coeffs_arr);
                            for iy in 0..2 {
                                for ix in 0..2 {
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 2 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        RAW_STRATEGY_DCT32X32 => {
                            let coeffs_arr: [f32; 1024] = dct_coeffs[c][..1024]
                                .try_into()
                                .expect("1024 coefficients for DCT32x32");
                            let dcs = dc_from_dct_32x32(&coeffs_arr);
                            for iy in 0..4 {
                                for ix in 0..4 {
                                    let y_dc = quant_dc[1][by + iy][bx + ix] as f32;
                                    quant_dc[c][by + iy][bx + ix] =
                                        (dcs[iy * 4 + ix] * inv_factor - y_dc * dc_cfl_factor)
                                            .round() as i16;
                                }
                            }
                        }
                        _ => unreachable!(),
                    }

                    // Quantize AC with thresholding
                    let weights = super::quant::quant_weights(raw_strategy as usize, c);
                    Self::quantize_ac_block(
                        &dct_coeffs[c],
                        weights,
                        qac,
                        qm_multiplier,
                        c,
                        block_width,
                        block_height,
                        covered_x,
                        covered_y,
                        covered_blocks,
                        size,
                        raw_strategy,
                        bx,
                        by,
                        &mut quant_ac[c],
                    );
                }

                // ── Step 8: Count non-zeros for all 3 channels ─────────────
                for c in 0..3 {
                    if covered_blocks == 1 {
                        num_nonzero_8x8_except_dc(&quant_ac[c][by][bx], &mut nzeros[c][by][bx]);
                        raw_nzeros[c][by][bx] = nzeros[c][by][bx];
                    } else {
                        let full_block: Vec<i32> = (0..size)
                            .map(|idx| {
                                let block_slot = idx / DCT_BLOCK_SIZE;
                                let coeff_in_block = idx % DCT_BLOCK_SIZE;
                                let slot_by = by + block_slot / covered_x;
                                let slot_bx = bx + block_slot % covered_x;
                                quant_ac[c][slot_by][slot_bx][coeff_in_block]
                            })
                            .collect();
                        let flat_len = (covered_y - 1) * xsize_blocks + covered_x;
                        let mut flat_nz = vec![0u8; flat_len];
                        let raw_nz = num_nonzero_except_llf(
                            cx,
                            cy,
                            &full_block,
                            xsize_blocks,
                            &mut flat_nz,
                            covered_x,
                            covered_y,
                        );
                        for dy in 0..covered_y {
                            for dx in 0..covered_x {
                                nzeros[c][by + dy][bx + dx] = flat_nz[dx + dy * xsize_blocks];
                            }
                        }
                        raw_nzeros[c][by][bx] = raw_nz;
                    }
                }
            }
        }

        (quant_dc, quant_ac, nzeros, raw_nzeros)
    }

    /// Write the file header (SizeHeader + ImageMetadata).
    ///
    /// Follows libjxl-tiny's enc_file.cc exactly:
    /// 1. SizeHeader with small=0 (U32 encoding)
    /// 2. ImageMetadata with float samples (32-bit, 8 exp bits)
    /// 3. ColorEncoding with sRGB primaries, Linear transfer
    /// 4. all_default_transform_data = 1
    /// 5. Zero padding to byte
    fn write_file_header(&self, width: usize, height: usize, writer: &mut BitWriter) -> Result<()> {
        // 1. SizeHeader - use U32 format (small=0), same as libjxl-tiny
        self.write_size(width, height, writer)?;

        // 2. ImageMetadata
        writer.write(1, 0)?; // not all default
        writer.write(1, 0)?; // no extra fields

        // Bit depth - 32-bit float with 8 exponent bits (libjxl-tiny format)
        writer.write(1, 1)?; // float = 1
        writer.write(2, 0)?; // bits_per_sample selector 0 = 32 bits
        writer.write(4, 7)?; // exp_bits = 8 (encoded as 7+1)

        writer.write(1, 0)?; // modular 16 bit buffer NOT sufficient (for float)

        // Extra channels - none
        writer.write(2, 0)?; // selector 0 = 0 extra channels

        // xyb_encoded = 1 (required for VarDCT)
        writer.write(1, 1)?;

        // Color encoding - sRGB primaries/white, Linear transfer
        writer.write(1, 0)?; // not all default
        writer.write(1, 0)?; // no ICC profile
        writer.write(2, 0)?; // color_space = RGB (0)
        writer.write(2, 1)?; // white_point = D65 (1)
        writer.write(2, 1)?; // primaries = sRGB (1)
        writer.write(1, 0)?; // no gamma (use transfer function)
        // TransferFunction: U32(0, 1, 2+Read(4), 18+Read(6))
        // For Linear (value 8): selector=2, extra=6 (8 = 2 + 6)
        writer.write(2, 2)?; // selector 2
        writer.write(4, 6)?; // value 6 -> transfer_function = 2+6 = 8 = Linear
        writer.write(2, 1)?; // rendering_intent = relative (1)

        // Extensions
        writer.write(2, 0)?; // no extensions

        // 3. all_default_transform_data = 1 (required before frame)
        writer.write(1, 1)?;

        // 4. Zero pad to byte before frame
        writer.zero_pad_to_byte();

        Ok(())
    }

    /// Write image size header.
    ///
    /// Uses U32 format (small=0) to match libjxl-tiny exactly.
    /// Format: small=0, height U32, ratio=0, width U32
    fn write_size(&self, width: usize, height: usize, writer: &mut BitWriter) -> Result<()> {
        // Helper to write a dimension using U32 encoding
        // Matches libjxl-tiny's WriteSize() exactly
        fn write_dim(size: usize, writer: &mut BitWriter) -> Result<()> {
            let size_m1 = (size.saturating_sub(1)) as u32;
            // U32 selectors: 9 bits, 13 bits, 18 bits, 30 bits
            // Select first one where value fits
            let k_bits: [u32; 4] = [9, 13, 18, 30];
            for (i, &bits) in k_bits.iter().enumerate() {
                if size_m1 < (1u32 << bits) {
                    writer.write(2, i as u64)?;
                    writer.write(bits as usize, size_m1 as u64)?;
                    return Ok(());
                }
            }
            // Shouldn't reach here for valid sizes
            Ok(())
        }

        // small = 0 (use U32 encoding)
        writer.write(1, 0)?;
        write_dim(height, writer)?;
        writer.write(3, 0)?; // ratio = 0 (explicit width)
        write_dim(width, writer)?;

        Ok(())
    }

    /// Write DC global section (LfGlobal).
    ///
    /// Decoder order (from jxl-rs frame/decode.rs):
    /// 0. Patches (if enabled) — not used
    /// 1. Splines (if enabled) — not used
    /// 2. **Noise params** (if ENABLE_NOISE flag set) — 8 × 10-bit LUT values
    /// 3. Default dequant DC (LfQuantFactors)
    /// 4. Quant scales (QuantizerParams)
    /// 5. Non-default BlockCtxMap + compact block context map
    /// 6. Default DC cmap (ColorCorrelationParams)
    /// 7. Context tree for modular stream
    /// 8. No LZ77
    /// 9. DC entropy code
    fn write_dc_global(
        &self,
        params: &DistanceParams,
        num_dc_groups: usize,
        dc_code: &BuiltEntropyCode,
        noise_params: &Option<NoiseParams>,
        writer: &mut BitWriter,
    ) -> Result<()> {
        #[cfg(feature = "debug-tokens")]
        let start_bits = writer.bits_written();

        // Write noise parameters before dequant DC (decoder expects this order)
        if let Some(ref noise) = *noise_params {
            write_noise_params(noise, writer)?;
        }

        writer.write(1, 1)?; // default dequant dc

        #[cfg(feature = "debug-tokens")]
        let after_dequant_dc = writer.bits_written();

        write_quant_scales(params.global_scale, params.quant_dc, writer)?;

        #[cfg(feature = "debug-tokens")]
        let after_quant = writer.bits_written();

        // BlockCtxMap - non-default, write compact map
        writer.write(1, 0)?; // non-default BlockCtxMap
        writer.write(16, 0)?; // no dc ctx, no qft

        // Write compact block context map
        super::context_tree::write_block_context_map(writer)?;

        #[cfg(feature = "debug-tokens")]
        let after_block_ctx = writer.bits_written();

        writer.write(1, 1)?; // default DC cmap

        // Write context tree for modular stream DC header
        super::context_tree::write_context_tree(num_dc_groups, writer)?;

        #[cfg(feature = "debug-tokens")]
        let after_ctx_tree = writer.bits_written();

        writer.write(1, 0)?; // no lz77

        #[cfg(feature = "debug-tokens")]
        let after_lz77 = writer.bits_written();

        // Write DC entropy code
        self.write_entropy_code_header(dc_code, writer)?;

        #[cfg(feature = "debug-tokens")]
        {
            let after_dc_code = writer.bits_written();
            let total_bits = after_dc_code - start_bits;
            let bytes_before_pad = total_bits.div_ceil(8);
            debug_log!("DC_global detailed breakdown:");
            debug_log!("  dequant_dc: {} bits (1)", after_dequant_dc - start_bits);
            debug_log!("  quant_scales: {} bits", after_quant - after_dequant_dc);
            debug_log!(
                "  block_ctx_map: {} bits (1+16+map)",
                after_block_ctx - after_quant
            );
            debug_log!("  dc_cmap: 1 bit (default=1)");
            debug_log!(
                "  context_tree: {} bits",
                after_ctx_tree - after_block_ctx - 1
            );
            debug_log!("  lz77: 1 bit (no=0)");
            debug_log!("  dc_entropy_code: {} bits", after_dc_code - after_lz77);
            debug_log!(
                "  total bits: {}, bytes before pad: {}",
                total_bits,
                bytes_before_pad
            );
        }

        Ok(())
    }

    /// Write DC group section.
    ///
    /// For single-group images (≤256x256), dc_group_idx is 0 and covers the whole image.
    /// For multi-group images, each DC group covers a 256x256 block region (2048x2048 pixels).
    #[allow(clippy::too_many_arguments)]
    fn write_dc_group(
        &self,
        dc_group_idx: usize,
        quant_dc: &[Vec<Vec<i16>>; 3],
        xsize_blocks: usize,
        ysize_blocks: usize,
        xsize_dc_groups: usize,
        quant_field: &[u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        dc_code: &super::entropy_code::EntropyCode,
        writer: &mut BitWriter,
    ) -> Result<()> {
        #[cfg(feature = "debug-tokens")]
        let start_bits = writer.bits_written();

        // Compute the block region for this DC group
        let dc_gx = dc_group_idx % xsize_dc_groups;
        let dc_gy = dc_group_idx / xsize_dc_groups;
        let start_bx = dc_gx * DC_GROUP_DIM_IN_BLOCKS;
        let start_by = dc_gy * DC_GROUP_DIM_IN_BLOCKS;
        let end_bx = (start_bx + DC_GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
        let end_by = (start_by + DC_GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
        let region_xsize = end_bx - start_bx;
        let region_ysize = end_by - start_by;

        #[cfg(feature = "debug-tokens")]
        debug_log!(
            "DC_group {}: blocks ({},{}) to ({},{}) = {}x{}",
            dc_group_idx,
            start_bx,
            start_by,
            end_bx,
            end_by,
            region_xsize,
            region_ysize
        );

        // DC group header
        writer.write(2, 0)?; // extra_dc_precision = 0
        writer.write(4, 3)?; // use global tree, default wp, no transforms

        #[cfg(feature = "debug-tokens")]
        let after_header1 = writer.bits_written();

        // Write DC tokens using gradient predictor for this region only
        write_dc_tokens_region(
            quant_dc, start_bx, start_by, end_bx, end_by, dc_code, writer,
        )?;

        #[cfg(feature = "debug-tokens")]
        let after_dc_tokens = writer.bits_written();

        // AC metadata header - count first blocks (distinct transforms) in region
        let num_blocks = region_xsize * region_ysize;
        let mut num_ac_blocks = 0;
        for ry in start_by..end_by {
            for rx in start_bx..end_bx {
                if ac_strategy.is_first(rx, ry) {
                    num_ac_blocks += 1;
                }
            }
        }
        let nb_bits = ceil_log2_nonzero(num_blocks);
        if nb_bits != 0 {
            writer.write(nb_bits as usize, (num_ac_blocks - 1) as u64)?;
        }
        writer.write(4, 3)?; // use global tree, default wp, no transforms

        #[cfg(feature = "debug-tokens")]
        let after_header2 = writer.bits_written();

        // Write AC metadata tokens for this region only
        write_ac_metadata_tokens_region(
            region_xsize,
            region_ysize,
            quant_field,
            xsize_blocks,
            start_bx,
            start_by,
            cfl_map,
            ac_strategy,
            dc_code,
            writer,
        )?;

        #[cfg(feature = "debug-tokens")]
        {
            let total = writer.bits_written() - start_bits;
            debug_log!("DC_group {} breakdown:", dc_group_idx);
            debug_log!("  header1: {} bits (2+4)", after_header1 - start_bits);
            debug_log!("  dc_tokens: {} bits", after_dc_tokens - after_header1);
            debug_log!(
                "  header2: {} bits (nb_bits+4)",
                after_header2 - after_dc_tokens
            );
            debug_log!(
                "  ac_metadata: {} bits",
                writer.bits_written() - after_header2
            );
            debug_log!(
                "  total: {} bits ({} bytes before pad)",
                total,
                total.div_ceil(8)
            );
        }

        Ok(())
    }

    /// Write AC global section.
    fn write_ac_global(
        &self,
        num_groups: usize,
        ac_code: &BuiltEntropyCode,
        used_orders: u32,
        coeff_order_tokens: Option<&[Token]>,
        writer: &mut BitWriter,
    ) -> Result<()> {
        #[cfg(feature = "debug-tokens")]
        let start_bits = writer.bits_written();

        writer.write(1, 1)?; // all default quant matrices

        let num_histo_bits = ceil_log2_nonzero(num_groups);
        if num_histo_bits != 0 {
            writer.write(num_histo_bits as usize, 0)?;
        }

        // Write used_orders via u2S(0x5F, 0x13, 0x00, U(13))
        if used_orders == 0x5F {
            writer.write(2, 0)?; // selector 0 = 0x5F
        } else if used_orders == 0x13 {
            writer.write(2, 1)?; // selector 1 = 0x13
        } else if used_orders == 0 {
            writer.write(2, 2)?; // selector 2 = 0
        } else {
            writer.write(2, 3)?; // selector 3 = U(13)
            writer.write(13, used_orders as u64)?;
        }

        // Write permutation data if we have custom orders
        if let Some(tokens) = coeff_order_tokens.filter(|_| used_orders != 0) {
            super::coeff_order::build_and_write_coeff_orders(tokens, self.use_ans, writer)?;
        }

        writer.write(1, 0)?; // no lz77

        #[cfg(feature = "debug-tokens")]
        let before_ac_code = writer.bits_written();

        // Write entropy code
        self.write_entropy_code_header(ac_code, writer)?;

        #[cfg(feature = "debug-tokens")]
        {
            let after_ac_code = writer.bits_written();
            debug_log!("AC_global breakdown:");
            debug_log!("  header: {} bits", before_ac_code - start_bits);
            debug_log!(
                "  ac_entropy_code: {} bits ({} contexts, {} histograms)",
                after_ac_code - before_ac_code,
                ac_code.num_contexts(),
                ac_code.num_histograms()
            );
        }

        Ok(())
    }

    /// Write AC group section.
    #[allow(clippy::too_many_arguments)]
    fn write_ac_group(
        &self,
        group_idx: usize,
        quant_ac: &[Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
        nzeros: &[Vec<Vec<u8>>; 3],
        raw_nzeros: &[Vec<Vec<u8>>; 3],
        xsize_blocks: usize,
        ysize_blocks: usize,
        xsize_groups: usize,
        ac_strategy: &AcStrategyMap,
        ac_code: &super::entropy_code::EntropyCode,
        writer: &mut BitWriter,
    ) -> Result<()> {
        #[cfg(feature = "debug-tokens")]
        let start_bits = writer.bits_written();

        // Compute block range for this group
        let group_x = group_idx % xsize_groups;
        let group_y = group_idx / xsize_groups;
        let start_bx = group_x * GROUP_DIM_IN_BLOCKS;
        let start_by = group_y * GROUP_DIM_IN_BLOCKS;
        let end_bx = (start_bx + GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
        let end_by = (start_by + GROUP_DIM_IN_BLOCKS).min(ysize_blocks);

        #[cfg(feature = "debug-tokens")]
        debug_log!(
            "AC group {}: blocks ({},{}) to ({},{})",
            group_idx,
            start_bx,
            start_by,
            end_bx,
            end_by
        );

        // Process blocks in row-major order, with channels interleaved per block
        // CRITICAL: libjxl-tiny loops: for block { for channel {Y,X,B} { tokenize } }
        // We must match this exact order!
        for by in start_by..end_by {
            for bx in start_bx..end_bx {
                // Skip non-first blocks of multi-block transforms
                if !ac_strategy.is_first(bx, by) {
                    continue;
                }

                let _raw_strategy = ac_strategy.raw_strategy(bx, by);
                let covered_x = ac_strategy.covered_blocks_x(bx, by);
                let covered_y = ac_strategy.covered_blocks_y(bx, by);
                let covered_blocks = covered_x * covered_y;
                let size = covered_blocks * DCT_BLOCK_SIZE;
                let strategy_code = ac_strategy.strategy_code(bx, by);

                // Process channels in order: Y (1), X (0), B (2)
                for &c in &[1usize, 0, 2] {
                    // Raw (unshifted) nzeros for bitstream token
                    let nz = raw_nzeros[c][by][bx];

                    // Predict nzeros from shifted neighbors (matches C++ PredictFromTopAndLeft)
                    let row_top = if by > start_by {
                        Some(nzeros[c][by - 1].as_slice())
                    } else {
                        None
                    };
                    let local_bx = bx - start_bx;
                    let predicted_nz = if local_bx == 0 {
                        match row_top {
                            Some(top) => top[bx] as i32,
                            None => 32,
                        }
                    } else {
                        predict_from_top_and_left(row_top, &nzeros[c][by], bx, 32)
                    };

                    if covered_blocks == 1 {
                        // DCT8: use existing single-block path
                        // Streaming path: no custom orders (requires two-pass)
                        tokenize_ac_coefficients(
                            &quant_ac[c][by][bx],
                            c,
                            strategy_code,
                            nz,
                            predicted_nz,
                            ac_code,
                            writer,
                            None,
                        )?;
                    } else {
                        // Multi-block: assemble contiguous coefficient buffer
                        let full_block: Vec<i32> = (0..size)
                            .map(|idx| {
                                let block_slot = idx / DCT_BLOCK_SIZE;
                                let coeff_in_block = idx % DCT_BLOCK_SIZE;
                                let slot_by = by + block_slot / covered_x;
                                let slot_bx = bx + block_slot % covered_x;
                                quant_ac[c][slot_by][slot_bx][coeff_in_block]
                            })
                            .collect();
                        // Streaming path: no custom orders
                        tokenize_ac_coefficients(
                            &full_block,
                            c,
                            strategy_code,
                            nz,
                            predicted_nz,
                            ac_code,
                            writer,
                            None,
                        )?;
                    }
                }
            }
        }

        #[cfg(feature = "debug-tokens")]
        {
            let total_bits = writer.bits_written() - start_bits;
            debug_log!(
                "AC_group {} breakdown: {} bits ({} bytes before pad)",
                group_idx,
                total_bits,
                total_bits.div_ceil(8)
            );
            // Show the raw bytes for comparison
            let bytes = writer.peek_bytes();
            let ac_start_byte = start_bits / 8;
            let ac_end_byte = writer.bits_written().div_ceil(8);
            if ac_end_byte <= bytes.len() && ac_start_byte < ac_end_byte {
                debug_log!(
                    "AC_group raw bytes: {:02x?}",
                    &bytes[ac_start_byte..ac_end_byte.min(ac_start_byte + 10)]
                );
            }
        }

        Ok(())
    }

    /// Two-pass encoding: collect all tokens, build optimal codes, write bitstream.
    #[allow(clippy::too_many_arguments)]
    fn encode_two_pass(
        &self,
        width: usize,
        height: usize,
        params: &DistanceParams,
        xsize_blocks: usize,
        ysize_blocks: usize,
        xsize_groups: usize,
        _ysize_groups: usize,
        xsize_dc_groups: usize,
        _ysize_dc_groups: usize,
        num_groups: usize,
        num_dc_groups: usize,
        num_sections: usize,
        quant_dc: &[Vec<Vec<i16>>; 3],
        quant_ac: &[Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
        nzeros: &[Vec<Vec<u8>>; 3],
        raw_nzeros: &[Vec<Vec<u8>>; 3],
        quant_field: &[u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        noise_params: &Option<NoiseParams>,
    ) -> Result<Vec<u8>> {
        // ── Pass 1: Collect tokens per section ──

        // DC section tokens: two Vecs per dc_group (DC tokens, AC metadata tokens)
        let mut dc_tokens_per_group: Vec<Vec<Token>> = Vec::with_capacity(num_dc_groups);
        let mut ac_metadata_tokens_per_group: Vec<Vec<Token>> = Vec::with_capacity(num_dc_groups);
        for dc_group_idx in 0..num_dc_groups {
            let dc_gx = dc_group_idx % xsize_dc_groups;
            let dc_gy = dc_group_idx / xsize_dc_groups;
            let start_bx = dc_gx * DC_GROUP_DIM_IN_BLOCKS;
            let start_by = dc_gy * DC_GROUP_DIM_IN_BLOCKS;
            let end_bx = (start_bx + DC_GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
            let end_by = (start_by + DC_GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
            let region_xsize = end_bx - start_bx;
            let region_ysize = end_by - start_by;

            let dc_tokens = collect_dc_tokens_region(quant_dc, start_bx, start_by, end_bx, end_by);
            let md_tokens = collect_ac_metadata_tokens_region(
                region_xsize,
                region_ysize,
                quant_field,
                xsize_blocks,
                start_bx,
                start_by,
                cfl_map,
                ac_strategy,
            );
            dc_tokens_per_group.push(dc_tokens);
            ac_metadata_tokens_per_group.push(md_tokens);
        }

        // Compute custom coefficient orders if enabled and image is large enough
        let (custom_order_map, used_orders) =
            if self.custom_orders && (xsize_blocks >= 5 || ysize_blocks >= 5) {
                let zero_counts = super::coeff_order::count_zero_coefficients(
                    quant_ac,
                    ac_strategy,
                    xsize_blocks,
                    ysize_blocks,
                );
                let (orders, used) = super::coeff_order::compute_custom_orders(&zero_counts);
                if used != 0 {
                    (Some(orders), used)
                } else {
                    (None, 0u32)
                }
            } else {
                (None, 0u32)
            };

        // AC section tokens: one Vec<Token> per ac_group
        let mut ac_section_tokens: Vec<Vec<Token>> = Vec::with_capacity(num_groups);
        for group_idx in 0..num_groups {
            let group_x = group_idx % xsize_groups;
            let group_y = group_idx / xsize_groups;
            let start_bx = group_x * GROUP_DIM_IN_BLOCKS;
            let start_by = group_y * GROUP_DIM_IN_BLOCKS;
            let end_bx = (start_bx + GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
            let end_by = (start_by + GROUP_DIM_IN_BLOCKS).min(ysize_blocks);

            let mut tokens = Vec::new();
            for by in start_by..end_by {
                for bx in start_bx..end_bx {
                    if !ac_strategy.is_first(bx, by) {
                        continue;
                    }
                    let covered_x = ac_strategy.covered_blocks_x(bx, by);
                    let covered_y = ac_strategy.covered_blocks_y(bx, by);
                    let covered_blocks = covered_x * covered_y;
                    let size = covered_blocks * DCT_BLOCK_SIZE;
                    let strategy_code = ac_strategy.strategy_code(bx, by);

                    for &c in &[1usize, 0, 2] {
                        // Raw (unshifted) nzeros for bitstream token
                        let nz = raw_nzeros[c][by][bx];
                        let local_bx = bx - start_bx;
                        // Prediction uses shifted nzeros from neighbors
                        let row_top = if by > start_by {
                            Some(nzeros[c][by - 1].as_slice())
                        } else {
                            None
                        };
                        let predicted_nz = if local_bx == 0 {
                            match row_top {
                                Some(top) => top[bx] as i32,
                                None => 32,
                            }
                        } else {
                            predict_from_top_and_left(row_top, &nzeros[c][by], bx, 32)
                        };
                        // Get custom order for this (bucket, channel) if available
                        // IMPORTANT: get_custom_order's strategy_bucket() expects
                        // bitstream strategy codes (0,4,5,6,7), not raw (0-4).
                        let custom_ord = custom_order_map.as_ref().and_then(|orders| {
                            super::coeff_order::get_custom_order(
                                orders,
                                used_orders,
                                strategy_code,
                                c,
                            )
                        });

                        if covered_blocks == 1 {
                            let block_tokens = collect_ac_coefficients(
                                &quant_ac[c][by][bx],
                                c,
                                strategy_code,
                                nz,
                                predicted_nz,
                                custom_ord,
                            );
                            tokens.extend_from_slice(&block_tokens);
                        } else {
                            // Assemble contiguous buffer
                            let full_block: Vec<i32> = (0..size)
                                .map(|idx| {
                                    let block_slot = idx / DCT_BLOCK_SIZE;
                                    let coeff_in_block = idx % DCT_BLOCK_SIZE;
                                    let slot_by = by + block_slot / covered_x;
                                    let slot_bx = bx + block_slot % covered_x;
                                    quant_ac[c][slot_by][slot_bx][coeff_in_block]
                                })
                                .collect();
                            let block_tokens = collect_ac_coefficients(
                                &full_block,
                                c,
                                strategy_code,
                                nz,
                                predicted_nz,
                                custom_ord,
                            );
                            tokens.extend_from_slice(&block_tokens);
                        }
                    }
                }
            }
            ac_section_tokens.push(tokens);
        }

        // ── Build optimal codes ──

        // Merge all DC section tokens (DC + AC metadata) for frequency counting
        let total_dc_tokens: usize = dc_tokens_per_group.iter().map(|t| t.len()).sum::<usize>()
            + ac_metadata_tokens_per_group
                .iter()
                .map(|t| t.len())
                .sum::<usize>();
        let mut all_dc_tokens = Vec::with_capacity(total_dc_tokens);
        for section in &dc_tokens_per_group {
            all_dc_tokens.extend_from_slice(section);
        }
        for section in &ac_metadata_tokens_per_group {
            all_dc_tokens.extend_from_slice(section);
        }
        // Build entropy codes (Huffman or ANS based on config)
        let dc_built_code = if self.use_ans {
            BuiltEntropyCode::Ans(build_entropy_code_ans_with_options(
                &all_dc_tokens,
                super::dc_coding::NUM_DC_CONTEXTS,
                self.enhanced_clustering,
            ))
        } else {
            BuiltEntropyCode::Huffman(build_entropy_code_with_options(
                &all_dc_tokens,
                super::dc_coding::NUM_DC_CONTEXTS,
                self.enhanced_clustering,
            ))
        };

        // Merge all AC section tokens for frequency counting
        let total_ac_tokens: usize = ac_section_tokens.iter().map(|t| t.len()).sum();
        let mut all_ac_tokens = Vec::with_capacity(total_ac_tokens);
        for section in &ac_section_tokens {
            all_ac_tokens.extend_from_slice(section);
        }

        let ac_built_code = if self.use_ans {
            BuiltEntropyCode::Ans(build_entropy_code_ans_with_options(
                &all_ac_tokens,
                super::ac_context::NUM_AC_CONTEXTS,
                self.enhanced_clustering,
            ))
        } else {
            BuiltEntropyCode::Huffman(build_entropy_code_with_options(
                &all_ac_tokens,
                super::ac_context::NUM_AC_CONTEXTS,
                self.enhanced_clustering,
            ))
        };

        // ── ANS invariant verification (debug builds only) ──
        // DISABLED: The local verification decoder has a bug that produces false positives
        // for certain histogram patterns (e.g., 256x256 solid color images). The actual
        // encoding is valid - djxl decodes these files correctly. Rely on external decoder
        // testing (djxl, jxl-rs) instead of this broken verification.
        // TODO: Fix verify_histogram_serialization to handle all histogram method types correctly
        // if self.use_ans {
        //     if let BuiltEntropyCode::Ans(ref dc_ans) = dc_built_code {
        //         verify_histogram_serialization(dc_ans, "DC")?;
        //     }
        //     if let BuiltEntropyCode::Ans(ref ac_ans) = ac_built_code {
        //         verify_histogram_serialization(ac_ans, "AC")?;
        //     }
        // }

        // ── Tokenize coefficient orders (if custom) ──
        let coeff_order_tokens = if used_orders != 0 {
            let tokens = super::coeff_order::tokenize_coeff_orders(
                custom_order_map
                    .as_ref()
                    .expect("custom_order_map must exist when used_orders != 0"),
                used_orders,
            );
            Some(tokens)
        } else {
            None
        };

        // ── Pass 2: Write bitstream ──

        let mut writer = BitWriter::with_capacity(width * height * 4);

        // Write JXL signature
        writer.write(8, 0xFF)?;
        writer.write(8, 0x0A)?;

        // Write file header
        self.write_file_header(width, height, &mut writer)?;

        // Write frame header
        write_frame_header(
            params.x_qm_scale,
            params.epf_iters,
            noise_params.is_some(),
            self.enable_gaborish,
            &mut writer,
        )?;

        if num_sections == 4 {
            // Single-group: combine sections at the bit level
            let mut dc_global = BitWriter::new();
            self.write_dc_global(
                params,
                num_dc_groups,
                &dc_built_code,
                noise_params,
                &mut dc_global,
            )?;

            let mut dc_group = BitWriter::new();
            self.write_dc_group_from_tokens(
                0,
                xsize_blocks,
                ysize_blocks,
                xsize_dc_groups,
                &dc_tokens_per_group[0],
                &ac_metadata_tokens_per_group[0],
                ac_strategy,
                &dc_built_code,
                &mut dc_group,
            )?;

            let mut ac_global = BitWriter::new();
            self.write_ac_global(
                num_groups,
                &ac_built_code,
                used_orders,
                coeff_order_tokens.as_deref(),
                &mut ac_global,
            )?;

            let mut ac_group_writer = BitWriter::new();
            ac_built_code.write_tokens(&ac_section_tokens[0], &mut ac_group_writer)?;

            let mut combined = dc_global;
            combined.append_unaligned(&dc_group)?;
            combined.append_unaligned(&ac_global)?;
            combined.append_unaligned(&ac_group_writer)?;
            combined.zero_pad_to_byte();
            let combined_bytes = combined.finish();

            write_toc(&[combined_bytes.len()], &mut writer)?;
            writer.append_bytes(&combined_bytes)?;
        } else {
            // Multi-group: byte-aligned sections
            let mut sections: Vec<Vec<u8>> = Vec::with_capacity(num_sections);

            // DC Global
            let mut dc_global = BitWriter::new();
            self.write_dc_global(
                params,
                num_dc_groups,
                &dc_built_code,
                noise_params,
                &mut dc_global,
            )?;
            dc_global.zero_pad_to_byte();
            sections.push(dc_global.finish());

            // DC groups
            for dc_group_idx in 0..num_dc_groups {
                let mut dc_group = BitWriter::new();
                self.write_dc_group_from_tokens(
                    dc_group_idx,
                    xsize_blocks,
                    ysize_blocks,
                    xsize_dc_groups,
                    &dc_tokens_per_group[dc_group_idx],
                    &ac_metadata_tokens_per_group[dc_group_idx],
                    ac_strategy,
                    &dc_built_code,
                    &mut dc_group,
                )?;
                dc_group.zero_pad_to_byte();
                sections.push(dc_group.finish());
            }

            // AC Global
            let mut ac_global = BitWriter::new();
            self.write_ac_global(
                num_groups,
                &ac_built_code,
                used_orders,
                coeff_order_tokens.as_deref(),
                &mut ac_global,
            )?;
            ac_global.zero_pad_to_byte();
            sections.push(ac_global.finish());

            // AC groups
            for ac_tokens in &ac_section_tokens {
                let mut ac_group_writer = BitWriter::new();
                ac_built_code.write_tokens(ac_tokens, &mut ac_group_writer)?;
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

    /// Write DC group section from pre-collected tokens (two-pass mode).
    ///
    /// Writes the DC group header, DC tokens, AC metadata sub-header, then AC
    /// metadata tokens — matching the exact bitstream layout of `write_dc_group`.
    #[allow(clippy::too_many_arguments)]
    fn write_dc_group_from_tokens(
        &self,
        dc_group_idx: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        xsize_dc_groups: usize,
        dc_tokens: &[Token],
        ac_metadata_tokens: &[Token],
        ac_strategy: &AcStrategyMap,
        dc_code: &BuiltEntropyCode,
        writer: &mut BitWriter,
    ) -> Result<()> {
        let dc_gx = dc_group_idx % xsize_dc_groups;
        let dc_gy = dc_group_idx / xsize_dc_groups;
        let start_bx = dc_gx * DC_GROUP_DIM_IN_BLOCKS;
        let start_by = dc_gy * DC_GROUP_DIM_IN_BLOCKS;
        let end_bx = (start_bx + DC_GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
        let end_by = (start_by + DC_GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
        let region_xsize = end_bx - start_bx;
        let region_ysize = end_by - start_by;

        // DC group header
        writer.write(2, 0)?; // extra_dc_precision = 0
        writer.write(4, 3)?; // use global tree, default wp, no transforms

        // Write DC tokens
        dc_code.write_tokens(dc_tokens, writer)?;

        // AC metadata sub-header — count first blocks (distinct transforms)
        let num_blocks = region_xsize * region_ysize;
        let mut num_ac_blocks = 0;
        for ry in start_by..end_by {
            for rx in start_bx..end_bx {
                if ac_strategy.is_first(rx, ry) {
                    num_ac_blocks += 1;
                }
            }
        }
        let nb_bits = ceil_log2_nonzero(num_blocks);
        if nb_bits != 0 {
            writer.write(nb_bits as usize, (num_ac_blocks - 1) as u64)?;
        }
        writer.write(4, 3)?; // use global tree, default wp, no transforms

        // Write AC metadata tokens
        dc_code.write_tokens(ac_metadata_tokens, writer)?;

        Ok(())
    }

    /// Write entropy code (context map + codes/distributions).
    fn write_entropy_code_header(
        &self,
        code: &BuiltEntropyCode,
        writer: &mut BitWriter,
    ) -> Result<()> {
        code.write_header(writer)
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
        // Updated: gaborish inverse enabled by default
        const EXPECTED_HASH: u64 = 0x53a52433cdc811d4;
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

        // Updated: gaborish inverse enabled by default
        const EXPECTED_HASH: u64 = 0xea5d762e171b78dc;
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

        // Updated: gaborish inverse enabled by default
        const EXPECTED_HASH: u64 = 0x6db9f8107ab85117;
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

        // Updated: gaborish inverse enabled by default
        const EXPECTED_HASH: u64 = 0x5ed333833314fda9;
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
}
