// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Main tiny encoder implementation.

use super::ac_group::{
    AC_STRATEGY_DCT8, collect_ac_coefficients, num_nonzero_8x8_except_dc,
    predict_from_top_and_left, tokenize_ac_coefficients,
};
use super::adaptive_quant::compute_adaptive_quant_field;
use super::chroma_from_luma::{CflMap, compute_cfl_map, ytob_ratio, ytox_ratio};
use super::common::*;
use super::dc_coding::{
    collect_ac_metadata_tokens_region, collect_dc_tokens_region, write_ac_metadata_tokens_region,
    write_dc_tokens_region,
};
use super::dct::dct_8x8;
use super::entropy_code::{build_entropy_code, write_tokens};
use super::frame::{DistanceParams, write_frame_header, write_quant_scales, write_toc};
use super::quant::INV_DC_QUANT;
use super::static_codes::{get_ac_entropy_code, get_dc_entropy_code};
use super::token::Token;
use crate::bit_writer::BitWriter;
use crate::color::xyb::linear_rgb_to_xyb;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;
use crate::error::Result;

/// Tiny JPEG XL encoder.
///
/// This is a simplified VarDCT encoder based on libjxl-tiny that uses:
/// - Only DCT8, DCT8x16, DCT16x8 transforms
/// - Only Huffman entropy coding
/// - Default zig-zag coefficient order
/// - Fixed context tree for DC
pub struct TinyEncoder {
    /// Target distance (quality). 1.0 = visually lossless.
    pub distance: f32,
    /// Use dynamic Huffman codes built from actual token frequencies.
    /// When false (default), uses pre-computed static codes (streaming, single-pass).
    /// When true, uses a two-pass mode: collect tokens first, build optimal codes, then write.
    pub optimize_codes: bool,
    /// Enable chroma-from-luma (CfL) optimization.
    /// When true (default), computes per-tile ytox/ytob values via least-squares fitting.
    /// When false, uses ytox=0, ytob=0 (no chroma decorrelation).
    pub cfl_enabled: bool,
}

impl Default for TinyEncoder {
    fn default() -> Self {
        Self {
            distance: 1.0,
            optimize_codes: false,
            cfl_enabled: true,
        }
    }
}

impl TinyEncoder {
    /// Create a new tiny encoder with the given distance.
    pub fn new(distance: f32) -> Self {
        Self {
            distance,
            optimize_codes: false,
            cfl_enabled: true,
        }
    }

    /// Encode an image in linear sRGB format.
    ///
    /// Input should be 3 channels (RGB) of f32 values in [0, 1] range.
    /// Values outside [0, 1] are allowed for out-of-gamut colors.
    pub fn encode(&self, width: usize, height: usize, linear_rgb: &[f32]) -> Result<Vec<u8>> {
        assert_eq!(linear_rgb.len(), width * height * 3);

        // Compute distance parameters
        let params = DistanceParams::compute(self.distance);

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

        // Convert to XYB
        let (xyb_x, xyb_y, xyb_b) = self.convert_to_xyb(width, height, linear_rgb);

        // Pad to block boundary
        let padded_width = xsize_blocks * BLOCK_DIM;
        let padded_height = ysize_blocks * BLOCK_DIM;

        // Compute adaptive per-block quantization field
        let quant_field = compute_adaptive_quant_field(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            width,
            height,
            xsize_blocks,
            ysize_blocks,
            self.distance,
            params.inv_scale,
        );

        // Compute per-tile chroma-from-luma map
        let cfl_map = if self.cfl_enabled {
            compute_cfl_map(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                width,
                height,
                xsize_blocks,
                ysize_blocks,
            )
        } else {
            CflMap::zeros(
                div_ceil(xsize_blocks, TILE_DIM_IN_BLOCKS),
                div_ceil(ysize_blocks, TILE_DIM_IN_BLOCKS),
            )
        };

        // Perform DCT and quantization
        let (quant_dc, quant_ac, nzeros) = self.transform_and_quantize(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            width,
            height,
            padded_width,
            padded_height,
            xsize_blocks,
            ysize_blocks,
            &params,
            &quant_field,
            &cfl_map,
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
                &quant_field,
                &cfl_map,
            );
        }

        // Get static entropy codes
        let dc_code = get_dc_entropy_code();
        let ac_code = get_ac_entropy_code();

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
        write_frame_header(params.x_qm_scale, params.epf_iters, &mut writer)?;
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
            self.write_dc_global(&params, num_dc_groups, &dc_code, &mut dc_global)?;

            let mut dc_group = BitWriter::new();
            self.write_dc_group(
                0,
                &quant_dc,
                xsize_blocks,
                ysize_blocks,
                xsize_dc_groups,
                &quant_field,
                &cfl_map,
                &dc_code,
                &mut dc_group,
            )?;

            let mut ac_global = BitWriter::new();
            self.write_ac_global(num_groups, &ac_code, &mut ac_global)?;

            let mut ac_group_writer = BitWriter::new();
            self.write_ac_group(
                0,
                &quant_ac,
                &nzeros,
                xsize_blocks,
                ysize_blocks,
                xsize_groups,
                &ac_code,
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

            // DC Global section
            let mut dc_global = BitWriter::new();
            self.write_dc_global(&params, num_dc_groups, &dc_code, &mut dc_global)?;
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
                    &dc_code,
                    &mut dc_group,
                )?;
                dc_group.zero_pad_to_byte();
                sections.push(dc_group.finish());
            }

            // AC Global section
            let mut ac_global = BitWriter::new();
            self.write_ac_global(num_groups, &ac_code, &mut ac_global)?;
            ac_global.zero_pad_to_byte();
            sections.push(ac_global.finish());

            // AC group sections
            for group_idx in 0..num_groups {
                let mut ac_group_writer = BitWriter::new();
                self.write_ac_group(
                    group_idx,
                    &quant_ac,
                    &nzeros,
                    xsize_blocks,
                    ysize_blocks,
                    xsize_groups,
                    &ac_code,
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

    /// Convert linear RGB to XYB color space.
    fn convert_to_xyb(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = width * height;
        let mut xyb_x = vec![0.0f32; n];
        let mut xyb_y = vec![0.0f32; n];
        let mut xyb_b = vec![0.0f32; n];

        for i in 0..n {
            let r = linear_rgb[i * 3];
            let g = linear_rgb[i * 3 + 1];
            let b = linear_rgb[i * 3 + 2];
            let (x, y, b_out) = linear_rgb_to_xyb(r, g, b);
            xyb_x[i] = x;
            xyb_y[i] = y;
            xyb_b[i] = b_out;
        }

        (xyb_x, xyb_y, xyb_b)
    }

    /// Perform DCT and quantization on all blocks.
    ///
    /// Returns (quantized_dc, quantized_ac, nzeros)
    fn transform_and_quantize(
        &self,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        width: usize,
        height: usize,
        _padded_width: usize,
        _padded_height: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        params: &DistanceParams,
        quant_field: &[u8],
        cfl_map: &CflMap,
    ) -> (
        [Vec<Vec<i16>>; 3],
        [Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
        [Vec<Vec<u8>>; 3],
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

        let mut nzeros: [Vec<Vec<u8>>; 3] = [
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
            vec![vec![0u8; xsize_blocks]; ysize_blocks],
        ];

        // Channel data
        let channels = [xyb_x, xyb_y, xyb_b];

        // Process each block
        // We need to:
        // 1. DCT all 3 channels
        // 2. Apply CFL to X and B channel AC coefficients (subtract Y * factor)
        // 3. Quantize all channels

        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                // Step 1: Extract and DCT all 3 channels
                let mut dct_blocks = [[0.0f32; DCT_BLOCK_SIZE]; 3];

                for c in 0..3 {
                    let mut block = [0.0f32; DCT_BLOCK_SIZE];
                    for dy in 0..BLOCK_DIM {
                        for dx in 0..BLOCK_DIM {
                            let py = (by * BLOCK_DIM + dy).min(height - 1);
                            let px = (bx * BLOCK_DIM + dx).min(width - 1);
                            block[dy * BLOCK_DIM + dx] = channels[c][py * width + px];
                        }
                    }

                    #[cfg(feature = "debug-tokens")]
                    if by == 0 && bx == 0 && c == 0 {
                        let block_sum: f32 = block.iter().sum();
                        debug_log!(
                            "Block[0,0,c=0]: sum={:.4}, first={:.4}",
                            block_sum,
                            block[0]
                        );
                    }

                    dct_8x8(&block, &mut dct_blocks[c]);

                    #[cfg(feature = "debug-tokens")]
                    if by == 0 && bx == 0 && c == 0 {
                        debug_log!("DCT[0,0,c=0]: DC={:.4}", dct_blocks[c][0]);
                    }
                }

                // Step 2: Apply CFL (chroma from luma) to X and B channels
                // Uses per-tile ytox/ytob values from the CfL map.
                //
                // Note: CfL is only applied to AC coefficients (idx 1..64).
                // DC has its own separate CfL mechanism in the DC quantization
                // step (the fixed dc_cfl_factor = 0.5 for B channel). The decoder
                // applies tile-level CfL only to AC, not DC.
                let tx = bx / TILE_DIM_IN_BLOCKS;
                let ty_cfl = by / TILE_DIM_IN_BLOCKS;
                let x_factor = ytox_ratio(cfl_map.ytox_at(tx, ty_cfl));
                let b_factor = ytob_ratio(cfl_map.ytob_at(tx, ty_cfl));
                for idx in 1..DCT_BLOCK_SIZE {
                    dct_blocks[0][idx] -= x_factor * dct_blocks[1][idx];
                    dct_blocks[2][idx] -= b_factor * dct_blocks[1][idx];
                }

                // Step 3: Quantize DC and AC for all channels
                // Process Y first (c=1), then X (c=0) and B (c=2) for CFL
                for &c in &[1usize, 0, 2] {
                    // Quantize DC with CFL (Chroma From Luma)
                    // Formula: qdc = round(dc * INV_DC_QUANT[c] * scale_dc - Y_dc * cfl_factor[c])
                    // where scale_dc = quant_dc * scale = quant_dc * global_scale / 65536
                    // CFL factors: X=0, Y=0, B=INV_DC_QUANT[2]*DC_QUANT[1]=256*(1/512)=0.5
                    let dc = dct_blocks[c][0];
                    let dc_scale = INV_DC_QUANT[c] * params.scale_dc;
                    let dc_cfl_factor = if c == 2 {
                        // B channel: subtract 0.5 * Y_dc for chroma correlation
                        // CFL factor = INV_DC_QUANT[2] * DC_QUANT[1] = 256 * (1/512) = 0.5
                        0.5f32
                    } else {
                        0.0f32
                    };
                    let y_dc = quant_dc[1][by][bx] as f32;
                    let qdc = (dc * dc_scale - y_dc * dc_cfl_factor).round() as i16;
                    quant_dc[c][by][bx] = qdc;

                    #[cfg(feature = "debug-tokens")]
                    if by == 0 && bx == 0 {
                        debug_log!(
                            "Quant[0,0,c={}]: dc={:.4}, scale_dc={:.6}, cfl={:.1}*{:.1}={:.1}, qdc={}",
                            c,
                            dc,
                            dc_scale,
                            y_dc,
                            dc_cfl_factor,
                            y_dc * dc_cfl_factor,
                            qdc
                        );
                    }

                    // Quantize AC coefficients (CFL already applied above for B channel)
                    // Formula: qval = round(coef * (1/weight) * qac)
                    // where qac = scale * raw_quant
                    //
                    // libjxl-tiny's kQuantWeights are small values (e.g., 0.0003).
                    // libjxl-tiny's InvMatrix = 1/kQuantWeights = large values (e.g., 3152).
                    // Quantization uses InvMatrix (large values).
                    //
                    // Our QUANT_WEIGHTS are the small values (same as kQuantWeights).
                    // So we need to DIVIDE by QUANT_WEIGHTS (equivalent to multiplying by InvMatrix).
                    //
                    // CRITICAL: Use per-channel weights! Each channel has different weights.
                    let qac = params.scale * quant_field[by * xsize_blocks + bx] as f32;
                    let weights = super::quant::quant_weights(0, c); // DCT8 strategy=0, per-channel

                    let mut qblock = [0i32; DCT_BLOCK_SIZE];
                    for idx in 0..DCT_BLOCK_SIZE {
                        if idx == 0 {
                            // DC is handled separately
                            qblock[0] = 0;
                        } else {
                            let coef = dct_blocks[c][idx];
                            let weight = weights[idx];
                            // DIVIDE by weight (equivalent to multiplying by InvMatrix = 1/weight)
                            let qval = (coef * qac / weight).round() as i32;
                            qblock[idx] = qval;
                        }
                    }
                    quant_ac[c][by][bx] = qblock;

                    // Count non-zeros
                    let _nz = num_nonzero_8x8_except_dc(&qblock, &mut nzeros[c][by][bx]);

                    #[cfg(feature = "debug-tokens")]
                    if by == 0 && bx == 0 && c == 1 {
                        let nonzero_coeffs: Vec<(usize, i32, f32)> = qblock
                            .iter()
                            .enumerate()
                            .filter(|&(_, v)| *v != 0)
                            .map(|(i, v)| (i, *v, dct_blocks[c][i]))
                            .collect();
                        debug_log!(
                            "AC Y[0,0] qac={:.4} (scale={:.4} * raw_quant={}), nonzero coeffs: {:?}",
                            qac,
                            params.scale,
                            quant_field[by * xsize_blocks + bx],
                            nonzero_coeffs
                        );
                        debug_log!(
                            "AC Y[0,0] DC={:.4}, coeff[63]={:.4} (checkerboard freq)",
                            dct_blocks[c][0],
                            dct_blocks[c][63]
                        );
                    }
                }
            }
        }

        (quant_dc, quant_ac, nzeros)
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

    /// Write DC global section.
    ///
    /// This follows the libjxl-tiny WriteDCGlobal pattern:
    /// 1. Default dequant DC
    /// 2. Quant scales
    /// 3. Non-default BlockCtxMap + compact block context map
    /// 4. Default DC cmap
    /// 5. Context tree for modular stream
    /// 6. No LZ77
    /// 7. DC entropy code
    fn write_dc_global(
        &self,
        params: &DistanceParams,
        num_dc_groups: usize,
        dc_code: &super::entropy_code::EntropyCode,
        writer: &mut BitWriter,
    ) -> Result<()> {
        #[cfg(feature = "debug-tokens")]
        let start_bits = writer.bits_written();

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
            let bytes_before_pad = (total_bits + 7) / 8;
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
    fn write_dc_group(
        &self,
        dc_group_idx: usize,
        quant_dc: &[Vec<Vec<i16>>; 3],
        xsize_blocks: usize,
        ysize_blocks: usize,
        xsize_dc_groups: usize,
        quant_field: &[u8],
        cfl_map: &CflMap,
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

        // AC metadata header - uses region block count
        let num_blocks = region_xsize * region_ysize;
        let num_ac_blocks = num_blocks; // All DCT8, so all blocks are first blocks
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
                (total + 7) / 8
            );
        }

        Ok(())
    }

    /// Write AC global section.
    fn write_ac_global(
        &self,
        num_groups: usize,
        ac_code: &super::entropy_code::EntropyCode,
        writer: &mut BitWriter,
    ) -> Result<()> {
        #[cfg(feature = "debug-tokens")]
        let start_bits = writer.bits_written();

        writer.write(1, 1)?; // all default quant matrices

        let num_histo_bits = ceil_log2_nonzero(num_groups);
        if num_histo_bits != 0 {
            writer.write(num_histo_bits as usize, 0)?;
        }

        writer.write(2, 3)?;
        writer.write(13, 0)?; // all default coeff order

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
                "  ac_entropy_code: {} bits ({} contexts, {} prefix codes)",
                after_ac_code - before_ac_code,
                ac_code.num_contexts,
                ac_code.num_prefix_codes
            );
        }

        Ok(())
    }

    /// Write AC group section.
    fn write_ac_group(
        &self,
        group_idx: usize,
        quant_ac: &[Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
        nzeros: &[Vec<Vec<u8>>; 3],
        xsize_blocks: usize,
        ysize_blocks: usize,
        xsize_groups: usize,
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

        // Debug: track bits for specific groups (600x600 = 75x75 blocks, groups 0,4,5)
        let debug_this_group = (group_idx == 0 || group_idx == 4 || group_idx == 5)
            && xsize_blocks == 75
            && ysize_blocks == 75;
        let mut group_total_bits = 0usize;
        let mut group_block_count = 0usize;

        // Process blocks in row-major order, with channels interleaved per block
        // CRITICAL: libjxl-tiny loops: for block { for channel {Y,X,B} { tokenize } }
        // We must match this exact order!
        for by in start_by..end_by {
            for bx in start_bx..end_bx {
                let block_start_bits = writer.bits_written();

                // Process channels in order: Y (1), X (0), B (2)
                for &c in &[1usize, 0, 2] {
                    let block = &quant_ac[c][by][bx];
                    let nz = nzeros[c][by][bx];

                    // Predict nzeros from neighbors
                    // At group boundaries, treat as edge (no neighbor from other groups)
                    let row_top = if by > start_by {
                        Some(nzeros[c][by - 1].as_slice())
                    } else {
                        None // First row of group: no top neighbor
                    };

                    // For left prediction, we need the position relative to group start
                    // If at first column of group, use default prediction
                    let local_bx = bx - start_bx;
                    let predicted_nz = if local_bx == 0 {
                        // First column of group: predict from top only (or default)
                        match row_top {
                            Some(top) => top[bx] as i32,
                            None => 32, // Default value
                        }
                    } else {
                        // Not first column: use standard prediction
                        predict_from_top_and_left(row_top, &nzeros[c][by], bx, 32)
                    };

                    // Validate nzeros matches actual count (debug only)
                    #[cfg(debug_assertions)]
                    {
                        let actual_nz = block[1..].iter().filter(|&&x| x != 0).count() as u8;
                        debug_assert_eq!(
                            nz, actual_nz,
                            "nzeros mismatch at c={} by={} bx={}: stored={} actual={}",
                            c, by, bx, nz, actual_nz
                        );
                    }

                    // Tokenize AC coefficients
                    #[cfg(feature = "debug-tokens")]
                    if by < start_by + 2 && bx < start_bx + 2 {
                        debug_log!(
                            "AC[c={},by={},bx={}]: nz={}, predicted={}",
                            c,
                            by,
                            bx,
                            nz,
                            predicted_nz
                        );
                    }
                    tokenize_ac_coefficients(
                        block,
                        c,
                        AC_STRATEGY_DCT8,
                        nz,
                        predicted_nz,
                        ac_code,
                        writer,
                    )?;
                }

                // Debug: track bits per block
                let block_bits = writer.bits_written() - block_start_bits;
                group_total_bits += block_bits;
                group_block_count += 1;

                // Log first few blocks and any anomalously large blocks
                if debug_this_group && (group_block_count <= 4 || block_bits > 100) {
                    let local_by = by - start_by;
                    let local_bx_dbg = bx - start_bx;
                    eprintln!(
                        "  G{} block ({},{}) = {} bits [nz: Y={}, X={}, B={}]",
                        group_idx,
                        local_by,
                        local_bx_dbg,
                        block_bits,
                        nzeros[1][by][bx],
                        nzeros[0][by][bx],
                        nzeros[2][by][bx]
                    );
                }
            }
        }

        // Debug summary for tracked groups
        if debug_this_group {
            eprintln!(
                "Group {} summary: {} blocks, {} total bits, {:.1} bits/block avg",
                group_idx,
                group_block_count,
                group_total_bits,
                group_total_bits as f64 / group_block_count as f64
            );
        }

        #[cfg(feature = "debug-tokens")]
        {
            let total_bits = writer.bits_written() - start_bits;
            debug_log!(
                "AC_group {} breakdown: {} bits ({} bytes before pad)",
                group_idx,
                total_bits,
                (total_bits + 7) / 8
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
        quant_field: &[u8],
        cfl_map: &CflMap,
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
            );
            dc_tokens_per_group.push(dc_tokens);
            ac_metadata_tokens_per_group.push(md_tokens);
        }

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
                    for &c in &[1usize, 0, 2] {
                        let block = &quant_ac[c][by][bx];
                        let nz = nzeros[c][by][bx];
                        let local_bx = bx - start_bx;
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
                        let block_tokens =
                            collect_ac_coefficients(block, c, AC_STRATEGY_DCT8, nz, predicted_nz);
                        tokens.extend_from_slice(&block_tokens);
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
        let dc_owned_code = build_entropy_code(&all_dc_tokens, super::dc_coding::NUM_DC_CONTEXTS);

        // Merge all AC section tokens for frequency counting
        let total_ac_tokens: usize = ac_section_tokens.iter().map(|t| t.len()).sum();
        let mut all_ac_tokens = Vec::with_capacity(total_ac_tokens);
        for section in &ac_section_tokens {
            all_ac_tokens.extend_from_slice(section);
        }
        let ac_owned_code = build_entropy_code(&all_ac_tokens, super::ac_context::NUM_AC_CONTEXTS);

        let dc_code = dc_owned_code.as_entropy_code();
        let ac_code = ac_owned_code.as_entropy_code();

        // ── Pass 2: Write bitstream ──

        let mut writer = BitWriter::with_capacity(width * height * 4);

        // Write JXL signature
        writer.write(8, 0xFF)?;
        writer.write(8, 0x0A)?;

        // Write file header
        self.write_file_header(width, height, &mut writer)?;

        // Write frame header
        write_frame_header(params.x_qm_scale, params.epf_iters, &mut writer)?;

        if num_sections == 4 {
            // Single-group: combine sections at the bit level
            let mut dc_global = BitWriter::new();
            self.write_dc_global(params, num_dc_groups, &dc_code, &mut dc_global)?;

            let mut dc_group = BitWriter::new();
            self.write_dc_group_from_tokens(
                0,
                xsize_blocks,
                ysize_blocks,
                xsize_dc_groups,
                &dc_tokens_per_group[0],
                &ac_metadata_tokens_per_group[0],
                &dc_code,
                &mut dc_group,
            )?;

            let mut ac_global = BitWriter::new();
            self.write_ac_global(num_groups, &ac_code, &mut ac_global)?;

            let mut ac_group_writer = BitWriter::new();
            write_tokens(&ac_section_tokens[0], &ac_code, &mut ac_group_writer)?;

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
            self.write_dc_global(params, num_dc_groups, &dc_code, &mut dc_global)?;
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
                    &dc_code,
                    &mut dc_group,
                )?;
                dc_group.zero_pad_to_byte();
                sections.push(dc_group.finish());
            }

            // AC Global
            let mut ac_global = BitWriter::new();
            self.write_ac_global(num_groups, &ac_code, &mut ac_global)?;
            ac_global.zero_pad_to_byte();
            sections.push(ac_global.finish());

            // AC groups
            for ac_tokens in &ac_section_tokens {
                let mut ac_group_writer = BitWriter::new();
                write_tokens(ac_tokens, &ac_code, &mut ac_group_writer)?;
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
        dc_code: &super::entropy_code::EntropyCode,
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
        write_tokens(dc_tokens, dc_code, writer)?;

        // AC metadata sub-header (comes between DC tokens and AC metadata tokens)
        let num_blocks = region_xsize * region_ysize;
        let num_ac_blocks = num_blocks;
        let nb_bits = ceil_log2_nonzero(num_blocks);
        if nb_bits != 0 {
            writer.write(nb_bits as usize, (num_ac_blocks - 1) as u64)?;
        }
        writer.write(4, 3)?; // use global tree, default wp, no transforms

        // Write AC metadata tokens
        write_tokens(ac_metadata_tokens, dc_code, writer)?;

        Ok(())
    }

    /// Write entropy code (context map + prefix codes).
    fn write_entropy_code_header(
        &self,
        code: &super::entropy_code::EntropyCode,
        writer: &mut BitWriter,
    ) -> Result<()> {
        super::entropy_code::write_entropy_code(code, writer)
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
    fn test_convert_to_xyb() {
        let encoder = TinyEncoder::new(1.0);

        // Gray pixel
        let linear_rgb = vec![0.5, 0.5, 0.5];
        let (x, y, b) = encoder.convert_to_xyb(1, 1, &linear_rgb);

        // Gray should have X ≈ 0 (equal L and M)
        assert!(x[0].abs() < 0.01, "X should be near zero for gray");
        assert!(y[0] > 0.0, "Y should be positive");
        assert!(b[0] > 0.0, "B should be positive");
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
}
