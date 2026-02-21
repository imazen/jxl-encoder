// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Bitstream writing: file/frame headers, DC/AC group encoding, two-pass orchestrator.

use super::ac_context::BlockCtxMap;
use super::ac_group::{
    collect_ac_coefficients_into, predict_from_top_and_left, tokenize_ac_coefficients,
};
use super::ac_strategy::AcStrategyMap;
use super::chroma_from_luma::CflMap;
use super::common::*;
use super::dc_coding::{collect_ac_metadata_tokens_region, collect_dc_tokens_wp};
use super::encoder::{BuiltEntropyCode, VarDctEncoder};
use super::frame::{DistanceParams, write_quant_scales, write_toc};
use super::noise::{NoiseParams, write_noise_params};
use crate::api::ProgressiveMode;
use crate::bit_writer::BitWriter;
#[cfg(feature = "debug-tokens")]
use crate::debug_log;

use crate::entropy_coding::encode::{
    build_entropy_code_ans_with_options, build_entropy_code_with_options,
};
use crate::entropy_coding::token::Token;
use crate::error::Result;
use crate::headers::color_encoding::{ColorEncoding, RenderingIntent};
use crate::headers::extra_channels::ExtraChannelInfo;
use crate::headers::file_header::{BitDepth, FileHeader, ImageMetadata};
use crate::headers::frame_header::{BlendMode, FrameHeader, FrameOptions};

/// Progressive pass configuration computed from ProgressiveMode.
struct ProgressivePassConfig {
    /// Number of passes (1 for Single mode).
    num_passes: u32,
    /// Shift per pass (num_passes - 1 elements). Last pass has implicit shift=0.
    /// The encoder right-shifts coefficients by this amount before encoding;
    /// the decoder left-shifts before accumulating.
    shifts: Vec<u32>,
    /// Number of downsampling brackets.
    num_ds: u32,
    /// Downsample factors per bracket (1, 2, 4, or 8).
    ds_downsample: Vec<u32>,
    /// Last pass index per bracket.
    ds_last_pass: Vec<u32>,
}

impl ProgressivePassConfig {
    fn from_mode(mode: ProgressiveMode) -> Self {
        match mode {
            ProgressiveMode::Single => Self {
                num_passes: 1,
                shifts: Vec::new(),
                num_ds: 0,
                ds_downsample: Vec::new(),
                ds_last_pass: Vec::new(),
            },
            ProgressiveMode::QuantizedAcFullAc => Self {
                // 2-pass: coarse (shift=1) → refinement (shift=0)
                num_passes: 2,
                shifts: vec![1],
                num_ds: 1,
                ds_downsample: vec![2],
                ds_last_pass: vec![0],
            },
            ProgressiveMode::DcVlfLfAc => Self {
                // 3-pass: very coarse (shift=2) → medium (shift=0) → final (shift=0)
                // Matches libjxl's kDcVlfLfAc preset
                num_passes: 3,
                shifts: vec![2, 0],
                num_ds: 2,
                ds_downsample: vec![8, 4],
                ds_last_pass: vec![0, 1],
            },
        }
    }

    fn is_progressive(&self) -> bool {
        self.num_passes > 1
    }

    fn shift_for_pass(&self, pass: usize) -> u32 {
        if pass < self.shifts.len() {
            self.shifts[pass]
        } else {
            0
        }
    }
}

/// Right-shift with symmetric rounding (libjxl convention).
/// The encoder right-shifts before encoding; the decoder left-shifts on decode.
fn shift_right_round(v: i32, shift: u32) -> i32 {
    if shift == 0 {
        return v;
    }
    let s = 1i32 << shift;
    if v >= 0 {
        (v + (s >> 1)) >> shift
    } else {
        -((-v + (s >> 1)) >> shift)
    }
}

/// Split a quantized coefficient block into per-pass residuals.
///
/// For each pass p:
/// - Compute `encoded = shift_right_round(residual, shift[p])`
/// - `decoded = encoded << shift[p]`
/// - residual for next pass = residual - decoded
///
/// Returns a vector of per-pass coefficient blocks (same layout as input).
fn split_coefficients_into_passes(
    coefficients: &[i32],
    pass_config: &ProgressivePassConfig,
) -> Vec<Vec<i32>> {
    let num_passes = pass_config.num_passes as usize;
    let size = coefficients.len();

    let mut per_pass: Vec<Vec<i32>> = Vec::with_capacity(num_passes);
    let mut residual: Vec<i32> = coefficients.to_vec();

    for pass in 0..num_passes {
        let shift = pass_config.shift_for_pass(pass);
        let mut pass_coeffs = vec![0i32; size];

        for (i, r) in residual.iter_mut().enumerate() {
            let encoded = shift_right_round(*r, shift);
            pass_coeffs[i] = encoded;
            let decoded = encoded << shift;
            *r -= decoded;
        }

        per_pass.push(pass_coeffs);
    }

    per_pass
}

impl VarDctEncoder {
    /// Build a `FileHeader` for VarDCT encoding from current encoder settings.
    ///
    /// This produces the same bitstream as the old hand-rolled `write_file_header()`,
    /// but uses the shared `FileHeader` struct used by both lossy and lossless paths.
    pub(crate) fn build_file_header(
        &self,
        width: usize,
        height: usize,
        has_alpha: bool,
    ) -> FileHeader {
        let bit_depth = if self.bit_depth_16 {
            BitDepth::uint16()
        } else {
            BitDepth::uint8()
        };

        let mut color_encoding = if self.is_grayscale {
            ColorEncoding::gray()
        } else {
            ColorEncoding::srgb()
        };
        // VarDCT uses Relative rendering intent (matches libjxl)
        color_encoding.rendering_intent = RenderingIntent::Relative;
        if self.icc_profile.is_some() {
            color_encoding.want_icc = true;
        }

        let extra_channels = if has_alpha {
            vec![ExtraChannelInfo::alpha()]
        } else {
            Vec::new()
        };

        FileHeader {
            width: width as u32,
            height: height as u32,
            metadata: ImageMetadata {
                bit_depth,
                color_encoding,
                extra_channels,
                xyb_encoded: true, // Required for VarDCT
                ..ImageMetadata::default()
            },
        }
    }

    /// Write the file header, ICC profile, and zero-pad to byte boundary.
    ///
    /// This replaces the old hand-rolled file header writer with the shared
    /// `FileHeader::write()` path, then appends ICC data and byte-aligns.
    pub(crate) fn write_file_header_and_pad(
        &self,
        width: usize,
        height: usize,
        has_alpha: bool,
        writer: &mut BitWriter,
    ) -> Result<()> {
        let file_header = self.build_file_header(width, height, has_alpha);
        file_header.write(writer)?;

        // Write ICC profile data if present (after header, before zero pad)
        if let Some(ref icc) = self.icc_profile {
            crate::icc::write_icc(icc, writer)?;
        }

        // Zero pad to byte before frame
        writer.zero_pad_to_byte();

        Ok(())
    }

    /// Write LZ77 header: either `Bool(false)` (1 bit) or `Bool(true)` + params.
    ///
    /// Serialization format (from libjxl `dec_ans.cc:308-316`):
    ///
    /// ```text
    /// Bool(enabled)
    /// if enabled:
    ///   U32(Val(224), Val(512), Val(4096), BitsOffset(15,8))  // min_symbol
    ///   U32(Val(3), Val(4), BitsOffset(2,5), BitsOffset(8,9)) // min_length
    ///   EncodeUintConfig(length_uint_config, log_alpha_size=8)
    /// ```
    pub(crate) fn write_lz77_header(
        lz77: Option<&crate::entropy_coding::lz77::Lz77Params>,
        writer: &mut BitWriter,
    ) -> Result<()> {
        crate::entropy_coding::lz77::write_lz77_header(lz77, writer)
    }

    /// Write DC global section (LfGlobal).
    ///
    /// Decoder order (from jxl-rs `frame/decode.rs`):
    ///
    /// 1. Patches (if enabled) — not used
    /// 2. Splines (if enabled) — not used
    /// 3. Noise params (if ENABLE_NOISE flag set) — 8 × 10-bit LUT values
    /// 4. Default dequant DC (LfQuantFactors)
    /// 5. Quant scales (QuantizerParams)
    /// 6. Non-default BlockCtxMap + compact block context map
    /// 7. Default DC cmap (ColorCorrelationParams)
    /// 8. Context tree for modular stream
    /// 9. LZ77 params (disabled or enabled with RLE config)
    /// 10. DC entropy code
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_dc_global(
        &self,
        params: &DistanceParams,
        num_dc_groups: usize,
        dc_code: &BuiltEntropyCode,
        noise_params: &Option<NoiseParams>,
        dc_lz77_params: Option<&crate::entropy_coding::lz77::Lz77Params>,
        block_ctx_map: &BlockCtxMap,
        learned_tree_tokens: Option<&[(u32, u32)]>,
        patches: Option<&super::patches::PatchesData>,
        splines: Option<&super::splines::SplinesData>,
        dc_quant_custom: Option<[f32; 3]>,
        writer: &mut BitWriter,
    ) -> Result<()> {
        #[cfg(feature = "debug-tokens")]
        let start_bits = writer.bits_written();

        // Write patches section before splines (JXL spec ordering in LfGlobal)
        if let Some(pd) = patches {
            #[cfg(feature = "trace-bitstream")]
            let patch_start = writer.bits_written();
            super::patches::encode_patches_section(pd, self.use_ans, writer)?;
            #[cfg(feature = "trace-bitstream")]
            {
                let patch_dict_bytes = (writer.bits_written() - patch_start).div_ceil(8);
                eprintln!(
                    "PATCHES: dict section = {} bytes ({} tokens)",
                    patch_dict_bytes,
                    pd.ref_positions.len() + pd.positions.len() * 3
                );
            }
        }

        // Write splines section (after patches, before noise)
        if let Some(sd) = splines {
            #[cfg(feature = "trace-bitstream")]
            eprintln!("SPLINES_SECTION: start at bit {}", writer.bits_written());
            super::splines::encode_splines_section(sd, writer)?;
            #[cfg(feature = "trace-bitstream")]
            eprintln!("SPLINES_SECTION: end at bit {}", writer.bits_written());
        }

        // Write noise parameters before dequant DC (decoder expects this order)
        if let Some(ref noise) = *noise_params {
            write_noise_params(noise, writer)?;
        }

        crate::f16::write_lf_quant(writer, dc_quant_custom)?;

        #[cfg(feature = "debug-tokens")]
        let after_dequant_dc = writer.bits_written();

        write_quant_scales(params.global_scale, params.quant_dc, writer)?;

        #[cfg(feature = "debug-tokens")]
        let after_quant = writer.bits_written();
        // BlockCtxMap
        if block_ctx_map.qf_thresholds.is_empty()
            && block_ctx_map.num_ctxs == super::ac_context::NUM_BLOCK_CTXS
        {
            // Default map: write non-default flag + hardcoded compact map
            writer.write(1, 0)?; // non-default BlockCtxMap
            writer.write(16, 0)?; // no dc ctx, no qft
            super::context_tree::write_block_context_map(writer)?;
        } else {
            // Adaptive map: write full header with QF thresholds and context map
            super::context_tree::write_block_ctx_map_adaptive(block_ctx_map, writer)?;
        }

        #[cfg(feature = "debug-tokens")]
        let after_block_ctx = writer.bits_written();

        writer.write(1, 1)?; // default DC cmap

        // Write context tree for modular stream DC header
        if let Some(tree_tokens) = learned_tree_tokens {
            super::context_tree::write_learned_context_tree(tree_tokens, num_dc_groups, writer)?;
        } else {
            super::context_tree::write_context_tree(num_dc_groups, writer)?;
        }

        #[cfg(feature = "debug-tokens")]
        let after_ctx_tree = writer.bits_written();

        // Write LZ77 params
        Self::write_lz77_header(dc_lz77_params, writer)?;

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
            debug_log!("  lz77: {} bits", after_lz77 - after_ctx_tree);
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_dc_group(
        &self,
        dc_group_idx: usize,
        quant_dc: &[Vec<Vec<i16>>; 3],
        xsize_blocks: usize,
        ysize_blocks: usize,
        xsize_dc_groups: usize,
        quant_field: &[u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        sharpness_map: Option<&[u8]>,
        dc_code: &crate::entropy_coding::encode::EntropyCode,
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
        super::dc_coding::write_dc_tokens_region(
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
        super::dc_coding::write_ac_metadata_tokens_region(
            region_xsize,
            region_ysize,
            quant_field,
            xsize_blocks,
            start_bx,
            start_by,
            cfl_map,
            ac_strategy,
            sharpness_map,
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
    #[allow(clippy::too_many_arguments)]
    /// Write HfGlobal section.
    ///
    /// For progressive encoding (`ac_codes.len() > 1`), the decoder reads per-pass
    /// data: used_orders, coeff_orders, and histograms for each pass.
    /// The dequant matrices and num_histograms are written once (shared).
    pub(crate) fn write_ac_global(
        &self,
        num_groups: usize,
        ac_codes: &[BuiltEntropyCode],
        used_orders: u32,
        coeff_order_tokens: Option<&[Token]>,
        ac_lz77_params: &[Option<crate::entropy_coding::lz77::Lz77Params>],
        writer: &mut BitWriter,
    ) -> Result<()> {
        #[cfg(feature = "debug-tokens")]
        let start_bits = writer.bits_written();

        writer.write(1, 1)?; // all default quant matrices

        let num_histo_bits = ceil_log2_nonzero(num_groups);
        if num_histo_bits != 0 {
            writer.write(num_histo_bits as usize, 0)?;
        }

        // Per-pass: used_orders, coeff_orders, histograms
        let num_passes = ac_codes.len();
        for pass in 0..num_passes {
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

            // Write LZ77 params for this pass
            Self::write_lz77_header(ac_lz77_params[pass].as_ref(), writer)?;

            #[cfg(feature = "debug-tokens")]
            let before_ac_code = writer.bits_written();

            // Write entropy code for this pass
            self.write_entropy_code_header(&ac_codes[pass], writer)?;

            #[cfg(feature = "debug-tokens")]
            {
                let after_ac_code = writer.bits_written();
                debug_log!("AC_global pass {} breakdown:", pass);
                debug_log!("  header: {} bits", before_ac_code - start_bits);
                debug_log!(
                    "  ac_entropy_code: {} bits ({} contexts, {} histograms)",
                    after_ac_code - before_ac_code,
                    ac_codes[pass].num_contexts(),
                    ac_codes[pass].num_histograms()
                );
            }
        }

        Ok(())
    }

    /// Write AC group section.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_ac_group(
        &self,
        group_idx: usize,
        quant_ac: &[Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
        nzeros: &[Vec<Vec<u8>>; 3],
        raw_nzeros: &[Vec<Vec<u16>>; 3],
        xsize_blocks: usize,
        ysize_blocks: usize,
        xsize_groups: usize,
        quant_field: &[u8],
        ac_strategy: &AcStrategyMap,
        block_ctx_map: &BlockCtxMap,
        ac_code: &crate::entropy_coding::encode::EntropyCode,
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

        // Pre-allocate scratch buffer for multi-block coefficient assembly (max DCT64x64 = 4096)
        const MAX_BLOCK_SIZE: usize = 4096;
        let mut full_block_scratch = vec![0i32; MAX_BLOCK_SIZE];

        // Process blocks in row-major order, with channels interleaved per block
        // CRITICAL: libjxl-tiny loops: for block { for channel {Y,X,B} { tokenize } }
        // We must match this exact order!
        for by in start_by..end_by {
            for bx in start_bx..end_bx {
                // Skip non-first blocks of multi-block transforms
                if !ac_strategy.is_first(bx, by) {
                    continue;
                }

                let raw_strategy = ac_strategy.raw_strategy(bx, by);
                let covered_x = ac_strategy.covered_blocks_x(bx, by);
                let covered_y = ac_strategy.covered_blocks_y(bx, by);
                let covered_blocks = covered_x * covered_y;
                let size = covered_blocks * DCT_BLOCK_SIZE;
                let _strategy_code = ac_strategy.strategy_code(bx, by);

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
                        // DCT8/DCT4X8/DCT8X4: use existing single-block path
                        // Streaming path: no custom orders (requires two-pass)
                        // tokenize_ac_coefficients expects raw_strategy, not bitstream code
                        let strategy_code = ac_strategy.strategy_code(bx, by);
                        let qf_val = quant_field[by * xsize_blocks + bx] as u32;
                        let block_ctx = block_ctx_map.block_context(c, strategy_code, qf_val);
                        tokenize_ac_coefficients(
                            &quant_ac[c][by][bx],
                            raw_strategy,
                            nz,
                            predicted_nz,
                            block_ctx,
                            block_ctx_map.num_ctxs,
                            ac_code,
                            writer,
                            None,
                        )?;
                    } else {
                        // Multi-block: assemble contiguous coefficient buffer in flat layout.
                        // tokenize_ac_coefficients uses COEFF_ORDER which indexes into a flat
                        // cx*8 × cy*8 layout (stride = cx*8), not 8x8 block slots.
                        //
                        // NOTE: For rectangular transforms, cx >= cy after swap, so stride = cx * 8.
                        // covered_x may differ from cx for DCT16x8/DCT8x16.
                        let (cx, _cy) = if covered_y > covered_x {
                            (covered_y, covered_x)
                        } else {
                            (covered_x, covered_y)
                        };
                        let transpose_slots = covered_y > covered_x;
                        let stride = cx * BLOCK_DIM;
                        let full_block = &mut full_block_scratch[..size];
                        #[allow(clippy::needless_range_loop)]
                        for idx in 0..size {
                            let y = idx / stride;
                            let x = idx % stride;
                            let coef_slot_y = y / BLOCK_DIM;
                            let coef_slot_x = x / BLOCK_DIM;
                            let pos_y = y % BLOCK_DIM;
                            let pos_x = x % BLOCK_DIM;
                            let pos_in_8x8 = pos_y * BLOCK_DIM + pos_x;
                            let (phys_row_off, phys_col_off) = if transpose_slots {
                                (coef_slot_x, coef_slot_y)
                            } else {
                                (coef_slot_y, coef_slot_x)
                            };
                            full_block[idx] =
                                quant_ac[c][by + phys_row_off][bx + phys_col_off][pos_in_8x8];
                        }

                        #[cfg(feature = "debug-tokens")]
                        if raw_strategy == 4 && c == 1 && bx == 0 && by == 0 {
                            // Debug: count nonzeros in full_block for DCT32x32
                            let nz_count = full_block.iter().filter(|&&v| v != 0).count();
                            eprintln!(
                                "[DCT32x32 debug] full_block for Y at (0,0): {} nonzeros out of {}",
                                nz_count, size
                            );
                            if nz_count > 0 && nz_count <= 20 {
                                for (i, &v) in full_block.iter().enumerate() {
                                    if v != 0 {
                                        eprintln!("  [{:4}] = {}", i, v);
                                    }
                                }
                            }
                        }
                        // Streaming path: no custom orders
                        // tokenize_ac_coefficients expects raw_strategy, not bitstream code
                        let strategy_code_2 = ac_strategy.strategy_code(bx, by);
                        let qf_val = quant_field[by * xsize_blocks + bx] as u32;
                        let block_ctx = block_ctx_map.block_context(c, strategy_code_2, qf_val);
                        tokenize_ac_coefficients(
                            full_block,
                            raw_strategy,
                            nz,
                            predicted_nz,
                            block_ctx,
                            block_ctx_map.num_ctxs,
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
    /// Encode a single frame to an existing BitWriter (no file header).
    ///
    /// Used by `encode_animation()` to write individual frames after the file header
    /// has already been written. The `frame_options` control animation-specific fields
    /// (duration, is_last, have_animation).
    #[allow(dead_code)]
    pub(crate) fn encode_frame_to_writer(
        &self,
        width: usize,
        height: usize,
        linear_rgb: &[f32],
        alpha: Option<&[u8]>,
        frame_options: &FrameOptions,
        writer: &mut BitWriter,
    ) -> Result<[u32; 19]> {
        // Reuse the full encode pipeline from encode() but write to an existing writer.
        // This duplicates some setup from encode(), but keeps the code paths separate.
        let xsize_blocks = div_ceil(width, BLOCK_DIM);
        let ysize_blocks = div_ceil(height, BLOCK_DIM);
        let xsize_groups = div_ceil(width, GROUP_DIM);
        let ysize_groups = div_ceil(height, GROUP_DIM);
        let xsize_dc_groups = div_ceil(width, DC_GROUP_DIM);
        let ysize_dc_groups = div_ceil(height, DC_GROUP_DIM);
        let num_groups = xsize_groups * ysize_groups;
        let num_dc_groups = xsize_dc_groups * ysize_dc_groups;
        let num_sections = 2 + num_dc_groups + num_groups;
        let padded_width = xsize_blocks * BLOCK_DIM;
        let padded_height = ysize_blocks * BLOCK_DIM;

        let (mut xyb_x, mut xyb_y, mut xyb_b) =
            self.convert_to_xyb_padded(width, height, padded_width, padded_height, linear_rgb);

        let noise_params = if self.enable_noise {
            let quality_coef = super::noise::noise_quality_coef(self.distance);
            let params = super::noise::estimate_noise_params(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                quality_coef,
            );
            if self.enable_denoise
                && let Some(ref p) = params
            {
                super::noise::denoise_xyb(
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

        let pixel_stats = super::frame::PixelStatsForChromacityAdjustment::calc(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            padded_width,
            padded_height,
        );
        let chromacity_x = pixel_stats.how_much_is_x_channel_pixelized();
        let chromacity_b = pixel_stats.how_much_is_b_channel_pixelized();

        if self.enable_gaborish {
            super::gaborish::gaborish_inverse(
                &mut xyb_x,
                &mut xyb_y,
                &mut xyb_b,
                padded_width,
                padded_height,
            );
        }

        let distance_for_iqf = if self.enable_gaborish {
            self.distance
        } else {
            self.distance * 0.62
        };

        let (mut quant_field_float, masking) = super::adaptive_quant::compute_quant_field_float(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            padded_width,
            padded_height,
            xsize_blocks,
            ysize_blocks,
            distance_for_iqf,
            self.profile.k_ac_quant,
        );

        let mut params = DistanceParams::compute_for_profile(self.distance, &self.profile);
        if self.profile.chromacity_adjustment {
            params.apply_chromacity_adjustment(chromacity_x, chromacity_b);
        }

        let mut quant_field =
            super::adaptive_quant::quantize_quant_field(&quant_field_float, params.inv_scale);

        let cfl_map = if self.cfl_enabled {
            super::chroma_from_luma::compute_cfl_map(
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

        let mask1x1 = if self.ac_strategy_enabled && self.pixel_domain_loss {
            Some(super::adaptive_quant::compute_mask1x1(
                &xyb_y,
                padded_width,
                padded_height,
            ))
        } else {
            None
        };

        let ac_strategy = if let Some(forced) = self.force_strategy {
            super::encoder::force_strategy_map(xsize_blocks, ysize_blocks, forced)
        } else if !self.ac_strategy_enabled {
            AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks)
        } else {
            super::ac_strategy::compute_ac_strategy(
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

        super::ac_strategy::adjust_quant_field_with_distance(
            &ac_strategy,
            &mut quant_field,
            self.distance,
        );
        super::ac_strategy::adjust_quant_field_float_with_distance(
            &ac_strategy,
            &mut quant_field_float,
            self.distance,
        );

        #[cfg(feature = "butteraugli-loop")]
        if self.butteraugli_iters > 0 {
            let initial_qf_float = quant_field_float.clone();
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
                None, // No patches in this code path
                None, // No splines in this code path
            );
        }

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

        let sharpness_map =
            if params.epf_iters > 0 && self.distance >= 0.5 && self.profile.epf_dynamic_sharpness {
                let mask = mask1x1.unwrap_or_else(|| {
                    super::adaptive_quant::compute_mask1x1(&xyb_y, padded_width, padded_height)
                });
                Some(super::epf::compute_epf_sharpness(
                    [&xyb_x, &xyb_y, &xyb_b],
                    &transform_out.quant_dc,
                    &transform_out.quant_ac,
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

        let strategy_counts = ac_strategy.strategy_histogram();

        self.encode_two_pass_to_writer(
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
            &transform_out.quant_dc,
            &transform_out.quant_ac,
            &transform_out.nzeros,
            &transform_out.raw_nzeros,
            &quant_field,
            &cfl_map,
            &ac_strategy,
            &noise_params,
            sharpness_map.as_deref(),
            alpha,
            Some(frame_options),
            None, // No patches in animation frames
            None, // No splines in animation frames
            None, // No LfFrame in animation frames
            writer,
        )?;

        Ok(strategy_counts)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_two_pass(
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
        raw_nzeros: &[Vec<Vec<u16>>; 3],
        quant_field: &[u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        noise_params: &Option<NoiseParams>,
        sharpness_map: Option<&[u8]>,
        alpha: Option<&[u8]>,
        patches: Option<&super::patches::PatchesData>,
        splines: Option<&super::splines::SplinesData>,
        float_dc: Option<&[Vec<f32>; 3]>,
    ) -> Result<Vec<u8>> {
        let mut writer = BitWriter::with_capacity(width * height * 4);

        // Write file header
        let has_alpha = alpha.is_some();
        self.write_file_header_and_pad(width, height, has_alpha, &mut writer)?;

        // Write LfFrame (separate DC frame) before other frames.
        // Must come before patches ref frame and main VarDCT frame.
        //
        // The encode returns decoded-back DC values (quantize→dequantize roundtrip)
        // matching libjxl's decode-back step (enc_cache.cc:195-222). These represent
        // the exact DC values the decoder will reconstruct from the LfFrame.
        let lf_dc_quant: Option<[f32; 3]> = if self.use_lf_frame
            && let Some(dc) = float_dc
        {
            let (_decoded_dc, dc_quant) = super::lf_frame::encode_lf_frame(
                dc,
                self.distance,
                xsize_blocks,
                ysize_blocks,
                self.use_ans,
                self.effort,
                &mut writer,
            )?;
            writer.zero_pad_to_byte();
            // Pass the LfFrame's dc_quant to the main frame's LfGlobal so the
            // decoder knows the correct dequantization factors for lf_image.
            Some(dc_quant)
        } else {
            None
        };

        // If patches present, write the reference frame before the main frame.
        // The reference frame is a modular FrameType::ReferenceOnly frame that
        // stores unique patch templates. The main frame then references it.
        if let Some(pd) = patches {
            #[cfg(feature = "trace-bitstream")]
            let ref_frame_start = writer.bits_written();
            #[cfg(feature = "trace-bitstream")]
            eprintln!(
                "PATCHES: writing reference frame at bit {} (byte {})",
                ref_frame_start,
                ref_frame_start / 8
            );
            super::patches::encode_reference_frame(pd, self.use_ans, &mut writer)?;
            writer.zero_pad_to_byte();
            #[cfg(feature = "trace-bitstream")]
            {
                let ref_frame_bytes = (writer.bits_written() - ref_frame_start).div_ceil(8);
                eprintln!(
                    "PATCHES: ref frame {}x{} = {} bytes, {} unique patches, {} occurrences",
                    pd.ref_width,
                    pd.ref_height,
                    ref_frame_bytes,
                    pd.ref_positions.len(),
                    pd.positions.len()
                );
            }
        }

        // Write main VarDCT frame (header + TOC + sections)
        self.encode_two_pass_to_writer(
            width,
            height,
            params,
            xsize_blocks,
            ysize_blocks,
            xsize_groups,
            _ysize_groups,
            xsize_dc_groups,
            _ysize_dc_groups,
            num_groups,
            num_dc_groups,
            num_sections,
            quant_dc,
            quant_ac,
            nzeros,
            raw_nzeros,
            quant_field,
            cfl_map,
            ac_strategy,
            noise_params,
            sharpness_map,
            alpha,
            None,
            patches,
            splines,
            lf_dc_quant,
            &mut writer,
        )?;

        Ok(writer.finish_with_padding())
    }

    /// Write a VarDCT frame to a BitWriter (two-pass mode).
    ///
    /// If `frame_options` is Some, overrides frame header fields (for animation).
    /// If None, uses default lossy frame header settings.
    #[allow(clippy::too_many_arguments)]
    fn encode_two_pass_to_writer(
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
        _num_sections: usize,
        quant_dc: &[Vec<Vec<i16>>; 3],
        quant_ac: &[Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
        nzeros: &[Vec<Vec<u8>>; 3],
        raw_nzeros: &[Vec<Vec<u16>>; 3],
        quant_field: &[u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        noise_params: &Option<NoiseParams>,
        sharpness_map: Option<&[u8]>,
        alpha: Option<&[u8]>,
        frame_options: Option<&FrameOptions>,
        patches: Option<&super::patches::PatchesData>,
        splines: Option<&super::splines::SplinesData>,
        dc_quant_custom: Option<[f32; 3]>,
        writer: &mut BitWriter,
    ) -> Result<()> {
        // ── Pass 1: Collect tokens per section ──

        // Build context tree and collect tokens.
        // When use_lf_frame: DC is in separate frame, only AC metadata tree/tokens needed.
        // Otherwise: merged WP DC + AC metadata tree with both token types.
        let (learned_tree_tokens, total_contexts, ac_meta_ctx_map);
        let mut dc_tokens_per_group: Vec<Vec<Token>> = Vec::with_capacity(num_dc_groups);
        let mut ac_metadata_tokens_per_group: Vec<Vec<Token>> = Vec::with_capacity(num_dc_groups);

        if self.use_lf_frame {
            // AC-metadata-only tree (no DC contexts needed)
            let (tree_tokens, num_ctx, ctx_map) = super::dc_tree_learn::ac_metadata_only_tree();
            learned_tree_tokens = Some(tree_tokens);
            total_contexts = num_ctx;
            ac_meta_ctx_map = ctx_map;

            // Only collect AC metadata tokens (no DC)
            for dc_group_idx in 0..num_dc_groups {
                let dc_gx = dc_group_idx % xsize_dc_groups;
                let dc_gy = dc_group_idx / xsize_dc_groups;
                let start_bx = dc_gx * DC_GROUP_DIM_IN_BLOCKS;
                let start_by = dc_gy * DC_GROUP_DIM_IN_BLOCKS;
                let end_bx = (start_bx + DC_GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
                let end_by = (start_by + DC_GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
                let region_xsize = end_bx - start_bx;
                let region_ysize = end_by - start_by;

                dc_tokens_per_group.push(Vec::new()); // no DC tokens

                let md_tokens = collect_ac_metadata_tokens_region(
                    region_xsize,
                    region_ysize,
                    quant_field,
                    xsize_blocks,
                    start_bx,
                    start_by,
                    cfl_map,
                    ac_strategy,
                    sharpness_map,
                );
                let md_tokens: Vec<Token> = md_tokens
                    .into_iter()
                    .map(|mut t| {
                        t.context = ac_meta_ctx_map[t.context as usize];
                        t
                    })
                    .collect();
                ac_metadata_tokens_per_group.push(md_tokens);
            }
        } else {
            // Build kWPFixedDC tree for DC context assignment.
            // Uses Weighted Predictor with balanced BSP on wp_max_error (property 15).
            // Matches libjxl's PredefinedTree(kWPFixedDC) at speed_tier <= kFalcon (effort >= 4).
            let total_dc_pixels = xsize_blocks * ysize_blocks * 3;
            let (wp_dc_tree, wp_dc_num_contexts) =
                super::dc_tree_learn::build_wp_fixed_dc_tree(total_dc_pixels, 8);

            let (wrapped_tokens, num_ctx, dc_remap, ctx_map) =
                super::dc_tree_learn::tree_tokens_with_ac_metadata_prefix(
                    &wp_dc_tree,
                    wp_dc_num_contexts,
                );

            learned_tree_tokens = Some(wrapped_tokens);
            total_contexts = num_ctx;
            ac_meta_ctx_map = ctx_map;
            let dc_ctx_remap = dc_remap;

            #[cfg(feature = "debug-tokens")]
            eprintln!(
                "WP fixed DC tree: dc_contexts={}, total={}, dc_remap={:?}, ac_map={:?}",
                wp_dc_num_contexts, total_contexts, dc_ctx_remap, ac_meta_ctx_map
            );

            for dc_group_idx in 0..num_dc_groups {
                let dc_gx = dc_group_idx % xsize_dc_groups;
                let dc_gy = dc_group_idx / xsize_dc_groups;
                let start_bx = dc_gx * DC_GROUP_DIM_IN_BLOCKS;
                let start_by = dc_gy * DC_GROUP_DIM_IN_BLOCKS;
                let end_bx = (start_bx + DC_GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
                let end_by = (start_by + DC_GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
                let region_xsize = end_bx - start_bx;
                let region_ysize = end_by - start_by;

                // Collect DC tokens using Weighted Predictor + kWPFixedDC tree
                let dc_tokens =
                    collect_dc_tokens_wp(quant_dc, &wp_dc_tree, start_bx, start_by, end_bx, end_by);
                let md_tokens = collect_ac_metadata_tokens_region(
                    region_xsize,
                    region_ysize,
                    quant_field,
                    xsize_blocks,
                    start_bx,
                    start_by,
                    cfl_map,
                    ac_strategy,
                    sharpness_map,
                );
                // Remap DC token contexts to match BFS ordering of merged tree.
                let dc_tokens: Vec<Token> = dc_tokens
                    .into_iter()
                    .map(|mut t| {
                        t.context = dc_ctx_remap[t.context as usize];
                        t
                    })
                    .collect();
                dc_tokens_per_group.push(dc_tokens);

                let md_tokens: Vec<Token> = md_tokens
                    .into_iter()
                    .map(|mut t| {
                        t.context = ac_meta_ctx_map[t.context as usize];
                        t
                    })
                    .collect();
                ac_metadata_tokens_per_group.push(md_tokens);
            }
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

        // Compute content-adaptive block context map
        let block_ctx_map = super::ac_context::compute_block_ctx_map(
            quant_field,
            ac_strategy,
            params.distance,
            xsize_blocks,
            ysize_blocks,
        );

        // ── Progressive pass configuration ──
        let pass_config = ProgressivePassConfig::from_mode(self.progressive);
        let num_passes = pass_config.num_passes as usize;
        #[cfg(feature = "debug-tokens")]
        eprintln!(
            "[PROGRESSIVE] mode={:?}, num_passes={}, num_groups={}",
            self.progressive, num_passes, num_groups
        );

        // Override num_sections for progressive: each pass has its own HfGroup sections
        let num_sections = 2 + num_dc_groups + num_groups * num_passes;

        // AC section tokens: per-pass per-group
        // ac_section_tokens_per_pass[pass][group] = Vec<Token>
        const MAX_BLOCK_SIZE: usize = 4096;
        let mut full_block_scratch = vec![0i32; MAX_BLOCK_SIZE];
        let mut pass_block_scratch = vec![0i32; MAX_BLOCK_SIZE];

        let mut ac_section_tokens_per_pass: Vec<Vec<Vec<Token>>> =
            vec![Vec::with_capacity(num_groups); num_passes];

        // For progressive encoding, we need per-pass nzeros grids for prediction.
        // pass_nzeros_grids[pass][channel][row][col] = shifted nzeros for that pass.
        let mut pass_nzeros_grids: Vec<[Vec<Vec<u8>>; 3]> = if pass_config.is_progressive() {
            (0..num_passes)
                .map(|_| core::array::from_fn(|_| vec![vec![0u8; xsize_blocks]; ysize_blocks]))
                .collect()
        } else {
            Vec::new()
        };

        for group_idx in 0..num_groups {
            let group_x = group_idx % xsize_groups;
            let group_y = group_idx / xsize_groups;
            let start_bx = group_x * GROUP_DIM_IN_BLOCKS;
            let start_by = group_y * GROUP_DIM_IN_BLOCKS;
            let end_bx = (start_bx + GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
            let end_by = (start_by + GROUP_DIM_IN_BLOCKS).min(ysize_blocks);

            let region_blocks = (end_bx - start_bx) * (end_by - start_by);

            // Initialize per-pass token vecs for this group
            let mut pass_tokens: Vec<Vec<Token>> = (0..num_passes)
                .map(|_| Vec::with_capacity(region_blocks * 64 * 3 / num_passes))
                .collect();

            for by in start_by..end_by {
                for bx in start_bx..end_bx {
                    if !ac_strategy.is_first(bx, by) {
                        continue;
                    }
                    let covered_x = ac_strategy.covered_blocks_x(bx, by);
                    let covered_y = ac_strategy.covered_blocks_y(bx, by);
                    let covered_blocks = covered_x * covered_y;
                    let size = covered_blocks * DCT_BLOCK_SIZE;
                    let raw_strategy = ac_strategy.raw_strategy(bx, by);
                    let strategy_code = ac_strategy.strategy_code(bx, by);

                    for &c in &[1usize, 0, 2] {
                        // Get custom order for this (bucket, channel) if available
                        let custom_ord = custom_order_map.as_ref().and_then(|orders| {
                            super::coeff_order::get_custom_order(
                                orders,
                                used_orders,
                                strategy_code,
                                c,
                            )
                        });

                        let qf_val = quant_field[by * xsize_blocks + bx] as u32;
                        let block_ctx = block_ctx_map.block_context(c, strategy_code, qf_val);

                        // Assemble the full coefficient block
                        let full_block: &[i32] = if covered_blocks == 1 {
                            &quant_ac[c][by][bx]
                        } else {
                            let (cx, _cy) = if covered_y > covered_x {
                                (covered_y, covered_x)
                            } else {
                                (covered_x, covered_y)
                            };
                            let transpose_slots = covered_y > covered_x;
                            let stride = cx * BLOCK_DIM;
                            let fb = &mut full_block_scratch[..size];
                            #[allow(clippy::needless_range_loop)]
                            for idx in 0..size {
                                let y = idx / stride;
                                let x = idx % stride;
                                let coef_slot_y = y / BLOCK_DIM;
                                let coef_slot_x = x / BLOCK_DIM;
                                let pos_y = y % BLOCK_DIM;
                                let pos_x = x % BLOCK_DIM;
                                let pos_in_8x8 = pos_y * BLOCK_DIM + pos_x;
                                let (phys_row_off, phys_col_off) = if transpose_slots {
                                    (coef_slot_x, coef_slot_y)
                                } else {
                                    (coef_slot_y, coef_slot_x)
                                };
                                fb[idx] =
                                    quant_ac[c][by + phys_row_off][bx + phys_col_off][pos_in_8x8];
                            }
                            &full_block_scratch[..size]
                        };

                        if !pass_config.is_progressive() {
                            // Single-pass: use original nzeros and collect directly
                            let nz = raw_nzeros[c][by][bx];
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

                            collect_ac_coefficients_into(
                                &mut pass_tokens[0],
                                full_block,
                                raw_strategy,
                                nz,
                                predicted_nz,
                                block_ctx,
                                block_ctx_map.num_ctxs,
                                custom_ord,
                            );
                        } else {
                            // Multi-pass: split coefficients and tokenize per-pass
                            let pass_blocks =
                                split_coefficients_into_passes(full_block, &pass_config);

                            for (pass, pass_coeffs) in pass_blocks.iter().enumerate() {
                                // Count non-zeros for this pass's coefficients
                                // (skip covered_blocks positions = LLF coefficients)
                                let pass_nz: u16 = pass_coeffs[covered_blocks..]
                                    .iter()
                                    .filter(|&&v| v != 0)
                                    .count()
                                    as u16;

                                // Compute shifted nzeros for prediction context
                                // Must use ceiling division to match decoder's shrc():
                                // (nzeros + covered_blocks - 1) >> log2(covered_blocks)
                                let log2_cb = covered_blocks.ilog2() as usize;
                                let shifted_nz = (pass_nz as usize + covered_blocks - 1) >> log2_cb;
                                let shifted_nz_u8 = shifted_nz.min(255) as u8;

                                // Store per-pass nzeros for neighbor prediction
                                for dy in 0..covered_y {
                                    for dx in 0..covered_x {
                                        pass_nzeros_grids[pass][c][by + dy][bx + dx] =
                                            shifted_nz_u8;
                                    }
                                }

                                // Predict nzeros from neighbors in this pass's grid
                                let local_bx = bx - start_bx;
                                let row_top = if by > start_by {
                                    Some(pass_nzeros_grids[pass][c][by - 1].as_slice())
                                } else {
                                    None
                                };
                                let predicted_nz = if local_bx == 0 {
                                    match row_top {
                                        Some(top) => top[bx] as i32,
                                        None => 32,
                                    }
                                } else {
                                    predict_from_top_and_left(
                                        row_top,
                                        &pass_nzeros_grids[pass][c][by],
                                        bx,
                                        32,
                                    )
                                };

                                // Tokenize this pass's coefficients
                                let pb = &mut pass_block_scratch[..size];
                                pb.copy_from_slice(pass_coeffs);
                                collect_ac_coefficients_into(
                                    &mut pass_tokens[pass],
                                    pb,
                                    raw_strategy,
                                    pass_nz,
                                    predicted_nz,
                                    block_ctx,
                                    block_ctx_map.num_ctxs,
                                    custom_ord,
                                );
                            }
                        }
                    }
                }
            }

            for (pass, tokens) in pass_tokens.into_iter().enumerate() {
                ac_section_tokens_per_pass[pass].push(tokens);
            }
        }

        // ── Apply LZ77 if enabled (ANS only, before building codes) ──

        let use_lz77 = self.enable_lz77 && self.use_ans;
        let mut dc_lz77_params: Option<crate::entropy_coding::lz77::Lz77Params> = None;
        let mut ac_lz77_params_per_pass: Vec<Option<crate::entropy_coding::lz77::Lz77Params>> =
            vec![None; num_passes];

        // Distance multiplier for special distance codes.
        // The decoder derives dist_multiplier = max(channel_widths) for each
        // modular subimage. The encoder must use the same multiplier so that
        // LZ77 distance symbols are interpreted correctly.
        //
        // DC subimage channels: 3 DC planes, each width = xsize_blocks
        // AC metadata subimage channels: EPF (w/64), CfL (w/64), BlockInfo (nb_blocks), QF (w/8)
        // AC VarDCT coefficients: not modular, decoder passes dist_multiplier=0
        let _dc_distance_multiplier = xsize_blocks as i32;
        let ac_distance_multiplier = 0i32;

        if use_lz77 {
            #[cfg(feature = "debug-tokens")]
            eprintln!(
                "[LZ77] Attempting LZ77 {:?} on DC ({} groups) and AC ({} groups)",
                self.lz77_method, num_dc_groups, num_groups
            );

            // Apply LZ77 to DC token streams (each DC group independently)
            // Use actual merged tree context count (WP DC + AC metadata), not old constant.
            let dc_num_ctx = total_contexts as usize;
            let merged_dc = {
                let mut m = Vec::new();
                for section in &dc_tokens_per_group {
                    m.extend_from_slice(section);
                }
                for section in &ac_metadata_tokens_per_group {
                    m.extend_from_slice(section);
                }
                m
            };
            #[cfg(feature = "debug-tokens")]
            eprintln!(
                "[LZ77] DC merged tokens: {}, num_contexts: {}",
                merged_dc.len(),
                dc_num_ctx
            );

            if let Some((lz77_tokens, params)) = crate::entropy_coding::lz77::apply_lz77(
                &merged_dc,
                dc_num_ctx,
                false,
                self.lz77_method,
                _dc_distance_multiplier,
            ) {
                #[cfg(feature = "debug-tokens")]
                eprintln!(
                    "[LZ77] DC LZ77 ACTIVATED: {} -> {} tokens",
                    merged_dc.len(),
                    lz77_tokens.len()
                );
                // Re-split LZ77 tokens back into per-group
                // For now, store merged LZ77 tokens and use single-group split
                dc_lz77_params = Some(params);
                // Replace per-group tokens with LZ77 versions
                // (apply per-group independently for correct splitting)
                let mut new_dc_per_group = Vec::with_capacity(num_dc_groups);
                let mut new_md_per_group = Vec::with_capacity(num_dc_groups);
                for i in 0..num_dc_groups {
                    // Compute per-group DC channel width for distance multiplier.
                    // DC subimage channels have width = group's block width.
                    let dc_gx = i % xsize_dc_groups;
                    let start_bx = dc_gx * DC_GROUP_DIM_IN_BLOCKS;
                    let end_bx = (start_bx + DC_GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
                    let group_dc_width = (end_bx - start_bx) as i32;

                    if let Some((lz77_dc, _)) = crate::entropy_coding::lz77::apply_lz77(
                        &dc_tokens_per_group[i],
                        dc_num_ctx,
                        false,
                        self.lz77_method,
                        group_dc_width,
                    ) {
                        new_dc_per_group.push(lz77_dc);
                    } else {
                        new_dc_per_group.push(dc_tokens_per_group[i].clone());
                    }

                    // AC metadata subimage has channels with different widths.
                    // Compute max(channel_widths) to match decoder's dist_multiplier.
                    let dc_gy = i / xsize_dc_groups;
                    let start_by = dc_gy * DC_GROUP_DIM_IN_BLOCKS;
                    let end_by = (start_by + DC_GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
                    let region_xblocks = end_bx - start_bx;
                    let mut num_ac_blocks = 0u32;
                    for ry in start_by..end_by {
                        for rx in start_bx..end_bx {
                            if ac_strategy.is_first(rx, ry) {
                                num_ac_blocks += 1;
                            }
                        }
                    }
                    // Metadata channels: EPF (w/8), CfL (w/8), BlockInfo (nb_blocks x 2), QF (bw x bh)
                    let epf_w = (region_xblocks * BLOCK_DIM).div_ceil(64) as u32;
                    let qf_w = region_xblocks as u32;
                    let md_dist_mult = epf_w.max(num_ac_blocks).max(qf_w) as i32;

                    if let Some((lz77_md, _)) = crate::entropy_coding::lz77::apply_lz77(
                        &ac_metadata_tokens_per_group[i],
                        dc_num_ctx,
                        false,
                        self.lz77_method,
                        md_dist_mult,
                    ) {
                        new_md_per_group.push(lz77_md);
                    } else {
                        new_md_per_group.push(ac_metadata_tokens_per_group[i].clone());
                    }
                }
                dc_tokens_per_group = new_dc_per_group;
                ac_metadata_tokens_per_group = new_md_per_group;
                let _ = lz77_tokens; // merged version not needed, per-group applied
            } else {
                #[cfg(feature = "debug-tokens")]
                eprintln!("[LZ77] DC LZ77 not beneficial (threshold not met)");
            }

            // Apply LZ77 to AC token streams per-pass (each pass independently)
            let ac_num_ctx = block_ctx_map.num_ac_contexts();
            for pass in 0..num_passes {
                let merged_ac = {
                    let mut m = Vec::new();
                    for section in &ac_section_tokens_per_pass[pass] {
                        m.extend_from_slice(section);
                    }
                    m
                };
                #[cfg(feature = "debug-tokens")]
                eprintln!(
                    "[LZ77] AC pass {} merged tokens: {}, num_contexts: {}",
                    pass,
                    merged_ac.len(),
                    ac_num_ctx
                );

                if let Some((_lz77_tokens, params)) = crate::entropy_coding::lz77::apply_lz77(
                    &merged_ac,
                    ac_num_ctx,
                    false,
                    self.lz77_method,
                    ac_distance_multiplier,
                ) {
                    #[cfg(feature = "debug-tokens")]
                    eprintln!(
                        "[LZ77] AC pass {} LZ77 ACTIVATED: {} -> {} tokens",
                        pass,
                        merged_ac.len(),
                        _lz77_tokens.len()
                    );
                    ac_lz77_params_per_pass[pass] = Some(params);
                    let mut new_sections = Vec::with_capacity(num_groups);
                    for tokens in &ac_section_tokens_per_pass[pass] {
                        if let Some((lz77_ac, _)) = crate::entropy_coding::lz77::apply_lz77(
                            tokens,
                            ac_num_ctx,
                            false,
                            self.lz77_method,
                            ac_distance_multiplier,
                        ) {
                            new_sections.push(lz77_ac);
                        } else {
                            new_sections.push(tokens.clone());
                        }
                    }
                    ac_section_tokens_per_pass[pass] = new_sections;
                } else {
                    #[cfg(feature = "debug-tokens")]
                    eprintln!(
                        "[LZ77] AC pass {} LZ77 not beneficial (threshold not met)",
                        pass
                    );
                }
            }
        }

        // ── Build optimal codes ──

        // Merge all DC section tokens (DC + AC metadata) for frequency counting
        // When using a learned DC tree, the number of contexts is:
        //   AC metadata contexts (0-10) + learned tree contexts (11+)
        // The decoder's MaConfig::parse reads Decoder::parse(ctx) where ctx is the number of tree leaves.
        // total_contexts from kWPFixedDC tree includes AC metadata (11) + DC tree contexts
        let base_dc_contexts = total_contexts as usize;
        let dc_num_contexts = if dc_lz77_params.is_some() {
            base_dc_contexts + 1 // +1 for LZ77 distance context
        } else {
            base_dc_contexts
        };
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
                dc_num_contexts,
                self.profile.enhanced_clustering_vardct,
                dc_lz77_params.as_ref(),
                None,
            ))
        } else {
            BuiltEntropyCode::Huffman(build_entropy_code_with_options(
                &all_dc_tokens,
                dc_num_contexts,
                self.profile.enhanced_clustering_vardct,
                dc_lz77_params.as_ref(),
            ))
        };

        // Build per-pass AC entropy codes
        let base_ac_num_contexts = block_ctx_map.num_ac_contexts();
        let mut ac_built_codes: Vec<BuiltEntropyCode> = Vec::with_capacity(num_passes);
        for pass in 0..num_passes {
            let ac_num_contexts = if ac_lz77_params_per_pass[pass].is_some() {
                base_ac_num_contexts + 1 // +1 for LZ77 distance context
            } else {
                base_ac_num_contexts
            };
            let total_ac_tokens: usize = ac_section_tokens_per_pass[pass]
                .iter()
                .map(|t| t.len())
                .sum();
            let mut all_ac_tokens = Vec::with_capacity(total_ac_tokens);
            for section in &ac_section_tokens_per_pass[pass] {
                all_ac_tokens.extend_from_slice(section);
            }

            let ac_built_code = if self.use_ans {
                BuiltEntropyCode::Ans(build_entropy_code_ans_with_options(
                    &all_ac_tokens,
                    ac_num_contexts,
                    self.profile.enhanced_clustering_vardct,
                    ac_lz77_params_per_pass[pass].as_ref(),
                    None,
                ))
            } else {
                BuiltEntropyCode::Huffman(build_entropy_code_with_options(
                    &all_ac_tokens,
                    ac_num_contexts,
                    self.profile.enhanced_clustering_vardct,
                    ac_lz77_params_per_pass[pass].as_ref(),
                ))
            };
            ac_built_codes.push(ac_built_code);
        }

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

        let has_alpha = alpha.is_some();
        let num_extra_channels = if has_alpha { 1 } else { 0 };

        // Write frame header
        {
            let mut fh = FrameHeader::lossy();
            fh.x_qm_scale = params.x_qm_scale;
            fh.b_qm_scale = params.b_qm_scale;
            fh.epf_iters = params.epf_iters;
            fh.gaborish = self.enable_gaborish;
            if noise_params.is_some() {
                fh.flags |= crate::headers::frame_header::ENABLE_NOISE;
            }
            if patches.is_some() {
                fh.flags |= crate::headers::frame_header::PATCHES_FLAG;
            }
            if splines.is_some() {
                fh.flags |= crate::headers::frame_header::SPLINES_FLAG;
            }
            if self.use_lf_frame {
                fh.flags |= crate::headers::frame_header::USE_LF_FRAME;
            }
            fh.ec_upsampling = vec![1; num_extra_channels];
            fh.ec_blend_modes = vec![BlendMode::Replace; num_extra_channels];

            // Progressive pass configuration
            if pass_config.is_progressive() {
                fh.num_passes = pass_config.num_passes;
                fh.pass_shifts = pass_config.shifts.clone();
                fh.num_ds = pass_config.num_ds;
                fh.ds_downsample = pass_config.ds_downsample.clone();
                fh.ds_last_pass = pass_config.ds_last_pass.clone();
            }

            // Apply animation frame options if provided
            if let Some(opts) = frame_options {
                fh.have_animation = opts.have_animation;
                fh.have_timecodes = opts.have_timecodes;
                fh.duration = opts.duration;
                fh.is_last = opts.is_last;
                if let Some(ref crop) = opts.crop {
                    fh.x0 = crop.x0;
                    fh.y0 = crop.y0;
                    fh.width = crop.width;
                    fh.height = crop.height;
                    fh.blend_mode = BlendMode::Replace;
                    fh.blend_source = 1;
                }
                // For animation, save non-last frames to reference slot 1
                // so crop frames can composite onto the previous canvas.
                if opts.have_animation && !opts.is_last {
                    fh.save_as_reference = 1;
                }
            }

            fh.write(writer)?;
        }

        let num_blocks = xsize_blocks * ysize_blocks;
        // Single combined section: only when 1 group AND 1 pass (non-progressive)
        if num_groups == 1 && num_dc_groups == 1 && num_passes == 1 {
            // Single-group: combine sections at the bit level
            let mut dc_global = BitWriter::with_capacity(4096);
            self.write_dc_global(
                params,
                num_dc_groups,
                &dc_built_code,
                noise_params,
                dc_lz77_params.as_ref(),
                &block_ctx_map,
                learned_tree_tokens.as_deref(),
                patches,
                splines,
                dc_quant_custom,
                &mut dc_global,
            )?;

            // Single-group alpha: all alpha data goes in the modular global sub-bitstream
            // within the DC global section, after the VarDCT DC entropy code.
            if let Some(alpha_data) = &alpha {
                Self::write_modular_alpha_global(alpha_data, width, height, &mut dc_global)?;
            }

            let mut dc_group = BitWriter::with_capacity(num_blocks * 10);
            self.write_dc_group_from_tokens(
                0,
                xsize_blocks,
                ysize_blocks,
                xsize_dc_groups,
                &dc_tokens_per_group[0],
                &ac_metadata_tokens_per_group[0],
                ac_strategy,
                &dc_built_code,
                dc_lz77_params.as_ref(),
                &mut dc_group,
            )?;

            let mut ac_global = BitWriter::with_capacity(4096);
            self.write_ac_global(
                num_groups,
                &ac_built_codes,
                used_orders,
                coeff_order_tokens.as_deref(),
                &ac_lz77_params_per_pass,
                &mut ac_global,
            )?;

            let mut ac_group_writer = BitWriter::with_capacity(num_blocks * 100);
            ac_built_codes[0].write_tokens(
                &ac_section_tokens_per_pass[0][0],
                ac_lz77_params_per_pass[0].as_ref(),
                &mut ac_group_writer,
            )?;

            let mut combined = dc_global;
            combined.append_unaligned(&dc_group)?;
            combined.append_unaligned(&ac_global)?;
            combined.append_unaligned(&ac_group_writer)?;
            combined.zero_pad_to_byte();
            let combined_bytes = combined.finish();

            write_toc(&[combined_bytes.len()], writer)?;
            writer.append_bytes(&combined_bytes)?;
        } else {
            // Multi-group: byte-aligned sections
            let mut sections: Vec<Vec<u8>> = Vec::with_capacity(num_sections);

            // DC Global
            let mut dc_global = BitWriter::with_capacity(4096);
            self.write_dc_global(
                params,
                num_dc_groups,
                &dc_built_code,
                noise_params,
                dc_lz77_params.as_ref(),
                &block_ctx_map,
                learned_tree_tokens.as_deref(),
                patches,
                splines,
                dc_quant_custom,
                &mut dc_global,
            )?;
            // Multi-group alpha: write empty modular global sub-bitstream.
            // Alpha channels are NOT meta_or_small for >256px images, so no data here.
            // The decoder still reads the GroupHeader + tree for the global section.
            if alpha.is_some() {
                Self::write_modular_empty_global(&mut dc_global)?;
            }
            dc_global.zero_pad_to_byte();
            sections.push(dc_global.finish());

            // DC groups
            let blocks_per_dc_group = (256 / 8) * (256 / 8);
            for dc_group_idx in 0..num_dc_groups {
                let mut dc_group = BitWriter::with_capacity(blocks_per_dc_group * 10);
                self.write_dc_group_from_tokens(
                    dc_group_idx,
                    xsize_blocks,
                    ysize_blocks,
                    xsize_dc_groups,
                    &dc_tokens_per_group[dc_group_idx],
                    &ac_metadata_tokens_per_group[dc_group_idx],
                    ac_strategy,
                    &dc_built_code,
                    dc_lz77_params.as_ref(),
                    &mut dc_group,
                )?;
                dc_group.zero_pad_to_byte();
                sections.push(dc_group.finish());
            }

            // AC Global (HfGlobal)
            let mut ac_global = BitWriter::with_capacity(4096);
            self.write_ac_global(
                num_groups,
                &ac_built_codes,
                used_orders,
                coeff_order_tokens.as_deref(),
                &ac_lz77_params_per_pass,
                &mut ac_global,
            )?;
            ac_global.zero_pad_to_byte();
            sections.push(ac_global.finish());

            // AC groups: Section order is pass-major, group-minor
            // Section index = 2 + num_dc_groups + pass * num_groups + group
            let blocks_per_ac_group = (256 / 8) * (256 / 8);
            for pass in 0..num_passes {
                for (group_idx, ac_tokens) in ac_section_tokens_per_pass[pass].iter().enumerate() {
                    let mut ac_group_writer = BitWriter::with_capacity(blocks_per_ac_group * 100);
                    ac_built_codes[pass].write_tokens(
                        ac_tokens,
                        ac_lz77_params_per_pass[pass].as_ref(),
                        &mut ac_group_writer,
                    )?;
                    // Multi-group alpha: write modular HF sub-bitstream only in LAST pass
                    if pass == num_passes - 1
                        && let Some(alpha_data) = &alpha
                    {
                        let group_x = group_idx % xsize_groups;
                        let group_y = group_idx / xsize_groups;
                        let x0 = group_x * GROUP_DIM;
                        let y0 = group_y * GROUP_DIM;
                        let gw = GROUP_DIM.min(width - x0);
                        let gh = GROUP_DIM.min(height - y0);
                        Self::write_modular_alpha_group(
                            alpha_data,
                            width,
                            x0,
                            y0,
                            gw,
                            gh,
                            &mut ac_group_writer,
                        )?;
                    }
                    ac_group_writer.zero_pad_to_byte();
                    sections.push(ac_group_writer.finish());
                }
            }

            let section_sizes: Vec<usize> = sections.iter().map(|s| s.len()).collect();

            #[cfg(feature = "debug-tokens")]
            {
                eprintln!(
                    "[SECTIONS] num_sections={}, num_passes={}, num_groups={}, num_dc_groups={}",
                    section_sizes.len(),
                    num_passes,
                    num_groups,
                    num_dc_groups
                );
                for (i, sz) in section_sizes.iter().enumerate() {
                    let label = if i == 0 {
                        "LfGlobal"
                    } else if i <= num_dc_groups {
                        "LfGroup"
                    } else if i == num_dc_groups + 1 {
                        "HfGlobal"
                    } else {
                        "HfGroup"
                    };
                    let pass_group = if i > num_dc_groups + 1 {
                        let idx = i - num_dc_groups - 2;
                        let pass = idx / num_groups;
                        let group = idx % num_groups;
                        format!(" (pass={}, group={})", pass, group)
                    } else {
                        String::new()
                    };
                    eprintln!("  section[{}]: {} = {} bytes{}", i, label, sz, pass_group);
                }
            }

            write_toc(&section_sizes, writer)?;
            for section in sections {
                writer.append_bytes(&section)?;
            }
        }

        Ok(())
    }

    /// Write DC group section from pre-collected tokens (two-pass mode).
    /// Write the modular global sub-bitstream for alpha in single-group VarDCT frames.
    ///
    /// For single-group images (≤256×256), the alpha channel is "meta_or_small" and
    /// goes entirely in the LfGlobal section. The decoder reads:
    ///   GroupHeader → (use_global_tree=0 → local tree) → entropy code → alpha pixels
    fn write_modular_alpha_global(
        alpha: &[u8],
        width: usize,
        height: usize,
        writer: &mut BitWriter,
    ) -> Result<()> {
        Self::write_modular_alpha_subbitstream(alpha, width, 0, 0, width, height, writer)
    }

    /// Write an empty modular global sub-bitstream for multi-group VarDCT frames with alpha.
    ///
    /// For multi-group images (>256×256), the alpha channel is NOT meta_or_small,
    /// so no alpha data belongs in the global section. The decoder reads the GroupHeader
    /// during `FullModularImage::read()`, then calls `decode_modular_subbitstream` with
    /// an empty buffer list (alpha assigned to HfGroups), which returns immediately.
    /// Only the GroupHeader is needed.
    fn write_modular_empty_global(writer: &mut BitWriter) -> Result<()> {
        // GroupHeader: use_global_tree=0, wp_params default=1, nb_transforms=0
        writer.write(1, 0)?; // use_global_tree = false
        writer.write(1, 1)?; // wp_params all_default = true
        writer.write(2, 0)?; // nb_transforms = 0
        Ok(())
    }

    /// Write a modular sub-bitstream for alpha data in a region (used by both global and HF groups).
    ///
    /// Format: GroupHeader → local tree → LZ77 header → entropy code → alpha residuals
    ///
    /// Uses gradient prediction (predictor 5) with LZ77 RLE for efficient encoding of
    /// mostly-uniform alpha channels (e.g. fully opaque screenshots). Each sub-bitstream
    /// is independent (fresh decoder state).
    fn write_modular_alpha_subbitstream(
        alpha: &[u8],
        stride: usize,
        x0: usize,
        y0: usize,
        region_width: usize,
        region_height: usize,
        writer: &mut BitWriter,
    ) -> Result<()> {
        use crate::modular::encode::{
            K_LZ77_MIN_LENGTH, K_LZ77_MIN_SYMBOL, Token, build_sparse_histogram,
            encode_hybrid_uint_000, encode_hybrid_uint_lz77_length, write_gradient_tree_tokens,
            write_hybrid_data_histogram, write_sparse_lz77_histogram,
            write_tree_histogram_for_gradient,
        };
        use crate::modular::predictor::pack_signed;

        // GroupHeader: use_global_tree=0, wp default, no transforms
        writer.write(1, 0)?; // use_global_tree = false
        writer.write(1, 1)?; // wp_params all_default = true
        writer.write(2, 0)?; // nb_transforms = 0

        // Local tree: gradient prediction, single context
        let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
        write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

        // Collect residuals with LZ77 RLE detection
        let mut tokens = Vec::new();
        let mut current_run = 0usize;
        let mut num_decoded = 0usize;
        let mut last_value = u32::MAX; // impossible initial value prevents LZ77 from first pixel

        for y in 0..region_height {
            for x in 0..region_width {
                let pixel = alpha[(y0 + y) * stride + (x0 + x)] as i32;

                let left = if x > 0 {
                    alpha[(y0 + y) * stride + (x0 + x - 1)] as i32
                } else if y > 0 {
                    alpha[(y0 + y - 1) * stride + x0] as i32
                } else {
                    0
                };
                let top = if y > 0 {
                    alpha[(y0 + y - 1) * stride + (x0 + x)] as i32
                } else {
                    left
                };
                let topleft = if x > 0 && y > 0 {
                    alpha[(y0 + y - 1) * stride + (x0 + x - 1)] as i32
                } else {
                    left
                };

                // ClampedGradient prediction
                let grad = left + top - topleft;
                let prediction = grad.clamp(left.min(top), left.max(top));
                let residual = pixel - prediction;
                let packed = pack_signed(residual);

                // LZ77 RLE: copies the last residual value
                let can_use_lz77 = num_decoded > 0 && packed == last_value;

                if can_use_lz77 {
                    current_run += 1;
                } else {
                    // Flush accumulated run
                    if current_run > K_LZ77_MIN_LENGTH {
                        tokens.push(Token::Lz77Run(current_run));
                        num_decoded += current_run;
                    } else {
                        for _ in 0..current_run {
                            tokens.push(Token::Raw(last_value));
                            num_decoded += 1;
                        }
                    }
                    current_run = 0;
                    tokens.push(Token::Raw(packed));
                    num_decoded += 1;
                    last_value = packed;
                }
            }
        }

        // Flush final run
        if current_run > K_LZ77_MIN_LENGTH {
            tokens.push(Token::Lz77Run(current_run));
        } else {
            for _ in 0..current_run {
                tokens.push(Token::Raw(last_value));
            }
        }

        // Check if we have LZ77 runs
        let num_lz77_runs = tokens
            .iter()
            .filter(|t| matches!(t, Token::Lz77Run(_)))
            .count();

        if num_lz77_runs > 0 {
            // LZ77-enabled path: sparse alphabet with LZ77 symbols
            let sparse_counts = build_sparse_histogram(&tokens);
            let (depths, codes) = write_sparse_lz77_histogram(writer, &sparse_counts)?;

            // Encode tokens
            for token in &tokens {
                match token {
                    Token::Raw(value) => {
                        let (tok, nbits, extra) = encode_hybrid_uint_000(*value);
                        let symbol = tok as usize;
                        if depths[symbol] > 0 {
                            writer.write(depths[symbol] as usize, codes[symbol] as u64)?;
                        }
                        if nbits > 0 {
                            writer.write(nbits as usize, extra as u64)?;
                        }
                    }
                    Token::Lz77Run(count) => {
                        let adjusted = count - K_LZ77_MIN_LENGTH;
                        let (tok, nbits, extra) = encode_hybrid_uint_lz77_length(adjusted as u32);
                        let symbol = K_LZ77_MIN_SYMBOL + tok as usize;
                        if depths[symbol] > 0 {
                            writer.write(depths[symbol] as usize, codes[symbol] as u64)?;
                        }
                        if nbits > 0 {
                            writer.write(nbits as usize, extra as u64)?;
                        }
                        // Distance symbol for distance=1 (RLE):
                        // SPECIAL_DISTANCES[1] = (1, 0) → distance = dist_multiplier*0 + 1 = 1
                        let (dist_tok, dist_nbits, dist_extra) = encode_hybrid_uint_000(1);
                        if depths[dist_tok as usize] > 0 {
                            writer.write(
                                depths[dist_tok as usize] as usize,
                                codes[dist_tok as usize] as u64,
                            )?;
                        }
                        if dist_nbits > 0 {
                            writer.write(dist_nbits as usize, dist_extra as u64)?;
                        }
                    }
                }
            }
        } else {
            // No LZ77 runs: use the simpler non-LZ77 path with HybridUint {4,2,0}
            use crate::entropy_coding::hybrid_uint::HybridUintConfig;
            let hybrid_config = HybridUintConfig {
                split_exponent: 4,
                split: 16,
                msb_in_token: 2,
                lsb_in_token: 0,
            };

            let mut max_token: u32 = 0;
            let mut histogram_data = Vec::with_capacity(tokens.len());
            for token in &tokens {
                if let Token::Raw(value) = token {
                    let (tok, extra_bits, num_extra) = hybrid_config.encode(*value);
                    max_token = max_token.max(tok);
                    histogram_data.push((tok, extra_bits, num_extra));
                }
            }

            let histogram_size = (max_token + 1) as usize;
            let mut histogram = vec![0u32; histogram_size];
            for &(tok, _, _) in &histogram_data {
                histogram[tok as usize] += 1;
            }

            let (depths, codes) = write_hybrid_data_histogram(writer, &histogram, max_token)?;

            for &(token, extra_bits, num_extra) in &histogram_data {
                let depth = depths[token as usize];
                let code = codes[token as usize];
                writer.write(depth as usize, code as u64)?;
                if num_extra > 0 {
                    writer.write(num_extra as usize, extra_bits as u64)?;
                }
            }
        }

        Ok(())
    }

    /// Write a modular HF group sub-bitstream for alpha in multi-group VarDCT frames.
    ///
    /// Each HF group gets its own independent modular sub-bitstream with a fresh
    /// GroupHeader, local tree, and entropy code.
    fn write_modular_alpha_group(
        alpha: &[u8],
        stride: usize,
        x0: usize,
        y0: usize,
        region_width: usize,
        region_height: usize,
        writer: &mut BitWriter,
    ) -> Result<()> {
        Self::write_modular_alpha_subbitstream(
            alpha,
            stride,
            x0,
            y0,
            region_width,
            region_height,
            writer,
        )
    }

    /// Writes the DC group header, DC tokens, AC metadata sub-header, then AC
    /// metadata tokens — matching the exact bitstream layout of `write_dc_group`.
    ///
    /// When `use_lf_frame` is true, DC tokens are empty and the DC modular
    /// sub-bitstream (extra_dc_precision + header + tokens) is skipped entirely.
    /// Only AC metadata (HF metadata) is written.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_dc_group_from_tokens(
        &self,
        dc_group_idx: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        xsize_dc_groups: usize,
        dc_tokens: &[Token],
        ac_metadata_tokens: &[Token],
        ac_strategy: &AcStrategyMap,
        dc_code: &BuiltEntropyCode,
        dc_lz77_params: Option<&crate::entropy_coding::lz77::Lz77Params>,
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

        // When use_lf_frame (dc_tokens empty), skip the VarDCT DC modular sub-bitstream.
        // The decoder skips decode_vardct_lf() when has_lf_frame() is true.
        if !self.use_lf_frame {
            // DC group header
            writer.write(2, 0)?; // extra_dc_precision = 0
            writer.write(4, 3)?; // use global tree, default wp, no transforms

            // Write DC tokens
            dc_code.write_tokens(dc_tokens, dc_lz77_params, writer)?;
        }

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
        dc_code.write_tokens(ac_metadata_tokens, dc_lz77_params, writer)?;

        Ok(())
    }

    /// Write entropy code (context map + codes/distributions).
    pub(crate) fn write_entropy_code_header(
        &self,
        code: &BuiltEntropyCode,
        writer: &mut BitWriter,
    ) -> Result<()> {
        code.write_header(writer)
    }
}
