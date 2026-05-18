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
    build_entropy_code_ans_from_token_groups, build_entropy_code_from_token_groups,
};
use crate::entropy_coding::token::Token;
use crate::error::Result;
use crate::headers::color_encoding::{ColorEncoding, ColorSpace, RenderingIntent};
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

// ── Extracted per-group functions for parallel dispatch ──

/// Tokenize a single DC group (LfFrame mode: AC metadata only, no DC tokens).
#[allow(clippy::too_many_arguments)]
fn tokenize_dc_group_lf_frame(
    dc_group_idx: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    xsize_dc_groups: usize,
    quant_field: &[u8],
    cfl_map: &CflMap,
    ac_strategy: &AcStrategyMap,
    sharpness_map: Option<&[u8]>,
    ac_meta_ctx_map: &[u32],
) -> (Vec<Token>, Vec<Token>) {
    let dc_gx = dc_group_idx % xsize_dc_groups;
    let dc_gy = dc_group_idx / xsize_dc_groups;
    let start_bx = dc_gx * DC_GROUP_DIM_IN_BLOCKS;
    let start_by = dc_gy * DC_GROUP_DIM_IN_BLOCKS;
    let end_bx = (start_bx + DC_GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
    let end_by = (start_by + DC_GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
    let region_xsize = end_bx - start_bx;
    let region_ysize = end_by - start_by;

    let dc_tokens = Vec::new(); // no DC tokens in LfFrame mode

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
            t.set_context(ac_meta_ctx_map[t.context() as usize]);
            t
        })
        .collect();

    (dc_tokens, md_tokens)
}

/// Tokenize a single DC group (WP DC mode: both DC and AC metadata tokens).
#[allow(clippy::too_many_arguments)]
fn tokenize_dc_group_wp(
    dc_group_idx: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    xsize_dc_groups: usize,
    quant_dc: &[Vec<Vec<i16>>; 3],
    quant_field: &[u8],
    cfl_map: &CflMap,
    ac_strategy: &AcStrategyMap,
    sharpness_map: Option<&[u8]>,
    wp_dc_tree: &super::dc_tree_learn::DcTree,
    dc_ctx_remap: &[u32],
    ac_meta_ctx_map: &[u32],
) -> (Vec<Token>, Vec<Token>) {
    let dc_gx = dc_group_idx % xsize_dc_groups;
    let dc_gy = dc_group_idx / xsize_dc_groups;
    let start_bx = dc_gx * DC_GROUP_DIM_IN_BLOCKS;
    let start_by = dc_gy * DC_GROUP_DIM_IN_BLOCKS;
    let end_bx = (start_bx + DC_GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
    let end_by = (start_by + DC_GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
    let region_xsize = end_bx - start_bx;
    let region_ysize = end_by - start_by;

    // Collect DC tokens using Weighted Predictor + kWPFixedDC tree
    let dc_tokens = collect_dc_tokens_wp(quant_dc, wp_dc_tree, start_bx, start_by, end_bx, end_by);
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
            t.set_context(dc_ctx_remap[t.context() as usize]);
            t
        })
        .collect();

    let md_tokens: Vec<Token> = md_tokens
        .into_iter()
        .map(|mut t| {
            t.set_context(ac_meta_ctx_map[t.context() as usize]);
            t
        })
        .collect();

    (dc_tokens, md_tokens)
}

/// Tokenize a single AC group, returning per-pass token Vecs.
///
/// Scratch buffers are allocated locally (per-call, not shared).
/// For progressive mode, a local nzeros grid covering only this group's blocks
/// is allocated and used for neighbor prediction.
#[allow(clippy::too_many_arguments)]
fn tokenize_ac_group(
    group_idx: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    xsize_groups: usize,
    quant_ac: &[Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
    nzeros: &[Vec<Vec<u8>>; 3],
    raw_nzeros: &[Vec<Vec<u16>>; 3],
    quant_field: &[u8],
    ac_strategy: &AcStrategyMap,
    block_ctx_map: &BlockCtxMap,
    custom_order_map: Option<&[Vec<Vec<u32>>]>,
    used_orders: u32,
    pass_config: &ProgressivePassConfig,
) -> Vec<Vec<Token>> {
    let group_x = group_idx % xsize_groups;
    let group_y = group_idx / xsize_groups;
    let start_bx = group_x * GROUP_DIM_IN_BLOCKS;
    let start_by = group_y * GROUP_DIM_IN_BLOCKS;
    let end_bx = (start_bx + GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
    let end_by = (start_by + GROUP_DIM_IN_BLOCKS).min(ysize_blocks);

    let region_blocks = (end_bx - start_bx) * (end_by - start_by);
    let num_passes = pass_config.num_passes as usize;

    // Per-call scratch buffers
    const MAX_BLOCK_SIZE: usize = 4096;
    let mut full_block_scratch = vec![0i32; MAX_BLOCK_SIZE];
    let mut pass_block_scratch = vec![0i32; MAX_BLOCK_SIZE];

    // Initialize per-pass token vecs for this group
    let mut pass_tokens: Vec<Vec<Token>> = (0..num_passes)
        .map(|_| Vec::with_capacity(region_blocks * 64 * 3 / num_passes))
        .collect();

    // For progressive encoding, allocate per-group local nzeros grids.
    // These cover only this group's block region, indexed by absolute coords.
    let mut pass_nzeros_grids: Vec<[Vec<Vec<u8>>; 3]> = if pass_config.is_progressive() {
        (0..num_passes)
            .map(|_| core::array::from_fn(|_| vec![vec![0u8; xsize_blocks]; ysize_blocks]))
            .collect()
    } else {
        Vec::new()
    };

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
                let custom_ord = custom_order_map.and_then(|orders| {
                    super::coeff_order::get_custom_order(orders, used_orders, strategy_code, c)
                });

                let qf_val = quant_field[by * xsize_blocks + bx] as u32;
                let block_ctx = block_ctx_map.block_context(c, strategy_code, qf_val);

                // Assemble the full coefficient block
                let full_block: &[i32] = if covered_blocks == 1 {
                    &quant_ac[c][by][bx]
                } else {
                    let (cx, cy) = if covered_y > covered_x {
                        (covered_y, covered_x)
                    } else {
                        (covered_x, covered_y)
                    };
                    let transpose_slots = covered_y > covered_x;
                    let stride = cx * BLOCK_DIM;
                    let fb = &mut full_block_scratch[..size];
                    // Nested loops eliminate per-element integer divisions
                    // (y/stride, x%stride, y/BLOCK_DIM, x/BLOCK_DIM, y%BLOCK_DIM, x%BLOCK_DIM)
                    for coef_slot_y in 0..cy {
                        for pos_y in 0..BLOCK_DIM {
                            let y = coef_slot_y * BLOCK_DIM + pos_y;
                            for coef_slot_x in 0..cx {
                                let (phys_row_off, phys_col_off) = if transpose_slots {
                                    (coef_slot_x, coef_slot_y)
                                } else {
                                    (coef_slot_y, coef_slot_x)
                                };
                                let row = &quant_ac[c][by + phys_row_off][bx + phys_col_off];
                                for pos_x in 0..BLOCK_DIM {
                                    let x = coef_slot_x * BLOCK_DIM + pos_x;
                                    fb[y * stride + x] = row[pos_y * BLOCK_DIM + pos_x];
                                }
                            }
                        }
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
                    let pass_blocks = split_coefficients_into_passes(full_block, pass_config);

                    for (pass, pass_coeffs) in pass_blocks.iter().enumerate() {
                        // Count non-zeros for this pass's coefficients
                        // (skip covered_blocks positions = LLF coefficients)
                        let pass_nz: u16 = pass_coeffs[covered_blocks..]
                            .iter()
                            .filter(|&&v| v != 0)
                            .count() as u16;

                        // Compute shifted nzeros for prediction context
                        let log2_cb = covered_blocks.ilog2() as usize;
                        let shifted_nz = (pass_nz as usize + covered_blocks - 1) >> log2_cb;
                        let shifted_nz_u8 = shifted_nz.min(255) as u8;

                        // Store per-pass nzeros for neighbor prediction
                        for dy in 0..covered_y {
                            for dx in 0..covered_x {
                                pass_nzeros_grids[pass][c][by + dy][bx + dx] = shifted_nz_u8;
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

    pass_tokens
}

/// Encode a single DC group section to bytes.
#[allow(clippy::too_many_arguments)]
fn encode_dc_group_section(
    enc: &VarDctEncoder,
    dc_group_idx: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    xsize_dc_groups: usize,
    dc_tokens: &[Token],
    ac_metadata_tokens: &[Token],
    ac_strategy: &AcStrategyMap,
    dc_built_code: &BuiltEntropyCode<'_>,
    dc_lz77_params: Option<&crate::entropy_coding::lz77::Lz77Params>,
) -> Result<Vec<u8>> {
    let blocks_per_dc_group = (256 / 8) * (256 / 8);
    let mut dc_group = BitWriter::with_capacity(blocks_per_dc_group * 10);
    enc.write_dc_group_from_tokens(
        dc_group_idx,
        xsize_blocks,
        ysize_blocks,
        xsize_dc_groups,
        dc_tokens,
        ac_metadata_tokens,
        ac_strategy,
        dc_built_code,
        dc_lz77_params,
        &mut dc_group,
    )?;
    dc_group.zero_pad_to_byte();
    Ok(dc_group.finish())
}

/// Encode a single AC group section to bytes (for a specific pass).
#[allow(clippy::too_many_arguments)]
fn encode_ac_group_section(
    ac_tokens: &[Token],
    ac_built_code: &BuiltEntropyCode<'_>,
    ac_lz77_params: Option<&crate::entropy_coding::lz77::Lz77Params>,
    // Extra channels (alpha + any non-alpha extras), travel as one
    // modular sub-bitstream per HF group in the LAST pass.
    extras: &[super::extras::VardctExtra<'_>],
    is_last_pass: bool,
    group_idx: usize,
    xsize_groups: usize,
    width: usize,
    height: usize,
    // Per-channel lossy quantizers (all-1 = lossless byte-identical
    // path). Length must equal `extras.len()` when non-empty; the
    // empty-extras path ignores this slice.
    extras_quantizers: &[u32],
) -> Result<Vec<u8>> {
    let blocks_per_ac_group = (256 / 8) * (256 / 8);
    let mut ac_group_writer = BitWriter::with_capacity(blocks_per_ac_group * 100);
    ac_built_code.write_tokens(ac_tokens, ac_lz77_params, &mut ac_group_writer)?;
    // Multi-group extras: write modular HF sub-bitstream only in LAST pass.
    if is_last_pass && !extras.is_empty() {
        let group_x = group_idx % xsize_groups;
        let group_y = group_idx / xsize_groups;
        let x0 = group_x * GROUP_DIM;
        let y0 = group_y * GROUP_DIM;
        let gw = GROUP_DIM.min(width - x0);
        let gh = GROUP_DIM.min(height - y0);
        VarDctEncoder::write_modular_extras_group_with_quant(
            extras,
            width,
            height,
            x0,
            y0,
            gw,
            gh,
            extras_quantizers,
            &mut ac_group_writer,
        )?;
    }
    ac_group_writer.zero_pad_to_byte();
    Ok(ac_group_writer.finish())
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
        extras_info: &[ExtraChannelInfo],
    ) -> FileHeader {
        let mut bit_depth = if self.bit_depth_16 {
            BitDepth::uint16()
        } else {
            BitDepth::uint8()
        };
        // Optional bits_per_sample override (#18 sub-feature). Keeps
        // float_sample / exponent_bits from the int default; only the
        // bits_per_sample field is replaced so callers can signal
        // narrower-precision integer input (10/12/14-bit).
        if let Some(bits) = self.bits_per_sample_override {
            bit_depth.bits_per_sample = bits;
        }

        let mut color_encoding = if let Some(ce) = self.color_encoding.clone() {
            // Explicit color encoding overrides source_gamma and defaults.
            if self.is_grayscale && ce.color_space != ColorSpace::Gray {
                ColorEncoding {
                    color_space: ColorSpace::Gray,
                    ..ce
                }
            } else {
                ce
            }
        } else if self.is_grayscale {
            if let Some(gamma) = self.source_gamma {
                ColorEncoding::gray_with_gamma(gamma)
            } else {
                ColorEncoding::gray()
            }
        } else if let Some(gamma) = self.source_gamma {
            ColorEncoding::with_gamma(gamma)
        } else {
            ColorEncoding::srgb()
        };
        // VarDCT uses Relative rendering intent (matches libjxl)
        color_encoding.rendering_intent = RenderingIntent::Relative;
        if self.icc_profile.is_some() {
            color_encoding.want_icc = true;
        }

        // Build the extra-channel list for the file header. When the
        // caller passes an alpha-typed extra we honour
        // `self.alpha_associated`; the rest are written through
        // exactly as the caller built them.
        let mut extra_channels: Vec<ExtraChannelInfo> = Vec::with_capacity(extras_info.len());
        for ec in extras_info {
            let mut ec = ec.clone();
            if ec.ec_type == crate::headers::extra_channels::ExtraChannelType::Alpha {
                ec.alpha_associated = self.alpha_associated;
            }
            extra_channels.push(ec);
        }

        // When upsampling > 1 (refs #12), the caller passed downsampled
        // (width, height); the file-header advertises the original
        // pre-downsample size that the decoder will produce after
        // applying the frame-header `upsampling` factor.
        let display_width = (width as u32).saturating_mul(self.upsampling);
        let display_height = (height as u32).saturating_mul(self.upsampling);
        FileHeader {
            width: display_width,
            height: display_height,
            metadata: ImageMetadata {
                bit_depth,
                color_encoding,
                extra_channels,
                xyb_encoded: true, // Required for VarDCT
                intensity_target: self.intensity_target,
                min_nits: self.min_nits,
                have_intrinsic_size: self.intrinsic_size.is_some(),
                intrinsic_width: self.intrinsic_size.map_or(0, |(w, _)| w),
                intrinsic_height: self.intrinsic_size.map_or(0, |(_, h)| h),
                ..ImageMetadata::default()
            },
            // Plumb the caller-selected upsampling mode + factor so
            // `FileHeader::write` can emit the matching `custom_weight`
            // LUT (libjxl `JxlEncoderSetUpsamplingMode`). Only takes
            // effect when `upsampling_factor > 1` AND a non-None /
            // non-(-1) mode was supplied.
            upsampling_mode: self.upsampling_mode,
            upsampling_factor: self.upsampling,
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
        extras_info: &[ExtraChannelInfo],
        writer: &mut BitWriter,
    ) -> Result<()> {
        let file_header = self.build_file_header(width, height, extras_info);
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
                        let (cx, cy) = if covered_y > covered_x {
                            (covered_y, covered_x)
                        } else {
                            (covered_x, covered_y)
                        };
                        let transpose_slots = covered_y > covered_x;
                        let stride = cx * BLOCK_DIM;
                        let full_block = &mut full_block_scratch[..size];
                        // Nested loops eliminate per-element integer divisions
                        for coef_slot_y in 0..cy {
                            for pos_y in 0..BLOCK_DIM {
                                let y = coef_slot_y * BLOCK_DIM + pos_y;
                                for coef_slot_x in 0..cx {
                                    let (phys_row_off, phys_col_off) = if transpose_slots {
                                        (coef_slot_x, coef_slot_y)
                                    } else {
                                        (coef_slot_y, coef_slot_x)
                                    };
                                    let row = &quant_ac[c][by + phys_row_off][bx + phys_col_off];
                                    for pos_x in 0..BLOCK_DIM {
                                        let x = coef_slot_x * BLOCK_DIM + pos_x;
                                        full_block[y * stride + x] = row[pos_y * BLOCK_DIM + pos_x];
                                    }
                                }
                            }
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
        extras: &[super::extras::VardctExtra<'_>],
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

        // Validate linear-RGB at intake. Mirrors the still-image entry point at
        // `vardct/encoder.rs:620-664`. The `forward_xyb` SIMD kernel uses
        // `mixed.max(0.0)` per channel, which silently coerces NaN to `0.0`
        // (IEEE-754 ordered max returns the non-NaN operand). That means a
        // caller-supplied NaN linear-RGB never reaches the XYB output. To
        // surface caller bugs (Error mode) or actively scrub them (Sanitize
        // mode) we must check / fix here, before forward_xyb runs.
        //
        // Without this scrub, float-input animation frames (RgbLinearF32,
        // RgbaLinearF32, GrayLinearF32, GrayAlphaLinearF32, …) carrying NaN /
        // ±Inf would silently encode wrong pixels — a divergence from the
        // still-image path where the same input would either error or sanitize.
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
                    let mut owned: alloc::vec::Vec<f32> = linear_rgb.to_vec();
                    let _ = jxl_simd::sanitize_finite(&mut owned);
                    Some(owned)
                }
            };
        let linear_rgb: &[f32] = sanitized_linear_rgb_storage
            .as_deref()
            .unwrap_or(linear_rgb);

        let (mut xyb_x, mut xyb_y, mut xyb_b) =
            self.convert_to_xyb_padded(width, height, padded_width, padded_height, linear_rgb)?;

        // Defense-in-depth XYB scan. Mirrors `vardct/encoder.rs:675`. Catches
        // downstream-bug non-finite values that should never appear on the
        // encode-fresh path because forward_xyb is finite-output-for-finite-input.
        super::encoder::validate_xyb_planes(
            self.non_finite_action,
            &mut xyb_x,
            &mut xyb_y,
            &mut xyb_b,
        )?;

        // Noise parameters. Four sources, in priority order — mirrors the
        // still-image entry point at `vardct/encoder.rs:677-737` (libjxl
        // `enc_frame.cc:680-689`). The animation path was previously only
        // wired for `enable_noise`, silently dropping `photon_noise_iso`
        // (caller-supplied ISO grain) and `manual_noise_lut` (caller-supplied
        // 8-point LUT) — divergence from the still-image path.
        // 1. `photon_noise_iso`: caller-supplied ISO value, bypasses
        //    content estimation. Matches libjxl --photon_noise.
        // 2. `manual_noise_lut`: caller-supplied 8-point LUT. Bypasses
        //    everything else.
        // 3. `enable_noise` + content estimation: scan flat patches,
        //    fit an 8-point LUT via SCG optimisation. Optional Wiener
        //    denoise pre-filter follows.
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

        // Detect and subtract patches on PRE-gaborish XYB. Mirrors the
        // still-image path at `vardct/encoder.rs:739-771`. Patches work in
        // the XYB domain: detect repeated rectangular elements, store unique
        // patterns in a reference frame, subtract from the image. Without
        // this, animation frames carrying screenshot/UI content (text glyphs,
        // icons, repeated buttons) emit the same template once per occurrence
        // — for typical UI-heavy GIF/APNG-style content this is a 30-50%
        // size penalty vs. the still-image path.
        //
        // Detection runs on PRE-gaborish XYB to match libjxl: the decoder
        // pipeline is IDCT → gaborish → EPF → patches add-back, so the
        // encoder must subtract patterns from the same XYB the decoder will
        // reconstruct (see `~/work/jxl-efforts/libjxl/lib/jxl/dec_cache.cc`
        // line 148-194 and the gpu encoder finding documented in
        // `MEMORY.md::screenshot_patches_landed_2026-05-15`).
        //
        // Cost-benefit gate is the same as the still-image path: only
        // applied in `EncoderMode::Experimental`. Reference mode follows
        // libjxl and uses patches unconditionally when detected.
        //
        // Distance-aware kMinPeak (W3-1 / commit 4fb0f52): libjxl
        // parity (=2) below d=1.0, W2-5 chunk 1 relaxation (=1) at
        // d>=1.0. See `vardct/encoder.rs::encode_inner` for why
        // RFC#45 chunk 3's per-patch gate does NOT lower this.
        let min_peak = if self.distance < 1.0 { 2 } else { 1 };
        // RFC#45 pick #5 chunk 3 per-patch cost gate — mirrors
        // `vardct/encoder.rs::encode_inner` (see comment there).
        let mut patches_data = if self.enable_patches {
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
        if matches!(self.encoder_mode, crate::api::EncoderMode::Experimental)
            && let Some(ref pd) = patches_data
            && !pd.is_cost_effective(self.distance, self.use_ans)
        {
            patches_data = None;
        }
        // Quantize ref_image so subtract/add use the same values the decoder
        // will reconstruct.
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

        // Compute adaptive per-block quantization field and masking on
        // PRE-gaborish XYB. libjxl computes InitialQuantField BEFORE
        // GaborishInverse (`enc_heuristics.cc:1117-1142`, comment:
        // "relies on pre-gaborish values"). Gaborish sharpening inflates
        // gradients which inflates masking → smaller quant values →
        // finer quantization → more bits.
        //
        // When gaborish is off, scale distance by 0.62 for the quant
        // field (matches libjxl `enc_heuristics.cc:1119`).
        //
        // Mirror of still-image ordering at vardct/encoder.rs:866 and
        // vardct/precomputed.rs:360. Previously this path applied
        // gaborish_inverse BEFORE compute_quant_field_float_with_budget,
        // matching neither libjxl nor our other still-image entry points.
        let distance_for_iqf = if self.enable_gaborish {
            self.distance
        } else {
            self.distance * 0.62
        };

        let (mut quant_field_float, masking) =
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
            )?;

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
            params.apply_chromacity_adjustment(chromacity_x, chromacity_b);
        }

        let mut quant_field =
            super::adaptive_quant::quantize_quant_field(&quant_field_float, params.inv_scale);

        // Compute per-pixel mask for pixel-domain loss on PRE-gaborish XYB
        // (matches libjxl `InitialQuantField` which produces
        // `initial_quant_masking1x1` before `GaborishInverse`).
        let mask1x1 = if self.ac_strategy_enabled && self.pixel_domain_loss {
            Some(super::adaptive_quant::compute_mask1x1_with_budget(
                &xyb_y,
                padded_width,
                padded_height,
                self.budget.as_ref(),
            )?)
        } else {
            None
        };

        // Apply gaborish inverse (5x5 sharpening) AFTER quant_field /
        // mask1x1 (computed on pre-gaborish XYB), but BEFORE CfL and AC
        // strategy. This matches libjxl `enc_heuristics.cc:1117-1142`
        // and the still-image paths at vardct/encoder.rs:973 and
        // vardct/precomputed.rs:403.
        if self.enable_gaborish {
            super::gaborish::gaborish_inverse(
                &mut xyb_x,
                &mut xyb_y,
                &mut xyb_b,
                padded_width,
                padded_height,
                self.budget.as_ref(),
            )?;
        }

        let mut cfl_map = if self.cfl_enabled {
            super::chroma_from_luma::compute_cfl_map(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                padded_width,
                padded_height,
                xsize_blocks,
                ysize_blocks,
                self.profile.cfl_newton,
                self.profile.cfl_newton_eps,
                self.profile.cfl_newton_max_iters,
            )
        } else {
            CflMap::zeros(
                div_ceil(xsize_blocks, TILE_DIM_IN_BLOCKS),
                div_ceil(ysize_blocks, TILE_DIM_IN_BLOCKS),
            )
        };

        // `ac_strategy` may be refined by the zensim loop below, which
        // splits large transforms with high perceptual error. The `mut`
        // is unused when the `zensim-loop` cargo feature is disabled.
        #[cfg_attr(not(feature = "zensim-loop"), allow(unused_mut))]
        let mut ac_strategy = if let Some(forced) = self.force_strategy {
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

        // CfL pass 2: recompute CfL map using actual AC strategies and per-block
        // quantization weighting. Uses the same FindBestMultiplier as pass 1 but
        // with strategy-specific DCTs and quant-weighted coefficients.
        // Gated at effort >= 7 (speed_tier <= kSquirrel) matching libjxl.
        //
        // ORDERING: CfL pass 2 must run BEFORE the butteraugli loop so the
        // loop's internal recon and the shipped bitstream both see the same
        // post-pass-2 cfl_map. Mirrors libjxl `enc_heuristics.cc:1190-1193`
        // (CfL2) → `:1250-1252` (FindBestQuantizer/buttloop) and the
        // still-image fix in commit d5e55c8a (drift investigation chunk-3,
        // 2026-05-15). Pre-fix the animation path skipped pass-2 entirely
        // even when `profile.cfl_two_pass` was true, so the shipped cfl_map
        // diverged from the still-image path's by the same per-block
        // refinement libjxl applies in `enc_heuristics.cc::CfL2`.
        if self.profile.cfl_two_pass && self.cfl_enabled {
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
                self.profile.cfl_newton,
                self.profile.cfl_newton_eps,
                self.profile.cfl_newton_max_iters,
            );
        }

        // Quantization loops: iteratively refine quant_field using perceptual
        // distance feedback. Butteraugli, ssim2 and zensim loops can stack:
        // butteraugli handles global convergence, ssim2 adds SSIM-aware spatial
        // fine-tuning, zensim refines AC strategy by splitting large transforms
        // with high perceptual error. Mirrors the still-image path at
        // `vardct/encoder.rs:1178-1259`. Pre-fix the animation path only honoured
        // `butteraugli_iters` — `ssim2_iters` and `zensim_iters` were silently
        // dropped. All three are wired through `encode_animation_lossy` to
        // `enc.{butteraugli,ssim2,zensim}_iters` (api.rs:6065-6076) so the
        // bug was a divergence between the wiring layer and the encoder body.
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
                patches_data.as_ref(),
                None, // No splines in animation frames
            )?;
        }

        // SSIM2 quantization loop: alternative to butteraugli using SSIM2 +
        // per-block RMSE. Same structure as butteraugli loop but faster per
        // iteration. Mirrors `vardct/encoder.rs:1208-1232`.
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
                None, // No splines in animation frames
            )?;
        }

        // Zensim quantization loop: uses zensim psychovisual metric +
        // per-pixel diffmap. Also refines AC strategy by splitting large
        // transforms with high perceptual error (hence `&mut ac_strategy`).
        // Mirrors `vardct/encoder.rs:1234-1259`.
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
                None, // No splines in animation frames
            )?;
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
        )?;

        let sharpness_map =
            if params.epf_iters > 0 && self.distance >= 0.5 && self.profile.epf_dynamic_sharpness {
                let mask = match mask1x1 {
                    Some(m) => m,
                    None => super::adaptive_quant::compute_mask1x1_with_budget(
                        &xyb_y,
                        padded_width,
                        padded_height,
                        self.budget.as_ref(),
                    )?,
                };
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
                    self.budget.as_ref(),
                )?)
            } else {
                None
            };

        let strategy_counts = ac_strategy.strategy_histogram();

        // If patches were detected, write the reference frame BEFORE the
        // main per-animation-frame VarDCT frame. The reference frame is a
        // modular FrameType::ReferenceOnly frame that stores unique patch
        // templates in the file's `save_as_reference` slot. The main frame
        // then references it via the patches block in LfGlobal.
        //
        // Mirrors the ordering in `encode_two_pass` (still-image entry
        // point) at the patches branch around line 1592 — but that path
        // also writes the file header itself. The animation entry point
        // already wrote the file header in `encode_animation_lossy`
        // (api.rs around line 6107) before the per-frame loop, so we only
        // emit the reference frame here.
        if let Some(ref pd) = patches_data {
            super::patches::encode_reference_frame(
                pd,
                self.use_ans,
                self.profile.patch_ref_tree_learning,
                writer,
                self.budget.as_ref(),
            )?;
            writer.zero_pad_to_byte();
        }

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
            extras,
            Some(frame_options),
            patches_data.as_ref(),
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
        extras: &[super::extras::VardctExtra<'_>],
        patches: Option<&super::patches::PatchesData>,
        splines: Option<&super::splines::SplinesData>,
        float_dc: Option<&[Vec<f32>; 3]>,
    ) -> Result<Vec<u8>> {
        let mut writer = BitWriter::with_capacity(width * height * 4);

        // Write file header. Every extra channel passed in here goes
        // into the file-header metadata; the writer below uses the
        // same list to produce the per-section modular sub-bitstream.
        let extras_info: Vec<ExtraChannelInfo> = extras.iter().map(|e| e.info.clone()).collect();
        self.write_file_header_and_pad(width, height, &extras_info, &mut writer)?;

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
            super::patches::encode_reference_frame(
                pd,
                self.use_ans,
                self.profile.patch_ref_tree_learning,
                &mut writer,
                self.budget.as_ref(),
            )?;
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
            extras,
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
        extras: &[super::extras::VardctExtra<'_>],
        frame_options: Option<&FrameOptions>,
        patches: Option<&super::patches::PatchesData>,
        splines: Option<&super::splines::SplinesData>,
        dc_quant_custom: Option<[f32; 3]>,
        writer: &mut BitWriter,
    ) -> Result<()> {
        let _phase_dbg = std::env::var_os("__JXL_ENC_PHASE_TIMING").is_some();
        let _t0 = std::time::Instant::now();
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

            // Only collect AC metadata tokens (no DC) — parallelizable
            let dc_results: Vec<(Vec<Token>, Vec<Token>)> =
                crate::parallel::parallel_map(num_dc_groups, |dc_group_idx| {
                    tokenize_dc_group_lf_frame(
                        dc_group_idx,
                        xsize_blocks,
                        ysize_blocks,
                        xsize_dc_groups,
                        quant_field,
                        cfl_map,
                        ac_strategy,
                        sharpness_map,
                        &ac_meta_ctx_map,
                    )
                });
            for (dc_tok, md_tok) in dc_results {
                dc_tokens_per_group.push(dc_tok);
                ac_metadata_tokens_per_group.push(md_tok);
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
                    num_dc_groups,
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

            // Collect DC + AC metadata tokens per group — parallelizable
            let dc_results: Vec<(Vec<Token>, Vec<Token>)> =
                crate::parallel::parallel_map(num_dc_groups, |dc_group_idx| {
                    tokenize_dc_group_wp(
                        dc_group_idx,
                        xsize_blocks,
                        ysize_blocks,
                        xsize_dc_groups,
                        quant_dc,
                        quant_field,
                        cfl_map,
                        ac_strategy,
                        sharpness_map,
                        &wp_dc_tree,
                        &dc_ctx_remap,
                        &ac_meta_ctx_map,
                    )
                });
            for (dc_tok, md_tok) in dc_results {
                dc_tokens_per_group.push(dc_tok);
                ac_metadata_tokens_per_group.push(md_tok);
            }
        }

        let _t_tok_dc = _t0.elapsed().as_secs_f64() * 1000.0;
        let _t_co = std::time::Instant::now();
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

        let _ms_co = _t_co.elapsed().as_secs_f64() * 1000.0;
        let _t_bcm = std::time::Instant::now();
        // Compute content-adaptive block context map
        let block_ctx_map = super::ac_context::compute_block_ctx_map(
            quant_field,
            ac_strategy,
            params.distance,
            xsize_blocks,
            ysize_blocks,
        );

        let _ms_bcm = _t_bcm.elapsed().as_secs_f64() * 1000.0;
        let _t_ac_tok = std::time::Instant::now();
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

        // AC section tokens: per-pass per-group — parallelizable
        // ac_section_tokens_per_pass[pass][group] = Vec<Token>
        let ac_results: Vec<Vec<Vec<Token>>> =
            crate::parallel::parallel_map(num_groups, |group_idx| {
                tokenize_ac_group(
                    group_idx,
                    xsize_blocks,
                    ysize_blocks,
                    xsize_groups,
                    quant_ac,
                    nzeros,
                    raw_nzeros,
                    quant_field,
                    ac_strategy,
                    &block_ctx_map,
                    custom_order_map.as_deref(),
                    used_orders,
                    &pass_config,
                )
            });

        // Transpose: ac_results[group][pass] → ac_section_tokens_per_pass[pass][group]
        let mut ac_section_tokens_per_pass: Vec<Vec<Vec<Token>>> =
            vec![Vec::with_capacity(num_groups); num_passes];
        for group_tokens in ac_results {
            for (pass, tokens) in group_tokens.into_iter().enumerate() {
                ac_section_tokens_per_pass[pass].push(tokens);
            }
        }

        let _ms_ac_tok = _t_ac_tok.elapsed().as_secs_f64() * 1000.0;
        let _t_lz77 = std::time::Instant::now();
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

        let _ms_lz77 = _t_lz77.elapsed().as_secs_f64() * 1000.0;
        let _t_codes = std::time::Instant::now();
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
        // Build entropy codes by iterating per-group tokens without merging.
        // This avoids allocating a merged Vec (which can be hundreds of MB).
        //
        // The DC code build and the per-pass AC code builds are independent
        // (they read disjoint token streams and write distinct outputs). Run
        // them in parallel via rayon::join: at 12 MP this drops ~40 ms off
        // the sequential build_codes phase.
        let base_ac_num_contexts = block_ctx_map.num_ac_contexts();
        let build_dc = || -> BuiltEntropyCode {
            let dc_groups: Vec<&[Token]> = dc_tokens_per_group
                .iter()
                .chain(ac_metadata_tokens_per_group.iter())
                .map(|v| v.as_slice())
                .collect();
            if self.use_ans {
                BuiltEntropyCode::Ans(build_entropy_code_ans_from_token_groups(
                    &dc_groups,
                    dc_num_contexts,
                    self.profile.enhanced_clustering_vardct,
                    self.profile.optimize_uint_configs_vardct,
                    dc_lz77_params.as_ref(),
                    None,
                ))
            } else {
                BuiltEntropyCode::Huffman(build_entropy_code_from_token_groups(
                    &dc_groups,
                    dc_num_contexts,
                    self.profile.enhanced_clustering_vardct,
                    dc_lz77_params.as_ref(),
                ))
            }
        };
        let build_ac_codes = || -> Vec<BuiltEntropyCode> {
            crate::parallel::parallel_map(num_passes, |pass| {
                let ac_num_contexts = if ac_lz77_params_per_pass[pass].is_some() {
                    base_ac_num_contexts + 1
                } else {
                    base_ac_num_contexts
                };
                let ac_groups: Vec<&[Token]> = ac_section_tokens_per_pass[pass]
                    .iter()
                    .map(|v| v.as_slice())
                    .collect();
                if self.use_ans {
                    BuiltEntropyCode::Ans(build_entropy_code_ans_from_token_groups(
                        &ac_groups,
                        ac_num_contexts,
                        self.profile.enhanced_clustering_vardct,
                        self.profile.optimize_uint_configs_vardct,
                        ac_lz77_params_per_pass[pass].as_ref(),
                        None,
                    ))
                } else {
                    BuiltEntropyCode::Huffman(build_entropy_code_from_token_groups(
                        &ac_groups,
                        ac_num_contexts,
                        self.profile.enhanced_clustering_vardct,
                        ac_lz77_params_per_pass[pass].as_ref(),
                    ))
                }
            })
        };
        let (dc_built_code, ac_built_codes) =
            crate::parallel::parallel_join(build_dc, build_ac_codes);

        let _ms_codes = _t_codes.elapsed().as_secs_f64() * 1000.0;
        let _t_pass2 = std::time::Instant::now();
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

        let num_extra_channels = extras.len();

        // Write frame header
        {
            let mut fh = FrameHeader::lossy();
            fh.x_qm_scale = params.x_qm_scale;
            fh.b_qm_scale = params.b_qm_scale;
            fh.epf_iters = params.epf_iters;
            fh.gaborish = self.enable_gaborish;
            fh.upsampling = self.upsampling;
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
                if let Some(tc) = opts.timecode {
                    fh.timecode = tc;
                }
                if let Some(ref name) = opts.name {
                    fh.name = name.clone();
                }
                if let Some(ref crop) = opts.crop {
                    fh.x0 = crop.x0;
                    fh.y0 = crop.y0;
                    fh.width = crop.width;
                    fh.height = crop.height;
                    fh.blend_mode = BlendMode::Replace;
                    fh.blend_source = 1;
                }
                // Per-frame blend override wins over the crop default.
                if let Some(mode) = opts.blend_mode {
                    fh.blend_mode = mode;
                }
                if let Some(source) = opts.blend_source {
                    fh.blend_source = source;
                }
                // For animation, save non-last frames to reference slot 1
                // so crop frames can composite onto the previous canvas.
                if opts.have_animation && !opts.is_last {
                    fh.save_as_reference = 1;
                }
                // Per-frame override wins last.
                if let Some(slot) = opts.save_as_reference {
                    fh.save_as_reference = slot;
                }
                // Reference-only frames are stored to a save slot but
                // NOT displayed during playback. The bitstream writer
                // (`headers/frame_header.rs::write`) gates duration /
                // is_last / x0,y0 / blending_info on the `normal_frame`
                // predicate (Regular or SkipProgressive), so simply
                // flipping `frame_type = ReferenceOnly` here is enough
                // to emit the correct bitstream layout. We also force
                // `is_last = false` (the spec disallows ReferenceOnly
                // as the last frame — public-API validation rejects
                // that combination), default the save slot to `1` when
                // the caller didn't pick one, and set
                // `save_before_ct = true` to match libjxl's
                // patches-frame defaults (patches.rs:1941).
                if opts.reference_only {
                    fh.frame_type = crate::headers::frame_header::FrameType::ReferenceOnly;
                    fh.is_last = false;
                    if opts.save_as_reference.is_none() && fh.save_as_reference == 0 {
                        fh.save_as_reference = 1;
                    }
                    fh.save_before_ct = true;
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

            // Single-group extras (alpha + any others): all data goes
            // in the modular global sub-bitstream within the DC global
            // section, after the VarDCT DC entropy code.
            if !extras.is_empty() {
                // Compute per-channel lossy quantizers (libjxl parity).
                // Alpha-typed extras read `alpha_distance`; others stay
                // at `q == 1` (lossless) until per-channel `ec_distance`
                // is wired through the public API. All-1 vector keeps
                // the lossless bit-identical path, so the default
                // `alpha_distance = None` is byte-for-byte identical
                // regardless of how many non-alpha extras follow.
                let quantizers = self.compute_extras_pixel_quantizers(extras);
                Self::write_modular_extras_global_with_quant(
                    extras,
                    width,
                    height,
                    &quantizers,
                    &mut dc_global,
                )?;
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
            // Multi-group extras: write empty modular global sub-bitstream.
            // Extra channels are NOT meta_or_small for >256px images,
            // so no per-channel data belongs in the global section.
            // The decoder still reads the GroupHeader + tree for the
            // global section.
            if !extras.is_empty() {
                Self::write_modular_empty_global(&mut dc_global)?;
            }
            dc_global.zero_pad_to_byte();
            sections.push(dc_global.finish());

            // DC groups — parallelizable
            let dc_group_sections: Vec<Vec<u8>> =
                crate::parallel::parallel_map_result(num_dc_groups, |dc_group_idx| {
                    encode_dc_group_section(
                        self,
                        dc_group_idx,
                        xsize_blocks,
                        ysize_blocks,
                        xsize_dc_groups,
                        &dc_tokens_per_group[dc_group_idx],
                        &ac_metadata_tokens_per_group[dc_group_idx],
                        ac_strategy,
                        &dc_built_code,
                        dc_lz77_params.as_ref(),
                    )
                })?;
            sections.extend(dc_group_sections);

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

            // AC groups: Section order is pass-major, group-minor — parallelizable
            // Section index = 2 + num_dc_groups + pass * num_groups + group
            //
            // Per-channel lossy quantizers (libjxl parity, all-`1`
            // vector keeps the lossless byte-identical path). Computed
            // once for the whole frame so all HF groups carry the same
            // multipliers; matches libjxl's per-channel `quants_[ch_id]`.
            // Mixed extras: alpha gets `alpha_distance`-derived `q`,
            // every other type stays at `q == 1` until per-channel
            // `ec_distance` is wired through the public API.
            let extras_quantizers: alloc::vec::Vec<u32> = if extras.is_empty() {
                alloc::vec::Vec::new()
            } else {
                self.compute_extras_pixel_quantizers(extras)
            };
            for pass in 0..num_passes {
                let is_last_pass = pass == num_passes - 1;
                let ac_group_sections: Vec<Vec<u8>> =
                    crate::parallel::parallel_map_result(num_groups, |group_idx| {
                        encode_ac_group_section(
                            &ac_section_tokens_per_pass[pass][group_idx],
                            &ac_built_codes[pass],
                            ac_lz77_params_per_pass[pass].as_ref(),
                            extras,
                            is_last_pass,
                            group_idx,
                            xsize_groups,
                            width,
                            height,
                            &extras_quantizers,
                        )
                    })?;
                sections.extend(ac_group_sections);
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

            // Center-first AC group reordering (#14). Identity prefix
            // for [LfGlobal, LfGroups..., HfGlobal], then AC groups
            // permuted by concentric-square distance from image
            // center. Only wired for num_passes == 1 (multi-pass
            // progressive + center-first interaction is a future
            // extension matching libjxl's per-pass loop).
            if self.center_first && num_groups > 1 && num_passes == 1 {
                use crate::vardct::coeff_order::compute_center_first_ac_permutation;
                use crate::vardct::frame::write_toc_with_permutation;
                // Caller-supplied center_x / center_y (libjxl
                // `cparams.center_x` / `center_y`); fall back to image
                // centre when unset.
                let cx = self
                    .center_x
                    .map(|x| x.min(width.saturating_sub(1) as u32))
                    .unwrap_or((width as u32) / 2);
                let cy = self
                    .center_y
                    .map(|y| y.min(height.saturating_sub(1) as u32))
                    .unwrap_or((height as u32) / 2);
                let ac_group_order =
                    compute_center_first_ac_permutation(xsize_groups, _ysize_groups, cx, cy);
                let mut inv_ac = vec![0u32; num_groups];
                for (on_disk_pos, &orig_idx) in ac_group_order.iter().enumerate() {
                    inv_ac[orig_idx as usize] = on_disk_pos as u32;
                }
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
                let mut new_sections: Vec<Vec<u8>> = (0..total).map(|_| Vec::new()).collect();
                for (logical_idx, section_data) in sections.into_iter().enumerate() {
                    let on_disk = permutation[logical_idx] as usize;
                    new_sections[on_disk] = section_data;
                }
                let permuted_sizes: Vec<usize> = new_sections.iter().map(|s| s.len()).collect();
                write_toc_with_permutation(&permuted_sizes, &permutation, self.use_ans, writer)?;
                for section in new_sections {
                    writer.append_bytes(&section)?;
                }
            } else {
                write_toc(&section_sizes, writer)?;
                for section in sections {
                    writer.append_bytes(&section)?;
                }
            }
        }

        let _ms_pass2 = _t_pass2.elapsed().as_secs_f64() * 1000.0;
        if _phase_dbg {
            eprintln!(
                "encode_two_pass: tok_dc={_t_tok_dc:.1} co={_ms_co:.1} bcm={_ms_bcm:.1} ac_tok={_ms_ac_tok:.1} lz77={_ms_lz77:.1} build_codes={_ms_codes:.1} pass2_write={_ms_pass2:.1}",
            );
        }
        Ok(())
    }

    /// Write DC group section from pre-collected tokens (two-pass mode).
    /// Write the modular global sub-bitstream for extras (alpha + others)
    /// in single-group VarDCT frames.
    ///
    /// For single-group images (≤256×256) every extra channel is
    /// "meta_or_small" and travels together in the LfGlobal section,
    /// in one sub-bitstream:
    ///   GroupHeader → (use_global_tree=0 → local tree) →
    ///   entropy code → channel-0 pixels → channel-1 pixels → …
    /// Write the single-group extras sub-bitstream into the DC global
    /// section. Applies the lossy alpha integer pixel quantizer `q`
    /// when `q > 1`; `q == 1` preserves the lossless path bit-for-bit.
    /// Currently only used for single-extra (alpha) lossy encoding;
    /// callers must enforce that constraint upstream.
    fn write_modular_extras_global_with_quant(
        extras: &[super::extras::VardctExtra<'_>],
        width: usize,
        height: usize,
        quantizers: &[u32],
        writer: &mut BitWriter,
    ) -> Result<()> {
        Self::write_modular_extras_subbitstream(
            extras, width, height, 0, 0, width, height, quantizers, writer,
        )
    }

    /// Write an empty modular global sub-bitstream for multi-group VarDCT frames with alpha.
    ///
    /// For multi-group images (>256×256), the alpha channel is NOT meta_or_small,
    /// so no alpha data belongs in the global section. The decoder reads the GroupHeader
    /// during `FullModularImage::read()`, then calls `decode_modular_subbitstream` with
    /// an empty buffer list (alpha assigned to HfGroups), which returns immediately.
    /// Only the GroupHeader is needed.
    fn write_modular_empty_global(writer: &mut BitWriter) -> Result<()> {
        // GroupHeader: use_global_tree=1, wp_params default=1, nb_transforms=0
        // Then 32-bit ANS initial state (no data follows)
        writer.write(1, 1)?; // use_global_tree = true
        writer.write(1, 1)?; // wp_params all_default = true
        writer.write(2, 0)?; // nb_transforms = 0
        // Initial ANS state: ANS_SIGNATURE << 16 = 0x130000
        writer.write(32, 0x130000)?;
        Ok(())
    }

    /// Write a modular sub-bitstream for N extra channels covering one
    /// rectangular region (used by both the global and per-HF-group paths).
    ///
    /// Format: GroupHeader → local tree → LZ77 header → entropy code →
    /// channel-0 residuals → channel-1 residuals → …
    ///
    /// Uses gradient prediction (predictor 5) with LZ77 RLE for
    /// efficient encoding of mostly-uniform extras (e.g. fully opaque
    /// alpha on a screenshot, or a flat-region depth map). All
    /// channels share one entropy code; the decoder pulls each
    /// channel's `channel_width × channel_height` tokens in the order
    /// the channels appear here.
    ///
    /// Channels must currently have `dim_shift == 0` — otherwise the
    /// per-channel region offsets and dimensions diverge from the
    /// VarDCT group grid and this writer would need to be plumbed with
    /// per-channel coords. `dim_shift > 0` for VarDCT extras is
    /// guarded upstream with an `Unsupported` error until we wire it.
    #[allow(clippy::too_many_arguments)]
    fn write_modular_extras_subbitstream(
        extras: &[super::extras::VardctExtra<'_>],
        image_width: usize,
        image_height: usize,
        x0: usize,
        y0: usize,
        region_width: usize,
        region_height: usize,
        quantizers: &[u32],
        writer: &mut BitWriter,
    ) -> Result<()> {
        use crate::modular::encode::{
            K_LZ77_MIN_LENGTH, K_LZ77_MIN_SYMBOL, Token, build_sparse_histogram,
            decompose_multiplier_pub, encode_hybrid_uint_000, encode_hybrid_uint_lz77_length,
            write_channel_split_tree_tokens, write_gradient_tree_tokens,
            write_gradient_tree_tokens_lossy, write_hybrid_data_histogram,
            write_sparse_lz77_histogram, write_tree_histogram_for_channel_split_lossy,
            write_tree_histogram_for_gradient, write_tree_histogram_for_gradient_lossy,
        };
        use crate::modular::predictor::pack_signed;

        debug_assert!(
            quantizers.is_empty() || quantizers.len() == extras.len(),
            "extras quantizers slice length must equal extras count when non-empty"
        );

        // GroupHeader: use_global_tree=0, wp default, no transforms
        writer.write(1, 0)?; // use_global_tree = false
        writer.write(1, 1)?; // wp_params all_default = true
        writer.write(2, 0)?; // nb_transforms = 0

        // Per-channel resolved quantizer (treat empty/short slice as
        // all-lossless so an internal caller can't accidentally enable
        // lossy on a channel that didn't opt in).
        let resolved_q = |ch: usize| -> u32 { quantizers.get(ch).copied().unwrap_or(1).max(1) };
        let any_lossy = (0..extras.len()).any(|i| resolved_q(i) > 1);

        // Local tree shape depends on the (set of) per-channel multipliers:
        //
        // - All channels lossless (`all q == 1`): single-leaf Gradient
        //   tree (byte-identical to the pre-pipeline lossless path).
        // - Exactly one channel with `q > 1` (always the alpha-only
        //   case today): single-leaf Gradient tree carrying that
        //   multiplier (byte-identical to the W6-3 single-extra lossy
        //   alpha path).
        // - Mixed: multi-leaf tree splitting on property 0 (channel
        //   index) so each channel gets its own multiplier. Mirrors
        //   libjxl's per-channel `quants_[ch_id]` (`enc_modular.cc:1027`)
        //   on a decoder-side decision tree.
        let use_channel_split = any_lossy && extras.len() > 1;
        // Per-channel multiplier vector used by both the tree writer
        // and the residual loop. Length 0 when `extras` is empty.
        let mut per_channel_q: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(extras.len());
        for i in 0..extras.len() {
            per_channel_q.push(resolved_q(i));
        }
        if use_channel_split {
            let (d, c) = write_tree_histogram_for_channel_split_lossy(writer, &per_channel_q)?;
            write_channel_split_tree_tokens(writer, &d, &c, &per_channel_q)?;
        } else if !any_lossy {
            let (d, c) = write_tree_histogram_for_gradient(writer)?;
            write_gradient_tree_tokens(writer, &d, &c)?;
        } else {
            // Exactly one channel is lossy. Find it and use its
            // multiplier on the (single-leaf) tree.
            let lossy_ch = (0..extras.len()).find(|&i| resolved_q(i) > 1).unwrap_or(0);
            let q_single = resolved_q(lossy_ch);
            let (mul_log, mul_bits) = decompose_multiplier_pub(q_single);
            let (d, c) = write_tree_histogram_for_gradient_lossy(writer, mul_log, mul_bits)?;
            write_gradient_tree_tokens_lossy(writer, &d, &c, mul_log, mul_bits)?;
        }

        // Collect residuals with LZ77 RLE detection.
        //
        // All channels share one token stream and one entropy code;
        // the decoder pulls each channel's tokens in sequence.
        let mut tokens = Vec::new();
        let mut current_run = 0usize;
        let mut num_decoded = 0usize;
        let mut last_value = u32::MAX; // impossible initial value prevents LZ77 from first pixel

        for (ch_idx, ec) in extras.iter().enumerate() {
            // dim_shift > 0 in lossy is guarded upstream — assert here
            // so a future caller can't silently produce a wrong bitstream.
            debug_assert_eq!(
                ec.info.dim_shift, 0,
                "write_modular_extras_subbitstream: dim_shift > 0 not supported yet"
            );

            let ch_w = ec.channel_width(image_width);
            let ch_x0 = x0 >> ec.info.dim_shift;
            let ch_y0 = y0 >> ec.info.dim_shift;
            let ch_rw = region_width >> ec.info.dim_shift;
            let ch_rh = region_height >> ec.info.dim_shift;
            let _ = image_height; // reserved for dim_shift > 0 plumbing

            // Flush any pending run from the *previous* channel before
            // resetting prediction state. Without this the run gets
            // re-attributed to the next channel as Raw(u32::MAX)
            // tokens, which the decoder can't parse.
            if current_run > 0 {
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
            }

            // Each channel starts a fresh prediction context: gradient
            // prediction references neighbours within the same
            // channel, never across channels. Reset the LZ77 anchor so
            // a uniform end-of-prev-channel doesn't run into the next.
            last_value = u32::MAX;

            // Per-channel pixel quantizer. The decoder picks this
            // multiplier via the channel-split tree (when present) or
            // via the single-leaf lossy / lossless tree. Lossless
            // (`q_ch == 1`) leaves the scratch buffer empty and falls
            // back to direct `ec.data.sample`.
            let q_ch = per_channel_q.get(ch_idx).copied().unwrap_or(1).max(1);
            let qi = q_ch as i32;
            let half = qi / 2;
            let mut quantized: alloc::vec::Vec<i32> = alloc::vec::Vec::new();
            if q_ch > 1 {
                quantized.reserve(ch_rw.saturating_mul(ch_rh));
                for y in 0..ch_rh {
                    for x in 0..ch_rw {
                        let s = ec.data.sample((ch_y0 + y) * ch_w + (ch_x0 + x));
                        let snapped = if s >= 0 {
                            ((s + half) / qi) * qi
                        } else {
                            -(((-s) + half) / qi) * qi
                        };
                        quantized.push(snapped);
                    }
                }
            }
            let read = |ry: usize, rx: usize| -> i32 {
                if q_ch > 1 {
                    quantized[ry * ch_rw + rx]
                } else {
                    ec.data.sample((ch_y0 + ry) * ch_w + (ch_x0 + rx))
                }
            };

            for y in 0..ch_rh {
                for x in 0..ch_rw {
                    let pixel = read(y, x);

                    let left = if x > 0 {
                        read(y, x - 1)
                    } else if y > 0 {
                        read(y - 1, 0)
                    } else {
                        0
                    };
                    let top = if y > 0 { read(y - 1, x) } else { left };
                    let topleft = if x > 0 && y > 0 {
                        read(y - 1, x - 1)
                    } else {
                        left
                    };

                    // ClampedGradient prediction
                    let grad = left + top - topleft;
                    let prediction = grad.clamp(left.min(top), left.max(top));
                    // For lossy: `pixel` and `prediction` are both
                    // multiples of q_ch, so the residual is exactly
                    // divisible. Divide so the decoder reconstructs
                    // `prediction + val * q_ch == pixel`.
                    let raw_residual = pixel - prediction;
                    let residual = if q_ch == 1 {
                        raw_residual
                    } else {
                        raw_residual / qi
                    };
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

    /// Write a modular HF group sub-bitstream for the extras in
    /// multi-group VarDCT frames.
    ///
    /// Each HF group gets its own independent modular sub-bitstream
    /// with a fresh GroupHeader, local tree, and entropy code. All
    /// extras share the one entropy code; the decoder pulls each
    /// channel's per-group region in turn.
    /// Write one HF-group extras sub-bitstream for a multi-group
    /// VarDCT frame. Threads a per-channel integer pixel quantizer
    /// slice for lossy encoding; an all-`1` slice preserves the
    /// lossless multi-group path bit-for-bit. Length must equal
    /// `extras.len()`.
    #[allow(clippy::too_many_arguments)]
    fn write_modular_extras_group_with_quant(
        extras: &[super::extras::VardctExtra<'_>],
        image_width: usize,
        image_height: usize,
        x0: usize,
        y0: usize,
        region_width: usize,
        region_height: usize,
        quantizers: &[u32],
        writer: &mut BitWriter,
    ) -> Result<()> {
        Self::write_modular_extras_subbitstream(
            extras,
            image_width,
            image_height,
            x0,
            y0,
            region_width,
            region_height,
            quantizers,
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
