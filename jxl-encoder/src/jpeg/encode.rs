// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JPEG lossless reencoding into JPEG XL VarDCT format.
//!
//! Converts parsed JPEG data (from `read_jpeg`) into a JXL codestream that
//! preserves the exact quantized DCT coefficients. The resulting JXL file
//! decodes to pixel-identical output as the original JPEG.

use super::data::*;
use super::jbrd::{encode_jbrd, extract_exif, extract_icc, extract_xmp};
use crate::BLOCK_SIZE;
use crate::bit_writer::BitWriter;
use crate::container::wrap_in_container_jxlp;
use crate::entropy_coding::encode::{
    OwnedAnsEntropyCode, build_entropy_code_ans, write_entropy_code_ans, write_tokens_ans,
};
use crate::entropy_coding::token::Token;
use crate::error::Result;
use crate::headers::color_encoding::ColorEncoding;
use crate::headers::file_header::{BitDepth, FileHeader, ImageMetadata};
use crate::headers::frame_header::{Encoding, FrameHeader};
use crate::vardct::ac_context;
use crate::vardct::ac_group::{collect_ac_coefficients_into, predict_from_top_and_left};
use crate::vardct::ac_strategy::AcStrategyMap;
use crate::vardct::chroma_from_luma::CflMap;
use crate::vardct::coeff_order::{
    NUM_ORDER_BUCKETS as NUM_ORDER_BUCKETS_JPEG, build_and_write_coeff_orders,
    compute_custom_orders, get_custom_order, tokenize_coeff_orders,
};
use crate::vardct::common::*;
use crate::vardct::dc_coding::{
    NUM_DC_CONTEXTS, collect_ac_metadata_tokens_region, collect_dc_tokens_region,
};
use crate::vardct::frame::{
    assemble_frame_sections, write_dc_group_from_tokens, write_quant_scales,
};

/// Number of JXL quant tables (from libjxl quant_weights.h).
const NUM_QUANT_TABLES: usize = 17;

/// Encode a parsed JPEG as a JXL codestream (lossless reencoding).
///
/// The output JXL will decode to pixel-identical results as the original JPEG.
/// This does NOT include the jbrd box — it produces a bare JXL codestream.
/// For byte-exact JPEG reconstruction, wrap in a container with a jbrd box.
pub fn encode_jpeg_to_jxl(jpeg: &JpegData) -> Result<Vec<u8>> {
    let (codestream, _split) = encode_jpeg_to_jxl_inner(jpeg)?;
    Ok(codestream)
}

/// Inner function that returns both codestream bytes and the file header size
/// (split point for jxlp box splitting when JBRD is needed).
fn encode_jpeg_to_jxl_inner(jpeg: &JpegData) -> Result<(Vec<u8>, usize)> {
    let width = jpeg.width as usize;
    let height = jpeg.height as usize;

    // Channel mapping: JXL c0=Cb, c1=Y, c2=Cr for YCbCr
    // JPEG components are typically: 0=Y, 1=Cb, 2=Cr
    // For grayscale (1 component), all JXL channels map to the single component.
    let jpeg_c_map: [usize; 3] = if jpeg.components.len() == 1 {
        [0, 0, 0] // grayscale: all channels reference component 0
    } else {
        match jpeg.component_type {
            JpegComponentType::YCbCr => [1, 0, 2], // JXL c0←JPEG Cb, c1←JPEG Y, c2←JPEG Cr
            _ => [0, 1, 2],                        // RGB or other: identity mapping
        }
    };

    let num_components = jpeg.components.len();
    if num_components != 3 && num_components != 1 {
        return Err(crate::error::Error::InvalidInput(format!(
            "JPEG reencoding requires 1 or 3 components, got {num_components}"
        )));
    }

    // Compute per-channel upsampling modes from sampling factors
    let jpeg_upsampling = if num_components == 3 {
        compute_jpeg_upsampling(jpeg, &jpeg_c_map)
    } else {
        [0; 3] // grayscale: no subsampling
    };

    // Compute per-channel actual downsampling shifts.
    // JXL stores the sampling factor (as log2) in jpeg_upsampling, not the shift.
    // The actual shift = max_raw_shift - raw_shift, matching libjxl's:
    //   HShift(c) = maxhs_ - kHShift[channel_mode_[c]]
    let max_raw_hs = jpeg_upsampling
        .iter()
        .map(|&u| JPEG_UPSAMPLING_H_SHIFT[u as usize])
        .max()
        .unwrap_or(0);
    let max_raw_vs = jpeg_upsampling
        .iter()
        .map(|&u| JPEG_UPSAMPLING_V_SHIFT[u as usize])
        .max()
        .unwrap_or(0);
    let channel_shifts: [(usize, usize); 3] = [
        (
            max_raw_hs - JPEG_UPSAMPLING_H_SHIFT[jpeg_upsampling[0] as usize],
            max_raw_vs - JPEG_UPSAMPLING_V_SHIFT[jpeg_upsampling[0] as usize],
        ),
        (
            max_raw_hs - JPEG_UPSAMPLING_H_SHIFT[jpeg_upsampling[1] as usize],
            max_raw_vs - JPEG_UPSAMPLING_V_SHIFT[jpeg_upsampling[1] as usize],
        ),
        (
            max_raw_hs - JPEG_UPSAMPLING_H_SHIFT[jpeg_upsampling[2] as usize],
            max_raw_vs - JPEG_UPSAMPLING_V_SHIFT[jpeg_upsampling[2] as usize],
        ),
    ];

    // Frame-level block dimensions: padded to multiples of the max sampling factor shift.
    // For 4:4:4 this produces identical results to the Y component's dimensions.
    // For 4:2:0, this rounds up to even block counts.
    let max_hs = channel_shifts.iter().map(|&(hs, _)| hs).max().unwrap_or(0);
    let max_vs = channel_shifts.iter().map(|&(_, vs)| vs).max().unwrap_or(0);
    let xsize_blocks = div_ceil(width, 8 << max_hs) << max_hs;
    let ysize_blocks = div_ceil(height, 8 << max_vs) << max_vs;

    // Map JPEG coefficients to JXL data structures
    // Each channel uses its native block dimensions (may differ for subsampled chroma)
    let (mut quant_dc, mut quant_ac, mut nzeros, mut raw_nzeros) =
        map_jpeg_coefficients(jpeg, &jpeg_c_map)?;

    let is_gray = num_components == 1;

    // For grayscale, zero-fill Cb and Cr channels (JXL c0 and c2).
    // libjxl does this in enc_frame.cc — only Y (c1) keeps actual data.
    if is_gray {
        for c in [0, 2] {
            for row in &mut quant_dc[c] {
                row.fill(0);
            }
            for row in &mut quant_ac[c] {
                for block in row.iter_mut() {
                    block.fill(0);
                }
            }
            for row in &mut nzeros[c] {
                row.fill(0);
            }
            for row in &mut raw_nzeros[c] {
                row.fill(0);
            }
        }
    }

    // All blocks use DCT8
    let ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);

    // JPEG-CfL search (refs #16): mirrors libjxl's
    // `force_cfl_jpeg_recompression` (default ON) at
    // `enc_frame.cc:855-941`. Only applies to 4:4:4 YCbCr JPEGs with
    // 3 components — for 4:2:0 / 4:2:2 / grayscale we keep the
    // zero-map. Each 8×8-block color tile gets a per-channel YtoX /
    // YtoB multiplier that maximizes zero AC coefficients in chroma
    // after subtracting `RatioJPEG(factor) * Y` (fixed-point).
    let xsize_tiles = div_ceil(xsize_blocks, TILE_DIM_IN_BLOCKS);
    let ysize_tiles = div_ceil(ysize_blocks, TILE_DIM_IN_BLOCKS);
    let is_444 = jpeg_upsampling.iter().all(|&u| u == 0);
    let cfl_map = if !is_gray && is_444 && num_components == 3 {
        // Build scaled_qtable per chroma channel: position-transposed
        // `(1 << 11) * qt_y[pos] / qt_c[pos]`. quant_ac is already
        // transposed (block[x*8+y] = JPEG[y*8+x]), so we use the
        // raw qtables and transpose to match.
        let qt = build_raw_qtables(jpeg, &jpeg_c_map)?;
        let mut scaled_qtable = [[0i32; 64]; 3];
        for c in 0..3 {
            for y in 0..8usize {
                for x in 0..8usize {
                    let coeffpos = y * 8 + x;
                    let transposed = x * 8 + y;
                    // qt is already transposed in build_raw_qtables, so
                    // index directly. Guard against zero divisor.
                    let qy = qt[64 + transposed];
                    let qc = qt[64 * c + transposed];
                    if qc != 0 {
                        scaled_qtable[c][coeffpos] = ((1 << 11) * qy) / qc;
                    }
                }
            }
        }
        let mut map = CflMap::zeros(xsize_tiles, ysize_tiles);
        // c=0 → JXL Cb (YtoX), c=2 → JXL Cr (YtoB)
        map.ytox = crate::vardct::chroma_from_luma::jpeg_cfl_search(
            0,
            xsize_blocks,
            ysize_blocks,
            &quant_ac[1],
            &quant_ac[0],
            &scaled_qtable[0],
        );
        map.ytob = crate::vardct::chroma_from_luma::jpeg_cfl_search(
            2,
            xsize_blocks,
            ysize_blocks,
            &quant_ac[1],
            &quant_ac[2],
            &scaled_qtable[2],
        );
        map
    } else {
        CflMap::zeros(xsize_tiles, ysize_tiles)
    };

    // Quant field: all 1s (JPEG already quantized)
    let quant_field = vec![1u8; xsize_blocks * ysize_blocks];

    // Group dimensions
    let xsize_groups = div_ceil(width, GROUP_DIM);
    let ysize_groups = div_ceil(height, GROUP_DIM);
    let xsize_dc_groups = div_ceil(width, DC_GROUP_DIM);
    let ysize_dc_groups = div_ceil(height, DC_GROUP_DIM);
    let num_groups = xsize_groups * ysize_groups;
    let num_dc_groups = xsize_dc_groups * ysize_dc_groups;
    // Build transposed quant tables for RAW encoding
    let raw_qtables = build_raw_qtables(jpeg, &jpeg_c_map)?;

    // DC dequantization values: dc_dequant[c] = Q_dc[c] / 2040.0
    let dc_dequant = build_dc_dequant(jpeg, &jpeg_c_map)?;

    // ── Pass 1: Collect all tokens ──

    // DC + AC metadata tokens per DC group
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

        let dc_tokens = collect_dc_tokens_region(
            &quant_dc,
            start_bx,
            start_by,
            end_bx,
            end_by,
            &channel_shifts,
        );
        let md_tokens = collect_ac_metadata_tokens_region(
            region_xsize,
            region_ysize,
            &quant_field,
            xsize_blocks,
            start_bx,
            start_by,
            &cfl_map,
            &ac_strategy,
            None, // no sharpness map for JPEG
        );
        dc_tokens_per_group.push(dc_tokens);
        ac_metadata_tokens_per_group.push(md_tokens);
    }

    // AC tokens per group — iterate blocks, call collect_ac_coefficients per block
    // Use the default 4-cluster block context map matching what we write in DC global.
    // JPEG reencoding has uniform QF=1 and all-DCT8, so adaptive context modeling
    // provides no benefit and compute_block_ctx_map would produce a different cluster
    // count than the hardcoded COMPACT_BLOCK_CONTEXT_MAP, causing decoder mismatch.
    let block_ctx_map = ac_context::BlockCtxMap::default();

    // EX-J17a: enable wire-format-safe per-channel custom coefficient orders.
    // The JPEG bridge is all-DCT8 so only bucket 0 can carry custom orders, but
    // the per-channel zero distribution still differs (Y vs Cb vs Cr), so
    // permuting positions can cluster zeros at the end of each block's scan
    // and shrink AC token totals. Spec-mandated per-block channel order
    // `[Y, X, B]` is unchanged; only the *position* permutation per channel
    // varies. compute_custom_orders has a built-in Lehmer cost-benefit gate
    // that returns used_orders=0 when the permutation overhead would exceed
    // the AC savings — same gate as the VarDCT path uses. Tiny images (under
    // 5 blocks per side) are skipped to mirror VarDCT's xsize_blocks/ysize_blocks
    // threshold at bitstream.rs:2067.
    //
    // We can't reuse `count_zero_coefficients` directly because chroma planes
    // are smaller than the luma grid under 4:2:0 / 4:2:2 / 4:4:0 subsampling;
    // count it manually per-channel using each channel's native dimensions.
    let (custom_order_map, used_orders) = if xsize_blocks >= 5 || ysize_blocks >= 5 {
        let mut zero_counts: Vec<Vec<Vec<i64>>> = (0..NUM_ORDER_BUCKETS_JPEG)
            .map(|_| vec![Vec::new(); 3])
            .collect();
        // Only bucket 0 (DCT8) is populated — every JPEG block is DCT8.
        for ch in &mut zero_counts[0] {
            *ch = vec![0i64; BLOCK_SIZE];
        }
        for c in 0..3 {
            let plane = &quant_ac[c];
            let cnt = &mut zero_counts[0][c];
            for row in plane {
                for block in row {
                    // Skip k=0 (DC) — it lives in quant_dc and is always 0
                    // in quant_ac; we set it to -1 below so it sorts to the
                    // front of the permutation as the LLF position
                    // (mirroring count_zero_coefficients's treatment for
                    // DCT8 bucket 0, coeff_order.rs:328-333).
                    for k in 1..BLOCK_SIZE {
                        if block[k] == 0 {
                            cnt[k] += 1;
                        }
                    }
                }
            }
            // Mark LLF position (DC = index 0 for DCT8 bucket 0).
            cnt[0] = -1;
        }
        let (orders, used) = compute_custom_orders(&zero_counts);
        if used != 0 {
            (Some(orders), used)
        } else {
            (None, 0u32)
        }
    } else {
        (None, 0u32)
    };

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
                // All DCT8, so every block is "first"
                let strategy_code = 0u8; // DCT8
                let raw_strategy = 0u8;

                for &c in &[1usize, 0, 2] {
                    // channel order: Y, X(Cb), B(Cr)
                    let (hs, vs) = channel_shifts[c];

                    // Skip non-aligned positions for subsampled channels
                    if hs > 0 && (bx & ((1 << hs) - 1)) != 0 {
                        continue;
                    }
                    if vs > 0 && (by & ((1 << vs) - 1)) != 0 {
                        continue;
                    }

                    // Convert to channel-local block coordinates
                    let ch_bx = bx >> hs;
                    let ch_by = by >> vs;
                    let ch_start_bx = start_bx >> hs;
                    let ch_start_by = start_by >> vs;

                    let nz = raw_nzeros[c][ch_by][ch_bx];
                    let local_bx = ch_bx - ch_start_bx;
                    let row_top = if ch_by > ch_start_by {
                        Some(nzeros[c][ch_by - 1].as_slice())
                    } else {
                        None
                    };
                    let predicted_nz = if local_bx == 0 {
                        match row_top {
                            Some(top) => top[ch_bx] as i32,
                            None => 32,
                        }
                    } else {
                        predict_from_top_and_left(row_top, &nzeros[c][ch_by], ch_bx, 32)
                    };
                    let qf_val = quant_field[by * xsize_blocks + bx] as u32;
                    let block_ctx = block_ctx_map.block_context(c, strategy_code, qf_val);
                    // EX-J17a: pass per-(bucket, channel) custom order if one was selected.
                    // raw_strategy is always 0 (DCT8) on the JPEG path, so this only
                    // consults bucket 0 of the orders map.
                    let custom_order = custom_order_map
                        .as_ref()
                        .and_then(|orders| get_custom_order(orders, used_orders, raw_strategy, c));
                    collect_ac_coefficients_into(
                        &mut tokens,
                        &quant_ac[c][ch_by][ch_bx],
                        raw_strategy,
                        nz,
                        predicted_nz,
                        block_ctx,
                        block_ctx_map.num_ctxs,
                        custom_order,
                    );
                }
            }
        }
        ac_section_tokens.push(tokens);
    }

    // ── Build entropy codes (ANS) ──

    let dc_num_contexts = NUM_DC_CONTEXTS;
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
    let dc_code = build_entropy_code_ans(&all_dc_tokens, dc_num_contexts);

    let ac_num_contexts = block_ctx_map.num_ac_contexts();
    let total_ac_tokens: usize = ac_section_tokens.iter().map(|t| t.len()).sum();
    let mut all_ac_tokens = Vec::with_capacity(total_ac_tokens);
    for section in &ac_section_tokens {
        all_ac_tokens.extend_from_slice(section);
    }
    let ac_code = build_entropy_code_ans(&all_ac_tokens, ac_num_contexts);

    // ── Pass 2: Write bitstream ──

    let mut writer = BitWriter::with_capacity(width * height * 4);

    // Extract ICC profile from JPEG APP2 markers (if present)
    let icc_profile = extract_icc(jpeg);

    // File header (write() includes the signature)
    let mut file_header = build_jpeg_file_header(width, height, is_gray);
    if icc_profile.is_some() {
        file_header.metadata.color_encoding.want_icc = true;
    }
    file_header.write(&mut writer)?;

    // Write ICC profile data after file header (PredictICC encoded)
    if let Some(ref icc) = icc_profile {
        crate::icc::write_icc(icc, &mut writer)?;
    }

    writer.zero_pad_to_byte();
    let file_header_bytes = writer.bytes_written();

    // Frame header
    let frame_header = build_jpeg_frame_header(jpeg, jpeg_upsampling);
    frame_header.write(&mut writer)?;

    // Build section content using shared infrastructure
    let write_tok = |tokens: &[Token], w: &mut BitWriter| -> Result<()> {
        write_tokens_ans(tokens, &dc_code, None, w)
    };

    // DC Global
    let mut dc_global = BitWriter::new();
    write_dc_global_jpeg(&dc_dequant, &dc_code, num_dc_groups, &mut dc_global)?;

    // DC Groups (using shared function from frame.rs)
    let mut dc_groups = Vec::with_capacity(num_dc_groups);
    for dc_group_idx in 0..num_dc_groups {
        let mut dc_group = BitWriter::new();
        write_dc_group_from_tokens(
            dc_group_idx,
            xsize_blocks,
            ysize_blocks,
            xsize_dc_groups,
            &dc_tokens_per_group[dc_group_idx],
            &ac_metadata_tokens_per_group[dc_group_idx],
            &ac_strategy,
            &write_tok,
            &mut dc_group,
        )?;
        dc_groups.push(dc_group);
    }

    // AC Global
    let mut ac_global = BitWriter::new();
    // EX-J17a: build Lehmer tokens for the per-channel custom orders selected
    // above (if any). Tokens are written between the used_orders selector
    // and the AC entropy code, matching VarDCT bitstream order.
    let coeff_order_tokens = if used_orders != 0 {
        let orders = custom_order_map
            .as_ref()
            .expect("custom_order_map must exist when used_orders != 0");
        Some(tokenize_coeff_orders(orders, used_orders))
    } else {
        None
    };
    write_ac_global_jpeg(
        &raw_qtables,
        num_groups,
        &ac_code,
        used_orders,
        coeff_order_tokens.as_deref(),
        &mut ac_global,
    )?;

    // AC Groups
    let mut ac_groups = Vec::with_capacity(num_groups);
    for ac_tokens in &ac_section_tokens {
        let mut ac_group_writer = BitWriter::new();
        write_tokens_ans(ac_tokens, &ac_code, None, &mut ac_group_writer)?;
        ac_groups.push(ac_group_writer);
    }

    // Assemble frame (shared single-group/multi-group assembly logic)
    assemble_frame_sections(dc_global, dc_groups, ac_global, ac_groups, &mut writer)?;

    Ok((writer.finish_with_padding(), file_header_bytes))
}

/// Encode a JPEG as a JXL container with JBRD for byte-exact reconstruction.
///
/// Returns a complete JXL container file with:
/// - `jxlp` boxes: VarDCT codestream split around the jbrd box
/// - `jbrd` box: JPEG Bitstream Reconstruction Data
/// - `Exif` box: EXIF metadata (if present in JPEG)
/// - `xml ` box: XMP metadata (if present in JPEG)
///
/// A decoder with JPEG reconstruction support (e.g., djxl --reconstruct_jpeg)
/// can produce a byte-exact copy of the original JPEG from this container.
pub fn encode_jpeg_to_jxl_container(jpeg: &JpegData) -> Result<Vec<u8>> {
    let (codestream, file_header_size) = encode_jpeg_to_jxl_inner(jpeg)?;
    let jbrd = encode_jbrd(jpeg)?;
    let exif = extract_exif(jpeg);
    let xmp = extract_xmp(jpeg);

    // Split codestream at file header boundary for jxlp box format.
    // libjxl requires the jbrd box to appear between the file header
    // and frame data, using jxlp (partial codestream) boxes.
    let cs_part1 = &codestream[..file_header_size];
    let cs_part2 = &codestream[file_header_size..];

    Ok(wrap_in_container_jxlp(
        cs_part1,
        cs_part2,
        &jbrd,
        exif.as_deref(),
        xmp.as_deref(),
    ))
}

/// Map JPEG coefficients into JXL quant_dc / quant_ac / nzeros arrays.
///
/// Each channel uses its component's native block dimensions (which differ
/// for subsampled chroma channels).
#[allow(clippy::type_complexity)]
fn map_jpeg_coefficients(
    jpeg: &JpegData,
    jpeg_c_map: &[usize; 3],
) -> Result<(
    [Vec<Vec<i16>>; 3],
    [Vec<Vec<[i32; BLOCK_SIZE]>>; 3],
    [Vec<Vec<u8>>; 3],
    [Vec<Vec<u16>>; 3],
)> {
    let mut quant_dc: [Vec<Vec<i16>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut quant_ac: [Vec<Vec<[i32; BLOCK_SIZE]>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut nzeros: [Vec<Vec<u8>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut raw_nzeros: [Vec<Vec<u16>>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for jxl_c in 0..3 {
        let jpeg_c = jpeg_c_map[jxl_c];
        let comp = &jpeg.components[jpeg_c];
        let xb = comp.width_in_blocks as usize;
        let yb = comp.height_in_blocks as usize;

        let mut dc_rows = Vec::with_capacity(yb);
        let mut ac_rows = Vec::with_capacity(yb);
        let mut nz_rows = Vec::with_capacity(yb);
        let mut raw_nz_rows = Vec::with_capacity(yb);

        for by in 0..yb {
            let mut dc_row = Vec::with_capacity(xb);
            let mut ac_row: Vec<[i32; BLOCK_SIZE]> = Vec::with_capacity(xb);
            let mut nz_row = Vec::with_capacity(xb);
            let mut raw_nz_row = Vec::with_capacity(xb);

            for bx in 0..xb {
                let blk_idx = by * xb + bx;
                let base = blk_idx * 64;

                // DC coefficient (natural order position 0)
                let dc = comp.coeffs[base];
                dc_row.push(dc);

                // AC coefficients with transposition: JXL block[x*8+y] = JPEG[y*8+x]
                let mut ac_block = [0i32; BLOCK_SIZE];
                let mut nz_count = 0u16;
                for y in 0..8 {
                    for x in 0..8 {
                        if x == 0 && y == 0 {
                            continue; // DC is separate
                        }
                        let natural_idx = y * 8 + x;
                        let transposed_idx = x * 8 + y;
                        ac_block[transposed_idx] = comp.coeffs[base + natural_idx] as i32;
                        if ac_block[transposed_idx] != 0 {
                            nz_count += 1;
                        }
                    }
                }
                ac_row.push(ac_block);

                // nzeros: for DCT8, shifted == raw (covered_blocks=1)
                nz_row.push(nz_count as u8);
                raw_nz_row.push(nz_count);
            }

            dc_rows.push(dc_row);
            ac_rows.push(ac_row);
            nz_rows.push(nz_row);
            raw_nz_rows.push(raw_nz_row);
        }

        quant_dc[jxl_c] = dc_rows;
        quant_ac[jxl_c] = ac_rows;
        nzeros[jxl_c] = nz_rows;
        raw_nzeros[jxl_c] = raw_nz_rows;
    }

    Ok((quant_dc, quant_ac, nzeros, raw_nzeros))
}

/// Build the transposed RAW quantization tables for JXL.
///
/// For each JXL channel c, builds a 64-entry table from the JPEG quant table,
/// with rows and columns swapped: qt_jxl[8*x+y] = qt_jpeg[8*y+x].
fn build_raw_qtables(jpeg: &JpegData, jpeg_c_map: &[usize; 3]) -> Result<Vec<i32>> {
    let mut qtables = vec![0i32; 3 * 64];
    for jxl_c in 0..3 {
        let jpeg_c = jpeg_c_map[jxl_c];
        let quant_idx = jpeg.components[jpeg_c].quant_idx as usize;
        let qt = &jpeg.quant[quant_idx].values;
        for y in 0..8 {
            for x in 0..8 {
                // Transpose: JXL stores coefficients transposed vs JPEG
                qtables[jxl_c * 64 + x * 8 + y] = qt[y * 8 + x];
            }
        }
    }
    Ok(qtables)
}

/// Build DC dequantization values for the DequantDC header section.
///
/// Returns the inverse DC quantization factors: `dc_dequant[c] = Q_dc[c] / 2040.0`
///
/// In libjxl, `SetDCQuant(dcquantization)` stores `dc_quant_[c] = 1/dcquantization[c]`
/// where `dcquantization[c] = 2040/Q_dc[c]`. The stored value `dc_quant_[c] = Q_dc[c]/2040`
/// is then written as `dc_quant_[c] * 128` in F16 format. The decoder reads this F16
/// and uses it directly in: `scale = m_lf * 512 / (global_scale * quant_lf)`.
fn build_dc_dequant(jpeg: &JpegData, jpeg_c_map: &[usize; 3]) -> Result<[f32; 3]> {
    let mut dc_dequant = [0.0f32; 3];
    for jxl_c in 0..3 {
        let jpeg_c = jpeg_c_map[jxl_c];
        let quant_idx = jpeg.components[jpeg_c].quant_idx as usize;
        let q_dc = jpeg.quant[quant_idx].values[0] as f32;
        dc_dequant[jxl_c] = q_dc / (255.0 * 8.0);
    }
    Ok(dc_dequant)
}

/// Build the JXL file header for JPEG reencoding.
fn build_jpeg_file_header(width: usize, height: usize, is_gray: bool) -> FileHeader {
    let color_encoding = if is_gray {
        // Grayscale sRGB with Relative rendering intent (matches libjxl SRGB(true))
        ColorEncoding {
            rendering_intent: crate::headers::color_encoding::RenderingIntent::Relative,
            ..ColorEncoding::gray()
        }
    } else {
        ColorEncoding::srgb() // RGB sRGB (all_default=true)
    };

    FileHeader {
        width: width as u32,
        height: height as u32,
        metadata: ImageMetadata {
            bit_depth: BitDepth::uint8(),
            color_encoding,
            extra_channels: Vec::new(),
            xyb_encoded: false, // JPEG is NOT in XYB
            ..ImageMetadata::default()
        },
        upsampling_mode: None,
        upsampling_factor: 1,
    }
}

/// Compute the jpeg_upsampling field from JPEG sampling factors.
///
/// JXL stores the sampling FACTOR (as log2), not the downsampling shift.
/// The actual downsampling shift used by the decoder is `maxhs - kHShift[mode]`.
/// This matches libjxl's `YCbCrChromaSubsampling::Set()` in frame_header.h.
///
/// Modes encode the component's own sampling factor:
/// - 0: factor 1×1 (h=1, v=1) — subsampled channels
/// - 1: factor 2×2 (h=2, v=2) — full-resolution luma in 4:2:0
/// - 2: factor 2×1 (h=2, v=1) — full-resolution luma in 4:2:2
/// - 3: factor 1×2 (h=1, v=2) — full-resolution luma in 4:4:0
fn compute_jpeg_upsampling(jpeg: &JpegData, jpeg_c_map: &[usize; 3]) -> [u8; 3] {
    let mut upsampling = [0u8; 3];
    for jxl_c in 0..3 {
        let jpeg_c = jpeg_c_map[jxl_c];
        let h = jpeg.components[jpeg_c].h_samp_factor;
        let v = jpeg.components[jpeg_c].v_samp_factor;
        // Store the sampling factor as log2 (matching libjxl convention)
        let hs = h.trailing_zeros();
        let vs = v.trailing_zeros();
        upsampling[jxl_c] = match (hs > 0, vs > 0) {
            (false, false) => 0,
            (true, true) => 1,
            (true, false) => 2,
            (false, true) => 3,
        };
    }
    upsampling
}

/// Build the JXL frame header for JPEG reencoding.
fn build_jpeg_frame_header(jpeg: &JpegData, jpeg_upsampling: [u8; 3]) -> FrameHeader {
    // libjxl always uses YCbCr for JPEG reencoding, including grayscale.
    // For grayscale (1 component), kYCbCr is forced in SetColorTransformFromJpegData.
    let is_ycbcr = jpeg.component_type == JpegComponentType::YCbCr || jpeg.components.len() == 1;
    FrameHeader {
        encoding: Encoding::VarDct,
        xyb_encoded: false,
        do_ycbcr: is_ycbcr,
        jpeg_upsampling,
        flags: 0x80, // SKIP_ADAPTIVE_LF_SMOOTHING
        gaborish: false,
        epf_iters: 0,
        x_qm_scale: 2,
        b_qm_scale: 2,
        ..FrameHeader::default()
    }
}

/// Write DC global section for JPEG reencoding.
///
/// Unlike the normal VarDCT path, JPEG reencoding uses:
/// - Custom DC dequantization values (not default)
/// - global_scale=65536, quant_dc=1
fn write_dc_global_jpeg(
    dc_dequant: &[f32; 3],
    dc_code: &OwnedAnsEntropyCode,
    num_dc_groups: usize,
    writer: &mut BitWriter,
) -> Result<()> {
    // No noise params for JPEG reencoding

    // DequantDC: custom values (not default)
    // The F16 value stored is dc_dequant[c] * 128.0
    // Decoder reads this and uses it directly in: scale = m_lf * 512 / (global_scale * quant_lf)
    writer.write(1, 0)?; // not all_default
    for &dcq in dc_dequant.iter() {
        write_f16(dcq * 128.0, writer)?;
    }

    // Quantizer params: global_scale=65536, quant_dc=1
    write_quant_scales(65536, 1, writer)?;

    // BlockCtxMap: write default (non-default header, but default compact map)
    writer.write(1, 0)?; // non-default BlockCtxMap
    writer.write(16, 0)?; // no dc ctx, no qft
    crate::vardct::context_tree::write_block_context_map(writer)?;

    // LfChannelCorrelation (CfL DC params)
    // For YCbCr mode, base_correlation_b must be 0.0 (not the XYB default of 1.0).
    // The default (all_default=1) uses base_correlation_b=1.0 which adds Y into the
    // Cr channel, corrupting chroma. We must write all_default=0 explicitly.
    writer.write(1, 0)?; // not all_default
    writer.write(2, 0)?; // colour_factor = 84 (U32 selector 0)
    write_f16(0.0, writer)?; // base_correlation_x = 0.0
    write_f16(0.0, writer)?; // base_correlation_b = 0.0 (NOT the XYB default 1.0)
    writer.write(8, 128)?; // x_factor_lf = 128 (signed 0)
    writer.write(8, 128)?; // b_factor_lf = 128 (signed 0)

    // Context tree for modular DC header
    crate::vardct::context_tree::write_context_tree(num_dc_groups, writer)?;

    // LZ77: disabled
    writer.write(1, 0)?;

    // DC entropy code
    write_entropy_code_ans(dc_code, writer)?;

    Ok(())
}

/// Write AC global section for JPEG reencoding.
///
/// Unlike normal VarDCT, this writes RAW quant matrices (not all_default).
fn write_ac_global_jpeg(
    raw_qtables: &[i32],
    num_groups: usize,
    ac_code: &OwnedAnsEntropyCode,
    used_orders: u32,
    coeff_order_tokens: Option<&[Token]>,
    writer: &mut BitWriter,
) -> Result<()> {
    // RAW quant matrices with JPEG quant tables
    writer.write(1, 0)?; // not all_default
    write_quant_matrices_jpeg(raw_qtables, writer)?;

    // num_histograms
    let num_histo_bits = ceil_log2_nonzero(num_groups);
    if num_histo_bits != 0 {
        writer.write(num_histo_bits as usize, 0)?;
    }

    // EX-J17a: write used_orders via u2S(0x5F, 0x13, 0x00, U(13)).
    // Mirrors VarDCT bitstream.rs:986-997 so the decoder reads the same
    // selector encoding on both paths.
    match used_orders {
        0x5F => writer.write(2, 0)?, // selector 0 = 0x5F
        0x13 => writer.write(2, 1)?, // selector 1 = 0x13
        0 => writer.write(2, 2)?,    // selector 2 = 0x00 (no custom orders)
        other => {
            writer.write(2, 3)?; // selector 3 = U(13)
            writer.write(13, other as u64)?;
        }
    }

    // EX-J17a: write the Lehmer permutation tokens for any (bucket, channel)
    // pair flagged in used_orders. Must come after used_orders and before the
    // AC entropy code header, matching the VarDCT write_ac_global flow.
    if let Some(tokens) = coeff_order_tokens.filter(|_| used_orders != 0) {
        // Always ANS on the JPEG path (use_ans=true), matching ac_code below.
        build_and_write_coeff_orders(tokens, true, writer)?;
    }

    // LZ77: disabled
    writer.write(1, 0)?;

    // AC entropy code
    write_entropy_code_ans(ac_code, writer)?;

    Ok(())
}

/// Write quantization matrices for JPEG reencoding.
///
/// Table 0 (DCT8) uses RAW mode with the JPEG quant tables.
/// Tables 1-16 use Library mode (predefined index 0).
fn write_quant_matrices_jpeg(raw_qtables: &[i32], writer: &mut BitWriter) -> Result<()> {
    for table_idx in 0..NUM_QUANT_TABLES {
        if table_idx == 0 {
            // RAW mode for DCT8
            writer.write(3, 7)?; // mode = kQuantModeRAW (7)

            // Write qtable_den as F16: 1.0 / (8 * 255) = 1/2040
            let qtable_den = 1.0f32 / (8.0 * 255.0);
            write_f16(qtable_den, writer)?;

            // Write the 8x8x3 quant table values as a modular sub-bitstream
            write_raw_quant_table_modular(raw_qtables, writer)?;
        } else {
            // Library mode (predefined table 0) for all other strategies
            // kCeilLog2NumPredefinedTables = 0, so no additional bits needed
            writer.write(3, 0)?; // mode = kQuantModeLibrary (0)
        }
    }
    Ok(())
}

/// Write a raw quant table as a modular-encoded 8x8 image with 3 channels.
///
/// This is a standalone modular sub-bitstream within the AC global section.
/// Structure: GroupHeader → MA tree (Decoder::parse with 6 ctx) → tree tokens
///          → Data entropy (Decoder::parse with 1 ctx) → data tokens
///
/// CRITICAL: When num_dist=1 (single leaf tree → 1 context for data), the decoder's
/// read_clusters() returns immediately without reading any context map bits.
/// We must NOT write a context map for the data entropy code.
fn write_raw_quant_table_modular(qtables: &[i32], writer: &mut BitWriter) -> Result<()> {
    use crate::modular::channel::{Channel, ModularImage};
    use crate::modular::section::collect_all_residuals;

    // Create a 3-channel 8x8 ModularImage from the quant table data
    let mut channels = Vec::with_capacity(3);
    for c in 0..3 {
        let data: Vec<i32> = (0..64).map(|i| qtables[c * 64 + i]).collect();
        channels.push(Channel::from_vec(data, 8, 8)?);
    }
    let image = ModularImage {
        channels,
        bit_depth: 8,
        is_grayscale: false,
        has_alpha: false,
    };

    // Collect gradient residuals using existing infrastructure
    let (residuals, _max_residual) = collect_all_residuals(&image);

    // GroupHeader: use_global_tree=false, wp_params default, no transforms
    writer.write(1, 0)?; // use_global_tree = false
    writer.write(1, 1)?; // wp_params all_default = true
    writer.write(2, 0)?; // nb_transforms = 0

    // Write tree entropy code (Decoder::parse with 6 contexts → writes context map)
    let (tree_depths, tree_codes) =
        crate::modular::encode::write_tree_histogram_for_gradient(writer)?;
    // Write tree tokens (single leaf: property=0, predictor=Gradient, offset=0, mul=1)
    crate::modular::encode::write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

    // Build ANS code and write data entropy header.
    // Uses write_ans_modular_header which correctly skips context map when num_dist=1.
    let (tokens, code) = crate::modular::encode::build_ans_modular_code(&residuals);
    crate::modular::encode::write_ans_modular_header(writer, &code)?;

    // Write data tokens
    crate::modular::encode::write_ans_modular_tokens(writer, &tokens, &code)?;

    Ok(())
}

/// Write an empty modular global sub-bitstream (no alpha, no extra channels).
// F16 functions delegated to shared f16 module.
#[cfg(not(test))]
use crate::f16::write_f16;
#[cfg(test)]
use crate::f16::{f32_to_f16_bits, write_f16};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        // 1.0 = 0x3C00 in f16
        assert_eq!(f32_to_f16_bits(1.0).unwrap(), 0x3C00);
        // 0.0 = 0x0000
        assert_eq!(f32_to_f16_bits(0.0).unwrap(), 0x0000);
        // -1.0 = 0xBC00
        assert_eq!(f32_to_f16_bits(-1.0).unwrap(), 0xBC00);
        // 1/2040 ≈ 0.0004902
        let qtable_den = 1.0f32 / 2040.0;
        let bits = f32_to_f16_bits(qtable_den).unwrap();
        // Should be a small positive denormalized or small normal value
        assert!(
            bits > 0 && bits < 0x4000,
            "qtable_den f16 bits = 0x{bits:04X}"
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "corpus-tests"),
        ignore = "requires codec corpus; enable `corpus-tests` feature"
    )]
    fn test_encode_real_jpeg() {
        crate::skip_without_corpus!();
        let path = format!(
            "{}/imageflow/test_inputs/orientation/Landscape_1.jpg",
            crate::test_helpers::corpus_dir().display()
        );
        let data = std::fs::read(path).expect("failed to read test JPEG");
        let jpeg = super::super::parse::read_jpeg(&data).expect("failed to parse JPEG");
        let jxl = encode_jpeg_to_jxl(&jpeg).expect("failed to encode JPEG to JXL");
        assert!(jxl.len() > 10, "JXL output too short: {} bytes", jxl.len());
        // Verify JXL signature
        assert_eq!(jxl[0], 0xFF);
        assert_eq!(jxl[1], 0x0A);
        eprintln!(
            "Encoded {}x{} JPEG to {} bytes JXL",
            jpeg.width,
            jpeg.height,
            jxl.len()
        );

        // Save for manual inspection
        crate::test_helpers::save_test_output("jpeg-reencoding", "landscape1.jxl", &jxl);
    }

    #[test]
    fn test_encode_420_jpeg() {
        let path =
            crate::test_helpers::output_dir_for("jpeg-reencoding", "").join("test128_420.jpg");
        let data = std::fs::read(&path).expect("failed to read test JPEG");
        let jpeg = super::super::parse::read_jpeg(&data).expect("failed to parse JPEG");

        // Verify it's actually 4:2:0
        assert_eq!(jpeg.components[0].h_samp_factor, 2);
        assert_eq!(jpeg.components[0].v_samp_factor, 2);
        assert_eq!(jpeg.components[1].h_samp_factor, 1);
        assert_eq!(jpeg.components[1].v_samp_factor, 1);

        let jxl = encode_jpeg_to_jxl(&jpeg).expect("failed to encode 4:2:0 JPEG to JXL");
        assert!(jxl.len() > 10, "JXL output too short: {} bytes", jxl.len());
        assert_eq!(jxl[0], 0xFF);
        assert_eq!(jxl[1], 0x0A);
        eprintln!(
            "Encoded {}x{} 4:2:0 JPEG to {} bytes JXL",
            jpeg.width,
            jpeg.height,
            jxl.len()
        );
    }

    #[test]
    fn test_compute_jpeg_upsampling() {
        // Build a fake JpegData for 4:2:0 (Y: h=2,v=2; Cb: h=1,v=1; Cr: h=1,v=1)
        let jpeg = JpegData {
            width: 128,
            height: 128,
            restart_interval: 0,
            app_data: Vec::new(),
            app_marker_type: Vec::new(),
            com_data: Vec::new(),
            quant: Vec::new(),
            huffman_code: Vec::new(),
            components: vec![
                JpegComponent {
                    id: 1,
                    h_samp_factor: 2,
                    v_samp_factor: 2,
                    quant_idx: 0,
                    width_in_blocks: 16,
                    height_in_blocks: 16,
                    coeffs: Vec::new(),
                },
                JpegComponent {
                    id: 2,
                    h_samp_factor: 1,
                    v_samp_factor: 1,
                    quant_idx: 1,
                    width_in_blocks: 8,
                    height_in_blocks: 8,
                    coeffs: Vec::new(),
                },
                JpegComponent {
                    id: 3,
                    h_samp_factor: 1,
                    v_samp_factor: 1,
                    quant_idx: 1,
                    width_in_blocks: 8,
                    height_in_blocks: 8,
                    coeffs: Vec::new(),
                },
            ],
            scan_info: Vec::new(),
            marker_order: Vec::new(),
            inter_marker_data: Vec::new(),
            tail_data: Vec::new(),
            has_zero_padding_bit: false,
            padding_bits: Vec::new(),
            component_type: JpegComponentType::YCbCr,
        };
        // JXL c_map: [1,0,2] for YCbCr (c0=Cb, c1=Y, c2=Cr)
        let c_map = [1usize, 0, 2];
        let up = compute_jpeg_upsampling(&jpeg, &c_map);
        // JXL stores the sampling FACTOR (log2), not the shift.
        // c0=Cb (h=1,v=1) → factor 1x1 → mode 0
        // c1=Y (h=2,v=2) → factor 2x2 → mode 1
        // c2=Cr (h=1,v=1) → factor 1x1 → mode 0
        assert_eq!(up, [0, 1, 0], "expected [0,1,0] for 4:2:0 YCbCr");

        // Test 4:2:2 (Y: h=2,v=1; Cb/Cr: h=1,v=1)
        let mut jpeg_422 = jpeg.clone();
        jpeg_422.components[0].v_samp_factor = 1;
        let up_422 = compute_jpeg_upsampling(&jpeg_422, &c_map);
        // c0=Cb (h=1,v=1) → mode 0
        // c1=Y (h=2,v=1) → factor 2x1 → mode 2
        // c2=Cr (h=1,v=1) → mode 0
        assert_eq!(up_422, [0, 2, 0], "expected [0,2,0] for 4:2:2 YCbCr");

        // Test 4:4:0 (Y: h=1,v=2; Cb/Cr: h=1,v=1)
        let mut jpeg_440 = jpeg.clone();
        jpeg_440.components[0].h_samp_factor = 1;
        let up_440 = compute_jpeg_upsampling(&jpeg_440, &c_map);
        // c0=Cb (h=1,v=1) → mode 0
        // c1=Y (h=1,v=2) → factor 1x2 → mode 3
        // c2=Cr (h=1,v=1) → mode 0
        assert_eq!(up_440, [0, 3, 0], "expected [0,3,0] for 4:4:0 YCbCr");

        // Test 4:4:4 (all h=1,v=1)
        let mut jpeg_444 = jpeg.clone();
        jpeg_444.components[0].h_samp_factor = 1;
        jpeg_444.components[0].v_samp_factor = 1;
        let up_444 = compute_jpeg_upsampling(&jpeg_444, &c_map);
        assert_eq!(up_444, [0, 0, 0], "expected [0,0,0] for 4:4:4 YCbCr");
    }
}
