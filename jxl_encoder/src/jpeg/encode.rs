// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! JPEG lossless reencoding into JPEG XL VarDCT format.
//!
//! Converts parsed JPEG data (from `read_jpeg`) into a JXL codestream that
//! preserves the exact quantized DCT coefficients. The resulting JXL file
//! decodes to pixel-identical output as the original JPEG.

use super::data::*;
use super::jbrd::{encode_jbrd, extract_exif, extract_xmp};
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
    let jpeg_c_map: [usize; 3] = match jpeg.component_type {
        JpegComponentType::YCbCr => [1, 0, 2], // JXL c0←JPEG Cb, c1←JPEG Y, c2←JPEG Cr
        _ => [0, 1, 2],                        // RGB or other: identity mapping
    };

    let num_components = jpeg.components.len();
    if num_components != 3 && num_components != 1 {
        return Err(crate::error::Error::InvalidInput(format!(
            "JPEG reencoding requires 1 or 3 components, got {num_components}"
        )));
    }

    // Determine block dimensions from the Y component (or first component)
    let y_jpeg_c = if num_components == 3 {
        jpeg_c_map[1]
    } else {
        0
    };
    let xsize_blocks = jpeg.components[y_jpeg_c].width_in_blocks as usize;
    let ysize_blocks = jpeg.components[y_jpeg_c].height_in_blocks as usize;

    // Map JPEG coefficients to JXL data structures
    let (quant_dc, quant_ac, nzeros, raw_nzeros) =
        map_jpeg_coefficients(jpeg, &jpeg_c_map, xsize_blocks, ysize_blocks)?;

    // All blocks use DCT8
    let ac_strategy = AcStrategyMap::new_dct8(xsize_blocks, ysize_blocks);

    // No chroma-from-luma for JPEG mode (YCbCr handles color decorrelation)
    let xsize_tiles = div_ceil(xsize_blocks, TILE_DIM_IN_BLOCKS);
    let ysize_tiles = div_ceil(ysize_blocks, TILE_DIM_IN_BLOCKS);
    let cfl_map = CflMap::zeros(xsize_tiles, ysize_tiles);

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

        let dc_tokens = collect_dc_tokens_region(&quant_dc, start_bx, start_by, end_bx, end_by);
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
                    let qf_val = quant_field[by * xsize_blocks + bx] as u32;
                    let block_ctx = block_ctx_map.block_context(c, strategy_code, qf_val);
                    collect_ac_coefficients_into(
                        &mut tokens,
                        &quant_ac[c][by][bx],
                        raw_strategy,
                        nz,
                        predicted_nz,
                        block_ctx,
                        block_ctx_map.num_ctxs,
                        None, // no custom coefficient order
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

    // File header (write() includes the signature)
    let file_header = build_jpeg_file_header(width, height);
    file_header.write(&mut writer)?;
    writer.zero_pad_to_byte();
    let file_header_bytes = writer.bytes_written();

    // Frame header
    let frame_header = build_jpeg_frame_header(jpeg);
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
    write_ac_global_jpeg(&raw_qtables, num_groups, &ac_code, &mut ac_global)?;

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
#[allow(clippy::type_complexity)]
fn map_jpeg_coefficients(
    jpeg: &JpegData,
    jpeg_c_map: &[usize; 3],
    xsize_blocks: usize,
    ysize_blocks: usize,
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
        let comp_xb = comp.width_in_blocks as usize;
        let comp_yb = comp.height_in_blocks as usize;

        // For 4:4:4, comp dimensions match Y dimensions.
        // TODO: handle subsampled JPEG (jpeg_upsampling field)
        let xb = comp_xb.min(xsize_blocks);
        let yb = comp_yb.min(ysize_blocks);

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
                let blk_idx = by * comp_xb + bx;
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
fn build_jpeg_file_header(width: usize, height: usize) -> FileHeader {
    let color_encoding = ColorEncoding::srgb(); // Perceptual rendering intent → all_default

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
    }
}

/// Build the JXL frame header for JPEG reencoding.
fn build_jpeg_frame_header(jpeg: &JpegData) -> FrameHeader {
    let is_ycbcr = jpeg.component_type == JpegComponentType::YCbCr;
    FrameHeader {
        encoding: Encoding::VarDct,
        xyb_encoded: false,
        do_ycbcr: is_ycbcr,
        jpeg_upsampling: [0; 3], // 4:4:4 (no upsampling)
        flags: 0x80,             // SKIP_ADAPTIVE_LF_SMOOTHING
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

    // used_orders via u2S(0x5F, 0x13, 0x00, U(13)): 0 = no custom orders
    writer.write(2, 2)?; // selector 2 = 0x00 (no custom orders)

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
/// Encode an f32 value as IEEE 754 half-precision (16 bits).
fn write_f16(value: f32, writer: &mut BitWriter) -> Result<()> {
    let bits = f32_to_f16_bits(value);
    writer.write(16, bits as u64)?;
    Ok(())
}

/// Convert f32 to IEEE 754 binary16 (half-precision) bit representation.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;

    if exp == 0 && mantissa == 0 {
        // Zero
        return (sign << 15) as u16;
    }

    if exp == 0xFF {
        // Inf or NaN - clamp to max finite
        return ((sign << 15) | 0x7BFF) as u16;
    }

    // Rebias exponent: f32 bias=127, f16 bias=15
    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        // Overflow → max finite f16
        return ((sign << 15) | 0x7BFF) as u16;
    }

    if new_exp <= 0 {
        // Denormalized or underflow
        if new_exp < -10 {
            // Too small
            return (sign << 15) as u16;
        }
        // Denormalized
        let m = mantissa | 0x80_0000;
        let shift = 1 - new_exp;
        let half_mantissa = (m >> (13 + shift)) as u16;
        return ((sign << 15) as u16) | half_mantissa;
    }

    // Normal case: round mantissa from 23 bits to 10 bits
    let half_mantissa = (mantissa >> 13) as u16;
    let half_exp = (new_exp as u16) << 10;
    ((sign << 15) as u16) | half_exp | half_mantissa
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        // 1.0 = 0x3C00 in f16
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
        // 0.0 = 0x0000
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        // -1.0 = 0xBC00
        assert_eq!(f32_to_f16_bits(-1.0), 0xBC00);
        // 1/2040 ≈ 0.0004902
        let qtable_den = 1.0f32 / 2040.0;
        let bits = f32_to_f16_bits(qtable_den);
        // Should be a small positive denormalized or small normal value
        assert!(
            bits > 0 && bits < 0x4000,
            "qtable_den f16 bits = 0x{bits:04X}"
        );
    }

    #[test]
    fn test_encode_real_jpeg() {
        let path =
            "/home/lilith/work/codec-corpus/imageflow/test_inputs/orientation/Landscape_1.jpg";
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
}
