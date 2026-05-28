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
    OwnedAnsEntropyCode, build_entropy_code_ans_with_options, write_entropy_code_ans,
    write_tokens_ans,
};
use crate::entropy_coding::token::Token;
use crate::error::Result;
use crate::headers::color_encoding::ColorEncoding;
use crate::headers::file_header::{BitDepth, FileHeader, ImageMetadata};
use crate::headers::frame_header::{Encoding, FrameHeader};
use crate::vardct::ac_context;
use crate::vardct::ac_group::{collect_ac_coefficients_into, predict_from_top_and_left};
use crate::vardct::ac_strategy::AcStrategyMap;
use crate::vardct::chroma_from_luma::{CFL_FIXED_POINT_PRECISION, CflMap, DEFAULT_COLOR_FACTOR};
use crate::vardct::coeff_order::{
    NUM_ORDER_BUCKETS as NUM_ORDER_BUCKETS_JPEG, build_and_write_coeff_orders,
    compute_custom_orders, get_custom_order, tokenize_coeff_orders,
};
use crate::vardct::common::*;
use crate::vardct::dc_coding::{
    NUM_DC_CONTEXTS, NUM_DC_CONTEXTS_JPEG_TRANSCODE, collect_ac_metadata_tokens_region,
    collect_ac_metadata_tokens_region_jpeg_transcode, collect_dc_tokens_region,
    collect_dc_tokens_region_jpeg_transcode,
};
use crate::vardct::frame::{
    assemble_frame_sections, write_dc_group_from_tokens, write_quant_scales,
};

/// Lever A (2026-05-28): when `true`, the JPEG path emits the libjxl-style
/// `kJpegTranscodeACMeta` context tree shape (single `Leaf(Zero)` for the
/// AC-metadata subtree, gradient-DC subtree unchanged) and emits all
/// AC-metadata data tokens as raw values in a single context.
///
/// **HONEST-STOP — default OFF, OPT-IN ONLY.** A 200-file paired bench
/// (`benchmarks/jpeg_bit_shaving_2026-05-28.tsv`) measured a **+10.7 %
/// regression** vs the existing gradient-fixed AC-metadata path at N=20:
/// 1.42 % → 10.71 % vs cjxl. Root cause: our CFL `i8` multipliers (range
/// ~±30 around `kOffset=127`) compress far better as gradient/left
/// residuals (typical |Δ| 1-5) than as raw values (|val| 10-50). The
/// `dc_global` section shrinks ~29 B/file from the smaller tree, but the
/// CFL data tokens in the DC groups balloon ~15 KB/file. Net regression
/// of ~15 380 B/file (+6.3 % per file).
///
/// libjxl pays this same cost on its JPEG transcode path; we are
/// *already* better than libjxl on the AC-metadata stream by using a
/// richer tree. Keep the API surface so future investigators can A/B
/// measure without re-implementing ~200 LOC of token collectors.
///
/// Set `JPEG_LEVER_A_ENABLE=1` to opt in.
fn jpeg_transcode_tree_enabled() -> bool {
    matches!(std::env::var_os("JPEG_LEVER_A_ENABLE"), Some(v) if v == "1")
}

/// Number of JXL quant tables (from libjxl quant_weights.h).
const NUM_QUANT_TABLES: usize = 17;

/// Encode a parsed JPEG as a JXL codestream (lossless reencoding).
///
/// The output JXL will decode to pixel-identical results as the original JPEG.
/// This does NOT include the jbrd box — it produces a bare JXL codestream.
/// For byte-exact JPEG reconstruction, wrap in a container with a jbrd box.
///
/// Uses the default effort level (7). For caller-supplied effort (which gates
/// AC-stream LZ77 + pair-merge clustering at effort >= 8), use
/// [`encode_jpeg_to_jxl_with_effort`].
pub fn encode_jpeg_to_jxl(jpeg: &JpegData) -> Result<Vec<u8>> {
    encode_jpeg_to_jxl_with_effort(jpeg, 7)
}

/// Encode a parsed JPEG as a JXL codestream at the given effort level.
///
/// At `effort >= 9`, enables kBest pair-merge histogram clustering on the AC
/// code (a partial port of libjxl's JPEG-mode VarDCT path at
/// `speed_tier <= kTortoise`; `enc_frame.cc:1267-1271` +
/// `enc_ans_params.h:60-75`). Measured -0.27 % avg vs the default-effort path
/// on a 10-file product-images corpus (2026-05-28).
///
/// libjxl additionally enables `uint_method = kBest` and RLE LZ77 at e9; both
/// are currently DEFAULT-OFF on our path because:
/// - Our `optimize_uint_configs_best_from_freqs` diverges from libjxl on JPEG
///   AC streams and regresses bytes by +0.5 % (measured). Filed for future
///   investigation; env hook `JPEG_E9_FORCE_UINT_OPT=1` re-enables.
/// - RLE LZ77's global savings threshold doesn't pass on the JPEG AC token
///   streams we produce; the multi-section helper is wired but never accepted.
///   Env hook `JPEG_E9_FORCE_LZ77=1` re-enables.
///
/// The DC code is left at the simple defaults regardless of effort — DC token
/// counts are tiny and clustering / LZ77 overhead would dominate.
///
/// Effort 0-8 is byte-identical to [`encode_jpeg_to_jxl`] (which uses effort 7).
pub fn encode_jpeg_to_jxl_with_effort(jpeg: &JpegData, effort: u8) -> Result<Vec<u8>> {
    let (codestream, _split) = encode_jpeg_to_jxl_inner(jpeg, effort)?;
    Ok(codestream)
}

/// Inner function that returns both codestream bytes and the file header size
/// (split point for jxlp box splitting when JBRD is needed).
fn encode_jpeg_to_jxl_inner(jpeg: &JpegData, effort: u8) -> Result<(Vec<u8>, usize)> {
    let width = jpeg.width as usize;
    let height = jpeg.height as usize;

    // Channel mapping: JXL c0=Cb, c1=Y, c2=Cr for YCbCr-flavored frames.
    //   - For semantic-YCbCr (3-comp, libjxl `color_transform == kYCbCr`):
    //     swap to put JPEG Y into the JXL c1 (luma) slot.
    //   - For semantic-RGB (3-comp, `color_transform == kNone`):
    //     identity — c0=R, c1=G, c2=B.
    //   - For grayscale (1-comp): all JXL channels reference comp[0]
    //     and chroma is zero-filled later.
    //
    // This MUST be driven by the semantic color transform (detected
    // from JFIF / Adobe APP14 / component IDs) and NOT by the JBRD
    // `component_type` tag, since the tag is purely a serialization
    // shortcut (only fires when IDs are literally (1,2,3) or
    // ('R','G','B')) and can be `Custom` for semantically-YCbCr
    // JPEGs with non-(1,2,3) IDs.
    let is_ycbcr_color_transform = detect_ycbcr_color_transform(jpeg);
    let jpeg_c_map: [usize; 3] = if jpeg.components.len() == 1 {
        [0, 0, 0] // grayscale: all channels reference component 0
    } else if is_ycbcr_color_transform {
        [1, 0, 2] // JXL c0←JPEG Cb, c1←JPEG Y, c2←JPEG Cr
    } else {
        [0, 1, 2] // RGB or other: identity mapping
    };

    let num_components = jpeg.components.len();
    if num_components != 3 && num_components != 1 {
        // The JXL spec's JBRD `num_components` U32 field encodes
        // `Val(1), Val(2), Val(3), Val(4)` — but the reference libjxl
        // decoder rejects any value other than 1 or 3 at
        // `lib/jxl/jpeg/jpeg_data.cc:180-182` with the same wording.
        // Encoding a 4-component (CMYK/YCCK) JPEG to JXL would produce
        // a bitstream that libjxl `cjxl --lossless_jpeg=1` itself
        // refuses to emit (`encode.cc:2131`: "Unsupported JPEG feature
        // (CMYK, arithmetic coding, etc.)") AND that `djxl
        // --reconstruct_jpeg` refuses to round-trip back. There is
        // therefore no spec-compatible path for 4-component JPEG
        // transcoding. Pixel-domain CMYK encoding (XYB + `kBlack`
        // extra channel) IS in the spec for non-JPEG sources, but it
        // requires a CMYK→RGB color-managed conversion that DOES NOT
        // round-trip back to byte-identical JPEG.
        return Err(crate::error::Error::InvalidInput(format!(
            "JPEG reencoding does not support {num_components}-component JPEGs (CMYK/YCCK): \
             the JBRD reconstruction format requires num_components ∈ {{1, 3}} per JXL spec \
             (libjxl jpeg_data.cc:180-182). Decode CMYK to RGB before encoding."
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

    // The DC offset is keyed on the wire-format `color_transform`
    // value (`is_ycbcr_color_transform` above):
    //   color_transform == kYCbCr → offset = 0
    //   color_transform == kNone  → offset = 1024 / qt_dc[c]
    // matching the decoder's `dcoff` (`dec_group.cc:244-247`,
    // applied as `jpeg_pos[0] = dc_rows[c] - dcoff[c]` on reconstruct).
    // libjxl forces grayscale to kYCbCr even with the kNone-style
    // detection because the chroma planes are zero-filled (no offset
    // matters) and the existing fast paths assume zero `dcoff`.
    let is_ycbcr_for_dc_offset = is_ycbcr_color_transform || jpeg.components.len() == 1;
    let dc_offset = compute_jpeg_dc_offset(jpeg, &jpeg_c_map, is_ycbcr_for_dc_offset);

    // Map JPEG coefficients to JXL data structures
    // Each channel uses its native block dimensions (may differ for subsampled chroma)
    let (mut quant_dc, mut quant_ac, mut nzeros, mut raw_nzeros) =
        map_jpeg_coefficients(jpeg, &jpeg_c_map, &dc_offset)?;

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
        // Build scaled_qtable per chroma channel for the JPEG-CfL search.
        //
        // libjxl reference (`enc_frame.cc:830-841`):
        //   for y, x in 0..8:
        //       coeffpos = y*8 + x                              // NATURAL
        //       scaled_qtable[c*64 + 8*x+y]                     // store slot
        //         = (1<<11) * qt[64 + coeffpos] / qt[c*64 + coeffpos]
        // where `qt` is libjxl's transposed-stored quant table
        // (`qt[c*64 + 8*x+y] = jpeg_quant[8*y+x]`, line 819).
        //
        // The CfL search then computes `row_m[coeffpos] * scaled_qtable[coeffpos]`
        // where `row_m` is *natural-order* JPEG coefficients (line 897-898).
        // Substituting the qt definition: `scaled_qtable[c*64 + i] =
        // (1<<11) * jpeg_natural_qy[i] / jpeg_natural_qc[i]` (since
        // both numerator and denominator transpose by the same amount,
        // the transpose cancels under like-indexing).
        //
        // OUR storage convention differs from libjxl in ONE place:
        // `quant_ac` is *also* transposed (`block[x*8+y] = JPEG[y*8+x]`,
        // see `map_jpeg_coefficients` line 619), while libjxl keeps
        // `row_m` in natural order. Our `build_raw_qtables` already
        // matches libjxl's transposed `qt` layout (line 660).
        //
        // Since our `luma_block[s] = jpeg_natural_y_coeff[transpose(s)]`,
        // the libjxl pairing translates to: `scaled_qtable[s]` must
        // equal `(1<<11) * jpeg_natural_qy[transpose(s)] /
        // jpeg_natural_qc[transpose(s)]`. With our transposed-stored
        // `qt`, the expression `qt[64 + s] / qt[c*64 + s]` already
        // evaluates to exactly that (both terms transpose, both reads
        // hit the same `s` slot). So the storage layout collapses:
        // index slot `s` directly on both sides.
        //
        // W44-161 (issue #63): pre-WIP code used
        // `scaled_qtable[c][coeffpos] = qt[64+transposed] / qt[c*64+transposed]`
        // — that paired `natural` qt-ratio storage slot with
        // `transposed` coefficient storage slot, giving the search
        // `coeff[transpose(s)] * ratio_at_natural(s)`, a mixed pairing.
        // W44-160 WIP swapped it to `scaled_qtable[c][transposed] =
        // qt[64+natural] / qt[c*64+natural]` — equivalent mismatch
        // in the opposite direction (same bug).
        //
        // Correct: `scaled_qtable[c][s] = (1<<11) * qt[64+s] / qt[c*64+s]`.
        let qt = build_raw_qtables(jpeg, &jpeg_c_map)?;
        let mut scaled_qtable = [[0i32; 64]; 3];
        for c in 0..3 {
            for s in 0..64usize {
                let qy = qt[64 + s];
                let qc = qt[64 * c + s];
                if qc != 0 {
                    scaled_qtable[c][s] = ((1 << 11) * qy) / qc;
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

        // W44-161 (issue #63): apply JPEG-CfL correction to chroma
        // coefficients. The decoder reads the per-tile multipliers
        // from `cfl_map` and ADDS the predicted chroma values during
        // reconstruction. Our encoded chroma must therefore carry the
        // *residual* (`QChroma - cfl_factor`), not the raw JPEG
        // coefficients, or the decoder's reconstructed chroma drifts
        // away from the original.
        //
        // Mirrors libjxl `enc_frame.cc:1015-1037`. libjxl iterates
        // natural (y, x) and writes the result to a transposed slot
        // `block[x*8+y]`. Our `quant_ac` is *already* transposed-
        // stored (`block[x*8+y] = JPEG_natural[y*8+x]`, see
        // `map_jpeg_coefficients` line 619), so we can iterate
        // storage slot `s` directly on both input (Y, QChroma) and
        // output (modified chroma) sides — both transposes cancel.
        //
        // Skip coeffpos=0 (DC) — libjxl CfL only modifies AC.
        let cfl_offset_x = 127i32;
        let fp_round = 1i32 << (CFL_FIXED_POINT_PRECISION - 1);
        for ty in 0..ysize_tiles {
            for tx in 0..xsize_tiles {
                let y0 = ty * TILE_DIM_IN_BLOCKS;
                let x0 = tx * TILE_DIM_IN_BLOCKS;
                let y1 = ((ty + 1) * TILE_DIM_IN_BLOCKS).min(ysize_blocks);
                let x1 = ((tx + 1) * TILE_DIM_IN_BLOCKS).min(xsize_blocks);
                let tile_idx = ty * xsize_tiles + tx;
                let ytox = map.ytox[tile_idx] as i32;
                let ytob = map.ytob[tile_idx] as i32;
                // RatioJPEG(factor) = factor << 11 / 84 — matches
                // libjxl `ColorCorrelation::RatioJPEG(cm[...])` at
                // line 1016 with `cm` = the raw signed-byte tile
                // value (kOffset already pre-applied by the decoder
                // when it computes the YtoX/YtoB ratio).
                let scale_x = (ytox * (1 << CFL_FIXED_POINT_PRECISION)) / DEFAULT_COLOR_FACTOR;
                let scale_b = (ytob * (1 << CFL_FIXED_POINT_PRECISION)) / DEFAULT_COLOR_FACTOR;
                let _ = cfl_offset_x; // kOffset=127 is applied by the search; map[] already stores the signed delta
                for by in y0..y1 {
                    for bx in x0..x1 {
                        // Y luma block (read-only; never modified)
                        let y_block = quant_ac[1][by][bx];
                        // X channel (c=0) and B channel (c=2) blocks
                        for &(c, scale) in &[(0usize, scale_x), (2usize, scale_b)] {
                            let chroma_block = &mut quant_ac[c][by][bx];
                            let mut nz_count: u16 = 0;
                            // Coefficient slot 0 = DC, unchanged in `quant_ac`
                            // (DC lives in `quant_dc`); skip s=0.
                            for s in 1..64usize {
                                let yv = y_block[s];
                                let coeff_scale = (scale * scaled_qtable[c][s] + fp_round)
                                    >> CFL_FIXED_POINT_PRECISION;
                                let cfl_factor =
                                    (yv * coeff_scale + fp_round) >> CFL_FIXED_POINT_PRECISION;
                                chroma_block[s] -= cfl_factor;
                                if chroma_block[s] != 0 {
                                    nz_count += 1;
                                }
                            }
                            // Recompute nzeros / raw_nzeros for the modified
                            // chroma block (subtraction may have created or
                            // destroyed zero coefficients).
                            raw_nzeros[c][by][bx] = nz_count;
                            nzeros[c][by][bx] = nz_count as u8;
                        }
                    }
                }
            }
        }

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
    let use_lever_a = jpeg_transcode_tree_enabled();
    // EX-J31 (2026-05-28): default to the Weighted-Predictor DC path
    // (kWPFixedDC), matching libjxl JPEG-transcode at speed_tier >= kSquirrel
    // (cjxl effort 7). libjxl encodes the JPEG DC stream with
    // `Predictor::Weighted` + a fixed BSP tree splitting on `wp_max_error`
    // (`enc_modular.cc:1584-1589`, `enc_encoding.cc:533-540`). We previously
    // used a clamped-gradient predictor, which produced ~+23% larger LfGroup
    // (DC) sections — the entire measured JPEG-in-JXL size gap vs cjxl lives
    // here (AC data is at parity). Env hook `JPEG_GRADIENT_DC=1` restores the
    // gradient path for A/B; lever A (JPEG_LEVER_A_ENABLE) is unaffected.
    let use_wp_dc = !use_lever_a && std::env::var_os("JPEG_GRADIENT_DC").is_none();

    // Build the kWPFixedDC tree + its AC-metadata-prefixed wrapper ONCE.
    // `dc_remap` maps the WP tree's leaf contexts into the combined
    // (DC-subtree + AC-meta-subtree) context space; `ac_meta_ctx_map` maps the
    // AC-metadata collector's contexts the same way. Mirrors the VarDCT
    // W44-57 path (`vardct/bitstream.rs`).
    let wp_dc_state: Option<(
        crate::vardct::dc_tree_learn::DcTree,
        Vec<(u32, u32)>,
        u32,
        Vec<u32>,
        [u32; crate::vardct::dc_tree_learn::NUM_AC_META_CONTEXTS as usize],
    )> = if use_wp_dc {
        let total_dc_pixels = xsize_blocks * ysize_blocks * 3;
        let (wp_tree, wp_num_ctx) =
            crate::vardct::dc_tree_learn::build_wp_fixed_dc_tree(total_dc_pixels, 8);
        let (wrapped, total_ctx, dc_remap, ac_map) =
            crate::vardct::dc_tree_learn::tree_tokens_with_ac_metadata_prefix(
                &wp_tree,
                wp_num_ctx,
                num_dc_groups,
            );
        Some((wp_tree, wrapped, total_ctx, dc_remap, ac_map))
    } else {
        None
    };

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

        let dc_tokens = if let Some((ref wp_tree, _, _, ref dc_remap, _)) = wp_dc_state {
            let mut t = crate::vardct::dc_coding::collect_dc_tokens_wp_region_jpeg(
                &quant_dc,
                wp_tree,
                start_bx,
                start_by,
                end_bx,
                end_by,
                &channel_shifts,
            );
            for tok in t.iter_mut() {
                tok.set_context(dc_remap[tok.context() as usize]);
            }
            t
        } else if use_lever_a {
            collect_dc_tokens_region_jpeg_transcode(
                &quant_dc,
                start_bx,
                start_by,
                end_bx,
                end_by,
                &channel_shifts,
            )
        } else {
            collect_dc_tokens_region(
                &quant_dc,
                start_bx,
                start_by,
                end_bx,
                end_by,
                &channel_shifts,
            )
        };
        let md_tokens = if let Some((_, _, _, _, ref ac_map)) = wp_dc_state {
            // WP-DC path uses the gradient-style AC-metadata collector (same
            // as the non-lever-A path); contexts are remapped into the
            // combined space via `ac_meta_ctx_map`.
            let mut t = collect_ac_metadata_tokens_region(
                region_xsize,
                region_ysize,
                &quant_field,
                xsize_blocks,
                start_bx,
                start_by,
                &cfl_map,
                &ac_strategy,
                None,
            );
            for tok in t.iter_mut() {
                tok.set_context(ac_map[tok.context() as usize]);
            }
            t
        } else if use_lever_a {
            collect_ac_metadata_tokens_region_jpeg_transcode(
                region_xsize,
                region_ysize,
                &quant_field,
                xsize_blocks,
                start_bx,
                start_by,
                &cfl_map,
                &ac_strategy,
                None,
            )
        } else {
            collect_ac_metadata_tokens_region(
                region_xsize,
                region_ysize,
                &quant_field,
                xsize_blocks,
                start_bx,
                start_by,
                &cfl_map,
                &ac_strategy,
                None,
            )
        };
        dc_tokens_per_group.push(dc_tokens);
        ac_metadata_tokens_per_group.push(md_tokens);
    }

    // Issue #65: port libjxl `enc_frame.cc:1049-1094` JPEG DC-quantile
    // context map. Build a luma DC histogram from `quant_dc[1]`, the
    // chroma quant table's first 5 AC entries (libjxl uses
    // `qt[1] + qt[2] + qt[3] + qt[4] + qt[5]` where `qt` is indexed
    // `kDCTBlockSize * c + 8*x + y` and `c=0` is the JXL X channel =
    // JPEG chroma component), then quantile-cut into up to 8 buckets
    // and emit a 3-channel ctx_map per the libjxl formula.
    //
    // JPEG reencoding still has uniform QF=1 and all-DCT8 so we don't
    // populate `qf_thresholds`. The DC-bucket axis carries the entropy
    // separation per the libjxl reference.
    //
    // dc_offset already pre-applied by `map_jpeg_coefficients` (it
    // adds `1024 / qt_dc[c]` for `color_transform == kNone`), so
    // `quant_dc[1][by][bx]` is the same value libjxl reads from
    // `inputjpeg[base] + 1024/qt_dc[c]` at enc_frame.cc:1001 and
    // bins via `dc_counts[clamp(idc + 1024, 0, 2047)]++` at line 1004.
    //
    // 4:2:0 subsampling: libjxl's loop only enters luma chroma-aligned
    // positions (`by == sby << vshift`), but for luma vshift=0 every
    // row is aligned, so `quant_dc[1]` covers every luma block once.
    let block_ctx_map = {
        let mut dc_counts = [0usize; 2048];
        let mut total_dc_luma = 0usize;
        // Histogram of luma DC (channel 1 in JXL = JPEG Y component).
        // quant_dc[1] is sized `comp_y.height_in_blocks ×
        // comp_y.width_in_blocks`; for the JPEG path luma stays at
        // full block resolution under any subsampling.
        for row in &quant_dc[1] {
            for &dc_val in row {
                let idx = (dc_val as i32 + 1024).clamp(0, 2047) as usize;
                dc_counts[idx] += 1;
                total_dc_luma += 1;
            }
        }
        // libjxl `enc_frame.cc:1057-1058`:
        // `qt[1] + qt[2] + qt[3] + qt[4] + qt[5]` with `qt` indexed
        // `kDCTBlockSize * c + 8*x + y` (transposed storage). Our
        // `raw_qtables` uses the same convention
        // (`qtables[jxl_c*64 + x*8 + y] = quant[y*8 + x]`,
        // `build_raw_qtables` line 875), so `raw_qtables[1..5]` is
        // `qt[0*64 + 1..5]` = JXL channel 0 = JPEG chroma slots
        // (x=0,y=1), (x=0,y=2), (x=0,y=3), (x=0,y=4), (x=0,y=5).
        let qt_ac_sum: u32 = raw_qtables[1..=5]
            .iter()
            .map(|&v| v.max(0) as u32)
            .sum::<u32>()
            .max(1);
        // EX-J15 lever: full-resolution chroma DC quantile mapping (one
        // context per luma DC bucket per channel) when num_dc_ctxs <= 5.
        // Default-OFF; gated by `EX_J15_FULL_CHROMA=1` env hook for A/B
        // measurement. Treats "0", "", "false", "off", "no" as OFF (so the
        // bench can pass `EX_J15_FULL_CHROMA=0` for the baseline arm). Any
        // other non-empty value triggers ON. Falls back to libjxl
        // half-resolution chroma when env is OFF OR num_dc_ctxs > 5.
        let use_ex_j15 = match std::env::var("EX_J15_FULL_CHROMA").ok() {
            Some(v) => {
                let v = v.trim().to_ascii_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "off" || v == "no")
            }
            None => false,
        };
        if use_ex_j15 {
            ac_context::BlockCtxMap::jpeg_dc_quantile_ex_j15(
                &dc_counts,
                total_dc_luma,
                qt_ac_sum,
                is_gray,
            )
        } else {
            ac_context::BlockCtxMap::jpeg_dc_quantile(&dc_counts, total_dc_luma, qt_ac_sum, is_gray)
        }
    };

    // Per-block DC bucket lookup table sized `xsize_blocks * ysize_blocks`.
    // libjxl `compressed_dc.cc:274-292` reads `quant_dc` (an `ImageB`)
    // populated only when `bctx.num_dc_ctxs > 1`. With only luma
    // thresholds set, the formula reduces to
    // `qdc_row[x] = sum(dc_thresholds[1] < quant_dc[1][by][bx])`
    // (i.e. how many luma thresholds the value exceeds → the bucket).
    let dc_buckets: Vec<u8> = if block_ctx_map.num_dc_ctxs > 1 {
        let thresholds = &block_ctx_map.dc_thresholds[1];
        let mut buckets = vec![0u8; xsize_blocks * ysize_blocks];
        // quant_dc[1] is the FULL luma plane; xsize_blocks/ysize_blocks
        // equal `comp_y.width/height_in_blocks` (luma is full-res).
        for by in 0..ysize_blocks {
            let row = &quant_dc[1].get(by);
            if let Some(row) = row {
                for bx in 0..xsize_blocks {
                    let dc_val = row.get(bx).copied().unwrap_or(0) as i32;
                    let bucket = thresholds.iter().filter(|&&t| dc_val > t).count() as u8;
                    buckets[by * xsize_blocks + bx] = bucket;
                }
            }
        }
        buckets
    } else {
        vec![0u8; xsize_blocks * ysize_blocks]
    };

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
    //
    // EX-J17b (lever #5): Sample 50% of blocks via xorshift128+ instead of
    // counting every block. Mirrors libjxl `enc_coeff_order.cc::ComputeCoeffOrder`
    // at speed >= kSquirrel with single strategy (DCT8). Every JPEG block is
    // DCT8 so we're always in the `current_used_orders == 1` case. The reduced
    // sample population is noisier per-position but admits cheaper orderings
    // (Lehmer cost depends on permutation distance from natural order; sampling
    // breaks tight ties that the all-blocks count would resolve in favour of
    // natural). Uses the same xorshift128+ seeds + threshold derivation as
    // libjxl so the sample mask is bit-identical between Rust and C++ when
    // input dimensions match.
    // EX-J27 (2026-05-28): custom orders (EX-J17a) verified as a real win:
    // disabling via env hook produced +0.433% regression (-44118 bytes /
    // 50 files) on the paired bench. cjxl-e7 also uses sampled custom
    // orders for JPEG. Keep the env hook for future investigators.
    let custom_orders_enabled = std::env::var_os("JPEG_NO_CUSTOM_ORDERS").is_none();
    let (custom_order_map, used_orders) =
        if custom_orders_enabled && (xsize_blocks >= 5 || ysize_blocks >= 5) {
            let mut zero_counts: Vec<Vec<Vec<i64>>> = (0..NUM_ORDER_BUCKETS_JPEG)
                .map(|_| vec![Vec::new(); 3])
                .collect();
            // Only bucket 0 (DCT8) is populated — every JPEG block is DCT8.
            for ch in &mut zero_counts[0] {
                *ch = vec![0i64; BLOCK_SIZE];
            }
            // libjxl xorshift128+ initial state (enc_coeff_order.cc:88-89).
            let mut xs_state: [u64; 2] = [0x94D049BB133111EB, 0xBF58476D1CE4E5B9];
            // block_fraction = 0.5 → threshold = (u64::MAX >> 32) * 0.5.
            let threshold: u64 = (((u64::MAX) >> 32) as f64 * 0.5f64) as u64;
            let mut use_sample = |state: &mut [u64; 2]| -> bool {
                let s1 = state[0];
                let s0 = state[1];
                let bits = s1.wrapping_add(s0);
                state[0] = s0;
                let mut s1 = s1 ^ (s1 << 23);
                s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
                state[1] = s1;
                (bits >> 32) <= threshold
            };
            for c in 0..3 {
                let plane = &quant_ac[c];
                let cnt = &mut zero_counts[0][c];
                for row in plane {
                    for block in row {
                        if !use_sample(&mut xs_state) {
                            continue;
                        }
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
            // EX-J29 (2026-05-28): HONEST-STOP. Tested libjxl-EXACT custom-order
            // admission (emit whenever nondefault, NO cost gate, matching
            // enc_coeff_order.cc:198-237) vs our cost-benefit gate on the JPEG
            // path. 50-file paired A/B: +239 bytes (49/50 tied, 1 worse). Our
            // gate ALREADY admits the order on essentially every JPEG file — the
            // gate decision savings>cost almost always passes for DCT8 bucket-0 —
            // so the gate is at parity with libjxl on JPEG and is correct on the
            // 1 file where it differs. Coefficient orders ARE NOT the gap.
            // Keep the gated path as default; the unconditional libjxl-exact
            // path stays available via env hook `JPEG_UNCONDITIONAL_ORDERS=1`
            // as a regression harness.
            let (orders, used) = if std::env::var_os("JPEG_UNCONDITIONAL_ORDERS").is_some() {
                crate::vardct::coeff_order::compute_custom_orders_unconditional(&zero_counts)
            } else {
                compute_custom_orders(&zero_counts)
            };
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
                    // Issue #65: DC bucket from luma DC (libjxl
                    // `dec_group.cc:491` uses `qdc_row[lbx]` for ALL
                    // channels, where `lbx` is the luma bx — i.e. the
                    // chroma block shares the luma block's DC bucket
                    // at the aligned luma position).
                    let dc_idx = dc_buckets[by * xsize_blocks + bx] as usize;
                    let block_ctx =
                        block_ctx_map.block_context_dc(c, strategy_code, qf_val, dc_idx);
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

    // EX-J31 (2026-05-28): enable libjxl-parity accurate ANS population cost
    // (real ComputeBest header+data, not the crude entropy + 5*alphabet
    // estimate) for the kBest clustering in all entropy-code builds below.
    // Scoped to the JPEG transcode path on this thread; default-ON, measured
    // -0.069pp on the 50-file corpus on top of WP-DC. `JPEG_CRUDE_ANS_COST=1`
    // restores the crude estimate for A/B. The guard lives to end of function,
    // covering the DC code, AC code, and any coeff-order entropy builds.
    let _accurate_ans_guard = if std::env::var_os("JPEG_CRUDE_ANS_COST").is_none() {
        Some(crate::entropy_coding::cluster::AccurateAnsCostGuard::new())
    } else {
        None
    };

    let dc_num_contexts = if let Some((_, _, total_ctx, _, _)) = wp_dc_state {
        // EX-J31: combined (kWPFixedDC DC subtree + AC-meta subtree) context
        // count from `tree_tokens_with_ac_metadata_prefix`.
        total_ctx as usize
    } else if use_lever_a {
        NUM_DC_CONTEXTS_JPEG_TRANSCODE
    } else {
        NUM_DC_CONTEXTS
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
    // Lever #3 (2026-05-28): match libjxl JPEG transcode path which sets
    // `uint_method = kNone` for non-modular paths (`enc_ans.cc:1361-1366`). Our
    // default helper uses kFast (4-config trial); disabling that for the JPEG
    // transcode reduces header overhead at the AC stream's per-context level
    // where the cluster pair-merge has already adapted the histograms.
    // EX-J22 (2026-05-28): pair-merge histogram clustering on the
    // combined DC + AC-metadata token stream at effort >= 7. Mirrors the
    // AC code path (line ~912 below) which has used `enhanced_clustering
    // = effort >= 7` since dddebe2c. The DC stream is small (a few
    // thousand tokens vs millions on AC) but kBest at e9 was reported
    // to deliver -0.27% on the AC code on a 10-file corpus, and the
    // signaling overhead amortization is comparable on DC. Env hook
    // JPEG_DC_NO_CLUSTERING=1 forces kFast for A/B measurement.
    let dc_enhanced_clustering = effort >= 7 && std::env::var_os("JPEG_DC_NO_CLUSTERING").is_none();
    let dc_code = build_entropy_code_ans_with_options(
        &all_dc_tokens,
        dc_num_contexts,
        /*enhanced_clustering=*/ dc_enhanced_clustering,
        /*optimize_uint_configs=*/ true,
        /*lz77=*/ None,
        /*total_pixel_hint=*/ Some(width * height),
    );

    let ac_num_contexts = block_ctx_map.num_ac_contexts();

    // ── Effort gates for the AC code (mirrors libjxl JPEG-mode VarDCT) ──
    //
    // libjxl `enc_frame.cc:1267-1271` constructs the AC code histogram params
    // as `HistogramParams(speed_tier, num_ac_contexts)`. The constructor in
    // `enc_ans_params.h:60-75` plus the per-frame override produce the
    // following table, where lower speed_tier == higher effort:
    //
    //   effort 1-8 (speed_tier > kTortoise):
    //     clustering = kFast, uint_method = kNone, lz77_method = kNone
    //   effort 9+ (speed_tier <= kTortoise):
    //     clustering = kBest, uint_method = kBest, lz77_method = kRLE
    //
    // ── Measurement on a 10-file corpus (2026-05-28, vs e7 baseline) ──
    //
    //   kBest clustering alone:                  -0.27 % avg, 10/10 win/tied
    //   kBest clustering + RLE LZ77:             -0.27 % avg (LZ77 global
    //                                            threshold never passes
    //                                            on these JPEG AC streams)
    //   kBest clustering + uint_method = kBest:  +0.51 % avg, REGRESSION
    //   all three combined:                      +0.51 % avg, REGRESSION
    //
    // Our `optimize_uint_configs_best_from_freqs` (kBest equivalent) picks
    // configs whose signaling overhead exceeds the data savings on JPEG AC
    // streams. The divergence vs libjxl is unresolved — filed as
    // jpeg-effort-cluster-lz77 follow-on. Our `apply_lz77_rle_multi_section`
    // matches the libjxl algorithm but the global savings threshold
    // (`total_symbols * 0.2 + 16`) doesn't pass on these JPEG AC streams in
    // either implementation; cjxl's e9 wins are clustering-driven, not LZ77.
    //
    // Decision: ship kBest clustering at e9 (the measurably-correct lever);
    // keep LZ77 + uint_opt wiring in place but default-OFF until divergence
    // root-causes are resolved. They can be flipped on via env hooks for
    // future investigation.
    let mut enhanced_clustering = effort >= 7;
    let mut lz77_method: Option<crate::entropy_coding::lz77::Lz77Method> = None;

    // ── Lever-experiment env hooks (off by default; for benching only) ──
    //
    // `JPEG_E9_NO_CLUSTERING=1`: forces kFast clustering on the AC code at
    //   effort >= 9 (default is kBest at e9).
    // `JPEG_E9_FORCE_LZ77=1`: re-enables the libjxl-parity RLE LZ77 transform
    //   on the AC code at effort >= 9. Default-off because the global
    //   savings threshold doesn't pass on JPEG AC streams; the wiring is
    //   retained so future investigations can A/B without re-implementing.
    // `JPEG_E9_FORCE_UINT_OPT=1`: re-enables `optimize_uint_configs=true` on
    //   the AC code at effort >= 9. Default-off (measured regression).
    if std::env::var_os("JPEG_E9_NO_CLUSTERING").is_some() {
        enhanced_clustering = false;
    }
    if effort >= 9 && std::env::var_os("JPEG_E9_FORCE_LZ77").is_some() {
        lz77_method = Some(crate::entropy_coding::lz77::Lz77Method::Rle);
    }

    // Distance multiplier for LZ77 special distance symbols.
    //
    // BUG FIX 2026-05-28: previously this was `xsize_blocks as i32`, which
    // mirrors the libjxl `image_widths[stream]` value used for MODULAR
    // subimage streams (DC, AC metadata). The JPEG-mode AC token stream is
    // the VarDCT-style coefficient stream — the decoder reads it via the
    // AC-group decode path which calls `SymbolReader::new(..., image_width: None)`
    // (see `zenjxl-decoder/src/frame/group.rs:386`), giving
    // `dist_multiplier = 0` decoder-side. The encoder must match: with
    // `distance_multiplier > 0` the RLE distance symbol is `1`, which the
    // decoder (dist_multiplier=0) interprets as distance=2 (`distance_sub_1
    // = distance_sym = 1` → `distance = 2`), corrupting the decoded stream
    // by repeating the symbol from two positions back instead of the
    // immediately previous one. With `distance_multiplier = 0`, the RLE
    // distance symbol is `0` → decoder reads `distance_sub_1 = 0` →
    // `distance = 1` = repeat-previous as intended.
    //
    // This matches the VarDct-side convention at
    // `vardct/bitstream.rs:3051` (`let ac_distance_multiplier = 0i32;`).
    let lz77_distance_multiplier = 0i32;

    // Apply LZ77 across ALL sections with a SINGLE global savings gate
    // (mirrors libjxl `ApplyLZ77_RLE` at `enc_lz77.cc:111-183`, which iterates
    // the section vector and aggregates `bit_decrease` + `total_symbols`
    // globally before thresholding once). The earlier per-section helper had
    // every section pass its own threshold, which rejected JPEG AC streams
    // even when libjxl accepts them — JPEG AC tokens are diverse per-group
    // but the aggregate has enough runs to clear the global gate. Currently
    // only RLE is wired here; greedy / optimal multi-section variants are
    // available via the single-section `apply_lz77` if a future caller opts
    // in (effort 10+ would be the natural home).
    let ac_lz77_section_tokens: Vec<Vec<Token>>;
    let ac_lz77_params: Option<crate::entropy_coding::lz77::Lz77Params>;
    match lz77_method {
        Some(crate::entropy_coding::lz77::Lz77Method::Rle) => {
            let section_slices: Vec<&[Token]> =
                ac_section_tokens.iter().map(|v| v.as_slice()).collect();
            match crate::entropy_coding::lz77::apply_lz77_rle_multi_section(
                &section_slices,
                ac_num_contexts,
                false, // force_huffman: ANS on the JPEG path
                lz77_distance_multiplier,
            ) {
                Some((transformed_sections, params)) => {
                    ac_lz77_section_tokens = transformed_sections;
                    ac_lz77_params = Some(params);
                }
                None => {
                    // Global gate failed — fall back to the un-transformed tokens.
                    ac_lz77_section_tokens = ac_section_tokens.clone();
                    ac_lz77_params = None;
                }
            }
        }
        Some(_) | None => {
            // No LZ77 (effort < 9) or non-RLE method (not wired for the
            // multi-section JPEG path) — pass-through.
            ac_lz77_section_tokens = ac_section_tokens.clone();
            ac_lz77_params = None;
        }
    }

    // Merge the (post-LZ77) token streams into a single buffer for entropy-
    // code construction. The merged stream is used only to build histograms /
    // context map / distributions; we write the per-section tokens (in
    // `ac_lz77_section_tokens`) at the end with the shared `ac_code`.
    let total_ac_tokens: usize = ac_lz77_section_tokens.iter().map(|t| t.len()).sum();
    let mut all_ac_tokens = Vec::with_capacity(total_ac_tokens);
    for section in &ac_lz77_section_tokens {
        all_ac_tokens.extend_from_slice(section);
    }

    // ENABLED at every effort: the ANS-distribution-header cost was added
    // to `optimize_uint_configs_{fast,best}_from_freqs` (commit on this
    // chain), so the optimizer now picks correctly. Bench: −0.148pp at e7
    // on the 200-file JPEG corpus. The previous fixed `(4, 2, 0)` default
    // was a workaround for the missing-header-cost bug; the workaround can
    // now be removed.
    //
    // `JPEG_E9_FORCE_UINT_OPT_OFF=1` environment hook lets future
    // investigators A/B if needed.
    let mut optimize_uint_configs_ac = true;
    if std::env::var_os("JPEG_E9_FORCE_UINT_OPT_OFF").is_some() {
        optimize_uint_configs_ac = false;
    }

    // BUG FIX 2026-05-28: when LZ77 is enabled the distance tokens use
    // `context = ac_num_contexts` (one past the per-block contexts), so the
    // entropy-code builder must accumulate histograms across
    // `ac_num_contexts + 1` slots, and the encoded context map must have
    // `ac_num_contexts + 1` entries (the decoder reads it that way at
    // `zenjxl-decoder/src/entropy_coding/decode.rs:605-612`). Previously the
    // call passed `ac_num_contexts`, which silently dropped every distance
    // token from histogram accumulation (`AccumulatedAnsData::add_token`
    // checks `if ctx < self.num_contexts`) and emitted a context map one
    // entry too short, producing a bitstream that no spec-conformant
    // decoder could parse.
    //
    // This mirrors the canonical pattern in the modular path at
    // `modular/encode.rs:2158-2162` and the VarDct AC path at
    // `vardct/bitstream.rs:3269-3273` (`ac_num_contexts = base + 1` when
    // `lz77_params.is_some()`).
    let ac_code_num_contexts = if ac_lz77_params.is_some() {
        ac_num_contexts + 1
    } else {
        ac_num_contexts
    };
    // EX-J23 (2026-05-28): HONEST-STOP. Tested ANSHistogramStrategy::Approximate
    // (libjxl-e7 default) on both DC and AC codes:
    // - DC: +20 bytes / 50 files (noise; few histograms, no signal).
    // - AC: +1979 bytes / 50 files (REGRESSION). On the AC stream the
    //   12-shift Precise grid finds tighter normalization that saves
    //   more bytes than the header overhead it incurs. cjxl-e7's
    //   choice of Approximate is a speed tradeoff, not a compression
    //   win. Reverted, kept Precise.
    //
    // EX-J24 (2026-05-28): HONEST-STOP. Tested removing total_pixel_hint
    // (the `min(num_contexts, total_pixels/2048)` cap) on the AC code:
    // result was 0 bytes / 50 files difference. At typical JPEG corpus
    // sizes the cap doesn't materially bind — kBest pair-merge converges
    // organically below the cap. Kept the existing hint for stability.
    let ac_code = build_entropy_code_ans_with_options(
        &all_ac_tokens,
        ac_code_num_contexts,
        enhanced_clustering,
        optimize_uint_configs_ac,
        ac_lz77_params.as_ref(),
        /*total_pixel_hint=*/ Some(width * height),
    );

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
    // EX-J31: when WP-DC is active, write the kWPFixedDC wrapped tree
    // (Predictor::Weighted DC subtree + AC-meta subtree) instead of the
    // static gradient context tree.
    let wp_dc_tree_tokens: Option<&[(u32, u32)]> = wp_dc_state
        .as_ref()
        .map(|(_, wrapped, _, _, _)| wrapped.as_slice());
    write_dc_global_jpeg(
        &dc_dequant,
        &dc_code,
        num_dc_groups,
        &block_ctx_map,
        use_lever_a,
        wp_dc_tree_tokens,
        &mut dc_global,
    )?;

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
        ac_lz77_params.as_ref(),
        &mut ac_global,
    )?;

    // AC Groups
    let mut ac_groups = Vec::with_capacity(num_groups);
    for ac_tokens in &ac_lz77_section_tokens {
        let mut ac_group_writer = BitWriter::new();
        write_tokens_ans(
            ac_tokens,
            &ac_code,
            ac_lz77_params.as_ref(),
            &mut ac_group_writer,
        )?;
        ac_groups.push(ac_group_writer);
    }

    // Bit-shaving probe: dump per-section bytes when JPEG_SECTION_DUMP=1.
    if std::env::var_os("JPEG_SECTION_DUMP").is_some() {
        let dc_groups_total: usize = dc_groups.iter().map(|w| w.bytes_written()).sum();
        let ac_groups_total: usize = ac_groups.iter().map(|w| w.bytes_written()).sum();
        eprintln!(
            "JPEG_SECTION_DUMP: file_hdr={} frame_hdr={}b dc_global={}B dc_groups[{}]={}B ac_global={}B ac_groups[{}]={}B total~{}B (dc_tokens={} ac_meta_tokens={} ac_tokens={})",
            file_header_bytes,
            writer.bytes_written() - file_header_bytes,
            dc_global.bytes_written(),
            num_dc_groups,
            dc_groups_total,
            ac_global.bytes_written(),
            num_groups,
            ac_groups_total,
            file_header_bytes
                + (writer.bytes_written() - file_header_bytes)
                + dc_global.bytes_written()
                + dc_groups_total
                + ac_global.bytes_written()
                + ac_groups_total,
            dc_tokens_per_group.iter().map(|t| t.len()).sum::<usize>(),
            ac_metadata_tokens_per_group
                .iter()
                .map(|t| t.len())
                .sum::<usize>(),
            ac_section_tokens.iter().map(|t| t.len()).sum::<usize>(),
        );
        eprintln!(
            "JPEG_SECTION_DUMP: AC num_contexts(in)={} clustered_histograms(out)={} used_orders={} | DC num_contexts(in)={} clustered_histograms(out)={}",
            ac_code.context_map.len(),
            ac_code.histograms.len(),
            used_orders,
            dc_code.context_map.len(),
            dc_code.histograms.len(),
        );
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
///
/// Uses the default effort level (7). For caller-supplied effort, use
/// [`encode_jpeg_to_jxl_container_with_effort`].
pub fn encode_jpeg_to_jxl_container(jpeg: &JpegData) -> Result<Vec<u8>> {
    encode_jpeg_to_jxl_container_with_effort(jpeg, 7)
}

/// Encode a JPEG as a JXL container at the given effort level.
///
/// See [`encode_jpeg_to_jxl_with_effort`] for the effort gates that affect the
/// codestream side. Container wrapping (JBRD / Exif / XMP boxes) is
/// effort-independent.
pub fn encode_jpeg_to_jxl_container_with_effort(jpeg: &JpegData, effort: u8) -> Result<Vec<u8>> {
    let (codestream, file_header_size) = encode_jpeg_to_jxl_inner(jpeg, effort)?;
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
/// Detect whether the frame's semantic color_transform is
/// `kYCbCr` (true) or `kNone` (false).
///
/// Mirrors libjxl `SetColorTransformFromJpegData`
/// (`lib/jxl/jpeg/enc_jpeg_data.cc:240-283`) exactly:
///
/// 1. **JFIF (APP0) present** → kYCbCr (always; even if Adobe APP14
///    says transform=0, the JFIF marker wins). This is the most
///    common case for camera / web JPEGs.
/// 2. **No JFIF, but Adobe APP14 with payload "Adobe...transform=0"
///    present** → kNone (RGB).
/// 3. **No JFIF, no Adobe APP14**: if 3 components AND component IDs
///    are literally ('R','G','B') = (82, 71, 66) → kNone (RGB).
/// 4. Otherwise → kYCbCr (default).
///
/// Grayscale (1 component) is handled by the caller (the
/// `nbcomp == 1` early-fast-path in libjxl's last line) — this
/// function returns the semantic transform; the caller forces
/// kYCbCr for grayscale separately.
///
/// Independent of the JBRD `component_type` tag (which is a
/// serialization shortcut keyed on LITERAL ID matches). A JPEG with
/// IDs (0, 1, 2) and the JFIF marker is semantically YCbCr but the
/// JBRD tag will be `Custom` (because (0,1,2) doesn't match the
/// (1,2,3) YCbCr ID-pattern).
fn detect_ycbcr_color_transform(jpeg: &JpegData) -> bool {
    // Rule 1: JFIF present → always kYCbCr
    if jpeg.marker_order.contains(&0xE0) {
        return true;
    }
    // Rule 2: Adobe APP14 → check transform byte
    let mut app_idx = 0usize;
    for &marker in &jpeg.marker_order {
        if marker & 0xF0 == 0xE0 {
            // APP marker — payload at app_data[app_idx]
            if app_idx < jpeg.app_data.len() {
                let data = &jpeg.app_data[app_idx];
                // libjxl checks marker == 0xEE and data.size() == 15
                //   (marker_byte + len_hi + len_lo + 'Adobe' + 7 bytes of payload).
                // Our `app_data[i]` is `[marker, len_hi, len_lo, payload...]`,
                // so total len 15 means payload is 12 bytes: 'Adobe'(5) + version(2)
                // + flags0(2) + flags1(2) + transform(1).
                // The transform byte is at app_data[14] in libjxl's layout
                // (= our index 14 too since marker_byte is at [0]).
                if marker == 0xEE && data.len() == 15 && data.len() > 7 && &data[3..8] == b"Adobe" {
                    let transform = data[14];
                    return transform != 0; // kYCbCr unless transform=0
                }
            }
            app_idx += 1;
        }
    }
    // Rule 3: no JFIF, no Adobe — guess from component IDs
    if jpeg.components.len() == 3
        && jpeg.components[0].id == b'R' as u32
        && jpeg.components[1].id == b'G' as u32
        && jpeg.components[2].id == b'B' as u32
    {
        return false; // explicit RGB IDs → kNone
    }
    // Rule 4: default → kYCbCr
    true
}

/// Compute the per-channel DC offset that the encoder must add to each
/// stored quantized DC value so the decoder's reconstruction
/// (`dc_value - dcoff[c]` in `dec_group.cc:417`) returns the original
/// JPEG DC.
///
/// For `color_transform == kYCbCr` (3-component YCbCr; and our forced
/// `is_ycbcr=true` grayscale path) → offset is 0.
///
/// For `color_transform == kNone` (3-component RGB; libjxl applies
/// this to grayscale too but we differ — see callsite comment) →
/// offset is `1024 / qt_dc[c]` per `enc_frame.cc:1005` mirrored by the
/// decoder's `dec_group.cc:246`.
fn compute_jpeg_dc_offset(jpeg: &JpegData, jpeg_c_map: &[usize; 3], is_ycbcr: bool) -> [i32; 3] {
    let mut offsets = [0i32; 3];
    if is_ycbcr {
        return offsets;
    }
    for jxl_c in 0..3 {
        let jpeg_c = jpeg_c_map[jxl_c];
        // Grayscale's [0,0,0] map keeps all three offsets aligned to comp[0],
        // but is_ycbcr is forced true for grayscale, so this branch is
        // reachable only for non-gray inputs in practice.
        if jpeg_c >= jpeg.components.len() {
            continue;
        }
        let qt_idx = jpeg.components[jpeg_c].quant_idx as usize;
        let qt_dc = jpeg.quant[qt_idx].values[0];
        if qt_dc > 0 {
            offsets[jxl_c] = 1024 / qt_dc;
        }
    }
    offsets
}

fn map_jpeg_coefficients(
    jpeg: &JpegData,
    jpeg_c_map: &[usize; 3],
    dc_offset: &[i32; 3],
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

                // DC coefficient (natural order position 0).
                //
                // For `color_transform == kNone` (RGB JPEGs) the
                // decoder subtracts `1024 / qt_dc[c]` from each DC
                // value when reconstructing the original JPEG
                // (dec_group.cc:417). Add the matching offset here so
                // the round-trip closes byte-exactly. For YCbCr
                // JPEGs `dc_offset[c]` is 0 and this is a no-op,
                // preserving the existing YCbCr/grayscale paths.
                //
                // The offset is small (~128 for typical luma DC quant
                // = 8) and the result still fits in i16 because JPEG
                // quantized DC is in [-1024, 1023] and the max offset
                // is `1024 / 1` = 1024.
                let dc = (comp.coeffs[base] as i32 + dc_offset[jxl_c]).clamp(-32768, 32767) as i16;
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
    // Mirror libjxl `SetColorTransformFromJpegData`: 3-component RGB
    // (Adobe APP14 transform=0 or explicit IDs (R,G,B), no JFIF) →
    // kNone (do_ycbcr = false); everything else, including grayscale,
    // → kYCbCr (do_ycbcr = true).
    let is_ycbcr = detect_ycbcr_color_transform(jpeg) || jpeg.components.len() == 1;
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
///
/// When `use_lever_a == true` the smaller JPEG-transcode context tree
/// (`kJpegTranscodeACMeta` + gradient-DC) is emitted in place of the
/// 313-token libjxl-tiny tree; the caller MUST have collected DC and
/// AC-metadata tokens with the matching `_jpeg_transcode` variants.
fn write_dc_global_jpeg(
    dc_dequant: &[f32; 3],
    dc_code: &OwnedAnsEntropyCode,
    num_dc_groups: usize,
    block_ctx_map: &ac_context::BlockCtxMap,
    use_lever_a: bool,
    wp_dc_tree_tokens: Option<&[(u32, u32)]>,
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

    // Issue #65: emit the JPEG DC-quantile BlockCtxMap (non-default
    // flag + DC threshold counts + threshold values + QF count +
    // QF values + entropy-coded ctx_map). Previously this was a
    // hardcoded `writer.write(16, 0)?` placeholder paired with the
    // compact-only `write_block_context_map`, which was incorrect once
    // we wanted to populate `dc_thresholds`. The adaptive writer
    // handles both legacy (all-empty) and JPEG (luma DC) paths.
    crate::vardct::context_tree::write_block_ctx_map_adaptive_with_mode(
        block_ctx_map,
        /*jpeg_mode=*/ true,
        writer,
    )?;

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

    // Context tree for modular DC + AC-metadata streams.
    // EX-J31: WP-DC mode writes the kWPFixedDC wrapped tree (Predictor::Weighted
    // DC subtree splitting on wp_max_error + AC-metadata subtree) produced by
    // `tree_tokens_with_ac_metadata_prefix`. Lever A: the JPEG-transcode
    // variant collapses the 11-leaf AC-metadata subtree into a single
    // Leaf(Zero). Default (neither): the static 313-token gradient tree.
    if let Some(tree_tokens) = wp_dc_tree_tokens {
        crate::vardct::context_tree::write_learned_context_tree(
            tree_tokens,
            num_dc_groups,
            writer,
        )?;
    } else if use_lever_a {
        crate::vardct::context_tree::write_jpeg_transcode_context_tree(num_dc_groups, writer)?;
    } else {
        crate::vardct::context_tree::write_context_tree(num_dc_groups, writer)?;
    }

    // LZ77: disabled
    writer.write(1, 0)?;

    // DC entropy code
    write_entropy_code_ans(dc_code, writer)?;

    Ok(())
}

/// Write AC global section for JPEG reencoding.
///
/// Unlike normal VarDCT, this writes RAW quant matrices (not all_default).
///
/// `ac_lz77`: if `Some`, writes an `lz77_enabled=1` header before the AC entropy
/// code. Must match the params used to transform `ac_section_tokens`. At
/// `effort >= 8` this is set by `encode_jpeg_to_jxl_with_effort` when the
/// per-section LZ77 cost gate passes for every section.
fn write_ac_global_jpeg(
    raw_qtables: &[i32],
    num_groups: usize,
    ac_code: &OwnedAnsEntropyCode,
    used_orders: u32,
    coeff_order_tokens: Option<&[Token]>,
    ac_lz77: Option<&crate::entropy_coding::lz77::Lz77Params>,
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

    // LZ77 header — written here per the entropy code spec; if `ac_lz77` is
    // `None`, this writes a single `0` bit (disabled), matching the legacy
    // effort-7 path byte-for-byte.
    crate::entropy_coding::lz77::write_lz77_header(ac_lz77, writer)?;

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
