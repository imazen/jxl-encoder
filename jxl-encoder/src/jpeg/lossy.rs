// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Lossy ("slightly lossy") JPEG → JXL recompression — the **PreserveJxl**
//! coefficient-domain path.
//!
//! Instead of decoding the JPEG to pixels and re-encoding (which resurrects
//! the frequencies the source already killed and re-ratchets the [0,255]
//! clamp — the classic generation-loss artifacts), this re-quantizes the
//! JPEG's own quantized DCT coefficients to a *coarser* quant matrix that is
//! a near-uniform scale of the source's own tables, then hands the result to
//! the existing lossless YCbCr transcode. The output is a YCbCr JXL whose
//! pixels are the coarsened image; no JBRD (we've gone lossy, so byte-exact
//! JPEG reconstruction is no longer meaningful).
//!
//! Why this beats a JPEG→JPEG recompressor at matched quality: the *same*
//! coarsened coefficients are entropy-coded by JXL's ANS + context models +
//! WP-DC instead of JPEG's fixed Huffman. Measured (Phase 0): −14% (gentle)
//! to −33% (aggressive) vs zenjpeg-recompress's JPEG at identical pixels,
//! the advantage *growing* with coarsening.
//!
//! Correctness rules borrowed from the zenjpeg-recompress research
//! (`RECOMPRESSION_COMPENDIUM.md` §10.2):
//! - **Same-family scale**: derive the target matrix by scaling the source's
//!   own dequant weights (keeps the per-position old/new ratio near-constant,
//!   a clean uniform rescale, minimal per-coefficient rounding). We do NOT
//!   impose an "ideal" matrix.
//! - **Build each unique quant table exactly once from the ORIGINAL.** JPEG's
//!   2-table layout shares one chroma table between Cb and Cr; scaling per
//!   *component* would scale the shared table twice (scale²) — a silent 2×
//!   chroma over-quantization. We scale the `quant` Vec (already deduplicated
//!   — one entry per unique table) and re-quantize coefficients reading the
//!   ORIGINAL weights, before overwriting.

use super::data::JpegData;

/// Coarsen a parsed JPEG's quantized DCT coefficients in the DCT domain by a
/// near-uniform `scale` (> 1.0 = coarser = smaller + slightly lossy).
///
/// Mutates `jpeg` in place: every quant table is replaced by
/// `round(scale · Q_source)` (clamped to `[1, 65535]` — JXL's RAW quant
/// matrix is not bound to JPEG's 8-bit ladder), and every coefficient is
/// re-quantized to the new grid as `round(level · Q_source / Q_target)`,
/// which preserves the dequantized DCT value (loss-minimal, no
/// cross-frequency coupling).
///
/// `scale <= 1.0` is a no-op (we never sharpen / add precision a JPEG lacks).
pub fn coarsen_coefficients(jpeg: &mut JpegData, scale: f32) {
    if !(scale > 1.0) || jpeg.quant.is_empty() {
        return;
    }

    // 1. Snapshot ORIGINAL weights per unique table, and compute the coarsened
    //    target weights once each (share-scale-safe: one entry per table).
    let orig: Vec<[i32; 64]> = jpeg.quant.iter().map(|t| t.values).collect();
    let mut target: Vec<[i32; 64]> = orig.clone();
    for tbl in target.iter_mut() {
        for v in tbl.iter_mut() {
            let scaled = ((*v as f32) * scale).round() as i32;
            *v = scaled.clamp(1, 65535);
        }
    }

    // 2. Re-quantize each component's coefficients using its table's ORIGINAL
    //    and TARGET weights. coeffs are 64-per-block, natural (row-major)
    //    order — same order as the quant tables.
    for comp in jpeg.components.iter_mut() {
        let qi = comp.quant_idx as usize;
        if qi >= orig.len() {
            continue;
        }
        let qs = &orig[qi];
        let qt = &target[qi];
        for block in comp.coeffs.chunks_mut(64) {
            for (k, level) in block.iter_mut().enumerate() {
                if *level == 0 {
                    continue;
                }
                let qsk = qs[k];
                let qtk = qt[k];
                if qsk == qtk {
                    continue;
                }
                // dequantized value preserved: new_level · qt ≈ level · qs.
                let dequant = *level as f64 * qsk as f64;
                let new_level = (dequant / qtk as f64).round();
                *level = new_level.clamp(i16::MIN as f64, i16::MAX as f64) as i16;
            }
        }
    }

    // 3. Commit the coarsened tables. Bump precision to 16-bit where any
    //    weight now exceeds the 8-bit ladder (the JXL RAW matrix stores i32,
    //    so this is just metadata consistency).
    for (i, tbl) in jpeg.quant.iter_mut().enumerate() {
        tbl.values = target[i];
        if target[i].iter().any(|&v| v > 255) {
            tbl.precision = 1;
        }
    }
}
