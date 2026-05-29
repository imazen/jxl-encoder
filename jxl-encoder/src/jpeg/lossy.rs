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
///
/// Equivalent to [`coarsen_coefficients_dz`] with `deadzone = 0.0`.
pub fn coarsen_coefficients(jpeg: &mut JpegData, scale: f32) {
    coarsen_coefficients_dz(jpeg, scale, 0.0);
}

/// Coarsen with an explicit AC **deadzone** widening `dz` in `[0.0, 0.5]`.
///
/// A coefficient is kept iff its re-quantized magnitude `|level·Qs/Qt|` is at
/// least `0.5 + dz` (standard JPEG quantization uses `0.5`); otherwise it is
/// zeroed. This smooths the RD knob: a plain uniform `scale` keeps the huge
/// ±1-AC population until `scale` crosses the round-to-zero threshold all at
/// once (a size cliff), whereas widening the deadzone removes the perceptually
/// cheap small-AC residue *gradually*. DC (`k == 0`) is never deadzoned (only
/// scaled) — zeroing DC causes blocking. Matches the mozjpeg/jpegli
/// deadzone-widening idea, applied in the transcode domain.
pub fn coarsen_coefficients_dz(jpeg: &mut JpegData, scale: f32, dz: f32) {
    if !(scale > 1.0) || jpeg.quant.is_empty() {
        return;
    }
    let dz = dz.clamp(0.0, 0.5);
    let keep_threshold = 0.5 + dz;

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
                // dequantized value preserved: new_level · qt ≈ level · qs.
                let dequant = *level as f64 * qsk as f64;
                let q = dequant / qtk as f64;
                // AC deadzone: drop perceptually-cheap small-magnitude AC.
                // DC (k==0) is only scaled, never deadzoned (avoids blocking).
                if k > 0 && q.abs() < keep_threshold as f64 {
                    *level = 0;
                    continue;
                }
                if qsk == qtk {
                    continue;
                }
                let new_level = q.round();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::data::{JpegComponent, JpegComponentType, JpegData, JpegQuantTable};

    fn bare(quant: Vec<JpegQuantTable>, components: Vec<JpegComponent>) -> JpegData {
        JpegData {
            width: 8,
            height: 8,
            restart_interval: 0,
            app_data: Vec::new(),
            app_marker_type: Vec::new(),
            com_data: Vec::new(),
            quant,
            huffman_code: Vec::new(),
            components,
            scan_info: Vec::new(),
            marker_order: Vec::new(),
            inter_marker_data: Vec::new(),
            tail_data: Vec::new(),
            has_zero_padding_bit: false,
            padding_bits: Vec::new(),
            component_type: JpegComponentType::YCbCr,
        }
    }

    fn tiny_jpeg(luma: [i16; 64], q: [i32; 64]) -> JpegData {
        bare(
            vec![JpegQuantTable {
                values: q,
                precision: 0,
                index: 0,
                is_last: true,
            }],
            vec![JpegComponent {
                id: 1,
                h_samp_factor: 1,
                v_samp_factor: 1,
                quant_idx: 0,
                width_in_blocks: 1,
                height_in_blocks: 1,
                coeffs: luma.to_vec(),
            }],
        )
    }

    #[test]
    fn scale_le_one_is_noop() {
        let mut c = [0i16; 64];
        c[0] = 40;
        c[1] = 7;
        let q = [8i32; 64];
        let mut j = tiny_jpeg(c, q);
        coarsen_coefficients(&mut j, 1.0);
        assert_eq!(j.components[0].coeffs[0], 40);
        assert_eq!(j.components[0].coeffs[1], 7);
        assert_eq!(j.quant[0].values[1], 8);
    }

    #[test]
    fn uniform_scale_preserves_dequant_value() {
        // level 40 @ q 8 => dequant 320. scale 2 => qt 16 => new level 20 => 20*16=320.
        let mut c = [0i16; 64];
        c[0] = 40; // DC
        c[1] = 40; // AC, dequant 320, well above deadzone
        let q = [8i32; 64];
        let mut j = tiny_jpeg(c, q);
        coarsen_coefficients(&mut j, 2.0);
        assert_eq!(j.quant[0].values[1], 16);
        assert_eq!(j.components[0].coeffs[1], 20); // 320/16
        // DC also scaled (not deadzoned): 40*8/16 = 20
        assert_eq!(j.components[0].coeffs[0], 20);
    }

    #[test]
    fn deadzone_zeros_small_ac_but_not_dc() {
        // AC level 1 @ q 8, scale 1.5 => qt 12, q-ratio = 1*8/12 = 0.667.
        // dz 0.2 => threshold 0.7 > 0.667 => AC zeroed. DC never deadzoned.
        let mut c = [0i16; 64];
        c[0] = 1; // DC level 1
        c[1] = 1; // AC level 1 (the noise we want to drop)
        let q = [8i32; 64];
        let mut j = tiny_jpeg(c, q);
        coarsen_coefficients_dz(&mut j, 1.5, 0.2);
        assert_eq!(j.components[0].coeffs[1], 0, "small AC must be deadzoned");
        assert_ne!(j.components[0].coeffs[0], 0, "DC must never be deadzoned");
    }

    #[test]
    fn shared_chroma_table_scaled_once() {
        // Two chroma components sharing quant_idx=1 must NOT double-scale the
        // shared table (the share-scale / scale-squared bug, COMPENDIUM §10.2).
        let mk = |qi| JpegComponent {
            id: 2,
            h_samp_factor: 1,
            v_samp_factor: 1,
            quant_idx: qi,
            width_in_blocks: 1,
            height_in_blocks: 1,
            coeffs: vec![0i16; 64],
        };
        let mut j = bare(
            vec![
                JpegQuantTable {
                    values: [8; 64],
                    precision: 0,
                    index: 0,
                    is_last: false,
                },
                JpegQuantTable {
                    values: [10; 64],
                    precision: 0,
                    index: 1,
                    is_last: true,
                },
            ],
            vec![mk(0), mk(1), mk(1)], // Cb and Cr share table 1
        );
        coarsen_coefficients(&mut j, 2.0);
        // table 1 scaled exactly once: 10*2 = 20, NOT 10*2*2 = 40.
        assert_eq!(
            j.quant[1].values[0], 20,
            "shared chroma table must scale once"
        );
    }
}
