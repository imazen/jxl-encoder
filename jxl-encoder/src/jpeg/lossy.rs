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
    coarsen_coefficients_planar(jpeg, scale, dz, scale, dz);
}

/// Coarsen with **separate luma and chroma** scale + deadzone.
///
/// Human vision is far less sensitive to chroma detail than luma, so chroma
/// can be coarsened harder (larger `chroma_scale` / `chroma_dz`) for nearly
/// free perceptual cost — a classic, large RD win. Each quant *table* is
/// classified luma vs chroma by the components that use it (YCbCr: component 0
/// is luma, the rest chroma; grayscale/RGB are all treated as luma), so the
/// share-scale-safe per-table scaling naturally applies the right class. Both
/// classes follow the same per-coefficient rule as [`coarsen_coefficients_dz`]
/// (dequant-value-preserving requant + DC-protected AC deadzone).
pub fn coarsen_coefficients_planar(
    jpeg: &mut JpegData,
    luma_scale: f32,
    luma_dz: f32,
    chroma_scale: f32,
    chroma_dz: f32,
) {
    let any = luma_scale > 1.0 || chroma_scale > 1.0;
    if !any || jpeg.quant.is_empty() {
        return;
    }
    let luma_dz = luma_dz.clamp(0.0, 0.5);
    let chroma_dz = chroma_dz.clamp(0.0, 0.5);

    // Classify each quant table: chroma iff EVERY component using it is chroma
    // (luma wins on a shared table — never over-coarsen luma). For YCbCr the
    // first component is luma; grayscale/RGB/Custom have no chroma plane.
    let is_ycbcr = matches!(jpeg.component_type, super::data::JpegComponentType::YCbCr)
        && jpeg.components.len() >= 3;
    let ntables = jpeg.quant.len();
    let mut tbl_used_by_luma = vec![false; ntables];
    let mut tbl_used_by_chroma = vec![false; ntables];
    for (ci, comp) in jpeg.components.iter().enumerate() {
        let qi = comp.quant_idx as usize;
        if qi >= ntables {
            continue;
        }
        let is_chroma = is_ycbcr && ci >= 1;
        if is_chroma {
            tbl_used_by_chroma[qi] = true;
        } else {
            tbl_used_by_luma[qi] = true;
        }
    }
    let tbl_is_chroma: Vec<bool> = (0..ntables)
        .map(|t| tbl_used_by_chroma[t] && !tbl_used_by_luma[t])
        .collect();

    // 1. Snapshot ORIGINAL weights; compute coarsened TARGET weights once per
    //    table using that table's class scale (share-scale-safe).
    let orig: Vec<[i32; 64]> = jpeg.quant.iter().map(|t| t.values).collect();
    let mut target: Vec<[i32; 64]> = orig.clone();
    for (t, tbl) in target.iter_mut().enumerate() {
        let s = if tbl_is_chroma[t] {
            chroma_scale
        } else {
            luma_scale
        };
        // NaN-safe: a NaN scale must take the lossless path, exactly
        // like s <= 1.0 (plain `s <= 1.0` would let NaN fall through).
        if s.is_nan() || s <= 1.0 {
            continue;
        }
        for v in tbl.iter_mut() {
            *v = (((*v as f32) * s).round() as i32).clamp(1, 65535);
        }
    }

    // 2. Re-quantize each component's coefficients using its table's class.
    for comp in jpeg.components.iter_mut() {
        let qi = comp.quant_idx as usize;
        if qi >= orig.len() {
            continue;
        }
        let qs = &orig[qi];
        let qt = &target[qi];
        let class_dz = if tbl_is_chroma[qi] {
            chroma_dz
        } else {
            luma_dz
        };
        let keep_threshold = 0.5 + class_dz as f64;
        for block in comp.coeffs.chunks_mut(64) {
            for (k, level) in block.iter_mut().enumerate() {
                if *level == 0 {
                    continue;
                }
                let qsk = qs[k];
                let qtk = qt[k];
                let dequant = *level as f64 * qsk as f64;
                let q = dequant / qtk as f64;
                if k > 0 && q.abs() < keep_threshold {
                    *level = 0;
                    continue;
                }
                if qsk == qtk {
                    continue;
                }
                *level = q.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
            }
        }
    }

    // 3. Commit the coarsened tables.
    for (i, tbl) in jpeg.quant.iter_mut().enumerate() {
        tbl.values = target[i];
        if target[i].iter().any(|&v| v > 255) {
            tbl.precision = 1;
        }
    }
}

/// The proven single-knob coarsening policy from the RD frontier
/// (`benchmarks/jpeg_lossy_rd_frontier_2026-05-28`): given one `scale`, derive
/// (luma_scale, luma_dz, chroma_scale, chroma_dz).
///
/// - **AC deadzone** grows with the scale (`0.30·(scale−1)`, capped 0.45). On
///   the frontier this is a *strict Pareto win* — at a fixed scale, widening
///   the deadzone is both smaller and higher quality on 8/10 files (the ±1 AC
///   residue is perceptually harmful noise). DC is never deadzoned (handled by
///   the coarsen routines).
/// - **Mild chroma lead**: chroma coarsens `1.4×` the luma *delta*
///   (`1 + (scale−1)·1.4`), so chroma is always slightly coarser than luma but
///   never aggressively so — chroma ≥2.5× luma was dominated on *every* metric.
///
/// `scale ≤ 1.0` maps to a true lossless no-op `(1, 0, 1, 0)`.
pub fn coarsen_policy(scale: f32) -> (f32, f32, f32, f32) {
    // NaN-safe: NaN maps to the lossless no-op, same as scale <= 1.0.
    if scale.is_nan() || scale <= 1.0 {
        return (1.0, 0.0, 1.0, 0.0);
    }
    let luma_dz = (0.30 * (scale - 1.0)).min(0.45);
    let chroma_scale = 1.0 + (scale - 1.0) * 1.4;
    let chroma_dz = (luma_dz + 0.05).min(0.45);
    (scale, luma_dz, chroma_scale, chroma_dz)
}

/// Coarsen with the bundled single-knob [`coarsen_policy`] (deadzone + mild
/// chroma lead). The caller's quality loop only has to move one `scale` dial.
pub fn coarsen_coefficients_auto(jpeg: &mut JpegData, scale: f32) {
    let (ls, ldz, cs, cdz) = coarsen_policy(scale);
    coarsen_coefficients_planar(jpeg, ls, ldz, cs, cdz);
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
    fn policy_lossless_at_scale_one() {
        assert_eq!(coarsen_policy(1.0), (1.0, 0.0, 1.0, 0.0));
        assert_eq!(coarsen_policy(0.5), (1.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn policy_chroma_leads_luma_mildly() {
        let (ls, ldz, cs, cdz) = coarsen_policy(2.0);
        assert_eq!(ls, 2.0);
        assert!(ldz > 0.0, "deadzone must be on above scale 1.0");
        // chroma leads luma but not aggressively: luma < chroma < 1.5x luma.
        assert!(cs > ls, "chroma must lead luma");
        assert!(cs < ls * 1.5, "chroma lead must be mild (<1.5x), got {cs}");
        assert!(cdz >= ldz);
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
