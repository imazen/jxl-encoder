// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.
#![forbid(unsafe_code)]

//! Extended feature extraction.
//!
//! Computes the same fields as
//! [`jxl_encoder::vardct::encoder::ZenanalyzeProxies`] (m3_colourfulness,
//! flat_color_block_ratio, edge_density, luma_var) PLUS the mask1x1
//! percentile features (`mask_p25`, `mask_median`) that the W44-150 /
//! W44-151 / W44-152 / W44-168 discriminators read.
//!
//! The mask1x1 percentiles come from the encoder's adaptive-quant
//! pipeline (`vardct::adaptive_quant::compute_mask1x1` → median + p25
//! of the masking field). For the sweep runner we replicate them
//! cheaply with a 2-D Laplacian on a downsampled luma plane — the exact
//! mask1x1 field is computed during encode and we don't surface it
//! externally yet.
//!
//! Future work: surface the encoder-internal mask1x1 percentiles via a
//! `__profile` build flag so we don't double-compute. Tracked under
//! W44-213+.

use serde::{Deserialize, Serialize};

/// Extended feature row attached to each [`crate::SweepCellRow`]. Lands
/// in the Parquet `feat_*` columns.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ExtendedFeatures {
    /// Zenanalyze M3 colourfulness (Hasler-Süsstrunk). Used by W44-91 /
    /// W44-98 / W44-99 discriminators.
    pub m3_colourfulness: f32,
    /// Fraction of 8×8 blocks where every channel's u8 range ≤ 4.
    /// Used by W44-91 / W44-96 fcbr discriminator.
    pub flat_color_block_ratio: f32,
    /// Fraction of interior pixels with Sobel |∇Y| > 30. Used by
    /// W44-96 discriminator.
    pub edge_density: f32,
    /// BT.601 luma variance on sRGB u8. Used by W44-176 terminal-class
    /// sub-discriminator.
    pub luma_var: f32,
    /// 25th percentile of mask1x1 (Laplacian-based masking) over 8×8
    /// blocks. Used by W44-150 / W44-151 / W44-168 discriminators.
    /// Computed via the same shape as `vardct::adaptive_quant`'s mask
    /// helper (cheap reproduction; encoder-exact path costs ~5–10 ms).
    pub mask_p25: f32,
    /// 50th percentile of mask1x1. Used by W22-1 / W44-29 / W44-65 /
    /// W44-168 screenshot discriminators.
    pub mask_median: f32,
    /// 75th percentile of mask1x1. Tail diagnostic.
    pub mask_p75: f32,
    /// Mean luma over the image (BT.601). Cheap; aids tone-mapping
    /// follow-on chunks.
    pub luma_mean: f32,
    /// Width × height (logical pixel count). Lands as f32 because
    /// downstream MLPs ingest f32 columns and we want the column
    /// dtype-stable.
    pub n_pixels: f32,
    /// Aspect ratio = `w / h`. Used by some tile-budget gates.
    pub aspect: f32,
    /// Bytes-per-pixel of the source (3 for Rgb8, 4 for Rgba8). Lands
    /// as f32 (single-value categorical proxy).
    pub bpp_source: f32,
    /// Average per-pixel byte entropy on a histogram of the RGB(A)
    /// source. Cheap; aids picker-class follow-on chunks.
    pub byte_entropy_bits: f32,
}

/// Compute the extended feature row from raw sRGB pixel bytes.
///
/// `bpp` is bytes-per-pixel (3 for RGB / BGR, 4 for RGBA / BGRA). The
/// `r_off`/`g_off`/`b_off` triple selects the byte offsets within each
/// pixel.
pub fn compute_extended_features(
    pixels: &[u8],
    width: usize,
    height: usize,
    bpp: usize,
    r_off: usize,
    g_off: usize,
    b_off: usize,
) -> ExtendedFeatures {
    // ZenanalyzeProxies is re-exported under the `__pre_quantized`
    // back-door module (see `jxl-encoder/src/lib.rs:478`). The proper
    // user-facing path was added in the same W44-91 commit but the
    // re-export only lives under `__pre_quantized` today.
    use jxl_encoder::__pre_quantized::ZenanalyzeProxies;
    let p = ZenanalyzeProxies::compute_srgb_u8(pixels, width, height, bpp, r_off, g_off, b_off);

    let n_pix = (width * height) as f64;
    let mut y_sum = 0.0f64;
    // Single-pass luma stats and a cheap mask1x1 reproduction (per-pixel
    // Laplacian; aggregated to 8×8 mean for the percentile computation).
    let mut luma_plane = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let off = (y * width + x) * bpp;
            let r = pixels[off + r_off] as f64;
            let g = pixels[off + g_off] as f64;
            let b = pixels[off + b_off] as f64;
            let yl = 0.299 * r + 0.587 * g + 0.114 * b;
            y_sum += yl;
            luma_plane[y * width + x] = yl.round().clamp(0.0, 255.0) as u8;
        }
    }
    let luma_mean = (y_sum / n_pix.max(1.0)) as f32;

    // Reproduction of mask1x1: per-pixel `1.0 / (log1p(|Y - avg4|) + 0.01)`
    // averaged on 8×8 blocks. Matches the shape of
    // `vardct::adaptive_quant::compute_mask1x1` (Laplacian → log1p →
    // recip), without the libjxl gamma curve — gives a comparable
    // distribution shape for sweep-row features; the exact thresholds
    // (50 / 80 / 85 / 95) the discriminators use are not strictly the
    // same as the encoder-internal mask, but the COLUMN is stable for
    // MLP training (the picker will learn its own thresholds from these
    // features).
    let mut block_mask_vals: Vec<f32> = Vec::with_capacity((width / 8) * (height / 8));
    let blocks_x = width / 8;
    let blocks_y = height / 8;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut acc = 0.0f32;
            for dy in 0..8 {
                for dx in 0..8 {
                    let yy = by * 8 + dy;
                    let xx = bx * 8 + dx;
                    let centre = luma_plane[yy * width + xx] as f32;
                    let n = if yy > 0 {
                        luma_plane[(yy - 1) * width + xx] as f32
                    } else {
                        centre
                    };
                    let s = if yy + 1 < height {
                        luma_plane[(yy + 1) * width + xx] as f32
                    } else {
                        centre
                    };
                    let w = if xx > 0 {
                        luma_plane[yy * width + xx - 1] as f32
                    } else {
                        centre
                    };
                    let e = if xx + 1 < width {
                        luma_plane[yy * width + xx + 1] as f32
                    } else {
                        centre
                    };
                    let avg = 0.25 * (n + s + w + e);
                    let diff = (centre - avg).abs();
                    let mask = 100.0 / (libm_log1p_f32(diff) + 0.01);
                    acc += mask;
                }
            }
            block_mask_vals.push(acc / 64.0);
        }
    }
    block_mask_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let percentile = |q: f64| -> f32 {
        if block_mask_vals.is_empty() {
            return 0.0;
        }
        let idx = (q * (block_mask_vals.len() - 1) as f64).round() as usize;
        block_mask_vals[idx.min(block_mask_vals.len() - 1)]
    };
    let mask_p25 = percentile(0.25);
    let mask_median = percentile(0.50);
    let mask_p75 = percentile(0.75);

    // Byte entropy on the RGB(A) histogram.
    let mut hist = [0u64; 256];
    for byte in pixels.iter() {
        hist[*byte as usize] += 1;
    }
    let total = pixels.len().max(1) as f64;
    let mut h = 0.0f64;
    for &c in hist.iter() {
        if c > 0 {
            let p = c as f64 / total;
            h -= p * p.log2();
        }
    }

    ExtendedFeatures {
        m3_colourfulness: p.m3_colourfulness,
        flat_color_block_ratio: p.flat_color_block_ratio,
        edge_density: p.edge_density,
        luma_var: p.luma_var,
        mask_p25,
        mask_median,
        mask_p75,
        luma_mean,
        n_pixels: (width * height) as f32,
        aspect: if height == 0 { 0.0 } else { width as f32 / height as f32 },
        bpp_source: bpp as f32,
        byte_entropy_bits: h as f32,
    }
}

#[inline]
fn libm_log1p_f32(x: f32) -> f32 {
    (1.0 + x).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_features_on_solid_grey() {
        // 32×32 solid mid-grey: mask should be saturated (no
        // gradient), m3 ≈ 0, edge_density ≈ 0, fcbr ≈ 1.0.
        let pixels = vec![128u8; 32 * 32 * 4];
        let f = compute_extended_features(&pixels, 32, 32, 4, 0, 1, 2);
        assert!(f.m3_colourfulness < 0.01, "m3 expected ~0, got {}", f.m3_colourfulness);
        assert!(f.flat_color_block_ratio > 0.99, "fcbr expected ~1, got {}", f.flat_color_block_ratio);
        assert!(f.edge_density < 0.01, "edge_density expected ~0, got {}", f.edge_density);
        // mask_median should be high (saturated by the +0.01 epsilon) on flat content
        assert!(f.mask_median > 50.0, "mask_median expected high on flat, got {}", f.mask_median);
        assert!(f.n_pixels == 1024.0);
        assert!(f.aspect == 1.0);
        assert!(f.bpp_source == 4.0);
    }

    #[test]
    fn smoke_features_on_2x2_checkerboard() {
        // 32×32 4x4-block binary checkerboard: m3 ≈ 0 (no chroma),
        // edge_density positive (Sobel sees boundaries between
        // blocks), fcbr low (each 8×8 block straddles a checker
        // boundary so range ≫ 4).
        // NOTE: a 1-pixel binary checker has edge_density = 0
        // because the Sobel kernel cancels on a fully-alternating
        // pattern (gx = gy = 0 everywhere). Larger checkers do
        // produce edges at block boundaries — which is what the
        // ZenanalyzeProxies::compute_srgb_u8 path measures.
        let mut pixels = vec![0u8; 32 * 32 * 4];
        for y in 0..32usize {
            for x in 0..32usize {
                let v = if ((x / 4) + (y / 4)) % 2 == 0 { 255u8 } else { 0u8 };
                let off = (y * 32 + x) * 4;
                pixels[off] = v;
                pixels[off + 1] = v;
                pixels[off + 2] = v;
                pixels[off + 3] = 255;
            }
        }
        let f = compute_extended_features(&pixels, 32, 32, 4, 0, 1, 2);
        assert!(
            f.edge_density > 0.05,
            "edge_density expected > 0.05 on 4x4 checker, got {}",
            f.edge_density
        );
        assert!(
            f.flat_color_block_ratio < 0.5,
            "fcbr expected low (8x8 blocks straddle), got {}",
            f.flat_color_block_ratio
        );
        assert!(
            f.m3_colourfulness < 1.0,
            "m3 expected low on greyscale checker, got {}",
            f.m3_colourfulness
        );
    }
}
