//! W44-91 proxy probe: measure cheap encoder-internal proxies on the
//! W44-79 wedge set to find a discriminator that catches 1189261 without
//! catching the 1025469-class REGRESSION-band images, using only data the
//! encoder hot path already computes (XYB planes + mask1x1).
//!
//! All values are taken via [`EncoderPrecomputed::compute`] to ensure they
//! exactly match what the encoder dispatch path sees (same SIMD XYB LUT
//! path, same pre-gaborish mask1x1, etc).
//!
//! Proxies tested (all derivable from XYB planes that already exist in
//! `EncoderPrecomputed`):
//!
//! - `chroma_var_x_pre_gab`: variance of pre-gaborish XYB X plane
//! - `chroma_var_b_pre_gab`: variance of pre-gaborish XYB B plane
//! - `chroma_proxy`: sqrt(var_x + var_b) — analog of zenanalyze
//!   `chroma_complexity = sqrt(cb_var + cr_var)`.
//! - `mask1x1_median`: existing pipeline statistic (post-gaborish path)
//! - `mask1x1_p10`, `mask1x1_p25`: low-percentile mask1x1 — captures
//!   whether the image has SOME high-detail regions even when median
//!   is moderate.
//! - `flat_x_block_ratio`: fraction of 8x8 blocks where X channel has
//!   std < threshold (proxy for `flat_color_block_ratio`)
//! - `xb_var_ratio`: variance of (X+B) / variance of Y — separates
//!   chroma-dominant content (high) from luma-dominant.

use jxl_encoder::__pre_quantized::{EffortProfile, EncoderPrecomputed};
use jxl_encoder::__test_exports::xyb::srgb_to_linear_value;
use jxl_encoder::api::EncoderMode;
use std::path::Path;

#[derive(Debug, Default)]
struct ImageProxies {
    name: String,
    width: u32,
    height: u32,
    mask1x1_median: f32,
    mask1x1_p10: f32,
    mask1x1_p25: f32,
    chroma_var_x_pre_gab: f32,
    chroma_var_b_pre_gab: f32,
    chroma_proxy: f32,
    flat_x_block_ratio: f32,
    xb_var_ratio: f32,
    masking_med: f32,
    masking_p10: f32,
    masking_p90: f32,
    high_qff_ratio: f32,
    m3_colourfulness: f32,
    fcbr_proxy: f32,
}

fn variance(plane: &[f32]) -> f32 {
    let n = plane.len() as f64;
    if n < 1.0 {
        return 0.0;
    }
    let mean = plane.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = plane
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    var as f32
}

fn percentile(values: &mut [f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() as f32 - 1.0) * p).floor() as usize;
    let idx = idx.min(values.len() - 1);
    values.select_nth_unstable_by(idx, |a, b| {
        a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
    });
    values[idx]
}

fn flat_x_block_ratio(xyb_x: &[f32], w: usize, h: usize, stride: usize, threshold: f32) -> f32 {
    let blocks_x = w / 8;
    let blocks_y = h / 8;
    if blocks_x == 0 || blocks_y == 0 {
        return 0.0;
    }
    let total = blocks_x * blocks_y;
    let mut flat = 0usize;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut sum = 0.0_f64;
            let mut sum_sq = 0.0_f64;
            for dy in 0..8 {
                for dx in 0..8 {
                    let v = xyb_x[(by * 8 + dy) * stride + (bx * 8 + dx)] as f64;
                    sum += v;
                    sum_sq += v * v;
                }
            }
            let mean = sum / 64.0;
            let var = (sum_sq / 64.0) - mean * mean;
            if var.max(0.0).sqrt() < threshold as f64 {
                flat += 1;
            }
        }
    }
    flat as f32 / total as f32
}

fn probe(path: &Path) -> ImageProxies {
    let img = image::open(path).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let raw = rgb.into_raw();
    // Convert to linear RGB the same way the encoder front-end does.
    let mut linear_rgb = vec![0.0_f32; w * h * 3];
    for i in 0..(w * h * 3) {
        linear_rgb[i] = srgb_to_linear_value(raw[i] as f32);
    }

    // --- Cheap zenanalyze-equivalent proxies computed directly from sRGB ---
    //
    // Hasler-Süsstrunk M3 colourfulness:
    //   rg = R − G, yb = 0.5·(R+G) − B (per pixel, sRGB u8 scale)
    //   M3 = sqrt(σ_rg² + σ_yb²) + 0.3 * sqrt(μ_rg² + μ_yb²)
    // This matches zenanalyze/src/tier1.rs (Hasler-Süsstrunk M3) exactly
    // up to lane order / FP precision.
    let mut rg_sum = 0.0_f64;
    let mut rg_sq_sum = 0.0_f64;
    let mut yb_sum = 0.0_f64;
    let mut yb_sq_sum = 0.0_f64;
    let n_pix = (w * h) as f64;
    for i in 0..(w * h) {
        let r = raw[i * 3] as f64;
        let g = raw[i * 3 + 1] as f64;
        let b = raw[i * 3 + 2] as f64;
        let rg = r - g;
        let yb = 0.5 * (r + g) - b;
        rg_sum += rg;
        rg_sq_sum += rg * rg;
        yb_sum += yb;
        yb_sq_sum += yb * yb;
    }
    let mu_rg = rg_sum / n_pix;
    let mu_yb = yb_sum / n_pix;
    let var_rg = (rg_sq_sum / n_pix - mu_rg * mu_rg).max(0.0);
    let var_yb = (yb_sq_sum / n_pix - mu_yb * mu_yb).max(0.0);
    let m3_colourfulness = (var_rg + var_yb).sqrt() + 0.3 * (mu_rg * mu_rg + mu_yb * mu_yb).sqrt();
    let m3_colourfulness = m3_colourfulness as f32;

    // flat_color_block_ratio analog on 8x8 sRGB blocks:
    //   Block is "flat" if max(stdev(R), stdev(G), stdev(B)) < threshold.
    // zenanalyze uses a threshold derived from BT.601 luma variance; we
    // match the same intuition by checking each channel's stdev directly.
    let blocks_x = w / 8;
    let blocks_y = h / 8;
    let mut flat_blocks = 0usize;
    let total_blocks = blocks_x * blocks_y;
    if total_blocks > 0 {
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                // Exact zenanalyze tier1.rs definition: flat = r_range <= 4
                // AND g_range <= 4 AND b_range <= 4.
                let mut r_min = 255u8;
                let mut r_max = 0u8;
                let mut g_min = 255u8;
                let mut g_max = 0u8;
                let mut b_min = 255u8;
                let mut b_max = 0u8;
                for dy in 0..8 {
                    for dx in 0..8 {
                        let off = ((by * 8 + dy) * w + (bx * 8 + dx)) * 3;
                        let r = raw[off];
                        let g = raw[off + 1];
                        let b = raw[off + 2];
                        if r < r_min {
                            r_min = r;
                        }
                        if r > r_max {
                            r_max = r;
                        }
                        if g < g_min {
                            g_min = g;
                        }
                        if g > g_max {
                            g_max = g;
                        }
                        if b < b_min {
                            b_min = b;
                        }
                        if b > b_max {
                            b_max = b;
                        }
                    }
                }
                let r_range = (r_max as i32) - (r_min as i32);
                let g_range = (g_max as i32) - (g_min as i32);
                let b_range = (b_max as i32) - (b_min as i32);
                if r_range <= 4 && g_range <= 4 && b_range <= 4 {
                    flat_blocks += 1;
                }
            }
        }
    }
    let fcbr_proxy = if total_blocks > 0 {
        flat_blocks as f32 / total_blocks as f32
    } else {
        0.0
    };
    // Build minimal effort profile (e7 reference).
    let profile = EffortProfile::lossy(7, EncoderMode::Reference);
    let pre = EncoderPrecomputed::compute(
        w,
        h,
        &linear_rgb,
        /*distance*/ 1.0,
        /*cfl_enabled*/ true,
        /*ac_strategy_enabled*/ true,
        /*pixel_domain_loss*/ true,
        /*enable_noise*/ false,
        /*enable_denoise*/ false,
        /*enable_gaborish*/ true,
        /*force_strategy*/ None,
        &profile,
        None,
    )
    .expect("EncoderPrecomputed::compute");

    let stride = pre.padded_width;
    // Pre-gaborish XYB if available (matches what mask1x1 was computed against)
    let (xyb_x_view, xyb_y_view, xyb_b_view) = match pre.xyb_pre_gaborish.as_ref() {
        Some(arr) => (arr[0].as_slice(), arr[1].as_slice(), arr[2].as_slice()),
        None => (
            pre.xyb_x.as_slice(),
            pre.xyb_y.as_slice(),
            pre.xyb_b.as_slice(),
        ),
    };

    // mask1x1 percentiles (only available when pixel_domain_loss is on)
    let (mask_med, mask_p10, mask_p25) = if let Some(mask) = pre.mask1x1.as_ref() {
        // mask is computed on stride==padded_width over height==padded_height,
        // but only the [0..width)x[0..height) region is meaningful.
        let mut buf: Vec<f32> = Vec::with_capacity(w * h);
        for y in 0..h {
            let row_off = y * stride;
            buf.extend_from_slice(&mask[row_off..row_off + w]);
        }
        let mut m1 = buf.clone();
        let med = percentile(&mut m1, 0.50);
        let mut m1 = buf.clone();
        let p10 = percentile(&mut m1, 0.10);
        let p25 = percentile(&mut buf, 0.25);
        (med, p10, p25)
    } else {
        (0.0, 0.0, 0.0)
    };

    // Variances on the unpadded region. Use the pre-gaborish view because
    // the mask1x1 was computed against that, so the discriminator should
    // be defined consistently against it.
    let mut x_unpadded: Vec<f32> = Vec::with_capacity(w * h);
    let mut b_unpadded: Vec<f32> = Vec::with_capacity(w * h);
    let mut y_unpadded: Vec<f32> = Vec::with_capacity(w * h);
    for y in 0..h {
        let off = y * stride;
        x_unpadded.extend_from_slice(&xyb_x_view[off..off + w]);
        b_unpadded.extend_from_slice(&xyb_b_view[off..off + w]);
        y_unpadded.extend_from_slice(&xyb_y_view[off..off + w]);
    }
    let var_x = variance(&x_unpadded);
    let var_b = variance(&b_unpadded);
    let var_y = variance(&y_unpadded);
    let chroma_proxy = (var_x + var_b).sqrt();
    let xb_var_ratio = if var_y > 1e-9 {
        (var_x + var_b) / var_y
    } else {
        0.0
    };
    let flat = flat_x_block_ratio(xyb_x_view, w, h, stride, 0.0005);

    // Look at masking field (per-block) stats — these capture the
    // adaptive_quant decisions which embed perceptual block-level
    // info similar to zenanalyze's flat_color_block_ratio (which is
    // also block-level).
    // masking is shape [xsize_blocks × ysize_blocks] (per-8x8-block).
    let _ = pre.masking.len(); // ensure we read it
    let mut masking_buf = pre.masking.clone();
    let masking_med = percentile(&mut masking_buf, 0.50);
    let mut masking_buf = pre.masking.clone();
    let masking_p10 = percentile(&mut masking_buf, 0.10);
    let mut masking_buf = pre.masking.clone();
    let masking_p90 = percentile(&mut masking_buf, 0.90);

    // Fraction of blocks with high quant_field_float (= adaptive_quant
    // demands MORE quant precision = high-detail block). High-detail
    // fraction is what discriminates colourful textured content from
    // flat photo content.
    let qff = &pre.quant_field_float;
    let qff_n = qff.len();
    let high_qff = if qff_n > 0 {
        let mut buf = qff.clone();
        let p90 = percentile(&mut buf, 0.90);
        let p50 = percentile(&mut qff.clone(), 0.50);
        // Spread: high p90/p50 ratio => more "spiky" detail distribution
        if p50 > 1e-6 { p90 / p50 } else { 0.0 }
    } else {
        0.0
    };

    let _ = (masking_med, masking_p10, masking_p90, high_qff);

    ImageProxies {
        name: path.file_name().unwrap().to_string_lossy().into_owned(),
        width: w as u32,
        height: h as u32,
        mask1x1_median: mask_med,
        mask1x1_p10: mask_p10,
        mask1x1_p25: mask_p25,
        chroma_var_x_pre_gab: var_x,
        chroma_var_b_pre_gab: var_b,
        chroma_proxy,
        flat_x_block_ratio: flat,
        xb_var_ratio,
        masking_med,
        masking_p10,
        masking_p90,
        high_qff_ratio: high_qff,
        m3_colourfulness,
        fcbr_proxy,
    }
}

fn main() {
    // Wedge set from W44-79 memo:
    // - TARGET: 1189261 (mask=69.08, W44-78 follow-on EV target)
    // - REGRESSION-band: 1025469, 1624487, 159550, 2079234, 2775196, 297394
    //   (mask1x1 in [50, 80] band, W44-78 variant-B widening regresses these)
    // - W44-78 reference cells that ALREADY fire (mask<50):
    //   1420710, 1531677, 1044329, 2389166, 3637739
    // - W44-78 reference that doesn't fire (mask>95): 1418519
    let base = Path::new("/home/lilith/work/codec-corpus/CID22/CID22-512/validation");
    let images = [
        ("TARGET", "1189261.png"),
        ("REGRESSION", "1025469.png"),
        ("REGRESSION", "1624487.png"),
        ("REGRESSION", "159550.png"),
        ("REGRESSION", "2079234.png"),
        ("REGRESSION", "2775196.png"),
        ("REGRESSION", "297394.png"),
        ("REFERENCE_FIRES", "1420710.png"),
        ("REFERENCE_FIRES", "1531677.png"),
        ("REFERENCE_FIRES", "1044329.png"),
        ("REFERENCE_FIRES", "2389166.png"),
        ("REFERENCE_FIRES", "3637739.png"),
        ("REFERENCE_SKIP", "1418519.png"),
    ];
    println!("# W44-91 proxy probe (via EncoderPrecomputed for pipeline parity)");
    println!(
        "class\timage\tw\th\tmask_med\tmask_p10\tmask_p25\tvar_x_pre\tvar_b_pre\tchroma_proxy\tflat_x_ratio\txb_var_ratio\tmasking_med\tmasking_p10\tmasking_p90\thigh_qff_ratio\tm3_colourfulness\tfcbr_proxy"
    );
    for (class, name) in &images {
        let p = base.join(name);
        if !p.exists() {
            eprintln!("MISSING: {}", p.display());
            continue;
        }
        let pr = probe(&p);
        println!(
            "{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.6}\t{:.6}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.3}\t{:.4}",
            class,
            pr.name,
            pr.width,
            pr.height,
            pr.mask1x1_median,
            pr.mask1x1_p10,
            pr.mask1x1_p25,
            pr.chroma_var_x_pre_gab,
            pr.chroma_var_b_pre_gab,
            pr.chroma_proxy,
            pr.flat_x_block_ratio,
            pr.xb_var_ratio,
            pr.masking_med,
            pr.masking_p10,
            pr.masking_p90,
            pr.high_qff_ratio,
            pr.m3_colourfulness,
            pr.fcbr_proxy,
        );
    }
}
