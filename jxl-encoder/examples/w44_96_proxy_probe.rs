//! W44-96 proxy probe: extend W44-91's encoder-internal proxy battery
//! to find a discriminator that splits the 4 mask<50 CID22 photos where
//! the W44-29 gate already fires:
//!
//! WANTS variant Z (dct32x32=1.20) lift:
//!   - 1420710 (mask=39.55) — wins -191 to -2423 bytes across e6-e9 d=5
//!   - 1531677 (mask=35.63) — wins -55 to -1149 bytes
//!
//! REJECTS variant Z (current default dct32x32=1.34 is optimal):
//!   - 2389166 (mask=46.24) — SSIM2 -0.82 on e6 d=4 (regresses FIXED→OPEN)
//!   - 3637739 (mask=47.80) — SSIM2 -0.43 / FIXED→OPEN
//!
//! Probes:
//! - W44-91 fields: mask1x1 percentiles, m3_colourfulness, fcbr_proxy,
//!   chroma proxies, flat_x_block_ratio, masking percentiles
//! - Additional candidates (per W44-95 closing note):
//!     - edge_density (Sobel-like high-freq energy)
//!     - high_freq_energy_ratio (post-XYB sum of higher-bin DCT energy)
//!     - palette_log2_size proxy (sRGB unique-color count log2)

use jxl_encoder::__pre_quantized::{EffortProfile, EncoderPrecomputed};
use jxl_encoder::api::EncoderMode;
use jxl_encoder::color::xyb::srgb_to_linear_value;
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Default)]
struct ImageProxies {
    name: String,
    width: u32,
    height: u32,
    // W44-91 fields (cheap sRGB-direct)
    m3_colourfulness: f32,
    fcbr_proxy: f32,
    // W44-91 EncoderPrecomputed-derived fields
    mask1x1_median: f32,
    mask1x1_p10: f32,
    mask1x1_p25: f32,
    mask1x1_p75: f32,
    chroma_var_x_pre_gab: f32,
    chroma_var_b_pre_gab: f32,
    chroma_proxy: f32,
    flat_x_block_ratio: f32,
    xb_var_ratio: f32,
    masking_med: f32,
    masking_p10: f32,
    masking_p90: f32,
    high_qff_ratio: f32,
    // W44-96 additional candidates
    edge_density: f32,           // Sobel-like gradient density on luma
    high_freq_energy_ratio: f32, // ratio of (block 8x8 high-freq energy) on Y plane
    palette_log2_size: f32,      // log2 of unique sRGB color count
    var_y: f32,                  // luma variance
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

/// Sobel-like edge density on luma. Returns the fraction of pixels with
/// gradient magnitude exceeding a threshold relative to per-image scale.
/// Cheap proxy for zenanalyze tier1 `edge_density`.
fn edge_density(raw: &[u8], w: usize, h: usize) -> f32 {
    if w < 3 || h < 3 {
        return 0.0;
    }
    let luma = |y: usize, x: usize| -> f32 {
        let off = (y * w + x) * 3;
        // BT.601 luma
        0.299 * raw[off] as f32 + 0.587 * raw[off + 1] as f32 + 0.114 * raw[off + 2] as f32
    };
    let mut grad_squares: Vec<f32> = Vec::with_capacity((w - 2) * (h - 2));
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            // Sobel Gx
            let gx = -luma(y - 1, x - 1) - 2.0 * luma(y, x - 1) - luma(y + 1, x - 1)
                + luma(y - 1, x + 1)
                + 2.0 * luma(y, x + 1)
                + luma(y + 1, x + 1);
            let gy = -luma(y - 1, x - 1) - 2.0 * luma(y - 1, x) - luma(y - 1, x + 1)
                + luma(y + 1, x - 1)
                + 2.0 * luma(y + 1, x)
                + luma(y + 1, x + 1);
            grad_squares.push(gx * gx + gy * gy);
        }
    }
    if grad_squares.is_empty() {
        return 0.0;
    }
    // edge if magnitude > ~30 (i.e. squared > 900). Sobel scale ~8x diff.
    let n = grad_squares.len();
    let edges = grad_squares.iter().filter(|&&g| g > 900.0).count();
    edges as f32 / n as f32
}

/// High-frequency energy ratio on Y plane. For each 8x8 block, compute
/// the fraction of total squared deviation that comes from the per-pixel
/// (val - 5x5 mean) signal. Higher = more high-freq content.
fn high_freq_energy_ratio_y(xyb_y: &[f32], w: usize, h: usize, stride: usize) -> f32 {
    let bx = w / 8;
    let by_n = h / 8;
    if bx == 0 || by_n == 0 {
        return 0.0;
    }
    let mut hf_sum = 0.0_f64;
    let mut total_sum = 0.0_f64;
    for by in 0..by_n {
        for bxi in 0..bx {
            let mut block_sum = 0.0_f64;
            let mut block_sum_sq = 0.0_f64;
            for dy in 0..8 {
                for dx in 0..8 {
                    let v = xyb_y[(by * 8 + dy) * stride + (bxi * 8 + dx)] as f64;
                    block_sum += v;
                    block_sum_sq += v * v;
                }
            }
            let mean = block_sum / 64.0;
            let var = (block_sum_sq / 64.0) - mean * mean;
            total_sum += var.max(0.0);
            // HF energy = variance minus low-freq variance approximation
            // (computed as variance of 2x2 averages — captures DC + a few low bins)
            let mut lf_sum = 0.0_f64;
            let mut lf_sum_sq = 0.0_f64;
            for dy in 0..4 {
                for dx in 0..4 {
                    let mut s = 0.0_f64;
                    for ddy in 0..2 {
                        for ddx in 0..2 {
                            s += xyb_y[(by * 8 + dy * 2 + ddy) * stride + (bxi * 8 + dx * 2 + ddx)]
                                as f64;
                        }
                    }
                    let avg = s / 4.0;
                    lf_sum += avg;
                    lf_sum_sq += avg * avg;
                }
            }
            let lf_mean = lf_sum / 16.0;
            let lf_var = (lf_sum_sq / 16.0) - lf_mean * lf_mean;
            let hf_var = (var - lf_var.max(0.0)).max(0.0);
            hf_sum += hf_var;
        }
    }
    if total_sum > 1e-9 {
        (hf_sum / total_sum) as f32
    } else {
        0.0
    }
}

/// Approximate zenanalyze `palette_log2_size`: log2 of unique sRGB
/// quantized colors (using 6-bit per channel quantization to bound RAM).
fn palette_log2_size(raw: &[u8], w: usize, h: usize) -> f32 {
    let mut seen: HashSet<u32> = HashSet::new();
    for i in 0..(w * h) {
        let r = (raw[i * 3] >> 2) as u32;
        let g = (raw[i * 3 + 1] >> 2) as u32;
        let b = (raw[i * 3 + 2] >> 2) as u32;
        seen.insert((r << 12) | (g << 6) | b);
    }
    let n = seen.len().max(1) as f32;
    n.log2()
}

fn probe(path: &Path) -> ImageProxies {
    let img = image::open(path).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let raw = rgb.into_raw();
    let mut linear_rgb = vec![0.0_f32; w * h * 3];
    for i in 0..(w * h * 3) {
        linear_rgb[i] = srgb_to_linear_value(raw[i] as f32);
    }

    // M3 colourfulness (Hasler-Süsstrunk) — sRGB u8 direct
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
    let m3 = (var_rg + var_yb).sqrt() + 0.3 * (mu_rg * mu_rg + mu_yb * mu_yb).sqrt();

    // FCBR
    let blocks_x = w / 8;
    let blocks_y = h / 8;
    let mut flat_blocks = 0usize;
    let total_blocks = blocks_x * blocks_y;
    if total_blocks > 0 {
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
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
                if (r_max as i32 - r_min as i32) <= 4
                    && (g_max as i32 - g_min as i32) <= 4
                    && (b_max as i32 - b_min as i32) <= 4
                {
                    flat_blocks += 1;
                }
            }
        }
    }
    let fcbr = if total_blocks > 0 {
        flat_blocks as f32 / total_blocks as f32
    } else {
        0.0
    };

    let ed = edge_density(&raw, w, h);
    let plog2 = palette_log2_size(&raw, w, h);

    // EncoderPrecomputed for XYB-derived stats
    let profile = EffortProfile::lossy(7, EncoderMode::Reference);
    let pre = EncoderPrecomputed::compute(
        w,
        h,
        &linear_rgb,
        /*distance*/ 5.0,
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
    let (xyb_x_view, xyb_y_view, xyb_b_view) = match pre.xyb_pre_gaborish.as_ref() {
        Some(arr) => (arr[0].as_slice(), arr[1].as_slice(), arr[2].as_slice()),
        None => (
            pre.xyb_x.as_slice(),
            pre.xyb_y.as_slice(),
            pre.xyb_b.as_slice(),
        ),
    };

    let (mask_med, mask_p10, mask_p25, mask_p75) = if let Some(mask) = pre.mask1x1.as_ref() {
        let mut buf: Vec<f32> = Vec::with_capacity(w * h);
        for y in 0..h {
            let row_off = y * stride;
            buf.extend_from_slice(&mask[row_off..row_off + w]);
        }
        let mut m1 = buf.clone();
        let med = percentile(&mut m1, 0.50);
        let mut m1 = buf.clone();
        let p10 = percentile(&mut m1, 0.10);
        let mut m1 = buf.clone();
        let p25 = percentile(&mut m1, 0.25);
        let p75 = percentile(&mut buf, 0.75);
        (med, p10, p25, p75)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

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

    let mut masking_buf = pre.masking.clone();
    let masking_med = percentile(&mut masking_buf, 0.50);
    let mut masking_buf = pre.masking.clone();
    let masking_p10 = percentile(&mut masking_buf, 0.10);
    let mut masking_buf = pre.masking.clone();
    let masking_p90 = percentile(&mut masking_buf, 0.90);

    let qff = &pre.quant_field_float;
    let qff_n = qff.len();
    let high_qff = if qff_n > 0 {
        let mut buf = qff.clone();
        let p90 = percentile(&mut buf, 0.90);
        let p50 = percentile(&mut qff.clone(), 0.50);
        if p50 > 1e-6 { p90 / p50 } else { 0.0 }
    } else {
        0.0
    };

    let hfer = high_freq_energy_ratio_y(xyb_y_view, w, h, stride);

    ImageProxies {
        name: path.file_name().unwrap().to_string_lossy().into_owned(),
        width: w as u32,
        height: h as u32,
        m3_colourfulness: m3 as f32,
        fcbr_proxy: fcbr,
        mask1x1_median: mask_med,
        mask1x1_p10: mask_p10,
        mask1x1_p25: mask_p25,
        mask1x1_p75: mask_p75,
        chroma_var_x_pre_gab: var_x,
        chroma_var_b_pre_gab: var_b,
        chroma_proxy,
        flat_x_block_ratio: flat,
        xb_var_ratio,
        masking_med,
        masking_p10,
        masking_p90,
        high_qff_ratio: high_qff,
        edge_density: ed,
        high_freq_energy_ratio: hfer,
        palette_log2_size: plog2,
        var_y,
    }
}

fn main() {
    let base = Path::new("/home/lilith/work/codec-corpus/CID22/CID22-512/validation");
    // W44-96 split target:
    //   WANT (auto-fire variant Z lift): 1420710, 1531677
    //   REJECT (regress under variant Z): 2389166, 3637739
    //
    // Also include 1044329, 1418519, 1189261 for triangulation:
    //   1044329 mask=48.03 — also "fires W44-29" per W44-91 list (control)
    //   1418519 mask=92    — above all gates, control
    //   1189261 mask=69    — W44-91-only fires (control)
    let images = [
        // The 5 CID22 photos where W44-29 fires (mask < 50):
        ("WANT_Z", "1420710.png"),   // mask=39.55 — WANT (variant Z helps)
        ("WANT_Z", "1531677.png"),   // mask=35.63 — WANT
        ("REJECT_Z", "2389166.png"), // mask=46.24 — REJECT (SSIM2 -0.82)
        ("REJECT_Z", "1044329.png"), // mask=48.03 — REJECT (W44-95 wider tested)
        ("REJECT_Z", "7062219.png"), // mask=47.80 — never measured under variant Z; verify proxies
        // Controls (don't currently fire W44-29 but useful for verifying)
        ("CONTROL", "3637739.png"), // mask=75.83 — outside both gates
        ("CONTROL", "1189261.png"), // mask=69 — W44-91 fires
        ("CONTROL", "1418519.png"), // mask=92
    ];
    println!(
        "# W44-96 proxy probe — split WANT_Z {{1420710,1531677}} from REJECT_Z {{2389166,3637739}}"
    );
    println!(
        "class\timage\tw\th\t\
m3_colour\tfcbr\tmask_med\tmask_p10\tmask_p25\tmask_p75\t\
var_x\tvar_b\tchroma_proxy\tflat_x_ratio\txb_var_ratio\t\
mask_med2\tmask_p10_2\tmask_p90_2\thigh_qff\t\
edge_density\thigh_freq_energy\tpalette_log2\tvar_y"
    );
    for (class, name) in &images {
        let p = base.join(name);
        if !p.exists() {
            eprintln!("MISSING: {}", p.display());
            continue;
        }
        let pr = probe(&p);
        println!(
            "{}\t{}\t{}\t{}\t\
{:.3}\t{:.5}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t\
{:.6}\t{:.6}\t{:.4}\t{:.4}\t{:.4}\t\
{:.4}\t{:.4}\t{:.4}\t{:.3}\t\
{:.4}\t{:.4}\t{:.3}\t{:.6}",
            class,
            pr.name,
            pr.width,
            pr.height,
            pr.m3_colourfulness,
            pr.fcbr_proxy,
            pr.mask1x1_median,
            pr.mask1x1_p10,
            pr.mask1x1_p25,
            pr.mask1x1_p75,
            pr.chroma_var_x_pre_gab,
            pr.chroma_var_b_pre_gab,
            pr.chroma_proxy,
            pr.flat_x_block_ratio,
            pr.xb_var_ratio,
            pr.masking_med,
            pr.masking_p10,
            pr.masking_p90,
            pr.high_qff_ratio,
            pr.edge_density,
            pr.high_freq_energy_ratio,
            pr.palette_log2_size,
            pr.var_y,
        );
    }
}
