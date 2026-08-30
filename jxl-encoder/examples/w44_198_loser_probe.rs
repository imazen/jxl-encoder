//! W44-198 photo strict-loser discriminator probe (measurement-only,
//! no production code change).
//!
//! Follow-on to W44-185 arc-closing bench analysis. The W44-185 zenjxl
//! TSV reveals 3637739 (CID22 photo) as the dominant Zenjxl Pareto-loser:
//! 83 strict-loss cells (bytes>+1% AND ssim2<-0.1) across the e5/e6/e7/
//! e8/e9 × d=0.25..5 grid. Other CID22 photos follow the same pattern
//! at lower frequencies (7062219: 51, 1420710: 43, 297394: 37, ...).
//!
//! This probe computes the encoder-internal `ZenanalyzeProxies` fields
//! (m3_colourfulness, flat_color_block_ratio, edge_density) plus
//! mask1x1 percentiles + XYB variances + luma stats for all 41 CID22
//! validation photos. Output TSV is the input to the discriminator
//! selection analysis: find a feature axis that splits the LOSER
//! class (3637739, 7062219, etc) from the WINNER class (1418519,
//! 1531677, 1025469 at d=3-5 e7+ where they BEAT cjxl on bytes).
//!
//! Class labels (per W44-185 TSV, counting strict-loss cells where
//! bytes>+1% AND ssim2<-0.1 across 100 cells/image):
//!   - LOSER_DOMINANT (n_loss > 50): 3637739 (83), 7062219 (51)
//!   - LOSER_HEAVY (30 <= n_loss <= 50): 1420710 (43), 297394 (37),
//!     1475938 (35), 1531677 (30)
//!   - LOSER_MILD (20 <= n_loss < 30): 1189261 (25), 1025469 (22)
//!   - WINNER (n_loss < 15): 1418519 (11), 2389166 (7)
//!
//! Goal: find proxy axis where LOSER_DOMINANT clusters distinctly from
//! WINNER (and ideally separates LOSER_HEAVY from WINNER too). If no
//! clean axis emerges, the W44-198 chunk HONEST-STOPs with the
//! diagnosis: this cluster needs structural / multi-week refactor, not
//! discriminator gating.
//!
//! NO production code change. Memory note + TSV + meta committed per
//! W44-198 Phase 1 acceptance gates.

use jxl_encoder::__pre_quantized::{EffortProfile, EncoderPrecomputed};
use jxl_encoder::__test_exports::xyb::srgb_to_linear_value;
use jxl_encoder::api::EncoderMode;
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Default)]
struct ImageProxies {
    name: String,
    width: u32,
    height: u32,
    // W44-91 / W44-96 ZenanalyzeProxies fields (cheap sRGB-direct)
    m3_colourfulness: f32,
    fcbr_proxy: f32,
    edge_density: f32,
    // mask1x1 percentiles (existing pipeline output)
    mask1x1_median: f32,
    mask1x1_p10: f32,
    mask1x1_p25: f32,
    mask1x1_p75: f32,
    mask1x1_p90: f32,
    // XYB-derived stats
    var_x: f32,
    var_y: f32,
    var_b: f32,
    chroma_proxy: f32,
    xb_var_ratio: f32,
    flat_x_block_ratio: f32,
    masking_med: f32,
    masking_p10: f32,
    masking_p90: f32,
    high_qff_ratio: f32,
    high_freq_energy_ratio: f32,
    palette_log2_size: f32,
    // Luma stats
    chroma_block_ratio: f32,
    low_luma_ratio: f32,
    high_luma_ratio: f32,
    mean_luma: f32,
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

/// Sobel-like edge density on luma. Matches the in-encoder
/// `ZenanalyzeProxies::edge_density` definition.
fn edge_density(raw: &[u8], w: usize, h: usize) -> f32 {
    if w < 3 || h < 3 {
        return 0.0;
    }
    let luma = |y: usize, x: usize| -> f32 {
        let off = (y * w + x) * 3;
        0.299 * raw[off] as f32 + 0.587 * raw[off + 1] as f32 + 0.114 * raw[off + 2] as f32
    };
    let mut edges = 0usize;
    let mut total = 0usize;
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let gx = -luma(y - 1, x - 1) - 2.0 * luma(y, x - 1) - luma(y + 1, x - 1)
                + luma(y - 1, x + 1)
                + 2.0 * luma(y, x + 1)
                + luma(y + 1, x + 1);
            let gy = -luma(y - 1, x - 1) - 2.0 * luma(y - 1, x) - luma(y - 1, x + 1)
                + luma(y + 1, x - 1)
                + 2.0 * luma(y + 1, x)
                + luma(y + 1, x + 1);
            let g_sq = gx * gx + gy * gy;
            if g_sq > 900.0 {
                edges += 1;
            }
            total += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        edges as f32 / total as f32
    }
}

/// HF energy ratio on Y plane (same as W44-96).
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

/// log2 of unique sRGB quantized colors (6 bits per channel).
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

    // === M3 colourfulness (Hasler-Süsstrunk) on sRGB u8 ===
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

    // === FCBR (zenanalyze tier1, per-channel range <= 4 on every 8×8) ===
    let blocks_x = w / 8;
    let blocks_y = h / 8;
    let total_blocks = blocks_x * blocks_y;
    let mut flat_blocks = 0usize;
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

    // === Luma statistics ===
    let mut luma_sum = 0.0_f64;
    let mut low_pix = 0usize;
    let mut high_pix = 0usize;
    for i in 0..(w * h) {
        let r = raw[i * 3] as f64;
        let g = raw[i * 3 + 1] as f64;
        let b = raw[i * 3 + 2] as f64;
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        luma_sum += y;
        if y < 32.0 {
            low_pix += 1;
        }
        if y > 224.0 {
            high_pix += 1;
        }
    }
    let mean_luma = (luma_sum / n_pix) as f32;
    let low_luma_ratio = low_pix as f32 / n_pix as f32;
    let high_luma_ratio = high_pix as f32 / n_pix as f32;

    // === EncoderPrecomputed for XYB-derived stats ===
    let profile = EffortProfile::lossy(7, EncoderMode::Reference);
    let pre = EncoderPrecomputed::compute(
        w,
        h,
        &linear_rgb,
        /*distance*/ 4.0,
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

    let (mask_med, mask_p10, mask_p25, mask_p75, mask_p90) =
        if let Some(mask) = pre.mask1x1.as_ref() {
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
            let mut m1 = buf.clone();
            let p75 = percentile(&mut m1, 0.75);
            let p90 = percentile(&mut buf, 0.90);
            (med, p10, p25, p75, p90)
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0)
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

    let bx_n = w / 8;
    let by_n = h / 8;
    let mut active = 0usize;
    let total_b = bx_n * by_n;
    if total_b > 0 {
        for by in 0..by_n {
            for bxi in 0..bx_n {
                let mut sx_sum = 0.0_f64;
                let mut sx_sq = 0.0_f64;
                let mut sb_sum = 0.0_f64;
                let mut sb_sq = 0.0_f64;
                for dy in 0..8 {
                    for dx in 0..8 {
                        let off = (by * 8 + dy) * stride + (bxi * 8 + dx);
                        let vx = xyb_x_view[off] as f64;
                        let vb = xyb_b_view[off] as f64;
                        sx_sum += vx;
                        sx_sq += vx * vx;
                        sb_sum += vb;
                        sb_sq += vb * vb;
                    }
                }
                let mx = sx_sum / 64.0;
                let mb = sb_sum / 64.0;
                let vx_var = (sx_sq / 64.0 - mx * mx).max(0.0);
                let vb_var = (sb_sq / 64.0 - mb * mb).max(0.0);
                if (vx_var + vb_var).sqrt() > 0.001 {
                    active += 1;
                }
            }
        }
    }
    let chroma_block_ratio = if total_b > 0 {
        active as f32 / total_b as f32
    } else {
        0.0
    };

    let mut masking_buf = pre.masking.clone();
    let masking_med = percentile(&mut masking_buf, 0.50);
    let mut masking_buf = pre.masking.clone();
    let masking_p10 = percentile(&mut masking_buf, 0.10);
    let mut masking_buf = pre.masking.clone();
    let masking_p90 = percentile(&mut masking_buf, 0.90);

    let qff = &pre.quant_field_float;
    let high_qff = if !qff.is_empty() {
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
        edge_density: ed,
        mask1x1_median: mask_med,
        mask1x1_p10: mask_p10,
        mask1x1_p25: mask_p25,
        mask1x1_p75: mask_p75,
        mask1x1_p90: mask_p90,
        var_x,
        var_y,
        var_b,
        chroma_proxy,
        xb_var_ratio,
        flat_x_block_ratio: flat,
        masking_med,
        masking_p10,
        masking_p90,
        high_qff_ratio: high_qff,
        high_freq_energy_ratio: hfer,
        palette_log2_size: plog2,
        chroma_block_ratio,
        low_luma_ratio,
        high_luma_ratio,
        mean_luma,
    }
}

/// Class labels for W44-198 strict-loser discriminator audit.
///
/// Counts come from W44-185 zenjxl TSV strict-loss filter
/// (bytes>+1% AND ssim2<-0.1) across 100 cells per image
/// (5 efforts × 20 distances).
///
/// LOSER_DOMINANT: n_loss > 50 (must be admitted to fix)
/// LOSER_HEAVY: 30 <= n_loss <= 50 (would benefit if discriminator widens)
/// LOSER_MILD: 20 <= n_loss < 30
/// WINNER: n_loss < 15 (must NOT be admitted; they beat cjxl on bytes
///         already, applying a lift would hurt them)
fn class_for(name: &str) -> &'static str {
    match name {
        // LOSER_DOMINANT (counts: 83, 51)
        "3637739.png" => "LOSER_DOMINANT",
        "7062219.png" => "LOSER_DOMINANT",
        // LOSER_HEAVY (counts: 43, 37, 35, 30)
        "1420710.png" => "LOSER_HEAVY",
        "297394.png" => "LOSER_HEAVY",
        "1475938.png" => "LOSER_HEAVY",
        "1531677.png" => "LOSER_HEAVY",
        // LOSER_MILD (counts: 25, 22)
        "1189261.png" => "LOSER_MILD",
        "1025469.png" => "LOSER_MILD",
        // WINNER (counts: 11, 7 — bytes-negative on most cells)
        "1418519.png" => "WINNER",
        "2389166.png" => "WINNER",
        // All other CID22 photos default to CONTROL (not in bench)
        _ => "CONTROL",
    }
}

fn main() {
    // 41 CID22 validation photos. Same set as W44-149.
    let cid22 = [
        "2936831.png",
        "7062219.png",
        "792079.png",
        "2389166.png",
        "7552578.png",
        "1531677.png",
        "1544947.png",
        "5458393.png",
        "2253934.png",
        "1044329.png",
        "1189261.png",
        "382297.png",
        "1418519.png",
        "1420710.png",
        "373965.png",
        "6078297.png",
        "3653963.png",
        "6292444.png",
        "297394.png",
        "2079234.png",
        "225228.png",
        "4215100.png",
        "159550.png",
        "2775196.png",
        "1475938.png",
        "2190188.png",
        "5055743.png",
        "1025469.png",
        "1624487.png",
        "162520.png",
        "2670327.png",
        "164595.png",
        "2736139.png",
        "3316926.png",
        "3156482.png",
        "844297.png",
        "1279330.png",
        "3637739.png",
        "3762075.png",
        "70497.png",
        "2887497.png",
    ];

    let cid22_base = Path::new("/home/lilith/work/codec-corpus/CID22/CID22-512/validation");

    println!(
        "class\timage\tw\th\t\
m3_colour\tfcbr\tedge_density\t\
mask_med\tmask_p10\tmask_p25\tmask_p75\tmask_p90\t\
var_x\tvar_y\tvar_b\tchroma_proxy\txb_var_ratio\tflat_x_ratio\t\
masking_med\tmasking_p10\tmasking_p90\thigh_qff\t\
high_freq_energy\tpalette_log2\t\
chroma_block_ratio\tlow_luma_ratio\thigh_luma_ratio\tmean_luma"
    );

    let mut probed = 0usize;
    let mut missing = 0usize;
    for name in &cid22 {
        let path = cid22_base.join(name);
        if !path.exists() {
            eprintln!("MISSING: {}", path.display());
            missing += 1;
            continue;
        }
        let pr = probe(&path);
        let class = class_for(name);
        probed += 1;
        println!(
            "{}\t{}\t{}\t{}\t\
{:.3}\t{:.5}\t{:.5}\t\
{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t\
{:.6}\t{:.6}\t{:.6}\t{:.5}\t{:.4}\t{:.4}\t\
{:.4}\t{:.4}\t{:.4}\t{:.3}\t\
{:.4}\t{:.3}\t\
{:.4}\t{:.4}\t{:.4}\t{:.2}",
            class,
            pr.name,
            pr.width,
            pr.height,
            pr.m3_colourfulness,
            pr.fcbr_proxy,
            pr.edge_density,
            pr.mask1x1_median,
            pr.mask1x1_p10,
            pr.mask1x1_p25,
            pr.mask1x1_p75,
            pr.mask1x1_p90,
            pr.var_x,
            pr.var_y,
            pr.var_b,
            pr.chroma_proxy,
            pr.xb_var_ratio,
            pr.flat_x_block_ratio,
            pr.masking_med,
            pr.masking_p10,
            pr.masking_p90,
            pr.high_qff_ratio,
            pr.high_freq_energy_ratio,
            pr.palette_log2_size,
            pr.chroma_block_ratio,
            pr.low_luma_ratio,
            pr.high_luma_ratio,
            pr.mean_luma,
        );
    }
    eprintln!("# probed {} CID22 photos ({} missing)", probed, missing);
}
