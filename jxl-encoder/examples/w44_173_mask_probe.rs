//! W44-173 mask1x1 probe: compute mask1x1 percentiles on clic_097cb426
//! so we can compare against W44-91/96/166/168 thresholds. Mirrors the
//! mask1x1 extraction block in `w44_96_proxy_probe.rs:340-356`.
//!
//! Output: stdout, also appended to
//!   benchmarks/w44_173_clic_097cb426_proxies_2026-05-21.txt
//!
//! Run:
//!   cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' \
//!     --example w44_173_mask_probe

use jxl_encoder::__pre_quantized::{EffortProfile, EncoderPrecomputed};
use jxl_encoder::api::EncoderMode;
use jxl_encoder::color::xyb::srgb_to_linear_value;
use std::path::PathBuf;

const SRC_PNG: &str =
    "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png";

fn percentile(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() - 1) as f32 * q).round() as usize;
    v[idx]
}

fn main() {
    let img = image::open(SRC_PNG).expect("load source");
    let rgb = img.to_rgb8();
    let raw = rgb.as_raw().clone();
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    eprintln!("Loaded clic_097cb426: {}x{}", w, h);

    let mut linear_rgb = Vec::with_capacity(w * h * 3);
    for px in raw.chunks(3) {
        linear_rgb.push(srgb_to_linear_value(px[0].into()));
        linear_rgb.push(srgb_to_linear_value(px[1].into()));
        linear_rgb.push(srgb_to_linear_value(px[2].into()));
    }

    for &(effort, distance) in &[(7u8, 5.0f32), (8u8, 5.0f32)] {
        let profile = EffortProfile::lossy(effort, EncoderMode::Reference);
        let pre = EncoderPrecomputed::compute(
            w,
            h,
            &linear_rgb,
            distance,
            true,  // cfl_enabled
            true,  // ac_strategy_enabled
            true,  // pixel_domain_loss
            false, // enable_noise
            false, // enable_denoise
            true,  // enable_gaborish
            None,  // force_strategy
            &profile,
            None,
        )
        .expect("EncoderPrecomputed::compute");

        let stride = pre.padded_width;
        let mut buf: Vec<f32> = Vec::with_capacity(w * h);
        if let Some(mask) = pre.mask1x1.as_ref() {
            for y in 0..h {
                let row_off = y * stride;
                buf.extend_from_slice(&mask[row_off..row_off + w]);
            }
        }
        let med = percentile(&mut buf.clone(), 0.50);
        let p10 = percentile(&mut buf.clone(), 0.10);
        let p25 = percentile(&mut buf.clone(), 0.25);
        let p75 = percentile(&mut buf.clone(), 0.75);
        let p90 = percentile(&mut buf.clone(), 0.90);

        // Masking field (different from mask1x1 — used in W44-29 gate)
        let mut masking_buf = pre.masking.clone();
        let masking_med = percentile(&mut masking_buf, 0.50);

        eprintln!("=== effort={} distance={} ===", effort, distance);
        eprintln!(
            "  mask1x1: median={:.2} p10={:.2} p25={:.2} p75={:.2} p90={:.2}",
            med, p10, p25, p75, p90
        );
        eprintln!("  masking_field_median={:.4}", masking_med);
    }

    // Append to proxies file
    let proxies_path = PathBuf::from(
        "/home/lilith/work/zen/jxl-encoder/benchmarks/w44_173_clic_097cb426_proxies_2026-05-21.txt",
    );
    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&proxies_path) {
        use std::io::Write;
        writeln!(f).unwrap();
        writeln!(f, "## mask1x1 percentiles (from W44-173 mask_probe)").unwrap();
        for &(effort, distance) in &[(7u8, 5.0f32), (8u8, 5.0f32)] {
            let profile = EffortProfile::lossy(effort, EncoderMode::Reference);
            let pre = EncoderPrecomputed::compute(
                w,
                h,
                &linear_rgb,
                distance,
                true,
                true,
                true,
                false,
                false,
                true,
                None,
                &profile,
                None,
            )
            .expect("EncoderPrecomputed::compute");
            let stride = pre.padded_width;
            let mut buf: Vec<f32> = Vec::with_capacity(w * h);
            if let Some(mask) = pre.mask1x1.as_ref() {
                for y in 0..h {
                    let row_off = y * stride;
                    buf.extend_from_slice(&mask[row_off..row_off + w]);
                }
            }
            let med = percentile(&mut buf.clone(), 0.50);
            let p25 = percentile(&mut buf.clone(), 0.25);
            let p75 = percentile(&mut buf.clone(), 0.75);
            let p90 = percentile(&mut buf.clone(), 0.90);
            writeln!(
                f,
                "e{}_d{:.0}\tmask1x1_med={:.2}\tmask1x1_p25={:.2}\tmask1x1_p75={:.2}\tmask1x1_p90={:.2}",
                effort, distance, med, p25, p75, p90
            )
            .unwrap();
        }
    }
}
