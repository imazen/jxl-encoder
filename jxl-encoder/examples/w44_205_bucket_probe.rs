//! W44-205 Phase 1: probe whether disabling coeff_orders buckets 2
//! (DCT16x16) and 4 (DCT16x8/DCT8x16) on top of W44-201's buckets 3+6
//! yields additional byte savings on Cluster #1 / #2 Pareto-loser
//! cells. Uses the existing `JXL_W44_201_DISABLE_BUCKETS` env hook to
//! test the extension WITHOUT shipping any production code, then
//! Phase 2 will wire it as a Section D gate.
//!
//! Variants:
//!   A = W44-201 production (buckets 3+6 disabled = current main)
//!   B = W44-201 + buckets 2+4 also disabled (W44-205 candidate)
//!   C = W44-201 + bucket 2 only also disabled (DCT16x16, isolation)
//!   D = W44-201 + bucket 4 only also disabled (DCT16x8/8x16, isolation)
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert __internal_recon_hook butteraugli-loop ssim2-loop parallel' \
//!     --example w44_205_bucket_probe

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::Path;

#[derive(Clone, Copy)]
struct Cell {
    label: &'static str,
    path: &'static str,
    distance: f32,
    effort: u8,
}

fn cells() -> Vec<Cell> {
    let mut v = Vec::new();
    // Phase-1 probe: focus on Cluster #1 LOSER_DOMINANT photos at e7 d=3,4,5
    // (densest Pareto-loser band per W44-204).
    for img in &["3637739", "297394", "7062219", "1475938"] {
        for &d in &[2.0_f32, 3.0, 4.0, 5.0] {
            v.push(Cell {
                label: Box::leak(format!("LOSER_{}_d{:.1}_e7", img, d).into_boxed_str()),
                path: Box::leak(
                    format!(
                        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/{}.png",
                        img
                    )
                    .into_boxed_str(),
                ),
                distance: d,
                effort: 7,
            });
        }
    }
    // PROTECT cells: W44-82 F-D OPEN cells (the gate's load-bearing test).
    // If buckets 2+4 add regressions here, we have an issue.
    for &(img, d) in &[
        ("1420710", 4.0_f32),
        ("1531677", 4.0),
        ("1189261", 4.0),
        ("1418519", 4.0),
        ("1418519", 5.0),
    ] {
        v.push(Cell {
            label: Box::leak(format!("PROTECT_{}_d{:.1}_e7", img, d).into_boxed_str()),
            path: Box::leak(
                format!(
                    "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/{}.png",
                    img
                )
                .into_boxed_str(),
            ),
            distance: d,
            effort: 7,
        });
    }
    // Screenshot PROTECT cells (W44-82 documented these don't admit
    // buckets 3+6 anyway; verify same for 2+4 too).
    for img in &["codec_wiki", "terminal", "imac_dark"] {
        for &d in &[1.0_f32, 4.0] {
            v.push(Cell {
                label: Box::leak(format!("SCRN_{}_d{:.1}_e7", img, d).into_boxed_str()),
                path: Box::leak(
                    format!("/home/lilith/work/codec-corpus/gb82-sc/{}.png", img).into_boxed_str(),
                ),
                distance: d,
                effort: 7,
            });
        }
    }
    v
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Some((rgb.into_raw(), w, h))
}

/// Encode under one of four configurations:
///   - A: W44-201 production (buckets 3+6 disabled = current main)
///   - B: + buckets 2+4 also disabled (W44-205 candidate)
///   - C: + bucket 2 only also disabled (DCT16x16)
///   - D: + bucket 4 only also disabled (DCT16x8/8x16)
#[derive(Clone, Copy)]
enum Variant {
    A,
    B,
    C,
    D,
}

fn encode(pixels: &[u8], w: u32, h: u32, distance: f32, effort: u8, v: Variant) -> usize {
    unsafe {
        std::env::remove_var("JXL_W44_201_DISABLE_BUCKETS");
        std::env::remove_var("JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS");
        // Production W44-201 (buckets 3+6 disabled) is the baseline; the
        // env hook adds more buckets on top.
        match v {
            Variant::A => {
                // No extra buckets disabled.
            }
            Variant::B => {
                std::env::set_var("JXL_W44_201_DISABLE_BUCKETS", "2,4");
            }
            Variant::C => {
                std::env::set_var("JXL_W44_201_DISABLE_BUCKETS", "2");
            }
            Variant::D => {
                std::env::set_var("JXL_W44_201_DISABLE_BUCKETS", "4");
            }
        }
    }
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl);
    let buf = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .expect("encode");
    buf.len()
}

fn main() {
    println!(
        "label\tw\th\tdistance\teffort\tA_bytes\tB_bytes\tC_bytes\tD_bytes\tdB_pct\tdC_pct\tdD_pct"
    );
    let mut sum_a = 0i64;
    let mut sum_b = 0i64;
    let mut sum_c = 0i64;
    let mut sum_d = 0i64;
    let mut worst_b: (f64, String) = (0.0, "".to_string());
    let mut worst_c: (f64, String) = (0.0, "".to_string());
    let mut worst_d: (f64, String) = (0.0, "".to_string());
    for cell in cells() {
        let Some((pixels, w, h)) = load_png(Path::new(cell.path)) else {
            eprintln!("skip {}: not found", cell.label);
            continue;
        };
        let a = encode(&pixels, w, h, cell.distance, cell.effort, Variant::A);
        let b = encode(&pixels, w, h, cell.distance, cell.effort, Variant::B);
        let c = encode(&pixels, w, h, cell.distance, cell.effort, Variant::C);
        let d = encode(&pixels, w, h, cell.distance, cell.effort, Variant::D);
        let pct_b = 100.0 * (b as i64 - a as i64) as f64 / a as f64;
        let pct_c = 100.0 * (c as i64 - a as i64) as f64 / a as f64;
        let pct_d = 100.0 * (d as i64 - a as i64) as f64 / a as f64;
        sum_a += a as i64;
        sum_b += b as i64;
        sum_c += c as i64;
        sum_d += d as i64;
        if pct_b > worst_b.0 {
            worst_b = (pct_b, cell.label.to_string());
        }
        if pct_c > worst_c.0 {
            worst_c = (pct_c, cell.label.to_string());
        }
        if pct_d > worst_d.0 {
            worst_d = (pct_d, cell.label.to_string());
        }
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:+.2}\t{:+.2}\t{:+.2}",
            cell.label, w, h, cell.distance, cell.effort, a, b, c, d, pct_b, pct_c, pct_d
        );
    }
    let pct_b_total = 100.0 * (sum_b - sum_a) as f64 / sum_a as f64;
    let pct_c_total = 100.0 * (sum_c - sum_a) as f64 / sum_a as f64;
    let pct_d_total = 100.0 * (sum_d - sum_a) as f64 / sum_a as f64;
    println!(
        "TOTAL\t-\t-\t-\t-\t{}\t{}\t{}\t{}\t{:+.2}\t{:+.2}\t{:+.2}",
        sum_a, sum_b, sum_c, sum_d, pct_b_total, pct_c_total, pct_d_total
    );
    eprintln!(
        "Variant B (buckets 2+4 + 3+6): TOTAL {:+.2}%, worst regression {:+.2}% on {}",
        pct_b_total, worst_b.0, worst_b.1
    );
    eprintln!(
        "Variant C (bucket 2 + 3+6):    TOTAL {:+.2}%, worst regression {:+.2}% on {}",
        pct_c_total, worst_c.0, worst_c.1
    );
    eprintln!(
        "Variant D (bucket 4 + 3+6):    TOTAL {:+.2}%, worst regression {:+.2}% on {}",
        pct_d_total, worst_d.0, worst_d.1
    );
}
