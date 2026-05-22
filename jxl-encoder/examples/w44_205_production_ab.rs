//! W44-205 Phase 3 validation A/B bench: production W44-205 (default-on,
//! buckets 2+4 disabled on top of W44-201 buckets 3+6) vs forced-legacy
//! W44-205 (`JXL_W44_205_FORCE_LEGACY_MEDIUM_BUCKETS=1` = pre-W44-205 =
//! W44-201 baseline). Confirms production behaviour matches the Phase-1
//! probe under `JXL_W44_201_DISABLE_BUCKETS=2,4`.
//!
//! Also includes a SCRN PROTECT set and Cluster #1 LOSER cells matching
//! W44-205 task spec.
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert __internal_recon_hook butteraugli-loop ssim2-loop parallel' \
//!     --example w44_205_production_ab > benchmarks/w44_205_production_ab_2026-05-22.tsv

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
    // Cluster #1 LOSER_DOMINANT photos (W44-204 #1 target).
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
    // PROTECT_W44_82: F-D OPEN cells (cost-benefit gate's load-bearing test).
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
    // SCRN PROTECT cells.
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

#[derive(Clone, Copy)]
enum Variant {
    /// A: forced-legacy W44-205 (= pre-W44-205 = W44-201 baseline).
    ALegacy,
    /// B: production W44-205 default (buckets 2+4 also disabled).
    BProd,
}

fn encode(pixels: &[u8], w: u32, h: u32, distance: f32, effort: u8, v: Variant) -> usize {
    unsafe {
        std::env::remove_var("JXL_W44_201_DISABLE_BUCKETS");
        std::env::remove_var("JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS");
        std::env::remove_var("JXL_W44_205_FORCE_LEGACY_MEDIUM_BUCKETS");
        match v {
            Variant::ALegacy => {
                // Force-legacy the W44-205 extension; W44-201 production
                // remains active.
                std::env::set_var("JXL_W44_205_FORCE_LEGACY_MEDIUM_BUCKETS", "1");
            }
            Variant::BProd => {
                // No env vars: production W44-205 default (both 3+6 and
                // 2+4 buckets disabled).
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
    println!("label\tw\th\tdistance\teffort\tA_bytes\tB_bytes\tdelta\tpct");
    let mut sum_a = 0i64;
    let mut sum_b = 0i64;
    let mut worst: (f64, String) = (0.0, "".to_string());
    let mut best: (f64, String) = (0.0, "".to_string());
    let mut wins = 0;
    let mut losses = 0;
    let mut ties = 0;
    for cell in cells() {
        let Some((pixels, w, h)) = load_png(Path::new(cell.path)) else {
            eprintln!("skip {}: not found", cell.label);
            continue;
        };
        let a = encode(&pixels, w, h, cell.distance, cell.effort, Variant::ALegacy);
        let b = encode(&pixels, w, h, cell.distance, cell.effort, Variant::BProd);
        let delta = b as i64 - a as i64;
        let pct = 100.0 * delta as f64 / a as f64;
        sum_a += a as i64;
        sum_b += b as i64;
        if pct > worst.0 {
            worst = (pct, cell.label.to_string());
        }
        if pct < best.0 {
            best = (pct, cell.label.to_string());
        }
        if delta < 0 {
            wins += 1;
        } else if delta > 0 {
            losses += 1;
        } else {
            ties += 1;
        }
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:+}\t{:+.2}%",
            cell.label, w, h, cell.distance, cell.effort, a, b, delta, pct
        );
    }
    let pct_total = 100.0 * (sum_b - sum_a) as f64 / sum_a as f64;
    println!(
        "TOTAL\t-\t-\t-\t-\t{}\t{}\t{:+}\t{:+.2}%",
        sum_a,
        sum_b,
        sum_b - sum_a,
        pct_total
    );
    eprintln!(
        "\nW44-205 production (B) vs W44-201 baseline (A):\n  TOTAL bytes: {:+.2}% ({:+} B)\n  wins: {}  losses: {}  ties: {}\n  worst regression: {:+.2}% on {}\n  best improvement: {:+.2}% on {}",
        pct_total,
        sum_b - sum_a,
        wins,
        losses,
        ties,
        worst.0,
        worst.1,
        best.0,
        best.1,
    );
}
