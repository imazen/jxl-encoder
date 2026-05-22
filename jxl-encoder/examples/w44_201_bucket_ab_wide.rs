//! W44-201 Phase 2 wide A/B test: A/B/C variants across photos +
//! screenshots + the W44-82 OPEN cells (1420710 e7 d=4) to make sure
//! the bucket disable does NOT regress the gate's previously-documented
//! load-bearing cases.
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert __internal_recon_hook butteraugli-loop ssim2-loop parallel' \
//!     --example w44_201_bucket_ab_wide

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
    // Original W44-201 LOSER / WINNER batch at e7 d=4
    for img in &["3637739", "1418519", "1025469", "1189261", "1420710", "1531677",
                 "2389166", "297394", "1475938"] {
        for &d in &[2.0_f32, 3.0, 4.0, 5.0] {
            v.push(Cell {
                label: Box::leak(format!("cid22_{}_d{:.1}", img, d).into_boxed_str()),
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
    // Screenshots from gb82-sc (test that the gate's W44-82 load-bearing
    // hypothesis doesn't regress on screen content)
    for img in &["codec_wiki", "imac_g3", "imac_dark", "terminal", "windows95"] {
        for &d in &[1.0_f32, 2.0, 4.0] {
            v.push(Cell {
                label: Box::leak(format!("gb82_{}_d{:.1}", img, d).into_boxed_str()),
                path: Box::leak(
                    format!("/home/lilith/work/codec-corpus/gb82-sc/{}.png", img)
                        .into_boxed_str(),
                ),
                distance: d,
                effort: 7,
            });
        }
    }
    // W44-82 spot cells (cited as the prior regression evidence for bucket 3+6 gating)
    for d in &[2.0_f32, 4.0] {
        v.push(Cell {
            label: Box::leak(format!("w44_82_spot_1420710_d{:.1}_e7", d).into_boxed_str()),
            path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
            distance: *d,
            effort: 7,
        });
    }
    v
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Some((rgb.into_raw(), w, h))
}

/// Encode under one of three configurations:
///   - `Variant::ALegacy`: pre-W44-201 baseline (cost-benefit admits all
///     buckets, no production fix) — achieved via the
///     `JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS=1` env hook.
///   - `Variant::BBucket3Only`: legacy mode + extra `DISABLE_BUCKETS=3`
///     (matches the pre-W44-201 "Variant B" measurement; not production
///     but kept for the W44-201 narrative table).
///   - `Variant::CProd`: production W44-201 fix (Zenjxl default = disable
///     buckets 3+6 by default).
#[derive(Clone, Copy)]
enum Variant {
    ALegacy,
    BBucket3Only,
    CProd,
}

fn encode(pixels: &[u8], w: u32, h: u32, distance: f32, effort: u8, v: Variant) -> usize {
    unsafe {
        // Clear all W44-201 env vars before each encode so configurations
        // don't leak between calls.
        std::env::remove_var("JXL_W44_201_DISABLE_BUCKETS");
        std::env::remove_var("JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS");
        match v {
            Variant::ALegacy => {
                std::env::set_var("JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS", "1");
            }
            Variant::BBucket3Only => {
                // Force legacy first to clear the production-gate `disable
                // buckets 3+6`, then explicitly disable bucket 3 only.
                std::env::set_var("JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS", "1");
                std::env::set_var("JXL_W44_201_DISABLE_BUCKETS", "3");
            }
            Variant::CProd => {
                // No env vars: production gate disables buckets 3+6 by default.
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
        "label\tw\th\tdistance\teffort\tA_bytes\tB_bytes\tC_bytes\tdelta_B\tdelta_C\tpct_B\tpct_C"
    );
    let mut sum_a = 0i64;
    let mut sum_b = 0i64;
    let mut sum_c = 0i64;
    let mut worst_b: (f64, String) = (0.0, "".to_string());
    let mut worst_c: (f64, String) = (0.0, "".to_string());
    for c in cells() {
        let Some((pixels, w, h)) = load_png(Path::new(c.path)) else {
            eprintln!("skip {}: not found", c.label);
            continue;
        };
        let a = encode(&pixels, w, h, c.distance, c.effort, Variant::ALegacy);
        let b = encode(&pixels, w, h, c.distance, c.effort, Variant::BBucket3Only);
        let cb = encode(&pixels, w, h, c.distance, c.effort, Variant::CProd);
        let da = b as i64 - a as i64;
        let dc = cb as i64 - a as i64;
        let pct_b = 100.0 * da as f64 / a as f64;
        let pct_c = 100.0 * dc as f64 / a as f64;
        sum_a += a as i64;
        sum_b += b as i64;
        sum_c += cb as i64;
        if pct_b > worst_b.0 {
            worst_b = (pct_b, c.label.to_string());
        }
        if pct_c > worst_c.0 {
            worst_c = (pct_c, c.label.to_string());
        }
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:+}\t{:+}\t{:+.2}%\t{:+.2}%",
            c.label, w, h, c.distance, c.effort, a, b, cb, da, dc, pct_b, pct_c
        );
    }
    println!(
        "TOTAL\t-\t-\t-\t-\t{}\t{}\t{}\t{:+}\t{:+}\t{:+.2}%\t{:+.2}%",
        sum_a,
        sum_b,
        sum_c,
        sum_b - sum_a,
        sum_c - sum_a,
        100.0 * (sum_b - sum_a) as f64 / sum_a as f64,
        100.0 * (sum_c - sum_a) as f64 / sum_a as f64
    );
    eprintln!("worst B regression: {:+.2}% on {}", worst_b.0, worst_b.1);
    eprintln!("worst C regression: {:+.2}% on {}", worst_c.0, worst_c.1);
}
