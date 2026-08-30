//! W44-34 verification: does forcing `try_dct64=true` (override the
//! `adapt_to_image_lossy` gate) close the 5-7% byte gap on 1418519?
//!
//! Tests the 5 OPEN cells from the W44-30 ledger refresh:
//!   1418519.png × {e6 d=1.0, e6 d=1.2, e6 d=1.6, e7 d=1.2, e7 d=1.6}
//!
//! Variant A: default (try_dct64 gated off by adapt_to_image_lossy on this
//!            512×512 < 500_000 pixel image at d < 2.0).
//! Variant B: __expert override LossyInternalParams::try_dct64 = Some(true).
//!            Bypasses the dispatch (api.rs:3795-3802) and forces full
//!            DCT64x64 / DCT64x32 / DCT32x64 evaluation.
//!
//! Build:
//!   cargo build -p jxl-encoder --release \
//!       --features 'parallel butteraugli-loop ssim2-loop __expert' \
//!       --example w44_34_1418519_dct64_verify
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!       --features 'parallel butteraugli-loop ssim2-loop __expert' \
//!       --example w44_34_1418519_dct64_verify

use std::path::PathBuf;
use std::time::Instant;

use jxl_encoder::LossyInternalParams;
use jxl_encoder::api::{LossyConfig, PixelLayout};

const CELLS: &[(u8, f32, &str)] = &[
    (6, 1.0, "e6_d1.0"),
    (6, 1.2, "e6_d1.2"),
    (6, 1.6, "e6_d1.6"),
    (7, 1.2, "e7_d1.2"),
    (7, 1.6, "e7_d1.6"),
];

fn load_png(path: &PathBuf) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap_or_else(|e| panic!("open {:?}: {}", path, e));
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.as_raw().clone(), w, h)
}

fn encode_a(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> (Vec<u8>, f64) {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode A");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes, ms)
}

fn encode_b(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> (Vec<u8>, f64) {
    let mut params = LossyInternalParams::default();
    params.try_dct64 = Some(true);
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_internal_params(params);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode B");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (bytes, ms)
}

fn main() {
    let img_path =
        PathBuf::from("/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png");
    let (rgb, w, h) = load_png(&img_path);
    println!("Loaded {:?} → {}×{} ({} bytes)", img_path, w, h, rgb.len());

    println!();
    println!("Cell        bytes_A   bytes_B   Δ_bytes  Δ_pct    ms_A     ms_B");
    println!("─────────  ────────  ────────  ───────  ──────  ───────  ───────");

    let mut total_a: i64 = 0;
    let mut total_b: i64 = 0;
    for &(effort, distance, label) in CELLS {
        // 3 warm-up + 3 measured per variant, interleaved, take min wall-time
        let mut a_bytes_set = Vec::new();
        let mut b_bytes_set = Vec::new();
        let mut a_ms = f64::INFINITY;
        let mut b_ms = f64::INFINITY;
        for _ in 0..3 {
            let (ba, ma) = encode_a(&rgb, w, h, distance, effort);
            let (bb, mb) = encode_b(&rgb, w, h, distance, effort);
            a_bytes_set.push(ba.len());
            b_bytes_set.push(bb.len());
            a_ms = a_ms.min(ma);
            b_ms = b_ms.min(mb);
        }
        let a = a_bytes_set[0] as i64;
        let b = b_bytes_set[0] as i64;
        let delta = b - a;
        let pct = (delta as f64) / (a as f64) * 100.0;
        println!(
            "{:9}  {:>8}  {:>8}  {:+7}  {:+6.2}%  {:>5.1}ms  {:>5.1}ms",
            label, a, b, delta, pct, a_ms, b_ms
        );
        total_a += a;
        total_b += b;
    }
    let tot_delta = total_b - total_a;
    let tot_pct = (tot_delta as f64) / (total_a as f64) * 100.0;
    println!("─────────  ────────  ────────  ───────  ──────");
    println!(
        "TOTAL      {:>8}  {:>8}  {:+7}  {:+6.2}%",
        total_a, total_b, tot_delta, tot_pct
    );
    println!();
    println!("Variant A: default (dispatch ON, try_dct64 gated off via adapt_to_image_lossy).");
    println!("Variant B: __expert LossyInternalParams::try_dct64 = Some(true) (force DCT64 eval).");
}
