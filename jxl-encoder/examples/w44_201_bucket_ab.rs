//! W44-201 Phase 2: A/B test of disabling DCT32x32 bucket in coeff_orders
//! for 3637739 LOSER + 8 other CID22 photos.
//!
//! Variants:
//!   A: control (Zenjxl default — all buckets eligible)
//!   B: disable bucket 3 (DCT32x32) only
//!   C: disable buckets 3 and 6 (DCT32x32 + DCT32x16/16x32)
//!
//! If B saves bytes on 3637739 LOSER without regressing other photos,
//! we have a candidate fix.
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert __internal_recon_hook butteraugli-loop ssim2-loop parallel' \
//!     --example w44_201_bucket_ab

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::Path;

const CID22_DIR: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";

const IMAGES: &[&str] = &[
    "3637739.png", // LOSER target
    "1418519.png", // WINNER baseline
    "1025469.png", // also flagged in W44-185 cluster
    "1189261.png",
    "1420710.png",
    "1531677.png",
    "2389166.png",
    "297394.png",
    "1475938.png",
];

const DISTANCE: f32 = 4.0;
const EFFORT: u8 = 7;

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).expect("png decode");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.into_raw(), w, h)
}

fn encode(pixels: &[u8], w: u32, h: u32, disable_spec: Option<&str>) -> usize {
    unsafe {
        match disable_spec {
            Some(spec) => std::env::set_var("JXL_W44_201_DISABLE_BUCKETS", spec),
            None => std::env::remove_var("JXL_W44_201_DISABLE_BUCKETS"),
        }
    }
    let cfg = LossyConfig::new(DISTANCE)
        .with_effort(EFFORT)
        .with_strategy(EncoderStrategy::Zenjxl);
    let buf = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .expect("encode");
    buf.len()
}

fn main() {
    println!("image\tw\th\tA_bytes\tB_bytes\tC_bytes\tdelta_B_vs_A\tdelta_C_vs_A\tpct_B\tpct_C");
    let mut sum_a = 0i64;
    let mut sum_b = 0i64;
    let mut sum_c = 0i64;
    for name in IMAGES {
        let path = format!("{}/{}", CID22_DIR, name);
        let (pixels, w, h) = load_png(Path::new(&path));
        let a = encode(&pixels, w, h, None);
        let b = encode(&pixels, w, h, Some("3"));
        let c = encode(&pixels, w, h, Some("3,6"));
        let da = b as i64 - a as i64;
        let dc = c as i64 - a as i64;
        let pct_b = 100.0 * da as f64 / a as f64;
        let pct_c = 100.0 * dc as f64 / a as f64;
        sum_a += a as i64;
        sum_b += b as i64;
        sum_c += c as i64;
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:+}\t{:+}\t{:+.2}%\t{:+.2}%",
            name, w, h, a, b, c, da, dc, pct_b, pct_c
        );
    }
    println!(
        "TOTAL\t-\t-\t{}\t{}\t{}\t{:+}\t{:+}\t{:+.2}%\t{:+.2}%",
        sum_a,
        sum_b,
        sum_c,
        sum_b - sum_a,
        sum_c - sum_a,
        100.0 * (sum_b - sum_a) as f64 / sum_a as f64,
        100.0 * (sum_c - sum_a) as f64 / sum_a as f64
    );
}
