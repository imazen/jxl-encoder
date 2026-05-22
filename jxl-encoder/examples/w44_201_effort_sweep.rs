//! W44-201 effort sweep: A/B/C variants across e4..=e9 for 3637739 + 1420710
//! to verify the bucket disable doesn't regress on other effort levels.

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::Path;

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).expect("png decode");
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.into_raw(), w, h)
}

/// Same Variant enum as `w44_201_bucket_ab_wide` — see that example's
/// header for the legacy/Bbucket3only/CProd semantics.
#[derive(Clone, Copy)]
enum Variant {
    ALegacy,
    BBucket3Only,
    CProd,
}

fn encode(pixels: &[u8], w: u32, h: u32, distance: f32, effort: u8, v: Variant) -> usize {
    unsafe {
        std::env::remove_var("JXL_W44_201_DISABLE_BUCKETS");
        std::env::remove_var("JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS");
        match v {
            Variant::ALegacy => {
                std::env::set_var("JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS", "1");
            }
            Variant::BBucket3Only => {
                std::env::set_var("JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS", "1");
                std::env::set_var("JXL_W44_201_DISABLE_BUCKETS", "3");
            }
            Variant::CProd => {}
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
    let imgs = &[
        (
            "3637739",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png",
        ),
        (
            "1420710",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
        ),
        (
            "1418519",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
        ),
        (
            "imac_g3",
            "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png",
        ),
    ];
    let efforts = [4u8, 5, 6, 7, 8];
    let distances = [1.0_f32, 2.0, 4.0, 6.0];

    println!("label\tdistance\teffort\tA_bytes\tB_bytes\tC_bytes\tdelta_B\tdelta_C\tpct_B\tpct_C");
    let mut sum_a = 0i64;
    let mut sum_b = 0i64;
    let mut sum_c = 0i64;
    let mut worst_b: (f64, String) = (0.0, "".to_string());
    let mut worst_c: (f64, String) = (0.0, "".to_string());
    for (label, path) in imgs {
        let (pixels, w, h) = load_png(Path::new(path));
        for &e in &efforts {
            for &d in &distances {
                let a = encode(&pixels, w, h, d, e, Variant::ALegacy);
                let b = encode(&pixels, w, h, d, e, Variant::BBucket3Only);
                let c = encode(&pixels, w, h, d, e, Variant::CProd);
                let da = b as i64 - a as i64;
                let dc = c as i64 - a as i64;
                let pct_b = 100.0 * da as f64 / a as f64;
                let pct_c = 100.0 * dc as f64 / a as f64;
                sum_a += a as i64;
                sum_b += b as i64;
                sum_c += c as i64;
                if pct_b > worst_b.0 {
                    worst_b = (pct_b, format!("{}_d{}_e{}", label, d, e));
                }
                if pct_c > worst_c.0 {
                    worst_c = (pct_c, format!("{}_d{}_e{}", label, d, e));
                }
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{:+}\t{:+}\t{:+.2}%\t{:+.2}%",
                    label, d, e, a, b, c, da, dc, pct_b, pct_c
                );
            }
        }
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
    eprintln!("worst B regression: {:+.2}% on {}", worst_b.0, worst_b.1);
    eprintln!("worst C regression: {:+.2}% on {}", worst_c.0, worst_c.1);
}
