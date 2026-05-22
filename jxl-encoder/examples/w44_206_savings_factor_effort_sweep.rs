//! W44-206 Phase-3 effort sweep: spot-check whether the multiplier
//! delivers additional wins at e=5/6 (below the buttloop) compared to e=7
//! (covered in Phases 1+2). If yes, we may want per-effort dispatch.
//!
//! Sub-cells: 5 representative photos × 4 efforts × d=4. Production
//! gates ON.

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::Path;

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Some((rgb.into_raw(), w, h))
}

fn set_env(savings_factor: Option<f32>) {
    // SAFETY: Single-threaded probe. No other threads access env vars
    // concurrently. Race-free in this main()-driven sequential bench.
    unsafe {
        match savings_factor {
            Some(f) => std::env::set_var("JXL_W44_201_SAVINGS_FACTOR", format!("{}", f)),
            None => std::env::remove_var("JXL_W44_201_SAVINGS_FACTOR"),
        }
        std::env::remove_var("JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS");
        std::env::remove_var("JXL_W44_205_FORCE_LEGACY_MEDIUM_BUCKETS");
    }
}

fn encode_zenjxl(pixels: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> usize {
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
    let images: Vec<(&str, &str)> = vec![
        (
            "3637739",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png",
        ),
        (
            "297394",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/297394.png",
        ),
        (
            "7062219",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/7062219.png",
        ),
        (
            "1475938",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png",
        ),
        (
            "1531677",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
        ),
    ];
    let efforts: [u8; 4] = [4, 5, 6, 7];
    let distances: [f32; 3] = [2.0, 3.0, 4.0];

    let factors = [1.0_f32, 0.3, 0.5, 0.7];

    print!("label\teffort\tdistance");
    for f in &factors {
        print!("\tf={:.1}", f);
    }
    print!("\tbest_factor\tbest_bytes\tdelta_vs_1.0\tpct");
    println!();

    let mut per_effort_sums: Vec<Vec<i64>> =
        efforts.iter().map(|_| vec![0; factors.len()]).collect();

    for (img_label, path) in &images {
        let Some((pixels, w, h)) = load_png(Path::new(path)) else {
            continue;
        };
        for (ei, &e) in efforts.iter().enumerate() {
            for &d in &distances {
                let mut row_bytes = vec![0usize; factors.len()];
                for (i, &f) in factors.iter().enumerate() {
                    set_env(Some(f));
                    row_bytes[i] = encode_zenjxl(&pixels, w, h, d, e);
                    per_effort_sums[ei][i] += row_bytes[i] as i64;
                }
                let (best_i, best_b) = row_bytes
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, b)| *b)
                    .unwrap();
                let baseline = row_bytes[0] as i64;
                let delta = *best_b as i64 - baseline;
                let pct = 100.0 * delta as f64 / baseline as f64;
                print!("{}\te{}\td{:.1}", img_label, e, d);
                for b in &row_bytes {
                    print!("\t{}", b);
                }
                print!(
                    "\t{}\t{}\t{:+}\t{:+.3}%",
                    factors[best_i], best_b, delta, pct
                );
                println!();
            }
        }
    }

    // Per-effort totals
    for (ei, &e) in efforts.iter().enumerate() {
        let sums = &per_effort_sums[ei];
        print!("PER_EFFORT_TOTAL\te{}\t-", e);
        for s in sums {
            print!("\t{}", s);
        }
        let (best_i, best_b) = sums.iter().enumerate().min_by_key(|(_, s)| *s).unwrap();
        let baseline = sums[0];
        let delta = *best_b - baseline;
        let pct = 100.0 * delta as f64 / baseline as f64;
        print!(
            "\t{}\t{}\t{:+}\t{:+.3}%",
            factors[best_i], best_b, delta, pct
        );
        println!();
    }
}
