//! W44-206 Phase-1 probe: recalibrate the W44-82 coeff_orders cost-model
//! `savings_factor` multiplier from 1.0 → {0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
//! with the W44-201 + W44-205 per-bucket gates DISABLED (env hooks
//! `JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS=1` +
//! `JXL_W44_205_FORCE_LEGACY_MEDIUM_BUCKETS=1`).
//!
//! Goal: measure the multiplier's effect IN ISOLATION (without the
//! production per-bucket gates compounding the result). If a single
//! recalibrated multiplier value subsumes the W44-201+W44-205 gates
//! (matches or beats their combined wins), C2 ships as a multiplier-only
//! fix. If it's additive, C2 ships as an additional gate on top.
//!
//! Cell set: union of W44-201-wide cells + W44-205 PROTECT/SCRN cells
//! that are not already represented. ~30 cells.
//!
//! Output: TSV with columns:
//!   label, w, h, distance, effort, f_baseline(=1.0), f_0_3, f_0_4, f_0_5,
//!   f_0_6, f_0_7, f_0_8, best_factor, best_bytes, best_pct_vs_baseline

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::Path;

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Some((rgb.into_raw(), w, h))
}

fn set_env(savings_factor: Option<f32>, disable_w44_201: bool, disable_w44_205: bool) {
    // SAFETY: Single-threaded probe. No other threads access env vars
    // concurrently. Race-free in this main()-driven sequential bench.
    unsafe {
        match savings_factor {
            Some(f) => std::env::set_var("JXL_W44_201_SAVINGS_FACTOR", format!("{}", f)),
            None => std::env::remove_var("JXL_W44_201_SAVINGS_FACTOR"),
        }
        if disable_w44_201 {
            std::env::set_var("JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS", "1");
        } else {
            std::env::remove_var("JXL_W44_201_FORCE_LEGACY_LARGE_BUCKETS");
        }
        if disable_w44_205 {
            std::env::set_var("JXL_W44_205_FORCE_LEGACY_MEDIUM_BUCKETS", "1");
        } else {
            std::env::remove_var("JXL_W44_205_FORCE_LEGACY_MEDIUM_BUCKETS");
        }
    }
}

fn encode_zenjxl(pixels: &[u8], w: u32, h: u32, distance: f32) -> usize {
    let cfg = LossyConfig::new(distance)
        .with_effort(7)
        .with_strategy(EncoderStrategy::Zenjxl);
    let buf = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .expect("encode");
    buf.len()
}

fn main() {
    // Cells: 16 from W44-205 production-AB (LOSER + PROTECT + SCRN) + 14
    // additional that test cross-corpus surface where the recalibration
    // might fire (windows95 / 1531677 d=5 / 1420710 d=5).
    let cells: Vec<(&str, &str, f32)> = vec![
        // ===== LOSER cluster (W44-205 wins) =====
        (
            "LOSER_3637739_d2.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png",
            2.0,
        ),
        (
            "LOSER_3637739_d3.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png",
            3.0,
        ),
        (
            "LOSER_3637739_d4.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png",
            4.0,
        ),
        (
            "LOSER_3637739_d5.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png",
            5.0,
        ),
        (
            "LOSER_297394_d2.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/297394.png",
            2.0,
        ),
        (
            "LOSER_297394_d3.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/297394.png",
            3.0,
        ),
        (
            "LOSER_297394_d4.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/297394.png",
            4.0,
        ),
        (
            "LOSER_297394_d5.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/297394.png",
            5.0,
        ),
        (
            "LOSER_7062219_d2.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/7062219.png",
            2.0,
        ),
        (
            "LOSER_7062219_d3.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/7062219.png",
            3.0,
        ),
        (
            "LOSER_7062219_d4.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/7062219.png",
            4.0,
        ),
        (
            "LOSER_7062219_d5.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/7062219.png",
            5.0,
        ),
        (
            "LOSER_1475938_d2.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png",
            2.0,
        ),
        (
            "LOSER_1475938_d3.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png",
            3.0,
        ),
        (
            "LOSER_1475938_d4.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png",
            4.0,
        ),
        (
            "LOSER_1475938_d5.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1475938.png",
            5.0,
        ),
        // ===== PROTECT (W44-82 F-D cells that ruled out gate removal) =====
        (
            "PROTECT_1420710_d4.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
            4.0,
        ),
        (
            "PROTECT_1531677_d4.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
            4.0,
        ),
        (
            "PROTECT_1189261_d4.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png",
            4.0,
        ),
        (
            "PROTECT_1418519_d4.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
            4.0,
        ),
        (
            "PROTECT_1418519_d5.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
            5.0,
        ),
        (
            "PROTECT_1420710_d5.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png",
            5.0,
        ),
        (
            "PROTECT_1531677_d5.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1531677.png",
            5.0,
        ),
        (
            "PROTECT_1189261_d3.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1189261.png",
            3.0,
        ),
        (
            "PROTECT_2389166_d4.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/2389166.png",
            4.0,
        ),
        (
            "PROTECT_1025469_d4.0_e7",
            "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png",
            4.0,
        ),
        // ===== SCRN cells (W44-205 had +0.14% noise on terminal_d4) =====
        (
            "SCRN_codec_wiki_d1.0_e7",
            "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
            1.0,
        ),
        (
            "SCRN_codec_wiki_d4.0_e7",
            "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
            4.0,
        ),
        (
            "SCRN_terminal_d1.0_e7",
            "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
            1.0,
        ),
        (
            "SCRN_terminal_d4.0_e7",
            "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
            4.0,
        ),
        (
            "SCRN_imac_dark_d1.0_e7",
            "/home/lilith/work/codec-corpus/gb82-sc/imac_dark.png",
            1.0,
        ),
        (
            "SCRN_imac_dark_d4.0_e7",
            "/home/lilith/work/codec-corpus/gb82-sc/imac_dark.png",
            4.0,
        ),
        (
            "SCRN_windows95_d1.0_e7",
            "/home/lilith/work/codec-corpus/gb82-sc/windows95.png",
            1.0,
        ),
        (
            "SCRN_windows95_d4.0_e7",
            "/home/lilith/work/codec-corpus/gb82-sc/windows95.png",
            4.0,
        ),
    ];

    let factors = [1.0_f32, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    // Header
    print!("label\tw\th\tdistance\teffort");
    for f in &factors {
        print!("\tf={:.1}", f);
    }
    print!("\tbest_factor\tbest_bytes\tbest_delta_vs_f1.0\tbest_pct");
    println!();

    let mut sums = vec![0i64; factors.len()];
    let mut count = 0;
    for (label, path, d) in &cells {
        let Some((pixels, w, h)) = load_png(Path::new(path)) else {
            eprintln!("# missing: {}", path);
            continue;
        };
        count += 1;
        let mut row_bytes = vec![0usize; factors.len()];
        for (i, &f) in factors.iter().enumerate() {
            // PHASE-1 PROBE: W44-201 + W44-205 gates DISABLED so the
            // multiplier's effect is measured in isolation.
            set_env(Some(f), true, true);
            row_bytes[i] = encode_zenjxl(&pixels, w, h, *d);
            sums[i] += row_bytes[i] as i64;
        }

        // Find best (smallest)
        let (best_i, best_b) = row_bytes
            .iter()
            .enumerate()
            .min_by_key(|(_, b)| *b)
            .unwrap();
        let baseline = row_bytes[0] as i64;
        let delta = *best_b as i64 - baseline;
        let pct = 100.0 * delta as f64 / baseline as f64;

        print!("{}\t{}\t{}\t{:.2}\t{}", label, w, h, d, 7);
        for b in &row_bytes {
            print!("\t{}", b);
        }
        print!(
            "\t{}\t{}\t{:+}\t{:+.3}%",
            factors[best_i], best_b, delta, pct
        );
        println!();
    }

    // TOTAL row
    print!("TOTAL\t-\t-\t-\t-");
    for s in &sums {
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

    // Per-factor delta vs baseline
    print!("delta_vs_f1.0\t-\t-\t-\t-");
    for s in &sums {
        let d = *s - sums[0];
        let p = 100.0 * d as f64 / sums[0] as f64;
        print!("\t{:+} ({:+.3}%)", d, p);
    }
    println!();

    eprintln!("# cells encoded: {} / {}", count, cells.len());
}
