//! Expanded-corpus calibration of the lossy `is_cost_effective` gate
//! (RFC#45 chunks 4-7, calibrated 2026-05-17 on 5 screenshots).
//!
//! The original chunks 4-7 fit `SAVINGS_BYTES_PER_PIXEL = 0.78` and the
//! `1/sqrt(d_clamped)` divisor against a 5-screenshot × 4-distance grid
//! (`benchmarks/patches_savings_calibrate_2026-05-17.tsv`). The W11-1
//! lossless backport extended to 11 gb82-sc screenshots; W12-1 honest
//! finding: "no NEW admissions on gb82-sc — same 8/8 cells admitted at
//! C=0.35 (lossless) as at C=0.45 (W11-1)."
//!
//! This harness probes whether a *larger* corpus (gb82-sc +
//! qoi-benchmark/screenshot_web — 25 unique screenshots total) shifts
//! the fit. For each (image, distance) cell we:
//!
//! 1. Encode patches-on in `EncoderMode::Experimental` (where the
//!    per-set gate fires). Capture `LastPatchesStats`.
//! 2. Encode patches-off in the same mode.
//! 3. Compute `actual_savings = bytes_off - bytes_on` and
//!    `empirical_bpp_sqrt_d = savings * sqrt(max(d, 1.0)) / pixels`
//!    (the chunk-5 model shape).
//!
//! Photos are included to confirm the no-regression invariant: patches
//! should produce no detection (or be rejected by the gate) on real
//! photos at all 4 distances.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --features __internals \
//!       --example patches_lossy_gate_expanded_corpus \
//!       > benchmarks/patches_lossy_gate_expanded_corpus_2026-05-17.tsv

use jxl_encoder::__internals::take_last_patches_stats;
use jxl_encoder::{EncoderMode, LossyConfig, PixelLayout};

fn main() {
    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));

    // 25 screenshots: 11 gb82-sc + 14 qoi-benchmark/screenshot_web.
    let screenshots = build_screenshot_corpus(&base);

    // 15 photos: 5 from CID22 validation + 5 from clic2025-1024 +
    // 5 from gb82 photo corpus.
    let photos = build_photo_corpus(&base);

    let distances = [0.5_f32, 1.0, 2.0, 4.0];

    println!(
        "class\timage\twidth\theight\tdistance\tbytes_off\tbytes_on\tactual_savings_B\t\
         total_patch_pixels\tunique_refs\tref_frame_pixels\toccurrences\t\
         empirical_bpp_sqrt_d"
    );

    for (label, set) in [("screenshot", &screenshots[..]), ("photo", &photos[..])] {
        for (name, path) in set {
            let Ok(img) = image::open(path) else {
                eprintln!("WARN: failed to open {path}");
                continue;
            };
            let img = img.to_rgb8();
            let (w, h) = (img.width(), img.height());
            let rgb = img.as_raw();
            for &d in &distances {
                let _ = take_last_patches_stats();

                // Experimental mode: per-set is_cost_effective gate is
                // the only patches gate. patches-on default.
                let bytes_on = match LossyConfig::new(d)
                    .with_mode(EncoderMode::Experimental)
                    .encode(rgb, w, h, PixelLayout::Rgb8)
                {
                    Ok(b) => b.len(),
                    Err(e) => {
                        eprintln!("encode on {name} d={d}: {e}");
                        continue;
                    }
                };
                let stats_on = take_last_patches_stats();

                let bytes_off = match LossyConfig::new(d)
                    .with_mode(EncoderMode::Experimental)
                    .with_patches(false)
                    .encode(rgb, w, h, PixelLayout::Rgb8)
                {
                    Ok(b) => b.len(),
                    Err(e) => {
                        eprintln!("encode off {name} d={d}: {e}");
                        continue;
                    }
                };
                let _ = take_last_patches_stats();

                let savings_signed = bytes_off as i64 - bytes_on as i64;
                let (total_patch_pixels, unique_refs, ref_frame_pixels, occurrences) =
                    match stats_on {
                        Some(s) => (
                            s.total_patch_pixels,
                            s.unique_refs_after_gate,
                            s.ref_frame_pixels_after_gate,
                            s.total_occurrences_after_gate,
                        ),
                        None => (0, 0, 0, 0),
                    };

                let d_clamped = (d.max(1.0) as f64).sqrt();
                let empirical_bpp_sqrt_d = if total_patch_pixels > 0 {
                    (savings_signed as f64) * d_clamped / (total_patch_pixels as f64)
                } else {
                    0.0
                };

                println!(
                    "{label}\t{name}\t{w}\t{h}\t{d:.2}\t{bytes_off}\t{bytes_on}\t\
                     {savings_signed}\t{total_patch_pixels}\t{unique_refs}\t\
                     {ref_frame_pixels}\t{occurrences}\t\
                     {empirical_bpp_sqrt_d:.6}",
                );
            }
        }
    }
}

fn build_screenshot_corpus(base: &str) -> Vec<(String, String)> {
    let gb82 = [
        "codec_wiki.png",
        "gmessages.png",
        "graph.png",
        "gui.png",
        "imac_dark.png",
        "imac_g3.png",
        "imac_g3_strip.png",
        "imessage.png",
        "terminal.png",
        "windows95.png",
        "windows.png",
    ];
    let qoi = [
        "amazon.com.png",
        "apple.com.png",
        "cnn.com.png",
        "creativecommons.org.png",
        "duckduckgo.com.png",
        "en.wikipedia.org.png",
        "imdb.com.png",
        "microsoft.com.png",
        "news.ycombinator.com.png",
        "nytimes.com.png",
        "phoboslab.org.png",
        "reddit.com.png",
        "stripe.com.png",
        "sublime.png",
    ];
    let mut v = Vec::with_capacity(gb82.len() + qoi.len());
    for n in &gb82 {
        v.push((
            (*n).to_string(),
            format!("{}/work/codec-corpus/gb82-sc/{}", base, n),
        ));
    }
    for n in &qoi {
        v.push((
            (*n).to_string(),
            format!(
                "{}/work/codec-corpus/qoi-benchmark/screenshot_web/{}",
                base, n
            ),
        ));
    }
    v
}

fn build_photo_corpus(base: &str) -> Vec<(String, String)> {
    let cid = format!("{}/work/codec-corpus/CID22/CID22-512/validation", base);
    let cid_photos = [
        "1025469.png",
        "1044329.png",
        "1189261.png",
        "1279330.png",
        "1418519.png",
    ];
    let clic = format!("{}/work/codec-corpus/clic2025-1024", base);
    let clic_photos = [
        "100f17892cf7bb12beb12479641fcc4c.png",
        "c80999cd5533e448167cd9b3ccfa2a37.png",
        "22ea12c903e41583b7c469cb86040157.png",
        "d1a9be98d1936065967adac50a6fb750.png",
        "0c49a5cce349020bbba2f97ae41e90ba.png",
    ];
    let gb82p = format!("{}/work/codec-corpus/gb82", base);
    let gb82_photos = [
        "rain-lossless.png",
        "path-lossless.png",
        "haze-lossless.png",
        "pixel-lossless.png",
        "guitar-lossless.png",
    ];

    let mut v = Vec::with_capacity(15);
    for n in &cid_photos {
        v.push(((*n).to_string(), format!("{}/{}", cid, n)));
    }
    for n in &clic_photos {
        v.push(((*n).to_string(), format!("{}/{}", clic, n)));
    }
    for n in &gb82_photos {
        v.push(((*n).to_string(), format!("{}/{}", gb82p, n)));
    }
    v
}
