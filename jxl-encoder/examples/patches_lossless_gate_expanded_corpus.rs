//! Expanded-corpus calibration of the lossless `is_cost_effective_lossless`
//! gate (RFC#45 lossless chunks 4-7, calibrated 2026-05-17 on 11 gb82-sc
//! screenshots — see `patches_lossless_calibrate_all`).
//!
//! W11-1 fit `SAVINGS_BYTES_PER_PIXEL_LOSSLESS = 0.45` on XYB-shape
//! overhead. W12-1 lossless chunk 5 (lossless-shape trial encoder)
//! tightened to `0.35`. W12-1 honest finding: "no NEW admissions on
//! gb82-sc — same 8/8 cells admitted."
//!
//! This harness probes whether a *larger* corpus (gb82-sc +
//! qoi-benchmark/screenshot_web — 25 unique screenshots total) shifts
//! the fit, plus a 15-photo no-regression panel.
//!
//! Per-cell output mirrors `patches_lossless_calibrate_all` so the
//! existing percentile / c_needed analysis tooling carries over.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --features __internals \
//!       --example patches_lossless_gate_expanded_corpus \
//!       > benchmarks/patches_lossless_gate_expanded_corpus_2026-05-17.tsv

use jxl_encoder::__internals::{
    find_and_build_patches_lossless, patches_data_stats, patches_trial_overhead,
    patches_trial_overhead_lossless,
};
use jxl_encoder::{LosslessConfig, PixelLayout};

fn main() {
    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));

    let screenshots = build_screenshot_corpus(&base);
    let photos = build_photo_corpus(&base);

    println!(
        "class\timage\twidth\theight\tbytes_off\tbytes_on\tactual_savings_B\t\
         total_patch_pixels\tunique_refs\tref_frame_pixels\toccurrences\t\
         ref_overhead_xyb_B\tref_overhead_lossless_B\tdict_overhead_B\t\
         total_overhead_xyb_B\ttotal_overhead_lossless_B\tempirical_bpp\t\
         c_needed_xyb\tc_needed_lossless\toverhead_overshoot_ratio"
    );

    let mut all_paths: Vec<(&'static str, String, String)> = Vec::new();
    for (n, p) in &screenshots {
        all_paths.push(("screenshot", n.clone(), p.clone()));
    }
    for (n, p) in &photos {
        all_paths.push(("photo", n.clone(), p.clone()));
    }

    for (label, name, path) in &all_paths {
        let Ok(img) = image::open(path) else {
            eprintln!("WARN: failed to open {path}");
            continue;
        };
        let img = img.to_rgb8();
        let (w, h) = (img.width(), img.height());
        let rgb = img.as_raw();

        let bytes_on = match LosslessConfig::new().encode(rgb, w, h, PixelLayout::Rgb8) {
            Ok(b) => b.len(),
            Err(e) => {
                eprintln!("encode on {name}: {e}");
                continue;
            }
        };

        let (
            total_patch_pixels,
            unique_refs,
            ref_frame_pixels,
            occurrences,
            ref_overhead_xyb_b,
            ref_overhead_lossless_b,
            dict_overhead_b,
        ) = match find_and_build_patches_lossless(rgb, w as usize, h as usize, 3, 8) {
            Some(pd) => {
                let (tpp, ur, rfp, occ) = patches_data_stats(&pd);
                let (rb_xyb, db) = patches_trial_overhead(&pd, true);
                let (rb_lossless, _) = patches_trial_overhead_lossless(&pd, 8, true);
                (tpp, ur, rfp, occ, rb_xyb, rb_lossless, db)
            }
            None => (0, 0, 0, 0, 0, 0, 0),
        };
        let total_overhead_xyb_b = ref_overhead_xyb_b + dict_overhead_b;
        let total_overhead_lossless_b = ref_overhead_lossless_b + dict_overhead_b;

        let bytes_off =
            match LosslessConfig::new()
                .with_patches(false)
                .encode(rgb, w, h, PixelLayout::Rgb8)
            {
                Ok(b) => b.len(),
                Err(e) => {
                    eprintln!("encode off {name}: {e}");
                    continue;
                }
            };

        let savings_signed = bytes_off as i64 - bytes_on as i64;
        let empirical_bpp = if total_patch_pixels > 0 {
            savings_signed as f64 / total_patch_pixels as f64
        } else {
            0.0
        };
        let c_needed_xyb = if total_patch_pixels > 0 {
            1.5 * total_overhead_xyb_b as f64 / total_patch_pixels as f64
        } else {
            0.0
        };
        let c_needed_lossless = if total_patch_pixels > 0 {
            1.5 * total_overhead_lossless_b as f64 / total_patch_pixels as f64
        } else {
            0.0
        };
        let overhead_overshoot_ratio = if ref_overhead_lossless_b > 0 {
            ref_overhead_xyb_b as f64 / ref_overhead_lossless_b as f64
        } else {
            0.0
        };

        println!(
            "{label}\t{name}\t{w}\t{h}\t{bytes_off}\t{bytes_on}\t\
             {savings_signed}\t{total_patch_pixels}\t{unique_refs}\t\
             {ref_frame_pixels}\t{occurrences}\t\
             {ref_overhead_xyb_b}\t{ref_overhead_lossless_b}\t{dict_overhead_b}\t\
             {total_overhead_xyb_b}\t{total_overhead_lossless_b}\t{empirical_bpp:.6}\t\
             {c_needed_xyb:.4}\t{c_needed_lossless:.4}\t{overhead_overshoot_ratio:.4}",
        );
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
