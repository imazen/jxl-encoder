//! Extended calibration harness — runs across ALL gb82-sc screenshots
//! to widen the fit beyond the 5 lossy-chunks selection.

use jxl_encoder::__internals::{
    find_and_build_patches_lossless, patches_data_stats, patches_trial_overhead,
    patches_trial_overhead_lossless,
};
use jxl_encoder::{LosslessConfig, PixelLayout};

fn main() {
    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));

    // All 11 gb82-sc screenshots.
    let dir = format!("{}/work/codec-corpus/gb82-sc", base);
    let names = [
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

    println!(
        "class\timage\twidth\theight\tbytes_off\tbytes_on\tactual_savings_B\t\
         total_patch_pixels\tunique_refs\tref_frame_pixels\toccurrences\t\
         ref_overhead_xyb_B\tref_overhead_lossless_B\tdict_overhead_B\t\
         total_overhead_xyb_B\ttotal_overhead_lossless_B\tempirical_bpp\t\
         c_needed_xyb\tc_needed_lossless\toverhead_overshoot_ratio"
    );

    // Photos for a no-detection sanity row.
    let cid = format!("{}/work/codec-corpus/CID22/CID22-512/validation", base);
    let photos = [
        ("1025469.png", format!("{}/1025469.png", cid)),
        ("1044329.png", format!("{}/1044329.png", cid)),
        ("1189261.png", format!("{}/1189261.png", cid)),
        ("1279330.png", format!("{}/1279330.png", cid)),
        ("1418519.png", format!("{}/1418519.png", cid)),
    ];

    let mut all_paths: Vec<(&str, String, String)> = names
        .iter()
        .map(|n| ("screenshot", n.to_string(), format!("{}/{}", dir, n)))
        .collect();
    for (n, p) in photos.iter() {
        all_paths.push(("photo", n.to_string(), p.clone()));
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
        // C needed to admit this cell at the chunk-6 1.5x safety margin:
        //   2 * (total_patch_pixels * C) >= 3 * total_overhead
        //   C >= 1.5 * total_overhead / total_patch_pixels
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
