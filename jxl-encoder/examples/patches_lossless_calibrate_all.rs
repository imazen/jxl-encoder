//! Extended calibration harness — runs across ALL gb82-sc screenshots
//! to widen the fit beyond the 5 lossy-chunks selection.

use jxl_encoder::__internals::{
    find_and_build_patches_lossless, patches_data_stats, patches_trial_overhead,
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
         ref_overhead_B\tdict_overhead_B\ttotal_overhead_B\tempirical_bpp"
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
            ref_overhead_b,
            dict_overhead_b,
        ) = match find_and_build_patches_lossless(rgb, w as usize, h as usize, 3, 8) {
            Some(pd) => {
                let (tpp, ur, rfp, occ) = patches_data_stats(&pd);
                let (rb, db) = patches_trial_overhead(&pd, true);
                (tpp, ur, rfp, occ, rb, db)
            }
            None => (0, 0, 0, 0, 0, 0),
        };
        let total_overhead_b = ref_overhead_b + dict_overhead_b;

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

        println!(
            "{label}\t{name}\t{w}\t{h}\t{bytes_off}\t{bytes_on}\t\
             {savings_signed}\t{total_patch_pixels}\t{unique_refs}\t\
             {ref_frame_pixels}\t{occurrences}\t\
             {ref_overhead_b}\t{dict_overhead_b}\t{total_overhead_b}\t{empirical_bpp:.6}",
        );
    }
}
