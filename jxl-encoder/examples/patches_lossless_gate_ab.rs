//! A/B harness for the lossless `is_cost_effective_lossless` gate
//! (RFC#45 chunks 4-7 backport to the lossless path).
//!
//! Pre-recalibration the lossless path had NO gate at all — every
//! detected patch (above the 1% coverage filter) shipped with its
//! reference frame + dictionary section overhead. On low-content
//! images that means we paid ref-frame overhead for marginal savings.
//!
//! This harness measures lossless bytes with patches detected-and-
//! admitted (default path) vs. patches forced-off, on 5 screenshots
//! and 5 photos. Reports per-image delta and whether the new
//! `is_cost_effective_lossless` gate's verdict matches the empirical
//! sign of `bytes_off - bytes_on`.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --features __internals \
//!       --example patches_lossless_gate_ab \
//!       > benchmarks/patches_lossless_gate_ab_2026-05-17.tsv

use jxl_encoder::__internals::{
    find_and_build_patches_lossless, is_cost_effective_lossless, patches_data_stats,
};
use jxl_encoder::{LosslessConfig, PixelLayout};

fn main() {
    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));

    let screenshots = [
        (
            "windows95.png",
            format!("{}/work/codec-corpus/gb82-sc/windows95.png", base),
        ),
        (
            "terminal.png",
            format!("{}/work/codec-corpus/gb82-sc/terminal.png", base),
        ),
        (
            "codec_wiki.png",
            format!("{}/work/codec-corpus/gb82-sc/codec_wiki.png", base),
        ),
        (
            "imac_g3.png",
            format!("{}/work/codec-corpus/gb82-sc/imac_g3.png", base),
        ),
        (
            "windows.png",
            format!("{}/work/codec-corpus/gb82-sc/windows.png", base),
        ),
    ];

    let cid = format!("{}/work/codec-corpus/CID22/CID22-512/validation", base);
    let photos = [
        ("1025469.png", format!("{}/1025469.png", cid)),
        ("1044329.png", format!("{}/1044329.png", cid)),
        ("1189261.png", format!("{}/1189261.png", cid)),
        ("1279330.png", format!("{}/1279330.png", cid)),
        ("1418519.png", format!("{}/1418519.png", cid)),
    ];

    println!(
        "class\timage\twidth\theight\tbytes_off\tbytes_on\tdelta_B\t\
         total_patch_pixels\tunique_refs\tref_frame_pixels\toccurrences\t\
         gate_says_effective\tempirical_win"
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

            // Default lossless (patches-on if effective).
            let bytes_on = match LosslessConfig::new().encode(rgb, w, h, PixelLayout::Rgb8) {
                Ok(b) => b.len(),
                Err(e) => {
                    eprintln!("encode on {name}: {e}");
                    continue;
                }
            };

            // patches-off forcing.
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
            let delta_b = bytes_off as i64 - bytes_on as i64;

            // Re-run detection to read the gate verdict + stats.
            let (
                total_patch_pixels,
                unique_refs,
                ref_frame_pixels,
                occurrences,
                gate_says_effective,
            ) = match find_and_build_patches_lossless(rgb, w as usize, h as usize, 3, 8) {
                Some(pd) => {
                    let (tpp, ur, rfp, occ) = patches_data_stats(&pd);
                    // Chunk-5 signature: bit_depth + use_ans
                    let eff = is_cost_effective_lossless(&pd, 8, true);
                    (tpp, ur, rfp, occ, eff)
                }
                None => (0, 0, 0, 0, false),
            };

            let empirical_win = delta_b > 0; // off > on means patches saved bytes

            println!(
                "{label}\t{name}\t{w}\t{h}\t{bytes_off}\t{bytes_on}\t{delta_b}\t\
                 {total_patch_pixels}\t{unique_refs}\t{ref_frame_pixels}\t{occurrences}\t\
                 {gate_says_effective}\t{empirical_win}",
            );
        }
    }
}
