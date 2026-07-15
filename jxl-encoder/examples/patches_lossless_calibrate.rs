//! Calibration harness for a lossless-mode `is_cost_effective_lossless`
//! gate (RFC#45 chunk 4-7 backport to lossless).
//!
//! The lossy `PatchesData::is_cost_effective` gate has been recalibrated
//! in chunks 4-7 (C=1.0 → 0.78 under 1/sqrt(d), 2.0× → 1.5× safety
//! margin, per-image trial-encode of both reference and dictionary
//! overhead). The lossless path (`find_and_build_lossless` →
//! `subtract_patches_modular`) ships every detected patch with only a
//! 1% coverage filter — no overhead/savings comparison.
//!
//! This harness fits a lossless equivalent. For each (image, content
//! class) cell we:
//!
//! 1. Encode with default patches-on path; capture the
//!    `LastPatchesStats` snapshot recorded by
//!    `find_and_build_with_per_patch_gate` (which is invoked via the
//!    no-per-patch-gate variant inside `find_and_build_lossless`).
//!    NOTE: the calibration sink is currently only wired into the
//!    LOSSY detector. For lossless we re-run detection here to capture
//!    the same telemetry.
//! 2. Encode with `with_patches(false)` (forces patches-off).
//! 3. `actual_savings = bytes_off - bytes_on`. Divide by
//!    `total_patch_pixels` to recover the empirical bytes-per-pixel
//!    savings constant — lossless has no distance axis, so the model
//!    shape is simply `pixels * C_LOSSLESS`.
//!
//! Output TSV columns:
//!   `class image width height bytes_off bytes_on actual_savings_B
//!    total_patch_pixels unique_refs ref_frame_pixels occurrences
//!    empirical_bpp` (where `bpp = savings / total_patch_pixels`).
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!       --features __internals --example patches_lossless_calibrate \
//!       > benchmarks/patches_lossless_savings_calibrate_2026-05-17.tsv 2>/dev/null

use jxl_encoder::__internals::{
    find_and_build_patches_lossless, patches_data_stats, patches_trial_overhead_lossless,
};
use jxl_encoder::{LosslessConfig, PixelLayout};

fn main() {
    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));

    // 5 screenshots from gb82-sc — mirror the lossy chunks 4-7 selection.
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

    // 5 photos from CID22 validation — mirror the lossy selection.
    let cid = format!("{}/work/codec-corpus/CID22/CID22-512/validation", base);
    let photos = [
        ("1025469.png", format!("{}/1025469.png", cid)),
        ("1044329.png", format!("{}/1044329.png", cid)),
        ("1189261.png", format!("{}/1189261.png", cid)),
        ("1279330.png", format!("{}/1279330.png", cid)),
        ("1418519.png", format!("{}/1418519.png", cid)),
    ];

    println!(
        "class\timage\twidth\theight\tbytes_off\tbytes_on\tactual_savings_B\t\
         total_patch_pixels\tunique_refs\tref_frame_pixels\toccurrences\t\
         ref_overhead_B\tdict_overhead_B\ttotal_overhead_B\tempirical_bpp"
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

            // Patches-on encode (default path: effort=7 enables patches).
            let bytes_on = match LosslessConfig::new().encode(rgb, w, h, PixelLayout::Rgb8) {
                Ok(b) => b.len(),
                Err(e) => {
                    eprintln!("encode on {name}: {e}");
                    continue;
                }
            };

            // Re-run detection (calibration-only) to capture telemetry
            // about what was actually detected for this image plus the
            // trial-encoded overhead the gate would see.
            let (
                total_patch_pixels,
                unique_refs,
                ref_frame_pixels,
                occurrences,
                ref_overhead_b,
                dict_overhead_b,
            ) = detect_lossless_stats_full(rgb, w as usize, h as usize, 3, 8);
            let total_overhead_b = ref_overhead_b + dict_overhead_b;

            // Patches-off encode.
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
}

/// Call the lossless detector directly to capture pre-encode telemetry
/// plus the trial-encoded `(ref_overhead, dict_overhead)` the gate
/// would see.
fn detect_lossless_stats_full(
    pixels: &[u8],
    w: usize,
    h: usize,
    num_channels: usize,
    bit_depth: u32,
) -> (usize, usize, usize, usize, usize, usize) {
    let Some(pd) = find_and_build_patches_lossless(pixels, w, h, num_channels, bit_depth) else {
        return (0, 0, 0, 0, 0, 0);
    };
    let (tpp, ur, rfp, occ) = patches_data_stats(&pd);
    // Lossless calibrate → the lossless trial-overhead estimator (the one the
    // production `is_cost_effective_lossless` gate uses), keyed on `bit_depth`.
    // (The lossy `patches_trial_overhead` takes a `distance` instead; this
    // example never had one — a stale call left by the lossless-variant split.)
    let (ref_b, dict_b) = patches_trial_overhead_lossless(&pd, bit_depth, true);
    (tpp, ur, rfp, occ, ref_b, dict_b)
}
