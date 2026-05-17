//! Calibration harness for `PatchesData::is_cost_effective`'s
//! per-pixel savings constant (RFC#45 chunk 4).
//!
//! The chunk-3 W5-5 bench (commit `f230dd1`, `patches_per_patch_gate_2026-05-17`)
//! surfaced a hard finding: the per-set `is_cost_effective` model uses a
//! `0.3 bytes/pixel` constant that **under-estimates** actual savings on
//! screenshot content by 3-5×. As a consequence the gate (currently only
//! active in `EncoderMode::Experimental`) would over-reject patches that
//! empirically win big.
//!
//! This harness fits the constant from data. For each (image, distance)
//! cell we:
//!
//! 1. Encode with default patches-on path; capture the
//!    `LastPatchesStats` snapshot recorded by
//!    `find_and_build_with_per_patch_gate` (total_patch_pixels +
//!    occurrences + ref-frame pixel count).
//! 2. Encode with `with_patches(false)` (forces the patches-off path).
//! 3. `actual_savings = bytes_off - bytes_on`. Divide by
//!    `total_patch_pixels * (1/max(distance, 1.0))` to recover the
//!    *empirical* bytes-per-pixel savings constant the model would need
//!    to match observation.
//!
//! Output TSV columns:
//!   `class image width height distance bytes_off bytes_on actual_savings_B
//!    total_patch_pixels unique_refs ref_frame_pixels occurrences
//!    empirical_bpp_d_clamped` (where `bpp_d_clamped = savings /
//!    (total_patch_pixels / max(distance, 1.0))`).
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!       --features __internals --example patches_savings_calibrate \
//!       > benchmarks/patches_savings_calibrate_2026-05-17.tsv 2>/dev/null

use jxl_encoder::__internals::take_last_patches_stats;
use jxl_encoder::{LossyConfig, PixelLayout};

fn main() {
    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));

    // 5 screenshots from gb82-sc — mirror W5-5's selection.
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

    // 5 photos from CID22 validation — mirror W5-5's selection.
    let cid = format!("{}/work/codec-corpus/CID22/CID22-512/validation", base);
    let photos = [
        ("1025469.png", format!("{}/1025469.png", cid)),
        ("1044329.png", format!("{}/1044329.png", cid)),
        ("1189261.png", format!("{}/1189261.png", cid)),
        ("1279330.png", format!("{}/1279330.png", cid)),
        ("1418519.png", format!("{}/1418519.png", cid)),
    ];

    let distances = [0.5_f32, 1.0, 2.0, 4.0];

    println!(
        "class\timage\twidth\theight\tdistance\tbytes_off\tbytes_on\tactual_savings_B\t\
         total_patch_pixels\tunique_refs\tref_frame_pixels\toccurrences\tempirical_bpp_d_clamped"
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
                // Drain any prior thread-local sample.
                let _ = take_last_patches_stats();

                // Patches-on encode (default Reference path).
                let bytes_on = match LossyConfig::new(d).encode(rgb, w, h, PixelLayout::Rgb8) {
                    Ok(b) => b.len(),
                    Err(e) => {
                        eprintln!("encode on {name} d={d}: {e}");
                        continue;
                    }
                };
                let stats_on = take_last_patches_stats();

                // Patches-off encode.
                let bytes_off = match LossyConfig::new(d).with_patches(false).encode(
                    rgb,
                    w,
                    h,
                    PixelLayout::Rgb8,
                ) {
                    Ok(b) => b.len(),
                    Err(e) => {
                        eprintln!("encode off {name} d={d}: {e}");
                        continue;
                    }
                };
                // Drain (patches-off run still calls the detection
                // function and writes to the slot? No — the call site
                // guards on `self.enable_patches`. So the slot will be
                // unchanged. We drain anyway for hygiene.)
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

                let empirical_bpp = if total_patch_pixels > 0 {
                    let d_clamped = d.max(1.0) as f64;
                    (savings_signed as f64) * d_clamped / (total_patch_pixels as f64)
                } else {
                    0.0
                };

                println!(
                    "{label}\t{name}\t{w}\t{h}\t{d:.2}\t{bytes_off}\t{bytes_on}\t\
                     {savings_signed}\t{total_patch_pixels}\t{unique_refs}\t\
                     {ref_frame_pixels}\t{occurrences}\t{empirical_bpp:.6}",
                );
            }
        }
    }
}
