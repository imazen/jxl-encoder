//! W44-20: codec_wiki patches coverage gap vs libjxl.
//!
//! Per W44-7 finding: codec_wiki.png at e7 d=3.0 has cjxl patches section
//! = 64579 bytes vs ours 10733 bytes (6.0× ratio). cjxl admits 6× more
//! content into the ref frame, which then saves more in the AC stream.
//!
//! Prints both the per-stage detection counters (which gate is dropping
//! candidates inside `find_text_like_patches_with_min_peak`?) and the
//! per-patch cost-gate stats (before/after the Rust-only `apply_per_patch_cost_gate`).
//!
//! Output TSV columns:
//!   image distance bytes_on bytes_off savings_B
//!   seeds bg_pct raw_ccs                      <- detection pipeline
//!   rej_no_border rej_inconsistent rej_too_large rej_no_similar rej_low_peak
//!   accepted_ccs accepted_pixels
//!   unique_before_minocc singletons_dropped
//!   final_unique final_occ final_coverage_pct
//!   refs_before_costgate refs_after_costgate gate_drop_pct
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!       --features __internals --example w44_20_codec_wiki_coverage

use jxl_encoder::__internals::{take_last_patches_detect_stats, take_last_patches_stats};
use jxl_encoder::{LossyConfig, PixelLayout};

fn main() {
    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));

    let images = [
        (
            "codec_wiki.png",
            format!("{}/work/codec-corpus/gb82-sc/codec_wiki.png", base),
        ),
        (
            "imac_g3.png",
            format!("{}/work/codec-corpus/gb82-sc/imac_g3.png", base),
        ),
        (
            "terminal.png",
            format!("{}/work/codec-corpus/gb82-sc/terminal.png", base),
        ),
        (
            "windows95.png",
            format!("{}/work/codec-corpus/gb82-sc/windows95.png", base),
        ),
    ];

    let distances = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    println!(
        "image\twidth\theight\tdistance\tbytes_on\tbytes_off\tsavings_B\t\
         seeds\tbg_pct\traw_ccs\t\
         rej_no_border\trej_inconsistent\trej_too_large\trej_no_similar\trej_low_peak\t\
         accepted_ccs\taccepted_pixels\t\
         unique_before_minocc\tsingletons_dropped\t\
         final_unique\tfinal_occ\tfinal_coverage_pct\t\
         refs_before_costgate\trefs_after_costgate\tcostgate_drop_pct"
    );

    for (name, path) in &images {
        let Ok(img) = image::open(path) else {
            eprintln!("WARN: failed to open {path}");
            continue;
        };
        let img = img.to_rgb8();
        let (w, h) = (img.width(), img.height());
        let rgb = img.as_raw();

        for &d in &distances {
            let _ = take_last_patches_stats();
            let _ = take_last_patches_detect_stats();

            let bytes_on = match LossyConfig::new(d).encode(rgb, w, h, PixelLayout::Rgb8) {
                Ok(b) => b.len(),
                Err(e) => {
                    eprintln!("encode on {name} d={d}: {e}");
                    continue;
                }
            };
            let cg = take_last_patches_stats();
            let det = take_last_patches_detect_stats();

            let bytes_off =
                match LossyConfig::new(d)
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
            let _ = take_last_patches_detect_stats();

            let savings = bytes_off as i64 - bytes_on as i64;
            let area = (w as f64) * (h as f64);
            let (
                seeds,
                bg_pct,
                raw_ccs,
                rej_nb,
                rej_inc,
                rej_lg,
                rej_ns,
                rej_lp,
                acc_cc,
                acc_px,
                uniq_min,
                singletons,
                fin_uniq,
                fin_occ,
                fin_cov_pct,
            ) = match det {
                Some(s) => (
                    s.num_seeds,
                    s.bg_count as f64 / area * 100.0,
                    s.raw_ccs,
                    s.reject_no_border,
                    s.reject_inconsistent,
                    s.reject_too_large,
                    s.reject_no_similar,
                    s.reject_low_peak,
                    s.accepted_ccs,
                    s.accepted_pixels,
                    s.unique_before_min_occ,
                    s.singletons_dropped,
                    s.final_unique,
                    s.final_occurrences,
                    s.final_total_patch_pixels as f64 / area * 100.0,
                ),
                None => (0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0),
            };
            let (cg_before, cg_after, cg_drop) = match cg {
                Some(s) => {
                    let drop = if s.unique_refs_before_gate > 0 {
                        (s.unique_refs_before_gate - s.unique_refs_after_gate) as f64
                            / s.unique_refs_before_gate as f64
                            * 100.0
                    } else {
                        0.0
                    };
                    (s.unique_refs_before_gate, s.unique_refs_after_gate, drop)
                }
                None => (0, 0, 0.0),
            };

            println!(
                "{name}\t{w}\t{h}\t{d:.2}\t{bytes_on}\t{bytes_off}\t{savings}\t\
                 {seeds}\t{bg_pct:.2}\t{raw_ccs}\t\
                 {rej_nb}\t{rej_inc}\t{rej_lg}\t{rej_ns}\t{rej_lp}\t\
                 {acc_cc}\t{acc_px}\t\
                 {uniq_min}\t{singletons}\t\
                 {fin_uniq}\t{fin_occ}\t{fin_cov_pct:.2}\t\
                 {cg_before}\t{cg_after}\t{cg_drop:.1}"
            );
        }
    }
}
