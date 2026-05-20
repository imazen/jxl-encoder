// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-112 — Layer-1.5 quant_field discriminator (buttloop INTERNAL vs PRODUCTION).
//!
//! Background: W44-111 ruled OUT butteraugli metric divergence as the root cause
//! of the recurring SSIM2 deficit fixed palliatively by W44-105/107/108/109.
//! The real cause is the buttloop's internal reconstruction diverging from the
//! decoder's output — per-channel max-abs in linear RGB is 0.05-0.17, ~50-170×
//! over the Layer-1 threshold.
//!
//! Three candidates remained after the W44-105/chunk-3 CfL pass-2 fix:
//!  1. **SetQuantField `inv_scale` drift** between buttloop and production
//!  2. **Parallel vs sequential `transform_and_quantize`** semantics
//!     (AdjustQuantBlockAC mid-pass vs after)
//!  3. **`reconstruct_xyb` vs decoder render-path divergence**
//!     (dequant_bias / AdjustQuantBias application order)
//!
//! This example compares the post-internal `quant_field` (captured by the
//! `recon_hook` after the FINAL buttloop iter's `transform_and_quantize_into`)
//! against the post-production `quant_field` (captured in
//! `transform_and_quantize_with_source` after the PARALLEL version's
//! `quant_adjustments` have been applied).
//!
//! Interpretation:
//!  - If `internal_qf == production_qf` per-block AND `internal_params ==
//!    production_params`: candidates 1 & 2 ruled OUT. Drill candidate #3
//!    (recon vs decoder) in W44-113.
//!  - If they differ: candidates 1 or 2 are in play. The
//!    per-block divergence pattern (which blocks differ + by how much)
//!    points at the next step.
//!
//! Run:
//!
//! ```bash
//! CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release \
//!     --example w44_112_qf_discriminator \
//!     --features '__internal_recon_hook butteraugli-loop' \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > benchmarks/w44_112_qf_discriminator_2026-05-20.tsv
//! ```
//!
//! Env vars:
//!   W44_112_CELLS=img1:d:eff,img2:d:eff,... (default: same 4 cells as W44-111)
//!   W44_112_PERBLOCK_DUMP=/tmp/prefix       (writes per-cell per-block CSV when set)

#[cfg(not(all(feature = "__internal_recon_hook", feature = "butteraugli-loop")))]
fn main() {
    eprintln!(
        "w44_112_qf_discriminator requires --features \
         '__internal_recon_hook butteraugli-loop'."
    );
    std::process::exit(2);
}

#[cfg(all(feature = "__internal_recon_hook", feature = "butteraugli-loop"))]
fn main() {
    inner::run();
}

#[cfg(all(feature = "__internal_recon_hook", feature = "butteraugli-loop"))]
mod inner {
    use jxl_encoder::vardct::__recon_hook;
    use jxl_encoder::{LossyConfig, PixelLayout};
    use std::io::Write as _;
    use std::path::PathBuf;

    struct Cell {
        image: String,
        path: PathBuf,
        distance: f32,
        effort: u8,
    }

    fn parse_cells() -> Vec<Cell> {
        let home = std::env::var("HOME").expect("HOME");
        let corpus = format!("{home}/work/codec-corpus");

        if let Ok(spec) = std::env::var("W44_112_CELLS") {
            spec.split(',')
                .filter_map(|s| {
                    let parts: Vec<&str> = s.split(':').collect();
                    if parts.len() != 3 {
                        eprintln!("bad cell spec '{}'", s);
                        return None;
                    }
                    let img = parts[0].to_string();
                    let d: f32 = parts[1].parse().ok()?;
                    let eff: u8 = parts[2].parse().ok()?;
                    let path = if img.contains('/') {
                        PathBuf::from(&img)
                    } else if img.starts_with("terminal")
                        || img.starts_with("codec_wiki")
                        || img.starts_with("imac")
                        || img.starts_with("graph")
                        || img.starts_with("windows")
                    {
                        PathBuf::from(format!("{corpus}/gb82-sc/{img}.png"))
                    } else {
                        PathBuf::from(format!("{corpus}/CID22/CID22-512/validation/{img}.png"))
                    };
                    Some(Cell {
                        image: img,
                        path,
                        distance: d,
                        effort: eff,
                    })
                })
                .collect()
        } else {
            // Match the W44-111 cell set so the comparison is apples-to-apples.
            vec![
                Cell {
                    image: "terminal".into(),
                    path: PathBuf::from(format!("{corpus}/gb82-sc/terminal.png")),
                    distance: 4.0,
                    effort: 8,
                },
                Cell {
                    image: "terminal".into(),
                    path: PathBuf::from(format!("{corpus}/gb82-sc/terminal.png")),
                    distance: 2.0,
                    effort: 8,
                },
                Cell {
                    image: "1025469".into(),
                    path: PathBuf::from(format!("{corpus}/CID22/CID22-512/validation/1025469.png")),
                    distance: 2.0,
                    effort: 8,
                },
                Cell {
                    image: "1025469".into(),
                    path: PathBuf::from(format!("{corpus}/CID22/CID22-512/validation/1025469.png")),
                    distance: 4.0,
                    effort: 8,
                },
            ]
        }
    }

    pub fn run() {
        let cells = parse_cells();
        let perblock_dump = std::env::var("W44_112_PERBLOCK_DUMP").ok();

        // TSV header
        println!(
            "image\tdistance\teffort\twidth\theight\t\
             xsize_blocks\tysize_blocks\tnum_blocks\t\
             qf_exact_matches\tqf_off_by_one\tqf_diff_2_to_5\tqf_diff_gt_5\t\
             qf_max_abs_diff\tqf_mean_abs_diff\tqf_internal_mean\tqf_production_mean\t\
             int_global_scale\tprod_global_scale\tint_scale\tprod_scale\t\
             int_inv_scale\tprod_inv_scale\tbytes"
        );

        for cell in &cells {
            if !cell.path.exists() {
                eprintln!("MISSING {}: {}", cell.image, cell.path.display());
                continue;
            }

            let img = image::open(&cell.path).expect("open image");
            let (w, h) = (img.width() as usize, img.height() as usize);
            let rgb_u8: Vec<u8> = img.to_rgb8().into_raw();

            // Enable both hooks. The buttloop populates InternalRecon
            // (slot 1) at the final iter; the production transform_and_quantize
            // populates ProductionQf (slot 2) after AdjustQuantBlockAC has been
            // applied in parallel.
            let _ = __recon_hook::take_last();
            let _ = __recon_hook::take_last_production_qf();
            __recon_hook::set_capture_enabled(true);
            __recon_hook::set_production_qf_capture_enabled(true);

            let cfg = LossyConfig::new(cell.distance).with_effort(cell.effort);
            let bitstream = cfg
                .encode(&rgb_u8, w as u32, h as u32, PixelLayout::Rgb8)
                .expect("encode");

            __recon_hook::set_capture_enabled(false);
            __recon_hook::set_production_qf_capture_enabled(false);

            let recon = __recon_hook::take_last().expect("recon hook");
            let prod_qf = __recon_hook::take_last_production_qf().expect("production qf hook");

            assert_eq!(recon.xsize_blocks, prod_qf.xsize_blocks);
            assert_eq!(recon.ysize_blocks, prod_qf.ysize_blocks);
            let num_blocks = recon.xsize_blocks * recon.ysize_blocks;
            assert_eq!(recon.quant_field_u8.len(), num_blocks);
            assert_eq!(prod_qf.quant_field_u8.len(), num_blocks);

            // Per-block diff stats.
            let mut exact = 0usize;
            let mut off_one = 0usize;
            let mut diff_2_5 = 0usize;
            let mut diff_gt_5 = 0usize;
            let mut max_diff = 0i32;
            let mut sum_abs_diff = 0i64;
            let mut sum_int = 0u64;
            let mut sum_prod = 0u64;

            // Optional: per-block CSV dump for the per-cell drill-down.
            let mut perblock_writer = perblock_dump.as_ref().and_then(|prefix| {
                let p = format!(
                    "{prefix}_{}_{:.1}_e{}.csv",
                    cell.image, cell.distance, cell.effort
                );
                std::fs::File::create(&p).ok()
            });
            if let Some(w) = perblock_writer.as_mut() {
                let _ = writeln!(
                    w,
                    "bx,by,raw_strategy,is_first,internal_qf,production_qf,diff"
                );
            }

            for by in 0..recon.ysize_blocks {
                for bx in 0..recon.xsize_blocks {
                    let idx = by * recon.xsize_blocks + bx;
                    let int_qf = recon.quant_field_u8[idx] as i32;
                    let prod_qf_v = prod_qf.quant_field_u8[idx] as i32;
                    let diff = (int_qf - prod_qf_v).abs();
                    if diff == 0 {
                        exact += 1;
                    } else if diff == 1 {
                        off_one += 1;
                    } else if diff <= 5 {
                        diff_2_5 += 1;
                    } else {
                        diff_gt_5 += 1;
                    }
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    sum_abs_diff += diff as i64;
                    sum_int += int_qf as u64;
                    sum_prod += prod_qf_v as u64;

                    if let Some(w) = perblock_writer.as_mut() {
                        let _ = writeln!(
                            w,
                            "{},{},{},{},{},{},{}",
                            bx,
                            by,
                            recon.raw_strategy[idx],
                            recon.is_first_block[idx] as u8,
                            int_qf,
                            prod_qf_v,
                            int_qf - prod_qf_v
                        );
                    }
                }
            }

            let mean_abs_diff = sum_abs_diff as f64 / num_blocks as f64;
            let int_mean = sum_int as f64 / num_blocks as f64;
            let prod_mean = sum_prod as f64 / num_blocks as f64;

            // Both internal (buttloop final-iter `current_params`) and
            // production-side `DistanceParams` are captured. Per code
            // inspection (jxl-encoder/src/vardct/butteraugli_loop.rs:1525-1535)
            // `final_params = compute_from_quant_field(target_distance,
            // quant_field_float)` is recomputed AFTER the loop with the SAME
            // float field that the final iter used, so the two should always
            // match. A divergence here would be an outright bug; matching
            // values let us rule out candidate #1 (params drift between
            // buttloop and production).
            let int_global_scale = recon.final_global_scale;
            let prod_global_scale = prod_qf.global_scale;
            let int_scale = recon.final_scale;
            let prod_scale = prod_qf.scale;
            let int_inv_scale = recon.final_inv_scale;
            let prod_inv_scale = prod_qf.inv_scale;

            println!(
                "{}\t{:.2}\t{}\t{}\t{}\t\
                 {}\t{}\t{}\t\
                 {}\t{}\t{}\t{}\t\
                 {}\t{:.4}\t{:.2}\t{:.2}\t\
                 {}\t{}\t{:.6}\t{:.6}\t\
                 {:.6}\t{:.6}\t{}",
                cell.image,
                cell.distance,
                cell.effort,
                w,
                h,
                recon.xsize_blocks,
                recon.ysize_blocks,
                num_blocks,
                exact,
                off_one,
                diff_2_5,
                diff_gt_5,
                max_diff,
                mean_abs_diff,
                int_mean,
                prod_mean,
                int_global_scale,
                prod_global_scale,
                int_scale,
                prod_scale,
                int_inv_scale,
                prod_inv_scale,
                bitstream.len()
            );
        }
    }
}
