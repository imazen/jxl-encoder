//! W44-31 per-distance `high_d_photo_hint` value sweep at d=5 and d=6.
//!
//! Per W44-30 follow-up (`cjxl_ledger_refresh_post_w44_3_w44_8_2026-05-19.md`
//! Post-W44-29 section): the W44-29 `(dct16=1.27, dct32=1.34)` tuple closed
//! 1531677.png at d=4 across e5..e9 (5 OPEN→FIXED) but the same content at
//! d=5 and d=6 stays OPEN. Hypothesis: same smooth-photo regime, just needs
//! more aggressive entropy_mul lowering at higher distances.
//!
//! Sweeps 4 candidate `(dct16, dct32)` tuples on the 2 OPEN F-D photos
//! (1531677.png, 1420710.png) at d∈{5, 6} and effort 5..=9, paired against
//! the W44-29 default (`dct16=1.27, dct32=1.34`) baseline. Writes a TSV
//! with byte / bfly / ssim2 deltas vs the W44-29 default for each (image,
//! effort, distance, tuple) cell.
//!
//! The winning tuple per-distance becomes the seed value for the
//! production `high_d_photo_smooth_suppressed_for_distance(d)` selector
//! (or whatever the parametrization shape ends up).
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_31_per_distance_sweep
//!
//! Output: `benchmarks/w44_31_per_distance_sweep_2026-05-18.tsv`.

#![allow(
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::too_many_arguments
)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::effort::{EntropyMulTable, LossyInternalParams};
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};

/// Sweep cells: 2 F-D photos × 2 distances × 5 efforts × 5 candidate tuples
/// = 100 encodes (plus 20 baseline). Should run in ~3-5 min on the 7950X.
const F_D_IMAGES: &[(&str, &str)] = &[
    ("cid22/1531677", "CID22/CID22-512/validation/1531677.png"),
    ("cid22/1420710", "CID22/CID22-512/validation/1420710.png"),
];

const EFFORTS: &[u8] = &[5, 6, 7, 8, 9];
const DISTANCES: &[f32] = &[5.0, 6.0];

/// Candidate `(dct16x16, dct32x32)` tuples. Per W44-30 recommendation
/// + AC strategy distribution analysis in `cjxl_parity_ledger_2026-05-18.tsv`:
///
/// At d=5/6 on these images, OUR encoder picks **more** DCT32 (1050–1189
/// vs cjxl 624–905) AND **fewer** DCT16 (4096–4377 vs cjxl 4684–5126)
/// compared to cjxl. To re-balance the distribution closer to cjxl we
/// want to:
///   - LOWER dct16 entropy_mul (make DCT16 cheaper to encourage it)
///   - RAISE dct32 entropy_mul (make DCT32 more expensive to discourage it)
///
/// This is the OPPOSITE direction from the initial W44-31 hypothesis
/// (which assumed monotonic-lower-as-distance-rises). The W44-30 memo
/// suggested "looser dct32 at d=5/6" but the ledger evidence flips that:
/// our dct32 is already TOO loose (over-picked vs cjxl), so tightening
/// it should help.
const CANDIDATES: &[(&str, f32, f32)] = &[
    ("w44_29_default", 1.27, 1.34), // baseline = current ship-state via auto-gate
    ("dct32_libjxl_ref", 1.27, 1.48), // dct32 at libjxl reference
    ("dct32_higher", 1.27, 1.62),   // dct32 above libjxl ref
    ("both_dir_a", 1.20, 1.48),     // lower dct16 + libjxl-ref dct32
    ("both_dir_b", 1.15, 1.62),     // most aggressive in re-balance direction
];

/// Screenshot regression panel — verify lowered tuples still produce
/// byte-identical screenshots (the auto-gate suppresses them by mask1x1
/// threshold, but a direct entropy_mul_table override bypasses the gate,
/// so we set `screenshot_lift_hint = Some(false)` AND `high_d_photo_hint
/// = Some(false)` to mimic the auto-gate's suppression behavior on
/// screenshots before applying the override).
const SCREEN_CELLS: &[(&str, &str, u8, f32)] = &[
    ("gb82/imac_g3", "gb82-sc/imac_g3.png", 7, 4.0),
    ("gb82/imac_g3", "gb82-sc/imac_g3.png", 7, 6.0),
    ("gb82/terminal", "gb82-sc/terminal.png", 7, 5.0),
];

fn build_table(dct16: f32, dct32: f32) -> EntropyMulTable {
    let mut t = EntropyMulTable::reference();
    t.dct16x16 = dct16;
    t.dct32x32 = dct32;
    // Match the production `high_d_photo_smooth_suppressed` table:
    // scale DCT16X32/DCT32X16 with DCT32X32 by the libjxl 1.49/1.48 ratio.
    t.dct16x32 = dct32 * (1.49 / 1.48);
    t
}

fn encode_with_table(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    dct16: f32,
    dct32: f32,
) -> Vec<u8> {
    let table = build_table(dct16, dct32);
    let mut params = LossyInternalParams::default();
    params.entropy_mul_table = Some(table);
    // CRITICAL: the W44-29 auto-gate OVERRIDES profile.entropy_mul_table
    // when (distance >= 4.0 && median(mask1x1) < 50.0) — see
    // `vardct/encoder.rs:2200-2226`. Without forcing the hint OFF, every
    // candidate in this sweep would receive the hardcoded
    // `high_d_photo_smooth_suppressed()` table (dct16=1.27, dct32=1.34)
    // and the internal-params override would be silently discarded,
    // producing byte-identical output across all sweep cells.
    //
    // Force `Some(false)` to suppress the auto-gate so the internal-params
    // table propagates to `compute_ac_strategy`. The screenshot panel below
    // also benefits — `with_high_d_photo_hint(Some(false))` is mutually
    // exclusive with `w22_1_lift` (which is opt-in via
    // `content_aware_entropy_mul=true`, not default), so the W22-1 gate
    // doesn't fire either.
    LossyConfig::new(d)
        .with_effort(effort)
        .with_high_d_photo_hint(Some(false))
        .with_internal_params(params)
        .encode(rgb_u8, w, h, PixelLayout::Rgb8)
        .unwrap()
}

fn decode_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn linear_to_srgb_u8(v: f32) -> u8 {
    let c = v.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

fn compute_metrics(
    bytes: &[u8],
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    bparams: &ButteraugliParams,
) -> (usize, f64, f64) {
    let (dw, dh, dec) = decode_linear(bytes).unwrap();
    let dec_pix: Vec<RGB<f32>> = dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
    let dec_lin = Img::new(dec_pix, dw, dh);
    let bfly = butteraugli_linear(orig_lin.as_ref(), dec_lin.as_ref(), bparams)
        .map(|r| r.score as f64)
        .unwrap_or(f64::NAN);
    let dec_srgb: Vec<[u8; 3]> = dec
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let ssim2 =
        fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), Img::new(dec_srgb, dw, dh).as_ref())
            .unwrap_or(f64::NAN);
    (bytes.len(), bfly, ssim2)
}

fn load_image(
    corpus: &Path,
    rel: &str,
) -> Option<(u32, u32, Vec<u8>, Vec<RGB<f32>>, Vec<[u8; 3]>)> {
    let path = corpus.join(rel);
    if !path.exists() {
        eprintln!("MISS {}", path.display());
        return None;
    }
    let img = image::open(&path).ok()?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let rgb_u8: Vec<u8> = rgb.as_raw().clone();
    let linear: Vec<RGB<f32>> = rgb
        .pixels()
        .map(|p| {
            RGB::new(
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
            )
        })
        .collect();
    let srgb_arr: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
    Some((w, h, rgb_u8, linear, srgb_arr))
}

fn main() {
    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus")),
    );
    let out_path = PathBuf::from("benchmarks/w44_31_per_distance_sweep_2026-05-18.tsv");
    if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let staging = PathBuf::from(format!("/tmp/w44_31_sweep_{}.tsv", std::process::id()));
    let mut out = std::fs::File::create(&staging).unwrap();
    writeln!(
        out,
        "class\timage\teffort\tdistance\tcandidate\tdct16\tdct32\tbytes\tbase_bytes\tbfly\tbase_bfly\tssim2\tbase_ssim2\tbytes_delta_pct\tbfly_delta_pct\tssim2_delta_abs"
    )
    .unwrap();
    let bparams = ButteraugliParams::default();

    // ─── F-D photo sweep ────────────────────────────────────────────────
    eprintln!("\n=== F-D photos ===");
    for &(label, rel) in F_D_IMAGES {
        let Some((w, h, rgb_u8, lin_pix, srgb_pix)) = load_image(&corpus, rel) else {
            continue;
        };
        let orig_lin = Img::new(lin_pix, w as usize, h as usize);
        let orig_srgb = Img::new(srgb_pix, w as usize, h as usize);

        for &effort in EFFORTS {
            for &d in DISTANCES {
                // Baseline = W44-29 default (dct16=1.27, dct32=1.34) — first candidate.
                let (b_dct16, b_dct32) = (CANDIDATES[0].1, CANDIDATES[0].2);
                let baseline_bytes = encode_with_table(&rgb_u8, w, h, effort, d, b_dct16, b_dct32);
                let (bb, bbfly, bssim2) =
                    compute_metrics(&baseline_bytes, &orig_lin, &orig_srgb, &bparams);

                eprintln!(
                    "{:<20} e={} d={:.1}  baseline (W44-29 default): {} B  bfly={:.4}  ssim2={:.4}",
                    label, effort, d, bb, bbfly, bssim2
                );

                for &(cand_name, dct16, dct32) in CANDIDATES {
                    let bytes = encode_with_table(&rgb_u8, w, h, effort, d, dct16, dct32);
                    let (b, bf, s) = compute_metrics(&bytes, &orig_lin, &orig_srgb, &bparams);
                    let bd_pct = (b as f64 - bb as f64) / bb as f64 * 100.0;
                    let bfd_pct = (bf - bbfly) / bbfly.max(1e-9) * 100.0;
                    let sd_abs = s - bssim2;
                    writeln!(
                        out,
                        "FD_PHOTO\t{}\t{}\t{:.3}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.4}\t{:+.4}\t{:+.4}",
                        label,
                        effort,
                        d,
                        cand_name,
                        dct16,
                        dct32,
                        b,
                        bb,
                        bf,
                        bbfly,
                        s,
                        bssim2,
                        bd_pct,
                        bfd_pct,
                        sd_abs
                    )
                    .unwrap();
                    out.flush().ok();
                    let identical = if b == bb { " IDENTICAL" } else { "" };
                    eprintln!(
                        "  {:>15}: {} B  Δbytes={:+.3}% Δbfly={:+.3}% Δssim2={:+.4}{}",
                        cand_name, b, bd_pct, bfd_pct, sd_abs, identical
                    );
                }
            }
        }
    }

    // ─── Screenshot regression panel ────────────────────────────────────
    // For each screenshot cell, baseline = W44-29 default with `screenshot_lift_hint`
    // active (auto-gate would fire on imac_g3/terminal because their mask1x1
    // medians exceed 95). The W44-29 default entropy_mul_table is applied
    // when screenshot_lift_hint = false. Then sweep alternate tuples and
    // confirm bytes stay within tight tolerance (<= 3% drift).
    eprintln!("\n=== Screenshot regression panel ===");
    for &(label, rel, effort, d) in SCREEN_CELLS {
        let Some((w, h, rgb_u8, lin_pix, srgb_pix)) = load_image(&corpus, rel) else {
            continue;
        };
        let orig_lin = Img::new(lin_pix, w as usize, h as usize);
        let orig_srgb = Img::new(srgb_pix, w as usize, h as usize);

        // Baseline = W44-29 default
        let (b_dct16, b_dct32) = (CANDIDATES[0].1, CANDIDATES[0].2);
        let baseline_bytes = encode_with_table(&rgb_u8, w, h, effort, d, b_dct16, b_dct32);
        let (bb, bbfly, bssim2) = compute_metrics(&baseline_bytes, &orig_lin, &orig_srgb, &bparams);
        eprintln!(
            "{:<20} e={} d={:.1}  baseline (W44-29 default direct override): {} B  bfly={:.4}  ssim2={:.4}",
            label, effort, d, bb, bbfly, bssim2
        );
        for &(cand_name, dct16, dct32) in CANDIDATES {
            let bytes = encode_with_table(&rgb_u8, w, h, effort, d, dct16, dct32);
            let (b, bf, s) = compute_metrics(&bytes, &orig_lin, &orig_srgb, &bparams);
            let bd_pct = (b as f64 - bb as f64) / bb as f64 * 100.0;
            let bfd_pct = (bf - bbfly) / bbfly.max(1e-9) * 100.0;
            let sd_abs = s - bssim2;
            writeln!(
                out,
                "SCREENSHOT\t{}\t{}\t{:.3}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.4}\t{:+.4}\t{:+.4}",
                label,
                effort,
                d,
                cand_name,
                dct16,
                dct32,
                b,
                bb,
                bf,
                bbfly,
                s,
                bssim2,
                bd_pct,
                bfd_pct,
                sd_abs
            )
            .unwrap();
            out.flush().ok();
            let identical = if b == bb { " IDENTICAL" } else { "" };
            eprintln!(
                "  {:>15}: {} B  Δbytes={:+.3}% Δbfly={:+.3}% Δssim2={:+.4}{}",
                cand_name, b, bd_pct, bfd_pct, sd_abs, identical
            );
        }
    }

    drop(out);
    std::fs::rename(&staging, &out_path).unwrap();
    eprintln!("\nTSV: {}", out_path.display());
}
