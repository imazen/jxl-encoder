//! W44-32 DCT64 entropy_mul sweep on F-D residual cells.
//!
//! Follow-on to W44-31 honest-stop (`cjxl_ledger_refresh_post_w44_3_w44_8_2026-05-19.md`).
//! W44-31 ruled out the EntropyMulTable's `(dct16, dct32)` levers at d=5/6 on
//! F-D residual cells: every candidate of the 4-tuple sweep produced LARGER bytes
//! than the W44-29 default. Item #2 in the recommended-next-chunks list was
//! "DCT64 entropy_mul investigation" — separate code path, separate constants.
//!
//! **Audit verdict**: libjxl's `entropy_mul64X64 = 2.25` and `entropy_mul64X32 = 2.25`
//! (`enc_ac_strategy.cc:896-897`) match our `EntropyMulTable::reference()` exactly.
//! No distance scaling in either path. The lever is at libjxl-parity values.
//!
//! However, the production encoder routes DCT64 entropy_mul through
//! `EntropyMulTable.dct64x64` / `dct64x32` via
//! `entropy_mul_for_strategy()` (vardct/ac_strategy.rs:715-716), so the same
//! override pattern as W44-31 (`LossyConfig::with_internal_params(...)`) lets
//! us empirically test whether lifting DCT64 from 2.25 → higher values shifts
//! bytes/quality on F-D residual cells.
//!
//! **Sweep design**: 2 F-D photos × 2 distances × 5 efforts × 4 DCT64 tuples
//! = 80 cells (plus 20 baseline). Each cell emits the strategy histogram so
//! we can verify whether DCT64 picks actually move with the entropy_mul lift.
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_32_dct64_sweep

#![allow(
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::too_many_arguments
)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{EntropyMulTable, LossyInternalParams};
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};

const F_D_IMAGES: &[(&str, &str)] = &[
    ("cid22/1531677", "CID22/CID22-512/validation/1531677.png"),
    ("cid22/1420710", "CID22/CID22-512/validation/1420710.png"),
];

const EFFORTS: &[u8] = &[5, 6, 7, 8, 9];
const DISTANCES: &[f32] = &[5.0, 6.0];

/// Candidate `(dct64x64, dct64x32)` tuples.
///
/// **Direction rationale**: W44-31's premise was "our DCT64 over-pick by +10%",
/// but the cited evidence (ledger column `jxl_ac_groups`) is BYTES-IN-AC-GROUPS
/// section, NOT block counts. We have no public-CLI mechanism to extract cjxl
/// AC-strategy distribution. So the W44-32 sweep instead lifts DCT64 entropy_mul
/// in BOTH directions and reads our own strategy_counts via EncodeStats.
///
/// libjxl reference = 2.25/2.25. Sweep both directions to find a local optimum
/// (if any exists for F-D content at d=5/6).
const CANDIDATES: &[(&str, f32, f32)] = &[
    ("libjxl_default", 2.25, 2.25),        // baseline = libjxl parity
    ("dct64_lift_25pct", 2.81, 2.81),      // discourage DCT64 (force more DCT32 and DCT16)
    ("dct64_lift_50pct", 3.375, 3.375),    // stronger discouragement
    ("dct64_lower_25pct", 1.69, 1.69),     // encourage DCT64 (force more 64-block picks)
    ("dct64_only_64x64_lift", 2.81, 2.25), // lift 64x64 only
];

fn build_table(dct64x64: f32, dct64x32: f32) -> EntropyMulTable {
    let mut t = EntropyMulTable::reference();
    // Keep the W44-29 high_d_photo_smooth_suppressed values for DCT16/DCT32
    // (those WIN on F-D content per the W44-29 acceptance gate).
    t.dct16x16 = 1.27;
    t.dct32x32 = 1.34;
    t.dct16x32 = 1.34 * (1.49 / 1.48);
    t.dct64x64 = dct64x64;
    t.dct64x32 = dct64x32;
    t
}

fn encode_with_table_and_stats(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    dct64x64: f32,
    dct64x32: f32,
) -> (Vec<u8>, [u32; 19]) {
    let table = build_table(dct64x64, dct64x32);
    let mut params = LossyInternalParams::default();
    params.entropy_mul_table = Some(table);
    // Force high_d_photo_hint OFF so the auto-gate doesn't OVERRIDE the
    // internal-params table at distance >= 4.0 (per W44-31 gotcha).
    let result = LossyConfig::new(d)
        .with_effort(effort)
        .with_strategy_overrides(jxl_encoder::api::StrategyOverrides {
            high_d_photo_hint: Some(false),
            ..Default::default()
        })
        .with_internal_params(params)
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode_with_stats(rgb_u8)
        .unwrap();
    let counts = *result.stats().strategy_counts();
    let bytes = result
        .data()
        .expect("encode_with_stats yielded no data")
        .to_vec();
    (bytes, counts)
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

fn format_strategy_counts(counts: &[u32; 19]) -> String {
    // Highlight DCT8 (0), DCT16x8 (1), DCT8x16 (2), DCT16x16 (3), DCT32x32 (4),
    // DCT32x16 (10), DCT16x32 (11), DCT64x64 (16), DCT64x32 (17), DCT32x64 (18).
    format!(
        "d8={} d16x8={} d8x16={} d16x16={} d32x32={} d16x32={} d32x16={} d64x64={} d64x32={} d32x64={}",
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        counts[4],
        counts[11],
        counts[10],
        counts[16],
        counts[17],
        counts[18]
    )
}

fn main() {
    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus")),
    );
    let out_path = PathBuf::from("benchmarks/w44_32_dct64_sweep_2026-05-18.tsv");
    if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let staging = PathBuf::from(format!("/tmp/w44_32_sweep_{}.tsv", std::process::id()));
    let mut out = std::fs::File::create(&staging).unwrap();
    writeln!(
        out,
        "image\teffort\tdistance\tcandidate\tdct64x64_mul\tdct64x32_mul\tbytes\tbase_bytes\tbfly\tbase_bfly\tssim2\tbase_ssim2\tbytes_delta_pct\tbfly_delta_pct\tssim2_delta_abs\tdct8\tdct16x8\tdct8x16\tdct16x16\tdct32x32\tdct16x32\tdct32x16\tdct64x64\tdct64x32\tdct32x64"
    )
    .unwrap();
    let bparams = ButteraugliParams::default();

    eprintln!("\n=== W44-32 DCT64 entropy_mul sweep on F-D residual cells ===");

    for &(label, rel) in F_D_IMAGES {
        let Some((w, h, rgb_u8, lin_pix, srgb_pix)) = load_image(&corpus, rel) else {
            continue;
        };
        let orig_lin = Img::new(lin_pix, w as usize, h as usize);
        let orig_srgb = Img::new(srgb_pix, w as usize, h as usize);

        for &effort in EFFORTS {
            for &d in DISTANCES {
                // Baseline = libjxl default (2.25, 2.25)
                let (b_dct64x64, b_dct64x32) = (CANDIDATES[0].1, CANDIDATES[0].2);
                let (baseline_bytes, baseline_counts) =
                    encode_with_table_and_stats(&rgb_u8, w, h, effort, d, b_dct64x64, b_dct64x32);
                let (bb, bbfly, bssim2) =
                    compute_metrics(&baseline_bytes, &orig_lin, &orig_srgb, &bparams);

                eprintln!(
                    "{:<20} e={} d={:.1}  baseline (libjxl 2.25/2.25): {} B  bfly={:.4}  ssim2={:.4}  | {}",
                    label,
                    effort,
                    d,
                    bb,
                    bbfly,
                    bssim2,
                    format_strategy_counts(&baseline_counts)
                );

                for &(cand_name, dct64x64, dct64x32) in CANDIDATES {
                    let (bytes, counts) =
                        encode_with_table_and_stats(&rgb_u8, w, h, effort, d, dct64x64, dct64x32);
                    let (b, bf, s) = compute_metrics(&bytes, &orig_lin, &orig_srgb, &bparams);
                    let bd_pct = (b as f64 - bb as f64) / bb as f64 * 100.0;
                    let bfd_pct = (bf - bbfly) / bbfly.max(1e-9) * 100.0;
                    let sd_abs = s - bssim2;
                    writeln!(
                        out,
                        "{}\t{}\t{:.3}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.4}\t{:+.4}\t{:+.4}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        label,
                        effort,
                        d,
                        cand_name,
                        dct64x64,
                        dct64x32,
                        b,
                        bb,
                        bf,
                        bbfly,
                        s,
                        bssim2,
                        bd_pct,
                        bfd_pct,
                        sd_abs,
                        counts[0], counts[1], counts[2], counts[3], counts[4],
                        counts[11], counts[10], counts[16], counts[17], counts[18]
                    )
                    .unwrap();
                    out.flush().ok();
                    let identical = if b == bb { " IDENTICAL" } else { "" };
                    eprintln!(
                        "  {:>22}: {} B  Δbytes={:+.3}% Δbfly={:+.3}% Δssim2={:+.4}{}  | {}",
                        cand_name,
                        b,
                        bd_pct,
                        bfd_pct,
                        sd_abs,
                        identical,
                        format_strategy_counts(&counts)
                    );
                }
            }
        }
    }

    drop(out);
    std::fs::rename(&staging, &out_path).unwrap();
    eprintln!("\nTSV: {}", out_path.display());
}
