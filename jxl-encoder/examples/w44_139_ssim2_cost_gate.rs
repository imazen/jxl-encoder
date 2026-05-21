// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-139 — SSIM2 cost gate measurement for the W44-138 Phase 1
//! photo-EPF divergence (+0.034 R max-abs at `after_epf` on
//! `1418519` d=5 e8 and similar smooth-photo cells at d>=3).
//!
//! Question: does the W44-138 per-stage divergence translate to a real
//! SSIM2/butteraugli quality cost in the SHIPPED bitstream, or is it
//! a numerical artifact invisible to perceptual metrics?
//!
//! Method: per-cell paired A/B
//!     A = `JXL_W44_117_DISABLE=1` → legacy uniform-4 EPF sharpness
//!         seed across all content (pre-W44-117 behaviour).
//!     B = production default (`AutoW44_117 { min_distance: 1.0 }`
//!         gated on `is_screenshot` via W44-118).
//!
//! Cell classes:
//!   1. Photo cells (W44-138 wedge) — W44-118 BLOCKS W44-117 → both
//!      modes byte-identical → SSIM2 delta MUST be 0. This is the
//!      "the +0.034 EPF divergence is not visible to production" anchor.
//!   2. Screenshot cells — W44-117 FIRES on B but not on A → measurable
//!      delta. This is the "what real-divergence-to-quality looks like"
//!      reference baseline.
//!
//! Verdict gate:
//!   * If photo SSIM2 delta is essentially zero (< 0.01) AND screenshot
//!     SSIM2 delta is substantial (>= 1.0, per W44-119 chain-disable
//!     baseline) → the +0.034 max-abs is real but ssim2-neutral on
//!     photos → honest-stop W44-138 Phase 2 P-1/P-2 root-cause arc.
//!   * If photo SSIM2 delta is materially non-zero → unexpected;
//!     re-investigate.
//!   * If we want to measure the SSIM2 impact of W44-117 firing ON
//!     photos (to evaluate P-2 admission), we'd need to bypass the
//!     W44-118 `is_screenshot` gate via a code patch — that is
//!     outside W44-139 scope (measurement-only chunk).
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!     --features 'butteraugli-loop ssim2-loop parallel' \
//!     --example w44_139_ssim2_cost_gate \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > benchmarks/w44_139_ssim2_cost_2026-05-20.tsv

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{Limits, LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::PathBuf;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82SC: &str = "/home/lilith/work/codec-corpus/gb82-sc";

/// (image, corpus, effort, distance, class).
/// Class = "photo" → W44-118 gate blocks W44-117 → A/B byte-identical;
///         "screen" → W44-117 fires in B → real measurable delta.
const CELLS: &[(&str, &str, u8, f32, &str)] = &[
    // PHOTO CLUSTER: W44-138 wedge cells where +0.034 R max-abs at
    // after_epf was measured. All blocked by W44-118 is_screenshot gate.
    ("1418519.png", "CID22", 8, 5.0, "photo"),
    ("1418519.png", "CID22", 8, 4.0, "photo"),
    ("1420710.png", "CID22", 8, 5.0, "photo"),
    ("1531677.png", "CID22", 8, 5.0, "photo"),
    // SCREENSHOT REFERENCE: W44-117 fires in B → measurable delta.
    // Per W44-119 chain-disable A/B baseline, terminal/codec_wiki d>=2
    // see -1.85 to -5.58 SSIM2 when the qac chain is removed; W44-117
    // alone closes 1-5 ssim2 points (per W44-138 meta).
    ("terminal.png", "gb82-sc", 8, 4.0, "screen"),
    ("codec_wiki.png", "gb82-sc", 8, 4.0, "screen"),
];

fn encode_shipped(rgb: &[u8], w: u32, h: u32, effort: u8, d: f32) -> Result<Vec<u8>, String> {
    // Lift the soft 2 GiB memory cap — codec_wiki e8 d=4 with the
    // butteraugli loop hits ~2.1 GiB working set (per W44-93 note).
    // 8 GiB is comfortable on the 128 GiB workstation; this is a
    // measurement harness, not production.
    let limits = Limits::new().with_max_memory_bytes(8 * 1024 * 1024 * 1024);
    LossyConfig::new(d)
        .with_effort(effort)
        .with_threads(8)
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(rgb)
        .map_err(|e| format!("encode failed: {e:?}"))
}

fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

fn srgb_u8_to_linear(rgb_u8: &[u8], w: u32, h: u32) -> Img<Vec<RGB<f32>>> {
    let lin: Vec<RGB<f32>> = rgb_u8
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear(c[0]),
                srgb_to_linear(c[1]),
                srgb_to_linear(c[2]),
            )
        })
        .collect();
    Img::new(lin, w as usize, h as usize)
}

#[derive(Clone, Copy, Default, Debug)]
struct Score {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
}

fn score_cell(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    d: f32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
) -> Option<Score> {
    let bitstream = match encode_shipped(rgb, w, h, effort, d) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  encode failed: {}", e);
            return None;
        }
    };
    let bytes = bitstream.len();

    let (dw, dh, decoded_linear) = decode_jxl_linear(&bitstream)?;
    if dw != w as usize || dh != h as usize {
        eprintln!("  decode dim mismatch: {}x{} vs {}x{}", dw, dh, w, h);
        return None;
    }

    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, dw, dh);
    let params = ButteraugliParams::default();
    let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), &params)
        .map(|r| r.score as f64)
        .unwrap_or(f64::NAN);

    let dec_srgb: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);

    Some(Score {
        bytes,
        butteraugli: bfly,
        ssim2,
    })
}

fn main() {
    eprintln!("W44-139 SSIM2 cost gate: A=JXL_W44_117_DISABLE=1 vs B=production default");
    eprintln!(
        "Cells: {} (4 photos with W44-118 gate blocking W44-117 + 2 screenshots where W44-117 fires)",
        CELLS.len()
    );
    eprintln!();

    println!(
        "image\tcorpus\teffort\tdistance\tclass\tA_bytes\tB_bytes\tbytes_delta_pct\tA_bfly\tB_bfly\tbfly_delta_pct\tA_ssim2\tB_ssim2\tssim2_delta_abs"
    );

    let mut images_cache: BTreeMap<
        String,
        (u32, u32, Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>),
    > = BTreeMap::new();

    let n_cells = CELLS.len();
    let mut cells_with_data = 0usize;
    let mut photo_ssim2_deltas: Vec<f64> = Vec::new();
    let mut screen_ssim2_deltas: Vec<f64> = Vec::new();

    for (i, &(image, corpus, effort, d, class)) in CELLS.iter().enumerate() {
        eprintln!(
            "[{}/{}] {} ({}) e{} d={} class={}",
            i + 1,
            n_cells,
            image,
            corpus,
            effort,
            d,
            class
        );

        let dir = match corpus {
            "CID22" => CID22,
            "gb82-sc" => GB82SC,
            _ => {
                eprintln!("  unknown corpus: {}", corpus);
                continue;
            }
        };
        let path = PathBuf::from(dir).join(image);
        if !path.exists() {
            eprintln!("  MISSING: {}", path.display());
            continue;
        }
        let cache_key = format!("{}/{}", corpus, image);

        let (w, h, raw, orig_linear_img, orig_srgb_img) =
            images_cache.entry(cache_key.clone()).or_insert_with(|| {
                let img = image::open(&path).expect("decode png");
                let (w, h) = img.dimensions();
                let rgb = img.to_rgb8().into_raw();
                let linear = srgb_u8_to_linear(&rgb, w, h);
                let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);
                (w, h, rgb, linear, srgb_img)
            });

        // A = legacy uniform-4 (pre-W44-117 behavior; force DISABLE).
        // SAFETY: single-threaded harness driver. The encoder uses an
        // internal rayon pool but this main loop is sequential between
        // A and B; the env var is read inside the encoder's setup
        // (`EncoderStrategy::resolve` env-var fallback layer) which
        // happens on the calling thread before any worker dispatch.
        unsafe { std::env::set_var("JXL_W44_117_DISABLE", "1") };
        let score_a = score_cell(raw, *w, *h, effort, d, orig_linear_img, orig_srgb_img);
        // SAFETY: same single-threaded harness driver as the set_var above.
        unsafe { std::env::remove_var("JXL_W44_117_DISABLE") };

        // B = production default (W44-117 gated on is_screenshot via W44-118)
        let score_b = score_cell(raw, *w, *h, effort, d, orig_linear_img, orig_srgb_img);

        match (score_a, score_b) {
            (Some(a), Some(b)) => {
                let bytes_delta_pct = 100.0 * (b.bytes as f64 - a.bytes as f64) / a.bytes as f64;
                let bfly_delta_pct = if a.butteraugli > 0.0 {
                    100.0 * (b.butteraugli - a.butteraugli) / a.butteraugli
                } else {
                    f64::NAN
                };
                let ssim2_delta_abs = b.ssim2 - a.ssim2;
                println!(
                    "{}\t{}\te{}\t{}\t{}\t{}\t{}\t{:+.3}\t{:.4}\t{:.4}\t{:+.3}\t{:.4}\t{:.4}\t{:+.4}",
                    image,
                    corpus,
                    effort,
                    d,
                    class,
                    a.bytes,
                    b.bytes,
                    bytes_delta_pct,
                    a.butteraugli,
                    b.butteraugli,
                    bfly_delta_pct,
                    a.ssim2,
                    b.ssim2,
                    ssim2_delta_abs,
                );
                match class {
                    "photo" => photo_ssim2_deltas.push(ssim2_delta_abs),
                    "screen" => screen_ssim2_deltas.push(ssim2_delta_abs),
                    _ => {}
                }
                cells_with_data += 1;
            }
            _ => {
                eprintln!("  one or both scores failed; skipping");
            }
        }
    }

    eprintln!();
    eprintln!("=== W44-139 VERDICT ===");
    eprintln!("Cells with data: {} / {}", cells_with_data, n_cells);

    let photo_mean = if photo_ssim2_deltas.is_empty() {
        f64::NAN
    } else {
        photo_ssim2_deltas.iter().sum::<f64>() / photo_ssim2_deltas.len() as f64
    };
    let photo_max_abs = photo_ssim2_deltas
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max);
    let screen_mean = if screen_ssim2_deltas.is_empty() {
        f64::NAN
    } else {
        screen_ssim2_deltas.iter().sum::<f64>() / screen_ssim2_deltas.len() as f64
    };
    let screen_max_abs = screen_ssim2_deltas
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max);

    eprintln!(
        "Photos (W44-118 BLOCKS W44-117): n={}, mean ssim2 Δ(B-A)={:+.4}, max|Δ|={:.4}",
        photo_ssim2_deltas.len(),
        photo_mean,
        photo_max_abs,
    );
    eprintln!(
        "Screens (W44-117 FIRES on B):   n={}, mean ssim2 Δ(B-A)={:+.4}, max|Δ|={:.4}",
        screen_ssim2_deltas.len(),
        screen_mean,
        screen_max_abs,
    );
    eprintln!();

    // Decision gate per the W44-139 task spec:
    //
    // Photo gate (W44-138 Phase 2 root-cause arc decision):
    //   - photo max|Δ| < 0.01 (noise floor): byte-identical AND
    //     ssim2-identical → W44-138 +0.034 R max-abs at after_epf is
    //     STRUCTURALLY INVISIBLE to production. The W44-118 is_screenshot
    //     gate already blocks W44-117 firing on photos, so there is no
    //     code path in which the divergence matters. → HONEST-STOP P-1
    //     (per-iter recompute) and P-2 (mask<50 admission) as currently
    //     specified, because both presuppose forcing W44-117 to fire
    //     somewhere on photos. To actually evaluate P-2 we would need a
    //     code patch to bypass the W44-118 gate (out of W44-139 scope).
    //
    //   - photo max|Δ| >= 0.1: unexpected (would contradict W44-138
    //     finding that A == B byte-identical on photos). Re-investigate
    //     the env-var fallback layer + EpfSharpnessSeed resolution.
    //
    // Screen baseline (metric internal-consistency check):
    //   - screen max|Δ| >= 0.5: meaningful SSIM2 movement when W44-117
    //     actually fires; confirms metric is sensitive enough to detect
    //     real EPF-stage changes. terminal d=4 e8 should land here
    //     (W44-138 meta: W44-117 saves 0.6% bytes on this cell).
    //
    //   - screen max|Δ| < 0.5: W44-117's contribution to SSIM2 alone
    //     (separated from the qac chain per W44-119) is smaller than
    //     expected. Not a measurement bug — just a calibration note.
    let photo_invisible = photo_max_abs < 0.01;
    let photo_actionable = photo_max_abs >= 0.1;
    let screen_baseline_meaningful = screen_max_abs >= 0.5;

    if photo_invisible && screen_baseline_meaningful {
        eprintln!(
            "VERDICT: HONEST-STOP W44-138 Phase 2 P-1/P-2 (root-cause arc) as currently specified"
        );
        eprintln!(
            "  - Photo max|Δssim2| < 0.01 → +0.034 R max-abs at after_epf is NOT visible in production"
        );
        eprintln!(
            "  - Screen baseline max|Δssim2| >= 0.5 → metric internally consistent ({} ssim2 pts on terminal)",
            screen_max_abs as f32
        );
        eprintln!();
        eprintln!("  Reasoning:");
        eprintln!("    The W44-138 +0.034 R max-abs is REAL (verified by per-stage capture) but");
        eprintln!("    occurs in a code path (buttloop EPF stage) that is GATED OFF on photos via");
        eprintln!("    the W44-118 `is_screenshot` predicate. Both A (LegacyUniform4) and B");
        eprintln!("    (AutoW44_117 default) take the same code path on photos → byte-identical →");
        eprintln!("    SSIM2-identical. The divergence has no production failure mode.");
        eprintln!();
        eprintln!("  Recommended action: update docs/LIBJXL_DIVERGENCES.md Section F to note");
        eprintln!("    'recon-vs-decoder +0.034 R max-abs at post-EPF on smooth photos at d>=3 is");
        eprintln!("    real but ssim2-neutral in production; no fix planned. To revisit, scope a");
        eprintln!("    separate chunk that bypasses the W44-118 is_screenshot gate AND validates");
        eprintln!("    no regression on the 1025469-class photos that W44-118 was designed to'");
        eprintln!("    protect.");
    } else if photo_actionable && screen_baseline_meaningful {
        eprintln!("VERDICT: PROCEED to W44-138 Phase 2 (P-1 or P-2)");
        eprintln!("  - Photo max|Δssim2| >= 0.1 → divergence has actionable quality cost");
        eprintln!(
            "  - Investigate why photo A != B (would contradict W44-138 byte-identical finding)"
        );
    } else if !screen_baseline_meaningful {
        eprintln!("VERDICT: SCREEN BASELINE LOW — max|Δssim2| < 0.5, calibration note only");
        eprintln!("  W44-117 alone (separated from qac chain) produces smaller deltas than the");
        eprintln!("  full-chain disable baseline (-1.85 to -5.58 SSIM2 per W44-119).");
        eprintln!("  Photo verdict still applies: see photo numbers above.");
    } else {
        eprintln!(
            "VERDICT: AMBIGUOUS — photo max|Δ| in [0.01, 0.1) band, screen baseline {}",
            if screen_baseline_meaningful {
                "OK"
            } else {
                "low"
            }
        );
    }
}
