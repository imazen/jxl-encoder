//! Zenjxl-vs-Libjxl per-image routing pilot (issue #74 "win or match always").
//!
//! The scoreboard's cjxl-dominant LOSSY cells are the biggest slice of the
//! "match always" gap. This harness measures, per (image, distance), whether
//! our production `EncoderStrategy::Zenjxl` content-aware lifts REGRESS vs our
//! own `EncoderStrategy::Libjxl` parity bundle — i.e. whether a strategy-level
//! zenanalyze router (route to Libjxl where our lift hurts) has any target.
//!
//! Classification per cell (Libjxl as baseline B, Zenjxl as Z):
//!   - Z_DOMINATES : Z bytes ≤ B AND Z ssim2 ≥ B AND Z butteraugli ≤ B (keep Z)
//!   - B_DOMINATES : the reverse (our lift strictly hurt → ROUTER TARGET)
//!   - TRADEOFF    : neither dominates (Pareto-incomparable)
//!   - EQUAL       : within noise on all axes (no lift fired)
//!
//! Metrics follow the CLAUDE.md rules: decode both with jxl-oxide in LINEAR
//! sRGB, Rust butteraugli + Rust fast-ssim2 (immune to the butteraugli_main
//! PNG-metadata bug). Smaller butteraugli = better; larger ssim2 = better.
//!
//! Run:
//!   cargo run -p jxl-encoder --release --features butteraugli-loop \
//!     --example zenjxl_vs_libjxl_routing_pilot [-- <output.tsv>]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 3.0, 5.0];
const EFFORT: u8 = 7;
// Noise floors below which a per-axis difference is "equal".
const BYTES_EPS_PCT: f64 = 0.10;
const SSIM2_EPS: f64 = 0.10;
const BFLY_EPS_PCT: f64 = 0.50;

struct Source {
    label: &'static str,
    class: &'static str,
    path: PathBuf,
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

fn encode(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    d: f32,
    strategy: EncoderStrategy,
) -> Result<Vec<u8>, String> {
    LossyConfig::new(d)
        .with_effort(EFFORT)
        .with_strategy(strategy)
        .encode(rgb_u8, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))
}

#[derive(Debug, Clone, Copy)]
struct Measure {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
    encode_ms: f64,
}

#[allow(clippy::too_many_arguments)]
fn measure(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    d: f32,
    strategy: EncoderStrategy,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> Result<Measure, String> {
    let t0 = Instant::now();
    let bytes = encode(rgb_u8, w, h, d, strategy)?;
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let (dw, dh, decoded_linear) =
        decode_jxl_linear(&bytes).ok_or_else(|| "decode failed".to_string())?;
    if dw != w as usize || dh != h as usize {
        return Err(format!("decoded {dw}x{dh} ≠ {w}x{h}"));
    }

    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), params)
        .map(|r| r.score)
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

    Ok(Measure {
        bytes: bytes.len(),
        butteraugli: bfly,
        ssim2,
        encode_ms,
    })
}

/// Classify Zenjxl (Z) vs Libjxl baseline (B). Smaller bytes/butteraugli
/// better; larger ssim2 better.
fn classify(z: &Measure, b: &Measure) -> &'static str {
    let bytes_pct = 100.0 * (z.bytes as f64 - b.bytes as f64) / b.bytes as f64;
    let bfly_pct = 100.0 * (z.butteraugli - b.butteraugli) / b.butteraugli.max(1e-6);
    let ssim2_abs = z.ssim2 - b.ssim2;

    let bytes_better = bytes_pct < -BYTES_EPS_PCT;
    let bytes_worse = bytes_pct > BYTES_EPS_PCT;
    let bfly_better = bfly_pct < -BFLY_EPS_PCT;
    let bfly_worse = bfly_pct > BFLY_EPS_PCT;
    let ssim_better = ssim2_abs > SSIM2_EPS;
    let ssim_worse = ssim2_abs < -SSIM2_EPS;

    let z_any_worse = bytes_worse || bfly_worse || ssim_worse;
    let z_any_better = bytes_better || bfly_better || ssim_better;

    if !z_any_worse && !z_any_better {
        "EQUAL"
    } else if !z_any_worse {
        "Z_DOMINATES"
    } else if !z_any_better {
        "B_DOMINATES" // our lift strictly hurt → router target
    } else {
        "TRADEOFF"
    }
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("benchmarks/zenjxl_vs_libjxl_routing_pilot_2026-07-14.tsv")
        });

    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus")),
    );

    let sources = vec![
        Source { label: "cid22/1418519", class: "photo", path: corpus.join("CID22/CID22-512/validation/1418519.png") },
        Source { label: "cid22/1531677", class: "photo", path: corpus.join("CID22/CID22-512/validation/1531677.png") },
        Source { label: "cid22/159550", class: "photo", path: corpus.join("CID22/CID22-512/validation/159550.png") },
        Source { label: "clic/11f2b039", class: "photo", path: corpus.join("clic2025-1024/11f2b039b293758398b1a7a8afa64bb2.png") },
        Source { label: "gb82/codec_wiki", class: "screen", path: corpus.join("gb82-sc/codec_wiki.png") },
        Source { label: "gb82/imac_dark", class: "screen", path: corpus.join("gb82-sc/imac_dark.png") },
        Source { label: "gb82/terminal", class: "screen", path: corpus.join("gb82-sc/terminal.png") },
        Source { label: "gb82/windows95", class: "screen", path: corpus.join("gb82-sc/windows95.png") },
        Source { label: "imazen26/web-loc", class: "screen", path: corpus.join("imazen-26/8100-lilith-web-screenshots/1920x1080/8174_web-screenshots_loc-free-to-use_dpr1_page1_1920x1080.png") },
        Source { label: "imazen26/doc-lee", class: "doc", path: corpus.join("imazen-26/5300-noaa-hurricane-documents/5334_noaa_nhc-al132023-lee_p21_2550x3300.png") },
    ];

    let staging = PathBuf::from(format!(
        "/tmp/zenjxl_vs_libjxl_routing_pilot_{}.tsv",
        std::process::id()
    ));
    let mut out = std::fs::File::create(&staging).expect("create staging tsv");
    use std::io::Write;
    writeln!(out, "image\tclass\tdistance\tlibjxl_bytes\tzenjxl_bytes\tbytes_delta_pct\tlibjxl_bfly\tzenjxl_bfly\tbfly_delta_pct\tlibjxl_ssim2\tzenjxl_ssim2\tssim2_delta_abs\tverdict").unwrap();

    let params = ButteraugliParams::default();
    let mut counts = std::collections::BTreeMap::<&str, u32>::new();

    for src in &sources {
        if !src.path.exists() {
            eprintln!("MISS {}", src.path.display());
            continue;
        }
        let img = match image::open(&src.path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("open {}: {e}", src.path.display());
                continue;
            }
        };
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8();
        let rgb_u8: &[u8] = rgb.as_raw();

        let linear_rgb: Vec<RGB<f32>> = rgb
            .pixels()
            .map(|p| {
                RGB::new(
                    srgb_to_linear(p[0]),
                    srgb_to_linear(p[1]),
                    srgb_to_linear(p[2]),
                )
            })
            .collect();
        let orig_linear_img = Img::new(linear_rgb, w as usize, h as usize);
        let orig_srgb_pixels: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        let orig_srgb_img = Img::new(orig_srgb_pixels, w as usize, h as usize);

        eprintln!("=== {} ({}, {}x{}) ===", src.label, src.class, w, h);

        for &d in DISTANCES {
            let b = match measure(
                rgb_u8,
                w,
                h,
                d,
                EncoderStrategy::Libjxl,
                &orig_linear_img,
                &orig_srgb_img,
                &params,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  d={d} libjxl: {e}");
                    continue;
                }
            };
            let z = match measure(
                rgb_u8,
                w,
                h,
                d,
                EncoderStrategy::Zenjxl,
                &orig_linear_img,
                &orig_srgb_img,
                &params,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  d={d} zenjxl: {e}");
                    continue;
                }
            };
            let verdict = classify(&z, &b);
            *counts.entry(verdict).or_insert(0) += 1;
            let bytes_pct = 100.0 * (z.bytes as f64 - b.bytes as f64) / b.bytes as f64;
            let bfly_pct = 100.0 * (z.butteraugli - b.butteraugli) / b.butteraugli.max(1e-6);
            let ssim_abs = z.ssim2 - b.ssim2;
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{}\t{:+.2}\t{:.3}\t{:.3}\t{:+.2}\t{:.3}\t{:.3}\t{:+.3}\t{}",
                src.label,
                src.class,
                d,
                b.bytes,
                z.bytes,
                bytes_pct,
                b.butteraugli,
                z.butteraugli,
                bfly_pct,
                b.ssim2,
                z.ssim2,
                ssim_abs,
                verdict
            )
            .unwrap();
            eprintln!(
                "  d={d}: {verdict}  bytes {:+.1}%  bfly {:+.1}%  ssim2 {:+.2}  (enc B={:.0}ms Z={:.0}ms)",
                bytes_pct, bfly_pct, ssim_abs, b.encode_ms, z.encode_ms
            );
        }
    }

    drop(out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::rename(&staging, &out_path)
        .or_else(|_| std::fs::copy(&staging, &out_path).map(|_| ()))
        .ok();
    eprintln!("\n=== VERDICT COUNTS ===");
    for (k, v) in &counts {
        eprintln!("  {k}: {v}");
    }
    eprintln!("wrote {}", out_path.display());
}
