//! Content-aware `entropy_mul` table A/B harness.
//!
//! Per-image paired A/B for `LossyConfig::with_content_aware_entropy_mul(on)`
//! against the same encode without the gate. The gate fires when the
//! per-encode `median(mask1x1)` exceeds 95 (screen / glyph / UI content).
//!
//! Methodology mirrors `quality_drift_investigation.rs`:
//! - Decode both with jxl-oxide in linear sRGB (CLAUDE.md jxl-oxide rule)
//! - Measure Rust butteraugli + Rust fast-ssim2 (avoid the
//!   `butteraugli_main` PNG-metadata bug documented in CLAUDE.md)
//!
//! Run:
//!   cargo run -p jxl-encoder --release --example content_aware_entropy_mul_ab
//!     [-- <output.tsv>]
//!
//! Default output: `benchmarks/content_aware_entropy_mul_2026-05-18.tsv`
//! (the path the audit memory expects).
//!
//! Cells: 5 CID22-512 photos × 5 gb82-sc screenshots × distances
//! {0.5, 1.0, 2.0} × {off, on} = 60 cells (effort 7 only — that's where
//! the cost-model lives and where the GPU-mirror lifted values were
//! bisected).

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0];
const EFFORT: u8 = 7;

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

fn encode(rgb_u8: &[u8], w: u32, h: u32, d: f32, content_aware: bool) -> Result<Vec<u8>, String> {
    // W44-130 Chunk D: `with_content_aware_entropy_mul` deleted —
    // policy enum subsumes the enable bit. `content_aware=true` →
    // `ScreenshotEntropyMulPolicy::Auto` (the mask1x1 auto-detector
    // matches the pre-Chunk-D enable-bit-on path);
    // `content_aware=false` → the Zenjxl default `Disabled` (matches
    // pre-Chunk-D enable-bit-off path).
    let mut custom = jxl_encoder::api::EncoderImprovementsCustom::default();
    if content_aware {
        custom.screenshot_entropy_mul = jxl_encoder::api::ScreenshotEntropyMulPolicy::Auto;
    }
    let cfg = LossyConfig::new(d)
        .with_effort(EFFORT)
        .with_strategy(jxl_encoder::api::EncoderStrategy::Custom(Box::new(custom)));
    cfg.encode(rgb_u8, w, h, PixelLayout::Rgb8)
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
    content_aware: bool,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> Result<Measure, String> {
    let t0 = Instant::now();
    let bytes = encode(rgb_u8, w, h, d, content_aware)?;
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let (dw, dh, decoded_linear) =
        decode_jxl_linear(&bytes).ok_or_else(|| "decode failed".to_string())?;
    if dw != w as usize || dh != h as usize {
        return Err(format!("decoded {}x{} ≠ {}x{}", dw, dh, w, h));
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_path = args
        .into_iter()
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("benchmarks/content_aware_entropy_mul_2026-05-18.tsv"));

    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus")),
    );

    // 5 photos × 5 screenshots
    let sources = vec![
        // CID22-512 photos (small detail-heavy validation set)
        Source {
            label: "cid22/1025469",
            class: "photo",
            path: corpus.join("CID22/CID22-512/validation/1025469.png"),
        },
        Source {
            label: "cid22/1189261",
            class: "photo",
            path: corpus.join("CID22/CID22-512/validation/1189261.png"),
        },
        Source {
            label: "cid22/1418519",
            class: "photo",
            path: corpus.join("CID22/CID22-512/validation/1418519.png"),
        },
        Source {
            label: "cid22/1531677",
            class: "photo",
            path: corpus.join("CID22/CID22-512/validation/1531677.png"),
        },
        Source {
            label: "cid22/159550",
            class: "photo",
            path: corpus.join("CID22/CID22-512/validation/159550.png"),
        },
        // gb82-sc screenshots (UI / terminal / glyph-heavy)
        Source {
            label: "gb82/codec_wiki",
            class: "screen",
            path: corpus.join("gb82-sc/codec_wiki.png"),
        },
        Source {
            label: "gb82/imac_dark",
            class: "screen",
            path: corpus.join("gb82-sc/imac_dark.png"),
        },
        Source {
            label: "gb82/imac_g3",
            class: "screen",
            path: corpus.join("gb82-sc/imac_g3.png"),
        },
        Source {
            label: "gb82/terminal",
            class: "screen",
            path: corpus.join("gb82-sc/terminal.png"),
        },
        Source {
            label: "gb82/windows95",
            class: "screen",
            path: corpus.join("gb82-sc/windows95.png"),
        },
    ];

    // Open output file. Writes one header row + per-cell rows.
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    // Stage TSV at /tmp first, atomic-mv at end (per the
    // bench-vs-jj failure-mode memory).
    let staging = PathBuf::from(format!(
        "/tmp/content_aware_entropy_mul_ab_{}.tsv",
        std::process::id()
    ));
    let mut out = std::fs::File::create(&staging).expect("create staging tsv");
    use std::io::Write;
    writeln!(
        out,
        "image\tclass\tdistance\tmode\tbytes\tbutteraugli\tssim2\tencode_ms\tbytes_delta_pct\tbfly_delta_pct\tssim2_delta_abs"
    )
    .unwrap();

    let params = ButteraugliParams::default();

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

        // Pre-build linear + sRGB references for measure().
        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                [
                    srgb_to_linear(p[0]),
                    srgb_to_linear(p[1]),
                    srgb_to_linear(p[2]),
                ]
            })
            .collect();
        let orig_pixels: Vec<RGB<f32>> = linear_rgb
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let orig_linear_img = Img::new(orig_pixels, w as usize, h as usize);
        let orig_srgb_pixels: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        let orig_srgb_img = Img::new(orig_srgb_pixels, w as usize, h as usize);

        eprintln!("=== {} ({}, {}x{}) ===", src.label, src.class, w, h);

        for &d in DISTANCES {
            let m_off = match measure(
                rgb_u8,
                w,
                h,
                d,
                false,
                &orig_linear_img,
                &orig_srgb_img,
                &params,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  OFF d={d}: {e}");
                    continue;
                }
            };
            let m_on = match measure(
                rgb_u8,
                w,
                h,
                d,
                true,
                &orig_linear_img,
                &orig_srgb_img,
                &params,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  ON  d={d}: {e}");
                    continue;
                }
            };

            let bytes_delta_pct =
                (m_on.bytes as f64 - m_off.bytes as f64) / m_off.bytes as f64 * 100.0;
            let bfly_delta_pct =
                (m_on.butteraugli - m_off.butteraugli) / m_off.butteraugli.max(1e-9) * 100.0;
            let ssim2_delta_abs = m_on.ssim2 - m_off.ssim2;

            writeln!(
                out,
                "{}\t{}\t{:.3}\toff\t{}\t{:.4}\t{:.4}\t{:.2}\t\t\t",
                src.label,
                src.class,
                d,
                m_off.bytes,
                m_off.butteraugli,
                m_off.ssim2,
                m_off.encode_ms,
            )
            .unwrap();
            writeln!(
                out,
                "{}\t{}\t{:.3}\ton\t{}\t{:.4}\t{:.4}\t{:.2}\t{:+.3}\t{:+.3}\t{:+.4}",
                src.label,
                src.class,
                d,
                m_on.bytes,
                m_on.butteraugli,
                m_on.ssim2,
                m_on.encode_ms,
                bytes_delta_pct,
                bfly_delta_pct,
                ssim2_delta_abs
            )
            .unwrap();
            out.flush().unwrap();
            eprintln!(
                "  d={d} OFF: {}B bfly={:.4} ssim2={:.4} {:.0}ms",
                m_off.bytes, m_off.butteraugli, m_off.ssim2, m_off.encode_ms,
            );
            eprintln!(
                "       ON: {}B bfly={:.4} ssim2={:.4} {:.0}ms  Δbytes={:+.2}% Δbfly={:+.2}% Δssim2={:+.3}",
                m_on.bytes,
                m_on.butteraugli,
                m_on.ssim2,
                m_on.encode_ms,
                bytes_delta_pct,
                bfly_delta_pct,
                ssim2_delta_abs
            );
        }
    }

    drop(out);
    // Atomic mv (per CLAUDE.md / jj_concurrent_workspace_failure_mode_2026-05-18.md).
    std::fs::rename(&staging, &out_path).expect("atomic mv staging → out_path");
    eprintln!("\nTSV: {}", out_path.display());
}
