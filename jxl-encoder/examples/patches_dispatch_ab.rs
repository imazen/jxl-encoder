//! Patches photo-skip dispatch A/B harness (W36-3).
//!
//! Per-image paired A/B for [`LossyConfig::with_patches_dispatch`]:
//! `AlwaysScan` (pre-W36-3 behavior — patches scan runs every image)
//! vs `Auto` (default — patches scan skipped when
//! `median(mask1x1) <= 95`, i.e. photo content). The expectation is:
//!
//! * **Photo** sources: `Auto` is identical bytes to `AlwaysScan`
//!   (patches scan would have produced empty output anyway) but
//!   ~25-30 ms/MP faster.
//! * **Screenshot** sources: `Auto` is byte-identical to `AlwaysScan`
//!   AND essentially the same wall clock (the dispatch gate fires,
//!   patches scan still runs).
//!
//! Methodology mirrors `content_aware_entropy_mul_ab.rs` /
//! `quality_drift_investigation.rs`:
//! - Decode both with jxl-oxide in linear sRGB (CLAUDE.md jxl-oxide rule)
//! - Measure Rust butteraugli + Rust fast-ssim2 (avoid the
//!   `butteraugli_main` PNG-metadata bug documented in CLAUDE.md)
//! - 5 paired iterations per cell, report median wall-clock so warm-up
//!   jitter washes out.
//!
//! Run:
//!   cargo run -p jxl-encoder --release --features 'std parallel butteraugli-loop' \
//!     --example patches_dispatch_ab \
//!     [-- <output.tsv>]
//!
//! Default output: `benchmarks/patches_dispatch_e7_2026-05-18.tsv`
//! (the path the W36-3 task expects).
//!
//! Cells: 5 CID22-512 photos × 5 gb82-sc screenshots × distances
//! {0.5, 1.0, 2.0} × {AlwaysScan, Auto} = 60 paired cells (effort 7 only
//! — that's where the patches scan is 25.2% of wall-clock per W36-1).

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{LossyConfig, PatchesDispatch, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0];
const EFFORT: u8 = 7;
const REPEATS: usize = 5;

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
    dispatch: PatchesDispatch,
) -> Result<Vec<u8>, String> {
    let cfg = LossyConfig::new(d).with_effort(EFFORT).with_strategy(
        jxl_encoder::api::EncoderStrategy::Custom(Box::new(
            jxl_encoder::api::EncoderImprovementsCustom {
                patches_dispatch: dispatch,
                ..Default::default()
            },
        )),
    );
    cfg.encode(rgb_u8, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))
}

#[derive(Debug, Clone, Copy)]
struct Measure {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
    encode_ms_median: f64,
    encode_ms_best: f64,
}

fn median_f64(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

#[allow(clippy::too_many_arguments)]
fn measure(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    d: f32,
    dispatch: PatchesDispatch,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> Result<Measure, String> {
    let mut times: Vec<f64> = Vec::with_capacity(REPEATS);
    let mut last_bytes: Option<Vec<u8>> = None;
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        let bytes = encode(rgb_u8, w, h, d, dispatch)?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        times.push(ms);
        last_bytes = Some(bytes);
    }
    let bytes = last_bytes.unwrap();

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

    let best = times.iter().copied().fold(f64::INFINITY, f64::min);
    let median = median_f64(&mut times);

    Ok(Measure {
        bytes: bytes.len(),
        butteraugli: bfly,
        ssim2,
        encode_ms_median: median,
        encode_ms_best: best,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_path = args
        .into_iter()
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("benchmarks/patches_dispatch_e7_2026-05-18.tsv"));

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

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    // Stage TSV at /tmp first, atomic-mv at end (per
    // jj_concurrent_workspace_failure_mode_2026-05-18.md).
    let staging = PathBuf::from(format!(
        "/tmp/patches_dispatch_ab_{}.tsv",
        std::process::id()
    ));
    let mut out = std::fs::File::create(&staging).expect("create staging tsv");
    use std::io::Write;
    writeln!(
        out,
        "image\tclass\tw\th\tdistance\tmode\tbytes\tbutteraugli\tssim2\tencode_ms_median\tencode_ms_best\tbytes_delta_pct\tbfly_delta_pct\tssim2_delta_abs\tms_delta_median\tms_delta_best\tms_per_mp_median_delta"
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
        let mp = (w as f64 * h as f64) / 1_000_000.0;
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

        eprintln!(
            "=== {} ({}, {}x{}, {:.2} MP) ===",
            src.label, src.class, w, h, mp
        );

        for &d in DISTANCES {
            // AlwaysScan: pre-W36-3 baseline behavior — run the patches
            // detector unconditionally.
            let m_always = match measure(
                rgb_u8,
                w,
                h,
                d,
                PatchesDispatch::AlwaysScan,
                &orig_linear_img,
                &orig_srgb_img,
                &params,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  AlwaysScan d={d}: {e}");
                    continue;
                }
            };
            // Auto: skip the scan when median(mask1x1) <= 95 (photo-class).
            let m_auto = match measure(
                rgb_u8,
                w,
                h,
                d,
                PatchesDispatch::Auto,
                &orig_linear_img,
                &orig_srgb_img,
                &params,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  Auto d={d}: {e}");
                    continue;
                }
            };

            let bytes_delta_pct =
                (m_auto.bytes as f64 - m_always.bytes as f64) / m_always.bytes as f64 * 100.0;
            let bfly_delta_pct = (m_auto.butteraugli - m_always.butteraugli)
                / m_always.butteraugli.max(1e-9)
                * 100.0;
            let ssim2_delta_abs = m_auto.ssim2 - m_always.ssim2;
            let ms_delta_median = m_auto.encode_ms_median - m_always.encode_ms_median;
            let ms_delta_best = m_auto.encode_ms_best - m_always.encode_ms_best;
            let ms_per_mp_median_delta = ms_delta_median / mp.max(1e-9);

            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{:.3}\talways\t{}\t{:.4}\t{:.4}\t{:.2}\t{:.2}\t\t\t\t\t\t",
                src.label,
                src.class,
                w,
                h,
                d,
                m_always.bytes,
                m_always.butteraugli,
                m_always.ssim2,
                m_always.encode_ms_median,
                m_always.encode_ms_best,
            )
            .unwrap();
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{:.3}\tauto\t{}\t{:.4}\t{:.4}\t{:.2}\t{:.2}\t{:+.4}\t{:+.4}\t{:+.4}\t{:+.2}\t{:+.2}\t{:+.2}",
                src.label,
                src.class,
                w,
                h,
                d,
                m_auto.bytes,
                m_auto.butteraugli,
                m_auto.ssim2,
                m_auto.encode_ms_median,
                m_auto.encode_ms_best,
                bytes_delta_pct,
                bfly_delta_pct,
                ssim2_delta_abs,
                ms_delta_median,
                ms_delta_best,
                ms_per_mp_median_delta
            )
            .unwrap();
            out.flush().unwrap();
            eprintln!(
                "  d={d} ALWAYS: {}B bfly={:.4} ssim2={:.3} med={:.0}ms best={:.0}ms",
                m_always.bytes,
                m_always.butteraugli,
                m_always.ssim2,
                m_always.encode_ms_median,
                m_always.encode_ms_best,
            );
            eprintln!(
                "       AUTO  : {}B bfly={:.4} ssim2={:.3} med={:.0}ms best={:.0}ms  Δbytes={:+.3}% Δmed={:+.1}ms ({:+.1}ms/MP)",
                m_auto.bytes,
                m_auto.butteraugli,
                m_auto.ssim2,
                m_auto.encode_ms_median,
                m_auto.encode_ms_best,
                bytes_delta_pct,
                ms_delta_median,
                ms_per_mp_median_delta,
            );
        }
    }

    drop(out);
    // Atomic mv (per CLAUDE.md / jj_concurrent_workspace_failure_mode_2026-05-18.md).
    std::fs::rename(&staging, &out_path).expect("atomic mv staging → out_path");
    eprintln!("\nTSV: {}", out_path.display());
}
