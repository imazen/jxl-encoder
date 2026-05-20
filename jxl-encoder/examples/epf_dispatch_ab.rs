//! W36-2 — A/B sweep for adaptive EPF sharpness dispatch.
//!
//! W36-1 (`70a48af9`) phase baseline:
//!   - e6 (176 ms mean): EPF sharpness 45.5% — single dominant phase
//!   - e7 (248 ms mean): EPF sharpness 33.8%, patches 25.2%
//!
//! `compute_epf_sharpness` runs a two-pass per-block search to pick
//! the EPF sharpness LUT index per 8x8 block. On smooth content the
//! search converges on the default sharpness value (4) almost
//! everywhere, so the work is wasted — we get the same bitstream
//! emitting the uniform default directly.
//!
//! This harness sweeps 10 images (5 CID22 photos + 5 GB82-SC
//! screenshots) × 3 distances {0.5, 1.0, 2.0} × 3 efforts {6, 7, 8}
//! × 3 dispatch modes {`AlwaysSelect`, `Auto`, `AlwaysDefault`} =
//! 270 cells. Per cell: wall-clock (3-run median), bytes,
//! butteraugli (Rust `butteraugli_linear` on `jxl-oxide` linear
//! decode — CLAUDE.md PNG-metadata-immune path), and SSIM2
//! (`fast_ssim2`).
//!
//! Output: `benchmarks/epf_dispatch_e6_e7_2026-05-18.tsv` (+ .meta)
//!
//! Reproducer:
//!   cargo run -p jxl-encoder --release --features 'std parallel butteraugli-loop' \
//!     --example epf_dispatch_ab -- \
//!     --out benchmarks/epf_dispatch_e6_e7_2026-05-18.tsv

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use jxl_encoder::{EpfDispatch, LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Convert linear light value to sRGB u8 using the correct sRGB
/// transfer function (NOT gamma 2.2). Matches
/// `tests/quality_compare.rs::linear_to_srgb_u8`.
fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0];
const EFFORTS: &[u8] = &[6, 7, 8];
const RUNS_PER_CELL: usize = 3;

fn dispatch_label(d: EpfDispatch) -> &'static str {
    match d {
        EpfDispatch::AlwaysSelect => "always_select",
        EpfDispatch::Auto => "auto",
        EpfDispatch::AlwaysDefault => "always_default",
    }
}

fn all_dispatches() -> [EpfDispatch; 3] {
    [
        EpfDispatch::AlwaysSelect,
        EpfDispatch::Auto,
        EpfDispatch::AlwaysDefault,
    ]
}

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

#[derive(Debug, Clone, Copy)]
struct Measure {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
    encode_ms: f64,
}

fn measure_cell(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    d: f32,
    effort: u8,
    dispatch: EpfDispatch,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> Result<Measure, String> {
    let cfg = LossyConfig::new(d)
        .with_effort(effort)
        .with_strategy(jxl_encoder::api::EncoderStrategy::Custom(Box::new(jxl_encoder::api::EncoderImprovementsCustom { epf_dispatch: dispatch, ..Default::default() })));

    let t0 = Instant::now();
    let bytes = cfg
        .encode(rgb_u8, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))?;
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

    // SSIM2 wants sRGB u8 inputs. Convert decoded linear → sRGB u8.
    let dec_srgb_pixels: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img = Img::new(dec_srgb_pixels, dw, dh);
    let s2 = fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);

    Ok(Measure {
        bytes: bytes.len(),
        butteraugli: bfly,
        ssim2: s2,
        encode_ms,
    })
}

fn median<T: Copy + PartialOrd>(mut v: Vec<T>) -> T {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut out_path = PathBuf::from("benchmarks/epf_dispatch_e6_e7_2026-05-18.tsv");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--out" && i + 1 < args.len() {
            out_path = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }

    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let cid_dir = format!("{}/work/codec-corpus/CID22/CID22-512/validation", base);
    let scrn_dir = format!("{}/work/codec-corpus/gb82-sc", base);

    let cid_names: &[&str] = &[
        "1025469.png",
        "1044329.png",
        "1189261.png",
        "1279330.png",
        "1418519.png",
    ];
    let scrn_names: &[&str] = &[
        "codec_wiki.png",
        "imac_g3.png",
        "terminal.png",
        "imac_dark.png",
        "windows.png",
    ];

    let mut sources: Vec<Source> = Vec::new();
    for n in cid_names {
        sources.push(Source {
            label: n,
            class: "photo",
            path: PathBuf::from(format!("{}/{}", cid_dir, n)),
        });
    }
    for n in scrn_names {
        sources.push(Source {
            label: n,
            class: "screen",
            path: PathBuf::from(format!("{}/{}", scrn_dir, n)),
        });
    }

    // Filter to images that exist on disk
    sources.retain(|s| s.path.exists());

    let dispatches = all_dispatches();
    let total_cells = sources.len() * EFFORTS.len() * DISTANCES.len() * dispatches.len();
    eprintln!(
        "Sweep: {} images × {} efforts × {} distances × {} dispatches × {} runs = {} encodes",
        sources.len(),
        EFFORTS.len(),
        DISTANCES.len(),
        dispatches.len(),
        RUNS_PER_CELL,
        total_cells * RUNS_PER_CELL
    );

    let mut header = String::from(
        "image\tclass\teffort\tdistance\tdispatch\twidth\theight\tbytes\tbutteraugli\tssim2\tencode_ms_med\tencode_ms_min",
    );
    let mut rows: Vec<String> = Vec::new();
    rows.push(header.clone());
    let _ = &mut header;

    let tmp = format!("/tmp/epf_dispatch_ab_{}.tsv.partial", std::process::id());
    {
        let mut f = fs::File::create(&tmp).expect("create tmp");
        writeln!(f, "{}", rows[0]).unwrap();
    }

    let params = ButteraugliParams::default();
    let t_start = Instant::now();
    let mut cell_idx = 0usize;

    for src in &sources {
        let img = match image::open(&src.path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("skip {}: {e}", src.path.display());
                continue;
            }
        };
        let (w, h) = (img.width(), img.height());
        let rgb = img.to_rgb8();
        let rgb_bytes = rgb.as_raw().clone();

        // Build linear reference (sRGB → linear) for butteraugli.
        let lin_pixels: Vec<RGB<f32>> = rgb_bytes
            .chunks(3)
            .map(|c| {
                RGB::new(
                    srgb_to_linear(c[0]),
                    srgb_to_linear(c[1]),
                    srgb_to_linear(c[2]),
                )
            })
            .collect();
        let orig_linear_img = Img::new(lin_pixels, w as usize, h as usize);
        // sRGB u8 reference for SSIM2.
        let srgb_pixels: Vec<[u8; 3]> = rgb_bytes.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let orig_srgb_img = Img::new(srgb_pixels, w as usize, h as usize);

        for &effort in EFFORTS {
            for &d in DISTANCES {
                for &dispatch in &dispatches {
                    cell_idx += 1;
                    eprintln!(
                        "[{}/{}] {} {} e{} d={} dispatch={}",
                        cell_idx,
                        total_cells,
                        src.class,
                        src.label,
                        effort,
                        d,
                        dispatch_label(dispatch)
                    );
                    let mut runs: Vec<Measure> = Vec::with_capacity(RUNS_PER_CELL);
                    for _ in 0..RUNS_PER_CELL {
                        match measure_cell(
                            &rgb_bytes,
                            w,
                            h,
                            d,
                            effort,
                            dispatch,
                            &orig_linear_img,
                            &orig_srgb_img,
                            &params,
                        ) {
                            Ok(m) => runs.push(m),
                            Err(e) => eprintln!("  err: {e}"),
                        }
                    }
                    if runs.is_empty() {
                        continue;
                    }

                    let bytes = median(runs.iter().map(|r| r.bytes as u64).collect::<Vec<_>>());
                    let bfly = median(runs.iter().map(|r| r.butteraugli).collect::<Vec<_>>());
                    let s2 = median(runs.iter().map(|r| r.ssim2).collect::<Vec<_>>());
                    let ms_med = median(runs.iter().map(|r| r.encode_ms).collect::<Vec<_>>());
                    let ms_min = runs
                        .iter()
                        .map(|r| r.encode_ms)
                        .fold(f64::INFINITY, f64::min);

                    let row = format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.2}\t{:.2}",
                        src.label,
                        src.class,
                        effort,
                        d,
                        dispatch_label(dispatch),
                        w,
                        h,
                        bytes,
                        bfly,
                        s2,
                        ms_med,
                        ms_min,
                    );
                    rows.push(row.clone());
                    // Incremental flush.
                    let mut f = fs::OpenOptions::new().append(true).open(&tmp).unwrap();
                    writeln!(f, "{}", row).unwrap();
                }
            }
        }
    }

    let dur = t_start.elapsed();
    eprintln!(
        "Sweep done in {:.1}s, wrote {} rows",
        dur.as_secs_f64(),
        rows.len() - 1
    );

    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::rename(&tmp, &out_path).expect("rename tmp → out");
    eprintln!("Wrote {}", out_path.display());
}
