//! W22-1 chunk-2 follow-on: CPU re-bisect of lifted `entropy_mul` values
//! for screen-content. W22-1 lifted IDENTITY=1.85, DCT2X2=1.15, AFV=0.95,
//! DCT4X8=0.98 (GPU-bisected). On the CPU encoder the screen-class
//! result was bytes Δ +0.7 % (range −1.27..+6.88 %), butteraugli swings up
//! to +20 %. The lift constants need to be smaller.
//!
//! This harness sweeps smaller IDENTITY × DCT2X2 lifts (AFV / DCT4X8 left
//! at the W22-1 values, one knob at a time) on 5 gb82-sc screenshots at
//! d∈{0.5, 1.0, 2.0}, then validates the winning tuple on 5 CID22-512
//! photos to confirm photo-class is byte-identical.
//!
//! Per-cell: paired off (reference table) vs on (lifted tuple via
//! `LossyInternalParams::entropy_mul_table`), butteraugli via
//! `jxl-oxide` linear decode + Rust `butteraugli_linear` (CLAUDE.md
//! PNG-metadata immune path). Stage-A TSV is written incrementally
//! (per-image flush) to a staging /tmp file, atomic-renamed at end
//! (per `jj_concurrent_workspace_failure_mode_2026-05-18.md`).
//!
//! Run:
//!   cargo run -p jxl-encoder --release --features '__expert butteraugli-loop' \
//!     --example cpu_entropy_mul_bisect [-- <out.tsv>]
//!
//! Default output: `benchmarks/cpu_entropy_mul_bisect_2026-05-18.tsv`.
//!
//! Cells:
//!   Stage A (screen): 5 screenshots × 5 IDENTITY × 3 DCT2X2 × 3 distances
//!     × {off, on} = 450 measurements. (Baseline `off` measured once per
//!     (image, distance) and reused — see `BaselineCache` below.)
//!   Stage B (photo validation): 5 photos × 1 winning tuple × 3 distances
//!     × {off, on} = 30 measurements.

#![allow(clippy::too_many_arguments, clippy::field_reassign_with_default)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::effort::{EntropyMulTable, LossyInternalParams};
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::HashMap;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0];
const EFFORT: u8 = 7;

// Lifted-value sweep grids. AFV + DCT4X8 are pinned at the W22-1 lifted
// values so we change one knob axis per stage. After chunk-2 picks a
// winning (IDENTITY, DCT2X2) we revisit AFV / DCT4X8 in chunk-3.
const IDENTITY_SWEEP: &[f32] = &[1.20, 1.30, 1.40, 1.50, 1.60];
const DCT2X2_SWEEP: &[f32] = &[0.95, 0.9975, 1.045]; // libjxl base + 5 % + 10 %
const AFV_PINNED: f32 = 0.95; // W22-1
const DCT4X8_PINNED: f32 = 0.98; // W22-1

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

fn build_entropy_table(identity: f32, dct2x2: f32) -> EntropyMulTable {
    let mut t = EntropyMulTable::reference();
    t.identity = identity;
    t.dct2x2 = dct2x2;
    t.afv = AFV_PINNED;
    t.dct4x8 = DCT4X8_PINNED;
    t
}

fn encode_baseline(rgb_u8: &[u8], w: u32, h: u32, d: f32) -> Result<Vec<u8>, String> {
    LossyConfig::new(d)
        .with_effort(EFFORT)
        .encode(rgb_u8, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("baseline encode failed: {e:?}"))
}

fn encode_lifted(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    d: f32,
    identity: f32,
    dct2x2: f32,
) -> Result<Vec<u8>, String> {
    let table = build_entropy_table(identity, dct2x2);
    let mut params = LossyInternalParams::default();
    params.entropy_mul_table = Some(table);
    LossyConfig::new(d)
        .with_effort(EFFORT)
        .with_internal_params(params)
        .encode(rgb_u8, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("lifted encode failed: {e:?}"))
}

#[derive(Debug, Clone, Copy)]
struct Measure {
    bytes: usize,
    butteraugli: f64,
    encode_ms: f64,
}

fn measure_bytes_and_bfly(
    bytes: Vec<u8>,
    w: u32,
    h: u32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    params: &ButteraugliParams,
    encode_ms: f64,
) -> Result<Measure, String> {
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
    Ok(Measure {
        bytes: bytes.len(),
        butteraugli: bfly,
        encode_ms,
    })
}

fn measure_baseline(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    d: f32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    params: &ButteraugliParams,
) -> Result<Measure, String> {
    let t0 = Instant::now();
    let bytes = encode_baseline(rgb_u8, w, h, d)?;
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    measure_bytes_and_bfly(bytes, w, h, orig_linear_img, params, encode_ms)
}

fn measure_lifted(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    d: f32,
    identity: f32,
    dct2x2: f32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    params: &ButteraugliParams,
) -> Result<Measure, String> {
    let t0 = Instant::now();
    let bytes = encode_lifted(rgb_u8, w, h, d, identity, dct2x2)?;
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    measure_bytes_and_bfly(bytes, w, h, orig_linear_img, params, encode_ms)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_path = args
        .into_iter()
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("benchmarks/cpu_entropy_mul_bisect_2026-05-18.tsv"));

    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus")),
    );

    let screen_sources = vec![
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

    let photo_sources = vec![
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
    ];

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let staging = PathBuf::from(format!(
        "/tmp/cpu_entropy_mul_bisect_{}.tsv",
        std::process::id()
    ));
    let mut out = std::fs::File::create(&staging).expect("create staging tsv");
    writeln!(
        out,
        "stage\timage\tclass\tdistance\tidentity\tdct2x2\tafv\tdct4x8\tbaseline_bytes\tlifted_bytes\tbaseline_bfly\tlifted_bfly\tbytes_delta_pct\tbfly_delta_pct\tbaseline_ms\tlifted_ms"
    )
    .unwrap();

    let bparams = ButteraugliParams::default();

    // ─── Stage A: screen-class IDENTITY × DCT2X2 sweep ──────────────────
    eprintln!("=== Stage A: screen sweep ===");
    eprintln!(
        "  grid: IDENTITY ∈ {:?}, DCT2X2 ∈ {:?}, distances ∈ {:?}",
        IDENTITY_SWEEP, DCT2X2_SWEEP, DISTANCES
    );
    eprintln!(
        "  AFV pinned at {}, DCT4X8 pinned at {} (W22-1 values)",
        AFV_PINNED, DCT4X8_PINNED
    );
    // Per-image baseline cache: (distance, image_label) → Measure
    let mut baselines: HashMap<(String, u32), Measure> = HashMap::new();

    for src in &screen_sources {
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

        eprintln!("--- {} ({}x{}) ---", src.label, w, h);

        for &d in DISTANCES {
            // Baseline once per (image, distance) — encoder is deterministic.
            let baseline = match measure_baseline(rgb_u8, w, h, d, &orig_linear_img, &bparams) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  baseline d={d}: {e}");
                    continue;
                }
            };
            let dkey = (d * 1000.0).round() as u32;
            baselines.insert((src.label.to_string(), dkey), baseline);
            eprintln!(
                "  baseline d={d}: {}B bfly={:.4} {:.0}ms",
                baseline.bytes, baseline.butteraugli, baseline.encode_ms
            );

            for &ident in IDENTITY_SWEEP {
                for &dct2 in DCT2X2_SWEEP {
                    let lifted = match measure_lifted(
                        rgb_u8,
                        w,
                        h,
                        d,
                        ident,
                        dct2,
                        &orig_linear_img,
                        &bparams,
                    ) {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("  ident={ident} dct2={dct2} d={d}: {e}");
                            continue;
                        }
                    };
                    let bytes_delta_pct = (lifted.bytes as f64 - baseline.bytes as f64)
                        / baseline.bytes as f64
                        * 100.0;
                    let bfly_delta_pct = (lifted.butteraugli - baseline.butteraugli)
                        / baseline.butteraugli.max(1e-9)
                        * 100.0;
                    writeln!(
                        out,
                        "A\t{}\t{}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:+.3}\t{:+.3}\t{:.2}\t{:.2}",
                        src.label,
                        src.class,
                        d,
                        ident,
                        dct2,
                        AFV_PINNED,
                        DCT4X8_PINNED,
                        baseline.bytes,
                        lifted.bytes,
                        baseline.butteraugli,
                        lifted.butteraugli,
                        bytes_delta_pct,
                        bfly_delta_pct,
                        baseline.encode_ms,
                        lifted.encode_ms
                    )
                    .unwrap();
                    out.flush().unwrap();
                    eprintln!(
                        "    ident={:.2} dct2={:.4} → Δbytes={:+.2}% Δbfly={:+.2}% ({}B vs {}B)",
                        ident, dct2, bytes_delta_pct, bfly_delta_pct, lifted.bytes, baseline.bytes
                    );
                }
            }
        }
    }

    // ─── Stage B: photo validation (bit-identical check) ─────────────────
    //
    // We need to know the winning tuple *before* running stage B. Since
    // the harness streams TSV rows, post-pick logic happens off-process
    // (review TSV, pick tuple, rerun with --photo-validation env var).
    // For now stage B runs the W22-1 tuple as a sanity baseline so the
    // TSV always has photo data; the actual chunk-2 picker re-runs this
    // example with the winning tuple set via env vars.
    let photo_ident = std::env::var("CPU_ENT_PHOTO_IDENT")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.85); // W22-1 default
    let photo_dct2 = std::env::var("CPU_ENT_PHOTO_DCT2")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.15); // W22-1 default

    eprintln!("\n=== Stage B: photo validation ===");
    eprintln!(
        "  tuple: IDENTITY={} DCT2X2={} (override via CPU_ENT_PHOTO_IDENT / CPU_ENT_PHOTO_DCT2)",
        photo_ident, photo_dct2
    );

    for src in &photo_sources {
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

        eprintln!("--- {} ({}x{}) ---", src.label, w, h);
        for &d in DISTANCES {
            let baseline = match measure_baseline(rgb_u8, w, h, d, &orig_linear_img, &bparams) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  baseline d={d}: {e}");
                    continue;
                }
            };
            let lifted = match measure_lifted(
                rgb_u8,
                w,
                h,
                d,
                photo_ident,
                photo_dct2,
                &orig_linear_img,
                &bparams,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  lifted d={d}: {e}");
                    continue;
                }
            };
            let bytes_delta_pct =
                (lifted.bytes as f64 - baseline.bytes as f64) / baseline.bytes as f64 * 100.0;
            let bfly_delta_pct = (lifted.butteraugli - baseline.butteraugli)
                / baseline.butteraugli.max(1e-9)
                * 100.0;
            writeln!(
                out,
                "B\t{}\t{}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:+.3}\t{:+.3}\t{:.2}\t{:.2}",
                src.label,
                src.class,
                d,
                photo_ident,
                photo_dct2,
                AFV_PINNED,
                DCT4X8_PINNED,
                baseline.bytes,
                lifted.bytes,
                baseline.butteraugli,
                lifted.butteraugli,
                bytes_delta_pct,
                bfly_delta_pct,
                baseline.encode_ms,
                lifted.encode_ms
            )
            .unwrap();
            out.flush().unwrap();
            eprintln!(
                "  d={d}: Δbytes={:+.3}% Δbfly={:+.3}% ({}B vs {}B)",
                bytes_delta_pct, bfly_delta_pct, lifted.bytes, baseline.bytes
            );
        }
    }

    drop(out);
    std::fs::rename(&staging, &out_path).expect("atomic mv staging → out_path");
    eprintln!("\nTSV: {}", out_path.display());
}
