//! W35-2 chunk-4: safe-class re-bisect of lifted `entropy_mul` values for
//! screen-content, EXCLUDING windows95 (the palette-4 outlier identified
//! by W35-1 chunk-1 as the smart-dispatch chunk-1 honest-stop class).
//!
//! W35-1 finding (chunk-1, 2026-05-18):
//!   The hint API (`with_screenshot_lift_hint`) correctly suppresses the
//!   W22-1 lift on windows95 (palette_log2_size = 4), but the W22-1
//!   default lift tuple (IDENT=1.85, DCT2X2=1.15) is broadly too
//!   aggressive on EVERY screenshot, not just windows95. `graph` d=0.5
//!   is +94 % bfly. No classifier rule can rescue an inherently-bad tuple.
//!
//! This chunk-4 harness drops windows95 from the bisect corpus and
//! re-runs on the 9 plog2 ≥ 7 screenshots, sweeping LOWER lift values
//! than the W23-2 stage-A grid:
//!
//!   IDENTITY ∈ {1.10, 1.15, 1.20, 1.25, 1.30}  (W23-2 used 1.20..1.60)
//!   DCT2X2   ∈ {1.04, 1.07, 1.10, 1.13}        (W23-2 used 0.95, 0.9975, 1.045)
//!
//! AFV + DCT4X8 stay at the W22-1 lifted values (0.95, 0.98) — one knob
//! axis at a time, following the same discipline as W23-2 stage A.
//!
//! Pass criteria for ship decision:
//!   1. avg screen-class bytes Δ  ≤ -0.30 %
//!   2. no individual cell shows |bfly Δ| > 3 %
//!   3. ≥ 80 % of cells show |bfly Δ| ≤ 2 %
//!
//! Photo validation (5 CID22-512 photos × winning tuple): MUST be
//! byte-identical (the `palette_log2_size > 6` gate excludes photos
//! anyway, but we measure to confirm the wiring stays bit-stable).
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert butteraugli-loop parallel' \
//!     --example entropy_mul_safe_class_bisect [-- <out.tsv>]
//!
//! Default output: `benchmarks/entropy_mul_safe_class_bisect_2026-05-18.tsv`.
//!
//! Cells:
//!   Stage A (screen): 9 screenshots × 5 IDENTITY × 4 DCT2X2 × 3 distances
//!     × {off, on} = 540 measurements + 27 baselines = 567 encodes.
//!   Stage B (photo validation): 5 photos × 1 winning tuple × 3 distances
//!     × {off, on} = 30 measurements. Photo tuple via env vars
//!     `SAFE_CLASS_PHOTO_IDENT` / `SAFE_CLASS_PHOTO_DCT2`
//!     (defaults to the IDENTITY=1.10, DCT2X2=1.04 lowest cell so the
//!     TSV always contains a sanity row even without manual review).

#![allow(clippy::too_many_arguments, clippy::field_reassign_with_default)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{EntropyMulTable, LossyInternalParams};
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::HashMap;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0];
const EFFORT: u8 = 7;

// W35-2 chunk-4 lift grids — LOWER than W23-2 (which never passed the
// |bfly| gate). DCT2X2 sweep stays close to libjxl reference (0.95),
// hugging from 1.04..1.13 (W22-1 was 1.15, W23-2 swept 0.95..1.045).
// IDENTITY sweep drops from W23-2's 1.20..1.60 to 1.10..1.30 — accept
// smaller bytes gain in exchange for stable quality.
const IDENTITY_SWEEP: &[f32] = &[1.10, 1.15, 1.20, 1.25, 1.30];
const DCT2X2_SWEEP: &[f32] = &[1.04, 1.07, 1.10, 1.13];
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
        .unwrap_or_else(|| {
            PathBuf::from("benchmarks/entropy_mul_safe_class_bisect_2026-05-18.tsv")
        });

    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus")),
    );

    // W35-2 chunk-4: 9 plog2 ≥ 7 screenshots from gb82-sc.
    // windows95 (plog2=4) DROPPED — handled via classifier hint suppression.
    let screen_sources = vec![
        Source {
            label: "gb82/codec_wiki",
            class: "screen",
            path: corpus.join("gb82-sc/codec_wiki.png"),
        },
        Source {
            label: "gb82/gmessages",
            class: "screen",
            path: corpus.join("gb82-sc/gmessages.png"),
        },
        Source {
            label: "gb82/graph",
            class: "screen",
            path: corpus.join("gb82-sc/graph.png"),
        },
        Source {
            label: "gb82/gui",
            class: "screen",
            path: corpus.join("gb82-sc/gui.png"),
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
            label: "gb82/imessage",
            class: "screen",
            path: corpus.join("gb82-sc/imessage.png"),
        },
        Source {
            label: "gb82/terminal",
            class: "screen",
            path: corpus.join("gb82-sc/terminal.png"),
        },
        Source {
            label: "gb82/windows",
            class: "screen",
            path: corpus.join("gb82-sc/windows.png"),
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
    // Atomic write per `jj_concurrent_workspace_failure_mode_2026-05-18.md`:
    // bench writes to /tmp staging file, then `rename` into the repo at end.
    let staging = PathBuf::from(format!(
        "/tmp/entropy_mul_safe_class_bisect_{}.tsv",
        std::process::id()
    ));
    let mut out = std::fs::File::create(&staging).expect("create staging tsv");
    writeln!(
        out,
        "stage\timage\tclass\tdistance\tidentity\tdct2x2\tafv\tdct4x8\tbaseline_bytes\tlifted_bytes\tbaseline_bfly\tlifted_bfly\tbytes_delta_pct\tbfly_delta_pct\tbaseline_ms\tlifted_ms"
    )
    .unwrap();

    let bparams = ButteraugliParams::default();

    // ─── Stage A: safe-class IDENTITY × DCT2X2 sweep ─────────────────────
    eprintln!("=== Stage A: safe-class screen sweep (windows95 EXCLUDED) ===");
    eprintln!(
        "  grid: IDENTITY ∈ {:?}, DCT2X2 ∈ {:?}, distances ∈ {:?}",
        IDENTITY_SWEEP, DCT2X2_SWEEP, DISTANCES
    );
    eprintln!(
        "  AFV pinned at {}, DCT4X8 pinned at {} (W22-1 values)",
        AFV_PINNED, DCT4X8_PINNED
    );
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

    // ─── Stage B: photo validation ──────────────────────────────────────
    //
    // Default to the lowest cell in the safe-class grid so the TSV
    // always contains a sanity row. Set via env vars after reviewing
    // stage A.
    let photo_ident = std::env::var("SAFE_CLASS_PHOTO_IDENT")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.10);
    let photo_dct2 = std::env::var("SAFE_CLASS_PHOTO_DCT2")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.04);

    eprintln!("\n=== Stage B: photo validation ===");
    eprintln!(
        "  tuple: IDENTITY={} DCT2X2={} (override via SAFE_CLASS_PHOTO_IDENT / SAFE_CLASS_PHOTO_DCT2)",
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
