//! W22-1 chunk-2 follow-on stage A2: test if removing the W22-1 AFV
//! (0.95) and DCT4X8 (0.98) lifts unlocks any winning tuple. Stage A
//! showed that even with the smallest IDENTITY ∈ {1.20..1.60} and
//! DCT2X2 ∈ {0.95, 0.9975, 1.045}, NO tuple passes the chunk-2 gate
//! (max(|Δbfly|) ≤ 2 %); excursions reach +18.45 % on terminal at d=2.
//!
//! Hypothesis: AFV lift 0.818 → 0.95 (+16 %) or DCT4X8 lift 0.859 →
//! 0.98 (+14 %) — both from the W22-1 GPU-bisected tuple — are the
//! dominant cause, not IDENTITY/DCT2X2. Stage A2 holds AFV + DCT4X8
//! at the libjxl reference values and re-sweeps the same
//! IDENTITY × DCT2X2 grid.
//!
//! Run:
//!   cargo run -p jxl-encoder --release --features '__expert butteraugli-loop' \
//!     --example cpu_entropy_mul_bisect_stage_a2 [-- <out.tsv>]
//!
//! Default output: `benchmarks/cpu_entropy_mul_bisect_stage_a2_2026-05-18.tsv`.
//!
//! Cells: 5 screenshots × 5 IDENTITY × 3 DCT2X2 × 3 distances = 225
//! + 15 baselines = 240 measurements. ~6 min on the 7950X.

#![allow(clippy::too_many_arguments, clippy::field_reassign_with_default)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::effort::{EntropyMulTable, LossyInternalParams};
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0];
const EFFORT: u8 = 7;

const IDENTITY_SWEEP: &[f32] = &[1.20, 1.30, 1.40, 1.50, 1.60];
const DCT2X2_SWEEP: &[f32] = &[0.95, 0.9975, 1.045];
// Stage A2: AFV + DCT4X8 at libjxl reference, NOT W22-1 lifts.
const AFV_REF: f32 = 0.817_794_9;
const DCT4X8_REF: f32 = 0.859_316_37;

struct Source {
    label: &'static str,
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
    t.afv = AFV_REF;
    t.dct4x8 = DCT4X8_REF;
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_path = args
        .into_iter()
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("benchmarks/cpu_entropy_mul_bisect_stage_a2_2026-05-18.tsv")
        });

    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus")),
    );

    let sources = vec![
        Source {
            label: "gb82/codec_wiki",
            path: corpus.join("gb82-sc/codec_wiki.png"),
        },
        Source {
            label: "gb82/imac_dark",
            path: corpus.join("gb82-sc/imac_dark.png"),
        },
        Source {
            label: "gb82/imac_g3",
            path: corpus.join("gb82-sc/imac_g3.png"),
        },
        Source {
            label: "gb82/terminal",
            path: corpus.join("gb82-sc/terminal.png"),
        },
        Source {
            label: "gb82/windows95",
            path: corpus.join("gb82-sc/windows95.png"),
        },
    ];

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let staging = PathBuf::from(format!(
        "/tmp/cpu_entropy_mul_bisect_stage_a2_{}.tsv",
        std::process::id()
    ));
    let mut out = std::fs::File::create(&staging).expect("create staging tsv");
    writeln!(
        out,
        "stage\timage\tclass\tdistance\tidentity\tdct2x2\tafv\tdct4x8\tbaseline_bytes\tlifted_bytes\tbaseline_bfly\tlifted_bfly\tbytes_delta_pct\tbfly_delta_pct\tbaseline_ms\tlifted_ms"
    )
    .unwrap();

    let bparams = ButteraugliParams::default();

    eprintln!("=== Stage A2: AFV + DCT4X8 at libjxl reference ===");
    eprintln!(
        "  grid: IDENTITY ∈ {:?}, DCT2X2 ∈ {:?}, distances ∈ {:?}",
        IDENTITY_SWEEP, DCT2X2_SWEEP, DISTANCES
    );
    eprintln!(
        "  AFV={} (reference), DCT4X8={} (reference) — NOT W22-1 lifts",
        AFV_REF, DCT4X8_REF
    );

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

        eprintln!("--- {} ({}x{}) ---", src.label, w, h);
        for &d in DISTANCES {
            let t0 = Instant::now();
            let baseline_bytes = match encode_baseline(rgb_u8, w, h, d) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("  baseline d={d}: {e}");
                    continue;
                }
            };
            let baseline_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let baseline = match measure_bytes_and_bfly(
                baseline_bytes,
                w,
                h,
                &orig_linear_img,
                &bparams,
                baseline_ms,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  baseline d={d}: {e}");
                    continue;
                }
            };
            eprintln!(
                "  baseline d={d}: {}B bfly={:.4} {:.0}ms",
                baseline.bytes, baseline.butteraugli, baseline.encode_ms
            );

            for &ident in IDENTITY_SWEEP {
                for &dct2 in DCT2X2_SWEEP {
                    let t1 = Instant::now();
                    let lifted_bytes = match encode_lifted(rgb_u8, w, h, d, ident, dct2) {
                        Ok(b) => b,
                        Err(e) => {
                            eprintln!("  ident={ident} dct2={dct2} d={d}: {e}");
                            continue;
                        }
                    };
                    let lifted_ms = t1.elapsed().as_secs_f64() * 1000.0;
                    let lifted = match measure_bytes_and_bfly(
                        lifted_bytes,
                        w,
                        h,
                        &orig_linear_img,
                        &bparams,
                        lifted_ms,
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
                        "A2\t{}\tscreen\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:+.3}\t{:+.3}\t{:.2}\t{:.2}",
                        src.label,
                        d,
                        ident,
                        dct2,
                        AFV_REF,
                        DCT4X8_REF,
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
                        "    ident={:.2} dct2={:.4} → Δbytes={:+.3}% Δbfly={:+.3}% ({}B vs {}B)",
                        ident, dct2, bytes_delta_pct, bfly_delta_pct, lifted.bytes, baseline.bytes
                    );
                }
            }
        }
    }

    drop(out);
    std::fs::rename(&staging, &out_path).expect("atomic mv staging → out_path");
    eprintln!("\nTSV: {}", out_path.display());
}
