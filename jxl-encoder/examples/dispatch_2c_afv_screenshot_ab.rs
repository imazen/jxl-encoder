// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Issue #43 chunk 2c: AFV/8×8-class transform lift on screenshots —
//! pinned probe + paired A/B bench.
//!
//! Background: `EffortProfile.try_dct4x8_afv` (gating the DCT4X8 / DCT8X4
//! / DCT4X4 / AFV0-3 evaluation block in AC strategy search) defaults to
//! `effort >= 6` — exact libjxl parity (AFV enters at e6/Wombat). The
//! issue-#43 chunk-2c idea ("auto-enable AFV on screenshots at e>=6") is
//! therefore a structural no-op as originally specced; the only residual
//! dispatch value is at **e5**, mirroring the shipped screenshot patches
//! lift (`adapt_to_image_content`, e ∈ {5,6}).
//!
//! Two subcommands:
//!
//! `--pin-probe --output <tsv>` — pinned-pair measurement via `__expert`
//! `LossyInternalParams { try_dct4x8_afv: Some(false|true) }`. Both arms
//! skip the per-image dispatch adapters identically (override-respect), so
//! the pair isolates the 8×8-class evaluation block itself:
//!   * e5 pairs: is the block ever PICKED at e5 on screenshots, and what
//!     is the bytes/quality delta? (If byte-identical → no win possible →
//!     honest-stop before any production gate.)
//!   * e6/e7 pairs: liveness proof that the block is active at e>=6 by
//!     default (PIN(false) != PIN(true)) — the empirical half of the
//!     "specced 2c gate is a no-op" verdict.
//!   * e7 DEF row: production default vs PIN(true) — on >=500k-px
//!     screenshots every profile adapter is a no-op at e7, so
//!     DEF == PIN(true) byte-identity directly demonstrates that
//!     "auto-enable at e>=6" cannot change production output.
//!
//! `--bench --output <tsv>` — production-context paired A/B for the
//! candidate e5 gate (only meaningful AFTER the gate lands in
//! `adapt_to_image_content`). The arm is selected per PROCESS by the
//! caller via the gate's own env hook:
//!   A = `JXL_DISPATCH_AFV_SCREENSHOT_DISABLE=1` (gate off — today's
//!       production path, patches lift intact)
//!   B = env unset (gate fires on Screenshot at e5)
//! Run alternating A/B processes (sample-major interleave) and pair rows
//! offline; the encoder is deterministic per (cell, arm) so bytes/sha
//! pair exactly and wall is min-per-(cell,arm).
//!
//! Build:
//!   cargo build -p jxl-encoder --release \
//!     --features '__expert __pre_quantized parallel butteraugli-loop ssim2-loop' \
//!     --example dispatch_2c_afv_screenshot_ab

#![allow(clippy::too_many_arguments)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, Limits, LossyConfig, PixelLayout};
use jxl_encoder::effort::LossyInternalParams;
use rgb::RGB;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

const CID22: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82: &str = "/home/lilith/work/codec-corpus/gb82-sc";
/// imazen-26 modern screenshot strata (#43 2c revalidation, 2026-06-11):
/// per-viewport web captures + mobile captures. The W44-164 classifier +
/// gate were calibrated on gb82-sc's 10 retro images; these cells answer
/// whether the in-band win generalizes to modern captures.
const IMAZEN26: &str = "/home/lilith/work/codec-corpus/imazen-26";

struct ImageSpec {
    class: &'static str,
    name: &'static str,
    path: &'static str,
}

/// Bench set: 7 gb82-sc screenshots (gate-target class; all classify
/// `Screenshot` under the W44-164 fcbr >= 0.35 discriminator — corpus
/// fcbr range 0.360–0.907) + 4 CID22 validation photos as no-fire
/// guards (fcbr <= 0.098; 297394 is the highest-fcbr photo = the
/// closest-to-threshold guard).
const BENCH_IMAGES: &[ImageSpec] = &[
    ImageSpec {
        class: "SCREENSHOT",
        name: "codec_wiki",
        path: "codec_wiki.png",
    },
    ImageSpec {
        class: "SCREENSHOT",
        name: "terminal",
        path: "terminal.png",
    },
    ImageSpec {
        class: "SCREENSHOT",
        name: "imac_g3",
        path: "imac_g3.png",
    },
    ImageSpec {
        class: "SCREENSHOT",
        name: "windows95",
        path: "windows95.png",
    },
    ImageSpec {
        class: "SCREENSHOT",
        name: "gui",
        path: "gui.png",
    },
    ImageSpec {
        class: "SCREENSHOT",
        name: "graph",
        path: "graph.png",
    },
    ImageSpec {
        class: "SCREENSHOT",
        name: "gmessages",
        path: "gmessages.png",
    },
    ImageSpec {
        class: "PHOTO",
        name: "1418519",
        path: "1418519.png",
    },
    ImageSpec {
        class: "PHOTO",
        name: "1025469",
        path: "1025469.png",
    },
    ImageSpec {
        class: "PHOTO",
        name: "2389166",
        path: "2389166.png",
    },
    ImageSpec {
        class: "PHOTO",
        name: "297394",
        path: "297394.png",
    },
    ImageSpec {
        class: "SCREENSHOT_I26",
        name: "web1440_wayback",
        path: "8100-lilith-web-screenshots/1440x900/8100_web-screenshots_archive-wayback-search_dpr1_page1_1440x900.png",
    },
    ImageSpec {
        class: "SCREENSHOT_I26",
        name: "web1440_exhibits",
        path: "8100-lilith-web-screenshots/1440x900/8101_web-screenshots_archives-exhibits_dpr1_page1_1440x900.png",
    },
    ImageSpec {
        class: "SCREENSHOT_I26",
        name: "web1920_wayback",
        path: "8100-lilith-web-screenshots/1920x1080/8159_web-screenshots_archive-wayback-search_dpr1_page1_1920x1080.png",
    },
    ImageSpec {
        class: "SCREENSHOT_I26",
        name: "web1920_exhibits",
        path: "8100-lilith-web-screenshots/1920x1080/8160_web-screenshots_archives-exhibits_dpr1_page1_1920x1080.png",
    },
    ImageSpec {
        class: "SCREENSHOT_I26",
        name: "web2880_wayback",
        path: "8100-lilith-web-screenshots/2880x1800/8219_web-screenshots_archive-wayback-search_dpr2_page1_2880x1800.png",
    },
    ImageSpec {
        class: "SCREENSHOT_I26",
        name: "web2880_exhibits",
        path: "8100-lilith-web-screenshots/2880x1800/8220_web-screenshots_archives-exhibits_dpr2_page1_2880x1800.png",
    },
    ImageSpec {
        class: "SCREENSHOT_I26",
        name: "web375_wayback",
        path: "8100-lilith-web-screenshots/375x667/8271_web-screenshots_archive-wayback-search_dpr1_page1_375x667.png",
    },
    ImageSpec {
        class: "SCREENSHOT_I26",
        name: "web375_exhibits",
        path: "8100-lilith-web-screenshots/375x667/8272_web-screenshots_archives-exhibits_dpr1_page1_375x667.png",
    },
    ImageSpec {
        class: "SCREENSHOT_I26",
        name: "web768_wayback",
        path: "8100-lilith-web-screenshots/768x1024/8330_web-screenshots_archive-wayback-search_dpr1_page1_768x1024.png",
    },
    ImageSpec {
        class: "SCREENSHOT_I26",
        name: "web768_exhibits",
        path: "8100-lilith-web-screenshots/768x1024/8331_web-screenshots_archives-exhibits_dpr1_page1_768x1024.png",
    },
    ImageSpec {
        class: "SCREENSHOT_I26",
        name: "mobile_imageflow",
        path: "8000-lilith-mobile-screenshots/8000_mobile-screenshots_imageflow-server_screenshot-20260526-065435-brave_1968x2184.png",
    },
    ImageSpec {
        class: "SCREENSHOT_I26",
        name: "mobile_brave1080",
        path: "8000-lilith-mobile-screenshots/8002_mobile-screenshots_imageflow-server_screenshot-20260526-065522-brave_1080x2520.png",
    },
];

/// Gate-relevant distances per the chunk-2c brief.
const DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 4.0];

fn full_path(spec: &ImageSpec) -> PathBuf {
    if spec.class == "SCREENSHOT_I26" {
        PathBuf::from(IMAZEN26).join(spec.path)
    } else if spec.class == "SCREENSHOT" {
        PathBuf::from(GB82).join(spec.path)
    } else {
        PathBuf::from(CID22).join(spec.path)
    }
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

fn compute_metrics(
    bytes: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> (f64, f64) {
    if let Some((dw, dh, dec)) = decode_jxl_linear(bytes) {
        let dec_pixels: Vec<RGB<f32>> = dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
        let dec_linear_img = Img::new(dec_pixels, dw, dh);
        let bfly = butteraugli_linear(orig_linear.as_ref(), dec_linear_img.as_ref(), params)
            .map(|r| r.score)
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
        let dec_srgb_img = Img::new(dec_srgb, dw, dh);
        let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref())
            .unwrap_or(f64::NAN);
        (bfly, ssim2)
    } else {
        (f64::NAN, f64::NAN)
    }
}

/// `afv_pin`: `None` = production default path (all dispatch adapters
/// active); `Some(v)` = `__expert` pinned `try_dct4x8_afv` (all dispatch
/// adapters skipped — override-respect).
fn encode_once(
    rgb: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    distance: f32,
    afv_pin: Option<bool>,
) -> (Vec<u8>, u128) {
    let mut cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_threads(1);
    if let Some(v) = afv_pin {
        let mut params = LossyInternalParams::default();
        params.try_dct4x8_afv = Some(v);
        cfg = cfg.with_internal_params(params);
    }
    let limits = Limits::new().with_max_memory_bytes(8 * 1024 * 1024 * 1024);
    let t0 = Instant::now();
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(rgb)
        .expect("encode");
    let ms = t0.elapsed().as_millis();
    (bytes, ms)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest.iter() {
        use std::fmt::Write;
        write!(&mut s, "{:02x}", b).unwrap();
    }
    s
}

struct LoadedImage {
    rgb: Vec<u8>,
    w: u32,
    h: u32,
    linear: Img<Vec<RGB<f32>>>,
    srgb: Img<Vec<[u8; 3]>>,
}

fn load(spec: &ImageSpec) -> LoadedImage {
    let p = full_path(spec);
    let img = image::open(&p).unwrap_or_else(|e| panic!("decode {}: {e}", p.display()));
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();
    let linear = srgb_u8_to_linear(&rgb, w, h);
    let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let srgb = Img::new(srgb_pixels, w as usize, h as usize);
    LoadedImage {
        rgb,
        w,
        h,
        linear,
        srgb,
    }
}

/// Pinned-pair probe. Per (image, effort, distance):
///   PIN_OFF = try_dct4x8_afv pinned false
///   PIN_ON  = try_dct4x8_afv pinned true
/// and at e7 additionally DEF (production default, no pin).
fn run_pin_probe(output_path: &str) {
    let mut out = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(output_path)
        .expect("open output");
    writeln!(
        out,
        "class\timage\teffort\tdistance\tarm\tbytes\tbfly\tssim2\tencode_ms\tsha256_16"
    )
    .unwrap();
    let bp = ButteraugliParams::default();
    // Probe grid: screenshots only (the gate-target class), d ∈ {1.0, 2.0}
    // (mid-band; enough for liveness + pick detection), e ∈ {5, 6, 7}.
    let probe_set: Vec<&ImageSpec> = BENCH_IMAGES
        .iter()
        .filter(|s| s.class == "SCREENSHOT")
        .collect();
    for spec in probe_set {
        let li = load(spec);
        eprintln!("[pin-probe] {} {}x{}", spec.name, li.w, li.h);
        for &effort in &[5u8, 6, 7] {
            for &distance in &[1.0f32, 2.0] {
                let mut arms: Vec<(&str, Option<bool>)> =
                    vec![("PIN_OFF", Some(false)), ("PIN_ON", Some(true))];
                if effort == 7 {
                    arms.push(("DEF", None));
                }
                for (arm, pin) in arms {
                    let (bytes, ms) = encode_once(&li.rgb, li.w, li.h, effort, distance, pin);
                    let (bfly, ssim2) = compute_metrics(&bytes, &li.linear, &li.srgb, &bp);
                    let sha = sha256_hex(&bytes);
                    let row = format!(
                        "{}\t{}\t{}\t{:.1}\t{}\t{}\t{:.5}\t{:.5}\t{}\t{}",
                        spec.class,
                        spec.name,
                        effort,
                        distance,
                        arm,
                        bytes.len(),
                        bfly,
                        ssim2,
                        ms,
                        &sha[..16],
                    );
                    println!("{}", row);
                    writeln!(out, "{}", row).unwrap();
                    out.flush().unwrap();
                }
            }
        }
    }
    eprintln!("[done] wrote {}", output_path);
}

/// Production-context A/B grid. Arm comes from the gate's env hook,
/// set per process by the caller (see module docs). Appends to the
/// output TSV so alternating A/B process invocations interleave.
/// `dump_dir`: when set, every encoded bitstream is persisted as
/// `<dir>/<image>_e<e>_d<d>_<arm>.jxl` (content-identical across
/// samples — determinism — so overwrites are harmless). Used for the
/// external-decoder roundtrip verification (djxl + jxl-rs + jxl-oxide).
fn run_bench(output_path: &str, sample: u32, dump_dir: Option<&str>) {
    let arm = if env::var("JXL_DISPATCH_AFV_SCREENSHOT_DISABLE").as_deref() == Ok("1") {
        "A_gate_off"
    } else {
        "B_gate_on"
    };
    let write_header = !std::path::Path::new(output_path).exists();
    let mut out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .expect("open output");
    if write_header {
        writeln!(
            out,
            "class\timage\teffort\tdistance\tarm\tsample\tbytes\tbfly\tssim2\tencode_ms\tsha256_16"
        )
        .unwrap();
    }
    let bp = ButteraugliParams::default();
    for spec in BENCH_IMAGES {
        let li = load(spec);
        eprintln!(
            "[bench {} s{}] {} {}x{}",
            arm, sample, spec.name, li.w, li.h
        );
        // e5 = gated cells; e6 = no-fire guard (gate is e5-only).
        for &effort in &[5u8, 6] {
            for &distance in DISTANCES {
                // Photos are no-fire guards; one distance pair is enough
                // at e6 to bound runtime (full distance grid at e5).
                if spec.class == "PHOTO" && effort == 6 && distance != 1.0 {
                    continue;
                }
                let (bytes, ms) = encode_once(&li.rgb, li.w, li.h, effort, distance, None);
                if let Some(dir) = dump_dir {
                    let p = std::path::Path::new(dir).join(format!(
                        "{}_e{}_d{}_{}.jxl",
                        spec.name, effort, distance, arm
                    ));
                    std::fs::write(&p, &bytes).expect("dump bitstream");
                }
                let (bfly, ssim2) = compute_metrics(&bytes, &li.linear, &li.srgb, &bp);
                let sha = sha256_hex(&bytes);
                let row = format!(
                    "{}\t{}\t{}\t{:.1}\t{}\t{}\t{}\t{:.5}\t{:.5}\t{}\t{}",
                    spec.class,
                    spec.name,
                    effort,
                    distance,
                    arm,
                    sample,
                    bytes.len(),
                    bfly,
                    ssim2,
                    ms,
                    &sha[..16],
                );
                println!("{}", row);
                writeln!(out, "{}", row).unwrap();
                out.flush().unwrap();
            }
        }
    }
    eprintln!(
        "[done arm={} sample={}] appended to {}",
        arm, sample, output_path
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let output_path = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str);
    if args.iter().any(|a| a == "--pin-probe") {
        run_pin_probe(output_path.expect("--output <tsv>"));
        return;
    }
    if args.iter().any(|a| a == "--bench") {
        let sample = args
            .iter()
            .position(|a| a == "--sample")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let dump_dir = args
            .iter()
            .position(|a| a == "--dump")
            .and_then(|i| args.get(i + 1))
            .map(String::as_str);
        if let Some(d) = dump_dir {
            std::fs::create_dir_all(d).expect("create dump dir");
        }
        run_bench(output_path.expect("--output <tsv>"), sample, dump_dir);
        return;
    }
    eprintln!(
        "usage: dispatch_2c_afv_screenshot_ab --pin-probe --output <tsv> | --bench --output <tsv> [--sample N] [--dump <dir>]"
    );
    std::process::exit(2);
}
