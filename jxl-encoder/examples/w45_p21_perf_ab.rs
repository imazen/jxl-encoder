//! W45-RECON part-20/21 perf A/B — encode wall time before vs after the
//! EPF `kMinSigma` gate (shared code) and the extras GlobalData stream-0
//! port (strict-only).
//!
//! Cells target the touched paths:
//! - photo + screenshot at e5/e7 (EPF kernels + epf_dispatch at e6-7)
//! - RGBA synthetics (extras Global stream, strict only)
//! - the byte-lock gradient cells themselves
//!
//! Run identically on the baseline build (dc15c5d2) and the candidate
//! build; compare min-of-N ms per cell.

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::{Path, PathBuf};
use std::time::Instant;

const TIME_ITERS: u32 = 5;

fn corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/lilith/work/codec-corpus"))
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32, PixelLayout)> {
    let img = image::open(path).ok()?;
    match img {
        image::DynamicImage::ImageRgb8(rgb) => Some((
            rgb.as_raw().clone(),
            rgb.width(),
            rgb.height(),
            PixelLayout::Rgb8,
        )),
        image::DynamicImage::ImageRgba8(rgba) => Some((
            rgba.as_raw().clone(),
            rgba.width(),
            rgba.height(),
            PixelLayout::Rgba8,
        )),
        other => {
            let rgb = other.to_rgb8();
            Some((
                rgb.as_raw().clone(),
                rgb.width(),
                rgb.height(),
                PixelLayout::Rgb8,
            ))
        }
    }
}

fn gradient_rgb_64x64() -> (Vec<u8>, u32, u32) {
    let (w, h) = (64usize, 64usize);
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            px.push(((x * 255) / w) as u8);
            px.push(((y * 255) / h) as u8);
            px.push(((x ^ y) & 0xFF) as u8);
        }
    }
    (px, w as u32, h as u32)
}

fn gradient_rgba_64x32() -> (Vec<u8>, u32, u32) {
    let (w, h) = (64usize, 32usize);
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            out[i] = (x * 255 / (w - 1)) as u8;
            out[i + 1] = (y * 255 / (h - 1)) as u8;
            out[i + 2] = 64;
            out[i + 3] = if (24..40).contains(&x) { 128 } else { 255 };
        }
    }
    (out, w as u32, h as u32)
}

/// Larger RGBA so the extras path is measurable: 512x512 gradient with a
/// two-level alpha checker/strip mix (palette-eligible: few colours).
fn gradient_rgba_512x512() -> (Vec<u8>, u32, u32) {
    let (w, h) = (512usize, 512usize);
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            out[i] = (x * 255 / (w - 1)) as u8;
            out[i + 1] = (y * 255 / (h - 1)) as u8;
            out[i + 2] = ((x ^ y) & 0xFF) as u8;
            out[i + 3] = if (x / 32 + y / 32) % 2 == 0 { 200 } else { 255 };
        }
    }
    (out, w as u32, h as u32)
}

struct Cell {
    name: &'static str,
    effort: u8,
    distance: f32,
    src: CellSrc,
}

enum CellSrc {
    Png(&'static str),
    Gen(fn() -> (Vec<u8>, u32, u32), PixelLayout),
}

const CELLS: &[Cell] = &[
    Cell {
        name: "cid22_1279330",
        effort: 5,
        distance: 1.0,
        src: CellSrc::Png("CID22/CID22-512/validation/1279330.png"),
    },
    Cell {
        name: "cid22_1279330",
        effort: 7,
        distance: 1.0,
        src: CellSrc::Png("CID22/CID22-512/validation/1279330.png"),
    },
    Cell {
        name: "terminal",
        effort: 5,
        distance: 1.0,
        src: CellSrc::Png("gb82-sc/terminal.png"),
    },
    Cell {
        name: "terminal",
        effort: 7,
        distance: 1.0,
        src: CellSrc::Png("gb82-sc/terminal.png"),
    },
    Cell {
        name: "grad64",
        effort: 7,
        distance: 1.0,
        src: CellSrc::Gen(gradient_rgb_64x64, PixelLayout::Rgb8),
    },
    Cell {
        name: "grad64",
        effort: 7,
        distance: 4.0,
        src: CellSrc::Gen(gradient_rgb_64x64, PixelLayout::Rgb8),
    },
    Cell {
        name: "rgba64x32",
        effort: 7,
        distance: 2.0,
        src: CellSrc::Gen(gradient_rgba_64x32, PixelLayout::Rgba8),
    },
    Cell {
        name: "rgba64x32",
        effort: 5,
        distance: 1.0,
        src: CellSrc::Gen(gradient_rgba_64x32, PixelLayout::Rgba8),
    },
    Cell {
        name: "rgba512",
        effort: 7,
        distance: 1.0,
        src: CellSrc::Gen(gradient_rgba_512x512, PixelLayout::Rgba8),
    },
    Cell {
        name: "rgba512",
        effort: 5,
        distance: 1.0,
        src: CellSrc::Gen(gradient_rgba_512x512, PixelLayout::Rgba8),
    },
];

fn run_cell(cell: &Cell, strategy: &EncoderStrategy) -> Option<String> {
    let (px, w, h, layout) = match &cell.src {
        CellSrc::Png(rel) => {
            let (p, w, h, l) = load_png(&corpus_dir().join(rel))?;
            (p, w, h, l)
        }
        CellSrc::Gen(f, l) => {
            let (p, w, h) = f();
            (p, w, h, *l)
        }
    };
    let mut best_ms = f64::INFINITY;
    let mut bytes_len = 0usize;
    for _ in 0..TIME_ITERS {
        let cfg = LossyConfig::new(cell.distance)
            .with_effort(cell.effort)
            .with_strategy(strategy.clone())
            .with_threads(1);
        let start = Instant::now();
        let bytes = cfg.encode(&px, w, h, layout).ok()?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if ms < best_ms {
            best_ms = ms;
            bytes_len = bytes.len();
        }
    }
    Some(format!("{bytes_len}\t{best_ms:.3}"))
}

fn main() {
    println!("name\teffort\tdistance\tstrategy\tbytes\tms_min");
    for cell in CELLS {
        for (label, strategy) in [
            ("zen", EncoderStrategy::Zenjxl),
            ("exact", EncoderStrategy::Libjxl),
        ] {
            match run_cell(cell, &strategy) {
                Some(tail) => println!(
                    "{}\t{}\t{}\t{label}\t{tail}",
                    cell.name, cell.effort, cell.distance
                ),
                None => println!(
                    "{}\t{}\t{}\t{label}\tFAILED\tFAILED",
                    cell.name, cell.effort, cell.distance
                ),
            }
        }
    }
}
