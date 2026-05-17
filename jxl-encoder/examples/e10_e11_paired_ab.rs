//! Paired A/B/C bench for RFC#45 chunk 1: e10/e11 extended butteraugli
//! search budgets vs e9 baseline.
//!
//! Per-cell A/B/C:
//!   A = e9  (baseline; libjxl kTortoise, 4 butteraugli iters)
//!   B = e10 (8 butteraugli iters; new in chunk 1)
//!   C = e11 (16 butteraugli iters, saturated at MAX_QUANT_LOOP_ITERS)
//!
//! Sample-major interleave keeps paired (A,B,C) thermally close at every
//! sample (CLAUDE.md zenbench discipline: randomized round-robin beats
//! criterion-style blocks for noise control).
//!
//! Each encode is followed by a jxl-oxide decode in linear sRGB, then
//! Rust butteraugli scoring — bypassing the `butteraugli_main` PNG-
//! metadata trap documented in project CLAUDE.md.
//!
//! Acceptance gate (per RFC#45 chunk-1 plan):
//!   e10 must produce ≤ e9 bytes at ≤ e9 butteraugli on ≥80% of cells.
//!   e11 must produce ≤ e10 bytes at ≤ e10 butteraugli on ≥80% of cells.
//!
//! Usage:
//!   cargo run -p jxl-encoder --release \
//!     --features 'std parallel butteraugli-loop' \
//!     --example e10_e11_paired_ab \
//!     > benchmarks/e10_e11_paired_ab_$(date +%Y-%m-%d).tsv
//!
//! Environment:
//!   SAMPLES=5         (default: 5; per-cell sample count, paired)
//!   THREADS=8         (default: 8)
//!   DISTANCES="0.5,1.0,2.0,4.0"  (default: 0.5,1.0,2.0,4.0)
//!   CORPUS_DIR        (default: /home/lilith/work/codec-corpus)

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;
use sha2::Digest;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Instant;

// 5 CID22-512 photos × 4 distances × 3 efforts × 5 samples = 300 encodes.
const IMAGES: &[&str] = &[
    "CID22/CID22-512/validation/1025469.png",
    "CID22/CID22-512/validation/1044329.png",
    "CID22/CID22-512/validation/1189261.png",
    "CID22/CID22-512/validation/1418519.png",
    "CID22/CID22-512/validation/2079234.png",
];

fn parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_distances() -> Vec<f32> {
    std::env::var("DISTANCES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<f32>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<f32>| !v.is_empty())
        .unwrap_or_else(|| vec![0.5, 1.0, 2.0, 4.0])
}

fn load_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
}

/// Decode JXL bytes via jxl-oxide in linear sRGB (per CLAUDE.md rule —
/// without `request_color_encoding(srgb_linear)` jxl-oxide returns sRGB
/// nonlinear and the butteraugli score is wrong).
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

fn srgb_u8_to_linear_img(rgb_u8: &[u8], w: u32, h: u32) -> Img<Vec<RGB<f32>>> {
    let pixels: Vec<RGB<f32>> = rgb_u8
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear(c[0]),
                srgb_to_linear(c[1]),
                srgb_to_linear(c[2]),
            )
        })
        .collect();
    Img::new(pixels, w as usize, h as usize)
}

fn measure_butteraugli(
    orig_linear: &Img<Vec<RGB<f32>>>,
    decoded_linear: &[f32],
    w: usize,
    h: usize,
) -> f64 {
    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_img = Img::new(dec_pixels, w, h);
    let params = ButteraugliParams::default();
    butteraugli_linear(orig_linear.as_ref(), dec_img.as_ref(), &params)
        .map(|r| r.score)
        .unwrap_or(f64::NAN)
}

#[derive(Debug, Clone, Copy)]
struct EncodeMeasure {
    bytes: usize,
    encode_ms: f64,
    butteraugli: f64,
    sha256_prefix: u64,
}

fn encode_at_effort(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    threads: usize,
    orig_linear: &Img<Vec<RGB<f32>>>,
) -> Option<EncodeMeasure> {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(threads);
    let start = Instant::now();
    let bytes = match cfg.encode(rgb, w, h, PixelLayout::Rgb8) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "ERROR encoding effort={} distance={} w={} h={}: {:?}",
                effort, distance, w, h, e
            );
            return None;
        }
    };
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut hasher = sha2::Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    let sha256_prefix = u64::from_be_bytes(digest[..8].try_into().unwrap());
    let (dw, dh, decoded) = decode_jxl_linear(&bytes)?;
    let bfly = measure_butteraugli(orig_linear, &decoded, dw, dh);
    Some(EncodeMeasure {
        bytes: bytes.len(),
        encode_ms,
        butteraugli: bfly,
        sha256_prefix,
    })
}

fn refresh_marker(activity: &str) {
    let repo_root = std::env::var("REPO_ROOT")
        .unwrap_or_else(|_| "/home/lilith/work/zen/jxl-encoder--e10-e11-chunk1".to_string());
    let path = format!("{}/.workongoing", repo_root);
    let _ = std::fs::write(
        path,
        format!("{} claude-e10-e11-chunk1 {}\n", iso_now(), activity),
    );
}

fn main() {
    let samples = parse_usize("SAMPLES", 5);
    let threads = parse_usize("THREADS", 8);
    let distances = parse_distances();
    let corpus = std::env::var("CORPUS_DIR")
        .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus"));
    let corpus = PathBuf::from(corpus);

    eprintln!(
        "# e10/e11 paired A/B/C (RFC#45 chunk 1)\n# distances: {:?}\n# images: {}\n# samples per cell (paired): {}\n# threads: {}",
        distances,
        IMAGES.len(),
        samples,
        threads
    );

    println!("# RFC#45 chunk 1: e10/e11 paired A/B/C bench");
    println!("# A = e9  (baseline, libjxl kTortoise, 4 butteraugli iters)");
    println!("# B = e10 (8 butteraugli iters)");
    println!("# C = e11 (16 butteraugli iters)");
    println!("# samples per cell (paired): {}", samples);
    println!("# threads: {}", threads);
    println!(
        "image\tdistance\teffort\tsample\tvariant\tbytes\tencode_ms\tbutteraugli\tsha256_prefix\twidth\theight"
    );

    // Load all images once.
    let images: Vec<(String, Vec<u8>, u32, u32, Img<Vec<RGB<f32>>>)> = IMAGES
        .iter()
        .filter_map(|rel| {
            let path = corpus.join(rel);
            let (rgb, w, h) = load_rgb(&path)?;
            let orig_linear = srgb_u8_to_linear_img(&rgb, w, h);
            let basename = path
                .file_stem()
                .map(|x| x.to_string_lossy().to_string())
                .unwrap_or_else(|| rel.to_string());
            Some((basename, rgb, w, h, orig_linear))
        })
        .collect();
    if images.len() < IMAGES.len() {
        eprintln!(
            "WARNING: only {}/{} images loaded",
            images.len(),
            IMAGES.len()
        );
    }
    if images.is_empty() {
        eprintln!(
            "FATAL: no images loaded; check CORPUS_DIR={}",
            corpus.display()
        );
        std::process::exit(1);
    }

    // Warm up once per (image, distance) at e9 only — amortizes first-run
    // effects (file mmap, rayon thread spin-up). e10/e11 share the same
    // pipeline plus extra iters, so warming e9 covers them.
    for (label, rgb, w, h, orig_linear) in &images {
        for &d in &distances {
            refresh_marker(&format!("warmup {} d{}", label, d));
            let _ = encode_at_effort(rgb, *w, *h, d, 9, threads, orig_linear);
        }
    }

    // Sample-major interleave: for each sample s in 1..=N, walk every
    // (image, distance) cell, run A then B then C contiguously. Paired
    // delta measurements are thermally close per CLAUDE.md zenbench
    // discipline.
    for s in 1..=samples {
        for (label, rgb, w, h, orig_linear) in &images {
            for &d in &distances {
                for (variant, effort) in [("A", 9u8), ("B", 10u8), ("C", 11u8)] {
                    refresh_marker(&format!(
                        "sample {}/{}: {} d{} e{} ({})",
                        s, samples, label, d, effort, variant
                    ));
                    let Some(m) = encode_at_effort(rgb, *w, *h, d, effort, threads, orig_linear)
                    else {
                        continue;
                    };
                    println!(
                        "{}\t{:.2}\t{}\t{}\t{}\t{}\t{:.2}\t{:.4}\t{:016x}\t{}\t{}",
                        label,
                        d,
                        effort,
                        s,
                        variant,
                        m.bytes,
                        m.encode_ms,
                        m.butteraugli,
                        m.sha256_prefix,
                        w,
                        h
                    );
                }
            }
        }
    }

    refresh_marker("paired A/B/C complete");
}

// ── Local ISO timestamp helper (mirrors bucket_dispatch_paired_ab.rs) ──
fn iso_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let s = secs as i64;
    let days = s / 86400;
    let secs_of_day = s % 86400;
    let hh = secs_of_day / 3600;
    let mm = (secs_of_day % 3600) / 60;
    let ss = secs_of_day % 60;
    let (yy, mo, dd) = days_to_ymd(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        yy, mo, dd, hh, mm, ss
    )
}

fn days_to_ymd(mut days: i64) -> (i32, u32, u32) {
    let mut year: i32 = 1970;
    loop {
        let leap = is_leap(year);
        let yd = if leap { 366 } else { 365 };
        if days < yd as i64 {
            break;
        }
        days -= yd as i64;
        year += 1;
    }
    let dim: [u32; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 0;
    let mut d = days as u32;
    for (i, &md) in dim.iter().enumerate() {
        if d < md {
            month = i;
            break;
        }
        d -= md;
    }
    (year, month as u32 + 1, d + 1)
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}
