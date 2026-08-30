//! Isolation A/B for RFC#45 chunk 3: lossy multi-seed butteraugli sweep.
//!
//! Holds effort fixed at e10 (8 buttloop iters) and e11 (16 buttloop iters),
//! and toggles `lossy_search_seeds` between 1 (libjxl-equivalent
//! single-seed `kInitMul=0.6`) and the profile default (2 at e10, 4 at
//! e11). This isolates the chunk-3 contribution from the chunk-1
//! extended-iter contribution.
//!
//! Per-cell pairing at each (image, distance, effort):
//!   A = seeds=1 (single, libjxl)
//!   B = seeds=default (multi: 2 at e10, 4 at e11)
//!
//! The picker's worst case is the libjxl basin (`init_mul_seeds[0]` is
//! always `LIBJXL_INIT_MUL=0.6`), so seeds=default bytes should be
//! ≤ seeds=1 bytes on cells where the picker's "smallest mean(qf)"
//! criterion is rewarded by the cost model (most photographic content
//! at d ≥ 1.0).
//!
//! Usage:
//!   cargo run -p jxl-encoder --release \
//!     --features 'std parallel butteraugli-loop __expert' \
//!     --example lossy_multiseed_isolate_ab \
//!     > benchmarks/lossy_multiseed_isolate_ab_$(date +%Y-%m-%d).tsv
//!
//! Environment (all optional):
//!   SAMPLES=2         (per-cell paired samples)
//!   THREADS=8         (rayon pool size)
//!   DISTANCES="1.0,2.0,4.0"
//!   CORPUS_DIR=/home/lilith/work/codec-corpus

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use jxl_encoder::LossyInternalParams;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;
use sha2::Digest;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Instant;

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
        .unwrap_or_else(|| vec![1.0, 2.0, 4.0])
}

fn load_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
}

fn short_sha(bytes: &[u8]) -> String {
    let h = sha2::Sha256::digest(bytes);
    let mut s = String::with_capacity(16);
    for &b in h.iter().take(8) {
        s.push_str(&format!("{b:02x}"));
    }
    s
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

fn encode_one(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    threads: usize,
    seeds_override: Option<u8>,
    orig_linear: &Img<Vec<RGB<f32>>>,
) -> Option<(usize, f64, f64, String)> {
    let mut cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(threads);
    if let Some(s) = seeds_override {
        // LossyInternalParams is #[non_exhaustive] — must construct via
        // Default then field-assign (struct-literal forbidden externally).
        let mut params = LossyInternalParams::default();
        params.lossy_search_seeds = Some(s);
        cfg = cfg.with_internal_params(params);
    }
    let t = Instant::now();
    let bytes = match cfg.encode(rgb, w, h, PixelLayout::Rgb8) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("encode err e{effort} d={distance}: {e:?}");
            return None;
        }
    };
    let encode_ms = t.elapsed().as_secs_f64() * 1e3;
    let (dw, dh, decoded) = decode_jxl_linear(&bytes)?;
    let bfly = measure_butteraugli(orig_linear, &decoded, dw, dh);
    Some((bytes.len(), encode_ms, bfly, short_sha(&bytes)))
}

fn main() {
    let samples = parse_usize("SAMPLES", 2);
    let threads = parse_usize("THREADS", 8);
    let distances = parse_distances();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();

    let corpus_dir = PathBuf::from(
        std::env::var("CORPUS_DIR")
            .unwrap_or_else(|_| "/home/lilith/work/codec-corpus".to_string()),
    );
    let images: Vec<PathBuf> = IMAGES.iter().map(|p| corpus_dir.join(p)).collect();

    println!("# lossy multi-seed isolation A/B (RFC#45 chunk 3)");
    println!("# distances: {distances:?}");
    println!("# images: {}", images.len());
    println!("# samples per cell (paired): {samples}");
    println!("# A = seeds=1 (libjxl single-seed kInitMul=0.6)");
    println!("# B = seeds=default (e10: 2, e11: 4)");
    println!("image\tdist\teffort\tsample\tlabel\tbytes\tencode_ms\tbutteraugli\tsha8\tw\th");

    // Sample-major interleave: outer over samples, then cells, then labels.
    for sample in 0..samples {
        for image_path in &images {
            let stem = image_path.file_stem().unwrap().to_str().unwrap();
            let Some((rgb, w, h)) = load_rgb(image_path) else {
                eprintln!("# skip (load failed): {}", image_path.display());
                continue;
            };
            let orig_linear = srgb_u8_to_linear_img(&rgb, w, h);
            for &dist in &distances {
                for &effort in &[10u8, 11u8] {
                    // A: seeds=1
                    if let Some((b, ms, bfly, sha)) =
                        encode_one(&rgb, w, h, dist, effort, threads, Some(1), &orig_linear)
                    {
                        println!(
                            "{stem}\t{dist:.2}\t{effort}\t{sample}\tA\t{b}\t{ms:.2}\t{bfly:.4}\t{sha}\t{w}\t{h}"
                        );
                    }
                    // B: seeds=default (None lets profile decide)
                    if let Some((b, ms, bfly, sha)) =
                        encode_one(&rgb, w, h, dist, effort, threads, None, &orig_linear)
                    {
                        println!(
                            "{stem}\t{dist:.2}\t{effort}\t{sample}\tB\t{b}\t{ms:.2}\t{bfly:.4}\t{sha}\t{w}\t{h}"
                        );
                    }
                }
            }
        }
    }
}
