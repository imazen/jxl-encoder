//! W38-2 wedge #1.1 — `fine_grained_step` libjxl-parity A/B at e9.
//!
//! Per W38-2 (`benchmarks/rd_curve_wedges_2026-05-18.md`), our e9
//! `fine_grained_step = 1` is inverted vs libjxl's
//! `enc_ac_strategy.cc:1046`:
//!
//! ```cpp
//! size_t step = cparams.speed_tier >= SpeedTier::kTortoise ? 2 : 1;
//! ```
//!
//! libjxl uses `step=2` at `speed_tier >= kTortoise` (which maps to
//! `effort 1..=9` on our scale), and `step=1` only at the slower-than-
//! kTortoise tiers (kGlacier / kTectonicPlate = our `effort 10..=12`
//! extension). We were doing 4× more 32×32 non-aligned search work at
//! e9 than libjxl AND consistently losing on the wedge audit's photo
//! cells at high distance.
//!
//! Acceptance gates (task brief):
//!   - wall-clock: `step=2` is 30-60% faster than `step=1` at e9
//!   - RD: bytes Δ within ±2%, butteraugli Δ within ±2%, ssim2 Δ within ±1
//!
//! Each encode is decoded with jxl-oxide in `srgb_linear` (per the
//! CLAUDE.md PNG-metadata-bug rule), scored with Rust butteraugli and
//! Rust fast-ssim2.
//!
//! Cells: 5 CID22-512 photos + 3 gb82-sc screenshots × 4 distances
//! {0.5, 1.0, 2.0, 4.0} × {A=step1 baseline, B=step2 fix} × 3 samples
//! = 192 encodes at e9 only.
//!
//! Per CLAUDE.md zenbench discipline, the sample-major round-robin keeps
//! paired (A, B) thermally close so the wall-clock delta is signal not
//! cooling noise. Output is TSV; pair them downstream.
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features 'std parallel butteraugli-loop ssim2-loop __expert' \
//!     --example fine_grained_step_libjxl_parity_ab \
//!     > benchmarks/fine_grained_step_libjxl_parity_2026-05-19.tsv

// Bench harness; favor readability of the inline measurement struct over
// extra type aliases. `encode_with_step` and the loaded-images tuple are
// both deliberately wide so the bench rows include every field the TSV
// downstream consumer expects (sister bench scripts use the same shape).
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use jxl_encoder::LossyInternalParams;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;
use sha2::Digest;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Instant;

// 5 CID22-512 photos + 3 gb82-sc screenshots — same wedge-audit corpus
// (matches `benchmarks/rd_curve_audit_vs_cjxl.tsv`).
const IMAGES: &[(&str, &str)] = &[
    ("1025469", "CID22/CID22-512/validation/1025469.png"),
    ("1418519", "CID22/CID22-512/validation/1418519.png"),
    ("1531677", "CID22/CID22-512/validation/1531677.png"),
    ("1189261", "CID22/CID22-512/validation/1189261.png"),
    ("1420710", "CID22/CID22-512/validation/1420710.png"),
    ("terminal", "gb82-sc/terminal.png"),
    ("codec_wiki", "gb82-sc/codec_wiki.png"),
    ("imac_g3", "gb82-sc/imac_g3.png"),
];

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 4.0];
const SAMPLES: usize = 3;
const EFFORT: u8 = 9;
const THREADS: usize = 8;

fn parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn load_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
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

fn srgb_u8_to_arr_img(rgb_u8: &[u8], w: u32, h: u32) -> Img<Vec<[u8; 3]>> {
    let pixels: Vec<[u8; 3]> = rgb_u8.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    Img::new(pixels, w as usize, h as usize)
}

#[derive(Debug, Clone, Copy)]
struct Measure {
    bytes: usize,
    encode_ms: f64,
    butteraugli: f64,
    ssim2: f64,
    sha256_prefix: u64,
}

fn encode_with_step(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    step: u8,
    threads: usize,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    bfly_params: &ButteraugliParams,
) -> Option<Measure> {
    // `LossyInternalParams` is `#[non_exhaustive]`, so struct-literal
    // construction is forbidden from outside the crate — build via
    // `Default` + field-assign (same pattern as `lossy_multiseed_isolate_ab`).
    let mut internal = LossyInternalParams::default();
    internal.fine_grained_step = Some(step);
    let cfg = LossyConfig::new(distance)
        .with_effort(EFFORT)
        .with_threads(threads)
        .with_internal_params(internal);
    let start = Instant::now();
    let bytes = match cfg.encode(rgb, w, h, PixelLayout::Rgb8) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "ERROR encoding step={} d={} w={} h={}: {:?}",
                step, distance, w, h, e
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
    let dec_linear_pixels: Vec<RGB<f32>> = decoded
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_linear_pixels, dw, dh);
    let bfly = butteraugli_linear(
        orig_linear_img.as_ref(),
        dec_linear_img.as_ref(),
        bfly_params,
    )
    .map(|r| r.score)
    .unwrap_or(f64::NAN);

    let dec_srgb_pixels: Vec<[u8; 3]> = decoded
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
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);

    Some(Measure {
        bytes: bytes.len(),
        encode_ms,
        butteraugli: bfly,
        ssim2,
        sha256_prefix,
    })
}

fn refresh_marker(activity: &str) {
    let repo_root = std::env::var("REPO_ROOT")
        .unwrap_or_else(|_| "/home/lilith/work/zen/jxl-encoder--fine-grained-step-fix".to_string());
    let path = format!("{}/.workongoing", repo_root);
    let _ = std::fs::write(
        path,
        format!("{} claude-fine-grained-step-fix {}\n", iso_now(), activity),
    );
}

fn main() {
    let samples = parse_usize("SAMPLES", SAMPLES);
    let threads = parse_usize("THREADS", THREADS);
    let corpus = std::env::var("CORPUS_DIR")
        .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus"));
    let corpus = PathBuf::from(corpus);

    eprintln!(
        "# W38-2 #1.1 fine_grained_step A/B at e{}\n# distances: {:?}\n# images: {}\n# samples per cell (paired): {}\n# threads: {}",
        EFFORT,
        DISTANCES,
        IMAGES.len(),
        samples,
        threads
    );

    println!("# W38-2 wedge #1.1: fine_grained_step libjxl parity at e9");
    println!("# A = step=1 (current default at e9; what we currently ship)");
    println!("# B = step=2 (libjxl enc_ac_strategy.cc:1046 kTortoise+ value)");
    println!("# samples per cell (paired): {}", samples);
    println!("# threads: {}", threads);
    println!(
        "image\tclass\tdistance\teffort\tsample\tvariant\tstep\tbytes\tencode_ms\tbutteraugli\tssim2\tsha256_prefix\twidth\theight"
    );

    let bfly_params = ButteraugliParams::default();

    // Load all images once + materialize the two reference forms used
    // for butteraugli (linear RGB f32) and ssim2 (sRGB u8 array).
    let images: Vec<(
        &str,
        &str,
        Vec<u8>,
        u32,
        u32,
        Img<Vec<RGB<f32>>>,
        Img<Vec<[u8; 3]>>,
    )> = IMAGES
        .iter()
        .filter_map(|(label, rel)| {
            let path = corpus.join(rel);
            let (rgb, w, h) = load_rgb(&path)?;
            let orig_linear = srgb_u8_to_linear_img(&rgb, w, h);
            let orig_srgb = srgb_u8_to_arr_img(&rgb, w, h);
            let class = if rel.starts_with("CID22/") {
                "photo"
            } else {
                "screenshot"
            };
            Some((*label, class, rgb, w, h, orig_linear, orig_srgb))
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

    // Warm up once per (image, distance) at step=2 — amortizes first-run
    // effects (file mmap, rayon thread spin-up). Both variants share the
    // same code path past `fine_grained_step`, so warming either suffices.
    for (label, _, rgb, w, h, orig_linear, orig_srgb) in &images {
        for &d in DISTANCES {
            refresh_marker(&format!("warmup {} d{}", label, d));
            let _ = encode_with_step(
                rgb,
                *w,
                *h,
                d,
                2,
                threads,
                orig_linear,
                orig_srgb,
                &bfly_params,
            );
        }
    }

    // Sample-major round-robin keeps A/B paired thermally close per
    // CLAUDE.md zenbench discipline. Order within cell: A then B.
    for s in 1..=samples {
        for (label, class, rgb, w, h, orig_linear, orig_srgb) in &images {
            for &d in DISTANCES {
                for (variant, step) in [("A", 1u8), ("B", 2u8)] {
                    refresh_marker(&format!(
                        "sample {}/{}: {} d{} ({} step={})",
                        s, samples, label, d, variant, step
                    ));
                    let Some(m) = encode_with_step(
                        rgb,
                        *w,
                        *h,
                        d,
                        step,
                        threads,
                        orig_linear,
                        orig_srgb,
                        &bfly_params,
                    ) else {
                        continue;
                    };
                    println!(
                        "{}\t{}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.4}\t{:.4}\t{:016x}\t{}\t{}",
                        label,
                        class,
                        d,
                        EFFORT,
                        s,
                        variant,
                        step,
                        m.bytes,
                        m.encode_ms,
                        m.butteraugli,
                        m.ssim2,
                        m.sha256_prefix,
                        w,
                        h
                    );
                }
            }
        }
    }

    refresh_marker("A/B complete");
}

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
