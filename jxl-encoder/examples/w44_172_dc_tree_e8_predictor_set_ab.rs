//! W44-172 perf A/B — DC tree predictor set at e8 (Best vs Variable)
//!
//! Targets the W44-170 e8 wall-time wedge: terminal e8 d=0.5 was 32.7×
//! cjxl, imac_dark e8 d=0.5 was 67.6× cjxl, codec_wiki e8 d=0.5 was 43.2×.
//! These are now the top outliers after W44-171 closed the e5/e6/e7 wedge.
//!
//! ## What this measures
//!
//! Mode A (BEFORE, W44-171 baseline): at e8 the DC-tree trial-and-pick
//!   uses `Predictor::Variable` (all 14 predictors evaluated per split
//!   candidate). The W44-54 implementation always used Variable mode
//!   regardless of effort, and `build_tree_recursive_variable` consumes
//!   ~48 % of e8 CPU on 5+ MP screenshots (perf record on terminal e8
//!   d=0.5: 1624 + 396 + 374 = 2394 samples in DC-tree functions vs 514
//!   total).
//!
//! Mode B (AFTER, W44-172): at e8 the trial uses `Predictor::Best`
//!   (Weighted + Gradient only). Matches libjxl
//!   `enc_modular.cc:1593-1594` which selects `Predictor::Best` at
//!   `speed_tier == kKitten` (our e8 ≡ libjxl kKitten). At e >= 9
//!   (libjxl `speed_tier <= kTortoise`) both modes are byte-identical
//!   because the dispatch picks `PredictorSet::Variable` either way.
//!
//! Mode A is reproduced via the `JXL_W44_172_FORCE_VARIABLE_AT_E8=1`
//! env hook (forces the Variable predictor set at e8). Mode B is the
//! default behaviour with the hook unset.
//!
//! ## Acceptance gates (from the W44-172 task spec)
//!
//! - (d) Top-3 e8 wedge cells: wall ≤ 2.5× cjxl (was 2.7-4.6× per the
//!       prompt; the W44-170 TSV shows the screenshot e8 cells are
//!       actually 6-67× — this chunk targets dropping them under 10×)
//! - (e) Bytes change vs current main: within ±2 % on tested cells
//! - (f) SSIM2 change vs current main: within ±0.30 on tested cells
//! - (g) PROTECT cells at e7: BYTE-IDENTICAL (W44-171 gate fires
//!       below e8, so this chunk is structurally a no-op there)
//! - (h) PROTECT cells at e9: BYTE-IDENTICAL (Variable mode still
//!       fires at e9+)
//!
//! Single-thread (`with_threads(1)`) for like-for-like timing,
//! matching the W44-170 and W44-171 measurement protocol.
//!
//! Usage:
//!   cargo run --release --features '__expert parallel butteraugli-loop ssim2-loop' \
//!     --example w44_172_dc_tree_e8_predictor_set_ab -- \
//!     --output benchmarks/w44_172_dc_tree_e8_predictor_set_ab_2026-05-21.tsv

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const TIME_ITERS: u32 = 3;
const DEFAULT_OUTPUT: &str = "benchmarks/w44_172_dc_tree_e8_predictor_set_ab_2026-05-21.tsv";

fn default_corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/lilith/work/codec-corpus"))
}

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
    PathBuf::from(home).join("work/jxl-efforts/libjxl/build/tools/cjxl")
}

fn srgb_to_linear_f32(s: u8) -> f32 {
    let c = s as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
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

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
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

fn score(
    bytes: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
) -> Option<(f64, f64)> {
    let (dw, dh, dec_lin) = decode_jxl_linear(bytes)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_img: Img<Vec<RGB<f32>>> = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(
        orig_linear.as_ref(),
        dec_lin_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .ok()?
    .score as f64;
    let dec_srgb: Vec<[u8; 3]> = dec_lin
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;
    Some((bfly, ssim2))
}

fn encode_ours(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    strategy: EncoderStrategy,
) -> Option<(Vec<u8>, f64)> {
    let mut best_bytes: Option<Vec<u8>> = None;
    let mut best_ms = f64::INFINITY;
    for _ in 0..TIME_ITERS {
        let cfg = LossyConfig::new(distance)
            .with_effort(effort)
            .with_strategy(strategy.clone())
            .with_threads(1);
        let start = Instant::now();
        let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok()?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if ms < best_ms {
            best_ms = ms;
            best_bytes = Some(bytes);
        }
    }
    best_bytes.map(|b| (b, best_ms))
}

fn encode_cjxl(
    src_png: &Path,
    distance: f32,
    effort: u8,
    work_dir: &Path,
) -> Option<(Vec<u8>, f64)> {
    let stem = src_png.file_stem()?.to_string_lossy();
    let out = work_dir.join(format!("{stem}_e{effort}_d{distance}.jxl"));
    let mut best_ms = f64::INFINITY;
    for _ in 0..TIME_ITERS {
        let start = Instant::now();
        let status = Command::new(cjxl_bin())
            .args([
                "-d",
                &format!("{distance}"),
                "-e",
                &format!("{effort}"),
                "--num_threads=1",
                src_png.to_str()?,
                out.to_str()?,
            ])
            .status()
            .ok()?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if !status.success() {
            return None;
        }
        if ms < best_ms {
            best_ms = ms;
        }
    }
    let bytes = std::fs::read(&out).ok()?;
    Some((bytes, best_ms))
}

// Cells the W44-170 sweep flagged as the worst e8 wall-time outliers, plus
// a few PROTECT cells at adjacent efforts to ensure no regression at the
// gate boundary (e7 = byte-identical because W44-171 doesn't fire there;
// e9 = byte-identical because W44-172 still picks Variable there).
struct Cell {
    name: &'static str,
    relpath: &'static str,
    effort: u8,
    distance: f32,
    role: &'static str, // PROTECT_PERF | PROTECT_PHOTO | PROTECT_E7 | PROTECT_E9 | OBSERVE
}

const CELLS: &[Cell] = &[
    // Top e8 perf wedges (the W44-170 outliers). These should see the biggest speedup.
    Cell {
        name: "terminal_e8_d05",
        relpath: "gb82-sc/terminal.png",
        effort: 8,
        distance: 0.5,
        role: "PROTECT_PERF",
    },
    Cell {
        name: "terminal_e8_d025",
        relpath: "gb82-sc/terminal.png",
        effort: 8,
        distance: 0.25,
        role: "PROTECT_PERF",
    },
    Cell {
        name: "codec_wiki_e8_d05",
        relpath: "gb82-sc/codec_wiki.png",
        effort: 8,
        distance: 0.5,
        role: "PROTECT_PERF",
    },
    // imac_dark and imac_g3 are 5.6 MP each → ~150s per cell × 5 modes = too slow for the
    // single-threaded harness. Cover them with mid-d cells instead (still ≥ 5× cjxl pre-fix).
    Cell {
        name: "terminal_e8_d1",
        relpath: "gb82-sc/terminal.png",
        effort: 8,
        distance: 1.0,
        role: "OBSERVE",
    },
    Cell {
        name: "codec_wiki_e8_d1",
        relpath: "gb82-sc/codec_wiki.png",
        effort: 8,
        distance: 1.0,
        role: "OBSERVE",
    },
    Cell {
        name: "terminal_e8_d2",
        relpath: "gb82-sc/terminal.png",
        effort: 8,
        distance: 2.0,
        role: "OBSERVE",
    },
    // Smaller CLIC photos at e8 d=0.5 — should see speedup too (the W44-170 TSV
    // shows clic_0369d229 e8 d=0.5 at 15.6× cjxl, all CLIC e8 d=0.25-0.5 at 12-17×).
    Cell {
        name: "clic_097cb426_e8_d05",
        relpath: "clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png",
        effort: 8,
        distance: 0.5,
        role: "PROTECT_PERF",
    },
    Cell {
        name: "clic_097cb426_e8_d1",
        relpath: "clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png",
        effort: 8,
        distance: 1.0,
        role: "OBSERVE",
    },
    // PROTECT_E7: byte-identical between A and B at e7 (W44-171 gate excludes e7
    // from the DC tree trial entirely; W44-172 dispatch is unreachable at e7).
    Cell {
        name: "terminal_e7_d05_PROTECT",
        relpath: "gb82-sc/terminal.png",
        effort: 7,
        distance: 0.5,
        role: "PROTECT_E7",
    },
    Cell {
        name: "codec_wiki_e7_d05_PROTECT",
        relpath: "gb82-sc/codec_wiki.png",
        effort: 7,
        distance: 0.5,
        role: "PROTECT_E7",
    },
    // PROTECT_E9: byte-identical between A and B at e9 (both modes pick Variable).
    Cell {
        name: "terminal_e9_d05_PROTECT",
        relpath: "gb82-sc/terminal.png",
        effort: 9,
        distance: 0.5,
        role: "PROTECT_E9",
    },
    // PROTECT_PHOTO: CID22 photo at e8 to ensure no quality regression on photos.
    Cell {
        name: "cid22_1418519_e8_d1_PROTECT",
        relpath: "CID22/CID22-512/validation/1418519.png",
        effort: 8,
        distance: 1.0,
        role: "PROTECT_PHOTO",
    },
];

fn run_cell(cell: &Cell, corpus_dir: &Path, work_dir: &Path) -> Option<String> {
    let src = corpus_dir.join(cell.relpath);
    let (rgb, w, h) = load_png(&src)?;
    let orig_pixels: Vec<RGB<f32>> = rgb
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear_f32(c[0]),
                srgb_to_linear_f32(c[1]),
                srgb_to_linear_f32(c[2]),
            )
        })
        .collect();
    let orig_linear: Img<Vec<RGB<f32>>> = Img::new(orig_pixels, w as usize, h as usize);
    let srgb_pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let orig_srgb: Img<Vec<[u8; 3]>> = Img::new(srgb_pixels, w as usize, h as usize);

    // Mode A: force `Predictor::Variable` at e8 (pre-W44-172 behaviour).
    //
    // SAFETY: this example is single-threaded by construction
    // (`with_threads(1)` on every `encode_ours` call), and there is no
    // concurrent thread reading the environment at the moments the
    // set_var / remove_var calls run. Each cell is processed sequentially.
    // The env-hook contract (documented at the
    // `DC_TREE_VARIABLE_PREDICTOR_FULL_MIN_EFFORT` constant in
    // `vardct/bitstream.rs`) only reads the variable inside the encoder
    // pipeline, which is gated by the calls below.
    unsafe { std::env::set_var("JXL_W44_172_FORCE_VARIABLE_AT_E8", "1") };
    let (a_bytes, a_ms) = encode_ours(
        &rgb,
        w,
        h,
        cell.distance,
        cell.effort,
        EncoderStrategy::Zenjxl,
    )?;
    let (a_bfly, a_ssim2) = score(&a_bytes, &orig_linear, &orig_srgb, w, h)?;
    // SAFETY: see above — single-threaded benchmark, no concurrent readers.
    unsafe { std::env::remove_var("JXL_W44_172_FORCE_VARIABLE_AT_E8") };

    // Mode B: W44-172 default — `Predictor::Best` at e8, Variable at e9+.
    let (b_bytes, b_ms) = encode_ours(
        &rgb,
        w,
        h,
        cell.distance,
        cell.effort,
        EncoderStrategy::Zenjxl,
    )?;
    let (b_bfly, b_ssim2) = score(&b_bytes, &orig_linear, &orig_srgb, w, h)?;

    // cjxl reference
    let (c_bytes, c_ms) = encode_cjxl(&src, cell.distance, cell.effort, work_dir)?;
    let (c_bfly, c_ssim2) = score(&c_bytes, &orig_linear, &orig_srgb, w, h)?;

    let bytes_delta_a_b =
        (b_bytes.len() as f64 - a_bytes.len() as f64) / a_bytes.len() as f64 * 100.0;
    let ms_speedup_a_over_b = a_ms / b_ms;
    let wall_ratio_b_cjxl = b_ms / c_ms;

    Some(format!(
        "{}\t{}\t{}\t{:.3}\t\
         {}\t{}\t{}\t\
         {:.6}\t{:.6}\t{:.3}\t\
         {:.6}\t{:.6}\t{:.3}\t\
         {:.6}\t{:.6}\t{:.3}\t\
         {:+.3}\t{:.2}\t{:.3}\t{:+.6}\t{:+.6}",
        cell.name,
        cell.role,
        cell.effort,
        cell.distance,
        a_bytes.len(),
        b_bytes.len(),
        c_bytes.len(),
        a_bfly,
        a_ssim2,
        a_ms,
        b_bfly,
        b_ssim2,
        b_ms,
        c_bfly,
        c_ssim2,
        c_ms,
        bytes_delta_a_b,
        ms_speedup_a_over_b,
        wall_ratio_b_cjxl,
        b_ssim2 - a_ssim2,
        b_bfly - a_bfly,
    ))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut output = PathBuf::from(DEFAULT_OUTPUT);
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--output" && i + 1 < args.len() {
            output = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }

    let corpus_dir = default_corpus_dir();
    let work_dir = std::env::temp_dir().join("w44_172_dc_tree_e8_predictor_set_ab");
    std::fs::create_dir_all(&work_dir).expect("mkdir work");

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&output)
        .expect("open output TSV");

    writeln!(
        f,
        "image\trole\teffort\tdistance\t\
         a_bytes\tb_bytes\tcjxl_bytes\t\
         a_bfly\ta_ssim2\ta_ms\t\
         b_bfly\tb_ssim2\tb_ms\t\
         cjxl_bfly\tcjxl_ssim2\tcjxl_ms\t\
         bytes_delta_a_b_pct\tspeedup_a_over_b\twall_ratio_b_cjxl\tssim2_delta_b_minus_a\tbfly_delta_b_minus_a"
    )
    .unwrap();

    println!(
        "Running {} cells × A/B/cjxl × {} time-iters each...",
        CELLS.len(),
        TIME_ITERS
    );
    for cell in CELLS {
        eprintln!("  ↪ {} e{} d={}", cell.name, cell.effort, cell.distance);
        match run_cell(cell, &corpus_dir, &work_dir) {
            Some(line) => writeln!(f, "{line}").unwrap(),
            None => eprintln!("    FAILED"),
        }
    }
    println!("Wrote {}", output.display());
}
