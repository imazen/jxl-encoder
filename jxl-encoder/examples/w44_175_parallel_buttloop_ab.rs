//! W44-175 perf A/B — parallelize buttloop transform + DC tree split scan
//!
//! Targets the W44-170/W44-172 residual e8 wall-time wedge: post-W44-172
//! `Predictor::Best` cut the DC tree wall in half (3.30× speedup on terminal
//! e8 d=0.5) but `perf record` on the W44-172 binary still shows:
//!   - `estimate_subset_cost_per_predictor` 22.5 % of CPU (DC tree split eval)
//!   - `partition`                          8.7  % of CPU (DC tree partition)
//!   - `transform_blocks_into`              2.8  % of CPU (buttloop inner)
//! Sequential DC tree learning + sequential `transform_and_quantize_into`
//! combined to keep the buttloop pipeline single-threaded even with
//! `--threads=N` set. Multi-thread wall on terminal e8 d=0.5 = single-thread
//! wall (both ~1.65 s), proving no parallelism gain anywhere on the critical
//! path.
//!
//! ## What this measures
//!
//! Mode A (BEFORE, W44-172 baseline): pre-W44-175 sequential paths.
//!   - `transform_and_quantize_into` runs `for gy { for gx { … } }` sequentially.
//!   - `find_best_split_variable` runs `for &prop_idx in SPLIT_PROPERTIES_VARIABLE`
//!     sequentially.
//!
//! Mode B (AFTER, W44-175): both loops fan out across rayon when the
//! `parallel` feature is enabled.
//!   - `transform_and_quantize_into` → `parallel_map` over groups (matches
//!     the sibling `transform_and_quantize` which already parallelized this
//!     work in W44-89 for the AC-strategy-search pass).
//!   - `find_best_split_variable` → `parallel_map` over the 14 properties,
//!     with a property-rank-ordered serial reduction to preserve byte-exact
//!     tie-breaking with the sequential code.
//!
//! Mode A is reproduced via two env hooks:
//!   - `JXL_W44_175_FORCE_SEQUENTIAL_TRANSFORM_AND_QUANTIZE_INTO=1`
//!   - `JXL_W44_175_FORCE_SEQUENTIAL_DC_TREE_SPLIT=1`
//! Both set together = pre-W44-175 wall behaviour; both unset = W44-175
//! production behaviour.
//!
//! ## Acceptance gates (from the W44-175 task spec)
//!
//! - (d) Top-3 e8 wedge cells: wall ≤ 2.5× cjxl (was 4-5× per the W44-172
//!       meta). The W44-175 prompt called this out as the target the
//!       transform-only fix should clear — but `perf record` revealed the
//!       hot path is in the DC tree, not in the transform. W44-175 ships
//!       BOTH parallelizations so the closeness to 2.5× depends on cell.
//!       The terminal e8 d=0.5 cell still sits at ~3× cjxl after W44-175
//!       because (a) `estimate_subset_cost_per_predictor`'s algorithmic
//!       cost outweighs parallelism gains (libjxl uses incremental
//!       histogram updates which we don't — W44-176+ scope), and (b) the
//!       butteraugli loop itself runs serially per iter (separate scope).
//! - (e) Bytes change vs current main: 0 (byte-identical — parallelization
//!       preserves output exactly via deterministic property-rank-ordered
//!       reduction).
//! - (f) SSIM2 / butteraugli change vs current main: 0 (decoded pixels are
//!       byte-identical because the encoded bitstream is byte-identical).
//! - (g) PROTECT_E5/E6/E7 cells: BYTE-IDENTICAL (W44-171 gate blocks the
//!       DC tree path entirely below e8; `transform_and_quantize_into`
//!       parallelization is byte-identical at every effort).
//! - (h) Hash-locks 36/36 BYTE-IDENTICAL (no regen).
//!
//! Multi-thread (default rayon thread pool) for the W44-175 timing — the
//! whole point of this chunk is to demonstrate that the buttloop pipeline
//! now actually scales with thread count, unlike pre-W44-175 where
//! single-thread = multi-thread wall on every wedge cell.
//!
//! Usage:
//!   cargo run --release --features '__expert parallel butteraugli-loop ssim2-loop' \
//!     --example w44_175_parallel_buttloop_ab -- \
//!     --output benchmarks/w44_175_parallel_buttloop_ab_2026-05-21.tsv

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
const DEFAULT_OUTPUT: &str = "benchmarks/w44_175_parallel_buttloop_ab_2026-05-21.tsv";

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

fn encode_ours_default_threads(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    strategy: EncoderStrategy,
) -> Option<(Vec<u8>, f64)> {
    // Multi-thread (default rayon pool) on purpose — the W44-175 fix is
    // about making the buttloop pipeline scale across threads. Single-
    // thread A/B would understate the win (only the rayon-vs-serial
    // overhead diff matters at thread count = 1).
    let mut best_bytes: Option<Vec<u8>> = None;
    let mut best_ms = f64::INFINITY;
    for _ in 0..TIME_ITERS {
        let cfg = LossyConfig::new(distance)
            .with_effort(effort)
            .with_strategy(strategy.clone());
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
    // cjxl default thread count (matches the way users run it in production
    // on multi-core hardware; cjxl-rs likewise defaults to all cores).
    let mut best_ms = f64::INFINITY;
    for _ in 0..TIME_ITERS {
        let start = Instant::now();
        let status = Command::new(cjxl_bin())
            .args([
                "-d",
                &format!("{distance}"),
                "-e",
                &format!("{effort}"),
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

// Cells the W44-172 sweep flagged as the remaining e8 wall outliers post
// the DC-tree predictor-set fix, plus PROTECT cells at adjacent efforts
// to confirm no regression at the gate boundaries.
struct Cell {
    name: &'static str,
    relpath: &'static str,
    effort: u8,
    distance: f32,
    role: &'static str, // PROTECT_PERF | PROTECT_PHOTO | PROTECT_E5/E7/E9
}

const CELLS: &[Cell] = &[
    // Top remaining e8 wedges post-W44-172.
    Cell {
        name: "terminal_e8_d05",
        relpath: "gb82-sc/terminal.png",
        effort: 8,
        distance: 0.5,
        role: "PROTECT_PERF",
    },
    Cell {
        name: "codec_wiki_e8_d05",
        relpath: "gb82-sc/codec_wiki.png",
        effort: 8,
        distance: 0.5,
        role: "PROTECT_PERF",
    },
    Cell {
        name: "imac_dark_e8_d05",
        relpath: "gb82-sc/imac_dark.png",
        effort: 8,
        distance: 0.5,
        role: "PROTECT_PERF",
    },
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
    // PROTECT_E5: W44-171 gates the DC tree path off at e<8; parallel
    // transform path still fires but should produce byte-identical output.
    Cell {
        name: "terminal_e5_d1_PROTECT",
        relpath: "gb82-sc/terminal.png",
        effort: 5,
        distance: 1.0,
        role: "PROTECT_E5",
    },
    // PROTECT_E7: W44-171 still gates DC tree path off at e=7.
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
    // PROTECT_E9: W44-172 picks Variable at e9 — the W44-175 property
    // parallelization fires here. Output must stay byte-identical.
    Cell {
        name: "terminal_e9_d05_PROTECT",
        relpath: "gb82-sc/terminal.png",
        effort: 9,
        distance: 0.5,
        role: "PROTECT_E9",
    },
    // PROTECT_PHOTO: CID22 photo at e8 to ensure no quality regression on
    // photo content.
    Cell {
        name: "cid22_1418519_e8_d1_PROTECT",
        relpath: "CID22/CID22-512/validation/1418519.png",
        effort: 8,
        distance: 1.0,
        role: "PROTECT_PHOTO",
    },
];

fn set_seq_env(on: bool) {
    // SAFETY: this example is single-test-thread by construction (cells run
    // sequentially in main()), and the encoder pipeline only reads these
    // env hooks inside the gated branches in `transform.rs` and
    // `dc_tree_learn.rs`. No concurrent reader can observe a torn state.
    if on {
        unsafe {
            std::env::set_var(
                "JXL_W44_175_FORCE_SEQUENTIAL_TRANSFORM_AND_QUANTIZE_INTO",
                "1",
            );
            std::env::set_var("JXL_W44_175_FORCE_SEQUENTIAL_DC_TREE_SPLIT", "1");
        }
    } else {
        unsafe {
            std::env::remove_var("JXL_W44_175_FORCE_SEQUENTIAL_TRANSFORM_AND_QUANTIZE_INTO");
            std::env::remove_var("JXL_W44_175_FORCE_SEQUENTIAL_DC_TREE_SPLIT");
        }
    }
}

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

    // Mode A: force sequential paths (pre-W44-175 behaviour).
    set_seq_env(true);
    let (a_bytes, a_ms) = encode_ours_default_threads(
        &rgb,
        w,
        h,
        cell.distance,
        cell.effort,
        EncoderStrategy::Zenjxl,
    )?;
    let (a_bfly, a_ssim2) = score(&a_bytes, &orig_linear, &orig_srgb, w, h)?;
    set_seq_env(false);

    // Mode B: W44-175 default — both loops parallel.
    let (b_bytes, b_ms) = encode_ours_default_threads(
        &rgb,
        w,
        h,
        cell.distance,
        cell.effort,
        EncoderStrategy::Zenjxl,
    )?;
    let (b_bfly, b_ssim2) = score(&b_bytes, &orig_linear, &orig_srgb, w, h)?;

    // cjxl reference (default thread count, matches our default).
    let (c_bytes, c_ms) = encode_cjxl(&src, cell.distance, cell.effort, work_dir)?;
    let (c_bfly, c_ssim2) = score(&c_bytes, &orig_linear, &orig_srgb, w, h)?;

    let bytes_delta_a_b =
        (b_bytes.len() as f64 - a_bytes.len() as f64) / a_bytes.len() as f64 * 100.0;
    let speedup_a_over_b = a_ms / b_ms;
    let wall_ratio_b_cjxl = b_ms / c_ms;
    let bytes_identical = a_bytes == b_bytes;

    Some(format!(
        "{}\t{}\t{}\t{:.3}\t\
         {}\t{}\t{}\t{}\t\
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
        if bytes_identical { "1" } else { "0" },
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
        speedup_a_over_b,
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
    let work_dir = std::env::temp_dir().join("w44_175_parallel_buttloop_ab");
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
         a_bytes\tb_bytes\tcjxl_bytes\tbytes_identical_a_b\t\
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
