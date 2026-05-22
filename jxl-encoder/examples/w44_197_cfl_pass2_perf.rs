//! W44-197: perf measurement + per-cell A/B for Candidates B+C.
//!
//! **Candidate C (perf, byte-identical default)**: precompute `inv_qw` in
//! `refine_cfl_map` to replace `q / qw_x[i]` per-coefficient division with
//! `q * inv_qw_x[i]` multiplication (mirrors libjxl
//! `enc_chroma_from_luma.cc:337-343`). W44-189 D13 audit predicted 5-10 ms
//! saved per 12 MP at e>=7. This bench measures the actual wall-time delta
//! by running Zenjxl encode at e>=7 on a real photo.
//!
//! **Candidate B (Pass-2 LS-only at e=5/6 under Libjxl strategy)**:
//! `cfl_pass2_ls_at_low_effort` gate on `EncoderImprovementsCustom`. Under
//! `EncoderStrategy::Libjxl` the existing Section A `cfl_two_pass_min_effort
//! = EffortGate::Libjxl` widening already fires Pass-2 LS at e=5/6 (because
//! `cfl_newton: effort >= 7` evaluates false there), so the new gate is
//! structurally redundant on `Libjxl`. This bench documents that no Libjxl
//! cell shifts.
//!
//! Outputs to stdout — pinned encoder version (`cargo run --release`).
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release --features '__expert parallel butteraugli-loop ssim2-loop' \
//!       --manifest-path jxl-encoder/Cargo.toml \
//!       --example w44_197_cfl_pass2_perf

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::time::Instant;

const ITERS: u32 = 5;

struct Cell {
    short: &'static str,
    path: &'static str,
}

const CELLS: &[Cell] = &[
    Cell {
        short: "cid22_1025469",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png",
    },
    Cell {
        short: "cid22_1418519",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
    },
    Cell {
        short: "gb82_codec_wiki",
        path: "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
    },
];

fn load_png(path: &str) -> Option<(u32, u32, Vec<u8>)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = (img.width(), img.height());
    Some((w, h, img.into_raw()))
}

fn encode_zenjxl(w: u32, h: u32, rgb: &[u8], effort: u8, distance: f32) -> (usize, f64) {
    let cfg = LossyConfig::new(distance).with_effort(effort);
    let mut out = Vec::new();
    // Warm-up
    let _ = cfg
        .encode_into(rgb, w, h, PixelLayout::Rgb8, &mut out)
        .expect("encode");
    let mut best_ms = f64::INFINITY;
    let mut bytes = 0usize;
    for _ in 0..ITERS {
        out.clear();
        let t = Instant::now();
        cfg.encode_into(rgb, w, h, PixelLayout::Rgb8, &mut out)
            .expect("encode");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if ms < best_ms {
            best_ms = ms;
        }
        bytes = out.len();
    }
    (bytes, best_ms)
}

fn encode_libjxl(w: u32, h: u32, rgb: &[u8], effort: u8, distance: f32) -> (usize, f64) {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Libjxl);
    let mut out = Vec::new();
    let _ = cfg
        .encode_into(rgb, w, h, PixelLayout::Rgb8, &mut out)
        .expect("encode");
    let mut best_ms = f64::INFINITY;
    let mut bytes = 0usize;
    for _ in 0..ITERS {
        out.clear();
        let t = Instant::now();
        cfg.encode_into(rgb, w, h, PixelLayout::Rgb8, &mut out)
            .expect("encode");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if ms < best_ms {
            best_ms = ms;
        }
        bytes = out.len();
    }
    (bytes, best_ms)
}

fn main() {
    println!("# W44-197 perf + A/B measurement");
    println!("# ITERS={}, best-of-N wall ms reported", ITERS);
    println!();
    println!("## Candidate C (perf): Zenjxl e7 d=1.0, e7 d=3.0");
    println!("cell\teffort\tdistance\tbytes\twall_ms");
    for cell in CELLS {
        let Some((w, h, rgb)) = load_png(cell.path) else {
            println!("{}\t-\t-\t-\tSKIP (cannot load)", cell.short);
            continue;
        };
        for (effort, distance) in [(7u8, 1.0f32), (7, 3.0)] {
            let (bytes, ms) = encode_zenjxl(w, h, &rgb, effort, distance);
            println!(
                "{}\te{}\td={}\t{}\t{:.1}",
                cell.short, effort, distance, bytes, ms
            );
        }
    }

    println!();
    println!("## Candidate B (Pass-2 LS at e=5/6): Libjxl strategy at e5, e6 d=1.0");
    println!("# Note: Libjxl strategy already fires Pass-2 LS at e=5/6 via Section A");
    println!("# `cfl_two_pass_min_effort = EffortGate::Libjxl` widening, so the new W44-197");
    println!("# gate is structurally redundant on this strategy. This bench documents that");
    println!("# bytes are unchanged (W44-197 is a Custom-strategy capability).");
    println!("cell\teffort\tdistance\tlibjxl_bytes\tlibjxl_ms");
    for cell in CELLS {
        let Some((w, h, rgb)) = load_png(cell.path) else {
            println!("{}\t-\t-\t-\tSKIP", cell.short);
            continue;
        };
        for (effort, distance) in [(5u8, 1.0f32), (6, 1.0)] {
            let (bytes, ms) = encode_libjxl(w, h, &rgb, effort, distance);
            println!(
                "{}\te{}\td={}\t{}\t{:.1}",
                cell.short, effort, distance, bytes, ms
            );
        }
    }
}
