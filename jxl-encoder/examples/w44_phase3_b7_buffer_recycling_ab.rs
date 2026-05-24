//! W44-phase3-B7a+b paired wall-time bench: CPU buttloop buffer recycling.
//!
//! Measures wall-clock time of `LossyConfig`-driven encode-with-buttloop on a
//! handful of photo cells at effort 9 (4-iter butteraugli loop). The B7
//! change recycles the diffmap output Vec (B7a) and the 3 subsample buffers
//! (B7b) across iters via the precompute crate's BufferPool, eliminating
//! ~16 MB + ~12 MB of fresh allocator traffic per 1024² encode.
//!
//! Paired A/B via env hook `JXL_W44_B7_DISABLE`:
//!  - mode A (DISABLE=1): pre-B7 path (`compare_linear_planar` + `into_buf`)
//!  - mode B (env unset): production path (`compare_linear_planar_into`)
//!
//! Same binary, same OS state, alternating per cell.
//!
//! Output: TSV with `cell,effort,distance,iter,mode_a_ms,mode_b_ms,delta_pct`.
//!
//! Numeric impact is expected to be modest on a single run because the libc
//! allocator's free + reuse is fast (~1 µs each); the win is mostly in
//! allocator pressure under load, not single-iter wall. Acceptance is
//! "no wall regression > 0.5 %".
//!
//! Run: `cargo run -p jxl-encoder --release --features '__expert butteraugli-loop parallel' --example w44_phase3_b7_buffer_recycling_ab -- <out.tsv>`

use std::path::PathBuf;
use std::time::Instant;

use image::ImageReader;
use jxl_encoder::api::{LossyConfig, PixelLayout};

const PHOTO_CELLS: &[&str] = &[
    "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
    "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1025469.png",
];

fn encode_once(rgb: &[u8], width: u32, height: u32, distance: f32, effort: u8) -> u128 {
    let cfg = LossyConfig::new(distance).with_effort(effort);
    let t0 = Instant::now();
    let _bytes = cfg
        .encode(rgb, width, height, PixelLayout::Rgb8)
        .expect("encode");
    t0.elapsed().as_micros()
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/w44_phase3_b7_buffer_recycling_ab.tsv".to_string());
    let mut lines: Vec<String> =
        vec!["cell\teffort\tdistance\titer\tmode_a_baseline_us\tmode_b_b7_us\tdelta_pct".to_string()];

    const TIME_ITERS: usize = 8;
    const DISTANCE: f32 = 1.0;
    const EFFORT: u8 = 9;

    // For heaptrack: setting MODE=A forces all encodes to mode A (pre-B7
    // baseline); MODE=B forces all to mode B (B7 production). Default
    // (env unset) runs interleaved pairs for wall-time measurement.
    let force_mode = std::env::var("MODE").ok();

    for cell in PHOTO_CELLS {
        let path = PathBuf::from(cell);
        if !path.exists() {
            eprintln!("SKIP {cell} (not found)");
            continue;
        }
        let img = ImageReader::open(&path)
            .expect("open")
            .decode()
            .expect("decode")
            .to_rgb8();
        let (w, h) = (img.width(), img.height());
        let rgb = img.into_raw();

        // Warm both paths once
        // SAFETY: tests use env vars to flip the path; this is not parallel.
        unsafe { std::env::set_var("JXL_W44_B7_DISABLE", "1") };
        let _ = encode_once(&rgb, w, h, DISTANCE, EFFORT);
        unsafe { std::env::remove_var("JXL_W44_B7_DISABLE") };
        let _ = encode_once(&rgb, w, h, DISTANCE, EFFORT);

        if let Some(mode) = force_mode.as_deref() {
            // Pure-mode run for heaptrack: same N iters, no interleave.
            if mode == "A" {
                unsafe { std::env::set_var("JXL_W44_B7_DISABLE", "1") };
            } else {
                unsafe { std::env::remove_var("JXL_W44_B7_DISABLE") };
            }
            for i in 0..TIME_ITERS {
                let us = encode_once(&rgb, w, h, DISTANCE, EFFORT);
                lines.push(format!(
                    "{}\t{}\t{}\t{}\t{}\tMODE_{}",
                    cell, EFFORT, DISTANCE, i, us, mode
                ));
            }
            continue;
        }

        // Interleaved: A B B A A B B A ... (mirror antisymmetric, removes
        // linear drift). Record per-iter pair.
        let mut a_times = Vec::with_capacity(TIME_ITERS);
        let mut b_times = Vec::with_capacity(TIME_ITERS);
        for i in 0..TIME_ITERS {
            if i % 2 == 0 {
                unsafe { std::env::set_var("JXL_W44_B7_DISABLE", "1") };
                a_times.push(encode_once(&rgb, w, h, DISTANCE, EFFORT));
                unsafe { std::env::remove_var("JXL_W44_B7_DISABLE") };
                b_times.push(encode_once(&rgb, w, h, DISTANCE, EFFORT));
            } else {
                unsafe { std::env::remove_var("JXL_W44_B7_DISABLE") };
                b_times.push(encode_once(&rgb, w, h, DISTANCE, EFFORT));
                unsafe { std::env::set_var("JXL_W44_B7_DISABLE", "1") };
                a_times.push(encode_once(&rgb, w, h, DISTANCE, EFFORT));
            }
        }
        for (i, (a_us, b_us)) in a_times.iter().zip(b_times.iter()).enumerate() {
            let delta_pct = (*b_us as f64 - *a_us as f64) / *a_us as f64 * 100.0;
            lines.push(format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{:+.3}",
                cell, EFFORT, DISTANCE, i, a_us, b_us, delta_pct
            ));
        }

        // Also emit aggregate median row for clarity.
        let mut a_sorted = a_times.clone();
        let mut b_sorted = b_times.clone();
        a_sorted.sort_unstable();
        b_sorted.sort_unstable();
        let a_median = a_sorted[a_sorted.len() / 2];
        let b_median = b_sorted[b_sorted.len() / 2];
        let median_delta = (b_median as f64 - a_median as f64) / a_median as f64 * 100.0;
        lines.push(format!(
            "{}\t{}\t{}\tMEDIAN\t{}\t{}\t{:+.3}",
            cell, EFFORT, DISTANCE, a_median, b_median, median_delta
        ));
    }

    std::fs::write(&out_path, lines.join("\n") + "\n").expect("write");
    eprintln!("wrote {} lines to {}", lines.len() - 1, out_path);
}
