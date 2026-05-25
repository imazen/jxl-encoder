// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! cvvdp-fork Phase 8b (2026-05-25) — diffmap distribution capture harness.
//!
//! For ~5 representative cells (CID22 photos + GB82-SC screenshots) at
//! 4 distances and effort 8 (buttloop fires), encode the same image with
//! BOTH cvvdp (C_GPU) and butteraugli (B_CPU) loops. The
//! [`maybe_dump_diffmap_stats`](crate::vardct::perceptual_backend::maybe_dump_diffmap_stats)
//! hook (gated on `JXL_PHASE8B_DIFFMAP_DUMP`) writes one TSV row per
//! per-iter compare. We post-process the dump file to compute the
//! cvvdp/butteraugli mean ratio that drives Phase 8c's
//! `CVVDP_DIFFMAP_RENORM_SCALE` constant.
//!
//! ## Per-image scope
//!
//! - 3 CID22 photos: `1025469.png`, `1418519.png`, `1189261.png`
//!   (mid-mass, low-detail, high-detail respectively).
//! - 2 GB82-SC screenshots: `terminal.png`, `imac_g3.png` (text-heavy
//!   + UI-heavy).
//! - Distances {0.5, 1.0, 2.0, 3.0}, effort 8 (buttloop fires AT e≥8
//!   per `libjxl` `enc_adaptive_quantization.cc:1282`).
//!
//! ## Output
//!
//! 1. Raw dump TSV at the path given via `--dump-out` (one row per
//!    backend per compare call across all encodes).
//! 2. Aggregate scale ratio TSV at `--out` summarising per-(image, d)
//!    cvvdp/butter mean ratios + global stats.
//!
//! ## How to read the result
//!
//! Look for the printed line `aggregate cvvdp/butter mean ratio = X`.
//! That's the inverse of the Phase 8c renorm scale: if cvvdp's mean is
//! 21× larger than butteraugli's, the renorm scale is `1/21 ≈ 0.0476`.
//!
//! ## Run
//!
//! ```bash
//! # Disable the renorm so we capture RAW cvvdp values for analysis.
//! JXL_CVVDP_DIFFMAP_RENORM_SCALE=1.0 \
//!   cargo run --release -p jxl-encoder \
//!   --features '__expert butteraugli-loop cvvdp-loop ssim2-loop parallel' \
//!   --example cvvdp_phase8b_diffmap_distribution -- \
//!   --dump-out /tmp/cvvdp_p8b_dump.tsv \
//!   --out benchmarks/cvvdp_diffmap_distribution_2026-05-25.tsv
//! ```

#![cfg(all(feature = "cvvdp-loop", feature = "butteraugli-loop"))]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};

const CID22_DIR: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82_SC_DIR: &str = "/home/lilith/work/codec-corpus/gb82-sc";

/// (corpus, filename). Path lookup goes to CID22_DIR or GB82_SC_DIR.
const FIXTURES: &[(&str, &str)] = &[
    ("CID22", "1025469.png"),
    ("CID22", "1418519.png"),
    ("CID22", "1189261.png"),
    ("GB82-SC", "terminal.png"),
    ("GB82-SC", "imac_g3.png"),
];

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 3.0];
const EFFORT: u8 = 8;

fn load_source(corpus: &str, name: &str) -> (Vec<u8>, u32, u32) {
    let p = match corpus {
        "CID22" => std::path::Path::new(CID22_DIR).join(name),
        "GB82-SC" => std::path::Path::new(GB82_SC_DIR).join(name),
        _ => panic!("unknown corpus {corpus}"),
    };
    let img = image::open(&p).unwrap_or_else(|e| panic!("open {}: {e}", p.display()));
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    (rgb.as_raw().clone(), w, h)
}

fn encode_cell(
    pixels: &[u8],
    w: u32,
    h: u32,
    d: f32,
    cvvdp_opt_in: bool,
) -> Result<(Vec<u8>, f64), Box<dyn std::error::Error>> {
    let cfg = LossyConfig::new(d)
        .with_strategy(EncoderStrategy::Zenjxl)
        .with_effort(EFFORT)
        .with_perceptual_metric(if cvvdp_opt_in {
            jxl_encoder::api::PerceptualMetric::Cvvdp
        } else {
            jxl_encoder::api::PerceptualMetric::Butteraugli
        });
    let t = Instant::now();
    let encoded = cfg.encode(pixels, w, h, PixelLayout::Rgb8)?;
    let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
    Ok((encoded, wall_ms))
}

struct DumpRow {
    backend: String,
    _compare_call: u32,
    _width: usize,
    _height: usize,
    _n: usize,
    mean: f64,
    _median: f64,
    _p25: f64,
    _p75: f64,
    _p95: f64,
    _max: f64,
    _score: f64,
    // Synthetic tag injected by the harness via the source-encode loop
    // (Vec is segmented per encode round, then we tag rows in order).
    tag: String,
}

fn parse_dump(path: &PathBuf) -> Vec<DumpRow> {
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) => panic!("Phase 8b dump file missing at {}: {e}", path.display()),
    };
    let mut rows = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 12 {
            continue;
        }
        let row = DumpRow {
            backend: parts[0].to_string(),
            _compare_call: parts[1].parse().unwrap_or(0),
            _width: parts[2].parse().unwrap_or(0),
            _height: parts[3].parse().unwrap_or(0),
            _n: parts[4].parse().unwrap_or(0),
            mean: parts[5].parse().unwrap_or(0.0),
            _median: parts[6].parse().unwrap_or(0.0),
            _p25: parts[7].parse().unwrap_or(0.0),
            _p75: parts[8].parse().unwrap_or(0.0),
            _p95: parts[9].parse().unwrap_or(0.0),
            _max: parts[10].parse().unwrap_or(0.0),
            _score: parts[11].parse().unwrap_or(0.0),
            tag: String::new(),
        };
        rows.push(row);
    }
    rows
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut dump_out: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--dump-out" && i + 1 < args.len() {
            dump_out = Some(PathBuf::from(&args[i + 1]));
            i += 2;
        } else if args[i] == "--out" && i + 1 < args.len() {
            out_path = Some(PathBuf::from(&args[i + 1]));
            i += 2;
        } else {
            eprintln!("Unknown arg: {}", args[i]);
            i += 1;
        }
    }
    let dump_out = dump_out.unwrap_or_else(|| PathBuf::from("/tmp/cvvdp_p8b_diffmap_dump.tsv"));
    let out_path = out_path
        .unwrap_or_else(|| PathBuf::from("benchmarks/cvvdp_diffmap_distribution_2026-05-25.tsv"));

    eprintln!(
        "[cvvdp_phase8b_diffmap_distribution] dump_out={} out={}",
        dump_out.display(),
        out_path.display()
    );

    // Clean prior dump file: O_APPEND would otherwise concat across runs.
    if dump_out.exists() {
        std::fs::remove_file(&dump_out)?;
    }

    // Set the env var BEFORE any encode begins so the backend dump hook fires.
    // SAFETY: single-threaded main, no concurrent env access. We are in fn
    // main, before any threads (rayon worker pool included) are spawned;
    // the cvvdp / butter backends only consult these vars from their own
    // compare_with_reference path, which runs synchronously in the same
    // thread that issued the encode call.
    unsafe {
        std::env::set_var("JXL_PHASE8B_DIFFMAP_DUMP", &dump_out);
    }
    // Disable renormalization for the capture run so we measure raw values.
    // Production runs (with the constant default) get a different ratio
    // post-renorm — that's the C_GPU_POST row when present.
    // SAFETY: see above.
    unsafe {
        std::env::set_var("JXL_CVVDP_DIFFMAP_RENORM_SCALE", "1.0");
    }

    // Encode all cells, writing a marker row INTO THE DUMP between cells
    // so we can later tag the rows by (image, distance, backend). The
    // dump backend tag is the C_GPU/_PRE/_POST/B_CPU label; we use the
    // mid-file order to associate consecutive rows of the same backend
    // with the most-recent encode setup.
    let mut tagged_rows: Vec<DumpRow> = Vec::new();
    let mut encode_log: Vec<(String, String, f32, String, usize, f64)> = Vec::new();

    for (corpus, name) in FIXTURES {
        eprintln!("[cvvdp_phase8b] loading {corpus}/{name}");
        let (pixels, w, h) = load_source(corpus, name);
        for &d in DISTANCES {
            // Record the row range before this (corpus, name, d, backend)
            // encode begins, so we can tag the new rows after the encode.

            // 1) Butteraugli (B_CPU). Mark before-position in the dump.
            let pre_b = dump_out_size(&dump_out);
            let (b_bytes, b_wall) = encode_cell(&pixels, w, h, d, false)?;
            let post_b = dump_out_size(&dump_out);
            encode_log.push((
                format!("{corpus}/{name}"),
                "B_CPU".to_string(),
                d,
                "OK".to_string(),
                b_bytes.len(),
                b_wall,
            ));
            tag_rows_in_range(
                &dump_out,
                &mut tagged_rows,
                pre_b,
                post_b,
                &format!("{corpus}/{name}|d={d}|B"),
            );

            // 2) CVVDP (C_GPU / silent-fallback). Mark before-position.
            let pre_c = dump_out_size(&dump_out);
            let (c_bytes, c_wall) = encode_cell(&pixels, w, h, d, true)?;
            let post_c = dump_out_size(&dump_out);
            encode_log.push((
                format!("{corpus}/{name}"),
                "C".to_string(),
                d,
                "OK".to_string(),
                c_bytes.len(),
                c_wall,
            ));
            tag_rows_in_range(
                &dump_out,
                &mut tagged_rows,
                pre_c,
                post_c,
                &format!("{corpus}/{name}|d={d}|C"),
            );

            eprintln!(
                "  {corpus}/{name} d={d:.2}: B={} bytes ({:.0}ms), C={} bytes ({:.0}ms), Δ%={:.1}",
                b_bytes.len(),
                b_wall,
                c_bytes.len(),
                c_wall,
                100.0 * (c_bytes.len() as f64 - b_bytes.len() as f64) / b_bytes.len() as f64
            );
        }
    }

    // Clear the env so nothing downstream gets confused.
    // SAFETY: still single-threaded at this point (all rayon work happened
    // inside the synchronous encode calls above which have returned).
    unsafe {
        std::env::remove_var("JXL_PHASE8B_DIFFMAP_DUMP");
        std::env::remove_var("JXL_CVVDP_DIFFMAP_RENORM_SCALE");
    }

    // Aggregate: per (image, d) mean of B_CPU rows vs mean of C_GPU_PRE rows.
    // Per-(image, d) ratio = cvvdp_mean / butter_mean.
    let mut by_cell: HashMap<String, (Vec<f64>, Vec<f64>)> = HashMap::new();
    for row in &tagged_rows {
        // Extract (cell, backend) from tag = "corpus/name|d=X|B" or |C
        let parts: Vec<&str> = row.tag.split('|').collect();
        if parts.len() < 3 {
            continue;
        }
        let cell = format!("{}|{}", parts[0], parts[1]);
        let entry = by_cell.entry(cell).or_default();
        match row.backend.as_str() {
            "B_CPU" => entry.0.push(row.mean),
            "C_GPU_PRE" | "C_CPU_PRE" => entry.1.push(row.mean),
            // Skip POST (renorm OFF for the capture; POST == PRE here).
            _ => {}
        }
    }

    // Write aggregate TSV.
    let mut out = File::create(&out_path)?;
    writeln!(
        out,
        "cell\tn_iters_b\tn_iters_c\tmean_b\tmean_c\tratio_c_over_b"
    )?;

    let mut all_ratios: Vec<f64> = Vec::new();
    let mut sum_b_mean = 0.0_f64;
    let mut sum_c_mean = 0.0_f64;
    let mut cell_keys: Vec<&String> = by_cell.keys().collect();
    cell_keys.sort();
    for cell in cell_keys {
        let (b_means, c_means) = &by_cell[cell];
        if b_means.is_empty() || c_means.is_empty() {
            continue;
        }
        let mean_b = b_means.iter().copied().sum::<f64>() / b_means.len() as f64;
        let mean_c = c_means.iter().copied().sum::<f64>() / c_means.len() as f64;
        let ratio = if mean_b > 0.0 {
            mean_c / mean_b
        } else {
            f64::NAN
        };
        writeln!(
            out,
            "{cell}\t{}\t{}\t{:.6e}\t{:.6e}\t{:.4}",
            b_means.len(),
            c_means.len(),
            mean_b,
            mean_c,
            ratio
        )?;
        if ratio.is_finite() {
            all_ratios.push(ratio);
            sum_b_mean += mean_b;
            sum_c_mean += mean_c;
        }
    }
    drop(out);

    // Print aggregate.
    if !all_ratios.is_empty() {
        all_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ratio = all_ratios[all_ratios.len() / 2];
        let n = all_ratios.len();
        let p25 = all_ratios[(n * 25 / 100).min(n - 1)];
        let p75 = all_ratios[(n * 75 / 100).min(n - 1)];
        let geo_mean = (all_ratios.iter().map(|r| r.ln()).sum::<f64>() / n as f64).exp();
        let global_ratio = sum_c_mean / sum_b_mean.max(f64::MIN_POSITIVE);
        let suggested_renorm = 1.0 / median_ratio;
        eprintln!();
        eprintln!("======== Phase 8b aggregate ========");
        eprintln!(
            "n cells: {} (encoded {} fixtures × {} distances)",
            all_ratios.len(),
            FIXTURES.len(),
            DISTANCES.len()
        );
        eprintln!("median cvvdp/butter mean ratio: {:.4}", median_ratio);
        eprintln!("p25..p75 range:                 [{:.4}, {:.4}]", p25, p75);
        eprintln!("global cvvdp/butter mean ratio: {:.4}", global_ratio);
        eprintln!("geometric mean ratio:           {:.4}", geo_mean);
        eprintln!();
        eprintln!(
            "===> Suggested Phase 8c CVVDP_DIFFMAP_RENORM_SCALE = 1 / median = {:.6}",
            suggested_renorm
        );
        eprintln!();
        eprintln!("Dump: {}", dump_out.display());
        eprintln!("Aggregate: {}", out_path.display());
    } else {
        eprintln!("[cvvdp_phase8b] no usable rows captured!");
    }

    Ok(())
}

// Return byte-size of the dump file, or 0 if missing. Used to mark
// pre/post encode positions in the file so we can later identify which
// rows came from which encode.
fn dump_out_size(path: &PathBuf) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

// Read the dump file from `pre_offset` to `post_offset`, parse the
// rows, tag them with `tag`, and append to `accumulator`.
fn tag_rows_in_range(
    path: &PathBuf,
    accumulator: &mut Vec<DumpRow>,
    pre_offset: u64,
    post_offset: u64,
    tag: &str,
) {
    if post_offset <= pre_offset {
        return;
    }
    let f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return,
    };
    use std::io::{Read, Seek, SeekFrom};
    let mut f = f;
    if f.seek(SeekFrom::Start(pre_offset)).is_err() {
        return;
    }
    let mut chunk = vec![0u8; (post_offset - pre_offset) as usize];
    if f.read_exact(&mut chunk).is_err() {
        return;
    }
    let s = String::from_utf8_lossy(&chunk);
    for line in s.lines() {
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 12 {
            continue;
        }
        accumulator.push(DumpRow {
            backend: parts[0].to_string(),
            _compare_call: parts[1].parse().unwrap_or(0),
            _width: parts[2].parse().unwrap_or(0),
            _height: parts[3].parse().unwrap_or(0),
            _n: parts[4].parse().unwrap_or(0),
            mean: parts[5].parse().unwrap_or(0.0),
            _median: parts[6].parse().unwrap_or(0.0),
            _p25: parts[7].parse().unwrap_or(0.0),
            _p75: parts[8].parse().unwrap_or(0.0),
            _p95: parts[9].parse().unwrap_or(0.0),
            _max: parts[10].parse().unwrap_or(0.0),
            _score: parts[11].parse().unwrap_or(0.0),
            tag: tag.to_string(),
        });
    }
    // Suppress unused-function warning by referencing the standalone parser.
    let _ = parse_dump;
}
