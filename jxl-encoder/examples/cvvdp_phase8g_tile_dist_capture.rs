// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! cvvdp-fork Phase 8g (2026-05-25) — per-block reducer tile_dist
//! distribution capture harness.
//!
//! For ~5 representative cells × 4 distances × 2 backends (butter +
//! cvvdp + Phase-8c renorm), encode each cell and dump the per-iter
//! `tile_dist` distribution stats. The Phase 8g `k_tile_norm`
//! refit reads the post-renorm cvvdp distribution and compares against
//! butteraugli's distribution at the SAME nominal distance band. The
//! load-bearing metric is the per-iter `bad_rate` = fraction of blocks
//! where `tile_dist > effective_metric_target_distance` (the
//! `diff > 1.0` predicate fire rate). Pareto-optimal cvvdp tuning is
//! when `bad_rate_c ≈ bad_rate_b` AT EACH iter index — that's when
//! cvvdp drives the same per-block adjustment magnitude as butteraugli
//! would for the same nominal quality target.
//!
//! ## Per-image scope
//!
//! Mirrors Phase 8c/8d: 3 CID22 photos + 2 GB82-SC screenshots × 4
//! distances {0.5, 1.0, 2.0, 3.0} × effort 8.
//!
//! ## Output
//!
//! 1. Raw per-iter dump TSV at `--dump-out` (one row per iter per
//!    encode across all encodes; written by
//!    `maybe_dump_tile_dist_stats_phase8g` in
//!    `vardct/perceptual_loop.rs`).
//! 2. Aggregate per-(image, d, backend) summary TSV at `--out`.
//!
//! ## Run
//!
//! ```bash
//! JXL_PHASE8G_TILE_DIST_DUMP=/tmp/cvvdp_p8g_tile_dist.tsv \
//!   cargo run --release -p jxl-encoder \
//!   --features '__expert butteraugli-loop cvvdp-loop ssim2-loop parallel' \
//!   --example cvvdp_phase8g_tile_dist_capture -- \
//!   --dump-out /tmp/cvvdp_p8g_tile_dist.tsv \
//!   --out benchmarks/cvvdp_block_signal_distribution_2026-05-25.tsv
//! ```

#![cfg(all(feature = "cvvdp-loop", feature = "butteraugli-loop"))]

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

use jxl_encoder::api::EncoderStrategy;
use jxl_encoder::{LossyConfig, PixelLayout};

const CID22_DIR: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82_SC_DIR: &str = "/home/lilith/work/codec-corpus/gb82-sc";

// Mirror Phase 8b/8c fixture set.
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
        .with_cvvdp_loop(if cvvdp_opt_in { Some(true) } else { None });
    let t = Instant::now();
    let encoded = cfg.encode(pixels, w, h, PixelLayout::Rgb8)?;
    let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
    Ok((encoded, wall_ms))
}

#[derive(Debug, Clone)]
struct DumpRow {
    backend: String,
    iter: u32,
    effective_target: f32,
    target_distance: f32,
    nblocks: u64,
    td_min: f32,
    td_max: f32,
    td_median: f32,
    td_p25: f32,
    td_p75: f32,
    td_p95: f32,
    td_mean: f64,
    bad_rate: f64,
    // Synthetic tag set by the harness via row ordering.
    tag: String,
}

fn parse_dump(path: &PathBuf) -> Vec<DumpRow> {
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) => panic!("Phase 8g dump file missing at {}: {e}", path.display()),
    };
    let mut rows = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.is_empty() || line.starts_with("backend\t") {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 13 {
            continue;
        }
        let row = DumpRow {
            backend: parts[0].to_string(),
            iter: parts[1].parse().unwrap_or(0),
            effective_target: parts[2].parse().unwrap_or(0.0),
            target_distance: parts[3].parse().unwrap_or(0.0),
            nblocks: parts[4].parse().unwrap_or(0),
            td_min: parts[5].parse().unwrap_or(0.0),
            td_max: parts[6].parse().unwrap_or(0.0),
            td_median: parts[7].parse().unwrap_or(0.0),
            td_p25: parts[8].parse().unwrap_or(0.0),
            td_p75: parts[9].parse().unwrap_or(0.0),
            td_p95: parts[10].parse().unwrap_or(0.0),
            td_mean: parts[11].parse().unwrap_or(0.0),
            bad_rate: parts[12].parse().unwrap_or(0.0),
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
    let dump_out = dump_out.unwrap_or_else(|| PathBuf::from("/tmp/cvvdp_p8g_tile_dist.tsv"));
    let out_path = out_path.unwrap_or_else(|| {
        PathBuf::from("benchmarks/cvvdp_block_signal_distribution_2026-05-25.tsv")
    });

    // Ensure the dump path is empty / fresh.
    if dump_out.exists() {
        std::fs::remove_file(&dump_out).ok();
    }

    // Force the dump hook on for the rest of the process via env.
    // SAFETY: single-threaded harness setup before any compute; no other
    // thread reads the env hook during this call.
    // (set_var is unsafe in 2024 edition but the harness controls when.)
    unsafe {
        std::env::set_var("JXL_PHASE8G_TILE_DIST_DUMP", &dump_out);
    }

    println!(
        "Phase 8g tile_dist capture: dump-out={} out={}",
        dump_out.display(),
        out_path.display()
    );

    // Encode each cell under both backends. Order them so the dump
    // file's row order tracks (corpus, name, d, backend).
    let mut tagged_runs: Vec<(String, String, f32, &'static str, usize)> = Vec::new();
    let row_cursor = 0usize;

    for (corpus, name) in FIXTURES {
        let (pixels, w, h) = load_source(corpus, name);
        for &d in DISTANCES {
            for &(opt_in, backend_tag) in &[(false, "butter"), (true, "cvvdp")] {
                // Record where rows for this run will start in the dump.
                let before = std::fs::metadata(&dump_out).map(|m| m.len()).unwrap_or(0);
                let (_enc, wall_ms) = encode_cell(&pixels, w, h, d, opt_in)?;
                let after = std::fs::metadata(&dump_out).map(|m| m.len()).unwrap_or(0);
                println!(
                    "{}/{} d={} backend={} wall={:.1}ms bytes_written={}",
                    corpus,
                    name,
                    d,
                    backend_tag,
                    wall_ms,
                    after - before
                );
                tagged_runs.push((
                    corpus.to_string(),
                    name.to_string(),
                    d,
                    backend_tag,
                    row_cursor,
                ));
                // We'll compute actual row counts after parse.
                let _ = row_cursor; // placeholder
            }
        }
    }

    // Parse the dump and aggregate.
    let mut rows = parse_dump(&dump_out);
    println!("Parsed {} dump rows", rows.len());

    // Order is deterministic: per encode, rows are appended iter=0..iters
    // for that encode only (single thread per encode body). We replay
    // the encode order to tag rows: per tagged_runs entry, consume
    // exactly the rows tagged with the matching backend until we see
    // the next-cell switch in iter count (iter resets to 0). We tag
    // by tag = "{corpus}|{name}|d={d}|backend={backend}".
    let mut tag_idx = 0usize;
    let mut prev_iter: i32 = -1;
    for r in rows.iter_mut() {
        if tag_idx >= tagged_runs.len() {
            break;
        }
        // Reset on iter rollover (iter=0 after iter>0 means new encode).
        if (r.iter as i32) <= prev_iter {
            // Move to next tag.
            tag_idx += 1;
            if tag_idx >= tagged_runs.len() {
                break;
            }
        }
        let (corpus, name, d, backend, _) = &tagged_runs[tag_idx];
        // Validate the backend matches what we expect on this row.
        // Mismatch means dump-row ordering diverged from encode order
        // (shouldn't happen single-threaded; defensive only).
        if r.backend != *backend {
            eprintln!(
                "warning: dump row backend {} != expected {} at tag_idx={}",
                r.backend, backend, tag_idx
            );
        }
        r.tag = format!("{corpus}|{name}|d={d}|backend={backend}");
        prev_iter = r.iter as i32;
    }

    // Write aggregate output with the tag column.
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out_file = std::fs::File::create(&out_path)?;
    writeln!(
        out_file,
        "tag\tbackend\titer\teffective_target\ttarget_distance\tnblocks\ttd_min\ttd_max\ttd_median\ttd_p25\ttd_p75\ttd_p95\ttd_mean\tbad_rate"
    )?;
    for r in &rows {
        writeln!(
            out_file,
            "{tag}\t{backend}\t{iter}\t{etgt}\t{tgt}\t{n}\t{minv}\t{maxv}\t{med}\t{p25}\t{p75}\t{p95}\t{mean:.6}\t{br:.6}",
            tag = r.tag,
            backend = r.backend,
            iter = r.iter,
            etgt = r.effective_target,
            tgt = r.target_distance,
            n = r.nblocks,
            minv = r.td_min,
            maxv = r.td_max,
            med = r.td_median,
            p25 = r.td_p25,
            p75 = r.td_p75,
            p95 = r.td_p95,
            mean = r.td_mean,
            br = r.bad_rate,
        )?;
    }
    println!("Wrote {} tagged rows to {}", rows.len(), out_path.display());

    // Per-cell summary: butter vs cvvdp bad_rate ratio at iter=0
    // (the iter that drives the largest per-block adjustment). Median
    // ratio across all cells suggests the linear scale fit.
    println!("\n## Per-cell summary (iter=0 only, the convergence-driving iter)");
    println!(
        "cell\td\tbadrate_butter\tbadrate_cvvdp\tratio_b_over_c\tmean_butter\tmean_cvvdp\ttd_ratio_b_over_c"
    );
    let mut ratios: Vec<f64> = Vec::new();
    let mut td_ratios: Vec<f64> = Vec::new();
    let mut bad_ratios: Vec<f64> = Vec::new();
    let mut per_distance_bad_butter: std::collections::BTreeMap<u32, Vec<f64>> =
        std::collections::BTreeMap::new();
    let mut per_distance_bad_cvvdp: std::collections::BTreeMap<u32, Vec<f64>> =
        std::collections::BTreeMap::new();
    for (corpus, name) in FIXTURES {
        for &d in DISTANCES {
            let tag_b = format!("{corpus}|{name}|d={d}|backend=butter");
            let tag_c = format!("{corpus}|{name}|d={d}|backend=cvvdp");
            let butter_row = rows.iter().find(|r| r.tag == tag_b && r.iter == 0);
            let cvvdp_row = rows.iter().find(|r| r.tag == tag_c && r.iter == 0);
            if let (Some(b), Some(c)) = (butter_row, cvvdp_row) {
                let ratio_bc = if c.bad_rate > 0.0 {
                    b.bad_rate / c.bad_rate
                } else {
                    f64::NAN
                };
                let mean_ratio = if c.td_mean > 0.0 {
                    b.td_mean / c.td_mean
                } else {
                    f64::NAN
                };
                println!(
                    "{}/{}\t{}\t{:.4}\t{:.4}\t{:.3}\t{:.6}\t{:.6}\t{:.3}",
                    corpus,
                    name,
                    d,
                    b.bad_rate,
                    c.bad_rate,
                    ratio_bc,
                    b.td_mean,
                    c.td_mean,
                    mean_ratio,
                );
                if ratio_bc.is_finite() {
                    ratios.push(ratio_bc);
                    bad_ratios.push(ratio_bc);
                }
                if mean_ratio.is_finite() {
                    td_ratios.push(mean_ratio);
                }
                // Bad-rate scale at this distance.
                let d_key = (d * 1000.0) as u32;
                per_distance_bad_butter
                    .entry(d_key)
                    .or_default()
                    .push(b.bad_rate);
                per_distance_bad_cvvdp
                    .entry(d_key)
                    .or_default()
                    .push(c.bad_rate);
            }
        }
    }

    let median = |mut xs: Vec<f64>| -> Option<f64> {
        if xs.is_empty() {
            None
        } else {
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            Some(xs[xs.len() / 2])
        }
    };
    if let Some(m) = median(ratios.clone()) {
        println!(
            "\nMedian (butter_bad_rate / cvvdp_bad_rate) across all cells iter=0: {:.3}",
            m
        );
    }
    if let Some(m) = median(td_ratios.clone()) {
        println!(
            "Median (butter_td_mean / cvvdp_td_mean) across all cells iter=0: {:.3}",
            m
        );
        println!(
            "Suggested k_tile_norm SCALE for cvvdp: ~{:.3} (multiplier on butteraugli's 1.2)",
            m
        );
    }

    println!("\n## Per-distance bad_rate summary (median across fixtures)");
    println!("distance\tn\tbutter_bad_rate_med\tcvvdp_bad_rate_med\tratio_b_over_c");
    for (&d_key, butter_rates) in &per_distance_bad_butter {
        let cvvdp_rates = per_distance_bad_cvvdp
            .get(&d_key)
            .cloned()
            .unwrap_or_default();
        let mb = median(butter_rates.clone()).unwrap_or(0.0);
        let mc = median(cvvdp_rates).unwrap_or(0.0);
        let r = if mc > 0.0 { mb / mc } else { f64::NAN };
        println!(
            "{}\t{}\t{:.4}\t{:.4}\t{:.3}",
            d_key as f32 / 1000.0,
            butter_rates.len(),
            mb,
            mc,
            r,
        );
    }

    Ok(())
}
