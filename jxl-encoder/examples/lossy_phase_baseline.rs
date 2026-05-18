//! Per-effort, per-phase wall-clock baseline for lossy encode.
//!
//! Profiles the default lossy encode pipeline across e6/e7/e8 and a
//! representative corpus (5 CID22-512 photos + 3 GB82-SC screenshots) at
//! d in {0.5, 1.0, 2.0, 4.0}. Each (image, effort, distance) cell is
//! encoded 3 times and the median wall-clock per phase is kept.
//!
//! Instrumentation: per-phase timings come from the existing
//! `__JXL_ENC_PHASE_TIMING` env-var path in `vardct::encoder::encode_inner`
//! and `vardct::bitstream::encode_two_pass_to_writer`, which both emit
//! a single `eprintln!` line per encode when the env var is set. This
//! example captures those lines via a wrapper `Stderr` redirect-based
//! approach: it runs each encode in-process and parses the captured
//! stderr.
//!
//! Output: `benchmarks/lossy_phase_baseline_2026-05-18.tsv` (+ .meta).
//!
//! Reproducer:
//!   cargo run -p jxl-encoder --release --features 'std parallel butteraugli-loop' \
//!     --example lossy_phase_baseline -- \
//!     --out benchmarks/lossy_phase_baseline_2026-05-18.tsv

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use jxl_encoder::{LossyConfig, PixelLayout};

const PHASES: &[&str] = &[
    "total",
    "xyb",
    "patches",
    "splines",
    "quant_field",
    "gaborish",
    "cfl1",
    "acstrat",
    "cfl2",
    "buttloop",
    "xform",
    "sharp",
    "entropy",
    "tok_dc",
    "co",
    "bcm",
    "ac_tok",
    "lz77",
    "build_codes",
    "pass2_write",
];

fn main() {
    // Detect worker subprocess mode first — must happen before printing the
    // parent's corpus banner so subprocess stdout/stderr stays clean.
    if std::env::var("__JXL_PHASE_WORKER").is_ok() {
        let img_path = std::env::var("__W_IMG").unwrap();
        let effort: u8 = std::env::var("__W_EFFORT").unwrap().parse().unwrap();
        let distance: f32 = std::env::var("__W_DIST").unwrap().parse().unwrap();
        worker_encode(&img_path, effort, distance);
        return;
    }

    // Parse --out path
    let args: Vec<String> = std::env::args().collect();
    let mut out_path = PathBuf::from("benchmarks/lossy_phase_baseline_2026-05-18.tsv");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--out" && i + 1 < args.len() {
            out_path = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }
    let runs_per_cell = 3usize;
    let efforts: &[u8] = &[6, 7, 8];
    let distances: &[f32] = &[0.5, 1.0, 2.0, 4.0];

    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let cid_dir = format!("{}/work/codec-corpus/CID22/CID22-512/validation", base);
    let scrn_dir = format!("{}/work/codec-corpus/gb82-sc", base);

    // Pick a fixed deterministic subset (alphabetical, first 5 / first 3).
    let cid_names = [
        "1025469.png",
        "1044329.png",
        "1189261.png",
        "1279330.png",
        "1418519.png",
    ];
    let scrn_names = ["codec_wiki.png", "imac_g3.png", "terminal.png"];

    let mut images: Vec<(String, String)> = Vec::new();
    for n in &cid_names {
        images.push((format!("cid22/{}", n), format!("{}/{}", cid_dir, n)));
    }
    for n in &scrn_names {
        images.push((format!("scrn/{}", n), format!("{}/{}", scrn_dir, n)));
    }

    eprintln!(
        "Corpus: {} images x {} efforts x {} distances x {} runs = {} encodes",
        images.len(),
        efforts.len(),
        distances.len(),
        runs_per_cell,
        images.len() * efforts.len() * distances.len() * runs_per_cell
    );

    // Build header
    let mut rows: Vec<String> = Vec::new();
    let mut header = String::from("image\teffort\tdistance\twidth\theight\tbytes");
    for p in PHASES {
        header.push('\t');
        header.push_str(&format!("ms_{}", p));
    }
    rows.push(header);

    let bin_path = std::env::current_exe().unwrap();
    eprintln!(
        "Using subprocess to capture per-phase eprintln: {}",
        bin_path.display()
    );

    let total_cells = images.len() * efforts.len() * distances.len();
    let mut cell_idx = 0usize;
    let t_start = Instant::now();
    for (img_label, img_path) in &images {
        // Pre-load to get dims
        let img = image::open(img_path).expect("open");
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        for &effort in efforts {
            for &distance in distances {
                cell_idx += 1;
                eprintln!(
                    "[{}/{}] {} e{} d={} ({}x{})",
                    cell_idx, total_cells, img_label, effort, distance, w, h
                );
                // Run N times, keep medians per phase
                let mut runs: Vec<PhaseResult> = Vec::with_capacity(runs_per_cell);
                for run in 0..runs_per_cell {
                    let r = subprocess_run(&bin_path, img_path, effort, distance);
                    match r {
                        Ok(pr) => runs.push(pr),
                        Err(e) => {
                            eprintln!("  run {} ERR: {}", run, e);
                        }
                    }
                }
                if runs.is_empty() {
                    eprintln!("  all runs failed, skipping cell");
                    continue;
                }
                let med = median_phases(&runs);
                let bytes_med = median_bytes(&runs);
                let mut row = format!(
                    "{}\t{}\t{}\t{}\t{}\t{}",
                    img_label, effort, distance, w, h, bytes_med
                );
                for p in PHASES {
                    let v = med.get(*p).copied().unwrap_or(0.0);
                    row.push('\t');
                    row.push_str(&format!("{:.2}", v));
                }
                rows.push(row);
            }
        }
    }
    let dur = t_start.elapsed();
    eprintln!("Sweep done in {:.1}s", dur.as_secs_f64());

    // Write atomically: tmp file then mv
    let tmp = format!(
        "/tmp/lossy_phase_baseline_{}.tsv.partial",
        std::process::id()
    );
    {
        let mut f = fs::File::create(&tmp).expect("create tmp");
        for r in &rows {
            writeln!(f, "{}", r).unwrap();
        }
    }
    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::rename(&tmp, &out_path).expect("rename");
    eprintln!("Wrote {}", out_path.display());
}

#[derive(Default, Clone)]
struct PhaseResult {
    bytes: usize,
    phases: std::collections::HashMap<String, f32>,
}

fn worker_encode(img_path: &str, effort: u8, distance: f32) {
    // Read image, encode once with phase timing, print bytes + timings to stdout
    let img = match image::open(img_path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("WORKER_ERR: {}", e);
            std::process::exit(1);
        }
    };
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());

    // Enable phase timing. Reading this env var is internal to the
    // encoder via `std::env::var_os` once per `encode_inner` call.
    // SAFETY: this is the worker subprocess — single-threaded at this
    // point (no other threads have been spawned yet), so the documented
    // race condition on `std::env::set_var` cannot fire.
    unsafe {
        std::env::set_var("__JXL_ENC_PHASE_TIMING", "1");
    }

    let cfg = LossyConfig::new(distance).with_effort(effort);
    // Warm up nothing — single encode per subprocess is the measurement.
    let t0 = Instant::now();
    let bytes = match cfg.encode(rgb.as_raw(), w, h, PixelLayout::Rgb8) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("WORKER_ENCODE_ERR: {}", e);
            std::process::exit(2);
        }
    };
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Print bytes + wall-clock total — the per-phase eprintln from the
    // encoder lands on stderr, which the parent captures separately.
    println!("BYTES {}", bytes.len());
    println!("TOTAL_MS {:.2}", total_ms);
}

fn subprocess_run(
    bin_path: &std::path::Path,
    img_path: &str,
    effort: u8,
    distance: f32,
) -> Result<PhaseResult, String> {
    let out = Command::new(bin_path)
        .env("__JXL_PHASE_WORKER", "1")
        .env("__W_IMG", img_path)
        .env("__W_EFFORT", effort.to_string())
        .env("__W_DIST", distance.to_string())
        .env("__JXL_ENC_PHASE_TIMING", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("spawn: {}", e))?;
    if !out.status.success() {
        return Err(format!(
            "exit {}: stderr={}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    let mut pr = PhaseResult::default();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("BYTES ") {
            pr.bytes = rest.trim().parse().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("TOTAL_MS ") {
            pr.phases
                .insert("total".into(), rest.trim().parse().unwrap_or(0.0));
        }
    }
    // Parse eprintln lines from encoder
    for line in stderr.lines() {
        // encode_inner: total=X xyb=Y patches=Z ...
        // encode_two_pass: tok_dc=X co=Y ...
        // encode_from_precomputed: ...
        if let Some(rest) = line.strip_prefix("encode_inner:") {
            parse_kv(&mut pr, rest);
        } else if let Some(rest) = line.strip_prefix("encode_two_pass:") {
            parse_kv(&mut pr, rest);
        }
    }
    Ok(pr)
}

fn parse_kv(pr: &mut PhaseResult, line: &str) {
    // tokens look like `key=value` possibly followed by unit ` ms,` or just whitespace
    for tok in line.split_whitespace() {
        let tok = tok.trim_end_matches(',');
        if let Some((k, v)) = tok.split_once('=') {
            // strip trailing "ms" if present
            let v = v.trim_end_matches("ms");
            if let Ok(parsed) = v.parse::<f32>() {
                // For "total" we prefer the wall-clock TOTAL_MS over the
                // encoder's internal `total` (they're equivalent within
                // a few microseconds anyway).
                if k == "total" && pr.phases.contains_key("total") {
                    continue;
                }
                pr.phases.insert(k.to_string(), parsed);
            }
        }
    }
}

fn median_bytes(runs: &[PhaseResult]) -> usize {
    let mut v: Vec<usize> = runs.iter().map(|r| r.bytes).collect();
    v.sort();
    v[v.len() / 2]
}

fn median_phases(runs: &[PhaseResult]) -> std::collections::HashMap<String, f32> {
    let mut out = std::collections::HashMap::new();
    for p in PHASES {
        let mut vals: Vec<f32> = runs
            .iter()
            .filter_map(|r| r.phases.get(*p).copied())
            .collect();
        if vals.is_empty() {
            continue;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = vals[vals.len() / 2];
        out.insert((*p).to_string(), m);
    }
    out
}
