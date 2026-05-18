//! W44-21 F-D Sub-I: per-position cost-input dump at chosen 32x32 wedge positions.
//!
//! Per the W44-21 plan: the W44-19 verdict (sub-chunk H) localised the F-D
//! wedge to the *cost-input side* of strategy selection (W44-15/17 confirmed
//! the comparison logic at parity, W44-11 confirmed post-selection
//! quantization at parity). This chunk dumps every cost-input component for
//! every 4-aligned 32x32 candidate region on the F-D wedge image
//! (`1531677.png` e7 d=6) so the divergent component can be pinpointed.
//!
//! The dump infrastructure lives in `vardct/ac_strategy_search.rs` —
//! `find_best_32x32_transform_impl` checks the `JXL_DUMP_POS=bx,by`
//! environment variable and writes a full cost-component breakdown
//! to stderr whenever its candidate 32x32 region matches `(bx, by)`.
//!
//! For each 4-aligned (bx, by) in [4..60] × [4..60] we capture the rect
//! decomposition vs square-transform cost gap and emit TSV columns:
//!   - bx, by
//!   - entropy_32x32 (DCT32X32 cost)
//!   - best_rival = min(cost_sub, cost_jxn, cost_nxj)
//!   - delta_pct = (entropy_32x32 - best_rival) / best_rival * 100
//!   - decision_32x32_wins (true if entropy_32x32 < min(cost_jxn, cost_nxj))
//!
//! The "borderline" wedge positions (delta_pct ∈ [0, +2%]) are the ones where
//! a small numerical bias in any cost-input could flip ours' decision from
//! "JxN" (or "sub") to "32x32" — matching cjxl's pick.
//!
//! W44-19 found 64 ours-Dct8 cells inside 8 cjxl-Dct32 parents on this image
//! (8 cells per parent). At ~4-5 of the ~50 4-aligned 32x32 candidates we'd
//! expect borderline-loss patterns. The exact root-cause input (entropy_mul
//! table values, mask1x1 propagation, quant_norm16 FMA ordering, etc.) is
//! NOT identifiable from ours' values alone — that requires patching libjxl
//! with mirror logging, which is out of scope for this read-only chunk
//! (see the W44-21 memory note for the recommended next steps).
//!
//! Run:
//! ```bash
//! cargo build -p jxl-encoder-cli --release
//! cargo run -p jxl-encoder --release \
//!   --example f_d_sub_i_cost_input_dump -- \
//!   --output benchmarks/f_d_sub_i_cost_input_dump_2026-05-19.tsv
//! ```
//!
//! Read-only investigation. Sibling workspace mandatory.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const IMAGE: &str = "1531677.png";
const EFFORT: u8 = 7;
const DISTANCE: f32 = 6.0;

fn corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/lilith/work/codec-corpus"))
}

fn image_path() -> PathBuf {
    corpus_dir()
        .join("CID22/CID22-512/validation")
        .join(IMAGE)
}

fn cjxl_rs_bin() -> PathBuf {
    // Prefer locally-built binary in the sibling workspace's target dir.
    let p = PathBuf::from("target/release/cjxl-rs");
    if p.exists() {
        p
    } else {
        PathBuf::from("./target/release/cjxl-rs")
    }
}

fn work_dir() -> PathBuf {
    PathBuf::from("/tmp/f_d_sub_i_cost_input_dump")
}

#[derive(Debug, Default)]
struct DumpResult {
    entropy_32x32: f32,
    entropy_32x16_0: f32,
    entropy_32x16_1: f32,
    entropy_16x32_0: f32,
    entropy_16x32_1: f32,
    cost_sub: f32,
    cost_jxn: f32,
    cost_nxj: f32,
    decision_32x32_wins: bool,
    decision_jxn_better_than_nxj: bool,
    visited: bool,
}

fn parse_dump(stderr: &str) -> DumpResult {
    let mut r = DumpResult::default();
    for line in stderr.lines() {
        if line.contains("=== W44-21 DUMP at") {
            r.visited = true;
        }
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("entropy_32x32=") {
            if let Some(v) = rest.split_whitespace().next() {
                if let Ok(f) = v.parse::<f32>() { r.entropy_32x32 = f; }
            }
        }
        if let Some(rest) = trimmed.strip_prefix("entropy_32x16_0(left)=") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if let Some(v) = parts.first() {
                if let Ok(f) = v.parse::<f32>() { r.entropy_32x16_0 = f; }
            }
            for p in &parts {
                if let Some(s) = p.strip_prefix("entropy_32x16_1(right)=") {
                    if let Ok(f) = s.parse::<f32>() { r.entropy_32x16_1 = f; }
                }
            }
        }
        if let Some(rest) = trimmed.strip_prefix("entropy_16x32_0(top)=") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if let Some(v) = parts.first() {
                if let Ok(f) = v.parse::<f32>() { r.entropy_16x32_0 = f; }
            }
            for p in &parts {
                if let Some(s) = p.strip_prefix("entropy_16x32_1(bot)=") {
                    if let Ok(f) = s.parse::<f32>() { r.entropy_16x32_1 = f; }
                }
            }
        }
        if let Some(rest) = trimmed.strip_prefix("cost_sub=") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if let Some(v) = parts.first() {
                if let Ok(f) = v.parse::<f32>() { r.cost_sub = f; }
            }
            for p in &parts {
                if let Some(s) = p.strip_prefix("cost_jxn=") {
                    if let Ok(f) = s.parse::<f32>() { r.cost_jxn = f; }
                }
                if let Some(s) = p.strip_prefix("cost_nxj=") {
                    if let Ok(f) = s.parse::<f32>() { r.cost_nxj = f; }
                }
            }
        }
        if trimmed.starts_with("DECISION:") {
            r.decision_32x32_wins = trimmed.contains("32x32_wins=true");
            r.decision_jxn_better_than_nxj = trimmed.contains("jxN_better_than_nxJ=true");
        }
    }
    r
}

fn run_at(bx: usize, by: usize, work: &Path) -> Option<DumpResult> {
    let bin = cjxl_rs_bin();
    if !bin.exists() {
        eprintln!("cjxl-rs not found at {} — build first", bin.display());
        return None;
    }
    let out_jxl = work.join(format!("dump_{bx}_{by}.jxl"));
    let pos = format!("{bx},{by}");
    let r = Command::new(&bin)
        .env("JXL_DUMP_POS", &pos)
        .arg(image_path())
        .arg(&out_jxl)
        .args(["-d", &DISTANCE.to_string()])
        .args(["-e", &EFFORT.to_string()])
        .output()
        .ok()?;
    let _ = fs::remove_file(&out_jxl);
    let stderr = String::from_utf8_lossy(&r.stderr).to_string();
    Some(parse_dump(&stderr))
}

fn main() {
    let mut output = PathBuf::from("benchmarks/f_d_sub_i_cost_input_dump_2026-05-19.tsv");
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--output" && i + 1 < args.len() {
            output = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }
    let work = work_dir();
    fs::create_dir_all(&work).expect("create work dir");
    if let Some(p) = output.parent() {
        fs::create_dir_all(p).expect("create output parent");
    }

    println!("W44-21 F-D Sub-I: per-position cost-input dump");
    println!("Image: {}  effort=e{EFFORT}  distance={DISTANCE}", IMAGE);
    println!("Output: {}", output.display());

    // All 4-aligned (bx, by) in [4..60] × [4..60]
    let mut rows = BTreeMap::new();
    for by in (4..=60).step_by(4) {
        for bx in (4..=60).step_by(4) {
            if let Some(r) = run_at(bx, by, &work) {
                if r.visited {
                    rows.insert((by, bx), r);
                }
            }
        }
    }

    let mut tsv = String::new();
    tsv.push_str("bx\tby\tentropy_32x32\tentropy_32x16_0\tentropy_32x16_1\tentropy_16x32_0\tentropy_16x32_1\tcost_sub\tcost_jxn\tcost_nxj\tbest_rival\tdelta_pct\tdec_32x32_wins\tdec_jxn_better\n");
    let mut borderline = 0;
    let mut total = 0;
    for ((by, bx), r) in &rows {
        let best_rival = r.cost_sub.min(r.cost_jxn).min(r.cost_nxj);
        let delta_pct = (r.entropy_32x32 - best_rival) / best_rival * 100.0;
        tsv.push_str(&format!(
            "{bx}\t{by}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.4}\t{}\t{}\n",
            r.entropy_32x32, r.entropy_32x16_0, r.entropy_32x16_1, r.entropy_16x32_0, r.entropy_16x32_1,
            r.cost_sub, r.cost_jxn, r.cost_nxj, best_rival, delta_pct,
            r.decision_32x32_wins, r.decision_jxn_better_than_nxj
        ));
        total += 1;
        if (0.0..=2.0).contains(&delta_pct) && !r.decision_32x32_wins {
            borderline += 1;
        }
    }

    fs::write(&output, &tsv).expect("write tsv");
    println!("\nVisited {total} 4-aligned 32x32 positions");
    println!("Borderline cases (32x32 loses by ≤2%): {borderline}");
    println!("These are the most likely cjxl-vs-ours wedge candidates.");

    // Meta file
    let meta_path = output.with_extension("meta");
    let meta = format!(
        "# W44-21 F-D Sub-I: per-position cost-input dump\n\
         # Image: {IMAGE} e{EFFORT} d={DISTANCE}\n\
         # Borderline-loss positions (delta_pct in [0,+2%], 32x32 loses):\n\
         #   These are candidates where a small numerical bias could flip the\n\
         #   decision from JxN/NxJ to 32x32 (matching cjxl's pick on the F-D\n\
         #   wedge).\n\
         # Read-only — no source mutation; the dump path is gated on\n\
         # `JXL_DUMP_POS={{bx,by}}` env var only.\n\
         total_positions: {total}\n\
         borderline_count: {borderline}\n"
    );
    fs::write(&meta_path, meta).expect("write meta");
    println!("Meta: {}", meta_path.display());
}
