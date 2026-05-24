// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
#![forbid(unsafe_code)]

//! `zenjxl-tuning-runner` — W44-212 single-cell sweep worker.
//!
//! Reads one [`zenjxl_tuning_runner::SweepCellSpec`] from JSON (CLI
//! arg or file), encodes via jxl-encoder, decodes via jxl-rs, scores
//! on GPU (or CPU fallback), writes one Parquet row.
//!
//! ## Usage
//!
//! ```text
//! zenjxl-tuning-runner \
//!     --cell '{"sweep_id":"W44-212-smoke","chunk_claim_id":"c1",
//!              "image_path":"/tmp/test.png","effort":7,"distance":1.0,
//!              "strategy":"zenjxl","metric_backend":"skip"}' \
//!     --output /tmp/out/c1.parquet
//! ```
//!
//! Or pass `--cell-file <path.json>` to avoid shell-escape pain.
//!
//! ## W44-PHASE4-M1 artifact persistence env flags (default OFF)
//!
//! Per the global CLAUDE.md `4. Always persist encoded variants when
//! encoding is expensive — NO EXCEPTIONS` rule (added 2026-05-24
//! after the W44-PHASE4-S1 incident discarded $30 of encoded bytes),
//! **production sweeps MUST set all three to `1`**:
//!
//! | env | what it adds |
//! |---|---|
//! | `W44_PHASE4_M1_SAVE_ENCODED=1` | stage encoded JXL bytes to artifacts dir, populate `encoded_jxl_sha256` + `encoded_jxl_r2_key` cols |
//! | `W44_PHASE4_M1_SAVE_DIFFMAP=1` | stage per-pixel butteraugli diffmap blob, populate `diffmap_r2_key` col |
//! | `W44_PHASE4_M1_COMPUTE_MULTIMETRIC=1` | populate `butter_max/p1/p2/p6` + `psnr_y/r/g/b` cols |
//! | `W44_PHASE4_M1_ARTIFACTS_DIR=<path>` | override the artifact-stage dir (default: `<output_parquet>/../artifacts/`) |
//!
//! The runner only stages artifacts to local disk; `worker.sh`
//! handles the actual R2 upload via `s5cmd cp` (matching the existing
//! per-cell Parquet upload pattern). Smoke tests + CI leave the flags
//! OFF for v1-shaped byte-identical output.
//!
//! Exit codes:
//! - 0 success
//! - 1 cell-spec parse error
//! - 2 cell ran but failed (load/encode/decode/write)
//! - 3 internal panic (Rust's default for `Result::unwrap` etc.)

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use zenjxl_tuning_runner::{SweepCellSpec, run_cell};

#[derive(Parser, Debug)]
#[command(name = "zenjxl-tuning-runner")]
#[command(version, about, long_about = None)]
struct Args {
    /// Inline JSON cell spec. Mutually exclusive with `--cell-file`.
    #[arg(long)]
    cell: Option<String>,

    /// Path to a JSON file containing the cell spec.
    #[arg(long)]
    cell_file: Option<PathBuf>,

    /// Output Parquet file path.
    #[arg(long, short = 'o')]
    output: PathBuf,

    /// Print the parsed cell spec + final row to stderr on success.
    #[arg(long)]
    verbose: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let cell_json = match (&args.cell, &args.cell_file) {
        (Some(s), None) => s.clone(),
        (None, Some(p)) => match std::fs::read_to_string(p) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[w44-212] read --cell-file {}: {e}", p.display());
                return ExitCode::from(1);
            }
        },
        (None, None) => {
            eprintln!("[w44-212] one of --cell or --cell-file is required");
            return ExitCode::from(1);
        }
        (Some(_), Some(_)) => {
            eprintln!("[w44-212] --cell and --cell-file are mutually exclusive");
            return ExitCode::from(1);
        }
    };

    let spec: SweepCellSpec = match serde_json::from_str(&cell_json) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[w44-212] cell JSON parse error: {e}");
            eprintln!("[w44-212] received: {cell_json}");
            return ExitCode::from(1);
        }
    };

    if args.verbose {
        eprintln!(
            "[w44-212] running cell sweep_id={} chunk={} image={} effort={} d={} strategy={}",
            spec.sweep_id,
            spec.chunk_claim_id,
            spec.image_path.display(),
            spec.effort,
            spec.distance,
            spec.strategy
        );
    }

    // Ensure parent dir for the output exists.
    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        eprintln!("[w44-212] mkdir -p {} failed: {e}", parent.display());
        return ExitCode::from(2);
    }

    match run_cell(&spec, &args.output) {
        Ok(row) => {
            if args.verbose {
                eprintln!(
                    "[w44-212] OK bytes={} encode_ms={:.1} decode_ms={:.1} ssim2={:?} butter={:?} cvvdp={:?}",
                    row.encoded_bytes,
                    row.encode_ms,
                    row.decode_ms,
                    row.ssim2,
                    row.butter_norm3,
                    row.cvvdp
                );
            }
            // Always emit a one-line summary to stdout so the bash
            // worker can `read` it for chunk progress reporting.
            println!(
                "{{\"status\":\"ok\",\"sweep_id\":\"{}\",\"chunk_claim_id\":\"{}\",\"encoded_bytes\":{},\"encode_ms\":{:.2},\"decode_ms\":{:.2},\"output\":\"{}\"}}",
                row.sweep_id,
                row.chunk_claim_id,
                row.encoded_bytes,
                row.encode_ms,
                row.decode_ms,
                args.output.display(),
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[w44-212] CELL FAILED: {e}");
            // Stdout still gets a JSON line so the bash worker can
            // record the failure without re-parsing stderr.
            println!(
                "{{\"status\":\"err\",\"sweep_id\":\"{}\",\"chunk_claim_id\":\"{}\",\"error\":\"{}\"}}",
                spec.sweep_id,
                spec.chunk_claim_id,
                e.to_string().replace('"', "'"),
            );
            ExitCode::from(2)
        }
    }
}
