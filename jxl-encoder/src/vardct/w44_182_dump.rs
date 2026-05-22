// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-182 per-tile CfL map dump.
//!
//! Env-var-gated diagnostic that records per-tile (`tx`, `ty`, `ytox`, `ytob`,
//! `pass`) values from `compute_cfl_map` (pass 1, `pass=1`) and
//! `refine_cfl_map` (pass 2, `pass=2`). Allows correlation with the
//! W44-178 per-block max-abs RGB shift map (`benchmarks/w44_178_recon_diff_clic_097cb426_2026-05-21.blocks.tsv`)
//! to identify whether CfL AC tile aggregation is the source of the
//! clic_097cb426 right-column ~0.008 RGB shift / -7 SSIM2 deficit.
//!
//! Set `JXL_W44_182_DUMP_CFL=<dir>` and re-encode. A single TSV named
//! `cfl_tiles.tsv` is written. Zero overhead when env var unset.
//!
//! Schema (5 cols, tab-separated):
//!   tx, ty, pass, ytox, ytob
//!
//! `pass=1` = `compute_cfl_map` result (forced DCT8 + Newton or fast).
//! `pass=2` = `refine_cfl_map` result (real ac_strategy + raw_quant_field).

#[cfg(feature = "std")]
use std::sync::Mutex;

#[cfg(feature = "std")]
fn dump_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("JXL_W44_182_DUMP_CFL").map(std::path::PathBuf::from)
}

#[cfg(feature = "std")]
static DUMP_STATE: Mutex<Option<DumpState>> = Mutex::new(None);

#[cfg(feature = "std")]
struct DumpState {
    file: std::io::BufWriter<std::fs::File>,
    rows: usize,
}

/// Initialize the dump (once per process) and write the TSV header.
#[cfg(feature = "std")]
fn ensure_initialized(dir: &std::path::Path) {
    let mut guard = DUMP_STATE.lock().unwrap();
    if guard.is_some() {
        return;
    }
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let path = dir.join("cfl_tiles.tsv");
    let Ok(file) = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)
    else {
        return;
    };
    use std::io::Write;
    let mut bw = std::io::BufWriter::new(file);
    let _ = writeln!(bw, "# W44-182 per-tile CfL map dump");
    let _ = writeln!(bw, "tx\tty\tpass\tytox\tytob");
    *guard = Some(DumpState { file: bw, rows: 0 });
}

/// Dump an entire CfL map (pass 1 or pass 2) — called AFTER
/// `compute_cfl_map` (pass=1) and AFTER `refine_cfl_map` (pass=2).
#[cfg(feature = "std")]
#[inline]
pub fn dump_map(pass: u8, xsize_tiles: usize, ysize_tiles: usize, ytox: &[i8], ytob: &[i8]) {
    let Some(dir) = dump_dir() else { return };
    ensure_initialized(&dir);
    let mut guard = DUMP_STATE.lock().unwrap();
    let Some(state) = guard.as_mut() else {
        return;
    };
    use std::io::Write;
    for ty in 0..ysize_tiles {
        for tx in 0..xsize_tiles {
            let idx = ty * xsize_tiles + tx;
            let _ = writeln!(
                state.file,
                "{}\t{}\t{}\t{}\t{}",
                tx, ty, pass, ytox[idx], ytob[idx]
            );
            state.rows += 1;
        }
    }
    let _ = state.file.flush();
}

#[cfg(feature = "std")]
pub fn flush() {
    let mut guard = DUMP_STATE.lock().unwrap();
    if let Some(state) = guard.as_mut() {
        use std::io::Write;
        let _ = state.file.flush();
    }
}

#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn flush() {}

#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn dump_map(_pass: u8, _xsize_tiles: usize, _ysize_tiles: usize, _ytox: &[i8], _ytob: &[i8]) {}
