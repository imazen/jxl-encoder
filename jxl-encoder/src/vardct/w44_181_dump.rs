// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-181 per-block DC quantization input dump.
//!
//! Env-var-gated diagnostic that records per-block DC quantization inputs
//! and BOTH the ours-expression result and the libjxl-expression result.
//!
//! Purpose: discriminate whether our `(dc * inv_factor - y_dc *
//! dc_cfl_factor).round() as i16` expression diverges from libjxl's
//! `std::round((dc - y_dc * y_factor * cfl_factor) * inv_factor)` due to
//! f32 evaluation-order precision (the two expressions are algebraically
//! equivalent at default cmap, but f32-order may differ by ±1 ULP at
//! rounding boundaries).
//!
//! Set `JXL_W44_181_DUMP_DC=<dir>` and re-encode. A single TSV named
//! `dc_quant_inputs.tsv` is written. Zero overhead when env var unset.
//!
//! Schema (8 cols, tab-separated):
//!   bx, by, channel, raw_strategy, dc, y_dc, inv_factor, dc_cfl_factor
//!
//! Post-process: for each row, compute:
//!   ours    = (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16
//!   libjxl  = ((dc - y_dc * dc_cfl_factor / inv_factor) * inv_factor).round() as i16
//! and count divergences. (Note: dc_cfl_factor / inv_factor is the f32
//! quantity libjxl precomputes per-channel as `y_factor * cfl_factor`.)
//!
//! Channel 1 (Y) is recorded with `y_dc = 0` and `dc_cfl_factor = 0.0` for
//! schema uniformity — the Y channel has no CfL subtraction.

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
use std::sync::Mutex;

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
static DUMP_HOOK_PRESENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
fn dump_dir() -> Option<std::path::PathBuf> {
    // Once-presence gate: probed per BLOCK from transform_blocks_into —
    // the raw env::var_os here was the residual ~12 % getenv share at
    // lossy e3/e4 after the first five hooks were gated
    // (perf_lossy_low_2026-06-10.meta addendum). Absent at first probe
    // => permanently disabled; present => legacy per-call re-reads.
    if !*DUMP_HOOK_PRESENT.get_or_init(|| std::env::var_os("JXL_W44_181_DUMP_DC").is_some()) {
        return None;
    }
    std::env::var_os("JXL_W44_181_DUMP_DC").map(std::path::PathBuf::from)
}

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
static DUMP_STATE: Mutex<Option<DumpState>> = Mutex::new(None);

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
struct DumpState {
    file: std::io::BufWriter<std::fs::File>,
    rows: usize,
}

/// Initialize the dump (once per process) and write the TSV header.
#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
fn ensure_initialized(dir: &std::path::Path) {
    let mut guard = DUMP_STATE.lock().unwrap();
    if guard.is_some() {
        return;
    }
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let path = dir.join("dc_quant_inputs.tsv");
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
    let _ = writeln!(bw, "# W44-181 per-block DC quant input dump (ours)");
    let _ = writeln!(
        bw,
        "bx\tby\tchannel\traw_strategy\tdc\ty_dc\tinv_factor\tdc_cfl_factor"
    );
    *guard = Some(DumpState { file: bw, rows: 0 });
}

/// Append a single DC sample. Cheap when dump_dir is None.
#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dump_dc(
    bx: usize,
    by: usize,
    channel: usize,
    raw_strategy: u8,
    dc: f32,
    y_dc: f32,
    inv_factor: f32,
    dc_cfl_factor: f32,
) {
    let Some(dir) = dump_dir() else { return };
    ensure_initialized(&dir);
    let mut guard = DUMP_STATE.lock().unwrap();
    let Some(state) = guard.as_mut() else { return };
    use std::io::Write;
    // Print floats with enough precision to reproduce the exact f32
    // (9 digits suffice for round-trip; we use {:e} format).
    let _ = writeln!(
        state.file,
        "{}\t{}\t{}\t{}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}",
        bx, by, channel, raw_strategy, dc, y_dc, inv_factor, dc_cfl_factor
    );
    state.rows += 1;
    let _ = state.file.flush();
}

#[cfg(not(all(feature = "std", feature = "__env_var_diagnostics")))]
#[inline(always)]
pub fn dump_dc(
    _bx: usize,
    _by: usize,
    _channel: usize,
    _raw_strategy: u8,
    _dc: f32,
    _y_dc: f32,
    _inv_factor: f32,
    _dc_cfl_factor: f32,
) {
}
