// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-76 per-block AC tokenization dump.
//!
//! Env-var-gated diagnostic that records `(by, bx, raw_strategy_wire,
//! channel, num_nonzeros, qac)` per call to the AC tokenizer. Used to
//! discriminate:
//!   (a) strategy-selection divergence vs libjxl, or
//!   (b) same-strategy-different-nzeros (i.e. quant-side) divergence.
//!
//! Set `JXL_W44_76_PER_BLOCK_DUMP=<dir>` and re-encode. A TSV named
//! `per_block_<group_idx>_<call_id>.tsv` is written for each AC group.
//! Zero overhead when env var unset (single `getenv` first call).
//!
//! CRITICAL (per W44-59 lesson): strategy is dumped in libjxl-wire space
//! via `STRATEGY_CODE_LUT[raw_strategy]` so joins against the libjxl
//! reference dump (`acs.RawStrategy()` is the wire enum) are safe.

#[cfg(feature = "std")]
use std::sync::Mutex;

#[cfg(feature = "std")]
use super::ac_strategy::STRATEGY_CODE_LUT;

#[cfg(feature = "std")]
fn dump_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("JXL_W44_76_PER_BLOCK_DUMP").map(std::path::PathBuf::from)
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
    let path = dir.join("per_block_ours.tsv");
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
    // Tab-separated: bx,by,strategy_wire,channel,nzeros,qac
    let _ = writeln!(bw, "# W44-76 per-block dump (ours, libjxl-wire strategy)");
    let _ = writeln!(bw, "bx\tby\tstrategy\tchannel\tnzeros\tqac");
    *guard = Some(DumpState { file: bw, rows: 0 });
}

/// Append a single (block, channel) tokenization sample.
///
/// `raw_strategy` is the *internal* Rust enum (DCT8=0, AFV0=12, etc.); this
/// fn applies `STRATEGY_CODE_LUT` to emit the libjxl-wire value (DCT8=0,
/// AFV0=14, etc.) for safe join with libjxl-side dumps.
///
/// `qac` is the per-block raw_quant (u8 from the quant_field).
#[cfg(feature = "std")]
pub fn dump_block(bx: usize, by: usize, raw_strategy: u8, channel: usize, nzeros: u16, qac: u8) {
    let Some(dir) = dump_dir() else { return };
    ensure_initialized(&dir);
    let mut guard = DUMP_STATE.lock().unwrap();
    let Some(state) = guard.as_mut() else { return };
    let strategy_wire = STRATEGY_CODE_LUT[raw_strategy as usize];
    use std::io::Write;
    let _ = writeln!(
        state.file,
        "{}\t{}\t{}\t{}\t{}\t{}",
        bx, by, strategy_wire, channel, nzeros, qac
    );
    state.rows += 1;
    // Flush after every row so a partial run still gives a complete file.
    // (Diagnostic build only; perf does not matter.)
    let _ = state.file.flush();
}

/// Explicitly flush + close the dump (call before exit if you want to be sure
/// the file is complete).
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

/// No-op when std is not available (dump requires std).
#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn dump_block(
    _bx: usize,
    _by: usize,
    _raw_strategy: u8,
    _channel: usize,
    _nzeros: u16,
    _qac: u8,
) {
}
