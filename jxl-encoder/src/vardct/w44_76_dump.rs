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

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
use std::sync::Mutex;

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
use super::ac_strategy::STRATEGY_CODE_LUT;

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
static DUMP_HOOK_PRESENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
fn dump_dir() -> Option<std::path::PathBuf> {
    // Perf: this gate is probed on the encode hot path; raw env::var_os
    // per probe (getenv + env RwLock + CStr scan) measured 25-35 % of
    // CPU at lossy e3/e4 (perf_lossy_low_2026-06-10.meta). The OnceLock
    // caches PRESENCE at first probe: absent => permanently disabled for
    // this process (zero further env reads); present => per-call
    // re-reads keep the documented repoint-between-images behaviour.
    // The hook must therefore be set before the process's first encode.
    if !*DUMP_HOOK_PRESENT.get_or_init(|| std::env::var_os("JXL_W44_76_PER_BLOCK_DUMP").is_some()) {
        return None;
    }
    std::env::var_os("JXL_W44_76_PER_BLOCK_DUMP").map(std::path::PathBuf::from)
}

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
static DUMP_STATE: Mutex<Option<DumpState>> = Mutex::new(None);

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
struct DumpState {
    file: std::io::BufWriter<std::fs::File>,
    rows: usize,
    dir: std::path::PathBuf,
}

/// Initialize the dump (or re-init if the env var now points to a different
/// directory than the last opened state — useful when an example driver
/// encodes multiple images in the same process and wants per-image dumps).
#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
fn ensure_initialized(dir: &std::path::Path) {
    let mut guard = DUMP_STATE.lock().unwrap();
    if let Some(state) = guard.as_ref()
        && state.dir == dir
    {
        return;
    }
    // Dir changed — flush old handle and re-init for the new dir.
    // We intentionally drop the old state below.
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
    *guard = Some(DumpState {
        file: bw,
        rows: 0,
        dir: dir.to_path_buf(),
    });
}

/// Append a single (block, channel) tokenization sample.
///
/// `raw_strategy` is the *internal* Rust enum (DCT8=0, AFV0=12, etc.); this
/// fn applies `STRATEGY_CODE_LUT` to emit the libjxl-wire value (DCT8=0,
/// AFV0=14, etc.) for safe join with libjxl-side dumps.
///
/// `qac` is the per-block raw_quant (u8 from the quant_field).
#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
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

/// No-op when std is not available (dump requires std).
#[cfg(not(all(feature = "std", feature = "__env_var_diagnostics")))]
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

// ── W44-201 per-position coefficient VALUE dump ──────────────────────
//
// Localizes the W44-200 finding: DCT32x32 Y custom scan order emits 308
// nonzero Lehmer codes in Zenjxl mode but only 5 in Libjxl mode on the
// 3637739 cell. Same `used_orders=0x5d` bitmask. The divergence MUST be
// in per-position zero patterns within DCT32x32 Y blocks (or whichever
// strategy/channel triggers the dump). This dump captures the raw
// quantized coefficient values per (bx, by, position) for whichever
// (strategy_wire, channel) combinations the caller wants.
//
// Set `JXL_W44_201_COEFFS_DUMP=<dir>` and optionally
// `JXL_W44_201_COEFFS_STRATEGY=<wire>` (default: 5 = DCT32X32) and
// `JXL_W44_201_COEFFS_CHANNEL=<c>` (default: 1 = Y). One TSV per
// caller-invocation is overwritten; zero overhead when env var unset.

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
static COEFFS_STATE: Mutex<Option<CoeffsDumpState>> = Mutex::new(None);

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
struct CoeffsDumpState {
    file: std::io::BufWriter<std::fs::File>,
    rows: usize,
    dir: std::path::PathBuf,
    target_strategy_wire: u8,
    target_channel: u8,
}

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
static COEFFS_HOOK_PRESENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
fn coeffs_dump_dir() -> Option<std::path::PathBuf> {
    // Same once-presence gate as `dump_dir` above — probed per block.
    if !*COEFFS_HOOK_PRESENT.get_or_init(|| std::env::var_os("JXL_W44_201_COEFFS_DUMP").is_some()) {
        return None;
    }
    std::env::var_os("JXL_W44_201_COEFFS_DUMP").map(std::path::PathBuf::from)
}

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
fn coeffs_target_strategy_wire() -> u8 {
    std::env::var("JXL_W44_201_COEFFS_STRATEGY")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(5) // DCT32X32 wire code
}

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
fn coeffs_target_channel() -> u8 {
    std::env::var("JXL_W44_201_COEFFS_CHANNEL")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(1) // Y channel
}

#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
fn ensure_coeffs_initialized(dir: &std::path::Path) {
    let mut guard = COEFFS_STATE.lock().unwrap();
    if let Some(state) = guard.as_ref()
        && state.dir == dir
    {
        return;
    }
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let path = dir.join("per_position_coeffs.tsv");
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
    let target_strategy_wire = coeffs_target_strategy_wire();
    let target_channel = coeffs_target_channel();
    let _ = writeln!(
        bw,
        "# W44-201 per-position coefficient dump (strategy_wire={}, channel={})",
        target_strategy_wire, target_channel
    );
    let _ = writeln!(bw, "bx\tby\tposition\tvalue");
    *guard = Some(CoeffsDumpState {
        file: bw,
        rows: 0,
        dir: dir.to_path_buf(),
        target_strategy_wire,
        target_channel,
    });
}

/// Append per-position coefficients for a block, but ONLY if the
/// (raw_strategy, channel) match the env-configured target.
///
/// `full_block` is the assembled coefficient block (size = coverage *
/// 64). `raw_strategy` is the INTERNAL Rust enum (DCT32X32 = 4); the
/// dump converts to the libjxl-wire code for env comparison.
#[cfg(all(feature = "std", feature = "__env_var_diagnostics"))]
pub fn dump_coeffs(bx: usize, by: usize, raw_strategy: u8, channel: usize, full_block: &[i32]) {
    let Some(dir) = coeffs_dump_dir() else {
        return;
    };
    ensure_coeffs_initialized(&dir);
    let mut guard = COEFFS_STATE.lock().unwrap();
    let Some(state) = guard.as_mut() else { return };
    let strategy_wire = STRATEGY_CODE_LUT[raw_strategy as usize];
    if strategy_wire != state.target_strategy_wire {
        return;
    }
    if channel as u8 != state.target_channel {
        return;
    }
    use std::io::Write;
    // Emit a sentinel row per block first so we can count blocks even when
    // all coefficients are zero (which DOES happen at high distances on
    // very smooth tiles). pos=-1, value=0 marks "block exists".
    let _ = writeln!(state.file, "{}\t{}\t-1\t0", bx, by);
    state.rows += 1;
    for (pos, &v) in full_block.iter().enumerate() {
        if v != 0 {
            // Only dump non-zero positions to keep file size manageable.
            // For zero-vs-nonzero comparison, that's all we need; the
            // post-processor reconstructs the per-position zero count.
            let _ = writeln!(state.file, "{}\t{}\t{}\t{}", bx, by, pos, v);
            state.rows += 1;
        }
    }
    let _ = state.file.flush();
}

#[cfg(not(all(feature = "std", feature = "__env_var_diagnostics")))]
#[inline(always)]
pub fn dump_coeffs(
    _bx: usize,
    _by: usize,
    _raw_strategy: u8,
    _channel: usize,
    _full_block: &[i32],
) {
}
