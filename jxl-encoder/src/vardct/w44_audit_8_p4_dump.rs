// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-AUDIT-8 Phase 4: per-DC-block dump for ours-vs-libjxl divergence
//! diagnosis on near-flat large-DCT blocks.
//!
//! Set `JXL_W44_AUDIT_8_P4_DUMP=<dir>` to record one row per (block, channel)
//! DC-quantization with columns:
//!
//!   bx, by, channel, raw_strategy, dc_raw, dc_quant
//!
//! `dc_raw` is the post-`dc_from_dct_NxN` float DC (input to quantization).
//! `dc_quant` is the rounded integer that ends up in `quant_dc[c]` (the
//! quantizer-domain value the decoder will dequantize). For X/B channels
//! the value already includes the libjxl CfL subtraction; for Y the
//! value is `(dc_raw * inv_factor).round() as i16`.
//!
//! Phase 4 mirror-side patch on libjxl writes `(bx, by, c, raw_strategy,
//! dc_raw, dc_quant)` from `enc_modular.cc::AddVarDCTDC` after the
//! `std::round(...)` quantize step. Both dumps are joined post-hoc on
//! `(bx, by, channel)` to localise DC-pipeline divergence on flat blocks.
//!
//! Zero overhead when env var is unset. `std`-gated.

#[cfg(feature = "std")]
use std::sync::Mutex;

#[cfg(feature = "std")]
static DUMP_HOOK_PRESENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[cfg(feature = "std")]
fn dump_dir() -> Option<std::path::PathBuf> {
    // Perf: this gate is probed on the encode hot path; raw env::var_os
    // per probe (getenv + env RwLock + CStr scan) measured 25-35 % of
    // CPU at lossy e3/e4 (perf_lossy_low_2026-06-10.meta). The OnceLock
    // caches PRESENCE at first probe: absent => permanently disabled for
    // this process (zero further env reads); present => per-call
    // re-reads keep the documented repoint-between-images behaviour.
    // The hook must therefore be set before the process's first encode.
    if !*DUMP_HOOK_PRESENT.get_or_init(|| std::env::var_os("JXL_W44_AUDIT_8_P4_DUMP").is_some()) {
        return None;
    }
    std::env::var_os("JXL_W44_AUDIT_8_P4_DUMP").map(std::path::PathBuf::from)
}

#[cfg(feature = "std")]
static DUMP_STATE: Mutex<Option<DumpState>> = Mutex::new(None);

#[cfg(feature = "std")]
struct DumpState {
    file: std::io::BufWriter<std::fs::File>,
    rows: usize,
}

#[cfg(feature = "std")]
fn ensure_initialized(dir: &std::path::Path) {
    let mut guard = DUMP_STATE.lock().unwrap();
    if guard.is_some() {
        return;
    }
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let path = dir.join("dc_per_block_ours.tsv");
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
    let _ = writeln!(bw, "# W44-AUDIT-8 Phase 4 per-DC-block dump (ours)");
    let _ = writeln!(bw, "bx\tby\tchannel\traw_strategy\tdc_raw\tdc_quant");
    *guard = Some(DumpState { file: bw, rows: 0 });
}

/// Emit one row. Cheap when env var unset.
///
/// Coordinates `(bx, by)` are absolute block indices (after adding any
/// rect/group origin). `dc_raw` is the post-transform float DC; `dc_quant`
/// is the rounded i16 that lands in the DC channel of the modular stream.
#[cfg(feature = "std")]
#[inline]
pub fn dump_dc(bx: usize, by: usize, channel: usize, raw_strategy: u8, dc_raw: f32, dc_quant: i16) {
    let Some(dir) = dump_dir() else { return };
    ensure_initialized(&dir);
    let mut guard = DUMP_STATE.lock().unwrap();
    let Some(state) = guard.as_mut() else { return };
    use std::io::Write;
    let _ = writeln!(
        state.file,
        "{}\t{}\t{}\t{}\t{:.9e}\t{}",
        bx, by, channel, raw_strategy, dc_raw, dc_quant
    );
    state.rows += 1;
    let _ = state.file.flush();
}

#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn dump_dc(
    _bx: usize,
    _by: usize,
    _channel: usize,
    _raw_strategy: u8,
    _dc_raw: f32,
    _dc_quant: i16,
) {
}
