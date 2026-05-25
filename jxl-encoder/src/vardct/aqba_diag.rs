// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-27: per-block `AdjustQuantBlockAC` firing-rate diagnostic.
//!
//! Gated behind the `investigate-adjust-quant-block-ac` feature. Counts how
//! many blocks fire each of the 6 heuristics (A/B/C/D/E/F) per
//! (raw_strategy, channel) tuple. At process exit (or when
//! [`emit_and_reset`] is called explicitly), writes a TSV to the path
//! named in the `JXL_AQBA_DIAG_TSV` env var, or stderr summary if unset.

#![cfg(feature = "investigate-adjust-quant-block-ac")]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Mutex;

// ── SA-A (2026-05-24) per-block dump extension ───────────────────────────
//
// W44-27 aggregates by (raw_strategy, channel). SA-A needs per-block
// (bx, by, channel, orig_quant, new_quant, flags, raw_strategy) so we can
// diff against a libjxl-instrumented dump on the same image.
//
// Activated by env var `JXL_AQBA_PERBLOCK_TSV=<path>`. Buffered per-thread,
// flushed to global on `flush_tl_to_global`, written by `emit_and_reset`.

#[derive(Debug, Clone, Copy)]
pub struct PerBlockRecord {
    pub bx: u32,
    pub by: u32,
    pub channel: u8,
    pub raw_strategy: u8,
    pub orig_quant: i32,
    pub new_quant: i32,
    pub heuristics: u8,
}

thread_local! {
    static TL_PERBLOCK: RefCell<Vec<PerBlockRecord>> = RefCell::new(Vec::new());
}

static GLOBAL_PERBLOCK: Mutex<Vec<PerBlockRecord>> = Mutex::new(Vec::new());

/// Record a per-block AdjustQuantBlockAC invocation when
/// `JXL_AQBA_PERBLOCK_TSV` env var is set.
pub fn record_perblock(
    bx: u32,
    by: u32,
    channel: u8,
    raw_strategy: u8,
    orig_quant: i32,
    new_quant: i32,
    heuristics: u8,
) {
    // Cheap env-var gate: only collect if path is set.
    if std::env::var_os("JXL_AQBA_PERBLOCK_TSV").is_none() {
        return;
    }
    TL_PERBLOCK.with(|tl| {
        tl.borrow_mut().push(PerBlockRecord {
            bx,
            by,
            channel,
            raw_strategy,
            orig_quant,
            new_quant,
            heuristics,
        });
    });
}

fn flush_tl_perblock() {
    TL_PERBLOCK.with(|tl| {
        let mut tl_vec = tl.borrow_mut();
        if tl_vec.is_empty() {
            return;
        }
        let mut g = GLOBAL_PERBLOCK.lock().unwrap();
        g.append(&mut tl_vec);
    });
}

/// Public wrapper for `flush_tl_perblock` so worker rayon tasks can call it
/// before returning (mirrors W44-27's `flush_tl_to_global`).
pub fn flush_tl_perblock_public() {
    flush_tl_perblock();
}

#[derive(Default, Debug, Clone, Copy)]
pub struct FiringCounts {
    pub total: u64,
    pub a: u64,
    pub b: u64,
    pub c: u64,
    pub d: u64,
    pub e: u64,
    pub f: u64,
    pub quant_unchanged: u64,
    pub quant_increased: u64,
    pub quant_decreased: u64,
    pub orig_quant_sum: u64,
    pub new_quant_sum: u64,
}

thread_local! {
    static TL_COUNTS: RefCell<BTreeMap<(u8, u8), FiringCounts>> = RefCell::new(BTreeMap::new());
}

static GLOBAL_COUNTS: Mutex<Option<BTreeMap<(u8, u8), FiringCounts>>> = Mutex::new(None);

/// Record one `AdjustQuantBlockAC` invocation.
pub fn record(
    raw_strategy: u8,
    channel: usize,
    heuristics_fired: u8,
    orig_quant: i32,
    new_quant: i32,
) {
    TL_COUNTS.with(|tl| {
        let mut map = tl.borrow_mut();
        let entry = map.entry((raw_strategy, channel as u8)).or_default();
        entry.total += 1;
        if heuristics_fired & 0x01 != 0 {
            entry.a += 1;
        }
        if heuristics_fired & 0x02 != 0 {
            entry.b += 1;
        }
        if heuristics_fired & 0x04 != 0 {
            entry.c += 1;
        }
        if heuristics_fired & 0x08 != 0 {
            entry.d += 1;
        }
        if heuristics_fired & 0x10 != 0 {
            entry.e += 1;
        }
        if heuristics_fired & 0x20 != 0 {
            entry.f += 1;
        }
        if new_quant == orig_quant {
            entry.quant_unchanged += 1;
        } else if new_quant > orig_quant {
            entry.quant_increased += 1;
        } else {
            entry.quant_decreased += 1;
        }
        entry.orig_quant_sum += orig_quant.max(0) as u64;
        entry.new_quant_sum += new_quant.max(0) as u64;
    });
}

/// Merge thread-local counts into the global aggregate. Called from
/// [`emit_and_reset`] but also exposed so callers (e.g. encode end) can
/// flush worker-thread state before reading the global TSV.
pub fn flush_tl_to_global() {
    TL_COUNTS.with(|tl| {
        let mut tl_map = tl.borrow_mut();
        if tl_map.is_empty() {
            return;
        }
        let mut g = GLOBAL_COUNTS.lock().unwrap();
        let g_map = g.get_or_insert_with(BTreeMap::new);
        // BTreeMap::drain not stabilized — take + clear.
        let taken = std::mem::take(&mut *tl_map);
        for ((strat, chan), counts) in taken.into_iter() {
            let entry = g_map.entry((strat, chan)).or_default();
            entry.total += counts.total;
            entry.a += counts.a;
            entry.b += counts.b;
            entry.c += counts.c;
            entry.d += counts.d;
            entry.e += counts.e;
            entry.f += counts.f;
            entry.quant_unchanged += counts.quant_unchanged;
            entry.quant_increased += counts.quant_increased;
            entry.quant_decreased += counts.quant_decreased;
            entry.orig_quant_sum += counts.orig_quant_sum;
            entry.new_quant_sum += counts.new_quant_sum;
        }
    });
}

/// Strategy-code → short label for the TSV. Matches `RAW_STRATEGY_*` from
/// `vardct/ac_strategy.rs` (this codebase's internal raw numbering, NOT the
/// JXL wire format).
fn strategy_label(s: u8) -> &'static str {
    match s {
        0 => "DCT8",
        1 => "DCT16X8",
        2 => "DCT8X16",
        3 => "DCT16X16",
        4 => "DCT32X32",
        5 => "DCT4X8",
        6 => "DCT8X4",
        7 => "DCT4X4",
        8 => "IDENTITY",
        9 => "DCT2X2",
        10 => "DCT32X16",
        11 => "DCT16X32",
        12 => "AFV0",
        13 => "AFV1",
        14 => "AFV2",
        15 => "AFV3",
        16 => "DCT64X64",
        17 => "DCT64X32",
        18 => "DCT32X64",
        _ => "unknown",
    }
}

fn channel_label(c: u8) -> &'static str {
    match c {
        0 => "X",
        1 => "Y",
        2 => "B",
        _ => "?",
    }
}

/// Emit per-block TSV to `JXL_AQBA_PERBLOCK_TSV` (if set). Header is
/// written only on first call; subsequent calls append. The header
/// columns match the libjxl side instrumentation in
/// `enc_adaptive_quantization.cc::AdjustQuantBlockAC` (SA-A 2026-05-24).
pub fn emit_perblock(tag: &str) {
    flush_tl_perblock();
    let path = match std::env::var("JXL_AQBA_PERBLOCK_TSV") {
        Ok(p) => p,
        Err(_) => return,
    };
    let records = {
        let mut g = GLOBAL_PERBLOCK.lock().unwrap();
        std::mem::take(&mut *g)
    };
    if records.is_empty() {
        return;
    }
    use std::io::Write;
    let path_obj = std::path::Path::new(&path);
    let already_exists = path_obj.exists();
    let mut f = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[aqba_diag perblock] cannot open {}: {}", path, e);
            return;
        }
    };
    if !already_exists {
        let _ = writeln!(
            f,
            "tag\tbx\tby\tchannel\traw_strategy\torig_quant\tnew_quant\theuristics_hex\tfired_a\tfired_b\tfired_c\tfired_d\tfired_e\tfired_f"
        );
    }
    for r in records.iter() {
        let h = r.heuristics;
        let _ = writeln!(
            f,
            "{tag}\t{bx}\t{by}\t{ch}\t{strat}\t{oq}\t{nq}\t0x{h:02x}\t{a}\t{b}\t{c}\t{d}\t{e}\t{ff}",
            tag = tag,
            bx = r.bx,
            by = r.by,
            ch = r.channel,
            strat = r.raw_strategy,
            oq = r.orig_quant,
            nq = r.new_quant,
            h = h,
            a = (h >> 0) & 1,
            b = (h >> 1) & 1,
            c = (h >> 2) & 1,
            d = (h >> 3) & 1,
            e = (h >> 4) & 1,
            ff = (h >> 5) & 1,
        );
    }
}

/// Emit the aggregated TSV and reset both thread-local and global counts.
/// Writes to the file named by `JXL_AQBA_DIAG_TSV`; if unset, writes a
/// summary to stderr.
pub fn emit_and_reset(tag: &str) {
    flush_tl_to_global();
    emit_perblock(tag);
    let g = {
        let mut g = GLOBAL_COUNTS.lock().unwrap();
        g.take().unwrap_or_default()
    };
    let path = std::env::var("JXL_AQBA_DIAG_TSV").ok();
    let mut buf = String::new();
    buf.push_str(
        "tag\tstrategy_code\tstrategy_label\tchannel\ttotal\ta\tb\tc\td\te\tf\tq_unchanged\tq_inc\tq_dec\torig_q_mean\tnew_q_mean\ta_pct\tb_pct\tc_pct\td_pct\te_pct\tf_pct\n",
    );
    for ((strat, chan), counts) in g.iter() {
        let total_f = counts.total.max(1) as f64;
        let orig_mean = counts.orig_quant_sum as f64 / total_f;
        let new_mean = counts.new_quant_sum as f64 / total_f;
        buf.push_str(&format!(
            "{tag}\t{strat}\t{slabel}\t{clabel}\t{total}\t{a}\t{b}\t{c}\t{d}\t{e}\t{f}\t{qu}\t{qi}\t{qd}\t{om:.3}\t{nm:.3}\t{ap:.3}\t{bp:.3}\t{cp:.3}\t{dp:.3}\t{ep:.3}\t{fp:.3}\n",
            tag = tag,
            strat = strat,
            slabel = strategy_label(*strat),
            clabel = channel_label(*chan),
            total = counts.total,
            a = counts.a,
            b = counts.b,
            c = counts.c,
            d = counts.d,
            e = counts.e,
            f = counts.f,
            qu = counts.quant_unchanged,
            qi = counts.quant_increased,
            qd = counts.quant_decreased,
            om = orig_mean,
            nm = new_mean,
            ap = counts.a as f64 / total_f,
            bp = counts.b as f64 / total_f,
            cp = counts.c as f64 / total_f,
            dp = counts.d as f64 / total_f,
            ep = counts.e as f64 / total_f,
            fp = counts.f as f64 / total_f,
        ));
    }
    if let Some(p) = path {
        // Append (so multi-image runs accumulate).
        use std::io::Write;
        let path_obj = std::path::Path::new(&p);
        let already_exists = path_obj.exists();
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&p)
        {
            Ok(mut f) => {
                let to_write = if already_exists {
                    // Strip header on append.
                    let nl = buf.find('\n').map(|i| i + 1).unwrap_or(0);
                    &buf[nl..]
                } else {
                    buf.as_str()
                };
                if let Err(e) = f.write_all(to_write.as_bytes()) {
                    eprintln!("[aqba_diag] failed to write {}: {}", p, e);
                }
            }
            Err(e) => eprintln!("[aqba_diag] cannot open {}: {}", p, e),
        }
    } else {
        eprintln!("[aqba_diag tag={tag}]\n{buf}");
    }
}
