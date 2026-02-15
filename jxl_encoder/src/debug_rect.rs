// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Spatial decision logging for encoder debugging.
//!
//! When the `debug-rect` feature is enabled, the [`debug_rect!`] macro logs every
//! encoder decision alongside the rectangle it affects. Logs are collected in a
//! global buffer and flushed to a sidecar CSV when the frame is complete.
//!
//! When the feature is disabled, the macro compiles to nothing.
//!
//! # CSV format
//!
//! ```text
//! stage,x,y,w,h,message
//! ```
//!
//! # Query workflow
//!
//! 1. Encode with `--features debug-rect` → produces `output.jxl.debug_rect.csv`
//! 2. Decode both our JXL and cjxl's JXL, diff the pixels to find divergent regions
//! 3. Call [`query_overlapping`] (or grep the CSV) for rectangles touching that region
//! 4. Read the `message` column to understand every decision that affected those pixels

#[cfg(feature = "debug-rect")]
use std::sync::Mutex;

/// Global log buffer, protected by a mutex.
/// Each entry is a pre-formatted CSV row (no newline).
/// Public so the `debug_rect!` macro can access it from other modules.
#[cfg(feature = "debug-rect")]
#[doc(hidden)]
pub static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Log a spatial decision.
///
/// # Parameters
/// - `stage`: short label for the encoder phase (e.g. `"patches/seed"`, `"patches/cc"`)
/// - `x, y, w, h`: the affected rectangle in image coordinates
/// - `msg`: a formatted message describing the decision
///
/// When `debug-rect` is disabled this is a no-op.
#[cfg(feature = "debug-rect")]
#[macro_export]
macro_rules! debug_rect {
    ($stage:expr, $x:expr, $y:expr, $w:expr, $h:expr, $($arg:tt)*) => {{
        let msg = format!($($arg)*);
        let row = format!("{},{},{},{},{},{}", $stage, $x, $y, $w, $h, msg.replace(',', ";"));
        if let Ok(mut buf) = $crate::debug_rect::LOG.lock() {
            buf.push(row);
        }
    }};
}

/// No-op version when feature is disabled — still evaluates arguments
/// to suppress unused variable warnings, but the optimizer eliminates everything.
#[cfg(not(feature = "debug-rect"))]
#[macro_export]
macro_rules! debug_rect {
    ($stage:expr, $x:expr, $y:expr, $w:expr, $h:expr, $($arg:tt)*) => {
        if false {
            // Ensure all arguments are type-checked and considered "used"
            let _ = ($stage, $x, $y, $w, $h);
            let _ = format_args!($($arg)*);
        }
    };
}

/// Clear the log buffer. Call at the start of each frame encode.
#[cfg(feature = "debug-rect")]
pub fn clear() {
    if let Ok(mut buf) = LOG.lock() {
        buf.clear();
    }
}

#[cfg(not(feature = "debug-rect"))]
pub fn clear() {}

/// Flush the log buffer to a CSV file. Call when the frame is done.
///
/// The file is written to `{base_path}.debug_rect.csv`.
/// If `base_path` is empty, writes to `debug_rect.csv` in the current directory.
#[cfg(feature = "debug-rect")]
pub fn flush(base_path: &str) {
    let path = if base_path.is_empty() {
        "debug_rect.csv".to_string()
    } else {
        format!("{base_path}.debug_rect.csv")
    };
    let rows = {
        let Ok(buf) = LOG.lock() else { return };
        buf.clone()
    };
    if rows.is_empty() {
        return;
    }
    let mut out = String::with_capacity(rows.len() * 80);
    out.push_str("stage,x,y,w,h,message\n");
    for row in &rows {
        out.push_str(row);
        out.push('\n');
    }
    if let Err(e) = std::fs::write(&path, &out) {
        eprintln!("debug_rect: failed to write {path}: {e}");
    } else {
        eprintln!("debug_rect: wrote {} rows to {path}", rows.len());
    }
}

#[cfg(not(feature = "debug-rect"))]
pub fn flush(_base_path: &str) {}

/// Return all log rows whose rectangle overlaps the query region.
///
/// Useful for programmatic queries: given a region of visual difference,
/// find every decision that touched it.
#[cfg(feature = "debug-rect")]
pub fn query_overlapping(qx: i64, qy: i64, qw: i64, qh: i64) -> Vec<String> {
    let Ok(buf) = LOG.lock() else {
        return Vec::new();
    };
    let mut hits = Vec::new();
    for row in buf.iter() {
        // Parse "stage,x,y,w,h,message"
        let parts: Vec<&str> = row.splitn(6, ',').collect();
        if parts.len() < 5 {
            continue;
        }
        let Ok(rx) = parts[1].parse::<i64>() else {
            continue;
        };
        let Ok(ry) = parts[2].parse::<i64>() else {
            continue;
        };
        let Ok(rw) = parts[3].parse::<i64>() else {
            continue;
        };
        let Ok(rh) = parts[4].parse::<i64>() else {
            continue;
        };
        // AABB overlap test
        if rx < qx + qw && rx + rw > qx && ry < qy + qh && ry + rh > qy {
            hits.push(row.clone());
        }
    }
    hits
}

#[cfg(not(feature = "debug-rect"))]
pub fn query_overlapping(_qx: i64, _qy: i64, _qw: i64, _qh: i64) -> Vec<String> {
    Vec::new()
}
