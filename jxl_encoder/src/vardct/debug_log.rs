// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Debug logging to file for the VarDCT encoder.
//!
//! When the `debug-tokens` feature is enabled, debug output goes to a file
//! instead of stderr, making it easy to grep without clobbering context.
//!
//! Usage:
//! ```ignore
//! debug_log!("DC token: ctx={}, value={}", ctx, value);
//! ```
//!
//! Output goes to `/tmp/jxl_enc_debug.log` (overwritten each run).

#[cfg(feature = "debug-tokens")]
use std::io::Write;
#[cfg(feature = "debug-tokens")]
use std::sync::Mutex;

#[cfg(feature = "debug-tokens")]
static DEBUG_LOG: Mutex<Option<std::fs::File>> = Mutex::new(None);

/// Initialize the debug log file. Called automatically on first use.
#[cfg(feature = "debug-tokens")]
pub fn init_debug_log() {
    let mut guard = DEBUG_LOG.lock().unwrap();
    if guard.is_none() {
        let file = std::fs::File::create("/tmp/jxl_enc_debug.log")
            .expect("Failed to create debug log file");
        *guard = Some(file);
    }
}

/// Write a line to the debug log file.
#[cfg(feature = "debug-tokens")]
pub fn write_debug_log(msg: &str) {
    init_debug_log();
    let mut guard = DEBUG_LOG.lock().unwrap();
    if let Some(ref mut file) = *guard {
        let _ = writeln!(file, "{}", msg);
    }
}

/// Flush the debug log file.
#[cfg(feature = "debug-tokens")]
pub fn flush_debug_log() {
    let mut guard = DEBUG_LOG.lock().unwrap();
    if let Some(ref mut file) = *guard {
        let _ = file.flush();
    }
}

/// Debug log macro - writes to /tmp/jxl_enc_debug.log when debug-tokens feature is enabled.
///
/// Usage: `debug_log!("message: {}", value);`
#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {
        #[cfg(feature = "debug-tokens")]
        {
            $crate::vardct::debug_log::write_debug_log(&format!($($arg)*));
        }
    };
}

/// Debug log macro that also flushes (use sparingly, for important checkpoints).
#[macro_export]
macro_rules! debug_log_flush {
    ($($arg:tt)*) => {
        #[cfg(feature = "debug-tokens")]
        {
            $crate::vardct::debug_log::write_debug_log(&format!($($arg)*));
            $crate::vardct::debug_log::flush_debug_log();
        }
    };
}
