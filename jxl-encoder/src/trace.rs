// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Zero-cost bitstream tracing for debugging encoder output.
//!
//! Enable with `--features trace-bitstream` to see exactly what's being
//! written to the bitstream, at what positions, and what it means.
//!
//! # Output Format
//!
//! Each write produces a line:
//! ```text
//! [bit_pos] SECTION.field: value (n_bits bits) = 0bXXXX
//! ```
//!
//! Example:
//! ```text
//! [0] FILE_HEADER.signature: 0xff0a (16 bits) = 0b0000101011111111
//! [16] FILE_HEADER.all_default: false (1 bit) = 0b0
//! [17] FRAME_HEADER.all_default: false (1 bit) = 0b0
//! ```
//!
//! # Usage
//!
//! Use the `trace_write!` macro instead of direct `writer.write()` calls:
//!
//! ```ignore
//! use crate::trace::trace_write;
//!
//! // Instead of: writer.write(16, 0xff0a)?;
//! trace_write!(writer, 16, 0xff0a, "FILE_HEADER.signature")?;
//!
//! // For boolean flags:
//! trace_write!(writer, 1, 0, "FRAME_HEADER.all_default", "false")?;
//!
//! // For enum values:
//! trace_write!(writer, 2, 0, "FRAME_HEADER.encoding", "VarDCT(0)")?;
//! ```
//!
//! # Sections
//!
//! Use `trace_section!` to track hierarchical context:
//!
//! ```ignore
//! trace_section!(begin "FRAME_HEADER");
//! // ... writes ...
//! trace_section!(end "FRAME_HEADER" @ writer);
//! ```

#[cfg(feature = "trace-bitstream")]
use std::cell::RefCell;
#[cfg(feature = "trace-bitstream")]
use std::fs::File;
#[cfg(feature = "trace-bitstream")]
use std::io::Write;

// Thread-local trace output file.
#[cfg(feature = "trace-bitstream")]
thread_local! {
    static TRACE_OUTPUT: RefCell<Option<File>> = const { RefCell::new(None) };
    static SECTION_STACK: RefCell<Vec<(&'static str, usize)>> = const { RefCell::new(Vec::new()) };
}

/// Initialize tracing to a file.
///
/// Call this at the start of encoding to capture trace output.
#[cfg(feature = "trace-bitstream")]
pub fn init_trace(path: &str) -> std::io::Result<()> {
    let file = File::create(path)?;
    TRACE_OUTPUT.with(|output| {
        *output.borrow_mut() = Some(file);
    });
    Ok(())
}

/// Initialize tracing to stderr (default).
#[cfg(feature = "trace-bitstream")]
pub fn init_trace_stderr() {
    // Default behavior - trace to stderr
}

/// Finalize tracing and flush output.
#[cfg(feature = "trace-bitstream")]
pub fn finish_trace() {
    TRACE_OUTPUT.with(|output| {
        if let Some(mut f) = output.borrow_mut().take() {
            let _ = f.flush();
        }
    });
}

/// Write a trace line.
#[cfg(feature = "trace-bitstream")]
pub fn trace_line(line: &str) {
    TRACE_OUTPUT.with(|output| {
        if let Some(ref mut f) = *output.borrow_mut() {
            let _ = writeln!(f, "{}", line);
        } else {
            eprintln!("{}", line);
        }
    });
}

/// Push a section onto the context stack.
#[cfg(feature = "trace-bitstream")]
pub fn push_section(name: &'static str, bit_pos: usize) {
    SECTION_STACK.with(|stack| {
        stack.borrow_mut().push((name, bit_pos));
    });
    trace_line(&format!("[{}] >>> BEGIN {}", bit_pos, name));
}

/// Pop a section from the context stack.
#[cfg(feature = "trace-bitstream")]
pub fn pop_section(name: &'static str, bit_pos: usize) {
    SECTION_STACK.with(|stack| {
        if let Some((popped, start_pos)) = stack.borrow_mut().pop() {
            if popped != name {
                trace_line(&format!(
                    "[{}] !!! SECTION MISMATCH: expected {}, got {}",
                    bit_pos, name, popped
                ));
            }
            trace_line(&format!(
                "[{}] <<< END {} ({} bits)",
                bit_pos,
                name,
                bit_pos - start_pos
            ));
        }
    });
}

/// Get current section prefix for field names.
#[cfg(feature = "trace-bitstream")]
pub fn section_prefix() -> String {
    SECTION_STACK.with(|stack| {
        stack
            .borrow()
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(".")
    })
}

/// Format bits as binary string.
#[cfg(feature = "trace-bitstream")]
pub fn format_bits(value: u64, n_bits: usize) -> String {
    if n_bits == 0 {
        return "0b(empty)".to_string();
    }
    let mut s = String::with_capacity(n_bits + 2);
    s.push_str("0b");
    for i in (0..n_bits).rev() {
        if (value >> i) & 1 == 1 {
            s.push('1');
        } else {
            s.push('0');
        }
    }
    s
}

/// Core trace write function.
#[cfg(feature = "trace-bitstream")]
#[inline]
pub fn trace_write_impl(
    bit_pos_before: usize,
    n_bits: usize,
    value: u64,
    field: &str,
    description: Option<&str>,
) {
    let bits_str = format_bits(value, n_bits);
    let desc = match description {
        Some(d) => format!(" // {}", d),
        None => String::new(),
    };

    let prefix = section_prefix();
    let full_field = if prefix.is_empty() {
        field.to_string()
    } else {
        format!("{}.{}", prefix, field)
    };

    trace_line(&format!(
        "[{:6}] {}: {} ({} bits) = {}{}",
        bit_pos_before, full_field, value, n_bits, bits_str, desc
    ));
}

// ============================================================================
// MACROS - These are the primary API
// ============================================================================

/// Trace and write bits to the bitstream.
///
/// Zero-cost when `trace-bitstream` feature is disabled.
///
/// # Syntax
///
/// ```ignore
/// // Basic write with field name:
/// trace_write!(writer, n_bits, value, "field_name")?;
///
/// // Write with description:
/// trace_write!(writer, n_bits, value, "field_name", "description")?;
/// ```
#[macro_export]
#[cfg(feature = "trace-bitstream")]
macro_rules! trace_write {
    ($writer:expr, $n_bits:expr, $value:expr, $field:expr) => {{
        let pos = $writer.bits_written();
        let result = $writer.write($n_bits, $value as u64);
        $crate::trace::trace_write_impl(pos, $n_bits, $value as u64, $field, None);
        result
    }};
    ($writer:expr, $n_bits:expr, $value:expr, $field:expr, $desc:expr) => {{
        let pos = $writer.bits_written();
        let result = $writer.write($n_bits, $value as u64);
        $crate::trace::trace_write_impl(pos, $n_bits, $value as u64, $field, Some($desc));
        result
    }};
}

#[macro_export]
#[cfg(not(feature = "trace-bitstream"))]
macro_rules! trace_write {
    ($writer:expr, $n_bits:expr, $value:expr, $field:expr) => {
        $writer.write($n_bits, $value as u64)
    };
    ($writer:expr, $n_bits:expr, $value:expr, $field:expr, $desc:expr) => {
        $writer.write($n_bits, $value as u64)
    };
}

/// Trace section boundaries.
///
/// # Syntax
///
/// ```ignore
/// trace_section!(begin "SECTION_NAME" @ writer);
/// // ... writes ...
/// trace_section!(end "SECTION_NAME" @ writer);
/// ```
#[macro_export]
#[cfg(feature = "trace-bitstream")]
macro_rules! trace_section {
    (begin $name:expr, $writer:expr) => {
        $crate::trace::push_section($name, $writer.bits_written())
    };
    (end $name:expr, $writer:expr) => {
        $crate::trace::pop_section($name, $writer.bits_written())
    };
}

#[macro_export]
#[cfg(not(feature = "trace-bitstream"))]
macro_rules! trace_section {
    (begin $name:expr, $writer:expr) => {};
    (end $name:expr, $writer:expr) => {};
}

/// Log a trace message without writing bits.
///
/// Useful for noting important state or decisions.
#[macro_export]
#[cfg(feature = "trace-bitstream")]
macro_rules! trace_note {
    ($writer:expr, $($arg:tt)*) => {
        $crate::trace::trace_line(&format!("[{:6}] NOTE: {}", $writer.bits_written(), format!($($arg)*)))
    };
}

#[macro_export]
#[cfg(not(feature = "trace-bitstream"))]
macro_rules! trace_note {
    ($writer:expr, $($arg:tt)*) => {};
}

/// Trace a byte append operation.
#[macro_export]
#[cfg(feature = "trace-bitstream")]
macro_rules! trace_bytes {
    ($writer:expr, $bytes:expr, $field:expr) => {{
        let pos = $writer.bits_written();
        let data: &[u8] = $bytes;
        let result = $writer.append_bytes(data);
        $crate::trace::trace_line(&format!(
            "[{:6}] {}: [{} bytes] {:02x?}",
            pos,
            $field,
            data.len(),
            &data[..data.len().min(32)]
        ));
        result
    }};
}

#[macro_export]
#[cfg(not(feature = "trace-bitstream"))]
macro_rules! trace_bytes {
    ($writer:expr, $bytes:expr, $field:expr) => {
        $writer.append_bytes($bytes)
    };
}

/// Debug print macro - only outputs when trace-bitstream feature is enabled.
/// Use this instead of eprintln! for debug output in encoder code.
#[macro_export]
#[cfg(feature = "trace-bitstream")]
macro_rules! debug_eprintln {
    ($($arg:tt)*) => {
        eprintln!($($arg)*)
    };
}

#[macro_export]
#[cfg(not(feature = "trace-bitstream"))]
macro_rules! debug_eprintln {
    ($($arg:tt)*) => {};
}

// Re-export macros at crate level
pub use debug_eprintln;
pub use trace_bytes;
pub use trace_note;
pub use trace_section;
pub use trace_write;

// ============================================================================
// NO-OP STUBS when feature is disabled
// ============================================================================

#[cfg(not(feature = "trace-bitstream"))]
pub fn init_trace(_path: &str) -> std::io::Result<()> {
    Ok(())
}

#[cfg(not(feature = "trace-bitstream"))]
pub fn init_trace_stderr() {}

#[cfg(not(feature = "trace-bitstream"))]
pub fn finish_trace() {}

#[cfg(test)]
mod tests {
    #[cfg(feature = "trace-bitstream")]
    use super::*;

    #[test]
    fn test_format_bits() {
        #[cfg(feature = "trace-bitstream")]
        {
            assert_eq!(format_bits(0b1010, 4), "0b1010");
            assert_eq!(format_bits(0b1, 1), "0b1");
            assert_eq!(format_bits(0b0, 1), "0b0");
            assert_eq!(format_bits(0xff0a, 16), "0b1111111100001010");
        }
    }
}
