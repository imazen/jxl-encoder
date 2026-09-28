// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Wire-level WP-header observer for `forced_wp_tests` (which lives in
//! `jxl-encoder` after the crate split — it needs the full encoder).
//! Zero cost unless a test installs an observation buffer.

use crate::bit_writer::BitWriter;
use alloc::vec::Vec;
use std::cell::RefCell;

std::thread_local! {
    static HEADERS: RefCell<Option<Vec<(usize, u64)>>> = const { RefCell::new(None) };
}

/// Record the last `start..bits_written` span as an LSB-first u64.
pub fn record_header(writer: &BitWriter, start: usize) {
    HEADERS.with_borrow_mut(|headers| {
        if let Some(headers) = headers {
            let end = writer.bits_written();
            let bytes = writer.peek_bytes();
            let bits = (start..end).enumerate().fold(0u64, |value, (shift, bit)| {
                value | (((bytes[bit / 8] >> (bit % 8)) & 1) as u64) << shift
            });
            headers.push((end - start, bits));
        }
    });
}

/// Install an empty observation buffer (thread-local).
pub fn begin_observation() {
    HEADERS.with_borrow_mut(|headers| *headers = Some(Vec::new()));
}

/// Take the observed `(n_bits, bits)` list and clear the buffer.
pub fn take_observed_headers() -> Vec<(usize, u64)> {
    HEADERS.with_borrow_mut(|headers| headers.take().unwrap())
}
