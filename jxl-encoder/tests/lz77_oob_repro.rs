// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression for an LZ77 hash-chain index-out-of-bounds panic at
//! `entropy_coding/lz77.rs` (DoS vector — any caller with untrusted
//! image input could trigger a panic).
//!
//! Surfaced in zen-metrics sweep v09 (`s3://zentrain/sweep-v09-2026-05-05/`):
//! chunk `zenjxl-full-009` crashed in `jxl-encoder-0.3.1` with
//!
//! ```text
//! thread 'main' panicked at jxl-encoder-0.3.1/src/entropy_coding/lz77.rs:630:28:
//! index out of bounds: the len is 262144 but the index is 1073788476
//! ```
//!
//! The trigger row was a tiny 96×48 photo render (`size-dense-renders/
//! 4cd6910a0b7b39365fda5df87618d091__sz96.png`, included as a fixture)
//! at distance ~4 with butteraugli iterations enabled. The bad index
//! `1073788476 = 0x4000B63C` corresponds to a hash-chain pointer that
//! escaped its window-sized array.
//!
//! ## Fix
//!
//! `find_matches` now masks every chain pointer load with `window_mask`
//! before using it as an index into window-sized arrays (`val`, `zeros`,
//! `chain`, `chainz`). One AND per follow; correctness-preserving since
//! all legitimate chain values fit in `[0, window_size)` already.

use jxl_encoder::{LossyConfig, PixelLayout};

const TRIGGER_PNG: &[u8] = include_bytes!("images/lz77_oob_trigger_sz96.png");

fn decode_trigger() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(TRIGGER_PNG).expect("decode trigger PNG");
    let (w, h) = (img.width(), img.height());
    let rgb = img.to_rgb8();
    (rgb.into_raw(), w, h)
}

#[test]
fn lz77_chain_does_not_panic_on_v09_trigger_image() {
    // The exact image, distance, and effort that crashed v09. Without the
    // window_mask fix this panics with `index out of bounds: the len is
    // 262144 but the index is 1073788476`.
    let (pixels, w, h) = decode_trigger();
    let bytes = LossyConfig::new(4.0)
        .with_effort(9)
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("encode failed");
    assert_eq!(&bytes[..2], &[0xFF, 0x0A], "missing JXL signature");
    assert!(bytes.len() < pixels.len(), "output suspiciously large");
}

#[test]
fn lz77_chain_does_not_panic_across_full_v09_grid() {
    // Full sweep config that v09 used — distance × effort × biters grid.
    // This is the configuration matrix the production sweep traverses.
    let (pixels, w, h) = decode_trigger();
    for distance in [0.5f32, 1.0, 2.0, 4.0, 8.0] {
        for effort in [5u8, 7, 9] {
            for biters in [0u32, 1, 2] {
                let bytes = LossyConfig::new(distance)
                    .with_effort(effort)
                    .with_butteraugli_iters(biters)
                    .encode(&pixels, w, h, PixelLayout::Rgb8)
                    .unwrap_or_else(|e| {
                        panic!(
                            "encode failed for d={} e={} biters={}: {:?}",
                            distance, effort, biters, e
                        )
                    });
                assert_eq!(
                    &bytes[..2],
                    &[0xFF, 0x0A],
                    "missing JXL signature for d={} e={} biters={}",
                    distance,
                    effort,
                    biters
                );
            }
        }
    }
}
