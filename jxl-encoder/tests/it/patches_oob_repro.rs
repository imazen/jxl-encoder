// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression for an OOB index panic in `vardct/patches.rs`
//! `find_text_like_patches` (DoS vector — same class as the LZ77 panic
//! patched in commit 1498053).
//!
//! Surfaced in zen-metrics sweep v11 on
//! `size-dense-renders/4cd...sz1280.png` after a butteraugli NaN
//! cascade left a corrupted BFS queue:
//!
//! ```text
//! thread '...' panicked at vardct/patches.rs:558:
//! index out of bounds: the len is 817920 but the index is 1073835134
//! ```
//!
//! The bad index `1073835134 = 0x4000B6BE` has bit 30 set — the same
//! corruption pattern as the LZ77 crash, suggesting a shared upstream
//! cause (NaN/Inf flowing into integer casts, possibly via the
//! `unsafe-performance` MaybeUninit scratch buffers — see
//! `vardct/common.rs`).
//!
//! ## Fix
//!
//! Three layers of defense added in this branch:
//!
//! 1. BFS pop in `find_text_like_patches` validates queue items at
//!    extraction time (commit 1498053).
//! 2. DFS pop in the same fn now does the same (this branch).
//! 3. NaN/Inf in XYB planes is sanitized to 0.0 before patches /
//!    splines / quant see them. The butteraugli loop also clamps NaN
//!    in `quant_field_float` after each iteration (this branch).
//!
//! The test below exercises the patches code path with the configurations
//! and content shapes that v11 reproduced on. It does NOT include the
//! exact `_sz1280.png` fixture (binary too large for inline inclusion;
//! tracked separately for fuzz-corpus pickup).

use jxl_encoder::{LossyConfig, PixelLayout};

/// Build a 1280×800 RGB8 image whose content shape triggers patches
/// detection (large flat areas, periodic repeats, hard edges) and is
/// known to push the butteraugli loop into the divergent regime that
/// previously cascaded NaN through to patches.
fn make_patches_trigger(width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            // Five-band layout: large solid background + repeated small
            // squares + alternating stripes + a soft gradient + noise.
            let band = x / 256;
            let (r, g, b) = match band % 5 {
                0 => (240u8, 240, 240), // background flat
                1 => {
                    // Periodic 16×16 squares (patch-friendly)
                    if (x / 16 + y / 16) & 1 == 0 {
                        (32, 32, 32)
                    } else {
                        (224, 224, 224)
                    }
                }
                2 => {
                    // 1-px stripes (high frequency, butteraugli-stressing)
                    if y & 1 == 0 { (255, 0, 0) } else { (0, 0, 255) }
                }
                3 => {
                    // Smooth gradient
                    let t = ((x % 256) as u32 * 255 / 255) as u8;
                    (t, t, t)
                }
                _ => {
                    // Pseudo-random per-pixel noise (deterministic)
                    let n = ((x * 31 + y * 17) ^ 0x55) as u8;
                    (n, n.wrapping_add(0x40), n.wrapping_add(0x80))
                }
            };
            out[i] = r;
            out[i + 1] = g;
            out[i + 2] = b;
        }
    }
    out
}

#[test]
fn patches_does_not_panic_on_high_freq_trigger() {
    let (w, h) = (1280u32, 800u32);
    let pixels = make_patches_trigger(w, h);

    // Exact configuration shape from the v11 sweep: high distance +
    // butteraugli iterations + e9 = the divergent regime where NaN
    // had been observed to reach patches detection.
    let bytes = LossyConfig::new(4.0)
        .with_effort(9)
        .with_butteraugli_iters(2)
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("encode panicked or errored on patches trigger");
    assert_eq!(&bytes[..2], &[0xFF, 0x0A], "missing JXL signature");
}

#[test]
fn patches_does_not_panic_across_distance_grid() {
    // Sweep across the distance × effort grid that v11 reproduced on.
    let (w, h) = (256u32, 256u32);
    let pixels = make_patches_trigger(w, h);
    for distance in [0.5f32, 1.0, 2.0, 4.0, 8.0] {
        for effort in [5u8, 7, 9] {
            let bytes = LossyConfig::new(distance)
                .with_effort(effort)
                .encode(&pixels, w, h, PixelLayout::Rgb8)
                .unwrap_or_else(|e| panic!("encode failed for d={distance} e={effort}: {e:?}"));
            assert_eq!(
                &bytes[..2],
                &[0xFF, 0x0A],
                "missing JXL signature for d={distance} e={effort}",
            );
        }
    }
}

#[test]
fn patches_handles_all_zero_input() {
    // All-zero RGB → XYB(0, 0, 0) (cbrt(bias) - cbrt(bias) = 0). This is
    // a finite, well-defined "all black" image — the encoder should
    // produce a valid bitstream without exercising any NaN-handling
    // path. Asserts upgraded in debug builds: `sanitize_xyb_planes`
    // would fire its debug_assert! if the XYB transform leaked
    // non-finite values for this input. (Originally written as a
    // "NaN-inducing" test under the mistaken belief that `log(0) = -Inf`
    // would fire — the real opsin transform is `cbrt(mixed + bias)`,
    // which is finite at zero. Renamed and re-purposed as a
    // "finite-input-stays-finite" smoke test.)
    let (w, h) = (256u32, 256u32);
    let pixels = vec![0u8; (w * h * 3) as usize];
    let bytes = LossyConfig::new(2.0)
        .with_effort(9)
        .with_butteraugli_iters(2)
        .encode(&pixels, w, h, PixelLayout::Rgb8)
        .expect("encode failed on all-zero (all-black) input");
    assert_eq!(&bytes[..2], &[0xFF, 0x0A], "missing JXL signature");
}
