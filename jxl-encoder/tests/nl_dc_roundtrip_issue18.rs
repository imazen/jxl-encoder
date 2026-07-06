// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression test for **imazen/zenjxl#18**: encoding at a near-lossless
//! butteraugli distance (<= 0.02) produced a BROKEN round-trip — the decoded
//! image scored ssim2 ~34 (PSNR ~15-35 dB) versus ~96 at distance 0.03, while
//! the encoder spent the MOST bits (largest file).
//!
//! Root cause (two layers):
//!  1. Quantised DC coefficients were stored as `i16`. At fine distances the DC
//!     `inv_factor` (`INV_DC_QUANT[c] * scale_dc * dc_mul`) grows large enough
//!     that legitimate DC coefficients exceed `i16::MAX`, and the saturating
//!     `.round() as i16` cast collapsed the low-frequency image → garbage.
//!     Fixed by widening DC storage to `i32` (matches the wire format, which
//!     already packs `i32` DC residuals, and libjxl's internal `i32` DC).
//!  2. The resulting large DC also broke spec-conformance below distance 0.03
//!     (imazen/jxl-encoder#94): the DC overflows the `i16` sample buffers a
//!     strict decoder uses for `modular_16bit_buffer_sufficient = true`,
//!     desynchronising the DC modular ANS stream. That was a separate stopgap
//!     `0.03` distance floor, now removed — #94 signals 32-bit modular buffers
//!     when the DC overflows `i16`, so sub-0.03 is genuinely conformant.
//!     `nl_dc_conformance_issue94.rs` is the dedicated jxl-oxide gate.
//!
//! This test drives the exact reported failure: encode a real high-contrast
//! sRGB image at distance 0.02 AND 0.03, decode with the pure-Rust
//! `zenjxl-decoder`, and assert the reconstruction is near-lossless and NOT
//! dramatically worse at the finer distance. It FAILS on pre-fix code
//! (`p02` ~15-35 dB, ~60-80 dB worse than `p03`) and PASSES post-fix.
//!
//! No runtime skip: the fixture is a committed file (`tests/images/`) and a
//! load failure is a hard panic, never a silent early return.

use jxl_encoder::{LossyConfig, PixelLayout};

/// Load the committed sRGB fixture and crop to a bounded, high-contrast region.
///
/// `frymire-srgb.png` is a vivid screen illustration — bright saturated blocks
/// give large Y-channel DC, the exact content class that overflowed the old
/// `i16` DC storage. The crop keeps the encode fast while preserving that DC
/// stress.
fn load_fixture() -> (Vec<u8>, u32, u32) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/frymire-srgb.png");
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("committed fixture {path} failed to load: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    let (cw, ch) = (w.min(384), h.min(384));
    let mut out = Vec::with_capacity((cw * ch * 3) as usize);
    for y in 0..ch {
        for x in 0..cw {
            out.extend_from_slice(&img.get_pixel(x, y).0);
        }
    }
    (out, cw, ch)
}

/// Encode `rgb` at `distance`, decode with `zenjxl-decoder`, return RGB PSNR
/// (dB) vs the original. `INFINITY` means a bit-exact reconstruction.
fn roundtrip_psnr(rgb: &[u8], w: u32, h: u32, distance: f32) -> f64 {
    let jxl = LossyConfig::new(distance)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .unwrap_or_else(|e| panic!("encode at distance {distance} failed: {e:?}"));

    let decoded = zenjxl_decoder::decode(&jxl)
        .unwrap_or_else(|e| panic!("decode of distance-{distance} bitstream failed: {e:?}"));

    assert_eq!(
        (decoded.width as u32, decoded.height as u32),
        (w, h),
        "decoded dimensions differ from source at distance {distance}"
    );
    assert_eq!(
        decoded.channels, 4,
        "expected RGBA output for a color image at distance {distance}"
    );
    assert_eq!(
        decoded.data.len(),
        (w * h * 4) as usize,
        "decoded buffer size mismatch at distance {distance}"
    );

    let mut sse = 0.0f64;
    for (src, dst) in rgb.chunks_exact(3).zip(decoded.data.chunks_exact(4)) {
        for c in 0..3 {
            let d = src[c] as i32 - dst[c] as i32;
            sse += (d * d) as f64;
        }
    }
    let n = (w * h * 3) as f64;
    let mse = sse / n;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

#[test]
fn near_lossless_distance_002_roundtrip_is_not_garbage_issue18() {
    let (rgb, w, h) = load_fixture();

    let p02 = roundtrip_psnr(&rgb, w, h, 0.02);
    let p03 = roundtrip_psnr(&rgb, w, h, 0.03);

    // Absolute quality: pre-fix the i16 DC saturation made this ~15-35 dB
    // (destroyed low-frequency structure). A conformant near-lossless encode
    // is far above 40 dB (measured ~58 dB on this fixture post-fix).
    assert!(
        p02 >= 40.0,
        "distance-0.02 round-trip PSNR {p02:.2} dB is garbage — issue #18 DC saturation \
         has regressed (expected near-lossless, >= 40 dB)"
    );

    // Monotonicity: a FINER distance must never decode dramatically worse than a
    // coarser one. Pre-fix, distance 0.02 (~15 dB) was ~80 dB worse than 0.03
    // (~96 dB) despite a larger file. Post-fix, 0.02 genuinely encodes at 0.02
    // (no floor) and decodes at least as well as 0.03 (finer = higher PSNR).
    assert!(
        p02 >= p03 - 1.0,
        "distance-0.02 PSNR {p02:.2} dB is worse than distance-0.03 {p03:.2} dB — \
         finer distance decoded worse (issue #18 monotonicity inversion)"
    );
}
