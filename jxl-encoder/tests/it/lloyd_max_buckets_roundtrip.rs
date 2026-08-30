//! Roundtrip validation for the EX-J5 reinterpretation:
//! **Lloyd-Max iterative clustering for MA-tree bucket boundaries on
//! residual-energy proxy properties (4 = `|N|`, 5 = `|W|`, 15 = `wp_max_error`)**.
//!
//! The flag is opt-in via `LosslessInternalParams::lloyd_max_buckets`. When
//! flipped on, the encoder picks different candidate splitvals for the
//! MA tree learner — this changes the chosen tree splits, the encoded
//! bytes, and the bitstream byte-stream. The decoder side is unchanged
//! (the spec property set is intact; only encoder-side splitval picks
//! differ), so all three reference decoders must accept the output and
//! reconstruct pixel-exact lossless output.
//!
//! Layer 1 (this file): jxl-oxide pixel-exact lossless decode on the
//! Lloyd-Max-encoded bitstream.
//!
//! Layer 2 (manual / future): djxl + jxl-rs decoder roundtrip. The
//! Lloyd-Max bitstream is fully spec-legal; the differences from the
//! default sort-quantile output live entirely in encoder-side splitval
//! choices that the tree's own header materialises before any decoder
//! ever sees them.

#![cfg(feature = "__expert")]

use std::io::Cursor;

use jxl_encoder::LosslessInternalParams;
use jxl_encoder::api::{LosslessConfig, PixelLayout};

/// Decode `jxl` via jxl-oxide and return the linear-sRGB pixel array
/// alongside (width, height). Pixel-exact lossless decode for an 8-bit
/// sRGB roundtrip requires no transfer-function conversion.
fn decode_to_rgb8(jxl: &[u8], w: u32, h: u32) -> Vec<u8> {
    let image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(jxl))
        .unwrap_or_else(|e| panic!("jxl-oxide parse failed: {:?}", e));
    assert_eq!(image.width(), w);
    assert_eq!(image.height(), h);
    let render = image
        .render_frame(0)
        .unwrap_or_else(|e| panic!("jxl-oxide render failed: {:?}", e));
    let stream = render.image_all_channels();
    let buf: Vec<f32> = stream.buf().to_vec();
    assert_eq!(buf.len(), (w as usize) * (h as usize) * 3);
    // Lossless 8-bit roundtrip: convert f32 ∈ [0,1] → u8 via 0..=255 round.
    buf.into_iter()
        .map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// Build a small synthetic RGB image with energy-shaped per-row activity
/// (alternating smooth bands + noisy bands). This exercises the
/// residual-energy proxy properties (`|N|`, `|W|`, `wp_max_error`) that
/// Lloyd-Max refines, without depending on an external corpus.
fn textured_rgb8() -> (Vec<u8>, u32, u32) {
    const W: u32 = 96;
    const H: u32 = 96;
    let mut pixels = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            let band = (y / 8) & 1;
            let (r, g, b) = if band == 0 {
                // Smooth gradient — low |N|, low |W|.
                (
                    (x * 255 / W) as u8,
                    ((x + y) * 127 / (W + H)) as u8,
                    (y * 255 / H) as u8,
                )
            } else {
                // Noisy texture — high |N|, high |W|, high wp_max_error.
                let mix = (x.wrapping_mul(31) ^ y.wrapping_mul(73)) as u8;
                (
                    mix.wrapping_add(13),
                    mix.wrapping_mul(3).wrapping_add(127),
                    mix.wrapping_mul(7).wrapping_add(64),
                )
            };
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    (pixels, W, H)
}

#[test]
fn lloyd_max_lossless_roundtrip_via_oxide() {
    let (rgb, w, h) = textured_rgb8();

    let mut params = LosslessInternalParams::default();
    params.lloyd_max_buckets = Some(true);
    let cfg = LosslessConfig::new()
        .with_effort(7)
        .with_threads(1)
        .with_internal_params(params);

    let bytes = cfg
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("Lloyd-Max lossless encode");

    // jxl-oxide → byte-identical reconstruction for lossless 8-bit RGB.
    let decoded = decode_to_rgb8(&bytes, w, h);
    assert_eq!(
        decoded, rgb,
        "Lloyd-Max-encoded bitstream must decode pixel-exact via jxl-oxide"
    );
}

#[test]
fn lloyd_max_changes_bytes_vs_sort_quantile() {
    // Sanity: confirm the flag actually has an effect on the encoded bytes
    // for an energy-shaped image. If the flag were inert (e.g. because the
    // tree learner happens to pick the same splits with either threshold
    // set on this input), this test would fail and the bench's headline
    // delta would be from another cause.
    let (rgb, w, h) = textured_rgb8();

    let bytes_default = LosslessConfig::new()
        .with_effort(7)
        .with_threads(1)
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("default lossless encode");

    let mut params = LosslessInternalParams::default();
    params.lloyd_max_buckets = Some(true);
    let bytes_lloyd = LosslessConfig::new()
        .with_effort(7)
        .with_threads(1)
        .with_internal_params(params)
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("Lloyd-Max lossless encode");

    assert_ne!(
        bytes_default, bytes_lloyd,
        "lloyd_max_buckets=true must change the encoded bytes on energy-shaped input"
    );
}

#[test]
fn lloyd_max_default_off_byte_identical_to_baseline() {
    // The opt-in design: `Some(false)` and `None` must both produce the
    // same bytes as the default (no flag set). This is what guarantees
    // hash-locks stay byte-identical at the default-OFF profile.
    let (rgb, w, h) = textured_rgb8();

    let bytes_default = LosslessConfig::new()
        .with_effort(7)
        .with_threads(1)
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("default lossless encode");

    let params_none = LosslessInternalParams::default();
    assert!(params_none.lloyd_max_buckets.is_none());
    let bytes_none = LosslessConfig::new()
        .with_effort(7)
        .with_threads(1)
        .with_internal_params(params_none)
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("none override lossless encode");

    let mut params_false = LosslessInternalParams::default();
    params_false.lloyd_max_buckets = Some(false);
    let bytes_false = LosslessConfig::new()
        .with_effort(7)
        .with_threads(1)
        .with_internal_params(params_false)
        .encode(&rgb, w, h, PixelLayout::Rgb8)
        .expect("Some(false) override lossless encode");

    assert_eq!(
        bytes_default, bytes_none,
        "None override must match default bytes"
    );
    assert_eq!(
        bytes_default, bytes_false,
        "Some(false) override must match default bytes"
    );
}
