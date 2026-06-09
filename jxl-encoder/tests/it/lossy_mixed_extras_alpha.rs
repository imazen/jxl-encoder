//! Roundtrip proof for the mixed-extras lossy-alpha pipeline (W8-2,
//! follow-on to W6-3 `bbf8a985`).
//!
//! W6-3 wired alpha lossy quantization but only fired when
//! `extras.len() == 1` — any image with alpha + a second extra (depth,
//! spot color, selection mask, ...) silently stayed all-lossless.
//! W8-2 dispatches the per-channel quantizer so alpha-typed extras get
//! the `alpha_distance`-derived `q` while non-alpha extras stay at
//! `q == 1` (lossless), even when present in the same frame.
//!
//! This test encodes a 32×32 RGB plane with **two** extra channels —
//! alpha (typed `ExtraChannelType::Alpha`) + depth (typed
//! `ExtraChannelType::Depth`) — at `alpha_distance=Some(10.0)`. At
//! `d_alpha=10.0` the alpha quantizer resolves to `q=15` (libjxl
//! `enc_modular.cc:973-1027`), so the alpha plane MUST come back with
//! measurable error vs the input (`MAE > 1.0`), while the depth plane
//! MUST be byte-identical (lossless `q=1` path).
//!
//! Decoded via jxl-rs (the primary roundtrip decoder per project
//! CLAUDE.md) — the extras come out as separate per-channel buffers
//! when `extra_channel_format` is `Some(...)` per channel.

#![cfg(feature = "rate-control")]

use jxl_encoder::api::ExtraChannel;
use jxl_encoder::{LossyConfig, PixelLayout};

const W: u32 = 32;
const H: u32 = 32;

/// Smooth multi-frequency RGB so VarDCT actually exercises the cost
/// model (no degenerate flat blocks). Returns interleaved u8 of
/// length `W * H * 3`.
fn rgb_buf() -> Vec<u8> {
    let mut out = vec![0u8; (W * H * 3) as usize];
    for y in 0..H {
        for x in 0..W {
            let idx = ((y * W + x) * 3) as usize;
            let fx = x as f32 / W as f32;
            let fy = y as f32 / H as f32;
            out[idx] = ((0.5 + 0.4 * (fx * 11.0).sin() * (fy * 9.0).cos()) * 255.0)
                .clamp(0.0, 255.0) as u8;
            out[idx + 1] = ((0.4 + 0.3 * (fx * 7.0).cos()) * 255.0).clamp(0.0, 255.0) as u8;
            out[idx + 2] = ((0.5 + 0.4 * (fy * 13.0).sin()) * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Non-trivial alpha plane: diamond mask + per-pixel modulation.
/// Spans the full 0..255 range so q=15 quantization produces clearly
/// detectable error on at least some pixels.
fn alpha_buf() -> Vec<u8> {
    let mut a = vec![0u8; (W * H) as usize];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let cx = W as i32 / 2;
            let cy = H as i32 / 2;
            let dx = (x - cx).abs();
            let dy = (y - cy).abs();
            let dist = dx + dy;
            let radius = (W.min(H) / 2) as i32;
            let base = if dist > radius {
                0
            } else {
                ((radius - dist).clamp(0, 31) as u8).saturating_mul(8)
            };
            let modulation = ((x ^ y) & 0x07) as u8;
            a[(y as u32 * W + x as u32) as usize] = base.saturating_add(modulation);
        }
    }
    a
}

/// Non-trivial depth plane: ramp + per-pixel modulation. The ramp is
/// deliberately distinct from the alpha shape so a regression that
/// stores `alpha` in `depth`'s slot (or vice versa) fails the
/// byte-identical check.
fn depth_buf() -> Vec<u8> {
    let mut d = vec![0u8; (W * H) as usize];
    for y in 0..H {
        for x in 0..W {
            let ramp = ((x * 7 + y * 3) & 0xFF) as u8;
            let modulation = ((x ^ y) & 0x0F) as u8;
            d[(y * W + x) as usize] = ramp.wrapping_add(modulation);
        }
    }
    d
}

/// Decode a JXL bitstream as RGB + N extras, all as u8 buffers.
///
/// Returns `(rgb_interleaved, vec_of_extra_planes)`. `extra_channels`
/// in the file header drives `extra_channel_format` — each extra gets
/// `Some(U8)` so jxl-rs decodes it into its own output buffer.
fn decode_jxl_rs_rgb_with_extras(data: &[u8]) -> (Vec<u8>, Vec<Vec<u8>>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder_init = decoder;
    let mut decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_init = fallback;
            }
            Err(e) => panic!("jxl-rs header decode error: {e:?}"),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let num_extras = basic_info.extra_channels.len();

    // Request RGB color + Some(U8) per extra so each lands in its own
    // output buffer (not interleaved into RGBA).
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: (0..num_extras)
            .map(|_| Some(JxlDataFormat::U8 { bit_depth: 8 }))
            .collect(),
    });

    let mut decoder_frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame info error: {e:?}"),
        }
    };

    let color_channels = 3usize;
    let mut color_image = Image::<u8>::new((width * color_channels, height)).expect("alloc rgb");
    let mut extra_images: Vec<Image<u8>> = (0..num_extras)
        .map(|_| Image::<u8>::new((width, height)).expect("alloc extra"))
        .collect();

    let mut buffers: Vec<JxlOutputBuffer<'_>> = Vec::with_capacity(1 + num_extras);
    buffers.push(JxlOutputBuffer::from_image_rect_mut(
        color_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * color_channels, height),
            })
            .into_raw(),
    ));
    for ec_image in extra_images.iter_mut() {
        buffers.push(JxlOutputBuffer::from_image_rect_mut(
            ec_image
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (width, height),
                })
                .into_raw(),
        ));
    }

    loop {
        match decoder_frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_frame = fallback;
            }
            Err(e) => panic!("jxl-rs frame decode error: {e:?}"),
        }
    }

    let mut rgb = Vec::with_capacity(width * height * color_channels);
    for y in 0..height {
        rgb.extend_from_slice(color_image.row(y));
    }
    let extras: Vec<Vec<u8>> = extra_images
        .iter()
        .map(|img| {
            let mut planar = Vec::with_capacity(width * height);
            for y in 0..height {
                planar.extend_from_slice(img.row(y));
            }
            planar
        })
        .collect();
    (rgb, extras)
}

/// Mean absolute error between two equal-length byte slices.
fn mae(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "mae buffer length mismatch");
    let sum: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.len() as f64
}

#[test]
fn mixed_extras_alpha_lossy_depth_lossless() {
    let rgb = rgb_buf();
    let alpha = alpha_buf();
    let depth = depth_buf();

    // Mixed-extras list: alpha at index 0, depth at index 1. The wire
    // order must come back unchanged so the decoder pulls each
    // channel's region in the same order the encoder wrote it.
    let extras = [
        ExtraChannel::from_alpha_buf(&alpha, /* associated = */ false),
        ExtraChannel::depth(&depth),
    ];

    // alpha_distance = 10.0 → q = 15 (libjxl's no-squeeze formula
    // floor((1 << 8) * 0.5 * (1.0 / 10.0) * 163.84 / 256) = 15) so
    // round-to-multiple-of-15 introduces up to ±7 per alpha pixel.
    let cfg = LossyConfig::new(1.0).with_alpha_distance(Some(10.0));
    let bytes = cfg
        .encode_request(W, H, PixelLayout::Rgb8)
        .with_extra_channels(&extras)
        .encode(&rgb)
        .expect("mixed-extras lossy encode (alpha_distance=10.0)");

    assert_eq!(
        &bytes[..2],
        &[0xFF, 0x0A],
        "encoded bitstream must start with the JXL signature"
    );

    let (_dec_rgb, dec_extras) = decode_jxl_rs_rgb_with_extras(&bytes);
    assert_eq!(
        dec_extras.len(),
        2,
        "file header must carry exactly two extra channels (alpha + depth)"
    );

    // Channel 0 must be the alpha plane (lossy: MAE > 1.0 at q=15).
    let alpha_mae = mae(&dec_extras[0], &alpha);
    assert!(
        alpha_mae > 1.0,
        "alpha extra must be lossy at alpha_distance=10.0 (got MAE = {alpha_mae:.3}); the \
         per-channel dispatch is silently keeping alpha lossless"
    );

    // Channel 1 must be the depth plane (lossless: byte-identical).
    // This is the regression guard for W8-2: without per-channel
    // dispatch the shared multiplier would have quantized depth too.
    assert_eq!(
        dec_extras[1], depth,
        "depth extra must round-trip byte-identical (lossless); the per-channel quantizer \
         dispatch is leaking alpha's `q` into the depth plane"
    );
}

#[test]
fn mixed_extras_alpha_lossless_depth_lossless_byte_identical() {
    // Guard the no-op path: with `alpha_distance = None` the
    // mixed-extras frame must be byte-identical to the lossless
    // alpha + lossless depth pre-W8-2 wire. Combined with the 36/36
    // hash-locks in `hash_lock_features` (which cover the single-extra
    // RGBA path), this proves W8-2 does not perturb the default wire
    // for either single-extra or mixed-extras frames.
    let rgb = rgb_buf();
    let alpha = alpha_buf();
    let depth = depth_buf();

    let extras = [
        ExtraChannel::from_alpha_buf(&alpha, /* associated = */ false),
        ExtraChannel::depth(&depth),
    ];

    let cfg_none = LossyConfig::new(1.0); // alpha_distance = None
    let cfg_zero = LossyConfig::new(1.0).with_alpha_distance(Some(0.0));

    let bytes_none = cfg_none
        .encode_request(W, H, PixelLayout::Rgb8)
        .with_extra_channels(&extras)
        .encode(&rgb)
        .expect("mixed-extras lossless encode (alpha_distance=None)");
    let bytes_zero = cfg_zero
        .encode_request(W, H, PixelLayout::Rgb8)
        .with_extra_channels(&extras)
        .encode(&rgb)
        .expect("mixed-extras lossless encode (alpha_distance=Some(0.0))");

    assert_eq!(
        bytes_none, bytes_zero,
        "alpha_distance=None and alpha_distance=Some(0.0) must produce byte-identical bitstreams \
         on mixed-extras (alpha + depth) frames"
    );

    // Both planes must round-trip byte-identical when alpha is lossless.
    let (_rgb_dec, dec_extras) = decode_jxl_rs_rgb_with_extras(&bytes_none);
    assert_eq!(
        dec_extras.len(),
        2,
        "default mixed-extras must keep both alpha + depth in the file header"
    );
    assert_eq!(
        dec_extras[0], alpha,
        "alpha must round-trip byte-identical when alpha_distance is None"
    );
    assert_eq!(
        dec_extras[1], depth,
        "depth must round-trip byte-identical when alpha_distance is None"
    );
}
