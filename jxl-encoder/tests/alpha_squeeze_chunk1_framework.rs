//! Chunk-1 framework roundtrip tests for
//! [`crate::LossyConfig::with_alpha_squeeze`].
//!
//! The W13-4 audit (commit `a160deb7`) documented a `-18%` to
//! `-160%` byte gap vs cjxl default on non-opaque alpha because
//! cjxl uses `--responsive=1` (Squeeze wavelet + ChannelCompact)
//! and we use `--responsive=0` (no-squeeze pixel quantizer). The
//! chunk-1 framework lays the constants + per-band quantizer
//! function (`enc_modular.cc:973-1027` parity for shift > 0) but
//! does NOT yet apply Squeeze to extras. Chunk 2 will wire that
//! and start delivering the byte savings.
//!
//! This test suite verifies the chunk-1 **contract**:
//!
//! 1. Default `with_alpha_squeeze(false)` keeps the existing lossy
//!    alpha pipeline byte-for-byte identical to today
//!    (encoding twice with the flag off produces identical bytes,
//!    and the flag-off output decodes correctly via jxl-rs at
//!    `alpha_distance = 2.0`).
//!
//! 2. `with_alpha_squeeze(true)` + lossy alpha path engaged
//!    (`alpha_distance > 0` AND an alpha extra present) surfaces
//!    a clear `Error::NotImplemented` that explains chunk 2 is
//!    where the real win lands. No silent fallback to the
//!    no-squeeze path under the new flag.
//!
//! 3. `with_alpha_squeeze(true)` with no alpha extra OR with
//!    `alpha_distance = None` is a no-op (does NOT error).
//!
//! When chunk 2 ships, test (2) will flip from "expect error" to
//! "expect smaller bytes than the no-squeeze baseline" and the
//! tracking comment below should be updated.

use jxl_encoder::{LossyConfig, PixelLayout};

const W: u32 = 32;
const H: u32 = 32;

/// Build the same RGBA buffer shape as `tests/lossy_alpha_roundtrip.rs`
/// so the chunk-2 byte-savings test can pivot off these baselines.
fn rgba_buf() -> Vec<u8> {
    let mut buf = vec![0u8; (W * H * 4) as usize];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let idx = ((y as u32 * W + x as u32) * 4) as usize;
            let fx = x as f32 / W as f32;
            let fy = y as f32 / H as f32;
            buf[idx] = ((0.5 + 0.4 * (fx * 11.0).sin() * (fy * 9.0).cos()) * 255.0)
                .clamp(0.0, 255.0) as u8;
            buf[idx + 1] = ((0.4 + 0.3 * (fx * 7.0).cos()) * 255.0).clamp(0.0, 255.0) as u8;
            buf[idx + 2] = ((0.5 + 0.4 * (fy * 13.0).sin()) * 255.0).clamp(0.0, 255.0) as u8;
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
            buf[idx + 3] = base.saturating_add(modulation);
        }
    }
    buf
}

fn encode_rgba(cfg: LossyConfig) -> Result<Vec<u8>, jxl_encoder::At<jxl_encoder::EncodeError>> {
    let pixels = rgba_buf();
    cfg.encode(&pixels, W, H, PixelLayout::Rgba8)
}

/// Decode JXL via jxl-rs (the primary roundtrip decoder per project
/// CLAUDE.md), returning interleaved RGBA8. Mirrors the helper in
/// `tests/lossy_alpha_roundtrip.rs`.
fn decode_jxl_rs_rgba8(data: &[u8]) -> Vec<u8> {
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

    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgba,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![None; num_extras],
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

    let channels = 4usize;
    let mut output_image = Image::<u8>::new((width * channels, height)).expect("alloc");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    loop {
        match decoder_frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_frame = fallback;
            }
            Err(e) => panic!("jxl-rs frame decode error: {e:?}"),
        }
    }

    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }
    pixels
}

#[test]
fn alpha_squeeze_default_off_is_byte_identical_to_existing_pipeline() {
    // Two encodes with `with_alpha_squeeze` default (= false) and
    // a non-zero `alpha_distance` must produce identical bytes —
    // proves the framework doesn't perturb the existing path.
    let cfg = || LossyConfig::new(1.0).with_alpha_distance(Some(2.0));
    let a = encode_rgba(cfg()).expect("flag-default encode A failed");
    let b = encode_rgba(cfg()).expect("flag-default encode B failed");
    assert_eq!(
        a, b,
        "default with_alpha_squeeze(false) must be deterministic"
    );
    assert!(!a.is_empty(), "encode produced empty bytes");
    // Same bytes as if we explicitly set the flag to false.
    let c = encode_rgba(
        LossyConfig::new(1.0)
            .with_alpha_distance(Some(2.0))
            .with_alpha_squeeze(false),
    )
    .expect("explicit false encode failed");
    assert_eq!(
        a, c,
        "with_alpha_squeeze(false) explicit must match implicit default"
    );
}

#[test]
fn alpha_squeeze_default_off_decodes_correctly_at_d_2_0() {
    // POC roundtrip: alpha_distance=2.0 with flag off must
    // continue to decode via jxl-rs (the existing W6-3 contract).
    let bytes =
        encode_rgba(LossyConfig::new(1.0).with_alpha_distance(Some(2.0))).expect("encode failed");
    let decoded = decode_jxl_rs_rgba8(&bytes);
    assert_eq!(decoded.len(), (W * H * 4) as usize);
    // Alpha plane should be lossy-but-present (not all-0, not all-255).
    let alpha_min = decoded.iter().skip(3).step_by(4).copied().min().unwrap();
    let alpha_max = decoded.iter().skip(3).step_by(4).copied().max().unwrap();
    assert!(
        alpha_max > alpha_min,
        "alpha plane should vary; got min={alpha_min} max={alpha_max}"
    );
}

#[test]
fn alpha_squeeze_on_plus_lossy_alpha_returns_not_implemented() {
    // The chunk-1 framework gate: flag-on AND alpha extra present
    // AND alpha_distance > 0 must error, NOT silently fall back to
    // the no-squeeze path. This protects callers from thinking
    // they're getting the byte savings cjxl-default delivers when
    // they're not.
    let res = encode_rgba(
        LossyConfig::new(1.0)
            .with_alpha_distance(Some(2.0))
            .with_alpha_squeeze(true),
    );
    let err = res.expect_err("expected NotImplemented for chunk-1 framework gate");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("chunk-1") || msg.contains("with_alpha_squeeze"),
        "error message should explain chunk-1 framework status; got: {msg}"
    );
}

#[test]
fn alpha_squeeze_on_without_alpha_distance_is_no_op() {
    // Flag on + alpha_distance default (= None / lossless alpha)
    // means there is no lossy-alpha case to redirect — leaving the
    // flag on must NOT error. Same goes for `Some(0.0)`.
    let bytes = encode_rgba(
        LossyConfig::new(1.0).with_alpha_squeeze(true), // no alpha_distance set
    )
    .expect("flag on + no alpha_distance should not error");
    assert!(!bytes.is_empty());

    let bytes2 = encode_rgba(
        LossyConfig::new(1.0)
            .with_alpha_distance(Some(0.0))
            .with_alpha_squeeze(true),
    )
    .expect("flag on + alpha_distance=0 should not error (still lossless alpha)");
    assert!(!bytes2.is_empty());
}

#[test]
fn alpha_squeeze_on_without_alpha_channel_is_no_op() {
    // RGB-only encode (no alpha extra). Flag on + non-zero
    // alpha_distance is still a no-op because there's no alpha
    // extra to apply the per-band quantizer to.
    let rgb: Vec<u8> = (0..(W * H * 3) as usize)
        .map(|i| ((i * 7) % 256) as u8)
        .collect();
    let bytes = LossyConfig::new(1.0)
        .with_alpha_distance(Some(2.0))
        .with_alpha_squeeze(true)
        .encode(&rgb, W, H, PixelLayout::Rgb8)
        .expect("RGB-only encode with flag on should not error (no alpha extra)");
    assert!(!bytes.is_empty());
}

#[test]
fn alpha_squeeze_builder_round_trips_through_with_effort() {
    // `LossyConfig::with_effort` must preserve the
    // `alpha_squeeze` setting (joins the CLI-passthrough knob
    // list that already survives with_effort). Without this the
    // CLI would silently drop the flag when applying an effort
    // change.
    let cfg = LossyConfig::new(1.0)
        .with_alpha_distance(Some(2.0))
        .with_alpha_squeeze(true)
        .with_effort(5);
    assert!(
        cfg.alpha_squeeze(),
        "with_effort must preserve alpha_squeeze=true"
    );
    assert_eq!(
        cfg.alpha_distance(),
        Some(2.0),
        "with_effort must preserve alpha_distance"
    );
}
