//! Roundtrip proof for the lossy alpha pipeline (follow-on to W4-2-r).
//!
//! `LossyConfig::with_alpha_distance(Some(d))` with `d > 0` engages a
//! separate integer quantizer on the alpha extras sub-bitstream (libjxl
//! `enc_modular.cc:973-1027` + `QuantizeChannel`). This test:
//!
//! 1. Verifies bytes differ from the lossless baseline at `d=1.0` and
//!    `d=10.0` (wiring proof — also covered in lossy_knobs_wiring.rs
//!    with d=2.0 / d=10.0).
//! 2. Verifies the decoded alpha plane changes vs the input when
//!    `d=10.0` (q=15) — the lossy contract.
//! 3. Verifies the decoded RGB plane is preserved bit-for-bit between
//!    alpha-lossless and alpha-lossy encodes (alpha_distance must not
//!    leak into the color path).
//!
//! Decoded via jxl-rs (the primary roundtrip decoder per project
//! CLAUDE.md).

use jxl_encoder::{LossyConfig, PixelLayout};

const W: u32 = 32;
const H: u32 = 32;

/// Multi-frequency RGBA buffer with a non-trivial alpha plane so a
/// regression that drops alpha or "fills with 0xFF" cannot pass.
fn rgba_buf() -> Vec<u8> {
    let mut buf = vec![0u8; (W * H * 4) as usize];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let idx = ((y as u32 * W + x as u32) * 4) as usize;
            // RGB: smooth multi-frequency content.
            let fx = x as f32 / W as f32;
            let fy = y as f32 / H as f32;
            buf[idx] = ((0.5 + 0.4 * (fx * 11.0).sin() * (fy * 9.0).cos()) * 255.0)
                .clamp(0.0, 255.0) as u8;
            buf[idx + 1] = ((0.4 + 0.3 * (fx * 7.0).cos()) * 255.0).clamp(0.0, 255.0) as u8;
            buf[idx + 2] = ((0.5 + 0.4 * (fy * 13.0).sin()) * 255.0).clamp(0.0, 255.0) as u8;
            // Alpha: diamond with per-pixel modulation. Spans the full
            // 0..255 range so quantization at q=15 produces a clearly
            // detectable error on at least some pixels.
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

/// Decode a JXL bitstream as RGBA8 via jxl-rs. Returns the interleaved
/// RGBA byte buffer.
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

/// Extract just the alpha plane (every 4th byte starting at offset 3).
fn alpha_only(rgba: &[u8]) -> Vec<u8> {
    rgba.iter().skip(3).step_by(4).copied().collect()
}

/// Extract just the RGB bytes (drop every 4th).
fn rgb_only(rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len() * 3 / 4);
    for ch in rgba.as_chunks::<4>().0 {
        out.extend_from_slice(&ch[..3]);
    }
    out
}

/// Mean absolute error between two equal-length byte slices.
fn mae(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.len() as f64
}

#[test]
fn alpha_distance_high_loses_alpha_precision() {
    let buf = rgba_buf();
    let input_alpha = alpha_only(&buf);

    let bytes_lossless = LossyConfig::new(1.0)
        .encode(&buf, W, H, PixelLayout::Rgba8)
        .expect("lossless alpha encode");
    let bytes_lossy = LossyConfig::new(1.0)
        .with_alpha_distance(Some(10.0))
        .encode(&buf, W, H, PixelLayout::Rgba8)
        .expect("lossy alpha encode (d=10)");

    // Wiring: bytes must differ.
    assert_ne!(
        bytes_lossless, bytes_lossy,
        "alpha_distance=Some(10.0) must engage the lossy alpha path"
    );

    let dec_lossless = decode_jxl_rs_rgba8(&bytes_lossless);
    let dec_lossy = decode_jxl_rs_rgba8(&bytes_lossy);

    // Lossless path: alpha exact (modular gradient + LZ77 is lossless).
    let lossless_alpha = alpha_only(&dec_lossless);
    assert_eq!(
        lossless_alpha, input_alpha,
        "lossless alpha path must preserve every alpha byte exactly \
         (regression: lossless path now corrupts alpha)"
    );

    // Lossy path: alpha must differ from the input by a measurable
    // amount (q=15 at d=10.0: round-to-multiple-of-15 introduces up
    // to ±7 per pixel; MAE on a non-flat alpha plane should be > 1).
    let lossy_alpha = alpha_only(&dec_lossy);
    let alpha_mae = mae(&lossy_alpha, &input_alpha);
    assert!(
        alpha_mae > 1.0,
        "lossy alpha at d=10.0 must produce MAE > 1.0 vs input alpha \
         (got {alpha_mae:.3}); the lossy pipeline is silently lossless"
    );

    // Color (RGB) must be byte-identical between the two encodes —
    // alpha_distance must not leak into the VarDCT color path.
    let rgb_l = rgb_only(&dec_lossless);
    let rgb_v = rgb_only(&dec_lossy);
    assert_eq!(
        rgb_l, rgb_v,
        "alpha_distance must not affect the color (RGB) plane"
    );
}

#[test]
fn alpha_distance_default_is_lossless() {
    // The no-argument default (`alpha_distance = None`) must keep the
    // alpha plane bit-identical to the input. This is the
    // backwards-compat guarantee for every existing RGBA caller.
    let buf = rgba_buf();
    let input_alpha = alpha_only(&buf);

    let bytes = LossyConfig::new(1.0)
        .encode(&buf, W, H, PixelLayout::Rgba8)
        .unwrap();
    let dec = decode_jxl_rs_rgba8(&bytes);
    let dec_alpha = alpha_only(&dec);
    assert_eq!(
        dec_alpha, input_alpha,
        "default (alpha_distance=None) must produce lossless alpha"
    );
}

/// RGBA buffer with non-trivial RGB content but a CONSTANT (all-255)
/// opaque alpha plane. Mirrors the W13-4 audit `red_night_opaque` shape
/// (`100% opaque`) so we can repro the ChannelCompact-for-extras win.
fn rgba_opaque_buf() -> Vec<u8> {
    let mut buf = vec![0u8; (W * H * 4) as usize];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let idx = ((y as u32 * W + x as u32) * 4) as usize;
            // RGB: non-flat so the VarDCT path actually does work.
            let fx = x as f32 / W as f32;
            let fy = y as f32 / H as f32;
            buf[idx] = ((0.5 + 0.4 * (fx * 11.0).sin() * (fy * 9.0).cos()) * 255.0)
                .clamp(0.0, 255.0) as u8;
            buf[idx + 1] = ((0.4 + 0.3 * (fx * 7.0).cos()) * 255.0).clamp(0.0, 255.0) as u8;
            buf[idx + 2] = ((0.5 + 0.4 * (fy * 13.0).sin()) * 255.0).clamp(0.0, 255.0) as u8;
            // Alpha: every pixel exactly 255 (the ChannelCompact gate).
            buf[idx + 3] = 255;
        }
    }
    buf
}

#[test]
fn opaque_alpha_survives_high_alpha_distance_via_channel_compact() {
    // W13-4 audit gap: lossy alpha pipeline at `alpha_distance=5.0`
    // computes `q=7` and snaps the constant `255` value to `252`
    // (`(255 + 3) / 7 * 7 = 36 * 7 = 252`), giving MAE=3.000 on a
    // 100%-opaque alpha plane vs cjxl-default's MAE=0.000.
    //
    // ChannelCompact-for-extras (single-channel `kPalette` with
    // `num_c=1, nb_colors=1`) preserves the constant exactly: the
    // palette meta channel holds `255` at `q=1` (meta channels skip
    // quantization) and the index channel is all zeros, so the
    // decoder reconstructs `palette[index=0] = 255`.
    //
    // This test asserts MAE = 0 at `alpha_distance=5.0` for a constant
    // alpha plane — the exact fix the W13-4 audit called for.
    let buf = rgba_opaque_buf();
    let input_alpha = alpha_only(&buf);
    assert!(
        input_alpha.iter().all(|&a| a == 255),
        "test image precondition: alpha must be 100% opaque"
    );

    let bytes = LossyConfig::new(1.0)
        .with_alpha_distance(Some(5.0))
        .encode(&buf, W, H, PixelLayout::Rgba8)
        .expect("opaque alpha + alpha_distance=5.0 encode");
    let dec = decode_jxl_rs_rgba8(&bytes);
    let dec_alpha = alpha_only(&dec);
    let alpha_mae_v = mae(&dec_alpha, &input_alpha);
    assert_eq!(
        alpha_mae_v, 0.0,
        "ChannelCompact must preserve constant alpha exactly at d=5.0 \
         (got MAE={alpha_mae_v:.4}, want 0.0); regression: \
         255→252 snap is back"
    );
    assert!(
        dec_alpha.iter().all(|&a| a == 255),
        "every decoded alpha byte must be exactly 255 (ChannelCompact win)"
    );
}

#[test]
fn opaque_alpha_survives_all_lossy_distances_via_channel_compact() {
    // Sweep the four distances from the W13-4 audit (0.5, 1.0, 2.0,
    // 5.0). At every distance, a 100%-opaque alpha plane must decode
    // bit-identically — the ChannelCompact `nb_colors=1` path is
    // independent of `q` (the meta channel is at `q=1`, the index
    // channel is all-zeros and `snap(0, q)=0`).
    let buf = rgba_opaque_buf();
    let input_alpha = alpha_only(&buf);
    for &ad in &[0.5_f32, 1.0, 2.0, 5.0] {
        let bytes = LossyConfig::new(1.0)
            .with_alpha_distance(Some(ad))
            .encode(&buf, W, H, PixelLayout::Rgba8)
            .unwrap_or_else(|e| panic!("encode failed at alpha_distance={ad}: {e:?}"));
        let dec = decode_jxl_rs_rgba8(&bytes);
        let dec_alpha = alpha_only(&dec);
        let alpha_mae_v = mae(&dec_alpha, &input_alpha);
        assert_eq!(
            alpha_mae_v, 0.0,
            "constant opaque alpha at alpha_distance={ad} must \
             roundtrip exactly via ChannelCompact (got MAE={alpha_mae_v:.4})"
        );
    }
}

/// Multi-group sibling of [`opaque_alpha_survives_high_alpha_distance_via_channel_compact`].
/// Uses a 400×267 RGBA image (the same shape as the W13-4 audit
/// `red_night_opaque` corpus image) so the extras sub-bitstream is
/// split across multiple HF groups. Each per-group region is also
/// constant-alpha, so each HF group's `write_modular_extras_subbitstream`
/// independently emits a `kPalette(num_c=1, nb_colors=1)` transform.
#[test]
fn opaque_alpha_multigroup_survives_high_alpha_distance_via_channel_compact() {
    const MW: u32 = 400;
    const MH: u32 = 267;
    let mut buf = vec![0u8; (MW * MH * 4) as usize];
    for y in 0..MH as i32 {
        for x in 0..MW as i32 {
            let idx = ((y as u32 * MW + x as u32) * 4) as usize;
            // Multi-frequency RGB so the VarDCT path actually does work.
            let fx = x as f32 / MW as f32;
            let fy = y as f32 / MH as f32;
            buf[idx] = ((0.5 + 0.4 * (fx * 11.0).sin() * (fy * 9.0).cos()) * 255.0)
                .clamp(0.0, 255.0) as u8;
            buf[idx + 1] = ((0.4 + 0.3 * (fx * 7.0).cos()) * 255.0).clamp(0.0, 255.0) as u8;
            buf[idx + 2] = ((0.5 + 0.4 * (fy * 13.0).sin()) * 255.0).clamp(0.0, 255.0) as u8;
            buf[idx + 3] = 255; // constant 100%-opaque alpha
        }
    }

    let bytes = LossyConfig::new(1.0)
        .with_alpha_distance(Some(5.0))
        .encode(&buf, MW, MH, PixelLayout::Rgba8)
        .expect("multi-group opaque-alpha + alpha_distance=5.0 encode");

    // Decode via jxl-rs and verify EVERY alpha byte is exactly 255.
    let dec = decode_multi(MW, MH, &bytes);
    let dec_alpha: Vec<u8> = dec.iter().skip(3).step_by(4).copied().collect();
    let max_err = dec_alpha.iter().map(|&a| 255i32 - a as i32).max().unwrap();
    let min_err = dec_alpha.iter().map(|&a| 255i32 - a as i32).min().unwrap();
    let alpha_mae_v: f64 = dec_alpha
        .iter()
        .map(|&a| (255i32 - a as i32).unsigned_abs() as f64)
        .sum::<f64>()
        / dec_alpha.len() as f64;
    assert_eq!(
        alpha_mae_v, 0.0,
        "multi-group opaque alpha must roundtrip exactly at d=5.0 via \
         per-HF-group ChannelCompact (got MAE={alpha_mae_v:.4}, \
         min_err={min_err}, max_err={max_err})"
    );
}

/// Decode an arbitrary-dimension JXL bitstream as RGBA8 via jxl-rs.
/// Used by the multi-group test (the original [`decode_jxl_rs_rgba8`]
/// helper hard-codes `W × H = 32 × 32`-shaped pieces of the test path).
fn decode_multi(width: u32, height: u32, data: &[u8]) -> Vec<u8> {
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
    let (w, h) = basic_info.size;
    assert_eq!(w as u32, width, "width mismatch");
    assert_eq!(h as u32, height, "height mismatch");
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
    let mut output_image = Image::<u8>::new((w * channels, h)).expect("alloc");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (w * channels, h),
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

    let mut pixels = Vec::with_capacity(w * h * channels);
    for y in 0..h {
        pixels.extend_from_slice(output_image.row(y));
    }
    pixels
}
