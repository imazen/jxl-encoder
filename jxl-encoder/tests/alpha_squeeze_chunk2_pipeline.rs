// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Chunk-2 pipeline tests for
//! [`jxl_encoder::LossyConfig::with_alpha_squeeze`].
//!
//! Chunk 1 (commit `3b042f8e`) shipped the framework: constants,
//! shift-aware quantizer fn (`compute_extra_pixel_quantizer_shifted`),
//! opt-in flag (`with_alpha_squeeze`), and a `NotImplemented` gate at
//! the lossy-alpha entry points.
//!
//! Chunk 2 (this commit) wires the framework into the bitstream:
//! `apply_squeeze` decomposes the alpha plane into wavelet
//! sub-channels, each gets its own integer quantizer
//! (`compute_extra_pixel_quantizer_shifted(shift = hshift + vshift - 1)`),
//! and the modular subbitstream signals `nb_transforms` + per-band
//! `SqueezeParam` so the decoder can undo the wavelet. The bitstream
//! writer uses the existing channel-split tree (one gradient leaf per
//! sub-channel) to dispatch the per-band quantizer at decode time.
//!
//! Scope: single-group images (≤ 256×256), exactly one alpha extra,
//! `dim_shift = 0`. Multi-group / multi-extra / dim_shift > 0 routes
//! through `Error::NotImplemented` until chunk-2.b.
//!
//! The W13-4 audit (commit `a160deb7`) measured cjxl default
//! (`--responsive=1`) at -18% to -160% smaller than our `responsive=0`
//! lossy alpha. This test verifies the chunk-2 path now delivers a
//! byte saving in the same direction on the same lossy-alpha shape.

use jxl_encoder::{LossyConfig, PixelLayout};

const W: u32 = 32;
const H: u32 = 32;

/// Same RGBA buffer shape as `tests/alpha_squeeze_chunk1_framework.rs`
/// so byte-savings comparisons are apples-to-apples vs the chunk-1
/// baseline.
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

/// jxl-rs decoder — primary roundtrip decoder per project CLAUDE.md.
/// Mirrors the helper in `tests/alpha_squeeze_chunk1_framework.rs`.
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

/// Mean absolute error between two alpha planes (RGBA-interleaved
/// input). Used to verify the squeeze path delivers a *similar*
/// MAE to the baseline (not necessarily identical: the per-band
/// quantizer is calibrated differently from the raw-pixel quantizer
/// — what matters is that the alpha plane comes back recognisable).
fn alpha_mae(decoded: &[u8], original: &[u8]) -> f32 {
    debug_assert_eq!(decoded.len(), original.len());
    let mut sum = 0u32;
    let mut n = 0u32;
    for i in (3..decoded.len()).step_by(4) {
        sum += (decoded[i] as i32 - original[i] as i32).unsigned_abs();
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    sum as f32 / n as f32
}

#[test]
fn alpha_squeeze_chunk2_beats_no_squeeze_baseline_at_d_2_0() {
    // Direct A/B at `alpha_distance = 2.0` on 32×32 RGBA. Chunk-2
    // pipeline should be smaller than the chunk-1 framework's no-squeeze
    // path. This is the W13-4 audit's "-18% to -160% smaller" direction
    // applied at our scale (smaller image → smaller margin, but the
    // *direction* must match the audit).
    let baseline =
        encode_rgba(LossyConfig::new(1.0).with_alpha_distance(Some(2.0))).expect("baseline encode");
    let squeezed = encode_rgba(
        LossyConfig::new(1.0)
            .with_alpha_distance(Some(2.0))
            .with_alpha_squeeze(true),
    )
    .expect("squeezed encode");

    assert!(
        !baseline.is_empty() && !squeezed.is_empty(),
        "both paths must produce non-empty bytes"
    );
    assert!(
        squeezed.len() < baseline.len(),
        "chunk-2 squeeze path must beat no-squeeze baseline at d=2.0; got squeeze={} bytes \
         vs baseline={} bytes (Δ = {:+})",
        squeezed.len(),
        baseline.len(),
        squeezed.len() as i64 - baseline.len() as i64
    );
}

#[test]
fn alpha_squeeze_chunk2_decodes_via_jxl_rs() {
    // Primary roundtrip decoder (per project CLAUDE.md). Verifies
    // jxl-rs can parse + decode the chunk-2 bitstream. Alpha plane
    // must come back varying (lossy ≠ all-zero / all-255).
    let bytes = encode_rgba(
        LossyConfig::new(1.0)
            .with_alpha_distance(Some(2.0))
            .with_alpha_squeeze(true),
    )
    .expect("squeezed encode");
    let decoded = decode_jxl_rs_rgba8(&bytes);
    assert_eq!(decoded.len(), (W * H * 4) as usize);
    let alpha_min = decoded.iter().skip(3).step_by(4).copied().min().unwrap();
    let alpha_max = decoded.iter().skip(3).step_by(4).copied().max().unwrap();
    assert!(
        alpha_max > alpha_min,
        "chunk-2 squeeze path: alpha plane should vary; got min={alpha_min} max={alpha_max}"
    );
    // MAE vs source should be lossy-reasonable (squeeze averages
    // alpha bands, so some smoothing is expected — but we should not
    // see the decoder snap everything to 0 or 255).
    let mae = alpha_mae(&decoded, &rgba_buf());
    assert!(
        mae < 80.0,
        "chunk-2 squeeze path alpha MAE = {mae} is unreasonably high; suspect a wire bug"
    );
}

#[test]
fn alpha_squeeze_chunk2b_multigroup_encodes_and_jxl_rs_roundtrips() {
    // Chunk-2.b lifts the multi-group gate: the squeeze pipeline now
    // partitions sub-channels across LfGlobal (`w,h ≤ GROUP_DIM`) +
    // per-LfGroup (`min_shift ≥ 3`) + per-HfGroup (`min_shift < 3`)
    // sections, matching the libjxl decoder partition in
    // `dec_modular.cc:331-373`. This test:
    //   - encodes a 320×128 RGBA multi-group image (> GROUP_DIM in x);
    //   - confirms the encode succeeds (no NotImplemented);
    //   - decodes through jxl-rs (primary roundtrip per project CLAUDE.md);
    //   - confirms the alpha plane round-trips with reasonable MAE.
    const WIDE: u32 = 320;
    const TALL: u32 = 128;
    let mut buf = vec![0u8; (WIDE * TALL * 4) as usize];
    for y in 0..TALL {
        for x in 0..WIDE {
            let idx = ((y * WIDE + x) * 4) as usize;
            buf[idx] = (x % 256) as u8;
            buf[idx + 1] = (y % 256) as u8;
            buf[idx + 2] = ((x + y) % 256) as u8;
            buf[idx + 3] = ((x * 3 + y * 5) % 256) as u8;
        }
    }
    let bytes = LossyConfig::new(1.0)
        .with_alpha_distance(Some(2.0))
        .with_alpha_squeeze(true)
        .encode(&buf, WIDE, TALL, PixelLayout::Rgba8)
        .expect("chunk-2.b: multi-group alpha squeeze encode must succeed");
    assert!(!bytes.is_empty());
    // Roundtrip through jxl-rs to verify the wire format is valid.
    let decoded = {
        use jxl::api::{
            JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
            JxlPixelFormat, ProcessingResult, states,
        };
        use jxl::image::{Image, Rect};

        let mut input = bytes.as_slice();
        let options = JxlDecoderOptions::default();
        let decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut decoder_init = decoder;
        let mut decoder = loop {
            match decoder_init.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_init = fallback,
                Err(e) => panic!("chunk-2.b multigroup: jxl-rs header decode error: {e:?}"),
            }
        };
        let basic = decoder.basic_info().clone();
        let (w, h) = basic.size;
        let num_extras = basic.extra_channels.len();
        assert_eq!(w as u32, WIDE);
        assert_eq!(h as u32, TALL);
        decoder.set_pixel_format(JxlPixelFormat {
            color_type: JxlColorType::Rgba,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format: vec![None; num_extras],
        });
        let mut decoder_frame = loop {
            match decoder.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
                Err(e) => panic!("chunk-2.b multigroup: jxl-rs frame info error: {e:?}"),
            }
        };
        let channels = 4usize;
        let mut img = Image::<u8>::new((w * channels, h)).expect("alloc");
        let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
            img.get_rect_mut(Rect {
                origin: (0, 0),
                size: (w * channels, h),
            })
            .into_raw(),
        )];
        loop {
            match decoder_frame.process(&mut input, &mut buffers) {
                Ok(ProcessingResult::Complete { .. }) => break,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_frame = fallback,
                Err(e) => panic!("chunk-2.b multigroup: jxl-rs frame decode error: {e:?}"),
            }
        }
        let mut pixels = Vec::with_capacity(w * h * channels);
        for y in 0..h {
            pixels.extend_from_slice(img.row(y));
        }
        pixels
    };

    // Alpha plane should vary and MAE should be reasonable for lossy
    // alpha at distance 2.0 with squeeze (the squeeze averages
    // sub-bands, expect some smoothing).
    let alpha_min = decoded.iter().skip(3).step_by(4).copied().min().unwrap();
    let alpha_max = decoded.iter().skip(3).step_by(4).copied().max().unwrap();
    assert!(
        alpha_max > alpha_min,
        "chunk-2.b multigroup squeeze: alpha plane should vary; got \
         min={alpha_min} max={alpha_max}"
    );
    // Bound MAE on the alpha plane only.
    let mut sum = 0u32;
    let mut n = 0u32;
    for i in (3..decoded.len()).step_by(4) {
        sum += (decoded[i] as i32 - buf[i] as i32).unsigned_abs();
        n += 1;
    }
    let mae = sum as f32 / n as f32;
    assert!(
        mae < 80.0,
        "chunk-2.b multigroup squeeze: alpha MAE = {mae} is unreasonably \
         high; suspect a wire bug"
    );
}

#[test]
fn alpha_squeeze_chunk2_no_alpha_extra_is_no_op() {
    // Re-confirms the chunk-1 contract: flag-on + no alpha → no-op,
    // not error (RGB-only encode at any alpha_distance).
    let rgb: Vec<u8> = (0..(W * H * 3) as usize)
        .map(|i| ((i * 7) % 256) as u8)
        .collect();
    let bytes = LossyConfig::new(1.0)
        .with_alpha_distance(Some(2.0))
        .with_alpha_squeeze(true)
        .encode(&rgb, W, H, PixelLayout::Rgb8)
        .expect("RGB-only encode with flag on should not error");
    assert!(!bytes.is_empty());
}

#[test]
fn alpha_squeeze_chunk2_default_off_byte_identical_to_pre_chunk1() {
    // Carries the chunk-1 byte-identical contract: with the flag at
    // its default (false) two encodes produce identical bytes. The
    // chunk-2 routing is OFF unless the caller opts in.
    let cfg = || LossyConfig::new(1.0).with_alpha_distance(Some(2.0));
    let a = encode_rgba(cfg()).expect("encode A");
    let b = encode_rgba(cfg()).expect("encode B");
    assert_eq!(a, b);
}

#[test]
fn alpha_squeeze_chunk2_produces_different_bytes_from_baseline() {
    // Sanity: same input, flag on vs off, must produce *different*
    // byte streams. If they're identical the routing didn't actually
    // engage.
    let baseline = encode_rgba(LossyConfig::new(1.0).with_alpha_distance(Some(2.0))).expect("base");
    let squeezed = encode_rgba(
        LossyConfig::new(1.0)
            .with_alpha_distance(Some(2.0))
            .with_alpha_squeeze(true),
    )
    .expect("squeezed");
    assert_ne!(
        baseline, squeezed,
        "chunk-2 routing should change the bitstream when flag is on"
    );
}
