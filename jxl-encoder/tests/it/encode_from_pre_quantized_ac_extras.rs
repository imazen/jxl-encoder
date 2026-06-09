// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Regression test for [`VarDctEncoder::encode_from_pre_quantized_ac_with_extras`].
//!
//! Mirrors [`encode_from_precomputed_extras.rs`] (the FU4 wrapper test)
//! but exercises the deeper GPU fast path. Where
//! `encode_from_precomputed` runs `transform_and_quantize` internally,
//! `encode_from_pre_quantized_ac` accepts pre-DCT'd / pre-quantized AC
//! coefficients (the GPU encoder uploads small per-block coefficient
//! buffers instead of re-running DCT on the CPU). Until this commit
//! that path passed `&[]` for extras into `encode_two_pass` — any
//! caller that wired alpha / depth / spot color / selection mask /
//! thermal / CFA channels through the precomputed entry would have
//! them silently dropped on the way out.
//!
//! The new entry point [`VarDctEncoder::encode_from_pre_quantized_ac_with_extras`]
//! validates extras (dim_shift = 0, sample-count = width * height) and
//! threads them through the final `encode_two_pass` call.
//!
//! ## What this test asserts
//!
//! Two encodes of the same 32×32 RGBA image at distance 1.0:
//!
//! * **A — extras-bearing**: builds an [`EncoderPrecomputed`] from the
//!   linear-RGB plane, runs CPU `transform_and_quantize_for_test` to
//!   produce the per-block AC structures (matching what the GPU
//!   pipeline would have uploaded), then calls
//!   [`VarDctEncoder::encode_from_pre_quantized_ac_with_extras`] with
//!   a single alpha extra channel. Decoded with jxl-rs as RGBA8 — the
//!   alpha plane MUST exactly match the input alpha (lossless: VarDCT
//!   extras are encoded as a modular sub-bitstream alongside the lossy
//!   color, with no quantization).
//!
//! * **B — color-only baseline**: same precomputed input + same AC
//!   structures, but called via the legacy
//!   [`VarDctEncoder::encode_from_pre_quantized_ac`] (no extras). The
//!   bitstream is RGB-only (no alpha channel in the file header).
//!   Decoded with jxl-rs as RGBA — the file header MUST carry zero
//!   extra channels and jxl-rs MUST synthesize opaque alpha.
//!
//! The bitstreams MUST differ (A carries an alpha sub-bitstream and an
//! extra-channel entry in the file header; B does not). Both MUST
//! decode round-trip via jxl-rs.

#![cfg(all(feature = "__pre_quantized", feature = "rate-control"))]

use jxl_encoder::__pre_quantized::DistanceParams;
use jxl_encoder::api::ExtraChannel;
use jxl_encoder::vardct::{EncoderPrecomputed, VarDctEncoder};

const W: usize = 32;
const H: usize = 32;

/// Build a 32x32 linear-RGB plane with multi-frequency content so the
/// VarDCT path actually exercises the cost model (no degenerate flat
/// blocks). Returns interleaved `f32` of length `W * H * 3`.
fn rgb_plane() -> Vec<f32> {
    let mut out = vec![0.0f32; W * H * 3];
    for y in 0..H {
        for x in 0..W {
            let fx = x as f32 / W as f32;
            let fy = y as f32 / H as f32;
            let idx = (y * W + x) * 3;
            out[idx] = (0.5 + 0.4 * (fx * 11.0).sin() * (fy * 9.0).cos()).clamp(0.05, 0.95);
            out[idx + 1] = (0.4 + 0.3 * (fx * 7.0).cos()).clamp(0.05, 0.95);
            out[idx + 2] = (0.5 + 0.4 * (fy * 13.0).sin()).clamp(0.05, 0.95);
        }
    }
    out
}

/// Build a 32x32 alpha plane with a non-trivial pattern (corners
/// transparent, body opaque, gradient edges) so a regression that
/// drops alpha and falls back to "fill 1.0" is caught immediately.
fn alpha_plane() -> Vec<u8> {
    let mut a = vec![0u8; W * H];
    for y in 0..H {
        for x in 0..W {
            // Diamond-shape mask + per-pixel modulation so no two
            // alpha values are identical (modulo the dominant value)
            // — a "fill with 0xFF" regression cannot pass this byte
            // comparison.
            let cx = W as i32 / 2;
            let cy = H as i32 / 2;
            let dx = (x as i32 - cx).abs();
            let dy = (y as i32 - cy).abs();
            let dist = dx + dy;
            let radius = (W.min(H) / 2) as i32;
            let base = if dist > radius {
                0
            } else {
                ((radius - dist).clamp(0, 255) as u8).saturating_mul(8)
            };
            // Modulate the LSB by a cheap pattern so the test detects
            // a hypothetical regression that fills with a constant.
            let modulation = ((x ^ y) & 0x07) as u8;
            a[y * W + x] = base.saturating_add(modulation);
        }
    }
    a
}

/// Decode a JXL bitstream as RGBA8 with jxl-rs. Returns
/// `(width, height, num_extras, rgba_bytes)`. When the file has no
/// alpha channel, jxl-rs synthesizes opaque alpha in the RGBA output;
/// the caller checks `num_extras` to distinguish "alpha was encoded"
/// from "synthesized opaque alpha".
fn decode_jxl_rs_rgba8(data: &[u8]) -> (u32, u32, usize, Vec<u8>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);

    // Process header
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

    // Frame info
    let mut decoder_frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame info error: {e:?}"),
        }
    };

    let channels = 4;
    let mut output_image =
        Image::<u8>::new((width * channels, height)).expect("alloc rgba8 buffer");
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
    (width as u32, height as u32, num_extras, pixels)
}

/// Build a precomputed shared by both encode paths so the alpha
/// channel is the only difference between bitstreams A and B.
fn build_precomputed(encoder: &VarDctEncoder, linear_rgb: &[f32]) -> EncoderPrecomputed {
    EncoderPrecomputed::compute(
        W,
        H,
        linear_rgb,
        encoder.distance,
        encoder.cfl_enabled,
        encoder.ac_strategy_enabled,
        encoder.pixel_domain_loss,
        encoder.enable_noise,
        encoder.enable_denoise,
        encoder.enable_gaborish,
        encoder.force_strategy,
        &encoder.profile,
        encoder.color_encoding.as_ref(),
    )
    .expect("EncoderPrecomputed::compute must succeed on 32x32 input")
}

#[test]
fn encode_from_pre_quantized_ac_with_extras_preserves_alpha_through_jxlrs_roundtrip() {
    let linear_rgb = rgb_plane();
    let alpha = alpha_plane();

    let encoder = VarDctEncoder::new(1.0);
    let precomputed = build_precomputed(&encoder, &linear_rgb);

    // Per `encode_from_pre_quantized_ac` docs: caller passes the
    // *un-adjusted* per-block u8 quant field. The entry point applies
    // `adjust_quant_field_with_distance` internally. To match that
    // contract on the AC structures, hand `transform_and_quantize` a
    // mutable copy that it can adjust — but feed the *original*
    // un-adjusted field to the entry point. This is the same pattern
    // jxl-encoder-gpu's `encode_from_pre_quantized_ac_self_check` uses.
    let n_blocks = precomputed.xsize_blocks * precomputed.ysize_blocks;
    let quant_field_unadjusted = vec![1u8; n_blocks];

    let params = DistanceParams::compute_for_profile(encoder.distance, &encoder.profile);
    let mut quant_field_for_xform = quant_field_unadjusted.clone();
    let to = encoder
        .transform_and_quantize_for_test(&precomputed, &mut quant_field_for_xform, &params)
        .expect("transform_and_quantize_for_test must succeed");

    // Path A: extras-bearing pre-quantized AC encode (the new entry).
    let alpha_extra = ExtraChannel::from_alpha_buf(&alpha, /* associated = */ false);
    let bytes_a = encoder
        .encode_from_pre_quantized_ac_with_extras(
            &precomputed,
            &quant_field_unadjusted,
            &to.quant_dc,
            &to.quant_ac,
            &to.nzeros,
            &to.raw_nzeros,
            &[alpha_extra],
        )
        .expect("encode_from_pre_quantized_ac_with_extras must accept one alpha extra");

    // Path B: color-only baseline (the legacy entry).
    let bytes_b = encoder
        .encode_from_pre_quantized_ac(
            &precomputed,
            &quant_field_unadjusted,
            &to.quant_dc,
            &to.quant_ac,
            &to.nzeros,
            &to.raw_nzeros,
        )
        .expect("encode_from_pre_quantized_ac must succeed without extras");

    // Sanity: both bitstreams carry the JXL signature.
    assert_eq!(
        &bytes_a[..2],
        &[0xFF, 0x0A],
        "path A: missing JXL signature"
    );
    assert_eq!(
        &bytes_b[..2],
        &[0xFF, 0x0A],
        "path B: missing JXL signature"
    );

    // Bitstreams must differ — A carries an extra-channel entry in the
    // file header AND a modular sub-bitstream for the alpha plane that
    // B does not. If they're byte-identical, the new method is a no-op
    // (i.e. extras are still being silently dropped in the
    // pre-quantized AC path).
    assert_ne!(
        bytes_a, bytes_b,
        "encode_from_pre_quantized_ac_with_extras produced byte-identical output to the \
         color-only encode_from_pre_quantized_ac — extras are being silently dropped in the \
         pre-quantized AC path"
    );

    // Decode A: must report 1 extra channel and reproduce alpha
    // pixel-for-pixel (VarDCT extras are encoded losslessly as a
    // modular sub-bitstream).
    let (wa, ha, num_extras_a, rgba_a) = decode_jxl_rs_rgba8(&bytes_a);
    assert_eq!(wa, W as u32, "path A decoded width mismatch");
    assert_eq!(ha, H as u32, "path A decoded height mismatch");
    assert_eq!(
        num_extras_a, 1,
        "path A: file header should carry exactly one extra channel (alpha), got {num_extras_a}"
    );
    assert_eq!(
        rgba_a.len(),
        W * H * 4,
        "path A: decoded RGBA byte count mismatch"
    );
    let mut decoded_alpha = vec![0u8; W * H];
    for i in 0..(W * H) {
        decoded_alpha[i] = rgba_a[i * 4 + 3];
    }
    assert_eq!(
        decoded_alpha, alpha,
        "path A: decoded alpha plane does not match input — extras were not encoded losslessly"
    );

    // Decode B: must report 0 extra channels (the color-only path
    // can't have invented one), and the synthesized alpha jxl-rs
    // returns must be uniform 0xFF (opaque) — meaning the file
    // header genuinely had no alpha channel, not a corrupt one.
    let (wb, hb, num_extras_b, rgba_b) = decode_jxl_rs_rgba8(&bytes_b);
    assert_eq!(wb, W as u32, "path B decoded width mismatch");
    assert_eq!(hb, H as u32, "path B decoded height mismatch");
    assert_eq!(
        num_extras_b, 0,
        "path B: color-only encode must produce zero extra channels in the header, got {num_extras_b}"
    );
    for i in 0..(W * H) {
        assert_eq!(
            rgba_b[i * 4 + 3],
            0xFF,
            "path B: jxl-rs should synthesize opaque alpha when the file has none, but pixel {i} is {}",
            rgba_b[i * 4 + 3]
        );
    }
}
