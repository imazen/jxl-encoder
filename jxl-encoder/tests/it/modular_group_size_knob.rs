// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! `LosslessConfig::with_modular_group_size` — `cjxl -g 0..3` knob.
//!
//! Mirrors libjxl `enc_params.h:modular_group_size_shift` /
//! `enc_frame.cc:336-358` (when `cparams.modular_group_size_shift == -1`
//! the heuristic picks a shift based on image dimensions; when set, it
//! is forwarded verbatim into `frame_header.group_size_shift`).
//!
//! At 600×600, the four legal shifts (0..=3) map to group dimensions
//! {128, 256, 512, 1024} and produce {5×5, 3×3, 2×2, 1×1} group grids
//! respectively, so the partitioning + TOC layout differ on every step.
//! Each setting must:
//!   1. produce a distinct bitstream (`-g 0` ≠ `-g 1` ≠ `-g 2` ≠ `-g 3`);
//!   2. round-trip through jxl-rs at pixel-exact precision (lossless);
//!   3. when the knob is `None`, the bytes must match the historical
//!      `shift = 1` (256-px) output so existing hash-locks stay green.

use jxl_encoder::{LosslessConfig, PixelLayout};

/// Minimal jxl-rs decode wrapper. Returns (width, height, rgb8 pixels).
/// Lossless decode → expect pixel-exact match with the encoder input.
fn decode_rgb8(data: &[u8]) -> (usize, usize, Vec<u8>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input during header");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs header decode error: {:?}", e),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let channels = 3;

    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![],
    });

    let mut decoder_frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input before frame");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame info error: {:?}", e),
        }
    };

    let mut output_image =
        Image::<u8>::new((width * channels, height)).expect("alloc output buffer");
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
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input during decode");
                }
                decoder_frame = fallback;
            }
            Err(e) => panic!("jxl-rs decode error: {:?}", e),
        }
    }

    let mut packed = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        packed.extend_from_slice(output_image.row(y));
    }

    (width, height, packed)
}

/// 600×600 mixed gradient. Big enough that the group count differs at
/// every legal `-g` setting (5×5 / 3×3 / 2×2 / 1×1 at shift 0/1/2/3),
/// small enough to stay quick.
fn mixed_gradient_600() -> Vec<u8> {
    let w: usize = 600;
    let h: usize = 600;
    let mut buf = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            // Mix of low- and high-frequency content so RCT / tree learning
            // see meaningful per-group statistics.
            let r = ((x * 3 + y) & 0xff) as u8;
            let g = ((x ^ (y * 5)) & 0xff) as u8;
            let b = ((x + y * 2) & 0xff) as u8;
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
    }
    buf
}

fn encode_with_g(data: &[u8], w: u32, h: u32, shift: Option<u8>) -> Vec<u8> {
    LosslessConfig::default()
        .with_effort(5)
        .with_modular_group_size(shift)
        .encode(data, w, h, PixelLayout::Rgb8)
        .unwrap_or_else(|e| panic!("encode failed at -g={shift:?}: {e}"))
}

#[test]
fn modular_group_size_none_matches_shift_1() {
    // `None` (default) must produce byte-identical output to `Some(1)`
    // (the historical 256-pixel default). This is the hash-lock invariant
    // for callers that never touch the knob.
    let data = mixed_gradient_600();
    let auto = encode_with_g(&data, 600, 600, None);
    let g1 = encode_with_g(&data, 600, 600, Some(1));
    assert_eq!(
        auto, g1,
        "default LosslessConfig must match `--modular_group_size 1` byte-for-byte \
         (current 256-px group default; otherwise existing hash-locks regress)"
    );
}

#[test]
fn modular_group_size_each_shift_changes_bytes() {
    let data = mixed_gradient_600();
    let bytes: Vec<(u8, Vec<u8>)> = (0u8..=3)
        .map(|s| (s, encode_with_g(&data, 600, 600, Some(s))))
        .collect();

    // Pairwise: all four bitstreams must differ (different group_dim
    // changes both the frame-header `group_size_shift` field and the
    // per-group entropy / TOC layout).
    for i in 0..bytes.len() {
        for j in (i + 1)..bytes.len() {
            let (si, bi) = &bytes[i];
            let (sj, bj) = &bytes[j];
            assert_ne!(
                bi, bj,
                "shifts {si} and {sj} produced byte-identical streams; the knob \
                 is not actually reaching the modular partitioner / frame header"
            );
        }
    }

    // Sanity: signatures + non-empty
    for (s, b) in &bytes {
        assert_eq!(&b[..2], &[0xFF, 0x0A], "bad JXL signature at -g={s}");
        assert!(b.len() > 32, "suspiciously small bitstream at -g={s}");
    }
}

#[test]
fn modular_group_size_roundtrips_each_shift() {
    // Lossless → every -g shift must decode pixel-exact through jxl-rs.
    let data = mixed_gradient_600();
    for shift in 0u8..=3 {
        let encoded = encode_with_g(&data, 600, 600, Some(shift));
        assert_eq!(&encoded[..2], &[0xFF, 0x0A], "bad signature at -g={shift}");
        let (dw, dh, decoded) = decode_rgb8(&encoded);
        assert_eq!((dw, dh), (600, 600), "decode dims mismatch at -g={shift}");
        assert_eq!(
            decoded.len(),
            data.len(),
            "decoded pixel count mismatch at -g={shift}"
        );
        assert_eq!(
            decoded, data,
            "lossless roundtrip mismatch at -g={shift}: decoded bytes != input"
        );
    }
}

#[test]
fn modular_group_size_g0_smaller_grid_at_600px() {
    // -g 0 (128px groups) on a 600×600 image creates a 5×5 = 25-group
    // grid; -g 3 (1024px groups) creates a single 1×1 group. The
    // single-group bitstream has no per-group TOC overhead and is
    // usually smaller on a 600px image where the per-group entropy
    // dominates. We don't pin the exact sign of the delta (it depends
    // on the content), but the two MUST differ.
    let data = mixed_gradient_600();
    let g0 = encode_with_g(&data, 600, 600, Some(0));
    let g3 = encode_with_g(&data, 600, 600, Some(3));
    assert_ne!(
        g0, g3,
        "-g 0 (128px, 5×5 groups) and -g 3 (1024px, single group) must produce \
         different bitstreams on a 600×600 image"
    );
}
