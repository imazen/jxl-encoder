//! Sectioned-tree lossless mode (imazen/jxl-encoder#96,
//! `JXL_LOSSLESS_LOCAL_TREES=1`): every PassGroup stream carries its own MA
//! tree (`use_global_tree = false`) learned from that group's samples only.
//! This test proves, in one process:
//!
//! 1. The mode ENGAGES on a multi-group image (bitstream differs from the
//!    global-tree encode).
//! 2. Both bitstreams decode PIXEL-EXACT via zenjxl-decoder.
//! 3. Clearing the env restores byte-identical global-tree output (the
//!    default path is untouched).
//!
//! Lives in the env-overrides binary because it mutates process env
//! (`env_serial` serializes with every other test here).

use jxl_encoder::{LosslessConfig, PixelLayout};

fn prng_rgb(w: usize, h: usize) -> Vec<u8> {
    // Small-period procedural content with real structure (gradients +
    // pseudo-noise) so trees are non-trivial in every group.
    let mut out = Vec::with_capacity(w * h * 3);
    let mut state: u32 = 0x1234_5678;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = (state >> 24) as u8;
            out.push(((x * 255 / w) as u8).wrapping_add(n & 0x1f));
            out.push(((y * 255 / h) as u8) ^ (n & 0x0f));
            out.push((((x + y) * 128 / (w + h)) as u8).wrapping_add(n >> 3));
        }
    }
    out
}

fn encode(pixels: &[u8], w: u32, h: u32) -> Vec<u8> {
    LosslessConfig::new()
        .with_effort(7)
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .expect("lossless encode")
}

/// Decode and return tightly-packed RGB8 (the decoder emits RGBA).
fn decode_pixels(bytes: &[u8], w: usize, h: usize) -> Vec<u8> {
    let decoded = zenjxl_decoder::decode(bytes).expect("zenjxl-decoder decode");
    assert_eq!((decoded.width, decoded.height), (w, h), "decoded dims");
    assert_eq!(decoded.channels, 4, "expected RGBA output");
    decoded
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect()
}

#[test]
fn local_trees_mode_roundtrips_and_default_is_untouched() {
    let _guard = crate::env_serial();
    // Multi-group: 512x512 = 2x2 groups at the default 256 group size.
    let (w, h) = (512usize, 512usize);
    let pixels = prng_rgb(w, h);

    // Global arms are PINNED with "0" (not merely unset): since the
    // 2026-08-19 Auto policy, an unset env under a parallel multi-thread
    // build resolves Auto -> sectioned at e7, which would collapse the
    // global-vs-local comparison below.
    // SAFETY: env mutation is serialized by `env_serial` (this binary's
    // process-wide mutex); no other thread reads env concurrently.
    unsafe { std::env::set_var("JXL_LOSSLESS_LOCAL_TREES", "0") };
    let global_a = encode(&pixels, w as u32, h as u32);

    // SAFETY: as above — serialized by `env_serial`.
    unsafe { std::env::set_var("JXL_LOSSLESS_LOCAL_TREES", "1") };
    let local = encode(&pixels, w as u32, h as u32);
    // SAFETY: as above — serialized by `env_serial`.
    unsafe { std::env::set_var("JXL_LOSSLESS_LOCAL_TREES", "0") };
    let global_b = encode(&pixels, w as u32, h as u32);
    // SAFETY: as above — serialized by `env_serial`.
    unsafe { std::env::remove_var("JXL_LOSSLESS_LOCAL_TREES") };

    assert_eq!(
        global_a, global_b,
        "pinned-global path must be byte-identical before/after toggling the mode"
    );
    assert_ne!(
        global_a, local,
        "the mode must actually engage (bitstreams should differ)"
    );

    let dec_global = decode_pixels(&global_a, w, h);
    let dec_local = decode_pixels(&local, w, h);
    assert_eq!(dec_global, pixels, "global-tree roundtrip must be exact");
    assert_eq!(dec_local, pixels, "local-tree roundtrip must be exact");

    // jxl-rs (the PRIMARY roundtrip decoder per project policy) must also
    // reconstruct both bitstreams pixel-exactly.
    let rs_global = decode_jxl_rs_rgb8(&global_a, w, h);
    let rs_local = decode_jxl_rs_rgb8(&local, w, h);
    assert_eq!(
        rs_global, pixels,
        "global-tree jxl-rs roundtrip must be exact"
    );
    assert_eq!(
        rs_local, pixels,
        "local-tree jxl-rs roundtrip must be exact"
    );
}

/// Decode via jxl-rs and return tightly-packed RGB8 (pattern per
/// tests/it/alpha_squeeze_chunk1_framework.rs).
fn decode_jxl_rs_rgb8(data: &[u8], w: usize, h: usize) -> Vec<u8> {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let mut decoder_init = JxlDecoder::<states::Initialized>::new(options);
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
    assert_eq!((width, height), (w, h), "jxl-rs dims");
    let num_extras = basic_info.extra_channels.len();

    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
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

    let channels = 3usize;
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

    let mut out = Vec::with_capacity(w * h * channels);
    for y in 0..h {
        out.extend_from_slice(&output_image.row(y)[..w * channels]);
    }
    out
}
