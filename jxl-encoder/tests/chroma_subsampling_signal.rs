//! Public API surface tests for `ChromaSubsampling` (issue #47).
//!
//! Pin the chunk-3 contract:
//!
//! 1. Enum default is `Full444` — never silently changes the
//!    bitstream of existing callers.
//! 2. `with_chroma_subsampling()` round-trips on the config (getter
//!    returns what was set; preserved across `with_effort()`).
//! 3. Per-mode `h_shifts()` / `v_shifts()` match libjxl's
//!    `YCbCrChromaSubsampling::HShift(c)` / `VShift(c)` tables in
//!    `[Cb, Y, Cr]` channel order.
//! 4. `Full444` actually encodes (default path; jxl-rs roundtrip
//!    proves the bitstream is well-formed).
//! 5. Every non-`Full444` mode returns `EncodeError::InvalidConfig`
//!    with a message that names the format tag — both via one-shot
//!    `encode()` AND via the streaming `LossyEncoder::finish()` path.

use jxl_encoder::{ChromaSubsampling, EncodeError, LossyConfig, PixelLayout};

const W: usize = 64;
const H: usize = 64;

fn make_rgb_buffer() -> Vec<u8> {
    let mut out = vec![0u8; W * H * 3];
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) * 3;
            out[i] = ((x * 255) / W.max(1)) as u8;
            out[i + 1] = ((y * 255) / H.max(1)) as u8;
            out[i + 2] = (((x + y) * 255) / (W + H).max(1)) as u8;
        }
    }
    out
}

fn decode_via_jxl_rs(bytes: &[u8]) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = bytes;
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

    let channels = 3;
    let mut output_image = Image::<u8>::new((width * channels, height)).expect("alloc rgb buffer");
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

    assert_eq!(width as usize, W);
    assert_eq!(height as usize, H);
}

fn assert_invalid_config_with_tag(err: &EncodeError, tag: &str) {
    match err {
        EncodeError::InvalidConfig { message } => {
            assert!(
                message.contains(tag),
                "InvalidConfig message must name the tag {tag:?}, got: {message}",
            );
            assert!(
                message.contains("chroma subsampling"),
                "InvalidConfig message must call out 'chroma subsampling', got: {message}",
            );
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn default_is_full444() {
    let cfg = LossyConfig::new(1.0);
    assert_eq!(cfg.chroma_subsampling(), ChromaSubsampling::Full444);
    assert!(cfg.chroma_subsampling().is_full());
}

#[test]
fn enum_default_is_full444() {
    assert_eq!(
        ChromaSubsampling::default(),
        ChromaSubsampling::Full444,
        "ChromaSubsampling::default must be Full444 — changing this would \
         alter every existing LossyConfig bitstream produced without an \
         explicit with_chroma_subsampling() call.",
    );
}

#[test]
fn h_v_shifts_match_libjxl_table() {
    // Channel order is [Cb, Y, Cr] (matches FrameHeader.jpeg_upsampling
    // layout in libjxl's frame_header.h).
    assert_eq!(ChromaSubsampling::Full444.h_shifts(), [0, 0, 0]);
    assert_eq!(ChromaSubsampling::Full444.v_shifts(), [0, 0, 0]);

    assert_eq!(ChromaSubsampling::Sub422.h_shifts(), [1, 0, 1]);
    assert_eq!(ChromaSubsampling::Sub422.v_shifts(), [0, 0, 0]);

    assert_eq!(ChromaSubsampling::Sub420.h_shifts(), [1, 0, 1]);
    assert_eq!(ChromaSubsampling::Sub420.v_shifts(), [1, 0, 1]);

    assert_eq!(ChromaSubsampling::Sub440.h_shifts(), [0, 0, 0]);
    assert_eq!(ChromaSubsampling::Sub440.v_shifts(), [1, 0, 1]);

    // Y (index 1) is never subsampled in any mode.
    for mode in [
        ChromaSubsampling::Full444,
        ChromaSubsampling::Sub422,
        ChromaSubsampling::Sub420,
        ChromaSubsampling::Sub440,
    ] {
        assert_eq!(mode.h_shifts()[1], 0, "Y h-shift must be 0 for {mode:?}");
        assert_eq!(mode.v_shifts()[1], 0, "Y v-shift must be 0 for {mode:?}");
    }
}

#[test]
fn is_full_only_for_full444() {
    assert!(ChromaSubsampling::Full444.is_full());
    assert!(!ChromaSubsampling::Sub422.is_full());
    assert!(!ChromaSubsampling::Sub420.is_full());
    assert!(!ChromaSubsampling::Sub440.is_full());
}

#[test]
fn tag_strings_match_industry_convention() {
    assert_eq!(ChromaSubsampling::Full444.tag(), "4:4:4");
    assert_eq!(ChromaSubsampling::Sub422.tag(), "4:2:2");
    assert_eq!(ChromaSubsampling::Sub420.tag(), "4:2:0");
    assert_eq!(ChromaSubsampling::Sub440.tag(), "4:4:0");
}

#[test]
fn with_chroma_subsampling_round_trips_on_config() {
    for mode in [
        ChromaSubsampling::Full444,
        ChromaSubsampling::Sub422,
        ChromaSubsampling::Sub420,
        ChromaSubsampling::Sub440,
    ] {
        let cfg = LossyConfig::new(1.0).with_chroma_subsampling(mode);
        assert_eq!(cfg.chroma_subsampling(), mode);
    }
}

/// Regression: `with_effort()` rebuilds the [`LossyConfig`] via
/// `Self::new_with_effort()`, which resets every field that isn't
/// explicitly preserved. Without the carry-over in `with_effort`, the
/// builder chain
/// `LossyConfig::new(d).with_chroma_subsampling(Sub420).with_effort(5)`
/// would silently revert to `Full444`, hiding the encoder's chunk-3
/// `InvalidConfig` guard. Pin the order independence so a future
/// refactor of `with_effort` can't lose the field again.
#[test]
fn with_chroma_subsampling_survives_with_effort() {
    let cfg = LossyConfig::new(1.0)
        .with_chroma_subsampling(ChromaSubsampling::Sub420)
        .with_effort(5);
    assert_eq!(
        cfg.chroma_subsampling(),
        ChromaSubsampling::Sub420,
        "with_effort() must preserve chroma_subsampling (see api.rs with_effort()).",
    );

    // Reverse order — also must end up Sub420.
    let cfg = LossyConfig::new(1.0)
        .with_effort(5)
        .with_chroma_subsampling(ChromaSubsampling::Sub420);
    assert_eq!(cfg.chroma_subsampling(), ChromaSubsampling::Sub420);
}

#[test]
fn full444_encodes_and_roundtrips_via_jxl_rs() {
    let rgb = make_rgb_buffer();
    let bytes = LossyConfig::new(1.0)
        .with_chroma_subsampling(ChromaSubsampling::Full444)
        .with_effort(5)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect("Full444 (default) must encode 64x64 RGB at d=1.0 e=5");

    assert!(bytes.len() > 32, "bitstream suspiciously small");
    assert_eq!(&bytes[..2], &[0xFF, 0x0A], "missing JXL signature");

    // jxl-rs roundtrip — proves the bitstream is well-formed.
    decode_via_jxl_rs(&bytes);
}

#[test]
fn sub420_returns_invalid_config_in_chunk3() {
    let rgb = make_rgb_buffer();
    let err = LossyConfig::new(1.0)
        .with_chroma_subsampling(ChromaSubsampling::Sub420)
        .with_effort(5)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect_err("Sub420 must return InvalidConfig in chunk 3 (signal-only on lossy path)");
    assert_invalid_config_with_tag(err.error(), "4:2:0");
}

#[test]
fn sub422_returns_invalid_config_in_chunk3() {
    let rgb = make_rgb_buffer();
    let err = LossyConfig::new(1.0)
        .with_chroma_subsampling(ChromaSubsampling::Sub422)
        .with_effort(5)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect_err("Sub422 must return InvalidConfig in chunk 3 (signal-only on lossy path)");
    assert_invalid_config_with_tag(err.error(), "4:2:2");
}

#[test]
fn sub440_returns_invalid_config_in_chunk3() {
    let rgb = make_rgb_buffer();
    let err = LossyConfig::new(1.0)
        .with_chroma_subsampling(ChromaSubsampling::Sub440)
        .with_effort(5)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect_err("Sub440 must return InvalidConfig in chunk 3 (signal-only on lossy path)");
    assert_invalid_config_with_tag(err.error(), "4:4:0");
}

/// Streaming `LossyEncoder::finish` must apply the same gate as
/// one-shot `EncodeRequest::encode`. Without this mirror, a caller
/// who uses the streaming API would silently bypass the chunk-3
/// check and hit a much-less-helpful error inside the VarDCT
/// pipeline.
#[test]
fn streaming_sub420_returns_invalid_config_in_chunk3() {
    let cfg = LossyConfig::new(1.0).with_chroma_subsampling(ChromaSubsampling::Sub420);
    let mut enc = cfg
        .encoder(W as u32, H as u32, PixelLayout::Rgb8)
        .expect("streaming encoder build must succeed");
    let rgb = make_rgb_buffer();
    for y in 0..H {
        let row_start = y * W * 3;
        let row_end = row_start + W * 3;
        enc.push_rows(&rgb[row_start..row_end], 1)
            .expect("push row");
    }
    let err = enc.finish().expect_err(
        "streaming Sub420 must return InvalidConfig in chunk 3 (signal-only on lossy path)",
    );
    assert_invalid_config_with_tag(err.error(), "4:2:0");
}
