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

/// Chunk-3 contract retained: without both `chroma-subsampling`
/// AND `jpeg-reencoding` cargo features the gate still ships
/// InvalidConfig. Once both features are on (chunk 4), the same
/// call routes to the Sub420 encoder and produces a valid bitstream
/// (`sub420_encodes_and_roundtrips_via_jxl_rs` below).
#[cfg(not(all(feature = "chroma-subsampling", feature = "jpeg-reencoding")))]
#[test]
fn sub420_returns_invalid_config_without_chunk4_features() {
    let rgb = make_rgb_buffer();
    let err = LossyConfig::new(1.0)
        .with_chroma_subsampling(ChromaSubsampling::Sub420)
        .with_effort(5)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect_err(
            "Sub420 must return InvalidConfig without the chunk-4 feature pair (chroma-subsampling + jpeg-reencoding)",
        );
    assert_invalid_config_with_tag(err.error(), "4:2:0");
}

/// Pre-chunk-5 fallback: when the `chroma-subsampling` +
/// `jpeg-reencoding` feature pair is OFF, Sub422 still surfaces
/// `InvalidConfig` (the JPEG-shaped path is the only Sub422 wire-up
/// today). Symmetric with [`sub420_returns_invalid_config_without_chunk4_features`].
#[cfg(not(all(feature = "chroma-subsampling", feature = "jpeg-reencoding")))]
#[test]
fn sub422_returns_invalid_config_without_chunk5_features() {
    let rgb = make_rgb_buffer();
    let err = LossyConfig::new(1.0)
        .with_chroma_subsampling(ChromaSubsampling::Sub422)
        .with_effort(5)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect_err(
            "Sub422 must return InvalidConfig without the chunk-5 feature pair (chroma-subsampling + jpeg-reencoding)",
        );
    assert_invalid_config_with_tag(err.error(), "4:2:2");
}

#[cfg(not(all(feature = "chroma-subsampling", feature = "jpeg-reencoding")))]
#[test]
fn sub440_returns_invalid_config_without_chunk5_features() {
    let rgb = make_rgb_buffer();
    let err = LossyConfig::new(1.0)
        .with_chroma_subsampling(ChromaSubsampling::Sub440)
        .with_effort(5)
        .encode_request(W as u32, H as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect_err(
            "Sub440 must return InvalidConfig without the chunk-5 feature pair (chroma-subsampling + jpeg-reencoding)",
        );
    assert_invalid_config_with_tag(err.error(), "4:4:0");
}

/// Streaming `LossyEncoder::finish` must apply the same gate as
/// one-shot `EncodeRequest::encode`. Chunk 4 keeps streaming Sub420
/// rejected (the streaming path eagerly linearises sRGB → f32 in
/// `push_rows`, so the JPEG-shaped Sub420 pipeline — which expects
/// raw u8 sRGB — cannot consume the buffer without a round-trip).
/// Chunk 5 extends to streaming once the round-trip is wired.
#[test]
fn streaming_sub420_still_invalid_config_in_chunk4() {
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
    let err = enc
        .finish()
        .expect_err("streaming Sub420 must still return InvalidConfig in chunk 4");
    assert_invalid_config_with_tag(err.error(), "4:2:0");
}

/// Chunk-4 acceptance via djxl (libjxl reference decoder). Encodes a
/// 256×256 Sub420 image, writes it to disk, and shells out to djxl to
/// confirm the libjxl decoder accepts the bitstream. Skipped when
/// djxl is not on $PATH so the regular CI run stays self-contained.
#[cfg(all(feature = "chroma-subsampling", feature = "jpeg-reencoding"))]
#[test]
fn sub420_decodes_via_djxl_when_available() {
    use std::process::Command;
    // Honour the user's djxl override; fall back to the libjxl build
    // tree path we have locally; finally fall back to whatever's on
    // $PATH. If none of these exist, we skip via env var (caller
    // controls whether to fail).
    let djxl_path = std::env::var("DJXL").unwrap_or_else(|_| {
        let local = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";
        if std::path::Path::new(local).exists() {
            local.to_string()
        } else {
            "djxl".to_string()
        }
    });
    // Verify the binary actually exists / runs before we encode.
    let ok = Command::new(&djxl_path).arg("--version").output().is_ok();
    if !ok {
        eprintln!(
            "skipping sub420_decodes_via_djxl_when_available: djxl not available at {djxl_path}"
        );
        return;
    }

    const W4: usize = 256;
    const H4: usize = 256;
    let mut rgb = vec![0u8; W4 * H4 * 3];
    for y in 0..H4 {
        for x in 0..W4 {
            let i = (y * W4 + x) * 3;
            let checker = (((x / 16) ^ (y / 16)) & 1) as u8 * 255;
            rgb[i] = ((x * 255) / W4) as u8;
            rgb[i + 1] = ((y * 255) / H4) as u8;
            rgb[i + 2] = checker;
        }
    }
    let bytes = LossyConfig::new(1.0)
        .with_chroma_subsampling(ChromaSubsampling::Sub420)
        .with_effort(5)
        .encode_request(W4 as u32, H4 as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect("Sub420 must encode for djxl roundtrip");
    let dir = std::env::temp_dir();
    let jxl_path = dir.join("jxl_encoder_chunk4_sub420.jxl");
    let png_path = dir.join("jxl_encoder_chunk4_sub420.png");
    std::fs::write(&jxl_path, &bytes).expect("write tmp jxl");
    let out = Command::new(&djxl_path)
        .arg(&jxl_path)
        .arg(&png_path)
        .output()
        .expect("run djxl");
    assert!(
        out.status.success(),
        "djxl rejected our Sub420 bitstream: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        png_path.exists() && std::fs::metadata(&png_path).unwrap().len() > 100,
        "djxl did not produce a sensible PNG output"
    );
    let _ = std::fs::remove_file(&jxl_path);
    let _ = std::fs::remove_file(&png_path);
}

// ── Chunk 4 end-to-end Sub420 — gated on chroma-subsampling + jpeg-reencoding ──

/// Chunk 4 acceptance: 256×256 RGB at d=1.0 with Sub420 routes
/// through `vardct::chroma_subsampling::encode_rgb8_sub420_via_jpeg_path`
/// and produces a decoder-valid bitstream. jxl-rs roundtrip proves
/// the dimensions + chroma shape decode without errors.
///
/// Bigger than the 64×64 default `W` / `H` so the path exercises
/// multi-DC-group framing too (>256 px would force multi-group; we
/// stay at single-group 256 so the helper test stays fast in CI).
#[cfg(all(feature = "chroma-subsampling", feature = "jpeg-reencoding"))]
#[test]
fn sub420_encodes_and_roundtrips_via_jxl_rs() {
    const W4: usize = 256;
    const H4: usize = 256;
    let mut rgb = vec![0u8; W4 * H4 * 3];
    for y in 0..H4 {
        for x in 0..W4 {
            let i = (y * W4 + x) * 3;
            // High-contrast checker + smooth gradient — exercises both
            // edge regions (chroma subsampling pain point) and smooth
            // regions (DCT8 quantisation).
            let checker = (((x / 16) ^ (y / 16)) & 1) as u8 * 255;
            rgb[i] = ((x * 255) / W4) as u8;
            rgb[i + 1] = ((y * 255) / H4) as u8;
            rgb[i + 2] = checker;
        }
    }
    let bytes = LossyConfig::new(1.0)
        .with_chroma_subsampling(ChromaSubsampling::Sub420)
        .with_effort(5)
        .encode_request(W4 as u32, H4 as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .expect("Sub420 must encode 256x256 RGB at d=1.0 (chunk 4)");

    assert_eq!(&bytes[..2], &[0xFF, 0x0A], "missing JXL signature");
    assert!(bytes.len() > 32, "bitstream suspiciously small");

    // jxl-rs roundtrip: parse header + decode pixels. Exercises the
    // full do_ycbcr=true + jpeg_upsampling=[0,1,0] decode path and
    // proves the per-channel block grids + RAW quant tables we
    // emitted match the decoder's expectations. The decode_via_jxl_rs
    // helper asserts that the decoded dimensions equal W/H — for the
    // chunk-4 test we re-implement inline against W4/H4.
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
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_init = fallback;
            }
            Err(e) => panic!("jxl-rs Sub420 header decode error: {e:?}"),
        }
    };
    let basic_info = decoder.basic_info().clone();
    let (dec_w, dec_h) = basic_info.size;
    assert_eq!(dec_w, W4, "decoded width mismatch");
    assert_eq!(dec_h, H4, "decoded height mismatch");
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
            Err(e) => panic!("jxl-rs Sub420 frame info error: {e:?}"),
        }
    };
    let channels = 3;
    let mut output_image = Image::<u8>::new((dec_w * channels, dec_h)).expect("alloc rgb buffer");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (dec_w * channels, dec_h),
            })
            .into_raw(),
    )];
    loop {
        match decoder_frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_frame = fallback;
            }
            Err(e) => panic!("jxl-rs Sub420 frame decode error: {e:?}"),
        }
    }
}

// ── Chunk 5: Sub422 / Sub440 end-to-end ─────────────────────────────────────

/// Encode 256×256 RGB via the requested non-`Full444` mode, then
/// roundtrip via jxl-rs. Returns the decoded basic-info dims so the
/// caller can pin the output shape matches input.
#[cfg(all(feature = "chroma-subsampling", feature = "jpeg-reencoding"))]
fn encode_and_roundtrip_via_jxl_rs(mode: ChromaSubsampling, w: usize, h: usize) -> (usize, usize) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let checker = (((x / 16) ^ (y / 16)) & 1) as u8 * 255;
            rgb[i] = ((x * 255) / w) as u8;
            rgb[i + 1] = ((y * 255) / h) as u8;
            rgb[i + 2] = checker;
        }
    }
    let bytes = LossyConfig::new(1.0)
        .with_chroma_subsampling(mode)
        .with_effort(5)
        .encode_request(w as u32, h as u32, PixelLayout::Rgb8)
        .encode(&rgb)
        .unwrap_or_else(|e| panic!("{:?} must encode {w}x{h} RGB at d=1.0: {e:?}", mode));

    assert_eq!(&bytes[..2], &[0xFF, 0x0A], "missing JXL signature");
    assert!(bytes.len() > 32, "bitstream suspiciously small");

    let mut input = bytes.as_slice();
    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut decoder_init = decoder;
    let mut decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_init = fallback;
            }
            Err(e) => panic!("jxl-rs {:?} header decode error: {e:?}", mode),
        }
    };
    let basic_info = decoder.basic_info().clone();
    let (dec_w, dec_h) = basic_info.size;
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
            Err(e) => panic!("jxl-rs {:?} frame info error: {e:?}", mode),
        }
    };
    let channels = 3;
    let mut output_image = Image::<u8>::new((dec_w * channels, dec_h)).expect("alloc rgb buffer");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (dec_w * channels, dec_h),
            })
            .into_raw(),
    )];
    loop {
        match decoder_frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_frame = fallback;
            }
            Err(e) => panic!("jxl-rs {:?} frame decode error: {e:?}", mode),
        }
    }
    (dec_w, dec_h)
}

#[cfg(all(feature = "chroma-subsampling", feature = "jpeg-reencoding"))]
#[test]
fn sub422_encodes_and_roundtrips_via_jxl_rs() {
    let (w, h) = encode_and_roundtrip_via_jxl_rs(ChromaSubsampling::Sub422, 256, 256);
    assert_eq!(w, 256);
    assert_eq!(h, 256);
}

#[cfg(all(feature = "chroma-subsampling", feature = "jpeg-reencoding"))]
#[test]
fn sub440_encodes_and_roundtrips_via_jxl_rs() {
    let (w, h) = encode_and_roundtrip_via_jxl_rs(ChromaSubsampling::Sub440, 256, 256);
    assert_eq!(w, 256);
    assert_eq!(h, 256);
}

/// djxl acceptance for Sub422 / Sub440. Skipped when djxl is not on
/// $PATH. Mirrors `sub420_decodes_via_djxl_when_available`.
#[cfg(all(feature = "chroma-subsampling", feature = "jpeg-reencoding"))]
#[test]
fn sub422_and_sub440_decode_via_djxl_when_available() {
    use std::process::Command;
    let djxl_path = std::env::var("DJXL").unwrap_or_else(|_| {
        let local = "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl";
        if std::path::Path::new(local).exists() {
            local.to_string()
        } else {
            "djxl".to_string()
        }
    });
    let ok = Command::new(&djxl_path).arg("--version").output().is_ok();
    if !ok {
        eprintln!(
            "skipping sub422_and_sub440_decode_via_djxl_when_available: djxl not available at {djxl_path}"
        );
        return;
    }

    const W4: usize = 256;
    const H4: usize = 256;
    let mut rgb = vec![0u8; W4 * H4 * 3];
    for y in 0..H4 {
        for x in 0..W4 {
            let i = (y * W4 + x) * 3;
            let checker = (((x / 16) ^ (y / 16)) & 1) as u8 * 255;
            rgb[i] = ((x * 255) / W4) as u8;
            rgb[i + 1] = ((y * 255) / H4) as u8;
            rgb[i + 2] = checker;
        }
    }
    for (tag, mode) in [
        ("sub422", ChromaSubsampling::Sub422),
        ("sub440", ChromaSubsampling::Sub440),
    ] {
        let bytes = LossyConfig::new(1.0)
            .with_chroma_subsampling(mode)
            .with_effort(5)
            .encode_request(W4 as u32, H4 as u32, PixelLayout::Rgb8)
            .encode(&rgb)
            .unwrap_or_else(|e| panic!("{mode:?} encode failed: {e:?}"));
        let dir = std::env::temp_dir();
        let jxl_path = dir.join(format!("jxl_encoder_chunk5_{tag}.jxl"));
        let png_path = dir.join(format!("jxl_encoder_chunk5_{tag}.png"));
        std::fs::write(&jxl_path, &bytes).expect("write tmp jxl");
        let out = Command::new(&djxl_path)
            .arg(&jxl_path)
            .arg(&png_path)
            .output()
            .expect("run djxl");
        assert!(
            out.status.success(),
            "djxl rejected {tag} bitstream: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            png_path.exists() && std::fs::metadata(&png_path).unwrap().len() > 100,
            "djxl did not produce a sensible PNG output for {tag}"
        );
        let _ = std::fs::remove_file(&jxl_path);
        let _ = std::fs::remove_file(&png_path);
    }
}

/// Streaming `LossyEncoder::finish` must continue to reject Sub422
/// and Sub440 with the same gate as Sub420 (streaming wire-up is a
/// separate piece of work).
#[test]
fn streaming_sub422_and_sub440_still_invalid_config() {
    let rgb = make_rgb_buffer();
    for (tag, mode) in [
        ("4:2:2", ChromaSubsampling::Sub422),
        ("4:4:0", ChromaSubsampling::Sub440),
    ] {
        let cfg = LossyConfig::new(1.0).with_chroma_subsampling(mode);
        let mut enc = cfg
            .encoder(W as u32, H as u32, PixelLayout::Rgb8)
            .expect("streaming encoder build must succeed");
        for y in 0..H {
            let row_start = y * W * 3;
            let row_end = row_start + W * 3;
            enc.push_rows(&rgb[row_start..row_end], 1)
                .expect("push row");
        }
        let err = enc
            .finish()
            .expect_err(&format!("streaming {tag} must still return InvalidConfig"));
        assert_invalid_config_with_tag(err.error(), tag);
    }
}
