// Integration test for the public JPEG → JXL lossless transcoding API
// (`LosslessConfig::encode_jpeg_transcode` and friends, plus
// `jpeg::is_jpeg_signature`).
//
// Exercises the public API surface only — does NOT call the internal
// `crate::jpeg::*` functions directly, so this also serves as a
// "the API is reachable from outside the crate" check.
//
// Verification layers (per CLAUDE.md "Proof-by-Tests" methodology):
//   Layer 1 — signature sniffing (no encode needed)
//   Layer 2 — encode JPEG bytes through `LosslessConfig::encode_jpeg_transcode`
//             and verify the result starts with the JXL container signature
//             AND contains a JBRD box
//   Layer 3 — encode JPEG bytes via the codestream-only variant and verify
//             the result starts with the bare JXL codestream signature
//   Layer 4 — round-trip pixels through jxl-rs and check dims+pixel range
//
// Note: byte-exact JPEG reconstruction via `djxl --reconstruct_jpeg` has
// pre-existing issues on the test fixture (see `test_jbrd_roundtrip_small`
// in `jpeg_reencoding.rs`); it is exercised through that test, not here.

#![cfg(feature = "jpeg-reencoding")]

use jxl_encoder::jpeg::is_jpeg_signature;
use jxl_encoder::{EncodeError, LosslessConfig};

/// Minimal in-tree JPEG fixture. Path is fixed; the file is synthesized
/// once and re-used across runs. We synthesize via the `image` crate
/// (dev-dep) rather than depending on `/mnt/v/...` so the test is
/// hermetic.
///
/// Test-parallel race protection: callers may be invoking this from
/// multiple threads simultaneously. We serialize the synthesis behind a
/// process-wide `OnceLock`-backed `Mutex` so the file is written exactly
/// once, then everyone reads from the same on-disk path.
fn ensure_test_jpeg() -> std::path::PathBuf {
    use std::sync::{Mutex, OnceLock};
    static SYNTH_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = SYNTH_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();

    // Use the dedicated test_jpegs subdir under output_dir so we don't
    // collide with the broken `output_dir_for` fixture-file paths that
    // jpeg_reencoding.rs uses. This dir is created fresh each test run.
    let dir = jxl_encoder::test_helpers::output_dir("jpeg_public_api");
    let path = dir.join("public_api_fixture.jpg");
    if path.exists()
        && std::fs::metadata(&path)
            .map(|m| m.is_file())
            .unwrap_or(false)
    {
        return path;
    }
    // Synthesize a tiny 32x32 RGB gradient JPEG via the `image` crate
    // (dev-dep). We use a non-trivial gradient so the encoder produces
    // real coefficient variance (vs a flat color, which has a degenerate
    // entropy distribution).
    let w: u32 = 32;
    let h: u32 = 32;
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x * 8) as u8);
            pixels.push((y * 8) as u8);
            pixels.push(((x + y) * 4) as u8);
        }
    }
    let img = image::RgbImage::from_raw(w, h, pixels).expect("image::RgbImage::from_raw");
    let mut bytes: Vec<u8> = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut bytes);
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut cursor, image::ImageFormat::Jpeg)
            .expect("image jpeg write");
    }
    std::fs::write(&path, &bytes).expect("write fixture");
    path
}

// ── Layer 1 — signature sniffing ──────────────────────────────────────

#[test]
fn layer1_is_jpeg_signature_positive() {
    let bytes = [0xFFu8, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
    assert!(is_jpeg_signature(&bytes));
}

#[test]
fn layer1_is_jpeg_signature_negative_png() {
    // PNG signature
    let bytes = [0x89u8, 0x50, 0x4E, 0x47];
    assert!(!is_jpeg_signature(&bytes));
}

#[test]
fn layer1_is_jpeg_signature_negative_empty() {
    assert!(!is_jpeg_signature(&[]));
    assert!(!is_jpeg_signature(&[0xFFu8]));
    assert!(!is_jpeg_signature(&[0xFFu8, 0xD8])); // SOI but no marker byte
}

// ── Layer 2 — encode_jpeg_transcode → JXL container with JBRD ────────

#[test]
fn layer2_encode_jpeg_transcode_produces_container_with_jbrd() {
    let path = ensure_test_jpeg();
    let jpeg = std::fs::read(&path).expect("read fixture");
    let jxl = LosslessConfig::new()
        .encode_jpeg_transcode(&jpeg)
        .expect("encode_jpeg_transcode");
    // JXL container: starts with `\0\0\0\x0c JXL `.
    assert!(jxl.len() >= 12, "output too short: {} bytes", jxl.len());
    assert_eq!(
        &jxl[..4],
        &[0x00, 0x00, 0x00, 0x0C],
        "container signature size"
    );
    assert_eq!(&jxl[4..8], b"JXL ", "container signature type");
    // JBRD box marker must appear somewhere in the container.
    let has_jbrd = jxl.windows(4).any(|w| w == b"jbrd");
    assert!(has_jbrd, "expected jbrd box in container");
    eprintln!(
        "encode_jpeg_transcode: {} JPEG bytes → {} JXL bytes (with JBRD)",
        jpeg.len(),
        jxl.len()
    );
}

#[test]
fn layer2_encode_jpeg_transcode_rejects_non_jpeg() {
    let cfg = LosslessConfig::new();
    let err = cfg
        .encode_jpeg_transcode(&[0x89, 0x50, 0x4E, 0x47])
        .expect_err("should reject PNG bytes");
    // Should be JpegParse, not Internal or InvalidInput.
    let (inner, _trace) = err.decompose();
    match inner {
        EncodeError::JpegParse { message } => {
            assert!(!message.is_empty(), "empty JpegParse message");
        }
        other => panic!("expected JpegParse, got {other:?}"),
    }
}

// ── Layer 3 — encode_jpeg_transcode_codestream → bare codestream ────

#[test]
fn layer3_encode_jpeg_transcode_codestream_produces_bare_jxl() {
    let path = ensure_test_jpeg();
    let jpeg = std::fs::read(&path).expect("read fixture");
    let jxl = LosslessConfig::new()
        .encode_jpeg_transcode_codestream(&jpeg)
        .expect("encode_jpeg_transcode_codestream");
    assert!(jxl.len() >= 2, "output too short: {} bytes", jxl.len());
    // Bare codestream starts with the 2-byte JXL signature 0xFF 0x0A.
    assert_eq!(jxl[0], 0xFF, "JXL signature byte 0");
    assert_eq!(jxl[1], 0x0A, "JXL signature byte 1");
    // No container, so no JBRD box, so no 'jbrd' bytes near the start.
    let head = &jxl[..jxl.len().min(64)];
    assert!(
        !head.windows(4).any(|w| w == b"jbrd"),
        "bare codestream should NOT have jbrd box"
    );
    eprintln!(
        "encode_jpeg_transcode_codestream: {} JPEG bytes → {} JXL bytes (bare)",
        jpeg.len(),
        jxl.len()
    );
}

// ── Layer 4 — pixel roundtrip through jxl-rs ─────────────────────────

#[test]
fn layer4_pixel_roundtrip_via_jxl_rs() {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let path = ensure_test_jpeg();
    let jpeg = std::fs::read(&path).expect("read fixture");
    let jxl = LosslessConfig::new()
        .encode_jpeg_transcode_codestream(&jpeg)
        .expect("encode codestream");

    let mut input: &[u8] = &jxl;
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                assert!(!input.is_empty(), "EOF during jxl-rs header");
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs header decode error: {e:?}"),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (w, h) = basic_info.size;

    let format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(format);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                assert!(!input.is_empty(), "EOF before jxl-rs frame");
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame info error: {e:?}"),
        }
    };

    let mut out_image = Image::<f32>::new((w * 3, h)).expect("alloc");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        out_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (w * 3, h),
            })
            .into_raw(),
    )];

    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                assert!(!input.is_empty(), "EOF during decode");
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs decode error: {e:?}"),
        }
    }
    assert_eq!(w, 32);
    assert_eq!(h, 32);
    // sanity check pixel range
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for y in 0..h {
        for &p in out_image.row(y) {
            if p < min {
                min = p;
            }
            if p > max {
                max = p;
            }
        }
    }
    assert!(
        min > -0.5 && max < 1.5,
        "pixels out of expected range: [{min}, {max}]"
    );
    eprintln!("jxl-rs decoded {w}x{h} RGB OK (pixel range [{min:.3}, {max:.3}])");
}
