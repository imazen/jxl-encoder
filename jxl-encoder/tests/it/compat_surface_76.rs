// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Reachability pin for the #76 (0.4.0) doc-hidden compat block in `lib.rs`.
//!
//! These items are **unsupported** — they may change freely — but they must
//! stay *reachable*, because downstream (zenjxl) consumes them today. This
//! file is an integration test on purpose: it links `jxl_encoder` as an
//! external crate, so it exercises the same public paths a consumer sees.
//! A `pub(crate)` item, or one dropped from the block, fails to compile here.
//!
//! Why it exists: #100. `container::wrap_in_container` was in that consumer
//! set but not in the compat block, so the 0.4.0 restructure silently removed
//! its public path and zenjxl's zencodec animation adapter had to carry a
//! local port of the box layout. Nothing pinned the block, so nothing caught
//! it. The rule this encodes: **every item in the compat block gets touched
//! here.**
//!
//! `wrap_in_container` is specifically the animation path's only metadata
//! route — the still and streaming paths thread EXIF/XMP through
//! `EncodeRequest` / `LossyEncoder::with_exif` and wrap internally, but
//! `encode_animation` has no metadata surface, so a post-encode wrap is it.
//! The test therefore wraps a *real animation codestream* rather than a
//! synthetic buffer, and checks all three decoders still accept the result.

use jxl_encoder::{AnimationFrame, AnimationParams, LosslessConfig, PixelLayout};

/// Solid-colour 64x64 RGB8 frame.
fn solid_rgb(r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(64 * 64 * 3);
    for _ in 0..64 * 64 {
        pixels.extend_from_slice(&[r, g, b]);
    }
    pixels
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Encode a 3-frame lossless animation and return the bare codestream.
fn three_frame_animation() -> Vec<u8> {
    let red = solid_rgb(255, 0, 0);
    let green = solid_rgb(0, 255, 0);
    let blue = solid_rgb(0, 0, 255);
    let frames = [
        AnimationFrame {
            pixels: &red,
            duration: 1,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &green,
            duration: 2,
            ..Default::default()
        },
        AnimationFrame {
            pixels: &blue,
            duration: 3,
            ..Default::default()
        },
    ];
    let animation = AnimationParams {
        tps_numerator: 10,
        tps_denominator: 1,
        num_loops: 0,
        premultiplied_alpha: false,
    };
    LosslessConfig::new()
        .encode_animation(64, 64, PixelLayout::Rgb8, &animation, &frames)
        .unwrap_or_else(|e| panic!("encode_animation failed: {e:?}"))
}

/// Decode every keyframe with jxl-oxide; returns (w, h, per-frame pixels).
fn decode_frames_oxide(data: &[u8]) -> (usize, usize, Vec<Vec<f32>>) {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .unwrap_or_else(|e| panic!("jxl-oxide decode failed: {e:?}"));
    let (w, h) = (image.width() as usize, image.height() as usize);
    let n = image.num_loaded_keyframes();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let render = image
            .render_frame(i)
            .unwrap_or_else(|e| panic!("jxl-oxide render_frame({i}) failed: {e:?}"));
        out.push(render.image_all_channels().buf().to_vec());
    }
    (w, h, out)
}

/// Decode the first frame with jxl-rs (the primary decoder). Returns
/// `(width, height, interleaved RGB f32)`.
///
/// `test_helpers::decode_with_jxl_rs` is `#[cfg(test)]`, i.e. unit-test only,
/// so an integration test has to drive the decoder itself.
fn decode_first_frame_jxlrs(data: &[u8]) -> (usize, usize, Vec<f32>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());

    let mut init = decoder;
    let mut decoder = loop {
        match init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => init = fallback,
            Err(e) => panic!("jxl-rs header decode failed: {e:?}"),
        }
    };

    let (width, height) = decoder.basic_info().size;
    const CHANNELS: usize = 3;
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    });

    let mut frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame-info decode failed: {e:?}"),
        }
    };

    let mut out = Image::<f32>::new((width * CHANNELS, height)).expect("allocate output");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        out.get_rect_mut(Rect {
            origin: (0, 0),
            size: (width * CHANNELS, height),
        })
        .into_raw(),
    )];
    loop {
        match frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => frame = fallback,
            Err(e) => panic!("jxl-rs frame decode failed: {e:?}"),
        }
    }

    let mut flat = Vec::with_capacity(width * height * CHANNELS);
    for y in 0..height {
        flat.extend_from_slice(out.row(y));
    }
    (width, height, flat)
}

/// #100: `wrap_in_container` must be reachable from an external consumer, and
/// what it produces must still decode.
///
/// Asserted, in order: the public path exists; the output is an ISOBMFF
/// container and the bare codestream is not; the `Exif` box carries the
/// 4-byte zero TIFF offset the spec requires ahead of the payload; the `xml `
/// box carries the XMP verbatim; and the wrapped stream renders identically
/// to the bare one in **jxl-rs, jxl-oxide and djxl**.
#[test]
fn wrap_in_container_is_reachable_and_its_output_still_decodes() {
    let bare = three_frame_animation();
    assert!(
        jxl_encoder::is_bare_codestream(&bare),
        "encode_animation should emit a bare codestream (it has no metadata surface)"
    );
    assert!(!jxl_encoder::is_container(&bare));

    // A minimal but real TIFF/Exif payload: little-endian header, one IFD
    // entry (Orientation = 1). Content is opaque to the wrapper; using a
    // well-formed one keeps the fixture honest for tools that parse it.
    let exif: &[u8] = &[
        0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, 0x01, 0x00, 0x12, 0x01, 0x03, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    let xmp: &[u8] = br#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta xmlns:x="adobe:ns:meta/"/><?xpacket end="w"?>"#;

    let wrapped = jxl_encoder::wrap_in_container(&bare, Some(exif), Some(xmp));

    assert!(
        jxl_encoder::is_container(&wrapped),
        "wrapped output must carry the ISOBMFF JXL container signature"
    );
    assert!(!jxl_encoder::is_bare_codestream(&wrapped));
    assert!(
        wrapped.len() > bare.len(),
        "wrapping must not drop the codestream: {} vs {}",
        wrapped.len(),
        bare.len()
    );

    // ISO/IEC 18181-2: the `Exif` box payload is a 4-byte big-endian offset
    // to the TIFF header (0 here) followed by the TIFF stream itself.
    let exif_at = find_subseq(&wrapped, b"Exif").expect("Exif box FourCC missing");
    let tiff_at = exif_at + 4;
    assert_eq!(
        &wrapped[tiff_at..tiff_at + 4],
        &[0, 0, 0, 0],
        "Exif box must begin with the 4-byte zero TIFF offset"
    );
    assert_eq!(
        &wrapped[tiff_at + 4..tiff_at + 4 + exif.len()],
        exif,
        "Exif payload must be carried verbatim"
    );

    let xml_at = find_subseq(&wrapped, b"xml ").expect("xml box FourCC missing");
    assert_eq!(
        &wrapped[xml_at + 4..xml_at + 4 + xmp.len()],
        xmp,
        "XMP payload must be carried verbatim"
    );

    // The wrapped stream must be equivalent to the bare one for every decoder.
    let (bw, bh, bare_frames) = decode_frames_oxide(&bare);
    let (ww, wh, wrapped_frames) = decode_frames_oxide(&wrapped);
    assert_eq!((bw, bh), (64, 64));
    assert_eq!((ww, wh), (bw, bh), "wrapping must not change dimensions");
    assert_eq!(
        wrapped_frames.len(),
        3,
        "expected 3 keyframes, got {}",
        wrapped_frames.len()
    );
    assert_eq!(
        wrapped_frames, bare_frames,
        "wrapping is metadata-only: decoded pixels must be identical"
    );

    // jxl-rs (primary decoder) must accept the container form too, and
    // render the same first frame it renders from the bare codestream.
    let (rw, rh, rs_wrapped) = decode_first_frame_jxlrs(&wrapped);
    assert_eq!((rw, rh), (64, 64), "jxl-rs dimensions from the container");
    let (_, _, rs_bare) = decode_first_frame_jxlrs(&bare);
    assert_eq!(
        rs_wrapped, rs_bare,
        "jxl-rs: wrapping is metadata-only, frame 0 must be bit-identical"
    );

    // djxl (libjxl v0.12) — decode-only; a nonzero exit means libjxl rejected
    // the boxes we wrote.
    let dir = jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "compat_surface_76");
    let path = dir.join("animation_wrapped.jxl");
    std::fs::write(&path, &wrapped).expect("write wrapped animation");
    let out = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&path)
        .arg("--disable_output")
        .arg("--num_threads=1")
        .output()
        .expect("run djxl v0.12");
    assert!(
        out.status.success(),
        "djxl rejected the wrapped animation container: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Every remaining item in the #76 compat block, touched so that removing one
/// fails to compile here rather than silently downstream (#100's failure mode).
#[test]
fn compat_block_76_items_stay_reachable() {
    let bare = three_frame_animation();

    // Container probes.
    assert!(jxl_encoder::is_bare_codestream(&bare));
    assert!(!jxl_encoder::is_container(&bare));

    // Gain-map box append.
    let jhgm: &[u8] = b"jhgm-payload";
    let with_gain_map = jxl_encoder::append_gain_map_box(&bare, jhgm);
    assert!(
        find_subseq(&with_gain_map, b"jhgm").is_some(),
        "append_gain_map_box must emit a jhgm box"
    );

    // Pre-encode estimators (zenjxl's admission pre-flight).
    // Signature: (w, h, input_bpp, has_alpha, is_lossless, effort).
    let est: jxl_encoder::EncodeEstimate =
        jxl_encoder::estimate_encode(64, 64, 3, false, true, 7).expect("estimate must be finite");
    assert!(
        est.peak_memory_bytes_min <= est.peak_memory_bytes
            && est.peak_memory_bytes <= est.peak_memory_bytes_max,
        "estimate band must be ordered: {} <= {} <= {}",
        est.peak_memory_bytes_min,
        est.peak_memory_bytes,
        est.peak_memory_bytes_max
    );
    let threaded = jxl_encoder::estimate_encode_threaded(64, 64, 3, false, true, 7, 4)
        .expect("threaded estimate must be finite");
    assert!(
        threaded.peak_memory_bytes >= est.peak_memory_bytes,
        "threading may only add memory"
    );
    let info: jxl_encoder::ThreadingInfo = jxl_encoder::encode_threading_info(true, 7);
    assert!(info.max_useful_threads >= 1);
}
