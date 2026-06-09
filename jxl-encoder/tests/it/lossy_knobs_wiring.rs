//! Integration tests for W4-2 lossy skeleton flag wiring.
//!
//! Verifies that each of the four lossy knobs (`alpha_distance`,
//! `group_order`, `center_x` / `center_y`, `upsampling_mode`) is plumbed
//! from [`LossyConfig`] through to the encoder and changes encoded bytes
//! in the expected way.
//!
//! The point of these tests is wiring proof, NOT quality calibration —
//! they assert that the knob affects the bitstream at all, not that the
//! affected bitstream is optimal. Lossy alpha and full upsampling LUT
//! semantics are tested elsewhere (and the alpha path in particular is
//! intentionally still lossless at all `alpha_distance` values for this
//! chunk; see the doc on [`LossyConfig::with_alpha_distance`]).

use jxl_encoder::{EpfDispatch, LossyConfig, PixelLayout, PixelLossDispatch};

fn rgb8_buf(w: u32, h: u32) -> Vec<u8> {
    (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect()
}

fn rgba8_buf(w: u32, h: u32) -> Vec<u8> {
    (0..(w * h * 4) as usize).map(|i| (i % 256) as u8).collect()
}

#[test]
fn upsampling_mode_changes_bytes_factor2() {
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let default_lut = LossyConfig::new(1.0)
        .with_resampling(2)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    let nearest_lut = LossyConfig::new(1.0)
        .with_resampling(2)
        .with_upsampling_mode(Some(0))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();

    assert_ne!(
        default_lut, nearest_lut,
        "upsampling_mode=Some(0) (nearest) must change the file-header \
         CustomTransformData block relative to the all-default fast path"
    );
    // Nearest LUT is one extra `!all_default` bit + the per-factor
    // weights; expect a strictly larger bitstream.
    assert!(
        nearest_lut.len() >= default_lut.len(),
        "nearest LUT encoder output ({}) should be >= default LUT ({}) — LUT bytes are appended",
        nearest_lut.len(),
        default_lut.len()
    );
}

#[test]
fn upsampling_mode_changes_bytes_factor4_pixel_dots() {
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let nearest = LossyConfig::new(1.0)
        .with_resampling(4)
        .with_upsampling_mode(Some(0))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    let dots = LossyConfig::new(1.0)
        .with_resampling(4)
        .with_upsampling_mode(Some(1))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();

    assert_ne!(
        nearest, dots,
        "upsampling_mode=Some(1) (pixel dots) at factor 4 has different \
         LUT slot values than mode=Some(0); encoded bytes must differ"
    );
}

#[test]
fn center_x_center_y_change_bytes_on_multigroup() {
    // 512x512 → 2x2 group grid → permutation is observable.
    let w = 512u32;
    let h = 512u32;
    let buf = rgb8_buf(w, h);

    let centered = LossyConfig::new(1.0)
        .with_group_order(Some(1))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    let off_centre = LossyConfig::new(1.0)
        .with_group_order(Some(1))
        .with_center_x(Some(0))
        .with_center_y(Some(0))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();

    assert_ne!(
        centered, off_centre,
        "shifting the AC group permutation centre to (0, 0) on a 2x2 \
         group grid must change the on-disk TOC ordering, which changes \
         encoded bytes"
    );
}

#[test]
fn group_order_one_implies_center_first() {
    // Same source, encode with `with_group_order(Some(1))` vs the
    // explicit `with_center_first(true)` setter. These should be
    // wire-equivalent (group_order=1 just flips center_first).
    let w = 512u32;
    let h = 512u32;
    let buf = rgb8_buf(w, h);

    let via_group_order = LossyConfig::new(1.0)
        .with_group_order(Some(1))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    let via_center_first = LossyConfig::new(1.0)
        .with_center_first(true)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    assert_eq!(
        via_group_order, via_center_first,
        "with_group_order(Some(1)) must produce the same bytes as \
         with_center_first(true) — they wire the same flag"
    );
}

#[test]
fn group_order_zero_disables_center_first() {
    // group_order=Some(0) (explicit scanline) cancels a previously-set
    // center_first. Verifies the with_group_order side-effect path.
    let w = 512u32;
    let h = 512u32;
    let buf = rgb8_buf(w, h);

    let scanline_default = LossyConfig::new(1.0)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    let scanline_explicit = LossyConfig::new(1.0)
        .with_center_first(true)
        .with_group_order(Some(0))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    assert_eq!(
        scanline_default, scanline_explicit,
        "with_group_order(Some(0)) after with_center_first(true) must \
         cancel center-first and produce default scanline bytes"
    );
}

#[test]
fn alpha_distance_unset_and_zero_are_lossless_byte_identical() {
    // `None` and `Some(0.0)` both mean "lossless alpha"; they MUST
    // emit identical bytes (default path, byte-for-byte equal to the
    // pre-pipeline lossless baseline). Guards the documented contract
    // on `LossyConfig::with_alpha_distance`.
    let w = 32u32;
    let h = 32u32;
    let buf = rgba8_buf(w, h);

    let unset = LossyConfig::new(1.0)
        .encode(&buf, w, h, PixelLayout::Rgba8)
        .unwrap();
    let zero = LossyConfig::new(1.0)
        .with_alpha_distance(Some(0.0))
        .encode(&buf, w, h, PixelLayout::Rgba8)
        .unwrap();

    assert_eq!(
        unset, zero,
        "alpha_distance=None and alpha_distance=Some(0.0) must produce \
         identical bytes (both mean lossless alpha)"
    );
}

#[test]
fn alpha_distance_nonzero_changes_bytes() {
    // alpha_distance > 0 with a single alpha extra channel must engage
    // the lossy alpha pipeline (separate modular multiplier on the
    // alpha sub-bitstream, matching libjxl
    // `enc_modular.cc:973-1027 + QuantizeChannel`). Bytes must differ
    // from the lossless baseline; this is the wiring proof for the
    // follow-on to W4-2-r.
    let w = 32u32;
    let h = 32u32;
    let buf = rgba8_buf(w, h);

    let lossless = LossyConfig::new(1.0)
        .encode(&buf, w, h, PixelLayout::Rgba8)
        .unwrap();
    // d=2 yields q ≈ 3 at 8-bit (libjxl formula: 0.025 * 2 * 1 *
    // 0.35 * 1.1 * 163.84 ≈ 3.15 → floor 3) so the tree carries
    // mul_bits=2, mul_log=0 and residuals divide by 3.
    let lossy_low = LossyConfig::new(1.0)
        .with_alpha_distance(Some(2.0))
        .encode(&buf, w, h, PixelLayout::Rgba8)
        .unwrap();
    // d=10 yields q ≈ 15 — definitely visible.
    let lossy_high = LossyConfig::new(1.0)
        .with_alpha_distance(Some(10.0))
        .encode(&buf, w, h, PixelLayout::Rgba8)
        .unwrap();

    assert_ne!(
        lossless, lossy_low,
        "alpha_distance=Some(2.0) must engage the lossy alpha tree \
         leaf (mul_bits=2, mul_log=0) and produce bytes different \
         from the lossless baseline"
    );
    assert_ne!(
        lossless, lossy_high,
        "alpha_distance=Some(10.0) must engage the lossy alpha tree \
         leaf (mul_bits=14, mul_log=0) and produce bytes different \
         from the lossless baseline"
    );
}

/// Decode through jxl-rs (primary decoder) — header parse + frame
/// render, returns (width, height) for cross-checks. Mirrors the
/// pattern used in `tests/content_class_dispatch_roundtrip.rs` so the
/// W12-4 audit's PARTIAL note on `--center_x/--center_y` ("wired but
/// lacks regression test") is closed with a real decoder roundtrip,
/// not just a "bytes differ" assertion.
fn decode_jxl_rs_rgb8_smoke(data: &[u8]) -> (u32, u32) {
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
    let mut output_image = Image::<u8>::new((width * channels, height)).expect("alloc rgb8 buffer");
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
    (width as u32, height as u32)
}

fn decode_jxl_oxide_smoke(data: &[u8]) -> (u32, u32) {
    use jxl_oxide::JxlImage;
    let image = JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("jxl-oxide: header parse must succeed");
    let header = image.image_header();
    let _frame = image
        .render_frame(0)
        .expect("jxl-oxide: render_frame must succeed");
    (header.size.width, header.size.height)
}

/// Closes the W12-4 audit's PARTIAL note on `--center_x` / `--center_y`
/// ("wired but lacks regression test"). The existing
/// `center_x_center_y_change_bytes_on_multigroup` test only asserts that
/// the bitstream bytes differ when the centre is shifted — it does NOT
/// verify the decoder still accepts the output. This test does both:
///
/// 1. A non-default centre permutation actually decodes through jxl-rs
///    (PRIMARY decoder) and jxl-oxide (SECONDARY) without errors.
/// 2. The header-reported dimensions round-trip back to the encoded
///    `(width, height)` — i.e. the AC group permutation does not corrupt
///    the file-header `SizeHeader`.
///
/// Tests three centres: image-centre default, top-left corner `(0, 0)`,
/// and bottom-right corner `(w-1, h-1)`. All three must produce a
/// 2×2-group bitstream that decodes pixel-shape-correctly.
#[test]
fn center_xy_decodes_through_jxl_rs_and_oxide() {
    // 512x512 → 2x2 group grid → permutation is observable in the TOC
    // section ordering. Group dim is 256 (matches libjxl `group_dim`).
    let w = 512u32;
    let h = 512u32;
    let buf = rgb8_buf(w, h);

    // (1) default centre (no override) — encoder falls back to image centre.
    let default_centre = LossyConfig::new(1.0)
        .with_effort(3)
        .with_group_order(Some(1))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();

    // (2) top-left centre — exercises the `dx > 0 && dy > 0` quadrant
    //     of `compute_center_first_ac_permutation`'s `side` derivation
    //     (within-group dx/dy = -128 from the central group's centre).
    let top_left = LossyConfig::new(1.0)
        .with_effort(3)
        .with_group_order(Some(1))
        .with_center_x(Some(0))
        .with_center_y(Some(0))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();

    // (3) bottom-left centre — different central group than default and
    //     top-left (group_x = 0, group_y = 1 → `cx=128, cy=384`).
    //     Together with the top-left case this exercises three distinct
    //     central groups out of the 2x2 grid, all reachable through
    //     `compute_center_first_ac_permutation`'s "central group containing
    //     the centre" logic (coeff_order.rs:660-667).
    let bottom_left = LossyConfig::new(1.0)
        .with_effort(3)
        .with_group_order(Some(1))
        .with_center_x(Some(0))
        .with_center_y(Some((h - 1) as i64))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();

    // Sanity: the three encodes produce distinct bitstreams (the
    // wiring assertion from `center_x_center_y_change_bytes_on_multigroup`,
    // extended to a third centre so the new test also stands alone).
    // Each of the three centres lands in a different central group of
    // the 2x2 grid (image-centre → group(1,1), top-left → group(0,0),
    // bottom-left → group(0,1)), so the central-first reorder picks a
    // different group first in each case.
    assert_ne!(default_centre, top_left);
    assert_ne!(default_centre, bottom_left);
    assert_ne!(top_left, bottom_left);

    // jxl-rs primary roundtrip — header dims must match for all three.
    for (label, bytes) in [
        ("default centre", &default_centre),
        ("top-left centre", &top_left),
        ("bottom-left centre", &bottom_left),
    ] {
        let (dw, dh) = decode_jxl_rs_rgb8_smoke(bytes);
        assert_eq!(dw, w, "{label}: jxl-rs decoded width must match input");
        assert_eq!(dh, h, "{label}: jxl-rs decoded height must match input");
    }

    // jxl-oxide secondary roundtrip — same assertion, secondary decoder.
    for (label, bytes) in [
        ("default centre", &default_centre),
        ("top-left centre", &top_left),
        ("bottom-left centre", &bottom_left),
    ] {
        let (dw, dh) = decode_jxl_oxide_smoke(bytes);
        assert_eq!(dw, w, "{label}: jxl-oxide decoded width must match input");
        assert_eq!(dh, h, "{label}: jxl-oxide decoded height must match input");
    }
}

/// Brotli quality knob (`--brotli-effort`, `with_brotli_metadata(q)`)
/// at q=11 (maximum) must produce a metadata box payload no larger
/// than q=1 (fastest). Confirms the value actually flows through to
/// the Brotli encoder — a previous wiring bug could silently pin the
/// quality at a default constant and this test would catch it.
///
/// Uses a 4 KB XMP payload of repeated structured XML — Brotli at
/// q=11 typically compresses this 5-10x smaller than at q=1.
///
/// The end-to-end bitstream must also still decode through jxl-rs +
/// jxl-oxide; the brob box is a JXL-spec container box that decoders
/// must accept (and either decompress to Exif/xml/jumb or skip).
#[cfg(feature = "brotli-metadata")]
#[test]
fn brotli_effort_q11_smaller_or_equal_to_q1_and_decodes() {
    use jxl_encoder::ImageMetadata;

    let w = 64u32;
    let h = 64u32;
    let pixels: Vec<u8> = (0..(w * h * 3) as usize)
        .map(|i| (i.wrapping_mul(31) % 251) as u8)
        .collect();

    // 4 KB of highly redundant XML — Brotli at high quality should
    // crush this far more than at low quality. Repeated structure
    // guarantees both q=1 and q=11 fall into the `brob` (compressed)
    // path, not the small-payload fallback to a plain `xml ` box.
    let xmp = "<rdf:Description><exif:CreatorTool>Test</exif:CreatorTool>\
               <dc:title>Repeated test payload for brotli quality A/B</dc:title></rdf:Description>"
        .repeat(48)
        .into_bytes();
    let meta = ImageMetadata::default().with_xmp(&xmp);

    let cfg = LossyConfig::new(1.0).with_effort(3);
    let bytes_q1 = cfg
        .clone()
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_metadata(&meta)
        .with_brotli_metadata(1)
        .encode(&pixels)
        .expect("encode with brotli q=1");
    let bytes_q11 = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_metadata(&meta)
        .with_brotli_metadata(11)
        .encode(&pixels)
        .expect("encode with brotli q=11");

    // Both must take the brob path (contain a `brob` box).
    let q1_has_brob = bytes_q1.windows(4).any(|x| x == b"brob");
    let q11_has_brob = bytes_q11.windows(4).any(|x| x == b"brob");
    assert!(q1_has_brob, "q=1 should still take the brob path");
    assert!(q11_has_brob, "q=11 should still take the brob path");

    // The codestream bytes are identical at both quality settings
    // (Brotli quality only affects container box payloads, not the
    // jxlp/jxlc codestream). So container size delta == brob payload
    // delta. q=11 must NOT be larger than q=1.
    assert!(
        bytes_q11.len() <= bytes_q1.len(),
        "brotli q=11 container ({}) must be <= q=1 container ({})",
        bytes_q11.len(),
        bytes_q1.len(),
    );

    // On a 4 KB repeated XML payload q=11 should win measurably —
    // require at least a 4-byte saving to guarantee the knob is
    // doing real work, not a silent no-op pinned at one quality
    // (sub-500-byte payloads bypass brob entirely; the precondition
    // above already excludes that case).
    assert!(
        bytes_q1.len() > bytes_q11.len(),
        "expected q=11 strictly smaller than q=1 on a redundant XML payload; \
         got q1={} q11={}. If equal, the brotli quality knob may not be wired.",
        bytes_q1.len(),
        bytes_q11.len(),
    );

    // Both bitstreams must decode end-to-end through jxl-rs (PRIMARY)
    // and jxl-oxide (SECONDARY). Decoders treat the brob box as an
    // opaque container entry — the codestream itself is unaffected.
    for (label, bytes) in [("q=1", &bytes_q1), ("q=11", &bytes_q11)] {
        let (dw, dh) = decode_jxl_rs_rgb8_smoke(bytes);
        assert_eq!(dw, w, "{label}: jxl-rs decoded width must match");
        assert_eq!(dh, h, "{label}: jxl-rs decoded height must match");
        let (dw, dh) = decode_jxl_oxide_smoke(bytes);
        assert_eq!(dw, w, "{label}: jxl-oxide decoded width must match");
        assert_eq!(dh, h, "{label}: jxl-oxide decoded height must match");
    }
}

/// `LosslessConfig::with_keep_invisible(false)` decode roundtrip
/// through jxl-rs (PRIMARY). Existing api_tests coverage uses
/// jxl-oxide only; this closes the audit gap by exercising jxl-rs on
/// the same skip-RGB output.
///
/// The decoded visible pixels (alpha=255) must round-trip pixel-exact;
/// invisible pixels (alpha=0) are allowed to differ from input (the
/// pre-pass zeroed them), which is the intended size-saving behaviour.
#[test]
fn lossless_keep_invisible_false_jxl_rs_roundtrip() {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};
    use jxl_encoder::LosslessConfig;

    let w = 16u32;
    let h = 16u32;
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    // Left half visible (structured), right half invisible (garbage).
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = (y * w as usize + x) * 4;
            if x < (w as usize) / 2 {
                pixels[idx] = (x * 17) as u8;
                pixels[idx + 1] = (y * 17) as u8;
                pixels[idx + 2] = ((x + y) * 9) as u8;
                pixels[idx + 3] = 255;
            } else {
                pixels[idx] = 0xAA;
                pixels[idx + 1] = 0xBB;
                pixels[idx + 2] = 0xCC;
                pixels[idx + 3] = 0;
            }
        }
    }

    let encoded = LosslessConfig::default()
        .with_keep_invisible(false)
        .encode(&pixels, w, h, PixelLayout::Rgba8)
        .expect("encode lossless RGBA with keep_invisible(false)");

    // jxl-rs decode with RGBA pixel format.
    let mut input: &[u8] = &encoded;
    let decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
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
    let basic = decoder.basic_info().clone();
    assert_eq!(basic.size, (w as usize, h as usize));
    // Lossless RGBA → 1 alpha extra channel.
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![
            Some(JxlDataFormat::U8 { bit_depth: 8 });
            basic.extra_channels.len()
        ],
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

    // RGB plane: width × 3 channels, packed [R, G, B, R, G, B, ...].
    let mut rgb_img = Image::<u8>::new((w as usize * 3, h as usize)).expect("alloc RGB plane");
    let mut alpha_img = Image::<u8>::new((w as usize, h as usize)).expect("alloc alpha plane");
    let mut buffers = vec![
        JxlOutputBuffer::from_image_rect_mut(
            rgb_img
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (w as usize * 3, h as usize),
                })
                .into_raw(),
        ),
        JxlOutputBuffer::from_image_rect_mut(
            alpha_img
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (w as usize, h as usize),
                })
                .into_raw(),
        ),
    ];
    loop {
        match decoder_frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_frame = fallback;
            }
            Err(e) => panic!("jxl-rs frame decode error: {e:?}"),
        }
    }

    // Visible pixels: pixel-exact RGB + alpha=255.
    for y in 0..h as usize {
        let rgb_row = rgb_img.row(y);
        let alpha_row = alpha_img.row(y);
        for x in 0..(w as usize) / 2 {
            let src_idx = (y * w as usize + x) * 4;
            for c in 0..3 {
                assert_eq!(
                    rgb_row[x * 3 + c],
                    pixels[src_idx + c],
                    "visible pixel ({x},{y}) channel {c} diverged on jxl-rs roundtrip",
                );
            }
            assert_eq!(
                alpha_row[x], 255,
                "visible pixel ({x},{y}) alpha must be 255 on jxl-rs roundtrip",
            );
        }
    }
    // Invisible pixels: alpha=0 is preserved bit-exact; RGB samples
    // were zeroed by the simplify_invisible pre-pass at the encoder
    // (api.rs:5634-5658) — the decoder must see those zeros back.
    for y in 0..h as usize {
        let rgb_row = rgb_img.row(y);
        let alpha_row = alpha_img.row(y);
        for x in (w as usize) / 2..w as usize {
            assert_eq!(
                alpha_row[x], 0,
                "invisible pixel ({x},{y}) alpha must remain 0",
            );
            for c in 0..3 {
                assert_eq!(
                    rgb_row[x * 3 + c],
                    0,
                    "invisible pixel ({x},{y}) channel {c} should be zeroed by pre-pass, got {}",
                    rgb_row[x * 3 + c],
                );
            }
        }
    }
}

// ─── W36-2: EpfDispatch wiring ─────────────────────────────────────────────

/// `LossyConfig::with_epf_dispatch` round-trips via `epf_dispatch()`.
// W44-130 Chunk D: `with_epf_dispatch` / `epf_dispatch()` setters
// deleted; behavioral coverage is now via the bytes-vary tests below
// (`epf_dispatch_always_default_changes_bytes_on_textured` +
// `epf_dispatch_auto_skips_on_flat_content`), which exercise the
// dispatch through the encode pipeline rather than the deleted
// getter — sufficient for the wiring contract this test covered.
// The `resolve_improvements()` method that produces the resolved
// policy is `pub(crate)`, so the in-process round-trip is covered
// by `api.rs::test_with_strategy_overrides_setter_roundtrip` (lib
// test).
//
// The `round_trips_through_config` smoke test is dropped here; the
// equivalent contract is preserved by the lib-test counterparts
// (`test_with_strategy_overrides_setter_roundtrip`,
// `test_dct_suppress_hint_api_roundtrip`, etc. in `api.rs`).

/// `EpfDispatch::AlwaysDefault` must change emitted bytes on
/// realistic content where the per-block search would otherwise
/// pick non-default sharpness values. On a synthetic checkerboard
/// (lots of strong edges) the per-block search picks varied
/// sharpness; forcing the uniform default produces a distinct
/// bitstream.
#[test]
fn epf_dispatch_always_default_changes_bytes_on_textured() {
    let w = 64u32;
    let h = 64u32;
    // Strong-edge checkerboard with 16-pixel cells — guarantees the
    // per-block search has work to do (the cell boundaries cross
    // many 8x8 block edges so different blocks favour different
    // sharpness values).
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let on = ((x / 16) + (y / 16)) % 2 == 0;
            let v = if on { 230u8 } else { 25u8 };
            buf.extend_from_slice(&[v, v, v]);
        }
    }
    // Effort 6 to make sure dynamic-sharpness gate is active. d=1.0
    // exceeds the 0.5 gate.
    // W44-130 Chunk D: setter deleted; dispatch lives on
    // `EncoderImprovementsCustom`.
    use jxl_encoder::api::{EncoderImprovementsCustom, EncoderStrategy};
    let mut cust = EncoderImprovementsCustom::default();
    cust.epf_dispatch = EpfDispatch::AlwaysSelect;
    let bytes_select = LossyConfig::new(1.0)
        .with_effort(6)
        .with_strategy(EncoderStrategy::Custom(Box::new(cust)))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("encode AlwaysSelect");
    let mut cust = EncoderImprovementsCustom::default();
    cust.epf_dispatch = EpfDispatch::AlwaysDefault;
    let bytes_default = LossyConfig::new(1.0)
        .with_effort(6)
        .with_strategy(EncoderStrategy::Custom(Box::new(cust)))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("encode AlwaysDefault");
    assert_ne!(
        bytes_select.len(),
        bytes_default.len(),
        "AlwaysDefault should produce a different byte length on textured \
         input than AlwaysSelect (select={}, default={})",
        bytes_select.len(),
        bytes_default.len()
    );
}

/// `EpfDispatch::Auto` must agree byte-for-byte with `AlwaysDefault`
/// on a perfectly flat image — the smoothness predicate fires
/// (mean(mask1x1) is saturated near 100) and the per-block search
/// is skipped.
#[test]
fn epf_dispatch_auto_skips_on_flat_content() {
    use jxl_encoder::api::{EncoderImprovementsCustom, EncoderStrategy};
    let w = 64u32;
    let h = 64u32;
    let buf = vec![128u8; (w * h * 3) as usize];
    let mut cust = EncoderImprovementsCustom::default();
    cust.epf_dispatch = EpfDispatch::Auto;
    let bytes_auto = LossyConfig::new(1.0)
        .with_effort(6)
        .with_strategy(EncoderStrategy::Custom(Box::new(cust)))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("encode Auto");
    let mut cust = EncoderImprovementsCustom::default();
    cust.epf_dispatch = EpfDispatch::AlwaysDefault;
    let bytes_default = LossyConfig::new(1.0)
        .with_effort(6)
        .with_strategy(EncoderStrategy::Custom(Box::new(cust)))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("encode AlwaysDefault");
    assert_eq!(
        bytes_auto, bytes_default,
        "Auto on flat content must produce identical bytes to AlwaysDefault \
         (smoothness predicate should skip the per-block search)"
    );
}

// ─── W38-2: PixelLossDispatch wiring ───────────────────────────────────────

// W44-130 Chunk D: `with_pixel_loss_dispatch` / `pixel_loss_dispatch()`
// setters deleted; behavioral coverage moves to the bytes-vary tests
// below. The round-trips-through-config + preserved-across-with-effort
// tests are dropped; equivalent lib-test contract lives in
// `api.rs::test_with_strategy_overrides_setter_roundtrip` and
// `test_with_strategy_preserved_across_with_effort`.

/// `PixelLossDispatch::AlwaysOff` on textured content must produce
/// a different bitstream than the byte-identical `AlwaysOn` default
/// — the pixel-domain loss term in the AC-strategy search cost is
/// removed, which changes which strategy wins for at least some
/// blocks at d=1.0 effort 5 (where the loss term is active).
#[test]
fn pixel_loss_dispatch_always_off_changes_bytes_on_textured() {
    let w = 64u32;
    let h = 64u32;
    // Strong-edge checkerboard with 16-pixel cells — guarantees the
    // AC-strategy search has work to do across multiple strategy
    // candidates, so the pixel-domain loss term participates.
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let on = ((x / 16) + (y / 16)) % 2 == 0;
            let v = if on { 230u8 } else { 25u8 };
            buf.extend_from_slice(&[v, v, v]);
        }
    }
    // Effort 5 (Hare) — `profile.pixel_domain_loss = true` and AC
    // strategy search is active.
    // W44-130 Chunk D: setter deleted; dispatch lives on
    // `EncoderImprovementsCustom`.
    use jxl_encoder::api::{EncoderImprovementsCustom, EncoderStrategy};
    let mut cust = EncoderImprovementsCustom::default();
    cust.pixel_loss_dispatch = PixelLossDispatch::AlwaysOn;
    let bytes_on = LossyConfig::new(1.0)
        .with_effort(5)
        .with_strategy(EncoderStrategy::Custom(Box::new(cust)))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("encode AlwaysOn");
    let mut cust = EncoderImprovementsCustom::default();
    cust.pixel_loss_dispatch = PixelLossDispatch::AlwaysOff;
    let bytes_off = LossyConfig::new(1.0)
        .with_effort(5)
        .with_strategy(EncoderStrategy::Custom(Box::new(cust)))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("encode AlwaysOff");
    assert_ne!(
        bytes_on,
        bytes_off,
        "AlwaysOff should produce different bytes than AlwaysOn on textured input \
         (on={} bytes, off={} bytes)",
        bytes_on.len(),
        bytes_off.len()
    );
}

/// `PixelLossDispatch::Auto` on a perfectly flat image (saturated
/// mask1x1, median > 80) must agree byte-for-byte with `AlwaysOff`
/// — the smoothness predicate fires and drops mask1x1 before the
/// AC-strategy search.
#[test]
fn pixel_loss_dispatch_auto_skips_on_flat_content() {
    use jxl_encoder::api::{EncoderImprovementsCustom, EncoderStrategy};
    let w = 64u32;
    let h = 64u32;
    let buf = vec![128u8; (w * h * 3) as usize];
    let mut cust = EncoderImprovementsCustom::default();
    cust.pixel_loss_dispatch = PixelLossDispatch::Auto;
    let bytes_auto = LossyConfig::new(1.0)
        .with_effort(5)
        .with_strategy(EncoderStrategy::Custom(Box::new(cust)))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("encode Auto");
    let mut cust = EncoderImprovementsCustom::default();
    cust.pixel_loss_dispatch = PixelLossDispatch::AlwaysOff;
    let bytes_off = LossyConfig::new(1.0)
        .with_effort(5)
        .with_strategy(EncoderStrategy::Custom(Box::new(cust)))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("encode AlwaysOff");
    assert_eq!(
        bytes_auto, bytes_off,
        "Auto on flat content must produce identical bytes to AlwaysOff \
         (smoothness predicate should drop mask1x1)"
    );
}

/// `PixelLossDispatch::AlwaysOn` (the default) must produce
/// byte-identical output to a freshly-constructed config without
/// any `with_pixel_loss_dispatch` call — this is the hash-lock
/// byte-identical contract.
#[test]
fn pixel_loss_dispatch_default_byte_identical_to_explicit_always_on() {
    use jxl_encoder::api::{EncoderImprovementsCustom, EncoderStrategy};
    let w = 64u32;
    let h = 64u32;
    let buf = rgb8_buf(w, h);
    let bytes_default = LossyConfig::new(1.0)
        .with_effort(5)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("encode default");
    let mut cust = EncoderImprovementsCustom::default();
    cust.pixel_loss_dispatch = PixelLossDispatch::AlwaysOn;
    let bytes_explicit = LossyConfig::new(1.0)
        .with_effort(5)
        .with_strategy(EncoderStrategy::Custom(Box::new(cust)))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .expect("encode explicit AlwaysOn");
    assert_eq!(
        bytes_default, bytes_explicit,
        "default config (implicit AlwaysOn) must produce byte-identical output \
         to explicit AlwaysOn"
    );
}
