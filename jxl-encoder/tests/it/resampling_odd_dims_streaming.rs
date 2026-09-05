//! Issue #101 companion to `resampling_odd_dims.rs`: the SizeHeader must
//! carry the SOURCE size on every resampling path, including the ones the
//! sibling module does not exercise.
//!
//! Background. Under libjxl's automatic 2× resampling rule
//! (`enc_frame.cc:108-114`; since the #101 follow-up reached only through
//! `with_auto_resampling(true)` or `EncoderStrategy::Libjxl` — the zen
//! strategies keep one regime) pixels at distance ≥ 10 are downsampled to
//! `ceil(w/2) × ceil(h/2)`, the frame header carries `upsampling = 2`, and
//! a decoder upsamples the coded frame and crops it to the SizeHeader size.
//! `build_file_header` used to rebuild the advertised size as
//! `coded × factor`, rounding every odd dimension UP (513×769 → 257×385 →
//! 514×770); fixed in 71e0f6af (`VarDctEncoder::display_dims`). The
//! two-pass frame header also hard-coded `ec_upsampling = 1`, making every
//! lossy+alpha encode at resampling > 1 an invalid stream; fixed in
//! 11828823. Issue #101 (2026-09-05) reported the dimension defect from a
//! sweep whose zenjxl build path-patched a checkout of this repo that
//! predated both fixes: 130 of its 585 persisted odd-dimension bitstreams
//! — exactly the 13 images × 10 distances at d ≥ 10 — declare a size one
//! pixel too large per odd axis.
//!
//! What this module adds over `resampling_odd_dims.rs` (one-shot Rgb8,
//! jxl-rs + jxl-oxide): the streaming `LossyEncoder` entry point, Rgba8
//! (the alpha plane is downsampled too, so `ec_upsampling` must follow),
//! multi-group coded cells (513×769 → 257×385 = 2×2 groups) in both
//! orientations, the issue's exact cell at both ends of the auto band,
//! streaming == one-shot byte identity (also for `already_downsampled`),
//! an `#[ignore]` djxl leg, and a decoder-independent LSB-first parse of
//! the codestream's own SizeHeader — decoders only reproduce what the
//! header says, so a decoder returning 514×770 for a header saying
//! 514×770 is correct and proves nothing on its own.

use jxl_encoder::api::{LossyConfig, PixelLayout};

// ─── minimal codestream SizeHeader parse (spec §SizeHeader) ────────────

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl BitReader<'_> {
    fn read(&mut self, n: u32) -> u64 {
        let mut v = 0u64;
        for i in 0..n {
            let byte = self.data[self.pos / 8];
            let bit = (byte >> (self.pos % 8)) & 1;
            v |= u64::from(bit) << i;
            self.pos += 1;
        }
        v
    }

    /// `U32(Bits(9)+1, Bits(13)+1, Bits(18)+1, Bits(30)+1)`.
    fn read_size_u32(&mut self) -> u64 {
        let sel = self.read(2) as usize;
        let bits = [9u32, 13, 18, 30][sel];
        self.read(bits) + 1
    }
}

/// Parse `(xsize, ysize)` straight out of the codestream's SizeHeader.
/// Mirrors the writer in `headers/file_header.rs::write_size_header`.
fn parse_size_header(codestream: &[u8]) -> (u32, u32) {
    assert_eq!(
        &codestream[..2],
        &[0xFF, 0x0A],
        "expected a bare JXL codestream signature"
    );
    let mut r = BitReader {
        data: &codestream[2..],
        pos: 0,
    };
    let small = r.read(1) == 1;
    let (ysize, ratio) = if small {
        ((r.read(5) + 1) * 8, r.read(3))
    } else {
        let y = r.read_size_u32();
        (y, r.read(3))
    };
    let xsize = match ratio {
        0 => {
            if small {
                (r.read(5) + 1) * 8
            } else {
                r.read_size_u32()
            }
        }
        1 => ysize,
        2 => ysize * 12 / 10,
        3 => ysize * 4 / 3,
        4 => ysize * 3 / 2,
        5 => ysize * 16 / 9,
        6 => ysize * 5 / 4,
        7 => ysize * 2,
        _ => unreachable!("3-bit ratio"),
    };
    (xsize as u32, ysize as u32)
}

// ─── decoders ─────────────────────────────────────────────────────────

/// jxl-rs (PRIMARY decoder per CLAUDE.md): returns the decoded
/// `(width, height)` plus the RGBA8 buffer (proves a full render, not
/// just a header parse).
fn decode_jxl_rs_rgba8(data: &[u8]) -> (u32, u32, Vec<u8>) {
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
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_init = fallback,
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
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
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
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_frame = fallback,
            Err(e) => panic!("jxl-rs frame decode error: {e:?}"),
        }
    }
    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }
    (width as u32, height as u32, pixels)
}

/// jxl-oxide (secondary): `(header dims, rendered frame dims)`.
fn decode_jxl_oxide_dims(data: &[u8]) -> ((u32, u32), (u32, u32)) {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("jxl-oxide header parse");
    let header = (image.width(), image.height());
    let frame = image.render_frame(0).expect("jxl-oxide render_frame(0)");
    let stream = frame.stream();
    (header, (stream.width(), stream.height()))
}

// ─── fixture + the shared assertion ───────────────────────────────────

/// Procedural fixture (zero committed bytes): a smooth gradient plus
/// xorshift noise so the VarDCT path has both DC and AC energy.
fn fixture(w: u32, h: u32, channels: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_u32 ^ (w << 16) ^ h;
    let mut px = Vec::with_capacity(w as usize * h as usize * channels);
    for y in 0..h {
        for x in 0..w {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (state & 0x1F) as u8;
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = ((x ^ y) & 0xFF) as u8;
            px.push(r.wrapping_add(noise));
            px.push(g.wrapping_add(noise));
            px.push(b);
            if channels == 4 {
                // Alpha varies too so the alpha downsample path is live.
                px.push(255u8.wrapping_sub(((x + y) & 0x7F) as u8));
            }
        }
    }
    px
}

/// The issue-#101 invariant: the codestream advertises the SOURCE size
/// and both decoders reproduce exactly that size.
fn assert_source_size(bytes: &[u8], w: u32, h: u32, ctx: &str) {
    let declared = parse_size_header(bytes);
    assert_eq!(
        declared,
        (w, h),
        "{ctx}: SizeHeader declares {declared:?}, source is ({w}, {h}) — issue #101"
    );
    let (oxide_header, oxide_rendered) = decode_jxl_oxide_dims(bytes);
    assert_eq!(oxide_header, (w, h), "{ctx}: jxl-oxide header dims");
    assert_eq!(oxide_rendered, (w, h), "{ctx}: jxl-oxide rendered dims");
    let (rw, rh, rgba) = decode_jxl_rs_rgba8(bytes);
    assert_eq!((rw, rh), (w, h), "{ctx}: jxl-rs decoded dims");
    assert_eq!(
        rgba.len(),
        w as usize * h as usize * 4,
        "{ctx}: jxl-rs RGBA8 buffer length"
    );
}

// ─── cells ────────────────────────────────────────────────────────────

/// Auto-resample (d ≥ 10) on odd dimensions, one-shot Rgb8. Includes the
/// exact cell from the issue (513×769 at d=10, declared 514×770 before
/// 71e0f6af) and its transpose, at both ends of the auto band.
#[test]
fn issue_101_auto_resample_odd_dims_one_shot_rgb8_header_is_source_size() {
    for &(w, h) in &[(65u32, 33u32), (33, 65), (513, 769), (769, 513)] {
        for &d in &[10.0f32, 25.0] {
            let cfg = LossyConfig::new(d)
                .with_effort(5)
                .with_auto_resampling(true);
            assert_eq!(
                cfg.effective_resampling(),
                2,
                "auto-resample must engage at d={d}"
            );
            let px = fixture(w, h, 3);
            let bytes = cfg
                .encode_request(w, h, PixelLayout::Rgb8)
                .encode(&px)
                .unwrap_or_else(|e| panic!("encode {w}x{h} d={d}: {e}"));
            assert_source_size(&bytes, w, h, &format!("one-shot rgb8 {w}x{h} d={d}"));
        }
    }
}

/// Streaming `LossyEncoder` path at d=10 with alpha (Rgba8): the alpha
/// plane is downsampled alongside RGB. Also pins streaming == one-shot
/// bytes (both routes share the downsample + header wiring).
#[test]
fn issue_101_auto_resample_odd_dims_streaming_rgba8_header_is_source_size() {
    for &(w, h) in &[(65u32, 33u32), (513, 769)] {
        let cfg = LossyConfig::new(10.0)
            .with_effort(5)
            .with_auto_resampling(true);
        let px = fixture(w, h, 4);
        let mut enc = cfg
            .encoder(w, h, PixelLayout::Rgba8)
            .expect("streaming encoder");
        enc.push_rows(&px, h).expect("push_rows");
        let streaming = enc.finish().expect("finish");
        assert_source_size(&streaming, w, h, &format!("streaming rgba8 {w}x{h} d=10"));

        let one_shot = cfg
            .encode_request(w, h, PixelLayout::Rgba8)
            .encode(&px)
            .expect("one-shot rgba8");
        assert_source_size(&one_shot, w, h, &format!("one-shot rgba8 {w}x{h} d=10"));
        assert_eq!(
            streaming, one_shot,
            "streaming and one-shot rgba8 {w}x{h} d=10 must be byte-identical"
        );
    }
}

/// Explicit `with_resampling(2|4|8)` on odd dimensions: the header must
/// still carry the source size for every factor (65×33 codes as
/// 33×17 / 17×9 / 9×5). Multi-group factor-4 cell: 1025×513 → 257×129.
#[test]
fn issue_101_explicit_resampling_odd_dims_header_is_source_size() {
    for &f in &[2u32, 4, 8] {
        let (w, h) = (65u32, 33u32);
        let cfg = LossyConfig::new(2.0).with_effort(5).with_resampling(f);
        assert_eq!(cfg.effective_resampling(), f);
        let px = fixture(w, h, 3);
        let bytes = cfg
            .encode_request(w, h, PixelLayout::Rgb8)
            .encode(&px)
            .unwrap_or_else(|e| panic!("encode {w}x{h} resampling={f}: {e}"));
        assert_source_size(&bytes, w, h, &format!("explicit resampling={f} {w}x{h}"));
    }
    let (w, h) = (1025u32, 513u32);
    let cfg = LossyConfig::new(2.0).with_effort(5).with_resampling(4);
    let px = fixture(w, h, 3);
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(&px)
        .expect("encode 1025x513 resampling=4");
    assert_source_size(&bytes, w, h, "explicit resampling=4 1025x513 (multi-group)");
}

/// `with_already_downsampled(true)`: the caller supplies coded-size
/// pixels and the header advertises `coded × N` (the only size the
/// encoder can know). Pins that the streaming path honours the flag
/// exactly like the one-shot path (it used to downsample such input a
/// second time) — byte-identical output on a multi-group cell.
#[test]
fn issue_101_already_downsampled_streaming_matches_one_shot() {
    let (w, h) = (300u32, 200u32); // coded size; header advertises 600×400
    let cfg = LossyConfig::new(2.0)
        .with_effort(5)
        .with_resampling(2)
        .with_already_downsampled(true);
    let px = fixture(w, h, 3);
    let one_shot = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(&px)
        .expect("one-shot already_downsampled");
    let mut enc = cfg
        .encoder(w, h, PixelLayout::Rgb8)
        .expect("streaming encoder");
    enc.push_rows(&px, h).expect("push_rows");
    let streaming = enc.finish().expect("finish");
    assert_eq!(
        parse_size_header(&one_shot),
        (2 * w, 2 * h),
        "one-shot header = coded × 2"
    );
    assert_eq!(
        parse_size_header(&streaming),
        (2 * w, 2 * h),
        "streaming header = coded × 2"
    );
    assert_eq!(
        streaming, one_shot,
        "streaming must honour already_downsampled exactly like one-shot"
    );
    let (rw, rh, _) = decode_jxl_rs_rgba8(&streaming);
    assert_eq!((rw, rh), (2 * w, 2 * h), "jxl-rs decodes to coded × 2");
}

/// Controls: even dimensions at d=10 with the rule opted in (unaffected
/// before and after) and odd dimensions just below the auto threshold.
#[test]
fn issue_101_controls_even_dims_and_below_threshold() {
    for &(w, h) in &[(64u32, 64u32), (66, 34)] {
        let px = fixture(w, h, 3);
        let bytes = LossyConfig::new(10.0)
            .with_effort(5)
            .with_auto_resampling(true)
            .encode_request(w, h, PixelLayout::Rgb8)
            .encode(&px)
            .expect("even-dim encode at d=10");
        assert_source_size(&bytes, w, h, &format!("even control {w}x{h} d=10"));
    }
    let (w, h) = (65u32, 33u32);
    let cfg = LossyConfig::new(9.9)
        .with_effort(5)
        .with_auto_resampling(true);
    assert_eq!(cfg.effective_resampling(), 1, "no auto-resample below d=10");
    let px = fixture(w, h, 3);
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(&px)
        .expect("odd-dim encode at d=9.9");
    assert_source_size(&bytes, w, h, "below-threshold control 65x33 d=9.9");
}

/// Reference-decoder leg: djxl (libjxl) must decode the issue's exact
/// cell to a 513×769 image. `#[ignore]` because CI runners carry no
/// libjxl build — the skip decision is the caller's (`--include-ignored`
/// locally, where `~/work/jxl-efforts/libjxl/build/tools/djxl` or a
/// PATH `djxl` exists); the test itself never skips.
#[test]
#[ignore = "requires djxl (libjxl); run with --include-ignored"]
fn issue_101_auto_resample_odd_dims_decodes_via_djxl_to_source_size() {
    use std::process::Command;
    // First candidate that actually runs (`--version` exits 0): an
    // explicit `DJXL_PATH`, then a PATH `djxl`, then the local libjxl
    // build. A candidate that exists but cannot load its shared
    // libraries is skipped, not silently trusted.
    let candidates: Vec<String> = std::env::var("DJXL_PATH")
        .ok()
        .into_iter()
        .chain([
            "djxl".to_string(),
            "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl".to_string(),
        ])
        .collect();
    let djxl_path = candidates
        .iter()
        .find(|c| {
            Command::new(c)
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
        })
        .cloned()
        .unwrap_or_else(|| panic!("no working djxl among {candidates:?}"));
    let (w, h) = (513u32, 769u32);
    let px = fixture(w, h, 3);
    let bytes = LossyConfig::new(10.0)
        .with_effort(5)
        .with_auto_resampling(true)
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(&px)
        .expect("encode 513x769 d=10");
    let dir = std::env::temp_dir();
    let jxl_path = dir.join("jxl_encoder_issue101_513x769_d10.jxl");
    let ppm_path = dir.join("jxl_encoder_issue101_513x769_d10.ppm");
    std::fs::write(&jxl_path, &bytes).expect("write tmp jxl");
    let out = Command::new(&djxl_path)
        .arg(&jxl_path)
        .arg(&ppm_path)
        .output()
        .unwrap_or_else(|e| panic!("run {djxl_path}: {e}"));
    assert!(
        out.status.success(),
        "djxl rejected the bitstream: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let decoded = image::open(&ppm_path).expect("read djxl PPM output");
    let _ = std::fs::remove_file(&jxl_path);
    let _ = std::fs::remove_file(&ppm_path);
    assert_eq!(
        (decoded.width(), decoded.height()),
        (w, h),
        "djxl decoded dims must equal the source size (issue #101)"
    );
}
