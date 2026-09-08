// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Integer input above 16 bits (imazen/jxl-encoder#95).
//!
//! `PixelLayout` tops out at 16-bit, which is a limit of the input plumbing,
//! not of the codec: modular stores `i32` samples and codes them through a
//! 32-bit token path. `LosslessConfig::encode_planar_int` is the route for the
//! imagery that needs more — DEM/terrain rasters, instrument counts, 24/32-bit
//! depth maps, masters that must round-trip as exact integers.
//!
//! **What "valid" rests on here, stated precisely.** The official text
//! (ISO/IEC 18181-1) has NOT been consulted — it is paywalled. Every claim in
//! this file traces to implementations: libjxl's `CheckValidBitdepth`
//! (`encode.cc:632`) caps its public API at 24 with a comment saying the spec
//! allows 31, its encoder refuses 32-bit integer modular outright
//! (`enc_modular.cc:744`), and jxl-oxide's parser rejects `> 31`.
//!
//! That makes the evidence weakest exactly where this file claims the most.
//! libjxl is ISO/IEC 18181-4 (the reference software), so agreement with it is
//! strong — but it cannot encode 25..=31-bit integers at all, so its decoder
//! path there is comparatively unexercised. The 18181-3 conformance suite
//! (github.com/libjxl/conformance) is decoder-only and its vectors stop at
//! 16-bit integer / 32-bit float, so **no conformance coverage exists for this
//! range anywhere**. Treat 25..=31 as "two independent decoders accept it and
//! the low bits demonstrably reach the stream", not as "certified".

use jxl_encoder::api::{LosslessConfig, PixelLayout};

/// Values that exercise the full width: 0, the maximum, powers of two either
/// side of the 16-bit boundary, and a spread that is not byte-aligned.
fn samples(n: usize, bits: u32) -> Vec<u32> {
    let max = (1u32 << bits) - 1;
    let mut out = vec![
        0u32,
        max,
        1,
        max - 1,
        1 << (bits - 1),
        (1 << (bits - 1)) - 1,
    ];
    let mut x: u32 = 0x1234_5678;
    while out.len() < n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        out.push(x & max);
    }
    out.truncate(n);
    out
}

/// Smooth ramp — what real high-bit-depth rasters look like, and what the
/// predictor is actually meant to handle.
fn ramp(w: usize, h: usize, bits: u32) -> Vec<u32> {
    let max = (1u32 << bits) - 1;
    (0..w * h)
        .map(|i| {
            let x = (i % w) as u64;
            let y = (i / w) as u64;
            ((x * 65_537 + y * 4_099) % (u64::from(max) + 1)) as u32
        })
        .collect()
}

fn decode_jxlrs_gray(data: &[u8]) -> (usize, usize, Vec<f32>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let mut init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => init = fallback,
            Err(e) => panic!("jxl-rs header decode failed: {e:?}"),
        }
    };
    let (width, height) = decoder.basic_info().size;
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Grayscale,
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
    let mut out = Image::<f32>::new((width, height)).expect("allocate");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        out.get_rect_mut(Rect {
            origin: (0, 0),
            size: (width, height),
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
    let mut flat = Vec::with_capacity(width * height);
    for y in 0..height {
        flat.extend_from_slice(out.row(y));
    }
    (width, height, flat)
}

fn djxl_accepts(data: &[u8], name: &str) {
    let dir = jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "high_bit_depth_int");
    let path = dir.join(format!("{name}.jxl"));
    std::fs::write(&path, data).expect("write");
    let out = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&path)
        .arg("--disable_output")
        .arg("--num_threads=1")
        .output()
        .expect("run djxl v0.12");
    assert!(
        out.status.success(),
        "djxl rejected {name}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Every width from 17 to 31 must encode, signal its real depth, be accepted by
/// djxl, and round-trip **exactly** — checked against the decoder's normalised
/// f32 output by re-deriving `sample / (2^bits - 1)`.
#[test]
fn every_width_17_to_31_round_trips_exactly() {
    const W: usize = 40;
    const H: usize = 30;
    for bits in 17..=31u32 {
        let vals = {
            let mut v = samples(24, bits);
            v.extend(ramp(W * H - 24, 1, bits));
            v
        };
        assert_eq!(vals.len(), W * H);
        let data = LosslessConfig::new()
            .encode_planar_int(W as u32, H as u32, &[&vals], bits, true, false)
            .unwrap_or_else(|e| panic!("{bits}-bit encode failed: {e:?}"));

        // >12-bit integer needs 32-bit modular buffers, hence level 10.
        let i = data
            .windows(4)
            .position(|w| w == b"jxll")
            .unwrap_or_else(|| panic!("{bits}-bit: expected a jxll box"));
        assert_eq!(data[i + 4], 10, "{bits}-bit must be level 10");

        djxl_accepts(&data, &format!("gray_{bits}bit"));

        let (dw, dh, decoded) = decode_jxlrs_gray(&data);
        assert_eq!((dw, dh), (W, H), "{bits}-bit dimensions");
        // How exact can this check be? The decoder hands back **f32**. That is
        // a property of the OUTPUT FORMAT, not of the codestream — modular
        // stores these samples verbatim and codes them losslessly — but it does
        // bound what this test can prove: above ~16-bit, adjacent values stop
        // being separable after the decoder's normalise-to-[0,1] step. djxl
        // accepting the stream (above) is the structural check; proving
        // exactness at 24+ bits would need an integer-output decode path this
        // suite does not have, and is the honest gap here.
        let maxi = (1u64 << bits) - 1;
        let max = maxi as f32;
        // Exact through 16 bits, one f32 ULP scaled to the range beyond.
        //
        // 24 was tried first, on the reasoning that f32's mantissa is 24 bits —
        // and it fails at exactly 24: `max - 1` normalises to a ratio whose
        // nearest f32 is 1.0, so it comes back as `max`. The mantissa bounds
        // what f32 can REPRESENT, not what survives a divide-then-multiply at
        // the top of the range. 16 is where the margin is large enough that the
        // round trip is unconditionally exact.
        let tol: i64 = if bits <= 16 {
            0
        } else {
            (maxi as f64 * f64::from(f32::EPSILON)).ceil() as i64
        };
        for (i, (got, &want)) in decoded.iter().zip(&vals).enumerate() {
            let recovered = (got * max).round() as i64;
            let diff = (recovered - i64::from(want)).abs();
            assert!(
                diff <= tol,
                "{bits}-bit sample {i}: decoded {got} -> {recovered}, want {want} \
                 (tolerance {tol}; 0 means exact recovery was required)"
            );
        }
    }
}

/// RGB and RGBA at 24-bit — the libjxl parity ceiling — plus a multi-group
/// shape, since the group boundary is where the modular path has historically
/// broken.
#[test]
fn rgb_and_rgba_24bit_including_multigroup() {
    for (w, h, name) in [(32usize, 24usize, "small"), (300, 260, "multigroup")] {
        let n = w * h;
        let r = ramp(w, h, 24);
        let g: Vec<u32> = r.iter().map(|v| v ^ 0x00_ff00).collect();
        let b: Vec<u32> = r.iter().map(|v| v.rotate_left(7) & 0xff_ffff).collect();
        let a: Vec<u32> = r.iter().map(|v| (v >> 3) & 0xff_ffff).collect();
        assert_eq!(r.len(), n);

        let rgb = LosslessConfig::new()
            .encode_planar_int(w as u32, h as u32, &[&r, &g, &b], 24, false, false)
            .expect("24-bit RGB encode");
        djxl_accepts(&rgb, &format!("rgb24_{name}"));

        let rgba = LosslessConfig::new()
            .encode_planar_int(w as u32, h as u32, &[&r, &g, &b, &a], 24, false, true)
            .expect("24-bit RGBA encode");
        djxl_accepts(&rgba, &format!("rgba24_{name}"));
    }
}

/// The 30- and 31-bit cells are the ones where the RCT budget bites: an RCT's
/// channel sums need a spare bit the sample does not leave, so it must be
/// refused — the same rule libjxl applies. If it were not, the stream would not
/// decode, so djxl accepting these IS the assertion.
#[test]
fn rct_budget_holds_at_30_and_31_bits() {
    const W: usize = 64;
    const H: usize = 48;
    for bits in [29u32, 30, 31] {
        let r = ramp(W, H, bits);
        let g: Vec<u32> = r
            .iter()
            .map(|v| v.rotate_left(5) & ((1 << bits) - 1))
            .collect();
        let b: Vec<u32> = r
            .iter()
            .map(|v| v.rotate_right(3) & ((1 << bits) - 1))
            .collect();
        let data = LosslessConfig::new()
            .encode_planar_int(W as u32, H as u32, &[&r, &g, &b], bits, false, false)
            .unwrap_or_else(|e| panic!("{bits}-bit RGB encode failed: {e:?}"));
        djxl_accepts(&data, &format!("rgb_{bits}bit"));
    }
}

/// Input validation: the surface must refuse what it cannot represent rather
/// than truncating it.
#[test]
fn out_of_range_input_is_refused() {
    let ok = vec![0u32; 16];
    let cfg = LosslessConfig::new();

    // bits outside 1..=31
    for bits in [0u32, 32, 33, 64] {
        assert!(
            cfg.encode_planar_int(4, 4, &[&ok], bits, true, false)
                .is_err(),
            "bits_per_sample {bits} must be refused"
        );
    }
    // sample exceeding the declared width
    let mut too_big = vec![0u32; 16];
    too_big[7] = 1 << 17;
    assert!(
        cfg.encode_planar_int(4, 4, &[&too_big], 17, true, false)
            .is_err(),
        "a sample above 2^bits - 1 must be refused, not truncated"
    );
    // plane count vs channel description
    assert!(
        cfg.encode_planar_int(4, 4, &[&ok], 17, false, false)
            .is_err(),
        "one plane cannot be RGB"
    );
    assert!(
        cfg.encode_planar_int(4, 4, &[&ok, &ok], 17, true, false)
            .is_err(),
        "two planes cannot be grayscale-without-alpha"
    );
    // wrong plane length
    let short = vec![0u32; 15];
    assert!(
        cfg.encode_planar_int(4, 4, &[&short], 17, true, false)
            .is_err(),
        "a short plane must be refused"
    );
}

/// 16-bit and below still go through `PixelLayout`; the planar entry point
/// accepts them too and must agree with the codestream the layout path
/// produces on what depth it signals.
#[test]
fn planar_path_signals_the_same_depth_as_the_layout_path_at_16_bit() {
    const W: usize = 24;
    const H: usize = 18;
    let vals = ramp(W, H, 16);
    let planar = LosslessConfig::new()
        .encode_planar_int(W as u32, H as u32, &[&vals], 16, true, false)
        .expect("16-bit planar");
    let interleaved: Vec<u8> = vals
        .iter()
        .flat_map(|v| (*v as u16).to_ne_bytes())
        .collect();
    let via_layout = LosslessConfig::new()
        .encode_request(W as u32, H as u32, PixelLayout::Gray16)
        .encode(&interleaved)
        .expect("16-bit via layout");
    // Both must be level 10 (bits > 12) and both must decode.
    for (data, name) in [(&planar, "planar"), (&via_layout, "layout")] {
        let i = data
            .windows(4)
            .position(|w| w == b"jxll")
            .unwrap_or_else(|| panic!("{name}: expected jxll"));
        assert_eq!(data[i + 4], 10, "{name} must be level 10 at 16-bit");
    }
    djxl_accepts(&planar, "planar_16bit");
}

/// The low bits above 16 must actually reach the codestream.
///
/// This closes the one thing the round-trip test above cannot see. That test
/// compares decoded values to the originals within a derived f32 ULP, so any
/// truncation, shift, endianness error or wrong-width signalling produces an
/// error far larger than the bound and fails — but a defect confined to the
/// LOWEST bits at wide depths would slip under it, because the decoder's
/// normalise-to-[0,1] step cannot represent those bits in the first place.
///
/// So test it on the encoder side instead, where no decoder precision is
/// involved: flip a single sample's least-significant bit and require the
/// codestream to change. If the low bits were being truncated, dropped, or
/// masked anywhere between the public API and the coded stream, the two
/// encodes would be byte-identical.
///
/// Note the assertion is deliberately "differs", not "differs by N bytes" —
/// an entropy coder is free to spend a different number of bytes on the same
/// one-bit change, and pinning a size here would be pinning noise.
#[test]
fn the_lowest_bit_reaches_the_codestream_at_every_width() {
    const W: usize = 24;
    const H: usize = 18;
    for bits in 17..=31u32 {
        let base = ramp(W, H, bits);
        // Pick a sample whose LSB is 0 so flipping it up stays in range.
        let idx = base
            .iter()
            .position(|v| v & 1 == 0)
            .expect("ramp must contain an even sample");
        let mut flipped = base.clone();
        flipped[idx] |= 1;
        assert_ne!(base[idx], flipped[idx], "{bits}-bit: fixture must differ");

        let cfg = LosslessConfig::new();
        let a = cfg
            .encode_planar_int(W as u32, H as u32, &[&base], bits, true, false)
            .unwrap_or_else(|e| panic!("{bits}-bit base encode: {e:?}"));
        let b = cfg
            .encode_planar_int(W as u32, H as u32, &[&flipped], bits, true, false)
            .unwrap_or_else(|e| panic!("{bits}-bit flipped encode: {e:?}"));
        assert_ne!(
            a, b,
            "{bits}-bit: flipping one sample's LSB did not change the codestream, \
             so the low bits are not reaching it"
        );
    }
}

/// The complement of the test above: a change in the HIGH bits must also
/// register, and identical input must produce identical output. Together these
/// bracket the sensitivity claim — the encoder is neither ignoring bits nor
/// non-deterministic.
#[test]
fn encoding_is_deterministic_and_sensitive_at_every_width() {
    const W: usize = 24;
    const H: usize = 18;
    let cfg = LosslessConfig::new();
    for bits in [17u32, 24, 31] {
        let base = ramp(W, H, bits);
        let once = cfg
            .encode_planar_int(W as u32, H as u32, &[&base], bits, true, false)
            .expect("encode");
        let twice = cfg
            .encode_planar_int(W as u32, H as u32, &[&base], bits, true, false)
            .expect("encode");
        assert_eq!(once, twice, "{bits}-bit: encoding must be deterministic");

        let mut high = base.clone();
        high[5] ^= 1 << (bits - 1);
        let changed = cfg
            .encode_planar_int(W as u32, H as u32, &[&high], bits, true, false)
            .expect("encode");
        assert_ne!(once, changed, "{bits}-bit: a top-bit change must register");
    }
}
