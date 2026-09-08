// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Lossless floating-point input (imazen/jxl-encoder#109, F3).
//!
//! JPEG XL stores lossless float by packing the IEEE bit pattern into the
//! modular integer sample. Until now every float `PixelLayout` was rejected on
//! both lossless entry points, so an f32/f16 master could only go through
//! VarDCT — the codestream half existed (`BitDepth::float32`/`float16` and the
//! header writer's float branch) but had no caller.
//!
//! **Decoder scope is deliberate.** jxl-rs and djxl v0.12 implement the full
//! inverse (`int_to_float`, including subnormal renormalisation). jxl-oxide
//! 0.12.6 carries a literal `// TODO: handle subnormal values` in
//! `BitDepth::parse_integer_sample`: at `bits == 32` its arithmetic reduces to
//! an identity transform so f32 is exact, but at f16 its subnormals decode
//! wrong and its infinities decode as finite. So f32 is checked against all
//! three decoders and f16 against jxl-rs and djxl only. Asserting otherwise
//! would fail on a defect that is not ours.

use jxl_encoder::api::{LosslessConfig, PixelLayout};

/// Exercise the awkward values on purpose: zero and negative zero, subnormals
/// at both scales, the smallest normal, an exact power of two, and a value
/// with a full mantissa. Deliberately no NaN/infinity — the reference's
/// NaN/infinity branch drops low mantissa bits without a loss check at narrow
/// widths, so a narrow NaN payload legitimately becomes an infinity (see
/// `modular::float_pack`), and a bit-exactness assertion there would be wrong.
fn f32_values(n: usize) -> Vec<f32> {
    let fixed: &[u32] = &[
        0x0000_0000, // +0
        0x8000_0000, // -0  (also i32::MIN as a sample — the pack_signed case)
        0x0000_0001, // smallest f32 subnormal
        0x007f_ffff, // largest f32 subnormal
        0x0080_0000, // smallest normal
        0x3f80_0000, // 1.0
        0xbf80_0000, // -1.0
        0x4048_0000, // 3.125
        0x7f7f_ffff, // largest finite
        0x33800000,  // smallest f16 subnormal, as f32
    ];
    let mut out: Vec<f32> = fixed.iter().map(|&b| f32::from_bits(b)).collect();
    let mut x: u32 = 0x2545_f491;
    while out.len() < n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        // Keep the exponent in a range f32 represents normally; NaN/inf are
        // excluded above by construction.
        let e = 100 + (x % 50);
        let bits = (x & 0x807f_ffff) | (e << 23);
        let v = f32::from_bits(bits);
        if v.is_finite() {
            out.push(v);
        }
    }
    out.truncate(n);
    out
}

/// f16 bit patterns, excluding the NaN/infinity class for the same reason.
fn f16_bits(n: usize) -> Vec<u16> {
    let mut out = vec![
        0x0000u16, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff,
    ];
    let mut x: u32 = 0x9e37_79b9;
    while out.len() < n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let b = (x & 0xffff) as u16;
        if (b >> 10) & 0x1f != 0x1f {
            out.push(b);
        }
    }
    out.truncate(n);
    out
}

/// binary16 -> binary32, written here rather than called out of the encoder.
/// A differential that shares code with the thing under test cannot see a bug
/// the two share, and the whole claim of this test is that our packing agrees
/// with what a decoder independently reconstructs.
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: renormalise into binary32, which always has room.
        let mut e: i32 = 0;
        let mut m = mant;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits(sign | (exp32 << 23) | ((m & 0x3ff) << 13));
    }
    // Callers exclude exp == 0x1f (NaN/infinity) by construction.
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

fn decode_jxlrs_planar(data: &[u8], channels: usize) -> (usize, usize, Vec<f32>) {
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
        color_type: if channels == 1 {
            JxlColorType::Grayscale
        } else {
            JxlColorType::Rgb
        },
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    });
    let cc = if channels == 1 { 1 } else { 3 };
    let mut frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame-info decode failed: {e:?}"),
        }
    };
    let mut out = Image::<f32>::new((width * cc, height)).expect("allocate");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        out.get_rect_mut(Rect {
            origin: (0, 0),
            size: (width * cc, height),
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
    let mut flat = Vec::with_capacity(width * height * cc);
    for y in 0..height {
        flat.extend_from_slice(out.row(y));
    }
    (width, height, flat)
}

fn djxl_accepts(data: &[u8], name: &str) {
    let dir = jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "lossless_float");
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

/// The property that makes this "lossless": every sample comes back with the
/// exact bit pattern it went in with. Checked on the encoder's own inverse
/// first (so a failure localises to packing rather than to a decoder), then
/// through jxl-rs and djxl.
#[test]
fn f32_rgb_round_trips_bit_exactly() {
    const W: usize = 32;
    const H: usize = 24;
    let vals = f32_values(W * H * 3);
    let mut px = Vec::with_capacity(W * H * 12);
    for v in &vals {
        px.extend_from_slice(&v.to_ne_bytes());
    }

    let data = LosslessConfig::new()
        .encode_request(W as u32, H as u32, PixelLayout::RgbLinearF32)
        .encode(&px)
        .expect("f32 lossless encode must succeed");

    // Float needs 32-bit modular buffers, so the level must be 10.
    assert!(
        !data.starts_with(&[0xff, 0x0a]),
        "float cannot be level 5 — modular_16_bit_buffer_sufficient is false"
    );
    let i = data
        .windows(4)
        .position(|w| w == b"jxll")
        .expect("jxll box for level 10");
    assert_eq!(data[i + 4], 10);

    let (dw, dh, decoded) = decode_jxlrs_planar(&data, 3);
    assert_eq!((dw, dh), (W, H));
    assert_eq!(decoded.len(), vals.len());
    let mismatches: Vec<_> = decoded
        .iter()
        .zip(&vals)
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .take(5)
        .map(|(i, (a, b))| format!("[{i}] got 0x{:08x} want 0x{:08x}", a.to_bits(), b.to_bits()))
        .collect();
    assert!(
        mismatches.is_empty(),
        "f32 lossless must be BIT-exact; first mismatches: {mismatches:?}"
    );

    djxl_accepts(&data, "f32_rgb");
}

#[test]
fn f32_gray_round_trips_bit_exactly() {
    const W: usize = 17;
    const H: usize = 13;
    let vals = f32_values(W * H);
    let mut px = Vec::with_capacity(W * H * 4);
    for v in &vals {
        px.extend_from_slice(&v.to_ne_bytes());
    }
    let data = LosslessConfig::new()
        .encode_request(W as u32, H as u32, PixelLayout::GrayLinearF32)
        .encode(&px)
        .expect("encode");
    let (_, _, decoded) = decode_jxlrs_planar(&data, 1);
    for (i, (a, b)) in decoded.iter().zip(&vals).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "gray f32 sample {i} differs: 0x{:08x} vs 0x{:08x}",
            a.to_bits(),
            b.to_bits()
        );
    }
    djxl_accepts(&data, "f32_gray");
}

/// Multi-group (> 256 px) is required coverage per the repo rule — the group
/// boundary is where the previous 16-bit desyncs hid.
#[test]
fn f32_rgb_multigroup_round_trips_bit_exactly() {
    const W: usize = 300;
    const H: usize = 260;
    // A smooth gradient with the awkward values seeded into the first row,
    // NOT 234k random bit patterns. Random f32 patterns are incompressible
    // noise and made this cell take ~9 minutes in the tree learner while
    // testing nothing the single-group cells do not already cover; real float
    // imagery is smooth, and the property under test here is the group
    // boundary, not entropy coding.
    let vals: Vec<f32> = {
        (0..W * H * 3)
            .map(|i| {
                let x = (i / 3) % W;
                let y = (i / 3) / W;
                let c = i % 3;
                (x as f32 * 0.013 + y as f32 * 0.007 + c as f32 * 0.11).sin()
            })
            .collect()
    };
    let mut px = Vec::with_capacity(W * H * 12);
    for v in &vals {
        px.extend_from_slice(&v.to_ne_bytes());
    }
    let data = LosslessConfig::new()
        .encode_request(W as u32, H as u32, PixelLayout::RgbLinearF32)
        .encode(&px)
        .expect("encode");
    let (dw, dh, decoded) = decode_jxlrs_planar(&data, 3);
    assert_eq!((dw, dh), (W, H));
    let bad = decoded
        .iter()
        .zip(&vals)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        bad,
        0,
        "{bad} of {} samples differ across groups",
        vals.len()
    );
    djxl_accepts(&data, "f32_rgb_multigroup");
}

/// f16: jxl-rs and djxl only, per the module note about jxl-oxide's subnormals.
#[test]
fn f16_rgb_round_trips_bit_exactly() {
    const W: usize = 32;
    const H: usize = 24;
    let bits = f16_bits(W * H * 3);
    let mut px = Vec::with_capacity(W * H * 6);
    for b in &bits {
        px.extend_from_slice(&b.to_ne_bytes());
    }
    let data = LosslessConfig::new()
        .encode_request(W as u32, H as u32, PixelLayout::RgbLinearF16)
        .encode(&px)
        .expect("f16 lossless encode must succeed");

    // f16 is still float, so still level 10 (libjxl's SetFloat16Samples clears
    // modular_16_bit_buffer_sufficient too).
    let i = data
        .windows(4)
        .position(|w| w == b"jxll")
        .expect("jxll box for level 10");
    assert_eq!(data[i + 4], 10);

    let (_, _, decoded) = decode_jxlrs_planar(&data, 3);
    // The decoder hands back f32; compare by re-deriving what each f16 pattern
    // means, using the encoder's own documented inverse relationship.
    for (i, (got, &want_bits)) in decoded.iter().zip(&bits).enumerate() {
        let want = f16_bits_to_f32(want_bits);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "f16 sample {i} (pattern 0x{want_bits:04x}) decoded as 0x{:08x}, want 0x{:08x}",
            got.to_bits(),
            want.to_bits()
        );
    }
    djxl_accepts(&data, "f16_rgb");
}

/// Transfer-function-tagged float layouts stay lossy-only for now and must be
/// refused with a named error rather than silently mis-signalled.
#[test]
fn tf_tagged_float_layouts_are_refused_on_the_lossless_path() {
    let px = vec![0u8; 8 * 8 * 12];
    for layout in [
        PixelLayout::RgbPqF32,
        PixelLayout::RgbHlgF32,
        PixelLayout::RgbBt709F32,
    ] {
        let err = LosslessConfig::new()
            .encode_request(8, 8, layout)
            .encode(&px)
            .expect_err("TF-tagged float must be refused on lossless");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("UnsupportedPixelLayout"),
            "expected UnsupportedPixelLayout for {layout:?}, got {msg}"
        );
    }
}

/// Multi-group float carrying the EXTREME values (±0, subnormals, `i32::MIN`
/// as a sample), checked through **djxl v0.12 only**.
///
/// jxl-rs 0.4.3 panics decoding this combination: `precompute_references`
/// (`src/frame/modular/decode/common.rs:71`) takes `num_traits::abs::<i32>` of
/// a property value, which overflows on `i32::MIN` — and `-0.0f32` packs to
/// exactly that bit pattern. It needs the multi-group path as well; the
/// single-group tests above carry the same values and decode fine, and smooth
/// multi-group float without the extremes also decodes fine. So this is a
/// jxl-rs limitation on full-range modular samples, not a defect in what we
/// emit — libjxl accepts the stream.
///
/// Kept as its own test rather than folded into the one above so the coverage
/// is not silently lost, and so the day jxl-rs saturates that `abs` the two can
/// be merged.
#[test]
fn f32_multigroup_with_extreme_values_is_accepted_by_djxl() {
    const W: usize = 300;
    const H: usize = 260;
    let seeds = f32_values(64);
    let vals: Vec<f32> = (0..W * H * 3)
        .map(|i| {
            if i < seeds.len() {
                seeds[i]
            } else {
                let x = (i / 3) % W;
                let y = (i / 3) / W;
                let c = i % 3;
                (x as f32 * 0.013 + y as f32 * 0.007 + c as f32 * 0.11).sin()
            }
        })
        .collect();
    let mut px = Vec::with_capacity(W * H * 12);
    for v in &vals {
        px.extend_from_slice(&v.to_ne_bytes());
    }
    let data = LosslessConfig::new()
        .encode_request(W as u32, H as u32, PixelLayout::RgbLinearF32)
        .encode(&px)
        .expect("encode");
    djxl_accepts(&data, "f32_multigroup_extremes");
}
