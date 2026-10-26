// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Lossy encodes code alpha losslessly, so 16-bit alpha must come back with
//! all 16 bits. It used to be quantized to 8 bits on the way in (found by the
//! conformance sweep: 65534 decoded as 65535, 3 as 0).

use jxl_encoder::{LossyConfig, PixelLayout};

/// Full jxl-rs decode; returns the last extra channel (alpha) scaled to
/// 0..=65535.
fn jxl_rs_alpha16(data: &[u8], label: &str) -> Vec<u16> {
    use jxl::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let mut init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => init = fallback,
            Err(e) => panic!("jxl-rs header {label}: {e:?}"),
        }
    };
    let (w, h) = decoder.basic_info().size;
    let format = decoder.current_pixel_format().clone();
    let nc = format.color_type.samples_per_pixel();
    let ne = format.extra_channel_format.len();
    assert!(ne >= 1, "{label}: no extra channel");
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![Some(JxlDataFormat::f32()); ne],
    });
    let mut frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame info {label}: {e:?}"),
        }
    };
    let mut color = Image::<f32>::new((w * nc, h)).unwrap();
    let mut extra: Vec<Image<f32>> = (0..ne)
        .map(|_| Image::<f32>::new((w, h)).unwrap())
        .collect();
    let mut bufs = vec![JxlOutputBuffer::from_image_rect_mut(
        color
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (w * nc, h),
            })
            .into_raw(),
    )];
    for e in &mut extra {
        bufs.push(JxlOutputBuffer::from_image_rect_mut(
            e.get_rect_mut(Rect {
                origin: (0, 0),
                size: (w, h),
            })
            .into_raw(),
        ));
    }
    loop {
        match frame.process(&mut input, &mut bufs) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => frame = fallback,
            Err(e) => panic!("jxl-rs decode {label}: {e:?}"),
        }
    }
    drop(bufs);
    let alpha = &extra[ne - 1];
    (0..h)
        .flat_map(|y| {
            alpha.row(y)[..w]
                .iter()
                .map(|v| (v * 65535.0).round() as u16)
        })
        .collect()
}

/// djxl v0.12 to a 16-bit PNG; returns its alpha channel.
fn djxl_alpha16(data: &[u8], label: &str, gray: bool) -> Vec<u16> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/lossy-alpha16");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join(format!("{label}.jxl"));
    let output = dir.join(format!("{label}.png"));
    std::fs::write(&input, data).unwrap();
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(&output)
        .arg("--num_threads=1")
        .output()
        .expect("run djxl v0.12");
    assert!(
        result.status.success(),
        "djxl {label}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let img = image::open(&output).expect("read djxl PNG");
    if gray {
        img.to_luma_alpha16().pixels().map(|p| p.0[1]).collect()
    } else {
        img.to_rgba16().pixels().map(|p| p.0[3]).collect()
    }
}

#[test]
fn lossy_16bit_alpha_keeps_all_bits() {
    for (w, h) in [(64usize, 32usize), (300, 260)] {
        let alpha: Vec<u16> = (0..w * h)
            .map(|i| ((i * 977 + i / w * 31) % 65536) as u16)
            .collect();
        for (layout, ch, gray) in [
            (PixelLayout::Rgba16, 4usize, false),
            (PixelLayout::GrayAlpha16, 2, true),
        ] {
            let mut px = Vec::with_capacity(w * h * ch * 2);
            for (i, a) in alpha.iter().enumerate() {
                for c in 0..ch - 1 {
                    let v = ((i % w) * 211 + (i / w) * 97 + c * 4099) % 65536;
                    px.extend_from_slice(&(v as u16).to_ne_bytes());
                }
                px.extend_from_slice(&a.to_ne_bytes());
            }
            for effort in [3u8, 7] {
                let label = format!("{layout:?}-{w}x{h}-e{effort}");
                let bytes = LossyConfig::new(1.0)
                    .with_effort(effort)
                    .encode(&px, w as u32, h as u32, layout)
                    .expect("lossy 16-bit alpha encode");
                assert_eq!(jxl_rs_alpha16(&bytes, &label), alpha, "jxl-rs {label}");
                assert_eq!(djxl_alpha16(&bytes, &label, gray), alpha, "djxl {label}");
            }
        }
    }
}

/// SimplifyInvisible may rewrite the colour of pixels whose alpha is exactly
/// 0. It used to decide from the 8-bit alpha copy, so a 16-bit alpha of
/// 100/65535 counted as invisible and its colour was smeared from the
/// neighbours (found by the conformance sweep).
#[test]
fn nearly_transparent_16bit_colour_is_kept() {
    let (w, h) = (64usize, 32usize);
    let mut px = Vec::with_capacity(w * h * 8);
    let mut red_square = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (rgb, a): ([u16; 3], u16) = if (40..56).contains(&x) && (8..24).contains(&y) {
                red_square.push(y * w + x);
                ([65535, 0, 0], 100)
            } else if x < 24 {
                (
                    [((x * 977) % 65536) as u16, 7, ((y * 131) % 65536) as u16],
                    0,
                )
            } else {
                ([0, 0, 65535], 65535)
            };
            for v in rgb.iter().chain(core::iter::once(&a)) {
                px.extend_from_slice(&v.to_ne_bytes());
            }
        }
    }
    let label = "nearly-transparent-rgba16";
    let bytes = LossyConfig::new(1.0)
        .with_effort(7)
        .encode(&px, w as u32, h as u32, PixelLayout::Rgba16)
        .expect("encode");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/lossy-alpha16");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join(format!("{label}.jxl"));
    let output = dir.join(format!("{label}.png"));
    std::fs::write(&input, &bytes).unwrap();
    let status = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(&output)
        .arg("--num_threads=1")
        .status()
        .expect("run djxl");
    assert!(status.success(), "djxl failed");
    let img = image::open(&output).unwrap().to_rgba16();
    let mean_red_err: f64 = red_square
        .iter()
        .map(|&i| {
            let p = img.as_raw()[i * 4..i * 4 + 3].to_vec();
            ((65535 - p[0] as i64).abs() + p[1] as i64 + p[2] as i64) as f64 / (3.0 * 65535.0)
        })
        .sum::<f64>()
        / red_square.len() as f64;
    assert!(
        mean_red_err < 0.1,
        "colour under alpha 100/65535 was rewritten: mean error {mean_red_err:.3}"
    );
}

/// Full jxl-rs decode of an RGBA file: interleaved RGB and alpha, f32 in the
/// file's colour space as the decoder returns them.
fn jxl_rs_rgba_f32(data: &[u8], label: &str) -> (Vec<f32>, Vec<f32>) {
    use jxl::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};
    let mut input = data;
    let mut init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => init = fallback,
            Err(e) => panic!("jxl-rs header {label}: {e:?}"),
        }
    };
    let (w, h) = decoder.basic_info().size;
    let format = decoder.current_pixel_format().clone();
    assert_eq!(format.color_type.samples_per_pixel(), 3, "{label}");
    let ne = format.extra_channel_format.len();
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![Some(JxlDataFormat::f32()); ne],
    });
    let mut frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame info {label}: {e:?}"),
        }
    };
    let mut color = Image::<f32>::new((w * 3, h)).unwrap();
    let mut alpha = Image::<f32>::new((w, h)).unwrap();
    let mut bufs = vec![
        JxlOutputBuffer::from_image_rect_mut(
            color
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (w * 3, h),
                })
                .into_raw(),
        ),
        JxlOutputBuffer::from_image_rect_mut(
            alpha
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (w, h),
                })
                .into_raw(),
        ),
    ];
    loop {
        match frame.process(&mut input, &mut bufs) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => frame = fallback,
            Err(e) => panic!("jxl-rs decode {label}: {e:?}"),
        }
    }
    drop(bufs);
    let rgb = (0..h)
        .flat_map(|y| color.row(y)[..w * 3].to_vec())
        .collect();
    let a = (0..h).flat_map(|y| alpha.row(y)[..w].to_vec()).collect();
    (rgb, a)
}

/// djxl v0.12 to PAM (which djxl writes without unpremultiplying): RGB and
/// alpha scaled to 0..=1.
fn djxl_rgba_pam(data: &[u8], label: &str) -> (Vec<f32>, Vec<f32>) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/lossy-premul");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join(format!("{label}.jxl"));
    let output = dir.join(format!("{label}.pam"));
    std::fs::write(&input, data).unwrap();
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(&output)
        .arg("--num_threads=1")
        .output()
        .expect("run djxl v0.12");
    assert!(result.status.success(), "djxl {label}");
    let b = std::fs::read(&output).unwrap();
    let end = b.windows(7).position(|w| w == b"ENDHDR\n").unwrap() + 7;
    let header = String::from_utf8_lossy(&b[..end]).into_owned();
    let field = |k: &str| -> usize {
        header
            .split(k)
            .nth(1)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    let (depth, maxval) = (field("DEPTH "), field("MAXVAL "));
    assert_eq!(depth, 4, "{label}: PAM depth");
    let n = if maxval > 255 { 2 } else { 1 };
    let samples: Vec<f32> = b[end..]
        .chunks(n)
        .map(|c| c.iter().fold(0u32, |v, &x| v << 8 | x as u32) as f32 / maxval as f32)
        .collect();
    let rgb = samples.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    let a = samples.chunks(4).map(|p| p[3]).collect();
    (rgb, a)
}

/// Premultiplied (associated) alpha is coded as given: libjxl never
/// unpremultiplies, and decoders return the samples premultiplied. The lossy
/// path used to divide colour by alpha before coding while still flagging the
/// alpha as associated, so decoders returned straight colour labelled
/// premultiplied (a half-transparent pixel came back 40 % too bright, a
/// nearly transparent one at full brightness: mean error 0.05 against 0.002
/// now). Decoded colour must match the source's premultiplied samples and
/// stay at or below its alpha.
#[test]
fn premultiplied_alpha_decodes_premultiplied() {
    let source = image::load_from_memory(include_bytes!("../images/frymire-srgb.png"))
        .unwrap()
        .to_rgb8();
    for (layout, w, h) in [
        (PixelLayout::Rgba8, 64u32, 32u32),
        (PixelLayout::Rgba16, 64, 32),
        (PixelLayout::Rgba8, 300, 260),
    ] {
        // Straight colour from the photo, alpha a horizontal ramp from 1/64
        // to 1, then premultiplied in the sample domain.
        let mut px = Vec::new();
        let mut want_rgb = Vec::new();
        let mut want_a = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let p = source.get_pixel(x % source.width(), y % source.height()).0;
                let a = (1 + x * 63 / (w - 1)) as f32 / 64.0;
                let premul = p.map(|c| c as f32 / 255.0 * a);
                want_rgb.extend_from_slice(&premul);
                let max = if layout == PixelLayout::Rgba8 {
                    255.0
                } else {
                    65535.0
                };
                want_a.push((a * max).round() / max);
                if layout == PixelLayout::Rgba8 {
                    px.extend(premul.map(|v| (v * 255.0).round() as u8));
                    px.push((a * 255.0).round() as u8);
                } else {
                    for v in premul.iter().chain(core::iter::once(&a)) {
                        px.extend_from_slice(&((v * 65535.0).round() as u16).to_ne_bytes());
                    }
                }
            }
        }
        let label = format!("premul-{layout:?}-{w}x{h}");
        let bytes = LossyConfig::new(1.0)
            .with_effort(7)
            .encode_request(w, h, layout)
            .with_premultiplied_alpha(true)
            .encode(&px)
            .expect("encode");
        for (who, (rgb, a)) in [
            ("djxl", djxl_rgba_pam(&bytes, &label)),
            ("jxl-rs", jxl_rs_rgba_f32(&bytes, &label)),
        ] {
            assert_eq!(rgb.len(), want_rgb.len(), "{label} {who}");
            let mean_err = rgb
                .iter()
                .zip(&want_rgb)
                .map(|(d, s)| (d - s).abs() as f64)
                .sum::<f64>()
                / rgb.len() as f64;
            let max_excess = rgb
                .chunks(3)
                .zip(&a)
                .map(|(c, a)| c.iter().fold(f32::MIN, |m, &v| m.max(v - a)))
                .fold(f32::MIN, f32::max);
            eprintln!(
                "{label} {who}: mean error {mean_err:.4}, max colour - alpha {max_excess:.3}"
            );
            let over_alpha = rgb
                .chunks(3)
                .zip(&a)
                .filter(|(c, a)| c.iter().any(|&v| v > *a + 0.05))
                .count();
            assert!(
                mean_err < 0.02,
                "{label} {who}: premultiplied colour mean error {mean_err:.4}"
            );
            // Lossy ringing on frymire's hard edges lifts a few pixels past
            // alpha (58 of 78,000 at 300x260, by up to 0.114); the old path
            // put the colour of every low-alpha pixel there.
            assert!(
                over_alpha * 200 < a.len(),
                "{label} {who}: {over_alpha} of {} pixels decode brighter than their alpha",
                a.len()
            );
            let alpha_err = a
                .iter()
                .zip(&want_a)
                .map(|(d, s)| (d - s).abs())
                .fold(0f32, f32::max);
            assert!(alpha_err < 1e-3, "{label} {who}: alpha off by {alpha_err}");
        }
    }
}
