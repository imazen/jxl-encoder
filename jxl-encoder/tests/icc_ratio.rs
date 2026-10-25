// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Issue #128: highly repetitive ICC profiles must stay decodable.
//!
//! libjxl's ICC reader rejects a stream as corrupt when, at any multiple of
//! 65536 decoded symbols, it has consumed fewer than 1/256 as many bytes.
//! LZ77 compresses a profile with a long zero or repeated tag payload far past
//! that, so the writer must fall back. Each profile here must decode in jxl-rs
//! and djxl v0.12, and both must return the profile byte for byte.
//!
//! Sizes stay below the pinned jxl-rs 0.4's 1 MiB cap on the encoded ICC
//! stream (`IccTooLarge`, a resource policy).

use jxl_encoder::{ImageMetadata, LosslessConfig, LossyConfig, PixelLayout};

/// A valid matrix/TRC RGB profile padded to `total` bytes by one private tag
/// whose payload is `fill` repeated.
fn padded_icc(total: usize, fill: &[u8]) -> Vec<u8> {
    fn s15(v: f64) -> [u8; 4] {
        ((v * 65536.0).round() as i32).to_be_bytes()
    }
    fn xyz(x: f64, y: f64, z: f64) -> Vec<u8> {
        let mut t = b"XYZ \0\0\0\0".to_vec();
        t.extend_from_slice(&s15(x));
        t.extend_from_slice(&s15(y));
        t.extend_from_slice(&s15(z));
        t
    }
    let curv: Vec<u8> = {
        let mut t = b"curv\0\0\0\0".to_vec();
        t.extend_from_slice(&1u32.to_be_bytes());
        t.extend_from_slice(&0x0233u16.to_be_bytes()); // 2.2 as u8Fixed8
        t.extend_from_slice(&[0, 0]);
        t
    };
    let desc: Vec<u8> = {
        let text = b"repetitive\0";
        let mut t = b"desc\0\0\0\0".to_vec();
        t.extend_from_slice(&(text.len() as u32).to_be_bytes());
        t.extend_from_slice(text);
        t.extend_from_slice(&[0u8; 11 + 67]);
        while !t.len().is_multiple_of(4) {
            t.push(0);
        }
        t
    };
    let tags: Vec<(&[u8; 4], Vec<u8>)> = vec![
        (b"desc", desc),
        (b"wtpt", xyz(0.9642, 1.0, 0.8249)),
        (b"rXYZ", xyz(0.4361, 0.2225, 0.0139)),
        (b"gXYZ", xyz(0.3851, 0.7169, 0.0971)),
        (b"bXYZ", xyz(0.1431, 0.0606, 0.7141)),
        (b"rTRC", curv.clone()),
        (b"gTRC", curv.clone()),
        (b"bTRC", curv),
    ];
    let data_start = 128 + 4 + 12 * (tags.len() + 1);
    let mut table = ((tags.len() + 1) as u32).to_be_bytes().to_vec();
    let mut body = Vec::new();
    for (sig, data) in &tags {
        table.extend_from_slice(*sig);
        table.extend_from_slice(&((data_start + body.len()) as u32).to_be_bytes());
        table.extend_from_slice(&(data.len() as u32).to_be_bytes());
        body.extend_from_slice(data);
    }
    let pad_len = total - data_start - body.len();
    table.extend_from_slice(b"zzzz");
    table.extend_from_slice(&((data_start + body.len()) as u32).to_be_bytes());
    table.extend_from_slice(&(pad_len as u32).to_be_bytes());
    body.extend_from_slice(b"zzzz\0\0\0\0");
    body.extend(fill.iter().copied().cycle().take(pad_len - 8));

    let mut icc = Vec::with_capacity(total);
    icc.extend_from_slice(&(total as u32).to_be_bytes());
    icc.extend_from_slice(&[0u8; 4]);
    icc.extend_from_slice(&0x0210_0000u32.to_be_bytes());
    icc.extend_from_slice(b"mntrRGB XYZ ");
    icc.extend_from_slice(&[0u8; 12]);
    icc.extend_from_slice(b"acsp");
    icc.extend_from_slice(&[0u8; 24]);
    icc.extend_from_slice(&0u32.to_be_bytes());
    icc.extend_from_slice(&s15(0.9642));
    icc.extend_from_slice(&s15(1.0));
    icc.extend_from_slice(&s15(0.8249));
    icc.extend_from_slice(&[0u8; 48]);
    assert_eq!(icc.len(), 128);
    icc.extend_from_slice(&table);
    icc.extend_from_slice(&body);
    assert_eq!(icc.len(), total);
    icc
}

/// Full jxl-rs decode; returns the embedded ICC profile.
fn jxl_rs_icc(data: &[u8], label: &str) -> Vec<u8> {
    use jxl::api::{
        JxlColorProfile, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let mut init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                assert!(!input.is_empty(), "jxl-rs header {label}: truncated");
                init = fallback
            }
            Err(e) => panic!("jxl-rs header {label}: {e:?}"),
        }
    };
    let icc = match decoder.embedded_color_profile() {
        JxlColorProfile::Icc(icc) => icc.clone(),
        _ => panic!("jxl-rs {label}: expected an ICC profile"),
    };
    let (width, height) = decoder.basic_info().size;
    let format = decoder.current_pixel_format().clone();
    let channels = format.color_type.samples_per_pixel();
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    });
    let mut frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame info {label}: {e:?}"),
        }
    };
    let mut color = Image::<f32>::new((width * channels, height)).expect("alloc color");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        color
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];
    loop {
        match frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => frame = fallback,
            Err(e) => panic!("jxl-rs frame decode {label}: {e:?}"),
        }
    }
    icc
}

/// Full djxl v0.12 decode; returns the original ICC profile.
fn djxl_icc(data: &[u8], label: &str) -> Vec<u8> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/icc-ratio");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join(format!("{label}.jxl"));
    let output = dir.join(format!("{label}.ppm"));
    let icc_out = dir.join(format!("{label}.icc"));
    std::fs::write(&input, data).unwrap();
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(&output)
        .arg(format!("--orig_icc_out={}", icc_out.display()))
        .arg("--num_threads=1")
        .output()
        .expect("run djxl v0.12");
    assert!(
        result.status.success(),
        "djxl {label}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    std::fs::read(&icc_out).expect("read djxl ICC output")
}

#[test]
fn repetitive_icc_profiles_decode_in_both_decoders() {
    let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i * 37 % 256) as u8).collect();
    for (name, fill) in [("zero", &[0u8][..]), ("repeat", &b"ABCDEFGHIJKLMNOP"[..])] {
        for total in [300_000usize, 1_000_000] {
            let icc = padded_icc(total, fill);
            let meta = ImageMetadata::new().with_icc_profile(&icc);
            for lossless in [true, false] {
                let label = format!(
                    "{name}-{total}-{}",
                    if lossless { "lossless" } else { "lossy" }
                );
                let request = |l: bool| {
                    if l {
                        LosslessConfig::new()
                            .encode_request(16, 16, PixelLayout::Rgb8)
                            .with_metadata(&meta)
                            .encode(&pixels)
                    } else {
                        LossyConfig::new(1.0)
                            .encode_request(16, 16, PixelLayout::Rgb8)
                            .with_metadata(&meta)
                            .encode(&pixels)
                    }
                };
                let jxl = request(lossless).unwrap_or_else(|e| panic!("encode {label}: {e}"));
                assert!(
                    jxl_rs_icc(&jxl, &label) == icc,
                    "jxl-rs {label}: ICC differs"
                );
                assert!(djxl_icc(&jxl, &label) == icc, "djxl {label}: ICC differs");
            }
        }
    }
}
