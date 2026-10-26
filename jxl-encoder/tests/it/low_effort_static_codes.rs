// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Lossy effort 1 and 2 write with static prefix codes whose alphabet has 64
//! symbols. Very fine quantization on HDR content needs larger symbols, and
//! the encode used to fail ("token symbol 64 exceeds the prefix-code
//! alphabet", found by the conformance sweep). Such frames now take the
//! two-pass path.

use jxl_encoder::{LossyConfig, PixelLayout};

fn decode_dims_jxl_rs(data: &[u8], label: &str) -> (usize, usize) {
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
    let mut color = Image::<f32>::new((w * nc, h)).unwrap();
    let mut bufs = vec![JxlOutputBuffer::from_image_rect_mut(
        color
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (w * nc, h),
            })
            .into_raw(),
    )];
    loop {
        match frame.process(&mut input, &mut bufs) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => frame = fallback,
            Err(e) => panic!("jxl-rs decode {label}: {e:?}"),
        }
    }
    (w, h)
}

fn djxl_decodes(data: &[u8], label: &str) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/low-effort-static");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join(format!("{label}.jxl"));
    let output = dir.join(format!("{label}.pfm"));
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
}

#[test]
fn fine_hdr_quantization_at_low_effort_encodes() {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus");
    let dir = corpus.get("gb82-sc").expect("gb82-sc corpus");
    let img = image::open(dir.join("windows.png"))
        .expect("windows.png")
        .to_rgb8();
    // Crops starting at row 120 include a bright title bar whose DC
    // quantizes past the static alphabet at d0.1 (the sweep's failing cell
    // was such a crop). Mirror-tiled past the image edge.
    let (iw, ih) = img.dimensions();
    let fold = |v: u32, n: u32| {
        let m = v % (2 * n);
        if m < n { m } else { 2 * n - 1 - m }
    };
    for (w, h, ox, oy) in [(260u32, 300u32, 120u32, 120u32), (300, 260, 60, 120)] {
        // PQ-coded signal: 8-bit screenshot samples as PQ code values.
        let mut px = Vec::with_capacity((w * h * 12) as usize);
        for y in 0..h {
            for x in 0..w {
                for c in img.get_pixel(fold(ox + x, iw), fold(oy + y, ih)).0 {
                    px.extend_from_slice(&(c as f32 / 255.0).to_ne_bytes());
                }
            }
        }
        for effort in [1u8, 2] {
            let label = format!("pq-d0.1-e{effort}-{w}x{h}");
            let bytes = LossyConfig::new(0.1)
                .with_effort(effort)
                .encode(&px, w, h, PixelLayout::RgbPqF32)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(
                decode_dims_jxl_rs(&bytes, &label),
                (w as usize, h as usize),
                "{label}"
            );
            djxl_decodes(&bytes, &label);
        }
    }
}

/// Effort 1-2 (zen strategies) write with static prefix tables. Those
/// tables cost ~820 bytes per file, so small images came out 3-16x larger
/// than the dynamic codes every other path builds (a 1x1 pixel: 875 B
/// against 63 B under `EncoderStrategy::Libjxl`), and no faster. Images up to
/// 128x128 now build dynamic codes; they must stay near the libjxl
/// strategy's size and decode in jxl-rs and djxl.
#[test]
fn small_images_at_low_effort_skip_static_tables() {
    use jxl_encoder::api::EncoderStrategy;
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus");
    let sources = [
        corpus
            .get("CID22/CID22-512/validation")
            .expect("CID22")
            .join("1279330.png"),
        corpus.get("gb82-sc").expect("gb82-sc").join("terminal.png"),
    ];
    for src in &sources {
        let img = image::open(src).expect("source").to_rgb8();
        for (w, h) in [(1u32, 1u32), (17, 9), (64, 64), (128, 128)] {
            let crop = image::imageops::crop_imm(&img, 100, 100, w, h).to_image();
            for effort in [1u8, 2] {
                let label = format!(
                    "{}-{w}x{h}-e{effort}",
                    src.file_stem().unwrap().to_string_lossy()
                );
                let encode = |s: EncoderStrategy| {
                    LossyConfig::new(1.0)
                        .with_effort(effort)
                        .with_strategy(s)
                        .encode(crop.as_raw(), w, h, PixelLayout::Rgb8)
                        .expect("encode")
                };
                let zen = encode(EncoderStrategy::Zenjxl);
                let reference = encode(EncoderStrategy::Libjxl);
                assert!(
                    zen.len() as f64 <= 1.25 * reference.len() as f64 + 16.0,
                    "{label}: {} B vs {} B under the libjxl strategy",
                    zen.len(),
                    reference.len()
                );
                assert_eq!(decode_dims_jxl_rs(&zen, &label), (w as usize, h as usize));
                djxl_decodes(&zen, &label);
            }
        }
    }
}
