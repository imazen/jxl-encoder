// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Images with many extra channels: lossless tree learning used to index
//! past its reference-property array (panic) with more than 9 same-size
//! channels, and the lossy extra-channel tree grows one level per channel.
//! More than 4 extra channels make the file Level 10, whose MA tree limit is
//! 2048 levels; every extra channel must still decode exactly.

use jxl_encoder::api::ExtraChannel;
use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};

/// Decodes through jxl-rs and returns every extra channel as f32 planes.
fn jxl_rs_extras(data: &[u8], label: &str) -> Vec<Vec<f32>> {
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
    let (width, height) = decoder.basic_info().size;
    let format = decoder.current_pixel_format().clone();
    let channels = format.color_type.samples_per_pixel();
    let extras = format.extra_channel_format.len();
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![Some(JxlDataFormat::f32()); extras],
    });
    let mut frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame info {label}: {e:?}"),
        }
    };
    let rect = |w: usize| Rect {
        origin: (0, 0),
        size: (w, height),
    };
    let mut color = Image::<f32>::new((width * channels, height)).expect("alloc color");
    let mut extra: Vec<Image<f32>> = (0..extras)
        .map(|_| Image::<f32>::new((width, height)).expect("alloc extra"))
        .collect();
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        color.get_rect_mut(rect(width * channels)).into_raw(),
    )];
    for e in &mut extra {
        buffers.push(JxlOutputBuffer::from_image_rect_mut(
            e.get_rect_mut(rect(width)).into_raw(),
        ));
    }
    loop {
        match frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => frame = fallback,
            Err(e) => panic!("jxl-rs frame decode {label}: {e:?}"),
        }
    }
    drop(buffers);
    extra
        .iter()
        .map(|e| {
            (0..height)
                .flat_map(|y| e.row(y)[..width].to_vec())
                .collect()
        })
        .collect()
}

/// Every decoded extra channel must reproduce its 8-bit source exactly.
fn assert_extras_exact(data: &[u8], planes: &[Vec<u8>], label: &str) {
    let decoded = jxl_rs_extras(data, label);
    assert_eq!(decoded.len(), planes.len(), "{label}: extra channel count");
    for (c, (d, p)) in decoded.iter().zip(planes).enumerate() {
        let bad = d
            .iter()
            .zip(p)
            .filter(|&(&v, &s)| (v * 255.0).round() as i32 != s as i32)
            .count();
        assert_eq!(bad, 0, "{label}: extra channel {c} has {bad} wrong samples");
    }
}

fn djxl_decodes(data: &[u8], label: &str) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/many-extras");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join(format!("{label}.jxl"));
    std::fs::write(&input, data).unwrap();
    let out = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg("--disable_output")
        .arg("--num_threads=1")
        .output()
        .expect("run djxl v0.12");
    assert!(
        out.status.success(),
        "djxl {label}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// 70 depth channels (73 coded channels), lossless and lossy.
#[test]
fn many_extra_channels_round_trip() {
    for (w, h) in [(16u32, 16u32), (300, 260)] {
        let n = (w * h) as usize;
        let rgb: Vec<u8> = (0..n * 3).map(|i| (i * 7 % 251) as u8).collect();
        let planes: Vec<Vec<u8>> = (0..70)
            .map(|c| (0..n).map(|i| ((i * (c + 3)) % 256) as u8).collect())
            .collect();
        let extras: Vec<ExtraChannel<'_>> = planes.iter().map(|p| ExtraChannel::depth(p)).collect();
        for lossless in [true, false] {
            let label = format!(
                "extras70-{w}x{h}-{}",
                if lossless { "lossless" } else { "lossy" }
            );
            let result = if lossless {
                LosslessConfig::new()
                    .encode_request(w, h, PixelLayout::Rgb8)
                    .with_extra_channels(&extras)
                    .encode(&rgb)
            } else {
                LossyConfig::new(1.0)
                    .with_effort(7)
                    .encode_request(w, h, PixelLayout::Rgb8)
                    .with_extra_channels(&extras)
                    .encode(&rgb)
            };
            let bytes = result.unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_extras_exact(&bytes, &planes, &label);
            djxl_decodes(&bytes, &label);
        }
    }
}

/// Constant extras become one-entry palettes (meta channels), which switch
/// the lossy extras writer to a channel-split tree with one split per coded
/// channel: a chain deeper than Level 5 allows, legal in this Level 10 file.
#[test]
fn many_constant_extra_channels_round_trip() {
    for (w, h) in [(16u32, 16u32), (300, 260)] {
        let n = (w * h) as usize;
        let rgb: Vec<u8> = (0..n * 3).map(|i| (i * 7 % 251) as u8).collect();
        let planes: Vec<Vec<u8>> = (0..70).map(|c| vec![(c * 3) as u8; n]).collect();
        let extras: Vec<ExtraChannel<'_>> = planes.iter().map(|p| ExtraChannel::depth(p)).collect();
        let label = format!("const-extras70-{w}x{h}");
        let bytes = LossyConfig::new(1.0)
            .with_effort(7)
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_extra_channels(&extras)
            .encode(&rgb)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_extras_exact(&bytes, &planes, &label);
        djxl_decodes(&bytes, &label);
    }
}
