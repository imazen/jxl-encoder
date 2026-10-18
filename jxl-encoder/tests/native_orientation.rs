//! Native orientation changes display geometry, never stored sample order.

use jxl_encoder::{
    AnimationFrame, AnimationParams, ImageMetadata, LosslessConfig, LossyConfig, Orientation,
    PixelLayout,
};
use zensim_decoder::api::{
    JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
    ProcessingResult,
};

fn decode(data: &[u8], adjust: bool) -> (usize, usize, u8, Vec<u8>) {
    let mut options = JxlDecoderOptions::default();
    options.adjust_orientation = adjust;
    let mut input = data;
    let ProcessingResult::Complete { result: mut header } =
        JxlDecoder::new(options).process(&mut input).unwrap()
    else {
        panic!("incomplete header")
    };
    let (width, height) = header.basic_info().size;
    let orientation = header.basic_info().orientation as u8;
    if !adjust {
        assert_eq!((width, height), (9, 5));
    }
    header.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![],
    });
    let ProcessingResult::Complete { result: frame } = header.process(&mut input).unwrap() else {
        panic!("incomplete frame")
    };
    let mut pixels = vec![0; width * height * 3];
    let output = JxlOutputBuffer::new(&mut pixels, height, width * 3);
    assert!(matches!(
        frame.process(&mut input, &mut [output]).unwrap(),
        ProcessingResult::Complete { .. }
    ));
    (width, height, orientation, pixels)
}

fn displayed(stored: &[u8], orientation: u8) -> (usize, usize, Vec<u8>) {
    let (width, height) = if orientation >= 5 { (5, 9) } else { (9, 5) };
    let mut pixels = vec![0; stored.len()];
    for y in 0..5 {
        for x in 0..9 {
            let (dx, dy) = match orientation {
                1 => (x, y),
                2 => (8 - x, y),
                3 => (8 - x, 4 - y),
                4 => (x, 4 - y),
                5 => (y, x),
                6 => (4 - y, x),
                7 => (4 - y, 8 - x),
                8 => (y, 8 - x),
                _ => unreachable!(),
            };
            pixels[(dy * width + dx) * 3..][..3].copy_from_slice(&stored[(y * 9 + x) * 3..][..3]);
        }
    }
    (width, height, pixels)
}

fn assert_libjxl_display(
    data: &[u8],
    name: &str,
    lossless: bool,
    expected: &(usize, usize, Vec<u8>),
) {
    let dir = jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "native-orientation");
    let input = dir.join(format!("{name}.jxl"));
    let output = dir.join(format!("{name}.ppm"));
    std::fs::write(&input, data).unwrap();
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(input)
        .arg(&output)
        .arg("--num_threads=1")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let ppm = std::fs::read(output).unwrap();
    let header = format!("P6\n{} {}\n255\n", expected.0, expected.1);
    assert!(ppm.starts_with(header.as_bytes()));
    if lossless {
        assert_eq!(&ppm[header.len()..], expected.2);
    }
}

#[test]
fn every_native_orientation_preserves_stored_pixels_and_display() {
    let source: Vec<u8> = (0..9 * 5 * 3).map(|n| (n * 37 % 251) as u8).collect();
    for lossless in [true, false] {
        let encode = |metadata: &ImageMetadata<'_>| {
            if lossless {
                LosslessConfig::new()
                    .encode_request(9, 5, PixelLayout::Rgb8)
                    .with_metadata(metadata)
                    .encode(&source)
                    .unwrap()
            } else {
                LossyConfig::new(1.0)
                    .with_effort(1)
                    .encode_request(9, 5, PixelLayout::Rgb8)
                    .with_metadata(metadata)
                    .encode(&source)
                    .unwrap()
            }
        };
        let reference = decode(&encode(&ImageMetadata::new()), false).3;
        if lossless {
            assert_eq!(reference, source);
        }
        for orientation in [
            Orientation::Identity,
            Orientation::FlipHorizontal,
            Orientation::Rotate180,
            Orientation::FlipVertical,
            Orientation::Transpose,
            Orientation::Rotate90CW,
            Orientation::AntiTranspose,
            Orientation::Rotate90CCW,
        ] {
            let data = encode(&ImageMetadata::new().with_orientation(orientation));
            let (w, h, tag, stored) = decode(&data, false);
            assert_eq!((w, h, tag), (9, 5, orientation as u8));
            assert_eq!(stored, reference);
            let (w, h, _, rendered) = decode(&data, true);
            let expected = displayed(&reference, orientation as u8);
            assert_eq!((w, h, rendered), expected);

            let name = format!("{lossless}-{}", orientation as u8);
            assert_libjxl_display(&data, &name, lossless, &expected);
        }
    }
}

#[test]
fn animation_metadata_preserves_native_orientation() {
    let source: Vec<u8> = (0..9 * 5 * 3).map(|n| (n * 37 % 251) as u8).collect();
    let frames = [AnimationFrame {
        pixels: &source,
        duration: 5,
        ..Default::default()
    }];
    for lossless in [true, false] {
        let encode = |orientation| {
            let metadata = ImageMetadata::new().with_orientation(orientation);
            if lossless {
                LosslessConfig::new()
                    .encode_animation_with_metadata(
                        9,
                        5,
                        PixelLayout::Rgb8,
                        &AnimationParams::default(),
                        &frames,
                        &metadata,
                    )
                    .unwrap()
            } else {
                LossyConfig::new(1.0)
                    .with_effort(1)
                    .encode_animation_with_metadata(
                        9,
                        5,
                        PixelLayout::Rgb8,
                        &AnimationParams::default(),
                        &frames,
                        &metadata,
                    )
                    .unwrap()
            }
        };
        let reference = decode(&encode(Orientation::Identity), false).3;
        if lossless {
            assert_eq!(reference, source);
        }
        let mut reference_cases = Vec::new();
        for orientation in [
            Orientation::Identity,
            Orientation::FlipHorizontal,
            Orientation::Rotate180,
            Orientation::FlipVertical,
            Orientation::Transpose,
            Orientation::Rotate90CW,
            Orientation::AntiTranspose,
            Orientation::Rotate90CCW,
        ] {
            let data = encode(orientation);
            let (w, h, tag, stored) = decode(&data, false);
            assert_eq!((w, h, tag), (9, 5, orientation as u8));
            assert_eq!(stored, reference);
            let (w, h, _, rendered) = decode(&data, true);
            let expected = displayed(&reference, orientation as u8);
            assert_eq!((w, h, rendered), expected);
            let name = format!("animation-{lossless}-{}", orientation as u8);
            reference_cases.push((data, name, expected));
        }
        for (data, name, expected) in reference_cases {
            assert_libjxl_display(&data, &name, lossless, &expected);
        }
    }
}
