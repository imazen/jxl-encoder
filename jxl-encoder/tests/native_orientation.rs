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

fn oracle_directory() -> &'static std::path::Path {
    static DIRECTORY: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    DIRECTORY.get_or_init(|| {
        let root = jxl_encoder::test_helpers::output_dir_for("jxl-encoder", "native-orientation");
        for run in 0u64.. {
            let directory = root.join(format!("run-{run}"));
            match std::fs::create_dir(&directory) {
                Ok(()) => return directory,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("create oracle directory {}: {error}", directory.display()),
            }
        }
        unreachable!("oracle run directory index exhausted")
    })
}

fn assert_libjxl_display(
    data: &[u8],
    name: &str,
    lossless: bool,
    expected: &(usize, usize, Vec<u8>),
) {
    let dir = oracle_directory();
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
            Orientation::FlipH,
            Orientation::Rotate180,
            Orientation::FlipV,
            Orientation::Transpose,
            Orientation::Rotate90,
            Orientation::Transverse,
            Orientation::Rotate270,
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
            Orientation::FlipH,
            Orientation::Rotate180,
            Orientation::FlipV,
            Orientation::Transpose,
            Orientation::Rotate90,
            Orientation::Transverse,
            Orientation::Rotate270,
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

fn exif_tag(tag: u16, big: bool) -> Vec<u8> {
    let mut tiff = if big {
        b"MM\0\x2a\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01".to_vec()
    } else {
        b"II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".to_vec()
    };
    tiff.extend(if big {
        tag.to_be_bytes()
    } else {
        tag.to_le_bytes()
    });
    tiff.extend([0; 6]);
    tiff
}

fn orient_pixels(source: &[u8], w: usize, h: usize, channels: usize, tag: u8) -> Vec<u8> {
    let dw = if tag >= 5 { h } else { w };
    let mut output = vec![0; source.len()];
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = match tag {
                1 => (x, y),
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (h - 1 - y, x),
                7 => (h - 1 - y, w - 1 - x),
                8 => (y, w - 1 - x),
                _ => unreachable!(),
            };
            output[(dy * dw + dx) * channels..][..channels]
                .copy_from_slice(&source[(y * w + x) * channels..][..channels]);
        }
    }
    output
}

fn decode_all_rgba(
    data: &[u8],
    adjust: bool,
    tag: u8,
    size: (usize, usize),
    intrinsic: Option<(usize, usize)>,
) -> Vec<Vec<u8>> {
    let mut options = JxlDecoderOptions::default();
    options.adjust_orientation = adjust;
    let mut input = data;
    let ProcessingResult::Complete {
        result: mut decoder,
    } = JxlDecoder::new(options).process(&mut input).unwrap()
    else {
        panic!("incomplete header")
    };
    let info = decoder.basic_info();
    assert_eq!(info.coded_size, size);
    assert_eq!(info.intrinsic_orientation as u8, tag);
    assert_eq!(info.orientation as u8, if adjust { 1 } else { tag });
    assert_eq!(info.intrinsic_size, intrinsic);
    let (w, h) = if adjust && tag >= 5 {
        (size.1, size.0)
    } else {
        size
    };
    assert_eq!(info.size, (w, h));
    decoder.set_pixel_format(JxlPixelFormat::rgba8(info.extra_channels.len()));
    let mut output = Vec::new();
    loop {
        let ProcessingResult::Complete { result: frame } = decoder.process(&mut input).unwrap()
        else {
            panic!("incomplete frame header")
        };
        let mut pixels = vec![0; w * h * 4];
        let ProcessingResult::Complete { result: next } = frame
            .process(
                &mut input,
                &mut [JxlOutputBuffer::new(&mut pixels, h, w * 4)],
            )
            .unwrap()
        else {
            panic!("incomplete frame")
        };
        output.push(pixels);
        decoder = next;
        if !decoder.has_more_frames() {
            break;
        }
    }
    output
}

fn real_rgba(w: u32, h: u32) -> Vec<u8> {
    let source = image::load_from_memory(include_bytes!("images/frymire-srgb.png"))
        .unwrap()
        .to_rgb8();
    image::imageops::crop_imm(&source, 0, 0, w, h)
        .to_image()
        .pixels()
        .enumerate()
        .flat_map(|(i, p)| [p[0], p[1], p[2], ((i * 73 + 11) % 256) as u8])
        .collect()
}

fn encode_rgba(
    source: &[u8],
    second: &[u8],
    size: (u32, u32),
    lossy: bool,
    animation: bool,
    metadata: &ImageMetadata<'_>,
) -> Vec<u8> {
    let lossy_config = LossyConfig::new(1.0).with_effort(1);
    let lossless_config = LosslessConfig::new().with_effort(1);
    let request = if lossy {
        lossy_config.encode_request(size.0, size.1, PixelLayout::Rgba8)
    } else {
        lossless_config.encode_request(size.0, size.1, PixelLayout::Rgba8)
    }
    .with_metadata(metadata);
    if animation {
        request
            .encode_animation(
                &AnimationParams::default(),
                &[
                    AnimationFrame::new(source, 10),
                    AnimationFrame::new(second, 20),
                ],
            )
            .unwrap()
    } else {
        request.encode(source).unwrap()
    }
}

/// Native JXL orientation is authoritative. EXIF/XMP bytes are pass-through
/// metadata: matching, conflicting, absent and invalid tags never compose a
/// second display transform. Explicit Identity differs from a baked rotation.
#[test]
fn native_exif_xmp_orientation_cartesian_product() {
    // This assignment is a compile-time contract: callers use the existing
    // zenpixels type directly, without conversion or a second orientation enum.
    let shared: zenpixels_convert::Orientation = Orientation::Rotate90;
    assert_eq!(shared.to_exif(), 6);
    let size = (32, 17);
    let source = real_rgba(size.0, size.1);
    let second: Vec<u8> = source
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[2], p[0], p[1], p[3]])
        .collect();
    let tags: Vec<Option<u16>> = std::iter::once(None).chain((0..=9).map(Some)).collect();
    let mut cells = 0;
    for lossy in [false, true] {
        for animation in [false, true] {
            let reference = encode_rgba(
                &source,
                &second,
                size,
                lossy,
                animation,
                &ImageMetadata::new(),
            );
            let stored = decode_all_rgba(
                &reference,
                false,
                1,
                (size.0 as usize, size.1 as usize),
                None,
            );
            for native in std::iter::once(None).chain(Orientation::ALL.into_iter().map(Some)) {
                for exif in &tags {
                    for xmp in &tags {
                        for big in [false, true] {
                            let exif_bytes = exif.map(|tag| exif_tag(tag, big));
                            let xmp_bytes = xmp.map(|tag| format!("<x:xmpmeta xmlns:x='adobe:ns:meta/' xmlns:tiff='http://ns.adobe.com/tiff/1.0/'><tiff:Orientation>{tag}</tiff:Orientation></x:xmpmeta>"));
                            let mut meta = ImageMetadata::new();
                            if let Some(tag) = native {
                                meta = meta.with_orientation(tag);
                            }
                            if let Some(bytes) = exif_bytes.as_deref() {
                                meta = meta.with_exif(bytes);
                            }
                            if let Some(bytes) = xmp_bytes.as_deref() {
                                meta = meta.with_xmp(bytes.as_bytes());
                            }
                            let tag = native.unwrap_or_default().to_exif();
                            let data = encode_rgba(&source, &second, size, lossy, animation, &meta);
                            let actual = decode_all_rgba(
                                &data,
                                false,
                                tag,
                                (size.0 as usize, size.1 as usize),
                                None,
                            );
                            assert_eq!(
                                actual, stored,
                                "stored: native={native:?}, Exif={exif:?}, XMP={xmp:?}, big={big}, lossy={lossy}, animation={animation}"
                            );
                            let rendered = decode_all_rgba(
                                &data,
                                true,
                                tag,
                                (size.0 as usize, size.1 as usize),
                                None,
                            );
                            for (actual, base) in rendered.iter().zip(&stored) {
                                assert_eq!(
                                    *actual,
                                    orient_pixels(base, size.0 as usize, size.1 as usize, 4, tag)
                                );
                            }
                            assert_eq!(rendered.len(), stored.len());
                            // Inspect the actual container payload independently of its decoder.
                            for (kind, bytes) in [
                                (b"Exif", exif_bytes.as_deref()),
                                (b"xml ", xmp_bytes.as_deref().map(str::as_bytes)),
                            ] {
                                assert_eq!(
                                    container_payload(&data, kind),
                                    bytes.map(|b| if kind == b"Exif" {
                                        [&[0; 4][..], b].concat()
                                    } else {
                                        b.to_vec()
                                    })
                                );
                            }
                            cells += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cells, 9 * 11 * 11 * 2 * 2 * 2);
}

fn container_payload(data: &[u8], kind: &[u8; 4]) -> Option<Vec<u8>> {
    if data.starts_with(&[0xff, 0x0a]) {
        return None;
    }
    let mut pos = 0;
    while pos < data.len() {
        let size = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        assert!(size >= 8 && pos + size <= data.len());
        if &data[pos + 4..pos + 8] == kind {
            return Some(data[pos + 8..pos + size].to_vec());
        }
        pos += size;
    }
    None
}

fn djxl_rgba16(data: &[u8], name: &str) -> (u32, u32, Vec<u8>) {
    let dir = oracle_directory();
    let input = dir.join(format!("{name}.jxl"));
    let output = dir.join(format!("{name}.png"));
    std::fs::write(&input, data).unwrap();
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(&output)
        .args(["--num_threads=1", "--bits_per_sample=16"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let image = image::open(output).unwrap().to_rgba16();
    (
        image.width(),
        image.height(),
        image
            .into_raw()
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect(),
    )
}

#[test]
fn orientations_preserve_intrinsic_size_and_multigroup_alpha() {
    for size in [(64, 32), (259, 133)] {
        let source = real_rgba(size.0, size.1);
        let second: Vec<u8> = source
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[1], p[2], p[0], p[3]])
            .collect();
        for lossy in [false, true] {
            for animation in [false, true] {
                for intrinsic in [None, Some((37, 23))] {
                    let base_meta = ImageMetadata::new();
                    let base_meta = if let Some((w, h)) = intrinsic {
                        base_meta.with_intrinsic_size(w, h)
                    } else {
                        base_meta
                    };
                    let baseline =
                        encode_rgba(&source, &second, size, lossy, animation, &base_meta);
                    let size_us = (size.0 as usize, size.1 as usize);
                    let intrinsic_us = intrinsic.map(|(w, h)| (w as usize, h as usize));
                    let stored = decode_all_rgba(&baseline, false, 1, size_us, intrinsic_us);
                    let label = format!(
                        "multi-{}-{}-{lossy}-{animation}-{intrinsic:?}",
                        size.0, size.1
                    );
                    let reference = djxl_rgba16(&baseline, &format!("{label}-base"));
                    for tag in Orientation::ALL {
                        let exif = exif_tag(if tag.to_exif() == 8 { 1 } else { 8 }, false);
                        let meta = base_meta.clone().with_orientation(tag).with_exif(&exif);
                        let data = encode_rgba(&source, &second, size, lossy, animation, &meta);
                        assert_eq!(
                            decode_all_rgba(&data, false, tag.to_exif(), size_us, intrinsic_us),
                            stored
                        );
                        let rendered =
                            decode_all_rgba(&data, true, tag.to_exif(), size_us, intrinsic_us);
                        assert_eq!(rendered.len(), stored.len());
                        for (actual, base) in rendered.iter().zip(&stored) {
                            assert_eq!(
                                *actual,
                                orient_pixels(base, size_us.0, size_us.1, 4, tag.to_exif())
                            );
                        }
                        let actual = djxl_rgba16(&data, &format!("{label}-{}", tag.to_exif()));
                        let dims = if tag.swaps_axes() {
                            (size.1, size.0)
                        } else {
                            size
                        };
                        assert_eq!((actual.0, actual.1), dims);
                        assert_eq!(
                            actual.2,
                            orient_pixels(&reference.2, size_us.0, size_us.1, 8, tag.to_exif()),
                            "djxl {label}, {tag:?}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(feature = "jpeg-reencoding")]
#[test]
fn jpeg_exif_xmp_combinations_preserve_original_bytes() {
    let tags: Vec<Option<u16>> = std::iter::once(None).chain((0..=9).map(Some)).collect();
    for size in [(64, 32), (259, 133)] {
        let rgba = real_rgba(size.0, size.1);
        let rgb: Vec<u8> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 90)
            .encode(&rgb, size.0, size.1, image::ExtendedColorType::Rgb8)
            .unwrap();
        let cfg = LosslessConfig::new().with_effort(3);
        let baseline = cfg.encode_jpeg_transcode(&jpeg).unwrap();
        let reference = djxl_rgba16(&baseline, &format!("jpeg-base-{}-{}", size.0, size.1));
        let size_us = (size.0 as usize, size.1 as usize);
        let stored = decode_all_rgba(&baseline, false, 1, size_us, None);
        for exif in &tags {
            for xmp in &tags {
                for big in [false, true] {
                    for reverse in [false, true] {
                        let exif_payload = exif.map(|tag| {
                            [b"Exif\0\0".as_slice(), exif_tag(tag, big).as_slice()].concat()
                        });
                        let xmp_payload=xmp.map(|tag|[b"http://ns.adobe.com/xap/1.0/\0".as_slice(),format!("<x:xmpmeta xmlns:x='adobe:ns:meta/' xmlns:tiff='http://ns.adobe.com/tiff/1.0/'><tiff:Orientation>{tag}</tiff:Orientation></x:xmpmeta>").as_bytes()].concat());
                        let mut payloads = [exif_payload.as_deref(), xmp_payload.as_deref()];
                        if reverse {
                            payloads.reverse();
                        }
                        let mut original = jpeg[..2].to_vec();
                        for payload in payloads.into_iter().flatten() {
                            original.extend([0xff, 0xe1]);
                            original
                                .extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
                            original.extend(payload);
                        }
                        original.extend(&jpeg[2..]);
                        let encoded = cfg.encode_jpeg_transcode(&original).unwrap();
                        let tag = exif
                            .and_then(|v| u8::try_from(v).ok())
                            .and_then(Orientation::from_exif)
                            .unwrap_or_default();
                        assert_eq!(
                            decode_all_rgba(&encoded, false, tag.to_exif(), size_us, None),
                            stored
                        );
                        let rendered =
                            decode_all_rgba(&encoded, true, tag.to_exif(), size_us, None);
                        assert_eq!(rendered.len(), 1);
                        assert_eq!(
                            rendered[0],
                            orient_pixels(&stored[0], size_us.0, size_us.1, 4, tag.to_exif())
                        );
                        assert_eq!(
                            zensim_decoder::reconstruct_jpeg(&encoded).unwrap().unwrap(),
                            original
                        );
                        let name = format!(
                            "jpeg-{}-{}-{exif:?}-{xmp:?}-{big}-{reverse}",
                            size.0, size.1
                        );
                        let actual = djxl_rgba16(&encoded, &name);
                        assert_eq!(
                            actual.2,
                            orient_pixels(&reference.2, size_us.0, size_us.1, 8, tag.to_exif())
                        );
                        let input = oracle_directory().join(format!("{name}.jxl"));
                        let output = oracle_directory().join(format!("{name}.jpg"));
                        let result =
                            std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(&input)
                                .arg(&output)
                                .args(["--reconstruct_jpeg", "--num_threads=1"])
                                .output()
                                .unwrap();
                        assert!(
                            result.status.success(),
                            "{name}: {}",
                            String::from_utf8_lossy(&result.stderr)
                        );
                        assert_eq!(std::fs::read(output).unwrap(), original, "djxl {name}");
                    }
                }
            }
        }
    }
}

/// Caller-selected because it requires the separately built public-C-API oracle.
/// This is not a runtime skip: enabling corpus-tests requires the oracle path.
#[cfg(feature = "corpus-tests")]
#[test]
fn original_icc_orientation_control() {
    let oracle=std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE").expect("caller must build and set LIBJXL_ORIGINAL_PROFILE_ORACLE; see just native-orientation-icc-check");
    let version = std::process::Command::new(&oracle)
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).unwrap().trim(),
        "libjxl original-profile oracle 12000"
    );
    let reference = |data: &[u8], name: &str| {
        let dir = oracle_directory();
        let input = dir.join(format!("{name}.jxl"));
        let stem = dir.join(name);
        std::fs::write(&input, data).unwrap();
        let result = std::process::Command::new(&oracle)
            .arg(&input)
            .arg(&stem)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let count: usize = String::from_utf8(result.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        (0..count)
            .map(|i| {
                let img = image::open(dir.join(format!("{name}-{i}.pam")))
                    .unwrap()
                    .to_rgba16();
                (
                    img.width(),
                    img.height(),
                    img.into_raw()
                        .into_iter()
                        .flat_map(u16::to_ne_bytes)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    for size in [(64, 32), (259, 133)] {
        let source = real_rgba(size.0, size.1);
        let second: Vec<u8> = source
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[2], p[0], p[1], p[3]])
            .collect();
        for lossy in [false, true] {
            for animation in [false, true] {
                let metadata = ImageMetadata::new()
                    .with_icc_profile(zenpixels_convert::icc_profiles::DISPLAY_P3_V4)
                    .with_intrinsic_size(37, 23);
                let baseline = encode_rgba(&source, &second, size, lossy, animation, &metadata);
                let label = format!("original-icc-{}-{}-{lossy}-{animation}", size.0, size.1);
                let base = reference(&baseline, &format!("{label}-base"));
                assert_eq!(base.len(), if animation { 2 } else { 1 });
                for tag in Orientation::ALL {
                    let exif = exif_tag(if tag.to_exif() == 8 { 1 } else { 8 }, true);
                    let meta = metadata.clone().with_orientation(tag).with_exif(&exif);
                    let encoded = encode_rgba(&source, &second, size, lossy, animation, &meta);
                    let frames = reference(&encoded, &format!("{label}-{}", tag.to_exif()));
                    assert_eq!(frames.len(), base.len());
                    let dims = if tag.swaps_axes() {
                        (size.1, size.0)
                    } else {
                        size
                    };
                    for (actual, expected) in frames.iter().zip(&base) {
                        assert_eq!((actual.0, actual.1), dims);
                        assert_eq!(
                            actual.2,
                            orient_pixels(
                                &expected.2,
                                size.0 as usize,
                                size.1 as usize,
                                8,
                                tag.to_exif()
                            ),
                            "{label} {tag:?}"
                        );
                    }
                }
            }
        }
    }
}

fn decode_rgb_and_depth(data: &[u8], adjust: bool) -> (usize, usize, Vec<u8>, Vec<u8>) {
    let mut options = JxlDecoderOptions::default();
    options.adjust_orientation = adjust;
    let mut input = data;
    let ProcessingResult::Complete { result: mut header } =
        JxlDecoder::new(options).process(&mut input).unwrap()
    else {
        panic!("incomplete header")
    };
    let (w, h) = header.basic_info().size;
    assert_eq!(header.basic_info().extra_channels.len(), 1);
    header.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![Some(JxlDataFormat::U8 { bit_depth: 8 })],
    });
    let ProcessingResult::Complete { result: frame } = header.process(&mut input).unwrap() else {
        panic!("incomplete frame header")
    };
    let mut rgb = vec![0; w * h * 3];
    let mut depth = vec![0; w * h];
    let mut outputs = [
        JxlOutputBuffer::new(&mut rgb, h, w * 3),
        JxlOutputBuffer::new(&mut depth, h, w),
    ];
    assert!(matches!(
        frame.process(&mut input, &mut outputs).unwrap(),
        ProcessingResult::Complete { .. }
    ));
    (w, h, rgb, depth)
}

#[test]
fn native_orientation_keeps_depth_aligned() {
    for (w, h) in [(64_u32, 32_u32), (259, 133)] {
        let rgba = real_rgba(w, h);
        let rgb: Vec<u8> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        let depth: Vec<u8> = (0..w * h)
            .map(|i| ((i * 29 + i / w * 73) % 256) as u8)
            .collect();
        let extras = [jxl_encoder::api::ExtraChannel::depth(&depth)];
        for lossy in [false, true] {
            let lossy_config = LossyConfig::new(1.0).with_effort(1);
            let lossless_config = LosslessConfig::new().with_effort(1);
            let encode = |tag| {
                let meta = ImageMetadata::new().with_orientation(tag);
                let request = if lossy {
                    lossy_config.encode_request(w, h, PixelLayout::Rgb8)
                } else {
                    lossless_config.encode_request(w, h, PixelLayout::Rgb8)
                };
                request
                    .with_metadata(&meta)
                    .with_extra_channels(&extras)
                    .encode(&rgb)
                    .unwrap()
            };
            let baseline = decode_rgb_and_depth(&encode(Orientation::Identity), false);
            assert_eq!(baseline.3, depth);
            for tag in Orientation::ALL {
                let data = encode(tag);
                let stored = decode_rgb_and_depth(&data, false);
                assert_eq!(stored, baseline);
                let rendered = decode_rgb_and_depth(&data, true);
                let dims = if tag.swaps_axes() {
                    (h as usize, w as usize)
                } else {
                    (w as usize, h as usize)
                };
                assert_eq!((rendered.0, rendered.1), dims);
                assert_eq!(
                    rendered.2,
                    orient_pixels(&baseline.2, w as usize, h as usize, 3, tag.to_exif())
                );
                assert_eq!(
                    rendered.3,
                    orient_pixels(&depth, w as usize, h as usize, 1, tag.to_exif())
                );
            }
        }
    }
}

/// The optional JPEG-shaped pixel path must follow the same native-tag policy.
#[cfg(all(feature = "jpeg-reencoding", feature = "chroma-subsampling"))]
#[test]
fn subsampled_native_exif_xmp_orientation_cartesian_product() {
    let size = (32, 17);
    let source: Vec<u8> = real_rgba(size.0, size.1)
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    let tags: Vec<Option<u16>> = std::iter::once(None).chain((0..=9).map(Some)).collect();
    let mut cells = 0;
    for mode in [
        jxl_encoder::ChromaSubsampling::Sub420,
        jxl_encoder::ChromaSubsampling::Sub422,
        jxl_encoder::ChromaSubsampling::Sub440,
    ] {
        let config = LossyConfig::new(0.1).with_chroma_subsampling(mode);
        let baseline_meta = ImageMetadata::new().with_intrinsic_size(37, 23);
        let baseline = config
            .encode_request(size.0, size.1, PixelLayout::Rgb8)
            .with_metadata(&baseline_meta)
            .encode(&source)
            .unwrap();
        let size_us = (size.0 as usize, size.1 as usize);
        let stored = decode_all_rgba(&baseline, false, 1, size_us, Some((37, 23)));
        let reference = djxl_rgba16(&baseline, &format!("subsampled-{mode:?}-baseline"));
        for native in std::iter::once(None).chain(Orientation::ALL.into_iter().map(Some)) {
            for exif in &tags {
                for xmp in &tags {
                    for big in [false, true] {
                        let exif_bytes = exif.map(|tag| exif_tag(tag, big));
                        let xmp_bytes = xmp.map(|tag| format!("<x:xmpmeta xmlns:x='adobe:ns:meta/' xmlns:tiff='http://ns.adobe.com/tiff/1.0/'><tiff:Orientation>{tag}</tiff:Orientation></x:xmpmeta>"));
                        let mut meta = baseline_meta.clone();
                        if let Some(tag) = native {
                            meta = meta.with_orientation(tag);
                        }
                        if let Some(bytes) = exif_bytes.as_deref() {
                            meta = meta.with_exif(bytes);
                        }
                        if let Some(bytes) = xmp_bytes.as_deref() {
                            meta = meta.with_xmp(bytes.as_bytes());
                        }
                        let tag = native.unwrap_or_default().to_exif();
                        let data = config
                            .encode_request(size.0, size.1, PixelLayout::Rgb8)
                            .with_metadata(&meta)
                            .encode(&source)
                            .unwrap();
                        assert_eq!(
                            decode_all_rgba(&data, false, tag, size_us, Some((37, 23))),
                            stored
                        );
                        let rendered = decode_all_rgba(&data, true, tag, size_us, Some((37, 23)));
                        assert_eq!(rendered.len(), stored.len());
                        for (actual, base) in rendered.iter().zip(&stored) {
                            assert_eq!(*actual, orient_pixels(base, size_us.0, size_us.1, 4, tag));
                        }
                        for (kind, bytes) in [
                            (b"Exif", exif_bytes.as_deref()),
                            (b"xml ", xmp_bytes.as_deref().map(str::as_bytes)),
                        ] {
                            assert_eq!(
                                container_payload(&data, kind),
                                bytes.map(|b| {
                                    if kind == b"Exif" {
                                        [&[0; 4][..], b].concat()
                                    } else {
                                        b.to_vec()
                                    }
                                })
                            );
                        }
                        // One independent reference permutation per native tag,
                        // with conflicting EXIF and XMP in the same stream.
                        if *exif == Some(8) && *xmp == Some(2) && !big {
                            let actual =
                                djxl_rgba16(&data, &format!("subsampled-{mode:?}-{native:?}"));
                            let dimensions = if tag >= 5 { (size.1, size.0) } else { size };
                            assert_eq!((actual.0, actual.1), dimensions);
                            assert_eq!(
                                actual.2,
                                orient_pixels(&reference.2, size_us.0, size_us.1, 8, tag)
                            );
                        }
                        cells += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cells, 3 * 9 * 11 * 11 * 2);
}

#[cfg(all(feature = "jpeg-reencoding", feature = "chroma-subsampling"))]
#[test]
fn subsampled_hdr_icc_orientation_keeps_samples() {
    let mut cells = 0;
    for size in [(32, 17), (259, 17)] {
        let source: Vec<u8> = real_rgba(size.0, size.1)
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        let size_us = (size.0 as usize, size.1 as usize);
        for mode in [
            jxl_encoder::ChromaSubsampling::Sub420,
            jxl_encoder::ChromaSubsampling::Sub422,
            jxl_encoder::ChromaSubsampling::Sub440,
        ] {
            let config = LossyConfig::new(0.1).with_chroma_subsampling(mode);
            for (name, color, profile) in [
                ("pq", jxl_encoder::ColorEncoding::bt2100_pq(), None),
                ("hlg", jxl_encoder::ColorEncoding::bt2100_hlg(), None),
                (
                    "p3",
                    jxl_encoder::ColorEncoding::srgb(),
                    Some(zenpixels_convert::icc_profiles::DISPLAY_P3_V4),
                ),
            ] {
                let mut meta = ImageMetadata::new()
                    .with_intrinsic_size(37, 23)
                    .with_intensity_target(4000.0);
                if let Some(profile) = profile {
                    meta = meta.with_icc_profile(profile);
                }
                let baseline = config
                    .encode_request(size.0, size.1, PixelLayout::Rgb8)
                    .with_metadata(&meta)
                    .with_color_encoding(color.clone())
                    .encode(&source)
                    .unwrap();
                let stored = decode_all_rgba(&baseline, false, 1, size_us, Some((37, 23)));
                let label = format!("subsampled-{mode:?}-{name}-{}-{}", size.0, size.1);
                let reference = djxl_rgba16(&baseline, &format!("{label}-base"));
                for tag in Orientation::ALL {
                    let exif = exif_tag(if tag.to_exif() == 8 { 1 } else { 8 }, false);
                    let xmp = b"<x:xmpmeta xmlns:x='adobe:ns:meta/' xmlns:tiff='http://ns.adobe.com/tiff/1.0/'><tiff:Orientation>2</tiff:Orientation></x:xmpmeta>";
                    let oriented = meta
                        .clone()
                        .with_orientation(tag)
                        .with_exif(&exif)
                        .with_xmp(xmp);
                    let data = config
                        .encode_request(size.0, size.1, PixelLayout::Rgb8)
                        .with_metadata(&oriented)
                        .with_color_encoding(color.clone())
                        .encode(&source)
                        .unwrap();
                    let image = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&data))
                        .unwrap();
                    assert_eq!(image.original_icc(), profile);
                    assert_eq!(
                        image.image_header().metadata.tone_mapping.intensity_target,
                        4000.0
                    );
                    assert_eq!(
                        decode_all_rgba(&data, false, tag.to_exif(), size_us, Some((37, 23))),
                        stored
                    );
                    let rendered =
                        decode_all_rgba(&data, true, tag.to_exif(), size_us, Some((37, 23)));
                    assert_eq!(rendered.len(), stored.len());
                    for (actual, base) in rendered.iter().zip(&stored) {
                        assert_eq!(
                            *actual,
                            orient_pixels(base, size_us.0, size_us.1, 4, tag.to_exif())
                        );
                    }
                    let actual = djxl_rgba16(&data, &format!("{label}-{}", tag.to_exif()));
                    let dimensions = if tag.swaps_axes() {
                        (size.1, size.0)
                    } else {
                        size
                    };
                    assert_eq!((actual.0, actual.1), dimensions);
                    assert_eq!(
                        actual.2,
                        orient_pixels(&reference.2, size_us.0, size_us.1, 8, tag.to_exif()),
                        "{label} {tag:?}"
                    );
                    cells += 1;
                }
            }
        }
    }
    assert_eq!(cells, 144);
}
