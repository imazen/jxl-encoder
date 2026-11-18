//! CMYK source codes, profile metadata and supported request routes.
#![cfg(all(feature = "__expert", feature = "std"))]
use jxl_encoder::{ImageMetadata, LosslessConfig, PixelLayout};
use std::{io::Write, path::Path};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;
fn profile() -> Vec<u8> {
    include_str!("fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|line| {
            (0..line.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
        })
        .collect()
}
fn new_output() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("cmyk-source-contract");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let out = root.join(format!("run-{i}"));
            match std::fs::create_dir(&out) {
                Ok(()) => Some(out),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{error}"),
            }
        })
        .unwrap()
}
fn check_codes(colors: &[f32], black: &[f32], source: &[u16], maximum: f32, label: &str) {
    assert_eq!(colors.len(), source.len() / 4 * 3, "{label}");
    assert_eq!(black.len(), source.len() / 4, "{label}");
    for (i, expected) in source.as_chunks::<4>().0.iter().enumerate() {
        for c in 0..3 {
            let actual = colors[i * 3 + c];
            assert!(actual.is_finite());
            assert_eq!(
                (actual * maximum).round() as u16,
                expected[c],
                "{label} CMY pixel{i} component{c}"
            );
        }
        assert!(black[i].is_finite());
        assert_eq!(
            (black[i] * maximum).round() as u16,
            expected[3],
            "{label} Black pixel{i}"
        );
    }
}
fn check_native(
    directory: &Path,
    bytes: &[u8],
    source: &[u16],
    width: usize,
    height: usize,
    maximum: f32,
    oracle: &std::ffi::OsStr,
) {
    std::fs::create_dir(directory).unwrap();
    let input = directory.join("source.jxl");
    std::fs::write(&input, bytes).unwrap();
    let result = std::process::Command::new(oracle)
        .arg(&input)
        .arg(directory.join("raw"))
        .arg("raw-cmy")
        .output()
        .unwrap();
    std::fs::write(directory.join("headers.tsv"), &result.stdout).unwrap();
    std::fs::write(directory.join("native.log"), &result.stderr).unwrap();
    assert!(
        result.status.success(),
        "{}: {}",
        directory.display(),
        String::from_utf8_lossy(&result.stderr)
    );
    let color = pfm::read_pfm_samples(&directory.join("raw-0.pfm"), (width, height));
    let black = pfm::read_pfm_samples(&directory.join("raw-0-black.pfm"), (width, height));
    check_codes(
        &color,
        &black,
        source,
        maximum,
        &directory.display().to_string(),
    );
}
#[test]
fn tagged_cmyk_lossless_keeps_every_source_code_in_primary_and_native_decoders() {
    let out = new_output();
    let profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let oracle =
        std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("caller-selected pinned raw CMY observer");
    let version = std::process::Command::new(&oracle)
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).unwrap().trim(),
        "libjxl CMYK raw-frame oracle 12000"
    );
    let mut journal = std::fs::File::create(out.join("samples.tsv")).unwrap();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for effort in [1, 3, 7] {
            for (width, height) in [(1_usize, 1_usize), (32, 17), (259, 17)] {
                for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
                    let wide = layout.is_16bit();
                    let maximum = if wide { 65535.0 } else { 255.0 };
                    let source: Vec<u16> = (0..width * height * 4)
                        .map(|i| {
                            if wide {
                                let edge = [0, 1, 255, 256, 257, 4095, 4096, 65534, 65535];
                                if i < edge.len() {
                                    edge[i]
                                } else {
                                    (i as u16).wrapping_mul(0x1f31).wrapping_add(0x534d)
                                }
                            } else {
                                ((i * 71 + i / 4 * 17) % 256) as u16
                            }
                        })
                        .collect();
                    let pixels: Vec<u8> = source
                        .iter()
                        .flat_map(|&value| {
                            if wide {
                                value.to_ne_bytes().to_vec()
                            } else {
                                vec![value as u8]
                            }
                        })
                        .collect();
                    let config = LosslessConfig::new()
                        .with_effort(effort)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode);
                    let request = || {
                        config
                            .encode_request(width as u32, height as u32, layout)
                            .with_metadata(&metadata)
                    };
                    let bytes = request().encode(&pixels).unwrap();
                    let stride = width * layout.bytes_per_pixel() + 11;
                    let mut padded = vec![0xa5; stride * height];
                    for y in 0..height {
                        padded[y * stride..][..width * layout.bytes_per_pixel()].copy_from_slice(
                            &pixels[y * width * layout.bytes_per_pixel()..]
                                [..width * layout.bytes_per_pixel()],
                        );
                    }
                    assert_eq!(
                        bytes,
                        request().with_row_stride(stride).encode(&padded).unwrap(),
                        "packed/strided CMYK"
                    );
                    let mut writer = Vec::new();
                    request().encode_to(&pixels, &mut writer).unwrap();
                    assert_eq!(bytes, writer, "writer CMYK");
                    let label = format!("{policy}-e{effort}-{layout:?}-{width}x{height}");
                    let parsed = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&bytes))
                        .unwrap();
                    assert_eq!(parsed.original_icc(), Some(profile.as_slice()));
                    let (colors, extras) = decoded_extras::decode_current_planes(&bytes, &label);
                    assert_eq!(extras.len(), 1);
                    check_codes(&colors, &extras[0], &source, maximum, &label);
                    check_native(
                        &out.join(&label),
                        &bytes,
                        &source,
                        width,
                        height,
                        maximum,
                        &oracle,
                    );
                    writeln!(journal,"{label}\tprimary/native exact source codes\tpacked/strided/writer identical").unwrap();
                    journal.flush().unwrap();
                }
            }
        }
    }
}

#[test]
fn separated_cmy_black_requests_preserve_codes_and_use_the_profile() {
    use jxl_encoder::{LossyConfig, api::ExtraChannel};
    let out = new_output();
    let profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let oracle =
        std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("caller-selected pinned raw CMY observer");
    let vectors: Vec<[f32; 3]> = include_str!("fixtures/cmyk_lcms_swatches.hex")
        .lines()
        .map(|line| {
            line.split_whitespace()
                .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect();
    let swatches = [
        [255_u8, 255, 255, 255],
        [0, 255, 255, 255],
        [255, 0, 255, 255],
        [255, 255, 0, 255],
        [255, 255, 255, 0],
        [64, 128, 192, 224],
    ];
    assert_eq!(vectors.len(), swatches.len());
    let mut journal = std::fs::File::create(out.join("separated.tsv")).unwrap();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            for layout in [
                PixelLayout::Rgb8,
                PixelLayout::Rgba8,
                PixelLayout::Bgr8,
                PixelLayout::Bgra8,
                PixelLayout::Rgb16,
                PixelLayout::Rgba16,
            ] {
                let wide = layout.is_16bit();
                let maximum = if wide { 65535.0 } else { 255.0 };
                let scale = if wide { 257 } else { 1 };
                let alpha = layout.has_alpha();
                for (case, (swatch, expected)) in swatches.into_iter().zip(&vectors).enumerate() {
                    let count = width * height;
                    let k8 = vec![swatch[3]; count];
                    let k16 = vec![u16::from(swatch[3]) * 257; count];
                    let extras = [if wide {
                        ExtraChannel::black_u16(&k16)
                    } else {
                        ExtraChannel::black(&k8)
                    }];
                    let mut pixel = swatch[..3].to_vec();
                    if matches!(layout, PixelLayout::Bgr8 | PixelLayout::Bgra8) {
                        pixel.swap(0, 2);
                    }
                    if alpha {
                        pixel.push(137);
                    }
                    let pixel: Vec<u8> = pixel
                        .into_iter()
                        .flat_map(|v| {
                            if wide {
                                (u16::from(v) * 257).to_ne_bytes().to_vec()
                            } else {
                                vec![v]
                            }
                        })
                        .collect();
                    let pixels = pixel.repeat(count);
                    for lossless in [true, false] {
                        let exact = LosslessConfig::new()
                            .with_effort(3)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode);
                        let lossy = LossyConfig::new(0.01)
                            .with_effort(3)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode);
                        let request = || {
                            if lossless {
                                exact.encode_request(width as u32, height as u32, layout)
                            } else {
                                lossy.encode_request(width as u32, height as u32, layout)
                            }
                            .with_metadata(&metadata)
                            .with_extra_channels(&extras)
                        };
                        let bytes = request().encode(&pixels).unwrap();
                        let stride = width * pixel.len() + 11;
                        let mut padded = vec![0xa5; stride * height];
                        for y in 0..height {
                            padded[y * stride..][..width * pixel.len()].copy_from_slice(
                                &pixels[y * width * pixel.len()..][..width * pixel.len()],
                            );
                        }
                        assert_eq!(
                            bytes,
                            request().with_row_stride(stride).encode(&padded).unwrap()
                        );
                        let mut destination = Vec::new();
                        request().encode_to(&pixels, &mut destination).unwrap();
                        assert_eq!(bytes, destination);
                        let label = format!(
                            "separated-{policy}-{layout:?}-{width}x{height}-{case}-{lossless}"
                        );
                        let directory = out.join(&label);
                        std::fs::create_dir(&directory).unwrap();
                        let input = directory.join("source.jxl");
                        std::fs::write(&input, &bytes).unwrap();
                        let (primary, planes) =
                            decoded_extras::decode_current_planes(&bytes, &label);
                        assert_eq!(planes.len(), 1 + usize::from(alpha));
                        let black = &planes[usize::from(alpha)];
                        assert!(black.iter().all(|v| v.is_finite()
                            && (*v * maximum).round() as u16 == u16::from(swatch[3]) * scale));
                        let native = std::process::Command::new(&oracle)
                            .arg(&input)
                            .arg(directory.join("native"))
                            .arg(if lossless { "raw-cmy" } else { "linear" })
                            .output()
                            .unwrap();
                        std::fs::write(directory.join("native.log"), &native.stderr).unwrap();
                        assert!(
                            native.status.success(),
                            "{label}: {}",
                            String::from_utf8_lossy(&native.stderr)
                        );
                        let colors =
                            pfm::read_pfm_samples(&directory.join("native-0.pfm"), (width, height));
                        let native_black = pfm::read_pfm_samples(
                            &directory.join("native-0-black.pfm"),
                            (width, height),
                        );
                        if alpha {
                            let native_alpha = pfm::read_pfm_samples(
                                &directory.join("native-0-alpha.pfm"),
                                (width, height),
                            );
                            for values in [&planes[0], &native_alpha] {
                                assert!(
                                    values.iter().all(|v| v.is_finite()
                                        && (*v * maximum).round() as u16 == 137 * scale),
                                    "exact alpha {label}"
                                );
                            }
                        }
                        if lossless {
                            let source: Vec<u16> = swatch
                                .iter()
                                .map(|v| u16::from(*v) * scale)
                                .collect::<Vec<_>>()
                                .repeat(count);
                            check_codes(&primary, black, &source, maximum, &label);
                            check_codes(&colors, &native_black, &source, maximum, &label);
                        } else {
                            assert!(native_black.iter().all(|v| v.is_finite()
                                && (*v * maximum).round() as u16 == u16::from(swatch[3]) * scale));
                            for values in [&primary, &colors] {
                                assert_eq!(values.len(), count * 3);
                                assert!(values.iter().all(|v| v.is_finite()));
                                let error = values
                                    .iter()
                                    .zip(expected.iter().cycle())
                                    .map(|(a, b)| (a - b).abs())
                                    .fold(0_f32, f32::max);
                                assert!(error <= 0.005, "{label}: physical error {error}");
                            }
                        }
                        writeln!(
                            journal,
                            "{label}\tcomplete primary/native samples; packed/strided/writer"
                        )
                        .unwrap();
                        journal.flush().unwrap();
                    }
                }
            }
        }
    }
}

#[test]
fn cmyk_streaming_reports_the_unsupported_layout_and_preserves_destination() {
    let profile = profile();
    for (_, strategy, mode) in encoder_policies::policies() {
        for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
            let pixels = vec![64_u8; 32 * 17 * layout.bytes_per_pixel()];
            let exact = LosslessConfig::new()
                .with_effort(3)
                .with_threads(1)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            let lossy = jxl_encoder::LossyConfig::new(1.0)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            let constructor_error = match exact.encoder(32, 17, layout) {
                Ok(_) => panic!("lossless CMYK streaming unexpectedly admitted"),
                Err(error) => error,
            };
            assert!(
                format!("{constructor_error:?}").contains("UnsupportedPixelLayout"),
                "{constructor_error:?}"
            );
            let mut lossy = lossy
                .encoder(32, 17, layout)
                .unwrap()
                .with_icc_profile(&profile);
            let error = lossy.push_rows(&pixels, 17).unwrap_err();
            assert!(
                format!("{error:?}").contains("UnsupportedPixelLayout"),
                "{error:?}"
            );
            let mut destination = b"existing stream destination".to_vec();
            let original = destination.clone();
            assert!(lossy.finish_into(&mut destination).is_err());
            assert_eq!(destination, original);
        }
    }
}

#[test]
fn cmyk_precision_and_admission_edges() {
    use jxl_encoder::{ColorEncoding, Limits, LossyConfig};
    let root = new_output();
    let profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("pinned native sample observer");
    let expected: Vec<f32> = include_str!("fixtures/cmyk_lcms_swatches.hex")
        .lines()
        .nth(5)
        .unwrap()
        .split_whitespace()
        .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
        .collect();
    for (policy, strategy, mode) in encoder_policies::policies() {
        let exact = LosslessConfig::new()
            .with_effort(3)
            .with_threads(1)
            .with_strategy(strategy.clone())
            .with_mode(mode);
        let lossy = LossyConfig::new(0.01)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(strategy)
            .with_mode(mode);
        for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
            let pixels = vec![64_u8; 32 * 17 * layout.bytes_per_pixel()];
            for result in [
                exact.encode(&pixels, 32, 17, layout),
                lossy.encode(&pixels, 32, 17, layout),
            ] {
                let error = result.unwrap_err();
                assert!(error.to_string().contains("CMYK ICC"), "{error}");
            }
            for gamma in [false, true] {
                let request = lossy
                    .encode_request(32, 17, layout)
                    .with_metadata(&metadata);
                let request = if gamma {
                    request.with_source_gamma(2.2)
                } else {
                    request.with_color_encoding(ColorEncoding::srgb())
                };
                let mut destination = b"existing destination".to_vec();
                let original = destination.clone();
                assert!(request.encode_to(&pixels, &mut destination).is_err());
                assert_eq!(destination, original);
            }
            let mut opaque = vec![0_u8; 132];
            opaque[16..20].copy_from_slice(b"CMYK");
            let raw = ImageMetadata::new().with_icc_profile(&opaque);
            let mut destination = b"existing destination".to_vec();
            let original = destination.clone();
            assert!(
                lossy
                    .encode_request(32, 17, layout)
                    .with_metadata(&raw)
                    .encode_to(&pixels, &mut destination)
                    .is_err()
            );
            assert_eq!(destination, original);
        }
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            let source: Vec<u16> = [64_u16, 128, 192, 224]
                .into_iter()
                .map(|v| ((u32::from(v) * 4095 + 127) / 255) as u16)
                .collect::<Vec<_>>()
                .repeat(width * height);
            let pixels: Vec<u8> = source.iter().flat_map(|v| v.to_ne_bytes()).collect();
            for lossless in [false, true] {
                for fallible in [false, true] {
                    let limits = Limits::default().with_fallible_alloc(fallible);
                    let request = if lossless {
                        exact.encode_request(width as u32, height as u32, PixelLayout::Cmyk16)
                    } else {
                        lossy.encode_request(width as u32, height as u32, PixelLayout::Cmyk16)
                    }
                    .with_metadata(&metadata)
                    .with_bits_per_sample(12)
                    .with_limits(&limits);
                    let bytes = request.encode(&pixels).unwrap();
                    let label = format!("precision-{policy}-{width}-{lossless}-{fallible}");
                    let dir = root.join(&label);
                    std::fs::create_dir(&dir).unwrap();
                    let input = dir.join("source.jxl");
                    std::fs::write(&input, &bytes).unwrap();
                    let (colors, extras) = decoded_extras::decode_current_planes(&bytes, &label);
                    assert_eq!(extras.len(), 1);
                    let native = std::process::Command::new(&oracle)
                        .arg(&input)
                        .arg(dir.join("native"))
                        .arg(if lossless { "raw-cmy" } else { "linear" })
                        .output()
                        .unwrap();
                    std::fs::write(dir.join("native.log"), &native.stderr).unwrap();
                    assert!(native.status.success(), "{label}");
                    let native_color =
                        pfm::read_pfm_samples(&dir.join("native-0.pfm"), (width, height));
                    let black =
                        pfm::read_pfm_samples(&dir.join("native-0-black.pfm"), (width, height));
                    if lossless {
                        check_codes(&colors, &extras[0], &source, 4095.0, &label);
                        check_codes(&native_color, &black, &source, 4095.0, &label);
                    } else {
                        for values in [&extras[0], &black] {
                            assert!(
                                values
                                    .iter()
                                    .all(|v| v.is_finite()
                                        && (*v * 4095.0).round() as u16 == source[3]),
                                "{label} 12-bit Black"
                            );
                        }
                        for values in [&colors, &native_color] {
                            assert!(values.iter().all(|v| v.is_finite()));
                            let error = values
                                .iter()
                                .zip(expected.iter().cycle())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0_f32, f32::max);
                            assert!(error <= 0.005, "{label} physical {error}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn independent_black_precision_is_not_replaced_by_color_precision() {
    use jxl_encoder::api::ExtraChannel;
    let root = new_output();
    let profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("pinned native sample observer");
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            for layout in [PixelLayout::Rgb16, PixelLayout::Rgba16] {
                let alpha = layout.has_alpha();
                let pixel: Vec<u16> = if alpha {
                    vec![1000, 2000, 3000, 3500]
                } else {
                    vec![1000, 2000, 3000]
                };
                let pixels: Vec<u8> = pixel
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>()
                    .repeat(width * height);
                let black = vec![0xe0e0_u16; width * height];
                let extras = [ExtraChannel::black_u16(&black)];
                let bytes = LosslessConfig::new()
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .encode_request(width as u32, height as u32, layout)
                    .with_metadata(&metadata)
                    .with_extra_channels(&extras)
                    .with_bits_per_sample(12)
                    .encode(&pixels)
                    .unwrap();
                let label = format!("separate-precision-{policy}-{width}-{layout:?}");
                let dir = root.join(&label);
                std::fs::create_dir(&dir).unwrap();
                let input = dir.join("source.jxl");
                std::fs::write(&input, &bytes).unwrap();
                let (color, planes) = decoded_extras::decode_current_planes(&bytes, &label);
                let native = std::process::Command::new(&oracle)
                    .arg(&input)
                    .arg(dir.join("native"))
                    .arg("raw-cmy")
                    .output()
                    .unwrap();
                std::fs::write(dir.join("native.log"), &native.stderr).unwrap();
                assert!(native.status.success());
                let native_color =
                    pfm::read_pfm_samples(&dir.join("native-0.pfm"), (width, height));
                let native_black =
                    pfm::read_pfm_samples(&dir.join("native-0-black.pfm"), (width, height));
                for values in [&color, &native_color] {
                    assert!(
                        values.iter().enumerate().all(|(i, v)| v.is_finite()
                            && (*v * 4095.0).round() as u16 == [1000, 2000, 3000][i % 3]),
                        "{label} exact CMY"
                    );
                }
                for values in [&planes[usize::from(alpha)], &native_black] {
                    for (index, value) in values.iter().enumerate() {
                        assert!(value.is_finite());
                        assert_eq!(
                            (*value * 65535.0).round() as u16,
                            0xe0e0,
                            "{label} exact 16-bit Black pixel{index}"
                        );
                    }
                }
                if alpha {
                    let native_alpha =
                        pfm::read_pfm_samples(&dir.join("native-0-alpha.pfm"), (width, height));
                    for values in [&planes[0], &native_alpha] {
                        assert!(
                            values
                                .iter()
                                .all(|v| v.is_finite() && (*v * 4095.0).round() as u16 == 3500),
                            "{label} exact 12-bit alpha"
                        );
                    }
                }
            }
        }
    }
}

fn relative_profile() -> Vec<u8> {
    let original = profile();
    let count = u32::from_be_bytes(original[128..132].try_into().unwrap()) as usize;
    let end = 132 + count * 12;
    let mut entries = Vec::new();
    let mut a2b = None;
    for tag in original[132..end].as_chunks::<12>().0 {
        let offset = u32::from_be_bytes(tag[4..8].try_into().unwrap());
        assert!(offset as usize >= end);
        let mut tag = tag.to_vec();
        tag[4..8].copy_from_slice(&(offset + 12).to_be_bytes());
        if &tag[..4] == b"A2B0" {
            a2b = Some(tag.clone());
        }
        entries.extend(tag);
    }
    let mut result = original[..128].to_vec();
    result[..4].copy_from_slice(&u32::try_from(original.len() + 12).unwrap().to_be_bytes());
    result[84..100].fill(0);
    result.extend_from_slice(&u32::try_from(count + 1).unwrap().to_be_bytes());
    result.extend(entries);
    result.extend_from_slice(b"A2B1");
    result.extend_from_slice(&a2b.unwrap()[4..]);
    result.extend_from_slice(&original[end..]);
    result
}
#[test]
fn cmyk_relative_and_fallback_tables_cover_effort_boundaries() {
    use jxl_encoder::LossyConfig;
    let root = new_output();
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("pinned native sample observer");
    let expected: Vec<f32> = include_str!("fixtures/cmyk_lcms_swatches.hex")
        .lines()
        .nth(5)
        .unwrap()
        .split_whitespace()
        .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
        .collect();
    for (name, profile) in [("fallback", profile()), ("relative", relative_profile())] {
        let metadata = ImageMetadata::new().with_icc_profile(&profile);
        for (policy, strategy, mode) in encoder_policies::policies() {
            for (width, height) in [(32_usize, 17_usize), (259, 17)] {
                for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
                    let wide = layout.is_16bit();
                    let maximum = if wide { 65535.0 } else { 255.0 };
                    let scale = if wide { 257 } else { 1 };
                    let source: Vec<u16> = [64_u16, 128, 192, 224]
                        .into_iter()
                        .map(|v| v * scale)
                        .collect::<Vec<_>>()
                        .repeat(width * height);
                    let pixels: Vec<u8> = source
                        .iter()
                        .flat_map(|v| {
                            if wide {
                                v.to_ne_bytes().to_vec()
                            } else {
                                vec![*v as u8]
                            }
                        })
                        .collect();
                    for (lossless, effort) in [
                        (true, 1),
                        (true, 7),
                        (true, 9),
                        (false, 1),
                        (false, 2),
                        (false, 3),
                        (false, 7),
                        (false, 8),
                        (false, 11),
                    ] {
                        let exact = LosslessConfig::new()
                            .with_effort(effort)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode);
                        let lossy = LossyConfig::new(0.01)
                            .with_effort(effort)
                            .with_threads(1)
                            .with_strategy(strategy.clone())
                            .with_mode(mode);
                        let request = if lossless {
                            exact.encode_request(width as u32, height as u32, layout)
                        } else {
                            lossy.encode_request(width as u32, height as u32, layout)
                        }
                        .with_metadata(&metadata);
                        let bytes = request.encode(&pixels).unwrap();
                        let label = format!(
                            "table-{name}-{policy}-{width}-{layout:?}-{lossless}-e{effort}"
                        );
                        let dir = root.join(&label);
                        std::fs::create_dir(&dir).unwrap();
                        let input = dir.join("source.jxl");
                        std::fs::write(&input, &bytes).unwrap();
                        let parsed = jxl_oxide::JxlImage::builder()
                            .read(std::io::Cursor::new(&bytes))
                            .unwrap();
                        assert_eq!(parsed.original_icc(), Some(profile.as_slice()));
                        let (colors, extras) =
                            decoded_extras::decode_current_planes(&bytes, &label);
                        let native = std::process::Command::new(&oracle)
                            .arg(&input)
                            .arg(dir.join("native"))
                            .arg(if lossless { "raw-cmy" } else { "linear" })
                            .output()
                            .unwrap();
                        std::fs::write(dir.join("native.log"), &native.stderr).unwrap();
                        assert!(native.status.success(), "{label}");
                        let native_colors =
                            pfm::read_pfm_samples(&dir.join("native-0.pfm"), (width, height));
                        let native_black =
                            pfm::read_pfm_samples(&dir.join("native-0-black.pfm"), (width, height));
                        if lossless {
                            check_codes(&colors, &extras[0], &source, maximum, &label);
                            check_codes(&native_colors, &native_black, &source, maximum, &label);
                        } else {
                            for values in [&extras[0], &native_black] {
                                assert!(
                                    values.iter().all(|v| v.is_finite()
                                        && (*v * maximum).round() as u16 == 224 * scale),
                                    "{label} exact Black"
                                );
                            }
                            for values in [&colors, &native_colors] {
                                assert!(values.iter().all(|v| v.is_finite()));
                                let error = values
                                    .iter()
                                    .zip(expected.iter().cycle())
                                    .map(|(a, b)| (a - b).abs())
                                    .fold(0_f32, f32::max);
                                assert!(error <= 0.005, "{label} physical {error}");
                            }
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn floating_cmy_with_separate_black_is_exact_lossless_and_refused_lossy() {
    use jxl_encoder::{LossyConfig, api::ExtraChannel};
    let root = new_output();
    let profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("pinned native sample observer");
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            for layout in [
                PixelLayout::RgbLinearF16,
                PixelLayout::RgbaLinearF16,
                PixelLayout::RgbLinearF32,
                PixelLayout::RgbaLinearF32,
            ] {
                for wide_black in [false, true] {
                    let alpha = layout.has_alpha();
                    let half = layout.is_f16();
                    let source = if half {
                        [0.5_f32, 0.25, 0.125, 0.5]
                    } else {
                        [0.1234567_f32, 0.2345678, 0.3456789, 0.3]
                    };
                    let channels = if alpha { 4 } else { 3 };
                    let bits = [0x3800_u16, 0x3400, 0x3000, 0x3800];
                    let pixel: Vec<u8> = if half {
                        bits[..channels]
                            .iter()
                            .flat_map(|v| v.to_ne_bytes())
                            .collect()
                    } else {
                        source[..channels]
                            .iter()
                            .flat_map(|v| v.to_ne_bytes())
                            .collect()
                    };
                    let pixels = pixel.repeat(width * height);
                    let k8 = vec![224_u8; width * height];
                    let k16 = vec![0xe0e0_u16; width * height];
                    let extras = [if wide_black {
                        ExtraChannel::black_u16(&k16)
                    } else {
                        ExtraChannel::black(&k8)
                    }];
                    let config = LosslessConfig::new()
                        .with_effort(3)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode);
                    let request = || {
                        config
                            .encode_request(width as u32, height as u32, layout)
                            .with_metadata(&metadata)
                            .with_extra_channels(&extras)
                    };
                    let bytes = request().encode(&pixels).unwrap();
                    let stride = width * pixel.len() + 7;
                    let mut padded = vec![0xa5; stride * height];
                    for y in 0..height {
                        padded[y * stride..][..width * pixel.len()].copy_from_slice(
                            &pixels[y * width * pixel.len()..][..width * pixel.len()],
                        );
                    }
                    assert_eq!(
                        bytes,
                        request().with_row_stride(stride).encode(&padded).unwrap()
                    );
                    let mut destination = Vec::new();
                    request().encode_to(&pixels, &mut destination).unwrap();
                    assert_eq!(bytes, destination);
                    let label = format!("float-{policy}-{width}-{layout:?}-{wide_black}");
                    let dir = root.join(&label);
                    std::fs::create_dir(&dir).unwrap();
                    let input = dir.join("source.jxl");
                    std::fs::write(&input, &bytes).unwrap();
                    let (color, planes) = decoded_extras::decode_current_planes(&bytes, &label);
                    let native = std::process::Command::new(&oracle)
                        .arg(&input)
                        .arg(dir.join("native"))
                        .arg("raw-cmy")
                        .output()
                        .unwrap();
                    std::fs::write(dir.join("native.log"), &native.stderr).unwrap();
                    assert!(native.status.success(), "{label}");
                    let native_color =
                        pfm::read_pfm_samples(&dir.join("native-0.pfm"), (width, height));
                    let native_black =
                        pfm::read_pfm_samples(&dir.join("native-0-black.pfm"), (width, height));
                    for values in [&color, &native_color] {
                        assert!(
                            values
                                .iter()
                                .enumerate()
                                .all(|(i, v)| v.to_bits() == source[i % 3].to_bits()),
                            "{label} exact float CMY"
                        );
                    }
                    let maximum = if wide_black { 65535.0 } else { 255.0 };
                    let code = if wide_black { 0xe0e0 } else { 224 };
                    for values in [&planes[usize::from(alpha)], &native_black] {
                        assert!(
                            values
                                .iter()
                                .all(|v| v.is_finite() && (*v * maximum).round() as u16 == code),
                            "{label} exact Black"
                        );
                    }
                    if alpha {
                        let native_alpha =
                            pfm::read_pfm_samples(&dir.join("native-0-alpha.pfm"), (width, height));
                        for values in [&planes[0], &native_alpha] {
                            assert!(
                                values.iter().all(|v| v.to_bits() == source[3].to_bits()),
                                "{label} exact alpha"
                            );
                        }
                    }
                    let config = LossyConfig::new(1.0)
                        .with_effort(3)
                        .with_strategy(strategy.clone())
                        .with_mode(mode);
                    let mut destination = b"existing destination".to_vec();
                    let original = destination.clone();
                    let error = config
                        .encode_request(width as u32, height as u32, layout)
                        .with_metadata(&metadata)
                        .with_extra_channels(&extras)
                        .encode_to(&pixels, &mut destination)
                        .unwrap_err();
                    assert!(
                        error.to_string().contains("integer RGB"),
                        "{label}: {error}"
                    );
                    assert_eq!(destination, original);
                }
            }
        }
    }
}

#[test]
fn tagged_hdr_float_cmy_with_black_reports_unsupported_lossless_and_lossy() {
    use jxl_encoder::{LossyConfig, api::ExtraChannel};
    let profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let black = vec![224; 32 * 17];
    let extras = [ExtraChannel::black(&black)];
    for (_, strategy, mode) in encoder_policies::policies() {
        for layout in [
            PixelLayout::RgbPqF32,
            PixelLayout::RgbaPqF32,
            PixelLayout::RgbHlgF32,
            PixelLayout::RgbaHlgF32,
            PixelLayout::RgbBt709F32,
            PixelLayout::RgbaBt709F32,
        ] {
            let pixels = 0.25_f32
                .to_ne_bytes()
                .repeat(32 * 17 * layout.bytes_per_pixel() / 4);
            let exact = LosslessConfig::new()
                .with_effort(3)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            let lossy = LossyConfig::new(1.0)
                .with_effort(3)
                .with_strategy(strategy.clone())
                .with_mode(mode);
            for lossless in [true, false] {
                let request = if lossless {
                    exact.encode_request(32, 17, layout)
                } else {
                    lossy.encode_request(32, 17, layout)
                }
                .with_metadata(&metadata)
                .with_extra_channels(&extras);
                let mut destination = b"existing destination".to_vec();
                let original = destination.clone();
                let error = request.encode_to(&pixels, &mut destination).unwrap_err();
                if lossless {
                    assert!(
                        format!("{error:?}").contains("UnsupportedPixelLayout"),
                        "{error}"
                    );
                } else {
                    assert!(error.to_string().contains("integer RGB"), "{error}");
                }
                assert_eq!(destination, original);
            }
        }
    }
}

#[test]
fn black_model_refusal_precedes_allocation_and_cmyk_conversion_refuses_bad_samples() {
    use jxl_encoder::{Limits, LossyConfig, api::ExtraChannel};
    let profile = profile();
    let cmyk = ImageMetadata::new().with_icc_profile(&profile);
    let rgb = ImageMetadata::new().with_icc_profile(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
    let blank = ImageMetadata::new();
    let black = vec![224_u8; 32 * 17];
    let extras = [ExtraChannel::black(&black)];
    let limits = Limits::default().with_max_memory_bytes(0);
    for (_, strategy, mode) in encoder_policies::policies() {
        let exact = LosslessConfig::new()
            .with_effort(3)
            .with_strategy(strategy.clone())
            .with_mode(mode);
        let lossy = LossyConfig::new(0.01)
            .with_effort(3)
            .with_strategy(strategy)
            .with_mode(mode);
        for (layout, metadata, attach) in [
            (PixelLayout::Rgb8, &cmyk, false),
            (PixelLayout::Rgb8, &rgb, true),
            (PixelLayout::Cmyk8, &blank, false),
        ] {
            for lossless in [false, true] {
                let pixels = vec![64_u8; 32 * 17 * layout.bytes_per_pixel()];
                let request = if lossless {
                    exact.encode_request(32, 17, layout)
                } else {
                    lossy.encode_request(32, 17, layout)
                }
                .with_metadata(metadata)
                .with_limits(&limits);
                let request = if attach {
                    request.with_extra_channels(&extras)
                } else {
                    request
                };
                let mut destination = b"existing destination".to_vec();
                let original = destination.clone();
                let error = request.encode_to(&pixels, &mut destination).unwrap_err();
                assert!(error.to_string().contains("CMYK ICC"), "{error}");
                assert_eq!(destination, original);
            }
        }
        let pixels = u16::MAX.to_ne_bytes().repeat(32 * 17 * 4);
        let mut destination = b"existing destination".to_vec();
        let original = destination.clone();
        let error = lossy
            .encode_request(32, 17, PixelLayout::Cmyk16)
            .with_metadata(&cmyk)
            .with_bits_per_sample(12)
            .encode_to(&pixels, &mut destination)
            .unwrap_err();
        assert!(error.to_string().contains("declared precision"), "{error}");
        assert_eq!(destination, original);
        let mut no_lut = profile.clone();
        let count = u32::from_be_bytes(no_lut[128..132].try_into().unwrap()) as usize;
        for tag in no_lut[132..132 + count * 12].as_chunks_mut::<12>().0 {
            if &tag[..4] == b"A2B0" {
                tag[..4].copy_from_slice(b"zzzz");
            }
        }
        let metadata = ImageMetadata::new().with_icc_profile(&no_lut);
        let pixels = vec![64_u8; 32 * 17 * 4];
        let error = lossy
            .encode_request(32, 17, PixelLayout::Cmyk8)
            .with_metadata(&metadata)
            .encode_to(&pixels, &mut destination)
            .unwrap_err();
        assert!(error.to_string().contains("CMYK"), "{error}");
        assert_eq!(destination, original);
    }
}
#[test]
fn associated_cmy_with_black_keeps_given_source_samples_and_exact_alpha() {
    use jxl_encoder::{LossyConfig, api::ExtraChannel};
    let root = new_output();
    let profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("pinned native sample observer");
    let expected: Vec<f32> = include_str!("fixtures/cmyk_lcms_swatches.hex")
        .lines()
        .nth(5)
        .unwrap()
        .split_whitespace()
        .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
        .collect();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            for layout in [PixelLayout::Rgba8, PixelLayout::Bgra8, PixelLayout::Rgba16] {
                for associated in [false, true] {
                    let wide = layout.is_16bit();
                    let scale = if wide { 257 } else { 1 };
                    let maximum = if wide { 65535.0 } else { 255.0 };
                    let mut pixel = [64_u16, 128, 192, 224];
                    if layout == PixelLayout::Bgra8 {
                        pixel.swap(0, 2);
                    }
                    let pixel: Vec<u8> = pixel
                        .into_iter()
                        .flat_map(|v| {
                            if wide {
                                (v * 257).to_ne_bytes().to_vec()
                            } else {
                                vec![v as u8]
                            }
                        })
                        .collect();
                    let pixels = pixel.repeat(width * height);
                    let k8 = vec![224_u8; width * height];
                    let k16 = vec![0xe0e0_u16; width * height];
                    let extras = [if wide {
                        ExtraChannel::black_u16(&k16)
                    } else {
                        ExtraChannel::black(&k8)
                    }];
                    let bytes = LossyConfig::new(0.01)
                        .with_effort(3)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .encode_request(width as u32, height as u32, layout)
                        .with_metadata(&metadata)
                        .with_extra_channels(&extras)
                        .with_premultiplied_alpha(associated)
                        .encode(&pixels)
                        .unwrap();
                    let parsed = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&bytes))
                        .unwrap();
                    assert_eq!(
                        parsed.image_header().metadata.ec_info[0].alpha_associated(),
                        Some(associated)
                    );
                    let label = format!("associated-{policy}-{width}-{layout:?}-{associated}");
                    let dir = root.join(&label);
                    std::fs::create_dir(&dir).unwrap();
                    let input = dir.join("source.jxl");
                    std::fs::write(&input, &bytes).unwrap();
                    let (colors, extras) = decoded_extras::decode_current_planes(&bytes, &label);
                    let native = std::process::Command::new(&oracle)
                        .arg(&input)
                        .arg(dir.join("native"))
                        .arg("linear")
                        .output()
                        .unwrap();
                    std::fs::write(dir.join("native.log"), &native.stderr).unwrap();
                    assert!(native.status.success(), "{label}");
                    let native_colors =
                        pfm::read_pfm_samples(&dir.join("native-0.pfm"), (width, height));
                    let native_black =
                        pfm::read_pfm_samples(&dir.join("native-0-black.pfm"), (width, height));
                    let native_alpha =
                        pfm::read_pfm_samples(&dir.join("native-0-alpha.pfm"), (width, height));
                    for values in [&extras[0], &extras[1], &native_alpha, &native_black] {
                        assert!(
                            values
                                .iter()
                                .all(|v| v.is_finite()
                                    && (*v * maximum).round() as u16 == 224 * scale),
                            "{label} exact alpha/Black"
                        );
                    }
                    for values in [&colors, &native_colors] {
                        assert!(values.iter().all(|v| v.is_finite()));
                        let error = values
                            .iter()
                            .zip(expected.iter().cycle())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0_f32, f32::max);
                        assert!(error <= 0.005, "{label} given-sample color {error}");
                    }
                }
            }
        }
    }
}

#[test]
fn shifted_black_storage_is_lossless_and_lossy_refuses_it() {
    use jxl_encoder::{LossyConfig, api::ExtraChannel};
    let root = new_output();
    let profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("pinned native sample observer");
    let (width, height) = (259_usize, 17_usize);
    let pixels = [64_u8, 128, 192].repeat(width * height);
    for (policy, strategy, mode) in encoder_policies::policies() {
        for shift in [0_u32, 1, 3] {
            for wide in [false, true] {
                let count = width.div_ceil(1_usize << shift) * height.div_ceil(1_usize << shift);
                let k8 = vec![224_u8; count];
                let k16 = vec![0xe0e0_u16; count];
                let extras = [if wide {
                    ExtraChannel::black_u16(&k16)
                } else {
                    ExtraChannel::black(&k8)
                }
                .with_dim_shift(shift)];
                let config = LosslessConfig::new()
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode);
                let bytes = config
                    .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                    .with_metadata(&metadata)
                    .with_extra_channels(&extras)
                    .encode(&pixels)
                    .unwrap();
                let label = format!("shifted-{policy}-{shift}-{wide}");
                let dir = root.join(&label);
                std::fs::create_dir(&dir).unwrap();
                let input = dir.join("source.jxl");
                std::fs::write(&input, &bytes).unwrap();
                let (color, planes) = decoded_extras::decode_current_planes(&bytes, &label);
                let native = std::process::Command::new(&oracle)
                    .arg(&input)
                    .arg(dir.join("native"))
                    .arg("raw-cmy")
                    .output()
                    .unwrap();
                std::fs::write(dir.join("native.log"), &native.stderr).unwrap();
                assert!(native.status.success(), "{label}");
                let native_color =
                    pfm::read_pfm_samples(&dir.join("native-0.pfm"), (width, height));
                let native_black =
                    pfm::read_pfm_samples(&dir.join("native-0-black.pfm"), (width, height));
                for values in [&color, &native_color] {
                    assert!(
                        values.iter().enumerate().all(|(i, v)| v.is_finite()
                            && (*v * 255.0).round() as u8 == [64, 128, 192][i % 3]),
                        "{label} exact CMY"
                    );
                }
                let maximum = if wide { 65535.0 } else { 255.0 };
                let code = if wide { 0xe0e0 } else { 224 };
                for values in [&planes[0], &native_black] {
                    assert!(
                        values
                            .iter()
                            .all(|v| v.is_finite() && (*v * maximum).round() as u16 == code),
                        "{label} constant Black after upsampling"
                    );
                }
                if shift != 0 {
                    let mut destination = b"existing output".to_vec();
                    let original = destination.clone();
                    let error = LossyConfig::new(1.0)
                        .with_effort(3)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                        .with_metadata(&metadata)
                        .with_extra_channels(&extras)
                        .encode_to(&pixels, &mut destination)
                        .unwrap_err();
                    assert!(error.to_string().contains("dim_shift"), "{error}");
                    assert_eq!(destination, original);
                }
            }
        }
    }
}

#[test]
fn reduced_cmyk_precision_keeps_physical_light_and_black_codes() {
    use jxl_encoder::{AnimationFrame, AnimationParams, LossyConfig};
    let output = new_output();
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("pinned native observer");
    let native = |bytes: &[u8], label: &str| {
        let dir = output.join(label);
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("source.jxl");
        std::fs::write(&path, bytes).unwrap();
        let result = std::process::Command::new(&oracle)
            .arg(path)
            .arg(dir.join("native"))
            .arg("linear")
            .output()
            .unwrap();
        std::fs::write(dir.join("native.log"), &result.stderr).unwrap();
        assert!(result.status.success(), "{label}");
        (
            pfm::read_pfm_samples(&dir.join("native-0.pfm"), (32, 17)),
            pfm::read_pfm_samples(&dir.join("native-0-black.pfm"), (32, 17)),
        )
    };
    let metadata_profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&metadata_profile);
    let width = 32_usize;
    let height = 17_usize;
    for (policy, strategy, mode) in encoder_policies::policies() {
        let config = LossyConfig::new(0.01)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(strategy)
            .with_mode(mode);
        for (layout, precisions) in [
            (PixelLayout::Cmyk8, [1_u32, 6, 8]),
            (PixelLayout::Cmyk16, [1, 12, 16]),
        ] {
            let wide = layout.is_16bit();
            let storage_maximum = if wide { 65535_u16 } else { 255 };
            for bits in precisions {
                let maximum = ((1_u32 << bits) - 1) as u16;
                // Matching rational ink fractions, including the one-bit endpoint.
                let codes = |max: u16| {
                    if bits == 1 {
                        [0, max, 0, max]
                    } else {
                        [0, max / 3, 2 * (max / 3), max]
                    }
                };
                let pack = |pixel: [u16; 4]| {
                    pixel
                        .into_iter()
                        .flat_map(|v| {
                            if wide {
                                v.to_ne_bytes().to_vec()
                            } else {
                                vec![v as u8]
                            }
                        })
                        .collect::<Vec<_>>()
                        .repeat(width * height)
                };
                let full = pack(codes(storage_maximum));
                let reduced = pack(codes(maximum));
                let prefix = format!("{policy}-{layout:?}-{bits}");
                let baseline = config
                    .encode_request(width as u32, height as u32, layout)
                    .with_metadata(&metadata)
                    .encode(&full)
                    .unwrap();
                let (expected, _) = decoded_extras::decode_current_planes(&baseline, &prefix);
                let (native_expected, _) = native(&baseline, &format!("{prefix}-full"));
                for animation in [false, true] {
                    let label = format!("{prefix}-reduced-{animation}");
                    let request = config
                        .encode_request(width as u32, height as u32, layout)
                        .with_metadata(&metadata)
                        .with_bits_per_sample(bits);
                    let bytes = if animation {
                        request
                            .encode_animation(
                                &AnimationParams::default(),
                                &[AnimationFrame::new(&reduced, 10)],
                            )
                            .unwrap()
                    } else {
                        request.encode(&reduced).unwrap()
                    };
                    let (native_colors, native_black) = native(&bytes, &label);
                    let (colors, planes) = decoded_extras::decode_current_planes(&bytes, &label);
                    for (actual, control) in
                        [(&colors, &expected), (&native_colors, &native_expected)]
                    {
                        assert_eq!(actual.len(), width * height * 3);
                        assert_eq!(control.len(), actual.len());
                        assert!(actual.iter().chain(control).all(|v| v.is_finite()));
                        let error = actual
                            .iter()
                            .zip(control)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0_f32, f32::max);
                        assert!(error <= 0.005, "{label} equivalent-ink light error {error}");
                    }
                    assert_eq!(planes.len(), 1);
                    for values in [&planes[0], &native_black] {
                        assert_eq!(values.len(), width * height);
                        assert!(
                            values.iter().all(|v| v.is_finite()
                                && (*v * f32::from(maximum)).round() as u16 == maximum),
                            "{label} Black precision"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(feature = "butteraugli-loop")]
#[test]
fn cmyk_opaque_icc_selector_does_not_select_an_hdr_source_or_loop() {
    use jxl_encoder::{
        AnimationFrame, AnimationParams, ColorEncoding, LoopTrace, LossyConfig, TransferFunction,
        api::ExtraChannel,
    };
    let output = new_output();
    let profile = profile();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);

    for (policy, strategy, mode) in encoder_policies::policies() {
        for (layout, animation) in [
            (PixelLayout::Cmyk8, false),
            (PixelLayout::Cmyk8, true),
            (PixelLayout::Cmyk16, false),
            (PixelLayout::Cmyk16, true),
            (PixelLayout::Rgb8, false),
            (PixelLayout::Rgba16, false),
        ] {
            let external = !layout.is_cmyk();
            let wide = layout.is_16bit();
            let has_alpha = layout.has_alpha();
            let mut pixel = vec![64_u8, 128, 192];
            if layout.is_cmyk() {
                pixel.push(224);
            } else if has_alpha {
                pixel.push(137);
            }
            let pixels: Vec<u8> = pixel
                .iter()
                .flat_map(|v| {
                    if wide {
                        (u16::from(*v) * 257).to_ne_bytes().to_vec()
                    } else {
                        vec![*v]
                    }
                })
                .collect::<Vec<_>>()
                .repeat(32 * 17);
            let k8 = vec![224_u8; 32 * 17];
            let k16 = vec![0xe0e0_u16; 32 * 17];
            let extras = [if wide {
                ExtraChannel::black_u16(&k16)
            } else {
                ExtraChannel::black(&k8)
            }];
            let extras = if external { &extras[..] } else { &[][..] };
            for budget in [None, Some(1)] {
                let mut config = LossyConfig::new(1.0)
                    .with_effort(8)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode);
                if let Some(n) = budget {
                    config = config.with_butteraugli_iters(n);
                }
                let trace = LoopTrace::new();
                let request = config
                    .encode_request(32, 17, layout)
                    .with_metadata(&metadata)
                    .with_extra_channels(extras)
                    .with_loop_trace(&trace);
                let baseline = if animation {
                    request
                        .encode_animation(
                            &AnimationParams::default(),
                            &[AnimationFrame::new(&pixels, 10)],
                        )
                        .unwrap()
                } else {
                    request.encode(&pixels).unwrap()
                };
                let expected: Vec<_> = trace
                    .take()
                    .into_iter()
                    .map(|frame| {
                        (
                            frame.metric,
                            frame.iter,
                            frame
                                .quant_field
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect();
                for transfer in [TransferFunction::Pq, TransferFunction::Hlg] {
                    let mut selector = ColorEncoding::srgb();
                    selector.want_icc = true;
                    selector.transfer_function = transfer;
                    selector.gamma = Some(f32::NAN);
                    let trace = LoopTrace::new();
                    let request = config
                        .encode_request(32, 17, layout)
                        .with_metadata(&metadata)
                        .with_extra_channels(extras)
                        .with_color_encoding(selector)
                        .with_loop_trace(&trace);
                    let bytes = if animation {
                        request
                            .encode_animation(
                                &AnimationParams::default(),
                                &[AnimationFrame::new(&pixels, 10)],
                            )
                            .unwrap()
                    } else {
                        request.encode(&pixels).unwrap()
                    };
                    let label =
                        format!("opaque-{policy}-{layout:?}-{animation}-{budget:?}-{transfer:?}");
                    let dir = output.join(&label);
                    std::fs::create_dir(&dir).unwrap();
                    std::fs::write(dir.join("baseline.jxl"), &baseline).unwrap();
                    std::fs::write(dir.join("selector.jxl"), &bytes).unwrap();
                    let actual: Vec<_> = trace
                        .take()
                        .into_iter()
                        .map(|frame| {
                            (
                                frame.metric,
                                frame.iter,
                                frame
                                    .quant_field
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect();
                    std::fs::write(
                        dir.join("traces.json"),
                        serde_json::to_vec(&(&expected, &actual)).unwrap(),
                    )
                    .unwrap();
                    assert_eq!(actual, expected, "{label} source/loop interpretation");
                    assert_eq!(bytes, baseline, "{label} source selector bytes");
                    let (colors, planes) = decoded_extras::decode_current_planes(&bytes, &label);
                    assert_eq!(colors.len(), 32 * 17 * 3);
                    assert!(colors.iter().all(|v| v.is_finite()));
                    let maximum = if wide { 65535.0 } else { 255.0 };
                    let code = if wide { 0xe0e0 } else { 224 };
                    assert_eq!(planes.len(), 1 + usize::from(has_alpha));
                    assert!(
                        planes
                            .last()
                            .unwrap()
                            .iter()
                            .all(|v| v.is_finite() && (*v * maximum).round() as u16 == code)
                    );
                    let native = std::process::Command::new(
                        std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("native observer"),
                    )
                    .arg(dir.join("selector.jxl"))
                    .arg(dir.join("native"))
                    .arg("linear")
                    .output()
                    .unwrap();
                    std::fs::write(dir.join("native.log"), &native.stderr).unwrap();
                    assert!(native.status.success(), "{label}");
                    let colors = pfm::read_pfm_samples(&dir.join("native-0.pfm"), (32, 17));
                    assert!(colors.iter().all(|v| v.is_finite()));
                    let black = pfm::read_pfm_samples(&dir.join("native-0-black.pfm"), (32, 17));
                    assert!(
                        black
                            .iter()
                            .all(|v| v.is_finite() && (*v * maximum).round() as u16 == code)
                    );
                }
            }
        }
    }
}
