//! Source interpretation when optional non-XYB output carries an ICC profile.
use super::*;
use jxl_encoder::{ChromaSubsampling, ImageMetadata};

// Native LCMS -> RGB8 -> cjpeg q100 -> cjxl/djxl v0.12 controls.
// Quantization is part of this RGB8 path; compare physical source error
// against both the original floor and the independently observed control.
fn native_quantization_bound(label: &str, source: [f32; 3]) -> f32 {
    let row = include_str!("optional_icc_native.tsv")
        .lines()
        .find(|row| row.starts_with(&format!("{label}\t")))
        .unwrap();
    let reference: Vec<f32> = row
        .split('\t')
        .skip(1)
        .map(|x| x.parse().unwrap())
        .collect();
    let quantization_error = reference
        .iter()
        .zip(source)
        .map(|(actual, target)| (actual - target).abs())
        .fold(0.0_f32, f32::max)
        + 2e-6;
    0.005_f32.max(quantization_error)
}

fn check_primary_profile_pixels(
    out: &std::path::Path,
    pixels: &[f32],
    width: usize,
    height: usize,
    linear_fallback: Option<[f32; 3]>,
) {
    assert_eq!(pixels.len(), width * height * 3);
    // jxl 0.8 update_default_output_options preserves the original profile for
    // non-XYB. XYB with ICC instead defaults to linear sRGB for float output.
    if let Some(expected) = linear_fallback {
        let error = pixels
            .iter()
            .zip(expected.iter().cycle())
            .map(|(actual, target)| (actual - target).abs())
            .fold(0_f32, f32::max);
        std::fs::write(out.join("primary-linear-error.txt"), format!("{error}\n")).unwrap();
        assert!(
            error <= 0.005,
            "primary linear-sRGB fallback samples: {error}"
        );
        return;
    }
    let oracle = std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
        .expect("optional ICC controls require the caller-selected v0.12 original-profile oracle");
    let version = std::process::Command::new(&oracle)
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).unwrap().trim(),
        "libjxl original-profile oracle 12000"
    );
    let native = std::process::Command::new(oracle)
        .arg(out.join("output.jxl"))
        .arg(out.join("original"))
        .output()
        .unwrap();
    std::fs::write(
        out.join("original.log"),
        [native.stdout, native.stderr].concat(),
    )
    .unwrap();
    assert!(native.status.success(), "native original-profile rendering");
    let native = image::open(out.join("original-0.pam")).unwrap().to_rgba16();
    assert_eq!(native.dimensions(), (width as u32, height as u32));
    let error = pixels
        .iter()
        .zip(
            native
                .pixels()
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]]),
        )
        .map(|(actual, code)| (actual - f32::from(code) / 65535.0).abs())
        .fold(0_f32, f32::max);
    std::fs::write(out.join("primary-original-error.txt"), format!("{error}\n")).unwrap();
    assert!(error <= 0.005, "primary original-profile samples: {error}");
}

#[test]
fn optional_subsampling_keeps_explicit_source_interpretation_with_icc_output() {
    let root = directory();
    let source = [128_u8, 64, 32];
    let pixels = source.repeat(32 * 17);
    let encoding = ColorEncoding::from_cicp(11, 8, 0, true).unwrap();
    let matrix = native_basis("p3-dci");
    let expected = matrix.map(|row| {
        row.into_iter()
            .zip(source)
            .map(|(coefficient, sample)| coefficient * f32::from(sample) / 255.0)
            .sum::<f32>()
    });
    let metadata =
        ImageMetadata::new().with_icc_profile(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
    let mut journal = std::fs::File::create(root.join("optional-icc.tsv")).unwrap();
    let mut failures = Vec::new();
    for mode in [
        ChromaSubsampling::Full444,
        ChromaSubsampling::Sub420,
        ChromaSubsampling::Sub422,
        ChromaSubsampling::Sub440,
    ] {
        for icc in [false, true] {
            let config = LossyConfig::new(0.01)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(EncoderStrategy::Libjxl)
                .with_chroma_subsampling(mode);
            let mut request = config
                .encode_request(32, 17, PixelLayout::Rgb8)
                .with_color_encoding(encoding.clone());
            if icc {
                request = request.with_metadata(&metadata);
            }
            let bytes = request.encode(&pixels).unwrap();
            let out = root.join(format!("{mode:?}-{icc}"));
            std::fs::create_dir(&out).unwrap();
            let rendered = render_case(&out, &bytes, 32, 17, false);
            check_primary_profile_pixels(
                &out,
                &rendered[0][0],
                32,
                17,
                (matches!(mode, ChromaSubsampling::Full444) && icc).then_some(expected),
            );
            for frames in &rendered {
                assert_eq!(frames.len(), 1);
                assert_eq!(frames[0].len(), 32 * 17 * 3);
                assert!(frames[0].iter().all(|sample| sample.is_finite()));
            }
            let error = rendered[1][0]
                .iter()
                .zip(expected.iter().cycle())
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0_f32, f32::max);
            writeln!(
                journal,
                "{}\t{icc}\t{error}\t{:?}",
                mode.tag(),
                &rendered[1][0][..3]
            )
            .unwrap();
            journal.flush().unwrap();
            let bound = if icc && !matches!(mode, ChromaSubsampling::Full444) {
                native_quantization_bound("p3-linear", expected)
            } else {
                0.005
            };
            if error > bound {
                failures.push(format!("{} icc={icc}: {error}, bound {bound}", mode.tag()));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}; {}", root.display());
}

#[test]
fn all_offered_subsampling_converts_explicit_source_to_icc_across_request_entries() {
    let root = directory();
    let mut journal = std::fs::File::create(root.join("offered-optional-icc.tsv")).unwrap();
    let encoding = ColorEncoding::from_cicp(11, 8, 0, true).unwrap();
    let source = [128_u8, 64, 32];
    let expected = native_basis("p3-dci").map(|row| {
        row.into_iter()
            .zip(source)
            .map(|(coefficient, sample)| coefficient * f32::from(sample) / 255.0)
            .sum::<f32>()
    });
    let mut failures = Vec::new();
    let mut cell = 0;
    for (policy, strategy, mode) in encoder_policies::policies() {
        for sampling in [
            ChromaSubsampling::Sub420,
            ChromaSubsampling::Sub422,
            ChromaSubsampling::Sub440,
        ] {
            for (width, height) in [(32_usize, 17_usize), (259, 17)] {
                for (profile_name, profile) in [
                    ("p3", zenpixels_convert::icc_profiles::DISPLAY_P3_V4),
                    ("2020", zenpixels_convert::icc_profiles::REC2020_V4),
                ] {
                    let out = root.join(format!("cell-{cell}"));
                    std::fs::create_dir(&out).unwrap();
                    let metadata = ImageMetadata::new().with_icc_profile(profile);
                    let config = LossyConfig::new(0.01)
                        .with_effort(3)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_chroma_subsampling(sampling);
                    let pixels = source.repeat(width * height);
                    let stride = width * 3 + 11;
                    let mut padded = vec![0xA5; stride * height];
                    for row in 0..height {
                        padded[row * stride..][..width * 3]
                            .copy_from_slice(&pixels[row * width * 3..][..width * 3]);
                    }
                    let request = || {
                        config
                            .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                            .with_color_encoding(encoding.clone())
                            .with_metadata(&metadata)
                    };
                    let encoded = request().encode(&pixels).unwrap();
                    assert_eq!(
                        request().with_row_stride(stride).encode(&padded).unwrap(),
                        encoded,
                        "{policy}: strided source"
                    );
                    let mut destination = Vec::new();
                    request().encode_to(&pixels, &mut destination).unwrap();
                    assert_eq!(destination, encoded, "{policy}: writer source");
                    let mut stream = config
                        .encoder(width as u32, height as u32, PixelLayout::Rgb8)
                        .unwrap()
                        .with_color_encoding(encoding.clone())
                        .with_icc_profile(profile);
                    for rows in pixels.chunks(width * 3 * 7) {
                        stream
                            .push_rows(rows, (rows.len() / (width * 3)) as u32)
                            .unwrap();
                    }
                    let mut output = b"existing streaming destination".to_vec();
                    let original = output.clone();
                    let error = stream.finish_into(&mut output).unwrap_err();
                    assert!(format!("{error:?}").contains("not yet wired"), "{error:?}");
                    assert_eq!(
                        output, original,
                        "unsupported streaming preserves destination"
                    );
                    let parsed = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&encoded))
                        .unwrap();
                    assert_eq!(parsed.original_icc(), Some(profile));
                    let rendered = render_case(&out, &encoded, width, height, false);
                    check_primary_profile_pixels(&out, &rendered[0][0], width, height, None);
                    assert!(rendered.iter().all(|frames| frames.len() == 1
                        && frames[0].len() == width * height * 3
                        && frames[0].iter().all(|value| value.is_finite())));
                    let error = rendered[1][0]
                        .iter()
                        .zip(expected.iter().cycle())
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0_f32, f32::max);
                    writeln!(
                        journal,
                        "{cell}\t{policy}\t{}\t{width}\t{height}\t{profile_name}\t{error}",
                        sampling.tag()
                    )
                    .unwrap();
                    journal.flush().unwrap();
                    if error
                        > native_quantization_bound(&format!("{profile_name}-linear"), expected)
                    {
                        failures.push(format!(
                            "{policy}/{} {width}x{height} {profile_name}: {error}",
                            sampling.tag()
                        ));
                    }
                    cell += 1;
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}; {}", root.display());
}

#[test]
fn unavailable_explicit_icc_conversion_is_an_error_and_raw_storage_remains_permissive() {
    let pixels = [128_u8, 64, 32].repeat(32 * 17);
    let mut opaque = vec![0xAB_u8; 132];
    opaque[16..20].copy_from_slice(b"RGB ");
    let metadata = ImageMetadata::new().with_icc_profile(&opaque);
    for (_policy, strategy, mode) in encoder_policies::policies() {
        for sampling in [
            ChromaSubsampling::Sub420,
            ChromaSubsampling::Sub422,
            ChromaSubsampling::Sub440,
        ] {
            let config = LossyConfig::new(0.01)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(strategy.clone())
                .with_mode(mode)
                .with_chroma_subsampling(sampling);
            let request = || {
                config
                    .encode_request(32, 17, PixelLayout::Rgb8)
                    .with_metadata(&metadata)
            };
            let raw = request()
                .encode(&pixels)
                .expect("opaque storage needs no CMS conversion");
            let parsed = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&raw))
                .unwrap();
            assert_eq!(parsed.original_icc(), Some(opaque.as_slice()));
            let mut selector = ColorEncoding::srgb();
            selector.want_icc = true;
            let selected = request()
                .with_color_encoding(selector)
                .encode(&pixels)
                .expect("an explicit ICC selector still stores raw profile bytes");
            assert_eq!(
                selected, raw,
                "ICC presence selector does not request simple-source conversion"
            );
            let mut destination = b"existing output".to_vec();
            let original = destination.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                request()
                    .with_color_encoding(ColorEncoding::from_cicp(11, 8, 0, true).unwrap())
                    .encode_to(&pixels, &mut destination)
            }));
            let error = result
                .expect("unavailable conversion must not panic")
                .unwrap_err();
            assert!(format!("{error:?}").contains("ICC"), "{error:?}");
            assert_eq!(destination, original);
            let mut stream = config
                .encoder(32, 17, PixelLayout::Rgb8)
                .unwrap()
                .with_icc_profile(&opaque)
                .with_color_encoding(ColorEncoding::from_cicp(11, 8, 0, true).unwrap());
            for rows in pixels.chunks(32 * 3 * 7) {
                stream
                    .push_rows(rows, (rows.len() / (32 * 3)) as u32)
                    .unwrap();
            }
            let error = stream.finish_into(&mut destination).unwrap_err();
            assert!(format!("{error:?}").contains("not yet wired"), "{error:?}");
            assert_eq!(
                destination, original,
                "unsupported streaming preserves destination"
            );
        }
    }
}

#[test]
fn optional_icc_conversion_consumes_explicit_source_gamma_precedence() {
    let root = directory();
    let source = [128_u8, 64, 32];
    let pixels = source.repeat(32 * 17);
    let encoding = ColorEncoding::from_cicp(11, 8, 0, true).unwrap();
    let expected = native_basis("p3-dci").map(|row| {
        row.into_iter()
            .zip(source)
            .map(|(coefficient, sample)| coefficient * (f32::from(sample) / 255.0).powi(2))
            .sum::<f32>()
    });
    let metadata =
        ImageMetadata::new().with_icc_profile(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
    for (cell, sampling) in [
        ChromaSubsampling::Full444,
        ChromaSubsampling::Sub420,
        ChromaSubsampling::Sub422,
        ChromaSubsampling::Sub440,
    ]
    .into_iter()
    .enumerate()
    {
        let out = root.join(format!("gamma-{cell}"));
        std::fs::create_dir(&out).unwrap();
        let bytes = LossyConfig::new(0.01)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_chroma_subsampling(sampling)
            .encode_request(32, 17, PixelLayout::Rgb8)
            .with_color_encoding(encoding.clone())
            .with_source_gamma(0.5)
            .with_metadata(&metadata)
            .encode(&pixels)
            .unwrap();
        let rendered = render_case(&out, &bytes, 32, 17, false);
        check_primary_profile_pixels(
            &out,
            &rendered[0][0],
            32,
            17,
            matches!(sampling, ChromaSubsampling::Full444).then_some(expected),
        );
        assert!(
            rendered[1][0]
                .iter()
                .zip(expected.iter().cycle())
                .all(|(actual, expected)| (actual - expected).abs() <= 0.005),
            "{} source gamma precedes declared transfer",
            sampling.tag()
        );
    }
}

#[test]
fn optional_icc_conversion_keeps_hdr_and_transfer_units() {
    let root = directory();
    let pixels = [128_u8, 64, 32].repeat(32 * 17);
    let mut cell = 0;
    let mut failures = Vec::new();
    let mut journal = std::fs::File::create(root.join("transfer-units.tsv")).unwrap();
    for transfer in [8, 16, 18, 1] {
        for intensity in [255.0, 1000.0, 10000.0] {
            let encoding = ColorEncoding::from_cicp(11, transfer, 0, true).unwrap();
            let metadata = ImageMetadata::new()
                .with_icc_profile(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
            let base = LossyConfig::new(0.01)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(EncoderStrategy::Libjxl);
            let mut reference = None;
            for sampling in [
                ChromaSubsampling::Full444,
                ChromaSubsampling::Sub420,
                ChromaSubsampling::Sub422,
                ChromaSubsampling::Sub440,
            ] {
                let out = root.join(format!("transfer-{cell}"));
                std::fs::create_dir(&out).unwrap();
                let bytes = base
                    .clone()
                    .with_chroma_subsampling(sampling)
                    .encode_request(32, 17, PixelLayout::Rgb8)
                    .with_color_encoding(encoding.clone())
                    .with_metadata(&metadata)
                    .with_intensity_target(intensity)
                    .encode(&pixels)
                    .unwrap();
                let rendered = render_case(&out, &bytes, 32, 17, false);
                if matches!(sampling, ChromaSubsampling::Full444) {
                    reference = Some(rendered[1][0].clone());
                } else {
                    check_primary_profile_pixels(&out, &rendered[0][0], 32, 17, None);
                    let error = rendered[1][0]
                        .iter()
                        .zip(reference.as_ref().unwrap())
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0_f32, f32::max);
                    writeln!(
                        journal,
                        "{transfer}\t{intensity}\t{}\t{error}",
                        sampling.tag()
                    )
                    .unwrap();
                    journal.flush().unwrap();
                    if error
                        > native_quantization_bound(
                            &format!("p3-tf{transfer}-nits{intensity}"),
                            reference.as_ref().unwrap()[..3].try_into().unwrap(),
                        )
                    {
                        failures.push(format!(
                            "transfer{transfer}/nits{intensity}/{}: {error}",
                            sampling.tag()
                        ));
                    }
                }
                cell += 1;
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}; {}", root.display());
}

#[test]
fn explicit_source_gamma_without_simple_encoding_converts_to_icc() {
    let root = directory();
    let input = [128_u8, 64, 32];
    let expected = input.map(|code| (f32::from(code) / 255.0).powi(2));
    for (cell, sampling) in [
        ChromaSubsampling::Full444,
        ChromaSubsampling::Sub420,
        ChromaSubsampling::Sub422,
        ChromaSubsampling::Sub440,
    ]
    .into_iter()
    .enumerate()
    {
        let out = root.join(format!("gamma-only-{cell}"));
        std::fs::create_dir(&out).unwrap();
        let metadata =
            ImageMetadata::new().with_icc_profile(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
        let config = LossyConfig::new(0.01)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_chroma_subsampling(sampling);
        let bytes = config
            .encode_request(32, 17, PixelLayout::Rgb8)
            .with_source_gamma(0.5)
            .with_metadata(&metadata)
            .encode(&input.repeat(32 * 17))
            .unwrap();
        let rendered = render_case(&out, &bytes, 32, 17, false);
        check_primary_profile_pixels(
            &out,
            &rendered[0][0],
            32,
            17,
            matches!(sampling, ChromaSubsampling::Full444).then_some(expected),
        );
        let error = rendered[1][0]
            .iter()
            .zip(expected.iter().cycle())
            .map(|(actual, target)| (actual - target).abs())
            .fold(0_f32, f32::max);
        assert!(
            error <= 0.005,
            "source gamma without simple encoding: {sampling:?}: {error}"
        );
    }
}

#[test]
fn cms_overflow_is_an_error_before_output_and_raw_profile_storage_still_works() {
    let hex: String = include_str!("../fixtures/icc_extreme_gamma.hex")
        .split_whitespace()
        .collect();
    let profile: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let pixels = [255_u8; 32 * 17 * 3];
    let root = directory();
    for (cell, sampling) in [
        ChromaSubsampling::Sub420,
        ChromaSubsampling::Sub422,
        ChromaSubsampling::Sub440,
    ]
    .into_iter()
    .enumerate()
    {
        let config = LossyConfig::new(0.01)
            .with_effort(3)
            .with_threads(1)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_chroma_subsampling(sampling);
        let request = || {
            config
                .encode_request(32, 17, PixelLayout::Rgb8)
                .with_metadata(&metadata)
        };
        let raw = request().encode(&pixels).unwrap();
        let out = root.join(format!("raw-{cell}"));
        std::fs::create_dir(&out).unwrap();
        let rendered = render_case(&out, &raw, 32, 17, false);
        assert!(
            rendered[1][0]
                .iter()
                .all(|value| (value - 1.0).abs() <= 0.005)
        );
        let mut destination = b"existing destination".to_vec();
        let original = destination.clone();
        let error = request()
            .with_color_encoding(ColorEncoding::from_cicp(1, 16, 0, true).unwrap())
            .with_intensity_target(255.0)
            .encode_to(&pixels, &mut destination)
            .unwrap_err();
        assert!(format!("{error:?}").contains("non-finite"), "{error:?}");
        assert_eq!(destination, original);
    }
}
