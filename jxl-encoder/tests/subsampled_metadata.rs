//! Optional subsampled RGB8 requests retain their declared color metadata.
#![cfg(all(feature = "chroma-subsampling", feature = "jpeg-reencoding"))]

use jxl_encoder::{ChromaSubsampling, ColorEncoding, ImageMetadata, LossyConfig, PixelLayout};
use std::path::{Path, PathBuf};

#[path = "../examples/distance_targeting_probe/decode.rs"]
mod decode;

const MODES: [ChromaSubsampling; 3] = [
    ChromaSubsampling::Sub420,
    ChromaSubsampling::Sub422,
    ChromaSubsampling::Sub440,
];

fn output_directory(label: &str) -> PathBuf {
    let root = jxl_encoder::test_helpers::output_dir(label);
    (0..)
        .find_map(|i| {
            let directory = root.join(format!("run-{i}"));
            match std::fs::create_dir(&directory) {
                Ok(()) => Some(directory),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{}: {error}", directory.display()),
            }
        })
        .unwrap()
}

fn reference_decode(
    bytes: &[u8],
    root: &Path,
    cell: usize,
    color: Option<&str>,
) -> image::ImageBuffer<image::Rgb<u16>, Vec<u16>> {
    let input = root.join(format!("{cell}.jxl"));
    let output = root.join(format!("{cell}.png"));
    std::fs::write(&input, bytes).unwrap();
    let mut command = std::process::Command::new(jxl_encoder::test_helpers::djxl_path());
    command
        .arg(input)
        .arg(&output)
        .args(["--num_threads=1", "--bits_per_sample=16"]);
    if let Some(color) = color {
        command.arg(format!("--color_space={color}"));
    }
    let result = command.output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    image::open(output).unwrap().to_rgb16()
}

#[test]
fn subsampled_rgb8_keeps_hdr_color_and_tone_mapping() {
    let root = output_directory("subsampled-hdr");
    let mut cell = 0;
    for mode in MODES {
        for (width, height) in [(32, 32), (32, 17), (259, 17)] {
            let signal = [102, 128, 153];
            let pixels = signal.repeat(width * height);
            for (name, color, target, reference_color) in [
                (
                    "Pq",
                    ColorEncoding::bt2100_pq(),
                    10000.0,
                    "RGB_D65_202_Rel_PeQ",
                ),
                (
                    "Hlg",
                    ColorEncoding::bt2100_hlg(),
                    1000.0,
                    "RGB_D65_202_Rel_HLG",
                ),
            ] {
                for variant in 0..4 {
                    let mut metadata = ImageMetadata::new().with_intrinsic_size(37, 23);
                    if variant > 0 {
                        metadata = metadata
                            .with_intensity_target(4000.0)
                            .with_min_nits(0.125)
                            .with_relative_to_max_display(false)
                            .with_linear_below(0.25);
                    }
                    let config = LossyConfig::new(0.1).with_chroma_subsampling(mode);
                    let mut request = config
                        .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                        .with_color_encoding(color.clone())
                        .with_metadata(&metadata);
                    let expected_target = match variant {
                        0 => target,
                        1 => 4000.0,
                        2 => {
                            request = request.with_intensity_target(255.0);
                            255.0
                        }
                        _ => {
                            request = request
                                .with_intensity_target(1000.0)
                                .with_min_nits(0.5)
                                .with_relative_to_max_display(true)
                                .with_linear_below(0.5);
                            1000.0
                        }
                    };
                    let encoded = request.encode(&pixels).unwrap();
                    let image = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&encoded))
                        .unwrap();
                    let header = &image.image_header().metadata;
                    let actual = format!("{:?}", header.colour_encoding);
                    assert!(
                        actual.contains(name),
                        "{mode:?}: expected {name}, got {actual}"
                    );
                    assert_eq!(header.tone_mapping.intensity_target, expected_target);
                    assert_eq!(
                        header.tone_mapping.min_nits,
                        if variant == 0 {
                            0.0
                        } else if variant == 3 {
                            0.5
                        } else {
                            0.125
                        }
                    );
                    assert_eq!(header.tone_mapping.relative_to_max_display, variant == 3);
                    assert_eq!(
                        header.tone_mapping.linear_below,
                        if variant == 0 {
                            0.0
                        } else if variant == 3 {
                            0.5
                        } else {
                            0.25
                        }
                    );
                    let intrinsic = header.intrinsic_size.as_ref().unwrap();
                    assert_eq!((intrinsic.width, intrinsic.height), (37, 23));
                    image.render_frame(0).unwrap();
                    let native = decode::verify_jxl_rs(&encoded, width, height);
                    let reference = reference_decode(&encoded, &root, cell, Some(reference_color));
                    assert_eq!(reference.dimensions(), (width as u32, height as u32));
                    for (i, &actual) in reference.as_raw().iter().enumerate() {
                        let expected = f32::from(signal[i % 3]) / 255.0;
                        assert!(
                            (f32::from(actual) / 65535.0 - expected).abs() < 0.005,
                            "djxl cell {cell}: sample {i}: {actual} vs {expected}"
                        );
                    }
                    // The upstream Rust decoder's HLG output conversion has a
                    // separate tracked conformance discrepancy (DECODER_FINDINGS Z1).
                    if name == "Pq" {
                        for (i, &actual) in native.iter().enumerate() {
                            let expected = f32::from(signal[i % 3]) / 255.0;
                            assert!(
                                (actual - expected).abs() < 0.005,
                                "native cell {cell}: {actual} vs {expected}"
                            );
                        }
                    }
                    cell += 1;
                }
            }
        }
    }
    assert_eq!(cell, 72);
}

#[test]
fn subsampled_rgb8_keeps_icc_profile() {
    let root = output_directory("subsampled-icc");
    let profile = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let metadata = ImageMetadata::new().with_icc_profile(profile);
    let mut cell = 0;
    for mode in MODES {
        for (width, height) in [(32, 32), (32, 17), (259, 17)] {
            let pixels = [102, 128, 153].repeat(width * height);
            let encoded = LossyConfig::new(0.1)
                .with_chroma_subsampling(mode)
                .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                .with_metadata(&metadata)
                .encode(&pixels)
                .unwrap();
            let image = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&encoded))
                .unwrap();
            assert_eq!(image.original_icc(), Some(profile), "{mode:?}");
            image.render_frame(0).unwrap();
            decode::verify_jxl_rs(&encoded, width, height);
            let reference = reference_decode(&encoded, &root, cell, None);
            assert_eq!(reference.dimensions(), (width as u32, height as u32));
            cell += 1;
        }
    }
    assert_eq!(cell, 9);
}

#[test]
fn subsampled_rgb8_gamma_and_incompatible_color_encoding() {
    let pixels = [102, 128, 153].repeat(32 * 17);
    for mode in MODES {
        let config = LossyConfig::new(0.1).with_chroma_subsampling(mode);
        let base = config
            .encode_request(32, 17, PixelLayout::Rgb8)
            .with_source_gamma(0.5);
        let explicit = config
            .encode_request(32, 17, PixelLayout::Rgb8)
            .with_source_gamma(0.5)
            .with_color_encoding(ColorEncoding::with_gamma(0.5))
            .encode(&pixels)
            .unwrap();
        assert_eq!(base.encode(&pixels).unwrap(), explicit);
        assert_ne!(
            config
                .encode_request(32, 17, PixelLayout::Rgb8)
                .encode(&pixels)
                .unwrap(),
            explicit
        );
        let image = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&explicit))
            .unwrap();
        let color = format!("{:?}", image.image_header().metadata.colour_encoding);
        assert!(color.contains("Gamma"), "{color}");
        image.render_frame(0).unwrap();
        decode::verify_jxl_rs(&explicit, 32, 17);
        assert!(
            config
                .encode_request(32, 17, PixelLayout::Rgb8)
                .with_source_gamma(0.5)
                .with_color_encoding(ColorEncoding::bt2100_pq())
                .encode(&pixels)
                .is_err()
        );
        assert!(
            config
                .encode_request(32, 17, PixelLayout::Rgb8)
                .with_color_encoding(ColorEncoding::gray())
                .encode(&pixels)
                .is_err()
        );
    }
}
