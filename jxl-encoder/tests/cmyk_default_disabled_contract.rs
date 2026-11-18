//! CMYK source behavior with the std Cargo feature disabled, and its std control.
#![cfg(feature = "__expert")]
use jxl_encoder::{ImageMetadata, LosslessConfig, LossyConfig, PixelLayout};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;
#[test]
fn cmyk_lossless_storage_and_explicit_conversion_availability() {
    let root = jxl_encoder::test_helpers::output_dir("cmyk-default-disabled");
    let out = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    std::fs::write(
        out.join("feature.txt"),
        format!("std Cargo feature: {}\n", cfg!(feature = "std")),
    )
    .unwrap();
    let profile: Vec<u8> = include_str!("fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|line| {
            (0..line.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
        })
        .collect();
    let metadata = ImageMetadata::new().with_icc_profile(&profile);
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("pinned native sample observer");
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (width, height) in [(32_usize, 17_usize), (259, 17)] {
            for layout in [PixelLayout::Cmyk8, PixelLayout::Cmyk16] {
                let wide = layout.is_16bit();
                let maximum = if wide { 65535.0 } else { 255.0 };
                let source: Vec<u16> = (0..width * height * 4)
                    .map(|i| {
                        if wide {
                            (i as u16).wrapping_mul(0x1f31).wrapping_add(0x534d)
                        } else {
                            ((i * 71 + i / 4 * 17) % 256) as u16
                        }
                    })
                    .collect();
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
                let exact = LosslessConfig::new()
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode);
                let request = || {
                    exact
                        .encode_request(width as u32, height as u32, layout)
                        .with_metadata(&metadata)
                };
                let bytes = request().encode(&pixels).unwrap();
                let stride = width * layout.bytes_per_pixel() + 7;
                let mut padded = vec![0xa5; stride * height];
                for y in 0..height {
                    padded[y * stride..][..width * layout.bytes_per_pixel()].copy_from_slice(
                        &pixels[y * width * layout.bytes_per_pixel()..]
                            [..width * layout.bytes_per_pixel()],
                    );
                }
                assert_eq!(
                    bytes,
                    request().with_row_stride(stride).encode(&padded).unwrap()
                );
                let mut writer = Vec::new();
                request().encode_into(&pixels, &mut writer).unwrap();
                assert_eq!(bytes, writer);
                let label = format!("{policy}-{width}-{layout:?}");
                let dir = out.join(&label);
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
                    assert_eq!(values.len(), width * height * 3);
                    for (i, v) in values.iter().enumerate() {
                        assert!(v.is_finite());
                        assert_eq!(
                            (*v * maximum).round() as u16,
                            source[i / 3 * 4 + i % 3],
                            "{label} CMY"
                        );
                    }
                }
                for values in [&planes[0], &native_black] {
                    assert_eq!(values.len(), width * height);
                    for (i, v) in values.iter().enumerate() {
                        assert!(v.is_finite());
                        assert_eq!(
                            (*v * maximum).round() as u16,
                            source[i * 4 + 3],
                            "{label} Black"
                        );
                    }
                }
                let lossy = LossyConfig::new(0.01)
                    .with_effort(3)
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode);
                let mut destination = b"existing destination".to_vec();
                let original = destination.clone();
                let result = lossy
                    .encode_request(width as u32, height as u32, layout)
                    .with_metadata(&metadata)
                    .encode_into(&pixels, &mut destination);
                if cfg!(feature = "std") {
                    result.unwrap();
                    let _ = decoded_extras::decode_current_planes(
                        &destination[original.len()..],
                        &label,
                    );
                } else {
                    let error = result.unwrap_err();
                    assert!(error.to_string().contains("requires std"), "{error}");
                    assert_eq!(destination, original);
                }
            }
        }
    }
}
