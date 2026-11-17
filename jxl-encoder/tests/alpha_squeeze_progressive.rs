//! Progressive alpha-band placement, isolated from multiblock color transforms.
#![cfg(feature = "__expert")]
use jxl_encoder::api::{EncoderMode, EncoderStrategy};
use jxl_encoder::{LossyConfig, PixelLayout, ProgressiveMode};
use sha2::{Digest, Sha256};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/native_planes.rs"]
mod native_planes;
#[path = "support/pfm.rs"]
mod pfm;

fn fresh_directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("alpha-squeeze-progressive");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{error}"),
            }
        })
        .unwrap()
}

#[test]
fn progressive_alpha_bands_keep_the_single_pass_final_samples() {
    let mut cases = Vec::new();
    for progressive in [
        ProgressiveMode::Single,
        ProgressiveMode::QuantizedAcFullAc,
        ProgressiveMode::DcVlfLfAc,
    ] {
        for (width, height, rows) in [(128, 64, true), (1024, 1024, false), (1025, 1024, true)] {
            cases.push((width, height, rows, progressive));
        }
    }
    check_cases(&cases);
}

fn check_cases(cases: &[(usize, usize, bool, ProgressiveMode)]) {
    let root = fresh_directory();
    let mut failures = Vec::new();
    // Alternation starts in the second squeeze direction. Its detail 210 is
    // divisible by the multiplier 15; the first direction has zero detail.
    // The odd final sample is 105, preserving the constant coarse average.
    // Pinned native source-sample checks below verify this selected oracle.
    for &(width, height, rows, progressive) in cases {
        let label = format!("strict-{width}x{height}-{progressive:?}");
        let out = root.join(&label);
        std::fs::create_dir(&out).unwrap();
        let pixels: Vec<u8> = (0..width * height)
            .flat_map(|i| {
                let coordinate = if rows { i / width } else { i % width };
                let extent = if rows { height } else { width };
                let alpha = if !extent.is_multiple_of(2) && coordinate + 1 == extent {
                    105
                } else if coordinate.is_multiple_of(2) {
                    0
                } else {
                    210
                };
                [64, 128, 192, alpha]
            })
            .collect();
        std::fs::write(out.join("input.rgba8"), &pixels).unwrap();
        let mut pam = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
        )
        .into_bytes();
        pam.extend_from_slice(&pixels);
        std::fs::write(out.join("input.pam"), pam).unwrap();
        let native = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
            .arg(out.join("input.pam"))
            .arg(out.join("native.jxl"))
            .args([
                "-d",
                "1",
                "-e",
                "3",
                "--alpha_distance=2",
                "--responsive=1",
                "--num_threads=1",
            ])
            .args(match progressive {
                ProgressiveMode::Single => &[][..],
                ProgressiveMode::QuantizedAcFullAc => &["--qprogressive_ac"][..],
                ProgressiveMode::DcVlfLfAc => &["--progressive_ac"][..],
            })
            .output()
            .unwrap();
        std::fs::write(out.join("native-encode.log"), &native.stderr).unwrap();
        assert!(
            native.status.success(),
            "pinned native encode: {}",
            String::from_utf8_lossy(&native.stderr)
        );
        let native_decode = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(out.join("native.jxl"))
            .arg(out.join("native.pfm"))
            .args(["--output_extra_channels", "--num_threads=1"])
            .output()
            .unwrap();
        std::fs::write(out.join("native-decode.log"), &native_decode.stderr).unwrap();
        assert!(native_decode.status.success(), "pinned native rendering");
        let native_alpha = pfm::read_pfm_samples(&out.join("native.pfm-ec1.pfm"), (width, height));
        assert_eq!(native_alpha.len(), width * height);
        for (i, value) in native_alpha.iter().enumerate() {
            assert_eq!(
                (value * 255.0).round(),
                f32::from(pixels[4 * i + 3]),
                "native source control {label} sample {i}"
            );
        }
        let bytes = LossyConfig::new(1.0)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_mode(EncoderMode::Reference)
            .with_effort(3)
            .with_threads(1)
            .with_force_strategy(Some(0))
            .with_progressive(progressive)
            .with_alpha_distance(Some(2.0))
            .with_alpha_squeeze(true)
            .encode(&pixels, width as u32, height as u32, PixelLayout::Rgba8)
            .unwrap();
        std::fs::write(out.join("input.jxl"), &bytes).unwrap();
        std::fs::write(
            out.join("input.sha256"),
            Sha256::digest(&bytes)
                .iter()
                .map(|v| format!("{v:02x}"))
                .collect::<String>(),
        )
        .unwrap();
        let primary =
            std::panic::catch_unwind(|| decoded_extras::decode_current_planes(&bytes, &label));
        let reference = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
            .arg(out.join("input.jxl"))
            .arg(out.join("output.pfm"))
            .args(["--output_extra_channels", "--num_threads=1"])
            .output()
            .unwrap();
        std::fs::write(out.join("decode.log"), &reference.stderr).unwrap();
        let mut observations = Vec::new();
        match primary {
            Ok((color, planes)) => {
                assert!(color.iter().all(|v| v.is_finite()));
                assert_eq!(planes.len(), 1);
                observations.push(("jxl-rs", planes[0].clone()));
            }
            Err(error) => {
                let message = error
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| error.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                assert!(
                    matches!((width, height), (131073, 17) | (17, 131073))
                        && message.contains("row < self.num_rows"),
                    "unclassified primary decoder failure {label}: {message}"
                );
                std::fs::write(out.join("decoder-gap.txt"), format!(
                    "Registry jxl 0.8 global-band row-bound panic: {message}. Native repaired decoder and djxl require exact alpha.\n")).unwrap();
                let (color, planes) = native_planes::decode(&bytes);
                assert!(color.iter().all(|v| v.is_finite()));
                assert_eq!(planes.len(), 1);
                observations.push(("pinned-native-jxl-rs", planes[0].clone()));
            }
        }
        if reference.status.success() {
            observations.push((
                "djxl",
                pfm::read_pfm_samples(&out.join("output.pfm-ec1.pfm"), (width, height)),
            ));
        } else {
            failures.push(format!("{label}: reference decoder failed"));
        }
        for (decoder, alpha) in observations {
            assert_eq!(alpha.len(), width * height);
            let wrong: Vec<_> = alpha
                .iter()
                .enumerate()
                .filter_map(|(i, value)| {
                    let expected = pixels[i * 4 + 3];
                    let actual = (value * 255.0).round();
                    (!value.is_finite() || actual != f32::from(expected))
                        .then_some((i, expected, *value))
                })
                .collect();
            let first = wrong.first();
            let record = format!("wrong={} first={first:?}\n", wrong.len());
            std::fs::write(out.join(format!("{decoder}-samples.txt")), &record).unwrap();
            if !wrong.is_empty() {
                failures.push(format!("{label} {decoder}: {record}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}; {}", root.display());
}

#[test]
fn progressive_without_extras_keeps_single_pass_color_samples() {
    let root = fresh_directory();
    for (width, height) in [(128, 64), (257, 257), (1025, 17)] {
        let pixels: Vec<u8> = (0..width * height).flat_map(|_| [64, 128, 192]).collect();
        let mut primary_control = None;
        let mut reference_control = None;
        for progressive in [
            ProgressiveMode::Single,
            ProgressiveMode::QuantizedAcFullAc,
            ProgressiveMode::DcVlfLfAc,
        ] {
            let label = format!("no-extras-{width}x{height}-{progressive:?}");
            let out = root.join(&label);
            std::fs::create_dir(&out).unwrap();
            let bytes = LossyConfig::new(1.0)
                .with_strategy(EncoderStrategy::Libjxl)
                .with_effort(3)
                .with_force_strategy(Some(0))
                .with_progressive(progressive)
                .with_threads(1)
                .encode(&pixels, width, height, PixelLayout::Rgb8)
                .unwrap();
            std::fs::write(out.join("input.jxl"), &bytes).unwrap();
            let (color, extra) = decoded_extras::decode_current_planes(&bytes, &label);
            assert!(extra.is_empty());
            assert!(color.iter().all(|v| v.is_finite()));
            let current: Vec<u32> = color.iter().map(|v| v.to_bits()).collect();
            if let Some(expected) = &primary_control {
                assert_eq!(&current, expected);
            } else {
                primary_control = Some(current);
            }
            let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                .arg(out.join("input.jxl"))
                .arg(out.join("output.pfm"))
                .args(["--num_threads=1"])
                .output()
                .unwrap();
            std::fs::write(out.join("decode.log"), &result.stderr).unwrap();
            assert!(result.status.success());
            let reference =
                pfm::read_pfm_samples(&out.join("output.pfm"), (width as usize, height as usize));
            assert!(reference.iter().all(|v| v.is_finite()));
            let current: Vec<u32> = reference.iter().map(|v| v.to_bits()).collect();
            if let Some(expected) = &reference_control {
                assert_eq!(&current, expected);
            } else {
                reference_control = Some(current);
            }
        }
    }
}

#[test]
fn zz_elongated_alpha_groups_keep_source_samples() {
    let mut cases = Vec::new();
    for progressive in [
        ProgressiveMode::Single,
        ProgressiveMode::QuantizedAcFullAc,
        ProgressiveMode::DcVlfLfAc,
    ] {
        for (width, height, rows) in [
            (131073, 17, true),
            (17, 131073, false),
            (131585, 17, true),
            (17, 131585, false),
        ] {
            cases.push((width, height, rows, progressive));
        }
    }
    check_cases(&cases);
}
