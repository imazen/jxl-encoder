//! Edge-tile alpha quantizer selection and complete decoded sample checks.
#![cfg(feature = "__expert")]
use jxl_encoder::api::{EncoderMode, EncoderStrategy};
use jxl_encoder::{LossyConfig, PixelLayout};
use sha2::{Digest, Sha256};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/pfm.rs"]
mod pfm;

fn fresh_directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("alpha-squeeze-contract");
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
fn edge_tiles_keep_the_surviving_alpha_band_quantizer() {
    check_cases(&[(1280, 1024, true), (1026, 1024, true), (1025, 1024, true)]);
}

#[test]
fn empty_corner_alpha_sections_write_no_phantom_payload() {
    check_cases(&[(257, 257, false)]);
}

fn check_cases(cases: &[(usize, usize, bool)]) {
    let root = fresh_directory();
    let mut failures = Vec::new();
    // Alternation starts in the second squeeze direction. Its detail 210 is
    // divisible by the multiplier 15; the first direction has zero detail.
    // The odd final sample is 105, preserving the constant coarse average.
    // Pinned native source-sample checks below verify this selected oracle.
    for &(width, height, rows) in cases {
        let label = format!("strict-{width}x{height}");
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
            Err(_) => failures.push(format!("{label}: primary decoder failed")),
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
