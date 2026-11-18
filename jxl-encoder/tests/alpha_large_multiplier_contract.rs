//! High-precision alpha multipliers through the shared modular tree writer.
#![cfg(feature = "__expert")]
use jxl_encoder::{LossyConfig, PixelLayout};
use std::path::PathBuf;
#[path = "support/alpha_samples.rs"]
mod alpha_samples;
use alpha_samples::{native_alpha, source};
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/pfm.rs"]
mod pfm;
#[cfg(feature = "rate-control")]
#[path = "support/encoder_policies.rs"]
mod policies;
fn directory() -> PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("alpha-large-multiplier-contract");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}

fn config() -> LossyConfig {
    LossyConfig::new(1.0)
        .with_effort(3)
        .with_threads(1)
        .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl)
        .with_force_strategy(Some(0))
        .with_patches(false)
        .with_gaborish(false)
        .with_epf_level(0)
        .with_simplify_invisible(false)
        .with_alpha_distance(Some(10.0))
        .with_alpha_squeeze(true)
}

#[test]
fn alpha_large_multiplier_still_writer_keeps_native_planes() {
    let root = directory();
    let layout = PixelLayout::Rgba16;
    let dims = (32, 17);
    let (pixels, alpha, max) = source(layout, dims.0, dims.1, 0);
    let expected = native_alpha(&root.join("reference"), dims, &alpha, max, 10.0, true);
    let bytes = config()
        .encode(&pixels, dims.0 as u32, dims.1 as u32, layout)
        .unwrap();
    std::fs::write(root.join("input.jxl"), &bytes).unwrap();
    let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(root.join("input.jxl"))
        .arg(root.join("native.pfm"))
        .args(["--num_threads=1", "--output_extra_channels"])
        .output()
        .unwrap();
    std::fs::write(root.join("native.log"), &native.stderr).unwrap();
    let primary = std::panic::catch_unwind(|| {
        decoded_frames::decode_current_frames(&bytes, "large alpha multiplier still")
    });
    assert!(
        native.status.success() && primary.is_ok(),
        "large-alpha tree: native success={} primary success={}",
        native.status.success(),
        primary.is_ok()
    );
    let primary = primary.unwrap();
    assert_eq!(primary.len(), 1);
    let native = pfm::read_pfm_samples(&root.join("native.pfm-ec1.pfm"), dims);
    for samples in [&primary[0].1[0], &native] {
        assert_eq!(samples.len(), expected.len());
        for (&actual, &expected) in samples.iter().zip(&expected) {
            assert!(actual.is_finite());
            assert_eq!((actual * f32::from(max)).round() as i64, expected);
        }
    }
}

#[cfg(feature = "rate-control")]
#[test]
fn alpha_large_multipliers_survive_request_controller_iterations() {
    let root = directory();
    for (name, strategy, mode) in policies::policies() {
        for layout in [
            PixelLayout::Rgba16,
            PixelLayout::GrayAlpha16,
            PixelLayout::RgbaLinearF32,
        ] {
            for iterations in [0, 2] {
                let out = root.join(format!("{name}-{layout:?}-i{iterations}"));
                std::fs::create_dir(&out).unwrap();
                let dims = (32, 17);
                let (pixels, alpha, max) = source(layout, dims.0, dims.1, 0);
                let expected = native_alpha(&out.join("reference"), dims, &alpha, max, 10.0, true);
                let cfg = config().with_strategy(strategy.clone()).with_mode(mode);
                let request = || {
                    cfg.encode_request(dims.0 as u32, dims.1 as u32, layout)
                        .with_rate_control(jxl_encoder::vardct::RateControlConfig {
                            max_iterations: iterations,
                            ..Default::default()
                        })
                };
                let bytes = request().encode(&pixels).unwrap();
                let row = dims.0 * layout.bytes_per_pixel();
                let stride = row + 5;
                let mut padded = vec![0xa5; stride * dims.1];
                for (src, dst) in pixels
                    .chunks_exact(row)
                    .zip(padded.chunks_exact_mut(stride))
                {
                    dst[..row].copy_from_slice(src);
                }
                assert_eq!(
                    request().with_row_stride(stride).encode(&padded).unwrap(),
                    bytes
                );
                std::fs::write(out.join("input.jxl"), &bytes).unwrap();
                let primary =
                    decoded_frames::decode_current_frames(&bytes, "controller large alpha");
                assert_eq!(primary.len(), 1);
                assert_eq!(primary[0].1.len(), 1);
                let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(out.join("input.jxl"))
                    .arg(out.join("native.pfm"))
                    .args(["--output_extra_channels", "--num_threads=1"])
                    .output()
                    .unwrap();
                std::fs::write(out.join("native.log"), &result.stderr).unwrap();
                assert!(result.status.success());
                let native = pfm::read_pfm_samples(&out.join("native.pfm-ec1.pfm"), dims);
                for samples in [&primary[0].1[0], &native] {
                    assert_eq!(samples.len(), expected.len());
                    for (&actual, &expected) in samples.iter().zip(&expected) {
                        assert!(actual.is_finite());
                        assert_eq!((actual * f32::from(max)).round() as i64, expected);
                    }
                }
                std::fs::write(
                    out.join("passed.json"),
                    serde_json::to_vec_pretty(&serde_json::json!({
                        "policy":name, "layout":format!("{layout:?}"), "cap":iterations,
                        "source_alpha_bits":16, "complete_native_alpha":true
                    }))
                    .unwrap(),
                )
                .unwrap();
            }
        }
    }
}
