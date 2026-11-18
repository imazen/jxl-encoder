//! Crops, references, identity deltas and metadata with selected alpha coding.
#![cfg(feature = "__expert")]
use jxl_encoder::{
    AnimationFrame, AnimationParams, BlendMode, ImageMetadata, Limits, LossyConfig, PixelLayout,
};
use std::path::{Path, PathBuf};
#[path = "support/alpha_samples.rs"]
mod alpha_samples;
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/pfm.rs"]
mod pfm;
#[path = "support/encoder_policies.rs"]
mod policies;

fn directory() -> PathBuf {
    let base = jxl_encoder::test_helpers::output_dir("animation-alpha-layers-contract");
    std::fs::create_dir_all(&base).unwrap();
    (0..)
        .find_map(|i| {
            let path = base.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{error}"),
            }
        })
        .unwrap()
}

fn config() -> LossyConfig {
    LossyConfig::new(1.0)
        .with_effort(3)
        .with_threads(1)
        .with_force_strategy(Some(0))
        .with_patches(false)
        .with_gaborish(false)
        .with_epf_level(0)
        .with_simplify_invisible(false)
        .with_alpha_distance(Some(10.0))
        .with_alpha_squeeze(true)
}

fn verify(out: &Path, bytes: &[u8], dims: (usize, usize), max: u16, expected: &[Vec<i64>]) {
    std::fs::write(out.join("input.jxl"), bytes).unwrap();
    let primary = decoded_frames::decode_current_frames(bytes, "alpha layers");
    assert_eq!(primary.len(), expected.len());
    let oracle = std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
        .expect("caller-selected native float observer");
    let version = std::process::Command::new(&oracle)
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).unwrap().trim(),
        "libjxl original-profile oracle 12000"
    );
    let result = std::process::Command::new(oracle)
        .arg(out.join("input.jxl"))
        .arg(out.join("native"))
        .arg("--original-profile-float")
        .output()
        .unwrap();
    std::fs::write(
        out.join("native.log"),
        [&result.stdout[..], &result.stderr[..]].concat(),
    )
    .unwrap();
    assert!(result.status.success(), "native full frames: {out:?}");
    assert_eq!(
        String::from_utf8_lossy(&result.stdout).trim(),
        format!("{} frames", expected.len())
    );
    for (index, reference) in expected.iter().enumerate() {
        let color = pfm::read_pfm_samples(&out.join(format!("native-{index}.pfm")), dims);
        assert_eq!(primary[index].0.len(), color.len());
        assert_eq!(primary[index].1.len(), 1);
        let native = pfm::read_pfm_samples(&out.join(format!("native-{index}-ec0.pfm")), dims);
        for samples in [&primary[index].1[0], &native] {
            assert_eq!(samples.len(), reference.len());
            for (sample, (&actual, &expected)) in samples.iter().zip(reference).enumerate() {
                assert!(actual.is_finite());
                assert_eq!(
                    (actual * f32::from(max)).round() as i64,
                    expected,
                    "{out:?} frame{index} sample{sample}"
                );
            }
        }
    }
    std::fs::write(
        out.join("passed.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "frames":expected.len(), "width":dims.0,"height":dims.1,"max_code":max
        }))
        .unwrap(),
    )
    .unwrap();
}

#[test]
fn alpha_animation_crops_identity_and_reference_layers_keep_selected_planes() {
    let root = directory();
    let dims = (257_usize, 17_usize);
    let layout = PixelLayout::Rgba16;
    let (first, alpha0, max) = alpha_samples::source(layout, dims.0, dims.1, 0);
    let (second, alpha1, _) = alpha_samples::source(layout, dims.0, dims.1, 73);
    let expected0 =
        alpha_samples::native_alpha(&root.join("reference-0"), dims, &alpha0, max, 10.0, true);
    let expected1 =
        alpha_samples::native_alpha(&root.join("reference-1"), dims, &alpha1, max, 10.0, true);
    let alpha_crop: Vec<_> = (0..8)
        .flat_map(|y| alpha0[y * dims.0..y * dims.0 + 8].iter().copied())
        .collect();
    let crop = alpha_samples::native_alpha(
        &root.join("reference-crop"),
        (8, 8),
        &alpha_crop,
        max,
        10.0,
        true,
    );
    let mut cropped = expected0.clone();
    for y in 0..8 {
        cropped[y * dims.0..y * dims.0 + 8].copy_from_slice(&crop[y * 8..y * 8 + 8]);
    }
    let mut envelopes = Vec::new();
    for (name, strategy, mode) in policies::policies() {
        for kind in ["crop", "identity", "hidden", "add"] {
            let out = root.join(format!("{name}-{kind}"));
            std::fs::create_dir(&out).unwrap();
            let cfg = config()
                .with_strategy(strategy.clone())
                .with_mode(mode)
                .with_auto_delta_frames(kind == "identity");
            let params = AnimationParams::default();
            let (frames, expected) = match kind {
                "crop" => (
                    vec![
                        AnimationFrame::new(&first, 10),
                        AnimationFrame::new(&first, 20),
                    ],
                    vec![expected0.clone(), cropped.clone()],
                ),
                "identity" => (
                    vec![
                        AnimationFrame::new(&first, 10),
                        AnimationFrame::new(&first, 20),
                    ],
                    vec![expected0.clone(), expected0.clone()],
                ),
                "hidden" => (
                    vec![
                        AnimationFrame::new(&first, 0)
                            .with_reference_only(true)
                            .with_save_as_reference(2),
                        AnimationFrame::new(&second, 20)
                            .with_blend_mode(BlendMode::Add)
                            .with_blend_source(2),
                    ],
                    vec![expected1.clone()],
                ),
                "add" => (
                    vec![
                        AnimationFrame::new(&first, 10),
                        AnimationFrame::new(&second, 20)
                            .with_blend_mode(BlendMode::Add)
                            .with_blend_source(1),
                    ],
                    vec![expected0.clone(), expected1.clone()],
                ),
                _ => unreachable!(),
            };
            let bytes = cfg
                .encode_request(dims.0 as u32, dims.1 as u32, layout)
                .encode_animation(&params, &frames)
                .unwrap();
            assert_eq!(
                cfg.encode_animation(dims.0 as u32, dims.1 as u32, layout, &params, &frames)
                    .unwrap(),
                bytes
            );
            verify(&out, &bytes, dims, max, &expected);
            envelopes.push(serde_json::json!({"path":out.join("input.jxl"),"frames":2,
                "reference":kind=="hidden","hidden_index":0,"slot":2,"blend":"Add","source_before_ct":false,
                "partial_area":kind=="crop" || kind=="identity",
                "assert_blend":if kind=="identity" || kind=="hidden" || kind=="add" {"Add"} else {"Replace"},
                "assert_extra_blends":[{"mode":if kind=="identity" {"Add"} else {"Replace"},"source":if kind=="crop" || kind=="identity" {1} else {0}}]}));
        }
    }
    let manifest = root.join("envelopes.json");
    std::fs::write(&manifest, serde_json::to_vec_pretty(&envelopes).unwrap()).unwrap();
    let result = std::process::Command::new("python3")
        .arg(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../scripts/check_explicit_blend_headers.py"),
        )
        .arg(manifest)
        .arg(root.join("observed-envelopes.json"))
        .output()
        .unwrap();
    std::fs::write(
        root.join("envelope.log"),
        [result.stdout, result.stderr].concat(),
    )
    .unwrap();
    assert!(
        result.status.success(),
        "actual alpha/crop/reference envelopes"
    );
}

#[test]
fn alpha_animation_metadata_limits_and_association_keep_selected_planes() {
    let root = directory();
    let dims = (257, 17);
    let layout = PixelLayout::Rgba16;
    let (first, alpha0, max) = alpha_samples::source(layout, dims.0, dims.1, 0);
    let (second, alpha1, _) = alpha_samples::source(layout, dims.0, dims.1, 73);
    let expected = [
        alpha_samples::native_alpha(&root.join("reference-0"), dims, &alpha0, max, 10.0, true),
        alpha_samples::native_alpha(&root.join("reference-1"), dims, &alpha1, max, 10.0, true),
    ];
    let frames = [
        AnimationFrame::new(&first, 10).with_blend_mode(BlendMode::Replace),
        AnimationFrame::new(&second, 20).with_blend_mode(BlendMode::Replace),
    ];
    let metadata =
        ImageMetadata::default().with_icc_profile(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
    let limits = Limits::default();
    let associate = |pixels: &[u8]| {
        let mut values = pixels.to_vec();
        for pixel in values.as_chunks_mut::<8>().0 {
            let alpha = u32::from(u16::from_ne_bytes([pixel[6], pixel[7]]));
            for sample in pixel[..6].as_chunks_mut::<2>().0 {
                let code = u32::from(u16::from_ne_bytes([sample[0], sample[1]]));
                sample.copy_from_slice(&(((code * alpha + 32767) / 65535) as u16).to_ne_bytes());
            }
        }
        values
    };
    let associated0 = associate(&first);
    let associated1 = associate(&second);
    let associated_frames = [
        AnimationFrame::new(&associated0, 10).with_blend_mode(BlendMode::Replace),
        AnimationFrame::new(&associated1, 20).with_blend_mode(BlendMode::Replace),
    ];
    for (name, strategy, mode) in policies::policies() {
        for entry in ["metadata", "limits", "associated"] {
            let out = root.join(format!("{name}-{entry}"));
            std::fs::create_dir(&out).unwrap();
            let cfg = config().with_strategy(strategy.clone()).with_mode(mode);
            let params = AnimationParams {
                premultiplied_alpha: entry == "associated",
                ..Default::default()
            };
            let bytes = match entry {
                "metadata" => {
                    let bytes = cfg
                        .encode_animation_with_metadata(
                            dims.0 as u32,
                            dims.1 as u32,
                            layout,
                            &params,
                            &frames,
                            &metadata,
                        )
                        .unwrap();
                    assert_eq!(
                        cfg.encode_request(dims.0 as u32, dims.1 as u32, layout)
                            .with_metadata(&metadata)
                            .encode_animation(&params, &frames)
                            .unwrap(),
                        bytes
                    );
                    let parsed = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&bytes))
                        .unwrap();
                    assert_eq!(parsed.original_icc(), metadata.icc_profile());
                    bytes
                }
                "limits" => {
                    let bytes = cfg
                        .encode_animation_with_limits(
                            dims.0 as u32,
                            dims.1 as u32,
                            layout,
                            &params,
                            &frames,
                            &limits,
                        )
                        .unwrap();
                    assert_eq!(
                        cfg.encode_request(dims.0 as u32, dims.1 as u32, layout)
                            .with_limits(&limits)
                            .encode_animation(&params, &frames)
                            .unwrap(),
                        bytes
                    );
                    bytes
                }
                "associated" => cfg
                    .encode_animation(
                        dims.0 as u32,
                        dims.1 as u32,
                        layout,
                        &params,
                        &associated_frames,
                    )
                    .unwrap(),
                _ => unreachable!(),
            };
            verify(&out, &bytes, dims, max, &expected);
        }
    }
}
