//! Animation alpha controls against independently encoded v0.12 alpha planes.
#![cfg(feature = "__expert")]
use jxl_encoder::{AnimationFrame, AnimationParams, BlendMode, LossyConfig, PixelLayout};
use std::path::{Path, PathBuf};
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/pfm.rs"]
mod pfm;
#[path = "support/encoder_policies.rs"]
mod policies;

fn directory() -> PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("animation-alpha-contract");
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

#[path = "support/alpha_samples.rs"]
mod alpha_samples;
use alpha_samples::{native_alpha, source};

fn check(root: &Path, label: &str, config: LossyConfig, layout: PixelLayout, dims: (usize, usize)) {
    let out = root.join(label);
    std::fs::create_dir(&out).unwrap();
    let (first, alpha0, max) = source(layout, dims.0, dims.1, 0);
    let (second, alpha1, _) = source(layout, dims.0, dims.1, 73);
    let distance = config.alpha_distance().unwrap_or(0.0);
    let expected = [
        native_alpha(
            &out.join("reference-0"),
            dims,
            &alpha0,
            max,
            distance,
            config.alpha_squeeze(),
        ),
        native_alpha(
            &out.join("reference-1"),
            dims,
            &alpha1,
            max,
            distance,
            config.alpha_squeeze(),
        ),
    ];
    if distance > 0.0 {
        assert!(
            expected[0]
                .iter()
                .zip(&alpha0)
                .any(|(&a, &b)| a != i64::from(b)),
            "native control must actually quantize alpha"
        );
    }
    let frames = [
        AnimationFrame::new(&first, 10).with_blend_mode(BlendMode::Replace),
        AnimationFrame::new(&second, 20).with_blend_mode(BlendMode::Replace),
    ];
    let params = AnimationParams::default();
    let bytes = config
        .encode_request(dims.0 as u32, dims.1 as u32, layout)
        .encode_animation(&params, &frames)
        .unwrap();
    assert_eq!(
        config
            .encode_animation(dims.0 as u32, dims.1 as u32, layout, &params, &frames)
            .unwrap(),
        bytes
    );
    std::fs::write(out.join("input.jxl"), &bytes).unwrap();
    let primary = decoded_frames::decode_current_frames(&bytes, label);
    assert_eq!(primary.len(), 2);
    let oracle = std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
        .expect("caller-selected v0.12 float color/extra-plane oracle");
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
        .arg(out.join("input.jxl"))
        .arg(out.join("native"))
        .arg("--source-fields-float")
        .output()
        .unwrap();
    std::fs::write(
        out.join("native.log"),
        [native.stdout, native.stderr].concat(),
    )
    .unwrap();
    assert!(
        native.status.success(),
        "native complete animation: {label}"
    );
    for frame in 0..2 {
        let color = pfm::read_pfm_samples(&out.join(format!("native-{frame}.pfm")), dims);
        assert_eq!(primary[frame].0.len(), color.len());
        assert_eq!(primary[frame].1.len(), 1);
        let native = pfm::read_pfm_samples(&out.join(format!("native-{frame}-ec0.pfm")), dims);
        assert_eq!(native.len(), expected[frame].len());
        assert_eq!(primary[frame].1[0].len(), expected[frame].len());
        for (decoder, samples) in [("primary", &primary[frame].1[0]), ("native", &native)] {
            for (index, (&actual, &expected)) in samples.iter().zip(&expected[frame]).enumerate() {
                assert!(actual.is_finite());
                assert_eq!(
                    (actual * f32::from(max)).round() as i64,
                    expected,
                    "{label}: {decoder} frame{frame} alpha sample{index}"
                );
                if distance <= 0.0 && (layout.is_f16() || layout.is_f32()) {
                    let code = if frame == 0 {
                        alpha0[index]
                    } else {
                        alpha1[index]
                    };
                    let original = match code {
                        0 => 0.0_f32,
                        32768 => 0.5,
                        _ => 1.0,
                    };
                    assert_eq!(actual.to_bits(), original.to_bits(), "lossless float alpha");
                }
            }
        }
    }
    std::fs::write(
        out.join("passed.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "layout":format!("{layout:?}"), "width":dims.0, "height":dims.1,
            "alpha_distance":distance, "squeeze":config.alpha_squeeze(),
            "displayed_frames":2, "independent_native_alpha":true
        }))
        .unwrap(),
    )
    .unwrap();
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
fn animation_alpha_distance_and_squeeze_reach_integer_frames() {
    check(
        &directory(),
        "strict-rgba8",
        config(),
        PixelLayout::Rgba8,
        (32, 17),
    );
}

#[test]
fn animation_alpha_distance_selects_integer_coding_for_float_frames() {
    check(
        &directory(),
        "strict-rgbaf32",
        config(),
        PixelLayout::RgbaLinearF32,
        (32, 17),
    );
}

#[test]
fn animation_alpha_policies_and_progressive_brackets_keep_native_planes() {
    let root = directory();
    for (name, strategy, mode) in policies::policies() {
        for progressive in [
            jxl_encoder::ProgressiveMode::Single,
            jxl_encoder::ProgressiveMode::QuantizedAcFullAc,
            jxl_encoder::ProgressiveMode::DcVlfLfAc,
        ] {
            for layout in [PixelLayout::Rgba8, PixelLayout::GrayAlpha16] {
                check(
                    &root,
                    &format!("{name}-{progressive:?}-{layout:?}"),
                    config()
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_progressive(progressive),
                    layout,
                    (32, 17),
                );
            }
        }
    }
}

#[test]
fn animation_alpha_layout_and_group_edges_keep_native_planes() {
    let root = directory();
    for layout in [
        PixelLayout::Rgba8,
        PixelLayout::Bgra8,
        PixelLayout::GrayAlpha8,
        PixelLayout::Rgba16,
        PixelLayout::GrayAlpha16,
        PixelLayout::RgbaLinearF16,
        PixelLayout::GrayAlphaLinearF16,
        PixelLayout::RgbaLinearF32,
        PixelLayout::GrayAlphaLinearF32,
        PixelLayout::RgbaPqF32,
        PixelLayout::RgbaHlgF32,
        PixelLayout::RgbaBt709F32,
    ] {
        check(
            &root,
            &format!("layout-{layout:?}"),
            config(),
            layout,
            (32, 17),
        );
    }
    for dims in [(257, 17), (1025, 17), (17, 1025), (2049, 17), (2049, 2048)] {
        check(
            &root,
            &format!("edge-{}x{}", dims.0, dims.1),
            config().with_alpha_distance(Some(25.0)),
            PixelLayout::Rgba8,
            dims,
        );
    }
}

#[test]
fn animation_alpha_zero_distance_and_unsqueezed_controls() {
    let root = directory();
    for layout in [
        PixelLayout::Rgba8,
        PixelLayout::GrayAlpha16,
        PixelLayout::RgbaLinearF32,
    ] {
        for distance in [None, Some(0.0)] {
            for squeeze in [false, true] {
                check(
                    &root,
                    &format!("{layout:?}-d{distance:?}-sq{squeeze}"),
                    config()
                        .with_alpha_distance(distance)
                        .with_alpha_squeeze(squeeze),
                    layout,
                    (32, 17),
                );
            }
        }
        check(
            &root,
            &format!("{layout:?}-unsqueezed-lossy"),
            config().with_alpha_squeeze(false),
            layout,
            (32, 17),
        );
    }
}
