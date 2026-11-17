//! Mixed sample formats and native-grid admission for larger default squeezes.
#![cfg(feature = "__expert")]
use jxl_encoder::api::{EncoderMode, EncoderStrategy, ExtraChannel, SectionedTrees};
use jxl_encoder::{LosslessConfig, PixelLayout};
use std::path::{Path, PathBuf};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/pfm.rs"]
mod pfm;

fn directory() -> PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("squeeze-layouts");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}

fn source(layout: PixelLayout, width: usize, height: usize) -> (Vec<u8>, Vec<u32>) {
    let colors = if layout.is_grayscale() { 1 } else { 3 };
    let mut bytes = Vec::new();
    let mut expected = Vec::new();
    let mut append = |value: u32| {
        if layout.is_f32() {
            bytes.extend_from_slice(&value.to_ne_bytes());
        } else if layout.is_f16() || layout.is_16bit() {
            bytes.extend_from_slice(&(value as u16).to_ne_bytes());
        } else {
            bytes.push(value as u8);
        }
    };
    for i in 0..width * height {
        let values: Vec<u32> = (0..colors)
            .map(|c| {
                if layout.is_f32() {
                    0x0300_0000 + (i * 37 + c * 997) as u32
                } else if layout.is_f16() {
                    if (i + c * 5) % 7 > 3 { 0x3c00 } else { 0 }
                } else {
                    ((i * 37 + c * 997 + 17) % if layout.is_16bit() { 65536 } else { 256 }) as u32
                }
            })
            .collect();
        for &value in &values {
            expected.push(if layout.is_f16() {
                if value == 0 { 0 } else { 1_f32.to_bits() }
            } else {
                value
            });
        }
        if matches!(layout, PixelLayout::Bgr8 | PixelLayout::Bgra8) {
            for &value in values.iter().rev() {
                append(value);
            }
        } else {
            for &value in &values {
                append(value);
            }
        }
        if layout.has_alpha() {
            append(if layout.is_f32() {
                0x0700_0000
            } else if layout.is_f16() {
                0x3c00
            } else if layout.is_16bit() {
                65535
            } else {
                255
            });
        }
    }
    assert_eq!(bytes.len(), width * height * layout.bytes_per_pixel());
    (bytes, expected)
}

fn check_color(samples: &[f32], expected: &[u32], layout: PixelLayout) {
    assert!(samples.iter().all(|v| v.is_finite()));
    let actual: Vec<u32> = if layout.is_f32() || layout.is_f16() {
        samples.iter().map(|v| v.to_bits()).collect()
    } else {
        let max = if layout.is_16bit() { 65535.0 } else { 255.0 };
        samples
            .iter()
            .map(|v| {
                let value = (v * max).round();
                assert!((0.0..=max).contains(&value));
                value as u32
            })
            .collect()
    };
    assert_eq!(actual, expected, "source colors for {layout:?}");
}

fn check_depth(samples: &[f32], wide: bool) {
    assert!(samples.iter().all(|v| v.is_finite()));
    let (max, expected) = if wide {
        (65535.0, 0xad5f_u32)
    } else {
        (255.0, 173)
    };
    assert!(
        samples.iter().all(|v| (v * max).round() == expected as f32),
        "exact native depth codes"
    );
}

struct LayoutCell<'a> {
    label: &'a str,
    layout: PixelLayout,
    width: usize,
    height: usize,
    shifted: bool,
}

fn check(config: &LosslessConfig, root: &Path, cell: LayoutCell<'_>) {
    let LayoutCell {
        label,
        layout,
        width,
        height,
        shifted,
    } = cell;
    let out = root.join(label);
    std::fs::create_dir(&out).unwrap();
    let colors = if layout.is_grayscale() { 1 } else { 3 };
    let alpha = usize::from(layout.has_alpha());
    let count = 20 - colors - alpha;
    let (pixels, expected) = source(layout, width, height);
    let full8 = vec![173; width * height];
    let full16 = vec![0xad5f_u16; width * height];
    let native8 = vec![173; width.div_ceil(8) * height.div_ceil(8)];
    let extras: Vec<_> = (0..count)
        .map(|index| {
            if shifted && index == 0 {
                ExtraChannel::depth(&native8).with_dim_shift(3)
            } else if index % 2 == 0 {
                ExtraChannel::depth(&full8)
            } else {
                ExtraChannel::depth_u16(&full16)
            }
        })
        .collect();
    let request = || {
        config
            .encode_request(width as u32, height as u32, layout)
            .with_extra_channels(&extras)
    };
    std::fs::write(out.join("source.bin"), &pixels).unwrap();
    let bytes = request().encode(&pixels).unwrap();
    std::fs::write(out.join("input.jxl"), &bytes).unwrap();
    if shifted {
        let plain = config
            .clone()
            .with_squeeze(false)
            .encode_request(width as u32, height as u32, layout)
            .with_extra_channels(&extras)
            .encode(&pixels)
            .unwrap();
        if layout.has_alpha() {
            assert_eq!(
                bytes, plain,
                "zero-residual implicit schedule uses the unsqueezed producer"
            );
        } else {
            assert_ne!(
                bytes, plain,
                "a shifted first extra skips chroma squeeze but permits the full-prefix schedule"
            );
        }
    }
    let row = width * layout.bytes_per_pixel();
    let stride = row + 3;
    let mut padded = vec![0xa5; stride * height];
    for (a, b) in pixels
        .chunks_exact(row)
        .zip(padded.chunks_exact_mut(stride))
    {
        b[..row].copy_from_slice(a);
    }
    assert_eq!(
        request().with_row_stride(stride).encode(&padded).unwrap(),
        bytes
    );
    let (main, planes) = decoded_extras::decode_current_planes(&bytes, label);
    check_color(&main, &expected, layout);
    assert_eq!(planes.len(), alpha + count);
    let alpha_bits = if layout.is_f32() {
        0x0700_0000
    } else {
        1_f32.to_bits()
    };
    if alpha == 1 {
        assert!(
            planes[0].iter().all(|v| v.to_bits() == alpha_bits),
            "exact alpha"
        );
    }
    for (index, plane) in planes[alpha..].iter().enumerate() {
        check_depth(plane, index % 2 == 1);
    }
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(out.join("input.jxl"))
        .arg(out.join("reference.pfm"))
        .args(["--output_extra_channels", "--num_threads=1"])
        .output()
        .unwrap();
    std::fs::write(out.join("decode.log"), &result.stderr).unwrap();
    assert!(result.status.success(), "{label}: reference rendering");
    check_color(
        &pfm::read_pfm_samples(&out.join("reference.pfm"), (width, height)),
        &expected,
        layout,
    );
    let digits = (alpha + count).to_string().len();
    for index in 0..alpha + count {
        let plane = pfm::read_pfm_samples(
            &out.join(format!("reference.pfm-ec{:0digits$}.pfm", index + 1)),
            (width, height),
        );
        if alpha == 1 && index == 0 {
            assert!(
                plane.iter().all(|v| v.to_bits() == alpha_bits),
                "reference alpha"
            );
        } else {
            check_depth(&plane, (index - alpha) % 2 == 1);
        }
    }
}

#[test]
fn supported_lossless_layouts_preserve_mixed_precisions_in_larger_default_squeezes() {
    let out = directory();
    for (name, strategy, mode) in [
        ("strict", EncoderStrategy::Libjxl, EncoderMode::Reference),
        (
            "experimental",
            EncoderStrategy::Zenjxl,
            EncoderMode::Experimental,
        ),
    ] {
        for layout in [
            PixelLayout::Rgb8,
            PixelLayout::Rgba8,
            PixelLayout::Bgr8,
            PixelLayout::Bgra8,
            PixelLayout::Gray8,
            PixelLayout::GrayAlpha8,
            PixelLayout::Rgb16,
            PixelLayout::Rgba16,
            PixelLayout::Gray16,
            PixelLayout::GrayAlpha16,
            PixelLayout::RgbLinearF16,
            PixelLayout::RgbaLinearF16,
            PixelLayout::GrayLinearF16,
            PixelLayout::GrayAlphaLinearF16,
            PixelLayout::RgbLinearF32,
            PixelLayout::RgbaLinearF32,
            PixelLayout::GrayLinearF32,
            PixelLayout::GrayAlphaLinearF32,
        ] {
            for (width, height, learned, ans) in [
                (17, 9, false, false),
                (17, 9, false, true),
                (17, 9, true, true),
                (259, 17, false, false),
                (259, 17, false, true),
                (259, 17, true, true),
            ] {
                let config = LosslessConfig::new()
                    .with_effort(if learned { 7 } else { 1 })
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_squeeze(true)
                    .with_tree_learning(learned)
                    .with_ans(ans)
                    .with_sectioned_trees(SectionedTrees::Off)
                    .with_patches(false)
                    .with_modular_palette_colors(Some(0))
                    .with_modular_group_size(Some(0));
                let label = format!("{name}-{layout:?}-{width}x{height}-learned{learned}-ans{ans}");
                check(
                    &config,
                    &out,
                    LayoutCell {
                        label: &label,
                        layout,
                        width,
                        height,
                        shifted: false,
                    },
                );
            }
        }
    }
}

#[test]
fn grayscale_alpha_with_shifted_depth_refuses_a_zero_residual_implicit_schedule() {
    let out = directory();
    for layout in [PixelLayout::GrayAlpha8, PixelLayout::GrayAlpha16] {
        let config = LosslessConfig::new()
            .with_effort(1)
            .with_threads(1)
            .with_strategy(EncoderStrategy::Libjxl)
            .with_squeeze(true)
            .with_tree_learning(false)
            .with_ans(false)
            .with_patches(false)
            .with_modular_palette_colors(Some(0));
        check(
            &config,
            &out,
            LayoutCell {
                label: &format!("{layout:?}"),
                layout,
                width: 17,
                height: 9,
                shifted: true,
            },
        );
    }
}

#[path = "support/encoder_policies.rs"]
mod encoder_policies;

#[test]
fn every_policy_reaches_no_rct_simple_and_learned_squeeze_producers() {
    let out = directory();
    for (name, strategy, mode) in encoder_policies::policies() {
        for layout in [
            PixelLayout::Gray8,
            PixelLayout::GrayAlpha8,
            PixelLayout::Gray16,
            PixelLayout::GrayAlpha16,
        ] {
            for (width, height) in [(17, 9), (259, 17)] {
                for (ans, learned) in [(false, false), (true, false), (true, true)] {
                    let config = LosslessConfig::new()
                        .with_effort(if learned { 7 } else { 1 })
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_squeeze(true)
                        .with_tree_learning(learned)
                        .with_ans(ans)
                        .with_sectioned_trees(SectionedTrees::Off)
                        .with_patches(false)
                        .with_modular_palette_colors(Some(0))
                        .with_modular_group_size(Some(0));
                    let label =
                        format!("{name}-{layout:?}-{width}x{height}-learned{learned}-ans{ans}");
                    check(
                        &config,
                        &out,
                        LayoutCell {
                            label: &label,
                            layout,
                            width,
                            height,
                            shifted: false,
                        },
                    );
                }
            }
        }
    }
}

#[test]
fn grayscale_with_a_shifted_first_extra_keeps_a_valid_implicit_schedule() {
    let out = directory();
    for (name, strategy, mode) in encoder_policies::policies() {
        for layout in [PixelLayout::Gray8, PixelLayout::Gray16] {
            for (ans, learned) in [(false, false), (true, false), (true, true)] {
                let config = LosslessConfig::new()
                    .with_effort(if learned { 7 } else { 1 })
                    .with_threads(1)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .with_squeeze(true)
                    .with_tree_learning(learned)
                    .with_ans(ans)
                    .with_patches(false)
                    .with_modular_palette_colors(Some(0));
                check(
                    &config,
                    &out,
                    LayoutCell {
                        label: &format!("{name}-{layout:?}-a{ans}-t{learned}"),
                        layout,
                        width: 17,
                        height: 9,
                        shifted: true,
                    },
                );
            }
        }
    }
}
