//! Native shifted extra grids must survive partial final groups.
#![cfg(feature = "__expert")]
use jxl_encoder::api::{EncoderMode, EncoderStrategy, ExtraChannel, SectionedTrees};
use jxl_encoder::{LosslessConfig, PixelLayout, RctType};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;

fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("shifted-extra-edges");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{}: {e}", p.display()),
            }
        })
        .unwrap()
}

fn base(strategy: EncoderStrategy, mode: EncoderMode, group: u8) -> LosslessConfig {
    LosslessConfig::new()
        .with_effort(1)
        .with_strategy(strategy)
        .with_mode(mode)
        .with_threads(1)
        .with_modular_group_size(Some(group))
        .with_squeeze(false)
        .with_patches(false)
        .with_tree_learning(false)
        .with_modular_palette_colors(Some(0))
        .with_modular_channel_colors_group_percent(Some(0.0))
        .with_force_rct(Some(RctType(0)))
}

fn pixels(width: usize, height: usize, pattern: bool, wide: bool) -> (Vec<u8>, Vec<u32>) {
    let codes: Vec<u8> = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                let v = if pattern && (x / 4 + y / 4) % 2 == 1 {
                    255
                } else {
                    0
                };
                [v; 3]
            })
        })
        .collect();
    let bits = codes
        .iter()
        .map(|v| (*v as f32 / 255.0).to_bits())
        .collect();
    let bytes = if wide {
        codes
            .iter()
            .flat_map(|v| (u16::from(*v) * 257).to_ne_bytes())
            .collect()
    } else {
        codes
    };
    (bytes, bits)
}

fn render(
    bytes: &[u8],
    out: &std::path::Path,
    name: &str,
    dims: (usize, usize),
    extra_count: usize,
    expected: Option<&[u32]>,
) -> (Option<Vec<f32>>, Option<Vec<f32>>) {
    let input = out.join(format!("{name}.jxl"));
    std::fs::write(&input, bytes).unwrap();
    let rust = std::panic::catch_unwind(|| decoded_extras::decode_current_planes(bytes, name))
        .ok()
        .map(|(color, p)| {
            if let Some(expected) = expected {
                assert_eq!(
                    color.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected,
                    "{name}: exact primary colors"
                );
            }
            assert_eq!(p.len(), extra_count);
            p.into_iter().last().unwrap()
        });
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(out.join(format!("{name}.pfm")))
        .arg("--output_extra_channels")
        .arg("--num_threads=1")
        .output()
        .unwrap();
    std::fs::write(
        out.join(format!("{name}.decode.log")),
        format!(
            "primary 0.8: {}\ndjxl: {}\n{}",
            rust.is_some(),
            result.status,
            String::from_utf8_lossy(&result.stderr)
        ),
    )
    .unwrap();
    let djxl = if result.status.success() {
        let color = pfm::read_pfm_samples(&out.join(format!("{name}.pfm")), dims);
        if let Some(expected) = expected {
            assert_eq!(
                color.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected,
                "{name}: exact reference colors"
            );
        }
        Some(pfm::read_pfm_samples(
            &out.join(format!("{name}.pfm-ec{extra_count}.pfm")),
            dims,
        ))
    } else {
        None
    };
    (rust, djxl)
}

struct Cell<'a> {
    label: &'a str,
    width: usize,
    height: usize,
    shift: u32,
    wide: bool,
    pattern: bool,
    exact_color: bool,
}
fn check(config: &LosslessConfig, out: &std::path::Path, cell: Cell<'_>) -> Vec<String> {
    let Cell {
        label,
        width,
        height,
        shift,
        wide,
        pattern,
        exact_color,
    } = cell;
    let (ew, eh) = (width.div_ceil(1 << shift), height.div_ceil(1 << shift));
    let (aw, ah) = (ew << shift, eh << shift);
    let mut native8 = vec![0_u8; ew * eh];
    *native8.last_mut().unwrap() = 173;
    let mut native16 = vec![0_u16; ew * eh];
    *native16.last_mut().unwrap() = 0xad5f;
    let native = if wide {
        native16.iter().flat_map(|v| v.to_ne_bytes()).collect()
    } else {
        native8.clone()
    };
    std::fs::write(out.join(format!("{label}.native")), native).unwrap();
    let extra = [if wide {
        ExtraChannel::depth_u16(&native16)
    } else {
        ExtraChannel::depth(&native8)
    }
    .with_dim_shift(shift)];
    let layout = if wide {
        PixelLayout::Rgb16
    } else {
        PixelLayout::Rgb8
    };
    let encode = |w: usize, h: usize| {
        let (pixels, expected) = pixels(w, h, pattern, wide);
        let bytes = config
            .encode_request(w as u32, h as u32, layout)
            .with_extra_channels(&extra)
            .encode(&pixels)
            .unwrap();
        let row_bytes = w * layout.bytes_per_pixel();
        let stride = row_bytes + 5;
        let mut padded = vec![0xa5; stride * h];
        for (src, dst) in pixels
            .chunks_exact(row_bytes)
            .zip(padded.chunks_exact_mut(stride))
        {
            dst[..row_bytes].copy_from_slice(src);
        }
        assert_eq!(
            config
                .encode_request(w as u32, h as u32, layout)
                .with_extra_channels(&extra)
                .with_row_stride(stride)
                .encode(&padded)
                .unwrap(),
            bytes,
            "{label}: strided source"
        );
        (bytes, expected)
    };
    let (aligned, aligned_color) = encode(aw, ah);
    let control = render(
        &aligned,
        out,
        &format!("{label}-aligned"),
        (aw, ah),
        1,
        exact_color.then_some(aligned_color.as_slice()),
    );
    let (odd, odd_color) = encode(width, height);
    let actual = render(
        &odd,
        out,
        &format!("{label}-odd"),
        (width, height),
        1,
        exact_color.then_some(odd_color.as_slice()),
    );
    let mut failures = Vec::new();
    for (decoder, actual, control) in [
        ("jxl-rs", actual.0, control.0),
        ("djxl", actual.1, control.1),
    ] {
        match (actual, control) {
            (Some(actual), Some(control)) => {
                assert_eq!(actual.len(), width * height);
                assert_eq!(control.len(), aw * ah);
                assert!(
                    control.iter().any(|v| *v > 0.0),
                    "{label}: source-grid marker vanished in aligned control"
                );
                let expected: Vec<_> = control
                    .chunks_exact(aw)
                    .take(height)
                    .flat_map(|row| row[..width].iter().map(|v| v.to_bits()))
                    .collect();
                if actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>() != expected {
                    failures.push(format!("{decoder} {label}: changed native source grid"));
                }
            }
            (a, c) => failures.push(format!(
                "{decoder} {label}: odd decoded={} aligned decoded={}",
                a.is_some(),
                c.is_some()
            )),
        }
    }
    failures
}

#[test]
fn shifted_depth_final_group_matches_same_native_grid_in_aligned_canvas() {
    let out = directory();
    let mut failures = Vec::new();
    for (width, height) in [(257, 2), (2, 257), (257, 257)] {
        failures.extend(check(
            &base(EncoderStrategy::Libjxl, EncoderMode::Reference, 1),
            &out,
            Cell {
                label: &format!("strict-{width}x{height}"),
                width,
                height,
                shift: 1,
                wide: false,
                pattern: false,
                exact_color: true,
            },
        ));
    }
    assert!(failures.is_empty(), "{}: {failures:?}", out.display());
}

#[test]
fn every_policy_preserves_shifted_edges_at_each_group_size() {
    let out = directory();
    let mut failures = Vec::new();
    for (name, strategy, mode) in encoder_policies::policies() {
        for group in 0..=3 {
            for shift in 0..=2 {
                for vertical in [false, true] {
                    for wide in [false, true] {
                        let (width, height) = if vertical {
                            (1 << shift, (128 << group) + 1)
                        } else {
                            ((128 << group) + 1, 1 << shift)
                        };
                        failures.extend(check(
                            &base(strategy.clone(), mode, group),
                            &out,
                            Cell {
                                label: &format!("{name}-g{group}-s{shift}-v{vertical}-wide{wide}"),
                                width,
                                height,
                                shift,
                                wide,
                                pattern: false,
                                exact_color: true,
                            },
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}: {failures:?}", out.display());
}

#[test]
fn shift_three_routes_preserve_the_native_grid() {
    let out = directory();
    let mut failures = Vec::new();
    for (name, strategy, mode) in encoder_policies::policies() {
        for group in 0..=3 {
            for vertical in [false, true] {
                let (width, height) = if vertical {
                    (8, (128 << group) + 1)
                } else {
                    ((128 << group) + 1, 8)
                };
                failures.extend(check(
                    &base(strategy.clone(), mode, group),
                    &out,
                    Cell {
                        label: &format!("{name}-g{group}-v{vertical}"),
                        width,
                        height,
                        shift: 3,
                        wide: false,
                        pattern: false,
                        exact_color: true,
                    },
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}: {failures:?}", out.display());
}

#[test]
fn palette_squeeze_and_tree_routes_preserve_shifted_extras() {
    let out = directory();
    let mut failures = Vec::new();
    for (name, strategy, mode) in encoder_policies::policies() {
        for style in [
            "ans",
            "huffman",
            "global",
            "local",
            "hybrid",
            "palette",
            "compact",
            "squeeze-fixed",
            "squeeze-tree",
            "lossy-palette",
        ] {
            let mut config = base(strategy.clone(), mode, 1);
            config = match style {
                "huffman" => config.with_ans(false),
                "global" => config
                    .with_effort(7)
                    .with_tree_learning(true)
                    .with_ans(true),
                "local" => config
                    .with_effort(7)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_sectioned_trees(SectionedTrees::On),
                "hybrid" => config
                    .with_effort(7)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_sectioned_trees(SectionedTrees::Hybrid),
                "palette" => config
                    .with_effort(7)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_modular_palette_colors(Some(256)),
                "compact" => config
                    .with_effort(7)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_modular_channel_colors_group_percent(Some(95.0)),
                "squeeze-fixed" => config.with_squeeze(true),
                "squeeze-tree" => config
                    .with_effort(7)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_squeeze(true),
                "lossy-palette" => config.with_effort(7).with_lossy_palette(true),
                "ans" => config.with_ans(true),
                _ => unreachable!(),
            };
            for shift in 1..=3 {
                failures.extend(check(
                    &config,
                    &out,
                    Cell {
                        label: &format!("{name}-{style}-s{shift}"),
                        width: 259,
                        height: 17,
                        shift,
                        wide: false,
                        pattern: style.contains("palette") || style == "compact",
                        exact_color: style != "lossy-palette",
                    },
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}: {failures:?}", out.display());
}

#[test]
fn shifted_channels_cross_lf_boundaries_with_selected_entropy_policies() {
    let out = directory();
    let mut failures = Vec::new();
    for group in 0..=3 {
        for style in ["ans", "huffman", "global", "local", "hybrid"] {
            let mut config = base(EncoderStrategy::Libjxl, EncoderMode::Reference, group);
            config = match style {
                "huffman" => config.with_ans(false),
                "global" => config
                    .with_effort(7)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_sectioned_trees(SectionedTrees::Off),
                "local" => config
                    .with_effort(7)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_sectioned_trees(SectionedTrees::On),
                "hybrid" => config
                    .with_effort(8)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_sectioned_trees(SectionedTrees::Hybrid),
                _ => config.with_ans(true),
            };
            for vertical in [false, true] {
                for wide in [false, true] {
                    let (width, height) = if vertical {
                        (8, (1024 << group) + 1)
                    } else {
                        ((1024 << group) + 1, 8)
                    };
                    failures.extend(check(
                        &config,
                        &out,
                        Cell {
                            label: &format!("lf-boundary-g{group}-{style}-v{vertical}-wide{wide}"),
                            width,
                            height,
                            shift: 3,
                            wide,
                            pattern: false,
                            exact_color: true,
                        },
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}: {failures:?}", out.display());
}

#[test]
fn shifted_boundaries_and_fixed_predictors_preserve_samples() {
    let out = directory();
    let mut failures = Vec::new();
    for group in 0..=3 {
        for shift in 1..=3 {
            for delta in [-1_isize, 0, 1] {
                for vertical in [false, true] {
                    let edge = ((128 << group) as isize + delta) as usize;
                    let (width, height) = if vertical { (1, edge) } else { (edge, 1) };
                    failures.extend(check(
                        &base(EncoderStrategy::Libjxl, EncoderMode::Reference, group),
                        &out,
                        Cell {
                            label: &format!("boundary-g{group}-s{shift}-d{delta}-v{vertical}"),
                            width,
                            height,
                            shift,
                            wide: false,
                            pattern: false,
                            exact_color: true,
                        },
                    ));
                }
            }
        }
    }
    for predictor in 0..=13 {
        for ans in [false, true] {
            let config = base(EncoderStrategy::Libjxl, EncoderMode::Reference, 1)
                .with_modular_predictor(Some(predictor))
                .with_ans(ans);
            failures.extend(check(
                &config,
                &out,
                Cell {
                    label: &format!("predictor-{predictor}-ans{ans}"),
                    width: 259,
                    height: 17,
                    shift: 3,
                    wide: true,
                    pattern: false,
                    exact_color: true,
                },
            ));
        }
    }
    assert!(failures.is_empty(), "{}: {failures:?}", out.display());
}

#[test]
fn strict_lf_producers_preserve_the_native_grid() {
    let out = directory();
    let mut failures = Vec::new();
    for lossy_palette in [false, true] {
        let config = base(EncoderStrategy::Libjxl, EncoderMode::Reference, 1)
            .with_effort(7)
            .with_lossy_palette(lossy_palette);
        failures.extend(check(
            &config,
            &out,
            Cell {
                label: if lossy_palette {
                    "lossy-palette-lf"
                } else {
                    "normal-lf"
                },
                width: 259,
                height: 17,
                shift: 3,
                wide: false,
                pattern: lossy_palette,
                exact_color: !lossy_palette,
            },
        ));
    }
    assert!(failures.is_empty(), "{}: {failures:?}", out.display());
}

#[test]
fn shifted_depth_is_independent_of_supported_source_layout_and_alpha_association() {
    let out = directory();
    for (policy, strategy, mode) in [
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
            for shift in [1, 3] {
                for associated in [false, true] {
                    for dims in [(17, 9), (259, 17)] {
                        check_layout(
                            &base(strategy.clone(), mode, 1),
                            &out,
                            LayoutCell {
                                label: &format!(
                                    "{policy}-{layout:?}-s{shift}-associated{associated}-{dims:?}"
                                ),
                                layout,
                                dims,
                                shift,
                                associated,
                                pattern: false,
                                native_wide: false,
                            },
                        );
                    }
                }
            }
        }
    }
}

struct LayoutCell<'a> {
    label: &'a str,
    layout: PixelLayout,
    dims: (usize, usize),
    shift: u32,
    associated: bool,
    pattern: bool,
    native_wide: bool,
}

fn check_layout(config: &LosslessConfig, out: &std::path::Path, cell: LayoutCell<'_>) {
    let LayoutCell {
        label,
        layout,
        dims: (width, height),
        shift,
        associated,
        pattern,
        native_wide,
    } = cell;
    let native = vec![173; width.div_ceil(1 << shift) * height.div_ceil(1 << shift)];
    let native16 = vec![0xad5f; native.len()];
    let extras = [if native_wide {
        ExtraChannel::depth_u16(&native16)
    } else {
        ExtraChannel::depth(&native)
    }
    .with_dim_shift(shift)];
    let mut source = vec![0; width * height * layout.bytes_per_pixel()];
    let mut expected = vec![0; width * height * if layout.is_grayscale() { 1 } else { 3 }];
    if pattern {
        assert!(matches!(
            layout,
            PixelLayout::Gray8 | PixelLayout::GrayAlpha8
        ));
        for (i, pixel) in source
            .chunks_exact_mut(layout.bytes_per_pixel())
            .enumerate()
        {
            let white = (i % width / 4 + i / width / 4) % 2 == 1;
            pixel[0] = if white { 255 } else { 0 };
            expected[i] = if white { 1_f32.to_bits() } else { 0 };
            if layout.has_alpha() {
                pixel[1] = 255;
            }
        }
    }
    let encode = || {
        config
            .encode_request(width as u32, height as u32, layout)
            .with_premultiplied_alpha(associated)
            .with_extra_channels(&extras)
    };
    let bytes = encode().encode(&source).unwrap();
    let row = width * layout.bytes_per_pixel();
    let stride = row + 3;
    let mut padded = vec![0xa5; stride * height];
    for (src, dst) in source
        .chunks_exact(row)
        .zip(padded.chunks_exact_mut(stride))
    {
        dst[..row].copy_from_slice(src);
    }
    assert_eq!(
        encode().with_row_stride(stride).encode(&padded).unwrap(),
        bytes
    );
    let (rust, reference) = render(
        &bytes,
        out,
        label,
        (width, height),
        1 + usize::from(layout.has_alpha()),
        Some(&expected),
    );
    for plane in [rust, reference] {
        let plane = plane.expect("complete layout/extra rendering");
        assert_eq!(plane.len(), width * height);
        assert!(
            plane.iter().all(|v| if native_wide {
                (*v * 65535.0).round() as u16 == 0xad5f
            } else {
                (*v * 255.0).round() as u8 == 173
            }),
            "{label}: constant native depth must remain constant after interpolation"
        );
    }
}

#[test]
fn grayscale_extras_do_not_become_color_transform_channels() {
    let out = directory();
    for (name, strategy, mode) in encoder_policies::policies() {
        for style in [
            "fixed",
            "huffman",
            "global",
            "local",
            "hybrid",
            "palette",
            "compact",
            "squeeze-fixed",
            "squeeze-tree",
            "lossy-palette",
        ] {
            let config = base(strategy.clone(), mode, 1);
            let config = match style {
                "huffman" => config.with_ans(false),
                "global" => config
                    .with_effort(7)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_sectioned_trees(SectionedTrees::Off),
                "local" => config
                    .with_effort(7)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_sectioned_trees(SectionedTrees::On),
                "hybrid" => config
                    .with_effort(8)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_sectioned_trees(SectionedTrees::Hybrid),
                "palette" => config
                    .with_effort(7)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_modular_palette_colors(Some(256)),
                "compact" => config
                    .with_effort(7)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_modular_channel_colors_group_percent(Some(95.0)),
                "squeeze-fixed" => config.with_squeeze(true),
                "squeeze-tree" => config
                    .with_effort(7)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_squeeze(true),
                "lossy-palette" => config.with_effort(7).with_lossy_palette(true),
                _ => config,
            };
            for dims in [(17, 9), (259, 17)] {
                for shift in [1, 3] {
                    check_layout(
                        &config,
                        &out,
                        LayoutCell {
                            label: &format!("gray-{name}-{style}-{dims:?}-s{shift}"),
                            layout: PixelLayout::GrayAlpha8,
                            dims,
                            shift,
                            associated: false,
                            pattern: true,
                            native_wide: false,
                        },
                    );
                }
            }
        }
    }
}

#[test]
fn constant_depth_preserves_all_native_bits_and_fixed_predictor_semantics() {
    let out = directory();
    for (name, strategy, mode) in encoder_policies::policies() {
        for native_wide in [false, true] {
            for predictor in 0..=13 {
                for ans in [false, true] {
                    let config = base(strategy.clone(), mode, 1)
                        .with_modular_predictor(Some(predictor))
                        .with_ans(ans);
                    check_layout(
                        &config,
                        &out,
                        LayoutCell {
                            label: &format!(
                                "constant-{name}-wide{native_wide}-p{predictor}-ans{ans}"
                            ),
                            layout: PixelLayout::Rgb8,
                            dims: (259, 17),
                            shift: 3,
                            associated: false,
                            pattern: false,
                            native_wide,
                        },
                    );
                }
            }
        }
    }
}

#[test]
fn real_rgb_and_depth_samples_survive_global_and_local_group_coding() {
    let out = directory();
    let source = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    let (width, height) = (259_usize, 133_usize);
    for (policy, strategy, mode) in [
        ("strict", EncoderStrategy::Libjxl, EncoderMode::Reference),
        (
            "experimental",
            EncoderStrategy::Zenjxl,
            EncoderMode::Experimental,
        ),
    ] {
        for tree_mode in [
            SectionedTrees::Off,
            SectionedTrees::On,
            SectionedTrees::Hybrid,
        ] {
            for shift in [1, 3] {
                let (ew, eh) = (width.div_ceil(1 << shift), height.div_ceil(1 << shift));
                let native: Vec<u8> = (0..eh)
                    .flat_map(|y| {
                        let source = &source;
                        (0..ew).map(move |x| {
                            let p = source
                                .get_pixel(
                                    ((x << shift).min(width - 1)) as u32,
                                    ((y << shift).min(height - 1)) as u32,
                                )
                                .0;
                            (u16::from(p[0]) + 5 * u16::from(p[1]) + 3 * u16::from(p[2])) as u8
                        })
                    })
                    .collect();
                let extras = [ExtraChannel::depth(&native).with_dim_shift(shift)];
                let config = base(strategy.clone(), mode, 1)
                    .with_effort(8)
                    .with_tree_learning(true)
                    .with_ans(true)
                    .with_sectioned_trees(tree_mode);
                let mut planes = Vec::new();
                for aligned in [true, false] {
                    let (w, h) = if aligned {
                        (ew << shift, eh << shift)
                    } else {
                        (width, height)
                    };
                    let mut rgb = Vec::with_capacity(w * h * 3);
                    for y in 0..h {
                        for x in 0..w {
                            rgb.extend(
                                source
                                    .get_pixel(x.min(width - 1) as u32, y.min(height - 1) as u32)
                                    .0,
                            );
                        }
                    }
                    let bytes = config
                        .encode_request(w as u32, h as u32, PixelLayout::Rgb8)
                        .with_extra_channels(&extras)
                        .encode(&rgb)
                        .unwrap();
                    let label = format!("real-{policy}-{tree_mode:?}-s{shift}-aligned{aligned}");
                    let (color, depth) = decoded_extras::decode_current_planes(&bytes, &label);
                    assert_eq!(
                        color
                            .iter()
                            .map(|v| (*v * 255.0).round() as u8)
                            .collect::<Vec<_>>(),
                        rgb,
                        "{label}: exact RGB codes"
                    );
                    assert_eq!(depth.len(), 1);
                    let (rust, reference) = render(&bytes, &out, &label, (w, h), 1, None);
                    assert_eq!(rust.as_ref().unwrap(), &depth[0]);
                    let png = out.join(format!("{label}.png"));
                    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(out.join(format!("{label}.jxl")))
                        .arg(&png)
                        .args(["--bits_per_sample=16", "--num_threads=1"])
                        .output()
                        .unwrap();
                    std::fs::write(out.join(format!("{label}.png.decode.log")), &result.stderr)
                        .unwrap();
                    assert!(result.status.success(), "{label}: reference PNG rendering");
                    assert_eq!(
                        image::open(png).unwrap().to_rgb16().as_raw(),
                        &rgb.iter().map(|v| u16::from(*v) * 257).collect::<Vec<_>>(),
                        "{label}: exact reference RGB codes"
                    );
                    planes.push((w, h, rust.unwrap(), reference.unwrap()));
                }
                for decoder in [false, true] {
                    let control = if decoder { &planes[0].3 } else { &planes[0].2 };
                    let actual = if decoder { &planes[1].3 } else { &planes[1].2 };
                    let expected: Vec<_> = control
                        .chunks_exact(planes[0].0)
                        .take(height)
                        .flat_map(|row| row[..width].iter().map(|v| v.to_bits()))
                        .collect();
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected,
                        "{policy} {tree_mode:?} s{shift}: reference={decoder} native grid"
                    );
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn threaded_shifted_routes_retain_native_grids() {
    let out = directory();
    let (width, height) = (513_usize, 259_usize);
    let mut failures = Vec::new();
    for (name, strategy, mode) in [
        ("strict", EncoderStrategy::Libjxl, EncoderMode::Reference),
        (
            "experimental",
            EncoderStrategy::Zenjxl,
            EncoderMode::Experimental,
        ),
    ] {
        for threads in [1, 2] {
            for style in ["huffman", "global", "local", "hybrid"] {
                let config = base(strategy.clone(), mode, 1)
                    .with_threads(threads)
                    .with_effort(7)
                    .with_modular_predictor(Some(0))
                    .with_tree_learning(style != "huffman")
                    .with_ans(style != "huffman")
                    .with_sectioned_trees(match style {
                        "local" => SectionedTrees::On,
                        "hybrid" => SectionedTrees::Hybrid,
                        _ => SectionedTrees::Off,
                    });
                let input = vec![0; width * height * 3];
                let native = vec![173; width.div_ceil(8) * height.div_ceil(8)];
                let extras = [ExtraChannel::depth(&native).with_dim_shift(3)];
                let result = config
                    .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
                    .with_extra_channels(&extras)
                    .encode_with_stats(&input)
                    .unwrap();
                assert_eq!(
                    result.stats().threads_used(),
                    threads as u32,
                    "{name}-{style}: requested worker count must be admitted"
                );
                for wide in [false, true] {
                    failures.extend(check(
                        &config,
                        &out,
                        Cell {
                            label: &format!("{name}-{style}-t{threads}-wide{wide}"),
                            width,
                            height,
                            shift: 3,
                            wide,
                            pattern: false,
                            exact_color: true,
                        },
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
