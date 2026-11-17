//! Binary32 RCT/squeeze admission through the public lossless entry routes.
#![cfg(feature = "__expert")]
use jxl_encoder::api::{EncoderMode, EncoderStrategy};
use jxl_encoder::{LosslessConfig, PixelLayout};
use sha2::{Digest, Sha256};
use std::{io::Write, path::Path};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;

fn same(actual: &[f32], expected: &[u32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: sample count");
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), *b, "{label}: sample {i}");
    }
}
fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("squeeze-f32-routes");
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
struct Cell<'a> {
    name: &'a str,
    strategy: EncoderStrategy,
    mode: EncoderMode,
    layout: PixelLayout,
    width: usize,
    height: usize,
    style: &'a str,
    entry: &'a str,
    magnitude: u32,
}
fn check(root: &Path, cell: Cell<'_>) -> String {
    let Cell {
        name,
        strategy,
        mode,
        layout,
        width,
        height,
        style,
        entry,
        magnitude,
    } = cell;
    let mut label = format!("{name}-{layout:?}-{width}x{height}-{style}-{entry}");
    if magnitude != 1 << 27 {
        label.push_str(&format!("-m{magnitude}"));
    }
    let out = root.join(&label);
    std::fs::create_dir(&out).unwrap();
    let mut colors = Vec::new();
    let mut raw = Vec::new();
    let alpha = 0x0700_0000_u32;
    for y in 0..height {
        for x in 0..width {
            let low = if width >= height { x < 2 } else { y < 2 };
            let (a, b) = if low {
                (magnitude, (-i32::try_from(magnitude).unwrap()) as u32)
            } else {
                ((-i32::try_from(magnitude).unwrap()) as u32, magnitude)
            };
            let pixel = if layout.is_grayscale() {
                vec![a]
            } else {
                vec![a, 0, b]
            };
            colors.extend_from_slice(&pixel);
            raw.extend(pixel.iter().flat_map(|v| v.to_ne_bytes()));
            if layout.has_alpha() {
                raw.extend(alpha.to_ne_bytes());
            }
        }
    }
    assert!(
        raw.as_chunks::<4>()
            .0
            .iter()
            .all(|v| f32::from_ne_bytes(*v).is_finite())
    );
    std::fs::write(out.join("source.native-f32"), &raw).unwrap();
    let hash = |data: &[u8]| -> String {
        Sha256::digest(data)
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect()
    };
    std::fs::write(out.join("source.sha256"), hash(&raw)).unwrap();
    let learned = style == "learned";
    let config = LosslessConfig::new()
        .with_effort(if learned { 7 } else { 1 })
        .with_threads(1)
        .with_strategy(strategy)
        .with_mode(mode)
        .with_squeeze(true)
        .with_ans(style != "huffman")
        .with_tree_learning(learned)
        .with_patches(false)
        .with_modular_palette_colors(Some(0))
        .with_modular_group_size(Some(0));
    let row = width * layout.bytes_per_pixel();
    let encoded = match entry {
        "request" => config
            .encode_request(width as u32, height as u32, layout)
            .encode(&raw)
            .unwrap(),
        "strided" => {
            let stride = row + 3;
            let mut padded = vec![0xa5; stride * height];
            for (a, b) in raw.chunks_exact(row).zip(padded.chunks_exact_mut(stride)) {
                b[..row].copy_from_slice(a);
            }
            config
                .encode_request(width as u32, height as u32, layout)
                .with_row_stride(stride)
                .encode(&padded)
                .unwrap()
        }
        "legacy" => config
            .encode(&raw, width as u32, height as u32, layout)
            .unwrap(),
        "stream" => {
            let mut encoder = config.encoder(width as u32, height as u32, layout).unwrap();
            let mut y = 0;
            while y < height {
                let n = (height - y).min(7);
                encoder
                    .push_rows(&raw[y * row..(y + n) * row], n as u32)
                    .unwrap();
                y += n;
            }
            encoder.finish().unwrap()
        }
        _ => panic!("unknown entry"),
    };
    std::fs::write(out.join("input.jxl"), &encoded).unwrap();
    std::fs::write(out.join("input.sha256"), hash(&encoded)).unwrap();
    let (color, planes) = decoded_extras::decode_current_planes(&encoded, &label);
    same(&color, &colors, &format!("{label}: primary colors"));
    assert_eq!(planes.len(), usize::from(layout.has_alpha()));
    if layout.has_alpha() {
        same(
            &planes[0],
            &vec![alpha; width * height],
            &format!("{label}: primary alpha"),
        );
    }
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(out.join("input.jxl"))
        .arg(out.join("output.pfm"))
        .args(["--output_extra_channels", "--num_threads=1"])
        .output()
        .unwrap();
    std::fs::write(out.join("decode.log"), &result.stderr).unwrap();
    assert!(result.status.success(), "{label}: reference rendering");
    same(
        &pfm::read_pfm_samples(&out.join("output.pfm"), (width, height)),
        &colors,
        &format!("{label}: reference colors"),
    );
    if layout.has_alpha() {
        same(
            &pfm::read_pfm_samples(&out.join("output.pfm-ec1.pfm"), (width, height)),
            &vec![alpha; width * height],
            &format!("{label}: reference alpha"),
        );
    }
    hash(&encoded)
}

#[test]
fn every_binary32_layout_and_public_entry_preserves_all_sample_bits() {
    let out = directory();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    writeln!(journal, "label\tstatus\tstream_sha256_or_failure").unwrap();
    let mut failures = Vec::new();
    for (name, strategy, mode) in encoder_policies::policies() {
        for layout in [
            PixelLayout::RgbLinearF32,
            PixelLayout::RgbaLinearF32,
            PixelLayout::GrayLinearF32,
            PixelLayout::GrayAlphaLinearF32,
        ] {
            for (width, height) in [(16, 1), (1, 16), (259, 17), (17, 259)] {
                for style in ["huffman", "ans", "learned"] {
                    let mut control = None;
                    for entry in ["request", "strided", "legacy", "stream"] {
                        let label = format!("{name}-{layout:?}-{width}x{height}-{style}-{entry}");
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            check(
                                &out,
                                Cell {
                                    name: &name,
                                    strategy: strategy.clone(),
                                    mode,
                                    layout,
                                    width,
                                    height,
                                    style,
                                    entry,
                                    magnitude: 1 << 27,
                                },
                            )
                        }));
                        match result {
                            Ok(hash) => {
                                if let Some(expected) = &control {
                                    assert_eq!(&hash, expected, "{label}: entry byte identity");
                                } else {
                                    control = Some(hash.clone());
                                }
                                writeln!(journal, "{label}\tPASS\t{hash}").unwrap();
                            }
                            Err(error) => {
                                let message = error
                                    .downcast_ref::<String>()
                                    .map(String::as_str)
                                    .or_else(|| error.downcast_ref::<&str>().copied())
                                    .unwrap_or("non-string panic");
                                let message = message.replace(['\n', '\t'], " ");
                                writeln!(journal, "{label}\tFAIL\t{message}").unwrap();
                                failures.push(label);
                            }
                        }
                        journal.flush().unwrap();
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "failed routes: {failures:?}");
}

#[test]
fn planar_integer_rct_budget_boundaries_preserve_representable_codes() {
    let out = directory();
    for (name, strategy, mode) in encoder_policies::policies() {
        for bits in [29, 30, 31] {
            for (width, height) in [(16_usize, 1_usize), (259, 17)] {
                for (gray, alpha) in [(false, false), (false, true), (true, false), (true, true)] {
                    let count = if gray { 1 } else { 3 };
                    let mut planes: Vec<Vec<u32>> = (0..count)
                        .map(|channel| {
                            (0..width * height)
                                .map(|i| [0, 1 << 26, (1 << 27) - 16][(i + channel) % 3])
                                .collect()
                        })
                        .collect();
                    if alpha {
                        planes.push(vec![1 << 26; width * height]);
                    }
                    let borrowed: Vec<_> = planes.iter().map(Vec::as_slice).collect();
                    for style in ["huffman", "ans", "learned"] {
                        let label = format!(
                            "planar-{name}-b{bits}-{width}x{height}-g{gray}-a{alpha}-{style}"
                        );
                        let path = out.join(&label);
                        std::fs::create_dir(&path).unwrap();
                        let learned = style == "learned";
                        let config = LosslessConfig::new()
                            .with_strategy(strategy.clone())
                            .with_mode(mode)
                            .with_effort(if learned { 7 } else { 1 })
                            .with_threads(1)
                            .with_squeeze(true)
                            .with_ans(style != "huffman")
                            .with_tree_learning(learned)
                            .with_patches(false)
                            .with_modular_palette_colors(Some(0))
                            .with_modular_group_size(Some(0));
                        let bytes = config
                            .encode_planar_int(
                                width as u32,
                                height as u32,
                                &borrowed,
                                bits,
                                gray,
                                alpha,
                            )
                            .unwrap();
                        std::fs::write(path.join("input.jxl"), &bytes).unwrap();
                        let (color, extra) = decoded_extras::decode_current_planes(&bytes, &label);
                        let expected: Vec<_> = (0..width * height)
                            .flat_map(|i| planes[..count].iter().map(move |p| p[i]))
                            .collect();
                        let maximum = ((1_u32 << bits) - 1) as f64;
                        let check = |samples: &[f32], expected: &[u32]| {
                            assert_eq!(samples.len(), expected.len(), "{label}: sample count");
                            for (i, (actual, expected)) in samples.iter().zip(expected).enumerate()
                            {
                                assert!(actual.is_finite(), "{label}: sample {i}");
                                assert_eq!(
                                    (f64::from(*actual) * maximum).round() as u32,
                                    *expected,
                                    "{label}: representable source code {i}"
                                );
                            }
                        };
                        check(&color, &expected);
                        assert_eq!(extra.len(), usize::from(alpha));
                        if alpha {
                            check(&extra[0], &planes[count]);
                        }
                        let result =
                            std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(path.join("input.jxl"))
                                .arg(path.join("output.pfm"))
                                .args(["--output_extra_channels", "--num_threads=1"])
                                .output()
                                .unwrap();
                        std::fs::write(path.join("decode.log"), &result.stderr).unwrap();
                        assert!(result.status.success(), "{label}: reference rendering");
                        check(
                            &pfm::read_pfm_samples(&path.join("output.pfm"), (width, height)),
                            &expected,
                        );
                        if alpha {
                            check(
                                &pfm::read_pfm_samples(
                                    &path.join("output.pfm-ec1.pfm"),
                                    (width, height),
                                ),
                                &planes[count],
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn source_squeeze_range_boundaries_preserve_binary32_sample_bits() {
    let out = directory();
    for (name, strategy, mode) in encoder_policies::policies() {
        for layout in [
            PixelLayout::RgbLinearF32,
            PixelLayout::RgbaLinearF32,
            PixelLayout::GrayLinearF32,
            PixelLayout::GrayAlphaLinearF32,
        ] {
            for (width, height) in [(16, 1), (259, 17)] {
                for style in ["huffman", "ans", "learned"] {
                    for magnitude in [(1 << 27) - 1, 1 << 27, (1 << 27) + 1] {
                        check(
                            &out,
                            Cell {
                                name: &name,
                                strategy: strategy.clone(),
                                mode,
                                layout,
                                width,
                                height,
                                style,
                                entry: "request",
                                magnitude,
                            },
                        );
                    }
                }
            }
        }
    }
}
