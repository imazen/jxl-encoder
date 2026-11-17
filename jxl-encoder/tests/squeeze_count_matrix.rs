//! Larger default squeeze schedules across offered policies and producers.
#![cfg(feature = "__expert")]
use jxl_encoder::api::{EncoderMode, EncoderStrategy, ExtraChannel, SectionedTrees};
use jxl_encoder::{LosslessConfig, PixelLayout};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::{Path, PathBuf},
};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/pfm.rs"]
mod pfm;

fn directory() -> PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("squeeze-count-matrix");
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

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}

fn codes(samples: &[f32], max: u16) -> Vec<u16> {
    assert!(
        samples.iter().all(|v| v.is_finite()),
        "finite decoded samples"
    );
    samples
        .iter()
        .map(|v| {
            let code = (v * f32::from(max)).round();
            assert!(
                (0.0..=f32::from(max)).contains(&code),
                "decoded integer range"
            );
            code as u16
        })
        .collect()
}

struct Cell<'a> {
    label: &'a str,
    width: usize,
    height: usize,
    channels: usize,
    wide: bool,
    expected_threads: Option<u32>,
    plain_control: bool,
}

fn check(config: &LosslessConfig, directory: &Path, cell: Cell<'_>, journal: &mut impl Write) {
    let Cell {
        label,
        width,
        height,
        channels,
        wide,
        expected_threads,
        plain_control,
    } = cell;
    let out = directory.join(label);
    std::fs::create_dir(&out).unwrap();
    let max = if wide { u16::MAX } else { 255 };
    let source: Vec<u16> = (0..width * height * 3)
        .map(|i| ((i * 37 + (i % 3) * 997 + 1) % (usize::from(max) + 1)) as u16)
        .collect();
    let pixels: Vec<u8> = if wide {
        source.iter().flat_map(|v| v.to_ne_bytes()).collect()
    } else {
        source.iter().map(|v| *v as u8).collect()
    };
    let native: Vec<Vec<u16>> = (0..channels - 3)
        .map(|c| {
            (0..width * height)
                .map(|i| ((i * 19 + c * 997 + 173) % (usize::from(max) + 1)) as u16)
                .collect()
        })
        .collect();
    let native8: Vec<Vec<u8>> = native
        .iter()
        .map(|p| p.iter().map(|v| *v as u8).collect())
        .collect();
    let extras: Vec<_> = if wide {
        native.iter().map(|p| ExtraChannel::depth_u16(p)).collect()
    } else {
        native8.iter().map(|p| ExtraChannel::depth(p)).collect()
    };
    let layout = if wide {
        PixelLayout::Rgb16
    } else {
        PixelLayout::Rgb8
    };
    let request = || {
        config
            .encode_request(width as u32, height as u32, layout)
            .with_extra_channels(&extras)
    };
    std::fs::write(out.join("source.bin"), &pixels).unwrap();
    let result = request().encode_with_stats(&pixels).unwrap();
    if let Some(expected) = expected_threads {
        assert_eq!(
            result.stats().threads_used(),
            expected,
            "{label}: actual workers"
        );
    }
    let bytes = result.data().unwrap();
    std::fs::write(out.join("input.jxl"), bytes).unwrap();
    if plain_control {
        let plain = config
            .clone()
            .with_squeeze(false)
            .encode_request(width as u32, height as u32, layout)
            .with_extra_channels(&extras)
            .encode(&pixels)
            .unwrap();
        assert_eq!(
            bytes, plain,
            "{label}: unsafe implicit schedule must retain unsqueezed bytes"
        );
    }
    let row = width * layout.bytes_per_pixel();
    let stride = row + 3;
    let mut padded = vec![0xa5; stride * height];
    for (input, output) in pixels
        .chunks_exact(row)
        .zip(padded.chunks_exact_mut(stride))
    {
        output[..row].copy_from_slice(input);
    }
    assert_eq!(
        request().with_row_stride(stride).encode(&padded).unwrap(),
        bytes,
        "{label}: stride"
    );
    let (color, planes) = decoded_extras::decode_current_planes(bytes, label);
    assert_eq!(codes(&color, max), source, "{label}: primary colors");
    assert_eq!(planes.len(), native.len());
    for (index, (plane, expected)) in planes.iter().zip(&native).enumerate() {
        assert_eq!(
            codes(plane, max),
            *expected,
            "{label}: primary depth {index}"
        );
    }
    let reference = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(out.join("input.jxl"))
        .arg(out.join("reference.pfm"))
        .args(["--output_extra_channels", "--num_threads=1"])
        .output()
        .unwrap();
    std::fs::write(out.join("decode.log"), &reference.stderr).unwrap();
    assert!(reference.status.success(), "{label}: reference rendering");
    assert_eq!(
        codes(
            &pfm::read_pfm_samples(&out.join("reference.pfm"), (width, height)),
            max
        ),
        source,
        "{label}: reference colors"
    );
    let digits = native.len().to_string().len();
    for (index, expected) in native.iter().enumerate() {
        let file = out.join(format!("reference.pfm-ec{:0digits$}.pfm", index + 1));
        assert_eq!(
            codes(&pfm::read_pfm_samples(&file, (width, height)), max),
            *expected,
            "{label}: reference depth {index}"
        );
    }
    writeln!(
        journal,
        "{label}\t{width}\t{height}\t{channels}\t{wide}\t{}\t{}",
        hash(&pixels),
        hash(bytes)
    )
    .unwrap();
    journal.flush().unwrap();
}

fn config(strategy: EncoderStrategy, mode: EncoderMode, style: &str) -> LosslessConfig {
    let learned = matches!(style, "tree-small" | "global" | "local" | "hybrid");
    LosslessConfig::new()
        .with_effort(if learned { 7 } else { 1 })
        .with_threads(1)
        .with_strategy(strategy)
        .with_mode(mode)
        .with_squeeze(true)
        .with_ans(!style.starts_with("huffman"))
        .with_tree_learning(learned)
        .with_sectioned_trees(match style {
            "local" => SectionedTrees::On,
            "hybrid" => SectionedTrees::Hybrid,
            _ => SectionedTrees::Off,
        })
        .with_patches(false)
        .with_modular_palette_colors(Some(0))
        .with_modular_group_size(Some(0))
}

#[test]
fn every_policy_preserves_count_boundaries_through_all_lossless_squeeze_producers() {
    let out = directory();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    writeln!(
        journal,
        "label\twidth\theight\tchannels\twide\tsource_sha256\tjxl_sha256"
    )
    .unwrap();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for wide in [false, true] {
            for channels in [19, 20, 21] {
                for (style, width, height) in [
                    ("huffman-small", 17, 9),
                    ("ans-small", 17, 9),
                    ("tree-small", 17, 9),
                    ("huffman-large", 259, 17),
                    ("ans-large", 259, 17),
                    ("global", 259, 17),
                    ("local", 259, 17),
                    ("hybrid", 259, 17),
                ] {
                    let label = format!("{policy}-{style}-n{channels}-wide{wide}");
                    check(
                        &config(strategy.clone(), mode, style),
                        &out,
                        Cell {
                            label: &label,
                            width,
                            height,
                            channels,
                            wide,
                            expected_threads: None,
                            plain_control: false,
                        },
                        &mut journal,
                    );
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn larger_squeeze_defaults_use_the_requested_parallel_route() {
    let out = directory();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    writeln!(
        journal,
        "label\twidth\theight\tchannels\twide\tsource_sha256\tjxl_sha256"
    )
    .unwrap();
    for (policy, strategy, mode) in [
        ("strict", EncoderStrategy::Libjxl, EncoderMode::Reference),
        (
            "experimental",
            EncoderStrategy::Zenjxl,
            EncoderMode::Experimental,
        ),
    ] {
        for threads in [1, 2] {
            for style in ["global", "local", "hybrid"] {
                let label = format!("{policy}-{style}-t{threads}");
                check(
                    &config(strategy.clone(), mode, style).with_threads(threads),
                    &out,
                    Cell {
                        label: &label,
                        width: 259,
                        height: 17,
                        channels: 20,
                        wide: true,
                        expected_threads: Some(threads as u32),
                        plain_control: false,
                    },
                    &mut journal,
                );
            }
        }
    }
}

#[test]
fn unsafe_implicit_dimensions_retain_unsqueezed_pixels_and_bytes() {
    let out = directory();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    writeln!(
        journal,
        "label\twidth\theight\tchannels\twide\tsource_sha256\tjxl_sha256"
    )
    .unwrap();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for wide in [false, true] {
            for (width, height) in [(8, 8), (1, 17), (17, 1)] {
                let label = format!("{policy}-fallback-{width}x{height}-wide{wide}");
                check(
                    &config(strategy.clone(), mode, "huffman-small"),
                    &out,
                    Cell {
                        label: &label,
                        width,
                        height,
                        channels: 20,
                        wide,
                        expected_threads: None,
                        plain_control: true,
                    },
                    &mut journal,
                );
            }
        }
    }
}

#[test]
fn maximum_admitted_extra_channel_count_uses_implicit_default() {
    let out = directory();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    for (name, strategy, mode) in encoder_policies::policies() {
        for wide in [false, true] {
            let label = format!("{name}-max256-wide{wide}");
            check(
                &config(strategy.clone(), mode, "ans-small"),
                &out,
                Cell {
                    label: &label,
                    width: 17,
                    height: 9,
                    channels: 259,
                    wide,
                    expected_threads: None,
                    plain_control: false,
                },
                &mut journal,
            );
        }
    }
}

#[test]
fn lower_count_one_axis_squeezes_keep_every_sample() {
    let out = directory();
    let mut journal = std::fs::File::create(out.join("completed.tsv")).unwrap();
    for (name, strategy, mode) in encoder_policies::policies()
        .into_iter()
        .filter(|(name, _, _)| name == "libjxl" || name == "experimental")
    {
        for wide in [false, true] {
            for (width, height) in [(1, 17), (17, 1)] {
                for style in ["huffman-small", "ans-small", "tree-small"] {
                    let label = format!("{name}-{width}x{height}-{style}-wide{wide}");
                    check(
                        &config(strategy.clone(), mode, style),
                        &out,
                        Cell {
                            label: &label,
                            width,
                            height,
                            channels: 3,
                            wide,
                            expected_threads: None,
                            plain_control: false,
                        },
                        &mut journal,
                    );
                }
            }
        }
    }
}
