//! E4 production regressions with observed tree selection and independent sample recovery.
use super::contract_decoded_frames as decoded_frames;
use super::tree::Tree;
use crate::api::SectionedTrees;
use crate::{LosslessConfig, PixelLayout, RctType};
use std::io::Write;
use std::sync::Mutex;
#[path = "../../tests/support/native_planes.rs"]
mod pinned_frames;

static TREES: Mutex<Option<Vec<Tree>>> = Mutex::new(None);
fn with_trees<R>(f: impl FnOnce(&mut Option<Vec<Tree>>) -> R) -> R {
    f(&mut TREES.lock().unwrap())
}
pub(super) fn record(tree: &Tree) {
    with_trees(|trees| {
        if let Some(trees) = trees {
            trees.push(tree.clone());
        }
    });
}
fn source(width: usize, height: usize, seed: u32) -> (Vec<i32>, Vec<u32>) {
    let mut state = seed;
    let reference: Vec<i32> = (0..width * height)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if state & 1 == 0 {
                -0x60000000
            } else {
                0x60000000
            }
        })
        .collect();
    let mut words = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let value = reference[y * width + x];
            let predicted = gradient(&reference, width, x, y);
            words.extend([
                value as u32,
                if value != predicted { 0x3f800000 } else { 0 },
                0,
            ]);
        }
    }
    (reference, words)
}
fn gradient(reference: &[i32], width: usize, x: usize, y: usize) -> i32 {
    let left = if x > 0 {
        reference[y * width + x - 1]
    } else {
        0
    };
    let top = if y > 0 {
        reference[(y - 1) * width + x]
    } else {
        left
    };
    let northwest = if x > 0 && y > 0 {
        reference[(y - 1) * width + x - 1]
    } else {
        left
    };
    (i64::from(left) + i64::from(top) - i64::from(northwest))
        .clamp(i64::from(left.min(top)), i64::from(left.max(top))) as i32
}
fn witness(
    tree: &Tree,
    reference: &[i32],
    words: &[u32],
    width: usize,
) -> Option<(usize, usize, usize, i32, i32)> {
    for y in 0..reference.len() / width {
        for x in 0..width {
            let value = reference[y * width + x];
            let predicted = gradient(reference, width, x, y);
            let narrow = value.wrapping_sub(predicted).wrapping_abs();
            let wide = (i64::from(value) - i64::from(predicted)).abs() as i32;
            let mut at = 0;
            loop {
                let node = &tree[at];
                let property = match node.property {
                    0 => 1, // actual untransformed channel one
                    1 => 0, // single-group stream
                    5 => {
                        // Absolute west sample of actual channel one.
                        let left = if x > 0 {
                            words[3 * (y * width + x - 1) + 1]
                        } else {
                            0
                        };
                        (left as i32).wrapping_abs()
                    }
                    18 => {
                        if (narrow <= node.splitval) != (wide <= node.splitval) {
                            return Some((x, y, at, narrow, wide));
                        }
                        narrow
                    }
                    _ => break, // other properties require a separate reachability proof
                };
                at = if property <= node.splitval {
                    node.lchild
                } else {
                    node.rchild
                };
            }
        }
    }
    None
}
fn native_words(
    path: &std::path::Path,
    output: &std::path::Path,
    width: usize,
    height: usize,
) -> Vec<u32> {
    let status = std::process::Command::new(crate::test_helpers::djxl_path())
        .arg(path)
        .arg(output)
        .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_Lin"])
        .output()
        .unwrap();
    std::fs::write(
        output.with_extension("log"),
        [&status.stdout[..], &status.stderr[..]].concat(),
    )
    .unwrap();
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let raw = std::fs::read(output).unwrap();
    let mut lines = raw.splitn(4, |&b| b == b'\n');
    assert_eq!(lines.next().unwrap(), b"PF");
    assert_eq!(
        lines.next().unwrap(),
        format!("{width} {height}").as_bytes()
    );
    let scale: f32 = std::str::from_utf8(lines.next().unwrap())
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(scale.abs(), 1.0);
    let words: Vec<u32> = lines
        .next()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| {
            if scale > 0.0 {
                u32::from_be_bytes(*v)
            } else {
                u32::from_le_bytes(*v)
            }
        })
        .collect();
    words
        .chunks_exact(width * 3)
        .rev()
        .flatten()
        .copied()
        .collect()
}
#[test]
#[ignore = "conditional selected-tree investigation; explicit caller required"]
fn selected_wide_reference_absolute_property_matches_native() {
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(run);
    #[cfg(not(feature = "parallel"))]
    run();
}
fn run() {
    let root = crate::test_helpers::output_dir("reference-property-contract");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let mut journal = std::fs::File::create(root.join("selection.tsv")).unwrap();
    let mut selected = false;
    for seed in [144u32, 7, 19, 41] {
        let (reference, words) = source(64, 32, seed);
        let input: Vec<u8> = words.iter().flat_map(|v| v.to_ne_bytes()).collect();
        if seed == 144 {
            use sha2::Digest;
            let canonical: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(
                sha2::Sha256::digest(&canonical)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>(),
                "e09dffe36aeaa8a5209d539420bd91ca4cff8210dbbc9c9a2ca84548f6e6c888",
                "the native positive control uses these exact source words"
            );
        }
        for learned in [false, true] {
            let config = LosslessConfig::new()
                .with_effort(9)
                .with_threads(1)
                .with_force_rct(Some(RctType::NONE))
                .with_patches(false)
                .with_squeeze(false)
                .with_modular_palette_colors(Some(0))
                .with_lz77(false)
                .with_tree_learning(learned)
                .with_modular_predictor(if learned { None } else { Some(0) })
                .with_modular_nb_prev_channels(Some(1))
                .with_tree_learning_sample_fraction(1.0)
                .with_sectioned_trees(SectionedTrees::Off)
                .with_strategy(crate::api::EncoderStrategy::Libjxl);
            with_trees(|trees| *trees = Some(Vec::new()));
            let bytes = config
                .encode_request(64, 32, PixelLayout::RgbLinearF32)
                .with_color_encoding(crate::ColorEncoding::linear_srgb())
                .encode(&input)
                .unwrap();
            let trees = with_trees(|trees| trees.take().unwrap());
            let out = root.join(format!("seed-{seed}-learned-{learned}"));
            std::fs::create_dir(&out).unwrap();
            let path = out.join("input.jxl");
            std::fs::write(&path, &bytes).unwrap();
            std::fs::write(out.join("trees.txt"), format!("{trees:#?}")).unwrap();
            let reached = if trees.len() == 1 {
                witness(&trees[0], &reference, &words, 64)
            } else {
                None
            };
            writeln!(
                journal,
                "seed={seed}\tlearned={learned}\tactual_tree_writes={}\twitness={reached:?}",
                trees.len()
            )
            .unwrap();
            journal.flush().unwrap();
            selected |= learned && reached.is_some();
            // crates.io jxl 0.8 shares the reviewed narrowing arithmetic.
            // Keep observing it, but use the pinned jxl-rs-derived decoder and
            // unmodified v0.12 as independent exact-sample acceptance gates.
            let current = std::panic::catch_unwind(|| {
                decoded_frames::decode_current_frames(
                    &bytes,
                    "actual selected reference-property tree",
                )
            });
            let current_exact = current.as_ref().is_ok_and(|decoded| {
                decoded.len() == 1
                    && decoded[0].1.is_empty()
                    && decoded[0].0.iter().map(|v| v.to_bits()).collect::<Vec<_>>() == words
            });
            writeln!(
                journal,
                "seed={seed}\tlearned={learned}\tcrates_io_jxl_08_exact={current_exact}"
            )
            .unwrap();
            journal.flush().unwrap();
            if !learned {
                assert!(current_exact, "fixed-tree positive control");
            }
            let (decoded, extras) = pinned_frames::decode(&bytes);
            assert!(extras.is_empty());
            assert_eq!(
                decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                words,
                "pinned jxl-rs-derived decoder: every binary32 source word"
            );
            assert_eq!(native_words(&path, &out.join("native.pfm"), 64, 32), words);
        }
    }
    assert!(
        selected,
        "no actual selected distinguishing property-18 path established; {}",
        root.display()
    );
}

#[test]
#[ignore = "explicit selected-reference-property caller matrix"]
fn selected_reference_property_all_policies_and_callers() {
    let root = crate::test_helpers::output_dir("reference-property-callers");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let mut journal = std::fs::File::create(root.join("callers.tsv")).unwrap();
    for (label, strategy, mode) in super::delta_palette_contract::policies() {
        for (width, height) in [(64usize, 32usize), (259, 17)] {
            let (reference, words) = source(width, height, 144);
            let input: Vec<u8> = words.iter().flat_map(|v| v.to_ne_bytes()).collect();
            for section in [
                SectionedTrees::Off,
                SectionedTrees::On,
                SectionedTrees::Hybrid,
            ] {
                let cfg = LosslessConfig::new()
                    .with_effort(9)
                    .with_threads(1)
                    .with_force_rct(Some(RctType::NONE))
                    .with_patches(false)
                    .with_squeeze(false)
                    .with_modular_palette_colors(Some(0))
                    .with_lz77(false)
                    .with_tree_learning(true)
                    .with_modular_nb_prev_channels(Some(1))
                    .with_tree_learning_sample_fraction(1.0)
                    .with_sectioned_trees(section)
                    .with_strategy(strategy.clone())
                    .with_mode(mode);
                for route in ["request", "strided", "streaming", "animation-request"] {
                    with_trees(|s| *s = Some(Vec::new()));
                    let bytes = match route {
                        "request" => cfg
                            .encode_request(width as u32, height as u32, PixelLayout::RgbLinearF32)
                            .with_color_encoding(crate::ColorEncoding::linear_srgb())
                            .encode(&input)
                            .unwrap(),
                        "strided" => {
                            let row = width * 12;
                            let stride = row + 7;
                            let mut padded = vec![0xd1; stride * height];
                            for (a, b) in
                                padded.chunks_exact_mut(stride).zip(input.chunks_exact(row))
                            {
                                a[..row].copy_from_slice(b);
                            }
                            cfg.encode_request(
                                width as u32,
                                height as u32,
                                PixelLayout::RgbLinearF32,
                            )
                            .with_color_encoding(crate::ColorEncoding::linear_srgb())
                            .with_row_stride(stride)
                            .encode(&padded)
                            .unwrap()
                        }
                        "streaming" => {
                            let mut enc = cfg
                                .encoder(width as u32, height as u32, PixelLayout::RgbLinearF32)
                                .unwrap()
                                .with_color_encoding(crate::ColorEncoding::linear_srgb());
                            for rows in input.chunks(width * 12 * 7) {
                                enc.push_rows(rows, (rows.len() / (width * 12)) as u32)
                                    .unwrap();
                            }
                            enc.finish().unwrap()
                        }
                        "animation-request" => {
                            let error = cfg
                                .encode_request(
                                    width as u32,
                                    height as u32,
                                    PixelLayout::RgbLinearF32,
                                )
                                .with_color_encoding(crate::ColorEncoding::linear_srgb())
                                .encode_animation(
                                    &crate::AnimationParams::default(),
                                    &[crate::AnimationFrame::new(&input, 10)],
                                )
                                .unwrap_err();
                            assert!(
                                matches!(
                                    error.error(),
                                    crate::api::EncodeError::UnsupportedPixelLayout(
                                        PixelLayout::RgbLinearF32
                                    )
                                ),
                                "documented integer-only lossless animation"
                            );
                            assert!(
                                with_trees(|s| s.take().unwrap()).is_empty(),
                                "refusal precedes tree emission"
                            );
                            writeln!(journal,"{label}\t{width}x{height}\t{section:?}\t{route}\tUnsupportedPixelLayout-before-tree").unwrap();
                            journal.flush().unwrap();
                            continue;
                        }
                        _ => unreachable!(),
                    };
                    let trees = with_trees(|s| s.take().unwrap());
                    let selected = if width == 64 && trees.len() == 1 {
                        witness(&trees[0], &reference, &words, width)
                    } else {
                        None
                    };
                    let out = root.join(format!("{label}-{width}x{height}-{section:?}-{route}"));
                    std::fs::create_dir(&out).unwrap();
                    let path = out.join("input.jxl");
                    std::fs::write(&path, &bytes).unwrap();
                    std::fs::write(out.join("trees.txt"), format!("{trees:#?}")).unwrap();
                    writeln!(journal,"{label}\t{width}x{height}\t{section:?}\t{route}\tactual_trees={}\twitness={selected:?}",trees.len()).unwrap();
                    journal.flush().unwrap();
                    assert!(!trees.is_empty(), "actual learned tree emitted");
                    // A different legal local tree may avoid this property.
                    // Exact full-source recovery remains mandatory either way.
                    let (pixels, extras) = pinned_frames::decode(&bytes);
                    assert!(extras.is_empty());
                    assert_eq!(
                        pixels.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        words,
                        "primary {label} {section:?} {route}"
                    );
                    assert_eq!(
                        native_words(&path, &out.join("native.pfm"), width, height),
                        words,
                        "native {label} {section:?} {route}"
                    );
                }
            }
        }
    }
}

#[path = "../../tests/support/pfm.rs"]
mod pfm;
#[test]
#[ignore = "explicit native float color and extra-channel buffers"]
fn selected_reference_property_floating_alpha_and_public_depth() {
    let oracle = std::path::PathBuf::from(
        std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
            .expect("explicit pinned v0.12 original-profile float oracle"),
    );
    let version = std::process::Command::new(&oracle)
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
    assert!(String::from_utf8_lossy(&version.stdout).contains("12000"));
    let root = crate::test_helpers::output_dir("reference-property-extra");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
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
        root.join("native-version.txt"),
        [version.stdout, version.stderr].concat(),
    )
    .unwrap();
    let mut journal = std::fs::File::create(root.join("callers.tsv")).unwrap();
    for (policy, strategy, mode) in super::delta_palette_contract::policies() {
        for (width, height) in [(64usize, 32usize), (259, 17)] {
            let (reference, labels) = source(width, height, 144);
            for (label, layout, colors, depth_case) in [
                ("rgba", PixelLayout::RgbaLinearF32, 3, false),
                ("gray-alpha", PixelLayout::GrayAlphaLinearF32, 1, false),
                ("rgb-depth", PixelLayout::RgbLinearF32, 3, true),
            ] {
                let mut words = Vec::new();
                let mut expected_color = Vec::new();
                let mut expected_alpha = Vec::new();
                for i in 0..width * height {
                    let color = if colors == 1 {
                        vec![reference[i] as u32]
                    } else {
                        vec![reference[i] as u32, labels[3 * i + 1], 0]
                    };
                    expected_color.extend_from_slice(&color);
                    words.extend(color);
                    if !depth_case {
                        let alpha = if labels[3 * i + 1] == 0 {
                            0x3e4ccccd
                        } else {
                            0x3f4ccccd
                        };
                        words.push(alpha);
                        expected_alpha.push(alpha);
                    }
                }
                let input: Vec<u8> = words.iter().flat_map(|v| v.to_ne_bytes()).collect();
                let depth: Vec<u16> = (0..width * height)
                    .map(|i| ((i * 197 + 13) % 65536) as u16)
                    .collect();
                let extras = if depth_case {
                    vec![crate::api::ExtraChannel::depth_u16(&depth)]
                } else {
                    Vec::new()
                };
                let cfg = LosslessConfig::new()
                    .with_effort(9)
                    .with_threads(1)
                    .with_force_rct(Some(RctType::NONE))
                    .with_patches(false)
                    .with_squeeze(false)
                    .with_modular_palette_colors(Some(0))
                    .with_lz77(false)
                    .with_tree_learning(true)
                    .with_modular_nb_prev_channels(Some(1))
                    .with_tree_learning_sample_fraction(1.0)
                    .with_sectioned_trees(SectionedTrees::Off)
                    .with_strategy(strategy.clone())
                    .with_mode(mode);
                let mut encoding = crate::ColorEncoding::linear_srgb();
                if colors == 1 {
                    encoding.color_space = crate::headers::color_encoding::ColorSpace::Gray;
                }
                let routes = if depth_case {
                    &["request", "strided", "writer"][..]
                } else {
                    &["request", "strided", "writer", "streaming"][..]
                };
                for &route in routes {
                    with_trees(|s| *s = Some(Vec::new()));
                    let request = cfg
                        .encode_request(width as u32, height as u32, layout)
                        .with_color_encoding(encoding.clone())
                        .with_extra_channels(&extras);
                    let bytes = if route == "streaming" {
                        let mut encoder = cfg
                            .encoder(width as u32, height as u32, layout)
                            .unwrap()
                            .with_color_encoding(encoding.clone());
                        let row = width * layout.bytes_per_pixel();
                        for rows in input.chunks(row * 7) {
                            encoder.push_rows(rows, (rows.len() / row) as u32).unwrap();
                        }
                        encoder.finish().unwrap()
                    } else if route == "strided" {
                        let row = width * layout.bytes_per_pixel();
                        let stride = row + 7;
                        let mut padded = vec![0xd1; stride * height];
                        for (a, b) in padded.chunks_exact_mut(stride).zip(input.chunks_exact(row)) {
                            a[..row].copy_from_slice(b);
                        }
                        request.with_row_stride(stride).encode(&padded).unwrap()
                    } else if route == "writer" {
                        let mut out = Vec::new();
                        request.encode_to(&input, &mut out).unwrap();
                        out
                    } else {
                        request.encode(&input).unwrap()
                    };
                    let trees = with_trees(|s| s.take().unwrap());
                    assert!(!trees.is_empty(), "actual learned tree emission");
                    let out = root.join(format!("{policy}-{width}x{height}-{label}-{route}"));
                    std::fs::create_dir(&out).unwrap();
                    let path = out.join("input.jxl");
                    std::fs::write(&path, &bytes).unwrap();
                    std::fs::write(out.join("trees.txt"), format!("{trees:#?}")).unwrap();
                    let (raw, planes) = pinned_frames::decode(&bytes);
                    let channels = raw.len() / (width * height);
                    assert!(channels == colors || channels == colors + usize::from(!depth_case));
                    let mut color = Vec::new();
                    let mut alpha = Vec::new();
                    for p in raw.chunks_exact(channels) {
                        color.extend(p[..colors].iter().map(|v| v.to_bits()));
                        if channels > colors {
                            alpha.push(p[colors].to_bits());
                        }
                    }
                    assert_eq!(color, expected_color, "every primary color word");
                    if depth_case {
                        assert_eq!(planes.len(), 1);
                        assert_eq!(
                            planes[0]
                                .iter()
                                .map(|v| (*v * 65535.0).round() as u16)
                                .collect::<Vec<_>>(),
                            depth,
                            "every primary depth code"
                        );
                    } else {
                        if !alpha.is_empty() {
                            assert_eq!(alpha, expected_alpha, "every inline primary alpha word");
                        }
                        if !planes.is_empty() {
                            assert_eq!(planes.len(), 1);
                            assert_eq!(
                                planes[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                expected_alpha,
                                "every separate primary alpha word"
                            );
                        } else {
                            assert!(!alpha.is_empty());
                        }
                    }
                    let native = std::process::Command::new(&oracle)
                        .arg(&path)
                        .arg(out.join("native"))
                        .arg("--source-fields-float")
                        .output()
                        .unwrap();
                    std::fs::write(
                        out.join("native.log"),
                        [native.stdout.as_slice(), native.stderr.as_slice()].concat(),
                    )
                    .unwrap();
                    assert!(
                        native.status.success(),
                        "{}",
                        String::from_utf8_lossy(&native.stderr)
                    );
                    let color = pfm::read_pfm_samples(&out.join("native-0.pfm"), (width, height));
                    assert_eq!(
                        color.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected_color,
                        "every native color word"
                    );
                    let plane =
                        pfm::read_pfm_samples(&out.join("native-0-ec0.pfm"), (width, height));
                    if depth_case {
                        assert_eq!(
                            plane
                                .iter()
                                .map(|v| (*v * 65535.0).round() as u16)
                                .collect::<Vec<_>>(),
                            depth,
                            "every native depth code"
                        );
                    } else {
                        assert_eq!(
                            plane.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            expected_alpha,
                            "every native alpha word"
                        );
                    }
                    writeln!(journal,"{policy}\t{width}x{height}\t{label}\t{route}\tactual_trees={}\texact-color-and-extra=true",trees.len()).unwrap();
                    journal.flush().unwrap();
                }
            }
        }
    }
}

#[test]
#[ignore = "explicit reference-property effort and Custom policy boundaries"]
fn selected_reference_property_effort_and_custom_boundaries() {
    use crate::api::{EncoderImprovementsCustom, EncoderMode, EncoderStrategy};
    let root = crate::test_helpers::output_dir("reference-property-boundaries");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let mut policies = vec![("libjxl".to_owned(), EncoderStrategy::Libjxl)];
    for repair in [false, true] {
        for buckets in [false, true] {
            policies.push((
                format!("custom-repair-{repair}-buckets-{buckets}"),
                EncoderStrategy::Custom(Box::new(EncoderImprovementsCustom {
                    lossless_tree_self_repair: repair,
                    lossless_large_tree_bucket_reduction: buckets,
                    ..Default::default()
                })),
            ));
        }
    }
    let mut journal = std::fs::File::create(root.join("callers.tsv")).unwrap();
    for (label, strategy) in policies {
        for mode in [EncoderMode::Reference, EncoderMode::Experimental] {
            for effort in [6, 7, 8, 9, 10, 11] {
                // Effort 8/9 is the actual property-set boundary. Six through
                // eleven also cover raw/exact bucket and multi-seed dispatch.
                for (width, height) in [(64usize, 32usize), (259, 17)] {
                    let (_, words) = source(width, height, 144);
                    let input: Vec<u8> = words.iter().flat_map(|v| v.to_ne_bytes()).collect();
                    let cfg = LosslessConfig::new()
                        .with_effort(effort)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_force_rct(Some(RctType::NONE))
                        .with_patches(false)
                        .with_modular_palette_colors(Some(0))
                        .with_lz77(false)
                        .with_tree_learning(true)
                        .with_modular_nb_prev_channels(Some(1))
                        .with_tree_learning_sample_fraction(1.0)
                        .with_sectioned_trees(SectionedTrees::Off);
                    // The safety gate must retain these wide samples when the
                    // optional squeeze permission is requested as well.
                    for squeeze in [false, true] {
                        with_trees(|s| *s = Some(Vec::new()));
                        let bytes = cfg
                            .clone()
                            .with_squeeze(squeeze)
                            .encode_request(width as u32, height as u32, PixelLayout::RgbLinearF32)
                            .with_color_encoding(crate::ColorEncoding::linear_srgb())
                            .encode(&input)
                            .unwrap();
                        let trees = with_trees(|s| s.take().unwrap());
                        assert!(!trees.is_empty());
                        if effort < 9 {
                            assert!(
                                trees
                                    .iter()
                                    .flat_map(|t| t.iter())
                                    .all(|n| n.property != 18),
                                "absolute reference properties are not offered below effort nine"
                            );
                        }
                        let out = root.join(format!(
                            "{label}-{mode:?}-e{effort}-{width}x{height}-squeeze-{squeeze}"
                        ));
                        std::fs::create_dir(&out).unwrap();
                        let path = out.join("input.jxl");
                        std::fs::write(&path, &bytes).unwrap();
                        std::fs::write(out.join("trees.txt"), format!("{trees:#?}")).unwrap();
                        let (pixels, extras) = pinned_frames::decode(&bytes);
                        assert!(extras.is_empty());
                        assert_eq!(
                            pixels.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            words
                        );
                        assert_eq!(
                            native_words(&path, &out.join("native.pfm"), width, height),
                            words
                        );
                        writeln!(journal,"{label}\t{mode:?}\te{effort}\t{width}x{height}\tsqueeze={squeeze}\tactual_trees={}\texact=true", trees.len()).unwrap();
                        journal.flush().unwrap();
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "explicit reference-property Huffman refusal control"]
fn reference_property_huffman_uses_fixed_trees() {
    let root = crate::test_helpers::output_dir("reference-property-huffman");
    std::fs::create_dir_all(&root).unwrap();
    let root = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    for (width, height) in [(64usize, 32usize), (259, 17)] {
        let (_, words) = source(width, height, 144);
        let input: Vec<u8> = words.iter().flat_map(|v| v.to_ne_bytes()).collect();
        with_trees(|s| *s = Some(Vec::new()));
        let bytes = LosslessConfig::new()
            .with_effort(9)
            .with_threads(1)
            .with_ans(false)
            .with_tree_learning(true)
            .with_modular_nb_prev_channels(Some(1))
            .with_force_rct(Some(RctType::NONE))
            .with_patches(false)
            .with_squeeze(false)
            .with_modular_palette_colors(Some(0))
            .with_lz77(false)
            .encode_request(width as u32, height as u32, PixelLayout::RgbLinearF32)
            .with_color_encoding(crate::ColorEncoding::linear_srgb())
            .encode(&input)
            .unwrap();
        let trees = with_trees(|s| s.take().unwrap());
        assert!(
            trees
                .iter()
                .flat_map(|t| t.iter())
                .all(|n| n.property != 18)
        );
        let path = root.join(format!("{width}x{height}.jxl"));
        std::fs::write(&path, &bytes).unwrap();
        std::fs::write(path.with_extension("trees.txt"), format!("{trees:#?}")).unwrap();
        let (pixels, extras) = pinned_frames::decode(&bytes);
        assert!(extras.is_empty());
        assert_eq!(
            pixels.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            words
        );
        assert_eq!(
            native_words(&path, &path.with_extension("pfm"), width, height),
            words
        );
    }
}
