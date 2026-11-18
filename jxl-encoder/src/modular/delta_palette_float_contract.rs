//! Float bit-pattern palette targets, observed before entropy serialization.
use super::super::contract_pfm as pfm;
use super::*;

fn half(word: u16) -> f32 {
    let exponent = i32::from((word >> 10) & 31);
    let mantissa = u32::from(word & 1023);
    assert_ne!(
        exponent, 31,
        "a finite source must not acquire a nonfinite palette target"
    );
    let magnitude = if exponent == 0 {
        mantissa as f32 * 2.0_f32.powi(-24)
    } else {
        (1024 + mantissa) as f32 * 2.0_f32.powi(exponent - 25)
    };
    if word & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn check(policies: Vec<Policy>) {
    let root = crate::test_helpers::output_dir("delta-palette-float");
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
    let oracle = std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
        .expect("caller must supply the pinned v0.12 original-profile float oracle");
    let version = std::process::Command::new(&oracle)
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).unwrap().trim(),
        "libjxl original-profile oracle 12000"
    );
    let mut journal = std::fs::File::create(root.join("observed.tsv")).unwrap();
    let mut selected = [false; 2];
    for bits in [16_u32, 32_u32] {
        for alpha in [false, true] {
            for normal in [false, true] {
                for (width, height) in [(32_u32, 32_u32), (259, 17)] {
                    let layout = match (bits, alpha) {
                        (16, false) => PixelLayout::RgbLinearF16,
                        (16, true) => PixelLayout::RgbaLinearF16,
                        (32, false) => PixelLayout::RgbLinearF32,
                        (32, true) => PixelLayout::RgbaLinearF32,
                        _ => unreachable!(),
                    };
                    let mut input = Vec::new();
                    let mut source_alpha = Vec::new();
                    for y in 0..height {
                        for x in 0..width {
                            let (xx, yy) = ((x % 32) as i32, (y % 32) as i32);
                            let rgb = if xx == 0 {
                                [128, 120, 136]
                            } else {
                                [
                                    128 + 4 * xx - 4 * yy,
                                    120 + 4 * xx - 3 * yy,
                                    136 + 3 * xx - 4 * yy,
                                ]
                            };
                            for sample in rgb {
                                if bits == 16 {
                                    let word = if normal {
                                        0x3800 + (sample - 128) * 8
                                    } else {
                                        sample
                                    };
                                    let word = u16::try_from(word).unwrap();
                                    assert!(half(word).is_finite());
                                    input.extend(word.to_ne_bytes());
                                } else {
                                    let word = if normal {
                                        0x3f00_0000 + (sample - 128) * 8192
                                    } else {
                                        sample
                                    };
                                    assert!(f32::from_bits(word as u32).is_finite());
                                    input.extend((word as u32).to_ne_bytes());
                                }
                            }
                            if alpha {
                                let word = 0x3800 + ((x + 31 * y) % 1024) as u16;
                                let value = half(word);
                                source_alpha.push(value.to_bits());
                                if bits == 16 {
                                    input.extend(word.to_ne_bytes());
                                } else {
                                    input.extend(value.to_ne_bytes());
                                }
                            }
                        }
                    }
                    for (policy, strategy, mode) in &policies {
                        for ans in [false, true] {
                            let name = format!(
                                "f{bits}-a{alpha}-normal{normal}-{width}x{height}-{policy}-ans{ans}"
                            );
                            let out = root.join(&name);
                            std::fs::create_dir(&out).unwrap();
                            let config = LosslessConfig::new()
                                .with_effort(7)
                                .with_threads(1)
                                .with_patches(false)
                                .with_squeeze(false)
                                .with_tree_learning(false)
                                .with_lz77(false)
                                .with_ans(ans)
                                .with_modular_predictor(Some(0))
                                .with_force_rct(Some(RctType::NONE))
                                .with_modular_palette_colors(Some(16))
                                .with_lossy_palette(true)
                                .with_strategy(strategy.clone())
                                .with_mode(*mode);
                            with_capture(|capture| {
                                *capture = Some(Capture {
                                    force: None,
                                    records: Vec::new(),
                                })
                            });
                            let bytes = config
                                .encode_request(width, height, layout)
                                .encode(&input)
                                .unwrap();
                            let capture = with_capture(|state| state.take().unwrap());
                            std::fs::write(out.join("input.bin"), &input).unwrap();
                            std::fs::write(out.join("output.jxl"), &bytes).unwrap();
                            std::fs::write(
                                out.join("record-count.txt"),
                                capture.records.len().to_string(),
                            )
                            .unwrap();
                            for (index, record) in capture.records.iter().enumerate() {
                                std::fs::write(
                                    out.join(format!("palette-{index}.i32le")),
                                    record
                                        .quantized
                                        .iter()
                                        .flatten()
                                        .flat_map(|v| v.to_le_bytes())
                                        .collect::<Vec<_>>(),
                                )
                                .unwrap();
                            }

                            assert_eq!(
                                capture.records.len(),
                                1,
                                "actual palette application {name}"
                            );
                            let record = &capture.records[0];
                            let witness = distinguishing_delta(record);
                            if witness.is_some() {
                                selected[usize::from(bits == 32)] = true;
                                assert_eq!(
                                    record.predictor, 5,
                                    "selected Gradient declaration {name}"
                                );
                            }
                            let target: Vec<u32> = record
                                .quantized
                                .iter()
                                .flatten()
                                .map(|word| {
                                    if bits == 16 {
                                        half(u16::try_from(*word).unwrap()).to_bits()
                                    } else {
                                        let v = f32::from_bits(*word as u32);
                                        assert!(v.is_finite(), "nonfinite palette target {name}");
                                        v.to_bits()
                                    }
                                })
                                .collect();
                            std::fs::write(out.join("input.bin"), &input).unwrap();
                            std::fs::write(out.join("output.jxl"), &bytes).unwrap();
                            std::fs::write(
                                out.join("target.u32le"),
                                target
                                    .iter()
                                    .flat_map(|word| word.to_le_bytes())
                                    .collect::<Vec<_>>(),
                            )
                            .unwrap();
                            writeln!(
                                journal,
                                "{name}\tdeltas={}\tpredictor={}\twitness={witness:?}",
                                record.nb_deltas, record.predictor
                            )
                            .unwrap();
                            journal.flush().unwrap();
                            let primary = decoded_frames::decode_current_frames(&bytes, &name);
                            assert_eq!(primary.len(), 1);
                            let channels = primary[0].0.len() / (width * height) as usize;
                            assert!(channels == 3 || (alpha && channels == 4));
                            let rgb: Vec<u32> = primary[0]
                                .0
                                .chunks_exact(channels)
                                .flat_map(|p| p[..3].iter().map(|v| v.to_bits()))
                                .collect();
                            assert_eq!(
                                rgb, target,
                                "complete primary selected float target {name}"
                            );
                            if alpha {
                                if channels == 4 {
                                    assert_eq!(
                                        primary[0]
                                            .0
                                            .chunks_exact(channels)
                                            .map(|p| p[3].to_bits())
                                            .collect::<Vec<_>>(),
                                        source_alpha
                                    );
                                }
                                if !primary[0].1.is_empty() {
                                    assert_eq!(primary[0].1.len(), 1);
                                    assert_eq!(
                                        primary[0].1[0]
                                            .iter()
                                            .map(|v| v.to_bits())
                                            .collect::<Vec<_>>(),
                                        source_alpha
                                    );
                                } else {
                                    assert_eq!(channels, 4);
                                }
                            } else {
                                assert!(primary[0].1.is_empty());
                            }
                            let native = std::process::Command::new(&oracle)
                                .arg(out.join("output.jxl"))
                                .arg(out.join("native"))
                                .arg("--source-fields-float")
                                .output()
                                .unwrap();
                            std::fs::write(
                                out.join("native.log"),
                                [native.stdout, native.stderr].concat(),
                            )
                            .unwrap();
                            assert!(native.status.success(), "{name}");
                            let native = pfm::read_pfm_samples(
                                &out.join("native-0.pfm"),
                                (width as usize, height as usize),
                            );
                            assert_eq!(
                                native.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                target,
                                "complete native selected float target {name}"
                            );
                            if alpha {
                                let native_alpha = pfm::read_pfm_samples(
                                    &out.join("native-0-ec0.pfm"),
                                    (width as usize, height as usize),
                                );
                                assert_eq!(
                                    native_alpha.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                    source_alpha
                                );
                            }
                            if policy == "libjxl" && ans && normal && !alpha && width == 32 {
                                assert!(
                                    witness.is_some(),
                                    "fault control needs a selected distinguishing float delta"
                                );
                                with_capture(|state| {
                                    *state = Some(Capture {
                                        force: Some(4),
                                        records: Vec::new(),
                                    })
                                });
                                let faulty = config
                                    .encode_request(width, height, layout)
                                    .encode(&input)
                                    .unwrap();
                                let control = with_capture(|state| state.take().unwrap());
                                assert_eq!(control.records.len(), 1);
                                assert_eq!(control.records[0].indices, record.indices);
                                assert_eq!(control.records[0].quantized, record.quantized);
                                assert_eq!(faulty.len(), bytes.len());
                                assert_eq!(
                                    faulty
                                        .iter()
                                        .zip(&bytes)
                                        .map(|(a, b)| (a ^ b).count_ones())
                                        .sum::<u32>(),
                                    1,
                                    "only declaration changes"
                                );
                                std::fs::write(out.join("select-only.jxl"), &faulty).unwrap();
                                let wrong = decoded_frames::decode_current_frames(
                                    &faulty,
                                    "float Select-only fault",
                                );
                                assert_ne!(
                                    wrong[0].0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                    target,
                                    "primary catches the declaration-only float fault"
                                );
                                let native = std::process::Command::new(&oracle)
                                    .arg(out.join("select-only.jxl"))
                                    .arg(out.join("select-only"))
                                    .arg("--source-fields-float")
                                    .output()
                                    .unwrap();
                                std::fs::write(
                                    out.join("select-only.log"),
                                    [native.stdout, native.stderr].concat(),
                                )
                                .unwrap();
                                assert!(native.status.success());
                                let wrong = pfm::read_pfm_samples(
                                    &out.join("select-only-0.pfm"),
                                    (width as usize, height as usize),
                                );
                                assert_ne!(
                                    wrong.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                    target,
                                    "native catches the declaration-only float fault"
                                );
                            }
                            let mut stream = config.encoder(width, height, layout).unwrap();
                            for rows in input.chunks(width as usize * layout.bytes_per_pixel() * 7)
                            {
                                stream
                                    .push_rows(
                                        rows,
                                        (rows.len() / (width as usize * layout.bytes_per_pixel()))
                                            as u32,
                                    )
                                    .unwrap();
                            }
                            assert_eq!(
                                stream.finish().unwrap(),
                                bytes,
                                "float streaming bytes {name}"
                            );
                            assert_eq!(
                                config.encode(&input, width, height, layout).unwrap(),
                                bytes,
                                "float legacy bytes {name}"
                            );
                            let mut written = Vec::new();
                            config
                                .encode_request(width, height, layout)
                                .encode_to(&input, &mut written)
                                .unwrap();
                            assert_eq!(written, bytes, "float writer bytes {name}");
                            let stride = width as usize * layout.bytes_per_pixel() + 7;
                            let row = width as usize * layout.bytes_per_pixel();
                            let mut padded = vec![0xd1; stride * height as usize];
                            for (dest, src) in
                                padded.chunks_exact_mut(stride).zip(input.chunks_exact(row))
                            {
                                dest[..row].copy_from_slice(src);
                            }
                            assert_eq!(
                                config
                                    .encode_request(width, height, layout)
                                    .with_row_stride(stride)
                                    .encode(&padded)
                                    .unwrap(),
                                bytes,
                                "float stride bytes {name}"
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(
        selected.into_iter().all(|v| v),
        "both float widths must select a distinguishing delta: {}",
        root.display()
    );
}

#[test]
#[ignore = "explicit float palette winner and target qualification"]
fn strict_delta_palette_float_selected_targets_match_both_decoders() {
    check(vec![policies().remove(0)]);
}

#[test]
#[ignore = "explicit all-policy float palette winner and target qualification"]
fn all_offered_delta_palette_float_selected_targets_match_both_decoders() {
    check(policies());
}

#[test]
#[ignore = "explicit float palette with independently typed caller depth"]
fn all_offered_delta_palette_float_with_public_integer_depth() {
    let base = crate::test_helpers::output_dir("delta-palette-float-depth");
    std::fs::create_dir_all(&base).unwrap();
    let root = (0..)
        .find_map(|i| {
            let path = base.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let oracle = std::env::var_os("LIBJXL_ORIGINAL_PROFILE_ORACLE")
        .expect("pinned original-profile float oracle");
    let version = std::process::Command::new(&oracle)
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).unwrap().trim(),
        "libjxl original-profile oracle 12000"
    );
    let mut journal = std::fs::File::create(root.join("completed.tsv")).unwrap();
    for (policy, strategy, mode) in policies() {
        for bits in [16, 32] {
            for (width, height) in [(32_u32, 32_u32), (259, 17)] {
                let layout = if bits == 16 {
                    PixelLayout::RgbLinearF16
                } else {
                    PixelLayout::RgbLinearF32
                };
                let mut input = Vec::new();
                for y in 0..height {
                    for x in 0..width {
                        let (xx, yy) = ((x % 32) as i32, (y % 32) as i32);
                        let color = if xx == 0 {
                            [128, 120, 136]
                        } else {
                            [
                                128 + 4 * xx - 4 * yy,
                                120 + 4 * xx - 3 * yy,
                                136 + 3 * xx - 4 * yy,
                            ]
                        };
                        for value in color {
                            if bits == 16 {
                                input.extend(
                                    u16::try_from(0x3800 + (value - 128) * 8)
                                        .unwrap()
                                        .to_ne_bytes(),
                                );
                            } else {
                                input.extend((0x3f00_0000 + (value - 128) * 8192).to_ne_bytes());
                            }
                        }
                    }
                }
                let depth: Vec<u16> = (0..width * height)
                    .map(|i| (i as u16).wrapping_mul(197).wrapping_add(13))
                    .collect();
                let extras = [crate::api::ExtraChannel::depth_u16(&depth)];
                let config = LosslessConfig::new()
                    .with_effort(7)
                    .with_threads(1)
                    .with_patches(false)
                    .with_squeeze(false)
                    .with_tree_learning(false)
                    .with_lz77(false)
                    .with_ans(true)
                    .with_modular_predictor(Some(0))
                    .with_force_rct(Some(RctType::NONE))
                    .with_modular_palette_colors(Some(16))
                    .with_lossy_palette(true)
                    .with_strategy(strategy.clone())
                    .with_mode(mode);
                let request = || {
                    config
                        .encode_request(width, height, layout)
                        .with_extra_channels(&extras)
                };
                with_capture(|state| {
                    *state = Some(Capture {
                        force: None,
                        records: Vec::new(),
                    })
                });
                let bytes = request().encode(&input).unwrap();
                let capture = with_capture(|state| state.take().unwrap());
                let label = format!("{policy}-f{bits}-{width}x{height}");
                let out = root.join(&label);
                std::fs::create_dir(&out).unwrap();
                std::fs::write(out.join("input.bin"), &input).unwrap();
                std::fs::write(
                    out.join("depth.u16le"),
                    depth
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                std::fs::write(out.join("output.jxl"), &bytes).unwrap();
                assert_eq!(capture.records.len(), 1, "actual palette {label}");
                let record = &capture.records[0];
                assert!(
                    distinguishing_delta(record).is_some(),
                    "selected float-depth delta {label}"
                );
                assert_eq!(record.predictor, 5);
                let target: Vec<u32> = record
                    .quantized
                    .iter()
                    .flatten()
                    .map(|word| {
                        if bits == 16 {
                            half(u16::try_from(*word).unwrap()).to_bits()
                        } else {
                            let v = f32::from_bits(*word as u32);
                            assert!(v.is_finite());
                            v.to_bits()
                        }
                    })
                    .collect();
                std::fs::write(
                    out.join("target.u32le"),
                    target
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let primary = decoded_frames::decode_current_frames(&bytes, &label);
                assert_eq!(primary.len(), 1);
                assert_eq!(primary[0].1.len(), 1);
                assert_eq!(
                    primary[0].0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    target
                );
                let native = std::process::Command::new(&oracle)
                    .arg(out.join("output.jxl"))
                    .arg(out.join("native"))
                    .arg("--source-fields-float")
                    .output()
                    .unwrap();
                std::fs::write(
                    out.join("native.log"),
                    [native.stdout, native.stderr].concat(),
                )
                .unwrap();
                assert!(native.status.success());
                let colors = pfm::read_pfm_samples(
                    &out.join("native-0.pfm"),
                    (width as usize, height as usize),
                );
                assert_eq!(
                    colors.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    target
                );
                let native_depth = pfm::read_pfm_samples(
                    &out.join("native-0-ec0.pfm"),
                    (width as usize, height as usize),
                );
                for values in [&primary[0].1[0], &native_depth] {
                    assert_eq!(values.len(), depth.len());
                    for (&value, &code) in values.iter().zip(&depth) {
                        assert!(value.is_finite());
                        assert_eq!((value * 65535.0).round() as u16, code);
                    }
                }
                let stride = width as usize * layout.bytes_per_pixel() + 7;
                let mut padded = vec![0xd1; stride * height as usize];
                for (target, source) in padded
                    .chunks_exact_mut(stride)
                    .zip(input.chunks_exact(width as usize * layout.bytes_per_pixel()))
                {
                    target[..source.len()].copy_from_slice(source);
                }
                assert_eq!(
                    request().with_row_stride(stride).encode(&padded).unwrap(),
                    bytes
                );
                let mut sink = Vec::new();
                request().encode_to(&input, &mut sink).unwrap();
                assert_eq!(sink, bytes);
                writeln!(journal,"{label}\tactual delta; complete quantized words and exact u16 depth; stride/writer bytes").unwrap();
                journal.flush().unwrap();
            }
        }
    }
}

#[path = "delta_palette_associated_contract.rs"]
mod associated;
