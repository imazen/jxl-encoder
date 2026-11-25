//! Selected float palette targets with genuine nonopaque associated alpha.
use super::*;

#[test]
#[ignore = "explicit all-policy associated float palette target qualification"]
fn all_offered_delta_palette_float_associated_alpha() {
    let base = crate::test_helpers::output_dir("delta-palette-float-associated");
    std::fs::create_dir_all(&base).unwrap();
    let root = (0..)
        .find_map(|i| {
            let p = base.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let (oracle, _) = crate::test_helpers::native_libjxl_oracle(
        "LIBJXL_ORIGINAL_PROFILE_ORACLE",
        "original-profile",
    );
    let alpha = half(0x3bff);
    assert!(alpha < 1.0);
    let mut journal = std::fs::File::create(root.join("completed.tsv")).unwrap();
    for (policy, strategy, mode) in policies() {
        for bits in [16, 32] {
            for (width, height) in [(32_u32, 32_u32), (259, 17)] {
                let layout = if bits == 16 {
                    PixelLayout::RgbaLinearF16
                } else {
                    PixelLayout::RgbaLinearF32
                };
                let mut input = Vec::new();
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
                                let word = u16::try_from(0x3800 + (sample - 128) * 8).unwrap();
                                assert!(half(word) <= alpha, "valid associated source");
                                input.extend(word.to_ne_bytes());
                            } else {
                                let word = (0x3f00_0000 + (sample - 128) * 8192) as u32;
                                assert!(f32::from_bits(word) <= alpha, "valid associated source");
                                input.extend(word.to_ne_bytes());
                            }
                        }
                        if bits == 16 {
                            input.extend(0x3bff_u16.to_ne_bytes());
                        } else {
                            input.extend(alpha.to_ne_bytes());
                        }
                    }
                }
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
                        .with_premultiplied_alpha(true)
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
                std::fs::write(out.join("output.jxl"), &bytes).unwrap();
                let parsed = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&bytes))
                    .unwrap();
                assert_eq!(
                    parsed.image_header().metadata.ec_info[0].alpha_associated(),
                    Some(true)
                );
                assert_eq!(capture.records.len(), 1, "actual associated palette");
                let record = &capture.records[0];
                assert!(
                    distinguishing_delta(record).is_some(),
                    "actual distinguishing delta {label}"
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
                let channels = primary[0].0.len() / (width * height) as usize;
                assert!(channels == 3 || channels == 4);
                assert_eq!(
                    primary[0]
                        .0
                        .chunks_exact(channels)
                        .flat_map(|p| p[..3].iter().map(|v| v.to_bits()))
                        .collect::<Vec<_>>(),
                    target
                );
                if channels == 4 {
                    assert!(
                        primary[0]
                            .0
                            .chunks_exact(channels)
                            .all(|p| p[3].to_bits() == alpha.to_bits())
                    );
                }
                if primary[0].1.is_empty() {
                    assert_eq!(channels, 4);
                } else {
                    assert_eq!(primary[0].1.len(), 1);
                    assert_eq!(primary[0].1[0].len(), (width * height) as usize);
                    assert!(
                        primary[0].1[0]
                            .iter()
                            .all(|v| v.to_bits() == alpha.to_bits())
                    );
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
                assert!(native.status.success());
                let colors = pfm::read_pfm_samples(
                    &out.join("native-0.pfm"),
                    (width as usize, height as usize),
                );
                assert_eq!(
                    colors.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    target
                );
                let native_alpha = pfm::read_pfm_samples(
                    &out.join("native-0-ec0.pfm"),
                    (width as usize, height as usize),
                );
                assert!(native_alpha.iter().all(|v| v.to_bits() == alpha.to_bits()));
                let mut stream = config
                    .encoder(width, height, layout)
                    .unwrap()
                    .with_premultiplied_alpha(true);
                for rows in input.chunks(width as usize * layout.bytes_per_pixel() * 7) {
                    stream
                        .push_rows(
                            rows,
                            (rows.len() / (width as usize * layout.bytes_per_pixel())) as u32,
                        )
                        .unwrap();
                }
                assert_eq!(
                    stream.finish().unwrap(),
                    bytes,
                    "associated streaming bytes"
                );
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
                writeln!(journal,"{label}\tassociated=true; actual delta; complete quantized words/exact alpha; stream/stride/writer bytes").unwrap();
                journal.flush().unwrap();
            }
        }
    }
}
