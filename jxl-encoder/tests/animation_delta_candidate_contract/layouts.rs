//! Signed deltas through every implemented integer layout builder.
use super::*;

fn check_layouts(
    policies: Vec<(
        String,
        jxl_encoder::api::EncoderStrategy,
        jxl_encoder::EncoderMode,
    )>,
) {
    let root = directory();
    let image = image::load_from_memory(include_bytes!("../images/frymire-srgb.png"))
        .unwrap()
        .to_rgb8();
    let mut journal = std::fs::File::create(root.join("layouts.tsv")).unwrap();
    for (policy, strategy, mode) in policies {
        for (layout, channels, gray, reversed, bits) in [
            (PixelLayout::Rgb8, 3, false, false, 8),
            (PixelLayout::Rgba8, 4, false, false, 8),
            (PixelLayout::Bgr8, 3, false, true, 8),
            (PixelLayout::Bgra8, 4, false, true, 8),
            (PixelLayout::Gray8, 1, true, false, 8),
            (PixelLayout::GrayAlpha8, 2, true, false, 8),
            (PixelLayout::Rgb16, 3, false, false, 16),
            (PixelLayout::Rgba16, 4, false, false, 16),
            (PixelLayout::Gray16, 1, true, false, 16),
            (PixelLayout::GrayAlpha16, 2, true, false, 16),
            (PixelLayout::Rgba16, 4, false, false, 12),
            (PixelLayout::GrayAlpha16, 2, true, false, 12),
        ] {
            let alpha = channels == if gray { 2 } else { 4 };
            let scale = match bits {
                8 => 1,
                12 => 8,
                16 => 257,
                _ => unreachable!(),
            };
            let max = (1u32 << bits) - 1;
            for (width, height, negative, associated) in
                [(64u32, 32u32, false, false), (259, 133, true, alpha)]
            {
                let label = format!(
                    "{policy}-{layout:?}-{bits}-{width}x{height}-neg{negative}-assoc{associated}"
                );
                let output = root.join(&label);
                std::fs::create_dir(&output).unwrap();
                let mut base = Vec::<u16>::new();
                let mut target = Vec::<u16>::new();
                for y in 0..height {
                    for x in 0..width {
                        let pixel = image.get_pixel(x, y).0;
                        for c in 0..channels {
                            let value = if alpha && c == channels - 1 {
                                180
                            } else {
                                let index = if gray {
                                    0
                                } else if reversed {
                                    2 - c
                                } else {
                                    c
                                };
                                u16::from(pixel[index]) / 2 + 10
                            } * scale;
                            base.push(value);
                            target.push(if x < width / 4 {
                                value
                            } else if negative {
                                value - scale
                            } else {
                                value + scale
                            });
                        }
                    }
                }
                let pack = |samples: &[u16]| -> Vec<u8> {
                    if bits == 8 {
                        samples.iter().map(|&v| v as u8).collect()
                    } else {
                        samples.iter().flat_map(|v| v.to_ne_bytes()).collect()
                    }
                };
                let buffers = [pack(&base), pack(&target)];
                let frames = [
                    AnimationFrame::new(&buffers[0], 10),
                    AnimationFrame::new(&buffers[1], 20),
                ];
                let animation = AnimationParams {
                    premultiplied_alpha: associated,
                    ..Default::default()
                };
                let config = LosslessConfig::new()
                    .with_effort(3)
                    .with_threads(1)
                    .with_tree_learning(false)
                    .with_patches(false)
                    .with_strategy(strategy.clone())
                    .with_mode(mode);
                let encode = |delta| {
                    config
                        .clone()
                        .with_auto_delta_frames(delta)
                        .encode_request(width, height, layout)
                        .with_bits_per_sample(bits)
                        .encode_animation(&animation, &frames)
                        .unwrap()
                };
                let control = encode(false);
                let bytes = encode(true);
                assert!(
                    bytes.len() < control.len(),
                    "{label}: must realize the full delta candidate"
                );
                let input = output.join("input.jxl");
                std::fs::write(&input, &bytes).unwrap();
                std::fs::write(output.join("control.jxl"), &control).unwrap();
                let manifest = output.join("streams.json");
                std::fs::write(&manifest,serde_json::to_vec(&vec![serde_json::json!({"path":input,"frames":2,"reference":false,"blend":"Add","source":1,"assert_blend":"Add","assert_source":1,"assert_extra_blends":if alpha {vec![serde_json::json!({"mode":"Add","source":1})]} else {vec![]},"partial":false})]).unwrap()).unwrap();
                let parsed = std::process::Command::new("python3")
                    .arg(
                        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                            .parent()
                            .unwrap()
                            .join("scripts/check_explicit_blend_headers.py"),
                    )
                    .arg(manifest)
                    .arg(output.join("headers.json"))
                    .output()
                    .unwrap();
                std::fs::write(
                    output.join("headers.log"),
                    [parsed.stdout, parsed.stderr].concat(),
                )
                .unwrap();
                assert!(parsed.status.success(), "{label}: independent frame fields");
                let decoded = decoded_frames::decode_current_frames(&bytes, &label);
                assert_eq!(decoded.len(), 2);
                for ((color, extras), expected) in decoded.iter().zip([&base, &target]) {
                    let colors = if gray { 1 } else { 3 };
                    assert_eq!(
                        color.len(),
                        width as usize * height as usize * colors,
                        "{label}"
                    );
                    assert_eq!(extras.len(), usize::from(alpha), "{label}");
                    for (i, pixel) in expected.chunks_exact(channels).enumerate() {
                        for c in 0..colors {
                            let source = if reversed { 2 - c } else { c };
                            assert_eq!(
                                (color[i * colors + c] * max as f32).round(),
                                f32::from(pixel[source]),
                                "{label}: primary color {i}/{c}"
                            );
                        }
                        if alpha {
                            assert_eq!(
                                (extras[0][i] * max as f32).round(),
                                f32::from(pixel[channels - 1]),
                                "{label}: primary alpha {i}"
                            );
                        }
                    }
                }
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(input)
                    .arg(output.join("frame.pam"))
                    .args(["--output_frames", "--num_threads=1"])
                    .output()
                    .unwrap();
                std::fs::write(output.join("native.log"), &native.stderr).unwrap();
                assert!(
                    native.status.success(),
                    "{label}: {}",
                    String::from_utf8_lossy(&native.stderr)
                );
                let mut paths: Vec<_> = std::fs::read_dir(&output)
                    .unwrap()
                    .map(|e| e.unwrap().path())
                    .filter(|p| p.extension().is_some_and(|e| e == "pam"))
                    .collect();
                paths.sort();
                assert_eq!(paths.len(), 2);
                for (path, expected) in paths.iter().zip([&base, &target]) {
                    let actual = image::open(path).unwrap().to_rgba16();
                    assert_eq!(actual.dimensions(), (width, height));
                    for (pixel, expected) in actual.pixels().zip(expected.chunks_exact(channels)) {
                        for (c, &sample) in pixel.0.iter().take(3).enumerate() {
                            let source = if gray {
                                0
                            } else if reversed {
                                2 - c
                            } else {
                                c
                            };
                            assert_eq!(
                                (f32::from(sample) * max as f32 / 65535.0).round(),
                                f32::from(expected[source]),
                                "{label}: native color"
                            );
                        }
                        let expected_alpha = if alpha {
                            u32::from(expected[channels - 1])
                        } else {
                            max
                        };
                        assert_eq!(
                            (f32::from(pixel.0[3]) * max as f32 / 65535.0).round(),
                            expected_alpha as f32,
                            "{label}: native alpha"
                        );
                    }
                }
                writeln!(
                    journal,
                    "{label}\tcontrol={}\tselected={}",
                    control.len(),
                    bytes.len()
                )
                .unwrap();
                journal.flush().unwrap();
            }
        }
    }
}
#[test]
fn strict_signed_integer_layouts() {
    check_layouts(vec![encoder_policies::policies().remove(0)]);
}
#[test]
fn all_offered_signed_integer_layouts() {
    check_layouts(encoder_policies::policies());
}
