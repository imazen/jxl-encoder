//! Automatic signed-delta selection preserves the ordinary color-tagged render.
use super::*;
fn check_profiles(
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
    for (policy, strategy, mode) in policies {
        for name in ["linear", "gamma", "p3-icc"] {
            for (width, height, alpha) in [(64u32, 32u32, false), (259, 133, true)] {
                let label = format!("{policy}-{name}-{width}x{height}");
                let out = root.join(&label);
                std::fs::create_dir(&out).unwrap();
                let channels = if alpha { 4 } else { 3 };
                let mut base = Vec::new();
                let mut target = Vec::new();
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..channels {
                            let value = if c == 3 {
                                180
                            } else {
                                image.get_pixel(x, y).0[c] / 2
                            };
                            base.push(value);
                            target.push(if x < width / 4 { value } else { value + 1 });
                        }
                    }
                }
                let layout = if alpha {
                    PixelLayout::Rgba8
                } else {
                    PixelLayout::Rgb8
                };
                let frames = [
                    AnimationFrame::new(&base, 10),
                    AnimationFrame::new(&target, 20),
                ];
                let metadata = jxl_encoder::ImageMetadata::new()
                    .with_icc_profile(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
                let config = LosslessConfig::new()
                    .with_effort(3)
                    .with_threads(1)
                    .with_tree_learning(false)
                    .with_patches(false)
                    .with_strategy(strategy.clone())
                    .with_mode(mode);
                let encode = |delta| {
                    let config = config.clone().with_auto_delta_frames(delta);
                    let request = config.encode_request(width, height, layout);
                    let request = match name {
                        "linear" => {
                            request.with_color_encoding(jxl_encoder::ColorEncoding::linear_srgb())
                        }
                        "gamma" => request
                            .with_color_encoding(jxl_encoder::ColorEncoding::with_gamma(1.0 / 2.2)),
                        "p3-icc" => request.with_metadata(&metadata),
                        _ => unreachable!(),
                    };
                    request
                        .encode_animation(&AnimationParams::default(), &frames)
                        .unwrap()
                };
                let control = encode(false);
                let bytes = encode(true);
                assert!(bytes.len() < control.len(), "{label}: actual winning delta");
                let parsed = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&bytes))
                    .unwrap();
                let header = parsed.frame(1).unwrap().header();
                assert_eq!(format!("{:?}", header.blending_info.mode), "Add");
                assert_eq!(header.blending_info.source, 1);
                for extra in &header.ec_blending_info {
                    assert_eq!(format!("{:?}", extra.mode), "Add");
                    assert_eq!(extra.source, 1);
                }
                let baseline = decoded_frames::decode_current_frames(&control, &label);
                let selected = decoded_frames::decode_current_frames(&bytes, &label);
                assert_eq!(baseline.len(), 2);
                assert_eq!(selected.len(), 2);
                for ((a, ae), (b, be)) in baseline.iter().zip(&selected) {
                    assert_eq!(a.len(), b.len());
                    assert_eq!(ae.len(), be.len());
                    assert!(
                        a.iter()
                            .zip(b)
                            .all(|(&a, &b)| (a * 255.0).round() == (b * 255.0).round()),
                        "{label}: primary rendered codes"
                    );
                    for (a, b) in ae.iter().zip(be) {
                        assert!(
                            a.iter()
                                .zip(b)
                                .all(|(&a, &b)| (a * 255.0).round() == (b * 255.0).round()),
                            "{label}: primary alpha codes"
                        );
                    }
                }
                let mut renders = Vec::new();
                for (variant, data) in [("control", control), ("selected", bytes)] {
                    let dir = out.join(variant);
                    std::fs::create_dir(&dir).unwrap();
                    let input = dir.join("input.jxl");
                    std::fs::write(&input, data).unwrap();
                    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(input)
                        .arg(dir.join("frame.pam"))
                        .args(["--output_frames", "--num_threads=1"])
                        .output()
                        .unwrap();
                    std::fs::write(dir.join("native.log"), &result.stderr).unwrap();
                    assert!(
                        result.status.success(),
                        "{label}: {}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    let mut paths: Vec<_> = std::fs::read_dir(&dir)
                        .unwrap()
                        .map(|e| e.unwrap().path())
                        .filter(|p| p.extension().is_some_and(|e| e == "pam"))
                        .collect();
                    paths.sort();
                    assert_eq!(paths.len(), 2);
                    renders.push(
                        paths
                            .iter()
                            .map(|p| image::open(p).unwrap().to_rgba8().into_raw())
                            .collect::<Vec<_>>(),
                    );
                }
                assert_eq!(
                    renders[0], renders[1],
                    "{label}: native rendered sample codes"
                );
            }
        }
    }
}
#[test]
fn strict_color_tagged_selected_deltas() {
    check_profiles(vec![encoder_policies::policies().remove(0)]);
}
#[test]
fn all_offered_color_tagged_selected_deltas() {
    check_profiles(encoder_policies::policies());
}
