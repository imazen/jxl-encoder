//! Complete raw rate-control rendering with and without the optional buttloop.
#![cfg(all(feature = "rate-control", feature = "__expert"))]
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/pfm.rs"]
mod pfm;
#[test]
fn rate_control_feature_keeps_native_and_primary_rendering() {
    use jxl_encoder::__test_exports::VarDctEncoder;
    use jxl_encoder::vardct::RateControlConfig;
    let base = jxl_encoder::test_helpers::output_dir("rate-control-feature-contract");
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
    let image = image::load_from_memory(include_bytes!("images/frymire.png"))
        .unwrap()
        .to_rgb8();
    for (width, height) in [(32_usize, 17_usize), (259, 17)] {
        let pixels = image::imageops::crop_imm(&image, 333, 135, width as u32, height as u32)
            .to_image()
            .into_raw();
        let linear: Vec<f32> = pixels
            .iter()
            .map(|v| {
                let v = f32::from(*v) / 255.0;
                if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            })
            .collect();
        for mode in [
            jxl_encoder::EncoderMode::Reference,
            jxl_encoder::EncoderMode::Experimental,
        ] {
            for iterations in [0, 2] {
                let out = root.join(format!("{width}x{height}-{mode:?}-i{iterations}"));
                std::fs::create_dir(&out).unwrap();
                let mut encoder = VarDctEncoder::new(0.1);
                encoder.profile = jxl_encoder::LossyConfig::new(0.1)
                    .with_effort(3)
                    .with_strategy(jxl_encoder::api::EncoderStrategy::Libjxl)
                    .resolved_profile();
                encoder.effort = 3;
                encoder.encoder_mode = mode;
                encoder.enable_patches = false;
                let config = RateControlConfig {
                    max_iterations: iterations,
                    ..Default::default()
                };
                let (raw, observed) = encoder
                    .encode_with_rate_control_config(width, height, &linear, &config)
                    .unwrap();
                assert!(observed <= iterations);
                let mut bytes = jxl_encoder::wrap_in_container(&raw, None, None);
                bytes.splice(32..32, b"\0\0\0\x09jxll\x0a".iter().copied());
                // This diagnostic wrapper permits either modular-buffer width;
                // the public raw method still returns its original codestream.
                std::fs::write(out.join("input.jxl"), &bytes).unwrap();
                let primary =
                    decoded_frames::decode_current_frames(&bytes, "rate-control feature closure");
                assert_eq!(primary.len(), 1);
                assert!(primary[0].1.is_empty());
                assert_eq!(primary[0].0.len(), width * height * 3);
                let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                    .arg(out.join("input.jxl"))
                    .arg(out.join("native.pfm"))
                    .arg("--num_threads=1")
                    .output()
                    .unwrap();
                std::fs::write(out.join("native.log"), &native.stderr).unwrap();
                assert!(native.status.success());
                let native = pfm::read_pfm_samples(&out.join("native.pfm"), (width, height));
                assert_eq!(native.len(), primary[0].0.len());
                assert!(
                    native
                        .iter()
                        .zip(&primary[0].0)
                        .all(|(a, b)| (a - b).abs() <= 0.001)
                );
                std::fs::write(
                    out.join("iterations.txt"),
                    format!("observed_iterations={observed}\n"),
                )
                .unwrap();
            }
        }
    }
}
