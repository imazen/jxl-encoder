//! Private rate-control level transport and selected-stream byte preservation.
use super::VarDctEncoder;
#[path = "../../tests/support/decoded_frames.rs"]
mod decoded_frames;
#[path = "../../tests/support/pfm.rs"]
mod pfm;

#[test]
fn rate_control_selected_level_covers_basic_header_and_frames() {
    let base = crate::test_helpers::output_dir("rate-control-level-transport");
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
    // Dimension boundary, modular-width promise, LF on/off and actual
    // iterative selection are independent constraints. Large area/LF has its
    // own retained integration replay; these cases keep the prototype small.
    for (label, width, height, source_float, lf, iterations, expected) in [
        ("small", 32, 17, false, false, 0, 5),
        ("dimension-boundary", 262144, 1, false, false, 0, 5),
        ("dimension-over", 262145, 1, false, false, 0, 10),
        ("float-promise", 32, 17, true, false, 0, 10),
        ("small-lf", 32, 17, false, true, 0, 5),
        ("iterative", 32, 17, false, false, 2, 5),
    ] {
        let out = root.join(label);
        std::fs::create_dir(&out).unwrap();
        let pixels = if label == "iterative" {
            (0..width * height * 3)
                .map(|i| ((i * 73) % 251) as f32 / 250.0)
                .collect()
        } else {
            vec![0.2158605_f32; width * height * 3]
        };
        let mut encoder = VarDctEncoder::new(1.0);
        encoder.effort = 1;
        encoder.profile = crate::LossyConfig::new(1.0)
            .with_effort(1)
            .with_strategy(crate::api::EncoderStrategy::Libjxl)
            .resolved_profile();
        encoder.force_strategy = Some(0);
        encoder.enable_gaborish = false;
        encoder.epf_level_override = Some(0);
        encoder.enable_patches = false;
        encoder.pixel_domain_loss = false;
        encoder.use_lf_frame = lf;
        if source_float {
            encoder.source_bit_depth = Some((true, 32, 8));
        }
        let config = crate::vardct::RateControlConfig {
            max_iterations: iterations,
            ..Default::default()
        };
        let (data, observed_iterations, level) = encoder
            .encode_with_rate_control_config_and_level_impl(width, height, &pixels, &config)
            .unwrap();
        std::fs::write(out.join("raw.jxl"), &data).unwrap();
        std::fs::write(
            out.join("observed.txt"),
            format!("level={level} iterations={observed_iterations}\n"),
        )
        .unwrap();
        assert_eq!(level, expected, "{label}");
        if label == "iterative" {
            assert!(observed_iterations > 0, "actual adjustment must run");
        }
        let legacy = encoder
            .encode_with_rate_control_config(width, height, &pixels, &config)
            .unwrap();
        assert_eq!(
            (data.as_slice(), observed_iterations),
            (legacy.0.as_slice(), legacy.1),
            "{label}"
        );
        let wrapped = crate::container::wrap_in_container_with_level(&data, None, None, level);
        let input = out.join("output.jxl");
        std::fs::write(&input, &wrapped).unwrap();
        let primary =
            std::panic::catch_unwind(|| decoded_frames::decode_current_frames(&wrapped, label));
        let native = std::process::Command::new(crate::test_helpers::djxl_path())
            .arg(&input)
            .arg(out.join("native.pfm"))
            .args(["--num_threads=1"])
            .output()
            .unwrap();
        std::fs::write(out.join("native.log"), &native.stderr).unwrap();
        assert!(
            native.status.success() && primary.is_ok(),
            "{label}: native={}, primary={}",
            native.status,
            primary.is_ok()
        );
        let primary = primary.unwrap();
        assert_eq!(primary.len(), 1);
        assert!(primary[0].1.is_empty());
        assert_eq!(primary[0].0.len(), width * height * 3);
        let samples = pfm::read_pfm_samples(&out.join("native.pfm"), (width, height));
        assert_eq!(samples.len(), primary[0].0.len());
        assert!(samples.iter().all(|v| v.is_finite()));
        let error = samples
            .iter()
            .zip(&primary[0].0)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f32, f32::max);
        assert!(error <= 0.001, "{label}: {error}");
    }
}
