//! Squeeze channel-count grammar and full pixel reconstruction.
#![cfg(feature = "__expert")]
use jxl_encoder::api::{EncoderMode, EncoderStrategy, ExtraChannel};
use jxl_encoder::{LosslessConfig, PixelLayout, RctType};
#[path = "support/decoded_extras.rs"]
mod decoded_extras;
#[path = "support/pfm.rs"]
mod pfm;

fn save_input(directory: &std::path::Path, name: &str, bytes: &[u8]) {
    use sha2::{Digest, Sha256};
    std::fs::write(directory.join(name), bytes).unwrap();
    let digest = Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect::<String>();
    std::fs::write(directory.join(format!("{name}.sha256")), digest + "\n").unwrap();
}

#[test]
fn twenty_channel_squeeze_preserves_every_sample() {
    let (w, h) = (16_usize, 16_usize);
    let native: Vec<Vec<u8>> = (0..17)
        .map(|c| {
            (0..w * h)
                .map(|i| ((i * 37 + c * 11) % 256) as u8)
                .collect()
        })
        .collect();
    let extras: Vec<_> = native.iter().map(|p| ExtraChannel::depth(p)).collect();
    let pixels: Vec<u8> = (0..w * h).flat_map(|i| [(i * 37 % 256) as u8; 3]).collect();
    let config = LosslessConfig::new()
        .with_effort(1)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_mode(EncoderMode::Reference)
        .with_squeeze(true)
        .with_ans(false)
        .with_patches(false)
        .with_tree_learning(false)
        .with_modular_palette_colors(Some(0))
        .with_force_rct(Some(RctType(0)));
    let out = jxl_encoder::test_helpers::output_dir("squeeze-contract");
    std::fs::create_dir_all(&out).unwrap();
    let run = (0..)
        .find_map(|i| {
            let p = out.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    save_input(&run, "color.rgb8", &pixels);
    for (index, plane) in native.iter().enumerate() {
        save_input(&run, &format!("depth-{index}.u8"), plane);
    }
    let bytes = config
        .encode_request(w as u32, h as u32, PixelLayout::Rgb8)
        .with_extra_channels(&extras)
        .encode(&pixels)
        .unwrap();
    std::fs::write(run.join("squeeze.jxl"), &bytes).unwrap();
    let (color, planes) = decoded_extras::decode_current_planes(&bytes, "twenty-channel squeeze");
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(run.join("squeeze.jxl"))
        .arg(run.join("squeeze.pfm"))
        .args(["--output_extra_channels", "--num_threads=1"])
        .output()
        .unwrap();
    std::fs::write(run.join("decode.log"), &result.stderr).unwrap();
    assert!(
        result.status.success(),
        "reference decode: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        color
            .iter()
            .map(|v| (*v * 255.0).round() as u8)
            .collect::<Vec<_>>(),
        pixels
    );
    let reference_color = pfm::read_pfm_samples(&run.join("squeeze.pfm"), (w, h));
    assert_eq!(
        reference_color
            .iter()
            .map(|v| (v * 255.0).round() as u8)
            .collect::<Vec<_>>(),
        pixels
    );
    assert_eq!(planes.len(), native.len());
    for (i, (plane, source)) in planes.iter().zip(&native).enumerate() {
        assert_eq!(
            plane
                .iter()
                .map(|v| (*v * 255.0).round() as u8)
                .collect::<Vec<_>>(),
            *source
        );
        let reference = pfm::read_pfm_samples(
            &run.join(format!(
                "squeeze.pfm-ec{:0width$}.pfm",
                i + 1,
                width = native.len().to_string().len()
            )),
            (w, h),
        );
        assert_eq!(
            reference
                .iter()
                .map(|v| (*v * 255.0).round() as u8)
                .collect::<Vec<_>>(),
            *source
        );
    }
}

#[test]
fn squeeze_bands_at_8x65665_keep_every_source_sample() {
    check_squeeze_bands(8, 65665);
}

#[test]
fn squeeze_bands_at_65665x8_keep_every_source_sample() {
    check_squeeze_bands(65665, 8);
}

fn check_squeeze_bands(width: usize, height: usize) {
    let out = jxl_encoder::test_helpers::output_dir("squeeze-tall-contract");
    std::fs::create_dir_all(&out).unwrap();
    let run = (0..)
        .find_map(|i| {
            let p = out.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let source: Vec<u8> = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                [
                    ((x * 17 + y * 3) % 256) as u8,
                    ((x * 5 + y * 13) % 256) as u8,
                    ((x * 29 + y * 7) % 256) as u8,
                ]
            })
        })
        .collect();
    let config = LosslessConfig::new()
        .with_effort(1)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_threads(1)
        .with_squeeze(true)
        .with_ans(false)
        .with_patches(false)
        .with_tree_learning(false)
        .with_modular_group_size(Some(1))
        .with_modular_palette_colors(Some(0));
    save_input(&run, &format!("{width}x{height}.rgb8"), &source);
    let encoded = config
        .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
        .encode(&source)
        .unwrap();
    let label = format!("{width}x{height}");
    let input = run.join(format!("{label}.jxl"));
    let output = run.join(format!("{label}.png"));
    save_input(&run, &format!("{label}.jxl"), &encoded);
    let primary =
        std::panic::catch_unwind(|| decoded_extras::decode_current_planes(&encoded, &label));
    if let Err(error) = &primary {
        let message = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap_or("Rust decoder panic");
        std::fs::write(run.join(format!("{label}.rust-error.log")), message).unwrap();
    }
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(&output)
        .arg("--num_threads=1")
        .output()
        .unwrap();
    std::fs::write(run.join(format!("{label}.decode.log")), &result.stderr).unwrap();
    assert!(
        result.status.success(),
        "{label}: pinned reference rendering"
    );
    assert_eq!(image::open(output).unwrap().to_rgb8().into_raw(), source);
    let (pixels, extras) = primary.expect("primary Rust decoder must fully render");
    assert!(extras.is_empty());
    assert_eq!(
        pixels
            .iter()
            .map(|v| (v * 255.0).round() as u8)
            .collect::<Vec<_>>(),
        source
    );
}

#[test]
fn f32_squeeze_admission_accounts_for_widening_rct() {
    let out = jxl_encoder::test_helpers::output_dir("squeeze-f32-contract");
    std::fs::create_dir_all(&out).unwrap();
    let run = (0..)
        .find_map(|i| {
            let p = out.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap();
    let positive = 0x0800_0000_u32;
    let negative = 0xf800_0000_u32;
    let bits: Vec<u32> = (0..16)
        .flat_map(|x| {
            if x < 2 {
                [positive, 0, negative]
            } else {
                [negative, 0, positive]
            }
        })
        .collect();
    assert!(bits.iter().all(|b| f32::from_bits(*b).is_finite()));
    let source: Vec<u8> = bits.iter().flat_map(|b| b.to_ne_bytes()).collect();
    save_input(&run, "source.rgb-f32-native", &source);
    let encoded = LosslessConfig::new()
        .with_effort(1)
        .with_threads(1)
        .with_strategy(EncoderStrategy::Libjxl)
        .with_squeeze(true)
        .with_ans(false)
        .with_patches(false)
        .with_tree_learning(false)
        .with_modular_palette_colors(Some(0))
        .encode_request(16, 1, PixelLayout::RgbLinearF32)
        .encode(&source)
        .unwrap();
    let input = run.join("input.jxl");
    let output = run.join("output.pfm");
    save_input(&run, "input.jxl", &encoded);
    let primary = std::panic::catch_unwind(|| {
        decoded_extras::decode_current_planes(&encoded, "f32-RCT-squeeze")
    });
    if let Ok((pixels, _)) = &primary {
        let raw: Vec<u8> = pixels.iter().flat_map(|v| v.to_ne_bytes()).collect();
        save_input(&run, "rust.rgb-f32-native", &raw);
    }
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(&output)
        .arg("--num_threads=1")
        .output()
        .unwrap();
    std::fs::write(run.join("decode.log"), &result.stderr).unwrap();
    assert!(result.status.success());
    let (pixels, extras) = primary.expect("primary Rust decoder must fully render");
    assert!(extras.is_empty());
    assert_eq!(pixels.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
    assert_eq!(
        pfm::read_pfm_samples(&output, (16, 1))
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        bits
    );
}
