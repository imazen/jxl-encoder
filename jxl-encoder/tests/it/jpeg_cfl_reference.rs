#![cfg(feature = "jpeg-reencoding")]

use super::jpeg_reencoding::decode_jxl_rs;
use jxl_encoder::LosslessConfig;
use std::path::Path;

fn decode_reference(input: &Path, output: &Path) {
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(input)
        .arg(output)
        .arg("--num_threads=1")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn jpeg_cfl_reference_pixels_match_at_color_tile_boundaries() {
    let source = image::load_from_memory(include_bytes!("../images/frymire-srgb.png"))
        .unwrap()
        .to_rgb8();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/jpeg-cfl-reference");
    for (w, h) in [(64, 32), (259, 133)] {
        let crop = image::imageops::crop_imm(&source, 0, 0, w, h).to_image();
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 90)
            .encode_image(&image::DynamicImage::ImageRgb8(crop))
            .unwrap();
        for effort in [3, 7] {
            let dir = root.join(format!("{w}x{h}-e{effort}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("input.jpg"), &jpeg).unwrap();
            let ours = LosslessConfig::new()
                .with_effort(effort)
                .encode_jpeg_transcode(&jpeg)
                .unwrap();
            std::fs::write(dir.join("ours.jxl"), &ours).unwrap();
            let result = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
                .arg(dir.join("input.jpg"))
                .arg(dir.join("reference.jxl"))
                .args(["--lossless_jpeg=1", "--num_threads=1", "-e"])
                .arg(effort.to_string())
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let reference = std::fs::read(dir.join("reference.jxl")).unwrap();
            let pixels = decode_jxl_rs(&ours);
            let expected = decode_jxl_rs(&reference);
            assert_eq!((pixels.0, pixels.1), (w as usize, h as usize));
            assert_eq!(
                (pixels.0, pixels.1, pixels.2.len()),
                (expected.0, expected.1, expected.2.len())
            );
            for (i, (&actual, &expected)) in pixels.2.iter().zip(&expected.2).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "jxl-rs {w}x{h} e{effort} sample {i}: {actual} vs {expected}"
                );
            }
            for name in ["ours", "reference"] {
                decode_reference(
                    &dir.join(format!("{name}.jxl")),
                    &dir.join(format!("{name}.png")),
                );
            }
            let pixels = image::open(dir.join("ours.png")).unwrap().to_rgb16();
            let expected = image::open(dir.join("reference.png")).unwrap().to_rgb16();
            assert_eq!(pixels.dimensions(), expected.dimensions());
            for (i, (&actual, &expected)) in
                pixels.as_raw().iter().zip(expected.as_raw()).enumerate()
            {
                assert_eq!(actual, expected, "djxl {w}x{h} e{effort} sample {i}");
            }
            assert_eq!(
                zensim_decoder::reconstruct_jpeg(&ours).unwrap().unwrap(),
                jpeg
            );
            decode_reference(&dir.join("ours.jxl"), &dir.join("reconstructed.jpg"));
            assert_eq!(std::fs::read(dir.join("reconstructed.jpg")).unwrap(), jpeg);
        }
    }
}
