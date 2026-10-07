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

/// Full render of `jxl` through jxl-oxide as raw f32 samples.
fn decode_jxl_oxide(jxl: &[u8]) -> (usize, usize, Vec<f32>) {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(jxl))
        .expect("jxl-oxide header");
    let render = image.render_frame(0).expect("jxl-oxide render");
    let frame = render.image_all_channels();
    (frame.width(), frame.height(), frame.buf().to_vec())
}

fn assert_bits_eq(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: sample count");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "{label} sample {i}: {a} vs {e}");
    }
}

/// Transcodes zenjpeg-encoded photos — three distinct quant tables (Y, Cb,
/// Cr), jpegli adaptive quantization, auto_optimize, 4:4:4 and 4:2:0,
/// progressive and baseline — and pins the two properties a JPEG transcode
/// owes its caller:
///
/// 1. Its rendered pixels equal libjxl v0.12's `cjxl --lossless_jpeg=1`
///    transcode of the same JPEG, bit for bit, through jxl-rs, jxl-oxide and
///    djxl. The JPEG XL render of a recompressed JPEG is defined by the
///    decoder (default AC quant bias, floating-point chroma-from-luma, no
///    per-sample clamp before YCbCr->RGB), so it is NOT a bit-exact JPEG
///    decode in libjxl either; matching the reference render is the
///    encoder's contract.
/// 2. The original JPEG reconstructs byte-exactly from the JBRD box through
///    zenjxl-decoder and djxl for both subsamplings, and through jxl-oxide
///    for 4:4:4. jxl-oxide (imazen fork 08395e61) panics reconstructing
///    4:2:0 transcodes from libjxl v0.12 itself (`jxl-grid` coordinate out
///    of range), so it is a decoder defect and is not asserted for 4:2:0.
///
/// Sources are real photos from the codec-corpus `imageflow` set, cropped to
/// a multi-group size with partial 16x16 MCUs and to a small odd size.
#[test]
#[cfg(not(target_family = "wasm"))]
fn jpeg_cfl_reference_zenjpeg_styles_match_libjxl_render_and_reconstruct() {
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

    let corpus = codec_corpus::Corpus::new().expect("codec-corpus init");
    let inputs = corpus
        .get("imageflow")
        .expect("codec-corpus imageflow")
        .join("test_inputs");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/jpeg-zenjpeg-reference");
    for (file, w, h) in [
        ("roof_test_800x600.jpg", 520u32, 392u32),
        ("red-leaf.jpg", 259, 133),
    ] {
        let source = std::fs::read(inputs.join(file)).unwrap();
        let decoded = zenjpeg::decoder::Decoder::new()
            .decode(&source, enough::Unstoppable)
            .unwrap();
        let (sw, sh) = decoded.dimensions();
        let pixels = decoded.pixels_u8().expect("RGB8 output");
        assert_eq!(pixels.len(), sw as usize * sh as usize * 3, "{file}");
        let (x0, y0) = ((sw - w) / 2, (sh - h) / 2);
        let mut crop = Vec::with_capacity((w * h * 3) as usize);
        for y in y0..y0 + h {
            let start = ((y * sw + x0) * 3) as usize;
            crop.extend_from_slice(&pixels[start..start + (w * 3) as usize]);
        }
        for (subsampling, sub_name) in [
            (ChromaSubsampling::None, "444"),
            (ChromaSubsampling::Quarter, "420"),
        ] {
            for progressive in [true, false] {
                for quality in [30.0f32, 90.0] {
                    let jpeg = EncoderConfig::ycbcr(Quality::ApproxJpegli(quality), subsampling)
                        .auto_optimize(true)
                        .aq_enabled(true)
                        // After auto_optimize, which forces progressive scans.
                        .progressive(progressive)
                        .encode_bytes(&crop, w, h, PixelLayout::Rgb8Srgb)
                        .unwrap();
                    let scan = if progressive { "prog" } else { "base" };
                    let cell = format!("{file}-{w}x{h}-{sub_name}-{scan}-q{quality}");

                    // The fixture must have the structure this test claims.
                    let parsed = jxl_encoder::jpeg::read_jpeg(&jpeg, None, None).unwrap();
                    assert_eq!(parsed.components.len(), 3, "{cell}");
                    assert_eq!(parsed.quant.len(), 3, "{cell}: quant table count");
                    let tables: Vec<_> = parsed.quant.iter().map(|q| q.values).collect();
                    assert!(
                        tables[0] != tables[1] && tables[1] != tables[2] && tables[0] != tables[2],
                        "{cell}: quant tables must be distinct"
                    );
                    let quant_idx: Vec<u32> =
                        parsed.components.iter().map(|c| c.quant_idx).collect();
                    assert_eq!(quant_idx, [0, 1, 2], "{cell}");
                    let luma_factor = if sub_name == "420" { 2 } else { 1 };
                    assert_eq!(
                        (
                            parsed.components[0].h_samp_factor,
                            parsed.components[0].v_samp_factor
                        ),
                        (luma_factor, luma_factor),
                        "{cell}"
                    );
                    assert_eq!(parsed.scan_info.len() > 1, progressive, "{cell}: scans");

                    for effort in [7u8, 9] {
                        let dir = root.join(format!("{cell}-e{effort}"));
                        std::fs::create_dir_all(&dir).unwrap();
                        std::fs::write(dir.join("input.jpg"), &jpeg).unwrap();
                        let ours = LosslessConfig::new()
                            .with_effort(effort)
                            .encode_jpeg_transcode(&jpeg)
                            .unwrap();
                        std::fs::write(dir.join("ours.jxl"), &ours).unwrap();
                        let result =
                            std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
                                .arg(dir.join("input.jpg"))
                                .arg(dir.join("reference.jxl"))
                                .args(["--lossless_jpeg=1", "--num_threads=1", "-e"])
                                .arg(effort.to_string())
                                .output()
                                .unwrap();
                        assert!(
                            result.status.success(),
                            "{cell}: {}",
                            String::from_utf8_lossy(&result.stderr)
                        );
                        let reference = std::fs::read(dir.join("reference.jxl")).unwrap();
                        let label = format!("{cell} e{effort}");

                        let (pw, ph, actual) = decode_jxl_rs(&ours);
                        let (rw, rh, expected) = decode_jxl_rs(&reference);
                        assert_eq!((pw, ph), (w as usize, h as usize), "{label}");
                        assert_eq!((rw, rh), (pw, ph), "{label}");
                        assert_bits_eq(&actual, &expected, &format!("jxl-rs {label}"));

                        let (ow, oh, actual) = decode_jxl_oxide(&ours);
                        let (_, _, expected) = decode_jxl_oxide(&reference);
                        assert_eq!((ow, oh), (w as usize, h as usize), "{label}");
                        assert_bits_eq(&actual, &expected, &format!("jxl-oxide {label}"));

                        for name in ["ours", "reference"] {
                            decode_reference(
                                &dir.join(format!("{name}.jxl")),
                                &dir.join(format!("{name}.png")),
                            );
                        }
                        let actual = image::open(dir.join("ours.png")).unwrap().to_rgb16();
                        let expected = image::open(dir.join("reference.png")).unwrap().to_rgb16();
                        assert_eq!(actual.dimensions(), (w, h), "{label}");
                        assert_eq!(actual.as_raw(), expected.as_raw(), "djxl {label}");

                        assert_eq!(
                            zensim_decoder::reconstruct_jpeg(&ours).unwrap().unwrap(),
                            jpeg,
                            "zenjxl-decoder JBRD {label}"
                        );
                        decode_reference(&dir.join("ours.jxl"), &dir.join("reconstructed.jpg"));
                        assert_eq!(
                            std::fs::read(dir.join("reconstructed.jpg")).unwrap(),
                            jpeg,
                            "djxl JBRD {label}"
                        );
                        if sub_name == "444" {
                            let image = jxl_oxide::JxlImage::builder()
                                .read(std::io::Cursor::new(&ours))
                                .unwrap();
                            let mut rebuilt = Vec::new();
                            image.reconstruct_jpeg(&mut rebuilt).unwrap();
                            assert_eq!(rebuilt, jpeg, "jxl-oxide JBRD {label}");
                        }
                    }
                }
            }
        }
    }
}
