//! Partial-MCU RGB8 inputs and malformed public JPEG coefficient layouts.
#![cfg(all(feature = "chroma-subsampling", feature = "jpeg-reencoding"))]

use jxl_encoder::{ChromaSubsampling, LossyConfig, PixelLayout};

#[path = "../examples/distance_targeting_probe/decode.rs"]
mod decode;

#[test]
fn subsampled_partial_mcu_shapes_decode_and_preserve_stride() {
    let source = image::load_from_memory(include_bytes!("images/frymire-srgb.png"))
        .unwrap()
        .to_rgb8();
    let root = jxl_encoder::test_helpers::output_dir("subsampled-odd-shapes");
    let output = (0..)
        .find_map(|i| {
            let path = root.join(format!("run-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{}: {error}", path.display()),
            }
        })
        .unwrap();
    let mut failures = Vec::new();
    let mut cells = 0;
    for mode in [
        ChromaSubsampling::Sub420,
        ChromaSubsampling::Sub422,
        ChromaSubsampling::Sub440,
    ] {
        for (width, height) in [
            (1, 1),
            (1, 17),
            (17, 1),
            (9, 9),
            (17, 13),
            (32, 17),
            (259, 133),
            (513, 257),
        ] {
            for gray in [false, true] {
                let mut pixels = image::imageops::crop_imm(&source, 0, 0, width, height)
                    .to_image()
                    .into_raw();
                if gray {
                    for pixel in pixels.as_chunks_mut::<3>().0 {
                        pixel[1] = pixel[0];
                        pixel[2] = pixel[0];
                    }
                }
                let label = format!("{mode:?}-{width}x{height}-gray{gray}");
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let config = LossyConfig::new(0.1).with_chroma_subsampling(mode);
                    let bytes = config
                        .encode_request(width, height, PixelLayout::Rgb8)
                        .encode(&pixels)
                        .unwrap();
                    let row_bytes = width as usize * 3;
                    let stride = row_bytes + 5;
                    let mut strided = vec![0xA7; stride * height as usize];
                    for (source, destination) in
                        pixels.chunks(row_bytes).zip(strided.chunks_mut(stride))
                    {
                        destination[..row_bytes].copy_from_slice(source);
                    }
                    assert_eq!(
                        config
                            .encode_request(width, height, PixelLayout::Rgb8)
                            .with_row_stride(stride)
                            .encode(&strided)
                            .unwrap(),
                        bytes
                    );
                    let native = decode::verify_jxl_rs(&bytes, width as usize, height as usize);
                    let input = output.join(format!("{label}.jxl"));
                    let decoded = output.join(format!("{label}.png"));
                    std::fs::write(&input, bytes).unwrap();
                    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(input)
                        .arg(&decoded)
                        .args(["--num_threads=1", "--bits_per_sample=16"])
                        .output()
                        .unwrap();
                    assert!(
                        result.status.success(),
                        "{}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    let reference = image::open(decoded).unwrap().to_rgb16();
                    assert_eq!(reference.dimensions(), (width, height));
                    assert_eq!(native.len(), pixels.len());
                    let mut source_error = 0.0f64;
                    for ((&native, &reference), &source) in
                        native.iter().zip(reference.as_raw()).zip(&pixels)
                    {
                        let actual = native.clamp(0.0, 1.0);
                        let reference = f32::from(reference) / 65535.0;
                        assert!(
                            (actual - reference).abs() < 1e-4,
                            "decoder pixels {actual} vs {reference}"
                        );
                        source_error += f64::from((actual - f32::from(source) / 255.0).abs());
                    }
                    if gray {
                        assert!(
                            source_error / pixels.len() as f64 <= 2.0 / 255.0,
                            "grayscale source mean error {}",
                            source_error / pixels.len() as f64
                        );
                    }
                }));
                if let Err(error) = result {
                    let message = error
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| error.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "non-string panic".into());
                    failures.push(format!("{label}: {message}"));
                }
                cells += 1;
            }
        }
    }
    assert_eq!(cells, 48);
    assert!(
        failures.is_empty(),
        "{} failing shapes:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn malformed_public_jpeg_geometry_returns_an_error() {
    let source = include_bytes!("fixtures/jbrd/base_a_420.jpg");
    let jpeg = jxl_encoder::jpeg::read_jpeg(source, None, None).unwrap();
    for mutation in 0..3 {
        let mut invalid = jpeg.clone();
        let component = &mut invalid.components[0];
        match mutation {
            0 => {
                component.height_in_blocks -= 1;
                component.coeffs.truncate(
                    component.width_in_blocks as usize * component.height_in_blocks as usize * 64,
                );
            }
            1 => {
                component.coeffs.pop();
            }
            _ => component.h_samp_factor = 0,
        }
        let result = std::panic::catch_unwind(|| jxl_encoder::jpeg::encode_jpeg_to_jxl(&invalid));
        assert!(
            result.is_ok(),
            "invalid geometry mutation {mutation} panicked"
        );
        assert!(
            result.unwrap().is_err(),
            "invalid geometry mutation {mutation} was accepted"
        );
    }
}
