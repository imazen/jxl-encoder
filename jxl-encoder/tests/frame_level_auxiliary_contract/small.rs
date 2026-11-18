//! Level-five controls for the container-owning still and streaming routes.
use super::*;

#[test]
fn small_frames_keep_level_five_across_offered_policies() {
    let root = directory();
    for (width, height) in [(9_u32, 3_u32), (129, 129), (300, 17)] {
        let pixels = vec![64_u8; width as usize * height as usize * 3];
        let stride = width as usize * 3 + 5;
        let mut padded = vec![0xA5; stride * height as usize];
        for row in padded.chunks_exact_mut(stride) {
            row[..width as usize * 3].fill(64);
        }
        for (policy, strategy, mode) in encoder_policies::policies() {
            for effort in [1, 3] {
                for lossless in [false, true] {
                    let label = format!("{width}x{height}-{policy}-e{effort}-lossless-{lossless}");
                    let config = jxl_encoder::LosslessConfig::new()
                        .with_effort(effort)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode);
                    let lossy = LossyConfig::new(1.0)
                        .with_effort(effort)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode);
                    let request = || {
                        if lossless {
                            config.encode_request(width, height, PixelLayout::Rgb8)
                        } else {
                            lossy.encode_request(width, height, PixelLayout::Rgb8)
                        }
                    };
                    let bytes = request().encode(&pixels).unwrap();
                    assert_eq!(level(&bytes), 5, "{label}");
                    assert_eq!(
                        request().with_row_stride(stride).encode(&padded).unwrap(),
                        bytes
                    );
                    let mut sink = Vec::new();
                    request().encode_to(&pixels, &mut sink).unwrap();
                    assert_eq!(sink, bytes, "writer {label}");
                    let streaming = if lossless {
                        let mut stream = config.encoder(width, height, PixelLayout::Rgb8).unwrap();
                        for row in pixels.chunks_exact(width as usize * 3) {
                            stream.push_rows(row, 1).unwrap();
                        }
                        stream.finish().unwrap()
                    } else {
                        let mut stream = lossy.encoder(width, height, PixelLayout::Rgb8).unwrap();
                        for row in pixels.chunks_exact(width as usize * 3) {
                            stream.push_rows(row, 1).unwrap();
                        }
                        stream.finish().unwrap()
                    };
                    assert_eq!(streaming, bytes, "streaming {label}");
                    let decoded = decoded_frames::decode_current_frames(&bytes, &label);
                    assert_eq!(decoded.len(), 1);
                    assert_eq!(decoded[0].0.len(), pixels.len());
                    assert!(decoded[0].1.is_empty());
                    let input = root.join(format!("{label}.jxl"));
                    let output = root.join(format!("{label}.png"));
                    std::fs::write(&input, &bytes).unwrap();
                    let native = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(&input)
                        .arg(&output)
                        .arg("--bits_per_sample=16")
                        .arg("--num_threads=1")
                        .output()
                        .unwrap();
                    std::fs::write(root.join(format!("{label}.log")), &native.stderr).unwrap();
                    assert!(
                        native.status.success(),
                        "{label}: {}",
                        String::from_utf8_lossy(&native.stderr)
                    );
                    let native = image::open(output).unwrap().to_rgb16();
                    assert_eq!(native.dimensions(), (width, height));
                    for ((&a, &b), &source) in decoded[0].0.iter().zip(native.as_raw()).zip(&pixels)
                    {
                        assert!(
                            (a.clamp(0.0, 1.0) - f32::from(b) / 65535.0).abs() <= 0.001,
                            "{label}"
                        );
                        if lossless {
                            assert_eq!((a * 255.0).round(), f32::from(source));
                            assert_eq!(b, u16::from(source) * 257);
                        }
                    }
                }
            }
        }
    }
}
