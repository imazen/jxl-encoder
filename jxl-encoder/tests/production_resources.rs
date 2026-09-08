//! Real-image resource and concurrency gates. The caller enables corpus-tests.
#![cfg(feature = "corpus-tests")]
use jxl_encoder::{EncodeError, Limits, LosslessConfig, LossyConfig, PixelLayout};
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "../examples/distance_targeting_probe/decode.rs"]
mod decode;

struct CancelAfter {
    allowed: usize,
    calls: AtomicUsize,
}
impl enough::Stop for CancelAfter {
    fn check(&self) -> Result<(), enough::StopReason> {
        if self.calls.fetch_add(1, Ordering::Relaxed) >= self.allowed {
            Err(enough::StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn photo(width: u32, height: u32) -> Vec<u8> {
    let path =
        jxl_encoder::test_helpers::corpus_dir().join("CID22/CID22-512/validation/1418519.png");
    image::open(path)
        .expect("production gate corpus")
        .crop_imm(0, 0, width, height)
        .to_rgb8()
        .into_raw()
}

#[test]
fn late_cancellation_preserves_output_and_does_not_poison_next_encode() {
    for (width, height) in [(63, 47), (511, 259)] {
        let pixels = photo(width, height);
        let limits = Limits::new()
            .with_max_memory_bytes(256 << 20)
            .with_fallible_alloc(true);
        for lossless in [false, true] {
            for allowed in [0, 2] {
                let stop = CancelAfter {
                    allowed,
                    calls: AtomicUsize::new(0),
                };
                let lossy = LossyConfig::new(4.0).with_effort(8).with_threads(1);
                let lossless_cfg = LosslessConfig::new().with_effort(7).with_threads(1);
                let request = if lossless {
                    lossless_cfg.encode_request(width, height, PixelLayout::Rgb8)
                } else {
                    lossy.encode_request(width, height, PixelLayout::Rgb8)
                };
                let mut destination = vec![1, 3, 5, 7];
                let result = request
                    .with_limits(&limits)
                    .with_stop(&stop)
                    .encode_into(&pixels, &mut destination);
                assert!(
                    matches!(&result, Err(e) if matches!(e.error(), EncodeError::Cancelled)),
                    "lossless={lossless} allowed={allowed} size={width}x{height} calls={} result={result:?}",
                    stop.calls.load(Ordering::Relaxed)
                );
                assert!(
                    stop.calls.load(Ordering::Relaxed) > allowed,
                    "late checkpoint was not reached"
                );
                assert_eq!(
                    destination,
                    [1, 3, 5, 7],
                    "cancelled encode appended partial output"
                );
                let next = if lossless {
                    lossless_cfg.encode(&pixels, width, height, PixelLayout::Rgb8)
                } else {
                    lossy.encode(&pixels, width, height, PixelLayout::Rgb8)
                }
                .expect("encode after cancellation");
                verify(&next, width, height);
            }
        }
    }
}

#[test]
fn concurrent_canonicalization_has_isolated_limits_and_identical_bytes() {
    let (width, height) = (511, 259);
    let rgb = photo(width, height);
    let limits = Limits::new()
        .with_max_memory_bytes(256 << 20)
        .with_fallible_alloc(true);
    let tight = Limits::new()
        .with_max_memory_bytes(1)
        .with_fallible_alloc(true);
    for transparent in [false, true] {
        let pixels: Vec<u8> = rgb
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], if transparent { p[0] } else { 255 }])
            .collect();
        for canonical in [false, true] {
            for effort in [5, 8] {
                let cfg = LossyConfig::new(4.0)
                    .with_effort(effort)
                    .with_threads(1)
                    .with_canonicalize_input(canonical);
                let baseline = cfg
                    .encode_request(width, height, PixelLayout::Rgba8)
                    .with_limits(&limits)
                    .encode(&pixels)
                    .unwrap();
                verify(&baseline, width, height);
                for concurrency in [1, 2, 4] {
                    let barrier = std::sync::Barrier::new(concurrency);
                    std::thread::scope(|scope| {
                        let handles: Vec<_> = (0..concurrency).map(|_| scope.spawn(|| {
                            barrier.wait();
                            let rejected = cfg.encode_request(width, height, PixelLayout::Rgba8)
                                .with_limits(&tight).encode(&pixels);
                            assert!(matches!(rejected, Err(e) if matches!(e.error(), EncodeError::LimitExceeded { .. })));
                            cfg.encode_request(width, height, PixelLayout::Rgba8)
                                .with_limits(&limits).encode(&pixels).unwrap()
                        })).collect();
                        for handle in handles {
                            assert_eq!(
                                handle.join().unwrap(),
                                baseline,
                                "concurrent request changed the encoded bytes"
                            );
                        }
                    });
                }
            }
        }
    }
}

fn verify(bytes: &[u8], width: u32, height: u32) {
    use sha2::{Digest, Sha256};
    decode::verify_jxl_rs(bytes, width as usize, height as usize);
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/production-resource-validation");
    std::fs::create_dir_all(&dir).unwrap();
    let digest: String = Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let path = dir.join(format!("{digest}.jxl"));
    std::fs::write(&path, bytes).unwrap();
    let output = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(path)
        .args(["--disable_output", "--num_threads=1"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "djxl: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn real_screenshot_streaming_keeps_one_shot_content_dispatch() {
    let source =
        image::open(jxl_encoder::test_helpers::corpus_dir().join("gb82-sc/codec_wiki.png"))
            .expect("codec_wiki corpus")
            .to_rgb8();
    let (width, height) = (513, 259);
    let pixels = image::imageops::crop_imm(
        &source,
        (source.width() - width) / 2,
        (source.height() - height) / 2,
        width,
        height,
    )
    .to_image()
    .into_raw();
    let cfg = LossyConfig::new(0.1 + 40.0 * (24.9 / 255.0))
        .with_effort(5)
        .with_threads(1);
    let baseline = cfg
        .encode(&pixels, width, height, PixelLayout::Rgb8)
        .unwrap();
    for rows in [1, 7, height] {
        let mut encoder = cfg.encoder(width, height, PixelLayout::Rgb8).unwrap();
        for chunk in pixels.chunks(width as usize * 3 * rows as usize) {
            encoder
                .push_rows(chunk, (chunk.len() / (width as usize * 3)) as u32)
                .unwrap();
        }
        let streamed = encoder.finish().unwrap();
        assert!(
            baseline == streamed,
            "content dispatch diverged at chunk_rows={rows}: one-shot {} bytes, streaming {}",
            baseline.len(),
            streamed.len()
        );
        verify(&streamed, width, height);
    }
}

/// Real spatial content in distinct storage/transfer representations. This
/// verifies entry-point equivalence, not HDR perceptual quality calibration.
#[test]
fn streaming_preserves_source_layout_and_color_metadata() {
    for (width, height) in [(63, 47), (511, 259)] {
        let rgb = photo(width, height);
        for layout in [
            PixelLayout::Rgb8,
            PixelLayout::Rgba8,
            PixelLayout::Bgr8,
            PixelLayout::Bgra8,
            PixelLayout::Gray8,
            PixelLayout::GrayAlpha8,
            PixelLayout::Rgb16,
            PixelLayout::Rgba16,
            PixelLayout::Gray16,
            PixelLayout::GrayAlpha16,
            PixelLayout::RgbLinearF32,
            PixelLayout::RgbaLinearF32,
            PixelLayout::GrayLinearF32,
            PixelLayout::GrayAlphaLinearF32,
            PixelLayout::RgbLinearF16,
            PixelLayout::RgbaLinearF16,
            PixelLayout::GrayLinearF16,
            PixelLayout::GrayAlphaLinearF16,
            PixelLayout::RgbPqF32,
            PixelLayout::RgbaPqF32,
            PixelLayout::RgbHlgF32,
            PixelLayout::RgbaHlgF32,
            PixelLayout::RgbBt709F32,
            PixelLayout::RgbaBt709F32,
        ] {
            let gray = matches!(
                layout,
                PixelLayout::Gray8
                    | PixelLayout::GrayAlpha8
                    | PixelLayout::Gray16
                    | PixelLayout::GrayAlpha16
                    | PixelLayout::GrayLinearF32
                    | PixelLayout::GrayAlphaLinearF32
                    | PixelLayout::GrayLinearF16
                    | PixelLayout::GrayAlphaLinearF16
            );
            let half = matches!(
                layout,
                PixelLayout::RgbLinearF16
                    | PixelLayout::RgbaLinearF16
                    | PixelLayout::GrayLinearF16
                    | PixelLayout::GrayAlphaLinearF16
            );
            let mut pixels = Vec::new();
            for p in rgb.chunks_exact(3) {
                let mut values = if gray { vec![p[0]] } else { p.to_vec() };
                if matches!(layout, PixelLayout::Bgr8 | PixelLayout::Bgra8) {
                    values.swap(0, 2);
                }
                if layout.has_alpha() {
                    values.push(p[1]);
                }
                let bytes_per_sample = layout.bytes_per_pixel() / values.len();
                for v in values {
                    match bytes_per_sample {
                        1 => pixels.push(v),
                        2 if half => {
                            // Exact binary16 encodings of k/16, k=0..15.
                            // Quantize fixture values only; no encoder conversion is shared.
                            const HALF: [u16; 16] = [
                                0, 0x2c00, 0x3000, 0x3200, 0x3400, 0x3500, 0x3600, 0x3700, 0x3800,
                                0x3880, 0x3900, 0x3980, 0x3a00, 0x3a80, 0x3b00, 0x3b80,
                            ];
                            pixels.extend_from_slice(&HALF[usize::from(v >> 4)].to_ne_bytes());
                        }
                        2 => pixels.extend_from_slice(&(u16::from(v) * 257).to_ne_bytes()),
                        4 => pixels.extend_from_slice(&(f32::from(v) / 255.0).to_ne_bytes()),
                        _ => unreachable!("fixture storage width"),
                    }
                }
            }
            for effort in [5, 8] {
                let cfg = LossyConfig::new(4.0).with_effort(effort).with_threads(1);
                let baseline = cfg.encode(&pixels, width, height, layout).unwrap();
                for rows in [1, 7] {
                    let mut encoder = cfg.encoder(width, height, layout).unwrap();
                    let row_bytes = width as usize * layout.bytes_per_pixel();
                    for chunk in pixels.chunks(row_bytes * rows) {
                        encoder
                            .push_rows(chunk, (chunk.len() / row_bytes) as u32)
                            .unwrap();
                    }
                    let streamed = encoder.finish().unwrap();
                    assert!(
                        baseline == streamed,
                        "layout={layout:?} effort={effort} chunk={rows} size={width}x{height}"
                    );
                }
                verify(&baseline, width, height);
            }
        }
    }
}

#[test]
fn streaming_preserves_explicit_hdr_intensity_equal_to_sdr_default() {
    for (width, height) in [(63, 47), (511, 259)] {
        let pixels: Vec<_> = photo(width, height)
            .iter()
            .flat_map(|&v| (f32::from(v) / 255.0).to_ne_bytes())
            .collect();
        let cfg = LossyConfig::new(4.0).with_effort(8).with_threads(1);
        let layout = PixelLayout::RgbPqF32;
        let baseline = cfg
            .encode_request(width, height, layout)
            .with_intensity_target(255.0)
            .encode(&pixels)
            .unwrap();
        let mut encoder = cfg
            .encoder(width, height, layout)
            .unwrap()
            .with_intensity_target(255.0);
        encoder.push_rows(&pixels, height).unwrap();
        assert!(
            encoder.finish().unwrap() == baseline,
            "explicit 255-nit override was lost"
        );
        verify(&baseline, width, height);
    }
}
