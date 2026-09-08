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
                assert!(matches!(result, Err(e) if matches!(e.error(), EncodeError::Cancelled)));
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
    let path = dir.join(format!("{:x}.jxl", Sha256::digest(bytes)));
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
