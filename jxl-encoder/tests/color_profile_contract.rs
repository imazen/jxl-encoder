//! ICC channel-model admission, Part 1 E.2.
#![cfg(feature = "__expert")]
use jxl_encoder::{AnimationFrame, AnimationParams, ImageMetadata, LosslessConfig, LossyConfig, PixelLayout};
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[path = "support/icc_contract.rs"]
mod icc_contract;
use icc_contract::{profiles, new_directory, preserve_unexpected_stream};

#[test]
fn mismatched_icc_channel_models_are_refused() {
    let profiles = profiles();
    let directory = new_directory("model-before");
    let mut accepted = Vec::new();
    for (name, strategy, mode) in encoder_policies::policies() {
        for gray in [false, true] {
            let layout = if gray { PixelLayout::Gray8 } else { PixelLayout::Rgb8 };
            let pixels = vec![64; 8 * 8 * layout.bytes_per_pixel()];
            let metadata = ImageMetadata::new().with_icc_profile(&profiles[usize::from(!gray)]);
            let lossy = LossyConfig::new(1.0).with_effort(1).with_strategy(strategy.clone()).with_mode(mode);
            let lossless = LosslessConfig::new().with_effort(1).with_strategy(strategy.clone()).with_mode(mode);
            for animation in [false, true] {
                let a = lossy.encode_request(8, 8, layout).with_metadata(&metadata);
                let b = lossless.encode_request(8, 8, layout).with_metadata(&metadata);
                let frames = [AnimationFrame::new(&pixels, 10)];
                let a = if animation { a.encode_animation(&AnimationParams::default(), &frames) } else { a.encode(&pixels) };
                let b = if animation { b.encode_animation(&AnimationParams::default(), &frames) } else { b.encode(&pixels) };
                for (kind, result) in [("lossy", a), ("lossless", b)] {
                    if let Ok(bytes) = result {
                        let label = format!("{name}-{gray}-{animation}-{kind}");
                        preserve_unexpected_stream(&directory, &label, &bytes);
                        accepted.push(label);
                    }
                }
            }
            let mut stream = lossy.encoder(8, 8, layout).unwrap().with_icc_profile(metadata.icc_profile().unwrap());
            stream.push_rows(&pixels, 8).unwrap();
            if let Ok(bytes) = stream.finish() {
                let label = format!("{name}-{gray}-lossy-stream");
                preserve_unexpected_stream(&directory, &label, &bytes);
                accepted.push(label);
            }
            let mut stream = lossless.encoder(8, 8, layout).unwrap().with_icc_profile(metadata.icc_profile().unwrap());
            stream.push_rows(&pixels, 8).unwrap();
            if let Ok(bytes) = stream.finish() {
                let label = format!("{name}-{gray}-lossless-stream");
                preserve_unexpected_stream(&directory, &label, &bytes);
                accepted.push(label);
            }
        }
    }
    assert!(accepted.is_empty(), "accepted mismatched ICC models: {accepted:?}; streams at {}", directory.display());
}

