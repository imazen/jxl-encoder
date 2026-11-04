//! Absolute PQ signal values must survive caller-selected display intensity.
use jxl_encoder::{
    AnimationFrame, AnimationParams, ColorEncoding, ImageMetadata, LossyConfig, PixelLayout,
};

#[path = "../examples/distance_targeting_probe/decode.rs"]
mod decode;

#[test]
fn pq_signal_is_independent_of_intensity_metadata() {
    let directory = std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
        .join("tmp")
        .join(format!("jxl-pq-intensity-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let mut cell = 0;
    for (w, h) in [(32, 17), (259, 17)] {
        for kind in 0..3 {
            let signal = [102_u8, 128, 153];
            let (layout, pixels) = match kind {
                0 => (PixelLayout::Rgb8, signal.repeat(w * h)),
                1 => (
                    PixelLayout::Rgb16,
                    signal
                        .repeat(w * h)
                        .into_iter()
                        .flat_map(|v| (u16::from(v) * 257).to_ne_bytes())
                        .collect(),
                ),
                _ => (
                    PixelLayout::RgbPqF32,
                    signal
                        .repeat(w * h)
                        .into_iter()
                        .flat_map(|v| (f32::from(v) / 255.0).to_ne_bytes())
                        .collect(),
                ),
            };
            for intensity in [255.0, 1000.0, 4000.0, 10000.0] {
                for request_override in [false, true] {
                    for animation in [false, true] {
                        // The animation API accepts integer source layouts.
                        if animation && kind == 2 {
                            continue;
                        }
                        let metadata =
                            ImageMetadata::new().with_intensity_target(if request_override {
                                600.0
                            } else {
                                intensity
                            });
                        let config = LossyConfig::new(0.01).with_effort(1).with_gaborish(false);
                        let mut request = config
                            .encode_request(w as u32, h as u32, layout)
                            .with_color_encoding(ColorEncoding::bt2100_pq())
                            .with_metadata(&metadata);
                        if request_override {
                            request = request.with_intensity_target(intensity);
                        }
                        let bytes = if animation {
                            request.encode_animation(
                                &AnimationParams::default(),
                                &[
                                    AnimationFrame::new(&pixels, 10),
                                    AnimationFrame::new(&pixels, 20),
                                ],
                            )
                        } else {
                            request.encode(&pixels)
                        }
                        .unwrap();
                        if !animation {
                            let mut streaming = config
                                .encoder(w as u32, h as u32, layout)
                                .unwrap()
                                .with_color_encoding(ColorEncoding::bt2100_pq())
                                .with_intensity_target(intensity);
                            let row_bytes = pixels.len() / h;
                            for rows in pixels.chunks(row_bytes * 7) {
                                streaming
                                    .push_rows(rows, (rows.len() / row_bytes) as u32)
                                    .unwrap();
                            }
                            assert_eq!(streaming.finish().unwrap(), bytes, "streaming cell {cell}");
                        }
                        let input = directory.join(format!("{cell}.jxl"));
                        std::fs::write(&input, &bytes).unwrap();
                        let native = decode::verify_jxl_rs(&bytes, w, h);
                        let output = directory.join(format!("{cell}.png"));
                        let result =
                            std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(&input)
                                .arg(&output)
                                .args([
                                    "--num_threads=1",
                                    "--bits_per_sample=16",
                                    "--color_space=RGB_D65_202_Rel_PeQ",
                                ])
                                .output()
                                .unwrap();
                        assert!(
                            result.status.success(),
                            "{}",
                            String::from_utf8_lossy(&result.stderr)
                        );
                        let reference = image::open(output).unwrap().to_rgb16();
                        assert_eq!(reference.dimensions(), (w as u32, h as u32));
                        for (name, decoded) in [
                            ("native", native),
                            (
                                "djxl",
                                reference
                                    .as_raw()
                                    .iter()
                                    .map(|&v| f32::from(v) / 65535.0)
                                    .collect(),
                            ),
                        ] {
                            for (i, actual) in decoded.iter().enumerate() {
                                let expected = f32::from(signal[i % 3]) / 255.0;
                                assert!(
                                    (actual - expected).abs() < 0.005,
                                    "{name} cell {cell}: {w}x{h}, {layout:?}, intensity={intensity}, request={request_override}, animation={animation}, sample {i}: {actual} vs {expected}"
                                );
                            }
                        }
                        cell += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cell, 80);
}
