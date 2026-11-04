//! Explicit HDR loop budgets must run the selected metric; defaults stay off.
#![cfg(all(feature = "std", feature = "butteraugli-loop"))]
use jxl_encoder::{ColorEncoding, HdrLoss, LoopTrace, LossyConfig, PixelLayout};
#[path = "../examples/distance_targeting_probe/decode.rs"]
mod decode;

#[test]
fn explicit_vdp2_iterations_run_on_hdr_and_default_stays_off() {
    let root = jxl_encoder::test_helpers::output_dir("hdr-loop-override");
    let dir = (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{}: {e}", p.display()),
            }
        })
        .unwrap();
    let mut failures = Vec::new();
    let mut cell = 0;
    for (w, h) in [(32_usize, 17_usize), (259, 17)] {
        for kind in 0..4 {
            let (layout, color) = match kind {
                0 => (PixelLayout::RgbPqF32, None),
                1 => (PixelLayout::RgbHlgF32, None),
                2 => (PixelLayout::Rgb8, Some(ColorEncoding::bt2100_pq())),
                _ => (PixelLayout::Rgb8, Some(ColorEncoding::bt2100_hlg())),
            };
            let pixels: Vec<u8> = (0..w * h * 3)
                .flat_map(|i| {
                    if kind < 2 {
                        ((i % 256) as f32 / 600.0).to_ne_bytes().to_vec()
                    } else {
                        vec![(i % 128) as u8]
                    }
                })
                .collect();
            for loss in [HdrLoss::Auto, HdrLoss::Vdp2] {
                let mut baseline = None;
                for iters in [None, Some(0), Some(2)] {
                    let mut config = LossyConfig::new(1.0).with_effort(8).with_hdr_loss(loss);
                    if let Some(n) = iters {
                        config = config.with_butteraugli_iters(n);
                    }
                    let trace = LoopTrace::new();
                    let mut request = config
                        .encode_request(w as u32, h as u32, layout)
                        .with_loop_trace(&trace);
                    if let Some(color) = &color {
                        request = request.with_color_encoding(color.clone());
                    }
                    let encoded = request.encode(&pixels).unwrap();
                    let frames = trace.take();
                    let wanted = if iters == Some(2) { 3 } else { 0 };
                    if frames.len() != wanted {
                        failures.push(format!("{w}x{h} kind={kind} loss={loss:?} iters={iters:?}: {} frames, expected {wanted}",frames.len()));
                    }
                    for frame in &frames {
                        assert_eq!(frame.metric, "vdp2");
                        assert!(frame.score.is_finite());
                        assert!(
                            frame
                                .recon
                                .iter()
                                .all(|p| p.len() == w * h && p.iter().all(|v| v.is_finite()))
                        );
                    }
                    if iters.is_none() {
                        baseline = Some(encoded.clone());
                    }
                    if iters == Some(0) {
                        assert_eq!(Some(&encoded), baseline.as_ref());
                    }
                    decode::verify_jxl_rs(&encoded, w, h);
                    let input = dir.join(format!("cell-{cell}.jxl"));
                    let output = dir.join(format!("cell-{cell}.pfm"));
                    std::fs::write(&input, &encoded).unwrap();
                    let rendered =
                        std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                            .arg(&input)
                            .arg(&output)
                            .arg("--num_threads=1")
                            .output()
                            .unwrap();
                    assert!(
                        rendered.status.success(),
                        "{}",
                        String::from_utf8_lossy(&rendered.stderr)
                    );
                    assert!(std::fs::metadata(output).unwrap().len() > 0);
                    cell += 1;
                }
            }
        }
    }
    assert_eq!(cell, 48);
    assert!(
        failures.is_empty(),
        "explicit HDR loop budget ignored: {failures:?}"
    );
}
