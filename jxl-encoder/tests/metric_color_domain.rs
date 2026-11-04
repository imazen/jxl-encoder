//! Reference and reconstruction supplied to a metric must share one RGB domain.
#![cfg(feature = "__external_metric")]

use jxl_encoder::{ExternalLoopMetric, ExternalMetric, LossyConfig, PixelLayout};
use std::sync::{Arc, Mutex};

#[path = "../examples/distance_targeting_probe/decode.rs"]
mod decode;

#[derive(Default)]
struct Samples {
    reference: Option<[f32; 3]>,
    reconstructed: Vec<[f32; 3]>,
}

struct Capture(Arc<Mutex<Samples>>);

impl ExternalLoopMetric for Capture {
    fn set_reference(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        _: usize,
        _: usize,
    ) -> Result<(), String> {
        self.0.lock().unwrap().reference = Some([r[0], g[0], b[0]]);
        Ok(())
    }

    fn compare(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        _: usize,
        width: usize,
        height: usize,
        diffmap: &mut Vec<f32>,
    ) -> Result<f64, String> {
        self.0
            .lock()
            .unwrap()
            .reconstructed
            .push([r[0], g[0], b[0]]);
        diffmap.clear();
        diffmap.resize(width * height, 0.0);
        Ok(0.0)
    }
}

fn capture(intensity: f32, encoding: Option<jxl_encoder::ColorEncoding>) -> (Samples, Vec<u8>) {
    let state = Arc::new(Mutex::new(Samples::default()));
    let metric = ExternalMetric::new(Capture(Arc::clone(&state)));
    let pixels: Vec<u8> = (0..32 * 32)
        .flat_map(|_| [0.1f32, 0.2, 0.3].into_iter().flat_map(f32::to_ne_bytes))
        .collect();
    let config = LossyConfig::new(0.01)
        .with_effort(8)
        .with_hdr_loss(jxl_encoder::HdrLoss::Butteraugli)
        .with_butteraugli_iters(1);
    let mut request = config
        .encode_request(32, 32, PixelLayout::RgbLinearF32)
        .with_intensity_target(intensity)
        .with_external_metric(&metric);
    if let Some(encoding) = encoding {
        request = request.with_color_encoding(encoding);
    }
    let encoded = request.encode(&pixels).unwrap();
    drop(metric);
    (
        Arc::try_unwrap(state).ok().unwrap().into_inner().unwrap(),
        encoded,
    )
}

#[test]
fn intensity_target_does_not_change_metric_rgb_units() {
    for intensity in [255.0, 1000.0, 10000.0] {
        let (samples, _) = capture(intensity, None);
        let reference = samples.reference.unwrap();
        assert!(!samples.reconstructed.is_empty());
        for reconstructed in samples.reconstructed {
            for c in 0..3 {
                assert!(
                    (reference[c] - reconstructed[c]).abs() < 0.01,
                    "intensity {intensity}, channel {c}: reference {reference:?}, reconstruction {reconstructed:?}"
                );
            }
        }
    }
}

#[test]
fn wide_gamut_reference_uses_the_reconstruction_primaries() {
    let (samples, _) = capture(255.0, Some(jxl_encoder::ColorEncoding::bt2100_pq()));
    let reference = samples.reference.unwrap();
    assert!(!samples.reconstructed.is_empty());
    for reconstructed in samples.reconstructed {
        for c in 0..3 {
            assert!(
                (reference[c] - reconstructed[c]).abs() < 0.01,
                "channel {c}: reference {reference:?}, reconstruction {reconstructed:?}"
            );
        }
    }
}

#[test]
fn metric_reconstruction_matches_independent_linear_decode() {
    let directory = std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
        .join("tmp")
        .join(format!("jxl138-metric-domain-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    for (i, (intensity, encoding)) in [
        (1000.0, None),
        (10000.0, Some(jxl_encoder::ColorEncoding::bt2100_pq())),
    ]
    .into_iter()
    .enumerate()
    {
        let (samples, bytes) = capture(intensity, encoding);
        let primary = decode::verify_jxl_rs(&bytes, 32, 32);
        let input = directory.join(format!("{i}.jxl"));
        std::fs::write(&input, bytes).unwrap();
        for linear in [false, true] {
            let output = directory.join(format!("{i}-{linear}.png"));
            let mut command = std::process::Command::new(jxl_encoder::test_helpers::djxl_path());
            command
                .arg(&input)
                .arg(&output)
                .args(["--num_threads=1", "--bits_per_sample=16"]);
            if linear {
                command.arg("--color_space=RGB_D65_SRG_Rel_Lin");
            }
            let result = command.output().unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let pixels = image::open(output).unwrap().to_rgb16();
            assert_eq!(pixels.dimensions(), (32, 32));
            for (c, &native) in primary.iter().take(3).enumerate() {
                let actual = pixels.get_pixel(0, 0).0[c] as f32 / 65535.0;
                let expected = if linear {
                    samples.reconstructed.last().unwrap()[c]
                } else {
                    native
                };
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "intensity={intensity}, linear={linear}, channel={c}: djxl {actual} vs {expected}"
                );
            }
        }
    }
}
