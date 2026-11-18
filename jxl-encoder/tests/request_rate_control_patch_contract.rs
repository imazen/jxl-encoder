//! Winning patch dictionaries through the request controller, LF and layout alpha.
#![cfg(all(
    feature = "__expert",
    feature = "rate-control",
    feature = "corpus-tests"
))]
use jxl_encoder::{LossyConfig, PixelLayout};
#[path = "support/decoded_frames.rs"]
mod decoded_frames;
#[path = "support/frame_level_helpers.rs"]
mod helpers;
#[path = "support/pfm.rs"]
mod pfm;
#[path = "support/encoder_policies.rs"]
mod policies;

#[test]
fn request_controller_keeps_winning_dictionary_with_lf_and_alpha() {
    let root = helpers::directory();
    let original =
        std::fs::read(jxl_encoder::test_helpers::corpus_dir().join("gb82-sc/imac_g3.png"))
            .expect("caller-provisioned real patch witness");
    std::fs::write(root.join("source.png"), &original).unwrap();
    let source = image::load_from_memory(&original).unwrap().to_rgb8();
    let tile = image::imageops::crop_imm(&source, 528, 512, 512, 512).to_image();
    let (width, height) = (2048_usize, 512_usize);
    let rgb: Vec<u8> = (0..height)
        .flat_map(|y| {
            let tile = &tile;
            (0..width).flat_map(move |x| tile.get_pixel((x % 512) as u32, y as u32).0)
        })
        .collect();
    let mut journal = std::fs::File::create(root.join("cases.tsv")).unwrap();
    use std::io::Write;
    for (policy, strategy, mode) in policies::policies() {
        for alpha in [false, true] {
            let layout = if alpha {
                PixelLayout::Rgba8
            } else {
                PixelLayout::Rgb8
            };
            let pixels: Vec<u8> = if alpha {
                rgb.as_chunks::<3>()
                    .0
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2], 137])
                    .collect()
            } else {
                rgb.clone()
            };
            let stride = width * layout.bytes_per_pixel() + 5;
            let mut padded = vec![0xA5; stride * height];
            for (out, input) in padded
                .chunks_exact_mut(stride)
                .zip(pixels.chunks_exact(width * layout.bytes_per_pixel()))
            {
                out[..input.len()].copy_from_slice(input);
            }
            for iterations in [0, 2] {
                let mut baseline: Option<Vec<f64>> = None;
                for lf in [false, true] {
                    let out = root.join(format!("{policy}-alpha{alpha}-i{iterations}-lf{lf}"));
                    std::fs::create_dir(&out).unwrap();
                    let cfg = LossyConfig::new(0.01)
                        .with_effort(7)
                        .with_threads(1)
                        .with_strategy(strategy.clone())
                        .with_mode(mode)
                        .with_patches(true)
                        .with_gaborish(false)
                        .with_epf_level(0)
                        .with_force_strategy(Some(0))
                        .with_lf_frame(lf);
                    #[cfg(feature = "butteraugli-loop")]
                    let cfg = cfg.with_butteraugli_iters(0);
                    let request = || {
                        cfg.encode_request(width as u32, height as u32, layout)
                            .with_rate_control(jxl_encoder::vardct::RateControlConfig {
                                max_iterations: iterations,
                                ..Default::default()
                            })
                    };
                    writeln!(
                        journal,
                        "{policy}\talpha{alpha}\ti{iterations}\tlf{lf}\tencoding"
                    )
                    .unwrap();
                    journal.flush().unwrap();
                    let bytes = request().encode(&pixels).unwrap();
                    assert_eq!(
                        request().with_row_stride(stride).encode(&padded).unwrap(),
                        bytes
                    );
                    std::fs::write(out.join("input.jxl"), &bytes).unwrap();
                    let parsed = jxl_oxide::JxlImage::builder()
                        .read(std::io::Cursor::new(&bytes))
                        .unwrap();
                    let references = (0..parsed.num_loaded_frames())
                        .filter(|&i| parsed.frame(i).unwrap().header().frame_type as u8 == 2)
                        .count();
                    assert_eq!(references, 1, "{out:?}: actual winning dictionary required");
                    assert_eq!(parsed.num_loaded_frames(), if lf { 3 } else { 2 });
                    let primary = decoded_frames::decode_current_frames(
                        &bytes,
                        "request controller dictionary",
                    );
                    assert_eq!(primary.len(), 1);
                    assert_eq!(primary[0].0.len(), rgb.len());
                    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                        .arg(out.join("input.jxl"))
                        .arg(out.join("native.pfm"))
                        .args(["--num_threads=1", "--color_space=RGB_D65_SRG_Rel_SRG"])
                        .output()
                        .unwrap();
                    std::fs::write(out.join("native.log"), &result.stderr).unwrap();
                    assert!(
                        result.status.success(),
                        "{}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    let native = pfm::read_pfm_samples(&out.join("native.pfm"), (width, height));
                    assert_eq!(native.len(), rgb.len());
                    assert!(
                        native
                            .iter()
                            .zip(&primary[0].0)
                            .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 0.001)
                    );
                    let errors: Vec<f64> = [&primary[0].0, &native]
                        .into_iter()
                        .map(|values| {
                            values
                                .iter()
                                .zip(&rgb)
                                .map(|(a, b)| (f64::from(*a) - f64::from(*b) / 255.0).abs())
                                .sum::<f64>()
                                / rgb.len() as f64
                        })
                        .collect();
                    assert!(
                        errors.iter().all(|e| *e <= 0.01),
                        "{out:?}: source MAE {errors:?}"
                    );
                    if let Some(base) = &baseline {
                        assert!(
                            errors.iter().zip(base).all(|(a, b)| *a < *b * 1.5 + 0.002),
                            "{out:?}: LF {errors:?} baseline {base:?}"
                        );
                    } else {
                        baseline = Some(errors.clone());
                    }
                    assert_eq!(primary[0].1.len(), usize::from(alpha));
                    if alpha {
                        assert_eq!(primary[0].1[0].len(), width * height);
                        assert!(
                            primary[0].1[0].iter().all(
                                |v| v.is_finite() && (*v * 65535.0).round() as u16 == 137 * 257
                            )
                        );
                        let result =
                            std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
                                .arg(out.join("input.jxl"))
                                .arg(out.join("alpha.png"))
                                .args(["--num_threads=1", "--bits_per_sample=16"])
                                .output()
                                .unwrap();
                        std::fs::write(out.join("alpha.log"), &result.stderr).unwrap();
                        assert!(result.status.success());
                        let image = image::open(out.join("alpha.png")).unwrap().to_rgba16();
                        assert_eq!(image.dimensions(), (width as u32, height as u32));
                        assert!(image.pixels().all(|p| p.0[3] == 137 * 257));
                    }
                    std::fs::write(out.join("observations.json"), serde_json::to_vec_pretty(&serde_json::json!({"level": helpers::level(&bytes), "reference_frames":references,"lf":lf,"source_mae":errors})).unwrap()).unwrap();
                    writeln!(
                        journal,
                        "{policy}\talpha{alpha}\ti{iterations}\tlf{lf}\tcomplete"
                    )
                    .unwrap();
                    journal.flush().unwrap();
                }
            }
        }
    }
}
