//! Decoder checks run in their own process because reconstruction hooks are global.
#![cfg(feature = "__internal_recon_hook")]

#[path = "../examples/distance_targeting_probe/decode.rs"]
mod decode;

use jxl_encoder::api::{EncoderImprovementsCustom, EncoderStrategy, EpfDispatch, EpfSharpnessSeed};
use jxl_encoder::vardct::__recon_hook;
use jxl_encoder::{LossyConfig, PixelLayout};

#[test]
fn visible_reconstruction_matches_both_decoders() {
    let source = image::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/images/frymire.png"
    ))
    .expect("tracked real-world graphics fixture");
    let artifacts =
        std::env::temp_dir().join(format!("jxl103-reconstruction-{}", std::process::id()));
    std::fs::create_dir_all(&artifacts).unwrap();
    let djxl = jxl_encoder::test_helpers::djxl_path();
    for (width, height) in [(64u32, 64u32), (259, 133), (512, 512)] {
        let pixels = source.crop_imm(0, 0, width, height).to_rgb8();
        for (strategy, epf, gab) in [(7u8, 0, false), (0, 0, true), (0, 3, true)] {
            // Pin both sides to the same sharpness policy. Selection of a
            // sharpness map is separate from reconstructing its decoded pixels.
            let config = LossyConfig::new(4.0)
                .with_effort(8)
                .with_force_strategy(Some(strategy))
                .with_epf_level(epf)
                .with_gaborish(gab)
                .with_strategy(EncoderStrategy::Custom(Box::new(
                    EncoderImprovementsCustom {
                        epf_dispatch: EpfDispatch::AlwaysDefault,
                        buttloop_epf_sharpness_seed: EpfSharpnessSeed::LegacyUniform4,
                        ..Default::default()
                    },
                )));
            let _ = __recon_hook::take_last();
            __recon_hook::set_capture_enabled(true);
            let encoded = config
                .encode(pixels.as_raw(), width, height, PixelLayout::Rgb8)
                .unwrap();
            __recon_hook::set_capture_enabled(false);
            let recon = __recon_hook::take_last().expect("final reconstruction captured");
            assert_eq!(
                (recon.width, recon.height),
                (width as usize, height as usize)
            );
            let primary = decode::verify_jxl_rs(&encoded, width as usize, height as usize);
            for (i, pixel) in primary.as_chunks::<3>().0.iter().enumerate() {
                for (c, (encoded, internal)) in pixel
                    .iter()
                    .zip([recon.r[i], recon.g[i], recon.b[i]])
                    .enumerate()
                {
                    let linear = decode::srgb_to_linear(*encoded);
                    assert!(
                        (linear - internal).abs() < 1e-3,
                        "{width}x{height}, strategy={strategy}, EPF={epf}, gab={gab}: pixel {i} channel {c}: {linear} vs {internal}"
                    );
                }
            }
            let path = artifacts.join(format!(
                "{width}x{height}-s{strategy}-epf{epf}-gab{gab}.jxl"
            ));
            std::fs::write(&path, encoded).unwrap();
            let decoded = std::process::Command::new(&djxl)
                .arg(&path)
                .args(["--disable_output", "--num_threads=1"])
                .output()
                .unwrap();
            std::fs::write(path.with_extension("djxl.log"), &decoded.stderr).unwrap();
            assert!(
                decoded.status.success(),
                "djxl: {}",
                String::from_utf8_lossy(&decoded.stderr)
            );
        }
    }
}
