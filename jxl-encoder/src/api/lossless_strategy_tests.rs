use super::*;

#[test]
fn lossless_strategy_controls_zen_policies() {
    for effort in [5, 8, 9, 10] {
        for pixels in [3_999_999, 4_000_000] {
            for strategy in [
                EncoderStrategy::Zenjxl,
                EncoderStrategy::LeanFaster,
                EncoderStrategy::Aggressive,
                EncoderStrategy::Libjxl,
            ] {
                let zen = !matches!(strategy, EncoderStrategy::Libjxl);
                let p = LosslessConfig::new()
                    .with_effort(effort)
                    .with_strategy(strategy)
                    .effective_profile_for_image(pixels);
                assert_eq!(p.tree_self_repair, zen);
                assert_eq!(p.tree_self_repair_allowed, zen);
                assert_eq!(p.lossless_large_tree_bucket_reduction, zen);
                let base = crate::effort::EffortProfile::lossless(effort, EncoderMode::default());
                let expected = if zen && effort >= 9 && pixels >= 4_000_000 {
                    192
                } else {
                    base.tree_max_buckets
                };
                assert_eq!(p.tree_max_buckets, expected);
            }
        }
    }
    for repair in [false, true] {
        for buckets in [false, true] {
            let custom = EncoderImprovementsCustom {
                lossless_tree_self_repair: repair,
                lossless_large_tree_bucket_reduction: buckets,
                ..Default::default()
            };
            let cfg = LosslessConfig::new()
                .with_effort(9)
                .with_strategy(EncoderStrategy::Custom(Box::new(custom)));
            let p = cfg.effective_profile_for_image(4_000_000);
            assert_eq!(p.tree_self_repair_allowed, repair);
            assert_eq!(p.tree_max_buckets, if buckets { 192 } else { 256 });
            #[cfg(feature = "__expert")]
            {
                let p = cfg
                    .with_internal_params(crate::effort::LosslessInternalParams {
                        tree_max_buckets: Some(224),
                        ..Default::default()
                    })
                    .effective_profile_for_image(4_000_000);
                assert_eq!(p.tree_max_buckets, 224);
            }
        }
    }
}

#[cfg(all(feature = "__expert", not(target_arch = "wasm32")))]
#[test]
fn lossless_strategy_blocks_legacy_env_override() {
    // Process-local OnceLock: set the environment before the child starts,
    // without mutating environment shared with other libtest threads.
    if std::env::var_os("JXL_LOSSLESS_STRATEGY_TEST_CHILD").is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "api::lossless_strategy_tests::lossless_strategy_blocks_legacy_env_override",
            ])
            .env("JXL_LOSSLESS_STRATEGY_TEST_CHILD", "1")
            .env("JXL_TREE_SELF_REPAIR", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("test result: ok. 1 passed"),
            "child must execute exactly one test: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        return;
    }
    for (strategy, expected) in [
        (EncoderStrategy::Zenjxl, true),
        (EncoderStrategy::Libjxl, false),
    ] {
        let p = LosslessConfig::new()
            .with_strategy(strategy)
            .effective_profile();
        assert_eq!(
            crate::modular::encode::tree_self_repair_should_try(
                p.tree_self_repair,
                p.tree_self_repair_allowed,
                8,
                256
            ),
            expected
        );
    }
}

#[cfg(all(feature = "__expert", not(target_arch = "wasm32")))]
#[test]
fn lossless_strategies_roundtrip_and_stream_identically() {
    let source = image::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/images/frymire-srgb.png"
    ))
    .unwrap();
    let dir = std::path::PathBuf::from(
        std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .unwrap(),
    )
    .join("tmp")
    .join(format!("lossless-strategies-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (width, height) in [(64, 64), (259, 133)] {
        let rgb = source.crop_imm(0, 0, width, height).to_rgb8();
        for effort in [5, 9] {
            for strategy in [
                EncoderStrategy::Zenjxl,
                EncoderStrategy::Libjxl,
                EncoderStrategy::LeanFaster,
                EncoderStrategy::Aggressive,
            ] {
                let label = format!("{width}x{height}-e{effort}-{strategy:?}");
                let cfg = LosslessConfig::new()
                    .with_effort(effort)
                    .with_threads(1)
                    .with_strategy(strategy);
                let bytes = cfg
                    .encode(rgb.as_raw(), width, height, PixelLayout::Rgb8)
                    .unwrap();
                let mut streaming = cfg.encoder(width, height, PixelLayout::Rgb8).unwrap();
                for rows in rgb.as_raw().chunks(7 * width as usize * 3) {
                    streaming
                        .push_rows(rows, (rows.len() / (width as usize * 3)) as u32)
                        .unwrap();
                }
                assert_eq!(streaming.finish().unwrap(), bytes, "{label}: streaming");
                let decoded = zenjxl_decoder::decode(&bytes).unwrap();
                assert_eq!(
                    (decoded.width, decoded.height, decoded.channels),
                    (width as usize, height as usize, 4)
                );
                let pixels: Vec<_> = decoded
                    .data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2]])
                    .collect();
                assert_eq!(pixels, *rgb.as_raw(), "{label}: Rust decoder");
                let jxl = dir.join(format!("{label}.jxl"));
                let png = dir.join(format!("{label}.png"));
                std::fs::write(&jxl, bytes).unwrap();
                let out = std::process::Command::new(crate::test_helpers::djxl_path())
                    .arg(jxl)
                    .arg(&png)
                    .output()
                    .unwrap();
                assert!(
                    out.status.success(),
                    "{label}: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
                assert_eq!(image::open(png).unwrap().to_rgb8(), rgb, "{label}: djxl");
            }
        }
    }
}
