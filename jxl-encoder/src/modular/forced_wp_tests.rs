//! Observe actual WP header bits; no encoding decisions are changed by this hook.

use crate::api::{LosslessConfig, PixelLayout, SectionedTrees};
use crate::bit_writer::BitWriter;
use crate::effort::LosslessInternalParams;
use crate::validation::ValidationError;
use std::cell::RefCell;
use std::path::PathBuf;

std::thread_local! {
    static HEADERS: RefCell<Option<Vec<(usize, u64)>>> = const { RefCell::new(None) };
}

pub(super) fn record_header(writer: &BitWriter, start: usize) {
    HEADERS.with_borrow_mut(|headers| {
        if let Some(headers) = headers {
            let end = writer.bits_written();
            let bytes = writer.peek_bytes();
            let bits = (start..end).enumerate().fold(0u64, |value, (shift, bit)| {
                value | (((bytes[bit / 8] >> (bit % 8)) & 1) as u64) << shift
            });
            headers.push((end - start, bits));
        }
    });
}

#[test]
fn invalid_wp_modes_fail_before_encoding() {
    for value in [5, 255] {
        let params = LosslessInternalParams {
            forced_wp_mode: Some(value),
            ..Default::default()
        };
        assert!(matches!(params.validate(),
            Err(ValidationError::ForcedWpModeOutOfRange { value: got }) if got == value));
        let cfg = LosslessConfig::new().with_internal_params(params);
        assert!(matches!(cfg.validate(),
            Err(ValidationError::ForcedWpModeOutOfRange { value: got }) if got == value));
        let err = cfg.encode(&[0; 3], 1, 1, PixelLayout::Rgb8).unwrap_err();
        assert!(err.to_string().contains("forced_wp_mode"), "{err}");
        let mut streaming = cfg.encoder(1, 1, PixelLayout::Rgb8).unwrap();
        let err = streaming.push_rows(&[0; 3], 1).unwrap_err();
        assert!(err.to_string().contains("forced_wp_mode"), "{err}");
    }
}

#[test]
fn every_mode_reaches_wire_and_both_decoders() {
    // With parallel enabled, keep both the observer and encode on the same
    // ambient worker. Other tests have separate thread-local observers.
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(check_all_modes);
    #[cfg(not(feature = "parallel"))]
    check_all_modes();
}

fn check_all_modes() {
    // Frozen LSB-first WP headers: default flag, seven 5-bit parameters,
    // four 4-bit weights. These distinguish all five parameter sets on wire.
    const EXPECTED: [(usize, u64); 5] = [
        (1, 1),
        (52, 0xbccd15c602210),
        (52, 0xcdcd4c0003a54),
        (52, 0xccdd05c100220),
        (52, 0xcccd230a52a94),
    ];
    let dir = PathBuf::from(
        std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .unwrap(),
    )
    .join("tmp")
    .join(format!("forced-wp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = image::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/images/frymire-srgb.png"
    ))
    .unwrap();
    for (width, height) in [(64, 64), (259, 133)] {
        let rgb = source.crop_imm(0, 0, width, height).to_rgb8();
        for squeeze in [false, true] {
            for sectioned in [
                SectionedTrees::Off,
                SectionedTrees::On,
                SectionedTrees::Hybrid,
            ] {
                for mode in 0..=4 {
                    let params = LosslessInternalParams {
                        forced_wp_mode: Some(mode),
                        wp_num_param_sets: Some(5),
                        ..Default::default()
                    };
                    let cfg = LosslessConfig::new()
                        .with_effort(7)
                        .with_internal_params(params)
                        .with_tree_learning(true)
                        .with_squeeze(squeeze)
                        .with_sectioned_trees(sectioned)
                        .with_modular_palette_colors(Some(0))
                        .with_modular_channel_colors_global_percent(Some(0.0))
                        .with_modular_channel_colors_group_percent(Some(0.0))
                        .with_patches(false)
                        .with_lz77(false)
                        .with_threads(0);
                    let label = format!("{width}x{height}-{squeeze}-{sectioned:?}-wp{mode}");
                    HEADERS.with_borrow_mut(|headers| *headers = Some(Vec::new()));
                    let encoded = cfg
                        .encode(rgb.as_raw(), width, height, PixelLayout::Rgb8)
                        .unwrap();
                    let headers = HEADERS.with_borrow_mut(|headers| headers.take().unwrap());
                    assert!(!headers.is_empty(), "no WP header observed: {label}");
                    assert!(
                        headers.iter().all(|h| *h == EXPECTED[mode as usize]),
                        "{label}: unexpected WP headers {headers:x?}"
                    );
                    let jxl = dir.join(format!("{label}.jxl"));
                    std::fs::write(&jxl, &encoded).unwrap();
                    let decoded = zenjxl_decoder::decode(&encoded).unwrap();
                    assert_eq!(
                        (decoded.width, decoded.height, decoded.channels),
                        (width as usize, height as usize, 4),
                        "{label}"
                    );
                    let pixels: Vec<_> = decoded
                        .data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .flat_map(|p| [p[0], p[1], p[2]])
                        .collect();
                    assert_eq!(pixels, *rgb.as_raw(), "{label}");
                    let png = dir.join(format!("{label}.png"));
                    let output = std::process::Command::new(crate::test_helpers::djxl_path())
                        .arg(&jxl)
                        .arg(&png)
                        .output()
                        .unwrap();
                    assert!(
                        output.status.success(),
                        "{label}: {}",
                        String::from_utf8_lossy(&output.stderr)
                    );
                    assert_eq!(image::open(png).unwrap().to_rgb8(), rgb, "{label}");
                }
            }
        }
    }
}
