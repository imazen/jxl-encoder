//! Default-disabled optional features retain the source-to-ICC contract.
#![cfg(all(
    feature = "__expert",
    feature = "jpeg-reencoding",
    feature = "chroma-subsampling"
))]
use jxl_encoder::{ChromaSubsampling, ColorEncoding, ImageMetadata, LossyConfig, PixelLayout};
#[path = "support/encoder_policies.rs"]
mod encoder_policies;
#[test]
fn default_disabled_features_convert_and_preserve_raw_icc_storage() {
    const {
        assert!(
            cfg!(feature = "std"),
            "jpeg-reencoding -> hdr-gainmap -> std"
        );
    }
    let pixels = [128_u8, 64, 32].repeat(32 * 17);
    let profile = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let metadata = ImageMetadata::new().with_icc_profile(profile);
    for (policy, strategy, mode) in encoder_policies::policies() {
        for sampling in [
            ChromaSubsampling::Sub420,
            ChromaSubsampling::Sub422,
            ChromaSubsampling::Sub440,
        ] {
            let config = LossyConfig::new(0.01)
                .with_effort(3)
                .with_threads(1)
                .with_strategy(strategy.clone())
                .with_mode(mode)
                .with_chroma_subsampling(sampling);
            let request = || {
                config
                    .encode_request(32, 17, PixelLayout::Rgb8)
                    .with_metadata(&metadata)
            };
            let raw = request().encode(&pixels).unwrap();
            let parsed = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&raw))
                .unwrap();
            assert_eq!(parsed.original_icc(), Some(profile));
            let mut selector = ColorEncoding::srgb();
            selector.want_icc = true;
            assert_eq!(
                request()
                    .with_color_encoding(selector.clone())
                    .encode(&pixels)
                    .unwrap(),
                raw
            );
            let mut gamma = None;
            for kind in ["simple", "gamma", "icc-gamma"] {
                let mut selected = request();
                selected = match kind {
                    "simple" => selected
                        .with_color_encoding(ColorEncoding::from_cicp(11, 8, 0, true).unwrap()),
                    "gamma" => selected.with_source_gamma(0.5),
                    _ => selected
                        .with_color_encoding(selector.clone())
                        .with_source_gamma(0.5),
                };
                let converted = selected.encode(&pixels).unwrap();
                assert_ne!(
                    converted, raw,
                    "{policy}/{sampling:?}/{kind}: conversion is consumed"
                );
                let parsed = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&converted))
                    .unwrap();
                assert_eq!(parsed.original_icc(), Some(profile));
                if kind == "gamma" {
                    gamma = Some(converted);
                } else if kind == "icc-gamma" {
                    assert_eq!(
                        Some(converted),
                        gamma,
                        "the ICC selector does not supply simple primaries"
                    );
                }
            }
        }
    }
}
