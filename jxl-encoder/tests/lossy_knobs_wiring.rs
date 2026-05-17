//! Integration tests for W4-2 lossy skeleton flag wiring.
//!
//! Verifies that each of the four lossy knobs (`alpha_distance`,
//! `group_order`, `center_x` / `center_y`, `upsampling_mode`) is plumbed
//! from [`LossyConfig`] through to the encoder and changes encoded bytes
//! in the expected way.
//!
//! The point of these tests is wiring proof, NOT quality calibration —
//! they assert that the knob affects the bitstream at all, not that the
//! affected bitstream is optimal. Lossy alpha and full upsampling LUT
//! semantics are tested elsewhere (and the alpha path in particular is
//! intentionally still lossless at all `alpha_distance` values for this
//! chunk; see the doc on [`LossyConfig::with_alpha_distance`]).

use jxl_encoder::{LossyConfig, PixelLayout};

fn rgb8_buf(w: u32, h: u32) -> Vec<u8> {
    (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect()
}

fn rgba8_buf(w: u32, h: u32) -> Vec<u8> {
    (0..(w * h * 4) as usize).map(|i| (i % 256) as u8).collect()
}

#[test]
fn upsampling_mode_changes_bytes_factor2() {
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let default_lut = LossyConfig::new(1.0)
        .with_resampling(2)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    let nearest_lut = LossyConfig::new(1.0)
        .with_resampling(2)
        .with_upsampling_mode(Some(0))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();

    assert_ne!(
        default_lut, nearest_lut,
        "upsampling_mode=Some(0) (nearest) must change the file-header \
         CustomTransformData block relative to the all-default fast path"
    );
    // Nearest LUT is one extra `!all_default` bit + the per-factor
    // weights; expect a strictly larger bitstream.
    assert!(
        nearest_lut.len() >= default_lut.len(),
        "nearest LUT encoder output ({}) should be >= default LUT ({}) — LUT bytes are appended",
        nearest_lut.len(),
        default_lut.len()
    );
}

#[test]
fn upsampling_mode_changes_bytes_factor4_pixel_dots() {
    let w = 32u32;
    let h = 32u32;
    let buf = rgb8_buf(w, h);

    let nearest = LossyConfig::new(1.0)
        .with_resampling(4)
        .with_upsampling_mode(Some(0))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    let dots = LossyConfig::new(1.0)
        .with_resampling(4)
        .with_upsampling_mode(Some(1))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();

    assert_ne!(
        nearest, dots,
        "upsampling_mode=Some(1) (pixel dots) at factor 4 has different \
         LUT slot values than mode=Some(0); encoded bytes must differ"
    );
}

#[test]
fn center_x_center_y_change_bytes_on_multigroup() {
    // 512x512 → 2x2 group grid → permutation is observable.
    let w = 512u32;
    let h = 512u32;
    let buf = rgb8_buf(w, h);

    let centered = LossyConfig::new(1.0)
        .with_group_order(Some(1))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    let off_centre = LossyConfig::new(1.0)
        .with_group_order(Some(1))
        .with_center_x(Some(0))
        .with_center_y(Some(0))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();

    assert_ne!(
        centered, off_centre,
        "shifting the AC group permutation centre to (0, 0) on a 2x2 \
         group grid must change the on-disk TOC ordering, which changes \
         encoded bytes"
    );
}

#[test]
fn group_order_one_implies_center_first() {
    // Same source, encode with `with_group_order(Some(1))` vs the
    // explicit `with_center_first(true)` setter. These should be
    // wire-equivalent (group_order=1 just flips center_first).
    let w = 512u32;
    let h = 512u32;
    let buf = rgb8_buf(w, h);

    let via_group_order = LossyConfig::new(1.0)
        .with_group_order(Some(1))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    let via_center_first = LossyConfig::new(1.0)
        .with_center_first(true)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    assert_eq!(
        via_group_order, via_center_first,
        "with_group_order(Some(1)) must produce the same bytes as \
         with_center_first(true) — they wire the same flag"
    );
}

#[test]
fn group_order_zero_disables_center_first() {
    // group_order=Some(0) (explicit scanline) cancels a previously-set
    // center_first. Verifies the with_group_order side-effect path.
    let w = 512u32;
    let h = 512u32;
    let buf = rgb8_buf(w, h);

    let scanline_default = LossyConfig::new(1.0)
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    let scanline_explicit = LossyConfig::new(1.0)
        .with_center_first(true)
        .with_group_order(Some(0))
        .encode(&buf, w, h, PixelLayout::Rgb8)
        .unwrap();
    assert_eq!(
        scanline_default, scanline_explicit,
        "with_group_order(Some(0)) after with_center_first(true) must \
         cancel center-first and produce default scanline bytes"
    );
}

#[test]
fn alpha_distance_lossless_path_byte_identical_today() {
    // alpha_distance is stored on the config and threaded to
    // VarDctEncoder, but the alpha extras subimage is still emitted
    // losslessly (gradient predictor + LZ77 RLE). Until the lossy
    // alpha pipeline lands, alpha_distance must be a no-op on bytes.
    //
    // This test guards the documented contract on
    // LossyConfig::with_alpha_distance so a future change to the
    // alpha path can flip this test deliberately rather than silently.
    let w = 32u32;
    let h = 32u32;
    let buf = rgba8_buf(w, h);

    let unset = LossyConfig::new(1.0)
        .encode(&buf, w, h, PixelLayout::Rgba8)
        .unwrap();
    let zero = LossyConfig::new(1.0)
        .with_alpha_distance(Some(0.0))
        .encode(&buf, w, h, PixelLayout::Rgba8)
        .unwrap();
    let nonzero = LossyConfig::new(1.0)
        .with_alpha_distance(Some(2.0))
        .encode(&buf, w, h, PixelLayout::Rgba8)
        .unwrap();

    assert_eq!(
        unset, zero,
        "alpha_distance=None and alpha_distance=Some(0.0) must produce \
         identical bytes (both mean lossless alpha)"
    );
    assert_eq!(
        unset, nonzero,
        "alpha_distance=Some(2.0) is recorded on the encoder but the \
         lossy alpha pipeline is not yet wired (see encoder docs); \
         bytes must be identical to the lossless baseline today. \
         When the lossy alpha path lands, flip this assertion to assert_ne!."
    );
}
