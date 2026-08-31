// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Resampling must never change the image's dimensions (T4, 2026-08-31).
//!
//! The encoder downsamples by `div_ceil(dim, factor)`, so reconstructing the
//! advertised size as `downsampled × factor` rounds **up** by as much as
//! `factor − 1` on any axis that is not a multiple of the factor. Before the
//! fix, a 1105-row input at factor 2 shipped a `SizeHeader` saying 1106 and
//! every decoder produced an image one row taller than the caller supplied.
//!
//! That reaches the DEFAULT path with no opt-in at all: `auto_resampling` is
//! on by default and selects factor 2 at `distance >= 10`, so any odd-height
//! or odd-width image encoded at `d >= 10` came back the wrong size.
//!
//! libjxl writes the true size into the `SizeHeader` and lets
//! `FrameDimensions::Set`'s `DivCeil(xsize_px, upsampling)` recover the coded
//! grid; the decoder crops the upsampled result back down. Verified against
//! `cjxl v0.12.0` on the same 1118×1105 input at `-d 12`: it writes
//! `ysize = 1105`, we wrote 1106.
//!
//! Why no earlier test caught it: every `resampling > 1` hash-lock cell uses a
//! 512×512 fixture, and 512 is divisible by 8, so the buggy product equals the
//! true size in every one of them. These cells deliberately use dimensions
//! that are **coprime-ish** to the factors — odd width AND odd height, plus
//! sizes that are not multiples of 4 or 8.

use jxl_encoder::{LossyConfig, PixelLayout};

/// Deterministic gradient with mild texture so the encode has real content at
/// every distance (a flat image can survive dimension bugs by accident).
fn gradient(w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            out[i] = (x * 255 / w.max(2).saturating_sub(1)) as u8;
            out[i + 1] = (y * 255 / h.max(2).saturating_sub(1)) as u8;
            out[i + 2] = ((x ^ y) & 0xFF) as u8;
        }
    }
    out
}

/// jxl-rs is the primary roundtrip decoder; returns the decoded frame size.
fn decode_dims_jxl_rs(data: &[u8]) -> (usize, usize) {
    use jxl::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult, states};

    let mut input = data;
    let mut decoder_init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_init = fallback,
            Err(e) => panic!("jxl-rs header decode error: {e:?}"),
        }
    };
    decoder.basic_info().size
}

/// jxl-oxide as the independent second opinion — and it renders, so a header
/// that merely *claims* the right size but cannot be rendered still fails.
fn decode_dims_jxl_oxide(data: &[u8]) -> (usize, usize) {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("jxl-oxide rejected the bitstream");
    let (w, h) = {
        let hdr = image.image_header();
        (hdr.size.width as usize, hdr.size.height as usize)
    };
    let render = image.render_frame(0).expect("jxl-oxide render failed");
    let fb = render.image_all_channels();
    assert_eq!(
        (fb.width(), fb.height()),
        (w, h),
        "jxl-oxide rendered {}x{} for a header claiming {}x{}",
        fb.width(),
        fb.height(),
        w,
        h
    );
    (w, h)
}

fn assert_dims_preserved(label: &str, bytes: &[u8], w: usize, h: usize) {
    let rs = decode_dims_jxl_rs(bytes);
    assert_eq!(
        rs,
        (w, h),
        "{label}: jxl-rs decoded {}x{}, input was {w}x{h}. Resampling must not \
         change the image size — the file header carries the ORIGINAL dims and \
         the decoder crops the upsampled result (libjxl `FrameDimensions::Set`, \
         `DivCeil(xsize_px, upsampling)`).",
        rs.0,
        rs.1
    );
    let ox = decode_dims_jxl_oxide(bytes);
    assert_eq!(
        ox,
        (w, h),
        "{label}: jxl-oxide decoded {}x{}, input was {w}x{h}",
        ox.0,
        ox.1
    );
}

/// Explicit `with_resampling(N)` on sizes that are not multiples of N.
#[test]
fn explicit_resampling_preserves_odd_dimensions() {
    // 259 and 133 are odd and not multiples of 4 or 8, so all three factors
    // exercise the `div_ceil` rounding on BOTH axes.
    for (w, h) in [(259usize, 133usize), (65, 65), (127, 255)] {
        let px = gradient(w, h);
        for factor in [2u32, 4, 8] {
            let bytes = LossyConfig::new(2.0)
                .with_effort(5)
                .with_resampling(factor)
                .encode(&px, w as u32, h as u32, PixelLayout::Rgb8)
                .unwrap_or_else(|e| panic!("{w}x{h} r{factor}: encode failed: {e:?}"));
            assert_dims_preserved(&format!("{w}x{h} with_resampling({factor})"), &bytes, w, h);
        }
    }
}

/// The default path: `auto_resampling` is on and selects factor 2 at
/// `distance >= 10`, with no caller opt-in. This is the arm that made the bug
/// reachable without anyone asking for resampling at all.
#[test]
fn auto_resampling_at_high_distance_preserves_odd_dimensions() {
    let (w, h) = (259usize, 133usize);
    let px = gradient(w, h);
    for d in [10.0f32, 12.0, 15.0] {
        let bytes = LossyConfig::new(d)
            .with_effort(5)
            .encode(&px, w as u32, h as u32, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("d={d}: encode failed: {e:?}"));
        assert_dims_preserved(&format!("{w}x{h} auto-resample at d={d}"), &bytes, w, h);
    }
}

/// Below the auto-resample threshold nothing resamples, so this is the control
/// arm: if it ever fails, the failure is not about resampling.
#[test]
fn no_resampling_below_threshold_preserves_odd_dimensions() {
    let (w, h) = (259usize, 133usize);
    let px = gradient(w, h);
    for d in [1.0f32, 4.0, 9.0] {
        let bytes = LossyConfig::new(d)
            .with_effort(5)
            .encode(&px, w as u32, h as u32, PixelLayout::Rgb8)
            .unwrap_or_else(|e| panic!("d={d}: encode failed: {e:?}"));
        assert_dims_preserved(&format!("{w}x{h} no-resample at d={d}"), &bytes, w, h);
    }
}

/// `with_already_downsampled` keeps the OTHER contract on purpose: the caller
/// hands over post-downsample pixels and asks for `dims × N` as the advertised
/// size. Pinned so the fix above cannot be "simplified" into overriding it.
#[test]
fn already_downsampled_still_advertises_dims_times_factor() {
    let (w, h) = (130usize, 67usize);
    let px = gradient(w, h);
    let bytes = LossyConfig::new(2.0)
        .with_effort(5)
        .with_resampling(2)
        .with_already_downsampled(true)
        .encode(&px, w as u32, h as u32, PixelLayout::Rgb8)
        .expect("already-downsampled encode failed");
    let rs = decode_dims_jxl_rs(&bytes);
    assert_eq!(
        rs,
        (w * 2, h * 2),
        "with_already_downsampled(true) must advertise dims x factor \
         ({}x{}), got {}x{}",
        w * 2,
        h * 2,
        rs.0,
        rs.1
    );
}
