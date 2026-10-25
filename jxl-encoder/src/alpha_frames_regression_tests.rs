// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! 0.3.2 regression gate for alpha-bearing shapes that produced files no
//! decoder accepts (each was rejected by jxl-rs AND djxl v0.12):
//!
//! - lossless GrayAlpha: the header declared no extra channel;
//! - lossy RGBA + progressive on a single-group image: alpha left LfGlobal;
//! - lossy RGBA + `with_lf_frame`: the LF frame carried no extra channel;
//! - patches + alpha: the reference frame carried no extra channel and patch
//!   positions lacked the per-extra-channel blend modes.
//!
//! Every case decodes through jxl-rs (alpha checked sample-exact; it is
//! coded losslessly in all of these) and fully decodes in djxl (`DJXL_PATH`, libjxl v0.12).

use crate::api::{LosslessConfig, LossyConfig, PixelLayout, ProgressiveMode};

/// Frymire (real graphics content) cropped to `w`x`h` with a deterministic
/// alpha ramp, as RGBA8.
fn frymire_rgba(w: u32, h: u32) -> Vec<u8> {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus");
    let dir = corpus
        .get("imageflow/test_inputs")
        .expect("imageflow corpus");
    let img = image::open(dir.join("frymire.png"))
        .expect("frymire.png")
        .to_rgba8();
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x % img.width(), y % img.height());
            out.extend([p[0], p[1], p[2], ((x * 3 + y * 5) % 256) as u8]);
        }
    }
    out
}

fn assert_alpha_roundtrip(jxl: &[u8], alpha: impl Iterator<Item = u8>, w: u32, h: u32, what: &str) {
    // Reference decoder (set DJXL_PATH to a libjxl v0.12 djxl). The
    // crates.io jxl-oxide 0.12 used by this branch's tests mis-reads some
    // multi-group modular frames that jxl-rs and djxl both accept.
    let r = crate::test_helpers::decode_with_djxl(jxl)
        .unwrap_or_else(|e| panic!("{what}: djxl: {e:?}"));
    assert_eq!(
        (r.width, r.height),
        (w as usize, h as usize),
        "{what}: djxl dims"
    );
    let d = crate::test_helpers::decode_with_jxl_rs(jxl)
        .unwrap_or_else(|e| panic!("{what}: jxl-rs: {e:?}"));
    assert_eq!((d.width, d.height), (w as usize, h as usize), "{what}");
    for (i, want) in alpha.enumerate() {
        let got = (d.pixels[i * d.channels + d.channels - 1] * 255.0).round();
        assert_eq!(got, want as f32, "{what}: jxl-rs alpha sample {i}");
    }
}

#[test]
fn lossless_gray_alpha_decodes() {
    for (w, h) in [(64u32, 32u32), (300, 260)] {
        let rgba = frymire_rgba(w, h);
        let ga: Vec<u8> = rgba.chunks(4).flat_map(|p| [p[1], p[3]]).collect();
        for e in [3u8, 7] {
            let jxl = LosslessConfig::new()
                .with_effort(e)
                .encode(&ga, w, h, PixelLayout::GrayAlpha8)
                .unwrap();
            let what = format!("lossless GrayAlpha8 {w}x{h} e{e}");
            assert_alpha_roundtrip(&jxl, ga.chunks(2).map(|p| p[1]), w, h, &what);
            let d = crate::test_helpers::decode_with_jxl_rs(&jxl).unwrap();
            for (i, p) in ga.chunks(2).enumerate() {
                let got = (d.pixels[i * d.channels] * 255.0).round();
                assert_eq!(got, p[0] as f32, "{what}: gray sample {i}");
            }
        }
    }
}

#[test]
fn lossy_rgba_progressive_and_lf_frame_decode() {
    for (w, h) in [(64u32, 32u32), (259, 133), (300, 260)] {
        let rgba = frymire_rgba(w, h);
        let alpha = || rgba.chunks(4).map(|p| p[3]);
        for (name, cfg) in [
            (
                "prog2",
                LossyConfig::new(2.0)
                    .with_effort(5)
                    .with_progressive(ProgressiveMode::QuantizedAcFullAc),
            ),
            (
                "prog3",
                LossyConfig::new(2.0)
                    .with_effort(5)
                    .with_progressive(ProgressiveMode::DcVlfLfAc),
            ),
            (
                "lf_frame",
                LossyConfig::new(2.0).with_effort(7).with_lf_frame(true),
            ),
        ] {
            let jxl = cfg.encode(&rgba, w, h, PixelLayout::Rgba8).unwrap();
            assert_alpha_roundtrip(&jxl, alpha(), w, h, &format!("lossy RGBA {w}x{h} {name}"));
        }
    }
}

/// The gb82 terminal screenshot reliably triggers patches.
#[test]
fn patches_with_alpha_decode() {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus");
    let dir = corpus.get("gb82-sc").expect("gb82-sc corpus");
    let img = image::open(dir.join("terminal.png"))
        .expect("terminal.png")
        .to_rgba8();
    let (w, h) = (img.width().min(600), img.height().min(400));
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y);
            rgba.extend([
                p[0],
                p[1],
                p[2],
                if (x / 16 + y / 16) % 3 == 0 { 128 } else { 255 },
            ]);
        }
    }
    let alpha = || rgba.chunks(4).map(|p| p[3]);
    for e in [5u8, 7] {
        let jxl = LosslessConfig::new()
            .with_effort(e)
            .with_patches(true)
            .encode(&rgba, w, h, PixelLayout::Rgba8)
            .unwrap();
        assert_alpha_roundtrip(&jxl, alpha(), w, h, &format!("lossless patches RGBA e{e}"));
    }
    for e in [7u8, 9] {
        let jxl = LossyConfig::new(1.0)
            .with_effort(e)
            .with_patches(true)
            .encode(&rgba, w, h, PixelLayout::Rgba8)
            .unwrap();
        assert_alpha_roundtrip(&jxl, alpha(), w, h, &format!("lossy patches RGBA e{e}"));
    }
}

/// Lossy effort 1 and 2 default to the single-pass static-code path, which
/// cannot write alpha; it used to drop alpha from the file. Frames it cannot
/// represent now take the two-pass path (0.3.4, backport of the main fix).
#[test]
fn lossy_low_effort_keeps_alpha() {
    for (w, h) in [(64u32, 32u32), (300, 260)] {
        let px = frymire_rgba(w, h);
        for effort in [1u8, 2] {
            let jxl = LossyConfig::new(1.0)
                .with_effort(effort)
                .encode(&px, w, h, PixelLayout::Rgba8)
                .expect("low-effort lossy RGBA encode");
            assert_alpha_roundtrip(
                &jxl,
                px.chunks(4).map(|p| p[3]),
                w,
                h,
                &format!("lossy RGBA {w}x{h} e{effort}"),
            );
        }
    }
}
