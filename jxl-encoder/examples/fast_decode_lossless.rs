// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Emit lossless files shaped for decoder fast paths (zenjxl-decoder
//! fast-decode exploration, 2026-10-01).
//!
//! Every profile is a spec-conformant bitstream; the profiles only restrict
//! the encoder's choices so that, after static-property pruning, each channel
//! is a single MA leaf with a fixed predictor. Decoders that specialise that
//! shape (zenjxl-decoder `SingleGradientOnly`, and the experimental top /
//! prefix paths) skip the tree walk entirely.
//!
//! Usage: `fast_decode_lossless <out_dir> <in.png>...`
//! Writes `<out_dir>/<stem>__<profile>.jxl` and prints `profile\tstem\tbytes`.
//! Also writes `<stem>__preset_e<N>.jxl` from `LosslessConfig::with_fast_decode`
//! at efforts 1, 3, 4, 7 and 9, in the PNG's own layout (gray, alpha and
//! 16-bit included). The fixed-predictor profiles are written for 8-bit RGB
//! inputs only.

use jxl_encoder::{LosslessConfig, PixelLayout};

/// (name, predictor, use_ans, lz77)
const PROFILES: &[(&str, u8, bool, bool)] = &[
    ("grad_ans", 5, true, false),
    ("grad_prefix", 5, false, false),
    ("top_ans", 2, true, false),
    ("top_prefix", 2, false, false),
    ("zero_prefix", 0, false, false),
];

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let out_dir = std::path::PathBuf::from(args.next().expect("out_dir"));
    std::fs::create_dir_all(&out_dir)?;
    for path in args {
        let dynimg = image::open(&path)?;
        let (w, h) = (dynimg.width(), dynimg.height());
        let (layout, native) = native_layout(&dynimg);
        let img = dynimg.to_rgb8();
        let stem = std::path::Path::new(&path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let profiles = if matches!(dynimg, image::DynamicImage::ImageRgb8(_)) {
            PROFILES
        } else {
            &[]
        };
        for &(name, pred, ans, lz77) in profiles {
            let cfg = LosslessConfig::new()
                .with_effort(4)
                .with_tree_learning(false)
                .with_modular_predictor(Some(pred))
                .with_ans(ans)
                .with_lz77(lz77)
                .with_patches(false)
                .with_squeeze(false)
                .with_modular_palette_colors(Some(0));
            let bytes = cfg
                .encode(img.as_raw(), w, h, PixelLayout::Rgb8)
                .map_err(|e| anyhow::anyhow!("{name} {path}: {e:?}"))?;
            std::fs::write(out_dir.join(format!("{stem}__{name}.jxl")), &bytes)?;
            println!("{name}\t{stem}\t{}", bytes.len());
        }
        for effort in [1u8, 3, 4, 7, 9] {
            let name = format!("preset_e{effort}");
            let bytes = LosslessConfig::new()
                .with_effort(effort)
                .with_fast_decode()
                .encode(&native, w, h, layout)
                .map_err(|e| anyhow::anyhow!("{name} {path}: {e:?}"))?;
            std::fs::write(out_dir.join(format!("{stem}__{name}.jxl")), &bytes)?;
            println!("{name}\t{stem}\t{}", bytes.len());
        }
    }
    Ok(())
}

/// The PNG's samples in their own layout, as native-endian bytes.
fn native_layout(img: &image::DynamicImage) -> (PixelLayout, Vec<u8>) {
    use image::DynamicImage as D;
    let wide = |v: &[u16]| v.iter().flat_map(|s| s.to_ne_bytes()).collect::<Vec<u8>>();
    match img {
        D::ImageLuma8(b) => (PixelLayout::Gray8, b.as_raw().clone()),
        D::ImageLumaA8(b) => (PixelLayout::GrayAlpha8, b.as_raw().clone()),
        D::ImageRgb8(b) => (PixelLayout::Rgb8, b.as_raw().clone()),
        D::ImageRgba8(b) => (PixelLayout::Rgba8, b.as_raw().clone()),
        D::ImageLuma16(b) => (PixelLayout::Gray16, wide(b.as_raw())),
        D::ImageLumaA16(b) => (PixelLayout::GrayAlpha16, wide(b.as_raw())),
        D::ImageRgb16(b) => (PixelLayout::Rgb16, wide(b.as_raw())),
        other => (PixelLayout::Rgba16, wide(other.to_rgba16().as_raw())),
    }
}
