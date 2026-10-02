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
        let img = image::open(&path)?.to_rgb8();
        let (w, h) = img.dimensions();
        let stem = std::path::Path::new(&path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        for &(name, pred, ans, lz77) in PROFILES {
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
    }
    Ok(())
}
