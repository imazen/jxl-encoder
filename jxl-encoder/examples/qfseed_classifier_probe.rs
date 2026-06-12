// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! GOAL_BEAT_CJXL wedge 2 probe (issue #74): does requiring the W44-164
//! classifier's Screenshot verdict (fcbr >= 0.35) on top of the W44-105
//! mask-median `is_screenshot` keep the gb82-sc qf-seed winners firing
//! while rejecting the ai-products class (smooth low-colour product
//! shots that trip the W44-108 m3<30 sub-gate at d>=2)?
//!
//! Run:
//!   cargo run --release -p jxl-encoder --example qfseed_classifier_probe \
//!     --features __expert -- <image.png> [more.png ...]
//!
//! With no args, runs the standing protection + negative set.

use jxl_encoder::__pre_quantized::ZenanalyzeProxies;
use std::path::{Path, PathBuf};

fn load_rgb8(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn main() {
    let home = std::env::var("HOME").unwrap();
    let corpus = std::env::var("CORPUS_ROOT").unwrap_or(format!("{home}/work/codec-corpus"));
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cases: Vec<(String, PathBuf)> = if args.is_empty() {
        let g = |n: &str| {
            (
                format!("PROTECT {n}"),
                Path::new(&corpus).join(format!("gb82-sc/{n}.png")),
            )
        };
        let im = |label: &str, rel: &str| (label.to_string(), Path::new(&corpus).join(rel));
        vec![
            g("graph"),
            g("imac_g3"),
            g("imac_dark"),
            g("gmessages"),
            g("gui"),
            g("terminal"),
            g("windows"),
            g("windows95"),
            g("codec_wiki"),
            g("imessage"),
            im(
                "NEG ai-products-9291",
                "imazen-26/9000-lilith-ai-products/9291_gen_products-beauty_bryn-birch-beard-oil-back_ingredients_p0062_1024x1536.png",
            ),
            im(
                "NEG ai-products-9307",
                "imazen-26/9000-lilith-ai-products/9307_gen_products-beauty_florwell-rose-oat-bar-soap_p0078_1024x1024.png",
            ),
            im(
                "NEG ai-products-9678",
                "imazen-26/9000-lilith-ai-products/9678_gen_products-grocery_cereal-box-coral_p0439_1024x1536.png",
            ),
        ]
    } else {
        args.into_iter()
            .map(|a| (a.clone(), PathBuf::from(a)))
            .collect()
    };
    println!(
        "{:<26} {:>9} {:>8} {:>9} {:>10} | fcbr>=0.35(=W44-164 Screenshot)",
        "case", "m3", "fcbr", "edge_d", "luma_var"
    );
    for (label, path) in cases {
        let Some((rgb, w, h)) = load_rgb8(&path) else {
            eprintln!("MISS: {}", path.display());
            continue;
        };
        let p = ZenanalyzeProxies::compute_srgb_u8(&rgb, w as usize, h as usize, 3, 0, 1, 2);
        println!(
            "{:<26} {:>9.2} {:>8.4} {:>9.4} {:>10.1} | {}",
            label,
            p.m3_colourfulness,
            p.flat_color_block_ratio,
            p.edge_density,
            p.luma_var,
            if p.flat_color_block_ratio >= 0.35 {
                "SCREENSHOT"
            } else {
                "not-screenshot"
            }
        );
    }
}
