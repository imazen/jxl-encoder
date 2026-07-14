// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Pin the e7 line-art effort-monotonicity regression on aliased line-art (#74).
//!
//! Aliased-polygon plots (imazen-26 7000-plots/line-*) lose +19..+52 % bytes
//! to cjxl at e7 d0.5 while our OWN e6 beats cjxl (-0.7 %) — an effort-
//! monotonicity violation. CLI A/B ruled out patches/tree_learning/try_dct64
//! and transform-size. This probe toggles each remaining e7-gated knob
//! individually via `LossyInternalParams` and reads the EXACT content proxies
//! the W44 gates dispatch on (`ZenanalyzeProxies`), producing a calibration
//! dataset: (proxies) -> (byte delta from disabling each knob).
//!
//! Result (benchmarks/cfl_two_pass_lineart_ablation_2026-07-14.tsv):
//! `cfl_two_pass` costs 8.8..16.5 % on aliased line-art and <=3.8 % (often
//! ~0) elsewhere — a Pareto win to disable on line-art, ~neutral otherwise.
//!
//! usage:
//!   line_art_cfl_probe <img.png> [img2.png ...]         # human table (d0.5 e7)
//!   line_art_cfl_probe --tsv out.tsv <img.png> ...      # + machine-readable TSV
//!   JXL_PROBE_DUMP_DIR=dir line_art_cfl_probe <img> ... # + dump base/nocfl2 .jxl

use jxl_encoder::__internals::ZenanalyzeProxies;
use jxl_encoder::{LossyConfig, LossyInternalParams, PixelLayout};

fn enc(rgb: &[u8], w: u32, h: u32, tweak: impl FnOnce(&mut LossyInternalParams)) -> usize {
    let mut p = LossyInternalParams::default();
    tweak(&mut p);
    LossyConfig::new(0.5)
        .with_effort(7)
        .with_internal_params(p)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .expect("encode")
        .len()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut tsv_path: Option<String> = None;
    if let Some(i) = args.iter().position(|a| a == "--tsv") {
        tsv_path = Some(args.remove(i + 1));
        args.remove(i);
    }
    let paths = args;
    assert!(
        !paths.is_empty(),
        "usage: line_art_cfl_probe [--tsv out] <img.png> ..."
    );

    let mut tsv = String::from(
        "image\twidth\theight\tm3_colourfulness\tflat_color_block_ratio\tedge_density\t\
         luma_var\tbase_bytes\tno_cfl2_pct\tno_chroma_pct\tall_off_pct\n",
    );
    println!(
        "{:<38} {:>7} {:>6} {:>6} {:>7} {:>8} {:>10} {:>9} {:>9}",
        "image", "m3", "fcbr", "edge", "lumavar", "base", "no_cfl2", "no_chroma", "all_off"
    );
    for path in &paths {
        let img = image::open(path).expect("open").to_rgb8();
        let (w, h) = (img.width(), img.height());
        let rgb = img.into_raw();

        // Exact gate proxies (bpp=3, r/g/b at 0/1/2 for Rgb8).
        let px = ZenanalyzeProxies::compute_srgb_u8(&rgb, w as usize, h as usize, 3, 0, 1, 2);

        let base = enc(&rgb, w, h, |_| {});
        let no_cfl2 = enc(&rgb, w, h, |p| p.cfl_two_pass = Some(false));
        let no_chroma = enc(&rgb, w, h, |p| p.chromacity_adjustment = Some(false));
        let all_off = enc(&rgb, w, h, |p| {
            p.cfl_two_pass = Some(false);
            p.chromacity_adjustment = Some(false);
        });

        // Dump base + no_cfl2 bitstreams for external decode+quality scoring
        // (set JXL_PROBE_DUMP_DIR). Confirms the byte win is a clean Pareto
        // win, not a quality tradeoff.
        if let Some(dir) = std::env::var_os("JXL_PROBE_DUMP_DIR") {
            let dir = std::path::PathBuf::from(dir);
            let name = std::path::Path::new(path)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap();
            let write = |suffix: &str, tweak: &mut dyn FnMut(&mut LossyInternalParams)| {
                let mut p = LossyInternalParams::default();
                tweak(&mut p);
                let bytes = LossyConfig::new(0.5)
                    .with_effort(7)
                    .with_internal_params(p)
                    .encode(&rgb, w, h, PixelLayout::Rgb8)
                    .expect("encode");
                std::fs::write(dir.join(format!("{name}.{suffix}.jxl")), bytes).unwrap();
            };
            write("base", &mut |_| {});
            write("nocfl2", &mut |p| p.cfl_two_pass = Some(false));
        }

        let pct = |b: usize| 100.0 * (b as f64 - base as f64) / base as f64;
        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        let short = &name[..name.len().min(38)];
        println!(
            "{:<38} {:>7.1} {:>6.3} {:>6.3} {:>7.0} {:>8} {:>+8.1}% {:>+8.1}% {:>+8.1}%",
            short,
            px.m3_colourfulness,
            px.flat_color_block_ratio,
            px.edge_density,
            px.luma_var,
            base,
            pct(no_cfl2),
            pct(no_chroma),
            pct(all_off),
        );
        tsv.push_str(&format!(
            "{name}\t{w}\t{h}\t{:.3}\t{:.4}\t{:.4}\t{:.1}\t{base}\t{:.2}\t{:.2}\t{:.2}\n",
            px.m3_colourfulness,
            px.flat_color_block_ratio,
            px.edge_density,
            px.luma_var,
            pct(no_cfl2),
            pct(no_chroma),
            pct(all_off),
        ));
    }
    if let Some(p) = tsv_path {
        std::fs::write(&p, tsv).expect("write tsv");
        eprintln!("wrote {p}");
    }
}
