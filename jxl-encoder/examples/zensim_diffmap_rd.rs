// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Diffmap RD harness (#38, 2026-05-27): encode a small multi-content corpus
//! with a chosen perceptual-loop metric/diffmap, save the decoded output, and
//! emit a manifest TSV. An EXTERNAL independent panel (`zen-metrics compare`:
//! butteraugli + dssim + ssim2) then scores the decoded outputs vs the clean
//! reference — the CIRCULARITY guard: a zensim/cvvdp-driven encode is NEVER
//! judged by the metric that drove it.
//!
//! Compares the three diffmap avenues for "which diffmap drives jxl-encoder
//! best" (user 2026-05-27):
//!   1. `--metric zensim` (+ ZENSIM_* env DiffmapOptions sweep)
//!   2. `--metric cvvdp`  (cvvdp's own diffmap)
//!   3. `--metric butteraugli` (the default baseline reference)
//!  (Avenue 2 — the #33 structural diffmap — is a separate code path added to
//!   the zensim loop; run it via `--metric zensim` once that env knob exists.)
//!
//! ## Run
//!
//! ```bash
//! cargo run --release -p jxl-encoder \
//!   --features "__expert butteraugli-loop zensim-loop cvvdp-loop-cpu ssim2-loop parallel" \
//!   --example zensim_diffmap_rd -- \
//!   --metric zensim --label zensim_v47_default \
//!   --out-dir /mnt/v/output/zensim/diffmap-rd-2026-05-27
//! ```
//!
//! Then score (independent panel) + build RD with the companion script.

#![cfg(feature = "zensim-loop")]

use std::fs;
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

use jxl_encoder::api::{EncoderStrategy, PerceptualMetric};
use jxl_encoder::{LossyConfig, PixelLayout};

const CID22_VAL_DIR: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82_SC_DIR: &str = "/home/lilith/work/codec-corpus/gb82-sc";

/// (name, content-class, absolute path). Default = the 3 calibration-seed
/// images; override with `--corpus-file <tsv>` (cols: path, name, class).
fn default_corpus() -> Vec<(String, String, String)> {
    vec![
        (
            "1418519".into(),
            "photo".into(),
            format!("{CID22_VAL_DIR}/1418519.png"),
        ),
        (
            "1025469".into(),
            "photo".into(),
            format!("{CID22_VAL_DIR}/1025469.png"),
        ),
        (
            "codec_wiki".into(),
            "screen".into(),
            format!("{GB82_SC_DIR}/codec_wiki.png"),
        ),
    ]
}

/// Parse a corpus TSV: each line `abs_path \t name \t class` (class optional,
/// defaults to "image"). Blank lines + `#` comments skipped.
fn corpus_from_file(path: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for line in std::fs::read_to_string(path).expect("corpus file").lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut f = line.split('\t');
        let p = f.next().unwrap_or("").to_string();
        let name = f.next().map(|s| s.to_string()).unwrap_or_else(|| {
            std::path::Path::new(&p)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        });
        let class = f.next().unwrap_or("image").to_string();
        if !p.is_empty() {
            out.push((name, class, p));
        }
    }
    out
}

const EFFORT: u8 = 8;

fn load_rgb8(path: &str) -> Result<(Vec<u8>, u32, u32), Box<dyn std::error::Error + Send + Sync>> {
    let img = image::open(path)?.to_rgb8();
    let (w, h) = (img.width(), img.height());
    Ok((img.into_raw(), w, h))
}

fn decode_jxl_srgb_u8(
    encoded: &[u8],
    w: u32,
    h: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let mut decoder = jxl_oxide::JxlImage::builder().read(Cursor::new(encoded))?;
    let frame = decoder.render_frame(0)?;
    let stream = frame.stream();
    let (dec_w, dec_h) = (stream.width(), stream.height());
    if dec_w != w || dec_h != h {
        return Err(format!("decoded dims {dec_w}x{dec_h} != {w}x{h}").into());
    }
    let ch = stream.channels() as usize;
    let mut f32buf: Vec<f32> = vec![0.0; (dec_w as usize) * (dec_h as usize) * ch];
    let mut s = stream;
    let _ = s.write_to_buffer(&mut f32buf);
    let n_px = (dec_w as usize) * (dec_h as usize);
    let mut rgb = Vec::with_capacity(n_px * 3);
    for i in 0..n_px {
        let r = (f32buf[i * ch].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = if ch >= 2 {
            (f32buf[i * ch + 1].clamp(0.0, 1.0) * 255.0).round() as u8
        } else {
            r
        };
        let b = if ch >= 3 {
            (f32buf[i * ch + 2].clamp(0.0, 1.0) * 255.0).round() as u8
        } else {
            r
        };
        rgb.extend_from_slice(&[r, g, b]);
    }
    Ok(rgb)
}

fn parse_metric(s: &str) -> PerceptualMetric {
    match s {
        "zensim" => PerceptualMetric::Zensim,
        #[cfg(any(feature = "cvvdp-loop", feature = "cvvdp-loop-cpu"))]
        "cvvdp" => PerceptualMetric::Cvvdp,
        "butteraugli" => PerceptualMetric::Butteraugli,
        other => {
            eprintln!("unknown/feature-gated metric '{other}', falling back to butteraugli");
            PerceptualMetric::Butteraugli
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut metric_s = "zensim".to_string();
    let mut label = "zensim_v47_default".to_string();
    let mut out_dir = PathBuf::from("/mnt/v/output/zensim/diffmap-rd-2026-05-27");
    // Diffmap-driven per-tile redistribution only runs when iters >= 1 (iter 0
    // is compare-only). With iters=0 the diffmap is computed but NEVER used, so
    // every DiffmapOptions/metric variant produces byte-identical output —
    // useless for a diffmap comparison. Default to 3 redistribution iters.
    let mut iters: u32 = 3;
    let mut corpus_file: Option<String> = None;
    let mut distances: Vec<f32> = vec![1.0, 2.0, 3.0];
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--metric" => metric_s = args.next().unwrap_or(metric_s),
            "--label" => label = args.next().unwrap_or(label),
            "--out-dir" => out_dir = PathBuf::from(args.next().unwrap_or_default()),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters),
            "--corpus-file" => corpus_file = args.next(),
            "--distances" => {
                if let Some(s) = args.next() {
                    distances = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                }
            }
            _ => {}
        }
    }
    let corpus = match &corpus_file {
        Some(f) => corpus_from_file(f),
        None => default_corpus(),
    };
    let metric = parse_metric(&metric_s);
    let decoded_dir = out_dir.join("decoded");
    let ref_dir = out_dir.join("ref");
    fs::create_dir_all(&decoded_dir)?;
    fs::create_dir_all(&ref_dir)?;

    let manifest_path = out_dir.join(format!("manifest_{label}.tsv"));
    let mut manifest =
        String::from("ref_path\tdist_path\tlabel\timage\tcorpus\tdistance\tbytes\tencode_ms\n");
    eprintln!(
        "[diffmap_rd] metric={metric_s} label={label} | ZENSIM_MASKING={} ZENSIM_SQRT={} ZENSIM_HF={} ZENSIM_EDGE_MSE={}",
        std::env::var("ZENSIM_MASKING").unwrap_or_else(|_| "default(8)".into()),
        std::env::var("ZENSIM_SQRT").unwrap_or_else(|_| "default(0)".into()),
        std::env::var("ZENSIM_HF").unwrap_or_else(|_| "default(1)".into()),
        std::env::var("ZENSIM_EDGE_MSE").unwrap_or_else(|_| "default(1)".into()),
    );

    eprintln!(
        "[diffmap_rd] corpus={} images, distances={:?}, iters={iters}",
        corpus.len(),
        distances
    );
    for (name, corpus_name, path) in &corpus {
        let (name, corpus_name) = (name.as_str(), corpus_name.as_str());
        let (pixels, w, h) = load_rgb8(path)?;
        // Save the clean reference once (as PNG) for the external scorer.
        let ref_png = ref_dir.join(format!("{name}.png"));
        if !ref_png.exists() {
            image::RgbImage::from_raw(w, h, pixels.clone())
                .ok_or("ref from_raw")?
                .save(&ref_png)?;
        }
        for &d in &distances {
            let t = Instant::now();
            let mut cfg = LossyConfig::new(d)
                .with_strategy(EncoderStrategy::Zenjxl)
                .with_effort(EFFORT)
                .with_perceptual_metric(metric);
            // Engage the diffmap-driven per-tile redistribution loop for the
            // active metric (iter 0 alone is compare-only → diffmap unused).
            // The quality loops are mutually exclusive — zero butteraugli's
            // default iters when driving with zensim/cvvdp.
            // zensim has a SEPARATE zensim_iters (and butteraugli_iters must be
            // 0 — mutually exclusive). cvvdp + butteraugli both drive the shared
            // buttloop via butteraugli_iters (perceptual_loop resolves
            // iters_override.unwrap_or(self.butteraugli_iters)); cvvdp is
            // selected by the perceptual-metric, not a separate iters field.
            cfg = match metric_s.as_str() {
                "zensim" => cfg.with_butteraugli_iters(0).with_zensim_iters(iters),
                _ => cfg.with_butteraugli_iters(iters),
            };
            let encoded = cfg.encode(&pixels, w, h, PixelLayout::Rgb8)?;
            let encode_ms = t.elapsed().as_secs_f64() * 1000.0;
            let bytes = encoded.len();
            let decoded = decode_jxl_srgb_u8(&encoded, w, h)?;
            let dist_png = decoded_dir.join(format!("{label}__{name}__d{d:.2}.png"));
            image::RgbImage::from_raw(w, h, decoded)
                .ok_or("dec from_raw")?
                .save(&dist_png)?;
            manifest.push_str(&format!(
                "{}\t{}\t{label}\t{name}\t{corpus_name}\t{d:.2}\t{bytes}\t{encode_ms:.1}\n",
                ref_png.display(),
                dist_png.display(),
            ));
            eprintln!("  {label} {name} d={d:.2} bytes={bytes} encode={encode_ms:.0}ms");
        }
    }
    fs::write(&manifest_path, &manifest)?;
    eprintln!("[diffmap_rd] wrote manifest {}", manifest_path.display());
    Ok(())
}
