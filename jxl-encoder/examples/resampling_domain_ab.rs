// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! A/B harness for the `with_resampling(N)` **colour-space domain** of the
//! encoder-side downsampler (issue #45, task T1).
//!
//! The decoder's upsampling stage runs on the XYB planes BEFORE the inverse
//! colour transform (libjxl `dec_cache.cc` puts `GetUpsamplingStage` ahead of
//! `GetXYBStage`; jxl-oxide matches), and libjxl's encoder correspondingly
//! downsamples the OPSIN image (`enc_frame.cc:740-763`
//! `DownsampleColorChannels(..., Image3F* opsin)` — sharper 2× AND the plain
//! box for 4×/8×). Downsampling linear RGB pre-XYB does not compose with that
//! round trip because averaging does not commute with the XYB nonlinearity.
//!
//! This harness measures the cost of that mismatch, and the gain from fixing
//! it, per cell: encode → decode in-process with jxl-oxide in **linear sRGB**
//! → score with the in-process Rust butteraugli against the linearized source.
//! That is the metadata-immune method the repo CLAUDE.md mandates — never
//! `butteraugli_main`.
//!
//! Run BEFORE and AFTER the change and diff the two TSVs:
//! ```text
//! RESAMP_AB_OUT=~/tmp/before.tsv nice -n 19 cargo run --release -p jxl-encoder \
//!     --example resampling_domain_ab
//! ```
//!
//! Env knobs (all optional):
//! - `RESAMP_AB_OUT`     — TSV output path (default: stdout only)
//! - `RESAMP_AB_IMAGES`  — how many CID22-512 validation PNGs (default 4)
//! - `RESAMP_AB_EFFORTS` — comma list (default `1,3,5,7,9,10`)
//! - `RESAMP_AB_RESAMP`  — comma list (default `2,4,8`)
//! - `RESAMP_AB_DIST`    — comma list (default `1.0,3.0,6.0`)
//! - `RESAMP_AB_CJXL`    — `0` to skip the libjxl reference arm (default: run
//!   it when `cjxl` is on PATH). The reference arm is the ground truth for
//!   "what does downsampling in the right domain buy", since libjxl has always
//!   downsampled the opsin image.
//! - `RESAMP_AB_LOCKCELLS` — `0` to skip the synthetic hash-lock-cell arm.

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::fmt::Write as _;

fn env_list(key: &str, default: &str) -> Vec<String> {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Locate `cjxl` on PATH (or the conventional local libjxl build path).
fn which_cjxl() -> Option<std::path::PathBuf> {
    if let Ok(out) = std::process::Command::new("which").arg("cjxl").output()
        && out.status.success()
    {
        let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !p.is_empty() {
            return Some(p.into());
        }
    }
    let home = std::env::var("HOME").ok()?;
    let built =
        std::path::PathBuf::from(format!("{home}/work/jxl-efforts/libjxl/build/tools/cjxl"));
    built.exists().then_some(built)
}

/// Decode `bytes` with jxl-oxide in linear sRGB and score it against
/// `orig` with the in-process Rust butteraugli. Returns `None` when the
/// stream does not decode — an undecodable cell is a failure to report,
/// never something to silently drop.
fn score(bytes: &[u8], orig: &Img<Vec<RGB<f32>>>, w: u32, h: u32, label: &str) -> Option<f64> {
    let mut image = match jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(bytes)) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("!! {label}: jxl-oxide parse failed: {e:?}");
            return None;
        }
    };
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = match image.render_frame(0) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("!! {label}: jxl-oxide render failed: {e:?}");
            return None;
        }
    };
    let fb = render.image_all_channels();
    if fb.width() != w as usize || fb.height() != h as usize {
        eprintln!(
            "!! {label}: decoded {}x{} != source {w}x{h}",
            fb.width(),
            fb.height()
        );
        return None;
    }
    let dec: Vec<RGB<f32>> = fb
        .buf()
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_img = Img::new(dec, w as usize, h as usize);
    let params = ButteraugliParams::default();
    match butteraugli_linear(orig.as_ref(), dec_img.as_ref(), &params) {
        Ok(r) => Some(r.score),
        Err(e) => {
            eprintln!("!! {label}: butteraugli failed: {e:?}");
            None
        }
    }
}

fn linearize(srgb: &[u8], w: u32, h: u32) -> Img<Vec<RGB<f32>>> {
    let lin: Vec<RGB<f32>> = srgb
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear(c[0]),
                srgb_to_linear(c[1]),
                srgb_to_linear(c[2]),
            )
        })
        .collect();
    Img::new(lin, w as usize, h as usize)
}

/// The `hash_lock_features.rs` `noise_rgb_512x512` fixture, byte-for-byte —
/// so the lock cells this harness reports on are the exact streams the
/// sidecar pins.
fn noise_rgb_512x512() -> Vec<u8> {
    let (w, h) = (512usize, 512usize);
    let mut out = vec![0u8; w * h * 3];
    let mut seed = 1337u64;
    for val in &mut out {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *val = (seed >> 56) as u8;
    }
    out
}

fn main() {
    let n_images: usize = std::env::var("RESAMP_AB_IMAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let efforts: Vec<u8> = env_list("RESAMP_AB_EFFORTS", "1,3,5,7,9,10")
        .iter()
        .map(|s| s.parse().expect("effort"))
        .collect();
    let resamplings: Vec<u32> = env_list("RESAMP_AB_RESAMP", "2,4,8")
        .iter()
        .map(|s| s.parse().expect("resampling"))
        .collect();
    let distances: Vec<f32> = env_list("RESAMP_AB_DIST", "1.0,3.0,6.0")
        .iter()
        .map(|s| s.parse().expect("distance"))
        .collect();
    let run_cjxl = std::env::var("RESAMP_AB_CJXL").as_deref() != Ok("0");
    let run_lock_cells = std::env::var("RESAMP_AB_LOCKCELLS").as_deref() != Ok("0");

    let mut tsv = String::new();
    tsv.push_str("image\twidth\theight\tdistance\teffort\tresampling\tarm\tbytes\tbutteraugli\n");
    let mut row = |image: &str,
                   w: u32,
                   h: u32,
                   d: f32,
                   e: u8,
                   r: u32,
                   arm: &str,
                   bytes: usize,
                   b: Option<f64>| {
        let bs = match b {
            Some(v) => format!("{v:.4}"),
            None => "UNDECODABLE".to_string(),
        };
        println!("{image}\t{w}\t{h}\td{d}\te{e}\tr{r}\t{arm}\t{bytes}\t{bs}");
        let _ = writeln!(
            tsv,
            "{image}\t{w}\t{h}\t{d}\t{e}\t{r}\t{arm}\t{bytes}\t{bs}"
        );
    };

    // ── Arm 1: the synthetic hash-lock fixture, at the exact cells the
    // sidecar pins (so the lock move is evidenced, not asserted). ────────
    if run_lock_cells {
        let px = noise_rgb_512x512();
        let orig = linearize(&px, 512, 512);
        for &(e, r) in &[(3u8, 2u32), (7, 2), (9, 2), (7, 4), (7, 8), (10, 2)] {
            let data = LossyConfig::new(1.0)
                .with_effort(e)
                .with_resampling(r)
                .encode(&px, 512, 512, PixelLayout::Rgb8)
                .unwrap_or_else(|err| panic!("lockcell e{e} r{r}: {err:?}"));
            let b = score(&data, &orig, 512, 512, &format!("lockcell e{e} r{r}"));
            row(
                "hashlock_noise_512",
                512,
                512,
                1.0,
                e,
                r,
                "ours",
                data.len(),
                b,
            );
        }
    }

    // ── Arm 2: the CID22-512 corpus grid. ────────────────────────────────
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus init");
    let dir = corpus
        .get("CID22/CID22-512/validation")
        .expect("codec-corpus CID22/CID22-512/validation");
    let mut pngs: Vec<_> = std::fs::read_dir(&dir)
        .expect("read CID22 dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
        .collect();
    pngs.sort();
    pngs.truncate(n_images);
    assert!(!pngs.is_empty(), "no CID22-512 validation PNGs found");

    let cjxl = if run_cjxl { which_cjxl() } else { None };
    if run_cjxl && cjxl.is_none() {
        eprintln!("note: cjxl not on PATH — libjxl reference arm skipped");
    }
    let tmp = std::env::temp_dir().join(format!("jxl_resamp_ab_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    for path in &pngs {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let img = image::open(path).unwrap().to_rgb8();
        let (w, h) = (img.width(), img.height());
        let srgb = img.as_raw().clone();
        let orig = linearize(&srgb, w, h);

        // Metadata-stripped rewrite for cjxl: the `image` crate drops the
        // ancillary gAMA/cHRM chunks that would otherwise change cjxl's
        // interpretation of the input (CLAUDE.md "PNG Color Metadata").
        let stripped = tmp.join(format!("{name}_stripped.png"));
        if cjxl.is_some() {
            img.save(&stripped).unwrap();
        }

        for &d in &distances {
            for &r in &resamplings {
                for &e in &efforts {
                    let data = LossyConfig::new(d)
                        .with_effort(e)
                        .with_resampling(r)
                        .encode(&srgb, w, h, PixelLayout::Rgb8)
                        .unwrap_or_else(|err| panic!("{name} d{d} e{e} r{r}: {err:?}"));
                    let b = score(&data, &orig, w, h, &format!("{name} d{d} e{e} r{r}"));
                    row(&name, w, h, d, e, r, "ours", data.len(), b);

                    if let Some(ref cjxl) = cjxl {
                        let out_jxl = tmp.join(format!("{name}_d{d}_e{e}_r{r}.jxl"));
                        let status = std::process::Command::new(cjxl)
                            .args([
                                stripped.to_str().unwrap(),
                                out_jxl.to_str().unwrap(),
                                "-d",
                                &format!("{d}"),
                                "-e",
                                &format!("{e}"),
                                &format!("--resampling={r}"),
                                "--num_threads=4",
                                "--quiet",
                            ])
                            .status()
                            .expect("spawn cjxl");
                        if status.success() {
                            let bytes = std::fs::read(&out_jxl).unwrap();
                            let b =
                                score(&bytes, &orig, w, h, &format!("cjxl {name} d{d} e{e} r{r}"));
                            row(&name, w, h, d, e, r, "cjxl", bytes.len(), b);
                            let _ = std::fs::remove_file(&out_jxl);
                        } else {
                            eprintln!("note: cjxl d{d} e{e} r{r} failed on {name} — arm skipped");
                        }
                    }
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);

    if let Ok(out) = std::env::var("RESAMP_AB_OUT") {
        let out = shellexpand_home(&out);
        std::fs::write(&out, &tsv).unwrap_or_else(|e| panic!("write {out}: {e}"));
        eprintln!("wrote {out}");
    }
}

fn shellexpand_home(p: &str) -> String {
    match (p.strip_prefix("~/"), std::env::var("HOME")) {
        (Some(rest), Ok(home)) => format!("{home}/{rest}"),
        _ => p.to_string(),
    }
}
