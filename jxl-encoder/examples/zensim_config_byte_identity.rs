// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Byte-identity harness for the zensim loop's configuration surface.
//!
//! Built for the config-over-flags Phase 1 migration
//! (`~/work/zen-workspace/CONFIG_OVER_FLAGS_2026-08-31.md`), whose defining
//! property is that it is BEHAVIOUR-PRESERVING: replacing 27 scattered
//! `env::var` reads with one typed config must move zero output bytes, with
//! and without each env var set.
//!
//! No hash lock covers either zensim path — both are behind `zensim-loop` AND
//! an explicit `PerceptualMetric::Zensim` opt-in — so the locks can only prove
//! the absence of collateral damage. This harness is the direct evidence:
//! encode a corpus sample and print a SHA256 per cell. Run it on both sides of
//! a change and `diff` the two TSVs; every line must match.
//!
//! `--route` picks WHICH zensim entry point runs, and the two are disjoint —
//! see [`Route`]. A harness that drives only one proves nothing about the
//! other, which is worth stating because it is easy to assume otherwise.
//!
//! One env arm per PROCESS, deliberately: `JXL_ZENSIM_RD_PROFILE` resolves
//! through a process-wide `OnceLock`, so a harness that mutated env between
//! encodes in one process would be measuring the first arm forever. The driver
//! for the `loop` route's env matrix is
//! `scripts/zensim-loop-eff/byte_identity_matrix.sh`.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release -p jxl-encoder --features zensim-loop \
//!   --example zensim_config_byte_identity -- \
//!   --corpus ~/work/codec-corpus/CID22/CID22-512/validation --limit 4
//! ```

#![cfg(feature = "zensim-loop")]

use jxl_encoder::api::{EncoderStrategy, PerceptualDevice, PerceptualMetric};
use jxl_encoder::{LossyConfig, PixelLayout};
use sha2::{Digest, Sha256};

/// Which zensim entry point to drive. There are TWO, they are reached by
/// different config, and a harness that only drives one proves nothing about
/// the other — which is exactly the trap this enum exists to stop.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Route {
    /// `with_zensim_iters(N)` → `VarDctEncoder::zensim_refine_quant_field`
    /// (`vardct/zensim_loop.rs`). The config-over-flags Phase 1 surface.
    Loop,
    /// `with_butteraugli_iters(N)` + `PerceptualMetric::Zensim` +
    /// `PerceptualDevice::Cpu` → the metric-agnostic buttloop driving
    /// `CpuZensimBackend` (`vardct/zensim_backend.rs`). Disjoint from `Loop`:
    /// neither route executes the other's code.
    Backend,
}

/// Distances × efforts the matrix walks. Deliberately small: what needs
/// covering here is CODE PATHS across a mechanical refactor, not an RD surface,
/// and every cell costs a full zensim loop (4 compares). One fine and one
/// aggressive distance exercise the controller at both ends; e6/e7 differ in
/// the surrounding VarDCT work the loop reconstructs through. Widen the arm
/// count in the driver script before widening this — arms are what find a
/// behaviour change, cells only make each arm louder.
const DISTANCES: &[f32] = &[1.0, 4.0];
const EFFORTS: &[u8] = &[6, 7];
const ZENSIM_ITERS: u32 = 3;

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    let digest = h.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest.iter() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// A deterministic synthetic fixture, used only when no corpus is given.
/// Procedural so the harness still runs on a box without the corpus; a real
/// corpus is what the acceptance evidence uses.
fn synthetic(w: u32, h: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 255) / w) as u8;
            let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
            let edge = if (y / 16) % 2 == 0 { 40u8 } else { 0 };
            pixels.extend_from_slice(&[
                base.wrapping_add(tex),
                base.wrapping_add(edge),
                (255 - base).wrapping_add(tex / 2),
            ]);
        }
    }
    pixels
}

fn main() {
    let mut corpus: Option<String> = None;
    let mut limit = 4usize;
    let mut label = String::from("arm");
    let mut route = Route::Loop;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                corpus = args.get(i + 1).cloned();
                i += 2;
            }
            "--limit" => {
                limit = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(limit);
                i += 2;
            }
            "--label" => {
                label = args.get(i + 1).cloned().unwrap_or(label);
                i += 2;
            }
            "--route" => {
                route = match args.get(i + 1).map(String::as_str) {
                    Some("loop") => Route::Loop,
                    Some("backend") => Route::Backend,
                    other => {
                        eprintln!("--route wants loop|backend, got {other:?}");
                        std::process::exit(2);
                    }
                };
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    // (name, width, height, rgb8 pixels)
    let mut images: Vec<(String, u32, u32, Vec<u8>)> = Vec::new();
    if let Some(dir) = &corpus {
        let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("corpus dir {dir}: {e}"))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "png"))
            .collect();
        paths.sort();
        for p in paths.into_iter().take(limit) {
            let img = image::open(&p)
                .unwrap_or_else(|e| panic!("open {}: {e}", p.display()))
                .to_rgb8();
            let (w, h) = img.dimensions();
            let name = p.file_stem().unwrap().to_string_lossy().into_owned();
            images.push((name, w, h, img.into_raw()));
        }
    }
    if images.is_empty() {
        images.push(("synthetic128".into(), 128, 128, synthetic(128, 128)));
        images.push(("synthetic320".into(), 320, 256, synthetic(320, 256)));
    }
    // ALWAYS append two fixtures whose dimensions are NOT multiples of 8, even
    // when a corpus was given. `padded_width = ceil(w/8)*8`, so a corpus of
    // 512×512 images has `padded_width == width` on every cell and exercises
    // only the tightly-packed branch of anything stride-sensitive. That is the
    // lock-coverage trap this repo has now hit twice (the r2 resampling cells
    // were all 512-wide, and 512 is divisible by 8, so a dimension bug survived
    // a "complete" lock grid). These two make the strided branch unconditional.
    images.push(("odd_263x139".into(), 263, 139, synthetic(263, 139)));
    images.push(("odd_505x257".into(), 505, 257, synthetic(505, 257)));

    let route_name = match route {
        Route::Loop => "loop",
        Route::Backend => "backend",
    };
    println!("# route\tlabel\timage\tw\th\tdistance\teffort\tbytes\tsha256");
    for (name, w, h, pixels) in &images {
        for &d in DISTANCES {
            for &e in EFFORTS {
                let base = LossyConfig::new(d)
                    .with_strategy(EncoderStrategy::Zenjxl)
                    .with_effort(e)
                    .with_perceptual_metric(PerceptualMetric::Zensim);
                let cfg = match route {
                    Route::Loop => base
                        .with_butteraugli_iters(0)
                        .with_zensim_iters(ZENSIM_ITERS),
                    // `PerceptualDevice::Cpu` is what selects `CpuZensimBackend`
                    // over the GPU one; the buttloop iteration count is what
                    // makes the loop run at all.
                    Route::Backend => base
                        .with_perceptual_device(PerceptualDevice::Cpu)
                        .with_butteraugli_iters(ZENSIM_ITERS)
                        .with_zensim_iters(0),
                };
                let encoded = cfg
                    .encode(pixels, *w, *h, PixelLayout::Rgb8)
                    .unwrap_or_else(|err| panic!("{name} d={d} e={e}: {err:?}"));
                println!(
                    "{route_name}\t{label}\t{name}\t{w}\t{h}\t{d}\t{e}\t{}\t{}",
                    encoded.len(),
                    sha256_hex(&encoded)
                );
            }
        }
    }
}
