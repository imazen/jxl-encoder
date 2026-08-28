//! Effort-ladder bench: bytes + wall per tier e9..=e13 on the imazen-26
//! gate fixtures at 256² / 1024² / 4 MP crops, lossless and lossy d1.0,
//! plus the `resampling = 2` lossy cell (the e10 iterative downsampler).
//!
//! Every stream is decoded in-process with jxl-oxide (dimensions
//! asserted, frame rendered) before its row is written — a row can never
//! report bytes for an undecodable stream.
//!
//! Usage (see `just effort-ladder-bench`):
//!   CODEC_CORPUS_DIR=~/work/codec-corpus nice -n 19 cargo run --release \
//!     -p jxl-encoder --features parallel --example effort_ladder_shift \
//!     > benchmarks/effort_ladder_shift_<date>.tsv
//!
//! Environment:
//!   CODEC_CORPUS_DIR  corpus root holding `imazen-26/` (the 5 gate
//!                     files from scripts/hunt/fetch_imazen26_gate_files.sh)
//!   EFFORTS           comma list, default "9,10,11,12,13"
//!   SIZES             comma list of square crop edges, default "256,1024,2048"
//!   THREADS           encoder threads per encode, default 4
//!   MODES             comma list of {lossless,lossy,lossy_r2}, default all
//!   FIXTURES          comma list of fixture keys (doc,plot,web,ai,scan), default all

use jxl_encoder::api::{LosslessConfig, LossyConfig, PixelLayout};
use std::path::{Path, PathBuf};
use std::time::Instant;

const FIXTURES: &[(&str, &str)] = &[
    (
        "doc",
        "imazen-26/5300-noaa-hurricane-documents/5308_noaa_nhc-al022024-beryl_p01_2550x3300.png",
    ),
    (
        "plot",
        "imazen-26/7000-lilith-plots/aliased-polygons/7026_plots_line-00081-s68404bb1_1024x1024.png",
    ),
    (
        "web",
        "imazen-26/8100-lilith-web-screenshots/1440x900/8106_web-screenshots_climate-news_dpr1_page1_1440x900.png",
    ),
    (
        "ai",
        "imazen-26/9226-lilith-ai-products/grocery/9678_gen_products-grocery_cereal-box-coral_p0439_1024x1536.png",
    ),
    (
        "scan",
        "imazen-26/6800-ia-scans-manuscript-text/6824_scans-text_redoute-fr-description_p0052_2415x3528.png",
    ),
];

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn corpus_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
    PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR").unwrap_or_else(|_| format!("{home}/work/codec-corpus")),
    )
}

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap_or_else(|e| {
        panic!(
            "failed to read `{}`: {e}. Fetch the gate fixtures with \
             CODEC_CORPUS_DIR=... scripts/hunt/fetch_imazen26_gate_files.sh",
            path.display()
        )
    });
    let rgb = img.to_rgb8();
    (rgb.as_raw().clone(), rgb.width(), rgb.height())
}

/// Centre crop (document scans carry blank margins at the corners).
fn crop(rgb: &[u8], w: u32, h: u32, edge: u32) -> Vec<u8> {
    let (w, h, e) = (w as usize, h as usize, edge as usize);
    let x0 = (w - e) / 2;
    let y0 = (h - e) / 2;
    let mut out = Vec::with_capacity(e * e * 3);
    for y in y0..y0 + e {
        out.extend_from_slice(&rgb[(y * w + x0) * 3..(y * w + x0 + e) * 3]);
    }
    out
}

fn decode_check(name: &str, data: &[u8], w: u32, h: u32) {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .unwrap_or_else(|e| panic!("{name}: jxl-oxide header parse failed: {e:?}"));
    let hdr = image.image_header();
    assert_eq!(
        (hdr.size.width, hdr.size.height),
        (w, h),
        "{name}: decoded dims"
    );
    image
        .render_frame(0)
        .unwrap_or_else(|e| panic!("{name}: jxl-oxide render failed: {e:?}"));
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn main() {
    let efforts: Vec<u8> = env_list("EFFORTS", "9,10,11,12,13")
        .iter()
        .map(|s| s.parse().expect("EFFORTS"))
        .collect();
    let sizes: Vec<u32> = env_list("SIZES", "256,1024,2048")
        .iter()
        .map(|s| s.parse().expect("SIZES"))
        .collect();
    let threads: usize = std::env::var("THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let modes = env_list("MODES", "lossless,lossy,lossy_r2");
    let fixtures = env_list("FIXTURES", "doc,plot,web,ai,scan");
    let corpus = corpus_dir();

    println!(
        "# effort_ladder_shift — commit {} host {} threads {}",
        git_commit(),
        hostname(),
        threads
    );
    println!(
        "# efforts {:?} sizes {:?} modes {:?}; every row decoded in-process with jxl-oxide",
        efforts, sizes, modes
    );
    println!("fixture\tcrop\tmode\teffort\tbytes\twall_ms");

    for (key, rel) in FIXTURES {
        if !fixtures.iter().any(|f| f == key) {
            continue;
        }
        let (rgb, w, h) = load_png(&corpus.join(rel));
        for &edge in &sizes {
            if edge > w || edge > h {
                continue;
            }
            let px = crop(&rgb, w, h, edge);
            for mode in &modes {
                for &effort in &efforts {
                    if mode == "lossy_r2" && effort > 10 {
                        // The downsampler is the only lossy e10 knob this
                        // cell isolates; higher tiers add nothing to it.
                        continue;
                    }
                    let name = format!("{key}_{edge}_{mode}_e{effort}");
                    let t0 = Instant::now();
                    let data = match mode.as_str() {
                        "lossless" => LosslessConfig::new()
                            .with_effort(effort)
                            .with_threads(threads)
                            .encode(&px, edge, edge, PixelLayout::Rgb8)
                            .unwrap_or_else(|e| panic!("{name}: {e}")),
                        "lossy" => LossyConfig::new(1.0)
                            .with_effort(effort)
                            .with_threads(threads)
                            .encode(&px, edge, edge, PixelLayout::Rgb8)
                            .unwrap_or_else(|e| panic!("{name}: {e}")),
                        "lossy_r2" => LossyConfig::new(1.0)
                            .with_effort(effort)
                            .with_resampling(2)
                            .with_threads(threads)
                            .encode(&px, edge, edge, PixelLayout::Rgb8)
                            .unwrap_or_else(|e| panic!("{name}: {e}")),
                        other => panic!("unknown MODES entry `{other}`"),
                    };
                    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    decode_check(&name, &data, edge, edge);
                    println!(
                        "{key}\t{edge}\t{mode}\t{effort}\t{}\t{wall_ms:.1}",
                        data.len()
                    );
                    eprintln!("{name}: {} B {wall_ms:.1} ms", data.len());
                }
            }
        }
    }
}
