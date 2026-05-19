//! W43-3 chunk 1 A/B bench: `HdrLoss::Butteraugli` vs `HdrLoss::Ssim2`
//! on 5 CID22-512 photos × {d=0.5, 1.0, 2.5, 4.0} × e8.
//!
//! Promotes [`jxl_encoder::HdrLoss::Ssim2`] to a first-class variant
//! by wiring the existing `ssim2-loop`-feature dispatch through the
//! public [`HdrLoss`] enum. This bench measures whether the swap is
//! net-positive on the CID22-512 photo subset before the chunk-2 A.9
//! decisive-rule eval (Mohammadi 6-stat panel) decides on a default
//! flip.
//!
//! Grid:
//! - 5 CID22-512 photos × {d=0.5, 1.0, 2.5, 4.0} × e8
//! - 2 modes per cell: HdrLoss::Butteraugli (PRE) vs HdrLoss::Ssim2 (POST)
//! - 40 cells total
//!
//! Metric capture: jxl-oxide `srgb_linear` decode + Rust
//! `butteraugli_linear` + `fast_ssim2::compute_ssimulacra2`
//! (CLAUDE.md-compliant — no `butteraugli_main`, no PNG metadata bug).
//!
//! Output:
//!   benchmarks/hdr_loss_ssim2_promotion_<UTC>.tsv  (per-cell paired)
//!   benchmarks/hdr_loss_ssim2_promotion_<UTC>.meta (provenance)
//!
//! Reproducer:
//!   cargo run -p jxl-encoder --release \
//!       --features 'std butteraugli-loop ssim2-loop' \
//!       --example hdr_loss_ssim2_promotion_ab

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::{HdrLoss, Limits, LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

// 5 photos covering smooth gradient (1025469), high-detail outdoor
// (1418519, 1531677), and two more from the broader CID22-512 pool.
// Picked to match the buttloop_distance_split_ab bench's photo set
// (1025469, 1418519, 1531677) plus two additional images sampled
// from CID22-512/training and validation directories.
const PHOTOS: &[&str] = &[
    "1025469.png",
    "1418519.png",
    "1531677.png",
    "1080721.png",
    "1189261.png",
];

const DISTANCES: &[f32] = &[0.5, 1.0, 2.5, 4.0];
const EFFORT: u8 = 8;

fn corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/lilith/work/codec-corpus"))
}

fn srgb_to_linear_f32(s: u8) -> f32 {
    let c = s as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

fn rgb_to_linear_img(rgb: &[u8], w: u32, h: u32) -> Img<Vec<RGB<f32>>> {
    let pixels: Vec<RGB<f32>> = rgb
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear_f32(c[0]),
                srgb_to_linear_f32(c[1]),
                srgb_to_linear_f32(c[2]),
            )
        })
        .collect();
    Img::new(pixels, w as usize, h as usize)
}

fn rgb_to_srgb_arr3(rgb: &[u8], w: u32, h: u32) -> Img<Vec<[u8; 3]>> {
    let pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    Img::new(pixels, w as usize, h as usize)
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

fn score(
    bytes: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
) -> Option<(f64, f64)> {
    let (dw, dh, dec_lin) = decode_jxl_linear(bytes)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_img: Img<Vec<RGB<f32>>> = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(
        orig_linear.as_ref(),
        dec_lin_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .ok()?
    .score as f64;
    let dec_srgb_pixels: Vec<[u8; 3]> = dec_lin
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb_pixels, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;
    Some((bfly, ssim2))
}

#[derive(Clone)]
struct Row {
    image: String,
    width: u32,
    height: u32,
    mode: &'static str, // "butteraugli" | "ssim2"
    effort: u8,
    distance: f32,
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
    encode_ms: f64,
}

fn row_header() -> &'static str {
    "image\twidth\theight\tmode\teffort\tdistance\tbytes\tbutteraugli\tssim2\tencode_ms\n"
}

fn row_tsv(r: &Row) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.6}\t{:.4}\t{:.2}\n",
        r.image,
        r.width,
        r.height,
        r.mode,
        r.effort,
        r.distance,
        r.bytes,
        r.butteraugli,
        r.ssim2,
        r.encode_ms,
    )
}

#[derive(Clone)]
struct Cell {
    image: String,
    path: PathBuf,
    distance: f32,
}

fn enumerate_cells() -> Vec<Cell> {
    let corpus = corpus_dir();
    let cid_dir_val = corpus.join("CID22/CID22-512/validation");
    let cid_dir_train = corpus.join("CID22/CID22-512/training");
    let mut cells = Vec::new();
    for name in PHOTOS {
        // Try validation first, fall back to training.
        let p = if cid_dir_val.join(name).exists() {
            cid_dir_val.join(name)
        } else if cid_dir_train.join(name).exists() {
            cid_dir_train.join(name)
        } else {
            eprintln!("skipping missing photo {name}");
            continue;
        };
        for &d in DISTANCES {
            cells.push(Cell {
                image: name.to_string(),
                path: p.clone(),
                distance: d,
            });
        }
    }
    cells
}

fn encode_and_score(
    rgb: &[u8],
    w: u32,
    h: u32,
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    d: f32,
    e: u8,
    loss: HdrLoss,
) -> Option<(usize, f64, f64, f64)> {
    let cfg = LossyConfig::new(d).with_effort(e).with_hdr_loss(loss);
    let lim = Limits::default().with_max_memory_bytes(8u64 * 1024 * 1024 * 1024);
    let t0 = Instant::now();
    let bytes = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_limits(&lim)
        .encode(rgb)
        .map_err(|err| eprintln!("encode failed: {err:?}"))
        .ok()?;
    let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (bfly, ssim2) = score(&bytes, orig_lin, orig_srgb, w, h)?;
    Some((bytes.len(), bfly, ssim2, enc_ms))
}

fn main() {
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let utc_label = format!("{now_secs}");
    let out_tmp = PathBuf::from(format!("/tmp/hdr_loss_ssim2_promotion_{utc_label}.tsv"));
    let out_final = PathBuf::from(format!(
        "benchmarks/hdr_loss_ssim2_promotion_2026-05-19.tsv"
    ));
    let meta_final = out_final.with_extension("meta");

    let cells = enumerate_cells();
    eprintln!(
        "cells planned: {} (×2 modes = {} encodes)",
        cells.len(),
        cells.len() * 2
    );

    let mut tmp = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&out_tmp)
        .expect("open tmp");
    tmp.write_all(row_header().as_bytes()).unwrap();

    let n_cells = cells.len();
    let mut rows: Vec<Row> = Vec::with_capacity(n_cells * 2);
    let mut failed: Vec<String> = Vec::new();

    for (idx, c) in cells.iter().enumerate() {
        let (rgb, w, h) = match load_png(&c.path) {
            Some(t) => t,
            None => {
                eprintln!("load failed: {}", c.path.display());
                failed.push(format!("load:{}", c.image));
                continue;
            }
        };
        let orig_lin = rgb_to_linear_img(&rgb, w, h);
        let orig_srgb = rgb_to_srgb_arr3(&rgb, w, h);

        for &(mode, loss) in &[
            ("butteraugli", HdrLoss::Butteraugli),
            ("ssim2", HdrLoss::Ssim2),
        ] {
            match encode_and_score(&rgb, w, h, &orig_lin, &orig_srgb, c.distance, EFFORT, loss) {
                Some((bytes, bfly, ssim2, enc_ms)) => {
                    let r = Row {
                        image: c.image.clone(),
                        width: w,
                        height: h,
                        mode,
                        effort: EFFORT,
                        distance: c.distance,
                        bytes,
                        butteraugli: bfly,
                        ssim2,
                        encode_ms: enc_ms,
                    };
                    tmp.write_all(row_tsv(&r).as_bytes()).unwrap();
                    tmp.flush().unwrap();
                    rows.push(r);
                }
                None => {
                    failed.push(format!(
                        "{}:e{}:d{:.2}:{}",
                        c.image, EFFORT, c.distance, mode
                    ));
                }
            }
        }

        // Refresh `.workongoing` marker every 2 cells (~30s of work).
        if idx % 2 == 0 {
            let _ = std::process::Command::new("sh")
                .arg("-c")
                .arg(format!(
                    "date -u +%Y-%m-%dT%H:%M:%SZ | xargs -I {{}} printf '%s claude-w43-3 hdrloss-ssim2-bench cell {}/{}\\n' {{}} > .workongoing",
                    idx + 1, n_cells
                ))
                .status();
            eprintln!("[{}/{}] {} d={:.2}", idx + 1, n_cells, c.image, c.distance);
        }
    }

    drop(tmp);

    std::fs::create_dir_all(out_final.parent().unwrap()).ok();
    std::fs::rename(&out_tmp, &out_final).expect("atomic mv tmp -> repo");

    // Paired aggregation: (image, distance) → (Butteraugli row, Ssim2 row).
    let mut summary = String::new();
    summary.push_str("\n=== Paired aggregates (ssim2 - butteraugli) per distance ===\n");
    summary.push_str(&format!(
        "{:<8} {:<6} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
        "dist", "n", "d_bytes%", "d_bfly%", "d_ssim2", "bt_bytes", "ss_bytes", "bt_bfly",
    ));

    type AggKey = u32; // distance × 100
    type PairBucket = Vec<(Row, Row)>;
    let mut by_dist: std::collections::BTreeMap<AggKey, PairBucket> =
        std::collections::BTreeMap::new();
    use std::collections::BTreeMap;
    let mut keyed: BTreeMap<(String, u32, &'static str), Row> = BTreeMap::new();
    for r in &rows {
        keyed.insert(
            (r.image.clone(), (r.distance * 100.0) as u32, r.mode),
            r.clone(),
        );
    }
    for r_bt in rows.iter().filter(|r| r.mode == "butteraugli") {
        let key = (r_bt.image.clone(), (r_bt.distance * 100.0) as u32, "ssim2");
        if let Some(r_ss) = keyed.get(&key) {
            by_dist
                .entry((r_bt.distance * 100.0) as u32)
                .or_default()
                .push((r_bt.clone(), r_ss.clone()));
        }
    }
    for (dx100, pairs) in &by_dist {
        let n = pairs.len();
        let mut sum_db = 0.0f64;
        let mut sum_dbf = 0.0f64;
        let mut sum_ds = 0.0f64;
        let mut sum_bt_b = 0.0f64;
        let mut sum_ss_b = 0.0f64;
        let mut sum_bt_bf = 0.0f64;
        for (bt, ss) in pairs {
            let db = (ss.bytes as f64 - bt.bytes as f64) / bt.bytes as f64 * 100.0;
            let dbf = (ss.butteraugli - bt.butteraugli) / bt.butteraugli * 100.0;
            let ds = ss.ssim2 - bt.ssim2;
            sum_db += db;
            sum_dbf += dbf;
            sum_ds += ds;
            sum_bt_b += bt.bytes as f64;
            sum_ss_b += ss.bytes as f64;
            sum_bt_bf += bt.butteraugli;
        }
        let d = *dx100 as f32 / 100.0;
        summary.push_str(&format!(
            "{:<8.2} {:<6} {:>+8.2} {:>+8.2} {:>+8.3} {:>10.0} {:>10.0} {:>10.4}\n",
            d,
            n,
            sum_db / n as f64,
            sum_dbf / n as f64,
            sum_ds / n as f64,
            sum_bt_b / n as f64,
            sum_ss_b / n as f64,
            sum_bt_bf / n as f64,
        ));
    }
    if !failed.is_empty() {
        summary.push_str(&format!("\nFAILED ({} cells):\n", failed.len()));
        for f in &failed {
            summary.push_str(&format!("  {f}\n"));
        }
    }

    eprintln!("{}", summary);

    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into());
    let mut meta = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&meta_final)
        .expect("meta open");
    writeln!(meta, "# W43-3 chunk 1 — HdrLoss::Ssim2 promotion A/B").unwrap();
    writeln!(meta, "# Generated: {utc_label}").unwrap();
    writeln!(meta, "# Host: {host}").unwrap();
    writeln!(
        meta,
        "# Promote HdrLoss::Ssim2 to first-class variant; bench vs HdrLoss::Butteraugli"
    )
    .unwrap();
    writeln!(
        meta,
        "# Grid: 5 CID22-512 photos × {{d=0.5, 1.0, 2.5, 4.0}} × e8 = 40 cells"
    )
    .unwrap();
    writeln!(
        meta,
        "# PRE  = HdrLoss::Butteraugli (default Auto resolves to Butteraugli on SDR)"
    )
    .unwrap();
    writeln!(
        meta,
        "# POST = HdrLoss::Ssim2 (routes butteraugli_iters budget through ssim2_refine_quant_field)"
    )
    .unwrap();
    writeln!(meta, "# Decoder: jxl-oxide srgb_linear (metadata-immune).").unwrap();
    writeln!(
        meta,
        "# Metric: Rust butteraugli_linear + fast_ssim2::compute_ssimulacra2."
    )
    .unwrap();
    writeln!(meta, "# Total rows: {}", rows.len()).unwrap();
    writeln!(meta, "# Failed cells: {}", failed.len()).unwrap();
    writeln!(meta, "#").unwrap();
    write!(meta, "{summary}").unwrap();
    drop(meta);

    eprintln!("wrote {}", out_final.display());
    eprintln!("wrote {}", meta_final.display());
}
