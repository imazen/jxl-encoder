//! Full RD curve audit of jxl-encoder vs cjxl across 15 distances × 5 efforts.
//!
//! Designed to find wedges where we lose to cjxl on bytes-vs-quality Pareto.
//! Per user direction (2026-05-18): step .2 in butteraugli distance from 0.2 to 3.0,
//! step 1 above 3.0, 15 bands max. Effort sweep e5..e9.
//!
//! Corpus: 5 CID22-512 photos + 3 gb82-sc screenshots (= 8 images).
//! Per-cell metrics: bytes, butteraugli, ssim2, encode_ms.
//! Decoder: jxl-oxide for linear-sRGB sources (per CLAUDE.md metric integrity).
//! Encoder reference: cjxl v0.12.0 with `-e <effort> -d <distance>`.
//!
//! Output: `benchmarks/rd_curve_audit_vs_cjxl_<UTC>.tsv` (one row per
//! `(image, effort, distance, encoder)` cell) + `.meta` sidecar with provenance.
//!
//! Reproducer:
//! ```bash
//! cargo run -p jxl-encoder --release --example rd_curve_audit_vs_cjxl
//! ```
//!
//! Per-row schema (tsv):
//!   image | size | encoder | effort | distance | bytes | butteraugli | ssim2 | encode_ms

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rayon::prelude::*;
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::Instant;

// 15 distance bands per user spec.
const DISTANCES: &[f32] = &[
    0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0,
];

const EFFORTS: &[u8] = &[5, 6, 7, 8, 9];

// 5 CID22-512 photo SHAs + 3 gb82-sc screenshots. CID22 validation is the
// "hold-out" set per the picker-train manifest convention.
const CID22_PHOTOS: &[&str] = &[
    "1025469.png",
    "1418519.png",
    "1531677.png",
    "1189261.png",
    "1420710.png",
];

const SCREENSHOTS: &[&str] = &["terminal.png", "codec_wiki.png", "imac_g3.png"];

fn corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/lilith/work/codec-corpus"))
}

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
    PathBuf::from(home).join("work/jxl-efforts/libjxl/build/tools/cjxl")
}

// `djxl` not used directly — we decode through jxl-oxide for metric integrity
// (no PNG/CMS round-trip per CLAUDE.md). Retained for future round-trip checks.
#[allow(dead_code)]
fn djxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("DJXL") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
    PathBuf::from(home).join("work/jxl-efforts/libjxl/build/tools/djxl")
}

// ── Helpers (mirrors lossy_pareto_calibrate) ─────────────────────────────────

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

/// Score a JXL bytestring against original linear + srgb references.
fn score_jxl(
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

    let dec_srgb: Vec<[u8; 3]> = dec_lin
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;

    Some((bfly, ssim2))
}

// ── Encoders ────────────────────────────────────────────────────────────────

fn encode_ours(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> Option<(Vec<u8>, f64)> {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Some((bytes, ms))
}

fn encode_cjxl(
    src_png: &Path,
    distance: f32,
    effort: u8,
    work_dir: &Path,
    tag: &str,
) -> Option<(Vec<u8>, f64)> {
    let out = work_dir.join(format!("{tag}_cjxl_e{effort}_d{:.2}.jxl", distance));
    let _ = std::fs::remove_file(&out);
    let start = Instant::now();
    let status = Command::new(cjxl_bin())
        .arg(src_png)
        .arg(&out)
        .args(["-e", &effort.to_string()])
        .args(["-d", &format!("{distance}")])
        .args(["--num_threads", "1"])
        .arg("--quiet")
        .output()
        .ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    if !status.status.success() {
        eprintln!(
            "cjxl failed for {} e{} d{}: {}",
            src_png.display(),
            effort,
            distance,
            String::from_utf8_lossy(&status.stderr)
        );
        return None;
    }
    let bytes = std::fs::read(&out).ok()?;
    let _ = std::fs::remove_file(&out);
    Some((bytes, ms))
}

// ── Args ────────────────────────────────────────────────────────────────────

struct Args {
    output: PathBuf,
    output_meta: PathBuf,
    threads: usize,
    distances: Vec<f32>,
    efforts: Vec<u8>,
    images: Option<Vec<String>>,
    smoke: bool,
    resume: bool,
}

fn parse_args() -> Args {
    let mut output: Option<PathBuf> = None;
    let mut threads: usize = 0;
    let mut distances: Vec<f32> = Vec::new();
    let mut efforts: Vec<u8> = Vec::new();
    let mut images: Option<Vec<String>> = None;
    let mut smoke = false;
    let mut resume = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--output" => output = Some(PathBuf::from(it.next().expect("--output VALUE"))),
            "--threads" => {
                threads = it
                    .next()
                    .expect("--threads VALUE")
                    .parse()
                    .expect("threads must be uint")
            }
            "--distances" => {
                distances = it
                    .next()
                    .expect("--distances LIST")
                    .split(',')
                    .map(|s| s.parse().expect("float"))
                    .collect()
            }
            "--efforts" => {
                efforts = it
                    .next()
                    .expect("--efforts LIST")
                    .split(',')
                    .map(|s| s.parse().expect("u8"))
                    .collect()
            }
            "--images" => {
                images = Some(
                    it.next()
                        .expect("--images LIST")
                        .split(',')
                        .map(|s| s.to_string())
                        .collect(),
                )
            }
            "--smoke" => smoke = true,
            "--resume" => resume = true,
            other => panic!("unknown arg: {other}"),
        }
    }
    if distances.is_empty() {
        distances = DISTANCES.to_vec();
    }
    if efforts.is_empty() {
        efforts = EFFORTS.to_vec();
    }
    if smoke {
        // Smoke test: 1 image, 3 distances, 2 efforts.
        images = Some(vec!["1025469.png".to_string()]);
        distances = vec![0.4, 1.0, 2.0];
        efforts = vec![5, 7];
    }
    let output = output.unwrap_or_else(|| {
        PathBuf::from(format!(
            "benchmarks/rd_curve_audit_vs_cjxl_{}.tsv",
            chrono_today()
        ))
    });
    let output_meta = output.with_extension("meta");
    Args {
        output,
        output_meta,
        threads,
        distances,
        efforts,
        images,
        smoke,
        resume,
    }
}

fn chrono_today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days = (secs / 86400) as i64;
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

// ── Image discovery ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct ImageEntry {
    name: String,        // file stem-ish display name
    class: &'static str, // "photo" | "screenshot"
    path: PathBuf,
}

fn discover_images(filter: &Option<Vec<String>>) -> Vec<ImageEntry> {
    let corpus = corpus_dir();
    let cid_dir = corpus.join("CID22/CID22-512/validation");
    let sc_dir = corpus.join("gb82-sc");
    let mut entries: Vec<ImageEntry> = Vec::new();
    for name in CID22_PHOTOS {
        let p = cid_dir.join(name);
        if p.exists() {
            entries.push(ImageEntry {
                name: name.to_string(),
                class: "photo",
                path: p,
            });
        }
    }
    for name in SCREENSHOTS {
        let p = sc_dir.join(name);
        if p.exists() {
            entries.push(ImageEntry {
                name: name.to_string(),
                class: "screenshot",
                path: p,
            });
        }
    }
    if let Some(filt) = filter {
        entries.retain(|e| filt.contains(&e.name));
    }
    entries
}

// ── Row writing (atomic tmp + append on commit) ─────────────────────────────

#[derive(Clone)]
struct Row {
    image: String,
    class: &'static str,
    width: u32,
    height: u32,
    encoder: &'static str,
    effort: u8,
    distance: f32,
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
    encode_ms: f64,
}

fn row_header() -> String {
    "image\tclass\twidth\theight\tencoder\teffort\tdistance\tbytes\tbutteraugli\tssim2\tencode_ms\n"
        .to_string()
}

fn row_tsv(r: &Row) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.6}\t{:.4}\t{:.2}\n",
        r.image,
        r.class,
        r.width,
        r.height,
        r.encoder,
        r.effort,
        r.distance,
        r.bytes,
        r.butteraugli,
        r.ssim2,
        r.encode_ms,
    )
}

fn write_atomic(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("tsv.tmp");
    {
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp)
            .expect("open tmp");
        f.write_all(contents.as_bytes()).expect("write");
        f.sync_all().ok();
    }
    std::fs::rename(&tmp, path).expect("atomic mv");
}

fn append_atomic(path: &Path, contents: &str) {
    // Read existing → append → atomic write.
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let mut combined = existing;
    combined.push_str(contents);
    write_atomic(path, &combined);
}

fn read_existing_keys(path: &Path) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    if let Ok(s) = std::fs::read_to_string(path) {
        for (i, line) in s.lines().enumerate() {
            if i == 0 {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 7 {
                continue;
            }
            // key = image | encoder | effort | distance
            let key = format!("{}|{}|{}|{}", cols[0], cols[4], cols[5], cols[6]);
            out.insert(key);
        }
    }
    out
}

fn row_key(image: &str, encoder: &str, effort: u8, distance: f32) -> String {
    format!("{}|{}|{}|{:.4}", image, encoder, effort, distance)
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }

    let images = discover_images(&args.images);
    if images.is_empty() {
        eprintln!("No images found. Check CODEC_CORPUS_DIR.");
        std::process::exit(2);
    }

    let total_cells = images.len() * args.efforts.len() * args.distances.len() * 2; // *2 encoders
    eprintln!(
        "RD audit: {} images × {} efforts × {} distances × 2 encoders = {} cells",
        images.len(),
        args.efforts.len(),
        args.distances.len(),
        total_cells,
    );

    // Workspace for cjxl outputs.
    let work_dir = std::env::temp_dir().join(format!("rd_audit_{}", std::process::id()));
    std::fs::create_dir_all(&work_dir).expect("create workdir");

    // Resume detection.
    let existing_keys = if args.resume {
        let k = read_existing_keys(&args.output);
        eprintln!(
            "Resume: {} existing rows in {}",
            k.len(),
            args.output.display()
        );
        k
    } else {
        // Always start fresh unless --resume.
        if args.output.exists() {
            let bak = args
                .output
                .with_extension(format!("tsv.bak-{}", std::process::id()));
            let _ = std::fs::rename(&args.output, &bak);
            eprintln!("Existing output backed up to {}", bak.display());
        }
        std::collections::HashSet::new()
    };

    // Header.
    if !args.output.exists() {
        write_atomic(&args.output, &row_header());
    }

    // Meta sidecar.
    let git_rev = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let host = Command::new("hostname")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let cjxl_ver = Command::new(cjxl_bin())
        .arg("--version")
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .to_string()
        })
        .unwrap_or_else(|| "unknown".into());
    let meta = format!(
        "# RD curve audit jxl-encoder vs cjxl\n\
         git_rev: {git_rev}\n\
         host: {host}\n\
         cjxl_version: {cjxl_ver}\n\
         distances: {:?}\n\
         efforts: {:?}\n\
         images: {:?}\n\
         corpus: {}\n\
         schema: image\\tclass\\twidth\\theight\\tencoder\\teffort\\tdistance\\tbytes\\tbutteraugli\\tssim2\\tencode_ms\n\
         decoder: jxl-oxide (request_color_encoding=srgb_linear)\n\
         metrics: butteraugli_linear (default params), fast_ssim2 (sRGB u8 from linear)\n\
         reproducer: cargo run -p jxl-encoder --release --example rd_curve_audit_vs_cjxl\n\
         smoke: {}\n\
         resume: {}\n",
        args.distances,
        args.efforts,
        images.iter().map(|i| &i.name).collect::<Vec<_>>(),
        corpus_dir().display(),
        args.smoke,
        args.resume,
    );
    write_atomic(&args.output_meta, &meta);

    // Build the cell grid: encode work first (us & cjxl parallel-friendly),
    // then commit per-image.
    let appender = Mutex::new(());

    for (i, img) in images.iter().enumerate() {
        eprintln!(
            "[{}/{}] {} class={} ({})",
            i + 1,
            images.len(),
            img.name,
            img.class,
            img.path.display()
        );
        let (rgb, w, h) = match load_png(&img.path) {
            Some(x) => x,
            None => {
                eprintln!("  ! failed to load {}", img.path.display());
                continue;
            }
        };
        let orig_linear = rgb_to_linear_img(&rgb, w, h);
        let orig_srgb = rgb_to_srgb_arr3(&rgb, w, h);

        // Build the (effort, distance, encoder) work list, dropping resumed cells.
        type Cell = (u8, f32, &'static str);
        let mut work: Vec<Cell> = Vec::new();
        for &e in &args.efforts {
            for &d in &args.distances {
                for enc in &["ours", "cjxl"] {
                    let k = row_key(&img.name, enc, e, d);
                    if existing_keys.contains(&k) {
                        continue;
                    }
                    work.push((e, d, *enc));
                }
            }
        }
        eprintln!("    {} cells to compute", work.len());

        let rows: Vec<Row> = work
            .par_iter()
            .filter_map(|(e, d, enc)| {
                let res = match *enc {
                    "ours" => {
                        let (bytes, ms) = encode_ours(&rgb, w, h, *d, *e)?;
                        let (bfly, ssim2) = score_jxl(&bytes, &orig_linear, &orig_srgb, w, h)?;
                        (bytes.len(), bfly, ssim2, ms)
                    }
                    "cjxl" => {
                        let (bytes, ms) = encode_cjxl(&img.path, *d, *e, &work_dir, &img.name)?;
                        let (bfly, ssim2) = score_jxl(&bytes, &orig_linear, &orig_srgb, w, h)?;
                        (bytes.len(), bfly, ssim2, ms)
                    }
                    _ => unreachable!(),
                };
                Some(Row {
                    image: img.name.clone(),
                    class: img.class,
                    width: w,
                    height: h,
                    encoder: enc,
                    effort: *e,
                    distance: *d,
                    bytes: res.0,
                    butteraugli: res.1,
                    ssim2: res.2,
                    encode_ms: res.3,
                })
            })
            .collect();

        // Atomic append per image (commit incrementally).
        let _g = appender.lock().unwrap();
        let mut chunk = String::new();
        for r in &rows {
            chunk.push_str(&row_tsv(r));
        }
        append_atomic(&args.output, &chunk);
        eprintln!("    + wrote {} rows", rows.len());
    }

    // Cleanup workdir.
    let _ = std::fs::remove_dir_all(&work_dir);

    eprintln!("\nDone. Output: {}", args.output.display());
    eprintln!("Meta:   {}", args.output_meta.display());
}
