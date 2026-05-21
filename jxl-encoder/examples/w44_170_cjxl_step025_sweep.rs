//! W44-170 comprehensive cjxl-parity sweep — step 0.25 distance, full effort
//! grid, both Libjxl and Zenjxl strategies, on a varied 20-image corpus.
//!
//! Per the 2026-05-21 user directive:
//!
//! > test with step 0.25 on a more varied and wider corpus vs cjxl at each
//! > effort level and with both libjxl mimic mode and zenjxl and record
//! > tables and make charts and identify cells for improvement. track and
//! > improve both average and outliers while keeping wall time consistent
//! > with effort range and competitive with libjxl. also make this code
//! > an expert of 20 years would be proud of
//!
//! ## Design
//!
//! Extends the `cjxl_parity_ledger` pattern with:
//! - **Manifest-driven corpus** (`--corpus-manifest <PATH>`): TSV of
//!   `relative_path<TAB>class<TAB>name`. Lines starting with `#` are
//!   comments. Paths resolve against `--corpus-dir` (default
//!   `$CODEC_CORPUS_DIR` or `/home/lilith/work/codec-corpus`).
//! - **Per-strategy output**: each `--strategy` value writes its own TSV
//!   (`<output_prefix>_<strategy>.tsv`). The cjxl reference encode is
//!   shared across strategies (cached in memory inside one run; on
//!   resume both TSVs replay the same cjxl numbers).
//! - **Step-0.25 distance grid**: 20 points from 0.25..=5.0 inclusive
//!   by default. Configurable via `--distance-step` / `--distance-min` /
//!   `--distance-max`.
//! - **Resumable per-cell writes**: each completed cell flushes the TSV
//!   atomically. Killed mid-run, the next invocation skips already-done
//!   cells (keyed on `(image, effort, distance)`).
//! - **Best-of-N timing**: `--time-iters N` (default 1). When `N>1`,
//!   `ours_ms` is the min over N encodes for that (image, effort,
//!   distance, strategy) cell. cjxl encode time is recorded from a
//!   single invocation per cell — the cjxl CLI's setup cost dominates
//!   single-image runs.
//! - **Failure-tolerant**: per-cell encode failures (OOM, decode
//!   mismatch) are recorded as a row with `status=FAILED` and the
//!   sweep continues.
//!
//! ## TSV schema (one row per (image, strategy, effort, distance) cell)
//!
//! ```text
//! image | class | width | height | strategy | effort | distance
//! | ours_bytes  | cjxl_bytes
//! | ours_ssim2  | cjxl_ssim2
//! | ours_bfly   | cjxl_bfly
//! | ours_ms     | cjxl_ms
//! | delta_bytes_pct  | delta_ssim2  | delta_bfly_pct  | delta_ms_pct
//! | status | commit | reproducer
//! ```
//!
//! - `delta_bytes_pct = (ours_bytes - cjxl_bytes) / cjxl_bytes * 100`
//! - `delta_ssim2     =  ours_ssim2 - cjxl_ssim2` (positive = ours better)
//! - `delta_bfly_pct  = (ours_bfly - cjxl_bfly) / cjxl_bfly * 100` (negative = ours better)
//! - `delta_ms_pct    = (ours_ms - cjxl_ms) / cjxl_ms * 100`
//! - `status ∈ {OK, FAILED}`. OK means both encodes + both decodes succeeded.
//!
//! ## Usage
//!
//! ```bash
//! # Smoke test (2 imgs × 1 effort × 3 distances × 2 strategies = 12 cells)
//! cargo run -p jxl-encoder --release \
//!   --features 'parallel butteraugli-loop ssim2-loop' \
//!   --example w44_170_cjxl_step025_sweep -- --smoke
//!
//! # Full sweep (20 imgs × 5 efforts × 20 distances × 2 strategies = 4000 cells)
//! cargo run -p jxl-encoder --release \
//!   --features 'parallel butteraugli-loop ssim2-loop' \
//!   --example w44_170_cjxl_step025_sweep -- \
//!   --corpus-manifest benchmarks/corpora/w44_170_varied_corpus.tsv \
//!   --output-prefix benchmarks/cjxl_step025 \
//!   --strategies zenjxl,libjxl
//! ```
//!
//! ## Provenance
//!
//! - commit: recorded per row + meta
//! - cjxl version: recorded in meta
//! - host: recorded in meta
//! - reproducer command: stored per-row so any individual cell can be
//!   re-run with the exact same flags
//!
//! Bench files this run produces:
//! - `<output_prefix>_<strategy>_<date>.tsv` per strategy
//! - `<output_prefix>_<date>.meta` (shared meta describing the run)

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rayon::prelude::*;
use rgb::RGB;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::Instant;

// ── Defaults ────────────────────────────────────────────────────────────────

const DEFAULT_DISTANCE_MIN: f32 = 0.25;
const DEFAULT_DISTANCE_MAX: f32 = 5.0;
const DEFAULT_DISTANCE_STEP: f32 = 0.25;
const DEFAULT_EFFORTS: &[u8] = &[5, 6, 7, 8, 9];

// ── External paths ──────────────────────────────────────────────────────────

fn default_corpus_dir() -> PathBuf {
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

// ── Color helpers (sRGB ↔ linear, matching cjxl_parity_ledger.rs) ───────────

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

/// Encode with our encoder under a given strategy. Records best-of-N
/// timing when `time_iters > 1`.
fn encode_ours(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    strategy: EncoderStrategy,
    time_iters: u32,
) -> Option<(Vec<u8>, f64)> {
    let iters = time_iters.max(1);
    let mut best_bytes: Option<Vec<u8>> = None;
    let mut best_ms = f64::INFINITY;
    for _ in 0..iters {
        let cfg = LossyConfig::new(distance)
            .with_effort(effort)
            .with_strategy(strategy.clone())
            .with_threads(1);
        let start = Instant::now();
        let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok()?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if ms < best_ms {
            best_ms = ms;
            best_bytes = Some(bytes);
        }
    }
    best_bytes.map(|b| (b, best_ms))
}

fn encode_cjxl(
    src_png: &Path,
    distance: f32,
    effort: u8,
    work_dir: &Path,
    tag: &str,
) -> Option<(Vec<u8>, f64)> {
    // Include a thread-unique suffix so two rayon workers that pick the
    // same (tag, effort, distance) at the same instant don't clobber each
    // other's output file. The cjxl_cache covers cross-strategy reuse;
    // this suffix covers the in-flight race that the cache cannot.
    let tid = std::thread::current().id();
    let nonce: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let out = work_dir.join(format!(
        "{tag}_cjxl_e{effort}_d{:.2}_{:?}_{nonce}.jxl",
        distance, tid
    ));
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

// ── Strategy plumbing ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum StrategyTag {
    Libjxl,
    Zenjxl,
}

impl StrategyTag {
    fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "libjxl" => Some(StrategyTag::Libjxl),
            "zenjxl" => Some(StrategyTag::Zenjxl),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            StrategyTag::Libjxl => "libjxl",
            StrategyTag::Zenjxl => "zenjxl",
        }
    }

    fn to_encoder_strategy(self) -> EncoderStrategy {
        match self {
            StrategyTag::Libjxl => EncoderStrategy::Libjxl,
            StrategyTag::Zenjxl => EncoderStrategy::Zenjxl,
        }
    }
}

// ── Manifest parsing ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct ImageEntry {
    name: String,
    class: String,
    path: PathBuf,
}

fn load_manifest(path: &Path, corpus_dir: &Path) -> std::io::Result<Vec<ImageEntry>> {
    let s = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 3 {
            eprintln!("manifest: skipping malformed line: {line}");
            continue;
        }
        let rel = cols[0];
        let class = cols[1];
        let name = cols[2];
        let full = corpus_dir.join(rel);
        if !full.exists() {
            eprintln!("manifest: missing image {} ({})", name, full.display());
            continue;
        }
        out.push(ImageEntry {
            name: name.to_string(),
            class: class.to_string(),
            path: full,
        });
    }
    Ok(out)
}

// ── Row schema ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct CellResult {
    image: String,
    class: String,
    width: u32,
    height: u32,
    strategy: StrategyTag,
    effort: u8,
    distance: f32,
    ours_bytes: u64,
    cjxl_bytes: u64,
    ours_ssim2: f64,
    cjxl_ssim2: f64,
    ours_bfly: f64,
    cjxl_bfly: f64,
    ours_ms: f64,
    cjxl_ms: f64,
    status: &'static str,
    commit: String,
    reproducer: String,
}

impl CellResult {
    fn delta_bytes_pct(&self) -> f64 {
        if self.cjxl_bytes == 0 {
            0.0
        } else {
            (self.ours_bytes as f64 - self.cjxl_bytes as f64) / self.cjxl_bytes as f64 * 100.0
        }
    }
    fn delta_ssim2(&self) -> f64 {
        self.ours_ssim2 - self.cjxl_ssim2
    }
    fn delta_bfly_pct(&self) -> f64 {
        if self.cjxl_bfly.abs() < 1e-9 {
            0.0
        } else {
            (self.ours_bfly - self.cjxl_bfly) / self.cjxl_bfly * 100.0
        }
    }
    fn delta_ms_pct(&self) -> f64 {
        if self.cjxl_ms.abs() < 1e-6 {
            0.0
        } else {
            (self.ours_ms - self.cjxl_ms) / self.cjxl_ms * 100.0
        }
    }
}

fn row_header() -> &'static str {
    "image\tclass\twidth\theight\tstrategy\teffort\tdistance\t\
     ours_bytes\tcjxl_bytes\t\
     ours_ssim2\tcjxl_ssim2\t\
     ours_bfly\tcjxl_bfly\t\
     ours_ms\tcjxl_ms\t\
     delta_bytes_pct\tdelta_ssim2\tdelta_bfly_pct\tdelta_ms_pct\t\
     status\tcommit\treproducer"
}

fn row_tsv(r: &CellResult) -> String {
    format!(
        "{image}\t{class}\t{w}\t{h}\t{strat}\t{e}\t{d:.4}\t\
         {ob}\t{cb}\t\
         {oss:.4}\t{css:.4}\t\
         {obf:.6}\t{cbf:.6}\t\
         {oms:.3}\t{cms:.3}\t\
         {dbp:.3}\t{dss:.4}\t{dbfp:.3}\t{dmp:.3}\t\
         {st}\t{commit}\t{repro}",
        image = r.image,
        class = r.class,
        w = r.width,
        h = r.height,
        strat = r.strategy.as_str(),
        e = r.effort,
        d = r.distance,
        ob = r.ours_bytes,
        cb = r.cjxl_bytes,
        oss = r.ours_ssim2,
        css = r.cjxl_ssim2,
        obf = r.ours_bfly,
        cbf = r.cjxl_bfly,
        oms = r.ours_ms,
        cms = r.cjxl_ms,
        dbp = r.delta_bytes_pct(),
        dss = r.delta_ssim2(),
        dbfp = r.delta_bfly_pct(),
        dmp = r.delta_ms_pct(),
        st = r.status,
        commit = r.commit,
        repro = r.reproducer,
    )
}

// ── Atomic ledger I/O ───────────────────────────────────────────────────────

fn write_atomic(path: &Path, contents: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!("tsv.tmp.{}", std::process::id()));
    {
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp)?;
        f.write_all(contents.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
struct CellKey {
    image: String,
    strategy: StrategyTag,
    effort: u8,
    distance_x10000: i32,
}

fn cell_key(image: &str, strategy: StrategyTag, effort: u8, distance: f32) -> CellKey {
    CellKey {
        image: image.to_string(),
        strategy,
        effort,
        distance_x10000: (distance * 10000.0).round() as i32,
    }
}

/// Load existing per-strategy TSV (if present). Header is regenerated on
/// write so column reordering between runs is OK.
fn read_existing(path: &Path) -> HashMap<CellKey, String> {
    let mut map: HashMap<CellKey, String> = HashMap::new();
    let s = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return map,
    };
    for (i, line) in s.lines().enumerate() {
        if i == 0 {
            continue; // header
        }
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 7 {
            continue;
        }
        let image = cols[0].to_string();
        let strategy = match StrategyTag::parse(cols[4]) {
            Some(s) => s,
            None => continue,
        };
        let effort: u8 = cols[5].parse().unwrap_or(0);
        let distance: f32 = cols[6].parse().unwrap_or(0.0);
        map.insert(
            cell_key(&image, strategy, effort, distance),
            line.to_string(),
        );
    }
    map
}

fn write_ledger(path: &Path, rows: &HashMap<CellKey, String>) -> std::io::Result<()> {
    let mut keys: Vec<&CellKey> = rows.keys().collect();
    keys.sort_by(|a, b| {
        a.image
            .cmp(&b.image)
            .then(a.effort.cmp(&b.effort))
            .then(a.distance_x10000.cmp(&b.distance_x10000))
            .then(a.strategy.as_str().cmp(b.strategy.as_str()))
    });
    let mut s = String::with_capacity(rows.len() * 320 + 256);
    s.push_str(row_header());
    s.push('\n');
    for k in keys {
        s.push_str(&rows[k]);
        s.push('\n');
    }
    write_atomic(path, &s)
}

// ── Arg parsing ─────────────────────────────────────────────────────────────

struct Args {
    output_prefix: PathBuf,
    output_meta: PathBuf,
    corpus_dir: PathBuf,
    manifest: PathBuf,
    threads: usize,
    distance_min: f32,
    distance_max: f32,
    distance_step: f32,
    efforts: Vec<u8>,
    strategies: Vec<StrategyTag>,
    image_filter: Option<Vec<String>>,
    time_iters: u32,
    smoke: bool,
}

fn parse_args() -> Args {
    let mut output_prefix: Option<PathBuf> = None;
    let mut corpus_dir: Option<PathBuf> = None;
    let mut manifest: Option<PathBuf> = None;
    let mut threads: usize = 0;
    let mut distance_min = DEFAULT_DISTANCE_MIN;
    let mut distance_max = DEFAULT_DISTANCE_MAX;
    let mut distance_step = DEFAULT_DISTANCE_STEP;
    let mut efforts: Vec<u8> = Vec::new();
    let mut strategies: Vec<StrategyTag> = Vec::new();
    let mut image_filter: Option<Vec<String>> = None;
    let mut time_iters: u32 = 1;
    let mut smoke = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--output-prefix" => {
                output_prefix = Some(PathBuf::from(it.next().expect("--output-prefix VALUE")))
            }
            "--corpus-dir" => {
                corpus_dir = Some(PathBuf::from(it.next().expect("--corpus-dir VALUE")))
            }
            "--corpus-manifest" => {
                manifest = Some(PathBuf::from(it.next().expect("--corpus-manifest VALUE")))
            }
            "--threads" => {
                threads = it
                    .next()
                    .expect("--threads VALUE")
                    .parse()
                    .expect("threads must be uint")
            }
            "--distance-min" => {
                distance_min = it
                    .next()
                    .expect("--distance-min VALUE")
                    .parse()
                    .expect("float")
            }
            "--distance-max" => {
                distance_max = it
                    .next()
                    .expect("--distance-max VALUE")
                    .parse()
                    .expect("float")
            }
            "--distance-step" => {
                distance_step = it
                    .next()
                    .expect("--distance-step VALUE")
                    .parse()
                    .expect("float")
            }
            "--efforts" => {
                efforts = it
                    .next()
                    .expect("--efforts LIST")
                    .split(',')
                    .map(|s| s.parse().expect("u8"))
                    .collect()
            }
            "--effort" => {
                let v: u8 = it.next().expect("--effort VALUE").parse().expect("u8");
                efforts = vec![v];
            }
            "--strategies" => {
                strategies = it
                    .next()
                    .expect("--strategies LIST")
                    .split(',')
                    .filter_map(StrategyTag::parse)
                    .collect();
            }
            "--strategy" => {
                let v = it.next().expect("--strategy VALUE");
                strategies = vec![StrategyTag::parse(&v).expect("invalid strategy")];
            }
            "--images" => {
                image_filter = Some(
                    it.next()
                        .expect("--images LIST")
                        .split(',')
                        .map(|s| s.to_string())
                        .collect(),
                )
            }
            "--image" => {
                let v = it.next().expect("--image VALUE").to_string();
                image_filter = Some(vec![v]);
            }
            "--time-iters" => {
                time_iters = it.next().expect("--time-iters VALUE").parse().expect("u32")
            }
            "--smoke" => smoke = true,
            other => panic!("unknown arg: {other}"),
        }
    }
    if efforts.is_empty() {
        efforts = DEFAULT_EFFORTS.to_vec();
    }
    if strategies.is_empty() {
        strategies = vec![StrategyTag::Zenjxl, StrategyTag::Libjxl];
    }
    if smoke {
        image_filter = Some(vec!["1025469".into(), "terminal".into()]);
        distance_min = 0.5;
        distance_max = 2.0;
        distance_step = 0.5;
        efforts = vec![7];
    }
    let corpus_dir = corpus_dir.unwrap_or_else(default_corpus_dir);
    let manifest =
        manifest.unwrap_or_else(|| PathBuf::from("benchmarks/corpora/w44_170_varied_corpus.tsv"));
    let date = chrono_today();
    let output_prefix = output_prefix.unwrap_or_else(|| PathBuf::from("benchmarks/cjxl_step025"));
    let output_meta = PathBuf::from(format!("{}_{}.meta", output_prefix.display(), date));
    Args {
        output_prefix,
        output_meta,
        corpus_dir,
        manifest,
        threads,
        distance_min,
        distance_max,
        distance_step,
        efforts,
        strategies,
        image_filter,
        time_iters,
        smoke,
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

fn build_distance_grid(min: f32, max: f32, step: f32) -> Vec<f32> {
    let mut out = Vec::new();
    let mut x = min;
    // Inclusive on max with floating-point tolerance.
    while x <= max + step * 0.5 {
        out.push((x * 10000.0).round() / 10000.0);
        x += step;
    }
    out
}

// ── Per-image precompute (shared across strategies and cells) ───────────────

struct PreparedImage {
    entry: ImageEntry,
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    linear: Img<Vec<RGB<f32>>>,
    srgb: Img<Vec<[u8; 3]>>,
}

fn prepare_image(entry: ImageEntry) -> Option<PreparedImage> {
    let (rgb, w, h) = load_png(&entry.path)?;
    let linear = rgb_to_linear_img(&rgb, w, h);
    let srgb = rgb_to_srgb_arr3(&rgb, w, h);
    Some(PreparedImage {
        entry,
        rgb,
        width: w,
        height: h,
        linear,
        srgb,
    })
}

// ── cjxl reference cache (per-image, per-(effort, distance)) ────────────────

#[derive(Clone)]
struct CjxlResult {
    bytes: u64,
    ssim2: f64,
    bfly: f64,
    ms: f64,
}

type CjxlCache = HashMap<(String, u8, i32), CjxlResult>;

fn get_or_compute_cjxl(
    cache: &Mutex<CjxlCache>,
    img: &PreparedImage,
    effort: u8,
    distance: f32,
    work_dir: &Path,
) -> Option<CjxlResult> {
    let key = (
        img.entry.name.clone(),
        effort,
        (distance * 10000.0).round() as i32,
    );
    if let Some(r) = cache.lock().unwrap().get(&key) {
        return Some(r.clone());
    }
    let (bytes, ms) = encode_cjxl(&img.entry.path, distance, effort, work_dir, &img.entry.name)?;
    let (bfly, ssim2) = score_jxl(&bytes, &img.linear, &img.srgb, img.width, img.height)?;
    let result = CjxlResult {
        bytes: bytes.len() as u64,
        ssim2,
        bfly,
        ms,
    };
    cache.lock().unwrap().insert(key, result.clone());
    Some(result)
}

// ── Per-cell driver ─────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_cell(
    img: &PreparedImage,
    strategy: StrategyTag,
    effort: u8,
    distance: f32,
    cjxl: &CjxlResult,
    time_iters: u32,
    commit: &str,
) -> CellResult {
    let reproducer = format!(
        "cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' --example w44_170_cjxl_step025_sweep -- --image {} --effort {} --distance-min {} --distance-max {} --distance-step {} --strategy {} --corpus-manifest benchmarks/corpora/w44_170_varied_corpus.tsv",
        img.entry.name,
        effort,
        distance,
        distance,
        distance,
        strategy.as_str(),
    );
    let strategy_enum = strategy.to_encoder_strategy();
    let ours = encode_ours(
        &img.rgb,
        img.width,
        img.height,
        distance,
        effort,
        strategy_enum,
        time_iters,
    );
    let (ours_bytes_vec, ours_ms) = match ours {
        Some(x) => x,
        None => {
            return failed_cell(img, strategy, effort, distance, cjxl, commit, &reproducer);
        }
    };
    let scored = score_jxl(
        &ours_bytes_vec,
        &img.linear,
        &img.srgb,
        img.width,
        img.height,
    );
    let (ours_bfly, ours_ssim2) = match scored {
        Some(x) => x,
        None => {
            return failed_cell(img, strategy, effort, distance, cjxl, commit, &reproducer);
        }
    };
    CellResult {
        image: img.entry.name.clone(),
        class: img.entry.class.clone(),
        width: img.width,
        height: img.height,
        strategy,
        effort,
        distance,
        ours_bytes: ours_bytes_vec.len() as u64,
        cjxl_bytes: cjxl.bytes,
        ours_ssim2,
        cjxl_ssim2: cjxl.ssim2,
        ours_bfly,
        cjxl_bfly: cjxl.bfly,
        ours_ms,
        cjxl_ms: cjxl.ms,
        status: "OK",
        commit: commit.to_string(),
        reproducer,
    }
}

fn failed_cell(
    img: &PreparedImage,
    strategy: StrategyTag,
    effort: u8,
    distance: f32,
    cjxl: &CjxlResult,
    commit: &str,
    reproducer: &str,
) -> CellResult {
    CellResult {
        image: img.entry.name.clone(),
        class: img.entry.class.clone(),
        width: img.width,
        height: img.height,
        strategy,
        effort,
        distance,
        ours_bytes: 0,
        cjxl_bytes: cjxl.bytes,
        ours_ssim2: 0.0,
        cjxl_ssim2: cjxl.ssim2,
        ours_bfly: 0.0,
        cjxl_bfly: cjxl.bfly,
        ours_ms: 0.0,
        cjxl_ms: cjxl.ms,
        status: "FAILED",
        commit: commit.to_string(),
        reproducer: reproducer.to_string(),
    }
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

    // Load corpus.
    let mut images = match load_manifest(&args.manifest, &args.corpus_dir) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load manifest {}: {}", args.manifest.display(), e);
            std::process::exit(2);
        }
    };
    if let Some(filt) = &args.image_filter {
        images.retain(|i| filt.contains(&i.name));
    }
    if images.is_empty() {
        eprintln!("No images matched manifest+filter.");
        std::process::exit(2);
    }
    eprintln!(
        "Loaded {} images from {}",
        images.len(),
        args.manifest.display()
    );

    let distances = build_distance_grid(args.distance_min, args.distance_max, args.distance_step);
    let efforts = args.efforts.clone();
    let strategies = args.strategies.clone();

    let work_dir = std::env::temp_dir().join(format!("w44_170_cjxl_sweep_{}", std::process::id()));
    std::fs::create_dir_all(&work_dir).expect("create workdir");

    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            Command::new("jj")
                .args(["log", "-r", "@", "-T", "commit_id", "--no-graph"])
                .output()
                .ok()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .filter(|s| !s.is_empty())
        })
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

    let host = Command::new("hostname")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());

    let date = chrono_today();

    // Per-strategy output paths.
    let strategy_paths: HashMap<StrategyTag, PathBuf> = strategies
        .iter()
        .map(|s| {
            (
                *s,
                PathBuf::from(format!(
                    "{}_{}_{}.tsv",
                    args.output_prefix.display(),
                    s.as_str(),
                    date
                )),
            )
        })
        .collect();

    // Load existing rows per strategy (resume support).
    let mut existing_per_strategy: HashMap<StrategyTag, HashMap<CellKey, String>> = HashMap::new();
    for (s, p) in &strategy_paths {
        existing_per_strategy.insert(*s, read_existing(p));
    }

    // Prepare images (load + linear + sRGB precompute).
    eprintln!(
        "Preparing {} images (load+linear+sRGB precompute)...",
        images.len()
    );
    let prepared: Vec<PreparedImage> = images
        .into_iter()
        .filter_map(|e| {
            let p = prepare_image(e);
            if p.is_none() {
                eprintln!("  warn: failed to load image (skipped)");
            }
            p
        })
        .collect();

    // Build cell list (skip already-done cells per strategy).
    let mut cells: Vec<(usize, StrategyTag, u8, f32)> = Vec::new();
    for (img_idx, img) in prepared.iter().enumerate() {
        for &e in &efforts {
            for &d in &distances {
                for &s in &strategies {
                    let key = cell_key(&img.entry.name, s, e, d);
                    if existing_per_strategy[&s].contains_key(&key) {
                        continue;
                    }
                    cells.push((img_idx, s, e, d));
                }
            }
        }
    }
    let total_cells = cells.len();
    eprintln!(
        "Sweep plan: {} images × {} efforts × {} distances × {} strategies = {} requested cells; {} not yet done.",
        prepared.len(),
        efforts.len(),
        distances.len(),
        strategies.len(),
        prepared.len() * efforts.len() * distances.len() * strategies.len(),
        total_cells,
    );
    eprintln!("  distance grid: {:?}", distances);
    eprintln!("  efforts:       {:?}", efforts);
    eprintln!(
        "  strategies:    {:?}",
        strategies.iter().map(|s| s.as_str()).collect::<Vec<_>>()
    );
    eprintln!("  commit:        {}", commit);
    eprintln!("  cjxl_version:  {}", cjxl_ver);
    eprintln!("  host:          {}", host);
    eprintln!("  date:          {}", date);
    eprintln!("  output prefix: {}", args.output_prefix.display());
    for (s, p) in &strategy_paths {
        eprintln!("    {} -> {}", s.as_str(), p.display());
    }
    eprintln!("  meta:          {}", args.output_meta.display());

    if total_cells == 0 {
        eprintln!("All cells already done — nothing to do.");
        write_meta(
            &args,
            &commit,
            &cjxl_ver,
            &host,
            &date,
            &strategy_paths,
            &distances,
            &efforts,
            &strategies,
            0.0,
            0,
        );
        return;
    }

    let cjxl_cache: Mutex<CjxlCache> = Mutex::new(HashMap::new());
    let rows_per_strategy: Mutex<HashMap<StrategyTag, HashMap<CellKey, String>>> =
        Mutex::new(existing_per_strategy);
    let completed = std::sync::atomic::AtomicUsize::new(0);
    let failed = std::sync::atomic::AtomicUsize::new(0);

    let started = Instant::now();

    cells.par_iter().for_each(|(img_idx, strategy, effort, distance)| {
        let img = &prepared[*img_idx];

        // cjxl reference (shared across strategies, cached).
        let cjxl = match get_or_compute_cjxl(&cjxl_cache, img, *effort, *distance, &work_dir) {
            Some(r) => r,
            None => {
                eprintln!(
                    "  ! cjxl failed: {} e{} d{:.2}",
                    img.entry.name, effort, distance
                );
                failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return;
            }
        };

        let cell = run_cell(
            img,
            *strategy,
            *effort,
            *distance,
            &cjxl,
            args.time_iters,
            &commit,
        );
        let line = row_tsv(&cell);
        let key = cell_key(&cell.image, cell.strategy, cell.effort, cell.distance);

        if cell.status != "OK" {
            failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        let mut guard = rows_per_strategy.lock().unwrap();
        let entry = guard.entry(cell.strategy).or_default();
        entry.insert(key, line.clone());

        let n = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

        // Flush each strategy's ledger every 16 cells (or at the very end).
        if n.is_multiple_of(16) || n == total_cells {
            for (s, p) in &strategy_paths {
                if let Some(rows) = guard.get(s)
                    && let Err(e) = write_ledger(p, rows)
                {
                    eprintln!("  ! flush failed for {}: {}", s.as_str(), e);
                }
            }
            eprintln!(
                "  [{}/{} {:.1}%] {} {} e{} d{:.2} ours_bytes={} cjxl_bytes={} Δbytes={:+.2}% Δss2={:+.3} Δms={:+.1}% status={}",
                n,
                total_cells,
                100.0 * n as f64 / total_cells as f64,
                cell.image,
                cell.strategy.as_str(),
                cell.effort,
                cell.distance,
                cell.ours_bytes,
                cell.cjxl_bytes,
                cell.delta_bytes_pct(),
                cell.delta_ssim2(),
                cell.delta_ms_pct(),
                cell.status,
            );
        }
    });

    // Final flush.
    {
        let guard = rows_per_strategy.lock().unwrap();
        for (s, p) in &strategy_paths {
            if let Some(rows) = guard.get(s)
                && let Err(e) = write_ledger(p, rows)
            {
                eprintln!("  ! final flush failed for {}: {}", s.as_str(), e);
            }
        }
    }

    let wall = started.elapsed().as_secs_f64();
    let n_failed = failed.load(std::sync::atomic::Ordering::Relaxed);
    write_meta(
        &args,
        &commit,
        &cjxl_ver,
        &host,
        &date,
        &strategy_paths,
        &distances,
        &efforts,
        &strategies,
        wall,
        n_failed,
    );
    let _ = std::fs::remove_dir_all(&work_dir);

    eprintln!(
        "\nDone in {:.1}s ({:.1}min). Cells: {} (failed: {}).",
        wall,
        wall / 60.0,
        total_cells,
        n_failed,
    );
    eprintln!("Meta: {}", args.output_meta.display());
    for (s, p) in &strategy_paths {
        eprintln!("  {}: {}", s.as_str(), p.display());
    }
}

#[allow(clippy::too_many_arguments)]
fn write_meta(
    args: &Args,
    commit: &str,
    cjxl_ver: &str,
    host: &str,
    date: &str,
    strategy_paths: &HashMap<StrategyTag, PathBuf>,
    distances: &[f32],
    efforts: &[u8],
    strategies: &[StrategyTag],
    wall: f64,
    n_failed: usize,
) {
    let mut paths_str = String::new();
    for (s, p) in strategy_paths {
        paths_str.push_str(&format!("  {}: {}\n", s.as_str(), p.display()));
    }
    let meta = format!(
        "# W44-170 comprehensive cjxl-parity sweep\n\
         date: {date}\n\
         commit: {commit}\n\
         host: {host}\n\
         cjxl_version: {cjxl_ver}\n\
         manifest: {manifest}\n\
         corpus_dir: {corpus_dir}\n\
         strategies: {strategies:?}\n\
         distance_grid: {distances:?}\n\
         efforts: {efforts:?}\n\
         time_iters: {time_iters}\n\
         smoke: {smoke}\n\
         wall_seconds: {wall:.1}\n\
         wall_minutes: {wall_minutes:.1}\n\
         cells_failed: {n_failed}\n\
         decoder: jxl-oxide (srgb_linear)\n\
         metrics: butteraugli_linear (default params); fast_ssim2 (sRGB u8 from linear)\n\
         tsv_outputs:\n{paths}\
         reproducer (full sweep):\n  \
         cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' \
         --example w44_170_cjxl_step025_sweep -- \
         --corpus-manifest {manifest} \
         --output-prefix {prefix} \
         --strategies zenjxl,libjxl\n",
        date = date,
        commit = commit,
        host = host,
        cjxl_ver = cjxl_ver,
        manifest = args.manifest.display(),
        corpus_dir = args.corpus_dir.display(),
        strategies = strategies.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        distances = distances,
        efforts = efforts,
        time_iters = args.time_iters,
        smoke = args.smoke,
        wall = wall,
        wall_minutes = wall / 60.0,
        n_failed = n_failed,
        paths = paths_str,
        prefix = args.output_prefix.display(),
    );
    let _ = write_atomic(&args.output_meta, &meta);
}
