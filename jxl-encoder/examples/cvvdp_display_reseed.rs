// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.
//
//! Phase 2 display-config backfill (RFC #4 + chunk spec 2026-05-26):
//! re-score the per-cell tracking corpus under EACH `DisplayConfig`
//! so the heuristic `WebSdr80 × {1.04, 1.12}` multipliers in
//! `vardct/cvvdp_targets.rs` can be replaced with measured per-distance
//! tables.
//!
//! ## Inputs
//!
//! Reads `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv` to get
//! the universe of (image_path, distance, effort) cells already in the
//! Phase 1 baseline (backend=B rows, 1,134 cells). For each unique cell:
//!
//! 1. Resolve the image path via the same corpus directory discovery
//!    (CID22-512/validation, gb82-sc, gb82-lossless extras).
//! 2. Encode ONCE with default `LossyConfig::new(distance).with_effort(effort)`
//!    (= the `B` backend → Zenjxl + butteraugli CPU buttloop).
//! 3. Decode via jxl-oxide to linear-RGB f32 planes.
//! 4. Score the (reference linear, decoded linear) pair under EACH of
//!    `DisplayConfig::{WebSdr80, Phone, Tv}` by constructing a fresh
//!    `cvvdp_gpu::CvvdpOpaque::new_with_geometry(.., display_model,
//!    display_geometry)` per (w, h, display) tuple.
//!
//! ## Outputs
//!
//! - `benchmarks/cvvdp_per_display_reseed_2026-05-26.tsv` (long format):
//!   `image_path	corpus	distance	effort	display_config	cvvdp_jod`
//!   3 rows per cell (one per display).
//! - `benchmarks/cvvdp_per_display_reseed_medians_2026-05-26.tsv` (wide):
//!   per-distance × per-display p25 / p50 / p75 of `cvvdp_jod`. Used to
//!   seed the new measured tables.
//!
//! ## Budget
//!
//! 1,134 cells × ~30 ms encode + ~3 × 100 ms scoring ≈ ~6.4 minutes on
//! lilith. Phase 1's `B` baseline encodes were 1.5-9 h because they
//! ALSO did 3-iter best-of-median encoding + GPU butteraugli scoring +
//! CPU butteraugli scoring. This harness only does ONE encode per cell
//! + 3 cvvdp scoring passes — much cheaper.
//!
//! ## Invocation
//!
//! ```sh
//! cargo run --release -p jxl-encoder \
//!     --features '__expert butteraugli-loop cvvdp-loop ssim2-loop parallel' \
//!     --example cvvdp_display_reseed -- \
//!     --tracking benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv \
//!     --output benchmarks/cvvdp_per_display_reseed_2026-05-26.tsv
//! ```
//!
//! Optional: `--max-cells N` to truncate; `--filter NAME` to limit to
//! one image basename substring. Resumable: re-running with the same
//! `--output` path skips cells already present.

use cvvdp_gpu::CvvdpOpaque;
use cvvdp_gpu::params::CvvdpParams;
use jxl_encoder::api::{DisplayConfig, LossyConfig, PixelLayout};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Cursor, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const CID22_VAL_DIR: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation";
const GB82_SC_DIR: &str = "/home/lilith/work/codec-corpus/gb82-sc";
const GB82_DIR: &str = "/home/lilith/work/codec-corpus/gb82";

#[derive(Clone)]
struct SourceImage {
    name: String,
    corpus: &'static str,
    path: PathBuf,
}

fn discover_corpus() -> BTreeMap<String, SourceImage> {
    let mut out = BTreeMap::new();

    if let Ok(rd) = std::fs::read_dir(CID22_VAL_DIR) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("png") {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            out.entry(name.clone()).or_insert(SourceImage {
                name,
                corpus: "CID22",
                path,
            });
        }
    }

    if let Ok(rd) = std::fs::read_dir(GB82_SC_DIR) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("png") {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            out.entry(name.clone()).or_insert(SourceImage {
                name,
                corpus: "GB82-SC",
                path,
            });
        }
    }

    let gb82 = Path::new(GB82_DIR);
    for name in &["baby-lossless.png", "bulb-lossless.png"] {
        let path = gb82.join(name);
        if path.is_file() {
            out.entry(name.to_string()).or_insert(SourceImage {
                name: name.to_string(),
                corpus: "W44-S1",
                path,
            });
        }
    }

    out
}

fn srgb_to_linear_u8(v: u8) -> f32 {
    let x = (v as f32) / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
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

/// Read existing (image, distance, effort, display) cells from the
/// output TSV so we can resume cleanly.
fn already_done(out_path: &Path) -> BTreeSet<(String, String, u8, String)> {
    let mut set = BTreeSet::new();
    let f = match std::fs::File::open(out_path) {
        Ok(f) => f,
        Err(_) => return set,
    };
    for (i, line) in BufReader::new(f).lines().enumerate() {
        if i == 0 {
            continue;
        }
        let Ok(line) = line else { continue };
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 5 {
            continue;
        }
        set.insert((
            cols[0].to_string(),
            cols[2].to_string(),
            cols[3].parse::<u8>().unwrap_or(0),
            cols[4].to_string(),
        ));
    }
    set
}

/// Read the universe of (image, distance, effort) cells from the
/// tracking TSV. Returns a deduplicated vector of cells keyed by
/// (image, distance, effort) where backend==B.
fn read_tracking_universe(tracking_path: &Path) -> Vec<(String, String, u8)> {
    let f = std::fs::File::open(tracking_path)
        .unwrap_or_else(|e| panic!("open tracking TSV {}: {e}", tracking_path.display()));
    let mut header: Option<Vec<String>> = None;
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let Ok(line) = line else { continue };
        let cols: Vec<&str> = line.split('\t').collect();
        if i == 0 {
            header = Some(cols.iter().map(|s| s.to_string()).collect());
            continue;
        }
        let h = header.as_ref().unwrap();
        let get = |name: &str| {
            h.iter()
                .position(|x| x == name)
                .and_then(|idx| cols.get(idx).copied())
        };
        let image = get("image").unwrap_or("").to_string();
        let backend = get("backend").unwrap_or("");
        let distance = get("distance").unwrap_or("").to_string();
        let effort: u8 = get("effort").unwrap_or("0").parse().unwrap_or(0);
        if backend != "B" {
            continue;
        }
        let key = (image.clone(), distance.clone(), effort);
        if seen.insert(key.clone()) {
            out.push((image, distance, effort));
        }
    }
    out
}

struct GpuPerDisplay {
    cvvdp: BTreeMap<String, CvvdpOpaque>, // key = display name
    last_w: u32,
    last_h: u32,
}

impl GpuPerDisplay {
    fn new() -> Self {
        Self {
            cvvdp: BTreeMap::new(),
            last_w: 0,
            last_h: 0,
        }
    }

    fn ensure(&mut self, w: u32, h: u32) {
        if w == self.last_w && h == self.last_h && !self.cvvdp.is_empty() {
            return;
        }
        self.cvvdp.clear();
        self.last_w = w;
        self.last_h = h;
        for &display in &[
            DisplayConfig::WebSdr80,
            DisplayConfig::Phone,
            DisplayConfig::Tv,
        ] {
            let params = CvvdpParams {
                display: display.display_model(),
                ..CvvdpParams::default()
            };
            let geometry = display.display_geometry();
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                CvvdpOpaque::new_with_geometry(
                    cvvdp_gpu::opaque::Backend::Cuda,
                    w,
                    h,
                    params,
                    geometry,
                )
            }));
            match res {
                Ok(Ok(c)) => {
                    self.cvvdp.insert(display_to_str(display).to_string(), c);
                }
                Ok(Err(e)) => {
                    eprintln!("WARN cvvdp-gpu init failed at {w}x{h} display={display:?}: {e:?}");
                }
                Err(_) => {
                    eprintln!("WARN cvvdp-gpu init panicked at {w}x{h} display={display:?}");
                }
            }
        }
    }

    fn score(
        &mut self,
        display: DisplayConfig,
        ref_rgb_u8: &[u8],
        dis_rgb_u8: &[u8],
    ) -> Option<f64> {
        let key = display_to_str(display);
        let c = self.cvvdp.get_mut(key)?;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            c.compute_srgb_u8(ref_rgb_u8, dis_rgb_u8)
        }));
        match res {
            Ok(Ok(s)) => Some(s.value),
            Ok(Err(e)) => {
                eprintln!("WARN compute_srgb_u8 err display={display:?}: {e:?}");
                None
            }
            Err(_) => {
                eprintln!("WARN compute_srgb_u8 panic display={display:?}");
                None
            }
        }
    }
}

fn display_to_str(d: DisplayConfig) -> &'static str {
    // `DisplayConfig` is non-exhaustive (#[non_exhaustive] in api.rs)
    // so we need an explicit fallback arm — newly added variants would
    // then surface as "unknown" in the bench rather than failing to
    // build, but the medians pipeline only aggregates the 3 known ones.
    match d {
        DisplayConfig::WebSdr80 => "WebSdr80",
        DisplayConfig::Phone => "Phone",
        DisplayConfig::Tv => "Tv",
        _ => "unknown",
    }
}

fn corpus_of(name: &str, corpus_map: &BTreeMap<String, SourceImage>) -> &'static str {
    corpus_map.get(name).map(|s| s.corpus).unwrap_or("unknown")
}

fn parse_distance(s: &str) -> f32 {
    s.parse::<f32>().unwrap_or(1.0)
}

struct Args {
    output: PathBuf,
    medians_output: PathBuf,
    tracking: PathBuf,
    filter: Option<String>,
    max_cells: Option<usize>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut output = PathBuf::from("benchmarks/cvvdp_per_display_reseed_2026-05-26.tsv");
    let mut medians_output =
        PathBuf::from("benchmarks/cvvdp_per_display_reseed_medians_2026-05-26.tsv");
    let mut tracking = PathBuf::from("benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv");
    let mut filter = None;
    let mut max_cells = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--output" => {
                output = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--medians-output" => {
                medians_output = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--tracking" => {
                tracking = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--filter" => {
                filter = Some(argv[i + 1].clone());
                i += 2;
            }
            "--max-cells" => {
                max_cells = Some(argv[i + 1].parse().expect("parse --max-cells"));
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }
    Args {
        output,
        medians_output,
        tracking,
        filter,
        max_cells,
    }
}

fn ensure_output_header(p: &Path) -> std::io::Result<()> {
    if p.exists() {
        return Ok(());
    }
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f = OpenOptions::new().create_new(true).write(true).open(p)?;
    writeln!(
        f,
        "image\tcorpus\tdistance\teffort\tdisplay_config\tcvvdp_jod"
    )?;
    Ok(())
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = p * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let t = rank - rank.floor();
    sorted[lo] * (1.0 - t) + sorted[hi] * t
}

fn write_medians_tsv(
    out_long: &Path,
    out_medians: &Path,
) -> std::io::Result<BTreeMap<(String, String), (f64, f64, f64)>> {
    // Read everything from the long TSV. distance + display → list of jod.
    let mut grouped: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    let f = std::fs::File::open(out_long)?;
    let mut header: Option<Vec<String>> = None;
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let Ok(line) = line else { continue };
        let cols: Vec<&str> = line.split('\t').collect();
        if i == 0 {
            header = Some(cols.iter().map(|s| s.to_string()).collect());
            continue;
        }
        let h = header.as_ref().unwrap();
        let get = |name: &str| {
            h.iter()
                .position(|x| x == name)
                .and_then(|idx| cols.get(idx).copied())
        };
        let distance = get("distance").unwrap_or("").to_string();
        let display = get("display_config").unwrap_or("").to_string();
        let jod_str = get("cvvdp_jod").unwrap_or("NA");
        if jod_str == "NA" {
            continue;
        }
        let Ok(jod) = jod_str.parse::<f64>() else {
            continue;
        };
        if !jod.is_finite() {
            continue;
        }
        grouped.entry((distance, display)).or_default().push(jod);
    }

    // Compute p25/p50/p75 per (distance, display).
    let mut quartiles: BTreeMap<(String, String), (f64, f64, f64)> = BTreeMap::new();
    for ((d, disp), values) in &mut grouped {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p25 = percentile_sorted(values, 0.25);
        let p50 = percentile_sorted(values, 0.50);
        let p75 = percentile_sorted(values, 0.75);
        quartiles.insert((d.clone(), disp.clone()), (p25, p50, p75));
    }

    if let Some(parent) = out_medians.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f = std::fs::File::create(out_medians)?;
    writeln!(
        f,
        "distance\tWebSdr80_p25\tWebSdr80_p50\tWebSdr80_p75\tPhone_p25\tPhone_p50\tPhone_p75\tTv_p25\tTv_p50\tTv_p75\tn_WebSdr80\tn_Phone\tn_Tv"
    )?;
    let distances = &["0.5", "1.0", "1.5", "2.0", "3.0", "4.0", "5.0"];
    for d in distances {
        let q_web = quartiles.get(&(d.to_string(), "WebSdr80".to_string()));
        let q_phone = quartiles.get(&(d.to_string(), "Phone".to_string()));
        let q_tv = quartiles.get(&(d.to_string(), "Tv".to_string()));
        let n_web = grouped
            .get(&(d.to_string(), "WebSdr80".to_string()))
            .map(|v| v.len())
            .unwrap_or(0);
        let n_phone = grouped
            .get(&(d.to_string(), "Phone".to_string()))
            .map(|v| v.len())
            .unwrap_or(0);
        let n_tv = grouped
            .get(&(d.to_string(), "Tv".to_string()))
            .map(|v| v.len())
            .unwrap_or(0);
        let fmt_q = |q: Option<&(f64, f64, f64)>| -> String {
            match q {
                Some((a, b, c)) => format!("{:.6}\t{:.6}\t{:.6}", a, b, c),
                None => "NA\tNA\tNA".to_string(),
            }
        };
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            d,
            fmt_q(q_web),
            fmt_q(q_phone),
            fmt_q(q_tv),
            n_web,
            n_phone,
            n_tv
        )?;
    }
    Ok(quartiles)
}

fn main() {
    let args = parse_args();
    eprintln!(
        "[cvvdp_display_reseed] tracking={} output={} medians={}",
        args.tracking.display(),
        args.output.display(),
        args.medians_output.display()
    );

    let corpus = discover_corpus();
    eprintln!("[cvvdp_display_reseed] {} images discovered", corpus.len());

    let universe = read_tracking_universe(&args.tracking);
    eprintln!(
        "[cvvdp_display_reseed] {} unique cells in tracking TSV (backend=B)",
        universe.len()
    );

    ensure_output_header(&args.output).expect("output header");
    let done = already_done(&args.output);
    eprintln!(
        "[cvvdp_display_reseed] {} (image, distance, effort, display) rows already in output",
        done.len()
    );

    let mut gpu = GpuPerDisplay::new();
    let started = Instant::now();
    let mut emitted = 0usize;
    let mut idx_global = 0usize;
    let total_rows = universe.len() * 3;
    let mut last_log = Instant::now();

    'outer: for (image_name, dist_str, effort) in &universe {
        idx_global += 3;
        if let Some(f) = &args.filter {
            if !image_name.contains(f) {
                continue;
            }
        }
        let Some(src) = corpus.get(image_name).cloned() else {
            eprintln!("WARN: image {image_name} not found in corpus; skip");
            continue;
        };
        // All 3 display rows for this cell? skip
        let displays = ["WebSdr80", "Phone", "Tv"];
        let all_done = displays.iter().all(|d| {
            done.contains(&(image_name.clone(), dist_str.clone(), *effort, d.to_string()))
        });
        if all_done {
            continue;
        }

        let img = match image::open(&src.path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("ERR open {}: {e}", src.path.display());
                continue;
            }
        };
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let src_pixels_u8 = rgb.as_raw().clone();
        let distance = parse_distance(dist_str);

        // Encode once (default Zenjxl + butteraugli; backend=B path).
        let cfg = LossyConfig::new(distance).with_effort(*effort);
        let jxl_bytes = match cfg.encode(&src_pixels_u8, w, h, PixelLayout::Rgb8) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "ERR encode {} d={} e={}: {e:?}",
                    image_name, dist_str, effort
                );
                continue;
            }
        };
        let Some((dw, dh, decoded_linear)) = decode_jxl_linear(&jxl_bytes) else {
            eprintln!(
                "ERR decode {} d={} e={}: jxl-oxide failed",
                image_name, dist_str, effort
            );
            continue;
        };
        if dw as u32 != w || dh as u32 != h {
            eprintln!("ERR decoded dims {}x{} != {}x{}", dw, dh, w, h);
            continue;
        }
        let decoded_srgb: Vec<u8> = decoded_linear
            .iter()
            .map(|v| linear_to_srgb_u8(*v))
            .collect();

        gpu.ensure(w, h);

        let mut f = OpenOptions::new()
            .append(true)
            .open(&args.output)
            .expect("open output for append");

        for &display in &[
            DisplayConfig::WebSdr80,
            DisplayConfig::Phone,
            DisplayConfig::Tv,
        ] {
            let key = display_to_str(display);
            let cell_key = (
                image_name.clone(),
                dist_str.clone(),
                *effort,
                key.to_string(),
            );
            if done.contains(&cell_key) {
                continue;
            }
            let jod = gpu.score(display, &src_pixels_u8, &decoded_srgb);
            let jod_str = match jod {
                Some(v) if v.is_finite() => format!("{:.6}", v),
                _ => "NA".to_string(),
            };
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}",
                image_name, src.corpus, dist_str, effort, key, jod_str
            )
            .expect("write row");
            emitted += 1;
        }

        if last_log.elapsed().as_secs() >= 30 || emitted % 60 == 0 {
            let el = started.elapsed().as_secs_f64();
            let rate = (emitted as f64) / el.max(0.001);
            let remaining = total_rows.saturating_sub(idx_global);
            let eta = remaining as f64 / rate.max(0.001);
            eprintln!(
                "[{} / {}] emitted {} rows, last={} d={} e={}, elapsed={:.0}s rate={:.1} rows/s eta={:.0}s",
                idx_global, total_rows, emitted, image_name, dist_str, effort, el, rate, eta
            );
            last_log = Instant::now();
        }

        if let Some(cap) = args.max_cells {
            if emitted >= cap * 3 {
                eprintln!("[cvvdp_display_reseed] hit --max-cells={cap}, stopping");
                break 'outer;
            }
        }
    }

    eprintln!(
        "[cvvdp_display_reseed] DONE: emitted {} rows in {:.0}s",
        emitted,
        started.elapsed().as_secs_f64()
    );

    // Aggregate into medians TSV
    let _ = corpus_of("", &corpus); // suppress unused warning if guard above never fires
    match write_medians_tsv(&args.output, &args.medians_output) {
        Ok(qs) => {
            eprintln!(
                "[cvvdp_display_reseed] wrote {} ({} (distance, display) buckets aggregated)",
                args.medians_output.display(),
                qs.len()
            );
        }
        Err(e) => {
            eprintln!("ERR writing medians {}: {e}", args.medians_output.display());
        }
    }

    // Suppress unused-import warnings if features change later.
    let _ = srgb_to_linear_u8(0);
}
