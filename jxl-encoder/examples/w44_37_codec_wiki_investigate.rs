//! W44-37 codec_wiki +3-7% bytes wedge investigation (sibling workspace
//! `~/work/zen/jxl-encoder--w44-37-codec-wiki-wedge`).
//!
//! Forked from `f_d_wedge_investigate.rs`. Differences:
//!   - CELLS list points at codec_wiki OPEN cells from W44-36 ledger.
//!   - source dir changed from CID22/CID22-512/validation/ → gb82-sc/.
//!   - output TSV named codec_wiki_wedge_sections_2026-05-19.tsv.
//!
//! Per-section accounting (from W44-36 ledger): the +3-7% bytes wedge
//! is concentrated in hf_global (+37-63% wider than cjxl on the 5 OPEN
//! cells) plus a smaller lf_groups overspend; codec_wiki uses patches
//! at parity (W44-20 already ruled out the patches-coverage hypothesis).
//!
//! This harness re-runs the AC strategy distribution + hf_mul stats from
//! W44-34 on codec_wiki specifically, to confirm whether the wedge is the
//! same "ours under-picks DCT64" pattern (as on 1418519 photo) or a
//! different mechanism (screenshot-specific cost model / content class).
//!
//! For each (image, effort, distance) cell:
//!   1. Encode with ours and with cjxl (effort/distance matched).
//!   2. Run `jxl-inspect export-csv --per-block` on both .jxl files.
//!   3. Aggregate per-block CSV into:
//!      - DCT-strategy histogram (varblock counts + 8x8-block-area weighted)
//!      - hf_mul (per-block raw_quant) statistics: median, mean, min, max
//!   4. Run `jxl-inspect inspect` and parse quant params (global_scale, quant_lf)
//!   5. Emit one TSV row per cell with all stats for both encoders.
//!
//! The expected wedge signature (per W38-2 WF1 + W44-5 F-D):
//!   - We pick MORE DCT8 / fewer DCT32/DCT16 → AC overspend (each block ships
//!     more LF + ringing coefficients).
//!   - LF group bytes ↓ vs cjxl (LF coefficients are encoded inside LF groups;
//!     more 8x8 blocks → LF coefs interpreted as the DC plane of the smallest
//!     transform, so they end up in AC instead of LF; net = LF starves).
//!
//! Run:
//! ```bash
//! cargo run -p jxl-encoder --release --features 'butteraugli-loop ssim2-loop' \
//!   --example f_d_wedge_investigate -- \
//!   --output benchmarks/f_d_photo_high_d_wedge_2026-05-19.tsv
//! ```
//!
//! Read-only investigation. No source/src changes; subprocess-driven.

use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

// ── Cells under investigation (selected from W44-5 ledger) ──────────────────

/// (image_name, effort, distance) — codec_wiki OPEN cells from W44-36 ledger.
/// 5 remaining OPEN codec_wiki cells (post-W44-35) + 2 reference FIXED cells.
const CELLS: &[(&str, u8, f32)] = &[
    ("codec_wiki.png", 6, 4.0), // +3.20% bytes, +7.96% bfly, -4.92 ssim2
    ("codec_wiki.png", 7, 3.0), // +4.14% bytes, +13.56% bfly, -2.65 ssim2
    ("codec_wiki.png", 7, 4.0), // +7.34% bytes, +7.19% bfly, -4.52 ssim2
    ("codec_wiki.png", 7, 5.0), // +6.16% bytes, +4.11% bfly, -2.41 ssim2
    ("codec_wiki.png", 7, 6.0), // +6.67% bytes, +3.26% bfly, -1.24 ssim2
    // Reference cells (FIXED at d=1.0/2.0 — sanity check for parity)
    ("codec_wiki.png", 7, 1.0), // FIXED, expected near-parity
    ("codec_wiki.png", 7, 2.0), // FIXED
];

fn corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/lilith/work/codec-corpus"))
}

fn cjxl_bin() -> PathBuf {
    std::env::var("CJXL").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from("/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl")
    })
}

fn jxl_inspect_bin() -> PathBuf {
    std::env::var("JXL_INSPECT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/home/lilith/work/jxl-efforts/jxl-oxide/target/release/jxl-inspect")
        })
}

fn output_dir() -> PathBuf {
    PathBuf::from("benchmarks")
}

fn work_dir() -> PathBuf {
    PathBuf::from("/tmp/f_d_wedge_investigate")
}

// ── PNG loader ──────────────────────────────────────────────────────────────

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

// ── Encoders ────────────────────────────────────────────────────────────────

fn encode_ours(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> Option<(Vec<u8>, f64)> {
    let cfg = LossyConfig::new(distance).with_effort(effort).with_threads(1);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Some((bytes, ms))
}

fn encode_cjxl(src_png: &Path, distance: f32, effort: u8, work: &Path) -> Option<(Vec<u8>, f64)> {
    let out = work.join(format!(
        "cjxl_{}_{}_e{}_d{:.2}.jxl",
        std::process::id(),
        src_png.file_stem().unwrap().to_string_lossy(),
        effort,
        distance
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

// ── Per-block CSV via jxl-inspect subprocess ────────────────────────────────

#[derive(Default, Debug, Clone)]
struct BlockStats {
    total_varblocks: u64,
    total_8x8_area: u64,
    // dct_type → (varblock_count, 8x8_block_area_count)
    by_type: HashMap<String, (u64, u64)>,
    hf_mul_values: Vec<i32>, // one per varblock (i.e. raw_quant in libjxl land)
}

impl BlockStats {
    fn median_hf_mul(&self) -> f64 {
        let mut v = self.hf_mul_values.clone();
        if v.is_empty() {
            return 0.0;
        }
        v.sort_unstable();
        let n = v.len();
        if n % 2 == 0 {
            (v[n / 2 - 1] as f64 + v[n / 2] as f64) / 2.0
        } else {
            v[n / 2] as f64
        }
    }

    fn mean_hf_mul(&self) -> f64 {
        if self.hf_mul_values.is_empty() {
            return 0.0;
        }
        self.hf_mul_values.iter().map(|&x| x as f64).sum::<f64>()
            / self.hf_mul_values.len() as f64
    }

    fn mad_hf_mul(&self) -> f64 {
        // Median absolute deviation
        let median = self.median_hf_mul();
        let mut devs: Vec<f64> = self
            .hf_mul_values
            .iter()
            .map(|&x| (x as f64 - median).abs())
            .collect();
        if devs.is_empty() {
            return 0.0;
        }
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = devs.len();
        if n % 2 == 0 {
            (devs[n / 2 - 1] + devs[n / 2]) / 2.0
        } else {
            devs[n / 2]
        }
    }

    fn pct_by_count(&self, dct_type: &str) -> f64 {
        if self.total_varblocks == 0 {
            return 0.0;
        }
        let count = self.by_type.get(dct_type).map(|&(c, _)| c).unwrap_or(0);
        count as f64 / self.total_varblocks as f64 * 100.0
    }

    fn pct_by_area(&self, dct_type: &str) -> f64 {
        if self.total_8x8_area == 0 {
            return 0.0;
        }
        let area = self.by_type.get(dct_type).map(|&(_, a)| a).unwrap_or(0);
        area as f64 / self.total_8x8_area as f64 * 100.0
    }
}

fn extract_block_stats(jxl_bytes: &[u8], work: &Path) -> Option<BlockStats> {
    let stem = format!("inspect_{}_{}", std::process::id(), rand_id());
    let jxl_path = work.join(format!("{stem}.jxl"));
    let csv_path = work.join(format!("{stem}.csv"));
    fs::write(&jxl_path, jxl_bytes).ok()?;
    let status = Command::new(jxl_inspect_bin())
        .arg("export-csv")
        .args(["-o", csv_path.to_string_lossy().as_ref()])
        .arg("--per-block")
        .arg(&jxl_path)
        .output()
        .ok()?;
    if !status.status.success() {
        eprintln!(
            "jxl-inspect failed: {}",
            String::from_utf8_lossy(&status.stderr)
        );
        let _ = fs::remove_file(&jxl_path);
        return None;
    }
    let csv = fs::read_to_string(&csv_path).ok()?;
    let _ = fs::remove_file(&jxl_path);
    let _ = fs::remove_file(&csv_path);

    let mut stats = BlockStats::default();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            // Header row
            continue;
        }
        // Columns: frame_idx,lf_group_idx,block_x,block_y,dct_type,size_w,size_h,hf_mul
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 8 {
            continue;
        }
        let dct_type = cols[4].to_string();
        let size_w: u64 = cols[5].parse().unwrap_or(0);
        let size_h: u64 = cols[6].parse().unwrap_or(0);
        let hf_mul: i32 = cols[7].parse().unwrap_or(0);
        let area = size_w * size_h;
        stats.total_varblocks += 1;
        stats.total_8x8_area += area;
        let entry = stats.by_type.entry(dct_type).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += area;
        stats.hf_mul_values.push(hf_mul);
    }
    Some(stats)
}

fn rand_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{nanos:x}")
}

// ── Quantization params via jxl-inspect inspect ─────────────────────────────

#[derive(Debug, Clone, Default)]
struct QuantParams {
    global_scale: u32,
    quant_lf: u32,
}

fn extract_quant_params(jxl_bytes: &[u8], work: &Path) -> Option<QuantParams> {
    let stem = format!("qparams_{}_{}", std::process::id(), rand_id());
    let jxl_path = work.join(format!("{stem}.jxl"));
    let dir_path = work.join(format!("{stem}.dir"));
    fs::write(&jxl_path, jxl_bytes).ok()?;
    let status = Command::new(jxl_inspect_bin())
        .arg("inspect")
        .args(["-o", dir_path.to_string_lossy().as_ref()])
        .arg(&jxl_path)
        .output()
        .ok()?;
    if !status.status.success() {
        let _ = fs::remove_file(&jxl_path);
        return None;
    }
    // Read frame_0/lf_global.json which holds the Quantizer block.
    // Per inspect output: annotations field holds the parsed values.
    // We grep for "global_scale" and "quant_lf" in the JSON.
    // (Falls back gracefully if not found.)
    let lf_global_path = dir_path.join("segments/frame_0/lf_global.json");
    let mut qp = QuantParams::default();
    if let Ok(s) = fs::read_to_string(&lf_global_path) {
        qp.global_scale = parse_field(&s, "global_scale").unwrap_or(0);
        qp.quant_lf = parse_field(&s, "quant_lf").unwrap_or(0);
    }
    let _ = fs::remove_dir_all(&dir_path);
    let _ = fs::remove_file(&jxl_path);
    Some(qp)
}

fn parse_field(json: &str, field: &str) -> Option<u32> {
    // Crude scan: find "<field>" : NUMBER, ...
    let needle = format!("\"{field}\"");
    let idx = json.find(&needle)?;
    let rest = &json[idx + needle.len()..];
    let colon = rest.find(':')?;
    let after = &rest[colon + 1..];
    let trimmed = after.trim_start();
    let end = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    trimmed[..end].parse::<u32>().ok()
}

// ── Per-cell investigation ──────────────────────────────────────────────────

#[derive(Debug)]
struct CellResult {
    image: String,
    effort: u8,
    distance: f32,
    ours_bytes: u64,
    cjxl_bytes: u64,
    ours_ms: f64,
    cjxl_ms: f64,
    ours_stats: BlockStats,
    cjxl_stats: BlockStats,
    ours_qp: QuantParams,
    cjxl_qp: QuantParams,
}

fn investigate_cell(image: &str, effort: u8, distance: f32, work: &Path) -> Option<CellResult> {
    // codec_wiki.png lives in gb82-sc/, other screenshots too.
    let src = corpus_dir().join("gb82-sc").join(image);
    let (rgb, w, h) = load_png(&src)?;
    let (ours_bytes, ours_ms) = encode_ours(&rgb, w, h, distance, effort)?;
    let (cjxl_bytes, cjxl_ms) = encode_cjxl(&src, distance, effort, work)?;
    let ours_stats = extract_block_stats(&ours_bytes, work)?;
    let cjxl_stats = extract_block_stats(&cjxl_bytes, work)?;
    let ours_qp = extract_quant_params(&ours_bytes, work).unwrap_or_default();
    let cjxl_qp = extract_quant_params(&cjxl_bytes, work).unwrap_or_default();
    Some(CellResult {
        image: image.to_string(),
        effort,
        distance,
        ours_bytes: ours_bytes.len() as u64,
        cjxl_bytes: cjxl_bytes.len() as u64,
        ours_ms,
        cjxl_ms,
        ours_stats,
        cjxl_stats,
        ours_qp,
        cjxl_qp,
    })
}

// ── TSV emit ────────────────────────────────────────────────────────────────

const DCT_TYPES: &[&str] = &[
    "Dct8", "Dct16", "Dct32", "Dct64", "Dct16x8", "Dct8x16", "Dct32x16", "Dct16x32", "Dct32x64",
    "Dct64x32", "Dct4x8", "Dct8x4", "Dct4", "Dct2", "Hornuss", "Afv0", "Afv1", "Afv2", "Afv3",
];

fn tsv_header() -> String {
    let mut s = String::from(
        "image\teffort\tdistance\tencoder\tbytes\tencode_ms\ttotal_varblocks\ttotal_8x8_area\t\
         hf_mul_median\thf_mul_mean\thf_mul_mad\thf_mul_min\thf_mul_max\tglobal_scale\tquant_lf",
    );
    for dt in DCT_TYPES {
        s.push_str(&format!("\t{dt}_pct"));
    }
    for dt in DCT_TYPES {
        s.push_str(&format!("\t{dt}_area_pct"));
    }
    s.push('\n');
    s
}

fn emit_row(out: &mut String, image: &str, effort: u8, distance: f32, encoder: &str, bytes: u64, ms: f64, st: &BlockStats, qp: &QuantParams) {
    let hf_min = st.hf_mul_values.iter().copied().min().unwrap_or(0);
    let hf_max = st.hf_mul_values.iter().copied().max().unwrap_or(0);
    out.push_str(&format!(
        "{}\t{}\t{:.4}\t{}\t{}\t{:.2}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}",
        image,
        effort,
        distance,
        encoder,
        bytes,
        ms,
        st.total_varblocks,
        st.total_8x8_area,
        st.median_hf_mul(),
        st.mean_hf_mul(),
        st.mad_hf_mul(),
        hf_min,
        hf_max,
        qp.global_scale,
        qp.quant_lf,
    ));
    for dt in DCT_TYPES {
        out.push_str(&format!("\t{:.4}", st.pct_by_count(dt)));
    }
    for dt in DCT_TYPES {
        out.push_str(&format!("\t{:.4}", st.pct_by_area(dt)));
    }
    out.push('\n');
}

// ── Pretty-print summary for the terminal ──────────────────────────────────

fn print_cell_summary(c: &CellResult) {
    println!("===== {} e{} d{:.1} =====", c.image, c.effort, c.distance);
    println!(
        "  bytes:   ours {}  cjxl {}  Δ={:+.1}%",
        c.ours_bytes,
        c.cjxl_bytes,
        (c.ours_bytes as f64 - c.cjxl_bytes as f64) / c.cjxl_bytes as f64 * 100.0
    );
    println!(
        "  varblocks: ours {}  cjxl {}  Δ={:+.1}%",
        c.ours_stats.total_varblocks,
        c.cjxl_stats.total_varblocks,
        (c.ours_stats.total_varblocks as f64 - c.cjxl_stats.total_varblocks as f64)
            / c.cjxl_stats.total_varblocks as f64
            * 100.0
    );
    println!(
        "  hf_mul (raw_quant):  ours med={:.1} mean={:.2} mad={:.2} [{}, {}]    cjxl med={:.1} mean={:.2} mad={:.2} [{}, {}]",
        c.ours_stats.median_hf_mul(),
        c.ours_stats.mean_hf_mul(),
        c.ours_stats.mad_hf_mul(),
        c.ours_stats.hf_mul_values.iter().copied().min().unwrap_or(0),
        c.ours_stats.hf_mul_values.iter().copied().max().unwrap_or(0),
        c.cjxl_stats.median_hf_mul(),
        c.cjxl_stats.mean_hf_mul(),
        c.cjxl_stats.mad_hf_mul(),
        c.cjxl_stats.hf_mul_values.iter().copied().min().unwrap_or(0),
        c.cjxl_stats.hf_mul_values.iter().copied().max().unwrap_or(0),
    );
    println!(
        "  global_scale: ours={} cjxl={}    quant_lf: ours={} cjxl={}",
        c.ours_qp.global_scale, c.cjxl_qp.global_scale, c.ours_qp.quant_lf, c.cjxl_qp.quant_lf
    );
    println!(
        "  AC strategy histogram (by 8x8-block-area %, sorted by largest gap):"
    );
    // Compute the area-delta per type and sort.
    let mut types_union: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for k in c.ours_stats.by_type.keys() {
        types_union.insert(k.clone());
    }
    for k in c.cjxl_stats.by_type.keys() {
        types_union.insert(k.clone());
    }
    let mut rows: Vec<(String, f64, f64, f64)> = types_union
        .iter()
        .map(|t| {
            let o = c.ours_stats.pct_by_area(t);
            let cj = c.cjxl_stats.pct_by_area(t);
            (t.clone(), o, cj, o - cj)
        })
        .collect();
    rows.sort_by(|a, b| b.3.abs().partial_cmp(&a.3.abs()).unwrap());
    for (t, o, cj, d) in rows.iter().take(8) {
        println!(
            "    {:12}  ours {:5.1}%   cjxl {:5.1}%   Δ {:+5.1}pp",
            t, o, cj, d
        );
    }
    println!();
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut output: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                output = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                i += 1;
            }
        }
    }
    let out_path = output
        .unwrap_or_else(|| output_dir().join("codec_wiki_wedge_sections_2026-05-19.tsv"));

    let work = work_dir();
    fs::create_dir_all(&work).expect("create work dir");
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }

    let mut tsv = tsv_header();
    let mut results: Vec<CellResult> = Vec::new();

    for &(image, effort, distance) in CELLS {
        eprintln!(">>> Investigating {} e{} d{}", image, effort, distance);
        match investigate_cell(image, effort, distance, &work) {
            Some(c) => {
                emit_row(
                    &mut tsv,
                    image,
                    effort,
                    distance,
                    "ours",
                    c.ours_bytes,
                    c.ours_ms,
                    &c.ours_stats,
                    &c.ours_qp,
                );
                emit_row(
                    &mut tsv,
                    image,
                    effort,
                    distance,
                    "cjxl",
                    c.cjxl_bytes,
                    c.cjxl_ms,
                    &c.cjxl_stats,
                    &c.cjxl_qp,
                );
                print_cell_summary(&c);
                results.push(c);
            }
            None => eprintln!("    SKIPPED — investigate_cell returned None"),
        }
    }

    fs::write(&out_path, tsv).expect("write tsv");
    eprintln!("Wrote {}", out_path.display());

    // Print aggregate summary across all cells
    println!("\n===== AGGREGATE (mean Δpp by 8x8-block-area, ours - cjxl) =====");
    let mut sum_by_type: HashMap<String, (f64, usize)> = HashMap::new();
    for c in &results {
        let mut types_union: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for k in c.ours_stats.by_type.keys() {
            types_union.insert(k.clone());
        }
        for k in c.cjxl_stats.by_type.keys() {
            types_union.insert(k.clone());
        }
        for t in &types_union {
            let d = c.ours_stats.pct_by_area(t) - c.cjxl_stats.pct_by_area(t);
            let entry = sum_by_type.entry(t.clone()).or_insert((0.0, 0));
            entry.0 += d;
            entry.1 += 1;
        }
    }
    let mut rows: Vec<(String, f64)> = sum_by_type
        .iter()
        .map(|(t, (s, n))| (t.clone(), if *n == 0 { 0.0 } else { *s / *n as f64 }))
        .collect();
    rows.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    for (t, d) in &rows {
        println!("  {:12}  Δ {:+6.2}pp", t, d);
    }

    // HF mul aggregate
    let mut ours_med_sum = 0.0;
    let mut cjxl_med_sum = 0.0;
    for c in &results {
        ours_med_sum += c.ours_stats.median_hf_mul();
        cjxl_med_sum += c.cjxl_stats.median_hf_mul();
    }
    let n = results.len() as f64;
    println!(
        "\n  HF mul median (avg across cells):  ours {:.2}  cjxl {:.2}  Δ={:+.2}",
        ours_med_sum / n,
        cjxl_med_sum / n,
        (ours_med_sum - cjxl_med_sum) / n
    );

    // global_scale + quant_lf aggregate
    let mut ours_gs = 0u64;
    let mut cjxl_gs = 0u64;
    let mut ours_qlf = 0u64;
    let mut cjxl_qlf = 0u64;
    for c in &results {
        ours_gs += c.ours_qp.global_scale as u64;
        cjxl_gs += c.cjxl_qp.global_scale as u64;
        ours_qlf += c.ours_qp.quant_lf as u64;
        cjxl_qlf += c.cjxl_qp.quant_lf as u64;
    }
    println!(
        "  global_scale (avg):  ours {:.1}  cjxl {:.1}  Δ={:+.1}",
        ours_gs as f64 / n,
        cjxl_gs as f64 / n,
        (ours_gs as f64 - cjxl_gs as f64) / n
    );
    println!(
        "  quant_lf (avg):  ours {:.1}  cjxl {:.1}  Δ={:+.1}",
        ours_qlf as f64 / n,
        cjxl_qlf as f64 / n,
        (ours_qlf as f64 - cjxl_qlf as f64) / n
    );
}
