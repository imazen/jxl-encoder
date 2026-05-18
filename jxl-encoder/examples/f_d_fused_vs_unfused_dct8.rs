//! W44-9 Sub-chunk B: A/B fused vs unfused DCT8 entropy on F-D wedge cells.
//!
//! Follow-on to W44-6 (`f_d_photo_high_d_ac_overspend_investigation_2026-05-19.md`).
//! W44-6 found we over-pick DCT8 by +2.92pp area vs cjxl on F-D photo cells,
//! confirmed every cost-model formula matches libjxl exactly, and queued three
//! Layer-2 sub-investigations. This is Sub-chunk B (hypothesis 3):
//!
//! > The fused DCT8 entropy path (`jxl_simd::fused_dct8_entropy`) keeps DCT
//! > coefficients in YMM registers and computes entropy via fused multiply-adds.
//! > The unfused fallback (`jxl_simd::fused_dct8_entropy_fallback`) extracts
//! > the block to a scratch buffer, runs `dct_8x8` then `entropy_estimate_coeffs`
//! > as separate kernels. The two might produce slightly different floating-point
//! > results on borderline blocks — giving DCT8 a small consistent cost advantage.
//!
//! Procedure per F-D cell:
//!   A. Encode with fused path (default, `force=false`).
//!   B. Encode with unfused fallback (`force=true`).
//!   C. Run `jxl-inspect export-csv --per-block` on both .jxl files.
//!   D. Diff AC strategy histograms, hf_mul stats, bytes.
//!   E. Decode both with jxl-oxide (linear sRGB) → Rust butteraugli + ssim2.
//!
//! Verdict gates:
//!   - CONFIRMED if mean DCT8 area shift (unfused − fused) ≤ −1pp AND
//!     |shift| ≥ 1pp on ≥ 4 of 7 cells (i.e. unfused picks meaningfully
//!     less DCT8 — matching the cjxl-side direction the W44-6 wedge identified).
//!   - REFUTED if max |DCT8 area shift| < 0.3pp across all cells.
//!   - INCONCLUSIVE otherwise.
//!
//! Run:
//! ```bash
//! cargo run -p jxl-encoder --release --features '__expert butteraugli-loop ssim2-loop' \
//!   --example f_d_fused_vs_unfused_dct8 -- \
//!   --output benchmarks/f_d_fused_vs_unfused_dct8_2026-05-19.tsv
//! ```

#![allow(clippy::too_many_arguments, clippy::manual_is_multiple_of)]

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::HashMap;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

// Cells selected from W44-6 memo (the 7 F-D cells with biggest ssim2 loss).
const CELLS: &[(&str, u8, f32)] = &[
    ("1531677.png", 7, 5.0),
    ("1531677.png", 7, 6.0),
    ("1531677.png", 7, 4.0),
    ("1420710.png", 7, 6.0),
    ("1420710.png", 7, 5.0),
    ("1189261.png", 7, 3.0),
    ("1189261.png", 7, 4.0),
];

fn corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/lilith/work/codec-corpus"))
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
    PathBuf::from("/tmp/f_d_fused_vs_unfused_dct8")
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

#[derive(Default, Debug, Clone)]
struct BlockStats {
    total_varblocks: u64,
    total_8x8_area: u64,
    by_type: HashMap<String, (u64, u64)>,
    hf_mul_values: Vec<i32>,
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
        self.hf_mul_values.iter().map(|&x| x as f64).sum::<f64>() / self.hf_mul_values.len() as f64
    }

    fn pct_by_area(&self, dct_type: &str) -> f64 {
        if self.total_8x8_area == 0 {
            return 0.0;
        }
        let area = self.by_type.get(dct_type).map(|&(_, a)| a).unwrap_or(0);
        area as f64 / self.total_8x8_area as f64 * 100.0
    }

    fn pct_by_count(&self, dct_type: &str) -> f64 {
        if self.total_varblocks == 0 {
            return 0.0;
        }
        let count = self.by_type.get(dct_type).map(|&(c, _)| c).unwrap_or(0);
        count as f64 / self.total_varblocks as f64 * 100.0
    }
}

fn rand_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{nanos:x}")
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
            continue;
        }
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

#[derive(Debug, Default, Clone)]
struct ArmMeasure {
    bytes: u64,
    encode_ms: f64,
    butteraugli: f64,
    ssim2: f64,
    stats: BlockStats,
}

fn measure_arm(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
    work: &Path,
) -> Option<ArmMeasure> {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1);
    let t0 = Instant::now();
    let bytes = cfg.encode(rgb_u8, w, h, PixelLayout::Rgb8).ok()?;
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let (dw, dh, decoded_linear) = decode_jxl_linear(&bytes)?;
    if dw != w as usize || dh != h as usize {
        eprintln!("    decode dim mismatch {dw}x{dh} vs {w}x{h}");
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), params)
        .map(|r| r.score)
        .unwrap_or(f64::NAN);

    let dec_srgb: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);

    let stats = extract_block_stats(&bytes, work)?;
    Some(ArmMeasure {
        bytes: bytes.len() as u64,
        encode_ms,
        butteraugli: bfly,
        ssim2,
        stats,
    })
}

#[derive(Debug)]
struct CellResult {
    image: String,
    effort: u8,
    distance: f32,
    fused: ArmMeasure,
    unfused: ArmMeasure,
}

fn investigate_cell(
    image: &str,
    effort: u8,
    distance: f32,
    work: &Path,
    params: &ButteraugliParams,
) -> Option<CellResult> {
    let src = corpus_dir().join("CID22/CID22-512/validation").join(image);
    let img = image::open(&src).ok()?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let rgb_u8: &[u8] = rgb.as_raw();

    // Pre-build reference once.
    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            [
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
            ]
        })
        .collect();
    let orig_pixels: Vec<RGB<f32>> = linear_rgb
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let orig_linear_img = Img::new(orig_pixels, w as usize, h as usize);
    let orig_srgb_pixels: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
    let orig_srgb_img = Img::new(orig_srgb_pixels, w as usize, h as usize);

    // Arm A: fused (default)
    jxl_encoder::vardct::set_force_unfused_dct8_entropy(false);
    jxl_encoder::vardct::reset_dct8_branch_counters();
    let fused = measure_arm(
        rgb_u8,
        w,
        h,
        distance,
        effort,
        &orig_linear_img,
        &orig_srgb_img,
        params,
        work,
    )?;
    let (fused_hits_a, unfused_hits_a) = jxl_encoder::vardct::dct8_branch_counters();
    if unfused_hits_a > 0 {
        eprintln!(
            "    WARN: fused arm leaked unfused branch hits: fused={} unfused={}",
            fused_hits_a, unfused_hits_a
        );
    }

    // Arm B: unfused fallback
    jxl_encoder::vardct::set_force_unfused_dct8_entropy(true);
    jxl_encoder::vardct::reset_dct8_branch_counters();
    let unfused = measure_arm(
        rgb_u8,
        w,
        h,
        distance,
        effort,
        &orig_linear_img,
        &orig_srgb_img,
        params,
        work,
    )?;
    let (fused_hits_b, unfused_hits_b) = jxl_encoder::vardct::dct8_branch_counters();
    eprintln!(
        "    branch hits: arm-A fused={} unfused={}  |  arm-B fused={} unfused={}",
        fused_hits_a, unfused_hits_a, fused_hits_b, unfused_hits_b
    );

    // Restore default
    jxl_encoder::vardct::set_force_unfused_dct8_entropy(false);

    Some(CellResult {
        image: image.to_string(),
        effort,
        distance,
        fused,
        unfused,
    })
}

const DCT_TYPES: &[&str] = &[
    "Dct8", "Dct16", "Dct32", "Dct64", "Dct16x8", "Dct8x16", "Dct32x16", "Dct16x32", "Dct32x64",
    "Dct64x32", "Dct4x8", "Dct8x4", "Dct4", "Dct2", "Hornuss", "Afv0", "Afv1", "Afv2", "Afv3",
];

fn tsv_header() -> String {
    let mut s = String::from(
        "image\teffort\tdistance\tarm\tbytes\tencode_ms\ttotal_varblocks\ttotal_8x8_area\t\
         hf_mul_median\thf_mul_mean\tbutteraugli\tssim2",
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

fn emit_row(out: &mut String, image: &str, effort: u8, distance: f32, arm: &str, m: &ArmMeasure) {
    out.push_str(&format!(
        "{}\t{}\t{:.4}\t{}\t{}\t{:.2}\t{}\t{}\t{:.4}\t{:.4}\t{:.6}\t{:.6}",
        image,
        effort,
        distance,
        arm,
        m.bytes,
        m.encode_ms,
        m.stats.total_varblocks,
        m.stats.total_8x8_area,
        m.stats.median_hf_mul(),
        m.stats.mean_hf_mul(),
        m.butteraugli,
        m.ssim2,
    ));
    for dt in DCT_TYPES {
        out.push_str(&format!("\t{:.4}", m.stats.pct_by_count(dt)));
    }
    for dt in DCT_TYPES {
        out.push_str(&format!("\t{:.4}", m.stats.pct_by_area(dt)));
    }
    out.push('\n');
}

fn print_cell(c: &CellResult) {
    println!("===== {} e{} d{:.1} =====", c.image, c.effort, c.distance);
    println!(
        "  bytes:        fused {}  unfused {}  Δ={:+.2}%",
        c.fused.bytes,
        c.unfused.bytes,
        (c.unfused.bytes as f64 - c.fused.bytes as f64) / c.fused.bytes as f64 * 100.0
    );
    println!(
        "  varblocks:    fused {}  unfused {}  Δ={:+.2}%",
        c.fused.stats.total_varblocks,
        c.unfused.stats.total_varblocks,
        (c.unfused.stats.total_varblocks as f64 - c.fused.stats.total_varblocks as f64)
            / c.fused.stats.total_varblocks as f64
            * 100.0
    );
    println!(
        "  butteraugli:  fused {:.4}  unfused {:.4}  Δ={:+.3}%",
        c.fused.butteraugli,
        c.unfused.butteraugli,
        (c.unfused.butteraugli - c.fused.butteraugli) / c.fused.butteraugli * 100.0
    );
    println!(
        "  ssim2:        fused {:.3}  unfused {:.3}  Δ={:+.4}",
        c.fused.ssim2,
        c.unfused.ssim2,
        c.unfused.ssim2 - c.fused.ssim2
    );

    let f_dct8 = c.fused.stats.pct_by_area("Dct8");
    let u_dct8 = c.unfused.stats.pct_by_area("Dct8");
    println!(
        "  DCT8 area:    fused {:.2}%  unfused {:.2}%  Δ {:+.3}pp",
        f_dct8,
        u_dct8,
        u_dct8 - f_dct8
    );

    let mut types_union: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for k in c.fused.stats.by_type.keys() {
        types_union.insert(k.clone());
    }
    for k in c.unfused.stats.by_type.keys() {
        types_union.insert(k.clone());
    }
    let mut rows: Vec<(String, f64, f64, f64)> = types_union
        .iter()
        .map(|t| {
            let f = c.fused.stats.pct_by_area(t);
            let u = c.unfused.stats.pct_by_area(t);
            (t.clone(), f, u, u - f)
        })
        .collect();
    rows.sort_by(|a, b| b.3.abs().partial_cmp(&a.3.abs()).unwrap());
    println!("  Top area shifts (fused → unfused):");
    for (t, f, u, d) in rows.iter().take(5) {
        println!(
            "    {:12}  fused {:5.2}%  unfused {:5.2}%  Δ {:+5.3}pp",
            t, f, u, d
        );
    }
    println!();
}

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
    let out_path =
        output.unwrap_or_else(|| output_dir().join("f_d_fused_vs_unfused_dct8_2026-05-19.tsv"));

    let work = work_dir();
    fs::create_dir_all(&work).expect("create work dir");
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }

    let params = ButteraugliParams::default();

    let mut tsv = tsv_header();
    let mut results: Vec<CellResult> = Vec::new();

    for &(image, effort, distance) in CELLS {
        eprintln!(">>> {} e{} d{}", image, effort, distance);
        match investigate_cell(image, effort, distance, &work, &params) {
            Some(c) => {
                emit_row(&mut tsv, image, effort, distance, "fused", &c.fused);
                emit_row(&mut tsv, image, effort, distance, "unfused", &c.unfused);
                print_cell(&c);
                results.push(c);
            }
            None => eprintln!("    SKIPPED — investigate_cell returned None"),
        }
    }

    fs::write(&out_path, tsv).expect("write tsv");
    eprintln!("Wrote {}", out_path.display());

    // Aggregate verdict
    println!(
        "\n===== AGGREGATE (mean Δ across {} cells, unfused − fused) =====",
        results.len()
    );
    if results.is_empty() {
        println!("  (no results)");
        return;
    }
    let n = results.len() as f64;
    let mean_bytes_pct: f64 = results
        .iter()
        .map(|c| (c.unfused.bytes as f64 - c.fused.bytes as f64) / c.fused.bytes as f64 * 100.0)
        .sum::<f64>()
        / n;
    let mean_bfly_pct: f64 = results
        .iter()
        .map(|c| (c.unfused.butteraugli - c.fused.butteraugli) / c.fused.butteraugli * 100.0)
        .sum::<f64>()
        / n;
    let mean_ssim2_delta: f64 = results
        .iter()
        .map(|c| c.unfused.ssim2 - c.fused.ssim2)
        .sum::<f64>()
        / n;

    let dct8_shifts: Vec<f64> = results
        .iter()
        .map(|c| c.unfused.stats.pct_by_area("Dct8") - c.fused.stats.pct_by_area("Dct8"))
        .collect();
    let mean_dct8_shift: f64 = dct8_shifts.iter().sum::<f64>() / n;
    let max_abs_dct8_shift: f64 = dct8_shifts
        .iter()
        .cloned()
        .fold(0.0_f64, |a, b| a.max(b.abs()));

    println!("  bytes Δ:              {:+.3}%", mean_bytes_pct);
    println!("  butteraugli Δ:        {:+.3}%", mean_bfly_pct);
    println!("  ssim2 Δ:              {:+.4}", mean_ssim2_delta);
    println!("  DCT8 area Δ (mean):   {:+.3}pp", mean_dct8_shift);
    println!("  DCT8 area Δ (max |·|): {:.3}pp", max_abs_dct8_shift);
    println!();
    println!("  Per-cell DCT8 area shift (unfused − fused, pp):");
    for (c, d) in results.iter().zip(dct8_shifts.iter()) {
        println!(
            "    {:20} e{} d{:.1}   Δ {:+6.3}pp",
            c.image, c.effort, c.distance, d
        );
    }

    println!("\n===== W44-9 VERDICT =====");
    let cells_with_neg_shift = dct8_shifts.iter().filter(|d| **d <= -1.0).count();
    let cells_all_below_03 = dct8_shifts.iter().all(|d| d.abs() < 0.3);
    if cells_with_neg_shift >= 4 && mean_dct8_shift <= -0.5 {
        println!(
            "  CONFIRMED: unfused path picks ≥1pp LESS DCT8 area on {}/7 cells",
            cells_with_neg_shift
        );
        println!(
            "  (mean shift {:+.3}pp, in cjxl direction). Hypothesis 3 supported:",
            mean_dct8_shift
        );
        println!("  the fused-AVX2 kernel's FMA op-ordering gives DCT8 a borderline");
        println!("  cost advantage. Next chunk: align fused kernel rounding/FMA order");
        println!("  to match unfused. Hash-lock impact: bitstream-affecting.");
    } else if cells_all_below_03 {
        println!("  REFUTED: |DCT8 area shift| < 0.3pp on all 7 cells.");
        println!(
            "  (max |Δ| {:.3}pp, mean {:+.3}pp). Fused vs unfused paths agree to",
            max_abs_dct8_shift, mean_dct8_shift
        );
        println!("  within strategy-selection precision. Hypothesis 3 is NOT the cause");
        println!("  of the +2.92pp DCT8 area over-pick. Escalate to Sub-chunk A (");
        println!("  quant_field SIMD drift dump vs libjxl debug build, 1-2 days).");
    } else {
        println!("  INCONCLUSIVE: shifts present but not in the W44-6 direction or not");
        println!(
            "  on majority of cells. mean Δ {:+.3}pp; max |Δ| {:.3}pp;",
            mean_dct8_shift, max_abs_dct8_shift
        );
        println!("  {}/7 cells with shift ≤ -1pp.", cells_with_neg_shift);
        println!("  Recommend: review per-cell breakdown above. If shifts are random");
        println!("  (some +, some -) → FMA noise, not systematic bias → REFUTED-equiv.");
        println!("  If shifts are consistent but small → may need larger corpus.");
    }
}
