//! W44-178 Phase 1: pixel-domain diff of clic_097cb426 e7 d=5 between our
//! encoder and cjxl, with per-block aggregation by AC strategy and region.
//!
//! Per W44-173 diagnosis: strategy maps + qac match cjxl yet right-column
//! DCT64X64 regions show -7 SSIM2 vs cjxl. This example reads BOTH bitstreams
//! through jxl-oxide (request linear sRGB output for byte-comparable XYB→RGB)
//! and writes:
//!
//!   * Per-pixel max-abs diff between ours-decoded and cjxl-decoded
//!   * Per-region (3x3 grid, 256x341 cells roughly) max-abs and mean-abs
//!   * Per-8x8-block max-abs with attribution to the OUR encoder's strategy
//!     (parsed from the W44-76 per-block dump generated as a side-effect)
//!   * Bytes histogram of large-diff blocks (>= 0.01 max-abs linear)
//!
//! The PRIMARY question this answers: is the bug localized to DCT64X64
//! first-blocks, to the right-column EDGE, or both?
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!       --features 'parallel butteraugli-loop ssim2-loop' \
//!       --manifest-path jxl-encoder/Cargo.toml \
//!       --example w44_178_recon_diff
//!
//! Output: benchmarks/w44_178_recon_diff_clic_097cb426_2026-05-21.{tsv,meta,blocks.tsv}

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const SRC_PNG: &str =
    "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png";
const EFFORT: u8 = 7;
const DISTANCE: f32 = 5.0;
const SHORT_NAME: &str = "clic_097cb426";

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
        return PathBuf::from(p);
    }
    PathBuf::from("/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl")
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

/// Decode bitstream into linear-RGB f32 planar via jxl-oxide.
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

fn encode_ours(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    strategy: EncoderStrategy,
) -> Option<Vec<u8>> {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_strategy(strategy);
    cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok()
}

fn encode_cjxl(src_png: &Path, distance: f32, effort: u8) -> Option<Vec<u8>> {
    let tmp = std::env::temp_dir().join(format!(
        "w44_178_cjxl_e{}_d{}.jxl",
        effort,
        distance.to_string().replace('.', "_")
    ));
    let bin = cjxl_bin();
    let status = Command::new(&bin)
        .arg(src_png)
        .arg(&tmp)
        .args(["-e", &effort.to_string()])
        .args(["-d", &distance.to_string()])
        .args(["--num_threads", "1"])
        .arg("--quiet")
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&tmp).ok()?;
    let _ = std::fs::remove_file(&tmp);
    Some(bytes)
}

/// Compute per-8x8-block max-abs (over the 64 pixels and 3 channels).
/// Returns: per-block max-abs in row-major (xsize_blocks * ysize_blocks).
fn per_block_max_abs(
    a: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> Vec<f64> {
    let mut per_block = vec![0.0_f64; xsize_blocks * ysize_blocks];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let y0 = by * 8;
            let x0 = bx * 8;
            let mut bmax = 0.0_f64;
            for py in 0..8 {
                let y = y0 + py;
                if y >= height {
                    break;
                }
                for px in 0..8 {
                    let x = x0 + px;
                    if x >= width {
                        break;
                    }
                    let i = (y * width + x) * 3;
                    for ch in 0..3 {
                        let d = (a[i + ch] - b[i + ch]).abs() as f64;
                        if d > bmax {
                            bmax = d;
                        }
                    }
                }
            }
            per_block[by * xsize_blocks + bx] = bmax;
        }
    }
    per_block
}

const STRATEGY_NAMES: [&str; 19] = [
    "DCT8",     // 0
    "DCT16X8",  // 1
    "DCT8X16",  // 2
    "DCT16X16", // 3
    "DCT32X32", // 4
    "DCT4X8",   // 5
    "DCT8X4",   // 6
    "DCT4X4",   // 7
    "IDENTITY", // 8
    "DCT2X2",   // 9
    "DCT32X16", // 10
    "DCT16X32", // 11
    "AFV0",     // 12
    "AFV1",     // 13
    "AFV2",     // 14
    "AFV3",     // 15
    "DCT64X64", // 16
    "DCT64X32", // 17
    "DCT32X64", // 18
];

/// Convert our raw_strategy to libjxl wire strategy. Our convention:
///   0=DCT8, 1=DCT16X8, 2=DCT8X16, 3=DCT16X16, 4=DCT32X32, 5=DCT4X8,
///   6=DCT8X4, 7=DCT4X4, 8=IDENTITY, 9=DCT2X2, 10=DCT32X16, 11=DCT16X32,
///   12-15=AFV0-3, 16=DCT64X64, 17=DCT64X32, 18=DCT32X64
///
/// Wire (libjxl):
///   0=DCT8, 1=Hornuss, 2=DCT2X2, 3=DCT4X4, 4=DCT16X16, 5=DCT32X32,
///   6=DCT16X8, 7=DCT8X16, 8=DCT32X8, 9=DCT8X32, 10=DCT32X16, 11=DCT16X32,
///   12=DCT4X8, 13=DCT8X4, 14=AFV0, 15=AFV1, 16=AFV2, 17=AFV3,
///   18=DCT64X64, 19=DCT64X32, 20=DCT32X64
const RAW_TO_WIRE: [u8; 19] = [
    0,  // DCT8
    6,  // DCT16X8
    7,  // DCT8X16
    4,  // DCT16X16
    5,  // DCT32X32
    12, // DCT4X8
    13, // DCT8X4
    3,  // DCT4X4
    1,  // IDENTITY (Hornuss)
    2,  // DCT2X2
    10, // DCT32X16
    11, // DCT16X32
    14, // AFV0
    15, // AFV1
    16, // AFV2
    17, // AFV3
    18, // DCT64X64
    19, // DCT64X32
    20, // DCT32X64
];

/// Parse the W44-76 per-block TSV produced by JXL_W44_76_PER_BLOCK_DUMP.
/// Returns per-(bx, by) (raw_strategy, covered_x, covered_y) for first-blocks
/// of the Y channel (channel == 1). Returns dense map of size
/// xsize_blocks * ysize_blocks, with raw_strategy=255 for non-first cells.
fn parse_per_block_dump(
    path: &Path,
    xsize_blocks: usize,
    ysize_blocks: usize,
) -> Vec<u8> {
    let mut strat = vec![255u8; xsize_blocks * ysize_blocks];
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("WARN: could not read {}: {}", path.display(), e);
            return strat;
        }
    };
    // Format from w44_76_dump.rs: TSV with columns bx, by, strategy, channel, ...
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 4 {
            continue;
        }
        let bx: usize = parts[0].parse().unwrap_or(usize::MAX);
        let by: usize = parts[1].parse().unwrap_or(usize::MAX);
        let strategy: u8 = parts[2].parse().unwrap_or(255);
        let channel: u8 = parts[3].parse().unwrap_or(255);
        if channel != 1 || bx >= xsize_blocks || by >= ysize_blocks {
            continue;
        }
        // strategy here is libjxl-wire code. Convert back to our raw_strategy
        // for direct comparison with our enum.
        let raw = wire_to_raw(strategy);
        // Mark first-block cell and also paint covered_blocks region with the
        // same raw code (so per-block diff aggregation knows the strategy
        // covering each 8x8 cell).
        let (cx, cy) = covered_dims(raw);
        for dy in 0..cy {
            for dx in 0..cx {
                let yy = by + dy;
                let xx = bx + dx;
                if yy < ysize_blocks && xx < xsize_blocks {
                    strat[yy * xsize_blocks + xx] = raw;
                }
            }
        }
    }
    strat
}

fn wire_to_raw(wire: u8) -> u8 {
    // Inverse of RAW_TO_WIRE.
    for (raw, &w) in RAW_TO_WIRE.iter().enumerate() {
        if w == wire {
            return raw as u8;
        }
    }
    255
}

fn covered_dims(raw: u8) -> (usize, usize) {
    // Mirrors COVERED_X/COVERED_Y in ac_strategy.rs.
    match raw {
        0 | 5 | 6 | 7 | 8 | 9 | 12 | 13 | 14 | 15 => (1, 1),
        1 => (1, 2),  // DCT16X8 = 1 col, 2 rows
        2 => (2, 1),  // DCT8X16 = 2 cols, 1 row
        3 => (2, 2),  // DCT16X16
        4 => (4, 4),  // DCT32X32
        10 => (2, 4), // DCT32X16 (cx=2, cy=4)
        11 => (4, 2), // DCT16X32 (cx=4, cy=2)
        16 => (8, 8), // DCT64X64
        17 => (4, 8), // DCT64X32
        18 => (8, 4), // DCT32X64
        _ => (1, 1),
    }
}

fn main() {
    eprintln!(
        "W44-178 Phase 1 — pixel-domain diff on {} e{} d={}",
        SHORT_NAME, EFFORT, DISTANCE
    );
    eprintln!("=========================================================");

    let (src_rgb, w, h) = load_png(Path::new(SRC_PNG)).expect("load source PNG");
    let width = w as usize;
    let height = h as usize;
    let xsize_blocks = width.div_ceil(8);
    let ysize_blocks = height.div_ceil(8);
    eprintln!(
        "src: {}×{}  blocks: {}×{}  pixels: {}",
        width,
        height,
        xsize_blocks,
        ysize_blocks,
        width * height
    );

    // Stage 1: encode with ours + cjxl
    eprintln!("\n[encode ours zenjxl] ...");
    let t = Instant::now();
    let ours = encode_ours(&src_rgb, w, h, DISTANCE, EFFORT, EncoderStrategy::Zenjxl)
        .expect("encode ours");
    eprintln!(
        "  ours bytes: {}  encode_ms: {:.0}",
        ours.len(),
        t.elapsed().as_secs_f64() * 1000.0
    );

    eprintln!("[encode cjxl] ...");
    let t = Instant::now();
    let cjxl_bytes = encode_cjxl(Path::new(SRC_PNG), DISTANCE, EFFORT).expect("encode cjxl");
    eprintln!(
        "  cjxl bytes: {}  encode_ms: {:.0}",
        cjxl_bytes.len(),
        t.elapsed().as_secs_f64() * 1000.0
    );

    // Stage 2: decode both via jxl-oxide → linear sRGB
    eprintln!("[decode ours via jxl-oxide linear sRGB] ...");
    let (dw, dh, ours_lin) = decode_jxl_linear(&ours).expect("decode ours");
    assert_eq!(dw, width);
    assert_eq!(dh, height);

    eprintln!("[decode cjxl via jxl-oxide linear sRGB] ...");
    let (cw, ch, cjxl_lin) = decode_jxl_linear(&cjxl_bytes).expect("decode cjxl");
    assert_eq!(cw, width);
    assert_eq!(ch, height);

    // Stage 3: per-pixel max-abs diff between ours and cjxl
    let mut total_sq_err = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut max_abs_loc = (0usize, 0usize, 'R');
    let mut over_001 = 0u64;
    let mut over_01 = 0u64;
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 3;
            for (ch, c) in ['R', 'G', 'B'].iter().enumerate() {
                let d = (ours_lin[i + ch] - cjxl_lin[i + ch]).abs() as f64;
                if d > max_abs {
                    max_abs = d;
                    max_abs_loc = (x, y, *c);
                }
                if d > 0.01 {
                    over_01 += 1;
                }
                if d > 0.001 {
                    over_001 += 1;
                }
                total_sq_err += d * d;
            }
        }
    }
    let rmse = (total_sq_err / (width * height * 3) as f64).sqrt();
    eprintln!(
        "\n[global] max_abs={:.4} @ ({},{}) ch={}  RMSE={:.5}  pixels>0.001={}  pixels>0.01={}",
        max_abs,
        max_abs_loc.0,
        max_abs_loc.1,
        max_abs_loc.2,
        rmse,
        over_001,
        over_01
    );

    // Stage 4: per-8x8-block max-abs
    let per_block_diff = per_block_max_abs(&ours_lin, &cjxl_lin, width, height, xsize_blocks, ysize_blocks);

    // Stage 5: encode AGAIN with the W44-76 dump enabled to learn our per-block strategy.
    let dump_dir = std::env::temp_dir().join("w44_178_ours_dump");
    let _ = std::fs::remove_dir_all(&dump_dir);
    let _ = std::fs::create_dir_all(&dump_dir);
    eprintln!("\n[re-encode ours with W44-76 per-block dump] ...");
    unsafe {
        std::env::set_var("JXL_W44_76_PER_BLOCK_DUMP", &dump_dir);
    }
    let _ = encode_ours(&src_rgb, w, h, DISTANCE, EFFORT, EncoderStrategy::Zenjxl)
        .expect("re-encode ours for dump");
    unsafe { std::env::remove_var("JXL_W44_76_PER_BLOCK_DUMP"); }

    let dump_path = dump_dir.join("per_block_ours.tsv");
    let our_strategy = parse_per_block_dump(&dump_path, xsize_blocks, ysize_blocks);

    // Stage 6: aggregate per-block diff by strategy (where the strategy
    // COVERS the cell, not just the first-block position).
    let mut by_strategy = vec![(0u64, 0.0_f64, 0.0_f64); 19]; // (count, sum_max, max_max)
    let mut over_01_count = 0u64;
    for i in 0..xsize_blocks * ysize_blocks {
        let s = our_strategy[i];
        if s as usize >= 19 {
            continue;
        }
        let d = per_block_diff[i];
        by_strategy[s as usize].0 += 1;
        by_strategy[s as usize].1 += d;
        if d > by_strategy[s as usize].2 {
            by_strategy[s as usize].2 = d;
        }
        if d > 0.01 {
            over_01_count += 1;
        }
    }

    // Stage 7: per-region (3x3) attribution. Region size in blocks ~= ysize_blocks/3.
    let mut region_max = [[0.0_f64; 3]; 3];
    let mut region_sum = [[0.0_f64; 3]; 3];
    let mut region_count = [[0u64; 3]; 3];
    let mut region_strat_count = [[[0u64; 19]; 3]; 3];
    let third_y = ysize_blocks / 3;
    let third_x = xsize_blocks / 3;
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let i = by * xsize_blocks + bx;
            let ry = if by < third_y { 0 } else if by < 2 * third_y { 1 } else { 2 };
            let rx = if bx < third_x { 0 } else if bx < 2 * third_x { 1 } else { 2 };
            let d = per_block_diff[i];
            if d > region_max[ry][rx] {
                region_max[ry][rx] = d;
            }
            region_sum[ry][rx] += d;
            region_count[ry][rx] += 1;
            let s = our_strategy[i] as usize;
            if s < 19 {
                region_strat_count[ry][rx][s] += 1;
            }
        }
    }

    // Output: TSV + meta
    let bench_dir = PathBuf::from("benchmarks");
    let _ = std::fs::create_dir_all(&bench_dir);
    let tsv_path = bench_dir.join("w44_178_recon_diff_clic_097cb426_2026-05-21.tsv");
    let meta_path = bench_dir.join("w44_178_recon_diff_clic_097cb426_2026-05-21.meta");
    let blocks_path = bench_dir.join("w44_178_recon_diff_clic_097cb426_2026-05-21.blocks.tsv");

    {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tsv_path)
            .unwrap();
        writeln!(f, "# W44-178 Phase 1 — pixel-domain diff aggregation (ours vs cjxl decoded)").unwrap();
        writeln!(f, "# image\teffort\tdistance\tours_bytes\tcjxl_bytes\tmax_abs\trmse\tpixels_over_0.001\tpixels_over_0.01\tblocks_over_0.01").unwrap();
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{}",
            SHORT_NAME,
            EFFORT,
            DISTANCE,
            ours.len(),
            cjxl_bytes.len(),
            max_abs,
            rmse,
            over_001,
            over_01,
            over_01_count,
        ).unwrap();
        writeln!(f, "# === per-strategy aggregation (ours' strategy, where strategy COVERS the cell) ===").unwrap();
        writeln!(f, "# raw_strategy\tname\tcell_count\tmean_max_abs\tmax_max_abs").unwrap();
        for (s, &(cnt, sum_m, max_m)) in by_strategy.iter().enumerate() {
            if cnt == 0 {
                continue;
            }
            writeln!(
                f,
                "{}\t{}\t{}\t{:.6}\t{:.6}",
                s,
                STRATEGY_NAMES[s],
                cnt,
                sum_m / cnt as f64,
                max_m,
            ).unwrap();
        }
        writeln!(f, "# === per-region (3x3 grid in BLOCKS) ===").unwrap();
        writeln!(f, "# region\tby_lo\tby_hi\tbx_lo\tbx_hi\tcell_count\tmean_max_abs\tmax_max_abs\ttop3_strategies").unwrap();
        for ry in 0..3 {
            for rx in 0..3 {
                let cnt = region_count[ry][rx];
                let mean = if cnt > 0 { region_sum[ry][rx] / cnt as f64 } else { 0.0 };
                let max = region_max[ry][rx];
                let by_lo = ry * third_y;
                let by_hi = if ry == 2 { ysize_blocks } else { (ry + 1) * third_y };
                let bx_lo = rx * third_x;
                let bx_hi = if rx == 2 { xsize_blocks } else { (rx + 1) * third_x };
                let mut strat_pairs: Vec<(usize, u64)> = region_strat_count[ry][rx]
                    .iter()
                    .enumerate()
                    .filter_map(|(s, &c)| if c > 0 { Some((s, c)) } else { None })
                    .collect();
                strat_pairs.sort_by(|a, b| b.1.cmp(&a.1));
                let top3: String = strat_pairs
                    .iter()
                    .take(3)
                    .map(|(s, c)| format!("{}={}", STRATEGY_NAMES[*s], c))
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(
                    f,
                    "r[{},{}]\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}",
                    ry, rx, by_lo, by_hi, bx_lo, bx_hi, cnt, mean, max, top3,
                ).unwrap();
            }
        }
    }

    {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&blocks_path)
            .unwrap();
        writeln!(f, "# W44-178 Phase 1 — per-block max-abs diff (linear RGB)").unwrap();
        writeln!(f, "# bx\tby\tour_strategy_raw\tour_strategy_name\tmax_abs_lin_rgb").unwrap();
        // Only emit blocks with diff > 0.001 to keep file manageable.
        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                let i = by * xsize_blocks + bx;
                let d = per_block_diff[i];
                if d < 0.001 {
                    continue;
                }
                let s = our_strategy[i] as usize;
                let name = if s < 19 { STRATEGY_NAMES[s] } else { "?" };
                writeln!(f, "{}\t{}\t{}\t{}\t{:.6}", bx, by, s, name, d).unwrap();
            }
        }
    }

    {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&meta_path)
            .unwrap();
        writeln!(f, "W44-178 Phase 1 — recon-diff for {} e{} d={}", SHORT_NAME, EFFORT, DISTANCE).unwrap();
        writeln!(f, "Date: 2026-05-21").unwrap();
        writeln!(f, "Image: {}×{} ({} blocks)", width, height, xsize_blocks * ysize_blocks).unwrap();
        writeln!(f).unwrap();
        writeln!(f, "Methodology:").unwrap();
        writeln!(f, "  1. Encode the source PNG once with our Zenjxl encoder (e{} d={}).", EFFORT, DISTANCE).unwrap();
        writeln!(f, "  2. Encode the source PNG once with cjxl (-e {} -d {} --num_threads 1).", EFFORT, DISTANCE).unwrap();
        writeln!(f, "  3. Decode BOTH bitstreams via jxl-oxide with linear sRGB output").unwrap();
        writeln!(f, "     (RenderingIntent::Relative) — same color path, same TF, no metadata bias.").unwrap();
        writeln!(f, "  4. Per-pixel max-abs and RMSE in linear-light space.").unwrap();
        writeln!(f, "  5. Per-8x8-block max-abs aggregated by OUR encoder's AC strategy.").unwrap();
        writeln!(f, "  6. Per-3x3 region attribution of WHICH strategies dominate WHERE the diff is.").unwrap();
        writeln!(f).unwrap();
        writeln!(f, "Headline:").unwrap();
        writeln!(f, "  global max_abs (linear RGB) = {:.6} @ ({},{}) ch={}", max_abs, max_abs_loc.0, max_abs_loc.1, max_abs_loc.2).unwrap();
        writeln!(f, "  global RMSE                 = {:.6}", rmse).unwrap();
        writeln!(f, "  pixels w/ diff > 0.001      = {}", over_001).unwrap();
        writeln!(f, "  pixels w/ diff > 0.01       = {}", over_01).unwrap();
        writeln!(f, "  blocks  w/ diff > 0.01      = {}", over_01_count).unwrap();
        writeln!(f).unwrap();
        writeln!(f, "Per-strategy mean of per-block-max-abs (top contributors):").unwrap();
        let mut sorted: Vec<(usize, u64, f64, f64)> = by_strategy
            .iter()
            .enumerate()
            .filter_map(|(i, &(c, s, m))| {
                if c > 0 {
                    Some((i, c, s / c as f64, m))
                } else {
                    None
                }
            })
            .collect();
        sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        for (s, c, mean, max) in sorted.iter().take(10) {
            writeln!(f, "  {:<10} cells={:<6} mean_max_abs={:.5} max_max_abs={:.5}", STRATEGY_NAMES[*s], c, mean, max).unwrap();
        }
        writeln!(f).unwrap();
        writeln!(f, "Per-region mean / max (3x3 grid):").unwrap();
        for ry in 0..3 {
            for rx in 0..3 {
                let cnt = region_count[ry][rx];
                let mean = if cnt > 0 { region_sum[ry][rx] / cnt as f64 } else { 0.0 };
                let max = region_max[ry][rx];
                writeln!(f, "  r[{},{}] cells={:<6} mean={:.5} max={:.5}", ry, rx, cnt, mean, max).unwrap();
            }
        }
        writeln!(f).unwrap();
        writeln!(f, "Outputs:").unwrap();
        writeln!(f, "  {}", tsv_path.display()).unwrap();
        writeln!(f, "  {}", blocks_path.display()).unwrap();
        writeln!(f).unwrap();
        writeln!(f, "Files (parsing):").unwrap();
        writeln!(f, "  W44-76 per-block dump: {}", dump_path.display()).unwrap();
    }

    eprintln!("\nWrote:");
    eprintln!("  {}", tsv_path.display());
    eprintln!("  {}", blocks_path.display());
    eprintln!("  {}", meta_path.display());
}
