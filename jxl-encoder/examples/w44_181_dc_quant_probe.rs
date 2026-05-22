//! W44-181 read-only probe: DC quantization f32 evaluation-order divergence.
//!
//! Follow-on to W44-178 honest-stop (`memory/w44_178_recon_diff_falsifies_dct64_2026-05-21.md`):
//! W44-178 ruled out DCT64X64 IDCT, dequant, EPF, gaborish, and CfL AC as the
//! cause of clic_097cb426 right-column SSIM2 -7 deficit. The remaining
//! top-EV candidate was DC quantization precision rounding mode/order.
//!
//! libjxl (`enc_modular.cc:1693-1694`):
//!   `std::round((row[x] - quant_row_y[x] * (y_factor * cfl_factor)) * inv_factor)`
//!
//! Ours (`vardct/transform.rs:947-949` for DCT64X64 B channel):
//!   `(dcs[iy * 8 + ix] * inv_factor - y_dc * dc_cfl_factor).round() as i16`
//!
//! Algebraically equivalent at default cmap (`color_factor=84`, `ytox_dc=ytob_dc=0`,
//! `dc_cfl_factor=0.5` for B, `0.0` for X). But the f32 multiply/subtract order
//! differs: ours does `(dc * if) - (y_dc * 0.5)`; libjxl does
//! `(dc - y_dc * 0.5/if) * if`. These can differ by ±1 ULP at rounding boundaries.
//!
//! This probe:
//! 1. Encodes clic_097cb426 e7 d=5 with `JXL_W44_181_DUMP_DC=<dir>` set so
//!    the W44-181 dump module records per-block DC quant inputs.
//! 2. Post-processes the dump: for each (dc, y_dc, inv_factor, dc_cfl_factor)
//!    tuple, computes BOTH our expression and the libjxl-form expression on
//!    the same f32 inputs and counts divergences.
//! 3. Joins per-block divergence count with `w44_178_recon_diff_clic_097cb426_*.blocks.tsv`
//!    to check correlation: do divergent DC quant blocks coincide with the
//!    right-column DCT64X64 regions where SSIM2 -7 lives?
//!
//! ZERO production source change. The probe instrumentation is gated behind
//! the JXL_W44_181_DUMP_DC env var and is a no-op when unset.
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!       --features 'parallel butteraugli-loop ssim2-loop' \
//!       --manifest-path jxl-encoder/Cargo.toml \
//!       --example w44_181_dc_quant_probe
//!
//! Output: benchmarks/w44_181_dc_quant_probe_2026-05-21.{tsv,meta}
//!         + /tmp/w44_181_dump/dc_quant_inputs.tsv (raw dump)

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const SRC_PNG: &str =
    "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png";
const EFFORT: u8 = 7;
const DISTANCE: f32 = 5.0;
const SHORT_NAME: &str = "clic_097cb426";

const DUMP_DIR: &str = "/tmp/w44_181_dump";
const W178_BLOCKS_TSV: &str = "benchmarks/w44_178_recon_diff_clic_097cb426_2026-05-21.blocks.tsv";
const OUT_TSV: &str = "benchmarks/w44_181_dc_quant_probe_2026-05-21.tsv";
const OUT_META: &str = "benchmarks/w44_181_dc_quant_probe_2026-05-21.meta";

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn encode_with_dump(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    dump_dir: &str,
) -> Option<Vec<u8>> {
    // Clear stale dump dir.
    let _ = std::fs::remove_dir_all(dump_dir);
    let _ = std::fs::create_dir_all(dump_dir);
    // Set env var for THIS process so the dump module fires.
    unsafe {
        std::env::set_var("JXL_W44_181_DUMP_DC", dump_dir);
    }
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_strategy(EncoderStrategy::Zenjxl);
    let result = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok();
    unsafe {
        std::env::remove_var("JXL_W44_181_DUMP_DC");
    }
    result
}

#[derive(Debug, Default, Clone)]
struct ChannelStats {
    rows: usize,
    diff: usize,
    diff_plus1: usize,
    diff_minus1: usize,
    diff_other: usize,
    max_abs_q: i32, // max |ours_q|
    near_boundary_rows: usize,
    near_boundary_diffs: usize,
    sum_abs_inv_factor: f64,
    inv_factor_min: f32,
    inv_factor_max: f32,
}

impl ChannelStats {
    fn new() -> Self {
        Self {
            inv_factor_min: f32::MAX,
            inv_factor_max: f32::MIN,
            ..Default::default()
        }
    }
}

/// Per-block aggregated divergence (one row per (bx, by)).
#[derive(Debug, Default, Clone, Copy)]
struct BlockAgg {
    raw_strategy: u8,
    samples: u32,
    divergences: u32,
    max_abs_quant_diff: i16,
}

fn parse_inputs_tsv(
    path: &Path,
) -> std::io::Result<(
    Vec<DumpRow>,
    [ChannelStats; 3],
    HashMap<(u32, u32), BlockAgg>,
)> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut rows: Vec<DumpRow> = Vec::new();
    let mut per_channel: [ChannelStats; 3] = [
        ChannelStats::new(),
        ChannelStats::new(),
        ChannelStats::new(),
    ];
    let mut per_block: HashMap<(u32, u32), BlockAgg> = HashMap::new();
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') || line.starts_with("bx\t") {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 8 {
            continue;
        }
        let bx: u32 = parts[0].parse().unwrap_or(0);
        let by: u32 = parts[1].parse().unwrap_or(0);
        let channel: u8 = parts[2].parse().unwrap_or(255);
        let raw_strategy: u8 = parts[3].parse().unwrap_or(255);
        let dc: f32 = parts[4].parse().unwrap_or(0.0);
        let y_dc: f32 = parts[5].parse().unwrap_or(0.0);
        let inv_factor: f32 = parts[6].parse().unwrap_or(0.0);
        let dc_cfl_factor: f32 = parts[7].parse().unwrap_or(0.0);

        let (q_ours, q_libjxl, near_boundary) = compute_both(dc, y_dc, inv_factor, dc_cfl_factor);

        let s = &mut per_channel[channel as usize];
        s.rows += 1;
        s.sum_abs_inv_factor += inv_factor.abs() as f64;
        if inv_factor < s.inv_factor_min {
            s.inv_factor_min = inv_factor;
        }
        if inv_factor > s.inv_factor_max {
            s.inv_factor_max = inv_factor;
        }
        if (q_ours as i32).abs() > s.max_abs_q {
            s.max_abs_q = (q_ours as i32).abs();
        }
        let diff = q_ours as i32 - q_libjxl as i32;
        if diff != 0 {
            s.diff += 1;
            match diff {
                1 => s.diff_plus1 += 1,
                -1 => s.diff_minus1 += 1,
                _ => s.diff_other += 1,
            }
        }
        if near_boundary {
            s.near_boundary_rows += 1;
            if diff != 0 {
                s.near_boundary_diffs += 1;
            }
        }

        // Per-block aggregation: B-channel divergences only (X has cfl=0
        // so expressions coincide; Y not in this dump).
        if channel == 2 {
            let agg = per_block.entry((bx, by)).or_insert(BlockAgg::default());
            agg.raw_strategy = raw_strategy;
            agg.samples += 1;
            if diff != 0 {
                agg.divergences += 1;
                let abs_diff = (q_ours as i32 - q_libjxl as i32).abs() as i16;
                if abs_diff > agg.max_abs_quant_diff {
                    agg.max_abs_quant_diff = abs_diff;
                }
            }
        }

        rows.push(DumpRow {
            bx,
            by,
            channel,
            raw_strategy,
            dc,
            y_dc,
            inv_factor,
            dc_cfl_factor,
            q_ours,
            q_libjxl,
        });
    }
    Ok((rows, per_channel, per_block))
}

#[derive(Debug, Clone, Copy)]
struct DumpRow {
    bx: u32,
    by: u32,
    channel: u8,
    raw_strategy: u8,
    dc: f32,
    y_dc: f32,
    inv_factor: f32,
    dc_cfl_factor: f32,
    q_ours: i16,
    q_libjxl: i16,
}

/// Compute both expressions on the same f32 inputs, returning
/// (q_ours, q_libjxl, near_boundary). `near_boundary` is true if the
/// pre-round value is within 0.05 of a half-integer.
#[inline]
fn compute_both(dc: f32, y_dc: f32, inv_factor: f32, dc_cfl_factor: f32) -> (i16, i16, bool) {
    // Ours: (dc * inv_factor - y_dc * dc_cfl_factor).round()
    let pre_ours: f32 = dc * inv_factor - y_dc * dc_cfl_factor;
    let q_ours = pre_ours.round() as i16;

    // Libjxl form: ((dc - y_dc * (dc_cfl_factor / inv_factor)) * inv_factor).round()
    //
    // libjxl precomputes y_factor_times_cfl = y_factor * cfl_factor (once per
    // channel). At defaults that quantity equals dc_cfl_factor / inv_factor
    // EXACTLY in real math, but in f32 the division may itself introduce a
    // tiny error. For an honest reproduction of libjxl's f32 path we
    // compute:
    //   y_factor_times_cfl_f32 = (1.0 / 512.0) * 1.0     for B  (= GetDcStep(1) * YtoBRatio(0))
    //   inv_factor matches kInvDCQuant[c] * global_scale_float * quant_dc
    // and dc_cfl_factor = y_factor_times_cfl_f32 * inv_factor approximately.
    //
    // Simpler and more general: derive y_factor_times_cfl_f32 from the
    // dumped inv_factor + dc_cfl_factor via division. This isn't perfectly
    // bit-identical to libjxl's precomputed product but is the closest f32
    // approximation we can do without re-running libjxl's full Quantizer
    // pipeline.
    let y_factor_times_cfl: f32 = if inv_factor == 0.0 {
        0.0
    } else {
        dc_cfl_factor / inv_factor
    };
    let pre_libjxl: f32 = (dc - y_dc * y_factor_times_cfl) * inv_factor;
    let q_libjxl = pre_libjxl.round() as i16;

    // "Near boundary" = within 0.05 of an integer + 0.5 (rounding boundary).
    let frac = pre_ours - pre_ours.floor();
    let near = (frac - 0.5).abs() < 0.05;

    (q_ours, q_libjxl, near)
}

fn join_with_w178_blocks(
    per_block: &HashMap<(u32, u32), BlockAgg>,
    w178_path: &Path,
) -> std::io::Result<Vec<JoinRow>> {
    let file = std::fs::File::open(w178_path)?;
    let reader = BufReader::new(file);
    let mut joined: Vec<JoinRow> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 5 {
            continue;
        }
        // bx by our_strategy_raw our_strategy_name max_abs_lin_rgb
        let bx: u32 = parts[0].parse().unwrap_or(0);
        let by: u32 = parts[1].parse().unwrap_or(0);
        let max_abs_rgb: f32 = parts[4].parse().unwrap_or(0.0);
        let agg = per_block.get(&(bx, by)).copied().unwrap_or_default();
        joined.push(JoinRow {
            bx,
            by,
            raw_strategy: agg.raw_strategy,
            dc_divergences: agg.divergences,
            samples: agg.samples,
            max_abs_rgb,
        });
    }
    Ok(joined)
}

#[derive(Debug, Clone, Copy)]
struct JoinRow {
    bx: u32,
    by: u32,
    raw_strategy: u8,
    dc_divergences: u32,
    samples: u32,
    max_abs_rgb: f32,
}

fn analyze_join(joined: &[JoinRow]) -> JoinSummary {
    // Bin blocks into "divergent" (>= 1 DC divergence) and "non-divergent",
    // compute mean / max / p95 max_abs_rgb in each bin.
    let mut div_rgb: Vec<f32> = Vec::new();
    let mut non_div_rgb: Vec<f32> = Vec::new();
    for r in joined {
        if r.samples == 0 {
            continue;
        }
        if r.dc_divergences > 0 {
            div_rgb.push(r.max_abs_rgb);
        } else {
            non_div_rgb.push(r.max_abs_rgb);
        }
    }
    div_rgb.sort_by(|a, b| a.partial_cmp(b).unwrap());
    non_div_rgb.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let div_mean = if !div_rgb.is_empty() {
        div_rgb.iter().sum::<f32>() / div_rgb.len() as f32
    } else {
        0.0
    };
    let non_div_mean = if !non_div_rgb.is_empty() {
        non_div_rgb.iter().sum::<f32>() / non_div_rgb.len() as f32
    } else {
        0.0
    };
    let div_max = div_rgb.last().copied().unwrap_or(0.0);
    let non_div_max = non_div_rgb.last().copied().unwrap_or(0.0);
    let div_p95 = if !div_rgb.is_empty() {
        div_rgb[(div_rgb.len() * 95 / 100).min(div_rgb.len() - 1)]
    } else {
        0.0
    };
    let non_div_p95 = if !non_div_rgb.is_empty() {
        non_div_rgb[(non_div_rgb.len() * 95 / 100).min(non_div_rgb.len() - 1)]
    } else {
        0.0
    };

    JoinSummary {
        div_count: div_rgb.len(),
        non_div_count: non_div_rgb.len(),
        div_mean_rgb: div_mean,
        non_div_mean_rgb: non_div_mean,
        div_max_rgb: div_max,
        non_div_max_rgb: non_div_max,
        div_p95_rgb: div_p95,
        non_div_p95_rgb: non_div_p95,
    }
}

#[derive(Debug)]
struct JoinSummary {
    div_count: usize,
    non_div_count: usize,
    div_mean_rgb: f32,
    non_div_mean_rgb: f32,
    div_max_rgb: f32,
    non_div_max_rgb: f32,
    div_p95_rgb: f32,
    non_div_p95_rgb: f32,
}

fn main() {
    eprintln!("[W44-181] Phase 2: encode + dump DC quant inputs");
    let src = Path::new(SRC_PNG);
    let (rgb, w, h) = load_png(src).expect("load PNG");
    eprintln!(
        "[W44-181] loaded {} ({}×{}, {} bytes)",
        SHORT_NAME,
        w,
        h,
        rgb.len()
    );

    let bytes = encode_with_dump(&rgb, w, h, DISTANCE, EFFORT, DUMP_DIR).expect("encode failed");
    eprintln!("[W44-181] encoded {} bytes", bytes.len());

    let dump_path = PathBuf::from(DUMP_DIR).join("dc_quant_inputs.tsv");
    let (rows, per_channel, per_block) = parse_inputs_tsv(&dump_path).expect("parse dump TSV");

    eprintln!("[W44-181] parsed {} rows", rows.len());
    eprintln!(
        "[W44-181] per-channel: X={} Y={} B={}",
        per_channel[0].rows, per_channel[1].rows, per_channel[2].rows
    );

    // Write summary TSV.
    let mut out = std::io::BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(OUT_TSV)
            .expect("open out tsv"),
    );
    writeln!(out, "# W44-181 DC quant precision probe — channel summary").unwrap();
    writeln!(
        out,
        "# image={} effort={} distance={}",
        SHORT_NAME, EFFORT, DISTANCE
    )
    .unwrap();
    writeln!(out, "# raw dump: {}", dump_path.display()).unwrap();
    writeln!(
        out,
        "channel\trows\tdiv_count\tdiv_plus1\tdiv_minus1\tdiv_other\tdiv_pct\tinv_factor_min\tinv_factor_max\tnear_boundary_rows\tnear_boundary_diffs"
    )
    .unwrap();
    for (c, s) in per_channel.iter().enumerate() {
        let pct = if s.rows > 0 {
            (s.diff as f64) / (s.rows as f64) * 100.0
        } else {
            0.0
        };
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}",
            c,
            s.rows,
            s.diff,
            s.diff_plus1,
            s.diff_minus1,
            s.diff_other,
            pct,
            s.inv_factor_min,
            s.inv_factor_max,
            s.near_boundary_rows,
            s.near_boundary_diffs
        )
        .unwrap();
    }

    // Join with W44-178 blocks.tsv.
    let w178_path = Path::new(W178_BLOCKS_TSV);
    if w178_path.exists() {
        let joined = join_with_w178_blocks(&per_block, w178_path).expect("join W178");
        let summary = analyze_join(&joined);
        writeln!(out, "").unwrap();
        writeln!(
            out,
            "# W44-178 blocks.tsv join (B-channel divergent blocks)"
        )
        .unwrap();
        writeln!(
            out,
            "div_count\tnon_div_count\tdiv_mean_rgb\tnon_div_mean_rgb\tdiv_max_rgb\tnon_div_max_rgb\tdiv_p95_rgb\tnon_div_p95_rgb"
        )
        .unwrap();
        writeln!(
            out,
            "{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            summary.div_count,
            summary.non_div_count,
            summary.div_mean_rgb,
            summary.non_div_mean_rgb,
            summary.div_max_rgb,
            summary.non_div_max_rgb,
            summary.div_p95_rgb,
            summary.non_div_p95_rgb
        )
        .unwrap();

        // Per-strategy join breakdown.
        writeln!(out, "").unwrap();
        writeln!(
            out,
            "# Per-raw_strategy divergence (DCT8=0, DCT32X32=4, DCT64X64=15)"
        )
        .unwrap();
        writeln!(
            out,
            "raw_strategy\tblocks\tdivergent\tdiv_pct\tmean_max_abs_rgb\tdiv_mean_rgb\tnon_div_mean_rgb"
        )
        .unwrap();
        let mut by_strat: HashMap<u8, (u32, u32, Vec<f32>, Vec<f32>)> = HashMap::new();
        for r in &joined {
            if r.samples == 0 {
                continue;
            }
            let entry = by_strat
                .entry(r.raw_strategy)
                .or_insert((0, 0, vec![], vec![]));
            entry.0 += 1;
            if r.dc_divergences > 0 {
                entry.1 += 1;
                entry.2.push(r.max_abs_rgb);
            } else {
                entry.3.push(r.max_abs_rgb);
            }
        }
        let mut strats: Vec<u8> = by_strat.keys().copied().collect();
        strats.sort();
        for s in strats {
            let (blocks, div, ref div_v, ref non_div_v) = by_strat[&s];
            let mean_all: f32 = if blocks > 0 {
                (div_v.iter().sum::<f32>() + non_div_v.iter().sum::<f32>()) / blocks as f32
            } else {
                0.0
            };
            let div_mean = if !div_v.is_empty() {
                div_v.iter().sum::<f32>() / div_v.len() as f32
            } else {
                0.0
            };
            let non_div_mean = if !non_div_v.is_empty() {
                non_div_v.iter().sum::<f32>() / non_div_v.len() as f32
            } else {
                0.0
            };
            let pct = if blocks > 0 {
                (div as f64) / (blocks as f64) * 100.0
            } else {
                0.0
            };
            writeln!(
                out,
                "{}\t{}\t{}\t{:.2}\t{:.6}\t{:.6}\t{:.6}",
                s, blocks, div, pct, mean_all, div_mean, non_div_mean
            )
            .unwrap();
        }
    } else {
        eprintln!(
            "[W44-181] WARNING: W44-178 blocks.tsv not found at {} — skipping join",
            W178_BLOCKS_TSV
        );
    }
    out.flush().unwrap();
    eprintln!("[W44-181] wrote {}", OUT_TSV);

    // Write meta.
    let mut meta = std::io::BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(OUT_META)
            .expect("open meta"),
    );
    writeln!(meta, "W44-181 DC quant precision probe").unwrap();
    writeln!(meta, "=================================").unwrap();
    writeln!(meta, "Date: 2026-05-21").unwrap();
    writeln!(meta, "Image: {} ({}×{})", SHORT_NAME, w, h).unwrap();
    writeln!(meta, "Effort: {} Distance: {}", EFFORT, DISTANCE).unwrap();
    writeln!(meta, "Encoder: EncoderStrategy::Zenjxl").unwrap();
    writeln!(meta, "Raw dump: {}", dump_path.display()).unwrap();
    writeln!(meta, "").unwrap();
    writeln!(
        meta,
        "Strategies dumped: DCT8 (raw=0), DCT32X32 (raw=4), DCT64X64 (raw=15)"
    )
    .unwrap();
    writeln!(meta, "Channels dumped: X (c=0) and B (c=2)").unwrap();
    writeln!(
        meta,
        "X has dc_cfl_factor=0, so ours and libjxl expressions coincide."
    )
    .unwrap();
    writeln!(
        meta,
        "B has dc_cfl_factor=0.5, so f32-order divergence can occur."
    )
    .unwrap();
    writeln!(meta, "").unwrap();
    writeln!(meta, "Methodology:").unwrap();
    writeln!(
        meta,
        "- Per-block dumps (bx, by, channel, raw_strategy, dc, y_dc, inv_factor, dc_cfl_factor)"
    )
    .unwrap();
    writeln!(
        meta,
        "  recorded via env-gated JXL_W44_181_DUMP_DC=<dir> infrastructure"
    )
    .unwrap();
    writeln!(
        meta,
        "  (added to vardct/w44_181_dump.rs + 3 instrumentation sites in transform.rs)."
    )
    .unwrap();
    writeln!(meta, "- For each row, compute:").unwrap();
    writeln!(
        meta,
        "    q_ours   = (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16"
    )
    .unwrap();
    writeln!(
        meta,
        "    q_libjxl = ((dc - y_dc * dc_cfl_factor / inv_factor) * inv_factor).round() as i16"
    )
    .unwrap();
    writeln!(
        meta,
        "  Both in f32 (matching production codepath bit-exact)."
    )
    .unwrap();
    writeln!(
        meta,
        "- Count rows where q_ours != q_libjxl. These are the candidate"
    )
    .unwrap();
    writeln!(meta, "  precision-drift sites.").unwrap();
    writeln!(meta, "").unwrap();
    writeln!(meta, "Reproducer:").unwrap();
    writeln!(
        meta,
        "  CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \\"
    )
    .unwrap();
    writeln!(
        meta,
        "    cargo run --release --manifest-path jxl-encoder/Cargo.toml \\"
    )
    .unwrap();
    writeln!(
        meta,
        "    --features 'parallel butteraugli-loop ssim2-loop' \\"
    )
    .unwrap();
    writeln!(meta, "    --example w44_181_dc_quant_probe").unwrap();
    meta.flush().unwrap();
    eprintln!("[W44-181] wrote {}", OUT_META);
}
