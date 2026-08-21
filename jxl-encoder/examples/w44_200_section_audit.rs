//! W44-200 Phase 1: per-section byte audit on 3637739 cid22 e7 d=4
//! Pareto-loser vs cjxl libjxl reference.
//!
//! Follow-on to W44-199 (`c8470c54`) which ruled out AC strategy
//! selection as the root cause of the +6.24% bytes on 3637739 e7 d=4
//! (strategy agreement 98.4%, qac matches 98.9%, Y/B nzeros at parity).
//! With strategy AT PARITY, the +6.24% byte gap must live in some
//! OTHER section: LfGlobal (DC quantizer + patches + splines + noise),
//! LfGroup (DC tokens + DC tree), HfGlobal (AC metadata + cluster
//! headers + context map), or AcGroups (per-coefficient value
//! encoding via ANS).
//!
//! Methodology:
//!
//! 1. Encode each of {3637739, 1418519} at e7 d=4 with:
//!    - OURS Zenjxl default (production)
//!    - OURS Libjxl strategy (`EncoderStrategy::Libjxl`)
//!    - cjxl libjxl v0.12.0 reference (`-e 7 -d 4`)
//!
//! 2. Parse encoded bytes via `jxl_frame::data::Toc` to extract
//!    per-section sizes — works for cjxl and OURS since both emit
//!    bit-compatible JXL bitstreams. For single-group images (≤256
//!    blocks per axis, i.e. ≤2048px), there's ONE combined section
//!    (`TocGroupKind::All`) in the TOC. To get sub-section breakdown
//!    on the OURS side, also run with `JXL_W44_200_SECTION_DUMP=<file>`
//!    which captures the per-sub-section bit counts (LfGlobal /
//!    LfGroup / HfGlobal / AcGroups + frame header + TOC overhead)
//!    inside `encode_two_pass_to_writer`.
//!
//! 3. Emit per-cell delta table. For each (image, encoder) the dump
//!    includes:
//!    - file_total_bytes (full output incl. container)
//!    - main_frame_bytes (sum of all main-frame TOC entries)
//!    - patches_frame_bytes (sum of all ReferenceOnly frames before
//!      the main frame; non-zero only when patches engaged)
//!    - other_overhead_bytes (file headers, ICC, container boxes)
//!    - per-sub-section bits (OURS only) via the env-var dump
//!
//! 4. Diagnostic conclusion: identifies the dominant overspending
//!    section on 3637739 vs 1418519. Suggests Phase 2 candidates.
//!
//! Output:
//!   benchmarks/w44_200_section_audit_2026-05-22.tsv
//!   benchmarks/w44_200_section_audit_2026-05-22.meta
//!   /tmp/w44_200_oursdump_{zenjxl,libjxl}.tsv  (OURS sub-section dumps)
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features '__expert butteraugli-loop ssim2-loop parallel' \
//!     --example w44_200_section_audit

use jxl_bitstream::{Bitstream, ContainerParser, ParseEvent};
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use jxl_frame::data::{Toc, TocGroupKind};
use jxl_frame::header::{FrameHeader, FrameType};
use jxl_image::ImageHeader;
use jxl_oxide_common::Bundle;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const IMG_3637739: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png";
const IMG_1418519: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png";
const DISTANCE: f32 = 4.0;
const EFFORT: u8 = 7;

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

// ── Section parsing (copied from cjxl_parity_ledger.rs, simplified) ─────────

#[derive(Default, Clone, Debug)]
struct Sections {
    /// `TocGroupKind::All` — single combined section for single-group images.
    all: u64,
    lf_global: u64,
    lf_groups: u64,
    hf_global: u64,
    ac_groups: u64,
    /// Per-AC-group sizes (one per `TocGroupKind::GroupPass`, in TOC order).
    ac_per_group: Vec<u32>,
    main_frame_total: u64,
    /// Sum of ReferenceOnly frames before the main frame.
    patches_total: u64,
    /// file_total - main_frame_total - patches_total (file headers / ICC / container)
    other_overhead: u64,
    file_total: u64,
}

fn parse_sections(bytes: &[u8]) -> Option<Sections> {
    let total = bytes.len() as u64;
    let mut parser = ContainerParser::new();
    let mut codestream: Vec<u8> = Vec::new();
    for ev in parser.feed_bytes(bytes) {
        match ev {
            Ok(ParseEvent::Codestream(payload)) => codestream.extend_from_slice(payload),
            Ok(_) => {}
            Err(_) => return None,
        }
    }
    if codestream.is_empty() {
        codestream = bytes.to_vec();
    }
    let mut bitstream = Bitstream::new(&codestream);
    let image_header = ImageHeader::parse(&mut bitstream, ()).ok()?;
    if image_header.metadata.colour_encoding.want_icc() {
        let _ = jxl_color::icc::read_icc(&mut bitstream).ok()?;
    }
    bitstream.zero_pad_to_byte().ok()?;

    let mut out = Sections {
        file_total: total,
        ..Default::default()
    };

    let mut frame_idx: usize = 0;
    loop {
        let frame_header = match FrameHeader::parse(&mut bitstream, &image_header) {
            Ok(h) => h,
            Err(_) => break,
        };
        let toc = match Toc::parse(&mut bitstream, &frame_header) {
            Ok(t) => t,
            Err(_) => break,
        };

        let mut all = 0u64;
        let mut lf_global = 0u64;
        let mut hf_global = 0u64;
        let mut lf_groups = 0u64;
        let mut ac_groups = 0u64;
        // Collect per-AC-group sizes in their natural (group_idx, pass_idx)
        // order, so the TOC's bitstream order doesn't muddle the per-group
        // analysis. We iterate the TOC and remember `(group_idx, pass_idx, size)`.
        let mut ac_per_group_raw: Vec<(u32, u32, u32)> = Vec::new();
        for grp in toc.iter_bitstream_order() {
            match grp.kind {
                TocGroupKind::All => all += grp.size as u64,
                TocGroupKind::LfGlobal => lf_global += grp.size as u64,
                TocGroupKind::HfGlobal => hf_global += grp.size as u64,
                TocGroupKind::LfGroup(_) => lf_groups += grp.size as u64,
                TocGroupKind::GroupPass {
                    pass_idx,
                    group_idx,
                } => {
                    ac_groups += grp.size as u64;
                    ac_per_group_raw.push((group_idx, pass_idx, grp.size));
                }
            }
        }
        // Sort by (group, pass) to get a stable per-group ordering.
        ac_per_group_raw.sort_by_key(|&(g, p, _)| (g, p));
        let ac_per_group_main: Vec<u32> = ac_per_group_raw.iter().map(|&(_, _, s)| s).collect();
        let frame_bytes = all + lf_global + hf_global + lf_groups + ac_groups;

        match frame_header.frame_type {
            FrameType::ReferenceOnly => {
                out.patches_total += frame_bytes;
            }
            _ => {
                out.all += all;
                out.lf_global += lf_global;
                out.lf_groups += lf_groups;
                out.hf_global += hf_global;
                out.ac_groups += ac_groups;
                out.ac_per_group = ac_per_group_main;
                out.main_frame_total += frame_bytes;
            }
        }

        let toc_total = toc.total_byte_size() as u64;
        bitstream.zero_pad_to_byte().ok()?;
        bitstream.skip_bits(toc_total as usize * 8).ok()?;

        if frame_header.is_last {
            break;
        }
        frame_idx += 1;
        if frame_idx > 32 {
            break;
        }
    }

    let accounted = out.main_frame_total + out.patches_total;
    out.other_overhead = out.file_total.saturating_sub(accounted);
    Some(out)
}

// ── Encoders ────────────────────────────────────────────────────────────────

fn encode_ours_with_strategy(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    strategy: EncoderStrategy,
    dump_label: Option<&str>,
    section_dump: Option<&Path>,
    hfg_dump: Option<&Path>,
    co_dump: Option<&Path>,
) -> Option<(Vec<u8>, f64)> {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_strategy(strategy);
    // SAFETY: single-threaded harness, called sequentially per cell.
    unsafe {
        if let (Some(label), Some(file)) = (dump_label, section_dump) {
            std::env::set_var("JXL_W44_200_SECTION_DUMP", file);
            std::env::set_var("JXL_W44_200_LABEL", label);
        } else {
            std::env::remove_var("JXL_W44_200_SECTION_DUMP");
            std::env::remove_var("JXL_W44_200_LABEL");
        }
        if let Some(file) = hfg_dump {
            std::env::set_var("JXL_W44_200_HFGLOBAL_DUMP", file);
        } else {
            std::env::remove_var("JXL_W44_200_HFGLOBAL_DUMP");
        }
        if let Some(file) = co_dump {
            std::env::set_var("JXL_W44_200_COEFFORDER_DUMP", file);
        } else {
            std::env::remove_var("JXL_W44_200_COEFFORDER_DUMP");
        }
    }
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    unsafe {
        std::env::remove_var("JXL_W44_200_SECTION_DUMP");
        std::env::remove_var("JXL_W44_200_LABEL");
        std::env::remove_var("JXL_W44_200_HFGLOBAL_DUMP");
        std::env::remove_var("JXL_W44_200_COEFFORDER_DUMP");
    }
    Some((bytes, ms))
}

fn encode_cjxl(src_png: &Path, distance: f32, effort: u8) -> Option<(Vec<u8>, f64)> {
    let tmp = std::env::temp_dir().join(format!(
        "w44_200_cjxl_{}_e{}_d{}.jxl",
        src_png.file_stem().unwrap().to_string_lossy(),
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
    let start = Instant::now();
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&tmp).ok()?;
    let _ = std::fs::remove_file(&tmp);
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Some((bytes, ms))
}

// ── Driver ──────────────────────────────────────────────────────────────────

struct CellResult {
    label: String,
    encoder: String,
    bytes: usize,
    encode_ms: f64,
    sections: Sections,
}

fn run_cell(
    label: &str,
    src_path: &Path,
    distance: f32,
    effort: u8,
    section_dump_file: &Path,
    hfg_dump_file: &Path,
    co_dump_file: &Path,
) -> Vec<CellResult> {
    let (rgb, w, h) = load_png(src_path).unwrap_or_else(|| panic!("load {}", src_path.display()));
    eprintln!("=== {} ({}×{}) e{} d={} ===", label, w, h, effort, distance);
    let mut results = Vec::new();

    // OURS Zenjxl (production default)
    if let Some((bytes, ms)) = encode_ours_with_strategy(
        &rgb,
        w,
        h,
        distance,
        effort,
        EncoderStrategy::Zenjxl,
        Some(&format!("{label}_zenjxl")),
        Some(section_dump_file),
        Some(hfg_dump_file),
        Some(co_dump_file),
    ) {
        let sections = parse_sections(&bytes).unwrap_or_default();
        eprintln!(
            "  zenjxl   : {} bytes  ({:.1} ms)  lf_glob={} lf_grp={} hf_glob={} ac_grp={} all={}",
            bytes.len(),
            ms,
            sections.lf_global,
            sections.lf_groups,
            sections.hf_global,
            sections.ac_groups,
            sections.all
        );
        results.push(CellResult {
            label: label.to_string(),
            encoder: "zenjxl".into(),
            bytes: bytes.len(),
            encode_ms: ms,
            sections,
        });
    }

    // OURS Libjxl strategy (cjxl-parity mode)
    if let Some((bytes, ms)) = encode_ours_with_strategy(
        &rgb,
        w,
        h,
        distance,
        effort,
        EncoderStrategy::Libjxl,
        Some(&format!("{label}_libjxl")),
        Some(section_dump_file),
        Some(hfg_dump_file),
        Some(co_dump_file),
    ) {
        let sections = parse_sections(&bytes).unwrap_or_default();
        eprintln!(
            "  libjxl   : {} bytes  ({:.1} ms)  lf_glob={} lf_grp={} hf_glob={} ac_grp={} all={}",
            bytes.len(),
            ms,
            sections.lf_global,
            sections.lf_groups,
            sections.hf_global,
            sections.ac_groups,
            sections.all
        );
        results.push(CellResult {
            label: label.to_string(),
            encoder: "ours_libjxl".into(),
            bytes: bytes.len(),
            encode_ms: ms,
            sections,
        });
    }

    // cjxl libjxl v0.12.0 reference
    if let Some((bytes, ms)) = encode_cjxl(src_path, distance, effort) {
        let sections = parse_sections(&bytes).unwrap_or_default();
        eprintln!(
            "  cjxl     : {} bytes  ({:.1} ms)  lf_glob={} lf_grp={} hf_glob={} ac_grp={} all={}",
            bytes.len(),
            ms,
            sections.lf_global,
            sections.lf_groups,
            sections.hf_global,
            sections.ac_groups,
            sections.all
        );
        results.push(CellResult {
            label: label.to_string(),
            encoder: "cjxl".into(),
            bytes: bytes.len(),
            encode_ms: ms,
            sections,
        });
    }

    results
}

fn main() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let bench_dir = workspace.join("benchmarks");
    std::fs::create_dir_all(&bench_dir).ok();
    let tsv_path = bench_dir.join("w44_200_section_audit_2026-05-22.tsv");
    let dump_path = PathBuf::from("/tmp/w44_200_ours_subsection_dump.tsv");
    let hfg_dump_path = PathBuf::from("/tmp/w44_200_ours_hfglobal_dump.tsv");
    let co_dump_path = PathBuf::from("/tmp/w44_200_ours_coefforder_dump.tsv");
    let _ = std::fs::remove_file(&dump_path);
    let _ = std::fs::remove_file(&hfg_dump_path);
    let _ = std::fs::remove_file(&co_dump_path);

    let mut tsv = File::create(&tsv_path).expect("create tsv");
    writeln!(
        tsv,
        "label\tencoder\tbytes\tencode_ms\tlf_global\tlf_groups\thf_global\tac_groups\tall\t\
         main_frame_total\tpatches_total\tother_overhead\tfile_total\tac_per_group"
    )
    .unwrap();

    // Optional CLI override: `<image.png> <distance> [label]` audits an
    // arbitrary cell instead of the historical W44-200 pair (added for
    // the 2026-08-21 mechanism-2 9016 investigation; the no-arg default
    // is unchanged).
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cells: Vec<(String, String, f32, u8)> = if args.len() >= 2 {
        let label = args
            .get(2)
            .cloned()
            .unwrap_or_else(|| "CLI_CELL".to_string());
        let effort: u8 = args
            .get(3)
            .map_or(EFFORT, |v| v.parse().expect("effort u8"));
        vec![(
            label,
            args[0].clone(),
            args[1].parse().expect("distance f32"),
            effort,
        )]
    } else {
        vec![
            (
                "3637739_LOSER".to_string(),
                IMG_3637739.to_string(),
                DISTANCE,
                EFFORT,
            ),
            (
                "1418519_WINNER".to_string(),
                IMG_1418519.to_string(),
                DISTANCE,
                EFFORT,
            ),
        ]
    };
    let mut all_results = Vec::new();
    for (label, src, distance, effort) in &cells {
        let (label, src) = (label.as_str(), src.as_str());
        let results = run_cell(
            label,
            Path::new(src),
            *distance,
            *effort,
            &dump_path,
            &hfg_dump_path,
            &co_dump_path,
        );
        for r in &results {
            let ac_per_group_str = r
                .sections
                .ac_per_group
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(
                tsv,
                "{}\t{}\t{}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                r.label,
                r.encoder,
                r.bytes,
                r.encode_ms,
                r.sections.lf_global,
                r.sections.lf_groups,
                r.sections.hf_global,
                r.sections.ac_groups,
                r.sections.all,
                r.sections.main_frame_total,
                r.sections.patches_total,
                r.sections.other_overhead,
                r.sections.file_total,
                ac_per_group_str,
            )
            .unwrap();
        }
        all_results.extend(results);
    }
    drop(tsv);
    eprintln!("\nTSV written: {}", tsv_path.display());
    if dump_path.exists() {
        eprintln!("OURS sub-section dump: {}", dump_path.display());
    }
    if hfg_dump_path.exists() {
        eprintln!("OURS HfGlobal dump:    {}", hfg_dump_path.display());
    }
    if co_dump_path.exists() {
        eprintln!("OURS CoeffOrder dump:  {}", co_dump_path.display());
    }

    // Print per-cell delta table.
    print_delta_table(&all_results);
}

fn print_delta_table(results: &[CellResult]) {
    eprintln!();
    eprintln!("════════════════════════════════════════════════════════════════════");
    eprintln!("PER-CELL Δ vs cjxl (positive = ours larger, negative = ours smaller)");
    eprintln!("════════════════════════════════════════════════════════════════════");

    let labels: Vec<&str> = ["3637739_LOSER", "1418519_WINNER"].into_iter().collect();
    for label in labels {
        let zenjxl = results
            .iter()
            .find(|r| r.label == label && r.encoder == "zenjxl");
        let libjxl_ours = results
            .iter()
            .find(|r| r.label == label && r.encoder == "ours_libjxl");
        let cjxl = results
            .iter()
            .find(|r| r.label == label && r.encoder == "cjxl");
        let Some(cjxl) = cjxl else {
            continue;
        };
        eprintln!();
        eprintln!("─── {label} (cjxl baseline) ───");
        let cjxl_ac_str: Vec<String> = cjxl
            .sections
            .ac_per_group
            .iter()
            .map(|s| format!("{:>5}", s))
            .collect();
        eprintln!(
            "  cjxl     : {:>8} B  (lf_glob={:>5} lf_grp={:>5} hf_glob={:>6} ac_grp={:>6})  ac=[{}]",
            cjxl.bytes,
            cjxl.sections.lf_global,
            cjxl.sections.lf_groups,
            cjxl.sections.hf_global,
            cjxl.sections.ac_groups,
            cjxl_ac_str.join(","),
        );
        for (name, r) in [("zenjxl", zenjxl), ("libjxl_o", libjxl_ours)] {
            let Some(r) = r else {
                continue;
            };
            let total_delta_pct = (r.bytes as f64 - cjxl.bytes as f64) / cjxl.bytes as f64 * 100.0;
            let ac_delta_str: Vec<String> = r
                .sections
                .ac_per_group
                .iter()
                .zip(cjxl.sections.ac_per_group.iter())
                .map(|(o, c)| format!("{:+5}", *o as i64 - *c as i64))
                .collect();
            eprintln!(
                "  {:8} : {:>8} B ({:+5.2}%)  all={:>6} (Δ={:+6}) lf_glob={:>5} (Δ={:+5}) lf_grp={:>5} (Δ={:+5}) hf_glob={:>6} (Δ={:+6}) ac_grp={:>6} (Δ={:+6}) acΔ=[{}]",
                name,
                r.bytes,
                total_delta_pct,
                r.sections.all,
                (r.sections.all as i64) - (cjxl.sections.all as i64),
                r.sections.lf_global,
                (r.sections.lf_global as i64) - (cjxl.sections.lf_global as i64),
                r.sections.lf_groups,
                (r.sections.lf_groups as i64) - (cjxl.sections.lf_groups as i64),
                r.sections.hf_global,
                (r.sections.hf_global as i64) - (cjxl.sections.hf_global as i64),
                r.sections.ac_groups,
                (r.sections.ac_groups as i64) - (cjxl.sections.ac_groups as i64),
                ac_delta_str.join(","),
            );
        }
    }
    eprintln!();
    eprintln!("NOTE: For single-group images (512×512), cjxl emits ONE combined");
    eprintln!("section under TocGroupKind::All (counted in `all` column). OURS likewise");
    eprintln!("emits one combined section for single-group/single-pass — sub-section");
    eprintln!("breakdown for OURS comes from /tmp/w44_200_ours_subsection_dump.tsv.");
}
