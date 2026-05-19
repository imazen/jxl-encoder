//! W44-42 codec_wiki post-cluster ANS / LZ77 / HybridUint audit (follow-on
//! to W44-37 + W44-39).
//!
//! W44-37 found the +3-7% bytes wedge concentrated in hf_global (+35-50%
//! wider than cjxl on 5 OPEN cells). W44-39 RULED OUT the block context
//! map count hypothesis: ours = cjxl on `num_block_clusters`,
//! `qf_thresholds.len()`, and `block_ctx_map.len()` at every cell.
//!
//! This probe drives BYTE ATTRIBUTION + post-cluster ANS state by re-parsing
//! the codestream with `jxl-frame::Toc` and emitting per-section byte counts:
//!
//!   - LfGlobal bytes
//!   - LfGroup(s) bytes
//!   - HfGlobal bytes
//!   - PassGroup bytes
//!   - frame header + TOC bytes
//!
//! Plus from `LfGlobal.vardct.hf_block_ctx` (already proven at parity in
//! W44-39): num_block_clusters, qf_thresholds.len(), ctx_map.len()
//!
//! Where the +35-50% hf_global overspend lives is the lever for the
//! next chunk. Possible levers (downstream of block ctx count):
//!   - hf_dist Decoder num_clusters (post-cluster count, 1..=128)
//!   - LZ77 enabled in hf_dist?
//!   - HybridUint config (4,2,0) vs other
//!   - dequant_matrices bytes (`all_default=1` bit vs custom matrices)
//!   - coeff_orders bytes (custom orders vs natural)
//!
//! Read-only — no source/src changes; subprocess-driven for cjxl.
//!
//! Run:
//! ```bash
//! cargo run -p jxl-encoder --release \
//!   --example w44_42_ans_lz77_probe -- \
//!   --output benchmarks/codec_wiki_section_attribution_2026-05-18.tsv
//! ```

use jxl_bitstream::{Bitstream, ContainerParser, ParseEvent};
use jxl_encoder::api::{LossyConfig, PixelLayout};
use jxl_frame::data::{HfGlobal, HfGlobalParams, LfGlobal, LfGlobalParams, Toc, TocGroupKind};
use jxl_frame::header::{Encoding, FrameHeader, FrameType};
use jxl_image::ImageHeader;
use jxl_oxide_common::Bundle;
use jxl_threadpool::JxlThreadPool;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ── Cells mirror W44-37/W44-39 ─────────────────────────────────────────────

const CELLS: &[(&str, u8, f32)] = &[
    ("codec_wiki.png", 6, 4.0), // OPEN +3.20% bytes
    ("codec_wiki.png", 7, 3.0), // OPEN +4.14%
    ("codec_wiki.png", 7, 4.0), // OPEN +7.34%
    ("codec_wiki.png", 7, 5.0), // OPEN +6.16%
    ("codec_wiki.png", 7, 6.0), // OPEN +6.67%
    // Reference cells (parity baseline)
    ("codec_wiki.png", 7, 1.0),
    ("codec_wiki.png", 7, 2.0),
];

fn corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/lilith/work/codec-corpus"))
}

fn cjxl_bin() -> PathBuf {
    std::env::var("CJXL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl"))
}

fn output_path() -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--output" {
            if let Some(p) = args.next() {
                return PathBuf::from(p);
            }
        }
    }
    PathBuf::from("benchmarks/codec_wiki_section_attribution_2026-05-18.tsv")
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn encode_ours(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> Option<Vec<u8>> {
    LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .ok()
}

fn encode_cjxl(src_png: &Path, distance: f32, effort: u8, work: &Path) -> Option<Vec<u8>> {
    let out = work.join(format!(
        "cjxl_{}_{}_e{}_d{:.2}.jxl",
        std::process::id(),
        src_png.file_stem().unwrap().to_string_lossy(),
        effort,
        distance
    ));
    let _ = std::fs::remove_file(&out);
    let status = Command::new(cjxl_bin())
        .arg(src_png)
        .arg(&out)
        .args(["-e", &effort.to_string()])
        .args(["-d", &format!("{distance}")])
        .args(["--num_threads", "1"])
        .arg("--quiet")
        .output()
        .ok()?;
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
    Some(bytes)
}

fn unbox_codestream(jxl: &[u8]) -> Vec<u8> {
    let mut parser = ContainerParser::new();
    let mut out: Vec<u8> = Vec::new();
    for ev in parser.feed_bytes(jxl) {
        match ev {
            Ok(ParseEvent::Codestream(payload)) => out.extend_from_slice(payload),
            Ok(_) => {}
            Err(_) => return jxl.to_vec(),
        }
    }
    if out.is_empty() { jxl.to_vec() } else { out }
}

#[derive(Default, Debug)]
struct SectionStats {
    total_bytes: usize,
    // codestream bytes (after demux)
    codestream_bytes: usize,
    // Frame-level
    frame_count: u32,
    // Main (non-patches) frame:
    main_frame_offset: usize,
    main_frame_total_bytes: usize, // total_byte_size of main frame TOC
    main_frame_header_bytes: usize,
    main_frame_lfglobal_bytes: usize,
    main_frame_lfgroups_bytes: usize,
    main_frame_hfglobal_bytes: usize,
    main_frame_passgroups_bytes: usize,
    // Patches frame (ReferenceOnly) total bytes if present
    patches_frame_bytes: usize,
    // hf_block_ctx (already at parity per W44-39, included for sanity)
    num_block_clusters: u32,
    num_qf_thresholds: usize,
    block_ctx_map_len: usize,
    // HfGlobal internals (when we can parse it):
    hfglobal_parsed: bool,
    num_passes: u32,
    num_hf_presets: u32,
    used_orders_pass0: u32,
    used_orders_count_pass0: u32, // popcount(used_orders)
    dequant_bytes: usize,         // dequant matrix set encoded bytes
    // Errors
    error: Option<String>,
}

fn probe(jxl: &[u8]) -> SectionStats {
    let mut s = SectionStats {
        total_bytes: jxl.len(),
        ..Default::default()
    };
    let cs = unbox_codestream(jxl);
    s.codestream_bytes = cs.len();
    let mut br = Bitstream::new(&cs);

    let image_header = match ImageHeader::parse(&mut br, ()) {
        Ok(h) => h,
        Err(e) => {
            s.error = Some(format!("image_header: {e}"));
            return s;
        }
    };
    if image_header.metadata.colour_encoding.want_icc() {
        if let Err(e) = jxl_color::icc::read_icc(&mut br) {
            s.error = Some(format!("icc: {e}"));
            return s;
        }
    }
    if br.zero_pad_to_byte().is_err() {
        s.error = Some("zero_pad_to_byte after header failed".to_string());
        return s;
    }

    loop {
        let frame_start_bits = br.num_read_bits();
        let frame_header = match FrameHeader::parse(&mut br, &image_header) {
            Ok(h) => h,
            Err(e) => {
                s.error = Some(format!("frame_header[{}]: {e}", s.frame_count));
                return s;
            }
        };
        let toc = match Toc::parse(&mut br, &frame_header) {
            Ok(t) => t,
            Err(e) => {
                s.error = Some(format!("toc[{}]: {e}", s.frame_count));
                return s;
            }
        };
        if br.zero_pad_to_byte().is_err() {
            s.error = Some("zero_pad after toc".to_string());
            return s;
        }
        let after_header_bits = br.num_read_bits();
        let frame_header_bytes = (after_header_bits - frame_start_bits + 7) / 8;
        let frame_total_bytes = toc.total_byte_size();

        if frame_header.frame_type == FrameType::ReferenceOnly {
            s.patches_frame_bytes += frame_header_bytes + frame_total_bytes;
            // Skip past the entire patches frame.
            if br.skip_bits(frame_total_bytes * 8).is_err() {
                s.error = Some("skip_bits past patches frame".to_string());
                return s;
            }
            s.frame_count += 1;
            if frame_header.is_last {
                s.error = Some("only patches frames found".to_string());
                return s;
            }
            continue;
        }

        if frame_header.encoding != Encoding::VarDct {
            s.error = Some(format!(
                "main frame encoding != VarDct ({:?})",
                frame_header.encoding
            ));
            return s;
        }

        s.main_frame_offset = frame_start_bits / 8;
        s.main_frame_total_bytes = frame_total_bytes;
        s.main_frame_header_bytes = frame_header_bytes;

        // Walk sections in BITSTREAM order so we can correlate br.num_read_bits()
        // with each group's bytes. The TOC `iter_bitstream_order()` returns
        // groups in the order they appear on the wire.
        let mut lf_global: Option<LfGlobal<i32>> = None;
        for (sec_idx, group) in toc.iter_bitstream_order().enumerate() {
            let sec_start_bits = br.num_read_bits();

            if group.kind == TocGroupKind::LfGlobal && lf_global.is_none() {
                // Parse LfGlobal so we can extract HfBlockContext and use it
                // later for HfGlobalParams.
                let params = LfGlobalParams::new(&image_header, &frame_header, None, false);
                match LfGlobal::<i32>::parse(&mut br, params) {
                    Ok(lfg) => {
                        if let Some(vardct) = lfg.vardct.as_ref() {
                            s.num_block_clusters = vardct.hf_block_ctx.num_block_clusters;
                            s.num_qf_thresholds = vardct.hf_block_ctx.qf_thresholds.len();
                            s.block_ctx_map_len = vardct.hf_block_ctx.block_ctx_map.len();
                        }
                        lf_global = Some(lfg);
                        // pad-to-byte after section.
                        let _ = br.zero_pad_to_byte();
                        let consumed_bits = br.num_read_bits() - sec_start_bits;
                        let consumed_bytes = (consumed_bits + 7) / 8;
                        s.main_frame_lfglobal_bytes += group.size as usize;
                        if consumed_bytes > group.size as usize {
                            eprintln!(
                                "    warn: LfGlobal parse consumed {} bytes but TOC size is {}",
                                consumed_bytes, group.size
                            );
                        }
                        let remaining = (group.size as usize).saturating_sub(consumed_bytes) * 8;
                        if remaining > 0 && br.skip_bits(remaining).is_err() {
                            s.error = Some("skip after LfGlobal".to_string());
                            return s;
                        }
                    }
                    Err(e) => {
                        s.error = Some(format!("lf_global parse: {e}"));
                        let consumed_bits = br.num_read_bits() - sec_start_bits;
                        let consumed_bytes = (consumed_bits + 7) / 8;
                        let remaining = (group.size as usize).saturating_sub(consumed_bytes) * 8;
                        let _ = br.skip_bits(remaining);
                    }
                }
            } else if group.kind == TocGroupKind::HfGlobal && lf_global.is_some() {
                // Read the raw HfGlobal section bytes for manual sub-section attribution.
                let raw_bits = group.size as usize * 8;
                let hf_global_start_byte = sec_start_bits / 8;
                let hf_global_end_byte = hf_global_start_byte + group.size as usize;
                let hf_bytes_slice = if hf_global_end_byte <= cs.len() {
                    cs[hf_global_start_byte..hf_global_end_byte].to_vec()
                } else {
                    Vec::new()
                };

                // Parse the dequant matrix set portion to attribute its bytes.
                let lfg_ref = lf_global.as_ref().unwrap();
                let pool = JxlThreadPool::none();
                let mut hf_br = Bitstream::new(&hf_bytes_slice);
                let dequant_start = hf_br.num_read_bits();
                let params = jxl_vardct::DequantMatrixSetParams::new(
                    image_header.metadata.bit_depth.bits_per_sample(),
                    frame_header.num_lf_groups(),
                    lfg_ref.gmodular.ma_config.as_ref(),
                    None,
                    &pool,
                );
                let dequant_bytes = match jxl_vardct::DequantMatrixSet::parse(&mut hf_br, params) {
                    Ok(_) => {
                        let consumed = hf_br.num_read_bits() - dequant_start;
                        (consumed + 7) / 8
                    }
                    Err(_) => 0,
                };
                s.dequant_bytes = dequant_bytes;

                // After dequant + num_hf_presets, read u2S for used_orders[pass 0].
                let num_groups = frame_header.num_groups();
                let num_histo_bits = num_groups.next_power_of_two().trailing_zeros() as usize;
                // Read the rest manually: num_hf_presets bits, then u2S for used_orders.
                if let Ok(_num_hf_presets_m1) = hf_br.read_bits(num_histo_bits) {
                    if let Ok(sel) = hf_br.read_bits(2) {
                        let used_orders_val: u32 = match sel {
                            0 => 0x5F,
                            1 => 0x13,
                            2 => 0,
                            _ => hf_br.read_bits(13).unwrap_or(0),
                        };
                        s.used_orders_pass0 = used_orders_val;
                        s.used_orders_count_pass0 = used_orders_val.count_ones();
                    }
                }

                // Now parse HfGlobal with the full bitstream (re-using br).
                let params2 = HfGlobalParams::new(
                    &image_header.metadata,
                    &frame_header,
                    lfg_ref,
                    None,
                    &pool,
                );
                match HfGlobal::parse(&mut br, params2) {
                    Ok(hf_global) => {
                        s.hfglobal_parsed = true;
                        s.num_passes = hf_global.hf_passes.len() as u32;
                        s.num_hf_presets = hf_global.num_hf_presets;
                        let _ = br.zero_pad_to_byte();
                        let consumed_bits = br.num_read_bits() - sec_start_bits;
                        let consumed_bytes = (consumed_bits + 7) / 8;
                        s.main_frame_hfglobal_bytes += group.size as usize;
                        let remaining = (group.size as usize).saturating_sub(consumed_bytes) * 8;
                        if remaining > 0 && br.skip_bits(remaining).is_err() {
                            s.error = Some("skip after HfGlobal".to_string());
                            return s;
                        }
                    }
                    Err(e) => {
                        s.error = Some(format!("hf_global parse: {e}"));
                        s.main_frame_hfglobal_bytes += group.size as usize;
                        let consumed_bits = br.num_read_bits() - sec_start_bits;
                        let consumed_bytes = (consumed_bits + 7) / 8;
                        let remaining = (group.size as usize).saturating_sub(consumed_bytes) * 8;
                        let _ = br.skip_bits(remaining);
                    }
                }
                let _ = raw_bits;
            } else {
                // Skip the whole section.
                if br.skip_bits((group.size as usize) * 8).is_err() {
                    s.error = Some(format!("skip section[{sec_idx}]"));
                    return s;
                }
                match group.kind {
                    TocGroupKind::LfGroup(_) => s.main_frame_lfgroups_bytes += group.size as usize,
                    TocGroupKind::HfGlobal => s.main_frame_hfglobal_bytes += group.size as usize,
                    TocGroupKind::GroupPass { .. } => {
                        s.main_frame_passgroups_bytes += group.size as usize
                    }
                    TocGroupKind::LfGlobal => {
                        s.main_frame_lfglobal_bytes += group.size as usize;
                    }
                    TocGroupKind::All => {
                        s.main_frame_lfglobal_bytes += group.size as usize;
                    }
                }
            }
        }

        s.frame_count += 1;
        return s;
    }
}

fn main() {
    let work = PathBuf::from("/tmp/w44_42_ans_lz77_probe");
    fs::create_dir_all(&work).expect("mkdir tmp");
    let out_path = output_path();
    if let Some(p) = out_path.parent() {
        fs::create_dir_all(p).ok();
    }

    let mut rows = Vec::new();
    rows.push(
        "image\teffort\tdistance\
        \tours_total\tcjxl_total\tdelta_bytes\tdelta_pct\
        \tours_patches\tcjxl_patches\
        \tours_lfgroups\tcjxl_lfgroups\
        \tours_lfglobal\tcjxl_lfglobal\
        \tours_hfglobal\tcjxl_hfglobal\tdelta_hfglobal_pct\
        \tours_dequant\tcjxl_dequant\tours_perm_etc\tcjxl_perm_etc\
        \tours_passgroups\tcjxl_passgroups\tdelta_passgroups_pct\
        \tours_n_blkclu\tcjxl_n_blkclu\
        \tours_n_hf_presets\tcjxl_n_hf_presets\
        \tours_err\tcjxl_err"
            .to_string(),
    );

    for &(name, effort, distance) in CELLS {
        let src_png = corpus_dir().join("gb82-sc").join(name);
        let Some((rgb, w, h)) = load_png(&src_png) else {
            eprintln!("skip {}: not found", src_png.display());
            continue;
        };
        eprintln!("== {name} e{effort} d{distance} ({w}×{h}) ==");
        let ours_bytes = match encode_ours(&rgb, w, h, distance, effort) {
            Some(b) => b,
            None => {
                eprintln!("  ours encode failed");
                continue;
            }
        };
        let cjxl_bytes = match encode_cjxl(&src_png, distance, effort, &work) {
            Some(b) => b,
            None => {
                eprintln!("  cjxl encode failed");
                continue;
            }
        };
        let ours = probe(&ours_bytes);
        let cjxl = probe(&cjxl_bytes);

        let delta_bytes = ours.total_bytes as i64 - cjxl.total_bytes as i64;
        let delta_pct = if cjxl.total_bytes > 0 {
            100.0 * delta_bytes as f64 / cjxl.total_bytes as f64
        } else {
            0.0
        };
        let delta_hfglobal_pct = if cjxl.main_frame_hfglobal_bytes > 0 {
            100.0
                * (ours.main_frame_hfglobal_bytes as i64 - cjxl.main_frame_hfglobal_bytes as i64)
                    as f64
                / cjxl.main_frame_hfglobal_bytes as f64
        } else {
            0.0
        };
        let delta_passgroups_pct = if cjxl.main_frame_passgroups_bytes > 0 {
            100.0
                * (ours.main_frame_passgroups_bytes as i64
                    - cjxl.main_frame_passgroups_bytes as i64) as f64
                / cjxl.main_frame_passgroups_bytes as f64
        } else {
            0.0
        };

        let ours_perm_etc = ours
            .main_frame_hfglobal_bytes
            .saturating_sub(ours.dequant_bytes);
        let cjxl_perm_etc = cjxl
            .main_frame_hfglobal_bytes
            .saturating_sub(cjxl.dequant_bytes);
        eprintln!(
            "  ours: total={} patches={} lfgrp={} lfglob={} hfglob={} (dequant={} perm_etc={}) passgrp={} n_blkclu={} n_hfpr={} used_orders=0x{:X} (pop={}) {}",
            ours.total_bytes,
            ours.patches_frame_bytes,
            ours.main_frame_lfgroups_bytes,
            ours.main_frame_lfglobal_bytes,
            ours.main_frame_hfglobal_bytes,
            ours.dequant_bytes,
            ours_perm_etc,
            ours.main_frame_passgroups_bytes,
            ours.num_block_clusters,
            ours.num_hf_presets,
            ours.used_orders_pass0,
            ours.used_orders_count_pass0,
            ours.error.as_deref().unwrap_or("")
        );
        eprintln!(
            "  cjxl: total={} patches={} lfgrp={} lfglob={} hfglob={} (dequant={} perm_etc={}) passgrp={} n_blkclu={} n_hfpr={} used_orders=0x{:X} (pop={}) {}",
            cjxl.total_bytes,
            cjxl.patches_frame_bytes,
            cjxl.main_frame_lfgroups_bytes,
            cjxl.main_frame_lfglobal_bytes,
            cjxl.main_frame_hfglobal_bytes,
            cjxl.dequant_bytes,
            cjxl_perm_etc,
            cjxl.main_frame_passgroups_bytes,
            cjxl.num_block_clusters,
            cjxl.num_hf_presets,
            cjxl.used_orders_pass0,
            cjxl.used_orders_count_pass0,
            cjxl.error.as_deref().unwrap_or("")
        );
        eprintln!(
            "  Δ: total={:+} ({:+.2}%) hfglob={:+.2}% (dequant Δ={:+} perm_etc Δ={:+}) passgrp={:+.2}%",
            delta_bytes,
            delta_pct,
            delta_hfglobal_pct,
            ours.dequant_bytes as i64 - cjxl.dequant_bytes as i64,
            ours_perm_etc as i64 - cjxl_perm_etc as i64,
            delta_passgroups_pct
        );

        rows.push(format!(
            "{name}\t{effort}\t{distance}\t{}\t{}\t{:+}\t{:+.2}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:+.2}\t{}\t{}\t{}\t{}\t{}\t{}\t{:+.2}\t{}\t{}\t{}\t{}\t{}\t{}",
            ours.total_bytes,
            cjxl.total_bytes,
            delta_bytes,
            delta_pct,
            ours.patches_frame_bytes,
            cjxl.patches_frame_bytes,
            ours.main_frame_lfgroups_bytes,
            cjxl.main_frame_lfgroups_bytes,
            ours.main_frame_lfglobal_bytes,
            cjxl.main_frame_lfglobal_bytes,
            ours.main_frame_hfglobal_bytes,
            cjxl.main_frame_hfglobal_bytes,
            delta_hfglobal_pct,
            ours.dequant_bytes,
            cjxl.dequant_bytes,
            ours_perm_etc,
            cjxl_perm_etc,
            ours.main_frame_passgroups_bytes,
            cjxl.main_frame_passgroups_bytes,
            delta_passgroups_pct,
            ours.num_block_clusters,
            cjxl.num_block_clusters,
            ours.num_hf_presets,
            cjxl.num_hf_presets,
            ours.error.unwrap_or_default(),
            cjxl.error.unwrap_or_default(),
        ));
    }

    fs::write(&out_path, rows.join("\n") + "\n").expect("write tsv");
    eprintln!("Wrote {}", out_path.display());
}
