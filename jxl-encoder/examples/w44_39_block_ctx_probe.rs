//! W44-39 codec_wiki block context map probe.
//!
//! Audit-only. For each (image, effort, distance) cell:
//!   1. Encode with ours and with cjxl.
//!   2. Parse both .jxl files using jxl-frame's low-level Bundle API.
//!   3. Extract `HfBlockContext` from the LfGlobalVarDct:
//!       - num_block_clusters
//!       - qf_thresholds.len()
//!       - block_ctx_map.len()
//!   4. Emit a TSV row per cell with our vs cjxl counts.
//!
//! Hypothesis H1 from W44-37: codec_wiki's hf_global overspend is caused by
//! our block context map having too many contexts (13) vs cjxl's. Measuring
//! that count is the prerequisite to any cluster-tuning chunk.
//!
//! Read-only — no source changes, no encoder tuning.
//!
//! Run:
//! ```bash
//! cargo run -p jxl-encoder --release \
//!   --example w44_39_block_ctx_probe -- \
//!   --output benchmarks/codec_wiki_block_ctx_2026-05-19.tsv
//! ```

use jxl_bitstream::{Bitstream, ContainerParser, ParseEvent};
use jxl_encoder::api::{LossyConfig, PixelLayout};
use jxl_frame::data::{LfGlobal, LfGlobalParams, Toc};
use jxl_frame::header::{Encoding, FrameHeader, FrameType};
use jxl_image::ImageHeader;
use jxl_oxide_common::Bundle;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ── Cells (mirrors W44-37) ──────────────────────────────────────────────────

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
        if a == "--output"
            && let Some(p) = args.next()
        {
            return PathBuf::from(p);
        }
    }
    PathBuf::from("benchmarks/codec_wiki_block_ctx_2026-05-19.tsv")
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
struct CtxStats {
    bytes: usize,
    num_block_clusters: u32,
    num_qf_thresholds: usize,
    block_ctx_map_len: usize,
    error: Option<String>,
}

fn probe_block_ctx(jxl: &[u8]) -> CtxStats {
    let mut s = CtxStats {
        bytes: jxl.len(),
        ..Default::default()
    };
    let cs = unbox_codestream(jxl);
    let mut br = Bitstream::new(&cs);

    let image_header = match ImageHeader::parse(&mut br, ()) {
        Ok(h) => h,
        Err(e) => {
            s.error = Some(format!("image_header: {e}"));
            return s;
        }
    };
    if image_header.metadata.colour_encoding.want_icc()
        && let Err(e) = jxl_color::icc::read_icc(&mut br)
    {
        s.error = Some(format!("icc: {e}"));
        return s;
    }
    if br.zero_pad_to_byte().is_err() {
        s.error = Some("zero_pad_to_byte after header failed".to_string());
        return s;
    }

    // Walk frames until we find the main (non-ReferenceOnly) frame.
    let mut frame_idx = 0u32;
    loop {
        let frame_header = match FrameHeader::parse(&mut br, &image_header) {
            Ok(h) => h,
            Err(e) => {
                s.error = Some(format!("frame_header[{frame_idx}]: {e}"));
                return s;
            }
        };
        let toc = match Toc::parse(&mut br, &frame_header) {
            Ok(t) => t,
            Err(e) => {
                s.error = Some(format!("toc[{frame_idx}]: {e}"));
                return s;
            }
        };

        // Each section is byte-aligned at start.
        if br.zero_pad_to_byte().is_err() {
            s.error = Some("zero_pad after toc".to_string());
            return s;
        }

        // For ReferenceOnly (patches) frames, skip the whole frame.
        if frame_header.frame_type == FrameType::ReferenceOnly {
            let toc_bytes = toc.total_byte_size() as usize;
            if br.skip_bits(toc_bytes * 8).is_err() {
                s.error = Some("skip_bits past patches frame".to_string());
                return s;
            }
            if frame_header.is_last {
                s.error = Some("only patches frames found".to_string());
                return s;
            }
            frame_idx += 1;
            continue;
        }

        // Main frame. Parse the LfGlobal section.
        // If single-entry TOC, just start parsing from the current position.
        // If multi-entry TOC, the first bitstream-order group is LfGlobal; iter to that.
        if frame_header.encoding != Encoding::VarDct {
            s.error = Some(format!(
                "main frame encoding != VarDct ({:?})",
                frame_header.encoding
            ));
            return s;
        }

        // The LfGlobal section starts at the current bitstream position (byte-aligned).
        let params = LfGlobalParams::new(&image_header, &frame_header, None, false);
        match LfGlobal::<i32>::parse(&mut br, params) {
            Ok(lf_global) => {
                if let Some(vardct) = lf_global.vardct.as_ref() {
                    s.num_block_clusters = vardct.hf_block_ctx.num_block_clusters;
                    s.num_qf_thresholds = vardct.hf_block_ctx.qf_thresholds.len();
                    s.block_ctx_map_len = vardct.hf_block_ctx.block_ctx_map.len();
                } else {
                    s.error = Some("frame has no vardct".to_string());
                }
                return s;
            }
            Err(e) => {
                s.error = Some(format!("lf_global parse: {e}"));
                return s;
            }
        }
    }
}

fn main() {
    let work = PathBuf::from("/tmp/w44_39_block_ctx_probe");
    fs::create_dir_all(&work).expect("mkdir tmp");
    let out_path = output_path();
    if let Some(p) = out_path.parent() {
        fs::create_dir_all(p).ok();
    }

    let mut rows = Vec::new();
    rows.push(
        "image\teffort\tdistance\tours_bytes\tcjxl_bytes\tours_n_clusters\tcjxl_n_clusters\
        \tours_n_qf_thresh\tcjxl_n_qf_thresh\tours_ctxmap_len\tcjxl_ctxmap_len\
        \tours_error\tcjxl_error"
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
        let ours_stat = probe_block_ctx(&ours_bytes);
        let cjxl_stat = probe_block_ctx(&cjxl_bytes);
        eprintln!(
            "  ours: n_clusters={} qf_thr={} ctxmap={} bytes={} {}",
            ours_stat.num_block_clusters,
            ours_stat.num_qf_thresholds,
            ours_stat.block_ctx_map_len,
            ours_stat.bytes,
            ours_stat.error.as_deref().unwrap_or("")
        );
        eprintln!(
            "  cjxl: n_clusters={} qf_thr={} ctxmap={} bytes={} {}",
            cjxl_stat.num_block_clusters,
            cjxl_stat.num_qf_thresholds,
            cjxl_stat.block_ctx_map_len,
            cjxl_stat.bytes,
            cjxl_stat.error.as_deref().unwrap_or("")
        );
        rows.push(format!(
            "{name}\t{effort}\t{distance}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            ours_stat.bytes,
            cjxl_stat.bytes,
            ours_stat.num_block_clusters,
            cjxl_stat.num_block_clusters,
            ours_stat.num_qf_thresholds,
            cjxl_stat.num_qf_thresholds,
            ours_stat.block_ctx_map_len,
            cjxl_stat.block_ctx_map_len,
            ours_stat.error.unwrap_or_default(),
            cjxl_stat.error.unwrap_or_default(),
        ));
    }

    fs::write(&out_path, rows.join("\n") + "\n").expect("write tsv");
    eprintln!("Wrote {}", out_path.display());
}
