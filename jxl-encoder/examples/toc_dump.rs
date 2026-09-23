//! W45-RECON part 10: minimal per-section TOC dump for parity diffing.
//! Prints LfGlobal/LfGroups/HfGlobal/per-AC-group sizes for each frame.
//!
//! `cargo run --example toc_dump --features __env_var_diagnostics -- file.jxl`

use jxl_bitstream::{Bitstream, ContainerParser, ParseEvent};
use jxl_frame::data::{Toc, TocGroupKind};
use jxl_frame::header::FrameHeader;
use jxl_image::ImageHeader;
use jxl_oxide_common::Bundle;

fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        dump(&path, &bytes);
    }
}

fn dump(path: &str, bytes: &[u8]) {
    let total = bytes.len();
    let mut parser = ContainerParser::new();
    let mut codestream: Vec<u8> = Vec::new();
    for ev in parser.feed_bytes(bytes) {
        match ev {
            Ok(ParseEvent::Codestream(payload)) => codestream.extend_from_slice(payload),
            Ok(_) => {}
            Err(e) => {
                eprintln!("{path}: container parse err {e:?}");
                return;
            }
        }
    }
    if codestream.is_empty() {
        codestream = bytes.to_vec();
    }
    let mut bitstream = Bitstream::new(&codestream);
    let image_header = ImageHeader::parse(&mut bitstream, ()).unwrap();
    if image_header.metadata.colour_encoding.want_icc() {
        let _ = jxl_color::icc::read_icc(&mut bitstream).unwrap();
    }
    bitstream.zero_pad_to_byte().unwrap();

    let mut frame_idx = 0usize;
    loop {
        let frame_header = match FrameHeader::parse(&mut bitstream, &image_header) {
            Ok(h) => h,
            Err(_) => break,
        };
        let toc = match Toc::parse(&mut bitstream, &frame_header) {
            Ok(t) => t,
            Err(_) => break,
        };
        let mut lf_global = 0u64;
        let mut hf_global = 0u64;
        let mut lf_groups = 0u64;
        let mut ac_groups = 0u64;
        let mut per_group: Vec<(u32, u32, u32)> = Vec::new();
        for grp in toc.iter_bitstream_order() {
            match grp.kind {
                TocGroupKind::LfGlobal => lf_global += grp.size as u64,
                TocGroupKind::HfGlobal => hf_global += grp.size as u64,
                TocGroupKind::LfGroup(_) => lf_groups += grp.size as u64,
                TocGroupKind::GroupPass {
                    pass_idx,
                    group_idx,
                } => {
                    ac_groups += grp.size as u64;
                    per_group.push((group_idx, pass_idx, grp.size));
                }
                _ => {}
            }
        }
        per_group.sort_by_key(|&(g, p, _)| (g, p));
        println!(
            "{path} frame{frame_idx} lf_global={lf_global} lf_groups={lf_groups} hf_global={hf_global} ac_total={ac_groups}"
        );
        for (g, p, s) in per_group {
            println!("  ac group={g} pass={p} size={s}");
        }
        // Skip frame data: advance bitstream past all sections.
        let mut consumed = 0u64;
        for grp in toc.iter_bitstream_order() {
            consumed += grp.size as u64;
        }
        // Bitstream is byte-aligned at section boundary; re-create
        // a bitstream over remaining bytes is hard — break after first frame.
        let _ = consumed;
        frame_idx += 1;
        if frame_idx > 0 {
            break;
        }
    }
    println!("{path} file_total={total}");
}
