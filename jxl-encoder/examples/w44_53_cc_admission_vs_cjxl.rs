//! W44-53: Compare patches admitted by ours vs cjxl on codec_wiki.
//!
//! Hypothesis H1: cjxl admits MORE patches via CC enumeration / admission
//! differences in `FindTextLikePatches`.  The 7 % bigger cjxl ref-frame at
//! d=3-6 could be 1-2 more actual patches admitted (not just looser
//! packing).
//!
//! Method:
//! 1. Encode codec_wiki with ours at d∈{3,4,5,6} → run instrumented
//!    detector via W44-20 `LastPatchesDetectStats` to get our
//!    `final_unique` and `final_occurrences`.
//! 2. Encode codec_wiki with cjxl at the same distances.
//! 3. Parse the main frame's LfGlobal of each cjxl output to extract its
//!    `Patches` struct and count `patches.len()` (unique ref-frame patches)
//!    and `sum(patch_targets.len())` (total occurrences).
//! 4. Print a side-by-side TSV.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!       --features __internals --example w44_53_cc_admission_vs_cjxl

use jxl_bitstream::{Bitstream, ContainerParser, ParseEvent};
use jxl_encoder::__internals::take_last_patches_detect_stats;
use jxl_encoder::{LossyConfig, PixelLayout};
use jxl_frame::data::{LfGlobal, LfGlobalParams, Toc, TocGroupKind};
use jxl_frame::header::{FrameHeader, FrameType};
use jxl_image::ImageHeader;
use jxl_oxide_common::Bundle;
use std::io::Write;

struct CjxlPatches {
    unique: usize,
    occurrences: usize,
    /// True if a ReferenceOnly frame was seen before the main frame.
    has_ref_frame: bool,
}

/// Walks frames; for the first RegularFrame, parses its single-section
/// LfGlobal (or the first LfGlobal section in multi-group) and reads its
/// `.patches` field.  cjxl screenshots typically encode the patches as
/// (a) one ReferenceOnly frame holding the ref-image followed by (b) the
/// main frame whose LfGlobal lists positions + ref indices.
fn parse_cjxl_patches(bytes: &[u8]) -> Option<CjxlPatches> {
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

    let mut has_ref = false;
    loop {
        let frame_header = FrameHeader::parse(&mut bitstream, &image_header).ok()?;
        let toc = Toc::parse(&mut bitstream, &frame_header).ok()?;
        let toc_total = toc.total_byte_size() as u64;
        bitstream.zero_pad_to_byte().ok()?;

        match frame_header.frame_type {
            FrameType::ReferenceOnly => {
                has_ref = true;
                // Skip the ref frame's bytes.
                bitstream.skip_bits(toc_total as usize * 8).ok()?;
            }
            _ => {
                // Main frame.  We need the LfGlobal section bytes only.
                // Use a fresh sub-bitstream pointed at the LfGlobal slice.
                // Calculate offset by reading section order.
                let mut lf_global_offset_bytes: usize = 0;
                let mut lf_global_size: u64 = 0;
                let mut found = false;
                // single-entry?  Then the "All" section starts at
                // bitstream's current byte position.
                if toc.is_single_entry() {
                    // The whole frame data is one section "All".
                    lf_global_offset_bytes = 0;
                    lf_global_size = toc_total;
                    found = true;
                } else {
                    let mut off: u64 = 0;
                    for grp in toc.iter_bitstream_order() {
                        if matches!(grp.kind, TocGroupKind::LfGlobal) {
                            lf_global_offset_bytes = off as usize;
                            lf_global_size = grp.size as u64;
                            found = true;
                            break;
                        }
                        off += grp.size as u64;
                    }
                }
                if !found {
                    return None;
                }
                // Reconstruct the codestream byte offset of LfGlobal.
                let cur_bit = bitstream.num_read_bits();
                let cur_byte = cur_bit / 8;
                let lf_start = cur_byte + lf_global_offset_bytes;
                let lf_end = (lf_start + lf_global_size as usize).min(codestream.len());
                let lf_slice = &codestream[lf_start..lf_end];

                let mut lf_bs = Bitstream::new(lf_slice);
                let params =
                    LfGlobalParams::new(&image_header, &frame_header, None, true);
                // i32 sample type is the default for VarDCT.
                let lf_global = LfGlobal::<i32>::parse(&mut lf_bs, params).ok()?;
                let unique = lf_global
                    .patches
                    .as_ref()
                    .map(|p| p.patches.len())
                    .unwrap_or(0);
                let occurrences = lf_global
                    .patches
                    .as_ref()
                    .map(|p| {
                        p.patches
                            .iter()
                            .map(|patch| patch.patch_targets.len())
                            .sum()
                    })
                    .unwrap_or(0);
                return Some(CjxlPatches {
                    unique,
                    occurrences,
                    has_ref_frame: has_ref,
                });
            }
        }

        if frame_header.is_last {
            break;
        }
    }
    None
}

fn cjxl_encode(src: &str, d: f32) -> Option<Vec<u8>> {
    let cjxl_bin = std::env::var("CJXL_BIN").unwrap_or_else(|_| {
        format!(
            "{}/work/jxl-efforts/libjxl/build/tools/cjxl",
            std::env::var("HOME").unwrap()
        )
    });
    let tmp = format!("/tmp/w44_53_cjxl_{}.jxl", std::process::id());
    let status = std::process::Command::new(&cjxl_bin)
        .args(["-e", "7", "-d", &format!("{d}"), src, &tmp, "--quiet"])
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&tmp).ok()?;
    let _ = std::fs::remove_file(&tmp);
    Some(bytes)
}

fn main() {
    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let png_path = format!("{}/work/codec-corpus/gb82-sc/codec_wiki.png", base);

    let img = match image::open(&png_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("failed to open {png_path}: {e}");
            std::process::exit(1);
        }
    };
    let img = img.to_rgb8();
    let (w, h) = (img.width(), img.height());
    let rgb = img.as_raw();

    let distances = [3.0_f32, 4.0, 5.0, 6.0];

    println!(
        "image\tdistance\t\
         ours_bytes\tours_final_unique\tours_final_occ\tours_accepted_ccs\tours_singletons\t\
         cjxl_bytes\tcjxl_unique\tcjxl_occurrences\tcjxl_has_ref_frame\t\
         delta_unique\tdelta_occurrences"
    );
    std::io::stdout().flush().ok();

    for &d in &distances {
        // Drain prior thread-local stats
        let _ = take_last_patches_detect_stats();
        let ours_bytes = match LossyConfig::new(d).encode(rgb, w, h, PixelLayout::Rgb8) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("ours encode d={d}: {e}");
                continue;
            }
        };
        let det = take_last_patches_detect_stats();

        let cjxl_bytes = match cjxl_encode(&png_path, d) {
            Some(b) => b,
            None => {
                eprintln!("cjxl encode d={d} failed");
                continue;
            }
        };
        let cjxl = parse_cjxl_patches(&cjxl_bytes);

        let (cjxl_unique, cjxl_occ, cjxl_has_ref) = match cjxl {
            Some(c) => (c.unique, c.occurrences, c.has_ref_frame),
            None => (0, 0, false),
        };
        let (ours_unique, ours_occ, ours_acc, ours_sing) = match det {
            Some(s) => (
                s.final_unique as usize,
                s.final_occurrences,
                s.accepted_ccs as usize,
                s.singletons_dropped as usize,
            ),
            None => (0, 0, 0, 0),
        };

        let dunique = cjxl_unique as i64 - ours_unique as i64;
        let docc = cjxl_occ as i64 - ours_occ as i64;
        println!(
            "codec_wiki.png\t{d:.2}\t\
             {}\t{}\t{}\t{}\t{}\t\
             {}\t{}\t{}\t{}\t\
             {}\t{}",
            ours_bytes.len(),
            ours_unique,
            ours_occ,
            ours_acc,
            ours_sing,
            cjxl_bytes.len(),
            cjxl_unique,
            cjxl_occ,
            cjxl_has_ref,
            dunique,
            docc,
        );
        std::io::stdout().flush().ok();
    }
}
