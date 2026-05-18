//! cjxl parity ledger — fast RD-diff toolchain (W44-1).
//!
//! Per the cjxl-parity-never-give-up directive (2026-05-19): build a
//! persistent ledger of every (image, effort, distance) cell where our
//! bytes-vs-quality Pareto loses to cjxl. Each row carries section-level
//! bitstream byte breakdown for both encoders so wedge investigations
//! (W42-1 pattern) can be triggered from the same TSV.
//!
//! Goal: <60s per (image, effort, distance) cell for hypothesis iteration.
//!
//! ## Schema (one row per cell)
//!
//! ```text
//! image | class | width | height | effort | distance
//! | jxl_bytes | cjxl_bytes | jxl_bfly | cjxl_bfly | jxl_ssim2 | cjxl_ssim2
//! | jxl_encode_ms | cjxl_encode_ms
//! | jxl_section_lf_global | jxl_section_lf_groups | jxl_section_hf_global | jxl_section_ac_groups | jxl_section_main_total | jxl_section_patches | jxl_section_other
//! | cjxl_section_lf_global | cjxl_section_lf_groups | cjxl_section_hf_global | cjxl_section_ac_groups | cjxl_section_main_total | cjxl_section_patches | cjxl_section_other
//! | bytes_delta_pct | bfly_delta_pct | ssim2_delta_abs
//! | status | last_investigated_commit | reproducer
//! ```
//!
//! `bytes_delta_pct = (jxl_bytes - cjxl_bytes) / cjxl_bytes * 100`
//! `bfly_delta_pct  = (jxl_bfly  - cjxl_bfly ) / cjxl_bfly  * 100`
//! `ssim2_delta_abs =  jxl_ssim2 - cjxl_ssim2`
//!
//! `status`:
//! - `OPEN` — we lose by more than thresholds (default: bytes +3% AND bfly +3% OR ssim2 -1.0)
//! - `FIXED` — within thresholds or beating cjxl
//!
//! ## Usage
//!
//! ```bash
//! # Seed run (parallel, 8 imgs × 5 efforts × 15 distances = 600 cells)
//! cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' \
//!   --example cjxl_parity_ledger -- --output benchmarks/cjxl_parity_ledger_$(date +%Y-%m-%d).tsv
//!
//! # Update specific cells (after a code change)
//! cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' \
//!   --example cjxl_parity_ledger -- \
//!   --ledger benchmarks/cjxl_parity_ledger_2026-05-19.tsv \
//!   --update --image 1025469.png --effort 7 --distance 1.0
//!
//! # Re-run only OPEN cells
//! cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' \
//!   --example cjxl_parity_ledger -- \
//!   --ledger benchmarks/cjxl_parity_ledger_2026-05-19.tsv \
//!   --update --status OPEN
//! ```

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_bitstream::{Bitstream, ContainerParser, ParseEvent};
use jxl_encoder::api::{LossyConfig, PixelLayout};
use jxl_frame::data::{Toc, TocGroupKind};
use jxl_frame::header::{FrameHeader, FrameType};
use jxl_image::ImageHeader;
use jxl_oxide_common::Bundle;
use rayon::prelude::*;
use rgb::RGB;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::Instant;

// ── Defaults (W38-2 corpus + W38-2 distance grid) ───────────────────────────

const DISTANCES: &[f32] = &[
    0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0,
];

const EFFORTS: &[u8] = &[5, 6, 7, 8, 9];

const CID22_PHOTOS: &[&str] = &[
    "1025469.png",
    "1418519.png",
    "1531677.png",
    "1189261.png",
    "1420710.png",
];

const SCREENSHOTS: &[&str] = &["terminal.png", "codec_wiki.png", "imac_g3.png"];

// Wedge thresholds — a cell is OPEN if we lose by all of these.
const BYTES_DELTA_PCT_THRESHOLD: f64 = 3.0;
const BFLY_DELTA_PCT_THRESHOLD: f64 = 3.0;
const SSIM2_DELTA_THRESHOLD: f64 = -1.0;

// ── External paths ──────────────────────────────────────────────────────────

fn corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/lilith/work/codec-corpus"))
}

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
    PathBuf::from(home).join("work/jxl-efforts/libjxl/build/tools/cjxl")
}

// ── Color helpers (sRGB ↔ linear) ───────────────────────────────────────────

fn srgb_to_linear_f32(s: u8) -> f32 {
    let c = s as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
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

fn rgb_to_linear_img(rgb: &[u8], w: u32, h: u32) -> Img<Vec<RGB<f32>>> {
    let pixels: Vec<RGB<f32>> = rgb
        .chunks(3)
        .map(|c| RGB::new(srgb_to_linear_f32(c[0]), srgb_to_linear_f32(c[1]), srgb_to_linear_f32(c[2])))
        .collect();
    Img::new(pixels, w as usize, h as usize)
}

fn rgb_to_srgb_arr3(rgb: &[u8], w: u32, h: u32) -> Img<Vec<[u8; 3]>> {
    let pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    Img::new(pixels, w as usize, h as usize)
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
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

fn score_jxl(
    bytes: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
) -> Option<(f64, f64)> {
    let (dw, dh, dec_lin) = decode_jxl_linear(bytes)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_img: Img<Vec<RGB<f32>>> = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(
        orig_linear.as_ref(),
        dec_lin_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .ok()?
    .score as f64;

    let dec_srgb: Vec<[u8; 3]> = dec_lin
        .chunks(3)
        .map(|c| [linear_to_srgb_u8(c[0]), linear_to_srgb_u8(c[1]), linear_to_srgb_u8(c[2])])
        .collect();
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;

    Some((bfly, ssim2))
}

// ── Section parsing (mirrors /tmp/jxl_toc_dump pattern from W42-1) ──────────

#[derive(Default, Clone, Copy, Debug)]
struct Sections {
    lf_global: u64,
    lf_groups: u64,
    hf_global: u64,
    ac_groups: u64,
    main_frame_total: u64,
    patches_total: u64, // sum of all ReferenceOnly frames before the main frame
    other_overhead: u64, // file_total - main_frame_total - patches_total
    file_total: u64,
}

fn parse_sections(bytes: &[u8]) -> Option<Sections> {
    let total = bytes.len() as u64;
    // Strip container if present.
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

        let mut lf_global = 0u64;
        let mut hf_global = 0u64;
        let mut all = 0u64;
        let mut lf_groups = 0u64;
        let mut ac_groups = 0u64;
        for grp in toc.iter_bitstream_order() {
            match grp.kind {
                TocGroupKind::All => all += grp.size as u64,
                TocGroupKind::LfGlobal => lf_global += grp.size as u64,
                TocGroupKind::HfGlobal => hf_global += grp.size as u64,
                TocGroupKind::LfGroup(_) => lf_groups += grp.size as u64,
                TocGroupKind::GroupPass { .. } => ac_groups += grp.size as u64,
            }
        }
        let frame_bytes = lf_global + hf_global + all + lf_groups + ac_groups;

        // Classify frame: ReferenceOnly is a patches frame; RegularFrame is main.
        match frame_header.frame_type {
            FrameType::ReferenceOnly => {
                out.patches_total += frame_bytes;
            }
            _ => {
                // Main frame — record breakdown. If multiple regular frames
                // (e.g. animation), sum into out (rare for our use case).
                if all > 0 {
                    // Single-section "All": treat as ac_groups for breakdown purposes.
                    out.ac_groups += all;
                }
                out.lf_global += lf_global;
                out.lf_groups += lf_groups;
                out.hf_global += hf_global;
                out.ac_groups += ac_groups;
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

fn encode_ours(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> Option<(Vec<u8>, f64)> {
    let cfg = LossyConfig::new(distance).with_effort(effort).with_threads(1);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Some((bytes, ms))
}

fn encode_cjxl(
    src_png: &Path,
    distance: f32,
    effort: u8,
    work_dir: &Path,
    tag: &str,
) -> Option<(Vec<u8>, f64)> {
    let out = work_dir.join(format!("{tag}_cjxl_e{effort}_d{:.2}.jxl", distance));
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

// ── Image discovery ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct ImageEntry {
    name: String,
    class: &'static str,
    path: PathBuf,
}

fn discover_images(filter: &Option<Vec<String>>) -> Vec<ImageEntry> {
    let corpus = corpus_dir();
    let cid_dir = corpus.join("CID22/CID22-512/validation");
    let sc_dir = corpus.join("gb82-sc");
    let mut entries: Vec<ImageEntry> = Vec::new();
    for name in CID22_PHOTOS {
        let p = cid_dir.join(name);
        if p.exists() {
            entries.push(ImageEntry {
                name: name.to_string(),
                class: "photo",
                path: p,
            });
        }
    }
    for name in SCREENSHOTS {
        let p = sc_dir.join(name);
        if p.exists() {
            entries.push(ImageEntry {
                name: name.to_string(),
                class: "screenshot",
                path: p,
            });
        }
    }
    if let Some(filt) = filter {
        entries.retain(|e| filt.iter().any(|n| e.name == *n));
    }
    entries
}

// ── Row schema ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Row {
    image: String,
    class: &'static str,
    width: u32,
    height: u32,
    effort: u8,
    distance: f32,
    jxl_bytes: u64,
    cjxl_bytes: u64,
    jxl_bfly: f64,
    cjxl_bfly: f64,
    jxl_ssim2: f64,
    cjxl_ssim2: f64,
    jxl_encode_ms: f64,
    cjxl_encode_ms: f64,
    jxl_sections: Sections,
    cjxl_sections: Sections,
    bytes_delta_pct: f64,
    bfly_delta_pct: f64,
    ssim2_delta_abs: f64,
    status: String,
    last_investigated_commit: String,
    reproducer: String,
}

fn compute_status(bytes_delta: f64, bfly_delta: f64, ssim2_delta: f64) -> &'static str {
    // OPEN if we lose by >3% bytes AND (>3% bfly OR <-1.0 ssim2).
    if bytes_delta > BYTES_DELTA_PCT_THRESHOLD
        && (bfly_delta > BFLY_DELTA_PCT_THRESHOLD || ssim2_delta < SSIM2_DELTA_THRESHOLD)
    {
        "OPEN"
    } else {
        "FIXED"
    }
}

fn make_reproducer(image: &str, effort: u8, distance: f32) -> String {
    format!(
        "cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' --example cjxl_parity_ledger -- --update --image {image} --effort {effort} --distance {distance}"
    )
}

fn build_row(
    img: &ImageEntry,
    effort: u8,
    distance: f32,
    rgb: &[u8],
    w: u32,
    h: u32,
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    work_dir: &Path,
    git_rev: &str,
) -> Option<Row> {
    let (j_bytes, j_ms) = encode_ours(rgb, w, h, distance, effort)?;
    let (c_bytes, c_ms) = encode_cjxl(&img.path, distance, effort, work_dir, &img.name)?;
    let (j_bfly, j_ssim2) = score_jxl(&j_bytes, orig_linear, orig_srgb, w, h)?;
    let (c_bfly, c_ssim2) = score_jxl(&c_bytes, orig_linear, orig_srgb, w, h)?;
    let j_sec = parse_sections(&j_bytes).unwrap_or_default();
    let c_sec = parse_sections(&c_bytes).unwrap_or_default();

    let bytes_delta_pct = if !c_bytes.is_empty() {
        (j_bytes.len() as f64 - c_bytes.len() as f64) / c_bytes.len() as f64 * 100.0
    } else {
        0.0
    };
    let bfly_delta_pct = if c_bfly > 1e-6 {
        (j_bfly - c_bfly) / c_bfly * 100.0
    } else {
        0.0
    };
    let ssim2_delta_abs = j_ssim2 - c_ssim2;
    let status = compute_status(bytes_delta_pct, bfly_delta_pct, ssim2_delta_abs).to_string();
    let reproducer = make_reproducer(&img.name, effort, distance);

    Some(Row {
        image: img.name.clone(),
        class: img.class,
        width: w,
        height: h,
        effort,
        distance,
        jxl_bytes: j_bytes.len() as u64,
        cjxl_bytes: c_bytes.len() as u64,
        jxl_bfly: j_bfly,
        cjxl_bfly: c_bfly,
        jxl_ssim2: j_ssim2,
        cjxl_ssim2: c_ssim2,
        jxl_encode_ms: j_ms,
        cjxl_encode_ms: c_ms,
        jxl_sections: j_sec,
        cjxl_sections: c_sec,
        bytes_delta_pct,
        bfly_delta_pct,
        ssim2_delta_abs,
        status,
        last_investigated_commit: git_rev.to_string(),
        reproducer,
    })
}

// ── TSV I/O ─────────────────────────────────────────────────────────────────

fn row_header() -> String {
    "image\tclass\twidth\theight\teffort\tdistance\t\
     jxl_bytes\tcjxl_bytes\tjxl_bfly\tcjxl_bfly\tjxl_ssim2\tcjxl_ssim2\t\
     jxl_encode_ms\tcjxl_encode_ms\t\
     jxl_lf_global\tjxl_lf_groups\tjxl_hf_global\tjxl_ac_groups\tjxl_main_total\tjxl_patches\tjxl_other\t\
     cjxl_lf_global\tcjxl_lf_groups\tcjxl_hf_global\tcjxl_ac_groups\tcjxl_main_total\tcjxl_patches\tcjxl_other\t\
     bytes_delta_pct\tbfly_delta_pct\tssim2_delta_abs\t\
     status\tlast_investigated_commit\treproducer\n"
        .to_string()
}

fn row_tsv(r: &Row) -> String {
    format!(
        "{image}\t{class}\t{w}\t{h}\t{e}\t{d:.4}\t\
         {jb}\t{cb}\t{jbfly:.6}\t{cbfly:.6}\t{jss:.4}\t{css:.4}\t\
         {jms:.2}\t{cms:.2}\t\
         {jlfg}\t{jlfgs}\t{jhfg}\t{jacg}\t{jmt}\t{jpt}\t{joth}\t\
         {clfg}\t{clfgs}\t{chfg}\t{cacg}\t{cmt}\t{cpt}\t{coth}\t\
         {bdp:.3}\t{fdp:.3}\t{sda:.4}\t\
         {st}\t{commit}\t{repro}\n",
        image = r.image,
        class = r.class,
        w = r.width,
        h = r.height,
        e = r.effort,
        d = r.distance,
        jb = r.jxl_bytes,
        cb = r.cjxl_bytes,
        jbfly = r.jxl_bfly,
        cbfly = r.cjxl_bfly,
        jss = r.jxl_ssim2,
        css = r.cjxl_ssim2,
        jms = r.jxl_encode_ms,
        cms = r.cjxl_encode_ms,
        jlfg = r.jxl_sections.lf_global,
        jlfgs = r.jxl_sections.lf_groups,
        jhfg = r.jxl_sections.hf_global,
        jacg = r.jxl_sections.ac_groups,
        jmt = r.jxl_sections.main_frame_total,
        jpt = r.jxl_sections.patches_total,
        joth = r.jxl_sections.other_overhead,
        clfg = r.cjxl_sections.lf_global,
        clfgs = r.cjxl_sections.lf_groups,
        chfg = r.cjxl_sections.hf_global,
        cacg = r.cjxl_sections.ac_groups,
        cmt = r.cjxl_sections.main_frame_total,
        cpt = r.cjxl_sections.patches_total,
        coth = r.cjxl_sections.other_overhead,
        bdp = r.bytes_delta_pct,
        fdp = r.bfly_delta_pct,
        sda = r.ssim2_delta_abs,
        st = r.status,
        commit = r.last_investigated_commit,
        repro = r.reproducer,
    )
}

fn write_atomic(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("tsv.tmp");
    {
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp)
            .expect("open tmp");
        f.write_all(contents.as_bytes()).expect("write");
        f.sync_all().ok();
    }
    std::fs::rename(&tmp, path).expect("atomic mv");
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
struct CellKey {
    image: String,
    effort: u8,
    distance_x10000: i32, // rounded to 4 decimals for stable equality
}

fn cell_key(image: &str, effort: u8, distance: f32) -> CellKey {
    CellKey {
        image: image.to_string(),
        effort,
        distance_x10000: (distance * 10000.0).round() as i32,
    }
}

/// Read existing ledger rows keyed by (image, effort, distance).
/// Returns the header line + a map keyed by CellKey containing the raw TSV row.
fn read_ledger(path: &Path) -> (Option<String>, HashMap<CellKey, String>) {
    let mut map: HashMap<CellKey, String> = HashMap::new();
    let mut header: Option<String> = None;
    if let Ok(s) = std::fs::read_to_string(path) {
        for (i, line) in s.lines().enumerate() {
            if i == 0 {
                header = Some(line.to_string());
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 6 {
                continue;
            }
            let image = cols[0].to_string();
            let effort: u8 = cols[4].parse().unwrap_or(0);
            let distance: f32 = cols[5].parse().unwrap_or(0.0);
            map.insert(cell_key(&image, effort, distance), line.to_string());
        }
    }
    (header, map)
}

fn write_ledger(path: &Path, header: &str, rows: &HashMap<CellKey, String>) {
    let mut keys: Vec<&CellKey> = rows.keys().collect();
    keys.sort_by(|a, b| {
        a.image
            .cmp(&b.image)
            .then(a.effort.cmp(&b.effort))
            .then(a.distance_x10000.cmp(&b.distance_x10000))
    });
    let mut s = String::with_capacity(rows.len() * 256 + header.len() + 1);
    s.push_str(header);
    s.push('\n');
    for k in keys {
        s.push_str(&rows[k]);
        s.push('\n');
    }
    write_atomic(path, &s);
}

// ── Args ────────────────────────────────────────────────────────────────────

struct Args {
    output: PathBuf,
    output_meta: PathBuf,
    threads: usize,
    distances: Vec<f32>,
    efforts: Vec<u8>,
    images: Option<Vec<String>>,
    update: bool,
    status_filter: Option<String>,
    smoke: bool,
}

fn parse_args() -> Args {
    let mut output: Option<PathBuf> = None;
    let mut ledger: Option<PathBuf> = None;
    let mut threads: usize = 0;
    let mut distances: Vec<f32> = Vec::new();
    let mut efforts: Vec<u8> = Vec::new();
    let mut images: Option<Vec<String>> = None;
    let mut update = false;
    let mut status_filter: Option<String> = None;
    let mut smoke = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--output" => output = Some(PathBuf::from(it.next().expect("--output VALUE"))),
            "--ledger" => ledger = Some(PathBuf::from(it.next().expect("--ledger VALUE"))),
            "--threads" => {
                threads = it
                    .next()
                    .expect("--threads VALUE")
                    .parse()
                    .expect("threads must be uint")
            }
            "--distances" => {
                distances = it
                    .next()
                    .expect("--distances LIST")
                    .split(',')
                    .map(|s| s.parse().expect("float"))
                    .collect()
            }
            "--distance" => {
                let v: f32 = it
                    .next()
                    .expect("--distance VALUE")
                    .parse()
                    .expect("float");
                distances = vec![v];
            }
            "--efforts" => {
                efforts = it
                    .next()
                    .expect("--efforts LIST")
                    .split(',')
                    .map(|s| s.parse().expect("u8"))
                    .collect()
            }
            "--effort" => {
                let v: u8 = it.next().expect("--effort VALUE").parse().expect("u8");
                efforts = vec![v];
            }
            "--images" => {
                images = Some(
                    it.next()
                        .expect("--images LIST")
                        .split(',')
                        .map(|s| s.to_string())
                        .collect(),
                )
            }
            "--image" => {
                let v = it.next().expect("--image VALUE").to_string();
                images = Some(vec![v]);
            }
            "--update" => update = true,
            "--status" => status_filter = Some(it.next().expect("--status VALUE").to_string()),
            "--smoke" => smoke = true,
            other => panic!("unknown arg: {other}"),
        }
    }
    if distances.is_empty() {
        distances = DISTANCES.to_vec();
    }
    if efforts.is_empty() {
        efforts = EFFORTS.to_vec();
    }
    if smoke {
        images = Some(vec!["1025469.png".to_string()]);
        distances = vec![0.4, 1.0, 2.0];
        efforts = vec![5, 7];
    }
    let output = ledger.clone().or(output).unwrap_or_else(|| {
        PathBuf::from(format!(
            "benchmarks/cjxl_parity_ledger_{}.tsv",
            chrono_today()
        ))
    });
    let output_meta = output.with_extension("meta");
    Args {
        output,
        output_meta,
        threads,
        distances,
        efforts,
        images,
        update,
        status_filter,
        smoke,
    }
}

fn chrono_today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days = (secs / 86400) as i64;
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }

    let images = discover_images(&args.images);
    if images.is_empty() {
        eprintln!("No images found. Check CODEC_CORPUS_DIR.");
        std::process::exit(2);
    }

    // Working dir for cjxl outputs.
    let work_dir = std::env::temp_dir().join(format!("cjxl_parity_ledger_{}", std::process::id()));
    std::fs::create_dir_all(&work_dir).expect("create workdir");

    let git_rev = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            // jj workspaces don't have a .git dir; fall back to `jj log`.
            Command::new("jj")
                .args(["log", "-r", "@", "-T", "commit_id", "--no-graph"])
                .output()
                .ok()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "unknown".into());

    // Load existing ledger (for --update or to dedup seed runs).
    let (_existing_header, mut existing_rows) = read_ledger(&args.output);
    let mut header_line = row_header().trim_end().to_string();
    // If the existing file has a different header, take precedence on the existing
    // (so a re-run preserves manually-added columns if any). For now, overwrite.
    header_line.clone_from(&row_header().trim_end().to_string());

    // Build cell list.
    let mut cells: Vec<(ImageEntry, u8, f32)> = Vec::new();
    if args.update {
        // Update mode: re-encode cells that match the filters.
        // If --status is set, take it from existing ledger; else from CLI image/effort/distance.
        if let Some(status) = &args.status_filter {
            for (k, line) in existing_rows.iter() {
                let cols: Vec<&str> = line.split('\t').collect();
                let row_status = cols.get(31).copied().unwrap_or("");
                if row_status != status {
                    continue;
                }
                if let Some(img) = images.iter().find(|i| i.name == k.image) {
                    let dist = k.distance_x10000 as f32 / 10000.0;
                    cells.push((img.clone(), k.effort, dist));
                }
            }
        } else {
            for img in &images {
                for &e in &args.efforts {
                    for &d in &args.distances {
                        cells.push((img.clone(), e, d));
                    }
                }
            }
        }
    } else {
        // Seed mode: every (image, effort, distance) — but skip cells already present.
        for img in &images {
            for &e in &args.efforts {
                for &d in &args.distances {
                    let k = cell_key(&img.name, e, d);
                    if !existing_rows.contains_key(&k) {
                        cells.push((img.clone(), e, d));
                    }
                }
            }
        }
    }

    eprintln!(
        "cjxl parity ledger: {} cells to compute (update={}, status_filter={:?})",
        cells.len(),
        args.update,
        args.status_filter
    );
    eprintln!("  output: {}", args.output.display());
    eprintln!("  meta:   {}", args.output_meta.display());
    eprintln!("  git_rev: {}", git_rev);

    let started = Instant::now();

    // Cache decoded images per (image path).
    let image_data: HashMap<String, (Vec<u8>, u32, u32, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>)> = images
        .iter()
        .filter_map(|img| {
            let (rgb, w, h) = load_png(&img.path)?;
            let lin = rgb_to_linear_img(&rgb, w, h);
            let sr = rgb_to_srgb_arr3(&rgb, w, h);
            Some((img.name.clone(), (rgb, w, h, lin, sr)))
        })
        .collect();

    // Parallel encode. Use a Mutex on the rows map for atomic per-cell writes.
    let rows_mtx = Mutex::new(&mut existing_rows);
    let completed = std::sync::atomic::AtomicUsize::new(0);
    let total = cells.len();

    cells.par_iter().for_each(|(img, e, d)| {
        let entry = match image_data.get(&img.name) {
            Some(x) => x,
            None => return,
        };
        let (rgb, w, h, lin, sr) = entry;
        let row = match build_row(img, *e, *d, rgb, *w, *h, lin, sr, &work_dir, &git_rev) {
            Some(r) => r,
            None => {
                eprintln!("  ! cell failed: {} e{} d{}", img.name, e, d);
                return;
            }
        };
        let line = row_tsv(&row);
        let key = cell_key(&row.image, row.effort, row.distance);
        let mut guard = rows_mtx.lock().unwrap();
        guard.insert(key, line.trim_end().to_string());
        // Write the ledger atomically every N cells to survive crashes.
        let n = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n % 16 == 0 || n == total {
            write_ledger(&args.output, &header_line, &guard);
            eprintln!(
                "  [{}/{}] last: {} e{} d{} status={} bytes_delta={:.2}% bfly_delta={:.2}%",
                n, total, row.image, row.effort, row.distance, row.status,
                row.bytes_delta_pct, row.bfly_delta_pct
            );
        }
    });

    // Final write.
    {
        let guard = rows_mtx.lock().unwrap();
        write_ledger(&args.output, &header_line, &guard);
    }

    let wall = started.elapsed().as_secs_f64();
    let n_rows = existing_rows.len();
    let cjxl_ver = Command::new(cjxl_bin())
        .arg("--version")
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .to_string()
        })
        .unwrap_or_else(|| "unknown".into());
    let host = Command::new("hostname")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let meta = format!(
        "# cjxl parity ledger\n\
         git_rev: {git_rev}\n\
         host: {host}\n\
         cjxl_version: {cjxl_ver}\n\
         distances: {:?}\n\
         efforts: {:?}\n\
         images: {:?}\n\
         corpus: {}\n\
         wall_seconds: {:.1}\n\
         total_rows: {}\n\
         cells_this_run: {}\n\
         update_mode: {}\n\
         status_filter: {:?}\n\
         smoke: {}\n\
         decoder: jxl-oxide (request_color_encoding=srgb_linear)\n\
         metrics: butteraugli_linear (default params), fast_ssim2 (sRGB u8 from linear)\n\
         section_parser: inline (jxl-frame::data::Toc, ContainerParser-stripped)\n\
         wedge_thresholds: bytes_delta_pct > {} AND (bfly_delta_pct > {} OR ssim2_delta_abs < {})\n\
         reproducer (seed): cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' --example cjxl_parity_ledger -- --output <path>\n\
         reproducer (update single cell): cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' --example cjxl_parity_ledger -- --ledger <path> --update --image <name> --effort <N> --distance <F>\n",
        args.distances,
        args.efforts,
        images.iter().map(|i| &i.name).collect::<Vec<_>>(),
        corpus_dir().display(),
        wall,
        n_rows,
        cells.len(),
        args.update,
        args.status_filter,
        args.smoke,
        BYTES_DELTA_PCT_THRESHOLD,
        BFLY_DELTA_PCT_THRESHOLD,
        SSIM2_DELTA_THRESHOLD,
    );
    write_atomic(&args.output_meta, &meta);

    // Cleanup workdir.
    let _ = std::fs::remove_dir_all(&work_dir);

    eprintln!(
        "\nDone in {:.1}s. Wrote {} cells. Total rows now: {}.",
        wall,
        cells.len(),
        n_rows
    );
    eprintln!("Output: {}", args.output.display());
    eprintln!("Meta:   {}", args.output_meta.display());

    // Quick summary by status.
    let mut open = 0usize;
    let mut fixed = 0usize;
    for line in existing_rows.values() {
        let cols: Vec<&str> = line.split('\t').collect();
        match cols.get(31).copied().unwrap_or("") {
            "OPEN" => open += 1,
            "FIXED" => fixed += 1,
            _ => {}
        }
    }
    eprintln!("Summary: {} OPEN, {} FIXED (of {})", open, fixed, n_rows);
}
