// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Weekly conformance sweep: random real-image encodes checked by three
//! decoders, libjxl v0.12's djxl, upstream jxl-rs (crates.io `jxl`, latest
//! release) and our fork zenjxl-decoder.
//!
//! Each cell is derived from `(seed, index)` alone, so a failure replays with
//! `--seed S --only N`. A cell crops a corpus image to a random size, picks a
//! pixel layout, lossless or lossy, an effort and option set, and sometimes
//! streaming input, an ICC profile or EXIF/XMP. Every decoder must:
//!
//! - decode the whole image at the right dimensions;
//! - lossless: reproduce every sample exactly at the layout's precision;
//! - lossy: reproduce alpha exactly, agree with djxl (Rust decoders), and stay
//!   under a loose error bound against the source (a garbage detector, not a
//!   quality gate);
//! - return an embedded ICC profile byte for byte.
//!
//! Streaming input must also produce the same bytes as the one-shot encode.
//!
//! Verdicts: `OK`; `REJECTED` (an encode error listed in
//! [`expected_rejection`]); `DECODER` when exactly one Rust decoder disagrees
//! while djxl and the other agree (a decoder finding, logged under
//! `decoder_findings/` for `docs/DECODER_FINDINGS.md`, not a failure); `FAIL`
//! otherwise (encode errors, djxl failures, or both Rust decoders against
//! djxl). Results go to `results.tsv`; each failure keeps its stream, source
//! pixels and a replay line under `failures/`. Exit status is 1 when any cell
//! fails.
//!
//! ```text
//! cargo run --release --manifest-path tools/conformance-sweep/Cargo.toml -- \
//!     --images <dir-or-png>... --out <dir> [--icc <dir>] [--seed N] [--cells N]
//!     [--minutes N] [--threads N] [--only N]
//! ```
//! The seed defaults to the week number, so weekly runs rotate coverage.
//! `DJXL_PATH` and `CJXL_PATH` select djxl and cjxl (resolved and
//! version-checked by `jxl_encoder::test_helpers`); cjxl encodes the
//! reference for the PQ, HLG and BT.709 float layouts.

use jxl_encoder::api::{EncoderStrategy, SectionedTrees};
use jxl_encoder::{
    ChromaSubsampling, ImageMetadata, LosslessConfig, LossyConfig, Lz77Method, PixelLayout,
    ProgressiveMode,
};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// ── Deterministic randomness ────────────────────────────────────────────────

struct Rng(u64);
impl Rng {
    fn for_cell(seed: u64, index: u64) -> Self {
        let mut r = Rng(seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03);
        for _ in 0..4 {
            r.next();
        }
        r
    }
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

// ── Inputs ──────────────────────────────────────────────────────────────────

struct Source {
    name: String,
    w: usize,
    h: usize,
    rgb: Vec<u8>,
}

fn load_sources(paths: &[PathBuf]) -> Vec<Source> {
    let mut files = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(p)
                .unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
                .map(|e| e.unwrap().path())
                .filter(|q| q.extension().is_some_and(|x| x.eq_ignore_ascii_case("png")))
                .collect();
            entries.sort();
            files.extend(entries);
        } else {
            files.push(p.clone());
        }
    }
    assert!(!files.is_empty(), "no input images");
    files
        .iter()
        .map(|f| {
            let img = image::open(f)
                .unwrap_or_else(|e| panic!("decode {}: {e}", f.display()))
                .to_rgb8();
            Source {
                name: f.file_name().unwrap().to_string_lossy().into_owned(),
                w: img.width() as usize,
                h: img.height() as usize,
                rgb: img.into_raw(),
            }
        })
        .collect()
}

struct Icc {
    name: String,
    data: Vec<u8>,
    gray: bool,
}

/// RGB- and gray-class ICC profiles (header colour space at bytes 16..20).
fn load_icc(dir: Option<&Path>) -> Vec<Icc> {
    let Some(dir) = dir else { return Vec::new() };
    let mut out: Vec<Icc> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(|e| {
            let p = e.unwrap().path();
            let data = std::fs::read(&p).ok()?;
            if data.len() < 132 || &data[36..40] != b"acsp" {
                return None;
            }
            let gray = match &data[16..20] {
                b"RGB " => false,
                b"GRAY" => true,
                _ => return None,
            };
            Some(Icc {
                name: p.file_name().unwrap().to_string_lossy().into_owned(),
                data,
                gray,
            })
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    assert!(
        !out.is_empty(),
        "no RGB or gray ICC profiles in {}",
        dir.display()
    );
    out
}

// ── Layouts and pixels ──────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    U8,
    U16,
    F16,
    F32,
}

const LAYOUTS: &[PixelLayout] = &[
    PixelLayout::Rgb8,
    PixelLayout::Rgba8,
    PixelLayout::Bgr8,
    PixelLayout::Bgra8,
    PixelLayout::Gray8,
    PixelLayout::GrayAlpha8,
    PixelLayout::Rgb16,
    PixelLayout::Rgba16,
    PixelLayout::Gray16,
    PixelLayout::GrayAlpha16,
    PixelLayout::RgbLinearF32,
    PixelLayout::RgbaLinearF32,
    PixelLayout::GrayLinearF32,
    PixelLayout::GrayAlphaLinearF32,
    PixelLayout::RgbLinearF16,
    PixelLayout::RgbaLinearF16,
    PixelLayout::GrayLinearF16,
    PixelLayout::GrayAlphaLinearF16,
    PixelLayout::RgbPqF32,
    PixelLayout::RgbaPqF32,
    PixelLayout::RgbHlgF32,
    PixelLayout::RgbaHlgF32,
    PixelLayout::RgbBt709F32,
    PixelLayout::RgbaBt709F32,
];

/// (channels, sample kind, bgr order, linear-light samples).
fn describe(layout: PixelLayout) -> (usize, Kind, bool, bool) {
    use PixelLayout::*;
    match layout {
        Rgb8 => (3, Kind::U8, false, false),
        Rgba8 => (4, Kind::U8, false, false),
        Bgr8 => (3, Kind::U8, true, false),
        Bgra8 => (4, Kind::U8, true, false),
        Gray8 => (1, Kind::U8, false, false),
        GrayAlpha8 => (2, Kind::U8, false, false),
        Rgb16 => (3, Kind::U16, false, false),
        Rgba16 => (4, Kind::U16, false, false),
        Gray16 => (1, Kind::U16, false, false),
        GrayAlpha16 => (2, Kind::U16, false, false),
        RgbLinearF32 => (3, Kind::F32, false, true),
        RgbaLinearF32 => (4, Kind::F32, false, true),
        GrayLinearF32 => (1, Kind::F32, false, true),
        GrayAlphaLinearF32 => (2, Kind::F32, false, true),
        RgbLinearF16 => (3, Kind::F16, false, true),
        RgbaLinearF16 => (4, Kind::F16, false, true),
        GrayLinearF16 => (1, Kind::F16, false, true),
        GrayAlphaLinearF16 => (2, Kind::F16, false, true),
        RgbPqF32 | RgbHlgF32 | RgbBt709F32 => (3, Kind::F32, false, false),
        RgbaPqF32 | RgbaHlgF32 | RgbaBt709F32 => (4, Kind::F32, false, false),
        other => panic!("layout {other:?} not swept"),
    }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xff) as i32;
    let m = x & 0x7f_ffff;
    if e == 255 {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return sign | 0x7c00;
    }
    // Round to nearest, ties to even; a mantissa carry correctly bumps the exponent.
    let round = |kept: u32, rem: u32, half: u32| {
        kept + (rem > half || (rem == half && kept & 1 == 1)) as u32
    };
    if e16 <= 0 {
        if e16 < -10 {
            return sign;
        }
        let mant = m | 0x80_0000;
        let shift = (14 - e16) as u32;
        return sign | round(mant >> shift, mant & ((1 << shift) - 1), 1 << (shift - 1)) as u16;
    }
    sign | round(((e16 as u32) << 10) | (m >> 13), m & 0x1fff, 0x1000) as u16
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 113u32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Encoder input bytes and the expected logical samples (R, G, B / gray,
/// then alpha) in the layout's own units (0..=255, 0..=65535 or float).
fn make_pixels(
    src: &Source,
    rng: &mut Rng,
    w: usize,
    h: usize,
    layout: PixelLayout,
) -> (Vec<u8>, Vec<f64>) {
    let (ch, kind, bgr, linear) = describe(layout);
    let has_alpha = ch == 2 || ch == 4;
    let (ox, oy) = (
        rng.below(src.w as u64) as usize,
        rng.below(src.h as u64) as usize,
    );
    let alpha_style = rng.below(4);
    let fold = |v: usize, n: usize| {
        let m = v % (2 * n);
        if m < n { m } else { 2 * n - 1 - m }
    };
    let mut bytes = Vec::with_capacity(w * h * ch * 4);
    let mut want = Vec::with_capacity(w * h * ch);
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (fold(ox + x, src.w), fold(oy + y, src.h));
            let p = &src.rgb[(sy * src.w + sx) * 3..][..3];
            let mut logical = [0f64; 4];
            if ch <= 2 {
                logical[0] =
                    (p[0] as f64 * 0.299 + p[1] as f64 * 0.587 + p[2] as f64 * 0.114) / 255.0;
            } else {
                for c in 0..3 {
                    logical[c] = p[c] as f64 / 255.0;
                }
            }
            if has_alpha {
                logical[ch - 1] = match alpha_style {
                    0 => 1.0,
                    1 => ((x * 7 + y * 3) % 256) as f64 / 255.0,
                    2 => {
                        if (x / 5 + y / 7) % 3 == 0 {
                            0.0
                        } else {
                            1.0
                        }
                    }
                    _ => p[1] as f64 / 255.0,
                };
            }
            let mut stored = [0f64; 4];
            for c in 0..ch {
                let is_alpha = has_alpha && c == ch - 1;
                let v = logical[c];
                stored[c] = match kind {
                    Kind::U8 => (v * 255.0).round(),
                    Kind::U16 => {
                        let base = (v * 65535.0).round() as u32;
                        (base ^ ((x as u32 * 31 + y as u32 * 17 + c as u32) & 0x3f)).min(65535)
                            as f64
                    }
                    Kind::F32 | Kind::F16 => {
                        let lin = if linear && !is_alpha {
                            srgb_to_linear(v as f32)
                        } else {
                            v as f32
                        };
                        let f = lin + (x as f32) * 1e-6;
                        if kind == Kind::F16 {
                            f16_bits_to_f32(f32_to_f16_bits(f)) as f64
                        } else {
                            f as f64
                        }
                    }
                };
            }
            for c in 0..ch {
                // Byte order: BGR layouts swap the first three channels.
                let lc = if bgr && c < 3 { 2 - c } else { c };
                let v = stored[lc];
                match kind {
                    Kind::U8 => bytes.push(v as u8),
                    Kind::U16 => bytes.extend_from_slice(&(v as u16).to_ne_bytes()),
                    Kind::F32 => bytes.extend_from_slice(&(v as f32).to_ne_bytes()),
                    Kind::F16 => bytes.extend_from_slice(&f32_to_f16_bits(v as f32).to_ne_bytes()),
                }
            }
            want.extend_from_slice(&stored[..ch]);
        }
    }
    (bytes, want)
}

fn scale(kind: Kind) -> f64 {
    match kind {
        Kind::U8 => 255.0,
        Kind::U16 => 65535.0,
        Kind::F16 | Kind::F32 => 1.0,
    }
}

// ── Cell configuration ──────────────────────────────────────────────────────

const SIZES: &[(usize, usize)] = &[
    (1, 1),
    (2, 3),
    (7, 5),
    (17, 9),
    (64, 48),
    (127, 129),
    (255, 257),
    (256, 256),
    (257, 255),
    (300, 260),
    (513, 769),
    (1100, 900),
    (2049, 300),
];

struct Cell {
    index: u64,
    src: usize,
    w: usize,
    h: usize,
    layout: PixelLayout,
    lossless: bool,
    effort: u8,
    distance: f32,
    opts: Vec<String>,
    streaming: Option<u32>,
    icc: Option<usize>,
    exif_xmp: bool,
}

impl Cell {
    fn describe(&self, sources: &[Source], iccs: &[Icc]) -> String {
        let mut s = format!(
            "{} {}x{} {:?} {} e{}",
            sources[self.src].name,
            self.w,
            self.h,
            self.layout,
            if self.lossless {
                "lossless".to_string()
            } else {
                format!("d{}", self.distance)
            },
            self.effort
        );
        for o in &self.opts {
            let _ = write!(s, " {o}");
        }
        if let Some(rows) = self.streaming {
            let _ = write!(s, " stream{rows}");
        }
        if let Some(i) = self.icc {
            let _ = write!(s, " icc={}", iccs[i].name);
        }
        if self.exif_xmp {
            s.push_str(" exif+xmp");
        }
        s
    }
}

fn plan(seed: u64, index: u64, sources: &[Source], iccs: &[Icc]) -> Cell {
    let mut r = Rng::for_cell(seed, index);
    let src = r.below(sources.len() as u64) as usize;
    let (mut w, mut h) = r.pick(SIZES);
    if r.chance(50) {
        std::mem::swap(&mut w, &mut h);
    }
    let layout = r.pick(LAYOUTS);
    let lossless = r.chance(45);
    let big = w * h > 600_000;
    let max_effort = if big {
        8
    } else if w * h > 150_000 {
        9
    } else {
        10
    };
    let effort = 1 + r.below(max_effort) as u8;
    let distance = r.pick(&[0.1f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 7.0, 10.0, 15.0, 25.0]);
    let mut opts = Vec::new();
    if lossless {
        if r.chance(25) {
            opts.push("squeeze".into());
        }
        if r.chance(15) {
            opts.push("no_ans".into());
        }
        if r.chance(20) {
            opts.push(format!(
                "lz77:{}",
                r.pick(&["rle", "greedy", "optimal", "off"])
            ));
        }
        if r.chance(20) {
            opts.push("patches".into());
        }
        if r.chance(15) {
            opts.push(format!(
                "strategy:{}",
                r.pick(&["libjxl", "lean", "aggressive"])
            ));
        }
        if r.chance(15) {
            opts.push(format!("sectioned:{}", r.pick(&["off", "on", "hybrid"])));
        }
        if r.chance(10) {
            opts.push(format!("faster_decoding:{}", 1 + r.below(4)));
        }
        if r.chance(10) {
            opts.push(format!("group_shift:{}", r.below(4)));
        }
    } else {
        if r.chance(20) {
            opts.push(format!("progressive:{}", r.pick(&["ac", "dc"])));
        }
        if r.chance(10) {
            opts.push("lf_frame".into());
        }
        if r.chance(10) {
            opts.push(format!("progressive_dc:{}", 1 + r.below(2)));
        }
        if r.chance(15) {
            opts.push("patches".into());
        }
        if r.chance(10) {
            opts.push(format!("resampling:{}", r.pick(&[2u32, 4, 8])));
        }
        if r.chance(10) {
            opts.push("noise".into());
        }
        if r.chance(10) {
            opts.push("error_diffusion".into());
        }
        if r.chance(10) {
            opts.push("no_gaborish".into());
        }
        if r.chance(10) {
            opts.push("no_ans".into());
        }
        if r.chance(15) {
            opts.push(format!(
                "strategy:{}",
                r.pick(&["libjxl", "lean", "aggressive"])
            ));
        }
        if r.chance(10) {
            opts.push(format!("epf:{}", r.below(4)));
        }
        if r.chance(10) {
            opts.push("dots".into());
        }
        if r.chance(10) {
            opts.push(format!("chroma:{}", r.pick(&["422", "420", "440"])));
        }
        if r.chance(10) {
            opts.push(format!("faster_decoding:{}", 1 + r.below(4)));
        }
    }
    let streaming = if r.chance(15) {
        Some(1 + r.below(h.max(1) as u64) as u32)
    } else {
        None
    };
    let (ch, _, _, _) = describe(layout);
    let want_gray = ch <= 2;
    let icc = if streaming.is_none() && !iccs.is_empty() && r.chance(if lossless { 30 } else { 10 })
    {
        let candidates: Vec<usize> = (0..iccs.len())
            .filter(|&i| iccs[i].gray == want_gray)
            .collect();
        if candidates.is_empty() {
            None
        } else {
            Some(r.pick(&candidates))
        }
    } else {
        None
    };
    let exif_xmp = streaming.is_none() && r.chance(10);
    Cell {
        index,
        src,
        w,
        h,
        layout,
        lossless,
        effort,
        distance,
        opts,
        streaming,
        icc,
        exif_xmp,
    }
}

/// Combinations the encoder rejects by design. Anything else that fails to
/// encode is a finding.
fn expected_rejection(cell: &Cell, err: &str) -> bool {
    use PixelLayout::*;
    // Transfer-coded float layouts are lossy-only.
    let hdr_float = matches!(
        cell.layout,
        RgbPqF32 | RgbaPqF32 | RgbHlgF32 | RgbaHlgF32 | RgbBt709F32 | RgbaBt709F32
    );
    if cell.lossless && hdr_float && err.contains("unsupported pixel layout") {
        return true;
    }
    // Subsampled lossy needs the `chroma-subsampling` and `jpeg-reencoding` features.
    let chroma = cell.opts.iter().any(|o| o.starts_with("chroma:"));
    chroma && err.contains("chroma subsampling") && err.contains("cargo features")
}

// ── Encoding ────────────────────────────────────────────────────────────────

fn lossless_config(cell: &Cell) -> LosslessConfig {
    let mut c = LosslessConfig::new().with_effort(cell.effort);
    for o in &cell.opts {
        let (k, v) = o.split_once(':').unwrap_or((o.as_str(), ""));
        c = match k {
            "squeeze" => c.with_squeeze(true),
            "no_ans" => c.with_ans(false),
            "lz77" => match v {
                "off" => c.with_lz77(false),
                "rle" => c.with_lz77(true).with_lz77_method(Lz77Method::Rle),
                "greedy" => c.with_lz77(true).with_lz77_method(Lz77Method::Greedy),
                _ => c.with_lz77(true).with_lz77_method(Lz77Method::Optimal),
            },
            "patches" => c.with_patches(true),
            "strategy" => c.with_strategy(strategy(v)),
            "sectioned" => c.with_sectioned_trees(match v {
                "off" => SectionedTrees::Off,
                "on" => SectionedTrees::On,
                _ => SectionedTrees::Hybrid,
            }),
            "faster_decoding" => c.with_faster_decoding(v.parse().unwrap()),
            "group_shift" => c.with_modular_group_size(Some(v.parse().unwrap())),
            _ => unreachable!("{o}"),
        };
    }
    c
}

fn lossy_config(cell: &Cell) -> LossyConfig {
    let mut c = LossyConfig::new(cell.distance).with_effort(cell.effort);
    for o in &cell.opts {
        let (k, v) = o.split_once(':').unwrap_or((o.as_str(), ""));
        c = match k {
            "progressive" => c.with_progressive(if v == "ac" {
                ProgressiveMode::QuantizedAcFullAc
            } else {
                ProgressiveMode::DcVlfLfAc
            }),
            "lf_frame" => c.with_lf_frame(true),
            "progressive_dc" => c.with_progressive_dc(v.parse().unwrap()),
            "patches" => c.with_patches(true),
            "resampling" => c.with_resampling(v.parse().unwrap()),
            "noise" => c.with_noise(true),
            "error_diffusion" => c.with_error_diffusion(true),
            "no_gaborish" => c.with_gaborish(false),
            "no_ans" => c.with_ans(false),
            "strategy" => c.with_strategy(strategy(v)),
            "epf" => c.with_epf_level(v.parse().unwrap()),
            "dots" => c.with_dot_detection(true),
            "chroma" => c.with_chroma_subsampling(match v {
                "422" => ChromaSubsampling::Sub422,
                "420" => ChromaSubsampling::Sub420,
                _ => ChromaSubsampling::Sub440,
            }),
            "faster_decoding" => c.with_faster_decoding(v.parse().unwrap()),
            _ => unreachable!("{o}"),
        };
    }
    c
}

fn strategy(v: &str) -> EncoderStrategy {
    match v {
        "libjxl" => EncoderStrategy::Libjxl,
        "lean" => EncoderStrategy::LeanFaster,
        _ => EncoderStrategy::Aggressive,
    }
}

const EXIF: &[u8] = b"MM\0*\0\0\0\x08\0\0\0\0\0\0";
const XMP: &[u8] = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"/></x:xmpmeta>";

fn encode(cell: &Cell, pixels: &[u8], iccs: &[Icc]) -> Result<Vec<u8>, String> {
    let (w, h) = (cell.w as u32, cell.h as u32);
    let mut meta = ImageMetadata::new();
    if let Some(i) = cell.icc {
        meta = meta.with_icc_profile(&iccs[i].data);
    }
    if cell.exif_xmp {
        meta = meta.with_exif(EXIF).with_xmp(XMP);
    }
    let r = if cell.lossless {
        lossless_config(cell)
            .encode_request(w, h, cell.layout)
            .with_metadata(&meta)
            .encode(pixels)
    } else {
        lossy_config(cell)
            .encode_request(w, h, cell.layout)
            .with_metadata(&meta)
            .encode(pixels)
    };
    r.map_err(|e| e.to_string())
}

fn encode_streaming(cell: &Cell, pixels: &[u8], rows: u32) -> Result<Vec<u8>, String> {
    let (w, h) = (cell.w as u32, cell.h as u32);
    let row_bytes = pixels.len() / cell.h;
    macro_rules! push_all {
        ($enc:expr) => {{
            let mut enc = $enc.map_err(|e| e.to_string())?;
            let mut y = 0u32;
            while y < h {
                let n = rows.min(h - y);
                enc.push_rows(
                    &pixels[y as usize * row_bytes..(y + n) as usize * row_bytes],
                    n,
                )
                .map_err(|e| e.to_string())?;
                y += n;
            }
            enc.finish().map_err(|e| e.to_string())
        }};
    }
    if cell.lossless {
        push_all!(lossless_config(cell).encoder(w, h, cell.layout))
    } else {
        push_all!(lossy_config(cell).encoder(w, h, cell.layout))
    }
}

// ── Decoders ────────────────────────────────────────────────────────────────

struct Decoded {
    w: usize,
    h: usize,
    /// Interleaved logical channels: colour, then extra channels.
    ch: usize,
    px: Vec<f32>,
    icc: Option<Vec<u8>>,
}

/// Upstream jxl-rs (crates.io `jxl`, latest release).
fn decode_jxl_rs(data: &[u8]) -> Result<Decoded, String> {
    use jxl::api::{
        Event, JxlColorProfile, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProfileLevel,
    };
    use jxl::image::{Image, Rect};
    let mut options = JxlDecoderOptions::default();
    options.max_profile_level = ProfileLevel::Main10;
    let mut dec = JxlDecoder::new(options);
    let mut input = data;
    let (mut w, mut h, mut nc, mut ne) = (0, 0, 0, 0);
    let mut icc = None;
    let mut bufs: Option<(Image<f32>, Vec<Image<f32>>)> = None;
    let mut first: Option<(Image<f32>, Vec<Image<f32>>)> = None;
    let alloc = |w: usize,
                 h: usize,
                 nc: usize,
                 ne: usize|
     -> Result<(Image<f32>, Vec<Image<f32>>), String> {
        let c = Image::new((w * nc, h)).map_err(|e| format!("{e:?}"))?;
        let e = (0..ne)
            .map(|_| Image::new((w, h)).map_err(|e| format!("{e:?}")))
            .collect::<Result<_, _>>()?;
        Ok((c, e))
    };
    loop {
        let event = {
            let mut out = bufs.as_mut().map(|(c, ex)| {
                let mut v = vec![JxlOutputBuffer::from_image_rect_mut(
                    c.get_rect_mut(Rect {
                        origin: (0, 0),
                        size: c.size(),
                    })
                    .into_raw(),
                )];
                for e in ex.iter_mut() {
                    v.push(JxlOutputBuffer::from_image_rect_mut(
                        e.get_rect_mut(Rect {
                            origin: (0, 0),
                            size: e.size(),
                        })
                        .into_raw(),
                    ));
                }
                v
            });
            dec.process(&mut input, out.as_deref_mut(), None)
                .map_err(|e| format!("{e:?}"))?
        };
        match event {
            Event::BasicInfo => {
                (w, h) = dec.basic_info().ok_or("no basic info")?.size;
                icc = match dec.embedded_color_profile() {
                    Some(JxlColorProfile::Icc(v)) => Some(v.clone()),
                    _ => None,
                };
                let f = dec.current_pixel_format().ok_or("no pixel format")?.clone();
                nc = f.color_type.samples_per_pixel();
                ne = f.extra_channel_format.len();
                dec.set_pixel_format(JxlPixelFormat {
                    color_type: f.color_type,
                    color_data_format: Some(JxlDataFormat::f32()),
                    extra_channel_format: vec![Some(JxlDataFormat::f32()); ne],
                })
                .map_err(|e| format!("{e:?}"))?;
                bufs = Some(alloc(w, h, nc, ne)?);
            }
            Event::FrameHeader => {}
            Event::FrameComplete { has_more_frames } => {
                let done = bufs.take();
                if first.is_none() {
                    first = done;
                }
                if has_more_frames {
                    bufs = Some(alloc(w, h, nc, ne)?);
                }
            }
            Event::Complete => break,
            Event::NeedMoreInput { .. } => {
                if input.is_empty() {
                    return Err("truncated".into());
                }
            }
        }
    }
    let (color, extra) = first.ok_or("no frame")?;
    let ch = nc + ne;
    let mut px = Vec::with_capacity(w * h * ch);
    for y in 0..h {
        let crow = color.row(y);
        for x in 0..w {
            px.extend_from_slice(&crow[x * nc..x * nc + nc]);
            for e in &extra {
                px.push(e.row(y)[x]);
            }
        }
    }
    Ok(Decoded { w, h, ch, px, icc })
}

/// zenjxl-decoder, our jxl-rs fork, with its resource limits lifted.
fn decode_zen(data: &[u8]) -> Result<Decoded, String> {
    use zenjxl_decoder::api::states::Initialized;
    use zenjxl_decoder::api::{
        JxlColorProfile, JxlDataFormat, JxlDecoder, JxlDecoderLimits, JxlDecoderOptions,
        JxlOutputBuffer, JxlPixelFormat, ProcessingResult,
    };
    let mut options = JxlDecoderOptions::default();
    options.limits = JxlDecoderLimits::unlimited();
    let mut input = data;
    let mut init: JxlDecoder<Initialized> = JxlDecoder::new(options);
    let mut dec = loop {
        match init
            .process(&mut input)
            .map_err(|e| format!("header {e:?}"))?
        {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err("header: truncated".into());
                }
                init = fallback
            }
        }
    };
    let icc = match dec.embedded_color_profile() {
        JxlColorProfile::Icc(v) => Some(v.clone()),
        _ => None,
    };
    let (w, h) = dec.basic_info().size;
    let f = dec.current_pixel_format().clone();
    let nc = f.color_type.samples_per_pixel();
    let ne = f.extra_channel_format.len();
    dec.set_pixel_format(JxlPixelFormat {
        color_type: f.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![Some(JxlDataFormat::f32()); ne],
    });
    let mut frame = loop {
        match dec
            .process(&mut input)
            .map_err(|e| format!("frame {e:?}"))?
        {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err("frame: truncated".into());
                }
                dec = fallback
            }
        }
    };
    let mut cb = vec![0u8; w * nc * 4 * h];
    let mut eb: Vec<Vec<u8>> = (0..ne).map(|_| vec![0u8; w * 4 * h]).collect();
    {
        let mut bufs = vec![JxlOutputBuffer::new(&mut cb, h, w * nc * 4)];
        for e in &mut eb {
            bufs.push(JxlOutputBuffer::new(e, h, w * 4));
        }
        loop {
            match frame
                .process(&mut input, &mut bufs)
                .map_err(|e| format!("decode {e:?}"))?
            {
                ProcessingResult::Complete { .. } => break,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decode: truncated".into());
                    }
                    frame = fallback
                }
            }
        }
    }
    let fl = |b: &[u8], i: usize| f32::from_ne_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
    let ch = nc + ne;
    let mut px = Vec::with_capacity(w * h * ch);
    for i in 0..w * h {
        for c in 0..nc {
            px.push(fl(&cb, i * nc + c));
        }
        for e in &eb {
            px.push(fl(e, i));
        }
    }
    Ok(Decoded { w, h, ch, px, icc })
}

/// djxl to a 16-bit PAM (all channels) or, for float layouts, a PFM (colour
/// only); samples normalized to 0..1, plus the original ICC profile.
fn decode_djxl(data: &[u8], dir: &Path, tag: &str, float: bool) -> Result<Decoded, String> {
    let jxl = dir.join(format!("{tag}.jxl"));
    let out = dir.join(format!("{tag}.{}", if float { "pfm" } else { "pam" }));
    let icc = dir.join(format!("{tag}.icc"));
    std::fs::write(&jxl, data).map_err(|e| e.to_string())?;
    let mut cmd = std::process::Command::new(jxl_encoder::test_helpers::djxl_path());
    cmd.arg(&jxl)
        .arg(&out)
        .arg(format!("--orig_icc_out={}", icc.display()))
        .arg("--num_threads=1");
    let o = cmd.output().map_err(|e| format!("run djxl: {e}"))?;
    let result = (|| {
        if !o.status.success() {
            let msg = String::from_utf8_lossy(&o.stderr);
            return Err(format!("djxl: {}", msg.lines().last().unwrap_or("?")));
        }
        let bytes = std::fs::read(&out).map_err(|e| e.to_string())?;
        let icc_bytes = std::fs::read(&icc).ok();
        let mut d = if float {
            read_pfm(&bytes)?
        } else {
            read_pam16(&bytes)?
        };
        d.icc = icc_bytes;
        Ok(d)
    })();
    for p in [&jxl, &out, &icc] {
        let _ = std::fs::remove_file(p);
    }
    result
}

fn read_pam16(b: &[u8]) -> Result<Decoded, String> {
    let mut pos = 0;
    let (mut w, mut h, mut depth, mut maxval) = (0, 0, 0, 0u32);
    loop {
        let end = b[pos..]
            .iter()
            .position(|&c| c == b'\n')
            .ok_or("pam header")?
            + pos;
        let line = std::str::from_utf8(&b[pos..end]).map_err(|e| e.to_string())?;
        pos = end + 1;
        let mut it = line.split_whitespace();
        match it.next() {
            Some("WIDTH") => w = it.next().unwrap().parse().unwrap(),
            Some("HEIGHT") => h = it.next().unwrap().parse().unwrap(),
            Some("DEPTH") => depth = it.next().unwrap().parse().unwrap(),
            Some("MAXVAL") => maxval = it.next().unwrap().parse().unwrap(),
            Some("ENDHDR") => break,
            _ => {}
        }
    }
    let n = w * h * depth;
    let wide = maxval > 255;
    if b.len() < pos + n * if wide { 2 } else { 1 } {
        return Err("pam truncated".into());
    }
    let px = (0..n)
        .map(|i| {
            let v = if wide {
                u16::from_be_bytes([b[pos + 2 * i], b[pos + 2 * i + 1]]) as u32
            } else {
                b[pos + i] as u32
            };
            v as f32 / maxval as f32
        })
        .collect();
    Ok(Decoded {
        w,
        h,
        ch: depth,
        px,
        icc: None,
    })
}

fn read_pfm(b: &[u8]) -> Result<Decoded, String> {
    let mut fields = Vec::new();
    let mut pos = 0;
    while fields.len() < 4 {
        while b[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let start = pos;
        while !b[pos].is_ascii_whitespace() {
            pos += 1;
        }
        fields.push(std::str::from_utf8(&b[start..pos]).unwrap().to_string());
    }
    pos += 1;
    let ch = if fields[0] == "PF" { 3 } else { 1 };
    let (w, h): (usize, usize) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    let little = fields[3].parse::<f32>().unwrap() < 0.0;
    let mut px = vec![0f32; w * h * ch];
    for y in 0..h {
        for i in 0..w * ch {
            let o = pos + ((h - 1 - y) * w * ch + i) * 4;
            let raw = [b[o], b[o + 1], b[o + 2], b[o + 3]];
            px[y * w * ch + i] = if little {
                f32::from_le_bytes(raw)
            } else {
                f32::from_be_bytes(raw)
            };
        }
    }
    Ok(Decoded {
        w,
        h,
        ch,
        px,
        icc: None,
    })
}

// ── Checks ──────────────────────────────────────────────────────────────────

/// Sample `c` of pixel `i` from a decode, mapping logical channel `c` of a
/// `want_ch`-channel layout onto the decoder's channel count.
fn sample(d: &Decoded, i: usize, c: usize, want_ch: usize) -> Option<f32> {
    let has_alpha = want_ch == 2 || want_ch == 4;
    let dc = if has_alpha && c == want_ch - 1 {
        if d.ch == want_ch || d.ch == 2 || d.ch == 4 {
            d.ch - 1
        } else {
            return None;
        }
    } else if want_ch <= 2 {
        0
    } else {
        c
    };
    if dc >= d.ch {
        return None;
    }
    Some(d.px[i * d.ch + dc])
}

fn check_lossless(
    d: &Decoded,
    want: &[f64],
    cell: &Cell,
    who: &str,
    color_only: bool,
) -> Result<(), String> {
    let (ch, kind, _, _) = describe(cell.layout);
    let s = scale(kind);
    let mut bad = 0usize;
    let mut first = None;
    for i in 0..cell.w * cell.h {
        for c in 0..ch {
            let is_alpha = (ch == 2 || ch == 4) && c == ch - 1;
            if color_only && is_alpha {
                continue;
            }
            let Some(v) = sample(d, i, c, ch) else {
                return Err(format!("{who}: channel {c} missing ({} decoded)", d.ch));
            };
            let got = v as f64 * s;
            let exp = want[i * ch + c];
            let ok = match kind {
                Kind::U8 | Kind::U16 => got.round() == exp,
                // Float samples must come back bit-exact as f32.
                Kind::F16 | Kind::F32 => (v as f64 == exp) || (v == exp as f32),
            };
            if !ok {
                bad += 1;
                first.get_or_insert((i, c, got, exp));
            }
        }
    }
    match first {
        None => Ok(()),
        Some((i, c, g, e)) => Err(format!(
            "{who}: {bad} samples differ; first pixel {i} ch {c}: {g} vs {e}"
        )),
    }
}

/// Lossy: alpha exact, mean colour error under a loose bound.
fn check_lossy(
    d: &Decoded,
    want: &[f64],
    cell: &Cell,
    who: &str,
    check_alpha: bool,
) -> Result<f64, String> {
    let (ch, kind, _, _) = describe(cell.layout);
    let s = scale(kind);
    let has_alpha = ch == 2 || ch == 4;
    let mut err_sum = 0f64;
    let mut n = 0usize;
    for i in 0..cell.w * cell.h {
        // The encoder may rewrite the colour of fully transparent pixels
        // (SimplifyInvisible), so they do not count towards colour error.
        let invisible = has_alpha && want[i * ch + ch - 1] == 0.0;
        for c in 0..ch {
            let is_alpha = has_alpha && c == ch - 1;
            if invisible && !is_alpha {
                continue;
            }
            if is_alpha && !check_alpha {
                continue;
            }
            let Some(v) = sample(d, i, c, ch) else {
                return Err(format!("{who}: channel {c} missing"));
            };
            let v = clamp_for(cell, kind, v);
            let exp = want[i * ch + c] / s;
            if is_alpha {
                if kind != Kind::F16
                    && kind != Kind::F32
                    && ((v as f64) * s).round() != want[i * ch + c]
                {
                    return Err(format!(
                        "{who}: alpha differs at pixel {i}: {} vs {}",
                        (v as f64 * s).round(),
                        want[i * ch + c]
                    ));
                }
                continue;
            }
            err_sum += (v as f64 - exp).abs();
            n += 1;
        }
    }
    Ok(if n > 0 { err_sum / n as f64 } else { 0.0 })
}

/// Integer outputs clamp to the sample range while lossy float output can
/// overshoot it, so integer layouts compare clamped values. PQ, HLG and
/// BT.709 float layouts hold transfer-coded signal in [0, 1]: lossy decodes
/// land outside it (cjxl's own output does the same, by up to 0.7 in PQ
/// near black), and decoders differ in whether they clamp
/// (docs/DECODER_FINDINGS.md R2), so those compare clamped values too.
fn clamp_for(cell: &Cell, kind: Kind, v: f32) -> f32 {
    match kind {
        Kind::U8 | Kind::U16 => v.clamp(0.0, 1.0),
        Kind::F16 | Kind::F32 if transfer_signal(cell.layout) => v.clamp(0.0, 1.0),
        Kind::F16 | Kind::F32 => v,
    }
}

fn transfer_signal(layout: PixelLayout) -> bool {
    matches!(
        layout,
        PixelLayout::RgbPqF32
            | PixelLayout::RgbaPqF32
            | PixelLayout::RgbHlgF32
            | PixelLayout::RgbaHlgF32
            | PixelLayout::RgbBt709F32
            | PixelLayout::RgbaBt709F32
    )
}

fn max_disagreement(a: &Decoded, b: &Decoded, cell: &Cell) -> f64 {
    let (ch, kind, _, _) = describe(cell.layout);
    let color = if ch <= 2 { 1 } else { 3 };
    let mut m = 0f64;
    for i in 0..cell.w * cell.h {
        for c in 0..color {
            if let (Some(x), Some(y)) = (sample(a, i, c, ch), sample(b, i, c, ch)) {
                m = m
                    .max((clamp_for(cell, kind, x) as f64 - clamp_for(cell, kind, y) as f64).abs());
            }
        }
    }
    m
}

enum Verdict {
    Ok,
    Rejected(String),
    /// One Rust decoder disagrees while djxl and the other agree: a finding
    /// against that decoder, tracked but not an encoder failure.
    DecoderFinding(&'static str, String),
    Fail(String),
}

/// Every check one decoder's output must pass. `float_djxl` marks djxl's PFM
/// output, which has no alpha channel.
fn check_one(
    d: &Decoded,
    who: &str,
    cell: &Cell,
    want: &[f64],
    iccs: &[Icc],
    pfm: bool,
) -> Result<f64, String> {
    if (d.w, d.h) != (cell.w, cell.h) {
        return Err(format!("{who}: decoded {}x{}", d.w, d.h));
    }
    if let Some(i) = cell.icc
        && d.icc.as_deref() != Some(&iccs[i].data[..])
    {
        return Err(format!("{who}: ICC profile differs"));
    }
    if cell.lossless {
        check_lossless(d, want, cell, who, pfm)?;
        return Ok(0.0);
    }
    if cell.icc.is_some() {
        // Decoders render into the profile's space; only the bytes are checked.
        return Ok(0.0);
    }
    // Resampling (explicit, or automatic at high distance under the libjxl
    // strategy) downsamples alpha with the colour, so it is not exact.
    let factor = lossy_config(cell).effective_resampling() as usize;
    let alpha_exact = factor == 1 && !pfm;
    let err = check_lossy(d, want, cell, who, alpha_exact)?;
    // Resampling discards detail on purpose; scale the bound with it.
    let bound = (0.06 + 0.02 * cell.distance as f64) * factor as f64;
    // Resampling an image only a few blocks across leaves too few coded
    // samples for an error bound to mean anything.
    let tiny_resampled = factor > 1 && cell.w.min(cell.h) < 8 * factor;
    if !tiny_resampled && err > bound {
        return Err(format!("{who}: mean error {err:.4} over bound {bound:.3}"));
    }
    Ok(err)
}

/// Mean error and size of libjxl's own encode of the same source, for the
/// transfer-coded float layouts. Their lossy error spans too wide a range
/// (PQ up to 0.17 mean on clean cells) for a fixed bound to catch real
/// defects, so the sweep holds ours to cjxl's error instead: same signal
/// (16-bit PPM tagged with the layout's colour space), distance, effort and
/// resampling, decoded and scored the same way.
fn cjxl_reference_error(cell: &Cell, want: &[f64], dir: &Path) -> Result<(f64, usize), String> {
    let (ch, kind, _, _) = describe(cell.layout);
    let s = scale(kind);
    let space = match cell.layout {
        PixelLayout::RgbPqF32 | PixelLayout::RgbaPqF32 => "RGB_D65_202_Rel_PeQ",
        PixelLayout::RgbHlgF32 | PixelLayout::RgbaHlgF32 => "RGB_D65_202_Rel_HLG",
        _ => "RGB_D65_SRG_Rel_709",
    };
    let tag = format!("ref{}", cell.index);
    let ppm = dir.join(format!("{tag}.ppm"));
    let jxl = dir.join(format!("{tag}.jxl"));
    let mut img = format!("P6\n{} {}\n65535\n", cell.w, cell.h).into_bytes();
    for i in 0..cell.w * cell.h {
        for c in 0..3 {
            let v = (want[i * ch + c] / s).clamp(0.0, 1.0);
            img.extend_from_slice(&((v * 65535.0).round() as u16).to_be_bytes());
        }
    }
    std::fs::write(&ppm, img).map_err(|e| e.to_string())?;
    let factor = lossy_config(cell).effective_resampling();
    let o = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
        .arg(&ppm)
        .arg(&jxl)
        .arg(format!("--distance={}", cell.distance))
        .arg(format!("--effort={}", cell.effort.clamp(1, 10)))
        .arg(format!("--resampling={factor}"))
        .arg("-x")
        .arg(format!("color_space={space}"))
        .arg("--num_threads=1")
        .output();
    let _ = std::fs::remove_file(&ppm);
    let o = o.map_err(|e| format!("run cjxl: {e}"))?;
    let bytes = std::fs::read(&jxl);
    let _ = std::fs::remove_file(&jxl);
    if !o.status.success() {
        let msg = String::from_utf8_lossy(&o.stderr);
        return Err(format!("cjxl: {}", msg.lines().last().unwrap_or("?")));
    }
    let bytes = bytes.map_err(|e| e.to_string())?;
    let d = decode_djxl(&bytes, dir, &tag, true)?;
    Ok((check_lossy(&d, want, cell, "cjxl", false)?, bytes.len()))
}

fn run_cell(
    seed: u64,
    cell: &Cell,
    sources: &[Source],
    iccs: &[Icc],
    scratch: &Path,
) -> (Verdict, Option<Vec<u8>>, Vec<u8>, Duration, usize) {
    let mut r = Rng::for_cell(seed ^ 0xA11CE, cell.index);
    let (pixels, want) = make_pixels(&sources[cell.src], &mut r, cell.w, cell.h, cell.layout);
    let t = Instant::now();
    let bytes = match encode(cell, &pixels, iccs) {
        Ok(b) => b,
        Err(e) if expected_rejection(cell, &e) => {
            return (Verdict::Rejected(e), None, pixels, t.elapsed(), 0);
        }
        Err(e) => {
            return (
                Verdict::Fail(format!("encode: {e}")),
                None,
                pixels,
                t.elapsed(),
                0,
            );
        }
    };
    let elapsed = t.elapsed();
    let len = bytes.len();
    let done = |v: Verdict| (v, Some(bytes.clone()), pixels.clone(), elapsed, len);
    if let Some(rows) = cell.streaming {
        match encode_streaming(cell, &pixels, rows) {
            Err(e) => return done(Verdict::Fail(format!("streaming encode: {e}"))),
            Ok(s) if s != bytes => {
                return done(Verdict::Fail(format!(
                    "streaming ({} B) differs from one-shot ({} B)",
                    s.len(),
                    bytes.len()
                )));
            }
            Ok(_) => {}
        }
    }
    let (_, kind, _, _) = describe(cell.layout);
    let float = matches!(kind, Kind::F16 | Kind::F32);
    let dj = decode_djxl(&bytes, scratch, &format!("c{}", cell.index), float);
    let rs = decode_jxl_rs(&bytes).map_err(|e| format!("jxl-rs: {e}"));
    let zen = decode_zen(&bytes).map_err(|e| format!("zenjxl: {e}"));
    let judge = |d: &Result<Decoded, String>, who: &str, pfm: bool| -> Result<(), String> {
        let d = d.as_ref().map_err(Clone::clone)?;
        check_one(d, who, cell, &want, iccs, pfm).map(|_| ())
    };
    let v_dj = judge(&dj, "djxl", float);
    let mut v_rs = judge(&rs, "jxl-rs", false);
    // Transfer-coded float layouts: hold our error to libjxl's on the same
    // source (`cjxl_reference_error`). Applied where the comparison is
    // like for like: both sides at least 32 px, and none of the options
    // cjxl is not given that change the error on purpose (photon noise, LF
    // frames, progressive DC). A file at least 15% smaller than cjxl's is
    // a different rate point, not a defect (e8+ runs smaller than cjxl at
    // the same distance), so only error at a comparable size fails.
    // Calibrated 2026-10-25 on seed 4: 2725 such cells, median error ratio
    // 1.00, none above 1.5x + 0.015.
    let mut v_dj = v_dj;
    if v_dj.is_ok()
        && !cell.lossless
        && cell.icc.is_none()
        && transfer_signal(cell.layout)
        && cell.w.min(cell.h) >= 32
        && !cell
            .opts
            .iter()
            .any(|o| o == "noise" || o == "lf_frame" || o.starts_with("progressive_dc"))
        && let Ok(d) = &dj
    {
        let ours = check_lossy(d, &want, cell, "djxl", false);
        v_dj = match (ours, cjxl_reference_error(cell, &want, scratch)) {
            (Ok(o), Ok((r, ref_len))) if o > 1.5 * r + 0.015 && len * 100 >= ref_len * 85 => {
                Err(format!(
                    "djxl: mean error {o:.4} over 1.5x cjxl's {r:.4} + 0.015 at {len} B vs {ref_len} B"
                ))
            }
            (Ok(_), Ok(_)) => Ok(()),
            (Err(e), _) | (_, Err(e)) => Err(format!("cjxl reference: {e}")),
        };
    }
    let mut v_zen = judge(&zen, "zenjxl", false);
    // Lossy: each Rust decoder must agree with djxl (transfer-coded float
    // output compared in the clamped signal range, see `clamp_for`). HLG is
    // excluded: jxl-rs and zenjxl-decoder agree with each other but differ
    // from djxl by up to 0.34 on bright pixels (seed 3, 46 of 2766 cells),
    // an inverse-OOTF difference still in triage (docs/DECODER_FINDINGS.md
    // Z1). All three are still checked against the source.
    let hlg = matches!(
        cell.layout,
        PixelLayout::RgbHlgF32 | PixelLayout::RgbaHlgF32
    );
    if !cell.lossless && cell.icc.is_none() && !hlg {
        let tol = if float { 0.02 } else { 3.0 / 255.0 };
        if let Ok(d) = &dj {
            for (v, other, who) in [(&mut v_rs, &rs, "jxl-rs"), (&mut v_zen, &zen, "zenjxl")] {
                if let (Ok(()), Ok(o)) = (&*v, other) {
                    let dis = max_disagreement(o, d, cell);
                    if dis > tol {
                        *v = Err(format!(
                            "{who}: differs from djxl by {dis:.4} (tolerance {tol:.4})"
                        ));
                    }
                }
            }
        }
    }
    let verdict = match (v_dj, v_rs, v_zen) {
        (Ok(()), Ok(()), Ok(())) => Verdict::Ok,
        (Ok(()), Err(e), Ok(())) => Verdict::DecoderFinding("jxl-rs", e),
        (Ok(()), Ok(()), Err(e)) => Verdict::DecoderFinding("zenjxl-decoder", e),
        (a, b, c) => {
            let msgs: Vec<String> = [a, b, c].into_iter().filter_map(Result::err).collect();
            Verdict::Fail(msgs.join("; "))
        }
    };
    done(verdict)
}

// ── Driver ──────────────────────────────────────────────────────────────────

fn iso_week_seed() -> u64 {
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        / 86_400;
    // Weeks since the epoch, Monday-aligned (1970-01-05 was a Monday).
    (days + 3) / 7
}

fn main() {
    let mut images = Vec::new();
    let (mut out, mut icc_dir) = (None, None);
    let (mut seed, mut cells, mut minutes, mut threads, mut only) =
        (None, 2000u64, 0u64, 0usize, None);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--images" => images.push(PathBuf::from(val())),
            "--out" => out = Some(PathBuf::from(val())),
            "--icc" => icc_dir = Some(PathBuf::from(val())),
            "--seed" => seed = Some(val().parse().unwrap()),
            "--cells" => cells = val().parse().unwrap(),
            "--minutes" => minutes = val().parse().unwrap(),
            "--threads" => threads = val().parse().unwrap(),
            "--only" => only = Some(val().parse::<u64>().unwrap()),
            other if !other.starts_with("--") => images.push(PathBuf::from(other)),
            other => panic!("unknown argument {other}"),
        }
    }
    let out = out.expect("--out <dir> is required");
    let seed = seed.unwrap_or_else(iso_week_seed);
    let threads = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        threads
    };
    let sources = load_sources(&images);
    let iccs = load_icc(icc_dir.as_deref());
    let failures_dir = out.join("failures");
    let findings_dir = out.join("decoder_findings");
    std::fs::create_dir_all(&findings_dir).unwrap();
    let scratch = out.join("scratch");
    std::fs::create_dir_all(&failures_dir).unwrap();
    std::fs::create_dir_all(&scratch).unwrap();
    // Check the decoder before spending time encoding.
    let _ = jxl_encoder::test_helpers::djxl_path();

    let indices: Vec<u64> = match only {
        Some(i) => vec![i],
        None => (0..cells).collect(),
    };
    eprintln!(
        "conformance sweep: seed {seed}, {} cells planned, {} images, {} ICC profiles, {threads} threads",
        indices.len(),
        sources.len(),
        iccs.len()
    );
    let deadline = (minutes > 0).then(|| Instant::now() + Duration::from_secs(minutes * 60));
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let rows = Mutex::new(Vec::<(u64, String)>::new());
    let (ok, rejected, failed) = (
        AtomicUsize::new(0),
        AtomicUsize::new(0),
        AtomicUsize::new(0),
    );
    let findings = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                if stop.load(Ordering::Relaxed) || deadline.is_some_and(|d| Instant::now() > d) {
                    stop.store(true, Ordering::Relaxed);
                    return;
                }
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some(&index) = indices.get(k) else { return };
                let cell = plan(seed, index, &sources, &iccs);
                let desc = cell.describe(&sources, &iccs);
                let (verdict, bytes, pixels, elapsed, len) = run_cell(seed, &cell, &sources, &iccs, &scratch);
                let (status, detail) = match &verdict {
                    Verdict::Ok => {
                        ok.fetch_add(1, Ordering::Relaxed);
                        ("OK", String::new())
                    }
                    Verdict::Rejected(e) => {
                        rejected.fetch_add(1, Ordering::Relaxed);
                        ("REJECTED", e.clone())
                    }
                    Verdict::DecoderFinding(dec, e) => {
                        findings.fetch_add(1, Ordering::Relaxed);
                        let base = findings_dir.join(format!("cell{index}"));
                        if let Some(b) = &bytes {
                            std::fs::write(base.with_extension("jxl"), b).unwrap();
                        }
                        std::fs::write(
                            base.with_extension("txt"),
                            format!("{dec}\n{desc}\n{e}\nreplay: conformance-sweep --seed {seed} --only {index} <same --images/--icc>\n"),
                        )
                        .unwrap();
                        ("DECODER", format!("{dec}: {e}"))
                    }
                    Verdict::Fail(e) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                        let base = failures_dir.join(format!("cell{index}"));
                        if let Some(b) = &bytes {
                            std::fs::write(base.with_extension("jxl"), b).unwrap();
                        }
                        std::fs::write(base.with_extension("input"), &pixels).unwrap();
                        std::fs::write(
                            base.with_extension("txt"),
                            format!("{desc}\n{e}\nreplay: conformance-sweep --seed {seed} --only {index} <same --images/--icc>\n"),
                        )
                        .unwrap();
                        eprintln!("FAIL cell {index}: {desc}: {e}");
                        ("FAIL", e.clone())
                    }
                };
                rows.lock().unwrap().push((
                    index,
                    format!("{index}\t{status}\t{}\t{len}\t{desc}\t{detail}", elapsed.as_millis()),
                ));
            });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by_key(|r| r.0);
    let mut tsv = format!(
        "# conformance sweep, seed {seed}\nindex\tstatus\tencode_ms\tbytes\tcell\tdetail\n"
    );
    for (_, r) in &rows {
        tsv.push_str(r);
        tsv.push('\n');
    }
    std::fs::write(out.join("results.tsv"), tsv).unwrap();
    let _ = std::fs::remove_dir_all(&scratch);
    let (ok, rejected, failed) = (ok.into_inner(), rejected.into_inner(), failed.into_inner());
    let findings = findings.into_inner();
    let summary = format!(
        "seed {seed}: {} cells run ({ok} ok, {rejected} rejected as expected, {failed} failed, {findings} decoder findings){}\n",
        rows.len(),
        if rows.len() < indices.len() {
            format!(", stopped at the time budget of {minutes} min")
        } else {
            String::new()
        }
    );
    std::fs::write(out.join("summary.txt"), &summary).unwrap();
    eprint!("{summary}");
    if failed > 0 {
        std::process::exit(1);
    }
}
