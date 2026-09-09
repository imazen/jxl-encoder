//! #110 item 1 — A/B the LZ77 hash function (shift-fold vs post-v0.12 Murmur).
//!
//! The arm is selected by `JXL_LZ77_MURMUR_HASH`, which is read ONCE per
//! process (it sits under the matcher inner loop), so each arm must be a
//! separate invocation. Run twice and join on the key columns.
//!
//! Hypothesis under test, from reading the hash rather than from upstream
//! (which ships no numbers): with `hash_shift = 5` and a 15-bit mask the old
//! fold sees only `data[pos]` bits 0..=14, `data[pos+1]` bits 0..=9 and
//! `data[pos+2]` bits 0..=4, so it is nearly blind above 2^15. Token values
//! that wide occur on the HIGH-BIT-DEPTH and FLOAT lossless paths, not on
//! 8-bit SDR. So the new hash should help disproportionately there — and may
//! do nothing at 8-bit. Both arms are measured so the null result is visible.
//!
//! LZ77 is only reachable at the efforts whose `lz77_method` is not `Rle`:
//! lossless e8 (Greedy) and e9+ (Optimal); lossy e9+ (Optimal).
//!
//! Usage: lz77_hash_ab <corpus-dir> <out.tsv> [--images N] [--size N]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use jxl_encoder::api::{LosslessConfig, LossyConfig, PixelLayout};

fn arg(name: &str, default: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
        .unwrap_or_else(|| default.to_string())
}

fn walk(d: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    let mut st = vec![d.to_path_buf()];
    while let Some(p) = st.pop() {
        if let Ok(rd) = std::fs::read_dir(&p) {
            for e in rd.flatten() {
                let q = e.path();
                if q.is_dir() { st.push(q) } else { v.push(q) }
            }
        }
    }
    v.sort();
    v
}

fn crop16(src: &[u16], sw: u32, sh: u32, n: u32) -> Option<Vec<u16>> {
    if sw < n || sh < n {
        return None;
    }
    let (x0, y0) = ((sw - n) / 2, (sh - n) / 2);
    let mut o = Vec::with_capacity((n * n * 3) as usize);
    for y in 0..n {
        let r = ((y0 + y) * sw + x0) as usize * 3;
        o.extend_from_slice(&src[r..r + (n * 3) as usize]);
    }
    Some(o)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let corpus = PathBuf::from(&a[1]);
    let out_path = PathBuf::from(&a[2]);
    let max_images: usize = arg("--images", "6").parse().unwrap();
    let n: u32 = arg("--size", "512").parse().unwrap();
    let efforts: Vec<u8> = arg("--efforts", "8,9")
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let do_lossy = arg("--lossy", "1") == "1";

    let arm = if std::env::var("JXL_LZ77_MURMUR_HASH").as_deref() == Ok("1") {
        "murmur"
    } else {
        "fold"
    };

    let mut out = std::fs::File::create(&out_path).unwrap();
    writeln!(out, "arm\timage\tpath_kind\tdepth\teffort\tbytes\tms").unwrap();

    let mut picked = 0usize;
    for f in walk(&corpus) {
        if picked >= max_images {
            break;
        }
        // Accept the corpus's HDR photos AND any other 16-bit RGB PNG (the
        // synthetic float line-art set), since LZ77 fires on repetitive
        // content and photo residuals are the case where it barely does.
        let fs = f.to_string_lossy();
        if !fs.ends_with(".png") {
            continue;
        }
        let Ok(img) = image::open(&f) else { continue };
        // Accept 8-bit sources too (widened by <<8) so the content grid covers
        // every imazen-26 stratum, not just the four photographic ones that
        // ship 16-bit HDR. For an ACCEPTANCE-RATE measurement what matters is
        // token-stream structure, not the low 8 bits.
        let (sw, sh, flat): (u32, u32, Vec<u16>) = if let Some(r) = img.as_rgb16() {
            (
                r.width(),
                r.height(),
                r.pixels().flat_map(|p| [p.0[0], p.0[1], p.0[2]]).collect(),
            )
        } else {
            let r = img.to_rgb8();
            (
                r.width(),
                r.height(),
                r.pixels()
                    .flat_map(|p| {
                        [
                            (p.0[0] as u16) << 8,
                            (p.0[1] as u16) << 8,
                            (p.0[2] as u16) << 8,
                        ]
                    })
                    .collect(),
            )
        };
        let Some(c16) = crop16(&flat, sw, sh, n) else {
            continue;
        };
        picked += 1;
        let name = f.file_name().unwrap().to_string_lossy().to_string();

        // Three token-width regimes on identical content.
        let b16: Vec<u8> = c16.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let b8: Vec<u8> = c16.iter().map(|v| (v >> 8) as u8).collect();
        let bf32: Vec<u8> = c16
            .iter()
            .flat_map(|v| (*v as f32 / 65535.0).to_ne_bytes())
            .collect();

        let mut row = |kind: &str, depth: &str, effort: u8, bytes: usize, ms: f64| {
            writeln!(
                out,
                "{arm}\t{name}\t{kind}\t{depth}\t{effort}\t{bytes}\t{ms:.1}"
            )
            .unwrap();
        };

        for &e in &efforts {
            for (depth, buf, layout) in [
                ("u8", &b8, PixelLayout::Rgb8),
                ("u16", &b16, PixelLayout::Rgb16),
                ("f32", &bf32, PixelLayout::RgbLinearF32),
            ] {
                let t = Instant::now();
                let d = LosslessConfig::new()
                    .with_effort(e)
                    .encode_request(n, n, layout)
                    .encode(buf)
                    .expect("lossless encode");
                row(
                    "lossless",
                    depth,
                    e,
                    d.len(),
                    t.elapsed().as_micros() as f64 / 1000.0,
                );
            }
        }

        // Lossy e9 also reaches Optimal LZ77 (effort.rs: lz77 at effort >= 9).
        if !do_lossy {
            eprintln!("{arm}: {name} done ({picked}/{max_images})");
            continue;
        }
        let t = Instant::now();
        let d = LossyConfig::new(1.0)
            .with_effort(9)
            .encode_request(n, n, PixelLayout::Rgb16)
            .encode(&b16)
            .expect("lossy encode");
        row(
            "lossy",
            "u16",
            9,
            d.len(),
            t.elapsed().as_micros() as f64 / 1000.0,
        );

        out.flush().unwrap();
        eprintln!("{arm}: {name} done ({picked}/{max_images})");
    }
    println!("{arm} -> {}", out_path.display());
}
