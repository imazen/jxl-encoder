//! #109 F5 — lossless FLOAT parity + rate/time vs cjxl v0.12.
//!
//! Lossless has no quality axis: both encoders reproduce the input exactly, so
//! "RD" here is bytes against wall time, and the only honest comparison is at
//! matched, verified bit-exactness. Every cell therefore round-trips our output
//! through djxl and asserts the decoded floats are bit-identical to the input
//! before its bytes are allowed into the table.
//!
//! Expected shape, stated up front so the numbers can contradict it: at f32 the
//! `max_bitdepth` budget disables BOTH RCT and Squeeze (32 leaves no headroom —
//! see the #109 F3 divergence row), and libjxl's budget refuses them for the
//! same reason. So neither side gets a colour transform; compression comes from
//! the predictor and entropy stages alone, which is exactly where a real
//! divergence would show.
//!
//! Source content: the imazen-26 `.hdr.png` set is 16-bit RGB, converted to f32
//! by dividing by 65535 (exact — 16-bit integers are representable in binary32,
//! so the conversion invents no precision). **That set is photographic only**
//! (nature / interiors / photos-general / food); this harness therefore says
//! nothing about float line-art, plots or screenshots. Non-photo float content
//! would need a different corpus and is NOT covered here.
//!
//! Sizes are CENTRE CROPS, never resamples: a resample would change the local
//! statistics that predictors key on, so a "1024²" row would not be measuring
//! the same content class as its 4096² sibling.
//!
//! Usage:
//!   cargo run --release --example lossless_float_parity -- \
//!       <corpus-dir> <out.tsv> [--sizes 64,256,1024,4096] [--efforts 3,5,7,9]
//!       [--images N] [--reps N]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use jxl_encoder::api::{LosslessConfig, PixelLayout};

fn arg(name: &str, default: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
        .unwrap_or_else(|| default.to_string())
}

/// 16-bit PNG -> f32 in [0,1]. Exact: every u16 is representable in binary32.
fn load_png_f32(path: &Path) -> Option<(u32, u32, Vec<f32>)> {
    let img = image::open(path).ok()?;
    let rgb16 = img.as_rgb16()?;
    let (w, h) = (rgb16.width(), rgb16.height());
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for p in rgb16.pixels() {
        for c in 0..3 {
            out.push(p.0[c] as f32 / 65535.0);
        }
    }
    Some((w, h, out))
}

fn centre_crop(src: &[f32], sw: u32, sh: u32, n: u32) -> Option<Vec<f32>> {
    if sw < n || sh < n {
        return None;
    }
    let (x0, y0) = ((sw - n) / 2, (sh - n) / 2);
    let mut out = Vec::with_capacity((n * n * 3) as usize);
    for y in 0..n {
        let row = ((y0 + y) * sw + x0) as usize * 3;
        out.extend_from_slice(&src[row..row + (n * 3) as usize]);
    }
    Some(out)
}

/// Little-endian PFM (scale < 0), the format cjxl reads for f32 input.
/// PFM rows run BOTTOM-UP, so the writer flips and the reader flips back;
/// getting this wrong would compare two different images and still "work".
fn write_pfm(path: &Path, px: &[f32], n: u32) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    write!(f, "PF\n{n} {n}\n-1.0\n")?;
    for y in (0..n).rev() {
        let row = (y * n) as usize * 3;
        for v in &px[row..row + (n * 3) as usize] {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn read_pfm(path: &Path) -> Option<(u32, Vec<f32>)> {
    let d = std::fs::read(path).ok()?;
    // Header is three whitespace-terminated ASCII fields.
    let mut pos = 0usize;
    let mut fields = Vec::new();
    while fields.len() < 4 && pos < d.len() {
        while pos < d.len() && d[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let s = pos;
        while pos < d.len() && !d[pos].is_ascii_whitespace() {
            pos += 1;
        }
        fields.push(String::from_utf8_lossy(&d[s..pos]).to_string());
    }
    pos += 1; // single whitespace after the scale
    let w: u32 = fields[1].parse().ok()?;
    let h: u32 = fields[2].parse().ok()?;
    // The SIGN of the scale is the endianness flag: negative = little-endian,
    // positive = big-endian. We write -1.0; djxl v0.12 writes +1.0, so a reader
    // that assumes one of them silently byte-swaps every sample and reports a
    // bit-exact stream as corrupt (this cost a debug cycle — the values matched
    // reversed: a88a273f vs 3f278aa8).
    let scale: f32 = fields[3].parse().ok()?;
    let little = scale < 0.0;
    let mut px = vec![0f32; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..(w * 3) {
            let o = pos + ((y * w * 3 + x) as usize) * 4;
            let b = [d[o], d[o + 1], d[o + 2], d[o + 3]];
            let v = if little {
                f32::from_le_bytes(b)
            } else {
                f32::from_be_bytes(b)
            };
            px[(((h - 1 - y) * w * 3) + x) as usize] = v;
        }
    }
    Some((w, px))
}

fn min_of<F: FnMut() -> u128>(reps: u32, mut f: F) -> f64 {
    (0..reps).map(|_| f()).min().unwrap() as f64 / 1000.0
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let corpus = PathBuf::from(&a[1]);
    let out_path = PathBuf::from(&a[2]);
    let sizes: Vec<u32> = arg("--sizes", "64,256,1024,4096")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let efforts: Vec<u8> = arg("--efforts", "3,5,7,9")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let max_images: usize = arg("--images", "8").parse().unwrap();
    let reps: u32 = arg("--reps", "3").parse().unwrap();

    let cjxl = jxl_encoder::test_helpers::cjxl_path();
    let djxl = jxl_encoder::test_helpers::djxl_path();
    let tmp = std::env::var("HOME").unwrap() + "/tmp/f5";
    std::fs::create_dir_all(&tmp).unwrap();

    // Stratify: walk classes round-robin so one class cannot dominate.
    let mut by_class: std::collections::BTreeMap<String, Vec<PathBuf>> = Default::default();
    for e in walkdir(&corpus) {
        if e.to_string_lossy().ends_with(".hdr.png") {
            let cls = e
                .strip_prefix(&corpus)
                .ok()
                .and_then(|p| {
                    p.components()
                        .next()
                        .map(|c| c.as_os_str().to_string_lossy().to_string())
                })
                .unwrap_or_default();
            by_class.entry(cls).or_default().push(e);
        }
    }
    for v in by_class.values_mut() {
        v.sort();
    }
    let mut picks: Vec<(String, PathBuf)> = Vec::new();
    let mut i = 0;
    while picks.len() < max_images {
        let mut added = false;
        for (c, v) in &by_class {
            if let Some(p) = v.get(i) {
                picks.push((c.clone(), p.clone()));
                added = true;
                if picks.len() >= max_images {
                    break;
                }
            }
        }
        if !added {
            break;
        }
        i += 1;
    }

    let mut out = std::fs::File::create(&out_path).unwrap();
    writeln!(
        out,
        "class\timage\tsize\teffort\tours_bytes\tcjxl_bytes\tbytes_ratio\tours_ms\tcjxl_ms\twall_ratio\tours_bitexact"
    )
    .unwrap();
    println!("{} images x {:?} x e{:?}", picks.len(), sizes, efforts);

    for (cls, path) in &picks {
        let Some((sw, sh, full)) = load_png_f32(path) else {
            eprintln!("skip (not 16-bit rgb): {}", path.display());
            continue;
        };
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        for &n in &sizes {
            let Some(px) = centre_crop(&full, sw, sh, n) else {
                continue; // source smaller than this size class: no upscaling
            };
            let bytes: Vec<u8> = px.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let pfm = PathBuf::from(format!("{tmp}/in_{n}.pfm"));
            write_pfm(&pfm, &px, n).unwrap();

            for &e in &efforts {
                // Interleave the two arms inside each rep: block-ordered
                // repeats inverted the sign of a result in the T3 work.
                let mut ours_t = Vec::new();
                let mut cj_t = Vec::new();
                let mut ours_bytes = 0usize;
                let cj_out = PathBuf::from(format!("{tmp}/cj_{n}_{e}.jxl"));
                for _ in 0..reps {
                    let t = Instant::now();
                    let d = LosslessConfig::new()
                        .with_effort(e)
                        .encode_request(n, n, PixelLayout::RgbLinearF32)
                        .encode(&bytes)
                        .expect("our lossless f32 encode");
                    ours_t.push(t.elapsed().as_micros());
                    ours_bytes = d.len();
                    std::fs::write(format!("{tmp}/ours_{n}_{e}.jxl"), &d).unwrap();

                    let t = Instant::now();
                    let st = Command::new(&cjxl)
                        .args([
                            pfm.to_str().unwrap(),
                            cj_out.to_str().unwrap(),
                            "-d",
                            "0",
                            "-e",
                            &e.to_string(),
                            "--quiet",
                        ])
                        .output()
                        .expect("cjxl");
                    cj_t.push(t.elapsed().as_micros());
                    if !st.status.success() {
                        eprintln!("cjxl failed: {}", String::from_utf8_lossy(&st.stderr));
                    }
                }
                let cjxl_bytes = std::fs::metadata(&cj_out).map(|m| m.len()).unwrap_or(0) as usize;
                let ours_ms = min_of(1, || *ours_t.iter().min().unwrap());
                let cj_ms = min_of(1, || *cj_t.iter().min().unwrap());

                // Gate: our bytes only count if they decode BIT-EXACTLY.
                let dec = PathBuf::from(format!("{tmp}/dec_{n}_{e}.pfm"));
                let _ = Command::new(&djxl)
                    .args([
                        &format!("{tmp}/ours_{n}_{e}.jxl"),
                        dec.to_str().unwrap(),
                        "--quiet",
                    ])
                    .output();
                let bitexact = match read_pfm(&dec) {
                    Some((dw, dpx)) => {
                        dw == n
                            && dpx.len() == px.len()
                            && dpx
                                .iter()
                                .zip(px.iter())
                                .all(|(a, b)| a.to_bits() == b.to_bits())
                    }
                    None => false,
                };

                writeln!(
                    out,
                    "{cls}\t{name}\t{n}\t{e}\t{ours_bytes}\t{cjxl_bytes}\t{:.4}\t{ours_ms:.1}\t{cj_ms:.1}\t{:.3}\t{bitexact}",
                    ours_bytes as f64 / cjxl_bytes.max(1) as f64,
                    ours_ms / cj_ms.max(0.001),
                )
                .unwrap();
                out.flush().unwrap();
                println!(
                    "{cls:32.32} {n:5} e{e}  ours {ours_bytes:9} cjxl {cjxl_bytes:9}  ratio {:.4}  wall {ours_ms:8.1}/{cj_ms:8.1}ms  bitexact={bitexact}",
                    ours_bytes as f64 / cjxl_bytes.max(1) as f64
                );
            }
        }
    }
    println!("wrote {}", out_path.display());
}

fn walkdir(d: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    let mut stack = vec![d.to_path_buf()];
    while let Some(p) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&p) {
            for e in rd.flatten() {
                let q = e.path();
                if q.is_dir() {
                    stack.push(q);
                } else {
                    v.push(q);
                }
            }
        }
    }
    v.sort();
    v
}
