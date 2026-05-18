//! HDR end-to-end RD sweep vs cjxl.
//!
//! Encodes synthetic HDR gradients (PQ / HLG / BT.709 video-range) at
//! several distance levels with both `jxl-encoder` (using
//! `RgbPqF32` / `RgbHlgF32` / `RgbBt709F32` PixelLayouts + the matching
//! `ColorEncoding` preset + `with_intensity_target`) and the libjxl
//! `cjxl` CLI (via a PFM dump + `-x color_space=Rec2100PQ` /
//! `Rec2100HLG` / `Rec709`).
//!
//! HDR-aware perceptual metrics are not available in-tree (Rust
//! butteraugli's display model assumes SDR ~80 nits intensity_target),
//! so this sweep reports bytes-parity only and writes a TSV under
//! `benchmarks/hdr_rd_sweep_<UTC>.tsv` plus a `.meta` sidecar with the
//! git revision, hostname, cjxl version, and reproducer command.
//!
//! Issue: jxl-encoder#44 / W4 — closes the "never RD-benchmarked"
//! item from the HDR implementation plan
//! (memory/hdr_encoding_implementation_plan_2026-05-17.md).
//!
//! Run with `cargo run -p jxl-encoder --release --example hdr_rd_sweep_vs_cjxl`.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use bytemuck::cast_slice;
use jxl_encoder::{ColorEncoding, LossyConfig, PixelLayout};

/// Width × height of every synthetic gradient. 256² is enough to
/// exercise the encoder's multi-block paths (32×32 of 8×8 blocks)
/// without bloating the sweep wall time.
const W: u32 = 256;
const H: u32 = 256;

/// Three HDR layouts under test. The tuple is
/// `(layout, jxl_encoder PixelLayout, cjxl shorthand, intensity_target_nits, label)`.
fn hdr_cases() -> Vec<(&'static str, PixelLayout, &'static str, f32)> {
    vec![
        ("pq", PixelLayout::RgbPqF32, "Rec2100PQ", 1000.0),
        ("hlg", PixelLayout::RgbHlgF32, "Rec2100HLG", 1000.0),
        // BT.709 video-range is SDR-like. libjxl's `-x color_space=`
        // parser only knows the `SRG`/`Cst`/`202`/`DCI` primaries
        // tokens (libjxl `lib/extras/dec/color_description.cc:43`),
        // not a separate `709` token — BT.709 primaries are
        // bit-identical to sRGB primaries, so we use `SRG` for the
        // primaries slot and the `709` transfer function. This is
        // what jxl-encoder also emits for `RgbBt709F32` when no
        // explicit `with_color_encoding` is set.
        (
            "bt709",
            PixelLayout::RgbBt709F32,
            "RGB_D65_SRG_Rel_709",
            100.0,
        ),
    ]
}

const DISTANCES: &[f32] = &[1.0, 2.0, 5.0];

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
    PathBuf::from(home).join("work/jxl-efforts/libjxl/build/tools/cjxl")
}

/// Synthesize a gradient appropriate for the given layout label. All
/// values are clamped to `[0.0, 1.0]` (the JXL convention for f32
/// gamma-encoded inputs).
///
/// * `pq` — 2D ramp 0.0..1.0 in Y (PQ codeword), with R/G/B offsets so
///   the gradient is not just gray and the chroma path fires.
/// * `hlg` — 2D ramp scaled to peak near the HLG nominal range top
///   (≈ 0.75 codeword = 75% luminance scene-light).
/// * `bt709` — 2D ramp on BT.709 codewords (0..1 maps to 0..1 video
///   linear before the OETF), with the same chroma split.
fn synth(label: &str) -> Vec<f32> {
    let mut out = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            let u = x as f32 / (W - 1) as f32;
            let v = y as f32 / (H - 1) as f32;
            // base luminance ramp
            let base = match label {
                "pq" => 0.05 + 0.95 * (0.5 * u + 0.5 * v),
                "hlg" => 0.05 + 0.7 * (0.5 * u + 0.5 * v),
                "bt709" => 0.05 + 0.9 * (0.5 * u + 0.5 * v),
                _ => unreachable!(),
            };
            // chroma offsets per channel so PQ EOTF + XYB matter
            let r = (base + 0.05).clamp(0.0, 1.0);
            let g = (base - 0.02).clamp(0.0, 1.0);
            let b = (base - 0.08).clamp(0.0, 1.0);
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    out
}

/// Write a 32-bit float RGB PFM (little-endian) at the canonical
/// `-1.0` scale factor cjxl expects for raw f32 codewords. The PFM
/// header is `PF\n<w> <h>\n-1.0\n` followed by pixels in **bottom-up**
/// row order (PFM convention).
fn write_pfm_le(path: &std::path::Path, w: u32, h: u32, pixels: &[f32]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "PF")?;
    writeln!(f, "{w} {h}")?;
    writeln!(f, "-1.0")?;
    // PFM is bottom-up; our pixel buffer is top-down. Write rows
    // bottom-to-top so cjxl sees the same gradient as jxl-encoder.
    let row_floats = (w * 3) as usize;
    for y in (0..h as usize).rev() {
        let start = y * row_floats;
        let end = start + row_floats;
        f.write_all(cast_slice(&pixels[start..end]))?;
    }
    Ok(())
}

/// Run cjxl and return the resulting bytes. Returns `None` if cjxl
/// exited non-zero (we tolerate this so the sweep still produces a
/// TSV row with `cjxl_bytes=-1`).
fn run_cjxl(
    cjxl: &std::path::Path,
    pfm: &std::path::Path,
    out: &std::path::Path,
    color_space: &str,
    distance: f32,
    intensity_target: f32,
) -> Option<u64> {
    let _ = std::fs::remove_file(out);
    let status = Command::new(cjxl)
        .arg(pfm)
        .arg(out)
        .arg("-d")
        .arg(format!("{distance}"))
        .arg("-x")
        .arg(format!("color_space={color_space}"))
        .arg("--intensity_target")
        .arg(format!("{intensity_target}"))
        .arg("--quiet")
        .output()
        .ok()?;
    if !status.status.success() {
        eprintln!(
            "cjxl failed ({}): {}",
            status.status,
            String::from_utf8_lossy(&status.stderr).trim()
        );
        return None;
    }
    std::fs::metadata(out).ok().map(|m| m.len())
}

fn cjxl_version(cjxl: &std::path::Path) -> String {
    Command::new(cjxl)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(|l| l.trim().to_string()))
        .unwrap_or_else(|| "unknown".into())
}

fn git_rev() -> String {
    Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn hostname() -> String {
    Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cjxl = cjxl_bin();
    if !cjxl.exists() {
        return Err(format!("cjxl not found at {} — set CJXL env var", cjxl.display()).into());
    }
    let cjxl_ver = cjxl_version(&cjxl);
    let git = git_rev();
    let host = hostname();

    // Output paths under repo root (the example runs from the workspace
    // root via `cargo run -p jxl-encoder --example ...`).
    let out_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("benchmarks");
    std::fs::create_dir_all(&out_root)?;
    let stamp = chrono_like_utc();
    let tsv_path = out_root.join(format!("hdr_rd_sweep_{stamp}.tsv"));
    let meta_path = out_root.join(format!("hdr_rd_sweep_{stamp}.meta"));
    let scratch = PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into()))
        .join("hdr_rd_sweep");
    std::fs::create_dir_all(&scratch)?;

    let mut tsv = std::fs::File::create(&tsv_path)?;
    writeln!(
        tsv,
        "layout\tdistance\twidth\theight\tintensity_target\tours_bytes\tcjxl_bytes\tratio_ours_over_cjxl\tours_bpp\tcjxl_bpp"
    )?;

    let mut meta = std::fs::File::create(&meta_path)?;
    writeln!(
        meta,
        "# HDR RD sweep vs cjxl (jxl-encoder#44 / W4 follow-up)"
    )?;
    writeln!(meta, "git_rev={git}")?;
    writeln!(meta, "host={host}")?;
    writeln!(meta, "cjxl_version={cjxl_ver}")?;
    writeln!(
        meta,
        "command=cargo run -p jxl-encoder --release --example hdr_rd_sweep_vs_cjxl"
    )?;
    writeln!(meta, "size={W}x{H}")?;
    writeln!(meta, "distances={:?}", DISTANCES)?;
    writeln!(meta, "effort=7 (jxl-encoder default), 7 (cjxl default)")?;
    writeln!(
        meta,
        "metric=bytes only (no HDR-aware perceptual metric in tree)"
    )?;
    writeln!(meta, "tsv={}", tsv_path.display())?;
    meta.flush()?;

    eprintln!(
        "{:<6} {:>4} {:>12} {:>12} {:>10} {:>10}",
        "layout", "d", "ours_bytes", "cjxl_bytes", "ratio", "Δ%"
    );
    eprintln!("{}", "-".repeat(64));

    let mut rows = 0u32;
    for (label, layout, cjxl_cs, intensity_nits) in hdr_cases() {
        let pixels_f32 = synth(label);
        let pixels_bytes: &[u8] = cast_slice(&pixels_f32);

        // Materialize PFM once per layout (distance does not change pixels).
        let pfm_path = scratch.join(format!("{label}.pfm"));
        write_pfm_le(&pfm_path, W, H, &pixels_f32)?;

        for &d in DISTANCES {
            // --- ours ---
            let ce = match label {
                "pq" => ColorEncoding::bt2100_pq(),
                "hlg" => ColorEncoding::bt2100_hlg(),
                "bt709" => {
                    // BT.709 primaries with BT.709 transfer; not a
                    // built-in preset, but with_color_encoding accepts
                    // any explicit ColorEncoding. We rely on the
                    // RgbBt709F32 layout to drive the inverse OETF.
                    // The header signal will come from PixelLayout
                    // alone when no encoding is set.
                    ColorEncoding::srgb()
                }
                _ => unreachable!(),
            };
            // `with_intensity_target` / `with_color_encoding` live on
            // `EncodeRequest` (not `LossyConfig` directly), per
            // api.rs:5017+5037 inside `impl<'a> EncodeRequest<'a>`.
            let cfg = LossyConfig::new(d);
            let mut req = cfg
                .encode_request(W, H, layout)
                .with_intensity_target(intensity_nits);
            if label != "bt709" {
                req = req.with_color_encoding(ce);
            }
            let ours = req.encode(pixels_bytes)?;

            // Write "ours" to disk alongside the cjxl output so an
            // operator can spot-check both with `djxl` post-sweep.
            let ours_path = scratch.join(format!("{label}_d{d}.ours.jxl"));
            std::fs::write(&ours_path, &ours)?;

            // Roundtrip-validate "ours" with jxl-oxide — confirms the
            // codestream is decodable end-to-end and the HDR signaling
            // we emit is well-formed. Skipped on bytes-only failure
            // paths because then there's nothing to validate.
            {
                let reader = std::io::Cursor::new(&ours);
                match jxl_oxide::JxlImage::builder().read(reader) {
                    Ok(img) => {
                        let ce_dbg = format!("{:?}", img.image_header().metadata.colour_encoding);
                        let expect_tf = match label {
                            "pq" => "Pq",
                            "hlg" => "Hlg",
                            "bt709" => "Bt709",
                            _ => "",
                        };
                        if !ce_dbg.contains(expect_tf) {
                            eprintln!(
                                "WARN: {label} d={d}: header missing expected TF '{expect_tf}': {ce_dbg}"
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!("ERR: {label} d={d}: jxl-oxide parse failed: {e}");
                    }
                }
            }

            // --- cjxl ---
            let out_path = scratch.join(format!("{label}_d{d}.cjxl.jxl"));
            let cjxl_bytes = run_cjxl(&cjxl, &pfm_path, &out_path, cjxl_cs, d, intensity_nits);

            let ours_bytes = ours.len() as u64;
            let cjxl_bytes_u = cjxl_bytes.unwrap_or(0);
            let ratio = if cjxl_bytes_u > 0 {
                ours_bytes as f64 / cjxl_bytes_u as f64
            } else {
                f64::NAN
            };
            let pixels = (W * H) as f64;
            let ours_bpp = (ours_bytes as f64 * 8.0) / pixels;
            let cjxl_bpp = if cjxl_bytes_u > 0 {
                (cjxl_bytes_u as f64 * 8.0) / pixels
            } else {
                f64::NAN
            };
            writeln!(
                tsv,
                "{label}\t{d}\t{W}\t{H}\t{intensity_nits}\t{ours_bytes}\t{}\t{ratio:.4}\t{ours_bpp:.4}\t{cjxl_bpp:.4}",
                if cjxl_bytes_u > 0 {
                    cjxl_bytes_u.to_string()
                } else {
                    "-1".to_string()
                }
            )?;
            tsv.flush()?;
            let delta = if cjxl_bytes_u > 0 {
                (ours_bytes as f64 - cjxl_bytes_u as f64) / cjxl_bytes_u as f64 * 100.0
            } else {
                f64::NAN
            };
            eprintln!(
                "{label:<6} {d:>4} {ours_bytes:>12} {:>12} {ratio:>10.3} {delta:>9.1}",
                if cjxl_bytes_u > 0 {
                    cjxl_bytes_u.to_string()
                } else {
                    "FAIL".to_string()
                }
            );
            rows += 1;
        }
    }

    eprintln!();
    eprintln!("wrote {rows} rows → {}", tsv_path.display());
    eprintln!("meta → {}", meta_path.display());
    Ok(())
}

/// `YYYYmmddTHHMMSSZ` UTC timestamp without pulling in `chrono`.
fn chrono_like_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Days since 1970-01-01 + h/m/s; cheap calendar math good through
    // 2099-12-31. Matches the convention used by other benchmark
    // scripts in this repo.
    let days = secs / 86_400;
    let h = (secs / 3_600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    let (mut y, mut d) = (1970i64, days as i64);
    loop {
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let ydays = if leap { 366 } else { 365 };
        if d < ydays {
            break;
        }
        d -= ydays;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let months: [i64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut mo = 0usize;
    while mo < 12 && d >= months[mo] {
        d -= months[mo];
        mo += 1;
    }
    let day = d + 1;
    let month = mo as i64 + 1;
    format!("{y:04}{month:02}{day:02}T{h:02}{m:02}{s:02}Z")
}
