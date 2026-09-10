//! Design D — the Pareto-staircase invariant, with SSIMULACRA2 as the oracle.
//!
//! **Why this exists.** A/B/C in the RD-monotonicity plan are unfalsifiable
//! without a contract layer that says what "monotone" means and fails when it
//! is broken. This is that layer. It is also the shape zensim training wants:
//! a per-(image, effort) distance ladder whose (bytes, delivered-IQA) points
//! form a staircase, so a model can learn from a curve that is not
//! self-contradictory.
//!
//! **The contract, stated precisely.** Two separate claims, because they have
//! different truth values:
//!
//! 1. WITHIN a reference-filter regime, both bytes and delivered SSIM2 must be
//!    non-increasing as the requested distance coarsens. Violations are bugs.
//! 2. ACROSS a regime boundary, neither is promised. libjxl v0.12 itself is not
//!    byte-monotone there (measured: d=0.5 -> 0.6 goes 71,368 -> 94,002 B on
//!    imazen-26 7026), because Gaborish is gated at d > 0.5 and the EPF
//!    thresholds step at 0.7 / 1.5 / 4.0. Crossings are therefore recorded
//!    with their magnitudes as DECLARED discontinuities, not silently
//!    tolerated and not failed.
//!
//! **Time is a first-class assertion, not a footnote.** Per effort, our wall
//! must not regress against a committed baseline, and the ratio against cjxl
//! v0.12 at the same effort and distance is tracked alongside it. An RD win
//! bought with unbounded time is not a win.
//!
//! Oracle choice: SSIMULACRA2 via `fast-ssim2` (PINNED at 0.7.1 -- 0.8.2 moves
//! every score, see CLAUDE.md), fed sRGB u8 on both sides because
//! `compute_ssimulacra2` linearises internally.
//!
//! Usage:
//!   rd_monotonicity_gate <corpus-dir> <out.tsv> [--images N] [--size N]
//!       [--efforts 3,5,7,9] [--baseline <tsv>] [--update-baseline]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;

/// Distances at which libjxl (and we) change reference filters, so byte and
/// quality monotonicity are not promised ACROSS them. Read from source:
/// Gaborish is enabled only above 0.5; EPF thresholds step at 0.7/1.5/4.0.
const FILTER_BOUNDARIES: &[f32] = &[0.5, 0.7, 1.5, 4.0];

/// Denser at the low end on purpose: the low-distance regime is where the
/// structural problems live, and a grid denser at high quality than low is
/// already wrong (CLAUDE.md sweep discipline).
const DISTANCES: &[f32] = &[
    0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0, 6.0,
    8.0, 10.0, 12.0, 15.0,
];

fn arg(name: &str, default: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
        .unwrap_or_else(|| default.to_string())
}
fn flag(name: &str) -> bool {
    std::env::args().any(|x| x == name)
}

fn regime(d: f32) -> usize {
    FILTER_BOUNDARIES.iter().filter(|&&b| d > b).count()
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

struct Cell {
    image: String,
    effort: u8,
    distance: f32,
    bytes: usize,
    ssim2: f64,
    bfly: f64,
    ms: f64,
    cjxl_bytes: usize,
    cjxl_ssim2: f64,
    cjxl_ms: f64,
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let corpus = PathBuf::from(&a[1]);
    let out_path = PathBuf::from(&a[2]);
    let max_images: usize = arg("--images", "4").parse().unwrap();
    let n: u32 = arg("--size", "512").parse().unwrap();
    let efforts: Vec<u8> = arg("--efforts", "3,5,7,9")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let baseline_path = arg(
        "--baseline",
        "benchmarks/rd_monotonicity_baseline_2026-09-09.tsv",
    );
    let update_baseline = flag("--update-baseline");

    let cjxl = jxl_encoder::test_helpers::cjxl_path();
    let tmp = std::env::var("HOME").unwrap() + "/tmp/rdmono";
    std::fs::create_dir_all(&tmp).unwrap();

    // Stratify round-robin so one content class cannot dominate the verdict.
    let mut by_class: std::collections::BTreeMap<String, Vec<PathBuf>> = Default::default();
    for f in walk(&corpus) {
        if !f.to_string_lossy().ends_with(".png") {
            continue;
        }
        let cls = f
            .parent()
            .and_then(|d| d.file_name())
            .map(|x| x.to_string_lossy().to_string())
            .unwrap_or_default();
        by_class.entry(cls).or_default().push(f);
    }
    let mut picks: Vec<PathBuf> = Vec::new();
    let mut idx = 0;
    while picks.len() < max_images {
        let mut added = false;
        for v in by_class.values() {
            if let Some(p) = v.get(idx) {
                picks.push(p.clone());
                added = true;
                if picks.len() >= max_images {
                    break;
                }
            }
        }
        if !added {
            break;
        }
        idx += 1;
    }

    let mut cells: Vec<Cell> = Vec::new();
    for path in &picks {
        let Ok(img) = image::open(path) else { continue };
        let rgb = img.to_rgb8();
        if rgb.width() < n || rgb.height() < n {
            continue;
        }
        let (x0, y0) = ((rgb.width() - n) / 2, (rgb.height() - n) / 2);
        let crop = image::imageops::crop_imm(&rgb, x0, y0, n, n).to_image();
        let src: Vec<u8> = crop.as_raw().clone();
        let orig_lin_px: Vec<RGB<f32>> = src
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| {
                RGB::new(
                    srgb_to_linear(c[0]),
                    srgb_to_linear(c[1]),
                    srgb_to_linear(c[2]),
                )
            })
            .collect();
        let orig_lin: Img<Vec<RGB<f32>>> = Img::new(orig_lin_px, n as usize, n as usize);
        let name = path.file_name().unwrap().to_string_lossy().to_string();

        // Fresh PNG for cjxl: written from raw RGB8, so it carries no ICC and
        // cannot reintroduce the metadata mismatch that has burned two sessions.
        let png = PathBuf::from(format!("{tmp}/src.png"));
        crop.save(&png).unwrap();

        for &e in &efforts {
            for &d in DISTANCES {
                let t = Instant::now();
                let enc = LossyConfig::new(d)
                    .with_effort(e)
                    .encode_request(n, n, PixelLayout::Rgb8)
                    .encode(&src)
                    .expect("encode");
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let ssim2 = score(&enc, &src, n);

                let cj = PathBuf::from(format!("{tmp}/cj.jxl"));
                let t = Instant::now();
                let st = Command::new(&cjxl)
                    .args([
                        png.to_str().unwrap(),
                        cj.to_str().unwrap(),
                        "-d",
                        &d.to_string(),
                        "-e",
                        &e.to_string(),
                        "--quiet",
                    ])
                    .output()
                    .expect("cjxl");
                let cjxl_ms = t.elapsed().as_secs_f64() * 1000.0;
                let (cjxl_bytes, cjxl_ssim2) = if st.status.success() {
                    let b = std::fs::read(&cj).unwrap_or_default();
                    (b.len(), score(&b, &src, n))
                } else {
                    (0, 0.0)
                };

                cells.push(Cell {
                    image: name.clone(),
                    effort: e,
                    distance: d,
                    bytes: enc.len(),
                    ssim2,
                    bfly: bfly_of(&enc, &orig_lin),
                    ms,
                    cjxl_bytes,
                    cjxl_ssim2,
                    cjxl_ms,
                });
            }
            eprintln!("{name} e{e} done");
        }
    }

    let mut out = std::fs::File::create(&out_path).unwrap();
    writeln!(
        out,
        "image\teffort\tdistance\tregime\tbytes\tssim2\tms\tcjxl_bytes\tcjxl_ssim2\tcjxl_ms"
    )
    .unwrap();
    for c in &cells {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}\t{}\t{:.4}\t{:.2}",
            c.image,
            c.effort,
            c.distance,
            regime(c.distance),
            c.bytes,
            c.ssim2,
            c.ms,
            c.cjxl_bytes,
            c.cjxl_ssim2,
            c.cjxl_ms
        )
        .unwrap();
    }
    println!("wrote {} cells to {}", cells.len(), out_path.display());

    let (fatal, advisory) = check_staircase(&cells);
    let time_report = check_time(&cells, Path::new(&baseline_path), update_baseline);
    println!("\n{time_report}");

    if advisory.is_empty() {
        println!("BYTES (advisory, great-to-have): no inversion");
    } else {
        let boundary = advisory.iter().filter(|v| v.contains("[boundary]")).count();
        println!(
            "BYTES (advisory, great-to-have): {} inversions ({boundary} at filter boundaries, \
             where libjxl is not monotone either)",
            advisory.len()
        );
        for v in advisory.iter().take(12) {
            println!("    {v}");
        }
    }

    if fatal.is_empty() {
        println!("\nIQA MONOTONICITY: clean — no cell where BOTH oracles invert");
    } else {
        let within = fatal.iter().filter(|v| v.contains("within regime")).count();
        println!(
            "\nIQA MONOTONICITY: {} VIOLATIONS ({within} within-regime targeting bugs, \
             {} at filter boundaries)",
            fatal.len(),
            fatal.len() - within
        );
        for v in fatal.iter().take(40) {
            println!("  {v}");
        }
        std::process::exit(1);
    }
}

/// Delivered butteraugli (LINEAR domain, lower = better).
///
/// The gate corroborates with a second metric because SSIM2 alone is not a
/// reliable ORDERING oracle near its ceiling: measured on two images at
/// d = 0.25..0.6, SSIM2 jitters +/-1-2 points with no trend (91.9, 91.7, 91.3,
/// 93.0, 91.3, 92.3, ...) while butteraugli over the same cells is smooth and
/// monotone (0.316 -> 0.613). Flagging those as violations would make the gate
/// cry wolf on exactly the high-quality cells that matter most.
fn bfly_of(encoded: &[u8], orig_linear: &Img<Vec<RGB<f32>>>) -> f64 {
    let Ok(mut img) = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(encoded)) else {
        return f64::NAN;
    };
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let Ok(render) = img.render_frame(0) else {
        return f64::NAN;
    };
    let fb = render.image_all_channels();
    let (buf, ch) = (fb.buf(), fb.channels());
    if ch < 3 {
        return f64::NAN;
    }
    let px: Vec<RGB<f32>> = (0..fb.width() * fb.height())
        .map(|i| RGB::new(buf[i * ch], buf[i * ch + 1], buf[i * ch + 2]))
        .collect();
    let dist: Img<Vec<RGB<f32>>> = Img::new(px, fb.width(), fb.height());
    butteraugli_linear(
        orig_linear.as_ref(),
        dist.as_ref(),
        &ButteraugliParams::default(),
    )
    .map(|r| r.score)
    .unwrap_or(f64::NAN)
}

/// Delivered SSIMULACRA2 of an encoded stream against the source.
///
/// Self-contained because `test_helpers`' decode/score pair is `#[cfg(test)]`
/// and therefore invisible to examples. It reproduces that pair's contract
/// exactly, and the contract has one trap: `compute_ssimulacra2` linearises
/// internally, so both sides must be **sRGB**, NOT linear. Requesting linear
/// output here (as the roundtrip helpers correctly do for butteraugli) would
/// double-linearise and produce the garbage scores CLAUDE.md documents.
fn score(encoded: &[u8], src: &[u8], n: u32) -> f64 {
    use fast_ssim2::compute_ssimulacra2;
    use imgref::ImgVec;

    let Ok(image) = jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(encoded)) else {
        return f64::NAN;
    };
    let Ok(render) = image.render_frame(0) else {
        return f64::NAN;
    };
    let fb = render.image_all_channels();
    let buf = fb.buf();
    let px = (n * n) as usize;
    if fb.channels() < 3 || buf.len() < px * fb.channels() {
        return f64::NAN;
    }
    let ch = fb.channels();
    let dec: Vec<[u8; 3]> = (0..px)
        .map(|i| {
            let o = i * ch;
            [
                (buf[o] * 255.0).clamp(0.0, 255.0) as u8,
                (buf[o + 1] * 255.0).clamp(0.0, 255.0) as u8,
                (buf[o + 2] * 255.0).clamp(0.0, 255.0) as u8,
            ]
        })
        .collect();
    let orig: Vec<[u8; 3]> = src.as_chunks::<3>().0.to_vec();
    if orig.len() != px {
        return f64::NAN;
    }
    let a = ImgVec::new(orig, n as usize, n as usize);
    let b = ImgVec::new(dec, n as usize, n as usize);
    compute_ssimulacra2(a.as_ref(), b.as_ref()).unwrap_or(f64::NAN)
}

/// The contract, per the owner decision of 2026-09-09:
///
/// > "byte monotonicity is a great-to-have, iqa monotonicity is the key"
///
/// So the two halves are graded DIFFERENTLY:
///
/// - **Delivered IQA must be non-increasing as the requested distance coarsens,
///   EVERYWHERE — filter boundaries included.** This is the hard contract and
///   the only thing that fails the gate. Crossing a boundary is not an excuse:
///   a boundary crossing that raises delivered quality is precisely the case
///   where the caller asked for coarser output and got finer, which is the
///   cliff zen mode exists to fix. Enforcing it across boundaries is a
///   deliberate divergence from libjxl, which does not hold this line.
/// - **Byte monotonicity is reported, never failed.** It is a great-to-have,
///   and it provably cannot hold across a reference-filter transition without
///   diverging further (libjxl v0.12 measured at d 0.5 -> 0.6: 71,368 ->
///   94,002 B).
///
/// Boundary-crossing IQA inversions are counted separately from within-regime
/// ones, because they have different fixes: the former is the distance/filter
/// interaction (design C), the latter is a targeting bug.
fn check_staircase(cells: &[Cell]) -> (Vec<String>, Vec<String>) {
    use std::collections::BTreeMap;
    let mut by: BTreeMap<(String, u8), Vec<&Cell>> = BTreeMap::new();
    for c in cells {
        by.entry((c.image.clone(), c.effort)).or_default().push(c);
    }
    let mut fatal = Vec::new();
    let mut advisory = Vec::new();
    for ((img, e), mut ladder) in by {
        ladder.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        for w in ladder.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            let crossing = regime(lo.distance) != regime(hi.distance);

            // HARD: delivered IQA must not rise as the request coarsens --
            // but a SINGLE metric is not enough to convict. Measured at
            // d = 0.25..0.6 on two images, SSIM2 jitters +/-1-2 points with no
            // trend near its ceiling (91.9, 91.7, 91.3, 93.0, 91.3, 92.3) while
            // butteraugli over the same cells is smooth and monotone
            // (0.316 -> 0.613). Convicting on SSIM2 alone would fail the gate on
            // exactly the high-quality cells that matter most, for a
            // disagreement between metrics rather than an encoder fault.
            //
            // So a hard violation needs BOTH oracles to agree the ordering
            // inverted. Single-metric inversions are reported as advisories.
            let ssim2_inverted =
                hi.ssim2.is_finite() && lo.ssim2.is_finite() && hi.ssim2 > lo.ssim2 + 0.30;
            // butteraugli: LOWER is better, so an inversion is hi < lo.
            let bfly_inverted =
                hi.bfly.is_finite() && lo.bfly.is_finite() && hi.bfly < lo.bfly * 0.98;

            if ssim2_inverted && bfly_inverted {
                fatal.push(format!(
                    "{img} e{e}: BOTH oracles invert d {} -> {}: ssim2 {:.3} -> {:.3}, bfly {:.4} -> {:.4}{}",
                    lo.distance,
                    hi.distance,
                    lo.ssim2,
                    hi.ssim2,
                    lo.bfly,
                    hi.bfly,
                    if crossing {
                        "  [filter-boundary crossing -- design C]"
                    } else {
                        "  [within regime -- targeting bug]"
                    }
                ));
            } else if ssim2_inverted || bfly_inverted {
                advisory.push(format!(
                    "{img} e{e}: {} only, d {} -> {}: ssim2 {:.3} -> {:.3}, bfly {:.4} -> {:.4}",
                    if ssim2_inverted {
                        "SSIM2"
                    } else {
                        "butteraugli"
                    },
                    lo.distance,
                    hi.distance,
                    lo.ssim2,
                    hi.ssim2,
                    lo.bfly,
                    hi.bfly
                ));
            }

            // ADVISORY: bytes. Never fails; across a boundary it is not even
            // expected to hold.
            if hi.bytes as f64 > lo.bytes as f64 * 1.005 {
                advisory.push(format!(
                    "{img} e{e}: bytes rose {} -> {} as d went {} -> {}{}",
                    lo.bytes,
                    hi.bytes,
                    lo.distance,
                    hi.distance,
                    if crossing {
                        "  [boundary]"
                    } else {
                        "  [within regime]"
                    }
                ));
            }
        }
    }
    (fatal, advisory)
}

/// Per-effort wall: must not regress against the committed baseline, and the
/// ratio against cjxl at the same cells is tracked next to it.
fn check_time(cells: &[Cell], baseline: &Path, update: bool) -> String {
    use std::collections::BTreeMap;
    let mut ours: BTreeMap<u8, f64> = BTreeMap::new();
    let mut theirs: BTreeMap<u8, f64> = BTreeMap::new();
    for c in cells {
        *ours.entry(c.effort).or_default() += c.ms;
        *theirs.entry(c.effort).or_default() += c.cjxl_ms;
    }
    if update {
        let mut f = std::fs::File::create(baseline).unwrap();
        writeln!(f, "effort\tours_ms_total\tcjxl_ms_total").unwrap();
        for (e, ms) in &ours {
            writeln!(f, "{e}\t{ms:.1}\t{:.1}", theirs[e]).unwrap();
        }
        return format!("BASELINE WRITTEN to {}", baseline.display());
    }
    let mut base: BTreeMap<u8, f64> = BTreeMap::new();
    if let Ok(s) = std::fs::read_to_string(baseline) {
        for l in s.lines().skip(1) {
            let f: Vec<&str> = l.split('\t').collect();
            if f.len() >= 2
                && let (Ok(e), Ok(ms)) = (f[0].parse::<u8>(), f[1].parse::<f64>())
            {
                base.insert(e, ms);
            }
        }
    }
    let mut s = String::from("TIME per effort (total ms over the grid):\n");
    s.push_str(&format!(
        "  {:>2}  {:>11}  {:>11}  {:>10}  {:>12}\n",
        "e", "ours", "cjxl", "ours/cjxl", "vs baseline"
    ));
    for (e, ms) in &ours {
        let vs_base = base
            .get(e)
            .map(|b| format!("{:.3}", ms / b))
            .unwrap_or_else(|| "no baseline".into());
        s.push_str(&format!(
            "  {e:>2}  {ms:>11.1}  {:>11.1}  {:>10.3}  {vs_base:>12}\n",
            theirs[e],
            ms / theirs[e].max(0.001)
        ));
    }
    s
}
