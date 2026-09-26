//! Design D — the Pareto-staircase invariant, with SSIMULACRA2 as the oracle.
//!
//! **Why this exists.** A/B/C in the RD-monotonicity plan are unfalsifiable
//! without a contract layer that says what "monotone" means and fails when it
//! is broken. This is that layer. It is also the shape zensim training wants:
//! a per-(image, effort) distance ladder whose (bytes, delivered-IQA) points
//! form a staircase, so a model can learn from a curve that is not
//! self-contradictory.
//!
//! Corroborated IQA inversions fail across every filter regime; byte inversions
//! are advisory. Known inversions remain visible without failing the run.
//! Missing inputs, failed encodes/decodes and non-finite scores fail the run.
//! Timings are diagnostics only: Rust encode wall and cjxl process wall have
//! different scopes, and the historical aggregate baseline is not grid-bound.
//!
//! Oracle choice: SSIMULACRA2 via `fast-ssim2` (PINNED at 0.7.1 -- 0.8.2 moves
//! every score, see CLAUDE.md), fed sRGB u8 on both sides because
//! `compute_ssimulacra2` linearises internally.
//!
//! Usage:
//!   rd_monotonicity_gate <corpus-dir> <out.tsv> [--images N] [--size N]
//!       [--efforts 3,5,7,9] [--distances 1,2] [--baseline <tsv>]
//!       [--update-baseline]
//! A single PNG may replace the corpus directory. Outputs must be new;
//! bitstreams, decoder logs, source crops and diffmaps live in <out.tsv>.artifacts.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[path = "distance_targeting_probe/decode.rs"]
mod decode;
#[path = "rd_monotonicity_gate/evidence.rs"]
mod evidence;

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;

/// Distances at which libjxl (and we) change reference filters, so byte and
/// boundary crossings are labelled separately. IQA is still checked. Source:
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

/// A stable identity for one adjacent-distance comparison, so a KNOWN
/// violation can be allow-listed without allow-listing a whole image.
fn violation_key(image: &str, effort: u8, lo: f32, hi: f32) -> String {
    format!("{image}\te{effort}\t{lo}\t{hi}")
}

/// Load the accepted-violation allowlist.
///
/// A gate that fails on day one for pre-existing issues gets disabled, not
/// fixed. This lets the gate go into automation NOW and fail only on NEW
/// regressions, while the known set stays visible in a committed file rather
/// than as a silently loosened threshold. Entries are
/// `image<TAB>eNN<TAB>lo_distance<TAB>hi_distance`; `#` comments allowed.
fn load_known(path: &str) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    {
        let text = std::fs::read_to_string(path).expect("read known violations");
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            out.insert(line.to_string());
        }
    }
    out
}

fn regime(d: f32) -> usize {
    FILTER_BOUNDARIES.iter().filter(|&&b| d > b).count()
}

fn walk(d: &Path) -> Vec<PathBuf> {
    if d.is_file() {
        return vec![d.to_path_buf()];
    }
    let mut v = Vec::new();
    for entry in std::fs::read_dir(d).expect("read corpus directory") {
        let path = entry.expect("read corpus entry").path();
        if path.is_dir() {
            v.extend(walk(&path));
        } else {
            v.push(path);
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
    assert!(
        a.len() >= 3,
        "usage: rd_monotonicity_gate <corpus-or-png> <new-output.tsv>"
    );
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
    assert!(max_images > 0 && n >= 8, "need images > 0 and size >= 8");
    assert!(
        !efforts.is_empty() && efforts.iter().all(|e| (1..=12).contains(e)),
        "invalid efforts"
    );
    let unique: std::collections::HashSet<_> = efforts.iter().collect();
    assert_eq!(unique.len(), efforts.len(), "duplicate efforts");
    let distances: Vec<f32> = arg(
        "--distances",
        &DISTANCES
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(","),
    )
    .split(',')
    .map(|v| v.parse().expect("distance"))
    .collect();
    assert!(
        distances.len() >= 2
            && distances.iter().all(|d| d.is_finite() && *d > 0.0)
            && distances.windows(2).all(|p| p[0] < p[1]),
        "distances must be finite, positive and strictly increasing"
    );
    let evidence = evidence::Evidence::new(&out_path, &a);
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&out_path)
        .expect("new TSV output");
    writeln!(out, "image\teffort\tdistance\tregime\tbytes\tssim2\tbfly\tms\tcjxl_bytes\tcjxl_ssim2\tcjxl_ms\tsource_sha256\tencoded_sha256\tcjxl_sha256\tdiffmap_sha256").unwrap();

    let cjxl = jxl_encoder::test_helpers::cjxl_path();
    let djxl = jxl_encoder::test_helpers::djxl_path();

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

    assert!(!picks.is_empty(), "no PNG inputs selected");
    let mut names = std::collections::HashSet::new();
    for path in &picks {
        assert!(
            names.insert(path.file_name().unwrap()),
            "duplicate input basename: {}",
            path.display()
        );
    }
    let mut cells: Vec<Cell> = Vec::new();
    for path in &picks {
        let img = image::open(path).unwrap_or_else(|e| panic!("input {}: {e}", path.display()));
        let rgb = img.to_rgb8();
        assert!(
            rgb.width() >= n && rgb.height() >= n,
            "input {} is smaller than requested crop {n}",
            path.display()
        );
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
        let source_hash = evidence::sha256(&std::fs::read(path).expect("source bytes"));
        let png = evidence.dir.join(format!("{source_hash}-{n}.source.png"));
        crop.save(&png).unwrap();

        for &e in &efforts {
            for &d in &distances {
                let t = Instant::now();
                let enc = LossyConfig::new(d)
                    .with_effort(e)
                    .encode_request(n, n, PixelLayout::Rgb8)
                    .encode(&src)
                    .expect("encode");
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let encoded_hash = evidence.retain(&enc, "jxl");
                evidence.verify(&enc, &encoded_hash, n, &djxl);
                let ssim2 = score(&enc, &src, n);

                let cj = evidence
                    .dir
                    .join(format!("{source_hash}-e{e}-d{d}.reference.jxl"));
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
                std::fs::write(
                    cj.with_extension("log"),
                    [&st.stdout[..], &st.stderr[..]].concat(),
                )
                .unwrap();
                assert!(
                    st.status.success(),
                    "cjxl failed: {}",
                    String::from_utf8_lossy(&st.stderr)
                );
                let reference = std::fs::read(&cj).expect("reference output");
                assert!(!reference.is_empty(), "empty reference output");
                let reference_hash = evidence.retain(&reference, "jxl");
                evidence.verify(&reference, &reference_hash, n, &djxl);
                let cjxl_bytes = reference.len();
                let cjxl_ssim2 = score(&reference, &src, n);
                let (bfly, diffmap_hash) = bfly_of(&enc, &orig_lin, &evidence);
                assert!(
                    ssim2.is_finite() && bfly.is_finite() && cjxl_ssim2.is_finite(),
                    "non-finite score: {name} e{e} d{d}"
                );
                writeln!(out, "{name}\t{e}\t{d}\t{}\t{}\t{ssim2:.4}\t{bfly:.5}\t{ms:.2}\t{cjxl_bytes}\t{cjxl_ssim2:.4}\t{cjxl_ms:.2}\t{source_hash}\t{encoded_hash}\t{reference_hash}\t{diffmap_hash}", regime(d), enc.len()).unwrap();
                out.flush().unwrap();
                eprintln!("{name} e{e} d{d}: retained and decoded both streams");

                cells.push(Cell {
                    image: name.clone(),
                    effort: e,
                    distance: d,
                    bytes: enc.len(),
                    ssim2,
                    bfly,
                    ms,
                    cjxl_bytes,
                    cjxl_ssim2,
                    cjxl_ms,
                });
            }
            eprintln!("{name} e{e} done");
        }
    }

    println!("wrote {} cells to {}", cells.len(), out_path.display());

    let known_path = arg("--known", "benchmarks/rd_monotonicity_known_violations.tsv");
    let known = load_known(&known_path);
    // Report the allowlist's reach explicitly. It keys on IMAGE FILENAME, and
    // nightly runs against the sha256-pinned `imazen-26-unprocessed` gate set
    // whose files differ (different pixels AND different names) from the
    // `png-v3` renders these entries were measured on. So a nightly line
    // reading "0 known" means the allowlist did not APPLY -- not that the known
    // violations are fixed. Printing the counts keeps that visible.
    println!(
        "allowlist: {} entries from {known_path}; {} of them name an image in this corpus",
        known.len(),
        known
            .iter()
            .filter(|k| {
                let name = k.split('\t').next().unwrap_or("");
                cells.iter().any(|c| c.image == name)
            })
            .count()
    );
    let (fatal, advisory, known_hits) = check_staircase(&cells, &known);
    let time_report = check_time(&cells, Path::new(&baseline_path), update_baseline);
    println!("\n{time_report}");

    if advisory.is_empty() {
        println!("ADVISORIES (bytes or single-metric inversions): no inversion");
    } else {
        let boundary = advisory.iter().filter(|v| v.contains("[boundary]")).count();
        println!(
            "ADVISORIES (bytes or single-metric inversions): {} inversions ({boundary} at filter boundaries, \
             where libjxl is not monotone either)",
            advisory.len()
        );
        for v in advisory.iter().take(12) {
            println!("    {v}");
        }
    }

    if !known_hits.is_empty() {
        println!(
            "\nKNOWN violations (allow-listed, not failing): {}",
            known_hits.len()
        );
        for v in known_hits.iter().take(20) {
            println!("    {v}");
        }
    }

    if fatal.is_empty() {
        println!(
            "\nIQA MONOTONICITY: clean — no NEW cell where both oracles invert ({} known)",
            known_hits.len()
        );
    } else {
        let within = fatal.iter().filter(|v| v.contains("within regime")).count();
        println!(
            "\nIQA MONOTONICITY: {} NEW VIOLATIONS ({within} within-regime targeting bugs, \
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
fn bfly_of(
    encoded: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    evidence: &evidence::Evidence,
) -> (f64, String) {
    let mut img = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(encoded))
        .expect("oxide header");
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).expect("oxide pixels");
    let fb = render.image_all_channels();
    let (buf, ch) = (fb.buf(), fb.channels());
    assert!(ch >= 3);
    assert_eq!(
        (fb.width(), fb.height()),
        (orig_linear.width(), orig_linear.height())
    );
    assert!(buf.iter().all(|v| v.is_finite()));
    let px: Vec<RGB<f32>> = (0..fb.width() * fb.height())
        .map(|i| RGB::new(buf[i * ch], buf[i * ch + 1], buf[i * ch + 2]))
        .collect();
    let dist: Img<Vec<RGB<f32>>> = Img::new(px, fb.width(), fb.height());
    let metric = butteraugli_linear(
        orig_linear.as_ref(),
        dist.as_ref(),
        &ButteraugliParams::default().with_compute_diffmap(true),
    )
    .expect("butteraugli");
    (
        metric.score,
        evidence.metric(&metric, fb.width(), fb.height()),
    )
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
    if fb.channels() < 3
        || buf.len() < px * fb.channels()
        || (fb.width(), fb.height()) != (n as usize, n as usize)
        || !buf.iter().all(|v| v.is_finite())
    {
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
fn check_staircase(
    cells: &[Cell],
    known: &std::collections::HashSet<String>,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    use std::collections::BTreeMap;
    let mut by: BTreeMap<(String, u8), Vec<&Cell>> = BTreeMap::new();
    for c in cells {
        by.entry((c.image.clone(), c.effort)).or_default().push(c);
    }
    let mut fatal = Vec::new();
    if cells.is_empty() {
        fatal.push("no measured cells".into());
    }
    for c in cells {
        if !c.ssim2.is_finite()
            || !c.bfly.is_finite()
            || !c.cjxl_ssim2.is_finite()
            || c.bytes == 0
            || c.cjxl_bytes == 0
        {
            fatal.push(format!(
                "invalid measurement: {} e{} d{}",
                c.image, c.effort, c.distance
            ));
        }
    }
    let mut advisory = Vec::new();
    let mut known_hits = Vec::new();
    for ((img, e), mut ladder) in by {
        ladder.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        if ladder.len() < 2 || ladder.windows(2).any(|p| p[0].distance >= p[1].distance) {
            fatal.push(format!("incomplete or duplicate ladder: {img} e{e}"));
        }
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
                let key = violation_key(&img, e, lo.distance, hi.distance);
                let sink = if known.contains(&key) {
                    &mut known_hits
                } else {
                    &mut fatal
                };
                sink.push(format!(
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
    (fatal, advisory, known_hits)
}

/// Diagnostic only: timings have different scopes and the old baseline is
/// not bound to a source/configuration grid. This does not enforce a time gate.
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
    let mut s = String::from(
        "TIME (diagnostic only; Rust encode vs cjxl process, historical baseline may use another grid):\n",
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(distance: f32) -> Cell {
        Cell {
            image: "fixture.png".into(),
            effort: 5,
            distance,
            bytes: 100,
            ssim2: 80.0,
            bfly: distance as f64,
            ms: 1.0,
            cjxl_bytes: 100,
            cjxl_ssim2: 80.0,
            cjxl_ms: 1.0,
        }
    }

    #[test]
    fn empty_run_cannot_pass() {
        assert!(!check_staircase(&[], &Default::default()).0.is_empty());
    }

    #[test]
    fn invalid_scores_cannot_pass_or_be_allowlisted() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for field in 0..3 {
                let mut cells = [cell(1.0), cell(2.0)];
                match field {
                    0 => cells[1].ssim2 = value,
                    1 => cells[1].bfly = value,
                    _ => cells[1].cjxl_ssim2 = value,
                }
                let known = [violation_key("fixture.png", 5, 1.0, 2.0)].into();
                assert!(
                    !check_staircase(&cells, &known).0.is_empty(),
                    "field {field}, value {value}"
                );
            }
        }
    }

    #[test]
    fn incomplete_or_duplicate_ladder_cannot_pass() {
        for cells in [vec![cell(1.0)], vec![cell(1.0), cell(1.0)]] {
            assert!(!check_staircase(&cells, &Default::default()).0.is_empty());
        }
    }

    #[test]
    fn corroborated_inversion_keeps_existing_thresholds() {
        let mut cells = [cell(1.0), cell(2.0)];
        cells[1].ssim2 = 81.0;
        cells[1].bfly = 0.9;
        assert_eq!(check_staircase(&cells, &Default::default()).0.len(), 1);
        let known = [violation_key("fixture.png", 5, 1.0, 2.0)].into();
        assert_eq!(check_staircase(&cells, &known).2.len(), 1);
        cells[1].bfly = 2.0;
        assert!(check_staircase(&cells, &Default::default()).0.is_empty());
    }
}
