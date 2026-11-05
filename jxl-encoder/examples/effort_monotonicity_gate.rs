//! Design B — the effort envelope: effort N may not lose to effort N-1.
//!
//! Companion to `rd_monotonicity_gate` (design D), which walks the DISTANCE
//! axis. This walks the EFFORT axis, where the failures look different: not
//! inversions but **dominance** failures, where a higher effort costs strictly
//! more time for no gain.
//!
//! Three gradings, and they are deliberately not the same severity:
//!
//! - **INVERSION (hard).** Effort N produces more bytes and worse quality
//!   than N-1 beyond the existing grading slack; a lossless byte rise alone
//!   fails. The lossy check requires both axes to worsen.
//! - **DOMINATED (hard for lossless, advisory for lossy).** Effort N produces
//!   equal byte counts to N-1 while costing measurably more wall. The
//!   caller paid for effort that did nothing. Measured instance this exists to
//!   catch: on the f32 lossless path e3 and e5 are byte-identical on 26/26
//!   (image, size) pairs, so e5 is e5 time for e3 bytes. Advisory on the lossy
//!   path only because a tie there can be a genuine quality-neutral plateau.
//! - **THIN (advisory).** Effort N costs > 1.5x the wall of N-1 for < 0.5 %
//!   byte gain. Not wrong, but it is what an effort ladder should not look
//!   like, and it is the shape that makes an effort level un-recommendable.
//!
//! Every selected input must complete its ladder. Streams and source crops
//! are retained beside the new output, with full Rust and djxl decoding.
//! Lossless has no quality axis, so its grading is bytes-and-wall only. Lossy
//! is scored with SSIMULACRA2 (fast-ssim2, PINNED 0.7.1; sRGB in, since
//! `compute_ssimulacra2` linearises internally).
//!
//! Usage:
//!   effort_monotonicity_gate <corpus> <out.tsv> [--images N] [--size N]
//!       [--efforts 1,3,5,7,9] [--distances 1.0,4.0] [--reps 3]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[path = "distance_targeting_probe/decode.rs"]
mod decode;
#[path = "effort_monotonicity_gate/evidence.rs"]
mod evidence;

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
        {
            for e in std::fs::read_dir(&p).expect("read corpus directory") {
                let e = e.expect("read corpus entry");
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
    path_kind: String, // "lossless" | "lossy d=X"
    effort: u8,
    bytes: usize,
    ssim2: f64,
    ms: f64,
    source_sha256: String,
    encoded_sha256: String,
}

fn score(encoded: &[u8], src: &[u8], n: u32) -> f64 {
    use fast_ssim2::compute_ssimulacra2;
    use imgref::ImgVec;
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(encoded))
        .expect("scoring header");
    let render = image.render_frame(0).expect("scoring frame");
    let fb = render.image_all_channels();
    let buf = fb.buf();
    assert_eq!((fb.width(), fb.height()), (n as usize, n as usize));
    assert!(
        buf.iter().all(|v| v.is_finite()),
        "non-finite scoring pixels"
    );
    let px = (n * n) as usize;
    let ch = fb.channels();
    assert!(ch >= 3 && buf.len() == px * ch, "scoring shape");
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
    assert_eq!(orig.len(), px, "source shape");
    let value = compute_ssimulacra2(
        ImgVec::new(orig, n as usize, n as usize).as_ref(),
        ImgVec::new(dec, n as usize, n as usize).as_ref(),
    )
    .expect("SSIMULACRA2 score");
    assert!(value.is_finite(), "non-finite score");
    value
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(
        a.len() >= 3,
        "usage: effort_monotonicity_gate <corpus> <out.tsv> [options]"
    );
    let corpus = PathBuf::from(&a[1]);
    let out_path = PathBuf::from(&a[2]);
    let max_images: usize = arg("--images", "6").parse().unwrap();
    let n: u32 = arg("--size", "512").parse().unwrap();
    let efforts: Vec<u8> = arg("--efforts", "1,3,5,7,9")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let distances: Vec<f32> = arg("--distances", "1.0,4.0")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let reps: u32 = arg("--reps", "3").parse().unwrap();

    assert!(
        max_images > 0 && reps > 0 && n >= 8,
        "positive images/repetitions and size >= 8 required"
    );
    assert!(
        efforts.len() >= 2 && efforts.iter().all(|e| (1..=11).contains(e)),
        "requires at least two valid efforts"
    );
    assert!(
        efforts.windows(2).all(|w| w[0] < w[1]),
        "efforts must increase without duplicates"
    );
    assert!(
        !distances.is_empty() && distances.iter().all(|d| d.is_finite() && *d > 0.0),
        "requires finite positive distances"
    );
    for (i, d) in distances.iter().enumerate() {
        assert!(!distances[..i].contains(d), "duplicate distance");
    }

    // Stratify so one content class cannot decide the verdict.
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
    let mut i = 0;
    while picks.len() < max_images {
        let mut added = false;
        for v in by_class.values() {
            if let Some(p) = v.get(i) {
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
        i += 1;
    }

    assert!(!picks.is_empty(), "no PNG inputs selected");
    let mut names = std::collections::HashSet::new();
    for path in &picks {
        assert!(
            names.insert(path.file_name().unwrap()),
            "duplicate input basename"
        );
    }
    let evidence = evidence::Evidence::new(&out_path, &a);
    let mut cells: Vec<Cell> = Vec::new();
    for path in &picks {
        let img = image::open(path).expect("open selected PNG");
        let rgb = img.to_rgb8();
        assert!(
            rgb.width() >= n && rgb.height() >= n,
            "selected PNG too small: {}",
            path.display()
        );
        let (x0, y0) = ((rgb.width() - n) / 2, (rgb.height() - n) / 2);
        let src: Vec<u8> = image::imageops::crop_imm(&rgb, x0, y0, n, n)
            .to_image()
            .as_raw()
            .clone();
        let name = path.file_name().unwrap().to_string_lossy().to_string();

        let source_sha256 = evidence.retain(&src, "rgb8");
        let original_hash =
            evidence.retain(&std::fs::read(path).expect("source PNG"), "source.png");
        evidence.retain(
            &serde_json::to_vec(&serde_json::json!({
                "source": path, "source_png_sha256": original_hash,
                "crop_rgb8_sha256": source_sha256, "crop_origin": [x0, y0], "size": n
            }))
            .unwrap(),
            "source.json",
        );
        for &e in &efforts {
            // Lossless: bytes and wall only, no quality axis.
            let mut best = f64::INFINITY;
            let mut bytes = 0usize;
            let mut encoded_sha256 = String::new();
            for _ in 0..reps {
                let t = Instant::now();
                let d = LosslessConfig::new()
                    .with_effort(e)
                    .encode(&src, n, n, PixelLayout::Rgb8)
                    .expect("lossless");
                best = best.min(t.elapsed().as_secs_f64() * 1000.0);
                bytes = d.len();
                let key = evidence.verify(&d, &src, n, true);
                assert!(
                    encoded_sha256.is_empty() || encoded_sha256 == key,
                    "nondeterministic lossless repeat"
                );
                encoded_sha256 = key;
            }
            cells.push(Cell {
                image: name.clone(),
                path_kind: "lossless".into(),
                effort: e,
                bytes,
                ssim2: f64::NAN,
                ms: best,
                source_sha256: source_sha256.clone(),
                encoded_sha256,
            });

            for &dist in &distances {
                let mut best = f64::INFINITY;
                let mut bytes = 0usize;
                let mut ss = f64::NAN;
                let mut encoded_sha256 = String::new();
                for _ in 0..reps {
                    let t = Instant::now();
                    let enc = LossyConfig::new(dist)
                        .with_effort(e)
                        .encode_request(n, n, PixelLayout::Rgb8)
                        .encode(&src)
                        .expect("lossy");
                    best = best.min(t.elapsed().as_secs_f64() * 1000.0);
                    bytes = enc.len();
                    let key = evidence.verify(&enc, &src, n, false);
                    assert!(
                        encoded_sha256.is_empty() || encoded_sha256 == key,
                        "nondeterministic lossy repeat"
                    );
                    encoded_sha256 = key;
                    if ss.is_nan() {
                        ss = score(&enc, &src, n);
                    }
                }
                cells.push(Cell {
                    image: name.clone(),
                    path_kind: format!("lossy d={dist}"),
                    effort: e,
                    bytes,
                    ssim2: ss,
                    ms: best,
                    source_sha256: source_sha256.clone(),
                    encoded_sha256,
                });
            }
        }
        eprintln!("{name} done");
    }

    assert_eq!(
        cells.len(),
        picks.len() * efforts.len() * (1 + distances.len()),
        "incomplete grid"
    );
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&out_path)
        .unwrap();
    writeln!(
        out,
        "image\tpath\teffort\tbytes\tssim2\tms\tsource_sha256\tencoded_sha256"
    )
    .unwrap();
    for c in &cells {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{:.4}\t{:.2}\t{}\t{}",
            c.image,
            c.path_kind,
            c.effort,
            c.bytes,
            c.ssim2,
            c.ms,
            c.source_sha256,
            c.encoded_sha256
        )
        .unwrap();
    }
    println!("wrote {} cells to {}", cells.len(), out_path.display());

    let (hard, advisory) = grade(&cells);
    if advisory.is_empty() {
        println!("\nADVISORY: none");
    } else {
        println!("\nADVISORY: {} findings", advisory.len());
        for v in advisory.iter().take(30) {
            println!("    {v}");
        }
    }
    if hard.is_empty() {
        println!("\nEFFORT ENVELOPE: clean — no effort loses to a cheaper one");
    } else {
        println!("\nEFFORT ENVELOPE: {} VIOLATIONS", hard.len());
        for v in hard.iter().take(40) {
            println!("  {v}");
        }
        std::process::exit(1);
    }
}

fn grade(cells: &[Cell]) -> (Vec<String>, Vec<String>) {
    use std::collections::BTreeMap;
    let mut by: BTreeMap<(String, String), Vec<&Cell>> = BTreeMap::new();
    for c in cells {
        by.entry((c.image.clone(), c.path_kind.clone()))
            .or_default()
            .push(c);
    }
    let (mut hard, mut adv) = (Vec::new(), Vec::new());
    for ((img, kind), mut ladder) in by {
        ladder.sort_by_key(|c| c.effort);
        let lossless = kind == "lossless";
        for w in ladder.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            // The bar is STRICT PARETO DOMINATION: effort N "loses" only when
            // it is worse on BOTH axes. An earlier draft flagged any byte rise
            // where quality merely failed to fall, which mislabels genuine
            // trades -- e5 -> e7 at +2.2 % bytes for +2.47 SSIM2 is a move
            // ALONG the curve, not a regression. That draft produced 5 false
            // positives on the first corpus it ran against.
            let bytes_worse = hi.bytes as f64 > lo.bytes as f64 * 1.005;
            let qual_worse =
                hi.ssim2.is_finite() && lo.ssim2.is_finite() && hi.ssim2 + 0.30 < lo.ssim2;

            if bytes_worse && qual_worse {
                hard.push(format!(
                    "{img} [{kind}]: e{} -> e{} DOMINATED -- bytes {} -> {} (+{:.1}%) AND ssim2 {:.3} -> {:.3} ({:+.3})",
                    lo.effort,
                    hi.effort,
                    lo.bytes,
                    hi.bytes,
                    100.0 * (hi.bytes as f64 / lo.bytes as f64 - 1.0),
                    lo.ssim2,
                    hi.ssim2,
                    hi.ssim2 - lo.ssim2
                ));
                continue;
            }
            // Lossless has no quality axis, so a byte rise there is unambiguous.
            if lossless && bytes_worse {
                hard.push(format!(
                    "{img} [{kind}]: e{} -> e{} bytes ROSE {} -> {} (lossless: no quality axis to trade against)",
                    lo.effort, hi.effort, lo.bytes, hi.bytes
                ));
                continue;
            }
            // A byte rise WITH a quality rise is a trade: recorded so the
            // ladder's shape stays visible, never failed on.
            if bytes_worse {
                adv.push(format!(
                    "{img} [{kind}]: e{} -> e{} trade: bytes +{:.1}% for ssim2 {:+.3}",
                    lo.effort,
                    hi.effort,
                    100.0 * (hi.bytes as f64 / lo.bytes as f64 - 1.0),
                    hi.ssim2 - lo.ssim2
                ));
            }
            // DOMINATED: identical bytes, measurably more time.
            if hi.bytes == lo.bytes && hi.ms > lo.ms * 1.10 {
                let msg = format!(
                    "{img} [{kind}]: e{} -> e{} equal byte counts ({} B) but {:.1}x the wall ({:.1} -> {:.1} ms)",
                    lo.effort,
                    hi.effort,
                    hi.bytes,
                    hi.ms / lo.ms,
                    lo.ms,
                    hi.ms
                );
                if lossless {
                    hard.push(msg);
                } else {
                    adv.push(msg);
                }
                continue;
            }
            // THIN: a lot more time for almost nothing.
            let gain = 1.0 - (hi.bytes as f64 / lo.bytes as f64);
            if hi.ms > lo.ms * 1.5 && gain < 0.005 {
                adv.push(format!(
                    "{img} [{kind}]: e{} -> e{} costs {:.1}x wall for {:.2}% bytes",
                    lo.effort,
                    hi.effort,
                    hi.ms / lo.ms,
                    gain * 100.0
                ));
            }
        }
    }
    (hard, adv)
}
