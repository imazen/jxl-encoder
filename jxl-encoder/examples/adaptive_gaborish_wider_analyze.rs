//! Aggregate the wider-corpus adaptive-Gaborish sweep into a default-on
//! decision report.
//!
//! Reads the TSV produced by `adaptive_gaborish_wider_corpus`, pairs each
//! (image, distance, effort) into (fixed, adapt), and prints:
//!   - per-class × per-distance × per-effort mean of bytes-delta-pct,
//!     butteraugli-delta-pct, ssim2-delta
//!   - top-3 wins (most-negative bytes-delta) and top-3 losses (most-positive
//!     butteraugli-delta) overall
//!   - default-on decision (per task spec):
//!     Photos: bytes Δ ≤ -1.0% mean AND butteraugli Δ ≤ +2% mean
//!             AND no cell shows butteraugli Δ > +5%
//!     Screenshots: bytes Δ ≤ +0.5% mean AND butteraugli Δ ≤ +2% mean
//!     e5 AND e7 both must pass.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --example adaptive_gaborish_wider_analyze \
//!       -- --in benchmarks/adaptive_gaborish_wider_corpus_2026-05-18.tsv

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Debug, Clone)]
struct Row {
    image: String,
    class: String,
    distance: f64,
    effort: u8,
    mode: String,
    bytes: u64,
    butteraugli: f64,
    ssim2: f64,
}

#[derive(Debug, Clone)]
struct Pair {
    image: String,
    class: String,
    distance: f64,
    effort: u8,
    fixed_bytes: u64,
    adapt_bytes: u64,
    fixed_bfly: f64,
    adapt_bfly: f64,
    fixed_ssim2: f64,
    adapt_ssim2: f64,
}

impl Pair {
    fn bytes_delta_pct(&self) -> f64 {
        100.0 * (self.adapt_bytes as f64 - self.fixed_bytes as f64) / (self.fixed_bytes as f64)
    }
    fn bfly_delta_pct(&self) -> f64 {
        if self.fixed_bfly <= 0.0 {
            0.0
        } else {
            100.0 * (self.adapt_bfly - self.fixed_bfly) / self.fixed_bfly
        }
    }
    fn ssim2_delta(&self) -> f64 {
        self.adapt_ssim2 - self.fixed_ssim2
    }
}

fn parse_in_path() -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--in" {
            if let Some(v) = args.next() {
                return PathBuf::from(v);
            }
        }
    }
    PathBuf::from("benchmarks/adaptive_gaborish_wider_corpus_2026-05-18.tsv")
}

fn read_rows(path: &PathBuf) -> Vec<Row> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("ERROR open {}: {}", path.display(), e);
        std::process::exit(1);
    });
    let r = BufReader::new(f);
    let mut out = Vec::new();
    for line in r.lines().map_while(Result::ok) {
        if line.starts_with('#') || line.starts_with("image\t") || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 10 {
            continue;
        }
        let Ok(distance) = cols[4].parse::<f64>() else {
            continue;
        };
        let Ok(effort) = cols[5].parse::<u8>() else {
            continue;
        };
        let Ok(bytes) = cols[7].parse::<u64>() else {
            continue;
        };
        let Ok(butteraugli) = cols[8].parse::<f64>() else {
            continue;
        };
        let Ok(ssim2) = cols[9].parse::<f64>() else {
            continue;
        };
        out.push(Row {
            image: cols[0].to_string(),
            class: cols[1].to_string(),
            distance,
            effort,
            mode: cols[6].to_string(),
            bytes,
            butteraugli,
            ssim2,
        });
    }
    out
}

fn pair_rows(rows: &[Row]) -> Vec<Pair> {
    let mut by_key: HashMap<(String, String, String, u8), [Option<Row>; 2]> = HashMap::new();
    for r in rows {
        // distance as string key to avoid float issues
        let d_key = format!("{:.2}", r.distance);
        let idx = if r.mode == "fixed" { 0 } else { 1 };
        let entry = by_key
            .entry((r.image.clone(), r.class.clone(), d_key, r.effort))
            .or_insert([None, None]);
        entry[idx] = Some(r.clone());
    }
    let mut pairs = Vec::new();
    for ((image, class, _d_key, effort), [fx, ad]) in by_key {
        let (Some(fx), Some(ad)) = (fx, ad) else {
            continue;
        };
        pairs.push(Pair {
            image,
            class,
            distance: fx.distance,
            effort,
            fixed_bytes: fx.bytes,
            adapt_bytes: ad.bytes,
            fixed_bfly: fx.butteraugli,
            adapt_bfly: ad.butteraugli,
            fixed_ssim2: fx.ssim2,
            adapt_ssim2: ad.ssim2,
        });
    }
    pairs
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        f64::NAN
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn main() {
    let path = parse_in_path();
    let rows = read_rows(&path);
    let pairs = pair_rows(&rows);
    println!(
        "# input: {} ({} rows, {} pairs)",
        path.display(),
        rows.len(),
        pairs.len()
    );

    // Group by (class, distance, effort)
    let mut groups: HashMap<(String, String, u8), Vec<Pair>> = HashMap::new();
    for p in &pairs {
        let d_key = format!("{:.2}", p.distance);
        groups
            .entry((p.class.clone(), d_key, p.effort))
            .or_default()
            .push(p.clone());
    }

    // Sorted output
    let mut keys: Vec<(String, String, u8)> = groups.keys().cloned().collect();
    keys.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)).then(a.1.cmp(&b.1)));

    println!("\n## Per-cell means (class × distance × effort)");
    println!(
        "class\tdistance\teffort\tn\tbytes_d%_mean\tbfly_d%_mean\tssim2_d_mean\tworst_bfly_d%"
    );
    for k in &keys {
        let g = &groups[k];
        let bytes_d: Vec<f64> = g.iter().map(|p| p.bytes_delta_pct()).collect();
        let bfly_d: Vec<f64> = g.iter().map(|p| p.bfly_delta_pct()).collect();
        let ssim2_d: Vec<f64> = g.iter().map(|p| p.ssim2_delta()).collect();
        let worst_bfly = bfly_d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
            k.0,
            k.1,
            k.2,
            g.len(),
            mean(&bytes_d),
            mean(&bfly_d),
            mean(&ssim2_d),
            worst_bfly
        );
    }

    // Aggregate by (class, effort) for default-on decision
    println!("\n## Default-on decision (class × effort means)");
    println!("class\teffort\tn\tbytes_d%_mean\tbfly_d%_mean\tssim2_d_mean\tworst_bfly_d%");
    let mut by_cls_eff: HashMap<(String, u8), Vec<Pair>> = HashMap::new();
    for p in &pairs {
        by_cls_eff
            .entry((p.class.clone(), p.effort))
            .or_default()
            .push(p.clone());
    }
    let mut ce_keys: Vec<(String, u8)> = by_cls_eff.keys().cloned().collect();
    ce_keys.sort();
    let mut all_pass = true;
    let mut per_pass: HashMap<(String, u8), bool> = HashMap::new();
    for k in &ce_keys {
        let g = &by_cls_eff[k];
        let bytes_d: Vec<f64> = g.iter().map(|p| p.bytes_delta_pct()).collect();
        let bfly_d: Vec<f64> = g.iter().map(|p| p.bfly_delta_pct()).collect();
        let ssim2_d: Vec<f64> = g.iter().map(|p| p.ssim2_delta()).collect();
        let worst_bfly = bfly_d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bytes_mean = mean(&bytes_d);
        let bfly_mean = mean(&bfly_d);
        let pass = if k.0 == "photo" {
            bytes_mean <= -1.0 && bfly_mean <= 2.0 && worst_bfly <= 5.0
        } else {
            bytes_mean <= 0.5 && bfly_mean <= 2.0
        };
        per_pass.insert(k.clone(), pass);
        if !pass {
            all_pass = false;
        }
        println!(
            "{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}",
            k.0,
            k.1,
            g.len(),
            bytes_mean,
            bfly_mean,
            mean(&ssim2_d),
            worst_bfly,
            if pass { "PASS" } else { "FAIL" }
        );
    }

    println!(
        "\n## Decision: {}",
        if all_pass {
            "FLIP DEFAULT-ON"
        } else {
            "KEEP OPT-IN (gate failed)"
        }
    );
    if !all_pass {
        println!("Failing cells:");
        for k in &ce_keys {
            if !per_pass[k] {
                println!("  - {} × e{}", k.0, k.1);
            }
        }
    }

    // Top-3 wins and top-3 losses (photos and screenshots separately)
    for class in ["photo", "screenshot"] {
        let mut cps: Vec<&Pair> = pairs.iter().filter(|p| p.class == class).collect();
        if cps.is_empty() {
            continue;
        }

        cps.sort_by(|a, b| {
            a.bytes_delta_pct()
                .partial_cmp(&b.bytes_delta_pct())
                .unwrap()
        });
        println!(
            "\n## {} — top-3 byte wins (most-negative bytes delta)",
            class
        );
        for p in cps.iter().take(3) {
            println!(
                "  {} d={} e={}  bytes {} -> {}  ({:+.2}%)  bfly {:.4} -> {:.4} ({:+.2}%)  ssim2 {:.2} -> {:.2} ({:+.2})",
                p.image,
                p.distance,
                p.effort,
                p.fixed_bytes,
                p.adapt_bytes,
                p.bytes_delta_pct(),
                p.fixed_bfly,
                p.adapt_bfly,
                p.bfly_delta_pct(),
                p.fixed_ssim2,
                p.adapt_ssim2,
                p.ssim2_delta()
            );
        }

        cps.sort_by(|a, b| b.bfly_delta_pct().partial_cmp(&a.bfly_delta_pct()).unwrap());
        println!(
            "\n## {} — top-3 butteraugli regressions (most-positive bfly delta)",
            class
        );
        for p in cps.iter().take(3) {
            println!(
                "  {} d={} e={}  bfly {:.4} -> {:.4} ({:+.2}%)  bytes {} -> {} ({:+.2}%)  ssim2 {:.2} -> {:.2} ({:+.2})",
                p.image,
                p.distance,
                p.effort,
                p.fixed_bfly,
                p.adapt_bfly,
                p.bfly_delta_pct(),
                p.fixed_bytes,
                p.adapt_bytes,
                p.bytes_delta_pct(),
                p.fixed_ssim2,
                p.adapt_ssim2,
                p.ssim2_delta()
            );
        }
    }
}
