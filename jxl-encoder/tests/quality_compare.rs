/// CSV-backed quality comparison: cjxl-rs vs cjxl reference data.
///
/// Embeds reference/cjxl_reference.csv and compares our encoder against
/// cjxl at a configurable effort level. Uses in-process Rust butteraugli
/// + fast-ssim2 — metadata-immune (no PNG color metadata issues).
///
/// Run with: `just quality-compare`
/// Override effort: `CJXL_EFFORT=5 just quality-compare`
/// Filter to one image: `QC_FILTER=1025469 just quality-compare`
/// Filter to one distance: `QC_DIST=2.0 just quality-compare`
/// Both: `QC_FILTER=1025469 QC_DIST=2.0 just quality-compare`
use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use rgb::RGB;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const CSV_DATA: &str = include_str!("../../reference/cjxl_reference.csv");

#[derive(Clone)]
struct CjxlEntry {
    corpus: String,
    image: String,
    distance: f32,
    size_bytes: usize,
    ssimulacra2: f64,
    butteraugli: f64,
}

fn parse_csv(effort: u32) -> Vec<CjxlEntry> {
    let mut entries = Vec::new();
    for line in CSV_DATA.lines() {
        if line.starts_with('#') || line.starts_with("corpus") || line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 9 {
            continue;
        }
        let e: u32 = match fields[5].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if e != effort {
            continue;
        }
        entries.push(CjxlEntry {
            corpus: fields[0].to_string(),
            image: fields[1].to_string(),
            distance: fields[4].parse().unwrap_or(0.0),
            size_bytes: fields[6].parse().unwrap_or(0),
            ssimulacra2: fields[7].parse().unwrap_or(0.0),
            butteraugli: fields[8].parse().unwrap_or(0.0),
        });
    }
    entries
}

struct SourceImage {
    name: String,
    corpus: String,
    path: PathBuf,
}

fn find_source_images(project_root: &std::path::Path) -> Vec<SourceImage> {
    let mut images = Vec::new();

    // frymire (local test image)
    let frymire = project_root.join("jxl_encoder/tests/images/frymire.png");
    if frymire.exists() {
        images.push(SourceImage {
            name: "frymire".into(),
            corpus: "frymire".into(),
            path: frymire,
        });
    }

    // CID22 via codec-corpus
    let corpus = match codec_corpus::Corpus::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("WARNING: codec-corpus init failed: {e}");
            None
        }
    };

    if let Some(ref cc) = corpus {
        for (subdir, corpus_name) in [
            ("CID22/CID22-512/validation", "cid22"),
            ("CID22/CID22-512/training", "cid22-train"),
        ] {
            match cc.get(subdir) {
                Ok(dir) => add_png_dir(&dir, corpus_name, &mut images),
                Err(e) => eprintln!("WARNING: codec-corpus get({subdir}) failed: {e}"),
            }
        }
    }

    images
}

fn add_png_dir(dir: &std::path::Path, corpus: &str, images: &mut Vec<SourceImage>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("png") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy();
        let short = &stem[..stem.len().min(8)];
        images.push(SourceImage {
            name: short.to_string(),
            corpus: corpus.to_string(),
            path,
        });
    }
}

/// Convert linear light value to sRGB u8 using the correct sRGB transfer function.
/// NOT gamma 2.2 — sRGB has a linear segment near black and uses exponent 1/2.4.
fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut jxl_image = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    jxl_image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = jxl_image.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

struct CompareResult {
    image: String,
    distance: f32,
    rs_size: usize,
    rs_bfly: f64,
    rs_ssim2: f64,
    cjxl_size: usize,
    cjxl_bfly: f64,
    cjxl_ssim2: f64,
    strategy_counts: [u32; 19],
}

fn process_image(src: &SourceImage, csv_entries: &[CjxlEntry]) -> Vec<CompareResult> {
    let img = match image::open(&src.path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("  WARNING: {}: failed to open: {e}", src.name);
            return Vec::new();
        }
    };
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();

    // sRGB u8 for SSIM2
    let original_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
    let orig_srgb_img = Img::new(original_srgb, w as usize, h as usize);

    // Linear f32 for encoder + butteraugli
    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            [
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
            ]
        })
        .collect();
    let orig_pixels: Vec<RGB<f32>> = linear_rgb
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let orig_linear_img = Img::new(orig_pixels, w as usize, h as usize);

    let params = ButteraugliParams::default();
    let mut results = Vec::new();

    // Sort by distance for consistent processing
    let mut sorted = csv_entries.to_vec();
    sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

    for entry in &sorted {
        // Encode with our encoder
        let encoder = jxl_encoder::vardct::VarDctEncoder::new(entry.distance);
        let rs_output = match encoder.encode(w as usize, h as usize, &linear_rgb, None) {
            Ok(out) => out,
            Err(e) => {
                eprintln!("  {} d={}: encode failed: {e:?}", src.name, entry.distance);
                continue;
            }
        };
        let rs_size = rs_output.data.len();
        let strategy_counts = rs_output.strategy_counts;

        // Decode our JXL with jxl-oxide in linear RGB
        let (_, _, rs_decoded) = match decode_jxl_linear(&rs_output.data) {
            Some(v) => v,
            None => {
                eprintln!("  {} d={}: decode failed", src.name, entry.distance);
                continue;
            }
        };

        // Butteraugli on linear RGB
        let rs_dec_pixels: Vec<RGB<f32>> = rs_decoded
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let rs_dec_img = Img::new(rs_dec_pixels, w as usize, h as usize);
        let rs_bfly = butteraugli_linear(orig_linear_img.as_ref(), rs_dec_img.as_ref(), &params)
            .unwrap_or_else(|e| {
                panic!(
                    "{} d={}: butteraugli failed: {e:?}",
                    src.name, entry.distance
                )
            })
            .score;

        // SSIM2 on sRGB u8 (must use correct sRGB TF, NOT gamma 2.2)
        let decoded_srgb: Vec<[u8; 3]> = rs_decoded
            .chunks(3)
            .map(|c| {
                [
                    linear_to_srgb_u8(c[0]),
                    linear_to_srgb_u8(c[1]),
                    linear_to_srgb_u8(c[2]),
                ]
            })
            .collect();
        let dec_srgb_img = Img::new(decoded_srgb, w as usize, h as usize);
        let rs_ssim2 =
            fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
                .unwrap_or_else(|e| {
                    panic!("{} d={}: ssim2 failed: {e:?}", src.name, entry.distance)
                });

        results.push(CompareResult {
            image: src.name.clone(),
            distance: entry.distance,
            rs_size,
            rs_bfly,
            rs_ssim2,
            cjxl_size: entry.size_bytes,
            cjxl_bfly: entry.butteraugli,
            cjxl_ssim2: entry.ssimulacra2,
            strategy_counts,
        });
    }

    results
}

/// Format distance for display: 0.25 -> "0.25", 0.5 -> "0.5", 1.0 -> "1.0"
fn fmt_dist(d: f32) -> String {
    let s = format!("{:.2}", d);
    let s = s.trim_end_matches('0');
    if s.ends_with('.') {
        format!("{s}0")
    } else {
        s.to_string()
    }
}

#[test]
#[ignore] // Run with: just quality-compare
fn quality_compare() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let project_root = std::path::Path::new(manifest_dir).parent().unwrap();

    let effort: u32 = std::env::var("CJXL_EFFORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    let entries = parse_csv(effort);
    if entries.is_empty() {
        eprintln!(
            "No CSV entries for effort {effort}. Run `bash scripts/generate_cjxl_reference.sh`."
        );
        return;
    }

    let sources = find_source_images(project_root);
    eprintln!("Found {} source images", sources.len());

    // Optional filters: QC_FILTER (image name substring), QC_DIST (distance)
    let filter_name = std::env::var("QC_FILTER").ok();
    let filter_dist: Option<f32> = std::env::var("QC_DIST").ok().and_then(|s| s.parse().ok());

    if let Some(ref f) = filter_name {
        eprintln!("Filtering images to: {f}");
    }
    if let Some(d) = filter_dist {
        eprintln!("Filtering distances to: {d}");
    }

    // Match sources to CSV entries
    let matched: Vec<(&SourceImage, Vec<CjxlEntry>)> = sources
        .iter()
        .filter(|src| {
            filter_name
                .as_ref()
                .is_none_or(|f| src.name.contains(f.as_str()))
        })
        .filter_map(|src| {
            let matching: Vec<CjxlEntry> = entries
                .iter()
                .filter(|e| e.corpus == src.corpus && e.image == src.name)
                .filter(|e| filter_dist.is_none_or(|d| (e.distance - d).abs() < 0.01))
                .cloned()
                .collect();
            if matching.is_empty() {
                None
            } else {
                Some((src, matching))
            }
        })
        .collect();

    let n_images = matched.len();
    let n_points: usize = matched.iter().map(|(_, e)| e.len()).sum();
    eprintln!("Matched {n_images} images ({n_points} data points) at effort {effort}");

    if matched.is_empty() {
        eprintln!(
            "No matching images. Ensure CSV has entries for loaded corpora at effort {effort}."
        );
        eprintln!(
            "Loaded corpora: frymire (local), cid22 (codec-corpus), cid22-train (codec-corpus)"
        );
        eprintln!("Run `bash scripts/generate_cjxl_reference.sh` to populate CSV.");
        return;
    }

    // Parallel encode+measure: work-stealing with bounded concurrency
    let results_mu: Mutex<Vec<CompareResult>> = Mutex::new(Vec::new());
    let next_idx = AtomicUsize::new(0);
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let done_count = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..n_threads {
            s.spawn(|| {
                loop {
                    let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= matched.len() {
                        break;
                    }
                    let (src, csv_entries) = &matched[idx];
                    let img_results = process_image(src, csv_entries);
                    let done = done_count.fetch_add(1, Ordering::Relaxed) + 1;
                    eprintln!(
                        "  [{done}/{n_images}] {} done ({} distances)",
                        src.name,
                        img_results.len()
                    );
                    results_mu.lock().unwrap().extend(img_results);
                }
            });
        }
    });

    let mut sorted = results_mu.into_inner().unwrap();
    sorted.sort_by(|a, b| {
        a.image
            .cmp(&b.image)
            .then(a.distance.partial_cmp(&b.distance).unwrap())
    });

    // Collect unique distances
    let mut distances: Vec<f32> = sorted.iter().map(|r| r.distance).collect();
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distances.dedup();

    // Print header
    eprintln!("\n=== Quality Comparison: cjxl-rs vs cjxl e{effort} ===");
    eprintln!("(Rust butteraugli + ssim2, metadata-immune)\n");
    eprintln!(
        "{:<10} {:>5}  {:>8} {:>8} {:>6}  {:>7} {:>7} {:>7}  {:>6} {:>6} {:>6}",
        "Image",
        "Dist",
        "RS_size",
        "C_size",
        "S%",
        "RS_bfly",
        "C_bfly",
        "B%",
        "RS_ss2",
        "C_ss2",
        "Δss2"
    );
    eprintln!("{}", "-".repeat(99));

    // Print per-image rows
    for r in &sorted {
        let size_pct = (r.rs_size as f64 / r.cjxl_size as f64 - 1.0) * 100.0;
        let bfly_pct = (r.rs_bfly / r.cjxl_bfly - 1.0) * 100.0;
        let ss2_delta = r.rs_ssim2 - r.cjxl_ssim2;
        eprintln!(
            "{:<10} {:>5}  {:>8} {:>8} {:>+5.1}%  {:>7.3} {:>7.3} {:>+6.1}%  {:>6.2} {:>6.2} {:>+5.2}",
            r.image,
            fmt_dist(r.distance),
            r.rs_size,
            r.cjxl_size,
            size_pct,
            r.rs_bfly,
            r.cjxl_bfly,
            bfly_pct,
            r.rs_ssim2,
            r.cjxl_ssim2,
            ss2_delta
        );
    }

    // Print strategy histogram when filtering (detailed single-image mode)
    if filter_name.is_some() || filter_dist.is_some() {
        const NAMES: [&str; 19] = [
            "DCT8", "DCT16x8", "DCT8x16", "DCT16x16", "DCT32x32", "DCT4x8", "DCT8x4", "DCT4x4",
            "IDENTITY", "DCT2X2", "DCT32x16", "DCT16x32", "AFV0", "AFV1", "AFV2", "AFV3",
            "DCT64x64", "DCT64x32", "DCT32x64",
        ];
        for r in &sorted {
            let total: u32 = r.strategy_counts.iter().sum();
            eprintln!(
                "\n  Strategy histogram for {} d={} ({total} transforms):",
                r.image,
                fmt_dist(r.distance)
            );
            for (i, &count) in r.strategy_counts.iter().enumerate() {
                if count > 0 {
                    let pct = 100.0 * count as f64 / total as f64;
                    eprintln!("    {:10}: {:6} ({:5.1}%)", NAMES[i], count, pct);
                }
            }
        }
    }

    // Per-distance averages
    eprintln!("{}", "-".repeat(99));
    for &dist in &distances {
        let rows: Vec<&CompareResult> = sorted
            .iter()
            .filter(|r| (r.distance - dist).abs() < 0.001)
            .collect();
        let n = rows.len() as f64;
        if n == 0.0 {
            continue;
        }
        let avg_rs_size = rows.iter().map(|r| r.rs_size).sum::<usize>() as f64 / n;
        let avg_c_size = rows.iter().map(|r| r.cjxl_size).sum::<usize>() as f64 / n;
        let avg_rs_bfly = rows.iter().map(|r| r.rs_bfly).sum::<f64>() / n;
        let avg_c_bfly = rows.iter().map(|r| r.cjxl_bfly).sum::<f64>() / n;
        let avg_rs_ssim2 = rows.iter().map(|r| r.rs_ssim2).sum::<f64>() / n;
        let avg_c_ssim2 = rows.iter().map(|r| r.cjxl_ssim2).sum::<f64>() / n;
        let avg_size_pct = (avg_rs_size / avg_c_size - 1.0) * 100.0;
        let avg_bfly_pct = (avg_rs_bfly / avg_c_bfly - 1.0) * 100.0;
        let avg_ss2_delta = avg_rs_ssim2 - avg_c_ssim2;
        eprintln!(
            "{:<10} {:>5}  {:>8.0} {:>8.0} {:>+5.1}%  {:>7.3} {:>7.3} {:>+6.1}%  {:>6.2} {:>6.2} {:>+5.2}",
            format!("Avg({})", rows.len()),
            fmt_dist(dist),
            avg_rs_size,
            avg_c_size,
            avg_size_pct,
            avg_rs_bfly,
            avg_c_bfly,
            avg_bfly_pct,
            avg_rs_ssim2,
            avg_c_ssim2,
            avg_ss2_delta
        );
    }

    // Grand average
    let n = sorted.len() as f64;
    if n > 0.0 {
        let grand_rs_size = sorted.iter().map(|r| r.rs_size).sum::<usize>() as f64 / n;
        let grand_c_size = sorted.iter().map(|r| r.cjxl_size).sum::<usize>() as f64 / n;
        let grand_rs_bfly = sorted.iter().map(|r| r.rs_bfly).sum::<f64>() / n;
        let grand_c_bfly = sorted.iter().map(|r| r.cjxl_bfly).sum::<f64>() / n;
        let grand_rs_ssim2 = sorted.iter().map(|r| r.rs_ssim2).sum::<f64>() / n;
        let grand_c_ssim2 = sorted.iter().map(|r| r.cjxl_ssim2).sum::<f64>() / n;
        let grand_size_pct = (grand_rs_size / grand_c_size - 1.0) * 100.0;
        let grand_bfly_pct = (grand_rs_bfly / grand_c_bfly - 1.0) * 100.0;
        let grand_ss2_delta = grand_rs_ssim2 - grand_c_ssim2;
        eprintln!(
            "\n{:<10} {:>5}  {:>8.0} {:>8.0} {:>+5.1}%  {:>7.3} {:>7.3} {:>+6.1}%  {:>6.2} {:>6.2} {:>+5.2}",
            format!("GRAND({})", sorted.len()),
            "",
            grand_rs_size,
            grand_c_size,
            grand_size_pct,
            grand_rs_bfly,
            grand_c_bfly,
            grand_bfly_pct,
            grand_rs_ssim2,
            grand_c_ssim2,
            grand_ss2_delta
        );
    }
    eprintln!();
}
