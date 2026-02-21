/// CSV-backed lossless compression comparison: cjxl-rs vs cjxl reference data.
///
/// Embeds reference/cjxl_lossless_reference.csv and compares our lossless encoder
/// against cjxl at a configurable effort level. Decodes our output with jxl-rs
/// and verifies pixel-exact accuracy via SHA-256 hash (stored in CSV from djxl decode).
///
/// Run with: `just lossless-compare`
/// Override effort: `CJXL_EFFORT=9 just lossless-compare`
use image::GenericImageView;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const CSV_DATA: &str = include_str!("../../reference/cjxl_lossless_reference.csv");

#[derive(Clone)]
struct CjxlLosslessEntry {
    corpus: String,
    image: String,
    effort: u8,
    size_bytes: usize,
    pixel_sha256: String,
}

fn parse_csv(effort: u8) -> Vec<CjxlLosslessEntry> {
    let mut entries = Vec::new();
    for line in CSV_DATA.lines() {
        if line.starts_with('#') || line.starts_with("corpus") || line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        // corpus,image,width,height,effort,size_bytes,pixel_sha256,enc_wall_s,enc_user_s,enc_sys_s
        if fields.len() < 7 {
            continue;
        }
        let e: u8 = match fields[4].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if e != effort {
            continue;
        }
        entries.push(CjxlLosslessEntry {
            corpus: fields[0].to_string(),
            image: fields[1].to_string(),
            effort: e,
            size_bytes: fields[5].parse().unwrap_or(0),
            pixel_sha256: fields[6].to_string(),
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
    let frymire = project_root.join("jxl_encoder/tests/images/frymire-srgb.png");
    if frymire.exists() {
        images.push(SourceImage {
            name: "frymire-".into(),
            corpus: "frymire".into(),
            path: frymire,
        });
    }

    // Corpora via codec-corpus
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
            ("gb82-sc", "gb82-sc"),
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

/// Decode JXL with jxl-rs, return raw u8 RGB pixel data (alpha discarded to match djxl PNM).
fn decode_jxl_pixels(jxl_bytes: &[u8]) -> Option<Vec<u8>> {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = jxl_bytes;
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return None;
                }
                decoder = fallback;
            }
            Err(_) => return None,
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let rgb_channels = 3;
    let num_extra = basic_info.extra_channels.len();

    // jxl-rs requires extra_channel_format to match num_extra_channels.
    // We request extra channels with None (discard) to satisfy the assertion,
    // but only keep RGB data for hashing (matching djxl PNM P6 output).
    let format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![Some(JxlDataFormat::f32()); num_extra],
    };
    decoder.set_pixel_format(format);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return None;
                }
                decoder = fallback;
            }
            Err(_) => return None,
        }
    };

    let mut output_image =
        Image::<f32>::new((width * rgb_channels, height)).expect("alloc output buffer");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * rgb_channels, height),
            })
            .into_raw(),
    )];

    // Allocate discard buffers for extra channels
    let mut extra_images: Vec<Image<f32>> = (0..num_extra)
        .map(|_| Image::<f32>::new((width, height)).expect("alloc extra buffer"))
        .collect();
    for extra in &mut extra_images {
        buffers.push(JxlOutputBuffer::from_image_rect_mut(
            extra
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (width, height),
                })
                .into_raw(),
        ));
    }

    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return None;
                }
                decoder = fallback;
            }
            Err(_) => return None,
        }
    }

    // Convert f32 RGB to u8 RGB (matching djxl PNM output: round(clamp(v,0,1)*255))
    let mut pixels = Vec::with_capacity(width * height * rgb_channels);
    for y in 0..height {
        let row = output_image.row(y);
        for &v in row {
            pixels.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    Some(pixels)
}

/// SHA-256 hash of raw pixel data, returned as hex string.
fn sha256_hex(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    let mut s = String::with_capacity(64);
    for b in hash.iter() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

struct CompareResult {
    image: String,
    corpus: String,
    rs_size: usize,
    cjxl_size: usize,
    pixel_exact: bool,
}

fn process_image(src: &SourceImage, entry: &CjxlLosslessEntry) -> Option<CompareResult> {
    let img = match image::open(&src.path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("  WARNING: {}: failed to open: {e}", src.name);
            return None;
        }
    };
    let (w, h) = img.dimensions();

    // Determine pixel layout and get raw bytes
    let (pixels, layout) = if img.color().has_alpha() {
        let rgba = img.to_rgba8();
        (rgba.into_raw(), jxl_encoder::PixelLayout::Rgba8)
    } else {
        let rgb = img.to_rgb8();
        (rgb.into_raw(), jxl_encoder::PixelLayout::Rgb8)
    };

    let config = jxl_encoder::LosslessConfig::new().with_effort(entry.effort);
    let rs_output = match config.encode(&pixels, w, h, layout) {
        Ok(out) => out,
        Err(e) => {
            eprintln!(
                "  {} e{}: encode failed: {e:?}",
                src.name, entry.effort
            );
            return None;
        }
    };
    let rs_size = rs_output.len();

    // Decode our JXL and verify pixel accuracy
    let pixel_exact = if entry.pixel_sha256.is_empty() {
        true
    } else {
        match decode_jxl_pixels(&rs_output) {
            Some(decoded_pixels) => {
                let rs_hash = sha256_hex(&decoded_pixels);
                if rs_hash != entry.pixel_sha256 {
                    eprintln!(
                        "  {} PIXEL MISMATCH: rs={} cjxl={}",
                        src.name,
                        &rs_hash[..16],
                        &entry.pixel_sha256[..16]
                    );
                    false
                } else {
                    true
                }
            }
            None => {
                eprintln!("  {} e{}: decode failed", src.name, entry.effort);
                false
            }
        }
    };

    Some(CompareResult {
        image: src.name.clone(),
        corpus: src.corpus.clone(),
        rs_size,
        cjxl_size: entry.size_bytes,
        pixel_exact,
    })
}

#[test]
#[ignore] // Run with: just lossless-compare
fn lossless_compare() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let project_root = std::path::Path::new(manifest_dir).parent().unwrap();

    let effort: u8 = std::env::var("CJXL_EFFORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    let entries = parse_csv(effort);
    if entries.is_empty() {
        eprintln!(
            "No CSV entries for effort {effort}. Run `just generate-lossless-reference`."
        );
        return;
    }

    let sources = find_source_images(project_root);
    eprintln!("Found {} source images", sources.len());

    // Match sources to CSV entries
    let matched: Vec<(&SourceImage, CjxlLosslessEntry)> = sources
        .iter()
        .filter_map(|src| {
            entries
                .iter()
                .find(|e| e.corpus == src.corpus && e.image == src.name)
                .map(|e| (src, e.clone()))
        })
        .collect();

    let n_images = matched.len();
    eprintln!("Matched {n_images} images at effort {effort}");

    if matched.is_empty() {
        eprintln!(
            "No matching images. Ensure CSV has entries for loaded corpora at effort {effort}."
        );
        eprintln!("Loaded corpora: frymire (local), cid22, cid22-train, gb82-sc (codec-corpus)");
        eprintln!("Run `just generate-lossless-reference` to populate CSV.");
        return;
    }

    // Parallel encode+decode+verify: work-stealing with bounded concurrency
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
                    let (src, entry) = &matched[idx];
                    if let Some(result) = process_image(src, entry) {
                        let done = done_count.fetch_add(1, Ordering::Relaxed) + 1;
                        let ok = if result.pixel_exact { "OK" } else { "MISMATCH" };
                        eprintln!(
                            "  [{done}/{n_images}] {} ({}) {ok}",
                            src.name, src.corpus
                        );
                        results_mu.lock().unwrap().push(result);
                    }
                }
            });
        }
    });

    let mut sorted = results_mu.into_inner().unwrap();
    sorted.sort_by(|a, b| a.corpus.cmp(&b.corpus).then(a.image.cmp(&b.image)));

    // Collect unique corpora
    let mut corpora: Vec<String> = sorted.iter().map(|r| r.corpus.clone()).collect();
    corpora.sort();
    corpora.dedup();

    // Count mismatches
    let mismatches: Vec<&CompareResult> = sorted.iter().filter(|r| !r.pixel_exact).collect();

    // Print header
    eprintln!("\n=== Lossless Compression: cjxl-rs vs cjxl e{effort} ===\n");
    eprintln!(
        "{:<10} {:<12} {:>8} {:>8} {:>7}  {:<5}",
        "Image", "Corpus", "RS_size", "C_size", "S%", "Exact"
    );
    eprintln!("{}", "-".repeat(56));

    // Print per-image rows
    for r in &sorted {
        let size_pct = (r.rs_size as f64 / r.cjxl_size as f64 - 1.0) * 100.0;
        let exact = if r.pixel_exact { "yes" } else { "NO" };
        eprintln!(
            "{:<10} {:<12} {:>8} {:>8} {:>+6.1}%  {:<5}",
            r.image, r.corpus, r.rs_size, r.cjxl_size, size_pct, exact
        );
    }

    // Per-corpus averages
    eprintln!("{}", "-".repeat(56));
    for corpus in &corpora {
        let rows: Vec<&CompareResult> = sorted.iter().filter(|r| &r.corpus == corpus).collect();
        let n = rows.len() as f64;
        if n == 0.0 {
            continue;
        }
        let avg_rs = rows.iter().map(|r| r.rs_size).sum::<usize>() as f64 / n;
        let avg_c = rows.iter().map(|r| r.cjxl_size).sum::<usize>() as f64 / n;
        let avg_pct = (avg_rs / avg_c - 1.0) * 100.0;
        let exact_count = rows.iter().filter(|r| r.pixel_exact).count();
        eprintln!(
            "{:<10} {:<12} {:>8.0} {:>8.0} {:>+6.1}%  {}/{}",
            format!("Avg({})", rows.len()),
            corpus,
            avg_rs,
            avg_c,
            avg_pct,
            exact_count,
            rows.len()
        );
    }

    // Grand average
    let n = sorted.len() as f64;
    if n > 0.0 {
        let grand_rs = sorted.iter().map(|r| r.rs_size).sum::<usize>() as f64 / n;
        let grand_c = sorted.iter().map(|r| r.cjxl_size).sum::<usize>() as f64 / n;
        let grand_pct = (grand_rs / grand_c - 1.0) * 100.0;
        let exact_count = sorted.iter().filter(|r| r.pixel_exact).count();
        eprintln!(
            "\n{:<10} {:<12} {:>8.0} {:>8.0} {:>+6.1}%  {}/{}",
            format!("GRAND({})", sorted.len()),
            "",
            grand_rs,
            grand_c,
            grand_pct,
            exact_count,
            sorted.len()
        );
    }
    eprintln!();

    // Fail if any pixel mismatches
    if !mismatches.is_empty() {
        panic!(
            "{} pixel mismatches detected (lossless output not pixel-exact)",
            mismatches.len()
        );
    }
}
