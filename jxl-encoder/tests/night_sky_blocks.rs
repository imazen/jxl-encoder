//! Per-8x8-block error comparison: our encoder vs cjxl at d=4.0 on night sky.
//! Run with: cargo test -p jxl-encoder --test night_sky_blocks --release -- --ignored --nocapture

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use rgb::RGB;
use std::io::Cursor;
use std::path::Path;

fn out_dir() -> String {
    jxl_encoder::test_helpers::output_dir("night_sky_coeff")
        .to_string_lossy()
        .into_owned()
}

/// Convert sRGB u8 (normalized to 0..1) to linear light using correct sRGB EOTF.
fn srgb_to_linear_val(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear light value to sRGB u8.
fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

/// Decode JXL bytes to linear f32 via jxl-oxide (requesting sRGB linear output).
fn decode_jxl_linear(bytes: &[u8]) -> (usize, usize, Vec<f32>) {
    let mut image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(bytes))
        .expect("jxl-oxide parse failed");
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).expect("jxl-oxide render failed");
    let fb = render.image_all_channels();
    (fb.width(), fb.height(), fb.buf().to_vec())
}

/// Decode JXL bytes to sRGB u8 via djxl.
fn decode_djxl_srgb(jxl_path: &str) -> Option<(usize, usize, Vec<u8>)> {
    let png_path = format!("{}.decoded.png", jxl_path);
    let ok = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .args([jxl_path, &png_path])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let img = image::open(&png_path).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let pixels: Vec<u8> = rgb.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect();
    Some((w, h, pixels))
}

/// Compute per-8x8-block MSE between two sRGB u8 images.
/// Returns a 2D grid of MSE values, one per 8x8 block.
fn per_block_mse(
    orig: &[u8],
    decoded: &[u8],
    width: usize,
    height: usize,
) -> (Vec<f64>, usize, usize) {
    let bw = width / 8;
    let bh = height / 8;
    let mut block_mses = vec![0.0f64; bw * bh];

    for by in 0..bh {
        for bx in 0..bw {
            let mut sum = 0.0f64;
            for dy in 0..8 {
                for dx in 0..8 {
                    let y = by * 8 + dy;
                    let x = bx * 8 + dx;
                    let idx = (y * width + x) * 3;
                    for c in 0..3 {
                        let diff = orig[idx + c] as f64 - decoded[idx + c] as f64;
                        sum += diff * diff;
                    }
                }
            }
            block_mses[by * bw + bx] = sum / (8.0 * 8.0 * 3.0);
        }
    }
    (block_mses, bw, bh)
}

/// Compute per-8x8-block max absolute error.
fn per_block_max_error(
    orig: &[u8],
    decoded: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, usize, usize) {
    let bw = width / 8;
    let bh = height / 8;
    let mut block_maxerr = vec![0u8; bw * bh];

    for by in 0..bh {
        for bx in 0..bw {
            let mut max_e = 0u8;
            for dy in 0..8 {
                for dx in 0..8 {
                    let y = by * 8 + dy;
                    let x = bx * 8 + dx;
                    let idx = (y * width + x) * 3;
                    for c in 0..3 {
                        let diff =
                            (orig[idx + c] as i16 - decoded[idx + c] as i16).unsigned_abs() as u8;
                        max_e = max_e.max(diff);
                    }
                }
            }
            block_maxerr[by * bw + bx] = max_e;
        }
    }
    (block_maxerr, bw, bh)
}

/// Compute per-8x8-block high-frequency energy (sum of |pixel - block_mean| for each channel).
/// High values indicate ringing/texture, low values indicate smooth blocks.
fn per_block_hf_energy(pixels: &[u8], width: usize, height: usize) -> (Vec<f64>, usize, usize) {
    let bw = width / 8;
    let bh = height / 8;
    let mut block_hf = vec![0.0f64; bw * bh];

    for by in 0..bh {
        for bx in 0..bw {
            // Compute block mean per channel
            let mut mean = [0.0f64; 3];
            for dy in 0..8 {
                for dx in 0..8 {
                    let y = by * 8 + dy;
                    let x = bx * 8 + dx;
                    let idx = (y * width + x) * 3;
                    for c in 0..3 {
                        mean[c] += pixels[idx + c] as f64;
                    }
                }
            }
            for m in &mut mean {
                *m /= 64.0;
            }

            // Sum of |pixel - mean| (proxy for HF energy)
            let mut hf = 0.0f64;
            for dy in 0..8 {
                for dx in 0..8 {
                    let y = by * 8 + dy;
                    let x = bx * 8 + dx;
                    let idx = (y * width + x) * 3;
                    for c in 0..3 {
                        hf += (pixels[idx + c] as f64 - mean[c]).abs();
                    }
                }
            }
            block_hf[by * bw + bx] = hf / (64.0 * 3.0);
        }
    }
    (block_hf, bw, bh)
}

/// Compute per-pixel error difference map: |our_error| - |cjxl_error| per pixel.
/// Positive = we're worse, negative = cjxl is worse.
fn error_diff_image(orig: &[u8], ours: &[u8], cjxl: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut img = vec![128u8; width * height * 3]; // gray = equal
    for i in 0..width * height {
        for c in 0..3 {
            let our_err = (orig[i * 3 + c] as f64 - ours[i * 3 + c] as f64).abs();
            let cjxl_err = (orig[i * 3 + c] as f64 - cjxl[i * 3 + c] as f64).abs();
            // diff > 0 means we're worse
            let diff = our_err - cjxl_err;
            // Map to 0-255: 128 = equal, >128 = we're worse (red), <128 = cjxl worse (blue)
            let val = (128.0 + diff * 8.0).clamp(0.0, 255.0) as u8;
            img[i * 3 + c] = val;
        }
    }
    img
}

#[test]
#[ignore]
fn test_night_sky_block_comparison() {
    use jxl_encoder::api::{LossyConfig, PixelLayout};

    let img_path = &format!(
        "{}/gb82/night-lossless.png",
        jxl_encoder::test_helpers::corpus_dir().display()
    );
    if !Path::new(img_path).exists() {
        eprintln!("SKIP: {} not found", img_path);
        return;
    }

    std::fs::create_dir_all(out_dir()).ok();

    let distance = 4.0f32;

    // Load image
    let img = image::open(img_path).expect("open failed");
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let rgb = img.to_rgb8();
    let original_srgb: Vec<u8> = rgb.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect();

    eprintln!("=== Night Sky Block Comparison (d={}) ===", distance);
    eprintln!("Image: {} ({}x{})", img_path, w, h);

    // --- Encode with our encoder ---
    let our_bytes = LossyConfig::new(distance)
        .encode(rgb.as_raw(), w as u32, h as u32, PixelLayout::Rgb8)
        .expect("our encode failed");
    let our_jxl_path = format!("{}/ours_d{}.jxl", out_dir(), distance);
    std::fs::write(&our_jxl_path, &our_bytes).unwrap();

    // Decode our output
    let (_, _, our_linear) = decode_jxl_linear(&our_bytes);
    let our_srgb: Vec<u8> = our_linear
        .chunks(3)
        .flat_map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();

    // --- Encode with cjxl ---
    let cjxl_path = format!("{}/cjxl_d{}.jxl", out_dir(), distance);
    let cjxl_ok = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
        .args([img_path, &cjxl_path, "-d", &distance.to_string(), "-e", "7"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !cjxl_ok {
        eprintln!("SKIP: cjxl encode failed");
        return;
    }

    // Decode cjxl output
    let cjxl_srgb = match decode_djxl_srgb(&cjxl_path) {
        Some((_, _, pixels)) => pixels,
        None => {
            eprintln!("SKIP: djxl decode failed");
            return;
        }
    };

    let cjxl_size = std::fs::metadata(&cjxl_path).map(|m| m.len()).unwrap_or(0);

    // --- Overall metrics ---
    // Butteraugli
    let orig_linear: Vec<RGB<f32>> = original_srgb
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear(c[0]),
                srgb_to_linear(c[1]),
                srgb_to_linear(c[2]),
            )
        })
        .collect();
    let orig_linear_img = Img::new(orig_linear, w, h);

    let our_linear_rgb: Vec<RGB<f32>> = our_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let our_bfly = butteraugli_linear(
        orig_linear_img.as_ref(),
        Img::new(our_linear_rgb, w, h).as_ref(),
        &ButteraugliParams::default(),
    )
    .expect("butteraugli failed")
    .score;

    // For cjxl, convert sRGB u8 to linear for butteraugli
    let cjxl_linear: Vec<RGB<f32>> = cjxl_srgb
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear(c[0]),
                srgb_to_linear(c[1]),
                srgb_to_linear(c[2]),
            )
        })
        .collect();
    let cjxl_bfly = butteraugli_linear(
        orig_linear_img.as_ref(),
        Img::new(cjxl_linear, w, h).as_ref(),
        &ButteraugliParams::default(),
    )
    .expect("butteraugli failed")
    .score;

    // SSIM2
    let orig_srgb_img: Vec<[u8; 3]> = original_srgb
        .chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let our_srgb_img: Vec<[u8; 3]> = our_srgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let cjxl_srgb_img: Vec<[u8; 3]> = cjxl_srgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

    let our_ss2 = fast_ssim2::compute_ssimulacra2(
        Img::new(orig_srgb_img.clone(), w, h).as_ref(),
        Img::new(our_srgb_img.clone(), w, h).as_ref(),
    )
    .unwrap_or(0.0);
    let cjxl_ss2 = fast_ssim2::compute_ssimulacra2(
        Img::new(orig_srgb_img, w, h).as_ref(),
        Img::new(cjxl_srgb_img, w, h).as_ref(),
    )
    .unwrap_or(0.0);

    eprintln!("\n--- Overall Metrics ---");
    eprintln!(
        "Ours:  {} bytes  bfly={:.3}  ss2={:.2}",
        our_bytes.len(),
        our_bfly,
        our_ss2
    );
    eprintln!(
        "cjxl:  {} bytes  bfly={:.3}  ss2={:.2}",
        cjxl_size, cjxl_bfly, cjxl_ss2
    );
    eprintln!(
        "Delta: size {:.1}%  bfly {:.1}%  ss2 {:.2}",
        (our_bytes.len() as f64 / cjxl_size as f64 - 1.0) * 100.0,
        (our_bfly / cjxl_bfly - 1.0) * 100.0,
        our_ss2 - cjxl_ss2
    );

    // --- Per-block analysis ---
    let bw = w / 8;
    let bh = h / 8;

    // Block MSE
    let (our_mse, _, _) = per_block_mse(&original_srgb, &our_srgb, w, h);
    let (cjxl_mse, _, _) = per_block_mse(&original_srgb, &cjxl_srgb, w, h);

    // Block max error
    let (our_maxerr, _, _) = per_block_max_error(&original_srgb, &our_srgb, w, h);
    let (cjxl_maxerr, _, _) = per_block_max_error(&original_srgb, &cjxl_srgb, w, h);

    // Block HF energy (in decoded images — high = ringing/texture)
    let (orig_hf, _, _) = per_block_hf_energy(&original_srgb, w, h);
    let (our_hf, _, _) = per_block_hf_energy(&our_srgb, w, h);
    let (cjxl_hf, _, _) = per_block_hf_energy(&cjxl_srgb, w, h);

    // Statistics
    let n = bw * bh;
    let our_mse_mean: f64 = our_mse.iter().sum::<f64>() / n as f64;
    let cjxl_mse_mean: f64 = cjxl_mse.iter().sum::<f64>() / n as f64;
    let our_maxerr_mean: f64 = our_maxerr.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let cjxl_maxerr_mean: f64 = cjxl_maxerr.iter().map(|&v| v as f64).sum::<f64>() / n as f64;

    eprintln!("\n--- Per-Block Statistics ({} blocks) ---", n);
    eprintln!(
        "Block MSE:     ours={:.2}  cjxl={:.2}  (ratio={:.3})",
        our_mse_mean,
        cjxl_mse_mean,
        our_mse_mean / cjxl_mse_mean
    );
    eprintln!(
        "Block MaxErr:  ours={:.1}  cjxl={:.1}",
        our_maxerr_mean, cjxl_maxerr_mean
    );

    // Find blocks where we're significantly worse
    let mut worse_blocks = 0usize;
    let mut much_worse_blocks = 0usize;
    let mut better_blocks = 0usize;
    for i in 0..n {
        if our_mse[i] > cjxl_mse[i] * 1.5 {
            worse_blocks += 1;
        }
        if our_mse[i] > cjxl_mse[i] * 3.0 {
            much_worse_blocks += 1;
        }
        if cjxl_mse[i] > our_mse[i] * 1.5 {
            better_blocks += 1;
        }
    }
    eprintln!(
        "Blocks where ours >1.5x cjxl MSE: {} ({:.1}%)",
        worse_blocks,
        worse_blocks as f64 / n as f64 * 100.0
    );
    eprintln!(
        "Blocks where ours >3x cjxl MSE:   {} ({:.1}%)",
        much_worse_blocks,
        much_worse_blocks as f64 / n as f64 * 100.0
    );
    eprintln!(
        "Blocks where cjxl >1.5x ours MSE: {} ({:.1}%)",
        better_blocks,
        better_blocks as f64 / n as f64 * 100.0
    );

    // HF energy analysis — blocks where original is smooth but decoded has ringing
    eprintln!("\n--- Ringing Analysis (HF energy in smooth blocks) ---");
    let mut our_ringing_count = 0usize;
    let mut cjxl_ringing_count = 0usize;
    let mut our_ringing_sum = 0.0f64;
    let mut cjxl_ringing_sum = 0.0f64;
    let smooth_threshold = 3.0; // original block HF < this = smooth
    let ringing_threshold = 2.0; // decoded HF exceeds original by this = ringing
    for i in 0..n {
        if orig_hf[i] < smooth_threshold {
            let our_excess = our_hf[i] - orig_hf[i];
            let cjxl_excess = cjxl_hf[i] - orig_hf[i];
            if our_excess > ringing_threshold {
                our_ringing_count += 1;
                our_ringing_sum += our_excess;
            }
            if cjxl_excess > ringing_threshold {
                cjxl_ringing_count += 1;
                cjxl_ringing_sum += cjxl_excess;
            }
        }
    }
    let smooth_count = orig_hf.iter().filter(|&&v| v < smooth_threshold).count();
    eprintln!(
        "Smooth blocks (orig HF < {}): {}",
        smooth_threshold, smooth_count
    );
    eprintln!(
        "Our ringing:  {} blocks ({:.1}% of smooth), avg excess={:.2}",
        our_ringing_count,
        our_ringing_count as f64 / smooth_count.max(1) as f64 * 100.0,
        if our_ringing_count > 0 {
            our_ringing_sum / our_ringing_count as f64
        } else {
            0.0
        }
    );
    eprintln!(
        "cjxl ringing: {} blocks ({:.1}% of smooth), avg excess={:.2}",
        cjxl_ringing_count,
        cjxl_ringing_count as f64 / smooth_count.max(1) as f64 * 100.0,
        if cjxl_ringing_count > 0 {
            cjxl_ringing_sum / cjxl_ringing_count as f64
        } else {
            0.0
        }
    );

    // --- Worst blocks detail ---
    eprintln!("\n--- Top 20 Worst Blocks (ours vs cjxl MSE) ---");
    let mut block_diffs: Vec<(usize, usize, f64, f64, f64)> = (0..n)
        .map(|i| {
            let bx = i % bw;
            let by = i / bw;
            (bx, by, our_mse[i], cjxl_mse[i], our_mse[i] - cjxl_mse[i])
        })
        .collect();
    block_diffs.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

    eprintln!(
        "{:>5} {:>5}  {:>10} {:>10} {:>10}  {:>8} {:>8}",
        "bx", "by", "our_mse", "cjxl_mse", "diff", "orig_hf", "our_hf"
    );
    for &(bx, by, our, cjxl, diff) in block_diffs.iter().take(20) {
        let idx = by * bw + bx;
        eprintln!(
            "{:>5} {:>5}  {:>10.2} {:>10.2} {:>10.2}  {:>8.2} {:>8.2}",
            bx, by, our, cjxl, diff, orig_hf[idx], our_hf[idx]
        );
    }

    // --- Save decoded images for visual comparison ---
    let our_png_path = format!("{}/ours_decoded.png", out_dir());
    let our_img = image::RgbImage::from_raw(w as u32, h as u32, our_srgb.clone()).unwrap();
    our_img.save(&our_png_path).unwrap();

    // Save error diff map
    let diff_map = error_diff_image(&original_srgb, &our_srgb, &cjxl_srgb, w, h);
    let diff_img = image::RgbImage::from_raw(w as u32, h as u32, diff_map).unwrap();
    let diff_path = format!("{}/error_diff_map.png", out_dir());
    diff_img.save(&diff_path).unwrap();

    // Save amplified error images (|original - decoded| * 8)
    let our_err_img: Vec<u8> = original_srgb
        .iter()
        .zip(our_srgb.iter())
        .map(|(&o, &d)| ((o as i16 - d as i16).unsigned_abs() * 8).min(255) as u8)
        .collect();
    let our_err_png = image::RgbImage::from_raw(w as u32, h as u32, our_err_img).unwrap();
    our_err_png
        .save(format!("{}/ours_error_8x.png", out_dir()))
        .unwrap();

    let cjxl_err_img: Vec<u8> = original_srgb
        .iter()
        .zip(cjxl_srgb.iter())
        .map(|(&o, &d)| ((o as i16 - d as i16).unsigned_abs() * 8).min(255) as u8)
        .collect();
    let cjxl_err_png = image::RgbImage::from_raw(w as u32, h as u32, cjxl_err_img).unwrap();
    cjxl_err_png
        .save(format!("{}/cjxl_error_8x.png", out_dir()))
        .unwrap();

    // Per-block MSE heatmap
    let mse_max = our_mse
        .iter()
        .chain(cjxl_mse.iter())
        .copied()
        .fold(0.0f64, f64::max);
    save_block_heatmap(
        &our_mse,
        bw,
        bh,
        mse_max,
        &format!("{}/ours_block_mse.png", out_dir()),
    );
    save_block_heatmap(
        &cjxl_mse,
        bw,
        bh,
        mse_max,
        &format!("{}/cjxl_block_mse.png", out_dir()),
    );

    // Diff heatmap: our MSE - cjxl MSE
    let mse_diff: Vec<f64> = our_mse
        .iter()
        .zip(cjxl_mse.iter())
        .map(|(o, c)| o - c)
        .collect();
    save_block_diff_heatmap(
        &mse_diff,
        bw,
        bh,
        &format!("{}/block_mse_diff.png", out_dir()),
    );

    // Montage: original | ours | cjxl | error diff
    let cjxl_decoded_path = format!("{}/cjxl_d{}.jxl.decoded.png", out_dir(), distance);
    let orig_copy_path = format!("{}/original.png", out_dir());
    img.save(&orig_copy_path).unwrap();

    let _ = std::process::Command::new("montage")
        .args([
            &orig_copy_path,
            &our_png_path,
            &cjxl_decoded_path,
            &diff_path,
            "-tile",
            "2x2",
            "-geometry",
            "+2+2",
            "-label",
            "",
            &format!("{}/montage.png", out_dir()),
        ])
        .status();

    let _ = std::process::Command::new("montage")
        .args([
            &format!("{}/ours_error_8x.png", out_dir()),
            &format!("{}/cjxl_error_8x.png", out_dir()),
            "-tile",
            "2x1",
            "-geometry",
            "+2+2",
            &format!("{}/error_montage.png", out_dir()),
        ])
        .status();

    eprintln!("\n--- Output Files ---");
    eprintln!("Decoded:       {}/ours_decoded.png", out_dir());
    eprintln!("Error diff:    {}/error_diff_map.png", out_dir());
    eprintln!(
        "Error 8x:      {}/ours_error_8x.png, cjxl_error_8x.png",
        out_dir()
    );
    eprintln!(
        "Block MSE:     {}/ours_block_mse.png, cjxl_block_mse.png",
        out_dir()
    );
    eprintln!("MSE diff:      {}/block_mse_diff.png", out_dir());
    eprintln!("Montage:       {}/montage.png", out_dir());
    eprintln!("Error montage: {}/error_montage.png", out_dir());

    // --- Strategy analysis (using internal encoder) ---
    let linear_rgb: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            [
                srgb_to_linear_val(p[0] as f32 / 255.0),
                srgb_to_linear_val(p[1] as f32 / 255.0),
                srgb_to_linear_val(p[2] as f32 / 255.0),
            ]
        })
        .collect();

    let encoder = jxl_encoder::vardct::VarDctEncoder::new(distance);
    let output = encoder
        .encode(w, h, &linear_rgb, None)
        .expect("internal encode failed");

    let strategy_names = [
        "DCT8", "HORNETS", "DCT2x2", "DCT4x4", "DCT16x16", "DCT32x32", "DCT16x8", "DCT8x16",
        "DCT32x16", "DCT16x32", "DCT4x8", "DCT8x4", "AFV0", "AFV1", "AFV2", "AFV3", "DCT64x64",
        "DCT64x32", "DCT32x64",
    ];

    eprintln!("\n--- AC Strategy Distribution ---");
    let total_transforms: u32 = output.strategy_counts.iter().sum();
    for (i, &count) in output.strategy_counts.iter().enumerate() {
        if count > 0 {
            eprintln!(
                "  {:>12}: {:>5} ({:.1}%)",
                strategy_names[i],
                count,
                count as f64 / total_transforms as f64 * 100.0
            );
        }
    }

    // --- Per-region coefficient analysis ---
    // Compare dark sky (top-left, low mean brightness) vs bright storefront (bottom-right)
    eprintln!("\n--- Regional Error Analysis ---");
    let regions = [
        ("dark_sky (0..36, 0..36)", 0usize, 0usize, 36usize, 36usize),
        ("mid_dark (0..36, 36..72)", 0, 36, 36, 72),
        ("storefront (36..72, 36..72)", 36, 36, 72, 72),
        ("bottom_left (36..72, 0..36)", 36, 0, 72, 36),
    ];

    for (name, by_start, bx_start, by_end, bx_end) in &regions {
        let mut our_region_mse = 0.0f64;
        let mut cjxl_region_mse = 0.0f64;
        let mut our_max = 0u8;
        let mut cjxl_max = 0u8;
        let mut block_count = 0usize;
        let mut orig_mean_brightness = 0.0f64;

        for by in *by_start..(*by_end).min(bh) {
            for bx in *bx_start..(*bx_end).min(bw) {
                let idx = by * bw + bx;
                our_region_mse += our_mse[idx];
                cjxl_region_mse += cjxl_mse[idx];
                our_max = our_max.max(our_maxerr[idx]);
                cjxl_max = cjxl_max.max(cjxl_maxerr[idx]);
                block_count += 1;

                // Compute mean brightness
                for dy in 0..8 {
                    for dx in 0..8 {
                        let y = by * 8 + dy;
                        let x = bx * 8 + dx;
                        let pidx = (y * w + x) * 3;
                        orig_mean_brightness += (original_srgb[pidx] as f64
                            + original_srgb[pidx + 1] as f64
                            + original_srgb[pidx + 2] as f64)
                            / 3.0;
                    }
                }
            }
        }
        let npix = (block_count * 64) as f64;
        orig_mean_brightness /= npix;
        our_region_mse /= block_count as f64;
        cjxl_region_mse /= block_count as f64;

        eprintln!(
            "  {} ({} blocks, mean brightness {:.0}):",
            name, block_count, orig_mean_brightness
        );
        eprintln!(
            "    MSE: ours={:.2}, cjxl={:.2} (ratio={:.3})",
            our_region_mse,
            cjxl_region_mse,
            our_region_mse / cjxl_region_mse.max(0.001)
        );
        eprintln!("    MaxErr: ours={}, cjxl={}", our_max, cjxl_max);
    }
}

/// Save a block-level heatmap as an upscaled image.
fn save_block_heatmap(values: &[f64], bw: usize, bh: usize, max_val: f64, path: &str) {
    let scale = 8; // Each block becomes 8x8 pixels
    let iw = bw * scale;
    let ih = bh * scale;
    let mut img = vec![0u8; iw * ih * 3];

    for by in 0..bh {
        for bx in 0..bw {
            let val = values[by * bw + bx];
            let t = (val / max_val).clamp(0.0, 1.0);
            // Hot colormap: black → red → yellow → white
            let (r, g, b) = if t < 0.33 {
                let s = t / 0.33;
                ((s * 255.0) as u8, 0, 0)
            } else if t < 0.66 {
                let s = (t - 0.33) / 0.33;
                (255, (s * 255.0) as u8, 0)
            } else {
                let s = (t - 0.66) / 0.34;
                (255, 255, (s * 255.0) as u8)
            };

            for dy in 0..scale {
                for dx in 0..scale {
                    let px = bx * scale + dx;
                    let py = by * scale + dy;
                    let idx = (py * iw + px) * 3;
                    img[idx] = r;
                    img[idx + 1] = g;
                    img[idx + 2] = b;
                }
            }
        }
    }

    let output = image::RgbImage::from_raw(iw as u32, ih as u32, img).unwrap();
    output.save(path).unwrap();
}

/// Save a block-level difference heatmap (blue = cjxl worse, red = we're worse).
fn save_block_diff_heatmap(values: &[f64], bw: usize, bh: usize, path: &str) {
    let scale = 8;
    let iw = bw * scale;
    let ih = bh * scale;
    let mut img = vec![128u8; iw * ih * 3]; // gray = equal

    // Find symmetric max for color range
    let abs_max = values.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if abs_max < 0.001 {
        let output = image::RgbImage::from_raw(iw as u32, ih as u32, img).unwrap();
        output.save(path).unwrap();
        return;
    }

    for by in 0..bh {
        for bx in 0..bw {
            let val = values[by * bw + bx];
            let t = (val / abs_max).clamp(-1.0, 1.0);
            // Red = we're worse (positive), Blue = cjxl worse (negative), Gray = equal
            let (r, g, b) = if t > 0.0 {
                let s = t;
                (
                    128 + (127.0 * s) as u8,
                    (128.0 * (1.0 - s)) as u8,
                    (128.0 * (1.0 - s)) as u8,
                )
            } else {
                let s = -t;
                (
                    (128.0 * (1.0 - s)) as u8,
                    (128.0 * (1.0 - s)) as u8,
                    128 + (127.0 * s) as u8,
                )
            };

            for dy in 0..scale {
                for dx in 0..scale {
                    let px = bx * scale + dx;
                    let py = by * scale + dy;
                    let idx = (py * iw + px) * 3;
                    img[idx] = r;
                    img[idx + 1] = g;
                    img[idx + 2] = b;
                }
            }
        }
    }

    let output = image::RgbImage::from_raw(iw as u32, ih as u32, img).unwrap();
    output.save(path).unwrap();
}
