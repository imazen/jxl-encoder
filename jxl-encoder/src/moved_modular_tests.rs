#![allow(unused_imports)]
// Tests relocated from jxl-modular during the crate split: they
// exercise the full encoder (LosslessConfig / decoder round-trips),
// which lives one crate up.

use crate::api::*;
use crate::bit_writer::BitWriter;
use crate::effort::EffortProfile;
use crate::entropy_coding::encode::*;
use crate::headers::color_encoding::*;
use crate::modular::channel::ModularImage;

// extracted from jxl-modular/src/effort.rs

#[test]
#[cfg(feature = "butteraugli-loop")]
fn test_init_mul_seeds_invariants() {
    use crate::vardct::butteraugli_loop::{LIBJXL_INIT_MUL, init_mul_seeds};
    // Index 0 must ALWAYS be the libjxl default so multi-seed can
    // never regress below single-seed worst-case.
    for seeds in [1, 2, 3, 4, 5, 10, 99, 255_u8] {
        let table = init_mul_seeds(seeds);
        assert!(!table.is_empty(), "seeds={seeds}: table empty");
        assert!(
            (table[0] - LIBJXL_INIT_MUL).abs() < f64::EPSILON,
            "seeds={seeds}: index 0 ({}) must equal LIBJXL_INIT_MUL ({LIBJXL_INIT_MUL})",
            table[0]
        );
        // Saturation cap: each seed is unique, no NaN/inf, bounded.
        for (i, &v) in table.iter().enumerate() {
            assert!(v.is_finite(), "seeds={seeds}[{i}]: non-finite {v}");
            assert!(
                (0.0..=1.0).contains(&v),
                "seeds={seeds}[{i}]: {v} outside [0, 1]"
            );
        }
    }
    // `0` defensively bumps to `1` (same single-seed behaviour).
    assert_eq!(init_mul_seeds(0).len(), 1);
    assert_eq!(init_mul_seeds(1).len(), 1);
    assert_eq!(init_mul_seeds(2).len(), 2);
    assert_eq!(init_mul_seeds(3).len(), 3);
    assert_eq!(init_mul_seeds(4).len(), 4);
    // Saturate at table length so requesting more is safe.
    assert_eq!(init_mul_seeds(255).len(), 4);
}

// extracted from jxl-modular/src/effort.rs

#[test]
fn test_butteraugli_iters_extended_tiers() {
    // RFC#45 chunks 1 + 2, renumbered by the 2026-08-29 ladder shift:
    // longer butteraugli search budgets at e11/e12/e13 on a
    // power-of-two ladder (8 → 16 → 32). e9 = libjxl kTortoise max
    // (4 iters); e10 keeps the 4-iter cap — libjxl kGlacier adds no
    // buttloop iterations, and new e10 = kGlacier parity + e9 extras.
    let p9 = EffortProfile::lossy(9, EncoderMode::Reference);
    let p10 = EffortProfile::lossy(10, EncoderMode::Reference);
    let p11 = EffortProfile::lossy(11, EncoderMode::Reference);
    let p12 = EffortProfile::lossy(12, EncoderMode::Reference);
    let p13 = EffortProfile::lossy(13, EncoderMode::Reference);
    assert_eq!(p9.butteraugli_iters, 4, "e9 = libjxl kTortoise default");
    assert_eq!(
        p10.butteraugli_iters, 4,
        "e10 = libjxl kGlacier (no extra iters)"
    );
    assert_eq!(p11.butteraugli_iters, 8, "e11 = 2× e9 budget");
    assert_eq!(p12.butteraugli_iters, 16, "e12 = 4× e9 budget");
    assert_eq!(
        p13.butteraugli_iters, 32,
        "e13 = 8× e9, saturated at MAX_QUANT_LOOP_ITERS = 32"
    );
    // Sanity: stays at saturation cap even if effort overshoots.
    // (The lossy() clamp pins at 13; verify the table never returns
    // anything above the loop's structural cap.)
    assert!(
        p13.butteraugli_iters <= crate::api::MAX_QUANT_LOOP_ITERS,
        "butteraugli_iters must not exceed MAX_QUANT_LOOP_ITERS"
    );
    // The cap itself stays 32 across the ladder shift.
    assert_eq!(
        crate::api::MAX_QUANT_LOOP_ITERS,
        32,
        "ITER_MAX stays 32 across the 2026-08-29 ladder shift"
    );
}

// extracted from jxl-modular/src/heuristics.rs
#[cfg(feature = "butteraugli-loop")]
#[test]
fn issue106_e8_comparison_scratch_exceeds_explicit_cap() {
    let e8 = super::estimate_encode_threaded(2550, 3300, 3, false, false, 8, 4).unwrap();
    let e7 = super::estimate_encode_threaded(2550, 3300, 3, false, false, 7, 4).unwrap();
    assert!(e8.peak_memory_bytes_max > 1_600_000_000);
    assert!(e7.peak_memory_bytes < 1_600_000_000);
    assert!(e8.peak_memory_bytes_max < crate::api::Limits::default_max_memory_bytes(false));
}

// extracted from jxl-modular/src/patches.rs
fn test_ref_frame_value_ranges() {
    crate::skip_without_corpus!();
    let path = crate::test_helpers::corpus_dir().join("gb82-sc/terminal.png");
    let img = image::open(&path).unwrap().to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels = img.as_raw();
    let n = w * h;
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for i in 0..n {
        r[i] = pixels[i * 3] as f32;
        g[i] = pixels[i * 3 + 1] as f32;
        b[i] = pixels[i * 3 + 2] as f32;
    }
    let mut x_out = vec![0.0f32; n];
    let mut y_out = vec![0.0f32; n];
    let mut b_out = vec![0.0f32; n];
    crate::color::xyb::srgb_image_to_xyb(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);

    let result = crate::vardct::patches::find_text_like_patches(
        [&x_out, &y_out, &b_out],
        w,
        h,
        w,
        true,
        None,
    )
    .unwrap();
    let patches_data = crate::vardct::patches::build_patches_data(result).unwrap();

    let ref_w = patches_data.ref_width;
    let ref_h = patches_data.ref_height;
    let ref_n = ref_w * ref_h;
    eprintln!("Reference frame: {ref_w}x{ref_h} = {ref_n} pixels");

    const INV_DC_QUANT_X: f32 = 4096.0;
    const INV_DC_QUANT_Y: f32 = 512.0;
    const INV_DC_QUANT_B: f32 = 256.0;

    // Compute integer channel ranges
    let mut ch_y_min = i32::MAX;
    let mut ch_y_max = i32::MIN;
    let mut ch_x_min = i32::MAX;
    let mut ch_x_max = i32::MIN;
    let mut ch_by_min = i32::MAX;
    let mut ch_by_max = i32::MIN;
    let mut nonzero_y = 0u32;
    let mut nonzero_x = 0u32;
    let mut nonzero_by = 0u32;

    for i in 0..ref_n {
        let y_int = crate::vardct::patches::safe_round_to_i32(
            patches_data.ref_image[1][i] * INV_DC_QUANT_Y,
        );
        let x_int = crate::vardct::patches::safe_round_to_i32(
            patches_data.ref_image[0][i] * INV_DC_QUANT_X,
        );
        let b_int = crate::vardct::patches::safe_round_to_i32(
            patches_data.ref_image[2][i] * INV_DC_QUANT_B,
        );
        let by_int = b_int - y_int;

        ch_y_min = ch_y_min.min(y_int);
        ch_y_max = ch_y_max.max(y_int);
        ch_x_min = ch_x_min.min(x_int);
        ch_x_max = ch_x_max.max(x_int);
        ch_by_min = ch_by_min.min(by_int);
        ch_by_max = ch_by_max.max(by_int);
        if y_int != 0 {
            nonzero_y += 1;
        }
        if x_int != 0 {
            nonzero_x += 1;
        }
        if by_int != 0 {
            nonzero_by += 1;
        }
    }

    eprintln!(
        "Channel Y:  range [{ch_y_min}, {ch_y_max}], {nonzero_y} nonzero ({:.1}%)",
        nonzero_y as f64 / ref_n as f64 * 100.0
    );
    eprintln!(
        "Channel X:  range [{ch_x_min}, {ch_x_max}], {nonzero_x} nonzero ({:.1}%)",
        nonzero_x as f64 / ref_n as f64 * 100.0
    );
    eprintln!(
        "Channel BY: range [{ch_by_min}, {ch_by_max}], {nonzero_by} nonzero ({:.1}%)",
        nonzero_by as f64 / ref_n as f64 * 100.0
    );
}

// extracted from jxl-modular/src/patches.rs
fn test_terminal_patch_coverage() {
    crate::skip_without_corpus!();
    let path = crate::test_helpers::corpus_dir().join("gb82-sc/terminal.png");
    let img = image::open(&path).unwrap().to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels = img.as_raw();
    eprintln!("Loaded terminal.png: {w}x{h}");

    // Convert to planar sRGB f32
    let n = w * h;
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for i in 0..n {
        r[i] = pixels[i * 3] as f32;
        g[i] = pixels[i * 3 + 1] as f32;
        b[i] = pixels[i * 3 + 2] as f32;
    }

    // Convert to XYB
    let mut x_out = vec![0.0f32; n];
    let mut y_out = vec![0.0f32; n];
    let mut b_out = vec![0.0f32; n];
    crate::color::xyb::srgb_image_to_xyb(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);

    // Run detection (eprintln stats from cfg(test) instrumentation)
    let result = crate::vardct::patches::find_text_like_patches(
        [&x_out, &y_out, &b_out],
        w,
        h,
        w,
        true,
        None,
    )
    .unwrap();

    // Print size distribution
    let mut size_dist: std::collections::HashMap<(usize, usize), (usize, usize)> =
        std::collections::HashMap::new();
    for p in &result {
        let entry = size_dist
            .entry((p.patch.xsize, p.patch.ysize))
            .or_insert((0, 0));
        entry.0 += 1; // unique patterns at this size
        entry.1 += p.positions.len(); // total occurrences
    }
    let mut sizes: Vec<_> = size_dist.into_iter().collect();
    sizes.sort_by_key(|&((w, h), _)| std::cmp::Reverse(w * h));
    eprintln!("\nPatch size distribution:");
    for ((pw, ph), (unique, occ)) in &sizes {
        eprintln!("  {pw}x{ph}: {unique} unique, {occ} occurrences");
    }

    // Print top patches by occurrence count
    let mut by_occ: Vec<_> = result.iter().enumerate().collect();
    by_occ.sort_by_key(|(_, p)| std::cmp::Reverse(p.positions.len()));
    eprintln!("\nTop 20 patches by occurrence:");
    for (i, (_, p)) in by_occ.iter().take(20).enumerate() {
        eprintln!(
            "  #{}: {}x{} with {} occurrences",
            i + 1,
            p.patch.xsize,
            p.patch.ysize,
            p.positions.len()
        );
    }

    // Analyze near-miss dedup: find singletons that are close to popular patterns
    // Count singleton dimensions
    let _all_patches = crate::vardct::patches::find_text_like_patches(
        [&x_out, &y_out, &b_out],
        w,
        h,
        w,
        true,
        None,
    )
    .unwrap();
    // Re-run to get raw CCs with their positions (need to access raw data)
    // For now, just analyze the final result's dimension distribution
    eprintln!("\nAnalyzing dedup quality...");

    // Build ALL patches including singletons (re-do dedup manually)
    // We'll work with what we have — check if similar-size patches exist
    // that differ only slightly in quantized values
    let mut all_by_dim: std::collections::HashMap<(usize, usize), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, p) in result.iter().enumerate() {
        all_by_dim
            .entry((p.patch.xsize, p.patch.ysize))
            .or_default()
            .push(i);
    }

    // Check for patches at same dimensions that could be merged with tolerance
    eprintln!("\nPer-dimension grouping (final patches only):");
    for ((pw, ph), indices) in &all_by_dim {
        if indices.len() >= 2 {
            // Compare pairs within same dimension
            let mut max_diff = 0i32;
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    let a = &result[indices[i]].patch;
                    let b_patch = &result[indices[j]].patch;
                    let mut diff = 0i32;
                    for c in 0..3 {
                        for k in 0..a.pixels[c].len() {
                            diff = diff
                                .max((a.pixels[c][k] as i32 - b_patch.pixels[c][k] as i32).abs());
                        }
                    }
                    max_diff = max_diff.max(diff);
                }
            }
            eprintln!(
                "  {pw}x{ph}: {} patterns, max quantized diff between any pair: {max_diff}",
                indices.len()
            );
        }
    }
}

// extracted from jxl-modular/src/modular/encode.rs

#[test]
fn test_ans_roundtrip_gray() {
    use crate::headers::{ColorEncoding, FileHeader};
    use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

    let data: Vec<u8> = vec![
        100, 101, 102, 103, 101, 102, 103, 104, 102, 103, 104, 105, 103, 104, 105, 106,
    ];
    let image = ModularImage::from_gray8(&data, 4, 4).unwrap();

    // Build full JXL bitstream with ANS modular
    let mut writer = BitWriter::new();
    let file_header = FileHeader::new_gray(4, 4);
    file_header.write(&mut writer).unwrap();
    writer.zero_pad_to_byte();

    let frame_options = FrameEncoderOptions {
        use_modular: true,
        effort: 7,
        use_ans: true,
        use_tree_learning: false,
        use_squeeze: false,
        ..Default::default()
    };
    let frame_encoder = FrameEncoder::new(4, 4, frame_options);
    let color_encoding = ColorEncoding::srgb();
    frame_encoder
        .encode_modular(&image, &color_encoding, &mut writer, None)
        .unwrap();

    let bytes = writer.finish_with_padding();
    eprintln!("ANS modular gray 4x4: {} bytes", bytes.len());

    // Decode with jxl-oxide
    let jxl_image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&bytes))
        .unwrap_or_else(|e| panic!("jxl-oxide parse failed: {}", e));

    assert_eq!(jxl_image.width(), 4);
    assert_eq!(jxl_image.height(), 4);

    let render = jxl_image
        .render_frame(0)
        .unwrap_or_else(|e| panic!("jxl-oxide render failed: {}", e));

    let fb = render.image_all_channels();
    let decoded_f32 = fb.buf();
    let decoded: Vec<u8> = decoded_f32
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();

    assert_eq!(
        decoded.len(),
        data.len(),
        "decoded size mismatch: {} vs {}",
        decoded.len(),
        data.len()
    );

    for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(
            orig, dec,
            "pixel {} differs: orig={} decoded={}",
            i, orig, dec
        );
    }
}

// extracted from jxl-modular/src/modular/encode.rs

#[test]
fn test_ans_roundtrip_gray_varied() {
    use crate::headers::{ColorEncoding, FileHeader};
    use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

    let data = vec![0u8, 64, 128, 192, 255, 100, 50, 200];
    let image = ModularImage::from_gray8(&data, 4, 2).unwrap();

    // First write with Huffman to get reference bytes
    {
        let mut writer = BitWriter::new();
        let file_header = FileHeader::new_gray(4, 2);
        file_header.write(&mut writer).unwrap();
        writer.zero_pad_to_byte();

        let frame_options = FrameEncoderOptions {
            use_modular: true,
            effort: 7,
            use_ans: false,
            use_tree_learning: false,
            use_squeeze: false,
            ..Default::default()
        };
        let frame_encoder = FrameEncoder::new(4, 2, frame_options);
        let color_encoding = ColorEncoding::srgb();
        frame_encoder
            .encode_modular(&image, &color_encoding, &mut writer, None)
            .unwrap();
        let huf_bytes = writer.finish_with_padding();
        eprintln!("Huffman modular gray varied 4x2: {} bytes", huf_bytes.len());
        eprintln!("Huffman bytes: {:02x?}", huf_bytes);
    }

    // Now write with ANS
    let mut writer = BitWriter::new();
    let file_header = FileHeader::new_gray(4, 2);
    file_header.write(&mut writer).unwrap();
    writer.zero_pad_to_byte();

    let frame_options = FrameEncoderOptions {
        use_modular: true,
        effort: 7,
        use_ans: true,
        use_tree_learning: false,
        use_squeeze: false,
        ..Default::default()
    };
    let frame_encoder = FrameEncoder::new(4, 2, frame_options);
    let color_encoding = ColorEncoding::srgb();
    frame_encoder
        .encode_modular(&image, &color_encoding, &mut writer, None)
        .unwrap();

    let bytes = writer.finish_with_padding();
    eprintln!("ANS modular gray varied 4x2: {} bytes", bytes.len());
    eprintln!("ANS bytes: {:02x?}", bytes);

    // Save for external debugging
    std::fs::write(std::env::temp_dir().join("ans_modular_varied.jxl"), &bytes).ok();

    let jxl_image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&bytes))
        .unwrap_or_else(|e| panic!("jxl-oxide parse failed: {}", e));

    let render = jxl_image
        .render_frame(0)
        .unwrap_or_else(|e| panic!("jxl-oxide render failed: {}", e));

    let fb = render.image_all_channels();
    let decoded_f32 = fb.buf();
    let decoded: Vec<u8> = decoded_f32
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();

    for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(
            orig, dec,
            "pixel {} differs: orig={} decoded={}",
            i, orig, dec
        );
    }
}

// extracted from jxl-modular/src/modular/encode.rs

#[test]
fn test_ans_roundtrip_rgb_gradient() {
    use crate::headers::{ColorEncoding, FileHeader};
    use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};

    let mut data = vec![0u8; 8 * 8 * 3];
    for y in 0..8 {
        for x in 0..8 {
            let idx = (y * 8 + x) * 3;
            data[idx] = (x * 32) as u8;
            data[idx + 1] = (y * 32) as u8;
            data[idx + 2] = ((x + y) * 16) as u8;
        }
    }
    let image = ModularImage::from_rgb8(&data, 8, 8).unwrap();

    let mut writer = BitWriter::new();
    let file_header = FileHeader::new_rgb(8, 8);
    file_header.write(&mut writer).unwrap();
    writer.zero_pad_to_byte();

    let frame_options = FrameEncoderOptions {
        use_modular: true,
        effort: 7,
        use_ans: true,
        use_tree_learning: false,
        use_squeeze: false,
        ..Default::default()
    };
    let frame_encoder = FrameEncoder::new(8, 8, frame_options);
    let color_encoding = ColorEncoding::srgb();
    frame_encoder
        .encode_modular(&image, &color_encoding, &mut writer, None)
        .unwrap();

    let bytes = writer.finish_with_padding();
    eprintln!("ANS modular RGB gradient 8x8: {} bytes", bytes.len());

    let jxl_image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&bytes))
        .unwrap_or_else(|e| panic!("jxl-oxide parse failed: {}", e));

    let render = jxl_image
        .render_frame(0)
        .unwrap_or_else(|e| panic!("jxl-oxide render failed: {}", e));

    let fb = render.image_all_channels();
    let decoded_f32 = fb.buf();
    let decoded: Vec<u8> = decoded_f32
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();

    assert_eq!(decoded.len(), data.len());
    let mut max_diff = 0i32;
    for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
        let diff = (orig as i32 - dec as i32).abs();
        if diff > max_diff {
            max_diff = diff;
            eprintln!(
                "pixel {} ch {}: orig={} decoded={} diff={}",
                i / 3,
                i % 3,
                orig,
                dec,
                diff
            );
        }
    }
    assert_eq!(max_diff, 0, "lossless roundtrip should have zero diff");
}

// extracted from jxl-modular/src/modular/encode.rs

/// Compare ANS vs Huffman file sizes for a lossless encode.
#[test]
fn test_ans_vs_huffman_size() {
    use crate::{LosslessConfig, PixelLayout};

    // Create a non-trivial 32x32 RGB image
    let mut data = vec![0u8; 32 * 32 * 3];
    for y in 0..32 {
        for x in 0..32 {
            let idx = (y * 32 + x) * 3;
            data[idx] = ((x * 8 + y * 2) % 256) as u8;
            data[idx + 1] = ((y * 8 + x * 3) % 256) as u8;
            data[idx + 2] = (((x + y) * 5) % 256) as u8;
        }
    }

    // Encode with Huffman
    let huf_encoded = LosslessConfig::new()
        .with_ans(false)
        .encode(&data, 32, 32, PixelLayout::Rgb8)
        .unwrap();

    // Encode with ANS
    let ans_encoded = LosslessConfig::new()
        .with_ans(true)
        .encode(&data, 32, 32, PixelLayout::Rgb8)
        .unwrap();

    eprintln!(
        "32x32 RGB: Huffman={} bytes, ANS={} bytes, savings={:.1}%",
        huf_encoded.len(),
        ans_encoded.len(),
        (1.0 - ans_encoded.len() as f64 / huf_encoded.len() as f64) * 100.0
    );

    // ANS should not be significantly larger than Huffman
    // (for small images the overhead can make ANS larger, but it should be close)
    assert!(
        ans_encoded.len() <= huf_encoded.len() + huf_encoded.len() / 5,
        "ANS should not be >20% larger than Huffman"
    );
}

// extracted from jxl-modular/src/headers/color_encoding.rs

// ---- Roundtrip decode tests with jxl-rs ----

#[test]
fn test_roundtrip_custom_white_point_d50() {
    // Encode a small image with D50 custom white point, decode with jxl-rs
    let width = 16u32;
    let height = 16u32;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let ce = ColorEncoding::with_custom_white_point(CIExy::new(0.3457, 0.3585));

    let encoded = crate::LosslessConfig::new()
        .encode_request(width, height, crate::PixelLayout::Rgb8)
        .with_color_encoding(ce)
        .encode(&pixels)
        .expect("encoding with custom white point should succeed");

    // Decode with jxl-rs (primary decoder)
    let decoded = crate::test_helpers::decode_with_jxl_rs(&encoded)
        .expect("jxl-rs should decode custom white point");
    assert_eq!(decoded.width, width as usize);
    assert_eq!(decoded.height, height as usize);
}

// extracted from jxl-modular/src/headers/color_encoding.rs

#[test]
fn test_roundtrip_custom_primaries() {
    // Encode with Adobe RGB-like custom primaries
    let width = 16u32;
    let height = 16u32;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 7) % 256) as u8)
        .collect();

    let ce = ColorEncoding::with_custom_primaries(CustomPrimaries {
        red: CIExy::new(0.6400, 0.3300),
        green: CIExy::new(0.2100, 0.7100),
        blue: CIExy::new(0.1500, 0.0600),
    });

    let encoded = crate::LosslessConfig::new()
        .encode_request(width, height, crate::PixelLayout::Rgb8)
        .with_color_encoding(ce)
        .encode(&pixels)
        .expect("encoding with custom primaries should succeed");

    let decoded = crate::test_helpers::decode_with_jxl_rs(&encoded)
        .expect("jxl-rs should decode custom primaries");
    assert_eq!(decoded.width, width as usize);
    assert_eq!(decoded.height, height as usize);
}

// extracted from jxl-modular/src/headers/color_encoding.rs

#[test]
fn test_roundtrip_custom_white_point_and_primaries() {
    // ProPhoto RGB: D50 white point + wide gamut primaries
    let width = 16u32;
    let height = 16u32;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 13) % 256) as u8)
        .collect();

    let ce = ColorEncoding::with_custom_white_point_and_primaries(
        CIExy::new(0.3457, 0.3585), // D50
        CustomPrimaries {
            red: CIExy::new(0.7347, 0.2653),
            green: CIExy::new(0.1596, 0.8404),
            blue: CIExy::new(0.0366, 0.0001),
        },
    );

    let encoded = crate::LosslessConfig::new()
        .encode_request(width, height, crate::PixelLayout::Rgb8)
        .with_color_encoding(ce)
        .encode(&pixels)
        .expect("encoding with custom WP + primaries should succeed");

    let decoded = crate::test_helpers::decode_with_jxl_rs(&encoded)
        .expect("jxl-rs should decode custom WP + primaries");
    assert_eq!(decoded.width, width as usize);
    assert_eq!(decoded.height, height as usize);
}

// extracted from jxl-modular/src/headers/color_encoding.rs

#[test]
fn test_from_cicp_roundtrip_srgb_with_jxl_rs() {
    // Encode a small image with CICP-derived sRGB encoding, decode with jxl-rs.
    let width = 16u32;
    let height = 16u32;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let ce = ColorEncoding::from_cicp(1, 13, 0, true).unwrap();
    let encoded = crate::LosslessConfig::new()
        .encode_request(width, height, crate::PixelLayout::Rgb8)
        .with_color_encoding(ce)
        .encode(&pixels)
        .expect("CICP-derived sRGB encoding should succeed");

    let decoded = crate::test_helpers::decode_with_jxl_rs(&encoded)
        .expect("jxl-rs should decode CICP-derived sRGB");
    assert_eq!(decoded.width, width as usize);
    assert_eq!(decoded.height, height as usize);
}

// extracted from jxl-modular/src/headers/color_encoding.rs

#[test]
fn test_from_cicp_roundtrip_bt2100_pq_with_jxl_rs() {
    // Encode a small image with BT.2100 PQ via CICP, decode with jxl-rs.
    let width = 16u32;
    let height = 16u32;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 7) % 256) as u8)
        .collect();

    let ce = ColorEncoding::from_cicp(9, 16, 0, true).unwrap();
    let encoded = crate::LosslessConfig::new()
        .encode_request(width, height, crate::PixelLayout::Rgb8)
        .with_color_encoding(ce)
        .encode(&pixels)
        .expect("CICP-derived BT.2100 PQ should succeed");

    let decoded = crate::test_helpers::decode_with_jxl_rs(&encoded)
        .expect("jxl-rs should decode CICP-derived BT.2100 PQ");
    assert_eq!(decoded.width, width as usize);
    assert_eq!(decoded.height, height as usize);
}
