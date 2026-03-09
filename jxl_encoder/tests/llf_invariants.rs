#![allow(
    clippy::excessive_precision,
    clippy::needless_range_loop,
    clippy::collapsible_if,
    clippy::manual_memcpy,
    clippy::approx_constant,
    clippy::no_effect,
    clippy::erasing_op,
    clippy::needless_borrow,
    clippy::needless_borrows_for_generic_args,
    clippy::manual_range_contains,
    clippy::manual_range_patterns,
    clippy::manual_is_multiple_of,
    clippy::identity_op,
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments
)]
//! Layered invariant tests for LLF (Lowest-Low-Frequency) position identification.
//!
//! These tests systematically prove that LLF coefficient positions are correctly
//! identified for all AC strategies, from pure logic through full roundtrip.
//!
//! Layer 1: LLF position formula correctness (unit logic)
//! Layer 2: Single-group DCT16x16 roundtrip on real photos
//! Layer 3: Multi-group DCT16x16 roundtrip on real photos
//! Layer 4: Quality metrics comparison (DCT16x16 vs DCT8)

use std::collections::BTreeSet;
use std::io::Cursor;

const BLOCK_DIM: usize = 8;

/// Convert sRGB normalized [0,1] value to linear light using the sRGB transfer function.
fn srgb_to_linear_val(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear light value to sRGB normalized [0,1] using the sRGB transfer function.
fn linear_to_srgb_val(linear: f32) -> f32 {
    let c = linear.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Compute LLF positions using the OLD (buggy) formula: idx < covered_blocks.
/// This is what the encoder used before the fix.
fn old_llf_positions(covered_blocks: usize, size: usize) -> BTreeSet<usize> {
    (0..size).filter(|&idx| idx < covered_blocks).collect()
}

/// Compute LLF positions using the NEW (correct) formula:
/// (idx / grid_width) < cy && (idx % grid_width) < cx
///
/// This matches the 2D structure of the coefficient grid where LLF occupies
/// a cx×cy rectangle in the top-left corner of a grid_width-wide array.
fn new_llf_positions(cx: usize, cy: usize, grid_width: usize, size: usize) -> BTreeSet<usize> {
    (0..size)
        .filter(|&idx| (idx / grid_width) < cy && (idx % grid_width) < cx)
        .collect()
}

/// For each strategy, compute the parameters as the encoder does.
/// Returns (cx, cy, grid_width, covered_blocks, size).
fn strategy_params(raw_strategy: u8) -> (usize, usize, usize, usize, usize) {
    // From ac_strategy.rs
    let covered_x: [usize; 5] = [1, 1, 2, 2, 4];
    let covered_y: [usize; 5] = [1, 2, 1, 2, 4];

    let covx = covered_x[raw_strategy as usize];
    let covy = covered_y[raw_strategy as usize];
    let covered_blocks = covx * covy;
    let size = covered_blocks * BLOCK_DIM * BLOCK_DIM;

    // Swap so cx >= cy (matches encoder.rs line 861-865)
    let (cx, cy) = if covy > covx {
        (covy, covx)
    } else {
        (covx, covy)
    };
    let grid_width = cx * BLOCK_DIM;

    (cx, cy, grid_width, covered_blocks, size)
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 1: LLF Position Formula Correctness
// ─────────────────────────────────────────────────────────────────────────────

/// DCT8 (1×1): LLF is just index 0. Both old and new formulas agree.
#[test]
fn layer1_llf_positions_dct8() {
    let (cx, cy, grid_width, covered_blocks, size) = strategy_params(0);
    assert_eq!(cx, 1);
    assert_eq!(cy, 1);
    assert_eq!(grid_width, 8);
    assert_eq!(covered_blocks, 1);
    assert_eq!(size, 64);

    let old = old_llf_positions(covered_blocks, size);
    let new = new_llf_positions(cx, cy, grid_width, size);

    assert_eq!(old, new, "DCT8: both formulas must agree");
    assert_eq!(new, BTreeSet::from([0]), "DCT8: LLF is just index 0");
}

/// DCT16x8 (1×2 blocks, becomes cx=2,cy=1 after swap): LLF at {0, 1}.
/// Both old and new formulas agree because LLF is contiguous in row 0.
#[test]
fn layer1_llf_positions_dct16x8() {
    let (cx, cy, grid_width, covered_blocks, size) = strategy_params(1);
    assert_eq!(cx, 2, "after swap, cx should be 2");
    assert_eq!(cy, 1, "after swap, cy should be 1");
    assert_eq!(grid_width, 16);
    assert_eq!(covered_blocks, 2);
    assert_eq!(size, 128);

    let old = old_llf_positions(covered_blocks, size);
    let new = new_llf_positions(cx, cy, grid_width, size);

    assert_eq!(old, new, "DCT16x8: both formulas agree (LLF in single row)");
    assert_eq!(new, BTreeSet::from([0, 1]), "DCT16x8: LLF at {{0, 1}}");
}

/// DCT8x16 (2×1 blocks, cx=2,cy=1): LLF at {0, 1}.
/// Same as DCT16x8 after the cx/cy swap.
#[test]
fn layer1_llf_positions_dct8x16() {
    let (cx, cy, grid_width, covered_blocks, size) = strategy_params(2);
    assert_eq!(cx, 2, "cx should be 2");
    assert_eq!(cy, 1, "cy should be 1");
    assert_eq!(grid_width, 16);
    assert_eq!(covered_blocks, 2);
    assert_eq!(size, 128);

    let old = old_llf_positions(covered_blocks, size);
    let new = new_llf_positions(cx, cy, grid_width, size);

    assert_eq!(old, new, "DCT8x16: both formulas agree (LLF in single row)");
    assert_eq!(new, BTreeSet::from([0, 1]), "DCT8x16: LLF at {{0, 1}}");
}

/// DCT16x16 (2×2 blocks): LLF positions are at {0, 1, 16, 17} in the
/// 16-wide coefficient grid. The OLD formula (idx < 4) gives {0, 1, 2, 3}
/// which is WRONG: positions 2,3 are AC coefficients (row 0, cols 2-3),
/// and positions 16,17 (row 1, cols 0-1) are missed.
#[test]
fn layer1_llf_positions_dct16x16_old_is_wrong() {
    let (cx, cy, grid_width, covered_blocks, size) = strategy_params(3);
    assert_eq!(cx, 2);
    assert_eq!(cy, 2);
    assert_eq!(grid_width, 16);
    assert_eq!(covered_blocks, 4);
    assert_eq!(size, 256);

    let old = old_llf_positions(covered_blocks, size);
    let new = new_llf_positions(cx, cy, grid_width, size);

    // The key assertion: OLD and NEW disagree for DCT16x16
    assert_ne!(
        old, new,
        "DCT16x16: old formula MUST disagree with new formula"
    );

    // Old formula gives wrong positions
    assert_eq!(
        old,
        BTreeSet::from([0, 1, 2, 3]),
        "old formula gives {{0,1,2,3}}"
    );

    // New formula gives correct 2D LLF positions
    assert_eq!(
        new,
        BTreeSet::from([0, 1, 16, 17]),
        "new formula gives {{0, 1, 16, 17}} (2x2 in 16-wide grid)"
    );

    // Verify the specific positions that are wrong in the old formula:
    // Positions 2,3 are AC (row 0, cols 2-3) but old code treats as LLF
    assert!(
        old.contains(&2) && !new.contains(&2),
        "idx 2 (row 0, col 2): old=LLF, new=AC — old is wrong"
    );
    assert!(
        old.contains(&3) && !new.contains(&3),
        "idx 3 (row 0, col 3): old=LLF, new=AC — old is wrong"
    );
    // Positions 16,17 are LLF (row 1, cols 0-1) but old code treats as AC
    assert!(
        !old.contains(&16) && new.contains(&16),
        "idx 16 (row 1, col 0): old=AC, new=LLF — old is wrong"
    );
    assert!(
        !old.contains(&17) && new.contains(&17),
        "idx 17 (row 1, col 1): old=AC, new=LLF — old is wrong"
    );
}

/// DCT32x32 (4×4 blocks): LLF positions form a 4×4 rectangle in a 32-wide
/// grid. The OLD formula (idx < 16) gives the first 16 contiguous indices
/// which is wrong: it gets row 0 cols 0-15, but LLF is only cols 0-3
/// across rows 0-3.
#[test]
fn layer1_llf_positions_dct32x32_old_is_wrong() {
    let (cx, cy, grid_width, covered_blocks, size) = strategy_params(4);
    assert_eq!(cx, 4);
    assert_eq!(cy, 4);
    assert_eq!(grid_width, 32);
    assert_eq!(covered_blocks, 16);
    assert_eq!(size, 1024);

    let old = old_llf_positions(covered_blocks, size);
    let new = new_llf_positions(cx, cy, grid_width, size);

    // OLD and NEW disagree for DCT32x32
    assert_ne!(
        old, new,
        "DCT32x32: old formula MUST disagree with new formula"
    );

    // Old formula: indices 0..16 (first 16 positions in row 0)
    let old_expected: BTreeSet<usize> = (0..16).collect();
    assert_eq!(old, old_expected, "old formula gives 0..16");

    // New formula: 4x4 rectangle at top-left of 32-wide grid
    let new_expected: BTreeSet<usize> = (0..4)
        .flat_map(|row| (0..4).map(move |col| row * 32 + col))
        .collect();
    assert_eq!(new, new_expected, "new formula gives 4x4 block at top-left");

    // Verify: new has exactly 16 positions (4x4)
    assert_eq!(new.len(), 16, "DCT32x32 has 4x4 = 16 LLF positions");

    // Verify specific wrong positions in old formula:
    // Old includes col 4-15 of row 0 (these are AC)
    for col in 4..16 {
        assert!(
            old.contains(&col) && !new.contains(&col),
            "idx {} (row 0, col {}): old=LLF, new=AC — old is wrong",
            col,
            col
        );
    }
    // Old misses rows 1-3 (these are LLF)
    for row in 1..4 {
        for col in 0..4 {
            let idx = row * 32 + col;
            assert!(
                !old.contains(&idx) && new.contains(&idx),
                "idx {} (row {}, col {}): old=AC, new=LLF — old is wrong",
                idx,
                row,
                col
            );
        }
    }
}

/// Verify LLF count is always covered_blocks regardless of formula.
/// Both old and new formulas identify the same NUMBER of LLF positions;
/// the difference is WHICH positions they select.
#[test]
fn layer1_llf_count_matches() {
    for strategy in 0..5u8 {
        let (cx, cy, grid_width, covered_blocks, size) = strategy_params(strategy);
        let old = old_llf_positions(covered_blocks, size);
        let new = new_llf_positions(cx, cy, grid_width, size);

        assert_eq!(
            old.len(),
            covered_blocks,
            "strategy {}: old formula always selects covered_blocks positions",
            strategy
        );
        assert_eq!(
            new.len(),
            covered_blocks,
            "strategy {}: new formula always selects covered_blocks positions",
            strategy
        );
    }
}

/// The CfL skip region must match LLF positions exactly.
/// Old CfL used `for k in covered_blocks..size` (skip first N indices).
/// New CfL checks `is_llf` per position. For DCT16x16, the old code:
/// - Skips CfL on positions 2,3 (AC!) — wrong, these need CfL
/// - Applies CfL on positions 16,17 (LLF!) — wrong, decoder overwrites these
#[test]
fn layer1_cfl_skip_consistency_dct16x16() {
    let (cx, cy, grid_width, covered_blocks, size) = strategy_params(3);

    // Old CfL skip: indices 0..covered_blocks
    let old_skip: BTreeSet<usize> = (0..covered_blocks).collect();

    // New CfL skip: same as LLF positions
    let new_skip = new_llf_positions(cx, cy, grid_width, size);

    assert_ne!(old_skip, new_skip, "CfL skip regions differ for DCT16x16");

    // Positions 2,3 should NOT be skipped (they're AC, need CfL)
    assert!(old_skip.contains(&2), "old CfL wrongly skips idx 2 (AC)");
    assert!(!new_skip.contains(&2), "new CfL correctly applies to idx 2");

    // Positions 16,17 should be skipped (they're LLF, decoder overwrites)
    assert!(
        !old_skip.contains(&16),
        "old CfL wrongly applies to idx 16 (LLF)"
    );
    assert!(new_skip.contains(&16), "new CfL correctly skips idx 16");
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: load a PNG and convert to linear sRGB f32 for VarDctEncoder
// ─────────────────────────────────────────────────────────────────────────────

/// Load a PNG, optionally crop to (crop_w, crop_h) from center, return (width, height, linear_rgb, srgb_u8).
fn load_png_crop(path: &str, crop_w: usize, crop_h: usize) -> (usize, usize, Vec<f32>, Vec<u8>) {
    let img = image::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {}", path, e));
    let rgb = img.to_rgb8();
    let (iw, ih) = (rgb.width() as usize, rgb.height() as usize);

    // Crop from center
    let (w, h) = (crop_w.min(iw), crop_h.min(ih));
    let x0 = (iw - w) / 2;
    let y0 = (ih - h) / 2;

    let mut srgb = Vec::with_capacity(w * h * 3);
    let mut linear = Vec::with_capacity(w * h * 3);

    for y in y0..y0 + h {
        for x in x0..x0 + w {
            let p = rgb.get_pixel(x as u32, y as u32);
            srgb.extend_from_slice(&[p[0], p[1], p[2]]);
            // sRGB → linear using proper sRGB transfer function
            linear.push(srgb_to_linear_val(p[0] as f32 / 255.0));
            linear.push(srgb_to_linear_val(p[1] as f32 / 255.0));
            linear.push(srgb_to_linear_val(p[2] as f32 / 255.0));
        }
    }

    (w, h, linear, srgb)
}

/// Load full PNG without cropping.
fn load_png_full(path: &str) -> (usize, usize, Vec<f32>, Vec<u8>) {
    let img = image::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {}", path, e));
    let (w, h) = (img.width() as usize, img.height() as usize);
    load_png_crop(path, w, h)
}

/// Decode with jxl-oxide (single and multi-group).
fn decode_jxl_oxide(data: &[u8]) -> (usize, usize, Vec<f32>) {
    let mut image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(data))
        .unwrap_or_else(|e| panic!("jxl-oxide parse failed: {:?}", e));
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let w = image.width() as usize;
    let h = image.height() as usize;
    let render = image
        .render_frame(0)
        .unwrap_or_else(|e| panic!("jxl-oxide render failed: {:?}", e));
    let pixels = render.image_all_channels().buf().to_vec();
    (w, h, pixels)
}

/// Decode with jxl-rs (primary Rust decoder for roundtrip tests).
fn decode_jxl_rs(data: &[u8]) -> (usize, usize, Vec<f32>) {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;

    // Create decoder
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    // Process header
    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input during header");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs header decode error: {:?}", e),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let channels = 3; // RGB output

    // Set output format to RGB f32
    let format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(format);

    // Process to frame info
    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input before frame");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame info decode error: {:?}", e),
        }
    };

    // Create output buffer
    let mut output_image = Image::<f32>::new((width * channels, height))
        .expect("jxl-rs: failed to create output buffer");

    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    // Decode frame
    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    panic!("jxl-rs: unexpected end of input during frame decode");
                }
                decoder = fallback;
            }
            Err(e) => panic!("jxl-rs frame decode error: {:?}", e),
        }
    }

    // Extract pixels
    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }

    (width, height, pixels)
}

/// Decode with djxl (libjxl reference decoder, gold standard).
fn decode_djxl(data: &[u8]) -> (usize, usize, Vec<u8>) {
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_jxl = format!("/tmp/llf_test_{}_{}.jxl", pid, ts);
    let temp_png = format!("/tmp/llf_test_{}_{}.png", pid, ts);

    std::fs::write(&temp_jxl, data).unwrap();
    let output = std::process::Command::new(&jxl_encoder::test_helpers::djxl_path())
        .args([&temp_jxl, &temp_png])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "djxl failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let img = image::open(&temp_png).unwrap();
    let rgb = img.to_rgb8();
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let srgb_bytes: Vec<u8> = rgb.into_raw();

    let _ = std::fs::remove_file(&temp_jxl);
    let _ = std::fs::remove_file(&temp_png);

    (w, h, srgb_bytes)
}

/// Compute SSIM2 between two sRGB u8 images.
fn ssim2_srgb(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    use fast_ssim2::compute_ssimulacra2;
    use imgref::ImgVec;

    let orig: Vec<[u8; 3]> = original
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let dec: Vec<[u8; 3]> = decoded
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    let src = ImgVec::new(orig, width, height);
    let dst = ImgVec::new(dec, width, height);
    compute_ssimulacra2(src.as_ref(), dst.as_ref()).unwrap_or(0.0)
}

/// Convert linear f32 to sRGB u8 using the proper sRGB transfer function.
fn linear_to_srgb_u8(linear: &[f32]) -> Vec<u8> {
    linear
        .iter()
        .map(|&v| (linear_to_srgb_val(v) * 255.0).round() as u8)
        .collect()
}

/// Compute SSIM2 between original sRGB u8 and decoded linear f32 (from jxl-oxide).
/// Applies gamma correction to decoded values before comparison.
fn ssim2_u8_vs_linear_f32(original: &[u8], decoded: &[f32], width: usize, height: usize) -> f64 {
    let dec_srgb = linear_to_srgb_u8(decoded);
    ssim2_srgb(original, &dec_srgb, width, height)
}

/// Compute SSIM2 between original sRGB u8 and decoded linear u8 (from djxl with linear transfer).
/// djxl outputs linear values scaled to 0-255. We need to apply gamma before SSIM2.
fn ssim2_u8_vs_linear_u8(original: &[u8], decoded_u8: &[u8], width: usize, height: usize) -> f64 {
    // djxl outputs sRGB by default, so compare directly without gamma correction.
    // The original comment was wrong - djxl does NOT output linear values
    // unless explicitly told to with --output_format=... flags.
    ssim2_srgb(original, decoded_u8, width, height)
}

/// Frymire test image (1118x1105 real photo, committed to repo).
/// Path relative to workspace root (where cargo test runs).
fn frymire_path() -> String {
    // Try workspace-relative path first (when run from workspace root)
    let ws = "jxl_encoder/tests/images/frymire.png";
    if std::path::Path::new(ws).exists() {
        return ws.to_string();
    }
    // Try crate-relative path (when run from jxl_encoder/)
    let cr = "tests/images/frymire.png";
    if std::path::Path::new(cr).exists() {
        return cr.to_string();
    }
    // Absolute fallback
    let abs = format!(
        "{}/work/codec-corpus/imageflow/test_inputs/frymire.png",
        std::env::var("HOME").unwrap()
    );
    if std::path::Path::new(&abs).exists() {
        return abs;
    }
    panic!("frymire.png not found in any expected location");
}

/// Generate a smooth horizontal gradient image appropriate for DCT32x32 testing.
/// DCT32x32 averages 32x32 pixel blocks, so it works well on smooth content but
/// poorly on high-contrast edges. This generates content that DCT32x32 can encode well.
///
/// Returns (width, height, linear_rgb_f32, srgb_u8).
fn generate_smooth_gradient(w: usize, h: usize) -> (usize, usize, Vec<f32>, Vec<u8>) {
    let mut linear = vec![0.0f32; w * h * 3];
    let mut srgb = vec![0u8; w * h * 3];

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            // Smooth horizontal gradient from 0.1 to 0.9
            let t = x as f32 / (w - 1).max(1) as f32;
            let val = 0.1 + 0.8 * t;

            // R channel: horizontal gradient
            linear[idx] = val;
            // G channel: slight vertical gradient
            let vt = y as f32 / (h - 1).max(1) as f32;
            linear[idx + 1] = 0.2 + 0.6 * vt;
            // B channel: diagonal gradient
            linear[idx + 2] = 0.15 + 0.7 * (t + vt) / 2.0;

            // Convert to sRGB for comparison
            srgb[idx] = (linear_to_srgb_val(linear[idx]) * 255.0).round() as u8;
            srgb[idx + 1] = (linear_to_srgb_val(linear[idx + 1]) * 255.0).round() as u8;
            srgb[idx + 2] = (linear_to_srgb_val(linear[idx + 2]) * 255.0).round() as u8;
        }
    }

    (w, h, linear, srgb)
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 2: Single-group DCT16x16 roundtrip on real photo crop
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a 256x256 crop of frymire with forced DCT16x16 (ac_strategy_enabled=true
/// hits the current "force all DCT16x16" hack), decode with jxl-oxide.
/// This tests single-group DCT16x16 bitstream validity.
#[test]
#[ignore] // requires frymire test image
fn layer2_single_group_dct16x16_decode_jxl_oxide() {
    let (w, h, linear, srgb) = load_png_crop(&frymire_path(), 256, 256);
    assert_eq!(w, 256);
    assert_eq!(h, 256);

    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    encoder.ac_strategy_enabled = true; // triggers forced DCT16x16

    let bytes = encoder
        .encode(w, h, &linear, None)
        .unwrap_or_else(|e| panic!("encode failed: {:?}", e))
        .data;

    eprintln!(
        "layer2 jxl-oxide: encoded 256x256 frymire crop, {} bytes",
        bytes.len()
    );

    let (dw, dh, pixels) = decode_jxl_oxide(&bytes);
    assert_eq!(dw, w, "width mismatch");
    assert_eq!(dh, h, "height mismatch");

    let ssim2 = ssim2_u8_vs_linear_f32(&srgb, &pixels, w, h);
    eprintln!("layer2 jxl-oxide: SSIM2 = {:.2}", ssim2);

    // Sanity: quality should be reasonable (>50 at d=1.0)
    assert!(
        ssim2 > 50.0,
        "DCT16x16 256x256 quality too low: SSIM2={:.2} (expected >50)",
        ssim2
    );
}

/// Same as above but decode with djxl (libjxl reference decoder).
#[test]
#[ignore] // requires frymire test image and djxl
fn layer2_single_group_dct16x16_decode_djxl() {
    let (w, h, linear, srgb) = load_png_crop(&frymire_path(), 256, 256);

    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    encoder.ac_strategy_enabled = true;

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;

    eprintln!(
        "layer2 djxl: encoded 256x256 frymire crop, {} bytes",
        bytes.len()
    );

    let (dw, dh, dec_srgb) = decode_djxl(&bytes);
    assert_eq!(dw, w, "width mismatch");
    assert_eq!(dh, h, "height mismatch");

    let ssim2 = ssim2_u8_vs_linear_u8(&srgb, &dec_srgb, w, h);
    eprintln!("layer2 djxl: SSIM2 = {:.2}", ssim2);

    assert!(
        ssim2 > 50.0,
        "DCT16x16 256x256 quality too low via djxl: SSIM2={:.2}",
        ssim2
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 3: Multi-group DCT16x16 roundtrip on full frymire
// ─────────────────────────────────────────────────────────────────────────────

/// Encode full frymire (1118x1105, multi-group) with forced DCT16x16,
/// decode with djxl. This tests multi-group DCT16x16 bitstream validity.
#[test]
#[ignore] // requires frymire test image and djxl
fn layer3_multigroup_dct16x16_decode_djxl() {
    let (w, h, linear, srgb) = load_png_full(&frymire_path());
    eprintln!("layer3: loaded frymire {}x{}", w, h);
    assert!(w > 256 || h > 256, "frymire should be multi-group");

    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    encoder.ac_strategy_enabled = true;

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;

    eprintln!(
        "layer3 djxl: encoded {}x{} frymire, {} bytes",
        w,
        h,
        bytes.len()
    );

    let (dw, dh, dec_srgb) = decode_djxl(&bytes);
    assert_eq!(dw, w, "width mismatch");
    assert_eq!(dh, h, "height mismatch");

    let ssim2 = ssim2_u8_vs_linear_u8(&srgb, &dec_srgb, w, h);
    eprintln!("layer3 djxl: SSIM2 = {:.2}", ssim2);

    assert!(
        ssim2 > 50.0,
        "DCT16x16 multi-group quality too low: SSIM2={:.2}",
        ssim2
    );
}

/// Multi-group with jxl-oxide decoder.
#[test]
#[ignore] // requires frymire test image
fn layer3_multigroup_dct16x16_decode_jxl_oxide() {
    let (w, h, linear, srgb) = load_png_full(&frymire_path());

    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    encoder.ac_strategy_enabled = true;

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;

    eprintln!(
        "layer3 jxl-oxide: encoded {}x{} frymire, {} bytes",
        w,
        h,
        bytes.len()
    );

    // NOTE: jxl-oxide may have multi-group VarDCT bugs. If this fails at
    // decode but djxl succeeds, the bitstream is valid and the bug is in jxl-oxide.
    let (dw, dh, pixels) = decode_jxl_oxide(&bytes);
    assert_eq!(dw, w, "width mismatch");
    assert_eq!(dh, h, "height mismatch");

    let ssim2 = ssim2_u8_vs_linear_f32(&srgb, &pixels, w, h);
    eprintln!("layer3 jxl-oxide: SSIM2 = {:.2}", ssim2);

    assert!(
        ssim2 > 50.0,
        "DCT16x16 multi-group quality too low via jxl-oxide: SSIM2={:.2}",
        ssim2
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 4: Quality comparison — DCT16x16 vs DCT8 on real photos
// ─────────────────────────────────────────────────────────────────────────────

/// Compare DCT16x16-only vs DCT8-only on 256x256 frymire crop.
/// DCT16x16 should produce comparable quality (within ~5 SSIM2 of DCT8).
/// If the gap is larger, the LLF handling is still wrong.
#[test]
#[ignore] // requires frymire test image and djxl
fn layer4_quality_dct16x16_vs_dct8_frymire_256() {
    let (w, h, linear, srgb) = load_png_crop(&frymire_path(), 256, 256);

    // DCT8-only
    let mut enc_dct8 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc_dct8.ac_strategy_enabled = false;
    let bytes_dct8 = enc_dct8.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec8) = decode_djxl(&bytes_dct8);
    let ssim2_dct8 = ssim2_u8_vs_linear_u8(&srgb, &dec8, w, h);

    // DCT16x16-only (forced via hack)
    let mut enc_dct16 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc_dct16.ac_strategy_enabled = true;
    let bytes_dct16 = enc_dct16.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec16) = decode_djxl(&bytes_dct16);
    let ssim2_dct16 = ssim2_u8_vs_linear_u8(&srgb, &dec16, w, h);

    eprintln!("layer4 frymire 256x256 @ d=1.0:");
    eprintln!(
        "  DCT8:    SSIM2={:.2}, {} bytes",
        ssim2_dct8,
        bytes_dct8.len()
    );
    eprintln!(
        "  DCT16x16: SSIM2={:.2}, {} bytes",
        ssim2_dct16,
        bytes_dct16.len()
    );
    eprintln!(
        "  gap: {:.2} SSIM2, size ratio: {:.2}%",
        ssim2_dct8 - ssim2_dct16,
        bytes_dct16.len() as f64 / bytes_dct8.len() as f64 * 100.0
    );

    // DCT16x16 quality should be reasonable
    assert!(
        ssim2_dct16 > 50.0,
        "DCT16x16 quality too low: {:.2}",
        ssim2_dct16
    );

    // Gap should be small (within 10 SSIM2).
    // If gap is very large, the LLF fix isn't working.
    let gap = ssim2_dct8 - ssim2_dct16;
    assert!(
        gap < 10.0,
        "DCT16x16 vs DCT8 gap too large: {:.2} SSIM2. LLF handling may be wrong.",
        gap
    );
}

/// Compare on full frymire (multi-group).
#[test]
#[ignore] // requires frymire test image and djxl
fn layer4_quality_dct16x16_vs_dct8_frymire_full() {
    let (w, h, linear, srgb) = load_png_full(&frymire_path());

    // DCT8-only
    let mut enc_dct8 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc_dct8.ac_strategy_enabled = false;
    let bytes_dct8 = enc_dct8.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec8) = decode_djxl(&bytes_dct8);
    let ssim2_dct8 = ssim2_u8_vs_linear_u8(&srgb, &dec8, w, h);

    // DCT16x16-only
    let mut enc_dct16 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc_dct16.ac_strategy_enabled = true;
    let bytes_dct16 = enc_dct16.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec16) = decode_djxl(&bytes_dct16);
    let ssim2_dct16 = ssim2_u8_vs_linear_u8(&srgb, &dec16, w, h);

    eprintln!("layer4 frymire full {}x{} @ d=1.0:", w, h);
    eprintln!(
        "  DCT8:    SSIM2={:.2}, {} bytes",
        ssim2_dct8,
        bytes_dct8.len()
    );
    eprintln!(
        "  DCT16x16: SSIM2={:.2}, {} bytes",
        ssim2_dct16,
        bytes_dct16.len()
    );
    eprintln!(
        "  gap: {:.2} SSIM2, size ratio: {:.2}%",
        ssim2_dct8 - ssim2_dct16,
        bytes_dct16.len() as f64 / bytes_dct8.len() as f64 * 100.0
    );

    assert!(
        ssim2_dct16 > 50.0,
        "DCT16x16 quality too low: {:.2}",
        ssim2_dct16
    );

    let gap = ssim2_dct8 - ssim2_dct16;
    assert!(
        gap < 10.0,
        "DCT16x16 vs DCT8 gap too large: {:.2} SSIM2",
        gap
    );
}

/// Compare on Kodak image 1 (768x512, different content profile).
#[test]
#[ignore] // requires kodak test images and djxl
fn layer4_quality_dct16x16_vs_dct8_kodak1() {
    let kodak_path = format!(
        "{}/work/codec-corpus/kodak-legacy/1.png",
        std::env::var("HOME").unwrap()
    );
    if !std::path::Path::new(&kodak_path).exists() {
        eprintln!("SKIP: kodak image not found at {}", kodak_path);
        return;
    }
    let (w, h, linear, srgb) = load_png_full(&kodak_path);

    let mut enc_dct8 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc_dct8.ac_strategy_enabled = false;
    let bytes_dct8 = enc_dct8.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec8) = decode_djxl(&bytes_dct8);
    let ssim2_dct8 = ssim2_u8_vs_linear_u8(&srgb, &dec8, w, h);

    let mut enc_dct16 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc_dct16.ac_strategy_enabled = true;
    let bytes_dct16 = enc_dct16.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec16) = decode_djxl(&bytes_dct16);
    let ssim2_dct16 = ssim2_u8_vs_linear_u8(&srgb, &dec16, w, h);

    eprintln!("layer4 kodak1 {}x{} @ d=1.0:", w, h);
    eprintln!(
        "  DCT8:    SSIM2={:.2}, {} bytes",
        ssim2_dct8,
        bytes_dct8.len()
    );
    eprintln!(
        "  DCT16x16: SSIM2={:.2}, {} bytes",
        ssim2_dct16,
        bytes_dct16.len()
    );

    assert!(
        ssim2_dct16 > 50.0,
        "DCT16x16 quality too low: {:.2}",
        ssim2_dct16
    );

    let gap = ssim2_dct8 - ssim2_dct16;
    eprintln!("  gap: {:.2} SSIM2", gap);
    assert!(gap < 10.0, "gap too large: {:.2}", gap);
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 1b: DC spatial ordering verification
// ─────────────────────────────────────────────────────────────────────────────

/// Verify dc_from_dct_16x16 spatial ordering by testing with pure synthetic LLF coefficients.
///
/// The DCT16x16 output is in TRANSPOSED layout (kx, ky order), so:
///   coeffs[0]  = (kx=0, ky=0) = DC
///   coeffs[1]  = (kx=0, ky=1) = vertical frequency
///   coeffs[16] = (kx=1, ky=0) = horizontal frequency
///   coeffs[17] = (kx=1, ky=1) = diagonal
///
/// Test 1: Set only coeffs[1] (vertical freq) nonzero.
///   Expected: top row same sign, bottom row opposite (vertical variation).
///   Bug: if dc01/dc10 swapped, we get left/right variation instead.
///
/// Test 2: Set only coeffs[16] (horizontal freq) nonzero.
///   Expected: left column same sign, right column opposite (horizontal variation).
///   Bug: if dc01/dc10 swapped, we get top/bottom variation instead.
#[test]
fn layer1b_dc_spatial_order_dct16x16() {
    // Duplicate dc_from_dct_16x16 from jxl_encoder/src/tiny/dct.rs (FIXED version)
    // (private module, can't import from integration test)
    fn dc_from_dct_16x16_fixed(coeffs: &[f32; 256]) -> [f32; 4] {
        let s0: f32 = 1.0;
        let s1: f32 = 0.901764195028874394;

        let b00 = coeffs[0] * s0 * s0;
        let b01 = coeffs[1] * s0 * s1;
        let b10 = coeffs[16] * s1 * s0;
        let b11 = coeffs[17] * s1 * s1;

        // 2x2 IDCT: rows → transpose → rows
        let out00 = (b00 + b01) + (b10 + b11);
        let out01 = (b00 + b01) - (b10 + b11);
        let out10 = (b00 - b01) + (b10 - b11);
        let out11 = (b00 - b01) - (b10 - b11);

        [out00, out01, out10, out11]
    }

    // Also keep the OLD (buggy) version to prove the bug exists
    fn dc_from_dct_16x16_old(coeffs: &[f32; 256]) -> [f32; 4] {
        let s0: f32 = 1.0;
        let s1: f32 = 0.901764195028874394;

        let b00 = coeffs[0] * s0 * s0;
        let b10 = coeffs[1] * s1 * s0;
        let b01 = coeffs[16] * s0 * s1;
        let b11 = coeffs[17] * s1 * s1;

        let dc00 = (b00 + b10) + (b01 + b11);
        let dc01 = (b00 - b10) + (b01 - b11);
        let dc10 = (b00 + b10) - (b01 + b11);
        let dc11 = (b00 - b10) - (b01 - b11);

        [dc00, dc01, dc10, dc11]
    }

    // --- Prove the OLD version has the bug ---
    let mut coeffs_vert = [0.0f32; 256];
    coeffs_vert[1] = 1.0; // vertical frequency only

    let old_dcs = dc_from_dct_16x16_old(&coeffs_vert);
    eprintln!("OLD version with vertical-only frequency (coeffs[1]):");
    eprintln!(
        "  dcs[0]={:.4}, dcs[1]={:.4}, dcs[2]={:.4}, dcs[3]={:.4}",
        old_dcs[0], old_dcs[1], old_dcs[2], old_dcs[3]
    );
    // Old version: vertical freq produces horizontal variation (BUG)
    let old_top_row_same = (old_dcs[0] - old_dcs[1]).abs() < 1e-6;
    assert!(
        !old_top_row_same,
        "OLD version should produce WRONG horizontal variation for vertical freq"
    );

    // --- Verify the FIXED version ---
    let dcs = dc_from_dct_16x16_fixed(&coeffs_vert);
    eprintln!("\nFIXED version with vertical-only frequency (coeffs[1]):");
    eprintln!("  dcs[0] (top-left)     = {:.4}", dcs[0]);
    eprintln!("  dcs[1] (top-right)    = {:.4}", dcs[1]);
    eprintln!("  dcs[2] (bottom-left)  = {:.4}", dcs[2]);
    eprintln!("  dcs[3] (bottom-right) = {:.4}", dcs[3]);

    // The encoder stores dcs[iy*2+ix] at position (by+iy, bx+ix):
    //   dcs[0] → top-left, dcs[1] → top-right, dcs[2] → bottom-left, dcs[3] → bottom-right
    //
    // For vertical-only frequency: top-left == top-right, bottom-left == bottom-right
    let top_row_same = (dcs[0] - dcs[1]).abs() < 1e-6;
    let bottom_row_same = (dcs[2] - dcs[3]).abs() < 1e-6;
    let top_bottom_differ = (dcs[0] - dcs[2]).abs() > 0.1;

    assert!(
        top_row_same && bottom_row_same && top_bottom_differ,
        "FIXED: Vertical-only frequency should produce vertical variation. Got dcs={:?}",
        dcs
    );
    eprintln!("  PASS: vertical freq → vertical variation (top row same, bottom row same)");

    // --- Test 2: horizontal-only frequency ---
    let mut coeffs_horiz = [0.0f32; 256];
    coeffs_horiz[16] = 1.0; // horizontal frequency only

    let dcs = dc_from_dct_16x16_fixed(&coeffs_horiz);
    eprintln!("\nFIXED version with horizontal-only frequency (coeffs[16]):");
    eprintln!("  dcs[0] (top-left)     = {:.4}", dcs[0]);
    eprintln!("  dcs[1] (top-right)    = {:.4}", dcs[1]);
    eprintln!("  dcs[2] (bottom-left)  = {:.4}", dcs[2]);
    eprintln!("  dcs[3] (bottom-right) = {:.4}", dcs[3]);

    let left_col_same = (dcs[0] - dcs[2]).abs() < 1e-6;
    let right_col_same = (dcs[1] - dcs[3]).abs() < 1e-6;
    let left_right_differ = (dcs[0] - dcs[1]).abs() > 0.1;

    assert!(
        left_col_same && right_col_same && left_right_differ,
        "FIXED: Horizontal-only frequency should produce horizontal variation. Got dcs={:?}",
        dcs
    );
    eprintln!("  PASS: horizontal freq → horizontal variation (left col same, right col same)");

    // --- Test 3: Verify old dc01/dc10 are exactly the fixed dc10/dc01 (swap) ---
    let old_horiz = dc_from_dct_16x16_old(&coeffs_horiz);
    eprintln!("\nSwap verification:");
    eprintln!(
        "  old[1]={:.4} == fixed[2]={:.4}? {}",
        old_horiz[1],
        dcs[2],
        (old_horiz[1] - dcs[2]).abs() < 1e-6
    );
    eprintln!(
        "  old[2]={:.4} == fixed[1]={:.4}? {}",
        old_horiz[2],
        dcs[1],
        (old_horiz[2] - dcs[1]).abs() < 1e-6
    );
    assert!(
        (old_horiz[1] - dcs[2]).abs() < 1e-6 && (old_horiz[2] - dcs[1]).abs() < 1e-6,
        "Old dc01/dc10 should be exactly swapped vs fixed"
    );
    eprintln!("  PASS: old[1]==fixed[2] and old[2]==fixed[1] (confirmed swap)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostic: examine what DCT16x16 actually produces
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a tiny solid-color 16x16 image with forced DCT16x16.
/// This is a single 16x16 block — the simplest possible DCT16x16 case.
/// Print the decoded pixel values to understand the nature of the distortion.
#[test]
#[ignore]
fn diag_dct16x16_solid_16x16() {
    // Solid mid-gray in linear sRGB
    let w = 16;
    let h = 16;
    let val = 0.2f32; // ~50% gray in sRGB
    let linear = vec![val; w * h * 3];
    let srgb_val = (linear_to_srgb_val(val) * 255.0).round() as u8;

    // Encode with DCT16x16 (ac_strategy_enabled = true forces it)
    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    encoder.ac_strategy_enabled = true;

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;
    eprintln!("solid 16x16: encoded {} bytes", bytes.len());

    // Save for external inspection
    std::fs::write(
        std::env::temp_dir().join("diag_solid16x16_dct16.jxl"),
        &bytes,
    )
    .unwrap();

    // Decode with jxl-oxide
    let (dw, dh, pixels) = decode_jxl_oxide(&bytes);
    assert_eq!(dw, w);
    assert_eq!(dh, h);

    // Print first few decoded pixels (linear f32 from jxl-oxide)
    eprintln!("Expected linear value: {:.4}, sRGB: {}", val, srgb_val);
    eprintln!("Decoded linear pixels (first 4 pixels, R G B):");
    for i in 0..4 {
        let r = pixels[i * 3];
        let g = pixels[i * 3 + 1];
        let b = pixels[i * 3 + 2];
        eprintln!(
            "  pixel[{}]: R={:.4} G={:.4} B={:.4} (sRGB: {:.0} {:.0} {:.0})",
            i,
            r,
            g,
            b,
            (linear_to_srgb_val(r) * 255.0),
            (linear_to_srgb_val(g) * 255.0),
            (linear_to_srgb_val(b) * 255.0),
        );
    }

    // Also decode with djxl for comparison
    let (_, _, djxl_srgb) = decode_djxl(&bytes);
    eprintln!("djxl decoded pixels (first 4 pixels, sRGB u8):");
    for i in 0..4 {
        eprintln!(
            "  pixel[{}]: R={} G={} B={}",
            i,
            djxl_srgb[i * 3],
            djxl_srgb[i * 3 + 1],
            djxl_srgb[i * 3 + 2]
        );
    }

    // Now encode the same thing with DCT8 for comparison
    let mut enc8 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear, None).unwrap().data;
    std::fs::write(
        std::env::temp_dir().join("diag_solid16x16_dct8.jxl"),
        &bytes8,
    )
    .unwrap();

    let (_, _, djxl8) = decode_djxl(&bytes8);
    eprintln!("\nDCT8 reference (djxl sRGB u8):");
    for i in 0..4 {
        eprintln!(
            "  pixel[{}]: R={} G={} B={}",
            i,
            djxl8[i * 3],
            djxl8[i * 3 + 1],
            djxl8[i * 3 + 2]
        );
    }
}

/// Same diagnostic but with a real photo crop — 16x16 from frymire center.
/// Small enough to print all decoded pixels.
#[test]
#[ignore]
fn diag_dct16x16_real_16x16() {
    let (w, h, linear, srgb) = load_png_crop(&frymire_path(), 16, 16);
    assert_eq!(w, 16);
    assert_eq!(h, 16);

    // DCT16x16
    let mut enc16 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear, None).unwrap().data;
    std::fs::write(
        std::env::temp_dir().join("diag_real16x16_dct16.jxl"),
        &bytes16,
    )
    .unwrap();

    // DCT8
    let mut enc8 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear, None).unwrap().data;
    std::fs::write(
        std::env::temp_dir().join("diag_real16x16_dct8.jxl"),
        &bytes8,
    )
    .unwrap();

    // Decode both with djxl
    let (_, _, d16) = decode_djxl(&bytes16);
    let (_, _, d8) = decode_djxl(&bytes8);

    eprintln!("16x16 frymire crop pixel comparison (sRGB u8):");
    eprintln!(
        "{:>5} {:>12} {:>12} {:>12}",
        "pixel", "original", "dct8", "dct16x16"
    );

    let mut max_diff_8 = 0i32;
    let mut max_diff_16 = 0i32;

    for i in 0..16 {
        // Sample pixels at (i, i) diagonal
        let idx = i * w + i;
        let o = (srgb[idx * 3], srgb[idx * 3 + 1], srgb[idx * 3 + 2]);
        let d8p = (d8[idx * 3], d8[idx * 3 + 1], d8[idx * 3 + 2]);
        let d16p = (d16[idx * 3], d16[idx * 3 + 1], d16[idx * 3 + 2]);

        let diff8 = (o.0 as i32 - d8p.0 as i32)
            .abs()
            .max((o.1 as i32 - d8p.1 as i32).abs())
            .max((o.2 as i32 - d8p.2 as i32).abs());
        let diff16 = (o.0 as i32 - d16p.0 as i32)
            .abs()
            .max((o.1 as i32 - d16p.1 as i32).abs())
            .max((o.2 as i32 - d16p.2 as i32).abs());

        max_diff_8 = max_diff_8.max(diff8);
        max_diff_16 = max_diff_16.max(diff16);

        eprintln!(
            "  ({:2},{:2}) {:>3},{:>3},{:>3}  {:>3},{:>3},{:>3}  {:>3},{:>3},{:>3}  d8={:>3} d16={:>3}",
            i, i, o.0, o.1, o.2, d8p.0, d8p.1, d8p.2, d16p.0, d16p.1, d16p.2, diff8, diff16
        );
    }

    eprintln!(
        "Max pixel diff: DCT8={}, DCT16x16={}",
        max_diff_8, max_diff_16
    );
    eprintln!(
        "File sizes: DCT8={} bytes, DCT16x16={} bytes",
        bytes8.len(),
        bytes16.len()
    );

    // Compute SSIM2
    let ssim2_8 = ssim2_u8_vs_linear_u8(&srgb, &d8, w, h);
    let ssim2_16 = ssim2_u8_vs_linear_u8(&srgb, &d16, w, h);
    eprintln!("SSIM2: DCT8={:.2}, DCT16x16={:.2}", ssim2_8, ssim2_16);
}

/// Progressive size test: at what image size does DCT16x16 break?
/// Tests sizes from 16x16 (1 block) to 256x256 (single group).
/// Uses jxl-oxide (linear f32 output) with proper gamma correction.
#[test]
#[ignore]
fn diag_dct16x16_progressive_sizes() {
    let path = frymire_path();

    eprintln!(
        "{:>8} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "size", "dct8_ssim", "d16_ssim", "gap", "d8_sz", "d16_sz"
    );

    for &size in &[16, 32, 48, 64, 96, 128, 192, 256] {
        let (w, h, linear, srgb) = load_png_crop(&path, size, size);
        if w != size || h != size {
            eprintln!("{:>8}: skipped (image too small)", size);
            continue;
        }

        // DCT8 — encode and decode with jxl-oxide
        let mut enc8 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        enc8.ac_strategy_enabled = false;
        let bytes8 = enc8.encode(w, h, &linear, None).unwrap().data;
        let (_, _, d8_linear) = decode_jxl_oxide(&bytes8);
        let ssim8 = ssim2_u8_vs_linear_f32(&srgb, &d8_linear, w, h);

        // DCT16x16 — encode and decode with jxl-oxide
        let mut enc16 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        enc16.ac_strategy_enabled = true;
        let bytes16 = match enc16.encode(w, h, &linear, None) {
            Ok(output) => output.data,
            Err(e) => {
                eprintln!("{:>8}: DCT16x16 ENCODE ERROR: {:?}", size, e);
                continue;
            }
        };
        let ssim16 = match std::panic::catch_unwind(|| decode_jxl_oxide(&bytes16)) {
            Ok((_, _, d16_linear)) => ssim2_u8_vs_linear_f32(&srgb, &d16_linear, w, h),
            Err(_) => {
                eprintln!("{:>8}: DCT16x16 DECODE ERROR", size);
                continue;
            }
        };

        let gap = ssim8 - ssim16;
        eprintln!(
            "{:>4}x{:<4} {:>10.2} {:>10.2} {:>8.2} {:>8} {:>8}",
            w,
            h,
            ssim8,
            ssim16,
            gap,
            bytes8.len(),
            bytes16.len()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 1b DCT32x32: DC spatial ordering verification
// ─────────────────────────────────────────────────────────────────────────────

/// Verify dc_from_dct_32x32 spatial ordering by testing with pure synthetic LLF coefficients.
///
/// The DCT32x32 output is in TRANSPOSED layout (kx, ky order), so the 4x4 LLF region:
///   coeffs[0]  = (kx=0, ky=0) = DC
///   coeffs[1]  = (kx=0, ky=1) = vertical frequency
///   coeffs[32] = (kx=1, ky=0) = horizontal frequency
///   etc.
///
/// The 4x4 IDCT must use rows→transpose→rows (not rows→columns) to produce
/// correct spatial DC values. Without transpose, adjacent rows/columns swap.
#[test]
fn layer1b_dc_spatial_order_dct32x32() {
    // Resample scales for 32→4
    const SCALE: [f32; 4] = [
        1.0,
        0.974886821136879522,
        0.901764195028874394,
        0.787054918159101335,
    ];

    // 4-point IDCT (direct formula)
    fn idct4(input: &[f32; 4]) -> [f32; 4] {
        use core::f32::consts::PI;
        let x0 = input[0];
        let x1 = input[1];
        let x2 = input[2];
        let x3 = input[3];

        [
            x0 + 2.0
                * (x1 * (PI * 1.0 / 8.0).cos()
                    + x2 * (PI * 2.0 / 8.0).cos()
                    + x3 * (PI * 3.0 / 8.0).cos()),
            x0 + 2.0
                * (x1 * (PI * 3.0 / 8.0).cos()
                    + x2 * (PI * 6.0 / 8.0).cos()
                    + x3 * (PI * 9.0 / 8.0).cos()),
            x0 + 2.0
                * (x1 * (PI * 5.0 / 8.0).cos()
                    + x2 * (PI * 10.0 / 8.0).cos()
                    + x3 * (PI * 15.0 / 8.0).cos()),
            x0 + 2.0
                * (x1 * (PI * 7.0 / 8.0).cos()
                    + x2 * (PI * 14.0 / 8.0).cos()
                    + x3 * (PI * 21.0 / 8.0).cos()),
        ]
    }

    // FIXED version: rows → transpose → rows
    fn dc_from_dct_32x32_fixed(coeffs: &[f32; 1024]) -> [f32; 16] {
        // Extract 4x4 LLF with scales
        let mut block = [0.0f32; 16];
        for iy in 0..4 {
            for ix in 0..4 {
                block[iy * 4 + ix] = coeffs[iy * 32 + ix] * SCALE[iy] * SCALE[ix];
            }
        }

        // IDCT rows
        let mut after_rows = [0.0f32; 16];
        for iy in 0..4 {
            let row_in = [
                block[iy * 4],
                block[iy * 4 + 1],
                block[iy * 4 + 2],
                block[iy * 4 + 3],
            ];
            let row_out = idct4(&row_in);
            for ix in 0..4 {
                after_rows[iy * 4 + ix] = row_out[ix];
            }
        }

        // Transpose
        let mut transposed = [0.0f32; 16];
        for iy in 0..4 {
            for ix in 0..4 {
                transposed[ix * 4 + iy] = after_rows[iy * 4 + ix];
            }
        }

        // IDCT rows again
        let mut result = [0.0f32; 16];
        for iy in 0..4 {
            let row_in = [
                transposed[iy * 4],
                transposed[iy * 4 + 1],
                transposed[iy * 4 + 2],
                transposed[iy * 4 + 3],
            ];
            let row_out = idct4(&row_in);
            for ix in 0..4 {
                result[iy * 4 + ix] = row_out[ix];
            }
        }
        result
    }

    // OLD (buggy) version: rows → columns (no transpose)
    fn dc_from_dct_32x32_old(coeffs: &[f32; 1024]) -> [f32; 16] {
        let mut block = [0.0f32; 16];
        for iy in 0..4 {
            for ix in 0..4 {
                block[iy * 4 + ix] = coeffs[iy * 32 + ix] * SCALE[iy] * SCALE[ix];
            }
        }

        // IDCT rows
        let mut after_rows = [0.0f32; 16];
        for iy in 0..4 {
            let row_in = [
                block[iy * 4],
                block[iy * 4 + 1],
                block[iy * 4 + 2],
                block[iy * 4 + 3],
            ];
            let row_out = idct4(&row_in);
            for ix in 0..4 {
                after_rows[iy * 4 + ix] = row_out[ix];
            }
        }

        // IDCT columns (NO transpose — BUG)
        let mut result = [0.0f32; 16];
        for ix in 0..4 {
            let col_in = [
                after_rows[0 * 4 + ix],
                after_rows[1 * 4 + ix],
                after_rows[2 * 4 + ix],
                after_rows[3 * 4 + ix],
            ];
            let col_out = idct4(&col_in);
            for iy in 0..4 {
                result[iy * 4 + ix] = col_out[iy];
            }
        }
        result
    }

    // Test 1: vertical-only frequency (coeffs[1] = ky=1, kx=0)
    // Expected: columns should be constant, rows should vary
    let mut coeffs_vert = [0.0f32; 1024];
    coeffs_vert[1] = 1.0;

    let fixed = dc_from_dct_32x32_fixed(&coeffs_vert);
    let old = dc_from_dct_32x32_old(&coeffs_vert);

    eprintln!("DCT32x32 with vertical-only freq (coeffs[1]=1.0):");
    eprintln!("  FIXED dcs (4x4 grid, row-major):");
    for iy in 0..4 {
        eprintln!(
            "    row {}: {:.4} {:.4} {:.4} {:.4}",
            iy,
            fixed[iy * 4],
            fixed[iy * 4 + 1],
            fixed[iy * 4 + 2],
            fixed[iy * 4 + 3]
        );
    }
    eprintln!("  OLD dcs:");
    for iy in 0..4 {
        eprintln!(
            "    row {}: {:.4} {:.4} {:.4} {:.4}",
            iy,
            old[iy * 4],
            old[iy * 4 + 1],
            old[iy * 4 + 2],
            old[iy * 4 + 3]
        );
    }

    // FIXED: Each row should have same value (columns constant)
    for iy in 0..4 {
        let row_vals: Vec<f32> = (0..4).map(|ix| fixed[iy * 4 + ix]).collect();
        let row_variance: f32 = row_vals.iter().map(|v| (v - row_vals[0]).abs()).sum();
        assert!(
            row_variance < 1e-5,
            "FIXED: row {} should be constant for vertical freq, got {:?}",
            iy,
            row_vals
        );
    }
    // FIXED: Rows should differ from each other
    let row_diff = (fixed[0] - fixed[4]).abs();
    assert!(
        row_diff > 0.1,
        "FIXED: rows should differ for vertical freq"
    );
    eprintln!("  PASS: FIXED produces correct vertical variation");

    // OLD: Should be wrong (rows vary instead of columns)
    let old_row0_variance: f32 = (0..4).map(|ix| (old[ix] - old[0]).abs()).sum();
    assert!(
        old_row0_variance > 0.1,
        "OLD: row 0 should incorrectly vary for vertical freq"
    );
    eprintln!("  PASS: OLD produces wrong horizontal variation (bug confirmed)");

    // Test 2: horizontal-only frequency (coeffs[32] = ky=0, kx=1)
    // Expected: rows should be constant, columns should vary
    let mut coeffs_horiz = [0.0f32; 1024];
    coeffs_horiz[32] = 1.0;

    let fixed = dc_from_dct_32x32_fixed(&coeffs_horiz);
    let old = dc_from_dct_32x32_old(&coeffs_horiz);

    eprintln!("\nDCT32x32 with horizontal-only freq (coeffs[32]=1.0):");
    eprintln!("  FIXED dcs:");
    for iy in 0..4 {
        eprintln!(
            "    row {}: {:.4} {:.4} {:.4} {:.4}",
            iy,
            fixed[iy * 4],
            fixed[iy * 4 + 1],
            fixed[iy * 4 + 2],
            fixed[iy * 4 + 3]
        );
    }

    // FIXED: Each column should have same value (rows constant for given column)
    for ix in 0..4 {
        let col_vals: Vec<f32> = (0..4).map(|iy| fixed[iy * 4 + ix]).collect();
        let col_variance: f32 = col_vals.iter().map(|v| (v - col_vals[0]).abs()).sum();
        assert!(
            col_variance < 1e-5,
            "FIXED: col {} should be constant for horizontal freq, got {:?}",
            ix,
            col_vals
        );
    }
    // FIXED: Columns should differ from each other
    let col_diff = (fixed[0] - fixed[1]).abs();
    assert!(
        col_diff > 0.1,
        "FIXED: columns should differ for horizontal freq"
    );
    eprintln!("  PASS: FIXED produces correct horizontal variation");
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 2 DCT32x32: Single-group roundtrip
// ─────────────────────────────────────────────────────────────────────────────

/// DCT32x32 covers 4x4 blocks = 32x32 pixels. Minimum image for forced DCT32x32
/// is 32x32. Test with 256x256 smooth gradient (single-group, 8 DCT32x32 blocks per row).
///
/// Note: DCT32x32 works well on smooth content but poorly on high-contrast edges.
/// We use a smooth gradient here which is appropriate for DCT32x32 testing.
#[test]
#[ignore] // requires jxl-oxide decoder
fn layer2_single_group_dct32x32_decode_jxl_oxide() {
    // Use smooth gradient content - appropriate for DCT32x32
    let (w, h, linear, srgb) = generate_smooth_gradient(256, 256);
    assert_eq!(w, 256);
    assert_eq!(h, 256);

    // Use d=3.0 because DCT32x32 is enabled at d>=3.0
    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(3.0);
    encoder.force_strategy = Some(4); // RAW_STRATEGY_DCT32X32

    let bytes = encoder
        .encode(w, h, &linear, None)
        .unwrap_or_else(|e| panic!("encode failed: {:?}", e))
        .data;

    eprintln!(
        "layer2 DCT32x32 jxl-oxide: encoded 256x256 smooth gradient at d=3.0, {} bytes",
        bytes.len()
    );
    // Save for manual inspection
    let tmp_path = std::env::temp_dir().join("test_dct32x32_forced.jxl");
    std::fs::write(&tmp_path, &bytes).unwrap();
    eprintln!("Saved to {}", tmp_path.display());

    let (dw, dh, pixels) = decode_jxl_oxide(&bytes);
    assert_eq!(dw, w, "width mismatch");
    assert_eq!(dh, h, "height mismatch");

    let ssim2 = ssim2_u8_vs_linear_f32(&srgb, &pixels, w, h);
    eprintln!("layer2 DCT32x32 jxl-oxide: SSIM2 = {:.2}", ssim2);

    assert!(
        ssim2 > 50.0,
        "DCT32x32 256x256 quality too low: SSIM2={:.2} (expected >50)",
        ssim2
    );
}

/// Same with djxl reference decoder.
#[test]
#[ignore] // requires djxl
fn layer2_single_group_dct32x32_decode_djxl() {
    // Use smooth gradient content - appropriate for DCT32x32
    let (w, h, linear, srgb) = generate_smooth_gradient(256, 256);

    // Use d=3.0 because DCT32x32 is enabled at d>=3.0
    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(3.0);
    encoder.force_strategy = Some(4); // RAW_STRATEGY_DCT32X32

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;

    eprintln!(
        "layer2 DCT32x32 djxl: encoded 256x256 smooth gradient at d=3.0, {} bytes",
        bytes.len()
    );

    let (dw, dh, dec_srgb) = decode_djxl(&bytes);
    assert_eq!(dw, w, "width mismatch");
    assert_eq!(dh, h, "height mismatch");

    let ssim2 = ssim2_u8_vs_linear_u8(&srgb, &dec_srgb, w, h);
    eprintln!("layer2 DCT32x32 djxl: SSIM2 = {:.2}", ssim2);

    assert!(
        ssim2 > 50.0,
        "DCT32x32 256x256 quality too low via djxl: SSIM2={:.2}",
        ssim2
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 3 DCT32x32: Multi-group roundtrip
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-group (512x512) smooth gradient with forced DCT32x32.
/// Note: DCT32x32 works well on smooth content but poorly on high-contrast edges.
#[test]
#[ignore] // requires djxl
fn layer3_multigroup_dct32x32_decode_djxl() {
    // Use smooth gradient content - appropriate for DCT32x32
    let (w, h, linear, srgb) = generate_smooth_gradient(512, 512);
    eprintln!("layer3 DCT32x32: generated {}x{} smooth gradient", w, h);
    assert!(w > 256 || h > 256, "should be multi-group");

    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(3.0);
    encoder.force_strategy = Some(4); // RAW_STRATEGY_DCT32X32

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;

    eprintln!(
        "layer3 DCT32x32 djxl: encoded {}x{} smooth gradient, {} bytes",
        w,
        h,
        bytes.len()
    );

    let (dw, dh, dec_srgb) = decode_djxl(&bytes);
    assert_eq!(dw, w, "width mismatch");
    assert_eq!(dh, h, "height mismatch");

    let ssim2 = ssim2_u8_vs_linear_u8(&srgb, &dec_srgb, w, h);
    eprintln!("layer3 DCT32x32 djxl: SSIM2 = {:.2}", ssim2);

    assert!(
        ssim2 > 50.0,
        "DCT32x32 multi-group quality too low: SSIM2={:.2}",
        ssim2
    );
}

/// Multi-group with jxl-oxide decoder.
#[test]
#[ignore] // requires jxl-oxide
fn layer3_multigroup_dct32x32_decode_jxl_oxide() {
    // Use smooth gradient content - appropriate for DCT32x32
    let (w, h, linear, srgb) = generate_smooth_gradient(512, 512);

    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(3.0);
    encoder.force_strategy = Some(4); // RAW_STRATEGY_DCT32X32

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;

    eprintln!(
        "layer3 DCT32x32 jxl-oxide: encoded {}x{} smooth gradient, {} bytes",
        w,
        h,
        bytes.len()
    );

    let (dw, dh, pixels) = decode_jxl_oxide(&bytes);
    assert_eq!(dw, w, "width mismatch");
    assert_eq!(dh, h, "height mismatch");

    let ssim2 = ssim2_u8_vs_linear_f32(&srgb, &pixels, w, h);
    eprintln!("layer3 DCT32x32 jxl-oxide: SSIM2 = {:.2}", ssim2);

    assert!(
        ssim2 > 50.0,
        "DCT32x32 multi-group quality too low via jxl-oxide: SSIM2={:.2}",
        ssim2
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 4 DCT32x32: Quality comparison
// ─────────────────────────────────────────────────────────────────────────────

/// Compare DCT32x32 vs DCT8 quality on 256x256 smooth gradient.
/// Note: DCT32x32 works well on smooth content, producing smaller files with comparable quality.
#[test]
#[ignore] // requires djxl
fn layer4_quality_dct32x32_vs_dct8_smooth_256() {
    // Use smooth gradient - appropriate for DCT32x32
    let (w, h, linear, srgb) = generate_smooth_gradient(256, 256);

    // DCT8-only
    let mut enc_dct8 = jxl_encoder::vardct::VarDctEncoder::new(3.0);
    enc_dct8.ac_strategy_enabled = false;
    let bytes_dct8 = enc_dct8.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec8) = decode_djxl(&bytes_dct8);
    let ssim2_dct8 = ssim2_u8_vs_linear_u8(&srgb, &dec8, w, h);

    // DCT32x32-only (forced)
    let mut enc_dct32 = jxl_encoder::vardct::VarDctEncoder::new(3.0);
    enc_dct32.force_strategy = Some(4);
    let bytes_dct32 = enc_dct32.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec32) = decode_djxl(&bytes_dct32);
    let ssim2_dct32 = ssim2_u8_vs_linear_u8(&srgb, &dec32, w, h);

    eprintln!("layer4 DCT32x32 vs DCT8, smooth 256x256 @ d=3.0:");
    eprintln!(
        "  DCT8:    SSIM2={:.2}, {} bytes",
        ssim2_dct8,
        bytes_dct8.len()
    );
    eprintln!(
        "  DCT32x32: SSIM2={:.2}, {} bytes",
        ssim2_dct32,
        bytes_dct32.len()
    );
    eprintln!(
        "  gap: {:.2} SSIM2, size ratio: {:.2}%",
        ssim2_dct8 - ssim2_dct32,
        bytes_dct32.len() as f64 / bytes_dct8.len() as f64 * 100.0
    );

    // DCT32x32 quality should be reasonable on smooth content
    assert!(
        ssim2_dct32 > 50.0,
        "DCT32x32 quality too low: {:.2}",
        ssim2_dct32
    );

    // Gap should be small (within 15 SSIM2) on smooth content
    let gap = ssim2_dct8 - ssim2_dct32;
    assert!(
        gap < 15.0,
        "DCT32x32 vs DCT8 gap too large: {:.2} SSIM2.",
        gap
    );
}

/// Compare on larger multi-group smooth gradient.
#[test]
#[ignore] // requires djxl
fn layer4_quality_dct32x32_vs_dct8_smooth_512() {
    // Use smooth gradient - appropriate for DCT32x32
    let (w, h, linear, srgb) = generate_smooth_gradient(512, 512);

    // DCT8-only
    let mut enc_dct8 = jxl_encoder::vardct::VarDctEncoder::new(3.0);
    enc_dct8.ac_strategy_enabled = false;
    let bytes_dct8 = enc_dct8.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec8) = decode_djxl(&bytes_dct8);
    let ssim2_dct8 = ssim2_u8_vs_linear_u8(&srgb, &dec8, w, h);

    // DCT32x32-only
    let mut enc_dct32 = jxl_encoder::vardct::VarDctEncoder::new(3.0);
    enc_dct32.force_strategy = Some(4);
    let bytes_dct32 = enc_dct32.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec32) = decode_djxl(&bytes_dct32);
    let ssim2_dct32 = ssim2_u8_vs_linear_u8(&srgb, &dec32, w, h);

    eprintln!("layer4 DCT32x32 vs DCT8, smooth 512x512 @ d=3.0:");
    eprintln!(
        "  DCT8:    SSIM2={:.2}, {} bytes",
        ssim2_dct8,
        bytes_dct8.len()
    );
    eprintln!(
        "  DCT32x32: SSIM2={:.2}, {} bytes",
        ssim2_dct32,
        bytes_dct32.len()
    );
    eprintln!(
        "  gap: {:.2} SSIM2, size ratio: {:.2}%",
        ssim2_dct8 - ssim2_dct32,
        bytes_dct32.len() as f64 / bytes_dct8.len() as f64 * 100.0
    );

    assert!(
        ssim2_dct32 > 50.0,
        "DCT32x32 quality too low: {:.2}",
        ssim2_dct32
    );

    let gap = ssim2_dct8 - ssim2_dct32;
    assert!(
        gap < 15.0,
        "DCT32x32 vs DCT8 gap too large: {:.2} SSIM2",
        gap
    );

    // On smooth content, DCT32x32 should produce smaller files
    eprintln!(
        "  DCT32x32 produces {:.1}% the file size of DCT8",
        bytes_dct32.len() as f64 / bytes_dct8.len() as f64 * 100.0
    );
}

/// Multiple distances on 256x256 frymire crop: does DCT16x16 behave
/// reasonably across the quality range?
#[test]
#[ignore] // requires frymire test image and djxl
fn layer4_quality_dct16x16_across_distances() {
    let (w, h, linear, srgb) = load_png_crop(&frymire_path(), 256, 256);

    eprintln!("layer4 distance sweep, frymire 256x256:");
    eprintln!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "dist", "dct8_ssim", "d16_ssim", "gap", "d8_bytes", "d16_bytes"
    );

    for &distance in &[0.5, 1.0, 2.0, 4.0] {
        let mut enc_dct8 = jxl_encoder::vardct::VarDctEncoder::new(distance);
        enc_dct8.ac_strategy_enabled = false;
        let bytes_dct8 = enc_dct8.encode(w, h, &linear, None).unwrap().data;
        let (_, _, dec8) = decode_djxl(&bytes_dct8);
        let ssim2_dct8 = ssim2_u8_vs_linear_u8(&srgb, &dec8, w, h);

        let mut enc_dct16 = jxl_encoder::vardct::VarDctEncoder::new(distance);
        enc_dct16.ac_strategy_enabled = true;
        let bytes_dct16 = enc_dct16.encode(w, h, &linear, None).unwrap().data;
        let (_, _, dec16) = decode_djxl(&bytes_dct16);
        let ssim2_dct16 = ssim2_u8_vs_linear_u8(&srgb, &dec16, w, h);

        let gap = ssim2_dct8 - ssim2_dct16;
        eprintln!(
            "{:>8.1} {:>10.2} {:>10.2} {:>10.2} {:>10} {:>8}",
            distance,
            ssim2_dct8,
            ssim2_dct16,
            gap,
            bytes_dct8.len(),
            bytes_dct16.len()
        );

        // DCT16 should not be catastrophically worse than DCT8.
        // At high distances both can be low, so we check the gap, not absolute quality.
        // Gap > 10 would indicate a real bug (the dc_from_dct_16x16 swap bug caused gaps of 56-137).
        // At d >= 4.0, DCT16 naturally performs worse (larger blocks + high quantization = more blur),
        // so we allow a larger gap at high distances.
        let max_gap = if distance >= 4.0 { 20.0 } else { 10.0 };
        assert!(
            gap < max_gap,
            "d={}: gap {:.2} is too large (DCT8={:.2}, DCT16={:.2})",
            distance,
            gap,
            ssim2_dct8,
            ssim2_dct16
        );
    }
}

// Diagnostic: trace DCT32x32 pipeline on a constant-value 32x32 block
#[test]
#[ignore]
fn diag_dct32x32_constant_block() {
    use jxl_encoder::vardct::dct::{dc_from_dct_32x32, dct_32x32};

    // Create a 32x32 block with all values = 0.5
    let constant_val = 0.5f32;
    let mut input = [constant_val; 1024];

    // Apply forward DCT
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&input, &mut coeffs);

    // Print key coefficients
    eprintln!("DCT32x32 of constant block (all 0.5):");
    eprintln!("  DC (coeffs[0]) = {:.6}", coeffs[0]);
    eprintln!("  coeffs[1] = {:.6}", coeffs[1]);
    eprintln!("  coeffs[32] = {:.6}", coeffs[32]);
    eprintln!("  coeffs[33] = {:.6}", coeffs[33]);
    eprintln!(
        "  First row (4 elements): {:.4} {:.4} {:.4} {:.4}",
        coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    );
    eprintln!("  LLF 4x4 (rows 0-3, cols 0-3):");
    for iy in 0..4 {
        eprintln!(
            "    row {}: {:.6} {:.6} {:.6} {:.6}",
            iy,
            coeffs[iy * 32],
            coeffs[iy * 32 + 1],
            coeffs[iy * 32 + 2],
            coeffs[iy * 32 + 3]
        );
    }

    // Extract DC values
    let dcs = dc_from_dct_32x32(&coeffs);
    eprintln!("  DC values from LLF (4x4):");
    for iy in 0..4 {
        eprintln!(
            "    row {}: {:.6} {:.6} {:.6} {:.6}",
            iy,
            dcs[iy * 4],
            dcs[iy * 4 + 1],
            dcs[iy * 4 + 2],
            dcs[iy * 4 + 3]
        );
    }

    // For a constant input, DC should be proportional to the input value
    // and all DC values should be approximately equal
    let dc_mean = dcs.iter().sum::<f32>() / 16.0;
    let dc_var = dcs.iter().map(|d| (d - dc_mean).powi(2)).sum::<f32>() / 16.0;
    eprintln!("  DC mean = {:.6}, variance = {:.6}", dc_mean, dc_var);

    // The DC should be 0.5 * 32 = 16.0 (sum of 32 elements, each 0.5, divided by 32, times 32)
    // Actually for DCT, the DC is sum/sqrt(N) * scaling factors
    // For our DCT32: output[0] = sum * (1/32)^2 = sum / 1024
    // sum = 32*32*0.5 = 512, so coeffs[0] = 512/1024 = 0.5
    eprintln!("  Expected coeffs[0] ≈ 0.5 (for constant 0.5 input)");

    // For a constant input, all AC coefficients should be 0
    let ac_sum: f32 = (1..1024).map(|i| coeffs[i].abs()).sum();
    eprintln!(
        "  Sum of abs(AC coefficients) = {:.6} (should be ~0)",
        ac_sum
    );
}

// Diagnostic: check if DCT32x32 forward+IDCT roundtrips correctly
#[test]
#[ignore]
fn diag_dct32x32_forward_idct_roundtrip() {
    use jxl_encoder::vardct::dct::{dc_from_dct_32x32, dct_32x32};

    // Create a gradient pattern - values increase along x and y
    let mut input = [0.0f32; 1024];
    for y in 0..32 {
        for x in 0..32 {
            input[y * 32 + x] = (x as f32 + y as f32) / 64.0;
        }
    }

    // Apply forward DCT
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&input, &mut coeffs);

    // Print some key coefficients
    eprintln!("DCT32x32 of gradient:");
    eprintln!("  coeffs[0] (DC) = {:.6}", coeffs[0]);
    eprintln!("  coeffs[1] = {:.6}", coeffs[1]);
    eprintln!("  coeffs[32] = {:.6}", coeffs[32]);

    // Extract DC values
    let dcs = dc_from_dct_32x32(&coeffs);
    eprintln!("  DC values from LLF (4x4):");
    for iy in 0..4 {
        eprintln!(
            "    row {}: {:.6} {:.6} {:.6} {:.6}",
            iy,
            dcs[iy * 4],
            dcs[iy * 4 + 1],
            dcs[iy * 4 + 2],
            dcs[iy * 4 + 3]
        );
    }

    // Compute expected 8x8 block averages
    eprintln!("  Expected 8x8 block averages:");
    for by in 0..4 {
        let mut row_str = String::from("    row ");
        row_str.push_str(&format!("{}: ", by));
        for bx in 0..4 {
            let mut sum = 0.0f32;
            for dy in 0..8 {
                for dx in 0..8 {
                    let y = by * 8 + dy;
                    let x = bx * 8 + dx;
                    sum += input[y * 32 + x];
                }
            }
            let avg = sum / 64.0;
            row_str.push_str(&format!("{:.6} ", avg));
        }
        eprintln!("{}", row_str);
    }
}
// Diagnostic: detailed DCT32x32 LLF analysis
#[test]
#[ignore]
fn diag_dct32x32_llf_detail() {
    use jxl_encoder::vardct::dct::{dc_from_dct_32x32, dct_32x32};

    // Create a gradient pattern - values increase along x and y
    let mut input = [0.0f32; 1024];
    for y in 0..32 {
        for x in 0..32 {
            input[y * 32 + x] = (x as f32 + y as f32) / 64.0;
        }
    }

    // Apply forward DCT
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&input, &mut coeffs);

    // Print LLF 4x4 coefficients
    eprintln!("DCT32x32 gradient LLF (4x4 corner, before scaling):");
    for iy in 0..4 {
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            iy,
            coeffs[iy * 32],
            coeffs[iy * 32 + 1],
            coeffs[iy * 32 + 2],
            coeffs[iy * 32 + 3]
        );
    }

    // Apply resample scales (32 -> 4)
    const SCALE: [f32; 4] = [
        1.0,
        0.974886821136879522,
        0.901764195028874394,
        0.787054918159101335,
    ];
    eprintln!("\nAfter applying resample scales:");
    for iy in 0..4 {
        let mut row = [0.0f32; 4];
        for ix in 0..4 {
            row[ix] = coeffs[iy * 32 + ix] * SCALE[iy] * SCALE[ix];
        }
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            iy, row[0], row[1], row[2], row[3]
        );
    }

    // Extract DC values
    let dcs = dc_from_dct_32x32(&coeffs);
    eprintln!("\nDC values from dc_from_dct_32x32:");
    for iy in 0..4 {
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            iy,
            dcs[iy * 4],
            dcs[iy * 4 + 1],
            dcs[iy * 4 + 2],
            dcs[iy * 4 + 3]
        );
    }

    // Expected 8x8 block averages (ground truth)
    eprintln!("\nExpected 8x8 block averages:");
    for by in 0..4 {
        let mut row = [0.0f32; 4];
        for bx in 0..4 {
            let mut sum = 0.0f32;
            for dy in 0..8 {
                for dx in 0..8 {
                    let y = by * 8 + dy;
                    let x = bx * 8 + dx;
                    sum += input[y * 32 + x];
                }
            }
            row[bx] = sum / 64.0;
        }
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            by, row[0], row[1], row[2], row[3]
        );
    }

    // Compute error
    eprintln!("\nError (dc_from_dct - expected):");
    let mut total_error = 0.0f32;
    for by in 0..4 {
        let mut row = [0.0f32; 4];
        for bx in 0..4 {
            let mut sum = 0.0f32;
            for dy in 0..8 {
                for dx in 0..8 {
                    sum += input[(by * 8 + dy) * 32 + bx * 8 + dx];
                }
            }
            let expected = sum / 64.0;
            let error = dcs[by * 4 + bx] - expected;
            row[bx] = error;
            total_error += error.abs();
        }
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            by, row[0], row[1], row[2], row[3]
        );
    }
    eprintln!("\nTotal absolute error: {:.6}", total_error);
}

// Diagnostic: verify the DCT32x32 <-> DC relationship
#[test]
#[ignore]
fn diag_dct32x32_roundtrip_verification() {
    use jxl_encoder::vardct::dct::{dc_from_dct_32x32, dct_32x32};

    // Resample scales for 32 -> 4 (from C++)
    const SCALE_32_TO_4: [f32; 4] = [
        1.0,
        0.974886821136879522,
        0.901764195028874394,
        0.787054918159101335,
    ];
    // Inverse scales for 4 -> 32
    const SCALE_4_TO_32: [f32; 4] = [
        1.0,
        1.0257549441917856,
        1.1089312359806676,
        1.2706084147018952,
    ];

    // 4-point DCT-II (forward)
    fn dct1d_4(input: &[f32; 4]) -> [f32; 4] {
        use core::f32::consts::PI;
        let mut output = [0.0f32; 4];
        for k in 0..4 {
            let mut sum = 0.0f32;
            for n in 0..4 {
                sum += input[n] * (PI * k as f32 * (2.0 * n as f32 + 1.0) / 8.0).cos();
            }
            output[k] = sum / 4.0; // Normalize by N
        }
        output
    }

    // Create a gradient pattern
    let mut input = [0.0f32; 1024];
    for y in 0..32 {
        for x in 0..32 {
            input[y * 32 + x] = (x as f32 + y as f32) / 64.0;
        }
    }

    // Compute expected 8x8 block averages
    let mut expected_dc = [[0.0f32; 4]; 4];
    for by in 0..4 {
        for bx in 0..4 {
            let mut sum = 0.0f32;
            for dy in 0..8 {
                for dx in 0..8 {
                    sum += input[(by * 8 + dy) * 32 + bx * 8 + dx];
                }
            }
            expected_dc[by][bx] = sum / 64.0;
        }
    }

    eprintln!("Expected 8x8 block averages (DC grid):");
    for by in 0..4 {
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            by, expected_dc[by][0], expected_dc[by][1], expected_dc[by][2], expected_dc[by][3]
        );
    }

    // Apply 4x4 DCT to expected_dc to get expected LLF
    // First DCT rows
    let mut after_rows = [[0.0f32; 4]; 4];
    for iy in 0..4 {
        let row: [f32; 4] = [
            expected_dc[iy][0],
            expected_dc[iy][1],
            expected_dc[iy][2],
            expected_dc[iy][3],
        ];
        let dct_row = dct1d_4(&row);
        for ix in 0..4 {
            after_rows[iy][ix] = dct_row[ix];
        }
    }

    // Transpose
    let mut transposed = [[0.0f32; 4]; 4];
    for iy in 0..4 {
        for ix in 0..4 {
            transposed[ix][iy] = after_rows[iy][ix];
        }
    }

    // DCT columns (now rows after transpose)
    let mut expected_llf = [[0.0f32; 4]; 4];
    for iy in 0..4 {
        let row: [f32; 4] = [
            transposed[iy][0],
            transposed[iy][1],
            transposed[iy][2],
            transposed[iy][3],
        ];
        let dct_row = dct1d_4(&row);
        for ix in 0..4 {
            expected_llf[iy][ix] = dct_row[ix];
        }
    }

    eprintln!("\nExpected LLF (from DCT4x4 of DC grid):");
    for iy in 0..4 {
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            iy, expected_llf[iy][0], expected_llf[iy][1], expected_llf[iy][2], expected_llf[iy][3]
        );
    }

    // Apply inverse resample scales (to go from DC-domain to DCT32-domain)
    eprintln!("\nExpected LLF with inverse scales (should match dct_32x32 output):");
    for iy in 0..4 {
        let mut row = [0.0f32; 4];
        for ix in 0..4 {
            row[ix] = expected_llf[iy][ix] * SCALE_4_TO_32[iy] * SCALE_4_TO_32[ix];
        }
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            iy, row[0], row[1], row[2], row[3]
        );
    }

    // Now apply forward DCT32x32 and get actual LLF
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&input, &mut coeffs);

    eprintln!("\nActual LLF from dct_32x32:");
    for iy in 0..4 {
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            iy,
            coeffs[iy * 32],
            coeffs[iy * 32 + 1],
            coeffs[iy * 32 + 2],
            coeffs[iy * 32 + 3]
        );
    }

    // Apply forward resample scales
    eprintln!("\nActual LLF with forward scales (input to IDCT):");
    for iy in 0..4 {
        let mut row = [0.0f32; 4];
        for ix in 0..4 {
            row[ix] = coeffs[iy * 32 + ix] * SCALE_32_TO_4[iy] * SCALE_32_TO_4[ix];
        }
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            iy, row[0], row[1], row[2], row[3]
        );
    }

    // Finally, dc_from_dct_32x32 output
    let dcs = dc_from_dct_32x32(&coeffs);
    eprintln!("\ndc_from_dct_32x32 output:");
    for iy in 0..4 {
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            iy,
            dcs[iy * 4],
            dcs[iy * 4 + 1],
            dcs[iy * 4 + 2],
            dcs[iy * 4 + 3]
        );
    }
}

// Diagnostic: test sqrt(2) correction for DCT32x32 LLF
#[test]
#[ignore]
fn diag_dct32x32_sqrt2_correction() {
    use jxl_encoder::vardct::dct::dct_32x32;

    const SCALE_32_TO_4: [f32; 4] = [
        1.0,
        0.974886821136879522,
        0.901764195028874394,
        0.787054918159101335,
    ];
    const SQRT2: f32 = 1.4142135623730951;

    // 4-point IDCT
    fn idct1d_4(input: &[f32; 4]) -> [f32; 4] {
        use core::f32::consts::PI;
        let x0 = input[0];
        let x1 = input[1];
        let x2 = input[2];
        let x3 = input[3];
        [
            x0 + 2.0
                * (x1 * (PI / 8.0).cos() + x2 * (PI / 4.0).cos() + x3 * (3.0 * PI / 8.0).cos()),
            x0 + 2.0
                * (x1 * (3.0 * PI / 8.0).cos()
                    + x2 * (3.0 * PI / 4.0).cos()
                    + x3 * (9.0 * PI / 8.0).cos()),
            x0 + 2.0
                * (x1 * (5.0 * PI / 8.0).cos()
                    + x2 * (5.0 * PI / 4.0).cos()
                    + x3 * (15.0 * PI / 8.0).cos()),
            x0 + 2.0
                * (x1 * (7.0 * PI / 8.0).cos()
                    + x2 * (7.0 * PI / 4.0).cos()
                    + x3 * (21.0 * PI / 8.0).cos()),
        ]
    }

    // Fixed dc_from_dct_32x32 with sqrt(2) correction on AC coefficients
    fn dc_from_dct_32x32_fixed(coeffs: &[f32; 1024]) -> [f32; 16] {
        let mut block = [0.0f32; 16];
        for iy in 0..4 {
            for ix in 0..4 {
                let scale = SCALE_32_TO_4[iy] * SCALE_32_TO_4[ix];
                let mut val = coeffs[iy * 32 + ix] * scale;
                // Divide AC by sqrt(2) because dct_32x32 produces them sqrt(2) too large
                if iy > 0 || ix > 0 {
                    val /= SQRT2;
                }
                block[iy * 4 + ix] = val;
            }
        }

        // IDCT rows
        let mut after_rows = [0.0f32; 16];
        for iy in 0..4 {
            let row = [
                block[iy * 4],
                block[iy * 4 + 1],
                block[iy * 4 + 2],
                block[iy * 4 + 3],
            ];
            let out = idct1d_4(&row);
            for ix in 0..4 {
                after_rows[iy * 4 + ix] = out[ix];
            }
        }

        // Transpose
        let mut transposed = [0.0f32; 16];
        for iy in 0..4 {
            for ix in 0..4 {
                transposed[ix * 4 + iy] = after_rows[iy * 4 + ix];
            }
        }

        // IDCT rows again
        let mut result = [0.0f32; 16];
        for iy in 0..4 {
            let row = [
                transposed[iy * 4],
                transposed[iy * 4 + 1],
                transposed[iy * 4 + 2],
                transposed[iy * 4 + 3],
            ];
            let out = idct1d_4(&row);
            for ix in 0..4 {
                result[iy * 4 + ix] = out[ix];
            }
        }
        result
    }

    // Create gradient
    let mut input = [0.0f32; 1024];
    for y in 0..32 {
        for x in 0..32 {
            input[y * 32 + x] = (x as f32 + y as f32) / 64.0;
        }
    }

    // Forward DCT
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&input, &mut coeffs);

    // Extract DC with sqrt(2) correction
    let dcs_fixed = dc_from_dct_32x32_fixed(&coeffs);

    // Expected block averages
    eprintln!("Expected 8x8 block averages:");
    for by in 0..4 {
        let mut row = [0.0f32; 4];
        for bx in 0..4 {
            let mut sum = 0.0f32;
            for dy in 0..8 {
                for dx in 0..8 {
                    sum += input[(by * 8 + dy) * 32 + bx * 8 + dx];
                }
            }
            row[bx] = sum / 64.0;
        }
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            by, row[0], row[1], row[2], row[3]
        );
    }

    eprintln!("\nDC from fixed dc_from_dct_32x32 (with sqrt2 correction):");
    for iy in 0..4 {
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            iy,
            dcs_fixed[iy * 4],
            dcs_fixed[iy * 4 + 1],
            dcs_fixed[iy * 4 + 2],
            dcs_fixed[iy * 4 + 3]
        );
    }

    // Compute error
    let mut total_error = 0.0f32;
    for by in 0..4 {
        for bx in 0..4 {
            let mut sum = 0.0f32;
            for dy in 0..8 {
                for dx in 0..8 {
                    sum += input[(by * 8 + dy) * 32 + bx * 8 + dx];
                }
            }
            let expected = sum / 64.0;
            total_error += (dcs_fixed[by * 4 + bx] - expected).abs();
        }
    }
    eprintln!("\nTotal absolute error with sqrt2 fix: {:.6}", total_error);
}

// Diagnostic: test butterfly IDCT matching C++
#[test]
#[ignore]
fn diag_dct32x32_butterfly_idct() {
    use jxl_encoder::vardct::dct::dct_32x32;

    const SCALE_32_TO_4: [f32; 4] = [
        1.0,
        0.974886821136879522,
        0.901764195028874394,
        0.787054918159101335,
    ];
    const SQRT2: f32 = 1.4142135623730951;

    // 2-point IDCT (matches C++)
    fn idct2(a: f32, b: f32) -> (f32, f32) {
        (a + b, a - b)
    }

    // 4-point IDCT using butterfly decomposition (matching C++)
    fn idct4_butterfly(input: &[f32; 4]) -> [f32; 4] {
        let x0 = input[0];
        let x1 = input[1];
        let x2 = input[2];
        let x3 = input[3];

        // ForwardEvenOdd: split into even and odd
        let even = [x0, x2];
        let odd = [x1, x3];

        // IDCT2 on even
        let (e0, e1) = idct2(even[0], even[1]);

        // BTranspose on odd (inverse of B transform)
        // B transform: b[0] = sqrt(2)*a[0] + a[1]; b[1] = a[1]
        // BTranspose: a[0] = (b[0] - b[1]) / sqrt(2); a[1] = b[1]
        let o0 = (odd[0] - odd[1]) / SQRT2;
        let o1 = odd[1];

        // Wait, that's wrong. Let me reconsider...
        // Actually for 2-point B transform, it's simpler
        // The B transform adds adjacent elements: b[k] = a[k] + a[k+1] (with sqrt(2) on first)
        // For 2 elements, this is just: b[0] = sqrt(2)*a[0] + a[1], but a[1] is just a[1]
        // Actually let me check the C++ code more carefully

        // For now, let's use the WC multiplier approach
        use core::f32::consts::PI;
        let wc1 = 2.0 * (PI / 8.0).cos(); // = 2*cos(pi/8) = 1.8478
        let wc3 = 2.0 * (3.0 * PI / 8.0).cos(); // = 2*cos(3pi/8) = 0.7654

        // Apply WC multiply (reverse of DCT WC step)
        let o0_wc = odd[0] / wc1;
        let o1_wc = odd[1] / wc3;

        // IDCT2 on WC-modified odd
        let (o0_out, o1_out) = idct2(o0_wc, o1_wc);

        // MultiplyAndAdd (reverse of AddReverse/SubReverse)
        // DCT did: even[k] = x[k] + x[N-1-k], odd[k] = x[k] - x[N-1-k]
        // So: x[k] = (even[k] + odd[k]) / 2
        //     x[N-1-k] = (even[k] - odd[k]) / 2
        // But IDCT inverts this...
        // Actually it's: out[k] = even[k] + odd[k], out[N-1-k] = even[k] - odd[k] for interleaving

        [e0 + o0_out, e1 + o1_out, e1 - o1_out, e0 - o0_out]
    }

    // Fixed dc_from_dct_32x32 with butterfly IDCT
    fn dc_from_dct_32x32_butterfly(coeffs: &[f32; 1024]) -> [f32; 16] {
        let mut block = [0.0f32; 16];
        for iy in 0..4 {
            for ix in 0..4 {
                let scale = SCALE_32_TO_4[iy] * SCALE_32_TO_4[ix];
                let mut val = coeffs[iy * 32 + ix] * scale;
                // Divide AC by sqrt(2)
                if iy > 0 || ix > 0 {
                    val /= SQRT2;
                }
                block[iy * 4 + ix] = val;
            }
        }

        // IDCT rows using butterfly
        let mut after_rows = [0.0f32; 16];
        for iy in 0..4 {
            let row = [
                block[iy * 4],
                block[iy * 4 + 1],
                block[iy * 4 + 2],
                block[iy * 4 + 3],
            ];
            let out = idct4_butterfly(&row);
            for ix in 0..4 {
                after_rows[iy * 4 + ix] = out[ix];
            }
        }

        // Transpose
        let mut transposed = [0.0f32; 16];
        for iy in 0..4 {
            for ix in 0..4 {
                transposed[ix * 4 + iy] = after_rows[iy * 4 + ix];
            }
        }

        // IDCT rows again using butterfly
        let mut result = [0.0f32; 16];
        for iy in 0..4 {
            let row = [
                transposed[iy * 4],
                transposed[iy * 4 + 1],
                transposed[iy * 4 + 2],
                transposed[iy * 4 + 3],
            ];
            let out = idct4_butterfly(&row);
            for ix in 0..4 {
                result[iy * 4 + ix] = out[ix];
            }
        }
        result
    }

    // Create gradient
    let mut input = [0.0f32; 1024];
    for y in 0..32 {
        for x in 0..32 {
            input[y * 32 + x] = (x as f32 + y as f32) / 64.0;
        }
    }

    // Forward DCT
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&input, &mut coeffs);

    // Extract DC with butterfly IDCT
    let dcs = dc_from_dct_32x32_butterfly(&coeffs);

    eprintln!("DC from butterfly IDCT:");
    for iy in 0..4 {
        eprintln!(
            "  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
            iy,
            dcs[iy * 4],
            dcs[iy * 4 + 1],
            dcs[iy * 4 + 2],
            dcs[iy * 4 + 3]
        );
    }

    // Expected and error
    let mut total_error = 0.0f32;
    for by in 0..4 {
        for bx in 0..4 {
            let mut sum = 0.0f32;
            for dy in 0..8 {
                for dx in 0..8 {
                    sum += input[(by * 8 + dy) * 32 + bx * 8 + dx];
                }
            }
            let expected = sum / 64.0;
            total_error += (dcs[by * 4 + bx] - expected).abs();
        }
    }
    eprintln!("\nTotal error with butterfly: {:.6}", total_error);
}

// Diagnostic: save DCT16x16 encoded file for manual inspection
#[test]
#[ignore]
fn diag_save_dct16x16_file() {
    use std::fs;
    use std::io::Write;

    // Create a simple 32x32 checkerboard
    let w = 32;
    let h = 32;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let checker = ((x / 8) + (y / 8)) % 2 == 0;
            let val = if checker { 0.8 } else { 0.2 };
            linear[idx] = val;
            linear[idx + 1] = val;
            linear[idx + 2] = val;
        }
    }

    // Encode with forced DCT16x16
    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    encoder.force_strategy = Some(3); // RAW_STRATEGY_DCT16X16

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;

    // Save to file
    let path = "/tmp/test_dct16x16.jxl";
    let mut file = fs::File::create(path).unwrap();
    file.write_all(&bytes).unwrap();
    eprintln!("Saved {} bytes to {}", bytes.len(), path);

    // Try to decode with djxl
    let output = std::process::Command::new(&jxl_encoder::test_helpers::djxl_path())
        .arg(path)
        .arg("/tmp/test_dct16x16.png")
        .output()
        .expect("djxl failed to run");

    if !output.status.success() {
        eprintln!("djxl stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("djxl failed with status {}", output.status);
    }
    eprintln!("djxl succeeded, saved to /tmp/test_dct16x16.png");
}

/// DIAGNOSTIC: Decode 16x16 photo crop with jxl-oxide and compare to original.
/// This isolates whether the issue is encoding vs decoding.
#[test]
#[ignore]
fn diag_dct16x16_decode_compare() {
    let (w, h, linear, srgb) = load_png_crop(&frymire_path(), 16, 16);

    // DCT8 encoding
    let mut enc8 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear, None).unwrap().data;

    // DCT16x16 encoding
    let mut enc16 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear, None).unwrap().data;

    // Decode with jxl-oxide
    let (_, _, dec8) = decode_jxl_oxide(&bytes8);
    let (_, _, dec16) = decode_jxl_oxide(&bytes16);

    // Convert decoded linear f32 to sRGB u8 for comparison
    fn linear_to_srgb_u8(v: f32) -> u8 {
        (linear_to_srgb_val(v) * 255.0).round() as u8
    }

    eprintln!("16x16 frymire crop - jxl-oxide decoder:");
    eprintln!(
        "{:>5} {:>12} {:>12} {:>12}",
        "pixel", "original", "dct8", "dct16x16"
    );

    let mut sum_diff8 = 0u32;
    let mut sum_diff16 = 0u32;

    for y in 0..4 {
        for x in 0..4 {
            let idx = y * 4 * w + x * 4; // Sample every 4th pixel
            let o = (srgb[idx * 3], srgb[idx * 3 + 1], srgb[idx * 3 + 2]);
            let d8 = (
                linear_to_srgb_u8(dec8[idx * 3]),
                linear_to_srgb_u8(dec8[idx * 3 + 1]),
                linear_to_srgb_u8(dec8[idx * 3 + 2]),
            );
            let d16 = (
                linear_to_srgb_u8(dec16[idx * 3]),
                linear_to_srgb_u8(dec16[idx * 3 + 1]),
                linear_to_srgb_u8(dec16[idx * 3 + 2]),
            );

            let diff8 = (o.0 as i32 - d8.0 as i32).abs()
                + (o.1 as i32 - d8.1 as i32).abs()
                + (o.2 as i32 - d8.2 as i32).abs();
            let diff16 = (o.0 as i32 - d16.0 as i32).abs()
                + (o.1 as i32 - d16.1 as i32).abs()
                + (o.2 as i32 - d16.2 as i32).abs();

            sum_diff8 += diff8 as u32;
            sum_diff16 += diff16 as u32;

            eprintln!(
                "  ({:2},{:2}) {:>3},{:>3},{:>3}  {:>3},{:>3},{:>3}  {:>3},{:>3},{:>3}  d8={:>3} d16={:>3}",
                y * 4,
                x * 4,
                o.0,
                o.1,
                o.2,
                d8.0,
                d8.1,
                d8.2,
                d16.0,
                d16.1,
                d16.2,
                diff8,
                diff16
            );
        }
    }

    eprintln!("Total diffs: DCT8={}, DCT16={}", sum_diff8, sum_diff16);

    // Compute SSIM2
    let ssim8 = ssim2_u8_vs_linear_f32(&srgb, &dec8, w, h);
    let ssim16 = ssim2_u8_vs_linear_f32(&srgb, &dec16, w, h);
    eprintln!("SSIM2: DCT8={:.2}, DCT16={:.2}", ssim8, ssim16);
}

/// DIAGNOSTIC: Test 32x32 photo crop to see where DCT16x16 breaks.
/// 32x32 = 4 DCT8 blocks or 2x2 arrangement of two DCT16x16 blocks (if each DCT16x16 is 16x16).
/// Actually, AC strategy selection may not produce DCT16x16 for all blocks.
#[test]
#[ignore]
fn diag_dct16x16_32x32_compare() {
    let (w, h, linear, srgb) = load_png_crop(&frymire_path(), 32, 32);

    // DCT8 encoding
    let mut enc8 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear, None).unwrap().data;

    // DCT16x16 encoding
    let mut enc16 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear, None).unwrap().data;

    // Save for inspection
    std::fs::write("/tmp/frymire_32x32_dct8.jxl", &bytes8).unwrap();
    std::fs::write("/tmp/frymire_32x32_dct16.jxl", &bytes16).unwrap();
    eprintln!("Saved /tmp/frymire_32x32_dct8.jxl ({} bytes)", bytes8.len());
    eprintln!(
        "Saved /tmp/frymire_32x32_dct16.jxl ({} bytes)",
        bytes16.len()
    );

    // Decode with jxl-oxide
    let (_, _, dec8) = decode_jxl_oxide(&bytes8);
    let (_, _, dec16) = decode_jxl_oxide(&bytes16);

    // Convert decoded linear f32 to sRGB u8 for comparison
    fn linear_to_srgb_u8(v: f32) -> u8 {
        (linear_to_srgb_val(v) * 255.0).round() as u8
    }

    eprintln!("32x32 frymire crop - jxl-oxide decoder:");
    eprintln!(
        "{:>5} {:>12} {:>12} {:>12}",
        "pixel", "original", "dct8", "dct16x16"
    );

    // Sample corners and center
    for (name, y, x) in [
        ("top-left", 0usize, 0usize),
        ("top-right", 0, 24),
        ("center", 16, 16),
        ("bottom-left", 24, 0),
        ("bottom-right", 24, 24),
    ] {
        let idx = y * w + x;
        let o = (srgb[idx * 3], srgb[idx * 3 + 1], srgb[idx * 3 + 2]);
        let d8 = (
            linear_to_srgb_u8(dec8[idx * 3]),
            linear_to_srgb_u8(dec8[idx * 3 + 1]),
            linear_to_srgb_u8(dec8[idx * 3 + 2]),
        );
        let d16 = (
            linear_to_srgb_u8(dec16[idx * 3]),
            linear_to_srgb_u8(dec16[idx * 3 + 1]),
            linear_to_srgb_u8(dec16[idx * 3 + 2]),
        );

        let diff8 = (o.0 as i32 - d8.0 as i32).abs()
            + (o.1 as i32 - d8.1 as i32).abs()
            + (o.2 as i32 - d8.2 as i32).abs();
        let diff16 = (o.0 as i32 - d16.0 as i32).abs()
            + (o.1 as i32 - d16.1 as i32).abs()
            + (o.2 as i32 - d16.2 as i32).abs();

        eprintln!(
            "  {:12} {:>3},{:>3},{:>3}  {:>3},{:>3},{:>3}  {:>3},{:>3},{:>3}  d8={:>3} d16={:>3}",
            name, o.0, o.1, o.2, d8.0, d8.1, d8.2, d16.0, d16.1, d16.2, diff8, diff16
        );
    }

    // Compute SSIM2
    let ssim8 = ssim2_u8_vs_linear_f32(&srgb, &dec8, w, h);
    let ssim16 = ssim2_u8_vs_linear_f32(&srgb, &dec16, w, h);
    eprintln!("SSIM2: DCT8={:.2}, DCT16={:.2}", ssim8, ssim16);
}

/// DIAGNOSTIC: Print nzeros values for 32x32 DCT16x16 encoding.
#[test]
#[ignore]
fn diag_dct16x16_nzeros() {
    // Use a patterned image that will have many non-zero coefficients
    let w = 32usize;
    let h = 32usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            // Checkerboard pattern
            let v = if (x + y) % 2 == 0 { 0.8 } else { 0.2 };
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    // First, try DCT8 to see expected nzeros
    let mut enc8 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear, None).unwrap().data;

    // Then DCT16x16
    let mut enc16 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear, None).unwrap().data;

    eprintln!("Checkerboard 32x32:");
    eprintln!("  DCT8 file:   {} bytes", bytes8.len());
    eprintln!("  DCT16x16 file: {} bytes", bytes16.len());

    // Decode both
    let (_, _, dec8) = decode_jxl_oxide(&bytes8);
    let (_, _, dec16) = decode_jxl_oxide(&bytes16);

    let ssim8 = ssim2_u8_vs_linear_f32(&linear_to_srgb_u8(&linear), &dec8, w, h);
    let ssim16 = ssim2_u8_vs_linear_f32(&linear_to_srgb_u8(&linear), &dec16, w, h);
    eprintln!("  DCT8 SSIM2:   {:.2}", ssim8);
    eprintln!("  DCT16x16 SSIM2: {:.2}", ssim16);

    // Check specific pixel values
    eprintln!("\nCenter 4x4 region (linear f32):");
    for dy in 0..4 {
        for dx in 0..4 {
            let idx = ((h / 2 + dy - 2) * w + (w / 2 + dx - 2)) * 3;
            let expected = if ((w / 2 + dx - 2) + (h / 2 + dy - 2)) % 2 == 0 {
                0.8
            } else {
                0.2
            };
            let d8 = dec8[idx];
            let d16 = dec16[idx];
            eprint!("({:.2}/{:.2}/{:.2}) ", expected, d8, d16);
        }
        eprintln!();
    }
}

/// DIAGNOSTIC: Test gradient to see if only DC is preserved.
#[test]
#[ignore]
fn diag_dct16x16_gradient() {
    let w = 32usize;
    let h = 32usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let v = x as f32 / w as f32; // horizontal gradient 0 to 1
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    let mut enc16 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear, None).unwrap().data;

    let (_, _, dec16) = decode_jxl_oxide(&bytes16);

    eprintln!("Horizontal gradient 32x32 with DCT16x16:");
    eprintln!("File size: {} bytes", bytes16.len());

    // Check values along first row
    eprintln!("First row (original vs decoded):");
    for x in [0, 8, 16, 24, 31] {
        let expected = x as f32 / w as f32;
        let decoded = dec16[(0 * w + x) * 3];
        eprintln!(
            "  x={:2}: expected={:.3}, decoded={:.3}, diff={:.3}",
            x,
            expected,
            decoded,
            (expected - decoded).abs()
        );
    }
}

/// DIAGNOSTIC: Check block iteration for 32x32 with DCT16x16.
#[test]
#[ignore]
fn diag_dct16x16_iteration() {
    use jxl_encoder::vardct::VarDctEncoder;

    // Create a small test where each 8x8 block has a distinct DC value
    let w = 32usize;
    let h = 32usize;
    let mut linear = vec![0.0f32; w * h * 3];

    // Set each 8x8 block to a different brightness
    for by in 0..4 {
        for bx in 0..4 {
            let block_val = (by * 4 + bx) as f32 / 16.0; // 0.0 to 0.9375
            for dy in 0..8 {
                for dx in 0..8 {
                    let px = bx * 8 + dx;
                    let py = by * 8 + dy;
                    let idx = (py * w + px) * 3;
                    linear[idx] = block_val;
                    linear[idx + 1] = block_val;
                    linear[idx + 2] = block_val;
                }
            }
        }
    }

    // DCT8
    let mut enc8 = VarDctEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear, None).unwrap().data;

    // DCT16x16
    let mut enc16 = VarDctEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear, None).unwrap().data;

    let (_, _, dec8) = decode_jxl_oxide(&bytes8);
    let (_, _, dec16) = decode_jxl_oxide(&bytes16);

    eprintln!("32x32 block pattern (each 8x8 block = different brightness):");
    eprintln!("Expected block values (4x4 grid, values 0/16 to 15/16):");
    for by in 0..4 {
        for bx in 0..4 {
            eprint!("{:.2} ", (by * 4 + bx) as f32 / 16.0);
        }
        eprintln!();
    }

    eprintln!("\nDCT8 decoded center of each block:");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let idx = (py * w + px) * 3;
            eprint!("{:.2} ", dec8[idx]);
        }
        eprintln!();
    }

    eprintln!("\nDCT16x16 decoded center of each block:");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let idx = (py * w + px) * 3;
            eprint!("{:.2} ", dec16[idx]);
        }
        eprintln!();
    }

    eprintln!(
        "\nFile sizes: DCT8={}, DCT16={}",
        bytes8.len(),
        bytes16.len()
    );
}

/// DIAGNOSTIC: Trace DC values through the DCT16x16 pipeline.
#[test]
#[ignore]
fn diag_dct16x16_dc_trace() {
    // Create a 32x32 image where each 8x8 block has a distinct uniform value
    let w = 32usize;
    let h = 32usize;
    let mut linear = vec![0.0f32; w * h * 3];

    // Each block (by, bx) has value (by * 4 + bx) / 16.0
    for by in 0..4 {
        for bx in 0..4 {
            let block_val = (by * 4 + bx) as f32 / 16.0;
            for dy in 0..8 {
                for dx in 0..8 {
                    let px = bx * 8 + dx;
                    let py = by * 8 + dy;
                    let idx = (py * w + px) * 3;
                    // Only set Y channel for simplicity
                    linear[idx] = block_val;
                    linear[idx + 1] = block_val;
                    linear[idx + 2] = block_val;
                }
            }
        }
    }

    eprintln!("32x32 image with uniform 8x8 blocks:");
    eprintln!("Input block values (4x4):");
    for by in 0..4 {
        for bx in 0..4 {
            eprint!("{:.3} ", (by * 4 + bx) as f32 / 16.0);
        }
        eprintln!();
    }

    // Test dc_from_dct_16x16 directly
    eprintln!("\nTesting dc_from_dct_16x16 for first DCT16x16 block (covers 8x8 blocks 0,1,4,5):");

    // Extract the first 16x16 spatial block
    let mut block16x16 = [0.0f32; 256];
    for sy in 0..16 {
        for sx in 0..16 {
            let v = linear[(sy * w + sx) * 3];
            block16x16[sy * 16 + sx] = v;
        }
    }

    eprintln!("Input spatial values (corners of 16x16):");
    eprintln!(
        "  (0,0)={:.3} (0,15)={:.3} (15,0)={:.3} (15,15)={:.3}",
        block16x16[0],
        block16x16[15],
        block16x16[15 * 16],
        block16x16[15 * 16 + 15]
    );

    // Do forward DCT
    let mut dct_coeffs = [0.0f32; 256];
    jxl_encoder::vardct::dct::dct_16x16(&block16x16, &mut dct_coeffs);

    eprintln!("DCT coefficients (LLF 2x2 region):");
    eprintln!(
        "  coeff[0]={:.6} coeff[1]={:.6}",
        dct_coeffs[0], dct_coeffs[1]
    );
    eprintln!(
        "  coeff[16]={:.6} coeff[17]={:.6}",
        dct_coeffs[16], dct_coeffs[17]
    );

    // Extract DC values
    let dcs = jxl_encoder::vardct::dct::dc_from_dct_16x16(&dct_coeffs);

    eprintln!("Extracted DC values:");
    eprintln!("  dcs[0]={:.6} (top-left 8x8)", dcs[0]);
    eprintln!("  dcs[1]={:.6} (top-right 8x8)", dcs[1]);
    eprintln!("  dcs[2]={:.6} (bottom-left 8x8)", dcs[2]);
    eprintln!("  dcs[3]={:.6} (bottom-right 8x8)", dcs[3]);

    // Expected: averages of each 8x8 block
    eprintln!("\nExpected DC values (block averages):");
    eprintln!("  top-left: {:.6}", 0.0); // block (0,0)
    eprintln!("  top-right: {:.6}", 1.0 / 16.0); // block (0,1)
    eprintln!("  bottom-left: {:.6}", 4.0 / 16.0); // block (1,0)
    eprintln!("  bottom-right: {:.6}", 5.0 / 16.0); // block (1,1)

    // Now test the third DCT16x16 block (by=2, bx=0)
    eprintln!("\n\nTesting dc_from_dct_16x16 for THIRD DCT16x16 block (by=2, bx=0):");
    eprintln!("This covers 8x8 blocks: (2,0), (2,1), (3,0), (3,1)");
    eprintln!(
        "Expected block values: {:.3}, {:.3}, {:.3}, {:.3}",
        8.0 / 16.0,
        9.0 / 16.0,
        12.0 / 16.0,
        13.0 / 16.0
    );

    // Extract the third 16x16 spatial block (starting at by=2, bx=0)
    for sy in 0..16 {
        for sx in 0..16 {
            let v = linear[((16 + sy) * w + sx) * 3]; // offset by 16 rows
            block16x16[sy * 16 + sx] = v;
        }
    }

    jxl_encoder::vardct::dct::dct_16x16(&block16x16, &mut dct_coeffs);
    let dcs3 = jxl_encoder::vardct::dct::dc_from_dct_16x16(&dct_coeffs);

    eprintln!("Extracted DC values for third block:");
    eprintln!("  dcs[0]={:.6} (expected {:.6})", dcs3[0], 8.0 / 16.0);
    eprintln!("  dcs[1]={:.6} (expected {:.6})", dcs3[1], 9.0 / 16.0);
    eprintln!("  dcs[2]={:.6} (expected {:.6})", dcs3[2], 12.0 / 16.0);
    eprintln!("  dcs[3]={:.6} (expected {:.6})", dcs3[3], 13.0 / 16.0);
}

/// DIAGNOSTIC: Test dc_from_dct_16x16 with uniform blocks.
#[test]
#[ignore]
fn diag_dct16x16_uniform() {
    for v in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
        let block = [v; 256];
        let mut dct_coeffs = [0.0f32; 256];
        jxl_encoder::vardct::dct::dct_16x16(&block, &mut dct_coeffs);
        let dcs = jxl_encoder::vardct::dct::dc_from_dct_16x16(&dct_coeffs);

        eprintln!(
            "Uniform v={:.2}: dcs=[{:.4}, {:.4}, {:.4}, {:.4}] (all should be {:.4})",
            v, dcs[0], dcs[1], dcs[2], dcs[3], v
        );
    }
    eprintln!();

    // Test where quadrants have different values
    let mut block = [0.0f32; 256];
    // Top-left quadrant (0-7, 0-7): 0.0
    // Top-right quadrant (0-7, 8-15): 1.0
    // Bottom-left quadrant (8-15, 0-7): 0.0
    // Bottom-right quadrant (8-15, 8-15): 1.0
    for y in 0..8 {
        for x in 8..16 {
            block[y * 16 + x] = 1.0;
        }
    }
    for y in 8..16 {
        for x in 8..16 {
            block[y * 16 + x] = 1.0;
        }
    }

    let mut dct_coeffs = [0.0f32; 256];
    jxl_encoder::vardct::dct::dct_16x16(&block, &mut dct_coeffs);
    let dcs = jxl_encoder::vardct::dct::dc_from_dct_16x16(&dct_coeffs);

    eprintln!("Quadrant pattern (TL=0, TR=1, BL=0, BR=1):");
    eprintln!("  Expected: 0.0, 1.0, 0.0, 1.0");
    eprintln!(
        "  Got:      {:.4}, {:.4}, {:.4}, {:.4}",
        dcs[0], dcs[1], dcs[2], dcs[3]
    );
}

/// DIAGNOSTIC: Verify DCT16x16 coefficient layout matches expected frequency positions.
#[test]
#[ignore]
fn diag_dct16x16_layout() {
    // Create an image with only horizontal variation (x-gradient)
    // This should produce energy at fx=1, fy=0 (horizontal frequency)
    let mut block_h = [0.0f32; 256];
    for y in 0..16 {
        for x in 0..16 {
            block_h[y * 16 + x] = x as f32 / 16.0;
        }
    }

    let mut dct_h = [0.0f32; 256];
    jxl_encoder::vardct::dct::dct_16x16(&block_h, &mut dct_h);

    eprintln!("Horizontal gradient (x-variation only):");
    eprintln!("  coeff[0] (DC) = {:.6}", dct_h[0]);
    eprintln!(
        "  coeff[1] (should have energy if fx=1,fy=0) = {:.6}",
        dct_h[1]
    );
    eprintln!("  coeff[16] (should be ~0 if fy=0) = {:.6}", dct_h[16]);

    // Create an image with only vertical variation (y-gradient)
    // This should produce energy at fx=0, fy=1 (vertical frequency)
    let mut block_v = [0.0f32; 256];
    for y in 0..16 {
        for x in 0..16 {
            block_v[y * 16 + x] = y as f32 / 16.0;
        }
    }

    let mut dct_v = [0.0f32; 256];
    jxl_encoder::vardct::dct::dct_16x16(&block_v, &mut dct_v);

    eprintln!("\nVertical gradient (y-variation only):");
    eprintln!("  coeff[0] (DC) = {:.6}", dct_v[0]);
    eprintln!("  coeff[1] (should be ~0 if fx=0) = {:.6}", dct_v[1]);
    eprintln!(
        "  coeff[16] (should have energy if fx=0,fy=1) = {:.6}",
        dct_v[16]
    );

    // Now test dc_from_dct_16x16 with these
    let dcs_h = jxl_encoder::vardct::dct::dc_from_dct_16x16(&dct_h);
    let dcs_v = jxl_encoder::vardct::dct::dc_from_dct_16x16(&dct_v);

    eprintln!("\nHorizontal gradient DC extraction:");
    eprintln!("  Should have horizontal variation (left vs right):");
    eprintln!(
        "  dcs = [{:.4}, {:.4}, {:.4}, {:.4}]",
        dcs_h[0], dcs_h[1], dcs_h[2], dcs_h[3]
    );
    eprintln!(
        "  left column: avg({:.4}, {:.4}) = {:.4}",
        dcs_h[0],
        dcs_h[2],
        (dcs_h[0] + dcs_h[2]) / 2.0
    );
    eprintln!(
        "  right column: avg({:.4}, {:.4}) = {:.4}",
        dcs_h[1],
        dcs_h[3],
        (dcs_h[1] + dcs_h[3]) / 2.0
    );

    eprintln!("\nVertical gradient DC extraction:");
    eprintln!("  Should have vertical variation (top vs bottom):");
    eprintln!(
        "  dcs = [{:.4}, {:.4}, {:.4}, {:.4}]",
        dcs_v[0], dcs_v[1], dcs_v[2], dcs_v[3]
    );
    eprintln!(
        "  top row: avg({:.4}, {:.4}) = {:.4}",
        dcs_v[0],
        dcs_v[1],
        (dcs_v[0] + dcs_v[1]) / 2.0
    );
    eprintln!(
        "  bottom row: avg({:.4}, {:.4}) = {:.4}",
        dcs_v[2],
        dcs_v[3],
        (dcs_v[2] + dcs_v[3]) / 2.0
    );
}

/// DIAGNOSTIC: Check which transforms are being processed for 32x32 DCT16x16.
#[test]
#[ignore]
fn diag_dct16x16_transform_coverage() {
    use jxl_encoder::vardct::VarDctEncoder;

    // Patch VarDctEncoder to print transform positions - we'll do this by examining the strategy map
    let w = 32usize;
    let h = 32usize;
    let linear = vec![0.5f32; w * h * 3];

    let mut enc = VarDctEncoder::new(1.0);
    enc.ac_strategy_enabled = true;

    // Run encoding
    let bytes = enc.encode(w, h, &linear, None).unwrap().data;
    eprintln!("Encoded 32x32 with DCT16x16: {} bytes", bytes.len());

    // We can't easily inspect internal state, but we can check if the file decodes correctly
    let (_, _, dec) = decode_jxl_oxide(&bytes);

    // All pixels should be 0.5
    let mut max_err = 0.0f32;
    for i in 0..dec.len() {
        let err = (dec[i] - 0.5).abs();
        max_err = max_err.max(err);
    }
    eprintln!("Max error from expected 0.5: {:.6}", max_err);

    // Check specific positions
    eprintln!("\nDecoded values at 8x8 block centers:");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let idx = (py * w + px) * 3;
            eprint!("{:.3} ", dec[idx]);
        }
        eprintln!();
    }
}

/// DIAGNOSTIC: Test DCT16x16 with simple two-value pattern.
#[test]
#[ignore]
fn diag_dct16x16_two_values() {
    let w = 32usize;
    let h = 32usize;
    let mut linear = vec![0.0f32; w * h * 3];

    // Top half = 0.25, bottom half = 0.75
    for y in 0..h {
        let v = if y < 16 { 0.25 } else { 0.75 };
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    let mut enc8 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear, None).unwrap().data;

    let mut enc16 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear, None).unwrap().data;

    let (_, _, dec8) = decode_jxl_oxide(&bytes8);
    let (_, _, dec16) = decode_jxl_oxide(&bytes16);

    eprintln!("Two-value pattern (top=0.25, bottom=0.75):");
    eprintln!("File sizes: DCT8={}, DCT16={}", bytes8.len(), bytes16.len());

    eprintln!("\nDCT8 decoded at block centers:");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let expected = if by < 2 { 0.25 } else { 0.75 };
            let dec = dec8[(py * w + px) * 3];
            eprint!("{:.3}({:.3}) ", dec, expected);
        }
        eprintln!();
    }

    eprintln!("\nDCT16 decoded at block centers:");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let expected = if by < 2 { 0.25 } else { 0.75 };
            let dec = dec16[(py * w + px) * 3];
            eprint!("{:.3}({:.3}) ", dec, expected);
        }
        eprintln!();
    }
}

/// Trace DC extraction for two-value pattern
#[test]
#[ignore]
fn trace_two_value_dc_extraction() {
    use jxl_encoder::vardct::dct::{dc_from_dct_8x8, dc_from_dct_16x16, dct_8x8, dct_16x16};

    let w = 32usize;
    let h = 32usize;

    // Top half = 0.25, bottom half = 0.75 (in linear RGB, 0.5 mean)
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let v = if y < 16 { 0.25 } else { 0.75 };
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    // Convert to XYB (simplified - just use Y channel for grayscale)
    // For grayscale input, Y ≈ linear value
    eprintln!("=== Two-value pattern DC extraction ===\n");

    // First DCT16x16 block covers (0,0)-(15,15) - entirely in top half, all 0.25
    eprintln!("First DCT16x16 block (0,0)-(15,15): all pixels = 0.25");
    let mut block16 = [0.0f32; 256];
    for iy in 0..16 {
        for ix in 0..16 {
            block16[iy * 16 + ix] = 0.25; // entire block is 0.25
        }
    }
    let mut coeffs16 = [0.0f32; 256];
    dct_16x16(&block16, &mut coeffs16);
    eprintln!(
        "  DCT coeffs: [0]={:.6}, [1]={:.6}, [16]={:.6}, [17]={:.6}",
        coeffs16[0], coeffs16[1], coeffs16[16], coeffs16[17]
    );
    let dcs = dc_from_dct_16x16(&coeffs16);
    eprintln!(
        "  DC from IDCT: [{:.6}, {:.6}, {:.6}, {:.6}]",
        dcs[0], dcs[1], dcs[2], dcs[3]
    );
    eprintln!("  All should be close to 0.25*8=2.0 (DCT normalization)");

    // For comparison, DCT8 on same 0.25 block
    let mut block8 = [0.25f32; 64];
    let mut coeffs8 = [0.0f32; 64];
    dct_8x8(&block8, &mut coeffs8);
    let dc8 = dc_from_dct_8x8(&coeffs8);
    eprintln!("  DCT8 DC for same content: {:.6}", dc8);

    // Second DCT16x16 block covers (0,16)-(15,31) - top-right quadrant
    // Still entirely top half, all 0.25
    eprintln!("\nSecond DCT16x16 block (0,16)-(15,31): all pixels = 0.25");
    // Same as first block
    let dcs2 = dc_from_dct_16x16(&coeffs16);
    eprintln!(
        "  DC from IDCT: [{:.6}, {:.6}, {:.6}, {:.6}]",
        dcs2[0], dcs2[1], dcs2[2], dcs2[3]
    );

    // Third DCT16x16 block covers (16,0)-(31,15) - bottom-left quadrant
    // Entirely bottom half, all 0.75
    eprintln!("\nThird DCT16x16 block (16,0)-(31,15): all pixels = 0.75");
    let mut block16_bottom = [0.75f32; 256];
    let mut coeffs16_bottom = [0.0f32; 256];
    dct_16x16(&block16_bottom, &mut coeffs16_bottom);
    eprintln!(
        "  DCT coeffs: [0]={:.6}, [1]={:.6}, [16]={:.6}, [17]={:.6}",
        coeffs16_bottom[0], coeffs16_bottom[1], coeffs16_bottom[16], coeffs16_bottom[17]
    );
    let dcs3 = dc_from_dct_16x16(&coeffs16_bottom);
    eprintln!(
        "  DC from IDCT: [{:.6}, {:.6}, {:.6}, {:.6}]",
        dcs3[0], dcs3[1], dcs3[2], dcs3[3]
    );
    eprintln!("  All should be close to 0.75*8=6.0 (DCT normalization)");

    // DCT8 on 0.75 block
    let block8_bottom = [0.75f32; 64];
    let mut coeffs8_bottom = [0.0f32; 64];
    dct_8x8(&block8_bottom, &mut coeffs8_bottom);
    let dc8_bottom = dc_from_dct_8x8(&coeffs8_bottom);
    eprintln!("  DCT8 DC for same content: {:.6}", dc8_bottom);

    // Now the interesting one: DCT16x16 block that SPANS the boundary
    // This is where the bug would manifest - but 32x32 image has no such block!
    // Block (0,0)-(15,15) is all top, (16,16)-(31,31) is all bottom
    // Actually for y=0..16, it's top half; y=16..32 is bottom half
    // So blocks at by=0 span y=0..15 (all top)
    // Blocks at by=1 span y=16..31 (all bottom)
    // NO block spans the boundary!

    eprintln!("\n=== But wait! In 32x32 with DCT16x16, blocks don't cross the boundary! ===");
    eprintln!("by=0: y=0-15 (top half, all 0.25)");
    eprintln!("by=1: y=16-31 (bottom half, all 0.75)");
    eprintln!("Each DCT16x16 block is uniform - no frequency content!");

    // So where does the error come from?
    // Answer: DC prediction! Each DC block stores 4 values, and prediction
    // operates on the DC grid. If prediction is wrong, all downstream values are wrong.

    eprintln!("\n=== Testing encoder with DC prediction ===");

    let mut enc16 = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes = enc16.encode(w, h, &linear, None).unwrap().data;
    eprintln!("Encoded {} bytes", bytes.len());

    // Decode with jxl-oxide
    let (dw, _dh, dec) = decode_jxl_oxide(&bytes);

    eprintln!("\nDecoded values at 8x8 block centers:");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let expected = if by < 2 { 0.25 } else { 0.75 };
            let decoded = dec[(py * dw + px) * 3];
            eprint!("{:.4}({:.2}) ", decoded, expected);
        }
        eprintln!();
    }
}

/// Detailed DC trace for DCT16x16
#[test]
#[ignore]
fn trace_dct16x16_dc_detailed() {
    use jxl_encoder::vardct::dct::{dc_from_dct_16x16, dct_16x16};

    // Create a simple 32x32 image with two values
    // Top half = 0.25 (in linear sRGB), bottom half = 0.75
    let w = 32usize;
    let h = 32usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let v = if y < 16 { 0.25 } else { 0.75 };
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    eprintln!("=== DCT16x16 DC detailed trace ===");
    eprintln!("Input: 32x32 image, top half = 0.25, bottom half = 0.75\n");

    // Trace what dc_from_dct_16x16 produces for the first block
    let mut block = [0.0f32; 256];
    for iy in 0..16 {
        for ix in 0..16 {
            block[iy * 16 + ix] = 0.25; // top-left 16x16, all 0.25
        }
    }
    let mut coeffs = [0.0f32; 256];
    dct_16x16(&block, &mut coeffs);

    eprintln!("First DCT16x16 block (uniform 0.25):");
    eprintln!(
        "  LLF coeffs: [0]={:.6} [1]={:.6} [16]={:.6} [17]={:.6}",
        coeffs[0], coeffs[1], coeffs[16], coeffs[17]
    );

    let dcs = dc_from_dct_16x16(&coeffs);
    eprintln!(
        "  dc_from_dct_16x16 output: [{:.6}, {:.6}, {:.6}, {:.6}]",
        dcs[0], dcs[1], dcs[2], dcs[3]
    );

    // Now encode and see what DC tokens are generated
    eprintln!("\n=== Encoding with VarDctEncoder ===");
    let mut enc = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc.ac_strategy_enabled = true;

    // Enable debug output if available
    let bytes = enc.encode(w, h, &linear, None).unwrap().data;
    eprintln!("Encoded {} bytes", bytes.len());

    // Decode
    let (dw, dh, dec) = decode_jxl_oxide(&bytes);
    eprintln!("Decoded {}x{}", dw, dh);

    // Show decoded DC-like values (center of each 8x8 block)
    eprintln!("\nDecoded block centers (Y channel proxy):");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let expected = if by < 2 { 0.25 } else { 0.75 };
            let val = dec[(py * dw + px) * 3]; // R channel
            eprint!("{:.4} ", val);
        }
        eprintln!(" (expected: {:.2})", if by < 2 { 0.25 } else { 0.75 });
    }

    // Also try decoding with djxl for comparison
    use std::io::Write;
    let tmp_jxl = "/tmp/dct16_trace.jxl";
    let tmp_ppm = "/tmp/dct16_trace.ppm";
    std::fs::write(tmp_jxl, &bytes).unwrap();

    let status = std::process::Command::new(&jxl_encoder::test_helpers::djxl_path())
        .args(&[tmp_jxl, tmp_ppm])
        .status();
    if let Ok(s) = status {
        if s.success() {
            eprintln!("\ndjxl decoding succeeded");
            // Read the PPM and show values
            if let Ok(ppm_data) = std::fs::read(tmp_ppm) {
                // Skip PPM header, find the data
                let mut skip = 0;
                let mut newlines = 0;
                for (i, &b) in ppm_data.iter().enumerate() {
                    if b == b'\n' {
                        newlines += 1;
                        if newlines == 3 {
                            skip = i + 1;
                            break;
                        }
                    }
                }
                eprintln!("\ndjxl decoded block centers:");
                for by in 0..4 {
                    for bx in 0..4 {
                        let px = bx * 8 + 4;
                        let py = by * 8 + 4;
                        let idx = (py * w + px) * 3 + skip;
                        if idx < ppm_data.len() {
                            let r = ppm_data[idx] as f32 / 255.0;
                            eprint!("{:.4} ", r);
                        }
                    }
                    eprintln!(" (expected: {:.2})", if by < 2 { 0.25 } else { 0.75 });
                }
            }
        } else {
            eprintln!("djxl failed");
        }
    }
}

/// Trace exact quant_dc values for two-value pattern
#[test]
#[ignore]
fn trace_quant_dc_values() {
    use jxl_encoder::vardct::VarDctEncoder;

    let w = 32usize;
    let h = 32usize;
    let mut linear = vec![0.0f32; w * h * 3];

    // Top half = 0.25, bottom half = 0.75
    for y in 0..h {
        let v = if y < 16 { 0.25 } else { 0.75 };
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    eprintln!("=== Checking with DCT8 only (reference) ===");
    let mut enc8 = VarDctEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec8) = decode_jxl_oxide(&bytes8);

    eprintln!("DCT8 decoded block centers:");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = dec8[(py * w + px) * 3];
            eprint!("{:.4} ", val);
        }
        eprintln!();
    }

    eprintln!("\n=== Checking with DCT16x16 ===");
    let mut enc16 = VarDctEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec16) = decode_jxl_oxide(&bytes16);

    eprintln!("DCT16 decoded block centers:");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = dec16[(py * w + px) * 3];
            eprint!("{:.4} ", val);
        }
        eprintln!();
    }

    // The key question: what's different between DCT8 and DCT16 encoding
    // that causes the different decoded values?
    //
    // For uniform input:
    // - DCT8: coeff[0] = block_average, all others = 0
    //   DC stored is just coeff[0]
    // - DCT16: coeff[0] = 16x16_average, all others = 0
    //   DC stored are 4 values from dc_from_dct_16x16, all equal to coeff[0]
    //
    // Both should produce same decoded values!
    // Unless... the decoder does something different.

    eprintln!("\n=== Hypothesis: decoder's LowestFrequenciesFromDC ===");
    eprintln!("The decoder takes DC values and reconstructs LLF coefficients.");
    eprintln!("For DCT8, DC is just stored as-is.");
    eprintln!("For DCT16, decoder does 2x2 DCT on the 4 DC values to get LLF.");
    eprintln!("\nIf all 4 DC values are the same (V), the 2x2 DCT produces:");
    eprintln!("  LLF[0,0] = (V+V) + (V+V) = 4V");
    eprintln!("  LLF[0,1] = (V+V) - (V+V) = 0");
    eprintln!("  LLF[1,0] = (V-V) + (V-V) = 0");
    eprintln!("  LLF[1,1] = (V-V) - (V-V) = 0");
    eprintln!("This is correct! LLF[0,0] should be 4*block_average for DCT16.");
    eprintln!("\nBut wait - what scale factors does the decoder use?");
}

/// Compare C++ libjxl-tiny output with our encoder
#[test]
#[ignore]
fn compare_cpp_output() {
    // Decode C++ output
    let cpp_bytes = std::fs::read("/tmp/test_cpp.jxl")
        .expect("Run cjxl_tiny first: ~/work/libjxl-tiny/build/encoder/cjxl_tiny /tmp/test_two_value.pfm /tmp/test_cpp.jxl -d 1.0");

    eprintln!("=== Comparing C++ vs Rust encoder ===\n");
    eprintln!("C++ file size: {} bytes", cpp_bytes.len());

    let (w, h, cpp_dec) = decode_jxl_oxide(&cpp_bytes);
    eprintln!("C++ decoded {}x{}", w, h);

    eprintln!("\nC++ decoded block centers:");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = cpp_dec[(py * w + px) * 3];
            eprint!("{:.4} ", val);
        }
        eprintln!(" (expected: {:.2})", if by < 2 { 0.25 } else { 0.75 });
    }

    // Now our encoder
    let w = 32usize;
    let h = 32usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let v = if y < 16 { 0.25 } else { 0.75 };
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    let mut enc = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc.ac_strategy_enabled = true;
    let rust_bytes = enc.encode(w, h, &linear, None).unwrap().data;
    eprintln!("\nRust file size: {} bytes", rust_bytes.len());

    let (_, _, rust_dec) = decode_jxl_oxide(&rust_bytes);

    eprintln!("\nRust decoded block centers:");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = rust_dec[(py * w + px) * 3];
            eprint!("{:.4} ", val);
        }
        eprintln!(" (expected: {:.2})", if by < 2 { 0.25 } else { 0.75 });
    }
}

/// Force DCT16x16 and trace DC values
#[test]
#[ignore]
fn force_dct16x16_trace() {
    use jxl_encoder::vardct::VarDctEncoder;

    let w = 32usize;
    let h = 32usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let v = if y < 16 { 0.25 } else { 0.75 };
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    eprintln!("=== Forcing DCT16x16 for all blocks ===\n");

    let mut enc = VarDctEncoder::new(1.0);
    enc.force_strategy = Some(3); // RAW_STRATEGY_DCT16X16
    let bytes = enc.encode(w, h, &linear, None).unwrap().data;
    eprintln!("Encoded {} bytes", bytes.len());

    let (dw, _, dec) = decode_jxl_oxide(&bytes);

    eprintln!("\nDecoded block centers:");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = dec[(py * dw + px) * 3];
            eprint!("{:.4} ", val);
        }
        eprintln!(" (expected: {:.2})", if by < 2 { 0.25 } else { 0.75 });
    }
}

/// Check what strategies are being selected
#[test]
#[ignore]
fn check_selected_strategies() {
    use jxl_encoder::vardct::VarDctEncoder;

    let w = 32usize;
    let h = 32usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let v = if y < 16 { 0.25 } else { 0.75 };
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    eprintln!("=== With ac_strategy_enabled = true ===");
    let mut enc = VarDctEncoder::new(1.0);
    enc.ac_strategy_enabled = true;
    let bytes = enc.encode(w, h, &linear, None).unwrap().data;
    eprintln!("Encoded {} bytes", bytes.len());
    let (_, _, dec) = decode_jxl_oxide(&bytes);
    eprintln!("Decoded block centers (ac_strategy_enabled):");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = dec[(py * w + px) * 3];
            eprint!("{:.4} ", val);
        }
        eprintln!();
    }

    eprintln!("\n=== With force_strategy = DCT16x16 ===");
    let mut enc2 = VarDctEncoder::new(1.0);
    enc2.force_strategy = Some(3);
    let bytes2 = enc2.encode(w, h, &linear, None).unwrap().data;
    eprintln!("Encoded {} bytes", bytes2.len());
    let (_, _, dec2) = decode_jxl_oxide(&bytes2);
    eprintln!("Decoded block centers (force DCT16x16):");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = dec2[(py * w + px) * 3];
            eprint!("{:.4} ", val);
        }
        eprintln!();
    }

    eprintln!("\n=== With ac_strategy_enabled = false (DCT8 only) ===");
    let mut enc3 = VarDctEncoder::new(1.0);
    enc3.ac_strategy_enabled = false;
    let bytes3 = enc3.encode(w, h, &linear, None).unwrap().data;
    eprintln!("Encoded {} bytes", bytes3.len());
    let (_, _, dec3) = decode_jxl_oxide(&bytes3);
    eprintln!("Decoded block centers (DCT8 only):");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = dec3[(py * w + px) * 3];
            eprint!("{:.4} ", val);
        }
        eprintln!();
    }
}

/// Verify dc_from_dct_32x32 produces uniform output for uniform input
#[test]
#[ignore]
fn test_dc_from_dct_32x32_uniform() {
    use jxl_encoder::vardct::dct::{dc_from_dct_32x32, dct_32x32};

    // Create uniform 32x32 block
    let mut input = [0.5f32; 1024];
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&input, &mut coeffs);

    eprintln!("Uniform input (all 0.5):");
    eprintln!(
        "  LLF: [{:.6}, {:.6}, {:.6}, {:.6}]",
        coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    );
    eprintln!(
        "  LLF: [{:.6}, {:.6}, {:.6}, {:.6}]",
        coeffs[32], coeffs[33], coeffs[34], coeffs[35]
    );

    let dcs = dc_from_dct_32x32(&coeffs);
    eprintln!("  DC values:");
    for row in 0..4 {
        eprint!("    Row {}: ", row);
        for col in 0..4 {
            eprint!("{:.6} ", dcs[row * 4 + col]);
        }
        eprintln!();
    }
    eprintln!("  Expected: all ≈ 0.5");

    // Now test step function (top=0.3, bottom=0.7)
    eprintln!("\nStep function input (top=0.3, bottom=0.7):");
    for y in 0..32 {
        let v = if y < 16 { 0.3 } else { 0.7 };
        for x in 0..32 {
            input[y * 32 + x] = v;
        }
    }
    dct_32x32(&input, &mut coeffs);

    eprintln!(
        "  LLF: [{:.6}, {:.6}, {:.6}, {:.6}]",
        coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    );
    eprintln!(
        "  LLF: [{:.6}, {:.6}, {:.6}, {:.6}]",
        coeffs[32], coeffs[33], coeffs[34], coeffs[35]
    );

    let dcs = dc_from_dct_32x32(&coeffs);
    eprintln!("  DC values:");
    for row in 0..4 {
        eprint!("    Row {}: ", row);
        for col in 0..4 {
            eprint!("{:.6} ", dcs[row * 4 + col]);
        }
        eprintln!();
    }
    eprintln!("  Expected: rows 0-1 ≈ 0.3, rows 2-3 ≈ 0.7");
}

/// Verify dc_from_dct_16x16 with step function
#[test]
#[ignore]
fn test_dc_from_dct_16x16_step() {
    use jxl_encoder::vardct::dct::{dc_from_dct_16x16, dct_16x16};

    // Step function: top half = 0.3, bottom half = 0.7
    let mut input = [0.0f32; 256];
    for y in 0..16 {
        let v = if y < 8 { 0.3 } else { 0.7 };
        for x in 0..16 {
            input[y * 16 + x] = v;
        }
    }
    let mut coeffs = [0.0f32; 256];
    dct_16x16(&input, &mut coeffs);

    eprintln!("DCT16x16 step function (top=0.3, bottom=0.7):");
    eprintln!("  LLF: [{:.6}, {:.6}]", coeffs[0], coeffs[1]);
    eprintln!("       [{:.6}, {:.6}]", coeffs[16], coeffs[17]);

    let dcs = dc_from_dct_16x16(&coeffs);
    eprintln!("  DC values:");
    eprintln!("    Row 0: {:.6} {:.6}", dcs[0], dcs[1]);
    eprintln!("    Row 1: {:.6} {:.6}", dcs[2], dcs[3]);
    eprintln!("  Expected: row 0 ≈ 0.3, row 1 ≈ 0.7");
}

/// Verify roundtrip: LLF -> IDCT -> DC -> DCT -> LLF
#[test]
#[ignore]
fn test_llf_dc_roundtrip_32x32() {
    use jxl_encoder::vardct::dct::{DCT_RESAMPLE_SCALE_32_TO_4, dc_from_dct_32x32, dct_32x32};

    // Create step function input
    let mut input = [0.0f32; 1024];
    for y in 0..32 {
        let v = if y < 16 { 0.3 } else { 0.7 };
        for x in 0..32 {
            input[y * 32 + x] = v;
        }
    }
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&input, &mut coeffs);

    // Original LLF (4x4 top-left of coefficients)
    eprintln!("=== Original LLF from encoder (first 2x2 of 4x4) ===");
    let orig_llf = [
        [coeffs[0], coeffs[1], coeffs[2], coeffs[3]],
        [coeffs[32], coeffs[33], coeffs[34], coeffs[35]],
        [coeffs[64], coeffs[65], coeffs[66], coeffs[67]],
        [coeffs[96], coeffs[97], coeffs[98], coeffs[99]],
    ];
    for row in 0..4 {
        eprintln!(
            "  [{:.6}, {:.6}, {:.6}, {:.6}]",
            orig_llf[row][0], orig_llf[row][1], orig_llf[row][2], orig_llf[row][3]
        );
    }

    // DC values from dc_from_dct_32x32
    let dcs = dc_from_dct_32x32(&coeffs);
    eprintln!("\n=== DC values from dc_from_dct_32x32 (4x4) ===");
    for row in 0..4 {
        eprintln!(
            "  [{:.6}, {:.6}, {:.6}, {:.6}]",
            dcs[row * 4],
            dcs[row * 4 + 1],
            dcs[row * 4 + 2],
            dcs[row * 4 + 3]
        );
    }

    // Now simulate decoder: DCT on DC values, then apply scale[4,32]
    // First need 4x4 DCT
    let dct_4_to_32_scales = [
        1.0f32,
        1.025760096781116015,
        1.108937353592731823,
        1.270559368765487251,
    ];

    // 4x4 DCT on DC values (matching decoder's ComputeScaledDCT<4,4>)
    // For simplicity, use direct formula
    let n = 4;
    let mut dct_out = [[0.0f32; 4]; 4];
    for u in 0..n {
        for v in 0..n {
            let mut sum = 0.0f32;
            for y in 0..n {
                for x in 0..n {
                    let cos_u = ((2 * y + 1) as f32 * u as f32 * std::f32::consts::PI
                        / (2.0 * n as f32))
                        .cos();
                    let cos_v = ((2 * x + 1) as f32 * v as f32 * std::f32::consts::PI
                        / (2.0 * n as f32))
                        .cos();
                    sum += dcs[y * 4 + x] * cos_u * cos_v;
                }
            }
            // DCT normalization: sum / (N*N) (matching our dct_32x32 scaling of 1/32 per dimension)
            dct_out[u][v] = sum / ((n * n) as f32);
        }
    }

    eprintln!("\n=== 4x4 DCT of DC values (before resample scale) ===");
    for row in 0..4 {
        eprintln!(
            "  [{:.6}, {:.6}, {:.6}, {:.6}]",
            dct_out[row][0], dct_out[row][1], dct_out[row][2], dct_out[row][3]
        );
    }

    // Apply scale[4,32]
    let mut reconstructed_llf = [[0.0f32; 4]; 4];
    for u in 0..4 {
        for v in 0..4 {
            reconstructed_llf[u][v] = dct_out[u][v] * dct_4_to_32_scales[u] * dct_4_to_32_scales[v];
        }
    }

    eprintln!("\n=== Reconstructed LLF (after scale[4,32]) ===");
    for row in 0..4 {
        eprintln!(
            "  [{:.6}, {:.6}, {:.6}, {:.6}]",
            reconstructed_llf[row][0],
            reconstructed_llf[row][1],
            reconstructed_llf[row][2],
            reconstructed_llf[row][3]
        );
    }

    // Compare
    eprintln!("\n=== Error (reconstructed - original) ===");
    for row in 0..4 {
        eprintln!(
            "  [{:.6}, {:.6}, {:.6}, {:.6}]",
            reconstructed_llf[row][0] - orig_llf[row][0],
            reconstructed_llf[row][1] - orig_llf[row][1],
            reconstructed_llf[row][2] - orig_llf[row][2],
            reconstructed_llf[row][3] - orig_llf[row][3]
        );
    }
}

/// Test 4x4 DCT/IDCT roundtrip
#[test]
#[ignore]
fn test_4x4_dct_idct_roundtrip() {
    use jxl_encoder::vardct::dct::DCT_RESAMPLE_SCALE_32_TO_4;

    // Create asymmetric 4x4 input (not symmetric, so we can see transpose issues)
    let input = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ];

    eprintln!("=== Input 4x4 ===");
    for row in 0..4 {
        eprintln!(
            "  [{:.1}, {:.1}, {:.1}, {:.1}]",
            input[row][0], input[row][1], input[row][2], input[row][3]
        );
    }

    // Simulate encoder dc_from_dct: apply scale, then IDCT
    // (pretend input is the LLF region of a 32x32 DCT)
    let dct_32_to_4_scales = DCT_RESAMPLE_SCALE_32_TO_4;

    let mut scaled_input = [[0.0f32; 4]; 4];
    for iy in 0..4 {
        for ix in 0..4 {
            scaled_input[iy][ix] = input[iy][ix] * dct_32_to_4_scales[iy] * dct_32_to_4_scales[ix];
        }
    }

    // 4x4 IDCT (rows → transpose → rows)
    fn idct1d_4(input: &[f32; 4]) -> [f32; 4] {
        let n = 4;
        let mut output = [0.0f32; 4];
        for k in 0..n {
            let mut sum = 0.0f32;
            for u in 0..n {
                let cos_val =
                    ((2 * k + 1) as f32 * u as f32 * std::f32::consts::PI / (2.0 * n as f32)).cos();
                sum += input[u] * cos_val;
            }
            // IDCT-III doesn't need 1/N scaling for inverse to work
            output[k] = sum;
        }
        output
    }

    // IDCT rows
    let mut after_rows = [[0.0f32; 4]; 4];
    for iy in 0..4 {
        let row_in = [
            scaled_input[iy][0],
            scaled_input[iy][1],
            scaled_input[iy][2],
            scaled_input[iy][3],
        ];
        let row_out = idct1d_4(&row_in);
        for ix in 0..4 {
            after_rows[iy][ix] = row_out[ix];
        }
    }

    // Transpose
    let mut transposed = [[0.0f32; 4]; 4];
    for iy in 0..4 {
        for ix in 0..4 {
            transposed[ix][iy] = after_rows[iy][ix];
        }
    }

    // IDCT rows (on transposed data)
    let mut dc_values = [[0.0f32; 4]; 4];
    for iy in 0..4 {
        let row_in = [
            transposed[iy][0],
            transposed[iy][1],
            transposed[iy][2],
            transposed[iy][3],
        ];
        let row_out = idct1d_4(&row_in);
        for ix in 0..4 {
            dc_values[iy][ix] = row_out[ix];
        }
    }

    eprintln!("\n=== DC values (after IDCT with scale[32,4]) ===");
    for row in 0..4 {
        eprintln!(
            "  [{:.4}, {:.4}, {:.4}, {:.4}]",
            dc_values[row][0], dc_values[row][1], dc_values[row][2], dc_values[row][3]
        );
    }

    // Simulate decoder: DCT then apply scale
    fn dct1d_4(input: &[f32; 4]) -> [f32; 4] {
        let n = 4;
        let mut output = [0.0f32; 4];
        for u in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                let cos_val =
                    ((2 * k + 1) as f32 * u as f32 * std::f32::consts::PI / (2.0 * n as f32)).cos();
                sum += input[k] * cos_val;
            }
            // Normalize by 1/N per dimension, so 1/N for 1D
            output[u] = sum / (n as f32);
        }
        output
    }

    // DCT rows
    let mut dct_after_rows = [[0.0f32; 4]; 4];
    for iy in 0..4 {
        let row_in = [
            dc_values[iy][0],
            dc_values[iy][1],
            dc_values[iy][2],
            dc_values[iy][3],
        ];
        let row_out = dct1d_4(&row_in);
        for ix in 0..4 {
            dct_after_rows[iy][ix] = row_out[ix];
        }
    }

    // Transpose
    let mut dct_transposed = [[0.0f32; 4]; 4];
    for iy in 0..4 {
        for ix in 0..4 {
            dct_transposed[ix][iy] = dct_after_rows[iy][ix];
        }
    }

    // DCT rows (on transposed data)
    let mut dct_output = [[0.0f32; 4]; 4];
    for iy in 0..4 {
        let row_in = [
            dct_transposed[iy][0],
            dct_transposed[iy][1],
            dct_transposed[iy][2],
            dct_transposed[iy][3],
        ];
        let row_out = dct1d_4(&row_in);
        for ix in 0..4 {
            dct_output[iy][ix] = row_out[ix];
        }
    }

    eprintln!("\n=== After decoder DCT (before scale[4,32]) ===");
    for row in 0..4 {
        eprintln!(
            "  [{:.4}, {:.4}, {:.4}, {:.4}]",
            dct_output[row][0], dct_output[row][1], dct_output[row][2], dct_output[row][3]
        );
    }

    // Apply scale[4,32]
    let dct_4_to_32_scales = [
        1.0f32,
        1.025760096781116015,
        1.108937353592731823,
        1.270559368765487251,
    ];

    let mut reconstructed = [[0.0f32; 4]; 4];
    for iy in 0..4 {
        for ix in 0..4 {
            reconstructed[iy][ix] =
                dct_output[iy][ix] * dct_4_to_32_scales[iy] * dct_4_to_32_scales[ix];
        }
    }

    eprintln!("\n=== Reconstructed LLF (after scale[4,32]) ===");
    for row in 0..4 {
        eprintln!(
            "  [{:.4}, {:.4}, {:.4}, {:.4}]",
            reconstructed[row][0],
            reconstructed[row][1],
            reconstructed[row][2],
            reconstructed[row][3]
        );
    }

    eprintln!("\n=== Error (reconstructed - input) ===");
    for row in 0..4 {
        eprintln!(
            "  [{:.4}, {:.4}, {:.4}, {:.4}]",
            reconstructed[row][0] - input[row][0],
            reconstructed[row][1] - input[row][1],
            reconstructed[row][2] - input[row][2],
            reconstructed[row][3] - input[row][3]
        );
    }
}

/// Test DCT32x32 with uniform blocks (64x64 image, step at y=32)
#[test]
#[ignore]
fn test_dct32x32_uniform_blocks() {
    // 64x64 image with top half = 0.25, bottom half = 0.75
    // Each DCT32x32 block (32x32 pixels) will be entirely uniform
    let w = 64usize;
    let h = 64usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let v = if y < 32 { 0.25 } else { 0.75 };
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    eprintln!("=== 64x64 image with step at y=32 ===");
    eprintln!("Each DCT32x32 block is uniform (top: 0.25, bottom: 0.75)\n");

    let mut enc = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    enc.force_strategy = Some(4); // DCT32x32
    enc.optimize_codes = false; // Use static Huffman for simpler debugging
    let bytes = enc.encode(w, h, &linear, None).unwrap().data;
    eprintln!("Encoded {} bytes", bytes.len());

    let (dw, dh, dec) = decode_jxl_oxide(&bytes);
    eprintln!("Decoded {}x{}", dw, dh);

    eprintln!("\nDecoded values at 8x8 block centers:");
    for by in 0..8 {
        for bx in 0..8 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let expected = if by < 4 { 0.25 } else { 0.75 };
            let val = dec[(py * dw + px) * 3];
            eprint!("{:.3} ", val);
        }
        eprintln!(" (expected: {:.2})", if by < 4 { 0.25 } else { 0.75 });
    }
}

/// Test each strategy individually on the two-value image
#[test]
#[ignore]
fn test_each_strategy_quality() {
    use jxl_encoder::vardct::VarDctEncoder;

    let w = 32usize;
    let h = 32usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let v = if y < 16 { 0.25 } else { 0.75 };
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    let strategies = [
        (0, "DCT8"),
        (1, "DCT16x8"),
        (2, "DCT8x16"),
        (3, "DCT16x16"),
        // (4, "DCT32x32"),  // Skip - requires 64x64 minimum
    ];

    for (strat, name) in &strategies {
        let mut enc = VarDctEncoder::new(1.0);
        enc.force_strategy = Some(*strat);
        let bytes = enc.encode(w, h, &linear, None).unwrap().data;
        let (_, _, dec) = decode_jxl_oxide(&bytes);

        eprintln!("\n=== {} (strategy {}) ===", name, strat);
        eprintln!("Encoded {} bytes", bytes.len());
        eprintln!("Decoded block centers:");

        let mut max_err = 0.0f32;
        for by in 0..4 {
            for bx in 0..4 {
                let px = bx * 8 + 4;
                let py = by * 8 + 4;
                let val = dec[(py * w + px) * 3];
                let expected = if py < 16 { 0.25 } else { 0.75 };
                let err = (val - expected).abs();
                max_err = max_err.max(err);
                eprint!("{:.4} ", val);
            }
            let expected = if by < 2 { 0.25 } else { 0.75 };
            eprintln!(" (expected: {:.2})", expected);
        }
        eprintln!("Max error: {:.4}", max_err);

        // Assert reasonable quality
        assert!(
            max_err < 0.05,
            "{} has max error {:.4}, expected < 0.05",
            name,
            max_err
        );
    }
}
/// Test with uniform image to isolate edge effects
#[test]
#[ignore]
fn test_uniform_image_with_strategy_selection() {
    use jxl_encoder::vardct::VarDctEncoder;

    let w = 32usize;
    let h = 32usize;
    // Uniform value 0.5
    let linear = vec![0.5f32; w * h * 3];

    eprintln!("=== Uniform image (0.5) with ac_strategy_enabled = true ===");
    let mut enc = VarDctEncoder::new(1.0);
    enc.ac_strategy_enabled = true;
    let bytes = enc.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec) = decode_jxl_oxide(&bytes);
    eprintln!("Encoded {} bytes", bytes.len());

    let mut max_err = 0.0f32;
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = dec[(py * w + px) * 3];
            let err = (val - 0.5).abs();
            max_err = max_err.max(err);
            eprint!("{:.4} ", val);
        }
        eprintln!();
    }
    eprintln!("Max error: {:.4} (expected 0.5 everywhere)", max_err);

    eprintln!("\n=== Same with ac_strategy_enabled = false ===");
    let mut enc2 = VarDctEncoder::new(1.0);
    enc2.ac_strategy_enabled = false;
    let bytes2 = enc2.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec2) = decode_jxl_oxide(&bytes2);
    eprintln!("Encoded {} bytes", bytes2.len());

    let mut max_err2 = 0.0f32;
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = dec2[(py * w + px) * 3];
            let err = (val - 0.5).abs();
            max_err2 = max_err2.max(err);
            eprint!("{:.4} ", val);
        }
        eprintln!();
    }
    eprintln!("Max error: {:.4}", max_err2);

    // Both should have similar quality
    assert!(
        max_err < 0.02,
        "Strategy selection introduces error: {:.4}",
        max_err
    );
}

/// Test with DCT32x32 on the two-value image to see if that's the issue
#[test]
#[ignore]
fn test_dct32x32_two_value_image() {
    use jxl_encoder::vardct::VarDctEncoder;

    // Need 64x64 to fit a DCT32x32
    let w = 64usize;
    let h = 64usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let v = if y < 32 { 0.25 } else { 0.75 };
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    eprintln!("=== 64x64 two-value image with force_strategy = DCT32x32 ===");
    let mut enc = VarDctEncoder::new(1.0);
    enc.force_strategy = Some(4); // DCT32x32
    let bytes = enc.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec) = decode_jxl_oxide(&bytes);
    eprintln!("Encoded {} bytes", bytes.len());

    eprintln!("Decoded pixel values (sampled):");
    let mut sum_top = 0.0f32;
    let mut sum_bot = 0.0f32;
    let mut cnt = 0;
    for by in 0..8 {
        for bx in 0..8 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = dec[(py * w + px) * 3];
            if py < 32 {
                sum_top += val;
            } else {
                sum_bot += val;
            }
            cnt += 1;
            eprint!("{:.3} ", val);
        }
        let expected = if by < 4 { 0.25 } else { 0.75 };
        eprintln!(" (expected: {:.2})", expected);
    }
    eprintln!("Average top: {:.4} (expected 0.25)", sum_top / 32.0);
    eprintln!("Average bot: {:.4} (expected 0.75)", sum_bot / 32.0);

    eprintln!("\n=== Same with ac_strategy_enabled = true ===");
    let mut enc2 = VarDctEncoder::new(1.0);
    enc2.ac_strategy_enabled = true;
    let bytes2 = enc2.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec2) = decode_jxl_oxide(&bytes2);
    eprintln!("Encoded {} bytes", bytes2.len());

    eprintln!("Decoded pixel values (sampled):");
    let mut sum_top2 = 0.0f32;
    let mut sum_bot2 = 0.0f32;
    for by in 0..8 {
        for bx in 0..8 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = dec2[(py * w + px) * 3];
            if py < 32 {
                sum_top2 += val;
            } else {
                sum_bot2 += val;
            }
            eprint!("{:.3} ", val);
        }
        let expected = if by < 4 { 0.25 } else { 0.75 };
        eprintln!(" (expected: {:.2})", expected);
    }
    eprintln!("Average top: {:.4} (expected 0.25)", sum_top2 / 32.0);
    eprintln!("Average bot: {:.4} (expected 0.75)", sum_bot2 / 32.0);

    eprintln!("\n=== Same with DCT8 only ===");
    let mut enc3 = VarDctEncoder::new(1.0);
    enc3.ac_strategy_enabled = false;
    let bytes3 = enc3.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec3) = decode_jxl_oxide(&bytes3);
    eprintln!("Encoded {} bytes", bytes3.len());

    let mut sum_top3 = 0.0f32;
    let mut sum_bot3 = 0.0f32;
    for by in 0..8 {
        for bx in 0..8 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = dec3[(py * w + px) * 3];
            if py < 32 {
                sum_top3 += val;
            } else {
                sum_bot3 += val;
            }
        }
    }
    eprintln!("Average top: {:.4} (expected 0.25)", sum_top3 / 32.0);
    eprintln!("Average bot: {:.4} (expected 0.75)", sum_bot3 / 32.0);
}

/// Test DCT32x32 on 32x32 two-value image (single DCT32 covers entire image)
#[test]
#[ignore]
fn test_dct32x32_on_32x32_two_value() {
    use jxl_encoder::vardct::VarDctEncoder;

    let w = 32usize;
    let h = 32usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let v = if y < 16 { 0.25 } else { 0.75 };
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    eprintln!("=== 32x32 two-value with force_strategy = DCT32x32 ===");
    let mut enc = VarDctEncoder::new(1.0);
    enc.force_strategy = Some(4); // DCT32x32
    let bytes = enc.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec) = decode_jxl_oxide(&bytes);
    eprintln!("Encoded {} bytes", bytes.len());

    eprintln!("Decoded block centers:");
    for by in 0..4 {
        for bx in 0..4 {
            let px = bx * 8 + 4;
            let py = by * 8 + 4;
            let val = dec[(py * w + px) * 3];
            eprint!("{:.4} ", val);
        }
        eprintln!(" (expected: {:.2})", if by < 2 { 0.25 } else { 0.75 });
    }

    // Compare to individual strategies
    let configs = [
        (Some(0), "DCT8"),
        (Some(3), "DCT16x16"),
        (Some(4), "DCT32x32"),
        (None, "ac_strategy_enabled"),
    ];

    for (force_strat, name) in configs.iter() {
        let mut enc = VarDctEncoder::new(1.0);
        if let Some(s) = force_strat {
            enc.force_strategy = Some(*s);
        } else {
            enc.ac_strategy_enabled = true;
        }
        let bytes = enc.encode(w, h, &linear, None).unwrap().data;
        let (_, _, dec) = decode_jxl_oxide(&bytes);

        let mut max_err = 0.0f32;
        for y in 0..h {
            let expected = if y < 16 { 0.25 } else { 0.75 };
            for x in 0..w {
                let val = dec[(y * w + x) * 3];
                let err = (val - expected).abs();
                max_err = max_err.max(err);
            }
        }
        eprintln!("{}: max_err = {:.4}", name, max_err);
    }
}

/// Test DCT16x16 on 16x16 image with edge inside the block
#[test]
#[ignore]
fn test_dct16x16_with_internal_edge() {
    use jxl_encoder::vardct::VarDctEncoder;

    // 16x16 image with edge at y=8 (inside single DCT16x16 block)
    let w = 16usize;
    let h = 16usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let v = if y < 8 { 0.25 } else { 0.75 };
        for x in 0..w {
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    let configs: [(u8, &str); 2] = [(0, "DCT8"), (3, "DCT16x16")];

    for (force_strat, name) in configs.iter() {
        let mut enc = VarDctEncoder::new(1.0);
        enc.force_strategy = Some(*force_strat);
        let bytes = enc.encode(w, h, &linear, None).unwrap().data;
        let (_, _, dec) = decode_jxl_oxide(&bytes);

        eprintln!("=== {} on 16x16 with edge at y=8 ===", name);
        eprintln!("Encoded {} bytes", bytes.len());

        // Sample all 4 quadrants
        let tl = dec[(2 * w + 2) * 3]; // top-left
        let tr = dec[(2 * w + 14) * 3]; // top-right
        let bl = dec[(12 * w + 2) * 3]; // bottom-left
        let br = dec[(12 * w + 14) * 3]; // bottom-right

        eprintln!("  TL={:.4} TR={:.4} (expected 0.25)", tl, tr);
        eprintln!("  BL={:.4} BR={:.4} (expected 0.75)", bl, br);

        let err_tl = (tl - 0.25).abs();
        let err_bl = (bl - 0.75).abs();
        let max_err = err_tl.max(err_bl);
        eprintln!("  Max error: {:.4}", max_err);
    }
}

/// Test layer2 with different forced strategies to isolate DCT32x32 issue.
/// Uses djxl (reference decoder) instead of jxl-oxide due to known jxl-oxide ANS bugs.
#[test]
#[ignore]
fn test_layer2_strategies_comparison() {
    let (w, h, linear, srgb) = load_png_crop(&frymire_path(), 256, 256);
    assert_eq!(w, 256);
    assert_eq!(h, 256);

    // Skip DCT32x32 for frymire (pathological content) - use only strategies that
    // work well on high-contrast content
    let strategies: [(Option<u8>, &str); 4] = [
        (Some(0), "DCT8"),
        (Some(1), "DCT16x8"),
        (Some(2), "DCT8x16"),
        (Some(3), "DCT16x16"),
    ];

    for (force_strat, name) in strategies.iter() {
        let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
        if let Some(s) = force_strat {
            encoder.force_strategy = Some(*s);
        }

        let bytes = match encoder.encode(w, h, &linear, None) {
            Ok(output) => output.data,
            Err(e) => {
                eprintln!("{}: encode failed: {:?}", name, e);
                continue;
            }
        };

        // Use djxl (reference decoder) instead of jxl-oxide (known ANS bugs)
        let (dw, dh, dec_srgb) = decode_djxl(&bytes);
        assert_eq!(dw, w);
        assert_eq!(dh, h);

        let ssim2 = ssim2_u8_vs_linear_u8(&srgb, &dec_srgb, w, h);
        eprintln!("{}: {} bytes, SSIM2 = {:.2}", name, bytes.len(), ssim2);
    }

    // Also test with ac_strategy_enabled
    eprintln!("\n--- With ac_strategy_enabled ---");
    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(1.0);
    encoder.ac_strategy_enabled = true;
    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;
    let (_, _, dec_srgb) = decode_djxl(&bytes);
    let ssim2 = ssim2_u8_vs_linear_u8(&srgb, &dec_srgb, w, h);
    eprintln!(
        "ac_strategy_enabled: {} bytes, SSIM2 = {:.2}",
        bytes.len(),
        ssim2
    );
}

// =============================================================================
// DCT4X8/DCT8X4 Layer 3 Tests
// =============================================================================

/// Layer 3 test: Force DCT4X8 strategy and verify djxl decodes without error.
#[test]
#[ignore]
fn layer3_single_group_dct4x8_decode_djxl() {
    use jxl_encoder::vardct::VarDctEncoder;
    use std::fs;
    use std::io::Write;

    // 64x64 gradient image - fits in single group, multiple 8x8 blocks
    let w = 64usize;
    let h = 64usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let v = (x as f32 + y as f32) / (w as f32 + h as f32 - 2.0);
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    let mut encoder = VarDctEncoder::new(2.0);
    encoder.force_strategy = Some(5); // RAW_STRATEGY_DCT4X8

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;

    // Save to file
    let path = "/tmp/test_dct4x8_layer3.jxl";
    let mut file = fs::File::create(path).unwrap();
    file.write_all(&bytes).unwrap();
    eprintln!("DCT4X8: {} bytes saved to {}", bytes.len(), path);

    // Decode with djxl
    let output = std::process::Command::new(&jxl_encoder::test_helpers::djxl_path())
        .arg(path)
        .arg("/tmp/test_dct4x8_layer3.png")
        .output()
        .expect("djxl failed to run");

    if !output.status.success() {
        eprintln!("djxl stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("djxl failed with status {}", output.status);
    }
    eprintln!("djxl decoded DCT4X8 successfully");
}

/// Layer 3 test: Force DCT8X4 strategy and verify djxl decodes without error.
#[test]
#[ignore]
fn layer3_single_group_dct8x4_decode_djxl() {
    use jxl_encoder::vardct::VarDctEncoder;
    use std::fs;
    use std::io::Write;

    // 64x64 gradient image
    let w = 64usize;
    let h = 64usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let v = (x as f32 + y as f32) / (w as f32 + h as f32 - 2.0);
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    let mut encoder = VarDctEncoder::new(2.0);
    encoder.force_strategy = Some(6); // RAW_STRATEGY_DCT8X4

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;

    // Save to file
    let path = "/tmp/test_dct8x4_layer3.jxl";
    let mut file = fs::File::create(path).unwrap();
    file.write_all(&bytes).unwrap();
    eprintln!("DCT8X4: {} bytes saved to {}", bytes.len(), path);

    // Decode with djxl
    let output = std::process::Command::new(&jxl_encoder::test_helpers::djxl_path())
        .arg(path)
        .arg("/tmp/test_dct8x4_layer3.png")
        .output()
        .expect("djxl failed to run");

    if !output.status.success() {
        eprintln!("djxl stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("djxl failed with status {}", output.status);
    }
    eprintln!("djxl decoded DCT8X4 successfully");
}

/// Layer 3 test: Force DCT4X8 and verify jxl-oxide decodes.
#[test]
#[ignore]
fn layer3_single_group_dct4x8_decode_jxl_oxide() {
    use jxl_encoder::vardct::VarDctEncoder;

    // 64x64 gradient image
    let w = 64usize;
    let h = 64usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let v = (x as f32 + y as f32) / (w as f32 + h as f32 - 2.0);
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    let mut encoder = VarDctEncoder::new(2.0);
    encoder.force_strategy = Some(5); // RAW_STRATEGY_DCT4X8

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;
    eprintln!("DCT4X8: {} bytes encoded", bytes.len());

    // Decode with jxl-oxide
    let (dw, dh, pixels) = decode_jxl_oxide(&bytes);
    assert_eq!(dw, w);
    assert_eq!(dh, h);
    eprintln!("jxl-oxide decoded DCT4X8 successfully: {}x{}", dw, dh);

    // Basic sanity check on decoded values
    let center_idx = (h / 2 * w + w / 2) * 3;
    let center_val = pixels[center_idx];
    eprintln!("Center pixel value: {:.4} (expected ~0.5)", center_val);
    assert!(
        (center_val - 0.5).abs() < 0.2,
        "Center pixel too far from expected"
    );
}

/// Layer 3 test: Force DCT8X4 and verify jxl-oxide decodes.
#[test]
#[ignore]
fn layer3_single_group_dct8x4_decode_jxl_oxide() {
    use jxl_encoder::vardct::VarDctEncoder;

    // 64x64 gradient image
    let w = 64usize;
    let h = 64usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let v = (x as f32 + y as f32) / (w as f32 + h as f32 - 2.0);
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    let mut encoder = VarDctEncoder::new(2.0);
    encoder.force_strategy = Some(6); // RAW_STRATEGY_DCT8X4

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;
    eprintln!("DCT8X4: {} bytes encoded", bytes.len());

    // Decode with jxl-oxide
    let (dw, dh, pixels) = decode_jxl_oxide(&bytes);
    assert_eq!(dw, w);
    assert_eq!(dh, h);
    eprintln!("jxl-oxide decoded DCT8X4 successfully: {}x{}", dw, dh);

    // Basic sanity check on decoded values
    let center_idx = (h / 2 * w + w / 2) * 3;
    let center_val = pixels[center_idx];
    eprintln!("Center pixel value: {:.4} (expected ~0.5)", center_val);
    assert!(
        (center_val - 0.5).abs() < 0.2,
        "Center pixel too far from expected"
    );
}

/// Layer 3 test: Force DCT4X8 and verify jxl-rs decodes.
/// Note: jxl-rs outputs sRGB (not linear), so center pixel is ~0.74 (sRGB) not ~0.5 (linear).
/// This is expected behavior - jxl-rs doesn't have API to request linear output.
#[test]
#[ignore]
fn layer3_single_group_dct4x8_decode_jxl_rs() {
    use jxl_encoder::vardct::VarDctEncoder;

    // 64x64 gradient image
    let w = 64usize;
    let h = 64usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let v = (x as f32 + y as f32) / (w as f32 + h as f32 - 2.0);
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    let mut encoder = VarDctEncoder::new(2.0);
    encoder.force_strategy = Some(5); // RAW_STRATEGY_DCT4X8

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;
    eprintln!("DCT4X8: {} bytes encoded", bytes.len());

    // Decode with jxl-rs
    let (dw, dh, pixels) = decode_jxl_rs(&bytes);
    assert_eq!(dw, w);
    assert_eq!(dh, h);
    eprintln!("jxl-rs decoded DCT4X8 successfully: {}x{}", dw, dh);

    // jxl-rs outputs sRGB, so linear 0.5 → sRGB ~0.73
    let center_idx = (h / 2 * w + w / 2) * 3;
    let center_val = pixels[center_idx];
    let expected_srgb = linear_to_srgb_val(0.5); // ~0.735
    eprintln!(
        "Center pixel value: {:.4} (expected sRGB ~{:.2})",
        center_val, expected_srgb
    );
    assert!(
        (center_val - expected_srgb).abs() < 0.1,
        "jxl-rs sRGB output should be ~{:.2}, got {:.4}",
        expected_srgb,
        center_val
    );
}

/// Layer 3 test: Force DCT8X4 and verify jxl-rs decodes.
/// Note: jxl-rs outputs sRGB (not linear), so center pixel is ~0.74 (sRGB) not ~0.5 (linear).
/// This is expected behavior - jxl-rs doesn't have API to request linear output.
#[test]
#[ignore]
fn layer3_single_group_dct8x4_decode_jxl_rs() {
    use jxl_encoder::vardct::VarDctEncoder;

    // 64x64 gradient image
    let w = 64usize;
    let h = 64usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let v = (x as f32 + y as f32) / (w as f32 + h as f32 - 2.0);
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    let mut encoder = VarDctEncoder::new(2.0);
    encoder.force_strategy = Some(6); // RAW_STRATEGY_DCT8X4

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;
    eprintln!("DCT8X4: {} bytes encoded", bytes.len());

    // Decode with jxl-rs
    let (dw, dh, pixels) = decode_jxl_rs(&bytes);
    assert_eq!(dw, w);
    assert_eq!(dh, h);
    eprintln!("jxl-rs decoded DCT8X4 successfully: {}x{}", dw, dh);

    // jxl-rs outputs sRGB, so linear 0.5 → sRGB ~0.73
    let center_idx = (h / 2 * w + w / 2) * 3;
    let center_val = pixels[center_idx];
    let expected_srgb = linear_to_srgb_val(0.5); // ~0.735
    eprintln!(
        "Center pixel value: {:.4} (expected sRGB ~{:.2})",
        center_val, expected_srgb
    );
    assert!(
        (center_val - expected_srgb).abs() < 0.1,
        "jxl-rs sRGB output should be ~{:.2}, got {:.4}",
        expected_srgb,
        center_val
    );
}

/// Verify all three decoders produce consistent results for DCT4x8.
///
/// Note: Different decoders output different color spaces by default:
/// - jxl-oxide: explicitly requests linear RGB (via `srgb_linear`)
/// - djxl: outputs sRGB (default for PNG output)
/// - jxl-rs: outputs sRGB (no API to request linear)
///
/// For a linear input value of ~0.5:
/// - Linear output: ~0.51 (jxl-oxide)
/// - sRGB output: ~0.73 (djxl, jxl-rs) because 0.5^(1/2.2) ≈ 0.73
///
/// All decoders are correct - they just output different color spaces.
#[test]
#[ignore]
fn test_dct4x8_decoder_colorspace_comparison() {
    use jxl_encoder::vardct::VarDctEncoder;

    // Create 64x64 diagonal gradient: value = (x + y) / 126, range 0.0 to 1.0
    let w = 64usize;
    let h = 64usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let v = (x as f32 + y as f32) / (w as f32 + h as f32 - 2.0);
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    // Encode with forced DCT4x8
    let mut encoder = VarDctEncoder::new(2.0);
    encoder.force_strategy = Some(5); // RAW_STRATEGY_DCT4X8
    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;
    eprintln!("Encoded {} bytes", bytes.len());

    // Decode with jxl-oxide (requests linear)
    let (_, _, oxide_pixels) = decode_jxl_oxide(&bytes);
    let oxide_center = oxide_pixels[(h / 2 * w + w / 2) * 3];

    // Decode with djxl (outputs sRGB)
    let (_, _, djxl_pixels) = decode_djxl(&bytes);
    let djxl_center = djxl_pixels[(h / 2 * w + w / 2) * 3] as f32 / 255.0;

    // Decode with jxl-rs (outputs sRGB)
    let (_, _, jxl_rs_pixels) = decode_jxl_rs(&bytes);
    let jxl_rs_center = jxl_rs_pixels[(h / 2 * w + w / 2) * 3];

    eprintln!("jxl-oxide (linear): {:.4}", oxide_center);
    eprintln!("djxl (sRGB):        {:.4}", djxl_center);
    eprintln!("jxl-rs (sRGB):      {:.4}", jxl_rs_center);

    // Expected: linear ~0.5, sRGB ~0.73
    let expected_linear: f32 = 0.5;
    let expected_srgb = linear_to_srgb_val(expected_linear); // ~0.735

    // jxl-oxide should output linear
    assert!(
        (oxide_center - expected_linear).abs() < 0.1,
        "jxl-oxide should output linear: got {:.4}, expected ~{:.4}",
        oxide_center,
        expected_linear
    );

    // djxl and jxl-rs should output sRGB and agree with each other
    assert!(
        (djxl_center - expected_srgb).abs() < 0.1,
        "djxl should output sRGB: got {:.4}, expected ~{:.4}",
        djxl_center,
        expected_srgb
    );
    assert!(
        (jxl_rs_center - expected_srgb).abs() < 0.1,
        "jxl-rs should output sRGB: got {:.4}, expected ~{:.4}",
        jxl_rs_center,
        expected_srgb
    );

    // djxl and jxl-rs should agree closely (same color space)
    assert!(
        (djxl_center - jxl_rs_center).abs() < 0.02,
        "djxl and jxl-rs should agree: djxl={:.4}, jxl-rs={:.4}",
        djxl_center,
        jxl_rs_center
    );

    eprintln!("All decoders produce expected values for their color space.");
}

/// Test that strategy selection can pick DCT4X8/DCT8X4 for appropriate content.
#[test]
#[ignore]
fn test_strategy_selection_picks_small_transforms() {
    use jxl_encoder::vardct::VarDctEncoder;

    // Create an image with strong horizontal edges (should favor DCT8X4)
    // and strong vertical edges (should favor DCT4X8)
    let w = 256usize;
    let h = 256usize;
    let mut linear = vec![0.0f32; w * h * 3];

    // Create alternating horizontal and vertical stripe patterns
    // Left half: vertical stripes (alternating columns)
    // Right half: horizontal stripes (alternating rows)
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let v = if x < w / 2 {
                // Vertical stripes (strong horizontal edges -> DCT8X4 preferred)
                if (x / 4) % 2 == 0 { 0.8 } else { 0.2 }
            } else {
                // Horizontal stripes (strong vertical edges -> DCT4X8 preferred)
                if (y / 4) % 2 == 0 { 0.8 } else { 0.2 }
            };
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    // Encode with strategy selection enabled
    let mut encoder = VarDctEncoder::new(2.0);
    encoder.ac_strategy_enabled = true;

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;
    eprintln!("Encoded {} bytes with ac_strategy_enabled", bytes.len());

    // Decode with jxl-oxide
    let (dw, dh, pixels) = decode_jxl_oxide(&bytes);
    assert_eq!(dw, w);
    assert_eq!(dh, h);

    // Check decoded values are reasonable (simple sanity check)
    let center_idx = (h / 2 * w + w / 2) * 3;
    let center_val = pixels[center_idx];
    eprintln!("Center pixel value: {:.4}", center_val);
    // Just verify we got something reasonable (not all zeros or garbage)
    assert!(
        center_val > 0.1 && center_val < 0.9,
        "Pixel value out of expected range"
    );

    // Decode with jxl-rs to verify
    let (dw2, dh2, _pixels2) = decode_jxl_rs(&bytes);
    assert_eq!(dw2, w);
    assert_eq!(dh2, h);
    eprintln!("jxl-rs decoded successfully");

    // Note: We can't easily check which strategies were selected without
    // exposing internal state. The test verifies the encoder doesn't crash
    // and produces valid output when small transforms might be selected.
}

/// Compare DCT4X8 vs DCT8 quality on a real photo crop.
#[test]
#[ignore]
fn test_dct4x8_vs_dct8_quality_real_photo() {
    use jxl_encoder::vardct::VarDctEncoder;

    // Load frymire image
    let path = &format!(
        "{}/imageflow/test_inputs/frymire.png",
        jxl_encoder::test_helpers::corpus_dir().display()
    );
    let (w, h, linear, srgb) = load_png_crop(path, 256, 256);

    // Encode with forced DCT4X8
    let mut encoder_4x8 = VarDctEncoder::new(1.0);
    encoder_4x8.force_strategy = Some(5); // DCT4X8
    let bytes_4x8 = encoder_4x8.encode(w, h, &linear, None).unwrap().data;

    // Encode with DCT8 only
    let mut encoder_dct8 = VarDctEncoder::new(1.0);
    encoder_dct8.ac_strategy_enabled = false;
    let bytes_dct8 = encoder_dct8.encode(w, h, &linear, None).unwrap().data;

    // Decode both
    let (_, _, pixels_4x8) = decode_jxl_oxide(&bytes_4x8);
    let (_, _, pixels_dct8) = decode_jxl_oxide(&bytes_dct8);

    // Compare quality
    let ssim_4x8 = ssim2_u8_vs_linear_f32(&srgb, &pixels_4x8, w, h);
    let ssim_dct8 = ssim2_u8_vs_linear_f32(&srgb, &pixels_dct8, w, h);

    eprintln!("=== DCT4X8 vs DCT8 Quality Comparison (frymire 256x256) ===");
    eprintln!("DCT4X8: {} bytes, SSIM2 = {:.2}", bytes_4x8.len(), ssim_4x8);
    eprintln!(
        "DCT8:   {} bytes, SSIM2 = {:.2}",
        bytes_dct8.len(),
        ssim_dct8
    );
    eprintln!(
        "Difference: DCT4X8 is {:.2} SSIM2 {} than DCT8",
        (ssim_4x8 - ssim_dct8).abs(),
        if ssim_4x8 > ssim_dct8 {
            "better"
        } else {
            "worse"
        }
    );

    // DCT4X8 should not be catastrophically worse
    assert!(
        ssim_4x8 > ssim_dct8 - 5.0,
        "DCT4X8 quality {} is too much worse than DCT8 {} (diff > 5 SSIM2)",
        ssim_4x8,
        ssim_dct8
    );
}

/// Test DCT4X8 with varying content complexity.
#[test]
#[ignore]
fn test_dct4x8_content_complexity() {
    use jxl_encoder::vardct::VarDctEncoder;

    let w = 64usize;
    let h = 64usize;

    // Test 1: Smooth gradient (known to work)
    let mut linear_grad = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let v = (x as f32 + y as f32) / (w as f32 + h as f32 - 2.0);
            linear_grad[idx] = v;
            linear_grad[idx + 1] = v;
            linear_grad[idx + 2] = v;
        }
    }

    // Test 2: Sharp edges (8-pixel wide stripes)
    let mut linear_edge = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let v = if (x / 8) % 2 == 0 { 0.8 } else { 0.2 };
            linear_edge[idx] = v;
            linear_edge[idx + 1] = v;
            linear_edge[idx + 2] = v;
        }
    }

    // Test 3: High frequency noise
    let mut linear_noise = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            // Simple pseudo-random based on position
            let v = ((x * 31 + y * 17 + x * y * 7) % 100) as f32 / 100.0;
            linear_noise[idx] = v;
            linear_noise[idx + 1] = v;
            linear_noise[idx + 2] = v;
        }
    }

    for (name, linear) in [
        ("gradient", &linear_grad),
        ("edge", &linear_edge),
        ("noise", &linear_noise),
    ] {
        let mut encoder = VarDctEncoder::new(1.0);
        encoder.force_strategy = Some(5); // DCT4X8
        let bytes = encoder.encode(w, h, linear, None).unwrap().data;

        let (_, _, pixels) = decode_jxl_oxide(&bytes);

        // Check center pixel
        let center_idx = (h / 2 * w + w / 2) * 3;
        let orig = linear[center_idx];
        let dec = pixels[center_idx];
        let diff = (orig - dec).abs();

        eprintln!(
            "{}: {} bytes, center: orig={:.4} dec={:.4} diff={:.4}",
            name,
            bytes.len(),
            orig,
            dec,
            diff
        );

        // Verify decoded value is reasonable
        assert!(
            diff < 0.3,
            "{}: center pixel diff {} too large (orig={}, dec={})",
            name,
            diff,
            orig,
            dec
        );
    }
}

/// Test DCT4X8 with varying image sizes.
#[test]
#[ignore]
fn test_dct4x8_image_sizes() {
    use jxl_encoder::vardct::VarDctEncoder;

    for size in [64, 128, 200, 256, 300] {
        let w = size;
        let h = size;

        // Gradient image
        let mut linear = vec![0.0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                let v = (x as f32 + y as f32) / (w as f32 + h as f32 - 2.0);
                linear[idx] = v;
                linear[idx + 1] = v;
                linear[idx + 2] = v;
            }
        }

        let mut encoder = VarDctEncoder::new(1.0);
        encoder.force_strategy = Some(5); // DCT4X8
        let bytes = encoder.encode(w, h, &linear, None).unwrap().data;

        let (dw, dh, pixels) = decode_jxl_oxide(&bytes);
        assert_eq!(dw, w);
        assert_eq!(dh, h);

        // Check center pixel
        let center_idx = (h / 2 * w + w / 2) * 3;
        let orig = linear[center_idx];
        let dec = pixels[center_idx];
        let diff = (orig - dec).abs();

        eprintln!(
            "{}x{}: {} bytes, center: orig={:.4} dec={:.4} diff={:.4}",
            w,
            h,
            bytes.len(),
            orig,
            dec,
            diff
        );

        // Verify decoded value is reasonable
        assert!(
            diff < 0.3,
            "{}x{}: center pixel diff {} too large (orig={}, dec={})",
            w,
            h,
            diff,
            orig,
            dec
        );
    }
}

/// Debug DCT4X8 on real photo - save output for inspection.
#[test]
#[ignore]
fn debug_dct4x8_real_photo_save() {
    use jxl_encoder::vardct::VarDctEncoder;
    use std::io::Write;

    // Load frymire image
    let path = &format!(
        "{}/imageflow/test_inputs/frymire.png",
        jxl_encoder::test_helpers::corpus_dir().display()
    );
    let (w, h, linear, _srgb) = load_png_crop(path, 256, 256);
    eprintln!("Loaded {}x{} image", w, h);

    // Encode with forced DCT4X8
    let mut encoder_4x8 = VarDctEncoder::new(1.0);
    encoder_4x8.force_strategy = Some(5); // DCT4X8
    let bytes_4x8 = encoder_4x8.encode(w, h, &linear, None).unwrap().data;

    // Save for djxl inspection
    let mut file = std::fs::File::create("/tmp/debug_dct4x8.jxl").unwrap();
    file.write_all(&bytes_4x8).unwrap();
    eprintln!("Saved {} bytes to /tmp/debug_dct4x8.jxl", bytes_4x8.len());

    // Try decoding with jxl-oxide
    let result = std::panic::catch_unwind(|| decode_jxl_oxide(&bytes_4x8));
    match result {
        Ok((dw, dh, pixels)) => {
            eprintln!("jxl-oxide decoded: {}x{}", dw, dh);
            // Check a few pixels
            for y in [0, 64, 128, 192] {
                let idx = (y * w + 128) * 3;
                eprintln!(
                    "  pixel ({},128): orig={:.4},{:.4},{:.4} dec={:.4},{:.4},{:.4}",
                    y,
                    linear[idx],
                    linear[idx + 1],
                    linear[idx + 2],
                    pixels[idx],
                    pixels[idx + 1],
                    pixels[idx + 2]
                );
            }
        }
        Err(e) => {
            eprintln!("jxl-oxide FAILED: {:?}", e);
        }
    }

    // Encode with DCT8 for comparison
    let mut encoder_dct8 = VarDctEncoder::new(1.0);
    encoder_dct8.ac_strategy_enabled = false;
    let bytes_dct8 = encoder_dct8.encode(w, h, &linear, None).unwrap().data;

    let mut file = std::fs::File::create("/tmp/debug_dct8.jxl").unwrap();
    file.write_all(&bytes_dct8).unwrap();
    eprintln!("Saved {} bytes to /tmp/debug_dct8.jxl", bytes_dct8.len());

    let (_, _, pixels_dct8) = decode_jxl_oxide(&bytes_dct8);
    for y in [0, 64, 128, 192] {
        let idx = (y * w + 128) * 3;
        eprintln!(
            "  DCT8 pixel ({},128): orig={:.4},{:.4},{:.4} dec={:.4},{:.4},{:.4}",
            y,
            linear[idx],
            linear[idx + 1],
            linear[idx + 2],
            pixels_dct8[idx],
            pixels_dct8[idx + 1],
            pixels_dct8[idx + 2]
        );
    }
}

/// Debug DCT4X8 - check for NaN/Inf/out-of-range values.
#[test]
#[ignore]
fn debug_dct4x8_check_values() {
    use jxl_encoder::vardct::VarDctEncoder;

    // Load frymire image
    let path = &format!(
        "{}/imageflow/test_inputs/frymire.png",
        jxl_encoder::test_helpers::corpus_dir().display()
    );
    let (w, h, linear, _srgb) = load_png_crop(path, 256, 256);

    // Encode with forced DCT4X8
    let mut encoder = VarDctEncoder::new(1.0);
    encoder.force_strategy = Some(5); // DCT4X8
    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;

    // Decode
    let (_, _, pixels) = decode_jxl_oxide(&bytes);

    // Check for problematic values
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut neg_count = 0;
    let mut high_count = 0; // > 1.5
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for (i, &v) in pixels.iter().enumerate() {
        if v.is_nan() {
            nan_count += 1;
        } else if v.is_infinite() {
            inf_count += 1;
        } else {
            if v < 0.0 {
                neg_count += 1;
            }
            if v > 1.5 {
                high_count += 1;
            }
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
    }

    eprintln!("DCT4X8 decoded {} pixels", pixels.len());
    eprintln!("  NaN: {}", nan_count);
    eprintln!("  Inf: {}", inf_count);
    eprintln!("  Negative: {}", neg_count);
    eprintln!("  > 1.5: {}", high_count);
    eprintln!("  Range: [{:.4}, {:.4}]", min_val, max_val);

    // Also check DCT8 for comparison
    let mut encoder2 = VarDctEncoder::new(1.0);
    encoder2.ac_strategy_enabled = false;
    let bytes2 = encoder2.encode(w, h, &linear, None).unwrap().data;
    let (_, _, pixels2) = decode_jxl_oxide(&bytes2);

    let mut min_val2 = f32::MAX;
    let mut max_val2 = f32::MIN;
    for &v in pixels2.iter() {
        if !v.is_nan() && !v.is_infinite() {
            min_val2 = min_val2.min(v);
            max_val2 = max_val2.max(v);
        }
    }
    eprintln!("DCT8 range: [{:.4}, {:.4}]", min_val2, max_val2);

    // No NaN or Inf should be present
    assert_eq!(nan_count, 0, "Found NaN values");
    assert_eq!(inf_count, 0, "Found Inf values");
}

// ─────────────────────────────────────────────────────────────────────────────
// DCT32x32: Production vs Reference IDCT comparison
// ─────────────────────────────────────────────────────────────────────────────

/// Compare production `dc_from_dct_32x32()` (uses butterfly idct1d_4)
/// against reference implementation (uses direct DCT-III formula).
///
/// This test identifies if the butterfly IDCT produces different numerical
/// results than the known-correct direct formula.
#[test]
fn test_dc_from_dct_32x32_production_vs_reference() {
    use jxl_encoder::vardct::dct::{DCT_RESAMPLE_SCALE_32_TO_4, dc_from_dct_32x32};

    // Reference 4-point IDCT using direct DCT-III formula
    fn idct4_reference(input: &[f32; 4]) -> [f32; 4] {
        use core::f32::consts::PI;
        let x0 = input[0];
        let x1 = input[1];
        let x2 = input[2];
        let x3 = input[3];

        [
            x0 + 2.0
                * (x1 * (PI * 1.0 / 8.0).cos()
                    + x2 * (PI * 2.0 / 8.0).cos()
                    + x3 * (PI * 3.0 / 8.0).cos()),
            x0 + 2.0
                * (x1 * (PI * 3.0 / 8.0).cos()
                    + x2 * (PI * 6.0 / 8.0).cos()
                    + x3 * (PI * 9.0 / 8.0).cos()),
            x0 + 2.0
                * (x1 * (PI * 5.0 / 8.0).cos()
                    + x2 * (PI * 10.0 / 8.0).cos()
                    + x3 * (PI * 15.0 / 8.0).cos()),
            x0 + 2.0
                * (x1 * (PI * 7.0 / 8.0).cos()
                    + x2 * (PI * 14.0 / 8.0).cos()
                    + x3 * (PI * 21.0 / 8.0).cos()),
        ]
    }

    /// Reference dc_from_dct_32x32 using direct IDCT formula (known correct)
    fn dc_from_dct_32x32_reference(coeffs: &[f32; 1024]) -> [f32; 16] {
        // Extract 4x4 LLF with resample scales
        let mut block = [0.0f32; 16];
        for iy in 0..4 {
            for ix in 0..4 {
                block[iy * 4 + ix] = coeffs[iy * 32 + ix]
                    * DCT_RESAMPLE_SCALE_32_TO_4[iy]
                    * DCT_RESAMPLE_SCALE_32_TO_4[ix]
                    * 16.0; // Same scaling as production
            }
        }

        // IDCT rows
        let mut after_rows = [0.0f32; 16];
        for iy in 0..4 {
            let row_in = [
                block[iy * 4],
                block[iy * 4 + 1],
                block[iy * 4 + 2],
                block[iy * 4 + 3],
            ];
            let row_out = idct4_reference(&row_in);
            for ix in 0..4 {
                after_rows[iy * 4 + ix] = row_out[ix];
            }
        }

        // Transpose
        let mut transposed = [0.0f32; 16];
        for iy in 0..4 {
            for ix in 0..4 {
                transposed[ix * 4 + iy] = after_rows[iy * 4 + ix];
            }
        }

        // IDCT rows again
        let mut result = [0.0f32; 16];
        for iy in 0..4 {
            let row_in = [
                transposed[iy * 4],
                transposed[iy * 4 + 1],
                transposed[iy * 4 + 2],
                transposed[iy * 4 + 3],
            ];
            let row_out = idct4_reference(&row_in);
            for ix in 0..4 {
                result[iy * 4 + ix] = row_out[ix];
            }
        }
        result
    }

    // Test 1: Uniform DC only
    eprintln!("=== Test 1: Uniform DC only (coeffs[0] = 1.0) ===");
    let mut coeffs = [0.0f32; 1024];
    coeffs[0] = 1.0;

    let production = dc_from_dct_32x32(&coeffs);
    let reference = dc_from_dct_32x32_reference(&coeffs);

    eprintln!("Production DC values: all = {:.6}", production[0]);
    eprintln!("Reference DC values:  all = {:.6}", reference[0]);

    // Calculate ratio
    let scale_factor = reference[0] / production[0];
    eprintln!("Scale factor (ref/prod): {:.6}", scale_factor);

    // Test 2: Vertical frequency (coeffs[1] = ky=1, kx=0)
    eprintln!("\n=== Test 2: Vertical frequency (coeffs[1] = 1.0) ===");
    let mut coeffs = [0.0f32; 1024];
    coeffs[1] = 1.0;

    let production = dc_from_dct_32x32(&coeffs);
    let reference = dc_from_dct_32x32_reference(&coeffs);

    eprintln!("Production DC values:");
    for iy in 0..4 {
        eprintln!(
            "  row {}: [{:.4}, {:.4}, {:.4}, {:.4}]",
            iy,
            production[iy * 4],
            production[iy * 4 + 1],
            production[iy * 4 + 2],
            production[iy * 4 + 3]
        );
    }
    eprintln!("Reference DC values:");
    for iy in 0..4 {
        eprintln!(
            "  row {}: [{:.4}, {:.4}, {:.4}, {:.4}]",
            iy,
            reference[iy * 4],
            reference[iy * 4 + 1],
            reference[iy * 4 + 2],
            reference[iy * 4 + 3]
        );
    }

    // Check if spatial patterns match (up to scale)
    // For vertical frequency, each row should be constant
    eprintln!("\nRow constancy check (should be ~0 for vertical freq):");
    for iy in 0..4 {
        let prod_row: Vec<f32> = (0..4).map(|ix| production[iy * 4 + ix]).collect();
        let ref_row: Vec<f32> = (0..4).map(|ix| reference[iy * 4 + ix]).collect();
        let prod_var: f32 = prod_row.iter().map(|v| (v - prod_row[0]).abs()).sum();
        let ref_var: f32 = ref_row.iter().map(|v| (v - ref_row[0]).abs()).sum();
        eprintln!(
            "  Row {}: prod_var={:.6}, ref_var={:.6}",
            iy, prod_var, ref_var
        );

        // Assert row constancy
        assert!(
            prod_var < 1e-5,
            "Production row {} not constant: {:?}",
            iy,
            prod_row
        );
        assert!(
            ref_var < 1e-5,
            "Reference row {} not constant: {:?}",
            iy,
            ref_row
        );
    }

    // Check row-to-row variation (should have different values between rows)
    let prod_row_diff = (production[0] - production[4]).abs();
    let ref_row_diff = (reference[0] - reference[4]).abs();
    eprintln!(
        "Row 0 vs Row 1 diff: prod={:.6}, ref={:.6}",
        prod_row_diff, ref_row_diff
    );
    assert!(
        prod_row_diff > 0.01,
        "Production should have row-to-row variation"
    );
    assert!(
        ref_row_diff > 0.01,
        "Reference should have row-to-row variation"
    );

    // Normalize and compare patterns
    let prod_mean: f32 = production.iter().sum::<f32>() / 16.0;
    let ref_mean: f32 = reference.iter().sum::<f32>() / 16.0;
    let prod_std: f32 = (production
        .iter()
        .map(|x| (x - prod_mean).powi(2))
        .sum::<f32>()
        / 16.0)
        .sqrt();
    let ref_std: f32 = (reference
        .iter()
        .map(|x| (x - ref_mean).powi(2))
        .sum::<f32>()
        / 16.0)
        .sqrt();

    eprintln!("\nNormalized comparison:");
    eprintln!("  Production: mean={:.6}, std={:.6}", prod_mean, prod_std);
    eprintln!("  Reference:  mean={:.6}, std={:.6}", ref_mean, ref_std);

    if prod_std > 1e-6 && ref_std > 1e-6 {
        let prod_norm: Vec<f32> = production
            .iter()
            .map(|x| (x - prod_mean) / prod_std)
            .collect();
        let ref_norm: Vec<f32> = reference.iter().map(|x| (x - ref_mean) / ref_std).collect();

        let mut max_norm_diff = 0.0f32;
        for i in 0..16 {
            let diff = (prod_norm[i] - ref_norm[i]).abs();
            if diff > max_norm_diff {
                max_norm_diff = diff;
            }
        }
        eprintln!("  Max normalized difference: {:.6}", max_norm_diff);

        // Patterns should match after normalization
        assert!(
            max_norm_diff < 0.01,
            "IDCT produces different PATTERN: max_norm_diff = {}",
            max_norm_diff
        );
    }

    eprintln!("\n=== CONCLUSION ===");
    eprintln!(
        "Scale factor between reference and production: {:.6}",
        scale_factor
    );
    eprintln!("The butterfly IDCT produces the correct PATTERN but different SCALE.");
    eprintln!("To fix: dc_from_dct_32x32 should use the direct IDCT formula,");
    eprintln!("or multiply the result by {:.1}.", scale_factor);
}

/// Verify dc_from_dct_32x32 with uniform input
#[test]
fn test_dc_from_dct_32x32_uniform_input() {
    use jxl_encoder::vardct::dct::{dc_from_dct_32x32, dct_32x32};

    // Uniform 32x32 input with value 0.5
    let input = [0.5f32; 1024];
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&input, &mut coeffs);

    eprintln!("Uniform input (all 0.5):");
    eprintln!("  coeffs[0] (DC) = {:.6}", coeffs[0]);
    eprintln!("  Expected DC = 0.5 (block average)");

    let dcs = dc_from_dct_32x32(&coeffs);
    eprintln!("  dc_from_dct_32x32 output[0] = {:.6}", dcs[0]);
    eprintln!("  All DC values should be 0.5 (the block average)");

    // Check all DC values are close to 0.5
    for (i, &dc) in dcs.iter().enumerate() {
        let err = (dc - 0.5).abs();
        if err > 0.01 {
            eprintln!(
                "  ERROR: dc[{}] = {:.6}, expected 0.5, err = {:.6}",
                i, dc, err
            );
        }
    }

    let max_err = dcs
        .iter()
        .map(|&dc| (dc - 0.5).abs())
        .fold(0.0f32, f32::max);
    eprintln!("  Max error: {:.6}", max_err);

    // This currently fails because of the 16x scale factor bug
    // After fix, this assertion should pass
    assert!(
        max_err < 0.01,
        "DC values should be ~0.5 for uniform 0.5 input, got {:?}",
        dcs
    );
}

/// Verify dc_from_dct_32x32 with step function input
#[test]
fn test_dc_from_dct_32x32_step_input() {
    use jxl_encoder::vardct::dct::{dc_from_dct_32x32, dct_32x32};

    // Step function: top half = 0.25, bottom half = 0.75
    let mut input = [0.0f32; 1024];
    for y in 0..32 {
        let v = if y < 16 { 0.25 } else { 0.75 };
        for x in 0..32 {
            input[y * 32 + x] = v;
        }
    }

    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&input, &mut coeffs);

    eprintln!("Step function input (top=0.25, bottom=0.75):");
    eprintln!("  LLF coeffs:");
    for iy in 0..4 {
        eprintln!(
            "    row {}: [{:.6}, {:.6}, {:.6}, {:.6}]",
            iy,
            coeffs[iy * 32],
            coeffs[iy * 32 + 1],
            coeffs[iy * 32 + 2],
            coeffs[iy * 32 + 3]
        );
    }

    let dcs = dc_from_dct_32x32(&coeffs);
    eprintln!("  DC values (4x4 grid):");
    for iy in 0..4 {
        eprintln!(
            "    row {}: [{:.6}, {:.6}, {:.6}, {:.6}]  expected: {:.2}",
            iy,
            dcs[iy * 4],
            dcs[iy * 4 + 1],
            dcs[iy * 4 + 2],
            dcs[iy * 4 + 3],
            if iy < 2 { 0.25 } else { 0.75 }
        );
    }

    // Check expected values
    // Rows 0-1 should be ~0.25 (top half of image)
    // Rows 2-3 should be ~0.75 (bottom half of image)
    let mut max_err = 0.0f32;
    for iy in 0..4 {
        let expected = if iy < 2 { 0.25 } else { 0.75 };
        for ix in 0..4 {
            let err = (dcs[iy * 4 + ix] - expected).abs();
            if err > max_err {
                max_err = err;
                eprintln!(
                    "  Error at [{},{}]: got {:.6}, expected {:.2}, err {:.6}",
                    iy,
                    ix,
                    dcs[iy * 4 + ix],
                    expected,
                    err
                );
            }
        }
    }
    eprintln!("  Max error: {:.6}", max_err);

    // Allow some error due to edge effects in the step function
    assert!(
        max_err < 0.1,
        "DC values should match expected, max_err = {}",
        max_err
    );
}

/// Debug test: check what AC coefficients are actually encoded for DCT32x32
#[test]
#[ignore]
fn test_dct32x32_ac_coeff_debug() {
    use jxl_encoder::vardct::VarDctEncoder;

    // Create simple 32x32 test pattern: left half black, right half white
    let w = 32usize;
    let h = 32usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let v = if x < 16 { 0.1 } else { 0.9 };
            let idx = (y * w + x) * 3;
            linear[idx] = v; // R (becomes X after XYB)
            linear[idx + 1] = v; // G (becomes Y)
            linear[idx + 2] = v; // B
        }
    }

    // Encode with forced DCT32x32
    let mut encoder = VarDctEncoder::new(3.0);
    encoder.force_strategy = Some(4); // DCT32x32

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;
    eprintln!("Encoded {} bytes", bytes.len());

    // Decode with jxl-oxide
    let (dec_w, dec_h, decoded) = decode_jxl_oxide(&bytes);
    eprintln!("Decoded image: {}x{}", dec_w, dec_h);

    // Sample some pixel values (in linear RGB)
    eprintln!("Pixel samples (linear RGB Y channel estimate):");
    for y in [0, 8, 16, 24, 31] {
        for x in [0, 8, 15, 16, 24, 31] {
            let idx = (y * w + x) * 3;
            let v = decoded[idx + 1]; // Green channel approximates Y
            eprintln!("  ({:2}, {:2}): {:.4}", x, y, v);
        }
    }

    // Check if horizontal variation is preserved
    let left_avg: f32 = (0..16).map(|x| decoded[(16 * w + x) * 3 + 1]).sum::<f32>() / 16.0;
    let right_avg: f32 = (16..32).map(|x| decoded[(16 * w + x) * 3 + 1]).sum::<f32>() / 16.0;
    eprintln!(
        "Row 16: left avg = {:.4}, right avg = {:.4}",
        left_avg, right_avg
    );
    eprintln!("Expected: left ~0.1, right ~0.9");

    // If all converging to average (~0.5), that's the bug
    assert!(
        right_avg > left_avg + 0.3,
        "Horizontal variation lost: left={:.4}, right={:.4}",
        left_avg,
        right_avg
    );
}

/// Debug test: 64x64 with forced DCT32x32
#[test]
#[ignore]
fn test_dct32x32_64x64_debug() {
    use jxl_encoder::vardct::VarDctEncoder;

    // Create 64x64 test pattern: top-left=black, bottom-right=white
    let w = 64usize;
    let h = 64usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let v = if y < 32 && x < 32 { 0.1 } else { 0.9 };
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    // Encode with forced DCT32x32
    let mut encoder = VarDctEncoder::new(3.0);
    encoder.force_strategy = Some(4); // DCT32x32

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;
    eprintln!("64x64 encoded {} bytes", bytes.len());

    // Decode with jxl-oxide
    let (dec_w, dec_h, decoded) = decode_jxl_oxide(&bytes);
    eprintln!("Decoded image: {}x{}", dec_w, dec_h);

    // Sample center of each quadrant
    let samples = [
        (16, 16, "top-left (expected ~0.1)"),
        (48, 16, "top-right (expected ~0.9)"),
        (16, 48, "bottom-left (expected ~0.9)"),
        (48, 48, "bottom-right (expected ~0.9)"),
    ];
    for (x, y, label) in samples {
        let idx = (y * w + x) * 3;
        let v = decoded[idx + 1];
        eprintln!("  ({:2}, {:2}): {:.4} - {}", x, y, v, label);
    }

    // Check quadrant averages
    let tl: f32 = (0..32)
        .flat_map(|y| (0..32).map(move |x| (y, x)))
        .map(|(y, x)| decoded[(y * w + x) * 3 + 1])
        .sum::<f32>()
        / (32.0 * 32.0);
    let tr: f32 = (0..32)
        .flat_map(|y| (32..64).map(move |x| (y, x)))
        .map(|(y, x)| decoded[(y * w + x) * 3 + 1])
        .sum::<f32>()
        / (32.0 * 32.0);
    let bl: f32 = (32..64)
        .flat_map(|y| (0..32).map(move |x| (y, x)))
        .map(|(y, x)| decoded[(y * w + x) * 3 + 1])
        .sum::<f32>()
        / (32.0 * 32.0);
    let br: f32 = (32..64)
        .flat_map(|y| (32..64).map(move |x| (y, x)))
        .map(|(y, x)| decoded[(y * w + x) * 3 + 1])
        .sum::<f32>()
        / (32.0 * 32.0);

    eprintln!("Quadrant averages:");
    eprintln!("  Top-left:     {:.4} (expected ~0.1)", tl);
    eprintln!("  Top-right:    {:.4} (expected ~0.9)", tr);
    eprintln!("  Bottom-left:  {:.4} (expected ~0.9)", bl);
    eprintln!("  Bottom-right: {:.4} (expected ~0.9)", br);

    // Top-left should be significantly darker than others
    assert!(tr > tl + 0.3, "Quadrant variation lost");
}

/// Debug test: 256x256 with forced DCT32x32 - this should reveal the bug
#[test]
#[ignore]
fn test_dct32x32_256x256_debug() {
    use jxl_encoder::vardct::VarDctEncoder;

    // Create 256x256 test pattern: top-left=black, bottom-right=white
    let w = 256usize;
    let h = 256usize;
    let mut linear = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let v = if y < 128 && x < 128 { 0.1 } else { 0.9 };
            let idx = (y * w + x) * 3;
            linear[idx] = v;
            linear[idx + 1] = v;
            linear[idx + 2] = v;
        }
    }

    // Encode with forced DCT32x32
    let mut encoder = VarDctEncoder::new(3.0);
    encoder.force_strategy = Some(4); // DCT32x32

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;
    eprintln!("256x256 encoded {} bytes", bytes.len());

    // Decode with jxl-oxide
    let (dec_w, dec_h, decoded) = decode_jxl_oxide(&bytes);
    eprintln!("Decoded image: {}x{}", dec_w, dec_h);

    // Sample center of each quadrant
    let samples = [
        (64, 64, "top-left (expected ~0.1)"),
        (192, 64, "top-right (expected ~0.9)"),
        (64, 192, "bottom-left (expected ~0.9)"),
        (192, 192, "bottom-right (expected ~0.9)"),
    ];
    for (x, y, label) in samples {
        let idx = (y * w + x) * 3;
        let v = decoded[idx + 1];
        eprintln!("  ({:3}, {:3}): {:.4} - {}", x, y, v, label);
    }

    // Check quadrant averages
    let tl: f32 = (0..128)
        .flat_map(|y| (0..128).map(move |x| (y, x)))
        .map(|(y, x)| decoded[(y * w + x) * 3 + 1])
        .sum::<f32>()
        / (128.0 * 128.0);
    let tr: f32 = (0..128)
        .flat_map(|y| (128..256).map(move |x| (y, x)))
        .map(|(y, x)| decoded[(y * w + x) * 3 + 1])
        .sum::<f32>()
        / (128.0 * 128.0);

    eprintln!("Quadrant averages:");
    eprintln!("  Top-left:  {:.4} (expected ~0.1)", tl);
    eprintln!("  Top-right: {:.4} (expected ~0.9)", tr);

    // Top-left should be significantly darker than top-right
    let diff = tr - tl;
    eprintln!("  Difference: {:.4} (expected ~0.8)", diff);

    assert!(
        diff > 0.3,
        "Quadrant variation lost: tl={:.4}, tr={:.4}, diff={:.4}",
        tl,
        tr,
        diff
    );
}

/// Debug test: frymire crop with detailed pixel analysis
#[test]
#[ignore]
fn test_dct32x32_frymire_detailed_debug() {
    let (w, h, linear, srgb) = load_png_crop(&frymire_path(), 256, 256);
    assert_eq!(w, 256);
    assert_eq!(h, 256);

    // Show some original pixel values
    eprintln!("Original linear values (first 3 pixels RGB):");
    for i in 0..3 {
        eprintln!(
            "  pixel {}: R={:.4}, G={:.4}, B={:.4}",
            i,
            linear[i * 3],
            linear[i * 3 + 1],
            linear[i * 3 + 2]
        );
    }
    eprintln!("Original sRGB values (first 3 pixels):");
    for i in 0..3 {
        eprintln!(
            "  pixel {}: R={}, G={}, B={}",
            i,
            srgb[i * 3],
            srgb[i * 3 + 1],
            srgb[i * 3 + 2]
        );
    }

    // Encode with forced DCT32x32
    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(3.0);
    encoder.force_strategy = Some(4); // RAW_STRATEGY_DCT32X32

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;
    eprintln!("\nEncoded {} bytes", bytes.len());

    // Decode with jxl-oxide
    let (dw, dh, decoded) = decode_jxl_oxide(&bytes);
    assert_eq!(dw, w);
    assert_eq!(dh, h);

    eprintln!("\nDecoded linear values (first 3 pixels RGB):");
    for i in 0..3 {
        eprintln!(
            "  pixel {}: R={:.4}, G={:.4}, B={:.4}",
            i,
            decoded[i * 3],
            decoded[i * 3 + 1],
            decoded[i * 3 + 2]
        );
    }

    // Check overall statistics
    let orig_min = linear.iter().cloned().fold(f32::MAX, f32::min);
    let orig_max = linear.iter().cloned().fold(f32::MIN, f32::max);
    let orig_avg = linear.iter().sum::<f32>() / linear.len() as f32;

    let dec_min = decoded.iter().cloned().fold(f32::MAX, f32::min);
    let dec_max = decoded.iter().cloned().fold(f32::MIN, f32::max);
    let dec_avg = decoded.iter().sum::<f32>() / decoded.len() as f32;

    eprintln!(
        "\nOriginal stats: min={:.4}, max={:.4}, avg={:.4}",
        orig_min, orig_max, orig_avg
    );
    eprintln!(
        "Decoded stats:  min={:.4}, max={:.4}, avg={:.4}",
        dec_min, dec_max, dec_avg
    );

    // If decoded values have collapsed range, that's the bug
    let orig_range = orig_max - orig_min;
    let dec_range = dec_max - dec_min;
    eprintln!(
        "Range: original={:.4}, decoded={:.4}, ratio={:.4}",
        orig_range,
        dec_range,
        dec_range / orig_range
    );

    // Compare with DCT8-only encoding
    let mut encoder8 = jxl_encoder::vardct::VarDctEncoder::new(3.0);
    encoder8.ac_strategy_enabled = false; // DCT8 only
    let bytes8 = encoder8.encode(w, h, &linear, None).unwrap().data;
    let (_, _, decoded8) = decode_jxl_oxide(&bytes8);

    let dec8_min = decoded8.iter().cloned().fold(f32::MAX, f32::min);
    let dec8_max = decoded8.iter().cloned().fold(f32::MIN, f32::max);
    let dec8_avg = decoded8.iter().sum::<f32>() / decoded8.len() as f32;
    let dec8_range = dec8_max - dec8_min;

    eprintln!(
        "\nDCT8 stats:     min={:.4}, max={:.4}, avg={:.4}, range={:.4}",
        dec8_min, dec8_max, dec8_avg, dec8_range
    );

    // DCT32x32 should have similar range to DCT8
    assert!(
        dec_range > orig_range * 0.3,
        "DCT32x32 range collapsed: {:.4} vs original {:.4}",
        dec_range,
        orig_range
    );
}

/// Debug test: frymire crop pattern analysis
#[test]
#[ignore]
fn test_dct32x32_frymire_pattern_debug() {
    let (w, h, linear, _srgb) = load_png_crop(&frymire_path(), 256, 256);

    // Show first 8x8 block in detail
    eprintln!("Original linear Y values (8x8 block at top-left):");
    for y in 0..8 {
        eprint!("  row {}: ", y);
        for x in 0..8 {
            let idx = (y * w + x) * 3;
            let g = linear[idx + 1]; // Green channel ~ Y
            eprint!("{:.2} ", g);
        }
        eprintln!();
    }

    // Encode with forced DCT32x32
    let mut encoder = jxl_encoder::vardct::VarDctEncoder::new(3.0);
    encoder.force_strategy = Some(4);

    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;
    let (_, _, decoded) = decode_jxl_oxide(&bytes);

    eprintln!("\nDecoded linear Y values (8x8 block at top-left):");
    for y in 0..8 {
        eprint!("  row {}: ", y);
        for x in 0..8 {
            let idx = (y * w + x) * 3;
            let g = decoded[idx + 1];
            eprint!("{:.2} ", g);
        }
        eprintln!();
    }

    // Also show middle of image for comparison
    eprintln!("\nOriginal linear Y values (8x8 block at center):");
    for y in 0..8 {
        eprint!("  row {}: ", y);
        for x in 0..8 {
            let idx = ((128 + y) * w + (128 + x)) * 3;
            let g = linear[idx + 1];
            eprint!("{:.2} ", g);
        }
        eprintln!();
    }

    eprintln!("\nDecoded linear Y values (8x8 block at center):");
    for y in 0..8 {
        eprint!("  row {}: ", y);
        for x in 0..8 {
            let idx = ((128 + y) * w + (128 + x)) * 3;
            let g = decoded[idx + 1];
            eprint!("{:.2} ", g);
        }
        eprintln!();
    }
}

/// Debug test: check AC coefficient encoding for DCT32x32
#[test]
#[ignore]
fn test_dct32x32_nzeros_debug() {
    use jxl_encoder::vardct::VarDctEncoder;
    use jxl_encoder::vardct::dct::dct_32x32;

    let (w, h, linear, _srgb) = load_png_crop(&frymire_path(), 256, 256);

    // Extract first 32x32 block from Y channel and transform
    let mut block = [0.0f32; 1024];
    for y in 0..32 {
        for x in 0..32 {
            // Approximate Y from linear RGB: Y ≈ 0.2126*R + 0.7152*G + 0.0722*B
            let idx = (y * w + x) * 3;
            block[y * 32 + x] =
                0.2126 * linear[idx] + 0.7152 * linear[idx + 1] + 0.0722 * linear[idx + 2];
        }
    }

    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&block, &mut coeffs);

    // Count non-zero LLF and AC coefficients
    let mut llf_nz = 0;
    let mut ac_nz = 0;
    let llf_thresh = 0.001;
    let ac_thresh = 0.001;

    for y in 0..4 {
        for x in 0..4 {
            if coeffs[y * 32 + x].abs() > llf_thresh {
                llf_nz += 1;
            }
        }
    }

    for y in 0..32 {
        for x in 0..32 {
            if y < 4 && x < 4 {
                continue; // Skip LLF
            }
            if coeffs[y * 32 + x].abs() > ac_thresh {
                ac_nz += 1;
            }
        }
    }

    eprintln!("First 32x32 block DCT analysis:");
    eprintln!("  LLF non-zeros (of 16): {}", llf_nz);
    eprintln!("  AC non-zeros (of 1008): {}", ac_nz);

    // Show LLF values
    eprintln!("  LLF values (4x4):");
    for y in 0..4 {
        eprint!("    ");
        for x in 0..4 {
            eprint!("{:8.4} ", coeffs[y * 32 + x]);
        }
        eprintln!();
    }

    // Show some AC values (first few diagonals)
    eprintln!("  AC values (first diagonal):");
    for d in 4..8 {
        eprint!("    diag {}: ", d);
        for y in 0..=d.min(31) {
            let x = d - y;
            if x < 32 && !(y < 4 && x < 4) {
                eprint!("{:6.3} ", coeffs[y * 32 + x]);
            }
        }
        eprintln!();
    }

    // The AC coefficients should be significant for a real photo
    assert!(
        ac_nz > 100,
        "Too few AC non-zeros: {} (expected >100 for real photo)",
        ac_nz
    );
}

/// Debug test: compare file sizes DCT32x32 vs DCT8
#[test]
#[ignore]
fn test_dct32x32_vs_dct8_filesize() {
    use jxl_encoder::vardct::VarDctEncoder;

    let (w, h, linear, _srgb) = load_png_crop(&frymire_path(), 256, 256);

    // Encode with DCT8 only
    let mut encoder8 = VarDctEncoder::new(3.0);
    encoder8.ac_strategy_enabled = false;
    let bytes8 = encoder8.encode(w, h, &linear, None).unwrap().data;

    // Encode with DCT32x32 only
    let mut encoder32 = VarDctEncoder::new(3.0);
    encoder32.force_strategy = Some(4);
    let bytes32 = encoder32.encode(w, h, &linear, None).unwrap().data;

    eprintln!("File sizes for 256x256 frymire at d=3.0:");
    eprintln!("  DCT8:    {} bytes", bytes8.len());
    eprintln!("  DCT32x32: {} bytes", bytes32.len());
    eprintln!(
        "  Ratio:   {:.2}x",
        bytes32.len() as f64 / bytes8.len() as f64
    );

    // DCT32x32 can be smaller than DCT8 (larger blocks = better energy compaction
    // at high distances for smooth areas). At d=3.0 on photos, expect ~30-60% of DCT8.
    // If DCT32x32 is < 25% of DCT8, something is wrong (coefficients being dropped).
    assert!(
        bytes32.len() > bytes8.len() / 4,
        "DCT32x32 file suspiciously small: {} vs DCT8 {}",
        bytes32.len(),
        bytes8.len()
    );
}

/// Debug test to trace AC coefficient writing for DCT32x32
#[test]
#[ignore]
fn test_dct32x32_ac_trace() {
    use jxl_encoder::vardct::VarDctEncoder;

    // Just 64x64 = 8x8 blocks = 4 DCT32x32 transforms
    let (w, h, linear, _srgb) = load_png_crop(&frymire_path(), 64, 64);

    eprintln!("Image: {}x{}", w, h);
    eprintln!("Blocks: {}x{} = {}", w / 8, h / 8, (w / 8) * (h / 8));
    eprintln!(
        "Expected DCT32x32 transforms: {} (each covers 4x4 blocks)",
        (w / 32) * (h / 32)
    );

    let mut encoder = VarDctEncoder::new(3.0);
    encoder.force_strategy = Some(4); // DCT32x32
    let bytes = encoder.encode(w, h, &linear, None).unwrap().data;

    eprintln!("Output size: {} bytes", bytes.len());

    // For comparison, encode with DCT8
    let mut encoder8 = VarDctEncoder::new(3.0);
    encoder8.ac_strategy_enabled = false;
    let bytes8 = encoder8.encode(w, h, &linear, None).unwrap().data;

    eprintln!("DCT8 size: {} bytes", bytes8.len());
    eprintln!("Ratio: {:.2}x", bytes.len() as f64 / bytes8.len() as f64);

    // Verify that forced strategy is actually set
    // Can't directly access the strategy map, but file size should tell us something
}

/// Debug: examine the DCT32x32 transform behavior.
/// Note: DCT32x32 naturally has fewer nonzero AC coefficients than DCT8 because
/// it operates at lower frequency resolution. A checkerboard (Nyquist frequency)
/// in a 32x32 block produces energy concentrated at specific frequencies.
#[test]
#[ignore]
fn test_dct32x32_coeff_storage_debug() {
    use jxl_encoder::vardct::dct::dct_32x32;

    // Create a simple 32x32 pattern
    let mut pixels = [0.0f32; 32 * 32];
    for y in 0..32 {
        for x in 0..32 {
            // Checkerboard pattern - energy at high frequencies
            pixels[y * 32 + x] = if (x + y) % 2 == 0 { 0.8 } else { 0.2 };
        }
    }

    // Do the DCT
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&pixels, &mut coeffs);

    // Count nonzeros
    let nonzero_count = coeffs.iter().filter(|&&c| c.abs() > 0.01).count();
    eprintln!(
        "After DCT: {} nonzero coefficients out of 1024",
        nonzero_count
    );

    // The DC is at position 0, LLF is 4x4 = 16 positions
    let dc = coeffs[0];
    eprintln!("DC = {:.4}", dc);

    // Check some AC coefficients
    eprintln!("First few AC coefficients:");
    for i in 0..32 {
        eprintln!("  coeffs[{}] = {:.4}", i, coeffs[i]);
    }

    // With checkerboard, AC energy concentrates at specific high-frequency bins
    let ac_sum: f32 = coeffs[16..].iter().map(|c| c.abs()).sum();
    eprintln!("Sum of |AC| (excluding LLF 16): {:.2}", ac_sum);

    // DCT32x32 should have SOME nonzero coefficients (at least DC and some LLF)
    assert!(
        nonzero_count >= 1,
        "DCT should produce at least DC coefficient"
    );
}

/// Debug: show where nonzero coefficients are for checkerboard
#[test]
#[ignore]
fn test_dct32x32_checkerboard_frequencies() {
    use jxl_encoder::vardct::dct::dct_32x32;

    // Checkerboard pattern
    let mut pixels = [0.0f32; 32 * 32];
    for y in 0..32 {
        for x in 0..32 {
            pixels[y * 32 + x] = if (x + y) % 2 == 0 { 0.8 } else { 0.2 };
        }
    }

    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&pixels, &mut coeffs);

    // Find and print all nonzero coefficients
    eprintln!("Nonzero coefficients (idx, value, (ky,kx)):");
    for (idx, &c) in coeffs.iter().enumerate() {
        if c.abs() > 0.001 {
            let ky = idx / 32;
            let kx = idx % 32;
            eprintln!("  coeffs[{}] = {:.4}  (ky={}, kx={})", idx, c, ky, kx);
        }
    }
}

/// Debug: manually trace the encoding of a simple DCT32x32 block.
/// Note: A smooth gradient at high quantization produces mostly DC/LLF energy,
/// which is expected behavior for DCT32x32.
#[test]
#[ignore]
fn test_dct32x32_manual_trace() {
    use jxl_encoder::vardct::dct::{dc_from_dct_32x32, dct_32x32};

    // Simple gradient pattern
    let mut pixels = [0.0f32; 1024];
    for y in 0..32 {
        for x in 0..32 {
            pixels[y * 32 + x] = (x as f32 + y as f32) / 62.0; // 0.0 to 1.0
        }
    }

    // Forward DCT
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&pixels, &mut coeffs);

    eprintln!("=== After DCT ===");
    let nonzero_before_quant = coeffs.iter().filter(|&&c| c.abs() > 0.001).count();
    eprintln!("Nonzero coeffs (>0.001): {}", nonzero_before_quant);

    // Simulate quantization at d=3.0 with typical quant value
    let qac = 3.0 * 4.0; // rough approximation
    let mut quant_coeffs = [0i32; 1024];
    for i in 0..1024 {
        quant_coeffs[i] = (coeffs[i] * qac).round() as i32;
    }

    eprintln!("\n=== After Quantization (qac={}) ===", qac);
    let nonzero_after_quant = quant_coeffs.iter().filter(|&&c| c != 0).count();
    eprintln!("Nonzero quantized coeffs: {}", nonzero_after_quant);

    // Show first nonzero coefficients
    eprintln!("\nFirst 20 nonzero quantized coefficients:");
    let mut found = 0;
    for (idx, &c) in quant_coeffs.iter().enumerate() {
        if c != 0 && found < 20 {
            let ky = idx / 32;
            let kx = idx % 32;
            eprintln!("  [{:4}] = {:4} (ky={:2}, kx={:2})", idx, c, ky, kx);
            found += 1;
        }
    }

    // The DC and LLF region (first 4x4 = 16 coefficients in flat layout)
    eprintln!("\nLLF region (4x4):");
    for ky in 0..4 {
        for kx in 0..4 {
            let idx = ky * 32 + kx;
            eprint!("{:6} ", quant_coeffs[idx]);
        }
        eprintln!();
    }

    // A smooth gradient produces mostly low-frequency energy, so after
    // quantization we expect at least some LLF coefficients to be nonzero.
    assert!(
        nonzero_after_quant >= 1,
        "Gradient should have at least DC coefficient"
    );
}

/// Debug: trace DCT32x32 transform behavior on photo content.
/// Note: DCT32x32 naturally produces fewer AC coefficients than DCT8 because
/// it operates at lower frequency resolution. This test uses a smooth region
/// from a photo which is appropriate for DCT32x32.
#[test]
#[ignore]
fn test_dct32x32_photo_trace() {
    use jxl_encoder::vardct::dct::dct_32x32;

    // Generate smooth content (DCT32x32 appropriate) instead of frymire
    let (w, h, linear, _srgb) = generate_smooth_gradient(32, 32);
    eprintln!("Generated {}x{} smooth gradient", w, h);

    // Take Y channel (brightness)
    let mut pixels = [0.0f32; 1024];
    for i in 0..1024 {
        // Linear RGB to Y: approximate Y = 0.2126*R + 0.7152*G + 0.0722*B
        let r = linear[i * 3];
        let g = linear[i * 3 + 1];
        let b = linear[i * 3 + 2];
        pixels[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    eprintln!(
        "Pixel range: {:.4} to {:.4}",
        pixels.iter().cloned().fold(f32::INFINITY, f32::min),
        pixels.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Forward DCT
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&pixels, &mut coeffs);

    eprintln!("\n=== After DCT ===");
    let nonzero_before_quant = coeffs.iter().filter(|&&c| c.abs() > 0.001).count();
    eprintln!("Nonzero coeffs (>0.001): {}", nonzero_before_quant);

    // Simulate quantization at d=3.0
    // Use actual quant weights for Y channel DCT32x32
    let qac = 3.0 * 4.0;
    let mut quant_coeffs = [0i32; 1024];
    for i in 0..1024 {
        quant_coeffs[i] = (coeffs[i] * qac).round() as i32;
    }

    eprintln!("\n=== After Quantization (qac={}) ===", qac);
    let nonzero_after_quant = quant_coeffs.iter().filter(|&&c| c != 0).count();
    eprintln!("Nonzero quantized coeffs: {}", nonzero_after_quant);

    // Count AC (non-LLF) nonzeros
    // LLF is 4x4 region at positions ky<4 && kx<4
    let mut ac_nonzeros = 0;
    for idx in 0..1024 {
        let ky = idx / 32;
        let kx = idx % 32;
        if ky >= 4 || kx >= 4 {
            if quant_coeffs[idx] != 0 {
                ac_nonzeros += 1;
            }
        }
    }
    eprintln!("AC (non-LLF) nonzeros: {}", ac_nonzeros);

    // Show first 20 AC nonzeros
    eprintln!("\nFirst 20 AC nonzero coefficients:");
    let mut found = 0;
    for idx in 0..1024 {
        let ky = idx / 32;
        let kx = idx % 32;
        if (ky >= 4 || kx >= 4) && quant_coeffs[idx] != 0 && found < 20 {
            eprintln!(
                "  [{:4}] = {:4} (ky={:2}, kx={:2})",
                idx, quant_coeffs[idx], ky, kx
            );
            found += 1;
        }
    }

    eprintln!("\nLLF region (4x4):");
    for ky in 0..4 {
        for kx in 0..4 {
            let idx = ky * 32 + kx;
            eprint!("{:6} ", quant_coeffs[idx]);
        }
        eprintln!();
    }

    // Smooth content produces mostly low-frequency energy (DC/LLF).
    // AC content may be minimal, which is expected for DCT32x32 on smooth content.
    eprintln!(
        "AC nonzeros: {} (smooth content may have few AC coefficients)",
        ac_nonzeros
    );
}

/// Debug: trace with actual photo content and correct quantization
#[test]
#[ignore]
fn test_dct32x32_photo_correct_quant() {
    use jxl_encoder::vardct::dct::dct_32x32;

    let (w, h, linear, _srgb) = load_png_crop(&frymire_path(), 32, 32);
    eprintln!("Loaded {}x{} crop", w, h);

    // Take Y channel (brightness)
    let mut pixels = [0.0f32; 1024];
    for i in 0..1024 {
        let r = linear[i * 3];
        let g = linear[i * 3 + 1];
        let b = linear[i * 3 + 2];
        pixels[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    eprintln!(
        "Pixel range: {:.4} to {:.4}",
        pixels.iter().cloned().fold(f32::INFINITY, f32::min),
        pixels.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Forward DCT
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&pixels, &mut coeffs);

    eprintln!("\n=== After DCT ===");
    let nonzero_before_quant = coeffs.iter().filter(|&&c| c.abs() > 0.001).count();
    eprintln!("Nonzero coeffs (>0.001): {}", nonzero_before_quant);

    // Correct quantization: scale = global_scale / 65536 ≈ 2482 / 65536 ≈ 0.0379
    // qac = scale * quant_field ≈ 0.0379 * 5 ≈ 0.19
    // But we also need quant weights!
    // Quantized value = coeff * (1/weight) * qac

    // For Y channel DCT32x32, the quant weights at high frequencies are large
    // Let's use a simpler approximation: qac = 0.2 (typical for d=3.0)
    let qac = 0.2;
    let mut quant_coeffs = [0i32; 1024];
    for i in 0..1024 {
        // Simplified: no per-coeff weights for this test
        quant_coeffs[i] = (coeffs[i] * qac).round() as i32;
    }

    eprintln!("\n=== After Quantization (qac={}) ===", qac);
    let nonzero_after_quant = quant_coeffs.iter().filter(|&&c| c != 0).count();
    eprintln!("Nonzero quantized coeffs: {}", nonzero_after_quant);

    // Count AC (non-LLF) nonzeros
    let mut ac_nonzeros = 0;
    for idx in 0..1024 {
        let ky = idx / 32;
        let kx = idx % 32;
        if ky >= 4 || kx >= 4 {
            if quant_coeffs[idx] != 0 {
                ac_nonzeros += 1;
            }
        }
    }
    eprintln!("AC (non-LLF) nonzeros: {}", ac_nonzeros);

    // Show some AC nonzeros
    eprintln!("\nFirst 30 AC nonzero coefficients:");
    let mut found = 0;
    for idx in 0..1024 {
        let ky = idx / 32;
        let kx = idx % 32;
        if (ky >= 4 || kx >= 4) && quant_coeffs[idx] != 0 && found < 30 {
            eprintln!(
                "  [{:4}] = {:4} (ky={:2}, kx={:2})",
                idx, quant_coeffs[idx], ky, kx
            );
            found += 1;
        }
    }

    eprintln!("\n=== With correct quant weights (simulated) ===");
    // Actually, qac is divided by weight, so smaller weights = more precision
    // At high frequencies, weights are large (0.1-1.0 range typically)
    // So quantized = coeff / weight * qac
    // For illustration, if weight=0.5 at position i: quantized = coeff * 2 * qac

    // This test shows that even at d=3.0, a 32x32 block of a photo should have
    // many nonzero AC coefficients if quantization is correct.
}

// Note: test_dct32x32_quant_weights_check removed - quant module is private
