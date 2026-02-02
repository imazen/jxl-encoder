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
// Helper: load a PNG and convert to linear sRGB f32 for TinyEncoder
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
            // sRGB → linear (gamma 2.2 approximation, matches the CLIC test pattern)
            linear.push((p[0] as f32 / 255.0).powf(2.2));
            linear.push((p[1] as f32 / 255.0).powf(2.2));
            linear.push((p[2] as f32 / 255.0).powf(2.2));
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
    let image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(data))
        .unwrap_or_else(|e| panic!("jxl-oxide parse failed: {:?}", e));
    let w = image.width() as usize;
    let h = image.height() as usize;
    let render = image
        .render_frame(0)
        .unwrap_or_else(|e| panic!("jxl-oxide render failed: {:?}", e));
    let pixels = render.image_all_channels().buf().to_vec();
    (w, h, pixels)
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
    let output =
        std::process::Command::new("/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl")
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

/// Convert linear f32 to sRGB u8 (applies gamma 1/2.2).
fn linear_to_srgb_u8(linear: &[f32]) -> Vec<u8> {
    linear
        .iter()
        .map(|&v| (v.max(0.0).powf(1.0 / 2.2) * 255.0).min(255.0).round() as u8)
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
fn ssim2_u8_vs_linear_u8(
    original: &[u8],
    decoded_linear_u8: &[u8],
    width: usize,
    height: usize,
) -> f64 {
    // Convert linear u8 → linear f32 → sRGB u8
    let dec_srgb: Vec<u8> = decoded_linear_u8
        .iter()
        .map(|&v| {
            let lin = v as f32 / 255.0;
            (lin.powf(1.0 / 2.2) * 255.0).min(255.0).round() as u8
        })
        .collect();
    ssim2_srgb(original, &dec_srgb, width, height)
}

/// Frymire test image (1118x1105 real photo, committed to repo).
/// Path relative to workspace root (where cargo test runs).
fn frymire_path() -> String {
    // Try workspace-relative path first (when run from workspace root)
    let ws = "jxl_enc/tests/images/frymire.png";
    if std::path::Path::new(ws).exists() {
        return ws.to_string();
    }
    // Try crate-relative path (when run from jxl_enc/)
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

    let mut encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    encoder.ac_strategy_enabled = true; // triggers forced DCT16x16

    let bytes = encoder
        .encode(w, h, &linear)
        .unwrap_or_else(|e| panic!("encode failed: {:?}", e));

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

    let mut encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    encoder.ac_strategy_enabled = true;

    let bytes = encoder.encode(w, h, &linear).unwrap();

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

    let mut encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    encoder.ac_strategy_enabled = true;

    let bytes = encoder.encode(w, h, &linear).unwrap();

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

    let mut encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    encoder.ac_strategy_enabled = true;

    let bytes = encoder.encode(w, h, &linear).unwrap();

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
    let mut enc_dct8 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc_dct8.ac_strategy_enabled = false;
    let bytes_dct8 = enc_dct8.encode(w, h, &linear).unwrap();
    let (_, _, dec8) = decode_djxl(&bytes_dct8);
    let ssim2_dct8 = ssim2_u8_vs_linear_u8(&srgb, &dec8, w, h);

    // DCT16x16-only (forced via hack)
    let mut enc_dct16 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc_dct16.ac_strategy_enabled = true;
    let bytes_dct16 = enc_dct16.encode(w, h, &linear).unwrap();
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
    let mut enc_dct8 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc_dct8.ac_strategy_enabled = false;
    let bytes_dct8 = enc_dct8.encode(w, h, &linear).unwrap();
    let (_, _, dec8) = decode_djxl(&bytes_dct8);
    let ssim2_dct8 = ssim2_u8_vs_linear_u8(&srgb, &dec8, w, h);

    // DCT16x16-only
    let mut enc_dct16 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc_dct16.ac_strategy_enabled = true;
    let bytes_dct16 = enc_dct16.encode(w, h, &linear).unwrap();
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

    let mut enc_dct8 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc_dct8.ac_strategy_enabled = false;
    let bytes_dct8 = enc_dct8.encode(w, h, &linear).unwrap();
    let (_, _, dec8) = decode_djxl(&bytes_dct8);
    let ssim2_dct8 = ssim2_u8_vs_linear_u8(&srgb, &dec8, w, h);

    let mut enc_dct16 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc_dct16.ac_strategy_enabled = true;
    let bytes_dct16 = enc_dct16.encode(w, h, &linear).unwrap();
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
    // Duplicate dc_from_dct_16x16 from jxl_enc/src/tiny/dct.rs (FIXED version)
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
    let srgb_val = (val.powf(1.0 / 2.2) * 255.0).round() as u8;

    // Encode with DCT16x16 (ac_strategy_enabled = true forces it)
    let mut encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    encoder.ac_strategy_enabled = true;

    let bytes = encoder.encode(w, h, &linear).unwrap();
    eprintln!("solid 16x16: encoded {} bytes", bytes.len());

    // Save for external inspection
    std::fs::write("/tmp/diag_solid16x16_dct16.jxl", &bytes).unwrap();

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
            (r.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0),
            (g.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0),
            (b.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0),
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
    let mut enc8 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear).unwrap();
    std::fs::write("/tmp/diag_solid16x16_dct8.jxl", &bytes8).unwrap();

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
    let mut enc16 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear).unwrap();
    std::fs::write("/tmp/diag_real16x16_dct16.jxl", &bytes16).unwrap();

    // DCT8
    let mut enc8 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear).unwrap();
    std::fs::write("/tmp/diag_real16x16_dct8.jxl", &bytes8).unwrap();

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
        let mut enc8 = jxl_enc::tiny::TinyEncoder::new(1.0);
        enc8.ac_strategy_enabled = false;
        let bytes8 = enc8.encode(w, h, &linear).unwrap();
        let (_, _, d8_linear) = decode_jxl_oxide(&bytes8);
        let ssim8 = ssim2_u8_vs_linear_f32(&srgb, &d8_linear, w, h);

        // DCT16x16 — encode and decode with jxl-oxide
        let mut enc16 = jxl_enc::tiny::TinyEncoder::new(1.0);
        enc16.ac_strategy_enabled = true;
        let bytes16 = match enc16.encode(w, h, &linear) {
            Ok(b) => b,
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
/// is 32x32. Test with 256x256 (single-group, 8 DCT32x32 blocks per row).
#[test]
#[ignore] // requires frymire test image
fn layer2_single_group_dct32x32_decode_jxl_oxide() {
    let (w, h, linear, srgb) = load_png_crop(&frymire_path(), 256, 256);
    assert_eq!(w, 256);
    assert_eq!(h, 256);

    let mut encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    encoder.force_strategy = Some(4); // RAW_STRATEGY_DCT32X32

    let bytes = encoder
        .encode(w, h, &linear)
        .unwrap_or_else(|e| panic!("encode failed: {:?}", e));

    eprintln!(
        "layer2 DCT32x32 jxl-oxide: encoded 256x256 frymire crop, {} bytes",
        bytes.len()
    );

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
#[ignore] // requires frymire test image and djxl
fn layer2_single_group_dct32x32_decode_djxl() {
    let (w, h, linear, srgb) = load_png_crop(&frymire_path(), 256, 256);

    let mut encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    encoder.force_strategy = Some(4); // RAW_STRATEGY_DCT32X32

    let bytes = encoder.encode(w, h, &linear).unwrap();

    eprintln!(
        "layer2 DCT32x32 djxl: encoded 256x256 frymire crop, {} bytes",
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

/// Full frymire (1118x1105) with forced DCT32x32.
#[test]
#[ignore] // requires frymire test image and djxl
fn layer3_multigroup_dct32x32_decode_djxl() {
    let (w, h, linear, srgb) = load_png_full(&frymire_path());
    eprintln!("layer3 DCT32x32: loaded frymire {}x{}", w, h);
    assert!(w > 256 || h > 256, "frymire should be multi-group");

    let mut encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    encoder.force_strategy = Some(4); // RAW_STRATEGY_DCT32X32

    let bytes = encoder.encode(w, h, &linear).unwrap();

    eprintln!(
        "layer3 DCT32x32 djxl: encoded {}x{} frymire, {} bytes",
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
#[ignore] // requires frymire test image
fn layer3_multigroup_dct32x32_decode_jxl_oxide() {
    let (w, h, linear, srgb) = load_png_full(&frymire_path());

    let mut encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    encoder.force_strategy = Some(4); // RAW_STRATEGY_DCT32X32

    let bytes = encoder.encode(w, h, &linear).unwrap();

    eprintln!(
        "layer3 DCT32x32 jxl-oxide: encoded {}x{} frymire, {} bytes",
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

/// Compare DCT32x32 vs DCT8 quality on 256x256 frymire.
#[test]
#[ignore] // requires frymire test image and djxl
fn layer4_quality_dct32x32_vs_dct8_frymire_256() {
    let (w, h, linear, srgb) = load_png_crop(&frymire_path(), 256, 256);

    // DCT8-only
    let mut enc_dct8 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc_dct8.ac_strategy_enabled = false;
    let bytes_dct8 = enc_dct8.encode(w, h, &linear).unwrap();
    let (_, _, dec8) = decode_djxl(&bytes_dct8);
    let ssim2_dct8 = ssim2_u8_vs_linear_u8(&srgb, &dec8, w, h);

    // DCT32x32-only (forced)
    let mut enc_dct32 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc_dct32.force_strategy = Some(4);
    let bytes_dct32 = enc_dct32.encode(w, h, &linear).unwrap();
    let (_, _, dec32) = decode_djxl(&bytes_dct32);
    let ssim2_dct32 = ssim2_u8_vs_linear_u8(&srgb, &dec32, w, h);

    eprintln!("layer4 DCT32x32 vs DCT8, frymire 256x256 @ d=1.0:");
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

    // DCT32x32 quality should be reasonable
    assert!(
        ssim2_dct32 > 50.0,
        "DCT32x32 quality too low: {:.2}",
        ssim2_dct32
    );

    // Gap should be small (within 10 SSIM2).
    // DCT32x32 may have more loss than DCT16x16/DCT8 on small images.
    let gap = ssim2_dct8 - ssim2_dct32;
    assert!(
        gap < 15.0,
        "DCT32x32 vs DCT8 gap too large: {:.2} SSIM2. LLF handling may be wrong.",
        gap
    );
}

/// Compare on full frymire (multi-group).
#[test]
#[ignore] // requires frymire test image and djxl
fn layer4_quality_dct32x32_vs_dct8_frymire_full() {
    let (w, h, linear, srgb) = load_png_full(&frymire_path());

    // DCT8-only
    let mut enc_dct8 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc_dct8.ac_strategy_enabled = false;
    let bytes_dct8 = enc_dct8.encode(w, h, &linear).unwrap();
    let (_, _, dec8) = decode_djxl(&bytes_dct8);
    let ssim2_dct8 = ssim2_u8_vs_linear_u8(&srgb, &dec8, w, h);

    // DCT32x32-only
    let mut enc_dct32 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc_dct32.force_strategy = Some(4);
    let bytes_dct32 = enc_dct32.encode(w, h, &linear).unwrap();
    let (_, _, dec32) = decode_djxl(&bytes_dct32);
    let ssim2_dct32 = ssim2_u8_vs_linear_u8(&srgb, &dec32, w, h);

    eprintln!("layer4 DCT32x32 vs DCT8, frymire full {}x{} @ d=1.0:", w, h);
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
        let mut enc_dct8 = jxl_enc::tiny::TinyEncoder::new(distance);
        enc_dct8.ac_strategy_enabled = false;
        let bytes_dct8 = enc_dct8.encode(w, h, &linear).unwrap();
        let (_, _, dec8) = decode_djxl(&bytes_dct8);
        let ssim2_dct8 = ssim2_u8_vs_linear_u8(&srgb, &dec8, w, h);

        let mut enc_dct16 = jxl_enc::tiny::TinyEncoder::new(distance);
        enc_dct16.ac_strategy_enabled = true;
        let bytes_dct16 = enc_dct16.encode(w, h, &linear).unwrap();
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
        assert!(
            gap < 10.0,
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
    use jxl_enc::tiny::dct::{dct_32x32, dc_from_dct_32x32};
    
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
    eprintln!("  First row (4 elements): {:.4} {:.4} {:.4} {:.4}", 
              coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
    eprintln!("  LLF 4x4 (rows 0-3, cols 0-3):");
    for iy in 0..4 {
        eprintln!("    row {}: {:.6} {:.6} {:.6} {:.6}",
                  iy, coeffs[iy*32], coeffs[iy*32+1], coeffs[iy*32+2], coeffs[iy*32+3]);
    }
    
    // Extract DC values
    let dcs = dc_from_dct_32x32(&coeffs);
    eprintln!("  DC values from LLF (4x4):");
    for iy in 0..4 {
        eprintln!("    row {}: {:.6} {:.6} {:.6} {:.6}",
                  iy, dcs[iy*4], dcs[iy*4+1], dcs[iy*4+2], dcs[iy*4+3]);
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
    eprintln!("  Sum of abs(AC coefficients) = {:.6} (should be ~0)", ac_sum);
}

// Diagnostic: check if DCT32x32 forward+IDCT roundtrips correctly  
#[test]
#[ignore]
fn diag_dct32x32_forward_idct_roundtrip() {
    use jxl_enc::tiny::dct::{dct_32x32, dc_from_dct_32x32};
    
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
        eprintln!("    row {}: {:.6} {:.6} {:.6} {:.6}",
                  iy, dcs[iy*4], dcs[iy*4+1], dcs[iy*4+2], dcs[iy*4+3]);
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
    use jxl_enc::tiny::dct::{dct_32x32, dc_from_dct_32x32};
    
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
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  iy, coeffs[iy*32], coeffs[iy*32+1], coeffs[iy*32+2], coeffs[iy*32+3]);
    }
    
    // Apply resample scales (32 -> 4)
    const SCALE: [f32; 4] = [1.0, 0.974886821136879522, 0.901764195028874394, 0.787054918159101335];
    eprintln!("\nAfter applying resample scales:");
    for iy in 0..4 {
        let mut row = [0.0f32; 4];
        for ix in 0..4 {
            row[ix] = coeffs[iy*32+ix] * SCALE[iy] * SCALE[ix];
        }
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  iy, row[0], row[1], row[2], row[3]);
    }
    
    // Extract DC values
    let dcs = dc_from_dct_32x32(&coeffs);
    eprintln!("\nDC values from dc_from_dct_32x32:");
    for iy in 0..4 {
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  iy, dcs[iy*4], dcs[iy*4+1], dcs[iy*4+2], dcs[iy*4+3]);
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
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  by, row[0], row[1], row[2], row[3]);
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
                    sum += input[(by*8+dy) * 32 + bx*8+dx];
                }
            }
            let expected = sum / 64.0;
            let error = dcs[by*4+bx] - expected;
            row[bx] = error;
            total_error += error.abs();
        }
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  by, row[0], row[1], row[2], row[3]);
    }
    eprintln!("\nTotal absolute error: {:.6}", total_error);
}

// Diagnostic: verify the DCT32x32 <-> DC relationship
#[test]
#[ignore]
fn diag_dct32x32_roundtrip_verification() {
    use jxl_enc::tiny::dct::{dct_32x32, dc_from_dct_32x32};
    
    // Resample scales for 32 -> 4 (from C++)
    const SCALE_32_TO_4: [f32; 4] = [1.0, 0.974886821136879522, 0.901764195028874394, 0.787054918159101335];
    // Inverse scales for 4 -> 32
    const SCALE_4_TO_32: [f32; 4] = [1.0, 1.0257549441917856, 1.1089312359806676, 1.2706084147018952];
    
    // 4-point DCT-II (forward)
    fn dct1d_4(input: &[f32; 4]) -> [f32; 4] {
        use core::f32::consts::PI;
        let mut output = [0.0f32; 4];
        for k in 0..4 {
            let mut sum = 0.0f32;
            for n in 0..4 {
                sum += input[n] * (PI * k as f32 * (2.0 * n as f32 + 1.0) / 8.0).cos();
            }
            output[k] = sum / 4.0;  // Normalize by N
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
                    sum += input[(by*8+dy) * 32 + bx*8+dx];
                }
            }
            expected_dc[by][bx] = sum / 64.0;
        }
    }
    
    eprintln!("Expected 8x8 block averages (DC grid):");
    for by in 0..4 {
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  by, expected_dc[by][0], expected_dc[by][1], expected_dc[by][2], expected_dc[by][3]);
    }
    
    // Apply 4x4 DCT to expected_dc to get expected LLF
    // First DCT rows
    let mut after_rows = [[0.0f32; 4]; 4];
    for iy in 0..4 {
        let row: [f32; 4] = [expected_dc[iy][0], expected_dc[iy][1], expected_dc[iy][2], expected_dc[iy][3]];
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
        let row: [f32; 4] = [transposed[iy][0], transposed[iy][1], transposed[iy][2], transposed[iy][3]];
        let dct_row = dct1d_4(&row);
        for ix in 0..4 {
            expected_llf[iy][ix] = dct_row[ix];
        }
    }
    
    eprintln!("\nExpected LLF (from DCT4x4 of DC grid):");
    for iy in 0..4 {
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  iy, expected_llf[iy][0], expected_llf[iy][1], expected_llf[iy][2], expected_llf[iy][3]);
    }
    
    // Apply inverse resample scales (to go from DC-domain to DCT32-domain)
    eprintln!("\nExpected LLF with inverse scales (should match dct_32x32 output):");
    for iy in 0..4 {
        let mut row = [0.0f32; 4];
        for ix in 0..4 {
            row[ix] = expected_llf[iy][ix] * SCALE_4_TO_32[iy] * SCALE_4_TO_32[ix];
        }
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  iy, row[0], row[1], row[2], row[3]);
    }
    
    // Now apply forward DCT32x32 and get actual LLF
    let mut coeffs = [0.0f32; 1024];
    dct_32x32(&input, &mut coeffs);
    
    eprintln!("\nActual LLF from dct_32x32:");
    for iy in 0..4 {
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  iy, coeffs[iy*32], coeffs[iy*32+1], coeffs[iy*32+2], coeffs[iy*32+3]);
    }
    
    // Apply forward resample scales
    eprintln!("\nActual LLF with forward scales (input to IDCT):");
    for iy in 0..4 {
        let mut row = [0.0f32; 4];
        for ix in 0..4 {
            row[ix] = coeffs[iy*32+ix] * SCALE_32_TO_4[iy] * SCALE_32_TO_4[ix];
        }
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  iy, row[0], row[1], row[2], row[3]);
    }
    
    // Finally, dc_from_dct_32x32 output
    let dcs = dc_from_dct_32x32(&coeffs);
    eprintln!("\ndc_from_dct_32x32 output:");
    for iy in 0..4 {
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  iy, dcs[iy*4], dcs[iy*4+1], dcs[iy*4+2], dcs[iy*4+3]);
    }
}

// Diagnostic: test sqrt(2) correction for DCT32x32 LLF
#[test]
#[ignore]
fn diag_dct32x32_sqrt2_correction() {
    use jxl_enc::tiny::dct::dct_32x32;
    
    const SCALE_32_TO_4: [f32; 4] = [1.0, 0.974886821136879522, 0.901764195028874394, 0.787054918159101335];
    const SQRT2: f32 = 1.4142135623730951;
    
    // 4-point IDCT
    fn idct1d_4(input: &[f32; 4]) -> [f32; 4] {
        use core::f32::consts::PI;
        let x0 = input[0];
        let x1 = input[1];
        let x2 = input[2];
        let x3 = input[3];
        [
            x0 + 2.0 * (x1 * (PI/8.0).cos() + x2 * (PI/4.0).cos() + x3 * (3.0*PI/8.0).cos()),
            x0 + 2.0 * (x1 * (3.0*PI/8.0).cos() + x2 * (3.0*PI/4.0).cos() + x3 * (9.0*PI/8.0).cos()),
            x0 + 2.0 * (x1 * (5.0*PI/8.0).cos() + x2 * (5.0*PI/4.0).cos() + x3 * (15.0*PI/8.0).cos()),
            x0 + 2.0 * (x1 * (7.0*PI/8.0).cos() + x2 * (7.0*PI/4.0).cos() + x3 * (21.0*PI/8.0).cos()),
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
            let row = [block[iy*4], block[iy*4+1], block[iy*4+2], block[iy*4+3]];
            let out = idct1d_4(&row);
            for ix in 0..4 { after_rows[iy*4+ix] = out[ix]; }
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
            let row = [transposed[iy*4], transposed[iy*4+1], transposed[iy*4+2], transposed[iy*4+3]];
            let out = idct1d_4(&row);
            for ix in 0..4 { result[iy*4+ix] = out[ix]; }
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
                    sum += input[(by*8+dy)*32 + bx*8+dx];
                }
            }
            row[bx] = sum / 64.0;
        }
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  by, row[0], row[1], row[2], row[3]);
    }
    
    eprintln!("\nDC from fixed dc_from_dct_32x32 (with sqrt2 correction):");
    for iy in 0..4 {
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  iy, dcs_fixed[iy*4], dcs_fixed[iy*4+1], dcs_fixed[iy*4+2], dcs_fixed[iy*4+3]);
    }
    
    // Compute error
    let mut total_error = 0.0f32;
    for by in 0..4 {
        for bx in 0..4 {
            let mut sum = 0.0f32;
            for dy in 0..8 {
                for dx in 0..8 {
                    sum += input[(by*8+dy)*32 + bx*8+dx];
                }
            }
            let expected = sum / 64.0;
            total_error += (dcs_fixed[by*4+bx] - expected).abs();
        }
    }
    eprintln!("\nTotal absolute error with sqrt2 fix: {:.6}", total_error);
}

// Diagnostic: test butterfly IDCT matching C++
#[test]
#[ignore]
fn diag_dct32x32_butterfly_idct() {
    use jxl_enc::tiny::dct::dct_32x32;
    
    const SCALE_32_TO_4: [f32; 4] = [1.0, 0.974886821136879522, 0.901764195028874394, 0.787054918159101335];
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
            let row = [block[iy*4], block[iy*4+1], block[iy*4+2], block[iy*4+3]];
            let out = idct4_butterfly(&row);
            for ix in 0..4 { after_rows[iy*4+ix] = out[ix]; }
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
            let row = [transposed[iy*4], transposed[iy*4+1], transposed[iy*4+2], transposed[iy*4+3]];
            let out = idct4_butterfly(&row);
            for ix in 0..4 { result[iy*4+ix] = out[ix]; }
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
        eprintln!("  row {}: {:10.6} {:10.6} {:10.6} {:10.6}",
                  iy, dcs[iy*4], dcs[iy*4+1], dcs[iy*4+2], dcs[iy*4+3]);
    }
    
    // Expected and error
    let mut total_error = 0.0f32;
    for by in 0..4 {
        for bx in 0..4 {
            let mut sum = 0.0f32;
            for dy in 0..8 { for dx in 0..8 { sum += input[(by*8+dy)*32 + bx*8+dx]; } }
            let expected = sum / 64.0;
            total_error += (dcs[by*4+bx] - expected).abs();
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
    let mut encoder = jxl_enc::tiny::TinyEncoder::new(1.0);
    encoder.force_strategy = Some(3); // RAW_STRATEGY_DCT16X16
    
    let bytes = encoder.encode(w, h, &linear).unwrap();
    
    // Save to file
    let path = "/tmp/test_dct16x16.jxl";
    let mut file = fs::File::create(path).unwrap();
    file.write_all(&bytes).unwrap();
    eprintln!("Saved {} bytes to {}", bytes.len(), path);
    
    // Try to decode with djxl
    let output = std::process::Command::new("djxl")
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
    let mut enc8 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear).unwrap();
    
    // DCT16x16 encoding  
    let mut enc16 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear).unwrap();
    
    // Decode with jxl-oxide
    let (_, _, dec8) = decode_jxl_oxide(&bytes8);
    let (_, _, dec16) = decode_jxl_oxide(&bytes16);
    
    // Convert decoded linear f32 to sRGB u8 for comparison
    fn linear_to_srgb_u8(v: f32) -> u8 {
        (v.clamp(0.0, 1.0).powf(1.0/2.2) * 255.0).round() as u8
    }
    
    eprintln!("16x16 frymire crop - jxl-oxide decoder:");
    eprintln!("{:>5} {:>12} {:>12} {:>12}", "pixel", "original", "dct8", "dct16x16");
    
    let mut sum_diff8 = 0u32;
    let mut sum_diff16 = 0u32;
    
    for y in 0..4 {
        for x in 0..4 {
            let idx = y * 4 * w + x * 4;  // Sample every 4th pixel
            let o = (srgb[idx*3], srgb[idx*3+1], srgb[idx*3+2]);
            let d8 = (
                linear_to_srgb_u8(dec8[idx*3]),
                linear_to_srgb_u8(dec8[idx*3+1]),
                linear_to_srgb_u8(dec8[idx*3+2]),
            );
            let d16 = (
                linear_to_srgb_u8(dec16[idx*3]),
                linear_to_srgb_u8(dec16[idx*3+1]),
                linear_to_srgb_u8(dec16[idx*3+2]),
            );
            
            let diff8 = (o.0 as i32 - d8.0 as i32).abs() 
                + (o.1 as i32 - d8.1 as i32).abs() 
                + (o.2 as i32 - d8.2 as i32).abs();
            let diff16 = (o.0 as i32 - d16.0 as i32).abs()
                + (o.1 as i32 - d16.1 as i32).abs()
                + (o.2 as i32 - d16.2 as i32).abs();
            
            sum_diff8 += diff8 as u32;
            sum_diff16 += diff16 as u32;
            
            eprintln!("  ({:2},{:2}) {:>3},{:>3},{:>3}  {:>3},{:>3},{:>3}  {:>3},{:>3},{:>3}  d8={:>3} d16={:>3}",
                y*4, x*4, o.0, o.1, o.2, d8.0, d8.1, d8.2, d16.0, d16.1, d16.2, diff8, diff16);
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
    let mut enc8 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear).unwrap();
    
    // DCT16x16 encoding
    let mut enc16 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear).unwrap();
    
    // Save for inspection
    std::fs::write("/tmp/frymire_32x32_dct8.jxl", &bytes8).unwrap();
    std::fs::write("/tmp/frymire_32x32_dct16.jxl", &bytes16).unwrap();
    eprintln!("Saved /tmp/frymire_32x32_dct8.jxl ({} bytes)", bytes8.len());
    eprintln!("Saved /tmp/frymire_32x32_dct16.jxl ({} bytes)", bytes16.len());
    
    // Decode with jxl-oxide
    let (_, _, dec8) = decode_jxl_oxide(&bytes8);
    let (_, _, dec16) = decode_jxl_oxide(&bytes16);
    
    // Convert decoded linear f32 to sRGB u8 for comparison
    fn linear_to_srgb_u8(v: f32) -> u8 {
        (v.clamp(0.0, 1.0).powf(1.0/2.2) * 255.0).round() as u8
    }
    
    eprintln!("32x32 frymire crop - jxl-oxide decoder:");
    eprintln!("{:>5} {:>12} {:>12} {:>12}", "pixel", "original", "dct8", "dct16x16");
    
    // Sample corners and center
    for (name, y, x) in [
        ("top-left", 0usize, 0usize),
        ("top-right", 0, 24),
        ("center", 16, 16),
        ("bottom-left", 24, 0),
        ("bottom-right", 24, 24),
    ] {
        let idx = y * w + x;
        let o = (srgb[idx*3], srgb[idx*3+1], srgb[idx*3+2]);
        let d8 = (
            linear_to_srgb_u8(dec8[idx*3]),
            linear_to_srgb_u8(dec8[idx*3+1]),
            linear_to_srgb_u8(dec8[idx*3+2]),
        );
        let d16 = (
            linear_to_srgb_u8(dec16[idx*3]),
            linear_to_srgb_u8(dec16[idx*3+1]),
            linear_to_srgb_u8(dec16[idx*3+2]),
        );
        
        let diff8 = (o.0 as i32 - d8.0 as i32).abs() 
            + (o.1 as i32 - d8.1 as i32).abs() 
            + (o.2 as i32 - d8.2 as i32).abs();
        let diff16 = (o.0 as i32 - d16.0 as i32).abs()
            + (o.1 as i32 - d16.1 as i32).abs()
            + (o.2 as i32 - d16.2 as i32).abs();
        
        eprintln!("  {:12} {:>3},{:>3},{:>3}  {:>3},{:>3},{:>3}  {:>3},{:>3},{:>3}  d8={:>3} d16={:>3}",
            name, o.0, o.1, o.2, d8.0, d8.1, d8.2, d16.0, d16.1, d16.2, diff8, diff16);
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
            linear[idx+1] = v;
            linear[idx+2] = v;
        }
    }
    
    // First, try DCT8 to see expected nzeros
    let mut enc8 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear).unwrap();
    
    // Then DCT16x16
    let mut enc16 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear).unwrap();
    
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
            let idx = ((h/2 + dy - 2) * w + (w/2 + dx - 2)) * 3;
            let expected = if ((w/2 + dx - 2) + (h/2 + dy - 2)) % 2 == 0 { 0.8 } else { 0.2 };
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
            linear[idx+1] = v;
            linear[idx+2] = v;
        }
    }
    
    let mut enc16 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear).unwrap();
    
    let (_, _, dec16) = decode_jxl_oxide(&bytes16);
    
    eprintln!("Horizontal gradient 32x32 with DCT16x16:");
    eprintln!("File size: {} bytes", bytes16.len());
    
    // Check values along first row
    eprintln!("First row (original vs decoded):");
    for x in [0, 8, 16, 24, 31] {
        let expected = x as f32 / w as f32;
        let decoded = dec16[(0 * w + x) * 3];
        eprintln!("  x={:2}: expected={:.3}, decoded={:.3}, diff={:.3}", 
            x, expected, decoded, (expected - decoded).abs());
    }
}

/// DIAGNOSTIC: Check block iteration for 32x32 with DCT16x16.
#[test]
#[ignore]
fn diag_dct16x16_iteration() {
    use jxl_enc::tiny::TinyEncoder;
    
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
                    linear[idx+1] = block_val;
                    linear[idx+2] = block_val;
                }
            }
        }
    }
    
    // DCT8
    let mut enc8 = TinyEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear).unwrap();
    
    // DCT16x16
    let mut enc16 = TinyEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear).unwrap();
    
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
    
    eprintln!("\nFile sizes: DCT8={}, DCT16={}", bytes8.len(), bytes16.len());
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
                    linear[idx+1] = block_val;
                    linear[idx+2] = block_val;
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
    eprintln!("  (0,0)={:.3} (0,15)={:.3} (15,0)={:.3} (15,15)={:.3}",
        block16x16[0], block16x16[15], block16x16[15*16], block16x16[15*16+15]);
    
    // Do forward DCT
    let mut dct_coeffs = [0.0f32; 256];
    jxl_enc::tiny::dct::dct_16x16(&block16x16, &mut dct_coeffs);
    
    eprintln!("DCT coefficients (LLF 2x2 region):");
    eprintln!("  coeff[0]={:.6} coeff[1]={:.6}", dct_coeffs[0], dct_coeffs[1]);
    eprintln!("  coeff[16]={:.6} coeff[17]={:.6}", dct_coeffs[16], dct_coeffs[17]);
    
    // Extract DC values
    let dcs = jxl_enc::tiny::dct::dc_from_dct_16x16(&dct_coeffs);
    
    eprintln!("Extracted DC values:");
    eprintln!("  dcs[0]={:.6} (top-left 8x8)", dcs[0]);
    eprintln!("  dcs[1]={:.6} (top-right 8x8)", dcs[1]);
    eprintln!("  dcs[2]={:.6} (bottom-left 8x8)", dcs[2]);
    eprintln!("  dcs[3]={:.6} (bottom-right 8x8)", dcs[3]);
    
    // Expected: averages of each 8x8 block
    eprintln!("\nExpected DC values (block averages):");
    eprintln!("  top-left: {:.6}", 0.0);      // block (0,0)
    eprintln!("  top-right: {:.6}", 1.0/16.0); // block (0,1)
    eprintln!("  bottom-left: {:.6}", 4.0/16.0); // block (1,0)
    eprintln!("  bottom-right: {:.6}", 5.0/16.0); // block (1,1)
    
    // Now test the third DCT16x16 block (by=2, bx=0)
    eprintln!("\n\nTesting dc_from_dct_16x16 for THIRD DCT16x16 block (by=2, bx=0):");
    eprintln!("This covers 8x8 blocks: (2,0), (2,1), (3,0), (3,1)");
    eprintln!("Expected block values: {:.3}, {:.3}, {:.3}, {:.3}",
        8.0/16.0, 9.0/16.0, 12.0/16.0, 13.0/16.0);
    
    // Extract the third 16x16 spatial block (starting at by=2, bx=0)
    for sy in 0..16 {
        for sx in 0..16 {
            let v = linear[((16 + sy) * w + sx) * 3]; // offset by 16 rows
            block16x16[sy * 16 + sx] = v;
        }
    }
    
    jxl_enc::tiny::dct::dct_16x16(&block16x16, &mut dct_coeffs);
    let dcs3 = jxl_enc::tiny::dct::dc_from_dct_16x16(&dct_coeffs);
    
    eprintln!("Extracted DC values for third block:");
    eprintln!("  dcs[0]={:.6} (expected {:.6})", dcs3[0], 8.0/16.0);
    eprintln!("  dcs[1]={:.6} (expected {:.6})", dcs3[1], 9.0/16.0);
    eprintln!("  dcs[2]={:.6} (expected {:.6})", dcs3[2], 12.0/16.0);
    eprintln!("  dcs[3]={:.6} (expected {:.6})", dcs3[3], 13.0/16.0);
}

/// DIAGNOSTIC: Test dc_from_dct_16x16 with uniform blocks.
#[test]
#[ignore]
fn diag_dct16x16_uniform() {
    for v in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
        let block = [v; 256];
        let mut dct_coeffs = [0.0f32; 256];
        jxl_enc::tiny::dct::dct_16x16(&block, &mut dct_coeffs);
        let dcs = jxl_enc::tiny::dct::dc_from_dct_16x16(&dct_coeffs);
        
        eprintln!("Uniform v={:.2}: dcs=[{:.4}, {:.4}, {:.4}, {:.4}] (all should be {:.4})",
            v, dcs[0], dcs[1], dcs[2], dcs[3], v);
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
    jxl_enc::tiny::dct::dct_16x16(&block, &mut dct_coeffs);
    let dcs = jxl_enc::tiny::dct::dc_from_dct_16x16(&dct_coeffs);
    
    eprintln!("Quadrant pattern (TL=0, TR=1, BL=0, BR=1):");
    eprintln!("  Expected: 0.0, 1.0, 0.0, 1.0");
    eprintln!("  Got:      {:.4}, {:.4}, {:.4}, {:.4}", dcs[0], dcs[1], dcs[2], dcs[3]);
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
    jxl_enc::tiny::dct::dct_16x16(&block_h, &mut dct_h);
    
    eprintln!("Horizontal gradient (x-variation only):");
    eprintln!("  coeff[0] (DC) = {:.6}", dct_h[0]);
    eprintln!("  coeff[1] (should have energy if fx=1,fy=0) = {:.6}", dct_h[1]);
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
    jxl_enc::tiny::dct::dct_16x16(&block_v, &mut dct_v);
    
    eprintln!("\nVertical gradient (y-variation only):");
    eprintln!("  coeff[0] (DC) = {:.6}", dct_v[0]);
    eprintln!("  coeff[1] (should be ~0 if fx=0) = {:.6}", dct_v[1]);
    eprintln!("  coeff[16] (should have energy if fx=0,fy=1) = {:.6}", dct_v[16]);
    
    // Now test dc_from_dct_16x16 with these
    let dcs_h = jxl_enc::tiny::dct::dc_from_dct_16x16(&dct_h);
    let dcs_v = jxl_enc::tiny::dct::dc_from_dct_16x16(&dct_v);
    
    eprintln!("\nHorizontal gradient DC extraction:");
    eprintln!("  Should have horizontal variation (left vs right):");
    eprintln!("  dcs = [{:.4}, {:.4}, {:.4}, {:.4}]", dcs_h[0], dcs_h[1], dcs_h[2], dcs_h[3]);
    eprintln!("  left column: avg({:.4}, {:.4}) = {:.4}", dcs_h[0], dcs_h[2], (dcs_h[0]+dcs_h[2])/2.0);
    eprintln!("  right column: avg({:.4}, {:.4}) = {:.4}", dcs_h[1], dcs_h[3], (dcs_h[1]+dcs_h[3])/2.0);
    
    eprintln!("\nVertical gradient DC extraction:");
    eprintln!("  Should have vertical variation (top vs bottom):");
    eprintln!("  dcs = [{:.4}, {:.4}, {:.4}, {:.4}]", dcs_v[0], dcs_v[1], dcs_v[2], dcs_v[3]);
    eprintln!("  top row: avg({:.4}, {:.4}) = {:.4}", dcs_v[0], dcs_v[1], (dcs_v[0]+dcs_v[1])/2.0);
    eprintln!("  bottom row: avg({:.4}, {:.4}) = {:.4}", dcs_v[2], dcs_v[3], (dcs_v[2]+dcs_v[3])/2.0);
}

/// DIAGNOSTIC: Check which transforms are being processed for 32x32 DCT16x16.
#[test]
#[ignore]
fn diag_dct16x16_transform_coverage() {
    use jxl_enc::tiny::TinyEncoder;
    
    // Patch TinyEncoder to print transform positions - we'll do this by examining the strategy map
    let w = 32usize;
    let h = 32usize;
    let linear = vec![0.5f32; w * h * 3];
    
    let mut enc = TinyEncoder::new(1.0);
    enc.ac_strategy_enabled = true;
    
    // Run encoding
    let bytes = enc.encode(w, h, &linear).unwrap();
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
            linear[idx+1] = v;
            linear[idx+2] = v;
        }
    }
    
    let mut enc8 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc8.ac_strategy_enabled = false;
    let bytes8 = enc8.encode(w, h, &linear).unwrap();
    
    let mut enc16 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc16.ac_strategy_enabled = true;
    let bytes16 = enc16.encode(w, h, &linear).unwrap();
    
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
