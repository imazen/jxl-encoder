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
    let (cx, cy) = if covy > covx { (covy, covx) } else { (covx, covy) };
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
    assert_ne!(old, new, "DCT16x16: old formula MUST disagree with new formula");

    // Old formula gives wrong positions
    assert_eq!(old, BTreeSet::from([0, 1, 2, 3]), "old formula gives {{0,1,2,3}}");

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
    assert_ne!(old, new, "DCT32x32: old formula MUST disagree with new formula");

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
    assert!(!old_skip.contains(&16), "old CfL wrongly applies to idx 16 (LLF)");
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
    let output = std::process::Command::new(
        "/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl",
    )
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

/// Compute SSIM2 between original sRGB u8 and decoded sRGB u8.
fn ssim2_u8(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    use fast_ssim2::compute_ssimulacra2;
    use imgref::ImgVec;

    let orig: Vec<[u8; 3]> = original.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let dec: Vec<[u8; 3]> = decoded.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();

    let src = ImgVec::new(orig, width, height);
    let dst = ImgVec::new(dec, width, height);
    compute_ssimulacra2(src.as_ref(), dst.as_ref()).unwrap_or(0.0)
}

/// Compute SSIM2 between original sRGB u8 and decoded linear f32.
fn ssim2_u8_vs_f32(original: &[u8], decoded: &[f32], width: usize, height: usize) -> f64 {
    // decoded is linear f32 from jxl-oxide, convert to sRGB u8
    let dec_u8: Vec<u8> = decoded
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    ssim2_u8(original, &dec_u8, width, height)
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

    let ssim2 = ssim2_u8_vs_f32(&srgb, &pixels, w, h);
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

    let ssim2 = ssim2_u8(&srgb, &dec_srgb, w, h);
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

    let ssim2 = ssim2_u8(&srgb, &dec_srgb, w, h);
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

    let ssim2 = ssim2_u8_vs_f32(&srgb, &pixels, w, h);
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
    let ssim2_dct8 = ssim2_u8(&srgb, &dec8, w, h);

    // DCT16x16-only (forced via hack)
    let mut enc_dct16 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc_dct16.ac_strategy_enabled = true;
    let bytes_dct16 = enc_dct16.encode(w, h, &linear).unwrap();
    let (_, _, dec16) = decode_djxl(&bytes_dct16);
    let ssim2_dct16 = ssim2_u8(&srgb, &dec16, w, h);

    eprintln!("layer4 frymire 256x256 @ d=1.0:");
    eprintln!("  DCT8:    SSIM2={:.2}, {} bytes", ssim2_dct8, bytes_dct8.len());
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
    let ssim2_dct8 = ssim2_u8(&srgb, &dec8, w, h);

    // DCT16x16-only
    let mut enc_dct16 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc_dct16.ac_strategy_enabled = true;
    let bytes_dct16 = enc_dct16.encode(w, h, &linear).unwrap();
    let (_, _, dec16) = decode_djxl(&bytes_dct16);
    let ssim2_dct16 = ssim2_u8(&srgb, &dec16, w, h);

    eprintln!("layer4 frymire full {}x{} @ d=1.0:", w, h);
    eprintln!("  DCT8:    SSIM2={:.2}, {} bytes", ssim2_dct8, bytes_dct8.len());
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
    let ssim2_dct8 = ssim2_u8(&srgb, &dec8, w, h);

    let mut enc_dct16 = jxl_enc::tiny::TinyEncoder::new(1.0);
    enc_dct16.ac_strategy_enabled = true;
    let bytes_dct16 = enc_dct16.encode(w, h, &linear).unwrap();
    let (_, _, dec16) = decode_djxl(&bytes_dct16);
    let ssim2_dct16 = ssim2_u8(&srgb, &dec16, w, h);

    eprintln!("layer4 kodak1 {}x{} @ d=1.0:", w, h);
    eprintln!("  DCT8:    SSIM2={:.2}, {} bytes", ssim2_dct8, bytes_dct8.len());
    eprintln!(
        "  DCT16x16: SSIM2={:.2}, {} bytes",
        ssim2_dct16,
        bytes_dct16.len()
    );

    assert!(ssim2_dct16 > 50.0, "DCT16x16 quality too low: {:.2}", ssim2_dct16);

    let gap = ssim2_dct8 - ssim2_dct16;
    eprintln!("  gap: {:.2} SSIM2", gap);
    assert!(gap < 10.0, "gap too large: {:.2}", gap);
}

/// Multiple distances on 256x256 frymire crop: does DCT16x16 behave
/// reasonably across the quality range?
#[test]
#[ignore] // requires frymire test image and djxl
fn layer4_quality_dct16x16_across_distances() {
    let (w, h, linear, srgb) = load_png_crop(&frymire_path(), 256, 256);

    eprintln!("layer4 distance sweep, frymire 256x256:");
    eprintln!("{:>8} {:>10} {:>10} {:>10} {:>10} {:>8}", "dist", "dct8_ssim", "d16_ssim", "gap", "d8_bytes", "d16_bytes");

    for &distance in &[0.5, 1.0, 2.0, 4.0] {
        let mut enc_dct8 = jxl_enc::tiny::TinyEncoder::new(distance);
        enc_dct8.ac_strategy_enabled = false;
        let bytes_dct8 = enc_dct8.encode(w, h, &linear).unwrap();
        let (_, _, dec8) = decode_djxl(&bytes_dct8);
        let ssim2_dct8 = ssim2_u8(&srgb, &dec8, w, h);

        let mut enc_dct16 = jxl_enc::tiny::TinyEncoder::new(distance);
        enc_dct16.ac_strategy_enabled = true;
        let bytes_dct16 = enc_dct16.encode(w, h, &linear).unwrap();
        let (_, _, dec16) = decode_djxl(&bytes_dct16);
        let ssim2_dct16 = ssim2_u8(&srgb, &dec16, w, h);

        let gap = ssim2_dct8 - ssim2_dct16;
        eprintln!(
            "{:>8.1} {:>10.2} {:>10.2} {:>10.2} {:>10} {:>8}",
            distance, ssim2_dct8, ssim2_dct16, gap, bytes_dct8.len(), bytes_dct16.len()
        );

        // Quality should be reasonable at each distance
        assert!(
            ssim2_dct16 > 30.0,
            "d={}: DCT16x16 quality {:.2} is catastrophically low",
            distance,
            ssim2_dct16
        );

        // Gap should not be huge
        assert!(
            gap < 15.0,
            "d={}: gap {:.2} is too large",
            distance,
            gap
        );
    }
}
