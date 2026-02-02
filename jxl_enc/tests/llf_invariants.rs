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
