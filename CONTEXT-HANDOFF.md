# Context Handoff: Non-DCT8 Quality Regression

**Date:** 2026-02-01
**Session:** DCT32 testing → quality regression investigation
**Status:** Root cause NOT YET IDENTIFIED. Multiple leads narrowed down.

## The Problem

When AC strategy selection is enabled (`ac_strategy_enabled = true`), any image
that gets non-DCT8 transforms (DCT16x8, DCT8x16, DCT16x16, DCT32x32) produces
**catastrophically bad output** (SSIM2 = -300 to -50). The file sizes are also
extremely small (2-3KB for a 256x256 photo at d=1.0, vs 23KB for DCT8-only).

This regression was introduced on Feb 1, 2026, between the ANS omit_pos fix
(`32d15d4`) and the custom coefficient ordering commit (`abd472d`). The last
known-good quality metrics (SSIM2 ~75 at d=1.0) were measured at or before
commit `5deda3c` (Jan 31).

## Proven Facts (DO NOT RE-INVESTIGATE)

| # | Fact | Evidence |
|---|------|----------|
| 1 | DCT8-only pipeline works | SSIM2 = 81.66 on 256x256 CLIC crop, d=1.0, djxl decode |
| 2 | Strategy selection produces catastrophic quality | SSIM2 = -313 on same image, same distance |
| 3 | Bug is NOT only in DCT32x32 | Disabling DCT32 in `find_best_32x32_transform` → still SSIM2 = -52 |
| 4 | Bug is NOT only in custom coeff ordering | `--no-custom-orders` (custom_orders=false) → still SSIM2 = -313 |
| 5 | Bug is NOT in the loop restructuring of `compute_ac_strategy` | Hash matches pre-DCT32 value (0x45d6d2bcd23d0b19) for 64x64 DCT8-only images |
| 6 | Bug is NOT in coeff_order.rs fix alone | Reverting the raw→bitstream code fix → still bad quality |
| 7 | 64x64 gradient also fails | `test_butteraugli_quality_gate`: Butteraugli=85.461 (expected <3.0). Gradient images trigger non-DCT8 selection |
| 8 | File sizes are 8-10x too small | 2734 bytes vs 23578 bytes → most AC coefficients are being zeroed or lost |

## Known Bugs (NOT YET FIXED)

### Bug A: `get_custom_order` raw/bitstream code mismatch

**File:** `jxl_enc/src/tiny/encoder.rs:1755`
**Issue:** `get_custom_order(orders, used_orders, raw_strategy, c)` passes
`raw_strategy` (0-4) but `strategy_bucket()` expects bitstream codes (0-26).

For DCT16x8: raw=1 → bucket=1 (wrong), should be raw→code 6 → bucket=4.

This causes the encoder to look up custom orders from the wrong bucket, either:
- Getting no order (bucket empty → None → fallback to natural order while bitstream has custom permutation)
- Getting wrong-sized order (64-element order for 128-coefficient transform)

**Fix needed:** Change line 1755 to:
```rust
let strategy_code = super::ac_strategy::STRATEGY_CODE_LUT[raw_strategy as usize];
super::coeff_order::get_custom_order(orders, used_orders, strategy_code, c)
```

**Impact:** Only affects two-pass mode with custom orders enabled. NOT the sole
cause of the regression (quality is still bad with custom_orders=false).

### Bug B (SUSPICION): Quantization grid coordinates may be wrong

**File:** `jxl_enc/src/tiny/encoder.rs:742-749`
**Issue:** `quantize_ac_block` computes the coefficient grid as:
```rust
let grid_width = covered_x * BLOCK_DIM;   // e.g., 1*8=8 for DCT16x8
let grid_height = covered_y * BLOCK_DIM;  // e.g., 2*8=16 for DCT16x8
```

But after the DCT layout swap (line 854), coefficients are stored in a
`(max(cx,cy)*8) × (min(cx,cy)*8)` layout. For DCT16x8 (covered_x=1, covered_y=2),
the layout is 8×16 (stride 16), not 16×8 (stride 8).

The thresholding at lines 597-598 uses y/x coordinates derived from this grid.
Wrong grid dimensions → wrong quadrant assignment → wrong thresholds. But the
threshold values (0.58-0.70) are close together, so this alone shouldn't cause
8.6x file size reduction.

**Status:** SUSPICION. Need to verify against C++ reference. The C++ uses
`(ROWS, COLS)` in QuantizeBlockAC, not (covered_x, covered_y).

## Uncommitted Changes

Only `jxl_enc/src/tiny/ac_group.rs` is modified:
- `coefficient_layout_order()` function for DCT32x32 (LLF-first ordering)
- Associated tests (`test_coeff_order_32x32_llf_first`, `test_coeff_order_16x16_llf_first`)

This change is correct and should be committed.

## Systematic Investigation Plan

Follow the proof-by-tests methodology: accumulate committed invariant tests that
stay in the codebase. Each layer proves or disproves a hypothesis. DO NOT GUESS.
Only move to the next layer when the current one passes.

### Test Images

Use REAL photos, not synthetic gradients:
- `~/work/codec-corpus/imageflow/test_inputs/frymire.png` (1118x1105)
- `~/work/codec-corpus/clic2025/final-test/*.png` (various sizes)
- `~/work/codec-corpus/kodak-legacy/*.png` (768x512)

Crop to 64x64 or 128x128 for fast unit-level tests. Use 256x256+ for integration.

### Layer 0: Baseline Quality Gate (COMMIT THIS FIRST)

Create `jxl_enc/tests/strategy_quality.rs`:

```rust
/// Regression test: strategy selection must not degrade quality catastrophically.
/// Encodes the same image with and without strategy selection.
/// The "ON" version must have SSIM2 >= 60 (current: -313).
#[test]
fn test_strategy_on_vs_off_quality() {
    // Load a 256x256 crop of a real CLIC image
    // Encode with ac_strategy_enabled=true and ac_strategy_enabled=false
    // Decode both with jxl-oxide
    // Compute SSIM2 (or butteraugli) for both
    // Assert ON quality >= 60 (generous threshold)
    // Assert ON-OFF gap < 5 SSIM2
}

/// Same test but at 64x64 to catch strategy selection on small images.
#[test]
fn test_strategy_64x64_quality() {
    // 64x64 crops can still trigger DCT16x16 selection
    // Uses real photo content, not gradients
}
```

### Layer 1: Forward DCT Correctness Per Strategy

Create tests that verify each forward DCT matches the reference C++ output
for known inputs. The inverse DCT (decoder side) is already validated.

```rust
/// DCT16x8 forward transform produces correct coefficients.
/// Uses a known pixel block and compares against C++ reference output.
#[test]
fn test_dct16x8_forward_correctness() {
    // Create a 16x8 pixel block with known content
    // Apply dct_16x8
    // Verify DC coefficient ≈ sum/sqrt(128)
    // Verify a few AC coefficients against hand-computed or C++ values
}

// Similar for dct_8x16, dct_16x16, dct_32x32
```

### Layer 2: DC Extraction from LLF

Test that `dc_from_dct_16x8`, `dc_from_dct_8x16`, `dc_from_dct_16x16`,
`dc_from_dct_32x32` produce correct DC values from known DCT output.

```rust
#[test]
fn test_dc_extraction_16x8() {
    // Create known DCT coefficients
    // Extract DC
    // Compare against expected DC value (block average * 8)
}
```

### Layer 3: Quantization Roundtrip Per Strategy

The most likely failure point. Test that `quantize_ac_block` followed by
the coefficient assembly loop produces reasonable quantized values.

```rust
/// Verify that non-DCT8 quantization doesn't zero out everything.
/// For a real photo block with typical coefficient magnitudes,
/// at least 30% of AC coefficients should be non-zero at d=1.0.
#[test]
fn test_quantization_nonzero_fraction() {
    // For each strategy (DCT8, DCT16x8, DCT8x16, DCT16x16):
    //   1. Load real photo pixels
    //   2. Apply forward DCT
    //   3. Quantize with typical qac value for d=1.0
    //   4. Count non-zero AC coefficients
    //   5. Assert non-zero fraction > 0.3 for DCT8
    //   6. Assert non-zero fraction > 0.15 for larger transforms
    //   7. Compare DCT8 non-zero fraction with larger transform fraction
    //      (should be within 3x, not 10x)
}
```

This test directly addresses the "8.6x smaller file size" symptom.

### Layer 4: Coefficient Assembly Roundtrip

Test that storing coefficients to `quant_ac[c][by][bx]` and reading them
back via the assembly loop produces identical values.

```rust
/// Verify coefficient storage and reassembly is lossless.
#[test]
fn test_coefficient_assembly_roundtrip() {
    // For each multi-block strategy:
    //   1. Create known 128/256/1024 coefficients
    //   2. Store via quantize_ac_block's mapping
    //   3. Read back via the assembly loop (lines 1587-1595)
    //   4. Assert exact equality
}
```

### Layer 5: Nzeros Counting

```rust
/// Verify non-zero counting matches for multi-block transforms.
#[test]
fn test_nzeros_multi_block() {
    // Create known quantized coefficients with specific zero pattern
    // Count via num_nonzero_except_llf
    // Assert raw_nzeros matches expected count
    // Assert shifted nzeros = raw_nzeros / covered_blocks
}
```

### Layer 6: Single-Strategy Full Pipeline

Force a specific strategy on ALL blocks and test the full encode-decode pipeline.
This isolates strategy encoding from strategy selection.

```rust
/// Encode a real photo with ALL blocks as DCT16x8 (no selection).
/// Decode and verify quality.
#[test]
fn test_forced_dct16x8_quality() {
    // Set up AcStrategyMap with all blocks as DCT16x8
    // Run full encode pipeline
    // Decode with jxl-oxide
    // Assert SSIM2 >= 50
}

// Similar for DCT8x16, DCT16x16
```

This test determines if the bug is in SELECTION or ENCODING.

### Layer 7: Strategy Selection Quality

Only reach this layer after Layers 1-6 all pass.

```rust
/// Test that strategy selection produces reasonable strategy mix.
/// On a real photo, most blocks should remain DCT8 at d=1.0.
#[test]
fn test_strategy_selection_mix() {
    // Encode with strategy selection
    // Count strategy distribution
    // Assert DCT8 > 50% of blocks
    // Assert no single non-DCT8 strategy > 40% of blocks
}
```

## Investigation Priorities (In Order)

1. **Commit the ac_group.rs change** (coefficient_layout_order for DCT32)
2. **Write and commit Layer 0 test** (quality gate — this is the failing regression test)
3. **Write and commit Layer 3 test** (quantization non-zero fraction)
   - This is the most likely failure point based on the 8.6x file size reduction
   - If non-zero fractions are comparable between DCT8 and larger transforms,
     the bug is in AC encoding, not quantization
   - If non-zero fractions are 10x smaller for larger transforms, the bug is
     in quantization (weights, grid coordinates, or thresholding)
4. **Fix Bug A** (`get_custom_order` raw/bitstream mismatch) — this is real but
   secondary to the main regression
5. **Investigate Bug B** (quantization grid coordinates) — compare against C++ reference

## Key Code Paths

All non-DCT8 paths branch at 14+ locations in encoder.rs. The critical ones:

| Location | What | Notes |
|----------|------|-------|
| encoder.rs:645-717 | `apply_dct()` | Dispatches to dct_16x8/8x16/16x16/32x32 |
| encoder.rs:854-858 | Layout swap | `if covered_y > covered_x { swap }` — converts to (cx>=cy) |
| encoder.rs:862 | qac computation | `params.scale * quant_field[by * xsize_blocks + bx]` |
| encoder.rs:881-930 | Y DC extraction | Match on raw_strategy, different dc_from_dct_* per strategy |
| encoder.rs:722-771 | `quantize_ac_block` | Grid coordinates: covered_x*8, covered_y*8 — may be wrong |
| encoder.rs:960-994 | Y roundtrip | Dequantize back for CfL |
| encoder.rs:1117-1136 | nzeros counting | Multi-block assembly → `num_nonzero_except_llf` |
| encoder.rs:1587-1595 | AC assembly (streaming) | Linear idx → 2D block mapping |
| encoder.rs:1772-1779 | AC assembly (two-pass) | Same mapping, with custom orders |

## Files to Read First

1. `jxl_enc/src/tiny/encoder.rs` — lines 830-1150 (`transform_and_quantize`) and 1537-1610 (streaming AC) and 1720-1790 (two-pass AC)
2. `jxl_enc/src/tiny/ac_strategy.rs` — `find_best_16x16_transform`, `find_best_32x32_transform`
3. `jxl_enc/src/tiny/ac_group.rs` — `tokenize_ac_coefficients`, `collect_ac_coefficients`, `ac_strategy_info`
4. `jxl_enc/src/tiny/quant.rs` — `quant_weights`, weight tables
5. `jxl_enc/src/tiny/coeff_order.rs` — `count_zero_coefficients`, `get_custom_order`

## Do NOT

- Re-test DCT8-only quality (already proven: 81.66 SSIM2)
- Re-test with DCT32 disabled (already proven: still bad)
- Re-test with coeff_order fix reverted (already proven: still bad)
- Compare against cjxl (full libjxl) — different encoder entirely
- Use synthetic gradients for quality tests — they mask real bugs
- Chase the custom ordering bug before fixing the fundamental non-DCT8 regression
