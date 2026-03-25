# Plan: Fix Non-Square DCT Transform Quality Bugs

> **Completed (February 2026).** All non-square DCT transforms and AFV modes described here have been fixed and enabled.

## Problem

All non-square transforms produce garbage butteraugli when forced:

| Transform | Raw | covered_x | covered_y | Forced bfly | Status |
|-----------|-----|-----------|-----------|-------------|--------|
| DCT16X8   | 1   | 1         | 2         | 2.59        | WORKS  |
| DCT8X16   | 2   | 2         | 1         | 3.03        | WORKS  |
| DCT32X16  | 10  | 2         | 4         | 114+        | BROKEN |
| DCT16X32  | 11  | 4         | 2         | 82+         | BROKEN |
| DCT64X32  | 17  | 4         | 8         | 109+        | BROKEN |
| DCT32X64  | 18  | 8         | 4         | 32-46       | BROKEN |
| AFV0-3    | 12-15| 1        | 1         | 7-8         | BROKEN (separate bug) |

Key observation: DCT16X8 and DCT8X16 work despite being non-square. The broken
transforms are all 32+ pixels in at least one dimension. DCT16X8/DCT8X16 use SIMD
implementations from jxl_simd, while the broken transforms use pure-Rust forward
DCTs in `dct/forward_large.rs`.

## Root Cause Analysis

### libjxl Output Convention

libjxl's `ComputeScaledDCT<ROWS, COLS>` always normalizes output to
`min(R,C) rows × max(R,C) cols`:

- `<32,16>`: ROWS≥COLS path (no final transpose) → output is 16×32, stride 32
- `<16,32>`: ROWS<COLS path (final transpose) → output is 16×32, stride 32
- Both produce the SAME layout: CoefficientLayout normalizes to (min, max)

### Our Implementation

Our `dct_32x16` and `dct_16x32` both output 16×32 stride-32, matching libjxl.
The CoefficientLayout swap in `transform.rs:376-382` uses `cx = max(covered_x, covered_y)`.

**The DCT output convention appears correct.** The bug is likely in one of:

1. **Pixel extraction layout** — how `apply_dct` reads pixels into the block buffer
2. **DC extraction** — `dc_from_dct_32x16` LLF region indices
3. **Quantization slot mapping** — `quantize_ac_block` index computation
4. **nzeros counting** — flat buffer indexing in `transform.rs:1009-1057`
5. **Tokenization** — coefficient scan order vs actual coefficient positions

## Diagnostic Strategy

### Phase 1: Bit-Level Comparison with libjxl (1-2 hours)

Use a tiny test image (32x16 constant or gradient) and compare coefficient outputs:

1. Encode with cjxl at effort 1 forcing DCT32x16
2. Decode with jxl-rs/jxl-oxide, extract raw coefficients
3. Encode same image with our encoder forcing DCT32x16
4. Compare coefficient values position-by-position

If coefficients differ at the forward DCT stage, the bug is in `dct/forward_large.rs`.
If coefficients match but the bitstream is wrong, the bug is in tokenization/assembly.

### Phase 2: Unit Test the Forward DCT (1 hour)

Write a roundtrip test: `dct_32x16` → `idct_32x16` (from jxl_simd) and verify
the output matches the input to within floating-point precision.

If the roundtrip fails, the forward DCT implementation is wrong.

### Phase 3: Trace Coefficient Flow (1-2 hours)

Add temporary debug logging to trace one block through the pipeline:
1. apply_dct: pixel extraction → DCT output (first 8 coefficients)
2. dc_from_dct: LLF extraction → DC values
3. quantize_ac_block: quantized coefficient positions
4. tokenize: which positions get which values

Compare each stage against libjxl trace output for the same block.

### Phase 4: Fix and Verify (1-2 hours)

Based on findings from Phases 1-3:
- Fix the identified bug(s)
- Run forced-strategy tests on 128x128 crops
- Run full regression tests
- Enable the fixed transforms in auto-selection

## Specific Code Locations to Investigate

### Forward DCT (`dct/forward_large.rs`)

- `dct_32x16` (lines 144-178): Output layout 16×32 stride 32
- `dct_16x32` (lines 186-222): Output layout 16×32 stride 32 (with final transpose)
- `dct_64x32` (lines ~430-470): Output layout 32×64 stride 64
- `dct_32x64` (lines ~470-510): Output layout 32×64 stride 64 (with final transpose)
- `dc_from_dct_32x16` (lines 230-268): LLF at `[r*32+c]`, resample scales
- `dc_from_dct_16x32` (lines 275-310): LLF at `[r*32+c]`, resample scales

### Pixel Extraction (`transform.rs:179-238`)

- DCT32x16: reads 32 rows × 16 cols, stride 16 → calls `dct_32x16`
- DCT16x32: reads 16 rows × 32 cols, stride 32 → calls `dct_16x32`
- DCT64x32: reads 64 rows × 32 cols, stride 32 → calls `dct_64x32`
- DCT32x64: reads 32 rows × 64 cols, stride 64 → calls `dct_32x64`

### Quantization (`quantize.rs:381-452`)

- CoefficientLayout normalization: `cx = max(covered_x, covered_y)`
- `transpose_slots` flag: set when `covered_y > covered_x`
- Slot indexing: `slot_y = sby / BLOCK_DIM`, `slot_x = sbx / BLOCK_DIM`
- When transposed: `slot_idx = slot_x * cy + slot_y`
- When not transposed: `slot_idx = slot_y * cx + slot_x`

### Flat Buffer Assembly (`bitstream.rs:562-593`, `bitstream.rs:1234-1265`)

- Same CoefficientLayout swap as quantize
- Stride = `cx * BLOCK_DIM`
- When transposed: reads `block[slot_x * cy + slot_y]`

### Coefficient Order (`ac_group.rs:103-137`)

- DCT32x16: `coefficient_layout_order(16, 32, 2, 4)` (after stride fix)
- DCT16x32: `coefficient_layout_order(16, 32, 2, 4)`
- Both should produce the same order for the same normalized 16×32 layout

## Test Plan (after fix)

1. Forward DCT roundtrip test for each non-square transform
2. Forced-strategy test on 128x128 crops: butteraugli < 5.0
3. Full regression: `just rd-regression` and `just rd-regression-hd`
4. Mixed-strategy test at d=2.0 and d=3.0: no quality regression

## Files to Modify

1. `jxl_encoder/src/vardct/dct/forward_large.rs` — likely fix location
2. `jxl_encoder/src/vardct/transform.rs` — pixel extraction or DC extraction
3. `jxl_encoder/src/vardct/ac_strategy_search.rs` — re-enable after fix
4. `jxl_encoder/tests/clic2025.rs` — update baselines

## Priority

Fix DCT32x16/DCT16x32 first (smaller, easier to test). If that works, apply the
same fix pattern to DCT64x32/DCT32x64. AFV0-3 is a separate bug class.
