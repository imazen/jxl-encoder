# Context Handoff - DCT4X8 Quality Fix Complete

**Date:** 2026-02-02
**Last Commits:**
- `ff804e5` - docs: update PROVEN_INVARIANTS.md with DCT4X8 Layer 4 completion
- `b74f3b6` - fix: implement correct DCT4X8/DCT8X4 parametric quantization weights

## What Was Accomplished

### DCT4X8/DCT8X4 Quality Bug - FIXED

The catastrophic quality issue with DCT4X8 encoding on real photos has been resolved.

**Root Cause:** The encoder was using DCT8 weights with position 8 boosted, but the jxl-oxide decoder expects **row-interleaved weights** generated from parametric band parameters:
1. Generate 8x4 matrix from `DCT4X8_BAND_PARAMS`
2. Duplicate each row to get 8x8 (matching interleaved coefficient layout)
3. Reciprocate all weights to match encoder convention

**Fix Applied:**
- Added `DCT4X8_BAND_PARAMS` to `quant.rs` (from jxl-oxide dequant.rs:44-48)
- Implemented `generate_dct4x8_weights()` with proper row duplication and reciprocation
- Re-enabled DCT4X8/DCT8X4 strategy selection (`k4x8mul2 = 0.88` in ac_strategy.rs)
- Fixed raw vs bitstream strategy code handling in multiple files

**Quality Improvement:**
| Metric | Before | After |
|--------|--------|-------|
| DCT4X8 range | [-1.47, 3.17] | [-0.18, 2.16] |
| Pixels >1.5 | 1165 | 30 |

**RD Regression:** PASSES with improvements
- frymire: 2.5% smaller, +0.93 SSIM2 at d=0.25
- img11: butteraugli 8% better
- img13: butteraugli 15% better

## Current State

### DCT4X8/DCT8X4 Feature - COMPLETE
All layers proven (see PROVEN_INVARIANTS.md):
- [x] Layer 1: Forward transforms, DC extraction, quant weights
- [x] Layer 2: Encoder integration, block context map
- [x] Layer 3: jxl-rs and jxl-oxide decode successfully
- [x] Layer 4: Quality metrics and RD regression pass

### Files Modified
- `jxl_enc/src/tiny/quant.rs` - Parametric weight generation
- `jxl_enc/src/tiny/ac_strategy.rs` - Re-enabled strategy selection
- `jxl_enc/src/tiny/ac_group.rs` - Fixed strategy code handling
- `jxl_enc/src/tiny/coeff_order.rs` - Fixed raw vs bitstream codes
- `jxl_enc/src/tiny/encoder.rs` - Fixed strategy code usage
- `jxl_enc/tests/dct4x8_diagnostic.rs` - New diagnostic test file

## Codebase Notes

### Strategy Code Confusion (RESOLVED)
There are TWO strategy code systems:
- **Raw codes (0-6):** Internal, used by `ac_strategy_info()`, `estimate_entropy()`
- **Bitstream codes:** 0=DCT8, 4=DCT16X16, 5=DCT32X32, 6=DCT8X16, 7=DCT16X8, 12=DCT4X8, 13=DCT8X4

`STRATEGY_CODE_LUT` converts raw → bitstream. Functions now document which they expect.

### Quantization Weight Convention
`QUANT_WEIGHTS` and `quant_weights()` return **inverted** weights (small values ~0.0003-0.002).
- Encoder uses `1.0 / weight` to quantize
- Decoder uses `weight` to dequantize
- Parametric formulas generate large values, must reciprocate before storing

## Potential Next Steps

1. **DCT32x32 Bug** - Still disabled (see Known Bugs in CLAUDE.md). The DC extraction has Gibbs phenomenon issues with 4-point IDCT.

2. **Further RD Tuning** - The `mul4x8` multiplier (0.88) could be tuned based on more extensive testing.

3. **Performance Optimization** - The parametric weight generation runs at startup via LazyLock; could precompute as const if needed.

## Test Commands

```bash
# Run DCT4X8 diagnostic tests
cargo test -p jxl_enc --test dct4x8_diagnostic -- --ignored --nocapture

# Run RD regression (requires clic2025 corpus)
just rd-regression

# Run all lib tests
cargo test -p jxl_enc --lib
```

## Important Files to Read

1. `PROVEN_INVARIANTS.md` - Layer-by-layer proof status
2. `CLAUDE.md` - Project instructions, known bugs, resolved bugs
3. `jxl_enc/src/tiny/quant.rs` - Weight generation (lines 460-545)
4. `jxl_enc/src/tiny/ac_strategy.rs` - Strategy selection (lines 435-445 for DCT4X8 multiplier)

---
**Delete this file after loading into new session.**
