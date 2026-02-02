# Context Handoff: DCT32x32 Bug Investigation

**Date:** 2026-02-01
**Branch:** main (ahead of origin by ~60 commits)
**Last commit:** b4cb866 style: cargo fmt

## What Was Done This Session

### Part 1: DCT16x16 Fix Completed (previous session)

Fixed two bugs causing catastrophic DCT16x16 quality loss. Bugs and fixes documented
in CLAUDE.md "Resolved Bugs" section. All DCT16x16 tests pass.

### Part 2: DCT32x32 Validation Started (this session)

Added `force_strategy` field to `TinyEncoder` for testing specific strategies:
```rust
encoder.force_strategy = Some(4); // RAW_STRATEGY_DCT32X32
```

Created DCT32x32 test infrastructure:
1. **Layer 1b test** (`layer1b_dc_spatial_order_dct32x32`) — PASSES
   - Proves `dc_from_dct_32x32` FIXED version produces correct spatial variation
   - Proves OLD (no transpose) version produces wrong (transposed) variation

2. **Layer 2-4 roundtrip tests** — FAILING
   - `layer2_single_group_dct32x32_decode_jxl_oxide`: SSIM2 = -67.00
   - `layer2_single_group_dct32x32_decode_djxl`: SSIM2 = -67.01
   - This is the same catastrophic pattern as DCT16x16 before the fix

## Active Bug: DCT32x32 Roundtrip Quality

**Symptom:** SSIM2 = -67 (catastrophic) when forcing DCT32x32 on 256x256 frymire crop.

**What we know:**
1. `dc_from_dct_32x32` math is correct (Layer 1b test proves it)
2. The fix added rows→transpose→rows pattern (matching C++)
3. LLF position formula uses the corrected 2D grid check

**What to investigate:**
1. Check if LLF positions for DCT32x32 are correctly computed in all paths:
   - `quantize_ac_block`: check `(idx / grid_width) < cy && (idx % grid_width) < cx`
   - Y roundtrip dequantization path
   - CfL skip regions
2. Check forward `dct_32x32` transform — does it produce coefficients in the expected layout?
3. Check if coefficient tokenization/ordering is correct for DCT32x32

**Debug approach:**
Create a diagnostic test that encodes a 32x32 solid block, forces DCT32x32,
and prints the raw coefficient values before/after quantization to trace the bug.

## Files Changed This Session

| File | Change |
|------|--------|
| `jxl_enc/src/tiny/encoder.rs` | Added `force_strategy: Option<u8>` field and helper |
| `jxl_enc/src/tiny/ac_strategy.rs` | Added `AcStrategyMap::force_strategy()` method |
| `jxl_enc/tests/llf_invariants.rs` | Added DCT32x32 Layer 1b, 2, 3, 4 tests |

## Current State

### Working tree
- Clean (user's DCT16x16 hack was stashed)

### Stashes
```
stash@{0}: temp: force-DCT16x16 hack for testing (user's, restore later)
stash@{1}: debug: forced DCT16x16 in ac_strategy.rs (older copy)
stash@{2}: pre-existing cluster.rs formatting changes
stash@{3}: WIP from earlier session
```

### Test results
- 9 non-ignored tests pass (Layer 1 + Layer 1b)
- Layer 2-4 DCT32x32 tests FAIL with SSIM2 = -67

## Tests in `llf_invariants.rs`

26 total tests. 9 run by default, 17 ignored (require test images).

| Layer | Tests | Status | What they prove |
|-------|-------|--------|-----------------|
| 1 | 7 | PASS | LLF position formulas |
| 1b DCT16 | 1 | PASS | dc_from_dct_16x16 spatial ordering |
| 1b DCT32 | 1 | PASS | dc_from_dct_32x32 spatial ordering |
| 2 DCT16 | 2 | PASS | 256x256 roundtrip |
| 2 DCT32 | 2 | **FAIL** | SSIM2 = -67 |
| 3 DCT16 | 2 | PASS | Multi-group roundtrip |
| 3 DCT32 | 2 | untested | Multi-group roundtrip |
| 4 DCT16 | 4 | PASS | Quality comparison |
| 4 DCT32 | 2 | untested | Quality comparison |
| diag | 3 | informational | Size analysis |

## Next Steps

1. **Debug DCT32x32 roundtrip** — the dc_from_dct_32x32 math is correct but something
   else in the DCT32x32 pipeline is wrong. Likely culprits:
   - Forward DCT32x32 producing wrong coefficient layout
   - Coefficient ordering/tokenization
   - Quant weight indexing

2. **Diagnostic test** — create `diag_dct32x32_solid_32x32` similar to the DCT16x16
   diagnostic that prints coefficients at each stage

3. **Check `dct_32x32` forward transform** — verify it matches C++ output

## Key Files

| File | Role |
|------|------|
| `jxl_enc/src/tiny/dct.rs:506-535` | `dct_32x32` forward transform |
| `jxl_enc/src/tiny/dct.rs:599-657` | `dc_from_dct_32x32` — fixed |
| `jxl_enc/src/tiny/encoder.rs:718-729` | DCT32x32 transform execution |
| `jxl_enc/src/tiny/encoder.rs:937-949` | DC extraction for DCT32x32 |
| `jxl_enc/src/tiny/ac_strategy.rs` | Strategy selection |
| `jxl_enc/tests/llf_invariants.rs` | Layered invariant tests |

## Run Tests

```bash
# Non-ignored tests (should pass)
cargo test -p jxl_enc --test llf_invariants

# DCT32x32 roundtrip tests (FAILING)
cargo test -p jxl_enc --test llf_invariants layer2_single_group_dct32x32 -- --ignored --nocapture

# All ignored tests
cargo test -p jxl_enc --test llf_invariants -- --ignored --nocapture
```
