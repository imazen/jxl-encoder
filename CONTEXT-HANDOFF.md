# Context Handoff: DCT16x16/32x32 Multi-Block Bug

**Date:** 2026-02-01
**Branch:** main

## CRITICAL FINDING

**DCT16x16 was NOT fixed as previously claimed.** Both DCT16x16 and DCT32x32 have the
same multi-block bug. Single-block encoding works perfectly; multi-block fails catastrophically.

## Bug Summary

| Condition | SSIM2 |
|-----------|-------|
| 16x16 image (1 DCT16x16 block) | 74.82 ✓ |
| 32x32 image (multiple blocks) | -63.89 ✗ |
| 256x256+ (many blocks) | -67 ✗ |

## Root Cause Analysis (UPDATED)

### Test Evidence

**Uniform value works:**
```
Input: all pixels = 0.5
Decoded: all pixels = 0.500 ✓
```

**Two-value pattern fails:**
```
Input: top half = 0.25, bottom half = 0.75
DCT8  decoded: [0.251, 0.251, 0.748, 0.748] ✓
DCT16 decoded: [0.179, 0.230, 0.878, 0.889] ✗ (contrast amplified)
```

**Quadrant pattern shows "leakage":**
```
Input: TL=0, TR=1, BL=0, BR=1
dc_from_dct_16x16 output: [0.0934, 0.9066, 0.0934, 0.9066]
Expected 8x8 averages:    [0.0, 1.0, 0.0, 1.0]
```

### Hypothesis

The `dc_from_dct_16x16()` produces IDCT-transformed values that contain "leakage"
between quadrants. This is mathematically correct for the 2x2 IDCT, BUT the decoder's
`LowestFrequenciesFromDC` expects **actual 8x8 block averages**, not IDCT-transformed values.

When decoder applies DCT on leaked values → double transformation → wrong LLF → wrong pixels.

For uniform input, leakage doesn't matter (all quadrants equal). For non-uniform input,
leakage + decoder DCT = contrast amplification.

## Diagnostic Tests Created

All in `jxl_enc/tests/llf_invariants.rs`:

1. `diag_dct16x16_progressive_sizes` - Shows 16x16 works, 32x32+ fails
2. `diag_dct16x16_dc_trace` - Shows extracted DC values have leakage
3. `diag_dct16x16_uniform` - Shows uniform works perfectly
4. `diag_dct16x16_two_values` - Shows contrast amplification bug clearly
5. `diag_dct16x16_layout` - Shows coefficient positions are correct
6. `diag_dct16x16_iteration` - Shows error grows with block position

## Key Files

| File | Lines | Function |
|------|-------|----------|
| `jxl_enc/src/tiny/dct.rs` | 360-389 | `dc_from_dct_16x16` |
| `jxl_enc/src/tiny/dct.rs` | 599-657 | `dc_from_dct_32x32` |
| `jxl_enc/src/tiny/encoder.rs` | 924-935 | DCT16x16 DC storage |
| `jxl_enc/src/tiny/encoder.rs` | 937-949 | DCT32x32 DC storage |

## Proposed Fix Direction

Change `dc_from_dct_16x16/32x32` to return actual 8x8 block averages instead of
IDCT-transformed values. This may require:

1. Different formula for DC extraction that doesn't have leakage
2. OR: Understanding what values the C++ encoder actually produces
3. OR: Understanding what format the decoder expects

## Next Steps

1. Build C++ libjxl-tiny and trace what DC values it produces for the two-value pattern
2. Compare with our output to find the mismatch
3. Look at jxl-oxide/jxl-rs decoder's `LowestFrequenciesFromDC` to understand expected input

## Test Commands

```bash
# Key diagnostic tests
cargo test -p jxl_enc --test llf_invariants diag_dct16x16_two_values -- --ignored --nocapture
cargo test -p jxl_enc --test llf_invariants diag_dct16x16_progressive_sizes -- --ignored --nocapture
cargo test -p jxl_enc --test llf_invariants diag_dct16x16_uniform -- --ignored --nocapture
```

## C++ Reference Files

- `~/work/jxl-efforts/libjxl/lib/jxl/enc_transforms-inl.h` - `DCFromLowestFrequencies`, `ReinterpretingIDCT`
- The C++ uses `ComputeScaledIDCT<2,2>` same as us - WHY does it work there?

## Working Tree

Clean. No uncommitted changes.
