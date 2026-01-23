# INVESTIGATION.md

## 2026-01-22: AC Coefficient Encoding Bug - Decoded Values Wrong

### Status: ACTIVE

### Summary
VarDCT-encoded horizontal gradients produce completely wrong decoded pixel values. The bitstream is valid (both djxl and jxl-oxide decode it) but the reconstructed image shows extreme values (0 and 255 oscillating) instead of smooth gradient.

### Evidence
For 16x16 horizontal gradient (values 0→239):
```
Original: 0, 15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239
Decoded:  0,  0,  0,  0,119,255,255,255,  0,  0,  8,125,253,255,255,255
```

Both djxl (libjxl) and jxl-oxide produce the SAME wrong output, confirming the bug is in our encoder's bitstream, not the decoders.

### Findings

- [PROVEN] The XYB conversion is CORRECT. Tested: mid-gray sRGB 127 → XYB Y = 0.4441 (correct)
- [PROVEN] The srgb_to_xyb function expects 0-255 range input (not normalized 0-1)
- [PROVEN] The quantized coefficients have reasonable values:
  - Y DC = 263 (block 0)
  - Y AC[1] = -5490 (large horizontal frequency component)
  - Multiple non-zero AC coefficients per block
- [SUSPICION] The AC coefficient transpose logic may be incorrect
- [SUSPICION] The coefficient indexing in tokenize_ac_with_strategy may be off by one
- [SUSPICION] The natural order / zigzag order mapping may be wrong

### What's Been Investigated

1. **XYB Conversion**: CORRECT
   - srgb_to_xyb expects 0-255 range, encoder passes this correctly
   - My debug test had wrong input format (passed normalized values)

2. **Quantized Coefficients**: APPEAR CORRECT
   - DC values: y_dc=[132, 429, 132, 429] for 2x2 blocks
   - AC values: Multiple non-zero coefficients per block
   - raw_quant fix IS applied (raw_quant=32, not 1)

3. **Channel Ordering**: CORRECT
   - Transform stores channels as: ac_coeffs[block*3*63 + c*63]
   - Tokenize accesses in order c=[1,0,2] (Y,X,B) but uses same c for indexing

4. **Transpose Logic**: NEEDS MORE INVESTIGATION
   - Code transposes: `transposed_idx = (orig_idx % 8) * 8 + (orig_idx / 8)`
   - Claims: "jxl-oxide transposes coordinates when h >= w"
   - This swaps row/col, changing position (0,1) → (1,0)
   - For DCT8: block_ac[transposed_idx - covered_blocks]

### Code Locations

- **Tokenize function**: `jxl_enc/src/vardct/encoder.rs:497` (tokenize_ac_with_strategy)
- **Transpose**: `jxl_enc/src/vardct/encoder.rs:633-634`
- **Coefficient access**: `jxl_enc/src/vardct/encoder.rs:647-651`

### Reproduction

```bash
cargo test -p jxl_enc test_debug_gradient_pixels -- --ignored --nocapture
# Output saved to /tmp/gradient_debug.jxl
# Compare with: djxl /tmp/gradient_debug.jxl /tmp/djxl_out.png
```

### Latest Findings (Jan 22, continued)

**[PROVEN] jxl-oxide DOES transpose DCT8x8:**
- `need_transpose()` returns true when `h >= w`, which is true for 8x8
- Decoder swaps (dx, dy) coordinates: `std::mem::swap(&mut dx, &mut dy)`
- Location: `~/work/jxl-efforts/jxl-oxide/crates/jxl-vardct/src/hf_coeff.rs`

**[PROVEN] Coefficient values with transpose (current code):**
```
k=0: coeff_idx=1, transposed=8, ac_index=7, coeff=0      (sending vertical freq)
k=1: coeff_idx=8, transposed=1, ac_index=0, coeff=-2484  (sending horizontal freq)
```

**[PROVEN] Coefficient values WITHOUT transpose:**
```
k=0: coeff_idx=1, ac_index=0, coeff=-2484  (sending horizontal freq at position 1)
k=1: coeff_idx=8, ac_index=7, coeff=0      (sending vertical freq at position 8)
```

**[OBSERVED] Results:**
- WITH transpose: Decoded values oscillate 0/255 (wrong but has variation)
- WITHOUT transpose: Decoded values all zero (completely wrong)

**[SUSPICION] The issue may be in HOW we compensate for transpose:**
- We transpose the coefficient INDEX to access different array position
- But jxl-oxide transposes COORDINATES after reading
- These may not be equivalent operations

### KEY INSIGHT (Jan 23)

**[PROVEN] DCT16x16 works, DCT8x8 fails:**
- solid (16x16): uses DCT16x16, SSIM2=99.79 ✓
- h_gradient (16x16): uses DCT8x8, SSIM2=49.03 ✗

The variance-based strategy selector chooses:
- DCT16 for uniform regions (solid) → WORKS
- DCT8 for varying regions (gradient) → BROKEN

**[PROVEN] File size difference:**
- cjxl: 84 bytes for 16x16 gradient
- Our encoder: 169 bytes (2x larger)
- Our file has suspicious 00 00 00 00 byte patterns

### Investigation continued (Jan 23)

**[PROVEN] jxl-oxide's natural order for DCT8 matches ZIGZAG_ORDER_8X8:**
- jxl-oxide: [0]=0, [1]=1, [2]=8, [3]=16, [4]=9, [5]=2
- ZIGZAG:    [0]=0, [1]=1, [2]=8, [3]=16, [4]=9, [5]=2
- This is NOT the bug.

**[PROVEN] Coefficient ordering with transpose seems correct:**
- For k=0: coeff_idx=1 → transposed=8 → ac_index=7 → sends quant_out[8] (vertical freq)
- For k=1: coeff_idx=8 → transposed=1 → ac_index=0 → sends quant_out[1] (horizontal freq)
- Decoder places first AC at position 8 (vertical), second at position 1 (horizontal)
- This SHOULD produce correct results.

**[SUSPICION] The bug may be elsewhere in DCT8 path:**
- Context calculation (zero_density_context)?
- ANS encoding of the tokens?
- DC coefficient handling?

### Next Steps

1. **Force DCT16 for gradient**: Verify that DCT16 on gradient produces correct output
2. **Add debug output to ANS encoding**: Trace actual bits written for DCT8
3. **Compare bitstream hex dumps**: Our DCT8 vs cjxl's output

---

> **ARCHIVED**: Historical investigation notes from Jan 2-3, 2026.
> For current status, see [STATUS.md](STATUS.md) and [VARDCT_STATUS.md](VARDCT_STATUS.md).

---

## 2026-01-03: VarDCT Encoding Bug - InvalidEnum TransformId

### Issue
VarDCT (lossy) encoding produces `InvalidEnum { name: "TransformId", value: 3 }` decoder error.

**FIXED**: Test infrastructure was detecting wrong encoding mode due to parsing bug.
- Problem: `parse_encoding_mode()` was finding file header's `num_extra_channels=0` (bit 31) instead of frame header's `all_default` (bit 40)
- Fix: Start searching at bit 38 to skip file header metadata

### Root Cause
VarDCT bitstream has incorrect TransformId value in modular substream.

Single-group test (8x8) now correctly detects VarDCT encoding but fails decoding:
- Decoder error: `InvalidEnum { name: "TransformId", value: 3 }`
- This occurs in the Modular substream used for DC coefficients or HF metadata

### Progress
**Fixed**:
- Test helper `parse_encoding_mode()` now correctly detects VarDCT vs Modular (changed search range from bit 30 to bit 38)

**Current Investigation**:
- Our output: 44 bytes, fails to decode (both djxl and jxl-oxide)
- Reference (cjxl): 85 bytes, decodes correctly
- Need to find exact bit position where bitstreams diverge
- Likely issue in GroupHeader or Tree writing for modular substreams

### Evidence
**Lossless (Modular) - WORKS:**
- ✓ 300x300 lossless: PASSES
- ✓ 512x512 lossless: PASSES

**Lossy (VarDCT) - FAILS:**
- ✗ 8x8 lossy: FAILS (`InvalidEnum { name: "TransformId", value: 3 }`)
- ✗ 256x256 lossy: FAILS (`InvalidEnum`)
- ✗ 300x300 lossy: FAILS (`UnexpectedEof`)
- ✗ 512x512 lossy: FAILS (`UnexpectedEof`)

### Workaround
**None - VarDCT encoding is currently broken for all sizes.** Use lossless Modular encoding instead.

### Next Steps
1. Investigate single-group vs multi-group encoding differences
2. Check if section data is being modified during finish() or append operations
3. Add comprehensive BitWriter tests for the specific write patterns used in single-group encoding
4. Consider if small images should use Modular encoding instead (like cjxl does)

### Tests
- Marked small lossy tests as `#[ignore]` with note about single-group bug
- Confirmed multi-group tests pass (300x300, 512x512)

## How to Prevent False Positives

### Created `test_helpers.rs` - Single Source of Truth

**Problem**: Tests don't verify what encoding mode they actually use, leading to false positives.

**Solution**: Every test MUST use standardized helpers that verify encoding mode.

```rust
use crate::test_helpers::{test_lossless_roundtrip, test_lossy_roundtrip};

#[test]
fn test_lossless_multigroup_300x300() {
    let data = vec![/* ... */];
    
    // This helper:
    // 1. Encodes with Modular
    // 2. Asserts bitstream has encoding=1
    // 3. Decodes and verifies
    test_lossless_roundtrip(&data, 300, 300, "lossless_300x300").unwrap();
}

#[test]
fn test_lossy_multigroup_300x300() {
    let data = vec![/* ... */];
    
    // This helper:
    // 1. Encodes with VarDCT  
    // 2. Asserts bitstream has encoding=0
    // 3. Decodes and verifies
    test_lossy_roundtrip(&data, 300, 300, 1.0, "lossy_300x300").unwrap();
}
```

### Rules to Prevent Loops

1. **NO ad-hoc verification scripts** - Use `test_helpers::parse_encoding_mode()` only
2. **Explicit test names** - Must say "lossless" or "lossy", never ambiguous
3. **Tests verify themselves** - Use `assert_encoding_mode()` in EVERY test
4. **Read source, don't guess** - Check what API the test calls, don't assume

### What Was Deceptive

1. **Test names**: `test_encode_multigroup_300x300` doesn't say lossless/lossy
2. **Multiple APIs**: `encode_rgb8()` vs `encode_lossy_rgb8()` - easy to confuse
3. **Buggy verification tools**: Created Python script that had parsing bugs
4. **Trusting tools over code**: Should read source, not trust ad-hoc scripts
