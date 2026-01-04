# VarDCT AC Coefficient Investigation - 2026-01-03

## Problem Statement

VarDCT (lossy) encoding produces files that parse successfully but fail to decode. All AC coefficients are being quantized to zero, resulting in empty Pass Group sections.

## Test Case

**Image**: 8x8 red/blue checkerboard pattern
**Code**: `test_encode_lossy_8x8` in `jxl_enc/src/encoder.rs`
```rust
// Pattern:
// R B R B R B R B
// B R B R B R B R
// R B R B R B R B
// ...
```

## Investigation Timeline

### Session 1: Initial Hypothesis (WRONG)
- **Hypothesis**: Size header encoding mismatch causing bitstream divergence
- **Finding**: Size header was actually correct! Both use `small=true`, ratio=1 for 8x8
- **Outcome**: Red herring - fixed minor metadata differences but not the core issue

### Session 2: Tokenization Analysis
- **Discovery**: Only 3 tokens generated for pass group (should be many more)
- **Token values**: `[(7, 0), (0, 0), (7, 0)]` - **ALL values are 0**!
- **Meaning**: All 3 blocks report `nzeros=0` (no non-zero AC coefficients)

### Session 3: Root Cause Found
**Added debug output in `tokenize_ac_with_strategy()`:**
```
AC_DEBUG_STRAT: block 0, channel 0: ac_start=0, ac_end=63, effective_ac.len()=63
AC_DEBUG_STRAT: first 10 coeffs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
AC_DEBUG_STRAT: nzeros = 0
```

**Conclusion**: All 189 AC coefficients (1 block × 3 channels × 63 ACs) are ZERO.

## Why This is Wrong

For a checkerboard pattern:
- **Spatial domain**: Alternating high-contrast pixels (red vs blue)
- **Frequency domain**: Should have strong high-frequency components
- **Expected DCT**: Large AC coefficients in horizontal/vertical frequency bins
- **Actual DCT**: ALL ZEROS

This indicates the DCT transform or quantization is broken.

## Code Flow Analysis

### Transform Pipeline
```
RGB input → XYB conversion → DCT transform → Quantization → AC coefficients
```

**Key functions**:
1. `transform_xyb_image_with_strategy()` - Main entry point
2. DCT transforms in `jxl_enc_transforms` crate
3. Quantization in `jxl_enc/src/vardct/transform.rs`

### Tokenization Pipeline (WORKING CORRECTLY)
```
AC coefficients → count non-zeros → create tokens → build histograms → encode
```

**Key finding**: Tokenization is faithful - it correctly encodes "all zeros" because that's what it receives.

## Data Points

### Our Encoder
- **File size**: 49 bytes
- **DC coefficients**: [2969, 619, 1996] - NON-ZERO (correct!)
- **AC coefficients**: All 189 are ZERO (WRONG!)
- **Quantizer**: global_scale=8813, quant_dc=10

### Reference (libjxl)
- **File size**: 65 bytes (16 bytes larger)
- **Decodes successfully**: Produces valid image
- **Implication**: Reference has non-zero AC coefficients encoded

## Next Investigation Steps

1. **Check DCT output BEFORE quantization**
   - Add debug output in transform.rs before quantization
   - Verify DCT is producing non-zero values for checkerboard

2. **Check quantization matrices**
   - Verify quantization parameters match libjxl
   - Check if quantization is too aggressive (zeroing everything)

3. **Extract libjxl reference code**
   - Find equivalent C++ DCT/quantization code
   - Compare step-by-step with our implementation
   - Create simplified reference document

4. **Test with simpler pattern**
   - Try solid color (should have DC only, ACs=0)
   - Try vertical gradient (should have vertical frequency ACs)
   - Try checkerboard (should have both horizontal and vertical ACs)

## Files Modified for Debugging

### Added Debug Output
- `jxl_enc/src/vardct/encoder.rs:508-513` - AC coefficient values
- `jxl_enc/src/vardct/encoder.rs:983` - Token values
- `jxl_enc/src/vardct/encoder.rs:991` - Distribution info
- `jxl_enc/src/headers/file_header.rs:213-224` - Size header debug

### Fixed Issues
- `jxl_enc/src/headers/file_header.rs:375` - `modular_16_bit_buffer_sufficient` logic

## References

- **libjxl source**: `~/work/jxl-efforts/libjxl`
- **Key C++ files**:
  - `lib/jxl/enc_transforms.cc/h` - DCT transforms
  - `lib/jxl/enc_modular.cc` - Quantization
  - `lib/jxl/ac_strategy.cc` - AC coefficient handling
  - `lib/jxl/dct_util.cc` - DCT utilities

## Conclusion

The issue is NOT in the encoding/tokenization logic. The transform pipeline (DCT or quantization) is producing all-zero AC coefficients for a checkerboard pattern, which is incorrect. We need to extract and compare the reference C++ implementation to find where our transform differs.
