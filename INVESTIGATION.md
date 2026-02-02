# INVESTIGATION.md

## 2026-02-02: Quantization Calibration Gap vs libjxl

### Status: ACTIVE

### Problem Statement

At the same distance parameter, our encoder produces 10-15% smaller files but 1-6 SSIM2 points worse quality compared to cjxl. The gap grows with distance:

| Distance | Size vs e7  | Quality vs e5 | Notes |
|----------|-------------|---------------|-------|
| d=1.0    | -13% smaller| -1.2 SSIM2    | Close |
| d=2.0    | -13% smaller| -2.4 SSIM2    | Gap growing |
| d=4.0    | -10% smaller| -3.5 SSIM2    | Significant |

Comparison baseline: cjxl e5 (Hare) has similar feature set to ours (DCT8 + some larger blocks, no error diffusion/splines/patches).

### Key Observation

The gap is NOT from missing features. Comparing e5→e7 shows only ~1 SSIM2 improvement from advanced features (error diffusion, splines, patches) at d=1.0. The gap we see vs e5 must be from quantization calibration differences.

### Hypotheses (to investigate)

1. **[THREAD] global_scale calibration** — `DistanceParams::compute()` may produce different `global_scale` than libjxl
2. **[THREAD] quant_dc calibration** — DC quantization step might differ
3. **[THREAD] Adaptive quantization masking** — Per-block `raw_quant` values differ in distribution
4. **[THREAD] AC strategy cost model** — We merge blocks too aggressively, picking larger transforms when smaller would preserve detail
5. **[THREAD] Coefficient thresholding** — `QuantizeBlockAC` threshold zeros too many coefficients
6. **[THREAD] Gaborish interaction** — Our sharpening kernel differs subtly from libjxl's

### Investigation Log

#### 2026-02-02 04:30 — Initial findings

**DistanceParams match libjxl-tiny exactly.** At d=2.0: global_scale=3670, quant_dc=10.

**File size comparison at d=2.0 on img10 (512x768):**
- libjxl-tiny: 53,895 bytes
- Our encoder: 37,782 bytes (-30% vs tiny)
- cjxl e7: 43,487 bytes (-13% vs e7)

**Quality comparison:**
- Our encoder: SSIM2 76.4
- cjxl e7: SSIM2 79.4

**Key insight:** We produce 30% smaller files than libjxl-tiny with identical DistanceParams. This isn't from DistanceParams differences — it's from our entropy coding (ANS + dynamic Huffman clustering) being more efficient.

**But we're also lower quality.** The gap must be in what we quantize, not how we encode it.

**Hypothesis update — kAcQuant is NOT the issue:**
- libjxl-tiny uses `kAcQuant = 0.8`
- Full libjxl uses `kAcQuant = 0.765` but in a DIFFERENT formula

#### 2026-02-02 04:45 — Initial hypothesis: content-adaptive global_scale

~~**[PROVEN] libjxl uses content-adaptive global_scale, we don't.**~~

**[RULED OUT] — See 05:10 update below.**

#### 2026-02-02 05:10 — Correction: global_scale is NOT content-adaptive in main path

**[PROVEN] libjxl's main encoding path uses FIXED global_scale, not content-adaptive.**

Reading `enc_heuristics.cc` carefully:

```cpp
// For Hare mode (effort 5) or slower with adaptive quant:
float q = 0.39 / cparams.butteraugli_distance;
quantizer.ComputeGlobalScaleAndQuant(quant_dc, q, 0);  // MAD=0 (fixed)

// Then later, just converts using the already-set global_scale:
quantizer.SetQuantFieldRect(initial_quant_field, r, &raw_quant_field);
```

The `SetQuantField` with median/MAD calculation exists (`quantizer.cc:90`) but is only called
from `enc_adaptive_quantization.cc` for specific refinement loops, NOT the main encoding path.

**Key insight:** In the main encoding path, libjxl uses a FIXED formula `q = 0.39/distance`
with MAD=0, very similar to our approach. The difference is the constant:
- Our encoder: AC_QUANT = 0.8 (from libjxl-tiny)
- libjxl Hare: q = 0.39

But wait — **AC_QUANT cancels out!** Tracing the quantization:
```
qac = params.scale * raw_quant
    = (global_scale / 65536) * (quant_field_float * inv_scale)
    = (global_scale / 65536) * (quant_field_float * 65536 / global_scale)
    = quant_field_float
```

Changing AC_QUANT (and thus global_scale) proportionally changes inv_scale, which
proportionally changes raw_quant, and the two cancel out in the final qac value!

**Tested this hypothesis:** Changed AC_QUANT from 0.8 to 0.39. Result: almost no change
in file size (140KB vs 141KB) or quality (75.7 vs 76.0). Confirms the cancellation.

**Real cause of quality gap must be elsewhere:**
- K_AC_QUANT in adaptive_quant.rs (ours: 0.8294, libjxl: 0.765)
- Adaptive quantization masking algorithm differences
- AC strategy cost model
- Missing features (DCT4x8, error diffusion)

#### 2026-02-02 05:15 — Testing K_AC_QUANT changes

**[TESTED] Changing K_AC_QUANT to 0.765 (libjxl) produces WORSE results:**

With K_AC_QUANT=0.765 and AC_QUANT=0.39 (matching libjxl ratio):
- File size: 133KB (smaller than baseline 143KB)
- SSIM2: 74.96 (worse than baseline 76.11)

The K_AC_QUANT/AC_QUANT ratio does affect quantization, but matching libjxl's
constants doesn't help. The quality difference must be in the adaptive quant
algorithm itself, not just the constants.

**[FIXED] Ported libjxl's AdjustQuantField mean/max blending:**

libjxl blends between max and mean for multi-block quant values based on distance:
- d ≤ 1.54: use max
- d > 2.82: use mean
- In between: linear interpolation

This helps at high distances but didn't close the gap significantly.

**Current quality gap vs cjxl e7:**

| Distance | Our Size | Our SSIM2 | cjxl Size | cjxl SSIM2 | Gap |
|----------|----------|-----------|-----------|------------|-----|
| d=2.0    | 143KB    | 76.1      | 162KB     | 81.2       | -5.1 |
| d=4.0    | 88KB     | 63.4      | 100KB     | 70.3       | -6.9 |

**Root cause is NOT K_AC_QUANT constants.** Must be in:
1. The adaptive quant masking computation itself (pre-erosion, fuzzy erosion, modulations)
2. Coefficient thresholding in QuantizeBlockAC
3. Missing features (DCT4x8, error diffusion, splines)

**Next steps:**
1. Compare pre-erosion/fuzzy erosion output between our code and libjxl
2. Check coefficient thresholding differences in QuantizeBlockAC
3. Consider implementing DCT4x8/DCT8x4 (helps at high distances)
4. Consider implementing error diffusion

---

### Frymire Bug (Separate Issue)

Frymire (1118x1105) shows catastrophic 13 SSIM2 gap at d=1.0 (74 vs 87) with 23% smaller files. This is NOT the same calibration issue — it's a specific bug with that image. Will investigate separately after general calibration gap is understood.

---

## 2026-01-30: Tiny Encoder Quality Ceiling — SSIM2 Plateaus at ~82.5

### Status: RESOLVED (adaptive quantization ported, commit b7b80fd)

### Summary

Lowering the `distance` parameter below 0.5 produces almost no quality improvement.
SSIM2 flatlines around 82.5 regardless of distance, and file sizes barely grow.
This indicates quantization is not actually getting finer at low distance values.

By comparison, cjxl (libjxl reference) continues improving to SSIM2 ~92 at d=0.05,
and its file sizes grow proportionally (8.8x from d=2.0 to d=0.01).

### Evidence

Test image: CLIC 2025 final-test landscape (2048x1360, 48 groups).
Measured 2026-01-30 with `fast-ssim2-cli`, decoded via jxl-oxide.

```
distance   tiny_bytes tiny_bpp  tiny_s2  |  cjxl_bytes cjxl_bpp  cjxl_s2
------------------------------------------------------------------------------
2.0000         795705     2.29    67.77  |      355520     1.02    72.88
1.0000         802145     2.30    79.43  |      609347     1.75    81.28
0.5000         892476     2.56    81.36  |     1015857     2.92    87.03
0.2500         901855     2.59    81.49  |     1474018     4.23    90.01
0.1000         985092     2.83    82.51  |     2382925     6.84    91.72
0.0500        1005328     2.89    82.46  |     3129919     8.99    92.27
0.0100        1025536     2.95    82.53  |     3129919     8.99    92.27
```

Key observations:
- **Tiny encoder:** d=0.5 → d=0.01 gains only +1.2 SSIM2 (81.36 → 82.53)
- **Tiny encoder:** file size grows only 1.15x over that range (892KB → 1026KB)
- **cjxl:** d=0.5 → d=0.01 gains +5.2 SSIM2 (87.03 → 92.27)
- **cjxl:** file size grows 3.1x over that range (1016KB → 3130KB)
- **cjxl also has a floor:** d=0.05 and d=0.01 produce identical output (3130KB, SSIM2=92.27)

### Findings

- [PROVEN] Quality stops improving below distance=0.5 for the tiny encoder
- [PROVEN] File sizes barely grow, confirming quantization isn't getting finer
- [SUSPICION] `DistanceParams::compute()` has `global_scale = clamp(scale, 1, scaled_quant_dc)` — the upper clamp on `scaled_quant_dc` may be binding at low distances, preventing global_scale from growing large enough
- [SUSPICION] `raw_quant_uniform()` uses hardcoded `0.73` quant field — at low distances, `inv_scale` changes but the uniform approximation may not produce enough dynamic range
- [SUSPICION] The sRGB gamma approximation (`powf(2.2)` / `powf(1.0/2.2)`) instead of the true sRGB transfer function contributes some fixed error, but this alone wouldn't explain the plateau since cjxl also goes through XYB and achieves 92+
- [THREAD] Need to print actual `global_scale`, `quant_dc`, `scale`, and `raw_quant` values at each distance to identify which parameter is clamping

### Likely Root Causes (ranked)

1. **Quantization parameter saturation**: The `DistanceParams` math clamps or saturates somewhere, so below d=0.5 the actual quantization step size stops shrinking. The fact that file sizes barely grow is strong evidence.

2. **Fixed `raw_quant` approximation**: `raw_quant_uniform()` uses a single hardcoded estimate (0.73) of the quant field. Without per-block adaptive quantization, every block gets the same quantization regardless of content. This caps quality for complex regions.

3. **sRGB gamma mismatch**: The encoder uses `powf(2.2)` to linearize, but the true sRGB transfer function has a linear segment below 0.0031308. The decoder (jxl-oxide) uses the correct sRGB EOTF. This creates a small but systematic mismatch in dark tones. However, this is a fixed ~0.5-1.0 SSIM2 penalty, not the cause of the plateau.

### Reproduction

```bash
cargo run --release --example distance_sweep
```

Output files in `/mnt/v/output/jxl-encoder-rs/distance_sweep/`.

### Next Steps

1. Print `DistanceParams` fields at each distance level to find which parameter clamps
2. Compare our `global_scale` / `quant_dc` values against libjxl-tiny at the same distances
3. If parameter saturation confirmed, fix the clamping math
4. Separately: implement per-block adaptive quantization (replaces `raw_quant_uniform()`)

---

## 2026-01-22: AC Coefficient Encoding Bug - Decoded Values Wrong

### Status: [RESOLVED] — Fixed Jan 27, 2026 (commits 25dfc9b, 7214a0c, ded4c0a)

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

### Status: [RESOLVED]

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

## 2026-02-02: Implementation Differences Between libjxl-tiny and Full libjxl

### Status: ACTIVE

### Summary

The tiny encoder was ported from libjxl-tiny, which has deliberate simplifications vs full libjxl.
These simplifications hurt quality at d≥2.0. The key differences are documented below.

### Findings

#### [PROVEN] kDampenRampStart difference (CRITICAL)

```
libjxl-tiny:  kDampenRampStart = 7.0
libjxl full:  kDampenRampStart = 2.0
```

**Impact**: Dampening reduces the effect of adaptive quant modulations at high distances.
At d=4.0:
- libjxl full: dampen = 1 - (4.0 - 2.0) / (14.0 - 2.0) = 0.833 (16.7% reduced)
- Our code: dampen = 1.0 (no reduction)

This means our adaptive quant has FULL effect even at high distances where it should be reduced.
The dampen factor blends toward a uniform `base_level` at high distances.

#### [PROVEN] kGam/kGamma sign difference (CRITICAL)

```
libjxl-tiny:  kGam = -0.15526878 * 0.693147 ≈ -0.1076 (NEGATIVE)
libjxl full:  kGamma = +0.1005613337192697 (POSITIVE)
```

**Impact**: GammaModulation adjusts quantization based on local luminance ratio.
The sign difference means the adjustment goes in OPPOSITE directions:
- Positive kGamma: high luminance ratio → MORE bits
- Negative kGam: high luminance ratio → FEWER bits

#### [PROVEN] kSGVOffset difference

```
libjxl-tiny:  kSGVOffset = 7.14672470003
libjxl full:  kSGVOffset = 7.7825991679894591
```

**Impact**: Used in RatioOfDerivativesOfCubicRootToSimpleGamma calculation.
Affects the gamma ratio computation but likely less impactful than the sign difference.

#### [PROVEN] GammaModulation accumulation difference

libjxl-tiny: Averages (ratio_r + ratio_g)/2 per pixel, then divides by 64
libjxl full: Adds ratio_r and ratio_g separately (128 terms), then multiplies by (0.5/64)

The normalization is equivalent, but the sign difference on kGamma is the key issue.

#### [PROVEN] MaskingSqrt constants differ

```
libjxl-tiny:  kLogOffset = 26.481471032459346, kMul = 211.50759899638012
libjxl full:  kLogOffset = 27.505837037000106, kMul = 211.66567973503678
```

Minor difference, likely not the main quality gap cause.

### Recommended Fixes (Priority Order)

1. **Change kDampenRampStart from 7.0 to 2.0** - This will dampen adaptive quant modulations
   earlier, making the quant field more uniform at high distances. This is the most likely
   fix for the d≥2.0 quality gap.

2. **Change kGam sign from negative to positive** - This changes GammaModulation behavior
   fundamentally. Need to test carefully as libjxl-tiny's choice may have been intentional
   for their simplified pipeline.

3. Update kSGVOffset and MaskingSqrt constants to match full libjxl.

### Next Steps

1. Test kDampenRampStart=2.0 alone on CLIC 2025 images at d=2.0 and d=4.0
2. Compare RD curves before and after
3. If successful, commit and add to regression tests
