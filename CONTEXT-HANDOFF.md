# Context Handoff - VarDCT Quality Bug Investigation

**Created**: Jan 22, 2026
**Session Focus**: Investigating why real photos encode with 4x larger files and 4x worse quality than libjxl

## Critical Finding: ROOT CAUSE IDENTIFIED

### The Bug

**Location**: `jxl_enc/src/vardct/transform.rs:60`

```rust
// CURRENT CODE - WRONG
let raw_quant = 1i32;  // Hardcoded!

// SHOULD BE
let raw_quant = quant_field.get(bx, by) as i32;  // Use per-block values
```

The `QuantField` is computed with adaptive per-block quantization values but **never used** during actual coefficient quantization.

### Why This Matters

For distance=1.0 with `global_scale=8813`:

| Parameter | Current (Wrong) | Expected (Correct) |
|-----------|-----------------|-------------------|
| raw_quant | 1 | ~38 |
| qac | 0.134 | 5.11 |
| Zeroing threshold (HF) | 0.019 | 0.0005 |

**Effect**: With `raw_quant=1`, the zeroing threshold is 38x higher, causing fine image detail to be discarded. This explains the blurry 8x8 blocks and poor SSIM2 scores.

### Quality Metrics

| Metric | Our Encoder | libjxl (cjxl) |
|--------|-------------|---------------|
| File size (1507x2048 photo, d=1.0) | 760KB | 184KB |
| SSIM2 | 23.5 | 82.6 |
| Bits per coefficient | 0.65 | 0.16 |

## Investigation Path (What Was Checked)

### 1. Quantization Formula Verification
- **Verified CORRECT**: `quantized = inv_dequant_matrix[i] * qac * coeff`
- Matches libjxl's `QuantizeBlockAC` in `enc_group.cc:94-98`
- Our weights (560 for Y DC, 196 for Y HF) match libjxl's `InvDequantMatrix`

### 2. Global Scale Computation
- **Verified CORRECT**: `global_scale ≈ 8813` for distance=1.0
- Computed from: `GLOBAL_SCALE_DENOM * quant_ac / quant_field_target`
- Where `quant_ac = 0.765 / distance` and `quant_field_target = 5.0`

### 3. Per-Block Quant Field
- **EXISTS but UNUSED**: `QuantField` struct in `heuristics/adaptive_quant.rs`
- Stores u8 values (1-255) per block based on variance analysis
- `base_quant ≈ 10` for distance=1.0
- `VarDctEncoder` computes it but `transform_and_quantize` ignores it

### 4. Threshold Value
- **Verified CLOSE**: Our `DEFAULT_THRESHOLD = 0.5` vs libjxl's `0.58`
- This is NOT the root cause (small difference)

### 5. Entropy Coding Efficiency
- **SECONDARY ISSUE**: 0.65 bits/coeff vs expected 0.1-0.3
- Likely caused by unusual coefficient distribution from wrong raw_quant
- May have additional ANS encoding issues

## Key Code Locations

### Quantization
- `jxl_enc/src/vardct/transform.rs:60` - **BUG LOCATION** (raw_quant=1)
- `jxl_enc/src/vardct/enc_coeff.rs:150` - Quantization formula (correct)
- `jxl_enc/src/vardct/quantizer.rs:64-92` - QuantizerParams::from_distance (correct)

### Quant Field (computed but unused)
- `jxl_enc/src/heuristics/adaptive_quant.rs:10` - QuantField struct
- `jxl_enc/src/heuristics/adaptive_quant.rs:40` - compute_adaptive()
- `jxl_enc/src/vardct/encoder.rs:135` - Where it's computed
- `jxl_enc/src/vardct/encoder.rs:1202` - Only used for metadata, not quantization!

### libjxl Reference
- `~/work/jxl-efforts/libjxl/lib/jxl/enc_group.cc:63-64` - qac computation
- `~/work/jxl-efforts/libjxl/lib/jxl/enc_group.cc:94-98` - quantization loop
- `~/work/jxl-efforts/libjxl/lib/jxl/quantizer.cc:84` - raw_quant from quant_field

## Fix Required

### Step 1: Update transform_and_quantize signature
```rust
pub fn transform_and_quantize(
    xyb_data: &[&[f32]; 3],
    width: usize,
    height: usize,
    quantizer: &QuantizerParams,
    quant_field: &QuantField,  // ADD THIS
) -> TransformedData {
```

### Step 2: Use per-block quant values
```rust
// At line 60, REPLACE:
let raw_quant = 1i32;

// WITH:
let raw_quant = quant_field.get(bx, by) as i32;
```

### Step 3: Update all callers
- `VarDctEncoder::encode_frame()` needs to pass `&self.quant_field`
- Similar fix needed in `transform_and_quantize_with_strategy()`

### Step 4: Verify
```bash
cargo test test_save_broken_image -- --ignored --nocapture
# Then visually compare /tmp/broken_decoded.png with original
# SSIM2 should improve from 23 to >70
```

## Test Commands

```bash
# Quality enforcement test (synthetic images)
cargo test test_vardct_quality_enforcement -- --ignored --nocapture

# Real photo quality test
cargo test test_save_broken_image -- --ignored --nocapture
# Output: /tmp/broken.jxl, /tmp/broken_decoded.png

# Compare with libjxl
~/work/jxl-efforts/libjxl/build/tools/cjxl \
  ~/work/codec-corpus/clic2025/validation/097cb426910ba8ce2525dd8bb7fb1777.png \
  /tmp/reference_d1.jxl -d 1.0
~/work/jxl-efforts/libjxl/build/tools/djxl /tmp/reference_d1.jxl /tmp/reference_d1.png

# SSIM2 comparison (if fast-ssim2 available)
ssim2 original.png decoded.png
```

## Python Analysis Scripts (in /tmp/)

- `/tmp/quant_formula.py` - Compares our formula vs potential alternatives
- `/tmp/check_weights.py` - Analyzes DCT8 perceptual weights and thresholds
- `/tmp/extract_quant.py` - Reads quantizer params from JXL files

## Commits Made This Session

1. `f699e9b` - docs: document raw_quant=1 bug as root cause of quality gap

## Previous Session Summary

The previous session fixed three bugs that caused VarDCT quality to go from SSIM2 -1000 to +90:
1. Transpose bug in `tokenize_ac_with_strategy`
2. Wrong LLF-to-DC scaling in `llf_to_dc_dct16`
3. DC quantization missing divide-by-8

These fixes made synthetic tests pass (SSIM2 85-95), but real photos still had issues (SSIM2 23).

## Next Steps

1. **Implement the fix** - Pass QuantField to transform functions, use per-block raw_quant
2. **Verify quality improves** - SSIM2 should go from 23 to >70 for real photos
3. **Check file size** - Should decrease from 760KB closer to libjxl's 184KB
4. **If still large**: Investigate ANS entropy coding efficiency separately

## User Messages This Session

1. "Please continue the conversation from where we left it off without asking the user any further questions. Continue with the last task that you were asked to work on."
2. "write full handoff"

## Important Notes

- The QuantField EXISTS and is COMPUTED correctly - it's just not USED
- This bug explains why synthetic tests pass but real photos fail
- Synthetic images have less fine detail, so losing it matters less
- The fix is straightforward but touches multiple functions

---
**DELETE THIS FILE** after loading into new session.
