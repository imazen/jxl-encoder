# Context Handoff - VarDCT Encoding Investigation

**Created**: Jan 22, 2026
**Updated**: Jan 23, 2026
**Session Focus**: Multi-group VarDCT encoding investigation

## Status: ENCODER WORKING - JXL-OXIDE BUG FOUND

Single-group images (≤256x256) work perfectly with jxl-oxide.
Multi-group images (>256x256) encode correctly but jxl-oxide fails to decode them.

**CRITICAL FINDING**: djxl (libjxl decoder) decodes our multi-group output CORRECTLY with RMSE=9.82 (good quality).
The issue is a bug in **jxl-oxide's multi-group VarDCT decoder**, NOT our encoder.

## Test Results

### Single-Group Tests (PASSING with jxl-oxide)
- 64x64 d=1: SSIM2=73.29 [OK]
- 128x128 d=1: SSIM2=63.66 [OK]
- 256x256 d=1: SSIM2=77.20 [OK]

### Multi-Group Tests (ENCODER CORRECT, jxl-oxide BUG)
- 257x257 d=1: jxl-oxide RMSE=81.49 (FAIL), djxl RMSE=9.82 (PASS!)
- 300x300 d=1: jxl-oxide SSIM2=-64.03 (FAIL), djxl decodes correctly

**Proof**: Our 257x257 output at `/tmp/test_257_ours.jxl`:
```bash
# Decodes correctly with djxl
/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl /tmp/test_257_ours.jxl /tmp/test_257_decoded.png
# Result: RMSE=9.82 (good quality)

# Fails with jxl-oxide (RMSE=81.49 - decoder bug)
```

## Known Bug: raw_quant Hardcoded to 1 (NOT YET FIXED)

**Location**: `jxl_enc/src/vardct/transform.rs:60`

```rust
// CURRENT CODE - WRONG
let raw_quant = 1i32;  // Hardcoded!

// SHOULD BE
let raw_quant = quant_field.get(bx, by) as i32;  // Use per-block values
```

**Impact**:
- Real photos encode 4x larger with 4x worse quality than libjxl
- File size (1507x2048 photo, d=1.0): Our 760KB vs libjxl 184KB
- SSIM2: Our 23.5 vs libjxl 82.6

**Synthetic tests still pass** because they have limited fine detail.

## Multi-Group Encoding Architecture

For images >256x256 pixels:
- Groups: ceil(width/256) × ceil(height/256) regular groups
- LF Groups: ceil(width/2048) × ceil(height/2048) LF groups (DC + metadata)
- Sections: LfGlobal, LfGroup[0..n], HfGlobal, PassGroup[0..m]

Each section is byte-padded independently and listed in TOC.

Key code paths:
- `frame_encoder.rs`: `encode_vardct_multi_group_clustered_old()` for multi-group
- `encoder.rs`: `tokenize_ac_coefficients_for_group()` for per-group tokenization
- `encoder.rs`: `write_pass_group_clustered()` for AC data per group

## Next Steps

1. **Report jxl-oxide bug** - Multi-group VarDCT decoder fails
2. **Update tests** - Use djxl as fallback for multi-group verification
3. **Fix raw_quant bug** - Use per-block quant field values
4. **Test with frymire.png** - Real photo quality after raw_quant fix

## Commands for Verification

```bash
# Single-group tests (should pass)
cargo test -p jxl_enc test_vardct_quality_enforcement -- --ignored --nocapture

# Verify multi-group with djxl (should pass)
cargo run -p jxl_enc --example test_multi_group --release
/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl /tmp/test_257_ours.jxl /tmp/decoded.png

# Compare decoded quality
python3 -c "from PIL import Image; import numpy as np; ..."
```

---
**DELETE THIS FILE** after loading into new session.
