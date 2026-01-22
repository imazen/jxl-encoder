# Context Handoff - VarDCT Quality Investigation

**Date**: Jan 22, 2026
**Session**: Investigating why VarDCT produces garbage images for >32x32

## CRITICAL BUG DISCOVERED

VarDCT encoder produces **valid bitstreams** that decode without errors, but the **actual image quality is catastrophically broken** for anything larger than 32x32 pixels.

### Quality Metrics

| Size | SSIM2 Score | Expected | Status |
|------|-------------|----------|--------|
| 8x8 | 84 | >50 | ✓ OK |
| 16x16 | 92 | >50 | ✓ OK |
| 32x32 | 7-18 | >50 | ⚠ Degrading |
| 64x64 | -562 | >50 | ❌ BROKEN |
| 128x128 | -1011 | >50 | ❌ BROKEN |
| 256x256 | -928 | >50 | ❌ BROKEN |
| 512x512 | -1000+ | >50 | ❌ BROKEN |

Butteraugli scores on 512x512 CLIC images: **110-180** (should be < 2.0)

### Visual Symptom (User Described)

> "The 8x8 grid has like a color that is correct, but it's either a flat or diagonal gradient to black, with only one color per block except it gradients to black."

This means:
- **DC coefficient** (block average color) is CORRECT
- **AC coefficients** (texture/detail within block) are BROKEN - either zeroed or corrupted

### Root Cause Hypothesis

The symptom suggests:
1. DC values are being encoded/decoded correctly
2. AC coefficients are either:
   - Not being written at all
   - Being quantized to near-zero
   - Having wrong coefficient ordering (zigzag/natural order mismatch)
   - Transpose issue affecting AC but not DC

Key code locations to investigate:
- `jxl_enc/src/vardct/encoder.rs:400-460` - AC coefficient tokenization
- `jxl_enc/src/vardct/encoder.rs:456-458` - Coefficient transpose logic
- `jxl_enc/src/vardct/transform.rs` - DCT and quantization

## HOW TESTS WERE CHEATED

Tests only checked `decode_ok` (no crash), never quality scores:

```rust
// OLD (cheating):
assert!(decode_failures.is_empty(), "...failures...");
// SSIM2 computed but NEVER CHECKED!

// NEW (fixed):
assert!(score > 50.0, "SSIM2 {} below threshold", score);
```

### Commit History of False Claims

| Commit | Claim | Reality |
|--------|-------|---------|
| bf6f0a2 | Fixed: tests weren't rendering | Exposed 7 real failures |
| 4e4f0ef | Corrected false claims | Honest: VarDCT broken |
| c884c12 | 73% decode success | Accurate |
| d385e41 | **100% decode success** | TRUE but images are GARBAGE |

## FILES CHANGED THIS SESSION

### Quality Enforcement Tests Added

1. **`jxl_enc/src/vardct_quality_tests.rs`**
   - `test_vardct_quality_thresholds` - Now asserts SSIM2 > 50
   - `test_vardct_quality_enforcement` - Tests 64-300px (IGNORED, will fail)

2. **`jxl_enc/src/encoder_tests.rs`**
   - `test_save_broken_image` - Saves original + decoded for visual comparison
   - `test_corpus_quality_sample` - Butteraugli on CLIC images

### Documentation Updated

- **`CLAUDE.md`** - Added "Known Bugs" section documenting the quality bug

## TO REPRODUCE VISUAL COMPARISON

```bash
# Generate broken image
cargo test --package jxl_enc test_save_broken_image -- --ignored --nocapture

# View side by side
display /home/lilith/work/codec-corpus/clic2025/validation/097cb426910ba8ce2525dd8bb7fb1777.png &
display /tmp/broken_decoded.png &
```

## TO RUN QUALITY ENFORCEMENT TEST

```bash
# This SHOULD FAIL until the bug is fixed
cargo test --package jxl_enc test_vardct_quality_enforcement -- --ignored --nocapture
```

## NEXT STEPS TO DEBUG

1. **Trace AC coefficients** - Add debug output showing:
   - Raw DCT output values
   - Quantized values
   - What gets written to bitstream
   - What decoder reads back

2. **Compare with libjxl** - Encode same image with cjxl, compare coefficient values

3. **Check coefficient ordering** - The transpose logic at line 456-458 may be wrong:
   ```rust
   let orig_idx = ZIGZAG_ORDER_8X8[k + 1];
   let transposed_idx = (orig_idx % 8) * 8 + (orig_idx / 8);
   let coeff = block_ac[transposed_idx - 1];
   ```

4. **Check quantization** - Values may be quantized too aggressively for larger images

## GIT STATUS

```
Branch: main (ahead of origin by 28 commits)
Latest commits:
- 0605145 docs: document CRITICAL VarDCT quality bug in CLAUDE.md
- 31b52f7 test: add quality enforcement to prevent false positive tests
- 3647f04 test: add corpus quality sampling with butteraugli
```

## KEY FILES

- `jxl_enc/src/vardct/encoder.rs` - Main VarDCT encoding logic
- `jxl_enc/src/vardct/transform.rs` - DCT transforms and quantization
- `jxl_enc/src/vardct/tokenize.rs` - Coefficient ordering (zigzag, natural order)
- `jxl_enc/src/vardct_quality_tests.rs` - Quality tests with SSIM2

## DELETE THIS FILE

After reading this into a new session, delete CONTEXT-HANDOFF.md.
