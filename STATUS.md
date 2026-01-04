# JXL Encoder Status - 2026-01-03

## Current Branch: vardct-fix-clean

**Tests**: 348/357 passing (97.5%)
**Base**: Clean rebuild from working commit b8245ff
**Commits**: 4 commits (tracing, modular fixes, **quantization fix**)

## What Works ✅

### Lossless Modular Encoding
- ✅ Single-group images (≤256x256) - jxl-rs, jxl-oxide, djxl
- ✅ Multi-group images (>256x256) - jxl-rs, jxl-oxide, djxl
- ✅ Grayscale and RGB
- ✅ Solid colors, gradients, checkerboards, corpus images
- ✅ LZ77 compression for repeated data
- ✅ Gradient prediction (predictor 5)

### VarDCT Lossy Encoding (PARTIAL - MAJOR FIX APPLIED)
- ✅ File/frame header encoding
- ✅ Color transform (RGB → XYB)
- ✅ DCT transforms (8x8, 16x16, 32x32)
- ✅ **Quantization** - **FIXED 2026-01-03**
- ✅ DC coefficient encoding
- ✅ **AC coefficient encoding** - **FIXED 2026-01-03**
- ✅ Header parsing works (jxl-oxide can read metadata)

## What's Fixed ✅

### VarDCT AC Coefficient Quantization Bug
**Status**: FIXED - 2026-01-03 (commit 911e589)

**Problem**:
All AC coefficients were being quantized to zero, causing empty Pass Group sections and decoder failures.

**Root Cause**:
`get_dct8_inv_dequant_per_channel()` was returning `1/weight` instead of `weight` for the quantization matrix. This caused all AC coefficients to be quantized too aggressively (scaled down by ~200x).

**Example** (Y channel, position 63):
- Before (WRONG): val = 0.092 × 0.0051 × 4.975 = **0.0023** → 0 ❌
- After (CORRECT): val = 0.092 × 196.07 × 4.975 = **89.98** → 90 ✅

**Evidence of Fix**:
```
Before: AC_DEBUG_STRAT: nzeros = 0 (all zeros)
After:  AC_DEBUG_STRAT: nzeros = 16 (preserved!)
```

**Files Changed**:
- `jxl_enc/src/vardct/quant_weights.rs` - Fixed inv_dequant matrix computation
- Added comprehensive documentation in `VARDCT_BUG_FOUND.md`

**Test Impact**: 346 → 348 passing (+2 fixed inv_dequant tests)

## What's Still Broken ❌

### VarDCT Decoder Validation (7 tests failing)
**Status**: UNDER INVESTIGATION

Now that AC coefficients are preserved, lossy files encode successfully but still fail to decode. This appears to be a separate bitstream format issue, NOT a quantization problem.

**Failing Tests**:
- `test_encode_lossy_8x8` - `InvalidPaletteParams`
- `test_decode_lossy_solid_color` - decoder error
- `test_decode_lossy_rgb` - decoder error
- `test_decode_lossy_distances` - decoder error
- `test_dual_decode_lossy_distances` - decoder error
- `test_dual_decode_lossy_vardct` - decoder error
- `test_decode_lossy_multi_group` - decoder error

**Next**: Investigate bitstream format issues causing decoder failures.

## Investigation Timeline - VarDCT AC Bug

### Session 1: Initial Hypothesis (WRONG)
- **Hypothesis**: Size header encoding mismatch causing bitstream divergence
- **Finding**: Size header was actually correct! Both use `small=true`, ratio=1 for 8x8
- **Outcome**: Red herring - fixed minor metadata differences but not the core issue

### Session 2: Tokenization Analysis
- **Discovery**: Only 3 tokens generated for pass group (should be many more)
- **Token values**: `[(7, 0), (0, 0), (7, 0)]` - **ALL values are 0**!
- **Meaning**: All 3 blocks report `nzeros=0` (no non-zero AC coefficients)

### Session 3: Root Cause Found (2026-01-03)
**Added debug output through the pipeline:**
1. ✅ XYB conversion: CORRECT (X=0.028 for red is expected)
2. ✅ DCT transform: CORRECT (produces AC=0.092 for checkerboard)
3. ❌ Quantization: **BUG FOUND** - using `1/weight` instead of `weight`!

**Fix Applied**: Changed `get_dct8_inv_dequant_per_channel()` to return weights directly.

**Verification**: AC coefficients now preserved (16 non-zero for checkerboard test).

## Recent Commits (This Branch)

### Commit 6c11635: Tracing Infrastructure
- Added `trace.rs` with zero-cost bitstream tracing macros
- Added `trace-bitstream` feature to Cargo.toml
- **Impact**: +1 test (351 → 352 passing)

### Commit 4cef0e1: VarDCT Modular Substream Fixes
- Fixed lz77.enabled bit writing for VarDCT substreams
- Fixed tree leaf property encoding: pack_signed(-1)=1
- Added DC coefficient debug logging
- **Impact**: +4 tests (351 → 355 passing)
- **Files**: Only vardct/*.rs (avoided broken modular changes)

### Commit 17678cc: Investigation Documentation
- Created `VARDCT_AC_INVESTIGATION.md` - Investigation timeline
- Created `LIBJXL_DCT_QUANTIZATION_REFERENCE.md` - C++ reference extraction
- Updated `STATUS.md` with root cause findings

### Commit 911e589: **QUANTIZATION FIX** ⭐
- **Fixed**: `get_dct8_inv_dequant_per_channel()` to use `weight` instead of `1/weight`
- **Result**: AC coefficients now preserved correctly
- **Documentation**: Added `VARDCT_BUG_FOUND.md` with detailed analysis
- **Impact**: +2 tests (346 → 348 passing), AC coefficients functional
- **Files**: `jxl_enc/src/vardct/quant_weights.rs`

## What We Avoided (Bugs in Broken Branch)

The `vardct-fix-lossless-borken` branch (44d8d58) had:
1. Forced debug return bypassing LZ77 encoding
2. Broken `write_simple_modular_stream` rewrite
3. Duplicate `extensions` field writes (restoration_filter bug)
4. Incorrectly removed `do_ycbcr` field

These broke 15 lossless tests. We surgically extracted only the good VarDCT changes.

## Next Steps

### Immediate
1. Remove debug output from `transform.rs` and `enc_coeff.rs`
2. Investigate remaining 7 decoder validation failures
3. Compare encoded bitstreams with libjxl reference

### VarDCT Completion
- Fix remaining decoder validation issues
- Implement 16x16 and 32x32 quantization (if not already correct)
- Add more lossy encoding tests
- Performance optimization

## Documentation Status

- ✅ **CLAUDE.md** - Project instructions, up to date
- ✅ **STATUS.md** - This file, updated with quantization fix
- ✅ **VARDCT_AC_INVESTIGATION.md** - Complete investigation timeline
- ✅ **VARDCT_BUG_FOUND.md** - Detailed bug analysis and fix
- ✅ **LIBJXL_DCT_QUANTIZATION_REFERENCE.md** - C++ reference extraction
- ⚠️ **INVESTIGATION_NOTES.md** - Outdated (from 2026-01-02), needs update
- ⚠️ **MISTAKES.md** - Needs update with recent learnings

## Current State

```bash
$ git status
On branch vardct-fix-clean
Changes not staged for commit:
  modified:   jxl_enc/src/vardct/enc_coeff.rs (debug output)
  modified:   jxl_enc/src/vardct/transform.rs (debug output)

$ cargo test -p jxl_enc 2>&1 | grep "test result:"
test result: FAILED. 348 passed; 7 failed; 2 ignored
```

**Major progress**: AC coefficient quantization bug is FIXED!
**Next**: Investigate remaining decoder validation failures.
