# JXL Encoder Status - 2026-01-03

## Current Branch: vardct-fix-clean

**Tests**: 355/361 passing (98.3%)  
**Base**: Clean rebuild from working commit b8245ff  
**Commits**: 2 incremental fixes applied cleanly

## What Works ✅

### Lossless Modular Encoding
- ✅ Single-group images (≤256x256) - jxl-rs, jxl-oxide, djxl
- ✅ Multi-group images (>256x256) - jxl-rs, jxl-oxide, djxl  
- ✅ Grayscale and RGB
- ✅ Solid colors, gradients, checkerboards, corpus images
- ✅ LZ77 compression for repeated data
- ✅ Gradient prediction (predictor 5)

### VarDCT Lossy Encoding (PARTIAL)
- ✅ File/frame header encoding
- ✅ Color transform (RGB → XYB)
- ✅ DCT transforms (8x8, 16x16, 32x32)
- ✅ Quantization
- ✅ DC coefficient encoding
- ✅ **Header parsing** works (jxl-oxide can read metadata)

## What's Broken ❌

### VarDCT AC Coefficient Loss
**Status**: ROOT CAUSE IDENTIFIED - 2026-01-03

**Symptoms**:
```
PASS_GROUP: 3 tokens to write
PASS_GROUP: token values: [(7, 0), (0, 0), (7, 0)]  ← All values are 0!
PASS_GROUP: alphabet_size = 1
PASS_GROUP: single symbol, returning
SECTION: Pass Group = 0 bytes  ← AC coefficients lost!
```

**Impact**:
- Files parse successfully (headers valid)
- Decoding pixels fails:
  - `test_encode_lossy_8x8`: `InvalidPaletteParams`
  - Other lossy tests: `InvalidEnum`, `ClusterHole` errors

**Root Cause** (CONFIRMED):
ALL AC coefficients are ZERO after DCT/quantization pipeline:
```
AC_DEBUG_STRAT: block 0, channel 0: first 10 coeffs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
AC_DEBUG_STRAT: nzeros = 0
```

For a red/blue checkerboard pattern (8x8), AC coefficients should be NON-ZERO to capture the high-frequency pattern. The issue is in the **DCT transform or quantization pipeline**, NOT in tokenization/encoding.

**Location**: `jxl_enc/src/vardct/transform.rs` - DCT transform and/or quantization
**Next**: Investigate why checkerboard DCT produces all-zero AC coefficients

## Recent Changes (This Branch)

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

## What We Avoided (Bugs in Broken Branch)

The `vardct-fix-lossless-borken` branch (44d8d58) had:
1. Forced debug return bypassing LZ77 encoding
2. Broken `write_simple_modular_stream` rewrite
3. Duplicate `extensions` field writes (restoration_filter bug)
4. Incorrectly removed `do_ycbcr` field

These broke 15 lossless tests. We surgically extracted only the good VarDCT changes.

## Next Steps for VarDCT AC Coefficient Fix

### Investigation Approach
1. ✅ Enable tracing (`--features trace-bitstream`)
2. ✅ Compare with libjxl reference (dump_bitstream example)
3. ✅ ~~Bug #1: Size header~~ - False alarm, was correct
4. ✅ **FOUND ROOT CAUSE**: All AC coefficients quantized to zero
5. ⏭️ Extract equivalent C++ code from libjxl for DCT/quantization
6. ⏭️ Compare our implementation with libjxl reference
7. ⏭️ Fix DCT/quantization to preserve AC coefficients
8. ⏭️ Verify with roundtrip test

### Test Strategy
- Use `test_roundtrip_lossy_rgb_d1` as primary test
- Enable tracing to see exact bit positions
- Compare against working libjxl output
- Un-ignore tests once fixed

## Documentation Status

- ✅ **CLAUDE.md** - Project instructions, up to date
- ✅ **STATUS.md** - This file, reflects current clean state
- ⚠️ **INVESTIGATION_NOTES.md** - Outdated (from 2026-01-02), needs update
- ⚠️ **MISTAKES.md** - Needs update with recent learnings
- ✅ **Test status report** - Created in /tmp/test_status_report.md

## Clean State Confirmed ✅

```bash
$ git status
On branch vardct-fix-clean
nothing to commit, working tree clean

$ cargo test -p jxl_enc 2>&1 | grep "test result:"
test result: ok. 355 passed; 0 failed; 2 ignored
```

**Ready to tackle VarDCT AC coefficient loss issue.**
