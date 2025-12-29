# Encoding Parity with libjxl

This document tracks progress toward achieving encoding parity with the libjxl reference encoder.

## Current Status

**Date:** 2025-12-28

The encoder produces valid JXL files that decode successfully, but currently outputs all-black pixels because the minimal encoder uses a single-symbol histogram (always outputs 0).

## Goals

1. **Lossless encoding** - Encode actual pixel values correctly
2. **Round-trip verification** - Encoded files decode to exact input pixels
3. **File size parity** - Comparable file sizes to libjxl for same settings

## Implementation Plan

### Phase 1: Basic Pixel Encoding
- [ ] Implement pack_signed/unpack_signed for residuals
- [ ] Compute actual residuals (pixel - prediction with Zero predictor)
- [ ] Build histogram from actual residual values
- [ ] Encode residuals with proper entropy coding

### Phase 2: Verification
- [ ] Create round-trip tests (encode -> decode -> compare)
- [ ] Test with various image patterns (flat, gradient, random)
- [ ] Verify bit-exact decode with libjxl decoder

### Phase 3: Optimization (if needed)
- [ ] Compare file sizes with libjxl
- [ ] Implement better predictors if needed
- [ ] Optimize histogram building

---

## Progress Log

### 2025-12-28: Initial Setup

**Completed:**
- Fixed SizeHeader encoding to match JXL spec
- Fixed frame header byte alignment (removed spurious padding)
- Encoder produces valid 16-byte JXL files for 2x2 black images
- Files successfully decode with libjxl djxl

**Current limitation:**
- All pixels decode as black (0,0,0) regardless of input
- Minimal encoder uses single-symbol histogram outputting only 0

**Next steps:**
1. Understand pack_signed encoding for residuals
2. Implement proper residual computation
3. Build histogram covering actual residual range
