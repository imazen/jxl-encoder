# Context Handoff: Architectural Shifts for Quality Parity

**Date**: 2026-02-02
**Status**: Ready for architectural work

## Current State

The jxl-encoder-rs encoder produces files that decode correctly with djxl, jxl-oxide, and jxl-rs. Quality at d≤1.0 is competitive with cjxl, but at d≥2.0 there's a ~3 SSIM2 gap.

### Quality Position (CLIC 2025 test image, 1360x2048)

| Distance | Our Size | Our SSIM2 | cjxl e7 Size | cjxl e7 SSIM2 | Gap |
|----------|----------|-----------|--------------|---------------|-----|
| d=1.0    | 729KB    | 85.0      | 815KB        | 86.8          | -1.8 |
| d=2.0    | 468KB    | 76.7      | 517KB        | 79.0          | -2.3 |

## What We Tried (Feb 2, 2026)

**Hypothesis**: Porting full libjxl adaptive quantization constants would improve quality at d≥2.0.

**Result**: Made quality WORSE (-0.6 SSIM2 at d=1.0, -1.3 SSIM2 at d=2.0).

**Why**: Full libjxl constants are tuned to work with features we don't have:
1. Error diffusion in main quantization loop
2. Splines and patches
3. 27 AC strategies with sophisticated selection
4. Different coefficient ordering logic

## Working Features

- [x] DCT8, DCT4x4, DCT4x8, DCT8x4, DCT16x16, DCT16x8, DCT8x16
- [x] ANS entropy coding (12% smaller than Huffman)
- [x] Custom coefficient ordering
- [x] Adaptive quantization (libjxl-tiny pipeline)
- [x] Chroma-from-luma
- [x] Error diffusion (opt-in)
- [x] Gaborish inverse
- [x] Noise synthesis

## Known Issues

- [ ] DCT32x32 DISABLED (DC extraction bug with 4-point IDCT)
- [ ] Quality gap at d≥2.0 vs cjxl

## Architectural Changes Needed

### Priority 1: DCT32x32 DC Extraction Fix

**Problem**: `dc_from_dct_32x32()` uses 4-point IDCT which can't represent step functions at position 2. Multi-block DCT32x32 produces catastrophic quality.

**Options**:
1. Use full 8x8 IDCT for DC extraction (like larger transforms)
2. Different mathematical formulation avoiding Gibbs phenomenon
3. Iterative refinement approach

**Location**: `jxl_enc/src/tiny/dct.rs`

### Priority 2: Coefficient Thresholding in QuantizeBlockAC

**Problem**: Current thresholding may zero too many coefficients at high distances.

**Investigation needed**: Compare our quantized coefficients with cjxl's output using the debugging facilities in libjxl.

**Location**: `jxl_enc/src/tiny/encoder.rs` (quantize_ac_block)

### Priority 3: AC Strategy Selection Cost Model

**Problem**: Strategy selection may not be optimal at high distances.

**Investigation needed**: Compare which blocks get which strategies vs cjxl.

**Location**: `jxl_enc/src/tiny/ac_strategy.rs`

### Priority 4 (Major): Content-Adaptive Features

**Splines**: Parametric encoding of smooth curves. Would help with power lines, horizons, edges.

**Patches**: Dictionary-based encoding for repeated patterns. Major help for screenshots/UI.

These are significant architectural additions that would require:
1. Detection algorithms
2. Bitstream encoding
3. Integration with existing pipeline

## Files to Read First

1. `INVESTIGATION.md` - Full debugging history and findings
2. `PROVEN_INVARIANTS.md` - Test coverage for features
3. `jxl_enc/src/tiny/encoder.rs` - Main encoding pipeline
4. `jxl_enc/src/tiny/ac_strategy.rs` - AC strategy selection
5. `jxl_enc/src/tiny/dct.rs` - Transform and DC extraction

## Test Commands

```bash
# Run all tests
cargo test -p jxl_enc

# Run RD regression (quality check)
just rd-regression

# Compare with cjxl
IMG=~/work/codec-corpus/clic2025/final-test/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png
cargo run --release -p jxl_enc_cli -- "$IMG" /tmp/ours.jxl -d 2.0
~/work/jxl-efforts/libjxl/build/tools/cjxl "$IMG" /tmp/cjxl.jxl -d 2.0 -e 7
~/work/jxl-efforts/libjxl/build/tools/ssimulacra2 /tmp/ours.jxl "$IMG"
~/work/jxl-efforts/libjxl/build/tools/ssimulacra2 /tmp/cjxl.jxl "$IMG"
```

## Reference Code

- Full libjxl: `~/work/jxl-efforts/libjxl/lib/jxl/`
- libjxl-tiny: `~/work/libjxl-tiny/encoder/`
- jxl-rs decoder: `~/work/jxl-rs/`
- jxl-oxide decoder: `~/work/jxl-efforts/jxl-oxide/`
