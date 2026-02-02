# Context Handoff — Feb 2, 2026 (Updated)

## Mission: Achieve libjxl Parity and Unify Encoders

The goal is to close the quality gap with full libjxl (cjxl) and merge the "tiny" encoder
into the main encoder path. The tiny encoder started as a port of libjxl-tiny but has
evolved beyond it. The original VarDCT encoder (`jxl_enc/src/vardct/`) failed completely
and should be abandoned.

## Session Update (Feb 2, 2026 - Latest)

### What Was Done This Session

1. **Comprehensive comparison** of our encoder vs full libjxl (see PROVEN_INVARIANTS.md)
2. **Implemented DCT4x8/DCT8x4 transforms** (Layer 1 complete):
   - `dct_4x8`, `dct_8x4` - base 32-coefficient transforms
   - `dct_4x8_full`, `dct_8x4_full` - full 8x8 pixel coverage
   - `dc_from_dct_4x8_full`, `dc_from_dct_8x4_full` - DC extraction
   - 10 new tests, all passing
3. **Added PROVEN_INVARIANTS.md** - layer-by-layer proof tracking system
4. **Documented libjxl reference constants** for DCT4x8 band parameters

### Commits This Session
```
d3089a3 docs: add invariant preservation rule and PROVEN_INVARIANTS.md
67c8678 feat: add DCT4x8/DCT8x4 forward transforms (Layer 1 proven)
de67393 docs: update PROVEN_INVARIANTS.md with libjxl constants
```

### Key Divergences Identified

| Area | Our Value | libjxl Value | Impact |
|------|-----------|--------------|--------|
| kAcQuant | 0.8294 | 0.765 | ~8% quantization difference |
| AC strategies | 5 | 27 | Missing small/large transforms |
| Multi-pass quant | No | Yes (FindBestQuantizer) | 5-7% at high compression |
| DC chroma correlation | No | Yes | 1-2% size |

## DCT4x8/DCT8x4 Status

**Layer 1 COMPLETE** - Transforms implemented and tested.

**Remaining work** (see PROVEN_INVARIANTS.md for details):
1. Quant weights for DCT4x8/DCT8x4 (can use DCT8 as placeholder)
2. Strategy codes (RAW_STRATEGY_DCT4X8=5, RAW_STRATEGY_DCT8X4=6, bitstream=12,13)
3. Encoder integration in `jxl_enc/src/tiny/encoder.rs`
4. Block context map entries for codes 12, 13
5. Layer 3 testing (djxl, jxl-rs, jxl-oxide)

**libjxl reference constants** recorded in PROVEN_INVARIANTS.md.

## Current RD Position vs cjxl e7

Single CLIC 2025 image (1024x1024), tested Feb 2, 2026:

| Distance | Our Size | Our SSIM2 | cjxl Size | cjxl SSIM2 | Gap |
|----------|----------|-----------|-----------|------------|-----|
| d=1.0    | ~514KB   | ~80.9     | ~520KB    | ~80.7      | **+0.2** |
| d=2.0    | 143KB    | 76.1      | 162KB     | 81.2       | **-5.1** |
| d=4.0    | 88KB     | 63.4      | 100KB     | 70.3       | **-6.9** |

**We match/beat cjxl at d≤1.0 but lose significantly at higher distances.**

## Files to Read First

1. **PROVEN_INVARIANTS.md** — Layer-by-layer tracking for DCT4x8 work
2. **CLAUDE.md** — Full project docs, known bugs, resolved bugs
3. `jxl_enc/src/tiny/dct.rs` — New DCT4x8/DCT8x4 transforms
4. `jxl_enc/src/tiny/encoder.rs` — Main encode function
5. `jxl_enc/src/tiny/ac_strategy.rs` — Strategy selection (needs DCT4x8 integration)

## Encoder Architecture

### Production Encoder: `jxl_enc/src/tiny/`

This is the working encoder. Key files:

- `encoder.rs` — Main encode pipeline (~2500 lines)
- `dct.rs` — DCT transforms including NEW dct_4x8_full/dct_8x4_full
- `adaptive_quant.rs` — Per-block quantization field computation
- `ac_strategy.rs` — DCT transform selection (8x8, 16x8, 8x16, 16x16)
- `quant.rs` — Quantization weights and coefficient quantization
- `entropy_code.rs` — Huffman and ANS entropy coding

### Dead Code: `jxl_enc/src/vardct/`

**DO NOT USE.** This was an earlier VarDCT implementation that never worked correctly.

## Test Commands

```bash
# Run DCT tests
cargo test -p jxl_enc tiny::dct::tests --release

# Run all tests
cargo test -p jxl_enc --release

# Run RD regression
just rd-regression

# Encode single image
./target/release/cjxl-rs -d 2.0 input.png output.jxl
```

## Suggested Next Steps

### Option A: Continue DCT4x8 Integration
1. Add quant weights (use DCT8 as placeholder initially)
2. Add strategy codes and encoder calls
3. Test with external decoders
4. Tune strategy selection thresholds

### Option B: Quick Wins
1. **DC chroma correlation** — Small addition to CfL pipeline (~1-2% size)
2. **kAcQuant change** — One constant, needs RD recalibration

### Option C: Deep Investigation
1. Compare adaptive quant intermediate values with libjxl
2. Check coefficient thresholding differences
3. Profile high-distance quality gap root cause

## Working Tree State

Clean. All tests pass. Latest commit: `de67393`.

---

**Delete this file after loading into new session.**
