# Context Handoff: jxl-encoder-rs

**Date**: Feb 3, 2026
**Status**: Clean working tree, 28 commits ahead of origin/main

## What Was Done This Session

### 1. DCT4X8/DCT8X4 Quant Weights (Already Done)
Discovered the parametric weights were already properly implemented. Fixed a stale
comment that incorrectly said "DCT8 weights as placeholder."

### 2. DCT32x32 DC Extraction (FIXED)
- Changed `dc_from_dct_32x32()` to use matched `idct1d_4` with 16x scaling
- DC extraction error: >100% → <0.5%
- Diagnostic tests pass with small errors (~0.003 per value)

### 3. DCT32x32 Strategy Selection (PARTIAL)
- **Enabled at d >= 3.0 only**
- At d < 3.0, ~0.5% DC error causes visible artifacts (butteraugli 4.1 vs 0.66)
- At high distances, compression benefit outweighs quality impact

## Outstanding Work (Priority Order)

### 1. Pixel-Domain Loss Parity (+1.2% gap)
The `--pixel-domain-loss` flag produces 1.2% larger files than coefficient-domain mode.

**Next step**: Instrumented side-by-side comparison with libjxl
- Add debug output at matching checkpoints in both codebases
- Compare: error coefficients, IDCT pixel errors, per-channel loss, final entropy
- Find first divergence point

**Key files**:
| Component | Our Code | libjxl Reference |
|-----------|----------|------------------|
| Entropy estimation | `ac_strategy.rs:380-620` | `enc_ac_strategy.cc:364-512` |
| IDCT application | `ac_strategy.rs:623-667` | `enc_ac_strategy.cc:456` |
| mask1x1 computation | `adaptive_quant.rs:506-639` | `enc_adaptive_quantization.cc:500-660` |

### 2. Quality Gap at d>=2.0 (~3-5 SSIM2 vs cjxl e7)
Investigated and ruled out as fixable by constant tuning. libjxl-tiny constants are
tuned together for their simplified pipeline. Options:
- Accept the gap
- Implement missing features (splines, patches, full 27 strategies)

### 3. DCT32x32 at Low Distances
Currently disabled at d < 3.0 due to quality impact. To enable:
- Improve DC extraction accuracy (currently ~0.5% error)
- Or tune strategy selection to be more conservative

### 4. Minor TODOs
- `encoder.rs`: verify_histogram_serialization needs fix for all histogram method types

## Test Commands

```bash
# Quick verification
cargo test -p jxl_enc

# RD regression (slow)
just rd-regression

# Quality gate test
cargo test -p jxl_enc --test clic2025 test_butteraugli_quality_gate -- --nocapture

# DCT32x32 diagnostics
cargo test -p jxl_enc --test llf_invariants diag_dct32x32 -- --ignored --nocapture

# File size comparison
TEST_IMG=~/work/codec-corpus/clic2025/validation/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png
cargo build --release -p jxl_enc_cli
./target/release/cjxl-rs "$TEST_IMG" /tmp/coeff.jxl -d 1.0
./target/release/cjxl-rs "$TEST_IMG" /tmp/pixel.jxl -d 1.0 --pixel-domain-loss
stat -c%s /tmp/coeff.jxl /tmp/pixel.jxl
```

## Recent Commits

```
7dfb1a2 docs: update CLAUDE.md with DCT32x32 fix status
3a848c1 feat: fix DCT32x32 DC extraction and enable at high distances
027c8bd docs: fix stale comment about DCT4X8/DCT8X4 quant weights
8dcf399 docs: add Outstanding Work section to CLAUDE.md
```

## Key Insight: DCT32x32 Quality Tradeoff

DCT32x32 covers 32x32 pixels (16 blocks). Even small DC errors (~0.5%) affect a large
area and are visible on smooth content. The fix works mathematically but the quality
impact at low distances is unacceptable. This is why it's only enabled at d >= 3.0.

To fully enable DCT32x32:
1. Need DC extraction with <0.1% error, OR
2. Accept that DCT32x32 is only beneficial at high compression ratios

---
**Delete this file after loading into new session.**
