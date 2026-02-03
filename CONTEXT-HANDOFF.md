# Context Handoff: Pixel-Domain Loss Parity with libjxl

**Date**: Feb 2, 2026
**Status**: Multiple fixes applied, ~2% file size gap remains vs coefficient-domain

## What Was Done This Session

Three fixes applied to `jxl_enc/src/tiny/ac_strategy.rs`:

1. **Fixed quant_norm16 computation** (lines 417-456)
   - libjxl uses different computation based on block count:
     - 1 block (DCT8): single quant value directly
     - 2 blocks (DCT16x8, DCT8x16): MAX of two quant values (NOT 16th norm!)
     - 4+ blocks (DCT16x16+): 16th norm
   - Reference: `lib/jxl/enc_ac_strategy.cc:383-410`

2. **Made K_COST2 threshold conditional** (lines 524-528)
   - `if q >= 1.5 { entropy_sum += K_COST2 }` is libjxl-tiny only
   - Full libjxl pixel-domain mode doesn't have this threshold
   - Now: `if !use_pixel_domain && q >= 1.5 { ... }`

3. **Made cost_of_1 term conditional** (lines 535-539)
   - `entropy_sum += nzeros_sum * cost_of_1` is libjxl-tiny only
   - Full libjxl pixel-domain mode doesn't have this per-nzero term
   - Now: `if !use_pixel_domain { entropy_sum += nzeros_sum * cost_of_1 }`

## Current Results

File sizes at d=1.0 on CLIC 2025 test image (1360x2048):
- **pixel-domain: 744,669 bytes** (+2.2% vs coefficient-domain)
- coefficient-domain: 728,745 bytes (baseline)
- DCT8-only: 740,996 bytes (+1.7% vs coefficient-domain)

Pixel-domain produces larger files than coefficient-domain. The gap narrowed
from ~16.5KB to ~16KB but the fundamental issue remains.

## Remaining Investigation Areas

### 1. IDCT Scaling Mismatch (HIGHEST PRIORITY)

Our IDCT functions may have different scaling than libjxl's TransformToPixels.
This would cause the loss term magnitude to be wrong.

**Verify by:**
```rust
// Add debug output in estimate_entropy_full
eprintln!("Block ({},{}) strategy={}: pixel_loss={:.2e}, entropy={:.2}",
    bx, by, strategy_name, total_pixel_loss, entropy);
```

Compare against libjxl's loss values for the same blocks.

### 2. Loss vs Entropy Weighting

The ratio `k_info_loss_mul * loss_scalar` vs `entropy` may be different.
At d=1.0, both k_info_loss_mul and cost_delta should equal their base values.

**Check:**
- Our entropy values match libjxl's for DCT8 blocks?
- Our loss_scalar values match libjxl's for the same input?

### 3. Mask1x1 Values

The mask1x1 computation might differ from libjxl. The mask values directly
scale the pixel error, so any difference would affect loss.

**Files to compare:**
- Our: `jxl_enc/src/tiny/adaptive_quant.rs` (`compute_mask1x1`)
- libjxl: `lib/jxl/enc_adaptive_quantization.cc`

## Test Commands

```bash
# Build release
cargo build --release -p jxl_enc_cli

# Test pixel-domain vs coefficient-domain
TEST_IMG=~/work/codec-corpus/clic2025/validation/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png
./target/release/cjxl-rs "$TEST_IMG" /tmp/pixel.jxl -d 1.0 --pixel-domain-loss
./target/release/cjxl-rs "$TEST_IMG" /tmp/coeff.jxl -d 1.0
./target/release/cjxl-rs "$TEST_IMG" /tmp/dct8.jxl -d 1.0 --dct8-only

# Compare sizes (goal: pixel ≤ coeff < dct8)
ls -la /tmp/*.jxl

# Run tests
cargo test -p jxl_enc
```

## Files Modified

- `jxl_enc/src/tiny/ac_strategy.rs` - Main entropy estimation logic
- `jxl_enc/src/tiny/dct.rs` - IDCT implementations (clippy fixes only)

## libjxl Reference

Key function: `EstimateEntropy` in `lib/jxl/enc_ac_strategy.cc:364-512`

Flow:
1. Transform pixels to DCT coefficients
2. For each coefficient: quantize, store error * dequant_matrix
3. IDCT error to pixel domain
4. Per-pixel masking with 8th power -> loss
5. entropy += cost_delta * sum(sqrt(|q|))
6. entropy += zeros_mul * (CeilLog2Nonzero(nbits + 17) + nbits)
7. X channel penalty (c==0 && num_blocks>=2): entropy *= w, loss *= w
8. loss_scalar = (sum(loss)/n)^(1/8) * n / quant_norm16
9. entropy *= entropy_mul
10. entropy += info_loss_mul * loss_scalar

## Recent Commits

- `11d9572` - fix: improve pixel-domain loss parity with libjxl
- `09c953d` - docs: update pixel-domain loss status after partial fix
- `0ca040e` - fix: normalize entropy_mul and fix X channel penalty timing
