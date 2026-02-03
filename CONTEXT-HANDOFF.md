# Context Handoff: Pixel-Domain Loss Parity with libjxl

**Date**: Feb 2, 2026
**Status**: Partial fix applied, further calibration needed to match libjxl

## What Was Done This Session

Two fixes were applied to `jxl_enc/src/tiny/ac_strategy.rs`:

1. **Normalized entropy_mul by DCT8's base value (0.8)** (lines 188-215)
   - libjxl divides all entropy_mul values by DCT8's base (0.8)
   - DCT8: 0.8/0.8 = 1.0, DCT16X8: 1.21/0.8 = 1.5125, DCT16X16: 1.34/0.8 = 1.675
   - Reference: `~/work/jxl-efforts/libjxl/lib/jxl/enc_ac_strategy.cc:584`

2. **Fixed X channel penalty timing** (lines 574-582)
   - Penalty now applies to TOTAL accumulated loss after adding X channel
   - Reference: `~/work/jxl-efforts/libjxl/lib/jxl/enc_ac_strategy.cc:500-501`

## Current Problem

Pixel-domain mode produces **larger** files than coefficient-domain:
- DCT8-only: 740,996 bytes
- Coefficient-domain: 728,745 bytes (-1.7%)
- Pixel-domain: 745,295 bytes (+0.6%)

This suggests the loss term is overweighted, causing strategies to be selected
that increase file size without quality benefit.

## libjxl Reference Code

The key function is `EstimateEntropy` in `enc_ac_strategy.cc:364-512`. Critical flow:

```cpp
// Line 418-440: Compute quantization error and store for IDCT
for (size_t i = 0; i < num_blocks * kDCTBlockSize; i += Lanes(df)) {
    const auto val = Mul(Sub(in, in_y), Mul(im, quant));  // quantize
    const auto rval = Round(val);
    const auto diff = Sub(val, rval);
    const auto m = Load(df, matrix + i);  // dequant matrix
    Store(Mul(m, diff), df, &mem[i]);     // error * matrix for IDCT
    entropy_v = Add(Sqrt(Abs(rval)), entropy_v);  // sqrt(|q|)
}

// Line 442-480: IDCT of error, then per-pixel masking with 8th power
TransformToPixels(acs.Strategy(), &mem[0], block, ...);
for (pixels) {
    auto in = Load(df8, block + pixel_offset);
    auto masku = Add(Load(df8, mask1x1_ptr), masku_off);
    in = Mul(masku, in);   // mask * error
    in = Mul(in, in);      // ^2
    in = Mul(in, in);      // ^4
    in = Mul(in, in);      // ^8
    lossc = Add(lossc, in);
}
lossc = Mul(Set(df8, kChannelMul[c]), lossc);
loss = Add(loss, lossc);

// Line 486-489: Entropy from sqrt(|q|) * cost_delta
entropy += config.cost_delta * GetLane(SumOfLanes(df, entropy_v));

// Line 496-502: X channel penalty on BOTH entropy AND loss
if (c == 0 && num_blocks >= 2) {
    float w = 1.0 + std::min(3.0, num_blocks / 8.0);
    entropy *= w;
    loss = Mul(loss, Set(df8, w));
}

// Line 503-509: Final computation
float loss_scalar = pow(GetLane(SumOfLanes(df8, loss)) / (num_blocks * kDCTBlockSize),
                        1.0f / 8.0f) * (num_blocks * kDCTBlockSize) / quant_norm16;
entropy *= entropy_mul;  // NORMALIZED entropy_mul
entropy += config.info_loss_multiplier * loss_scalar;
```

## HIGH PRIORITY BUG: quant_norm16 for 2-Block Transforms

**This is likely the main remaining calibration issue.**

libjxl uses different quant_norm16 computation based on block count (lines 383-410):

```cpp
if (num_blocks == 1) {
    quant_norm16 = config.Quant(x/8, y/8);
} else if (num_blocks == 2) {
    quant_norm16 = std::max(q1, q2);  // ← MAX, not 16th norm!
} else {
    // 4+ blocks: use 16th norm
    for (blocks) {
        qval *= qval; qval *= qval; qval *= qval;  // qval^8
        quant_norm16 += qval * qval;  // qval^16
    }
    quant_norm16 /= num_blocks;
    quant_norm16 = FastPowf(quant_norm16, 1.0f / 16.0f);
}
```

**Our code** (lines 434-454) always uses the 16th norm, even for 2-block transforms!

### Fix Required

In `estimate_entropy_full` around line 430, change the quant_norm16 computation:

```rust
// Current (WRONG for 2-block):
for iy in 0..cy {
    for ix in 0..cx {
        let qval = quant_field[idx];
        let q16 = qval.powi(16);
        quant_norm16 += q16;
    }
}
quant_norm16 /= num_blocks as f32;
quant_norm16 = quant_norm16.powf(1.0 / 16.0);
```

Should be:

```rust
// Matches libjxl behavior
if use_pixel_domain {
    if num_blocks == 1 {
        quant_norm16 = quant_field[by * xsize_blocks + bx];
    } else if num_blocks == 2 {
        // Use MAX for 2-block transforms (DCT16x8, DCT8x16)
        let q1 = quant_field[by * xsize_blocks + bx];
        let q2 = if cy == 2 {
            quant_field[(by + 1) * xsize_blocks + bx]
        } else {
            quant_field[by * xsize_blocks + bx + 1]
        };
        quant_norm16 = q1.max(q2);
    } else {
        // 4+ blocks: use 16th norm
        for iy in 0..cy {
            for ix in 0..cx {
                let idx = (by + iy) * xsize_blocks + bx + ix;
                let qval = quant_field[idx];
                let q2 = qval * qval;
                let q4 = q2 * q2;
                let q8 = q4 * q4;
                let q16 = q8 * q8;
                quant_norm16 += q16;
            }
        }
        quant_norm16 /= num_blocks as f32;
        quant_norm16 = quant_norm16.powf(1.0 / 16.0);
    }
}
```

## Other Areas to Verify (Lower Priority)

### 1. IDCT Output Layout
Our IDCT functions may output in different layout than libjxl's `TransformToPixels`.
libjxl outputs in:
```
block[(iy * 8 + dy) * (covered_blocks_x * 8) + (ix * 8 + dx)]
```

**Files**:
- Our code: `jxl_enc/src/tiny/dct.rs` (idct_8x8, idct_16x8, idct_8x16, idct_16x16)
- libjxl: `lib/jxl/enc_transforms.cc` (TransformToPixels)

### 2. Error Coefficient Computation
Verify our `weights[i]` matches libjxl's dequant matrix values.

### 3. Add Debug Logging
```rust
#[cfg(feature = "debug-strategy")]
eprintln!("Strategy {} ({},{}): entropy={:.2}, loss={:.2}, quant_norm16={:.4}",
    strategy_name, bx, by, entropy, loss_scalar, quant_norm16);
```

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

# Run unit tests
cargo test -p jxl_enc tiny::ac_strategy -- --nocapture

# Run roundtrip tests
cargo test -p jxl_enc test_roundtrip -- --nocapture
```

## Files to Modify

1. `jxl_enc/src/tiny/ac_strategy.rs` - Fix quant_norm16 computation (HIGH PRIORITY)
2. `jxl_enc/src/tiny/dct.rs` - Verify IDCT layout if needed (LOW PRIORITY)

## Reference Files in libjxl

- `lib/jxl/enc_ac_strategy.cc` - EstimateEntropy function (lines 364-512)
- `lib/jxl/enc_transforms.cc` - TransformToPixels (IDCT)
- `lib/jxl/quant_weights.h` - Quantization weight matrices

## Recent Commits

- `09c953d` - docs: update pixel-domain loss status after partial fix
- `0ca040e` - fix: normalize entropy_mul and fix X channel penalty timing in pixel-domain loss
