# Context Handoff: Pixel-Domain Loss Exact Parity with libjxl

**Date**: Feb 3, 2026
**Goal**: Make `--pixel-domain-loss` produce files ≤ coefficient-domain size (exact libjxl match)
**Current Gap**: +1.2% (down from +2.2%)

## Current State

File sizes on CLIC 2025 test image (1360x2048) at d=1.0:
```
DCT8-only:          740,996 bytes
Coefficient-domain: 733,576 bytes (baseline)
Pixel-domain:       742,412 bytes (+1.2% vs coeff)
```

## What Was Fixed This Session (Feb 3)

### 1. LLF Coefficient Inclusion (commit 9ce557b)
**File**: `ac_strategy.rs:511-514`

Removed incorrect skip that excluded LLF coefficients from entropy estimation:
```rust
// REMOVED: if i < num_blocks { continue; }
// Both libjxl and libjxl-tiny process ALL coefficients including LLF
```

### 2. Matched IDCT Implementations (commits 9ce557b, 1a0650d)
**File**: `dct.rs:684-762`

Added fast IDCTs that exactly reverse forward DCT:
- `idct1d_2`, `idct1d_4`, `idct1d_8`, `idct1d_16`
- Updated `idct_8x8`, `idct_16x16`, `idct_16x8`, `idct_8x16`
- Roundtrip error: ~0.00002 (was ~110 with old formula-based IDCT)

### 3. Symmetric5 Blur on mask1x1 (commit e5855b0)
**File**: `adaptive_quant.rs:540-639`

Implemented libjxl's `BlurMasking` 5x5 convolution:
```rust
// Weights from enc_adaptive_quantization.cc
const W_R: f32 = 0.364911248;   // adjacent
const W_D: f32 = 0.05;          // diagonal
const W_R2: f32 = 0.1688888021; // far h/v
const W_L: f32 = 0.221069183;   // knight-move
const W_D2: f32 = 0.306563504;  // far diagonal
```

## What Still Differs (Root Cause of +1.2% Gap)

The remaining gap is **loss calculation constant calibration**. Our matched IDCT produces correct roundtrip, but libjxl's constants were calibrated with their specific IDCT output magnitudes.

### Suspected Issues

1. **IDCT absolute scaling** - libjxl's `TransformToPixels` may output different absolute values than our matched IDCT. The loss constants compensate for that specific scale.

2. **Channel multipliers** - `CHANNEL_MUL = [8.2^8, 1.0, 1.03^8]` are huge. Small IDCT scale differences get amplified 8th-power.

3. **Possible constant drift** - The distance-scaled formulas might have subtle coefficient differences.

## Next Step: Instrumented Side-by-Side Comparison

Add debug output to BOTH codebases, compare values at each checkpoint:

### libjxl Instrumentation Points
```cpp
// enc_ac_strategy.cc:435 - After error coefficient computation
Store(Mul(m, diff), df, &mem[i]);
printf("C++ err_coeff[%zu] = %.6e\n", i, mem[i]);

// enc_ac_strategy.cc:456 - After TransformToPixels
TransformToPixels(acs.Strategy(), &mem[0], block, ...);
printf("C++ pixel_err[0..3] = %.6e %.6e %.6e %.6e\n", block[0], block[1], block[2], block[3]);

// enc_ac_strategy.cc:479 - Per-channel loss
lossc = Mul(Set(df8, kChannelMul[c]), lossc);
printf("C++ ch%d loss_sum = %.6e\n", c, GetLane(SumOfLanes(df8, lossc)));

// enc_ac_strategy.cc:505 - Final loss scalar
printf("C++ loss_scalar=%.6f quant_norm16=%.6f entropy=%.6f\n", loss_scalar, quant_norm16, entropy);
```

### Our Instrumentation Points
```rust
// ac_strategy.rs:508
error_coeffs[i] = weights[i] * diff;
eprintln!("RS err_coeff[{}] = {:.6e}", i, error_coeffs[i]);

// ac_strategy.rs:555 (after IDCT)
eprintln!("RS pixel_err[0..3] = {:.6e} {:.6e} {:.6e} {:.6e}",
    pixel_error[0], pixel_error[1], pixel_error[2], pixel_error[3]);

// ac_strategy.rs:588 (after channel loop)
eprintln!("RS ch{} loss_sum = {:.6e}", c, channel_loss);

// ac_strategy.rs:608
eprintln!("RS loss_scalar={:.6} quant_norm16={:.6} entropy={:.6}", loss_scalar, quant_norm16, entropy);
```

### Test Procedure

1. Use small test image (64x64) with a single forced transform
2. Run both encoders with debug output
3. Find first divergence point
4. Fix that specific mismatch
5. Repeat until values match

## Key Files

| Component | Our Code | libjxl Reference |
|-----------|----------|------------------|
| Entropy estimation | `ac_strategy.rs:380-620` | `enc_ac_strategy.cc:364-512` |
| IDCT application | `ac_strategy.rs:623-667` | `enc_ac_strategy.cc:456` |
| mask1x1 computation | `adaptive_quant.rs:506-639` | `enc_adaptive_quantization.cc:500-660` |
| IDCT implementations | `dct.rs:684-928` | `dec_transforms_inl.h` |

## Test Commands

```bash
# Quick file size comparison
TEST_IMG=~/work/codec-corpus/clic2025/validation/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png
cargo build --release -p jxl_enc_cli
./target/release/cjxl-rs "$TEST_IMG" /tmp/coeff.jxl -d 1.0
./target/release/cjxl-rs "$TEST_IMG" /tmp/pixel.jxl -d 1.0 --pixel-domain-loss
stat -c%s /tmp/coeff.jxl /tmp/pixel.jxl

# Run all tests
cargo test -p jxl_enc

# Strategy A/B comparison (slow, ~10min)
cargo test -p jxl_enc --test clic2025 test_strategy_ab_comparison -- --ignored --nocapture
```

## Build libjxl with Debug

```bash
cd ~/work/jxl-efforts/libjxl
# Add printf statements to enc_ac_strategy.cc
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
./build/tools/cjxl test.png test.jxl -d 1.0 -e 7
```

## Commits This Session

```
50ae708 docs: update handoff with pixel-domain loss fixes status
1a0650d feat: implement matched idct1d_16 and update 2D IDCTs
e5855b0 feat: add Symmetric5 blur to mask1x1 for pixel-domain loss
9ce557b fix: improve pixel-domain loss accuracy
```

## Previous Session Commits (Feb 2)

```
c117721 docs: update handoff with pixel-domain loss fixes status
11d9572 fix: improve pixel-domain loss parity with libjxl
09c953d docs: update pixel-domain loss status after partial fix
0ca040e fix: normalize entropy_mul and fix X channel penalty timing
```

---
**Delete this file after loading into new session.**
