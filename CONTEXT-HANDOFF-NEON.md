# NEON Implementation Handoff

## Goal

Add AArch64 NEON SIMD implementations for the 13 jxl_simd modules that currently only have x86_64 AVX2 dispatch. The encoder already produces byte-identical output on AArch64 (scalar fallback), but VarDCT lossy encoding is ~26x slower under QEMU emulation vs native x86_64 with AVX2.

## Current State

### Cross-Platform Consistency: VERIFIED
- 36 hash-locked tests pass on x86_64, WASM (wasmtime), and AArch64 (QEMU)
- FNV-1a hash (deterministic, not DefaultHasher) in `jxl_encoder/tests/hash_lock_features.rs`
- Output bytes are identical across all three targets
- **CRITICAL: NEON implementations must NOT change output bytes** — hash tests will catch regressions

### SIMD Coverage

**Already have NEON (2/15 modules):**
- `jxl_simd/src/gab.rs` — Gaborish smoothing
- `jxl_simd/src/gaborish5x5.rs` — 5x5 convolution

**Need NEON (13 modules), ranked by VarDCT hotpath impact:**

1. **dct8.rs** — Forward 8x8 DCT (hottest path, every block)
2. **quantize.rs** — AC coefficient quantization
3. **xyb.rs** — Linear RGB → XYB color space conversion
4. **transpose.rs** — 8x8 matrix transpose
5. **dequant.rs** — Dequantization (for reconstruction loop)
6. **idct16.rs** — Inverse DCT 16x16/8x16/16x8 (reconstruction)
7. **dct16.rs** — Forward DCT 16x16/8x16/16x8
8. **pixel_loss.rs** — Pixel-domain loss computation
9. **mask1x1.rs** — Per-pixel masking field
10. **epf.rs** — Edge-preserving filter (3 steps)
11. **block_l2.rs** — Block L2 norm
12. **entropy.rs** — Entropy estimation helpers
13. **gab_coeffs.rs** — Gaborish coefficient computation

### Architecture Pattern

All modules use the `archmage` crate for safe SIMD dispatch:

```rust
// In jxl_simd/src/dct8.rs (x86_64 example):
#[cfg(target_arch = "x86_64")]
fn dct8_avx2(token: archmage::X64V3Token, ...) { ... }

// Scalar fallback (always present):
fn dct8_scalar(...) { ... }

// Dispatch:
pub fn dct_8x8(...) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = archmage::X64V3Token::summon() {
        return dct8_avx2(token, ...);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = archmage::NeonToken::summon() {
        return dct8_neon(token, ...);
    }
    dct8_scalar(...)
}
```

The existing NEON modules (gab.rs, gaborish5x5.rs) demonstrate the pattern. Use `archmage::NeonToken` for dispatch, `magetypes` for cross-platform vector types where applicable.

### NEON Intrinsics

Use `core::arch::aarch64::*` intrinsics within `archmage::NeonToken` guarded functions. The token guarantees NEON is available. Common patterns:

- `vld1q_f32` / `vst1q_f32` — Load/store 4xf32
- `vfmaq_f32` — Fused multiply-add
- `vmulq_f32` / `vaddq_f32` / `vsubq_f32` — Arithmetic
- `vdupq_n_f32` — Broadcast scalar
- `vtrn1q_f32` / `vtrn2q_f32` — Transpose pairs
- `vzip1q_f32` / `vzip2q_f32` — Interleave

NEON has 128-bit vectors (4xf32), not 256-bit like AVX2 (8xf32). Most AVX2 code processes two NEON-width chunks. The DCT and transpose kernels will need restructuring, not just 1:1 intrinsic swaps.

## Build & Test Commands

```bash
# Native x86_64
cargo test --tests

# WASM
CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --" cargo test --test hash_lock_features --target wasm32-wasip1 --no-default-features --features safe-mode

# AArch64 (requires Docker for cross)
CROSS_CONTAINER_OPTS="--volume /home/lilith/work:/home/lilith/work" cross test --test hash_lock_features -p jxl-encoder --target aarch64-unknown-linux-gnu --no-default-features --features safe-mode

# Benchmark all platforms
just bench-platforms

# Full cross-platform hash verification
just test-wasm  # WASM core tests
just test-aarch64  # AArch64 via cross
```

## Validation Strategy

1. **Hash lock tests must pass on all 3 platforms** — output bytes must not change
2. **Unit tests per module** — each NEON function should have a test comparing NEON vs scalar output on identical inputs
3. **Run `just bench-platforms`** after each module — track speedup on AArch64 (QEMU) as a proxy

## Performance Baseline (Feb 2026)

| Encode | x86_64 (AVX2) | WASM (scalar) | AArch64 QEMU (scalar) |
|--------|---------------|---------------|----------------------|
| 256x256 lossless | 1.1ms | 1.1ms (1.0x) | 10.0ms (9x) |
| 256x256 lossy d=1 | 20.9ms | 35.5ms (1.7x) | 696.8ms (33x) |
| 1024x1024 lossless | 103ms | 126ms (1.2x) | 354ms (3.4x) |
| 1024x1024 lossy d=1 | 410ms | 699ms (1.7x) | 10798ms (26x) |

Note: AArch64 numbers are QEMU emulation, not native. Native ARM performance will be much better. The lossless path (modular) is CPU-light — lossy VarDCT is where NEON matters.

## Files

- SIMD modules: `jxl_simd/src/*.rs`
- SIMD crate root: `jxl_simd/src/lib.rs` (`#![no_std]`, `extern crate alloc`)
- Hash-locked tests: `jxl_encoder/tests/hash_lock_features.rs`
- Benchmark: `jxl_encoder/examples/wasm_bench.rs`
- Justfile: `justfile` (targets: bench-platforms, test-wasm, test-aarch64)
- Cross config: `Cross.toml`

## Important Constraints

- `#![forbid(unsafe_code)]` in main encoder crate — SIMD intrinsics go in `jxl_simd` only
- `jxl_simd` is `no_std + alloc`
- The archmage token pattern must be followed — no raw `is_aarch64_feature_detected!` checks
- Don't change scalar fallback code — it defines the reference output
- All 684+ existing tests must continue passing
