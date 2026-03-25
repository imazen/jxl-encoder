# magetypes 0.5 & archmage 0.5 — NEON Implementation Guide

> **Version note:** This guide was written for archmage 0.5. The current version is 0.9.x. Core patterns (`#[archmage::arcane]`, token-based dispatch) remain the same, but some API details may differ.

## Quick Reference

**magetypes 0.5** provides cross-platform SIMD types with natural operators.
**archmage 0.5** provides capability tokens for safe SIMD dispatch.

Both support **AArch64 NEON (128-bit)**, **x86-64 AVX2 (256-bit)**, and **WASM SIMD128**.

---

## 1. Tokens and Dispatch

### NeonToken

```rust
use archmage::{NeonToken, SimdToken};

// Runtime check (elided if compiled with -C target-cpu=native)
if let Some(token) = NeonToken::summon() {
    // NEON is available
    process_neon(token, data);
}
```

### Standard Dispatch Pattern

```rust
pub fn my_function(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            my_function_avx2(token, input, output);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            my_function_neon(token, input, output);
            return;
        }
    }

    my_function_scalar(input, output);
}
```

This is the pattern used throughout jxl_simd (see dct8.rs, quantize.rs, etc.).

---

## 2. f32x4 API (NEON 128-bit vectors)

### Construction

```rust
use magetypes::simd::f32x4;

// Broadcast single value to all lanes
let v = f32x4::splat(token, 2.5);

// Load from memory (reads 4 elements)
let v = f32x4::from_slice(token, &data[offset..]);

// From array literal
let v = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);

// Zero vector
let v = f32x4::zero(token);

// From raw NEON register (inside #[arcane])
use core::arch::aarch64::float32x4_t;
let v = f32x4::from_float32x4_t(token, neon_reg);
```

### Arithmetic (Natural Operators)

```rust
let a = f32x4::splat(token, 2.0);
let b = f32x4::splat(token, 3.0);

let sum = a + b;        // [5.0, 5.0, 5.0, 5.0]
let diff = a - b;       // [-1.0, -1.0, -1.0, -1.0]
let prod = a * b;       // [6.0, 6.0, 6.0, 6.0]
let quot = a / b;       // [0.667, 0.667, 0.667, 0.667]
```

### Methods

```rust
// Fused multiply-add (single rounding, faster than a * b + c)
let fma = a.mul_add(b, c);  // a * b + c

// Element-wise operations
let abs_v = v.abs();        // absolute value
let rounded = v.round();    // round to nearest integer
let max_v = a.max(b);       // element-wise max
let min_v = a.min(b);       // element-wise min

// Comparisons (return mask vectors)
let mask = a.simd_ge(b);    // a >= b (element-wise)
let mask = a.simd_lt(b);    // a < b
let mask = a.simd_eq(b);    // a == b

// Blend (conditional select)
let result = f32x4::blend(mask, if_true, if_false);

// Horizontal reduction
let sum: f32 = v.reduce_add();  // sum of all 4 lanes

// Store to memory
v.store((&mut output[offset..offset+4]).try_into().unwrap());

// Extract raw register (for custom intrinsics)
let raw: float32x4_t = v.raw();
```

---

## 3. i32x4 API (NEON 128-bit integer vectors)

Same pattern as f32x4:

```rust
use magetypes::simd::i32x4;

let v = i32x4::splat(token, 42);
let v = i32x4::from_slice(token, &data[..]);
let v = i32x4::zero(token);

// Arithmetic
let sum = a + b;
let prod = a * b;

// Comparisons
let mask = a.simd_ge(b);

// Store
v.store((&mut output[..4]).try_into().unwrap());
```

### Type Conversions

```rust
// f32x4 → i32x4 (truncate toward zero)
let f = f32x4::from_array(token, [3.9, -2.1, 5.5, 0.3]);
let i = f.to_i32x4();  // [3, -2, 5, 0]

// i32x4 → f32x4
let i = i32x4::splat(token, 7);
let f = i.to_f32x4();  // [7.0, 7.0, 7.0, 7.0]
```

---

## 4. #[arcane] Macro

The `#[arcane]` macro makes NEON intrinsics safe without `unsafe`:

```rust
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn compute_neon(token: archmage::NeonToken, data: &[f32]) -> f32 {
    use magetypes::simd::f32x4;

    let mut sum = f32x4::zero(token);
    for chunk in (0..data.len()).step_by(4) {
        let v = f32x4::from_slice(token, &data[chunk..]);
        sum = sum + v * v;  // sum of squares
    }
    sum.reduce_add()
}
```

**What it does:**
- Generates inner function with `#[target_feature(enable = "neon")]`
- Makes NEON intrinsics and `magetypes` calls safe (no `unsafe` needed)
- Token parameter is proof that CPU support was verified via `summon()`

---

## 5. Complete Real Example: Gaborish 5x5 (from jxl_simd)

```rust
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn gaborish_5x5_neon(
    token: archmage::NeonToken,
    output: &mut [f32],
    input: &[f32],
    width: usize,
    height: usize,
    wc: f32,
    wr: f32,
    wd: f32,
    w_big_r: f32,
    wl: f32,
    w_big_d: f32,
) {
    use magetypes::simd::f32x4;

    // Broadcast weights to all SIMD lanes
    let wc_v = f32x4::splat(token, wc);
    let wr_v = f32x4::splat(token, wr);
    let wd_v = f32x4::splat(token, wd);
    let w_big_r_v = f32x4::splat(token, w_big_r);
    let wl_v = f32x4::splat(token, wl);
    let w_big_d_v = f32x4::splat(token, w_big_d);

    for y in 2..height-2 {
        let r_m2 = (y - 2) * width;
        let r_m1 = (y - 1) * width;
        let r_0 = y * width;
        let r_p1 = (y + 1) * width;
        let r_p2 = (y + 2) * width;

        let simd_end = width - 4;
        for x in (2..simd_end).step_by(4) {
            // Load center pixel
            let center = f32x4::from_slice(token, &input[r_0 + x..]);

            // Load rook neighbors (4 directions)
            let r_sum = f32x4::from_slice(token, &input[r_0 + x - 1..])
                + f32x4::from_slice(token, &input[r_0 + x + 1..])
                + f32x4::from_slice(token, &input[r_m1 + x..])
                + f32x4::from_slice(token, &input[r_p1 + x..]);

            // Load diagonal neighbors
            let d_sum = f32x4::from_slice(token, &input[r_m1 + x - 1..])
                + f32x4::from_slice(token, &input[r_m1 + x + 1..])
                + f32x4::from_slice(token, &input[r_p1 + x - 1..])
                + f32x4::from_slice(token, &input[r_p1 + x + 1..]);

            // 5x5 convolution with fused multiply-add
            let result = wc_v.mul_add(center,
                wr_v.mul_add(r_sum,
                    wd_v * d_sum  // ... + more terms
                )
            );

            // Store result
            result.store((&mut output[r_0 + x..r_0 + x + 4]).try_into().unwrap());
        }
    }
}
```

**Key patterns:**
- `f32x4::splat()` for broadcasting constants
- `f32x4::from_slice()` for loading data
- Natural arithmetic: `a + b`, `a * b`
- Chained `mul_add()` for efficiency
- `.store()` for writing results
- Process 4 elements per iteration

---

## 6. AVX2 vs NEON Porting Checklist

| Step | AVX2 (256-bit) | NEON (128-bit) |
|------|----------------|----------------|
| **1. Vector type** | `f32x8` | `f32x4` |
| **2. Token type** | `X64V3Token` | `NeonToken` |
| **3. Loop stride** | `.step_by(8)` | `.step_by(4)` |
| **4. Platform guard** | `#[cfg(target_arch = "x86_64")]` | `#[cfg(target_arch = "aarch64")]` |
| **5. API calls** | No change needed | Same methods |

### Example Conversion

**AVX2 version (8 lanes):**
```rust
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn process_avx2(token: archmage::X64V3Token, data: &[f32]) {
    use magetypes::simd::f32x8;

    for i in (0..data.len()).step_by(8) {
        let v = f32x8::from_slice(token, &data[i..]);
        let result = v * f32x8::splat(token, 2.0);
        result.store((&mut output[i..i+8]).try_into().unwrap());
    }
}
```

**NEON version (4 lanes):**
```rust
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn process_neon(token: archmage::NeonToken, data: &[f32]) {
    use magetypes::simd::f32x4;  // Changed from f32x8

    for i in (0..data.len()).step_by(4) {  // Changed from step_by(8)
        let v = f32x4::from_slice(token, &data[i..]);  // Changed type
        let result = v * f32x4::splat(token, 2.0);  // Changed type
        result.store((&mut output[i..i+4]).try_into().unwrap());  // Changed range
    }
}
```

**Only 3 changes:**
1. `f32x8` → `f32x4` (3 occurrences)
2. `.step_by(8)` → `.step_by(4)`
3. `i+8` → `i+4` in store range

The arithmetic code is **identical** — magetypes provides source-level portability.

---

## 7. Important Notes

### No gab_coeffs.rs Module

The handoff document mentions `gab_coeffs.rs`, but **it doesn't exist**. The coefficients are defined inline in:
- `gab.rs` — 3x3 decoder blur weights
- `gaborish5x5.rs` — 5x5 encoder sharpening weights

### Store Requires try_into()

```rust
// CORRECT
v.store((&mut output[..4]).try_into().unwrap());

// WRONG (won't compile)
v.store(&mut output[..4]);
```

The `.store()` method requires `&mut [f32; 4]` (fixed-size array), not `&mut [f32]` (slice).

### Bounds Check Elimination

Pre-slice arrays to known lengths before inner loops:

```rust
// SLOW: bounds check per SIMD load
for i in (0..len).step_by(4) {
    let v = f32x4::from_slice(token, &data[i..]);
}

// FAST: compiler proves bounds at compile time
let row = &data[y * width .. (y+1) * width];  // Pre-slice to row
for x in (0..width).step_by(4) {
    let v = f32x4::from_slice(token, &row[x..]);
}
```

### Remainder Loop

Always handle non-SIMD-aligned remainder with scalar code:

```rust
let chunks = len / 4;
for i in (0..chunks * 4).step_by(4) {
    // SIMD path
}
for i in chunks * 4 .. len {
    // Scalar remainder
}
```

---

## 8. Full API Reference

### f32x4 Methods

| Method | Description | Example |
|--------|-------------|---------|
| `splat(token, val)` | Broadcast scalar | `f32x4::splat(token, 1.5)` |
| `from_slice(token, &[f32])` | Load from memory | `f32x4::from_slice(token, &data[i..])` |
| `from_array(token, [f32; 4])` | From array literal | `f32x4::from_array(token, [1., 2., 3., 4.])` |
| `zero(token)` | Zero vector | `f32x4::zero(token)` |
| `from_float32x4_t(token, reg)` | From raw NEON | `f32x4::from_float32x4_t(token, neon_reg)` |
| `.mul_add(b, c)` | Fused multiply-add | `a.mul_add(b, c)  // a*b + c` |
| `.abs()` | Absolute value | `v.abs()` |
| `.round()` | Round to nearest | `v.round()` |
| `.max(other)` | Element-wise max | `a.max(b)` |
| `.min(other)` | Element-wise min | `a.min(b)` |
| `.simd_ge(other)` | Element-wise >= | `a.simd_ge(b)` |
| `.simd_lt(other)` | Element-wise < | `a.simd_lt(b)` |
| `.simd_eq(other)` | Element-wise == | `a.simd_eq(b)` |
| `.blend(mask, a, b)` | Conditional select | `f32x4::blend(mask, if_true, if_false)` |
| `.reduce_add()` | Horizontal sum | `let sum: f32 = v.reduce_add()` |
| `.store(&mut [f32; 4])` | Store to memory | `v.store((&mut out[..4]).try_into().unwrap())` |
| `.raw()` | Extract NEON register | `let reg: float32x4_t = v.raw()` |
| `.to_i32x4()` | Convert to i32 | `let i = f.to_i32x4()` |

### i32x4 Methods

Same as f32x4 except:
- No `.abs()`, `.round()`, `.mul_add()` (not applicable to integers)
- `.to_f32x4()` converts to float

---

## 9. Resources

- **archmage docs**: https://docs.rs/archmage/0.9
- **magetypes docs**: https://docs.rs/magetypes/0.9
- **Existing NEON code**: `jxl_simd/src/gaborish5x5.rs` (lines with `#[cfg(target_arch = "aarch64")]`)
- **AVX2 examples**: All other kernels in `jxl_simd/src/` (dct8.rs, quantize.rs, etc.)

---

## Summary

**magetypes 0.5** provides a unified SIMD API across x86-64, AArch64, and WASM. For NEON:

1. Use `f32x4` (4 lanes) instead of `f32x8` (8 lanes)
2. Use `NeonToken` from `archmage::NeonToken::summon()`
3. Process 4 elements at a time instead of 8
4. All API calls (splat, from_slice, mul_add, store) are identical
5. Arithmetic operators (+, -, *, /) work naturally
6. Use `#[arcane]` macro to make NEON intrinsics safe

Most porting work is mechanical: change vector width from 8→4, adjust loop strides, and the rest of the code stays the same.
