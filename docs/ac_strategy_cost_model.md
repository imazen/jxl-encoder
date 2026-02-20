# AC Strategy Cost Model Reference

Complete reference for all constants, formulas, and comparison logic in the AC strategy
selection pipeline. Covers both our encoder and libjxl (the reference implementation).

Last updated: 2026-02-19

## Core Cost Function: `estimate_entropy_full`

**Our file**: `jxl_encoder/src/vardct/ac_strategy.rs` (function at line ~700)
**libjxl file**: `lib/jxl/enc_ac_strategy.cc` (function `EstimateEntropy` at line ~418)

### Final Cost Formula

**Pixel-domain mode** (default at effort >= 5):
```
final_cost = entropy × entropy_mul + k_info_loss_mul × loss_scalar
```

**Coefficient-domain mode** (libjxl-tiny style, effort < 5):
```
final_cost = entropy + mask_val × (k_info_loss_mul × info_loss_sum + 50.4684 × sqrt(num_blocks × info_loss2_sum))
```

**Key rule**: `entropy_mul` applies ONLY to entropy, NOT to loss.
(libjxl enc_ac_strategy.cc:508-509)

---

## Entropy Multiplier Constants

8x8-class transforms are **normalized by DCT8's raw value (0.8)**.
Larger transforms use **raw values** directly.

| Strategy | Raw | Normalized (÷0.8) | Which Used | libjxl Source |
|----------|-----|-------------------|------------|---------------|
| DCT8 | 0.8 | **1.0** | Normalized | :525 |
| DCT4X8/DCT8X4 | 0.8593 | **1.074** | Normalized | :541 |
| DCT4X4 | 1.08 | **1.35** | Normalized | :530 |
| IDENTITY | 1.0428 | **1.304** | Normalized | :548 |
| DCT2X2 | 0.95 | **1.188** | Normalized | :535 |
| AFV0-3 | 0.8178 | **1.022** | Normalized | :553-576 |
| DCT16X8/DCT8X16 | **1.21** | — | Raw | :892 |
| DCT16X16 | **1.34** | — | Raw | :893 |
| DCT32X16/DCT16X32 | **1.49** | — | Raw | :894 |
| DCT32X32 | **1.48** | — | Raw | :895 |
| DCT64X32/DCT32X64 | **2.25** | — | Raw | :896 |
| DCT64X64 | **2.25** | — | Raw | :897 |

**Normalization**: libjxl enc_ac_strategy.cc:584:
`entropy_mul = tx.entropy_mul / kTransforms8x8[0].entropy_mul`

Our code: `ac_strategy.rs:entropy_mul_for_strategy()` at line 383.

---

## Distance-Scaled Constants

**Formula** (`ac_strategy.rs:344`, libjxl enc_ac_strategy.cc:1111):

```
ratio = (distance + 0.1373) / (1.0 + 0.1373)

k_info_loss_mul = 1.2        × ratio^0.3368
k_zeros_mul     = 9.3089     × ratio^0.5099
k_cost_delta    = 10.8333    × ratio^0.3670
```

| Constant | Base Value | Power | Purpose |
|----------|-----------|-------|---------|
| `k_info_loss_mul` | 1.2 | 0.3368 | Weight of pixel loss in final cost |
| `k_zeros_mul` | 9.3089 | 0.5099 | Weight of nzeros bit count |
| `k_cost_delta` | 10.8333 | 0.3670 | Weight of coefficient entropy |

---

## Channel Constants

| Constant | X (c=0) | Y (c=1) | B (c=2) | Source |
|----------|---------|---------|---------|--------|
| Mask offset | 12.0 | 0.0 | 4.0 | ac_strategy.rs:318, libjxl:447-449 |
| Channel mul | 8.2^8 = 2.088e7 | 1.0 | 1.03^8 = 1.267 | ac_strategy.rs:324, libjxl:480-484 |

---

## X Channel Penalty (Large Transforms)

Applied when `c == 0 && num_blocks >= 2`:

```
w = 1.0 + min(3.0, num_blocks / 8.0)
entropy *= w      // applied INSIDE channel loop after c=0
total_loss *= w   // applied INSIDE channel loop after c=0
```

| Transform | num_blocks | w |
|-----------|-----------|---|
| DCT8 | 1 | 1.0 (no penalty) |
| DCT16x8 | 2 | 1.25 |
| DCT16x16 | 4 | 1.5 |
| DCT32x32 | 16 | 3.0 |
| DCT64x64 | 64 | 4.0 |

Penalty ordering: After c=0 channel, the accumulated entropy and loss
include ONLY X channel contribution. The `*= w` multiplies everything
accumulated so far. Y and B contributions are added afterwards without penalty.

libjxl enc_ac_strategy.cc:496-501 — identical.

---

## Loss Scalar Formula

```
n = num_blocks × 64
loss_scalar = (total_pixel_loss / n)^(1/8) × n / quant_norm16
```

Where `total_pixel_loss = sum over pixels of (mask[px] × error[px])^8 × channel_mul[c]`
with X channel penalty applied.

---

## Quant Norm Computation

| num_blocks | Method | Formula |
|-----------|--------|---------|
| 1 | Direct | `quant_field[by * xsize_blocks + bx]` |
| 2 | MAX | `max(q1, q2)` |
| 4+ | L16 norm | `(mean(q^16))^(1/16)` |

libjxl enc_ac_strategy.cc:384-413 — identical.

---

## External Multipliers (Applied to EstimateEntropy Return Value)

### Pixel-Domain Mode

| Transform Class | External Mul | Formula |
|----------------|-------------|---------|
| All 8x8-class (DCT8, DCT4x8, DCT8x4, DCT4x4, ID, DCT2x2, AFV) | `mul8x8` | `1.0 + (-0.4) / (distance + 1.4)` |
| All larger transforms | `1.0` | (entropy_mul already applied internally) |

At d=1.0: `mul8x8 = 1.0 - 0.167 = 0.833`
At d=0.5: `mul8x8 = 1.0 - 0.211 = 0.789`
At d=2.0: `mul8x8 = 1.0 - 0.118 = 0.882`

libjxl enc_ac_strategy.cc:863-866: `k8x8mul1=-0.4, k8x8mul2=1.0, k8x8base=1.4`

### Coefficient-Domain Mode

Formula: `mul = mul2 + mul1 / (distance + base)`

| Transform | mul1 | mul2 | base | At d=1.0 |
|-----------|------|------|------|----------|
| DCT8 | -0.4125 | 0.8052 | 1.4 | 0.633 |
| DCT16x8 | -0.55 | 0.9020 | 1.6 | 0.690 |
| DCT16x16 | -0.65 | 0.88 | 1.8 | 0.648 |
| DCT4x8 | -0.375 | 0.66 | 1.3 | 0.497 |
| DCT4x4 | -0.3375 | 0.6375 | 1.2 | 0.484 |

Note: DCT8 has a 0.75 factor baked in (`effort.rs:220`).

---

## Entropy Mul Adjustments (Additive)

### kFavor2X2AtHighQuality (IDENTITY, DCT2X2 only)

```
if distance < 5.0:
    weight = ((5.0 - distance) / 5.0)^2
    adjust = -0.4 × weight     // SUBTRACTS from entropy_mul
else:
    adjust = 0.0
```

At d=1.0: adjust = -0.4 × 0.64 = **-0.256** (makes IDENTITY/DCT2X2 cheaper)

### kAvoidEntropyOfTransforms (DCT4x8, DCT8x4, DCT4x4, AFV only)

```
if distance > 4.0:
    if distance < 12.0:
        mul = 8.0 / (distance - 4.0)
    else:
        mul = 1.0
    adjust = 0.5 × mul         // ADDS to entropy_mul
else:
    adjust = 0.0
```

Active only at d > 4.0.

---

## Base Cost (Coefficient-Domain Only)

```
pixel-domain:      base_cost_8x8 = 0.0
coefficient-domain: base_cost_8x8 = 3.0 × mul8x8
```

Added to each single-block transform cost.

---

## Strategy Selection: 16x16 Region

**File**: `ac_strategy_search.rs:find_best_16x16_transform`

```
// Single-block costs already include mul8x8 + base_cost
cost_all_single = e[0][0] + e[0][1] + e[1][0] + e[1][1]

// Rectangular: min of merged vs split per half
cost16x8 = min(e_left,  e[0][0]+e[1][0]) + min(e_right,  e[0][1]+e[1][1])
cost8x16 = min(e_top,   e[0][0]+e[0][1]) + min(e_bottom, e[1][0]+e[1][1])

// Square
cost16x16 = estimate_entropy(DCT16X16)  // mul16x16=1.0 in pixel-domain

best_rect  = min(cost16x8, cost8x16)
best_large = min(best_rect, cost16x16)

if best_large >= cost_all_single → keep singles
elif cost16x16 <= best_rect       → use DCT16x16
elif cost16x8 < cost8x16          → try 16x8 per-column
else                               → try 8x16 per-row
```

**favor_single_mul**: Applied to single-block costs during non-aligned evaluation pass.
Default 1.0 (aligned pass). Values < 1.0 make singles cheaper → harder for large transforms.

libjxl enc_ac_strategy.cc:792-823 — structurally identical.

---

## Strategy Selection: 32x32 Region

**File**: `ac_strategy_search.rs:find_best_32x32_transform`

```
// Gate: d < 0.5 → skip, use four 16x16 sub-evaluations only
if distance < 0.5: return

// Evaluate 32x32 candidates
entropy_32x32   = estimate_entropy(DCT32X32)     // mul=1.0 pixel-domain
entropy_32x16_0 = estimate_entropy(DCT32X16, bx)
entropy_32x16_1 = estimate_entropy(DCT32X16, bx+2)
entropy_16x32_0 = estimate_entropy(DCT16X32, by)
entropy_16x32_1 = estimate_entropy(DCT16X32, by+2)

// Run four 16x16 sub-evaluations (sets ac_strategy)
for each 2x2 sub-region:
    find_best_16x16_transform(...)

// Re-estimate cost of what 16x16 chose
cost_sub = sum over is_first blocks:
    mul_for_strategy × estimate_entropy(selected_strategy)

// Compare all options
best = min(cost_sub, entropy_32x32, entropy_32x16_total, entropy_16x32_total)
```

Note: `sub_mul8x8` at 32x32 level uses same formula as top-level mul8x8.

---

## Strategy Selection: 64x64 Region

**File**: `ac_strategy_search.rs:find_best_64x64_transform`

Same structure as 32x32 but with DCT64x64, DCT64x32, DCT32x64 candidates.
Distance gate: `d >= 1.0`.

---

## Distance Gates (Our Encoder vs libjxl)

| Transform | Our Gate | libjxl Gate |
|-----------|----------|-------------|
| DCT16x16 | none | none |
| DCT32x32 | d >= 0.5 | none |
| DCT32x16/16x32 | d >= 0.5 | none |
| DCT64x64 | d >= 1.0 | none |
| DCT64x32/32x64 | d >= 1.0 | none |

libjxl evaluates ALL strategies at ALL distances. Our gates exist because our cost model
under-penalizes large transforms at very low distances, causing butteraugli regressions.

---

## Effort Gating

| Feature | Min Effort | libjxl Equivalent |
|---------|-----------|-------------------|
| DCT4x8/AFV evaluation | 6 | kWombat (speed_tier 4) |
| DCT32 evaluation | 5 | kHare (speed_tier 5) |
| DCT64 evaluation | 7 | kSquirrel (speed_tier 3) |
| Non-aligned eval pass | 6 | kWombat |
| Fine-grained step=1 | 9 | kTortoise |
| AdjustQuantBlockAC | 5 | kHare |
| Pixel-domain loss | 5 | kHare |
| Butteraugli loop | 8 | kKitten (speed_tier 2) |

---

## Known Differences vs libjxl

1. **Distance gates**: We gate large transforms; libjxl doesn't. (Workaround for cost model imprecision.)

2. **Boundary crossing checks**: libjxl has `MultiBlockTransformCrossesHorizontalBoundary` /
   `MultiBlockTransformCrossesVerticalBoundary` to reject transforms spanning incompatible
   regions. We don't implement this.

3. **Entropy estimate propagation**: libjxl stores updated entropy_estimate after each merge
   level via `SetEntropyForTransform`. Our code re-evaluates from scratch at each level.
   Functionally equivalent but may produce slightly different results due to floating-point
   ordering.

4. **DCT64x64 in merge loop**: libjxl evaluates DCT64x64 through
   `FindBestFirstLevelDivisionForSquare(8, true, ...)` called from the DCT32X64 merge entry.
   DCT64x64 is NOT a separate entry in `kTransformsForMerge` (line 919 is commented out).

5. **Coefficient-domain multiplier constants**: Our DCT8 coefficient-domain multiplier has
   a 0.75 factor baked in (effort.rs:220). This doesn't affect pixel-domain mode.
