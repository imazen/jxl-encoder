# libjxl AC Strategy Selection: Complete Decision Tree

Full extraction from libjxl source (`lib/jxl/enc_ac_strategy.cc`, `enc_adaptive_quantization.cc`,
`enc_heuristics.cc`, `quantizer.cc`, `quantizer-inl.h`, `common.h`, `enc_params.h`).

## SpeedTier / Effort Mapping

```
speed_tier    name        effort = 10 - speed_tier
-1            kTectonicPlate   11
 0            kGlacier         10
 1            kTortoise         9
 2            kKitten           8
 3            kSquirrel         7   (DEFAULT)
 4            kWombat           6
 5            kHare             5
 6            kCheetah          4
 7            kFalcon           3
 8            kThunder          2
 9            kLightning        1
```

**Pattern**: libjxl checks `speed_tier <= X` which means `effort >= (10 - X)`.

---

## 1. Top-Level Flow

Source: `enc_heuristics.cc:1088-1207`

```
ProcessHeuristics():
    1. Compute initial quant field (effort-gated)
    2. Apply gaborish inverse (if enabled)
    3. Find best dequant matrices
    4. CfL pass 1 (effort >= 7, without AC strategy/quant)
    5. Per 64x64 tile (8x8 blocks):
        a. Choose block sizes (AC strategy selection)
        b. AdjustQuantField (mean-max mixing for multi-block)
        c. SetQuantFieldRect
        d. CfL pass 2 (effort >= 5, with AC strategy + quant)
    6. Finalize
```

### Initial Quant Field (effort-gated)

```
if effort < 5 (speed_tier > kHare):
    q = 0.79 / butteraugli_distance
    quant_field = flat(q)
    masking = 1.0 / (q + 0.001)
    global_scale from ComputeGlobalScaleAndQuant(quant_dc, q, 0)

if effort >= 5 (speed_tier <= kHare):
    d_for_iqf = butteraugli_distance * (gab ? 1.0 : 0.62)
    quant_field = InitialQuantField(d_for_iqf, opsin, 1.0)
    q = 0.39 / butteraugli_distance
    global_scale from ComputeGlobalScaleAndQuant(quant_dc, q, 0)
```

---

## 2. All 27 AC Strategy Types

| Code | Name | Block Size | Coverage (8x8 blocks) | Used by Encoder |
|------|------|-----------|----------------------|-----------------|
| 0 | DCT | 8x8 | 1x1 = 1 | Yes |
| 1 | IDENTITY | 8x8 | 1x1 = 1 | Yes |
| 2 | DCT2X2 | 8x8 | 1x1 = 1 | Yes |
| 3 | DCT4X4 | 8x8 | 1x1 = 1 | Yes |
| 4 | DCT16X16 | 16x16 | 2x2 = 4 | Yes |
| 5 | DCT32X32 | 32x32 | 4x4 = 16 | Yes |
| 6 | DCT16X8 | 16x8 | 1x2 = 2 | Yes |
| 7 | DCT8X16 | 8x16 | 2x1 = 2 | Yes |
| 8 | DCT32X8 | 32x8 | 1x4 = 4 | **No** (commented out) |
| 9 | DCT8X32 | 8x32 | 4x1 = 4 | **No** (commented out) |
| 10 | DCT32X16 | 32x16 | 2x4 = 8 | Yes |
| 11 | DCT16X32 | 16x32 | 4x2 = 8 | Yes |
| 12 | DCT4X8 | 8x8 | 1x1 = 1 | Yes |
| 13 | DCT8X4 | 8x8 | 1x1 = 1 | Yes |
| 14 | AFV0 | 8x8 | 1x1 = 1 | Yes |
| 15 | AFV1 | 8x8 | 1x1 = 1 | Yes |
| 16 | AFV2 | 8x8 | 1x1 = 1 | Yes |
| 17 | AFV3 | 8x8 | 1x1 = 1 | Yes |
| 18 | DCT64X64 | 64x64 | 8x8 = 64 | Yes |
| 19 | DCT64X32 | 64x32 | 4x8 = 32 | Yes |
| 20 | DCT32X64 | 32x64 | 8x4 = 32 | Yes |
| 21-26 | DCT128+ | 128+ | 16+ | **No** (commented out) |

**19 of 27 are used**. DCT32X8, DCT8X32, and all DCT128+ are never evaluated.

---

## 3. Strategy Gating by Effort Level

### Which strategies are available

| Effort | 8x8-Class Strategies | Merge Strategies (16x16+) |
|--------|---------------------|--------------------------|
| 1-4 | DCT8 only (FillDCT8, no search) | None |
| 5 | DCT, DCT4X4, DCT2X2, IDENTITY | DCT16X8/8X16, DCT16X16, DCT32X16/16X32, DCT32X32, DCT64X32/32X64, DCT64X64 |
| 6 | + DCT4X8, DCT8X4, AFV0-3 (all 10) | Same + non-aligned 16x16 search |
| 7 | All 10 | Same |
| 8 | All 10 | Same |
| 9 | All 10 | Same (non-aligned 32x32 at step=2) |
| 10 | All 10 | Same (non-aligned 32x32 at step=1) |

### 8x8 transform `encoding_speed_tier_max_limit`

A transform is evaluated when `limit >= speed_tier`:

| Transform | Max Limit | Available at |
|-----------|----------|-------------|
| DCT8 | 9 | All effort levels |
| DCT4X4 | 5 | effort >= 5 (kHare) |
| DCT2X2 | 5 | effort >= 5 |
| DCT4X8 | 4 | effort >= 6 (kWombat) |
| DCT8X4 | 4 | effort >= 6 |
| IDENTITY | 5 | effort >= 5 |
| AFV0-3 | 4 | effort >= 6 |

### Merge transform gates

| Transform | decode_tier_max | encode_tier_max | Priority |
|-----------|---------------|-----------------|----------|
| DCT16X8/8X16 | 4 | 5 | 2 |
| DCT16X32/32X16 | 4 | 4 | 4 |
| DCT64X32/32X64 | 1 | 3 | 6 |

DCT32X32 additional gate: `decoding_speed_tier < 4` (typically true since default is 0).

---

## 4. Distance-Scaled Cost Model Constants

Source: `enc_ac_strategy.cc:1107-1123`

```
kBias = 0.13731742964354549
ratio = (butteraugli_distance + kBias) / (1.0 + kBias)
      = (d + 0.137) / 1.137

info_loss_multiplier = 1.2       * ratio^0.33678
zeros_mul            = 9.30891   * ratio^0.50991
cost_delta           = 10.83327  * ratio^0.36703
```

At d=1.0: ratio=1.0, all constants = base values.

| d | ratio | info_loss_mul | zeros_mul | cost_delta |
|---|-------|--------------|-----------|------------|
| 0.25 | 0.340 | 0.833 | 5.377 | 7.597 |
| 0.5 | 0.560 | 0.960 | 6.929 | 8.860 |
| 1.0 | 1.000 | 1.200 | 9.309 | 10.833 |
| 2.0 | 1.879 | 1.508 | 12.938 | 13.812 |
| 3.0 | 2.759 | 1.724 | 15.759 | 15.962 |
| 5.0 | 4.518 | 2.044 | 20.241 | 19.148 |

---

## 5. The `mul8x8` Post-Hoc Multiplier

Source: `enc_ac_strategy.cc:863-866`

```
mul8x8 = 1.0 + (-0.4) / (butteraugli_target + 1.4)
```

Applied to ALL 8x8-class entropies AFTER `EstimateEntropy` returns, BEFORE comparison with
larger transforms. Always < 1.0, giving 8x8 transforms a built-in cost advantage.

| d | mul8x8 | 8x8 advantage |
|---|--------|--------------|
| 0.5 | 0.789 | 21.1% cheaper |
| 1.0 | 0.833 | 16.7% cheaper |
| 2.0 | 0.882 | 11.8% cheaper |
| 5.0 | 0.938 | 6.3% cheaper |

---

## 6. `entropy_mul` Per Transform

### 8x8-class transforms (normalized by DCT8's 0.8)

Source: `enc_ac_strategy.cc:525-576, 584`

```
entropy_mul_used = tx.entropy_mul / 0.8   (normalized so DCT8 = 1.0)
```

| Transform | Raw | Normalized |
|-----------|-----|-----------|
| DCT8 | 0.80000 | 1.000 |
| DCT4X4 | 1.08000 | 1.350 |
| DCT2X2 | 0.95000 | 1.188 |
| DCT4X8 | 0.85932 | 1.074 |
| DCT8X4 | 0.85932 | 1.074 |
| IDENTITY | 1.04275 | 1.303 |
| AFV0-3 | 0.81779 | 1.022 |

### Larger transforms (NOT normalized, used directly)

Source: `enc_ac_strategy.cc:892-920`

| Transform | entropy_mul |
|-----------|------------|
| DCT16X8 / DCT8X16 | 1.21 |
| DCT16X16 | 1.34 |
| DCT16X32 / DCT32X16 | 1.49 |
| DCT32X32 | 1.48 |
| DCT64X32 / DCT32X64 | 2.25 |
| DCT64X64 | 2.25 |

### kFavor2X2 adjustment (DCT2X2 and IDENTITY only, d < 5.0)

Source: `enc_ac_strategy.cc:585-591`

```
if d < 5.0:
    weight = ((5.0 - d) / 5.0)^2
    entropy_mul -= 0.4 * weight
```

| d | weight | reduction | DCT2X2 effective | IDENTITY effective |
|---|--------|-----------|-----------------|-------------------|
| 0.5 | 0.810 | -0.324 | 0.864 | 0.979 |
| 1.0 | 0.640 | -0.256 | 0.932 | 1.047 |
| 2.0 | 0.360 | -0.144 | 1.044 | 1.159 |
| 3.0 | 0.160 | -0.064 | 1.124 | 1.239 |
| 5.0+ | 0.000 | 0.000 | 1.188 | 1.303 |

### kAvoidEntropyOfTransforms (NOT DCT/DCT2X2/IDENTITY, d > 4.0)

Source: `enc_ac_strategy.cc:592-601`

```
if d > 4.0:
    if d < 12.0: mul = 8.0 / (d - 4.0)
    else: mul = 1.0
    entropy_mul += 0.5 * mul
```

Affects DCT4X4, DCT4X8, DCT8X4, AFV0-3. Massive penalty at d just above 4.0.

---

## 7. `EstimateEntropy` — The Complete Cost Function

Source: `enc_ac_strategy.cc:364-511`

### Step 1: Compute `quant_norm16`

```
num_blocks = covered_blocks_x * covered_blocks_y

if num_blocks == 1:
    quant_norm16 = Quant(bx, by)

elif num_blocks == 2:
    quant_norm16 = max(Quant(block0), Quant(block1))

else (4+):
    sum = 0
    for each sub-block (ix, iy):
        q = Quant(bx+ix, by+iy)
        q = q^16                         // via 4 squarings
        sum += q
    quant_norm16 = (sum / num_blocks)^(1/16)
```

### Step 2: Forward DCT

```
for c in {0, 1, 2}:
    TransformFromPixels(strategy, pixels[c], block + c*size, ...)
```

### Step 3: Per-channel coefficient quantization + entropy + loss

```
for c in {0=X, 1=Y, 2=B}:
    cmap_factor = cmap_factors[c]        // 0 for Y

    for i in 0..num_blocks*64:
        // Quantize
        in     = block[c*size + i]
        in_y   = block[size + i] * cmap_factor
        val    = (in - in_y) * inv_matrix[i] * quant_norm16
        rval   = Round(val)
        diff   = val - rval

        // Dequantized error (for IDCT)
        mem[i] = matrix[i] * diff

        // Entropy: sqrt cost model
        entropy_v += sqrt(|rval|)
        if rval != 0: nzeros_v += 1

    // Coefficient entropy cost
    entropy += cost_delta * sum(entropy_v)

    // Nzeros cost
    nbits = CeilLog2Nonzero(num_nzeros + 1) + 1
    entropy += zeros_mul * (CeilLog2Nonzero(nbits + 17) + nbits)

    // Pixel-domain loss (IDCT of quantization error)
    TransformToPixels(strategy, mem, pixel_error_block, ...)

    channel_offsets = [12.0, 0.0, 4.0]
    kChannelMul = [8.2^8, 1.0, 1.03^8]    // = [2.052e7, 1.0, 1.267]

    for each pixel (px, py) in block:
        masked = (mask1x1[px, py] + channel_offsets[c]) * pixel_error[px, py]
        masked = masked^8                  // 8th power norm
        loss_c += masked

    loss_c *= kChannelMul[c]
    loss += loss_c

    // X channel penalty (c=0 and multi-block only)
    if c == 0 && num_blocks >= 2:
        w = 1.0 + min(3.0, num_blocks / 8.0)
        entropy *= w
        loss *= w
```

### Step 4: Final combination

```
// Average loss, undo 8th power, scale by area, normalize by quant
loss_scalar = (loss / (num_blocks * 64))^(1/8)
            * (num_blocks * 64) / quant_norm16

// Apply per-transform bias to entropy only
entropy *= entropy_mul

// Add pixel-domain loss
entropy += info_loss_multiplier * loss_scalar

return entropy
```

### X channel penalty factor `w`

| num_blocks | w |
|-----------|---|
| 1 | 1.0 (no penalty) |
| 2 | 1.25 |
| 4 | 1.5 |
| 8 | 2.0 |
| 16 | 3.0 |
| 24+ | 4.0 (capped) |

---

## 8. Phase 1: Best 8x8 Transform per Block

Source: `enc_ac_strategy.cc:860-878`

```
for each 8x8 block (ix, iy) in the 64x64 tile:
    FindBest8x8Transform(config, ix, iy, cmap_factors, ...)
    ac_strategy.Set(bx + ix, by + iy, best_type)
    entropy_estimate[iy * 8 + ix] = best_entropy * mul8x8
```

`FindBest8x8Transform` (lines 513-613):
1. For each enabled transform in `kTransforms8x8[]`:
   a. Compute `entropy_mul = tx.entropy_mul / 0.8`
   b. Apply kFavor2X2 (DCT2X2/IDENTITY, d<5)
   c. Apply kAvoidEntropyOfTransforms (non-DCT/2X2/ID, d>4)
   d. Call `EstimateEntropy(acs, entropy_mul, ...)`
   e. Track minimum entropy and corresponding transform
2. Return best transform + its entropy

---

## 9. Phase 2: Hierarchical Merging

Source: `enc_ac_strategy.cc:880-1057`

### Pass 1: Aligned merges (kTransformsForMerge loop)

The merge table is iterated in order. When the SECOND entry of a priority pair is reached at
an aligned position, `FindBestFirstLevelDivisionForSquare` evaluates square + both rect options.

**Priority 2 — DCT16X8/8X16/16X16** (at 2-aligned positions):

```
for each position where (cy | cx) % 2 == 0 && enough room:
    FindBestFirstLevelDivisionForSquare(
        blocks=2,
        allow_square=true,
        entropy_mul_JXK=1.21,     // DCT16X8/DCT8X16
        entropy_mul_JXJ=1.34      // DCT16X16
    )
```

**Priority 4 — DCT32X16/16X32/32X32** (at 4-aligned positions):

```
for each position where (cy | cx) % 4 == 0 && enough room:
    FindBestFirstLevelDivisionForSquare(
        blocks=4,
        allow_square=enable_32x32,   // gated: decoding_speed_tier < 4
        entropy_mul_JXK=1.49,        // DCT32X16/DCT16X32
        entropy_mul_JXJ=1.48         // DCT32X32
    )
```

**Priority 6 — DCT64X32/32X64/64X64** (at 8-aligned positions):

```
for each position where (cy | cx) % 8 == 0 && enough room:
    FindBestFirstLevelDivisionForSquare(
        blocks=8,
        allow_square=true,
        entropy_mul_JXK=2.25,        // DCT64X32/DCT32X64
        entropy_mul_JXJ=2.25         // DCT64X64
    )
```

Odd-positioned blocks use `TryMergeAcs` for individual rectangle merges.

### Pass 2: Non-aligned 16x16 (effort >= 6)

```
if speed_tier >= kHare: return   // effort < 6, skip

for cy in 0..rect.ysize()-1:
    for cx in 0..rect.xsize()-1:
        if (cy | cx) % 2 != 0:   // only non-aligned
            FindBestFirstLevelDivisionForSquare(2, true, ..., 1.21, 1.34)
```

### Pass 3: Non-aligned 32x32 (effort >= 6)

```
step = (speed_tier >= kTortoise) ? 2 : 1
     = (effort <= 9) ? 2 : 1

for cy in 0..rect.ysize()-3  step step:
    for cx in 0..rect.xsize()-3  step step:
        if (cy | cx) % 4 == 0: continue   // already tried
        FindBestFirstLevelDivisionForSquare(4, enable_32x32, ..., 1.49, 1.48)
```

---

## 10. `FindBestFirstLevelDivisionForSquare`

Source: `enc_ac_strategy.cc:703-825`

The core decision function. Given an NxN square (N blocks on each side), chooses between:
- **JXJ** (square): e.g., DCT16X16, DCT32X32, DCT64X64
- **Two JXK** (vertical split): e.g., two DCT16X8s, two DCT32X16s
- **Two KXJ** (horizontal split): e.g., two DCT8X16s, two DCT16X32s
- **Keep existing** (sum of current sub-block entropies)

### Boundary checks

```
Check all 4 edges of the NxN square for multi-block transforms crossing them.
If any crossing found: bail out, can't analyze this square.

allow_JXK = no vertical crossing at midpoint
allow_KXJ = no horizontal crossing at midpoint
```

### Aggregate current entropies into quadrants

```
float entropy[2][2] = {}
for dy in 0..N:
    for dx in 0..N:
        entropy[dy / half][dx / half] += entropy_estimate[(cy+dy)*8 + (cx+dx)]
```

### Evaluate candidates

```
if allow_JXK:
    if left_half not already JXK:
        EstimateEntropy(JXK, entropy_mul_JXK, ...) -> entropy_JXK_left
    if right_half not already JXK:
        EstimateEntropy(JXK, entropy_mul_JXK, ...) -> entropy_JXK_right

if allow_KXJ:
    if top_half not already KXJ:
        EstimateEntropy(KXJ, entropy_mul_JXK, ...) -> entropy_KXJ_top
    if bottom_half not already KXJ:
        EstimateEntropy(KXJ, entropy_mul_JXK, ...) -> entropy_KXJ_bottom

if allow_square:
    EstimateEntropy(JXJ, entropy_mul_JXJ, ...) -> entropy_JXJ
```

### Decision

```
// Best vertical split: each half independently picks JXK or keeps existing
costJxN = min(entropy_JXK_left,  entropy[0][0] + entropy[1][0])
        + min(entropy_JXK_right, entropy[0][1] + entropy[1][1])

// Best horizontal split: each half independently picks KXJ or keeps existing
costNxJ = min(entropy_KXJ_top,    entropy[0][0] + entropy[0][1])
        + min(entropy_KXJ_bottom, entropy[1][0] + entropy[1][1])

if entropy_JXJ < costJxN && entropy_JXJ < costNxJ:
    -> Use square transform (JXJ)
elif costJxN < costNxJ:
    -> Vertical orientation wins
    -> Left half:  use JXK if JXK < existing, else keep
    -> Right half: use JXK if JXK < existing, else keep
else:
    -> Horizontal orientation wins
    -> Top half:    use KXJ if KXJ < existing, else keep
    -> Bottom half: use KXJ if KXJ < existing, else keep
```

Each half is decided independently. One half can take the rectangle while the other keeps
its smaller blocks.

---

## 11. `TryMergeAcs`

Source: `enc_ac_strategy.cc:618-653`

Simple merge for edge cases. Used for non-square-aligned leftover blocks.

```
// Check priority overlap
for each sub-block (ix, iy) covered by candidate:
    if priority[(cy+iy)*8 + (cx+ix)] >= candidate_priority:
        return  // overlap, reject
    entropy_current += entropy_estimate[...]

// Evaluate candidate
EstimateEntropy(candidate, entropy_mul, ...) -> entropy_candidate

// Compare
if entropy_candidate >= entropy_current:
    return  // reject, not cheaper

// Accept: update priorities, zero out old entropies, set new strategy
for each sub-block:
    entropy_estimate[...] = 0
    priority[...] = candidate_priority
ac_strategy.Set(bx+cx, by+cy, candidate)
entropy_estimate[cy*8 + cx] = entropy_candidate
```

---

## 12. `mask1x1` Computation

Source: `enc_adaptive_quantization.cc:498-526, 634-662`

### Step 1: Per-pixel Laplacian

```
For each pixel (x, y) in Y channel of XYB image:
    base = 0.25 * (Y[x,y-1] + Y[x,y+1] + Y[x-1,y] + Y[x+1,y])
    gammac = RatioOfDerivativesOfCubicRootToSimpleGamma(Y[x,y] + 0.019)
    diff = |gammac * (Y[x,y] - base)|
    diff = log1p(diff)
    mask1x1[x,y] = 1.0 / (diff + 0.01)
```

`RatioOfDerivativesOfCubicRootToSimpleGamma` constants:
```
kSGmul = 226.77216153508914
kSGmul2 = 1.0 / 73.377132366608819
kSGRetMul = kSGmul2 * 18.6580932135 * ln(2)
kSGVOffset = 7.7825991679894591
match_gamma_offset = 0.019
```

### Step 2: Symmetric5 blur

```
kFilterMask1x1[5] = {
    0.364911248,   // diagonal
    0.05,          // distance-2 h/v
    0.1688888021,  // L-shaped (sqrt(5))
    0.221069183,   // distance-2 diagonal
    0.306563504    // direct h/v neighbors
}
center_weight = 1.0
normalize = 1.0 / (1.0 + 4*(k[0] + k[1] + k[2] + k[4] + 2*k[3]))
```

---

## 13. `AdjustQuantField` (Post-Strategy Mean-Max Mixing)

Source: `enc_adaptive_quantization.cc:1198-1248`

Called AFTER AC strategy selection. For multi-block transforms (coverage >= 4), replaces
per-8x8-block quant values with an interpolation between max and mean.

```
kLimit = 1.54138
kMul   = 0.56391
kMin   = 0.0

if butteraugli_target > kLimit:
    mean_max_mixer = 1.0 - (butteraugli_target - kLimit) * kMul
    mean_max_mixer = max(mean_max_mixer, kMin)
else:
    mean_max_mixer = 1.0

For blocks with coverage >= 4:
    quant = mean_max_mixer * max + (1 - mean_max_mixer) * mean
For blocks with coverage < 4:
    quant = max    (pure max, no mixing)
```

| d | mean_max_mixer | Behavior |
|---|---------------|----------|
| 1.0 | 1.0 | Pure max |
| 1.5 | 1.0 | Pure max |
| 2.0 | 0.741 | 74% max + 26% mean |
| 3.0 | 0.178 | 18% max + 82% mean |
| 3.32 | 0.0 | Pure mean |
| 5.0+ | 0.0 | Pure mean |

---

## 14. `AdjustQuantBlockAC` (Dead-Zone Thresholds)

Source: `enc_group.cc:104-328`

### Effort gating

```
if effort >= 5 (speed_tier <= kHare):
    Full AdjustQuantBlockAC with adaptive thresholds

if effort < 5:
    Fixed thresholds:
    Y:    {0.56, 0.62, 0.62, 0.62}
    X, B: {0.58, 0.62, 0.62, 0.62}
```

### Initial thresholds (effort >= 5)

```
{0.58, 0.64, 0.64, 0.64}
```

### Multi-block threshold reduction

```
For all channels, if multi-block (xsize > 1 || ysize > 1):
    thresholds[i] -= clamp(0.003 * xsize * ysize, 0, 0.08)
    thresholds[i] = max(thresholds[i], 0.54)

For X and B channels only, if coverage >= 4:
    thresholds[i] -= 0.00744 * xsize * ysize
    thresholds[i] = max(thresholds[i], 0.5)
```

### Skipped strategies (no adjustment)

```
kPartialBlockKinds = IDENTITY | DCT2X2 | DCT4X4 | DCT4X8 | DCT8X4 | AFV0-3
```

### Y channel special adjustments

```
// Quadrant zero-check: if any quadrant has 0 nonzeros with error > 0.46
//   quant += 1, adjust thresholds to 0.9999 * hfMaxError

// High-frequency corner: per-channel multipliers {70, 30, 60}
//   if mul[c] * sum_of_highest_freq >= all_nonzeros:
//       quant += mul[c] * sum_of_highest_freq / all_nonzeros

// DCT8 flatness: if total_hf_nonzeros < 11: quant += 1

// Large block error/vals ratio:
//   kQuantNormalizer = 2.2942708343284721
//   step = error / (kMul1 * pixels + kMul2 * vals), clamped to [0, 2]
//   quant += step

// Reduce quant in highly active areas:
//   activity = min(min_quadrant_nz / block_area, 15)
//   quant = max(quant - activity, max(4, quant / 2))
//   for Y: thresholds[1..3] += 0.01 * activity
```

---

## 15. `AdjustQuantBias` (Decoder Dequantization Bias)

Source: `quantizer-inl.h:35-67`

```
kDefaultQuantBias[4] = {
    0.94535,   // X: 1.0 - 0.05465
    0.92995,   // Y: 1.0 - 0.07005
    0.95006,   // B: 1.0 - 0.04994
    0.145      // bias numerator
}

if quant == 0:  return 0
if |quant| == 1: return sign(quant) * kDefaultQuantBias[channel]
else:            return quant - 0.145 / quant
```

---

## 16. `x_qm_mul` and `b_qm_mul` Computation

Source: `enc_cache.cc:78-79, enc_frame.cc:647-673`

```
x_qm_scale = 3                              // base
if original_butteraugli_distance > 2.5: x_qm_scale++   // 4
if original_butteraugli_distance > 5.5: x_qm_scale++   // 5
if original_butteraugli_distance > 9.5: x_qm_scale++   // 6

// At effort >= 7: also pixel-based adjustment
x_qm_scale = max(x_qm_scale, 2 + HowMuchIsXChannelPixelized())
b_qm_scale = 2 + HowMuchIsBChannelPixelized()

x_qm_multiplier = 1.25^(x_qm_scale - 2)    // 1.25 at scale=3
b_qm_multiplier = 1.25^(b_qm_scale - 2)     // 1.0 at scale=2
```

---

## 17. `ComputeGlobalScaleAndQuant`

Source: `quantizer.cc:45-76`

```
kGlobalScaleDenom = 65536
kGlobalScaleNumerator = 4096
kQuantFieldTarget = 5

scale = 65536 * (quant_median - quant_median_absd) / 5
scale = clamp(scale, 1, 32768)

// Ensure quant_dc >= 10
scaled_quant_dc = (int)(quant_dc * 4096 * 1.6)
if scale > scaled_quant_dc:
    scale = max(scaled_quant_dc, 1)

global_scale = (int)scale
inv_global_scale = kGlobalScaleNumerator / global_scale
                 = 4096.0 / global_scale

quant_dc = (int)(quant_dc * inv_global_scale + 0.5)
quant_dc = min(quant_dc, 65536)
```

When `quant_median_absd = 0` (flat field or initial call):
```
scale = 65536 * quant_median / 5
```

---

## 18. Butteraugli Quantization Loop

Source: `enc_adaptive_quantization.cc:929-1115`

### Gating

```
Runs only when:
    speed_tier <= kKitten (effort >= 8)
    AND linear image is available
    AND NOT max_error_mode
```

### Iteration counts

```
effort 8 (kKitten):   2 iterations + 1 final
effort 9+ (kTortoise): 4 iterations + 1 final
```

### kPow (adjustment strength per iteration)

```
kPow[8]    = { 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
kPowMod[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
```

### kOriginalComparisonRound = 1

At iteration 1:
```
kInitMul = 0.6
clamp = 0.4 * current_quant + 0.6 * initial_quant
quant = max(quant, clamp)
```

### Per-iteration update

```
diff = tile_distmap / original_butteraugli

if cur_pow > 0 (iterations 0, 1):
    if diff <= 1.0:
        quant *= diff^cur_pow       // quality is good, reduce quant
    else:
        quant *= diff               // quality is bad, increase quant

if cur_pow == 0 (iterations 2+):
    if diff > 1.0:
        quant *= diff               // only increase where bad
    // no change where quality is already good
```

### Deviation bounds

```
initial_qf_ratio = max(initial_qf) / min(initial_qf)
qf_max_deviation_low = sqrt(250 / initial_qf_ratio)
asymmetry = min(2, qf_max_deviation_low)
qf_lower = min(initial_qf) / (asymmetry * qf_max_deviation_low)
qf_higher = max(initial_qf) * (qf_max_deviation_low / asymmetry)
// Ensure qf_higher / qf_lower < 253
```

### TileDistMap

```
kTileNorm = 1.2
kBorderMul = 0.98
kCornerMul = 0.7

// Per-block from per-pixel butteraugli diffmap:
// 16th norm with border/corner weighting
dist_norm = sum(v^16 * weight)
tile_dist = kTileNorm * (dist_norm / pixels)^(1/16)
```

---

## 19. Adaptive Quantization Map (Per-Block)

Source: `enc_adaptive_quantization.cc:315-611`

### Pipeline

```
1. Pre-erosion (4x subsampled Laplacian):
   - Per-pixel: gammac * (pixel - avg_neighbors), squared, clamp at 0.2
   - MaskingSqrt: 0.25 * sqrt(v * sqrt(kMul * 1e8) + 27.506)
   - Subsample 4x: sum of 4 consecutive * 0.25

2. FuzzyErosion (3x3 neighborhood, 4 smallest values):
   - Weights: kMulBase = {0.125, 0.1, 0.09, 0.06}
   - Distance-adjusted: kMulAdd = {0, -0.1, -0.09, -0.06}
   - if d < 2.0: mul = (2.0 - d) / 2.0, weights += mul * kMulAdd
   - Normalize by kTotal = 0.29960

3. ComputeMaskForAcStrategyUse: 1.0 / (out_val + 0.001)

4. PerBlockModulations (per 8x8 block):
   a. ComputeMask (polynomial in 1/v):
      kBase=-0.7647, kMul0=0.8006
      result = kBase + k4/(v^2 + k_off4) + k2/(v + k_off2) + k3/(v^2 + k_off3)

   b. GammaModulation: 0.1006 * log2(avg_ratio_of_derivatives) + val
      kBias=0.16, computed from R=(Y+0.16-X), G=(Y+0.16+X)

   c. HfModulation: sum of clamped 4-connected pixel diffs * -0.38 + 0.42
      kMul_y=-0.38, kOffset=0.42, valmin_y=0.0206

   d. BlueModulation: penalize when B > Y (blue excess)
      kLimit=0.01047, kOffset=0.00320, kMul=0.90591

   e. Final: pow(2, val * 1.4427) * mul + add
      where mul, add depend on scale and dampen
      dampen = clamp(1.0 - (d-2.0)/12.0, 0, 1)
```

---

## 20. Complete Decision Tree Summary

```
AC Strategy Selection (per 64x64 pixel tile):

GATE: effort <= 4 → DCT8 everywhere, no search
GATE: effort >= 5 → full pipeline below

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: For each 8x8 block, find best 8x8-class transform    │
│                                                                 │
│   Candidates (effort-gated):                                    │
│     e5:  DCT8, DCT4X4, DCT2X2, IDENTITY                       │
│     e6+: + DCT4X8, DCT8X4, AFV0, AFV1, AFV2, AFV3             │
│                                                                 │
│   For each candidate:                                           │
│     1. entropy_mul = raw / 0.8                                  │
│     2. if DCT2X2/IDENTITY & d<5: entropy_mul -= 0.4*((5-d)/5)² │
│     3. if !DCT/2X2/ID & d>4: entropy_mul += 0.5 * penalty_mul  │
│     4. cost = EstimateEntropy(strategy, entropy_mul, ...)       │
│     5. Pick minimum cost                                        │
│                                                                 │
│   Store: entropy_estimate[block] = best_cost * mul8x8           │
│          (mul8x8 = 1.0 - 0.4/(d+1.4), always < 1.0)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: Hierarchical merge (bottom-up)                         │
│                                                                 │
│   Pass 1: ALIGNED positions                                     │
│                                                                 │
│   Level 1 (priority 2): At every 2-aligned position             │
│     → FindBestFirstLevelDivisionForSquare(2)                    │
│     → Evaluates: DCT16X8(1.21), DCT8X16(1.21), DCT16X16(1.34) │
│     → Decision: square vs vert-split vs horiz-split vs keep     │
│                                                                 │
│   Level 2 (priority 4): At every 4-aligned position             │
│     → FindBestFirstLevelDivisionForSquare(4)                    │
│     → Evaluates: DCT32X16(1.49), DCT16X32(1.49), DCT32X32(1.48)│
│     → DCT32X32 gated by decoding_speed_tier < 4                │
│                                                                 │
│   Level 3 (priority 6): At every 8-aligned position             │
│     → FindBestFirstLevelDivisionForSquare(8)                    │
│     → Evaluates: DCT64X32(2.25), DCT32X64(2.25), DCT64X64(2.25)│
│                                                                 │
│   Odd-positioned leftovers: TryMergeAcs for individual rects   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (effort >= 6 only)
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: Non-aligned search                                     │
│                                                                 │
│   Pass 2: Non-aligned 16x16 (every odd position)               │
│     → FindBestFirstLevelDivisionForSquare(2)                    │
│                                                                 │
│   Pass 3: Non-aligned 32x32                                     │
│     → step=2 at effort 6-9, step=1 at effort 10+               │
│     → FindBestFirstLevelDivisionForSquare(4)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ POST-PROCESSING                                                 │
│                                                                 │
│   AdjustQuantField: mean-max mixing for multi-block transforms  │
│   SetQuantFieldRect: apply to raw quant field                   │
│   CfL pass 2: recompute with AC strategy + quant (effort >= 5)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 21. Key Implementation Notes

1. **No distance gates on strategies**: libjxl evaluates ALL enabled strategies regardless of
   distance. The only distance-dependent behavior is through `entropy_mul` adjustments and
   the distance-scaled cost model constants.

2. **mul8x8 is always < 1.0**: This gives 8x8 transforms a built-in advantage. Larger
   transforms must overcome this bias. The advantage is strongest at low distances.

3. **entropy_mul normalization asymmetry**: 8x8-class transforms have their entropy_mul
   divided by 0.8 (DCT8 normalizes to 1.0). Larger transforms use their raw values (1.21,
   1.34, etc.) directly. But the 8x8 entropies have ALREADY been multiplied by mul8x8
   (<1.0), so the effective comparison baseline is lower.

4. **Each half decided independently**: In `FindBestFirstLevelDivisionForSquare`, each half
   of a rectangle independently decides whether to use the larger transform or keep existing
   blocks. This is NOT all-or-nothing for rectangles.

5. **Priority prevents overlap**: Higher-priority transforms (later in the hierarchy) cannot
   overwrite blocks already claimed at the same or higher priority. This prevents, e.g.,
   DCT64X32 from conflicting with DCT32X64.

6. **DCT32X8/DCT8X32 are dead code**: Present in the enum but never evaluated by any code path.

7. **DCT128+ are dead code**: Commented out, dequant matrices not even computed for them.
