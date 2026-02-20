# Encoder-Side Reconstruction: Rust vs libjxl

Instruction-level comparison of the encoder-side reconstruction pipeline between
`jxl-encoder-rs` and `libjxl`. This pipeline simulates what the decoder produces
from quantized coefficients, enabling the butteraugli quantization loop and EPF
sharpness selection.

**Scope**: Only the reconstruction path used during encoding (butteraugli loop +
AR heuristics). Not the decoder itself, not the forward encoding path.

**Source files compared**:
- Rust: `jxl_encoder/src/vardct/reconstruct.rs`, `gaborish.rs`, `epf.rs`, `butteraugli_loop.rs`
- libjxl: `enc_adaptive_quantization.cc`, `enc_heuristics.cc`, `dec_cache.cc`,
  `render_pipeline/stage_epf.cc`, `enc_gaborish.cc`, `loop_filter.cc`, `epf.h`

---

## Pipeline Overview

Both pipelines follow the same logical sequence:

```
quantized coefficients
    → dequantize AC (with quant bias adjustment)
    → restore CfL (X += ytox*Y, B += ytob*Y)
    → restore LLF from DC (inverse Hadamard/DCT of DC grid)
    → IDCT (strategy-specific)
    → gaborish smooth (3x3 decoder blur, if enabled)
    → EPF (edge-preserving filter, 1-3 steps based on epf_iters)
    → add patches (if present)
    → add splines (if present)
    → XYB → linear RGB
    → butteraugli comparison
```

The difference is in *how* these steps are orchestrated, not *what* they compute.

---

## Architectural Difference: Render Pipeline vs Direct Calls

### libjxl

The butteraugli loop calls `RoundtripImage` (`enc_adaptive_quantization.cc:840-924`),
which:
1. Creates a full decoder `PassesDecoderState`
2. Re-encodes coefficients via `InitializePassesEncoder` (token serialization)
3. Constructs the complete render pipeline via `dec_state->PreparePipeline()`
4. Decodes each group through `DecodeGroupForRoundtrip` → pipeline stages

The pipeline stages run in order (`dec_cache.cc:116-312`):
```
chroma upsampling → gab → EPF0 → EPF1 → EPF2 → patches → splines
→ upsampling → noise → XYB→RGB → CMS/tone mapping
```

This means the butteraugli loop roundtrip goes through the **real decoder path**,
including actual token encoding and decoding. Any bitstream serialization bug would
manifest as a reconstruction error during the loop.

### Rust

The butteraugli loop (`butteraugli_loop.rs:121-390`) calls functions directly:
```rust
// Step 2: Forward quantize
self.transform_and_quantize_into(...);

// Step 3: Reconstruct
let mut planes = reconstruct_xyb(quant_dc, quant_ac, ...);

if self.enable_gaborish {
    gab_smooth(&mut planes, padded_width, padded_height);
}

if current_params.epf_iters > 0 {
    epf::apply_epf(&mut planes, ...);
}

if let Some(pd) = patches_data {
    super::patches::add_patches(&mut planes, padded_width, pd);
}

if let Some(sd) = splines_data {
    super::splines::add_splines(&mut planes, padded_width, width, height, sd);
}

xyb_to_linear_rgb_planar(...);
```

This skips token serialization/deserialization entirely. The dequantization reads
directly from the quantized coefficient arrays (`quant_dc`, `quant_ac`) rather
than going through the bitstream. Functionally equivalent for pixel reconstruction
but does not validate the bitstream roundtrip.

**Impact**: None on reconstruction accuracy. A bitstream encoding bug would be
invisible during the butteraugli loop but would be caught by decoder validation tests.

---

## Dequantization (`reconstruct_xyb`)

### Formula — MATCH

Both use the same dequantization formula:
```
pixel_coeff = adjust_quant_bias(quantized_int, channel) * weight / (qac * qm_mul)
```
where:
- `qac = global_scale/65536 * raw_quant_field[block]`
- `qm_mul`: 1.0 for Y, `1.25^(x_qm_scale - 2)` for X, `1.25^(b_qm_scale - 2)` for B
- `weight = quant_weights[strategy][channel][position]` (parametric, all_default=true)
- `adjust_quant_bias` uses `kDefaultQuantBias` (identical values in both)

### SIMD Dispatch — Difference in Structure

**libjxl**: Dequantization happens inside `DecodeGroupForRoundtrip`, which feeds into the
render pipeline. The IDCT is dispatched via Highway's `HWY_DYNAMIC_DISPATCH`.

**Rust**: `reconstruct_xyb` dispatches to architecture-specific entry points:
```rust
// reconstruct.rs:52-96
#[cfg(target_arch = "x86_64")]
if let Some(token) = jxl_simd::X64V3Token::summon() {
    return reconstruct_xyb_avx2(token, ...);
}
#[cfg(target_arch = "aarch64")]
if let Some(token) = jxl_simd::NeonToken::summon() {
    return reconstruct_xyb_neon(token, ...);
}
reconstruct_xyb_impl(...)  // scalar fallback
```

The `#[archmage::arcane]` annotation propagates `#[target_feature]` so the IDCT
functions within can use SIMD instructions. The underlying math is identical.

### DCT8 Fast Path — Rust-Specific Optimization

Rust has a dedicated DCT8 fast path (`reconstruct.rs:222-301`) that combines
dequantization, CfL, and IDCT for the common DCT8 case using a SIMD kernel:
```rust
jxl_simd::dequant_block_dct8(
    &quant_ac[0][by][bx], &quant_ac[1][by][bx], &quant_ac[2][by][bx],
    weights_x, weights_y, weights_b,
    qac_qm, x_factor, b_factor,
    &mut dq_x, &mut dq_y, &mut dq_b,
);
```

libjxl does not have an equivalent fused kernel — its dequant, CfL, and IDCT
are separate render pipeline stages. The output is numerically equivalent but
may differ at the level of floating-point rounding due to operation reordering.

---

## CfL (Chroma-from-Luma) Restoration

### LLF Position Handling — Different Approach, Same Result

**libjxl**: The decoder's `DequantBlock` applies CfL to **all** coefficient
positions including LLF. Then `LowestFrequenciesFromDC` overwrites the LLF
positions with DC-derived values, discarding the CfL result at those positions.

**Rust**: Explicitly skips LLF positions in the CfL loop (`reconstruct.rs:370-385`):
```rust
for idx in 0..size {
    let y = idx / block_width;
    let x = idx % block_width;
    let is_llf_pos = y < cy && x < cx;

    if !is_llf_pos {
        let y_val = dequant_scratch[1][idx];
        dequant_scratch[0][idx] += x_factor * y_val;
        dequant_scratch[2][idx] += b_factor * y_val;
    }
}
```

**Impact**: None. Both produce the same output because the LLF values are
determined by `restore_llf_from_dc`, not by CfL. The Rust approach avoids
redundant computation.

### DC CfL — MATCH

Both use `dc_cfl_factor = 0.5` for the B channel, 0.0 for X and Y:
```rust
// reconstruct.rs:260
let dc_cfl_factor_b: f32 = 0.5;
```

This is the fixed DC-level CfL, separate from the per-tile AC-level CfL factors
(`ytox`, `ytob`).

---

## LLF Restoration from DC

### Algorithm — MATCH

Both use the same inverse of the decoder's `LowestFrequenciesFromDC`:
- **DCT8/DCT4x4/DCT4x8/etc. (single block)**: LLF = dequantized DC at position [0]
- **DCT16x8/DCT8x16**: 2-point inverse Hadamard with `DCT_RESAMPLE_SCALE_16_TO_2`
- **DCT16x16**: 2x2 inverse Hadamard with `DCT_RESAMPLE_SCALE_16_TO_2`
- **DCT32x32**: 4x4 forward DCT of DC grid, divided by `DCT_RESAMPLE_SCALE_32_TO_4 * 16`
- **DCT64x64**: 8x8 forward DCT of DC grid, divided by `DCT_RESAMPLE_SCALE_64_TO_8`
- **DCT32x16/DCT16x32**: Mixed-size forward DCT, appropriate scale factors
- **DCT64x32/DCT32x64**: Mixed-size forward DCT with `dct1d_8` gain correction

The resample scale constants, Hadamard transforms, and gain factors are identical.

---

## IDCT

### Strategy Coverage — MATCH

Both support all 19 strategies through IDCT:

| Strategy | libjxl Function | Rust Function |
|----------|----------------|---------------|
| DCT8 | `ComputeScaledIDCT<8,8>` | `idct_8x8` |
| DCT4x4 | Sub-block IDCT with DC combining | `idct_for_strategy(DCT4X4)` |
| DCT4x8 | Sub-block IDCT with DC combining | `idct_for_strategy(DCT4X8)` |
| DCT8x4 | Sub-block IDCT with DC combining | `idct_for_strategy(DCT8X4)` |
| DCT16x8 | `ComputeScaledIDCT<16,8>` | `idct_16x8` |
| DCT8x16 | `ComputeScaledIDCT<8,16>` | `idct_8x16` |
| DCT16x16 | `ComputeScaledIDCT<16,16>` | `idct_16x16` |
| DCT32x32 | `ComputeScaledIDCT<32,32>` | `idct_32x32` |
| DCT32x16 | `ComputeScaledIDCT<32,16>` | `idct_32x16` |
| DCT16x32 | `ComputeScaledIDCT<16,32>` | `idct_16x32` |
| DCT64x64 | `ComputeScaledIDCT<64,64>` | `idct_64x64` |
| DCT64x32 | `ComputeScaledIDCT<64,32>` | `idct_64x32` |
| DCT32x64 | `ComputeScaledIDCT<32,64>` | `idct_32x64` |
| IDENTITY | Identity inverse | `inverse_identity_transform` |
| DCT2x2 | 2x2 inverse | `inverse_dct2x2_transform` |
| AFV0-3 | Corner DCT inverse | `inverse_afv_transform` |

### Transpose Handling — Minor Difference

**Rust**: Only DCT16X8 needs an explicit transpose before IDCT (`reconstruct.rs:397`):
```rust
let needs_transpose = raw_strategy == RAW_STRATEGY_DCT16X8;
```

The encoder stores coefficients in "post-swap layout" (cx >= cy). Most Rust IDCT
functions expect this layout directly. `idct_16x8` uses gather/scatter and expects
natural pixel layout (16 rows, 8 cols), requiring a transpose from the 8-col-first
storage.

`idct_32x16` and `idct_64x32` handle their internal transposes — no external
transpose needed.

**libjxl**: The render pipeline handles layout transformation through its
row-based processing model, with column strides managed by `RenderPipelineInput`.

**Impact**: None on output values. Internal implementation detail.

---

## Gaborish Smooth (Decoder-Side 3x3 Blur)

### Constants — MATCH

```
w1_base = 0.104699568 * 1.1 = 0.115169525
w2_base = 0.055680538 * 1.1 = 0.061248592
```

Source:
- libjxl `loop_filter.cc:32-50`: `visitor->F16(1.1 * 0.104699568f, &gab_x_weight1)`
- Rust `reconstruct.rs:1024-1025`: `let w1_base = 0.104_699_57_f32 * 1.1;`

All three channels use the same weights (no per-channel variation in either codebase).

### Normalization — MATCH

```
div = 1.0 + 4.0 * (w1 + w2)
w_center = 1.0 / div
w1_normalized = w1_base / div
w2_normalized = w2_base / div
```

### Implementation

**libjxl**: `GaborishStage` in the render pipeline, row-based with Highway SIMD,
uses `WeightsSymmetric3` kernel infrastructure.

**Rust**: `gab_smooth` in `reconstruct.rs:1022-1038` dispatches to
`jxl_simd::gab_smooth_channel` per channel with a shared scratch buffer.

Both use edge replication at boundaries.

---

## Gaborish Inverse (Encoder-Side 5x5 Sharpening)

### Constants — MATCH

```rust
// Both codebases, identical:
const K_GABORISH: [f64; 5] = [
    -0.09495815671340026,   // orthogonal dist 1
    -0.041031725066768575,  // diagonal dist sqrt(2)
     0.013710004822696948,  // orthogonal dist 2
     0.006510206083837737,  // knight's move
    -0.0014789063378272242, // corner dist 2*sqrt(2)
];
```

Source:
- libjxl `enc_gaborish.cc:30-33`
- Rust `gaborish.rs:30-36`

### Channel Multipliers — MATCH

Both use `mul = [1.0, 1.0, 1.0]` for all three XYB channels.

Source:
- libjxl `enc_heuristics.cc:1137-1140`: `GaborishInverse(opsin, rect, {1.0, 1.0, 1.0}, pool)`
- Rust `gaborish.rs:88-91`: `apply_channel(..., 1.0)` for each

---

## Edge-Preserving Filter (EPF)

### Constants — ALL MATCH

| Constant | libjxl | Rust | Source |
|----------|--------|------|--------|
| `kInvSigmaNum` | `-1.1715728752538099024f` | `-1.171_572_9` | `epf.h:19` / `epf.rs:22` |
| `epf_quant_mul` | `0.46f` | `0.46` | `loop_filter.cc` / `epf.rs:25` |
| `epf_pass0_sigma_scale` | `0.9f` | `0.9` | `loop_filter.cc` / `epf.rs:26` |
| `epf_pass2_sigma_scale` | `6.5f` | `6.5` | `loop_filter.cc` / `epf.rs:27` |
| `epf_border_sad_mul` | `2.0/3.0` | `2.0/3.0` | `loop_filter.cc` / `epf.rs:28` |
| Channel scales | `[40.0, 5.0, 3.5]` | `[40.0, 5.0, 3.5]` | `loop_filter.cc` / `epf.rs:31` |
| Sharp LUT | `i / (entries-1)` | `[0, 1/7, ..., 1]` | `loop_filter.cc` / `epf.rs:34-43` |

### Sigma Computation — MATCH

```
sigma_quant = epf_quant_mul / (quant_scale * raw_quant * kInvSigmaNum)
sigma = sigma_quant * epf_sharp_lut[sharpness]
inv_sigma = 1.0 / sigma
```

### Weight Function — MATCH

```
w = max(0, sad * inv_sigma + 1)
```

### Step Definitions — MATCH

| Step | Kernel | SAD Pattern | Sigma Scale | Border |
|------|--------|------------|-------------|--------|
| Step 0 / EPF0 | 12-point 5x5 plus | 3x3 plus (5 SAD points) | `pass0_sigma_scale * 1.65` | 3 |
| Step 1 / EPF1 | 4-point 3x3 cross | 3x3 plus (5 SAD points) | `1.65` | 2 |
| Step 2 / EPF2 | 4-point 3x3 cross | single-pixel (1 SAD point) | `pass2_sigma_scale * 1.65` | 1 |

### Step Activation — DIFFERENCE (BUG at epf_iters=1)

**libjxl** (`dec_cache.cc:155-165`):
```cpp
if (lf.epf_iters >= 3) AddStage(EPF0);  // heaviest
if (lf.epf_iters >= 1) AddStage(EPF1);  // medium — runs ALWAYS when EPF is on
if (lf.epf_iters >= 2) AddStage(EPF2);  // lightest
```

**Rust** (`epf.rs:251-307`):
```rust
if epf_iters >= 3 { epf_step0(...); }  // heaviest — MATCH
if epf_iters >= 2 { epf_step1(...); }  // medium — should be >= 1
// step2 always runs when epf_iters >= 1   // lightest — should be >= 2
```

**Behavior at each `epf_iters` value:**

| `epf_iters` | libjxl | Rust | Match? |
|-------------|--------|------|--------|
| 0 | nothing | nothing | YES |
| 1 | EPF1 (medium, multi-point SAD) | step2 (lightest, single-pixel SAD) | **NO** |
| 2 | EPF1 + EPF2 (medium + light) | step1 + step2 (medium + light) | YES |
| 3 | EPF0 + EPF1 + EPF2 | step0 + step1 + step2 | YES |

**The activation conditions for steps 1 and 2 are swapped.** At `epf_iters=1`, Rust
applies the lightest filter (single-pixel SAD comparison, 3x3 effective) instead of the
medium filter (multi-point SAD, 5x5 effective).

**Practical impact**: Low. The encoder typically uses `epf_iters=2` (default for most
distances) or `epf_iters=3`. The `epf_iters=1` case only occurs at very high distances
where EPF quality matters least.

### Step 0 SIMD — Difference

**libjxl**: All three EPF stages use Highway SIMD with `HWY_DYNAMIC_DISPATCH`.

**Rust**: Step 0 is pure scalar (`epf.rs:141-218` — nested loops with `get_pixel`
coordinate clamping). Steps 1 and 2 dispatch to `jxl_simd::epf_step1` /
`jxl_simd::epf_step2` with AVX2/NEON paths.

**Impact**: Performance only (step 0 is ~3x slower than it could be with SIMD).
No output difference.

### Border Handling — Different Approach, Same Semantic

**libjxl**: The render pipeline infrastructure handles borders via stage padding
requirements (`Settings::Symmetric(shift=0, border=N)` for border N). The pipeline
ensures rows beyond the image boundary are available as mirrored/replicated rows.

**Rust**: Uses per-pixel coordinate clamping (`epf.rs:86-92`):
```rust
fn get_pixel(plane: &[f32], x: isize, y: isize, width: usize, height: usize) -> f32 {
    let cx = x.clamp(0, width as isize - 1) as usize;
    let cy = y.clamp(0, height as isize - 1) as usize;
    plane[cy * width + cx]
}
```

Both implement edge replication (nearest border pixel). For interior pixels the
output is identical. At image edges, any floating-point difference is limited to
the EPF border region (1-3 pixels deep).

---

## EPF Sharpness Selection (AR Heuristics)

### Candidates — MATCH

```
distance > 4.5: candidates = [0, 4]
distance <= 4.5: candidates = [0, 2, 7]
```

Source:
- libjxl `enc_heuristics.cc:912-920`
- Rust `epf.rs:422-426`

### Reconstruction Per Candidate

**libjxl**: For each candidate, fills the entire `epf_sharpness` image with the
candidate value, then calls `ReconstructImage` which runs the full render pipeline
(`enc_heuristics.cc:929-943`):
```cpp
for (uint8_t val : epf_steps) {
    FillPlane(val, &epf_sharpness, Rect(epf_sharpness));
    Image3F decoded = ReconstructImage(frame_header, shared, enc_state->coeffs, pool);
    // compute per-block error...
}
```

**Rust**: Reconstructs the base image (dequant + CfL + LLF + IDCT + gab) ONCE,
then clones and applies EPF separately per candidate (`epf.rs:428-493`):
```rust
let mut base_recon = reconstruct_xyb(...);
if enable_gaborish { gab_smooth(&mut base_recon, ...); }

for &sharpness_val in candidates {
    let mut recon = base_recon.clone();
    uniform_sharpness.fill(sharpness_val);
    let inv_sigma = compute_inv_sigma_map(..., &uniform_sharpness, ...);
    apply_epf_with_scratch(&mut recon, &inv_sigma, ...);
    let errors = compute_block_l2_errors(...);
    error_maps.push(errors);
}
```

**Behavioral difference**: None. The dequant→CfL→LLF→IDCT→gab steps produce
identical output regardless of sharpness (sharpness only affects EPF). The Rust
approach is an optimization that avoids redundant reconstruction.

### Block L2 Error Metric — MATCH

Channel weights:
```
kW = [12.339445295782363, 1.0, 0.2]   // X, Y, B
```

Source:
- libjxl `enc_heuristics.cc:858-891` (line 885: `constexpr double kW[] = {12.339445295782363, 1.0, 0.2}`)
- Rust `jxl_simd/src/block_l2.rs`: `CHANNEL_WEIGHTS = [12.339_445, 1.0, 0.2]`

The Rust value is truncated to f32 precision (12.339445 vs 12.339445295782363).
At f32 these are the same bit pattern.

### Two-Pass Selection Algorithm — MATCH

**Pass 1: Greedy with neighbor preference**

For each block, find the candidate with lowest error (with `K_FAVOR_NO_SMOOTHING = 0.99`
bias for sharpness=0). If a neighbor's sharpness produced lower error on that neighbor's
block, prefer the neighbor's value. Accumulate context histograms.

**Pass 2: Context-based re-weighting**

Compute per-context multipliers using:
```
c3 = max(c3clamp, c3base^distance)
mul = 1.0 / (1.0 + c5 * ln(1 + ratio) / distance)
if candidate == 0: mul *= c3
```

### AR Heuristics Constants — MATCH

| Constant | libjxl | Rust |
|----------|--------|------|
| `c3base` | `0.98017198824148288` | `0.980_172` |
| `c3clamp` | `0.85970338919928291` | `0.859_703_4` |
| `c5` | `0.1087690359555803` | `0.108_769_04` |
| `kFavorNoSmoothing` | `0.99` | `0.99` |
| Default sharpness | `4` | `4` |

Rust values are truncated to f32 precision. The first significant difference
is at the 7th decimal place, well below f32 precision (7 significant digits).

### Integer Division for Context Ratios — MATCH

**Critical parity detail**: libjxl uses `size_t / size_t` (integer division) for
`ctx_histo[val] / totals[context]`. For count < total (the common case), this
yields 0, making `log1p(0) = 0` and the entropy-based multiplier = 1.0. The
context refinement is effectively a no-op for most blocks — only the c3 bias
for sharpness=0 has real effect.

Rust matches this exactly (`epf.rs:594-595`):
```rust
// Integer division to match libjxl's size_t / size_t
let ratio = count / totals[ctx];
```

This was previously an f32 division bug, fixed in commit `ce7f0f9`.

---

## Butteraugli Quantization Loop

### Activation — MATCH

Both gate the butteraugli loop at effort >= 8 (speed_tier <= kKitten):
- libjxl `enc_adaptive_quantization.cc:1282`: `cparams.speed_tier <= SpeedTier::kKitten`
- Rust: `butteraugli_iters > 0` only when effort >= 8

### Iteration Count — MATCH

| Effort | libjxl | Rust |
|--------|--------|------|
| 8 (kKitten) | `kDefaultButteraugliIters = 2` | 2 |
| 9+ (kTortoise) | `kMaxButteraugliIters = 4` | 4 |

### EPF Sharpness During Loop — MATCH

**Both** use fixed sharpness=4 during the butteraugli loop:
- libjxl `enc_heuristics.cc:1248-1249`:
  ```cpp
  FillPlane(static_cast<uint8_t>(4), &epf_sharpness, Rect(epf_sharpness));
  JXL_RETURN_IF_ERROR(FindBestQuantizer(...));
  ```
- Rust `butteraugli_loop.rs:108`:
  ```rust
  let sharpness = vec![4u8; num_blocks];
  ```

The actual per-block EPF sharpness is computed AFTER the butteraugli loop finishes:
- libjxl: `ComputeARHeuristics` called in `enc_frame.cc:1132`, after `InitializePassesEncoder`
- Rust: `compute_epf_sharpness` called separately after the butteraugli loop

### Loop Body Steps — MATCH

Both implement the same sequence per iteration:

1. **SetQuantField**: Recompute `global_scale` from float quant field (median/MAD),
   convert float → u8 via `round(qf * inv_global_scale + 0.5)`
2. **Encode + Reconstruct**: Forward quantize → decoder-side reconstruction
3. **Butteraugli comparison**: Compare reconstructed linear RGB against original
4. **TileDistMap**: 16th-power norm of per-pixel butteraugli distortion per block
5. **Original comparison round** (iter == 1): Constrain toward initial field
6. **Quant field adjustment**: Scale bad blocks up, optionally reduce good blocks

### Butteraugli Comparator — Different Implementation

**libjxl**: Uses internal `JxlButteraugliComparator` wrapping `ButteraugliComparator`
(`enc_butteraugli_comparator.h`). The reference image is set once as an `Image3F`
(linear sRGB float planes). Comparison produces a `diffmap` (per-pixel scores) and
a scalar `score`.

**Rust**: Uses the external `butteraugli` crate (`butteraugli_loop.rs:75-86`):
```rust
let reference = butteraugli::ButteraugliReference::new_linear(
    linear_rgb, width, height, butteraugli_params,
)?;
let result = reference.compare_linear_planar(&recon_r, &recon_g, &recon_b, padded_width)?;
```

Both implement the same butteraugli algorithm. The external crate is a Rust port
of the same code. Floating-point divergence between the two implementations may
cause slightly different `diffmap` values, leading to slightly different quant
field adjustments per iteration.

**Intensity target**: Both use 80.0 (Rust: `butteraugli_params.with_intensity_target(80.0)`).

### TileDistMap — Minor Precision Difference

**libjxl**: Accumulates the 16th-power norm in `float` (f32):
```cpp
float v = row_distmap[px];
v *= v; v *= v; v *= v; v *= v;  // v^16
dist_norm += v;
```

**Rust**: Accumulates in `f64`, then casts to `f32` at the end
(`butteraugli_loop.rs:231-247`):
```rust
let v = diffmap_buf[py * width + px] as f64;
let v2 = v * v; let v4 = v2 * v2;
let v8 = v4 * v4; let v16 = v8 * v8;
dist_norm += v16;
```
```rust
let td = K_TILE_NORM * (dist_norm / pixels).sqrt().sqrt().sqrt().sqrt() as f32;
```

**Impact**: The 16th power amplifies small differences dramatically. A pixel value
of 2.0 becomes 65536.0 in the norm. f64 accumulation avoids overflow on blocks
with high distortion (distortion > ~15 per pixel would overflow f32 at the 16th power).
The final result is cast to f32, so the difference is only in the accumulation
intermediate values.

### K_TILE_NORM — MATCH

Both use `K_TILE_NORM = 1.2`.

### Deviation Bounds — MATCH

```
initial_qf_ratio = max(initial_qf) / min(initial_qf)
qf_max_deviation_low = sqrt(250 / initial_qf_ratio)
asymmetry = min(2, qf_max_deviation_low)
qf_lower = min(initial_qf) / (asymmetry * qf_max_deviation_low)
qf_higher = max(initial_qf) * (qf_max_deviation_low / asymmetry)
```

### Original Comparison Round — MATCH

At iteration 1 (`K_ORIGINAL_COMPARISON_ROUND = 1`):
```
if cur_qf < clamp_val:
    clamp_val = (1 - K_INIT_MUL) * cur_qf + K_INIT_MUL * init_qf
    qf = clamp(clamp_val, qf_lower, qf_higher)
```

Where `K_INIT_MUL = 0.6`.

### Pow Schedule — MATCH

```
iter 0, 1: pow = 0.2
iter 2+: pow = 0.0 (only fix bad blocks)
```

The pow modifier is `kPowMod = 0` for all iterations (distance-independent).

### Minimum Step Check — MATCH

When a bad block's quant field adjustment rounds to the same integer value,
both add one quantizer step:
```
if floor(old * inv_global_scale + 0.5) == floor(new * inv_global_scale + 0.5):
    qf = old + quantizer_scale
```

---

## XYB → Linear RGB Conversion

### Approach

**libjxl**: The `XYBStage` in the render pipeline converts XYB to linear sRGB,
including the inverse opsin absorbance matrix, cube root inverse, and bias subtraction.

**Rust**: `xyb_to_linear_rgb_planar` (`reconstruct.rs:1066-1076`) dispatches to
`jxl_simd::xyb_to_linear_rgb_planar`, which implements the same inverse transform.

### Constants

The opsin absorbance matrix, bias values (`kOpsinAbsorbanceBias`), and the
cube-root gamma inverse are all defined by the JXL specification. Both
implementations use the same constants.

---

## Patches and Splines in Reconstruction

**libjxl**: Patches and splines are render pipeline stages that run after EPF
and before XYB conversion.

**Rust**: Added as explicit function calls after EPF (`butteraugli_loop.rs:182-188`):
```rust
if let Some(pd) = patches_data {
    super::patches::add_patches(&mut planes, padded_width, pd);
}
if let Some(sd) = splines_data {
    super::splines::add_splines(&mut planes, padded_width, width, height, sd);
}
```

The XYB-domain patch/spline application is the same operation: add pre-computed
XYB values to the reconstructed planes at specific pixel locations.

---

## FindBestQuantizationMaxError — Not Implemented

**libjxl** has `FindBestQuantizationMaxError` (`enc_adaptive_quantization.cc:1117`),
which targets per-block maximum RGB error instead of butteraugli distance. This is
used for LfFrame intermediate levels (`cparams.max_error_mode = true`).

**Rust**: LfFrame encoding does not use max error mode. The LfFrame distance is
simply scaled (`distance * 0.02` for final level, `distance * 0.1` for intermediate)
without per-block RGB error targeting.

**Impact**: LfFrame intermediate levels may have different quality distribution
(libjxl bounds max per-block error; Rust uses uniform distance scaling). Since
LfFrame is used for progressive display at very low resolution, the visual
difference is negligible.

---

## Summary of Differences

### Actual Behavioral Differences

1. **EPF step activation at `epf_iters=1`**: Rust runs the lightest filter (step2,
   single-pixel SAD) instead of the medium filter (EPF1, multi-point SAD). Steps 1/2
   activation thresholds are swapped. No impact at `epf_iters >= 2`.

2. **TileDistMap f64 accumulation**: Rust accumulates the 16th-power norm in f64
   instead of f32. Marginally more precise; prevents overflow at extreme distortion
   values.

3. **External butteraugli crate**: Different implementation of the same algorithm.
   Floating-point differences in the diffmap may cause slightly different quant
   field evolution during the loop. Both converge to similar quality.

4. **No max-error mode for LfFrame**: Intermediate LfFrame levels use uniform
   distance scaling instead of per-block max RGB error targeting.

### Implementation Differences (Same Output)

5. **Direct function calls vs render pipeline**: Rust skips token serialization
   during the butteraugli loop roundtrip.

6. **CfL skips LLF explicitly**: Rust doesn't apply CfL to LLF positions (libjxl
   applies then overwrites). Same result.

7. **Base reconstruction reuse in AR heuristics**: Rust reconstructs once and
   clones for each sharpness candidate. libjxl reconstructs fully per candidate.

8. **EPF step 0 scalar-only**: Performance difference, not output difference.

9. **Border handling via clamping vs pipeline padding**: Same edge replication
   semantic.

10. **DCT8 fused dequant+CfL+IDCT kernel**: Rust optimization for the most
    common strategy. May have f32 rounding differences from operation reordering.

### Confirmed Matches

All constants verified identical between codebases:
- Gaborish kernel weights (encoder 5x5 and decoder 3x3)
- EPF sigma parameters, channel scales, sharp LUT
- AR heuristics constants (c3base, c3clamp, c5, kFavorNoSmoothing)
- Integer division parity for context ratios
- Block L2 distance channel weights
- Butteraugli loop parameters (iters, pow schedule, deviation bounds, kInitMul)
- Fixed sharpness=4 during butteraugli loop
- DC CfL factor (0.5 for B channel)
- Quant bias adjustment (kDefaultQuantBias)
- All resample scale constants for LLF restoration
