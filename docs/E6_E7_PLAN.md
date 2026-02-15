# Plan: e6/e7 Feature Parity

## Goal

Reach full libjxl effort 6 (Wombat) and effort 7 (Squirrel) VarDCT feature parity.
We already have all the AC strategies. The remaining gaps are cost model refinements
and perceptual tuning that improve compression efficiency without adding new transforms.

## Prerequisites: Regression Locks

Before any feature work, update and tighten the RD regression baselines to lock current
quality. This prevents silent regressions as we add features.

### Step 0: Update RD regression baselines

1. Fix the `test_rd_regression` CID22 download path (some images 404 at d=0.50).
2. Re-record baselines at d=0.25, d=0.50, d=2.0, d=3.0 after the AFV fix.
3. Add d=1.0 baselines (our most common operating point, currently untested).
4. Tighten margins: size 3% (was 5%), butteraugli 5% (was 10%).
5. Run both tests, verify green, commit.

Each subsequent feature must pass these locks before merge.

## Feature Plan

### Phase 1: e6 (Wombat) parity

#### 1a. Non-aligned AC strategy matching

**What:** libjxl at e6+ tries 16x8, 8x16, 16x16 blocks at non-2-aligned positions
(odd cx/cy within the 64x64 tile). Our search only evaluates 2-aligned positions.

**Where:** `enc_ac_strategy.cc:1030-1044` — after the aligned search, iterates over
`(cy, cx)` where `(cy | cx) % 2 != 0` and calls `FindBestFirstLevelDivisionForSquare`
with size=2 (16x16 search).

**Impact:** Medium. Catches cases where the optimal 16x16 block straddles a 2-aligned
boundary. Most benefit on images with strong diagonal or off-grid structure.

**Implementation:**
- In `ac_strategy_search.rs`, after the main 2-aligned loop, add a second pass over
  non-2-aligned positions within each 64x64 tile.
- Only evaluate 16x8/8x16/16x16 (not 32x32+) at these positions, matching libjxl.
- Guard behind `distance >= some_threshold` if needed to avoid overhead at low distance.

**Files:** `jxl_encoder/src/vardct/ac_strategy_search.rs`

### Phase 2: e7 (Squirrel) parity

#### 2a. Pixel-based chromacity adjustment (x_qm_scale, b_qm_scale)

**What:** libjxl's `PixelStatsForChromacityAdjustment` scans the opsin image to detect
how much the X and B channels are "pixelized" (contain per-pixel high-frequency chroma
detail). Sets `frame_header.x_qm_scale` and `frame_header.b_qm_scale` (range 0-6,
default 2) to control how aggressively chroma channels are quantized relative to luma.

**Where:** `enc_frame.cc:662-673` — guarded by `speed_tier <= kSquirrel`.

**Impact:** Small but measurable. Adjusts chroma quantization per-image. Images with
fine color detail (e.g., colorful textures, CG renders) get less chroma quantization.
Images with smooth color (landscapes, portraits) get more aggressive chroma compression.

**Implementation:**
- Port `PixelStatsForChromacityAdjustment` from `enc_frame.cc`.
- Compute after XYB conversion, before frame header finalization.
- Set `x_qm_scale` and `b_qm_scale` in the frame header.
- Currently we write these as defaults (2); this just makes them content-adaptive.

**Files:** `jxl_encoder/src/vardct/encoder.rs`, `jxl_encoder/src/frame_header.rs`

#### 2b. CfL map before AC strategy selection

**What:** libjxl at e7 computes a preliminary CfL (chroma-from-luma) map BEFORE choosing
AC strategies. This means the strategy cost evaluation uses accurate chroma correlation
factors, improving block size decisions for images with strong chroma patterns.

**Where:** `enc_heuristics.cc:1169-1174` — guarded by `speed_tier <= kSquirrel`.
Calls `cfl_heuristics.ComputeTile()` with no AC strategy / quant field (preliminary).
Then after strategy selection, CfL is recomputed with full context at Hare and below.

**Impact:** Small. Strategy selection already works well with post-hoc CfL. The benefit
is that strategy cost estimates are slightly more accurate when chroma is decorrelated
first. Mostly helps images with strong chroma gradients.

**Implementation:**
- In the encode pipeline, compute CfL before `find_best_ac_strategy`.
- Pass the CfL map to the strategy search so cost evaluation uses decorrelated chroma.
- After strategy selection, recompute CfL with the chosen strategies (already done).
- This is a pipeline reordering, not new algorithm code.

**Files:** `jxl_encoder/src/vardct/encoder.rs`, `jxl_encoder/src/vardct/ac_strategy_search.rs`

#### 2c. Non-aligned 32x32 matching (step=2)

**What:** libjxl at e7 extends non-aligned matching to 32x32 blocks (and 16x32/32x16).
At e7, uses step=2 (every other position). At e9 (Tortoise), step=1 (all positions).

**Where:** `enc_ac_strategy.cc:1046-1054` — after non-aligned 16x16 matching.

**Impact:** Small. Catches optimal 32x32 blocks at non-4-aligned positions. Only relevant
at d>=2.0 where 32x32 blocks are enabled.

**Implementation:**
- Extend the non-aligned pass from Phase 1a to also try 32x32 at step=2.
- Only at d>=2.0 (where DCT32x32 is enabled).

**Files:** `jxl_encoder/src/vardct/ac_strategy_search.rs`

### Phase 3: Deferred (high complexity, niche benefit)

These are e7 features we intentionally skip for now:

- **Splines:** Parametric curve encoding. High complexity (detection, fitting, subtraction,
  residual encoding). Benefits specific content (power lines, horizons, smooth gradients).
  Not worth the implementation cost for general-purpose encoding.

- **Patches/dots dictionary:** Repeated pattern detection. High complexity (search, matching,
  dictionary encoding). Benefits screenshots and UI content. Would need a separate content
  detection heuristic to avoid overhead on photos.

## Verification

After each phase:
1. `cargo test --lib` — all unit tests pass
2. `cargo test --release -p jxl-encoder --test clic2025 test_rd_regression -- --ignored` — regression locks hold
3. `cargo test --release -p jxl-encoder --test clic2025 test_rd_regression_high_distance -- --ignored` — high-distance locks hold
4. `cargo clippy --all-targets` — clean
5. Verify decode with jxl-rs and djxl on at least one image per distance level

## Expected Impact

| Feature | Expected size reduction | Expected quality change |
|---------|----------------------|----------------------|
| Non-aligned 16x16 matching | 0.5-2% at d<=1.0 | Neutral to +0.1 SSIM2 |
| Pixel chromacity adjustment | 0-0.5% | +0.1-0.3 SSIM2 on colorful images |
| CfL before strategy | 0-0.5% | Neutral |
| Non-aligned 32x32 matching | 0-1% at d>=2.0 | Neutral |

Combined: ~1-3% size reduction at equal quality, small quality improvements on specific content.
