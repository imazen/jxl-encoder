# Plan: e6/e7 Feature Parity — COMPLETE

## Status: All phases complete (2026-02-15)

All features needed for libjxl effort 6 (Wombat) and effort 7 (Squirrel) VarDCT
feature parity are implemented. The remaining gaps vs e7 are Phase 3 features
(splines, patches) which are intentionally deferred.

## Completed Work

### Step 0: Update RD regression baselines ✓
- Re-recorded baselines at d=0.25, d=0.50, d=1.0, d=2.0, d=3.0
- Tightened margins: size 3% (was 5%), butteraugli 5% (was 10%)
- Commit: `e75f478`

### Phase 1a: Non-aligned AC strategy matching ✓
- Non-aligned 16x16/16x8/8x16 at odd cx/cy positions
- Non-aligned 32x32/32x16/16x32 at non-4-aligned positions (d>=2.0)
- Save/restore prevents single-block re-evaluation from overriding aligned-pass choices
- `favor_single_mul` (libjxl's mul8x8) raises bar for multi-block at low distances
- 0.0-0.7% smaller files, no quality regressions
- Commit: `825e7c5`

### Phase 2a: Pixel-based chromacity adjustment ✓
- Already implemented: `PixelStatsForChromacityAdjustment` in `frame.rs`
- Computed from pre-gaborish XYB, applied to distance params
- x_qm_scale = max(distance_based, 2 + pixel_pixelized)
- b_qm_scale = 2 + pixel_pixelized

### Phase 2b: CfL before AC strategy selection ✓
- Already implemented: CfL map computed before `compute_ac_strategy`
- ytox/ytob values threaded through to entropy estimation

### Phase 2c: Non-aligned 32x32 matching ✓
- Implemented as part of Phase 1a (step=2, d>=2.0 only)

## Phase 3: Deferred (high complexity, niche benefit)

- **Splines:** Parametric curve encoding. High complexity.
- **Patches/dots dictionary:** Repeated pattern detection. High complexity.
