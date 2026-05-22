# Tuning sweep queue

Top-N sweep targets ranked by EV, drawing from:
- W44-210-E `High-EV deviation candidates` (8 explicit, ranked by audit)
- W44-210-D high-uncertainty edges (distance-aware bands not yet bisected
  on the full ledger)
- W44-210-A refactor candidates (consolidate duplicate thresholds — one
  parameter to sweep instead of N)
- W44-210-C flagged "never re-bisected since W44-XX" entries
- W44-204 residual cluster analysis (which images STILL Pareto-lose)

Each entry: const(s), expected EV, sweep grid, risk, prerequisite
chunks. Maintained per the
[`TUNING_RELATIONS.md`](../../docs/TUNING_RELATIONS.md) mandatory rule.

---

## P1: ablate `K_INFO_LOSS_MULTIPLIER2` + `K_COST2` (W44-210-E #1)

- **Consts**:
  - `K_INFO_LOSS_MULTIPLIER2 = 50.4684` (`vardct/ac_strategy.rs:925`)
  - `K_COST2 = 4.462815` (`vardct/ac_strategy.rs:926`)
- **Hypothesis**: dead code. Appears in `info_loss_score = k_info_loss_mul * info_loss_sum + K_INFO_LOSS_MULTIPLIER2 * infoloss2`,
  but `infoloss2` does not exist in libjxl source (`enc_ac_strategy.cc`).
  W44-210-E flagged as OURS-ONLY math, never bisected.
- **Sweep grid**: 2-point A/B (default vs `K_INFO_LOSS_MULTIPLIER2 = 0.0`,
  `K_COST2 = 0.0`)
- **Expected EV**: HIGH
  - If byte-identical: confirms dead code (refactor: remove the term),
    unblocks understanding of the `infoloss2` accumulation site
  - If non-zero delta: exposes hidden mechanism, demands proper sweep
    over a real range
- **Risk**: LOW (binary ablation; either no effect or measurable delta
  on full ledger)
- **Prerequisite**: none (can run today)
- **Bench**: paired-A/B across the 595-cell ledger + 36 hash-locks
- **Bucket**: NOT-IN-LIBJXL, room=high
- **Owner reference**: W44-210-E Section 1.5 + High-EV Deviation #1

## P2: Legacy `profile.k8x8/k16x8/k16x16/k4x8/k4x4` re-bisect or remove (W44-210-E #2)

- **Consts**: 5 `EffortProfile` `(mul1, mul2, base)` triples (coef-domain
  legacy path)
- **Hypothesis**: only fires when `pixel_domain_loss=false` (off by
  default since W22-1+). The `0.75` baked-in factor in
  `k8x8 = (-0.55*0.75, 1.0735758*0.75, 1.4)` is OURS, undocumented.
  Either (a) the legacy path is dead and should be removed, or (b) the
  constants were never re-bisected after pixel-domain shipped.
- **Sweep grid**:
  - Phase 1: audit whether `pixel_domain_loss=false` is reachable in any
    production strategy (Libjxl / LeanFaster / Zenjxl / Aggressive). If
    NO, remove the path entirely.
  - Phase 2 (if reachable): full 21-q × 4-size × 4-mode grid bisect on
    the 5 triples, focused on `lossy_experimental` profile.
- **Expected EV**: MED (path likely dead — high refactor value)
- **Risk**: MED (refactor touches `EffortProfile` API)
- **Prerequisite**: audit reachability of `pixel_domain_loss=false`
- **Bucket**: DEVIATED, room=medium

## P3: variant-Z `dct16x16 = 1.27` re-bisect (W44-210-E #6)

- **Consts**: variant Z `dct16x16` field across `_z` / `_z_high_colour` /
  `_z_low_colour` / 3× d_high tables (8 tables total in
  `EntropyMulTable::high_d_photo_smooth_suppressed_z*` family)
- **Hypothesis**: like `dct32x32`, may have post-W44-148 era better
  value. The 1.27 value comes from the W44-28 sweep (~5.2% cheaper than
  reference 1.34) on a narrower corpus than the current ledger. Stayed
  constant across W44-148/154/156 cycles even though `dct32x32` was
  bisected multiple times.
- **Sweep grid**: 5-7 values × W44-204 residual cluster (3637739,
  297394, 7062219, 1418519, 1420710, 1531677)
  - Values: {1.20, 1.22, 1.24, 1.27, 1.30, 1.34}
- **Expected EV**: MED-HIGH (similar to W44-148 dct32x32 chain — both
  values control the same DCT16/32 mix)
- **Risk**: MED (could regress same way W44-148 initially did at 1.20)
- **Prerequisite**: W44-211+ tuning extraction infrastructure (so A/B
  can flip table values without rebuild)
- **Bucket**: NOT-IN-LIBJXL, room=medium
- **DO NOT**: bisect outside [1.20, 1.34] — outside reference value
  loses W44-28 motivation

## P4: `tree_threshold_base = 75.0 + 14.0 * speed_tier` (W44-210-E #3)

- **Consts**: ID3-style heuristic in tree learning; slope (14.0) and
  intercept (75.0) never directly swept
- **Hypothesis**: no direct libjxl analog. Both terms shipped at
  W36-1 era; corpus + ML data pipeline has evolved substantially
  since.
- **Sweep grid**: 2D bisect (intercept × slope) on the full ML data
  pipeline discipline (21-quality × 4-size × 4-mode grid per CLAUDE.md).
  Suggested:
  - intercept ∈ {50, 75, 100, 125}
  - slope ∈ {0, 7, 14, 21, 28}
  - = 20-cell grid × 4-mode × 4-size × 21-q = 6720 cells
- **Expected EV**: MED (bytes-modest, perf-positive if higher threshold
  reduces tree-learning iters)
- **Risk**: MED (lossless path; affects every modular-tree-learned
  encode)
- **Prerequisite**: ML pipeline budget for the dense grid
- **Bucket**: DEVIATED, room=medium

## P5: `tree_max_samples_fixed = 65_000` at e<=4 re-bisect (W44-210-E #4)

- **Const**: `EffortProfile.tree_max_samples_fixed = 65_000` (at e<=4)
- **Hypothesis**: chosen W36-1 era to bound build time; never
  re-measured against modern parallel-tree-learning code (W36-4+).
  Modern parallel implementation may admit larger sample budget at no
  wall cost.
- **Sweep grid**: {32_000, 65_000, 130_000, 260_000, 520_000} ×
  16 representative images × {e2, e3, e4}
- **Expected EV**: MED (perf trade — bytes saved vs wall)
- **Risk**: LOW (bounded growth; existing alloc is fine to 2× larger)
- **Prerequisite**: none
- **Bucket**: DEVIATED, room=medium

## P6: Splines `MIN_GRAD_MAG / MIN_EIG_RATIO / SIGMA_MIN / SIGMA_MAX / COST_BENEFIT_MARGIN` (W44-210-E #5)

- **Consts**: 10 NOT-IN-LIBJXL constants in `vardct/splines.rs`
- **Hypothesis**: splines is opt-in via `LossyConfig::with_splines()`;
  empirical bisect on a varied corpus (power-line photos, horizons,
  hair) never done. Defaults shipped at the dev's intuition.
- **Sweep grid**: corpus-specific (need spline-rich content corpus
  curated first)
  - `MIN_GRAD_MAG`: {0.10, 0.12, 0.15, 0.20, 0.25}
  - `MIN_EIG_RATIO`: {3.0, 5.0, 7.0, 10.0}
  - `SIGMA_MIN`: {0.4, 0.6, 0.8}
  - `SIGMA_MAX`: {2.0, 4.0, 6.0}
  - `COST_BENEFIT_MARGIN`: {1.5, 2.0, 3.0}
- **Expected EV**: HIGH if anyone wants splines to fire by default;
  LOW if it stays opt-in
- **Risk**: MED-HIGH (opt-in-only mechanism; default behavior
  unaffected)
- **Prerequisite**: spline-rich content corpus + auto-detection
  promotion decision
- **Bucket**: NOT-IN-LIBJXL, room=high

## P7: `W44_176_TERMINAL_CLASS_LUMA_VAR` band (W44-210-E #7)

- **Consts**:
  - `W44_176_TERMINAL_CLASS_LUMA_VAR_MIN = 1500.0`
  - `W44_176_TERMINAL_CLASS_LUMA_VAR_MAX = 2200.0`
- **Hypothesis**: derived from a 17-image probe (8 gb82-sc + 6 CID22 +
  3 borderline). 2200 upper cap set at +50% margin above terminal
  (1706) below imac_dark (3303), never tested against arbitrary
  terminal-class screenshots outside the probe corpus.
- **Sweep grid**: extend probe corpus to 50+ terminal/imac/imessage-class
  screenshots
  - `MIN`: {1200, 1500, 1700}
  - `MAX`: {1900, 2200, 2500, 3000}
- **Expected EV**: HIGH in coverage (could either generalize or shrink
  the band)
- **Risk**: MED (W44-109 terminal-class exclude is load-bearing for
  terminal e7 wins; bad threshold change could re-introduce regressions)
- **Prerequisite**: corpus extension
- **Bucket**: NOT-IN-LIBJXL, room=high

## P8: `W44-82` Lehmer per-entry cost heuristic re-tune (W44-210-E #11)

- **Consts**: inline `0.5 / 1.5 + log2(v+1)` per-entry cost approximation
  in `vardct/coeff_order.rs:556-560`
- **Hypothesis**: the per-entry approximation has never been validated
  against actual emitted Lehmer code byte counts. W44-201/W44-205
  measurements showed savings estimate (1 bit/zero) overshoots
  empirical 0.3-0.5 bits/zero. The per-bucket SKIP is the production
  fix, but the cost-side heuristic itself was never tuned.
- **Sweep grid**: per-bucket measured Lehmer byte count → fit linear
  model with new constants
- **Expected EV**: LOW (W44-201/W44-205 already extract most of the
  value)
- **Risk**: MED (gate is load-bearing; W44-82 measured +1595 B
  regression on gate removal)
- **Prerequisite**: per-bucket emit-count instrumentation
- **Bucket**: DEVIATED, room=medium

## P9: Refactor cluster — `mask_p25 = 85.0` to single shared const

- **Consts**: `W44_166_VARIANT_Z_PHOTO_MASK_P25_MIN`,
  `W44_150_PHOTO_W44_117_MASK_P25_MIN`, `W44_151_HIGH_MASK_P25_MIN`,
  `W44_168_SMOOTH_MASK_P25_MIN` — all = 85.0
- **Hypothesis**: 4 sites share an identical value derived from W44-149
  audit (1418519 mask_p25=88.88 with 11pp safety margin). Refactor to
  `SMART_ZENJXL_PHOTO_MASK_P25_MIN` would (a) make the relationship
  explicit, (b) enable single-point bisect for any future widening.
- **Sweep grid**: post-refactor, bisect {75, 80, 85, 90}
- **Expected EV**: MED — refactor enables the bisect at lower cost
- **Risk**: LOW (refactor is mechanical, all 4 sites byte-identical
  semantics)
- **Prerequisite**: refactor commit
- **Bucket**: NOT-IN-LIBJXL, room=high
- **W44-210-A finding**: noted as refactor candidate

## P10: Refactor cluster — `mask_median = 95.0` to single shared const

- **Consts**: `CONTENT_AWARE_SCREENSHOT_MEDIAN_THRESHOLD`,
  `BUTTLOOP::SCREENSHOT_MEDIAN_THRESHOLD`,
  `W44_168_SCREENSHOT_MEDIAN_MIN`, `splines::SCREENSHOT_MEDIAN_MASK_THRESHOLD`
- **Hypothesis**: 4 sites share `= 95.0`. Same refactor opportunity
  as P9. CAVEAT — W44-210-A flags "verify every consumer agrees on
  the semantic (some use it as screenshot-class predicate, some as
  high-confidence smooth predicate)" before hoisting.
- **Sweep grid**: post-refactor (and only if semantic-audit passes),
  bisect {90, 92.5, 95, 97.5}
- **Expected EV**: MED — refactor + single sweep
- **Risk**: MED (semantic-audit required first; failed audit means
  consts must stay separate)
- **Prerequisite**: semantic audit
- **Bucket**: NOT-IN-LIBJXL, room=high

## P11: Hoist `k32x32`/`k32x16`/`k64x32` triples to `EffortProfile` (W44-210-A refactor)

- **Consts**: 3-copy (`k32x32mul1/mul2/base`), 3-copy
  (`k32x16mul1/mul2/base`), 2-copy (`k64x32mul1/mul2/base`) inline
  duplicates in `vardct/ac_strategy_search.rs`
- **Hypothesis**: hoisting to `EffortProfile.k32x32` / `.k32x16` / `.k64x32`
  enables the picker to tune them per-content-class via
  `StrategyOverrides` (mirroring the existing `k8x8`/`k16x8`/`k16x16`/`k4x8`/`k4x4` slots).
- **Sweep grid**: post-hoist, full content-class × per-effort grid
- **Expected EV**: HIGH (currently no per-class tuning possible at
  large-DCT cost-model level; the 3-copy inline pattern blocks the
  picker)
- **Risk**: LOW (refactor; existing values byte-identical post-hoist)
- **Prerequisite**: refactor
- **Bucket**: SAME (currently libjxl-matched but inaccessible to picker)

## P12: distance-aware `W44_156_VARIANT_Z_D_HIGH_THRESHOLD` widen test (W44-210-E #8)

- **Const**: `W44_156_VARIANT_Z_D_HIGH_THRESHOLD = 5.5`
- **Hypothesis**: derived from a 20-cell bisect that found 5.0 and 5.5
  produced byte-identical results on the test cells. The 5.5 ships as
  the safer narrower band, but the wider 5.0 might admit useful cells
  on a wider corpus.
- **Sweep grid**: {4.5, 5.0, 5.5, 6.0} × wider corpus (12+ smooth-photo
  cells outside W44-154 test set)
- **Expected EV**: LOW-MEDIUM
- **Risk**: LOW (W44-156 measured 5.0/5.5 byte-identical on test
  cells — extension is monotone)
- **Prerequisite**: corpus extension
- **Bucket**: NOT-IN-LIBJXL, room=medium

## P13: `W44_120 + W44_140` EPF seed sub-cluster (W44-210-E #9)

- **Consts**:
  - `W44_120_EPF_SEED_MIN_DISTANCE = 1.0`
  - `W44_140_EPF_SEED_FADE_MAX = 1.5`
- **Hypothesis**: pareto-optimal on the bisect ranges tested (0.8/1.0/1.2/1.5
  + 1.5/2.0/3.0). BUT the W44-141 cluster (codec_wiki e8/e9 d=1.2/1.6/1.8)
  showed the bisect missed a sub-cluster.
- **Sweep grid**: finer bisect
  - `MIN_DISTANCE`: {0.9, 1.0, 1.1, 1.2}
  - `FADE_MAX`: {1.3, 1.5, 1.7, 2.0}
  - = 16-cell × W44-141 sub-cluster
- **Expected EV**: MED (resolves a documented bisect gap)
- **Risk**: LOW (existing distance-window pattern is well-understood)
- **Prerequisite**: none
- **Bucket**: NOT-IN-LIBJXL, room=medium

---

## Notes

- **Budget rule**: any sweep informing source constants MUST follow
  CLAUDE.md "Sweep / Calibration / Source-informing Benchmark
  Discipline (CRITICAL)" — 4-dimension grid (size × quality × mode ×
  content). Single-image benches are forbidden.
- **Ranking philosophy**: P1-P3 are the highest EV per measurement
  cost (P1 is binary ablation; P3 leverages an existing audit gap).
  P9-P11 are refactor-first chunks that enable cheaper future
  sweeps. P4-P8, P12, P13 are corpus-bound and require more setup.
- **Concurrent ship constraint**: don't sweep two consts that share an
  edge in [`TUNING_RELATIONS.md` Section 4](../../docs/TUNING_RELATIONS.md#section-4-edges)
  in the same chunk — measure one, ship, refresh ledger, then move
  to the next.
- **All entries cross-reference** the W44-210-E "High-EV deviation
  candidates" section + W44-210-A inventory rows. Maintainers MUST
  update this queue when a P-numbered candidate ships (move to
  Section G of `LIBJXL_DIVERGENCES.md` history).

## Provenance

| input | source |
|---|---|
| 8 explicit High-EV candidates | W44-210-E §High-EV Deviation #1-#11 |
| Refactor candidates (P9-P11) | W44-210-A §Cross-section observations |
| EPF seed sub-cluster (P13) | W44-210-E #9 + W44-141 ledger refresh |
| Distance threshold extension (P12) | W44-210-E #8 |
| Residual cluster context | W44-204 zenjxl attack ranking memo |
