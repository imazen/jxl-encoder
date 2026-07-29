# C3b — attribution-map steering A/B in the zensim loop (zensim task #67, 2026-07-29)

Pre-registered A/B: does the C3a FUSED attribution map (score + SAT steering
map from one `compute_with_ref_score_and_attribution` call) beat the
ModelSensitivity fold at closed-loop target hitting? Companion: zensim
`benchmarks/attribution_map_c1_2026-07-29.md` (C1-C3a).

## Setup

- Fixtures: city/dog/girl 576² (`/mnt/v/output/zensim/diffmap-coherence-2026-07-18/`).
- Targets: native zensim {75, 85, 92}; shared damped controller
  (`JXL_ZENSIM_TARGET_SCORE`, exponent 0.6, ×1.35 clamp, tol 0.25) — ALL arms
  share it so the A/B isolates the steering map.
- Arms (`JXL_ZENSIM_MODEL_MAP`): `baseline` (Trained fold, no model map),
  `abs` (ModelSensitivity fold, s_k from iter-1 central differences),
  `attr` (C3a fused score+map, per-tile `query_rect`, tile-level clamp-at-0,
  IDENTICAL normalization/blend tail), `attr-stale` (attr steering with the
  PREVIOUS iteration's map — prices the stale-scalar single-pass lever).
- Bakes (372-class, `JXL_ZENSIM_RD_PROFILE=bake:`): `v47_strict_qat_native`
  (A-class MLP) and `b_sdr_linear_cid80_inclwinsor_dense_dial` (shipped-B
  linear). Budget 6 redistribution iters (7 compares max), effort 8.
- Judge: decoded output rescored vs ref with the SAME bake (native score).
  Data: `zensim_attr_ab_{v47A,shippedB}_2026-07-29.tsv`.

## Result: the pre-registered gate is NOT met

Median |achieved − target| (9 cells/arm), decoded-judged:

| bake | baseline | abs (fold) | **attr** | attr-stale |
|---|--:|--:|--:|--:|
| v47A | 0.244 | **0.594** | 0.807 | 0.589 |
| shippedB | 1.174 | **0.982** | 1.507 | 1.355 |

Per-cell attr-vs-abs: v47A 1W/7L (1 degenerate no-steer cell: dog@75
converged before any model iteration); shippedB 1W/8L. attr does NOT beat
the fold at equal budget, and does not match it with fewer iterations
(median iters 7 vs 7). Per-target medians show the one attr win class:
shippedB@75 attr 0.299 vs abs 0.842; everywhere else fold ≥ attr.

**Steering-signal check (not the failure mode):** `JXL_ZENSIM_ATTR_PROBE` on
city@85/v47A shows the attr tile field spanning the FULL dynamic range
(min 0.6 = ratio-0 clamp, max 3.3 = ratio_max clamp) — the map
differentiates 8px tiles strongly; the loss is in how that allocation
interacts with the shared redistribution+controller dynamics, not signal
degeneracy. Also NOT a scorer-path artifact: the fused arm's
in-loop→decoded offset is SMALLER than the fold's (median +0.033 vs +0.073
v47A; +0.071 vs +0.154 shippedB).

## The stale-map lever is VIABLE (the forward-looking answer)

`attr-stale` ≈ `attr` in every slice — and slightly BETTER on median (v47A
0.589 vs 0.807; shippedB 1.355 vs 1.507). A one-iteration-stale attribution
map costs nothing at these loop dynamics, which is precisely the semantic
precondition for the C3a "stale-scalar single-pass" perf endpoint (fold the
density in-kernel using the previous compare's pooled scalars → the ≤1.1×
marginal). The perf case survives; the QUALITY case for attr-vs-fold in
THIS loop does not (yet).

## Wall time (576², medians)

ms/compare: baseline 36.5 | abs 60.7 | attr 68.9 (v47A) — the fused call
adds **+8.2 ms/compare over the fold arm** (matches C3a's 9.0 ms marginal
prediction; ~12 % of the steered-iteration cost, ~+7 % of total encode).
First model iteration adds a one-off ~150 ms (372 central differences for
s_k) in ALL model arms. shippedB: 39.8 | 45.9 | 51.7.

## Honest caveats

1. |err| under a shared controller measures {map × redistribution ×
   controller} jointly; several cells are controller-dominated (t=75
   overshoot from the d=2.5 seed; t=92 saturates the qf deviation clamp for
   every arm — nobody reaches 92).
2. dog@75 converged pre-steering in all arms (identical bytes) — degenerate.
3. n = 3 fixtures × 3 targets × 2 bakes; medians over 9 cells/arm/bake.

## Pre-existing dep failures (disclosed, NOT from this change)

`cargo test --lib` with `zensim-loop` shows 3 failures in
`vardct::zensim_backend::tests::cpu_zensim_*` — a LATENT zensim bug on the
sub-64 linear-planar path (`precompute_reference_linear_planar` lacks the
sub-64 reflect-pad its ImageSource twin has → panic at
`streaming.rs` mean-offset on 32² fixtures). Reproduced standalone against
zensim main AND at pre-program commit `ea565b71`; the same 3 tests fail on
plain jxl-encoder main (`9d711ee4`) with the current path dep. Owned by
zensim (read-only for this phase); reported upstream via the program
coordinator. All other tests: 1520 passed.

## Repro

```sh
cargo build --release -p jxl-encoder \
  --features "__expert butteraugli-loop zensim-loop ssim2-loop parallel" \
  --example zensim_diffmap_rd
zensim_diffmap_rd --corpus-file corpus576.tsv --zensim-targets 75,85,92 \
  --arms baseline,abs,attr,attr-stale --bake <372-bake.bin> --iters 6 \
  --label v47A --out-dir <dir>
```
