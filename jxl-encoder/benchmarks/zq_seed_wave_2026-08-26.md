# jxl-encoder Zq seed head — pre-registered wave (2026-08-26)

**Goal (criterion 4)**: a content-aware iteration-1 seed for the zensim target
controller, replacing the content-blind 3-step staircase
(`zensim_diffmap_rd.rs::seed_distance_for_target`: 0.9 / 1.5 / 2.5 by target
band). Mirrors zenjpeg's shipped `zq_seed` (wave:
`zenjpeg/benchmarks/zq_seed_wave_2026-08-26.md`) and zenavif's `q0_head`.

## Data (frozen)
- `/mnt/v/output/canonical-picker-2026-07-01-zensimA/zenjxl_lossy/{train,validate}.parquet`
  (726,705 train rows; origin-clean split). Curve key `(origin_id, cell, width,
  height)`; **80,745 of 80,745 curves carry the full 9-point q grid**
  {5,25,30,40,50,60,70,80,90} (no coarse-plan exclusions needed).
- Labels: `q*(t)` = leftmost crossing of the PAVA-isotonic q→`score_zensim`
  curve at t ∈ {40,45,…,90} (zensimA-era scores — same registered limitation
  as the zenjpeg wave: seed-only role, search corrects residual).
- **Unit bridge (frozen)**: the head is fit in ENCODER-QUALITY units; the
  runtime converts with the same public `jxl_encoder::quality_to_distance`
  the sweep's zenjxl encodes went through. No hand-rolled mapping.

## Features + model (frozen)
Same candidate pool, transforms, basis, robust-L1 fit, and greedy
LOO-origin-p90 selection (max 8 features, seeded 80k selection subsample) as
the zenjpeg wave — per-codec copy of `scripts/fit_zq_seed.py` by the
loop-ownership directive.

## Gates (frozen BEFORE any fit runs)
- **G-J1 (diagnostic, reported)**: validate |q0−q*| p50/p90.
- **G-J2 (THE decision gate — the real census, not a sim)**: the in-loop
  controller (damped step + secant + redistribution) is not faithfully
  portable to stored q→score curves, so NO offline sim is claimed. Decision =
  a 27-cell census A/B on the registered instrument (9 refs × targets
  {70,80,88}, current loop defaults, k2 emit-best protocol): arm A = the
  staircase seed, arm B = the head seed (in-binary via zenanalyze under
  `JXL_ZENSIM_SEED_HEAD=1`). **PASS iff median decoded |achieved−target|
  improves ≥ 15% AND the ±2-hit count does not regress.**
- **G-J3 (safety, by construction)**: head returns `Option`; any feature
  failure ⇒ staircase (never degrades current behavior).

## Endgame (frozen)
PASS ⇒ consts module `jxl-encoder/src/zq_seed.rs` (pure fn, q-domain, callers
bridge via `quality_to_distance`), seed wiring in the A/B binary, census A/B
TSV committed here, plan/memory updated. FAIL ⇒ numbers committed here; the
staircase stays.
