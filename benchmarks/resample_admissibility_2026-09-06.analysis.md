# resampling admissibility on imazen-26 — benchmarks/resample_admissibility_2026-09-06.cells.tsv

4644 encode cells, 2236 decision cells, 42 images, 23 strata, efforts [5, 8], sizes [512, 1024].

A cell counts as admissible only when 2x beats full-res by more than 2% at matched butteraugli.

## 1. Is 2x ever the cheaper regime?

- admissible: **71/2236** cells (3%)
- unreachable at any bitrate (2x cannot match the full-res quality): 1601/2236 (72%)
- when admissible, byte saving: median 13%, max 37%

| stratum | cells | admissible | unreachable | first d where 2x wins | best saving |
|---|--:|--:|--:|--:|--:|
| ai-clipart | 104 | 0 (0%) | 97% | — | — |
| ai-illustrations | 104 | 4 (4%) | 53% | 15.0 | 24% |
| ai-products | 156 | 2 (1%) | 67% | 18.0 | 19% |
| epa-report | 52 | 1 (2%) | 87% | 25.0 | 27% |
| manuscript-illustrations | 104 | 6 (6%) | 57% | 15.0 | 34% |
| manuscript-text | 104 | 0 (0%) | 62% | — | — |
| mobile-screenshots | 104 | 2 (2%) | 71% | 18.0 | 22% |
| museum-aic | 52 | 1 (2%) | 35% | 15.0 | 4% |
| museum-met | 52 | 3 (6%) | 44% | 8.0 | 18% |
| noaa-documents | 104 | 4 (4%) | 75% | 15.0 | 20% |
| nps-brochures | 104 | 3 (3%) | 88% | 25.0 | 22% |
| patents | 104 | 0 (0%) | 100% | — | — |
| patents-gray-jpg | 52 | 0 (0%) | 71% | — | — |
| photos-food | 52 | 1 (2%) | 67% | 25.0 | 11% |
| photos-general | 104 | 8 (8%) | 57% | 10.0 | 13% |
| photos-interiors | 104 | 3 (3%) | 61% | 18.0 | 32% |
| photos-nature | 104 | 0 (0%) | 66% | — | — |
| photos-people | 52 | 3 (6%) | 35% | 12.0 | 21% |
| photos-png | 104 | 15 (14%) | 41% | 4.0 | 37% |
| plots | 156 | 0 (0%) | 100% | — | — |
| renders | 52 | 15 (29%) | 50% | 8.0 | 31% |
| textures | 52 | 0 (0%) | 44% | — | — |
| web-screenshots | 260 | 0 (0%) | 98% | — | — |

### by requested distance (all strata pooled)

| d | cells | admissible | unreachable | mean ratio (reachable) |
|--:|--:|--:|--:|--:|
| 1 | 172 | 0 (0%) | 100% | — |
| 2 | 172 | 0 (0%) | 100% | — |
| 3 | 172 | 0 (0%) | 99% | 2.26 |
| 4 | 172 | 1 (1%) | 98% | 1.51 |
| 5 | 172 | 0 (0%) | 95% | 2.03 |
| 6 | 172 | 0 (0%) | 89% | 2.28 |
| 8 | 172 | 5 (3%) | 77% | 2.17 |
| 10 | 172 | 3 (2%) | 68% | 1.68 |
| 12 | 172 | 5 (3%) | 54% | 1.79 |
| 15 | 172 | 11 (6%) | 47% | 1.55 |
| 18 | 172 | 16 (9%) | 38% | 1.43 |
| 21 | 172 | 13 (8%) | 35% | 1.35 |
| 25 | 172 | 17 (10%) | 30% | 1.35 |

## 2. Is the resampling floor a sound gate?

The floor bounds the regime, so `floor > bfly_full(d)` should imply unreachable. Violations would mean the bound is not a bound.

- cells with a floor measured: 2236
- effort-matched kernel: says unreachable on 1618/2236 cells; of those actually reachable on the measured grid (bound violations): **19**
- iterative kernel (the tighter bound): says unreachable on 1527/2236 cells; of those actually reachable on the measured grid (bound violations): **8**
- floor says reachable but the grid could not reach it: 2 (expected — the floor needs infinite bitrate; the grid stops at internal distance 0.3)

### headroom (bfly_full / floor) vs outcome

| bfly_full/floor | cells | admissible | mean ratio |
|---|--:|--:|--:|
| 0–0.5 | 1052 | 0 (0%) | — |
| 0.5–1 | 566 | 1 (0%) | 2.93 |
| 1–1.5 | 285 | 11 (4%) | 1.82 |
| 1.5–2 | 190 | 21 (11%) | 1.37 |
| 2–3 | 117 | 23 (20%) | 1.19 |
| ≥3 | 26 | 15 (58%) | 0.97 |

### is the bound tight? (best measured 2x quality vs the floor)

If the 2x regime saturates at its floor, the floor is not merely an upper bound on quality but a good PREDICTION of it — which is what makes a floor gate decisive rather than merely safe.

- best measured 2x butteraugli / floor over 168 image×size×effort cells: median **0.98**, p10 0.92, p90 1.00
- (1.00 = the grid reached the floor exactly; >1 = the grid's finest 2x setting is still short of it)

## 2c. Within-regime monotonicity on the full-res ladder (no switch involved)

Bytes should fall as the requested distance rises. Violations here are the quantiser's own, not a regime switch — and they bound how monotone any single-regime encoder can be on this corpus.

| stratum | steps | bytes rise | max rise | quality improves | max bfly improvement |
|---|--:|--:|--:|--:|--:|
| ai-clipart | 96 | 8 | +58.4% | 17 | 1.47 |
| ai-illustrations | 96 | 0 | +0.0% | 10 | 0.78 |
| ai-products | 144 | 3 | +96.5% | 9 | 2.24 |
| epa-report | 48 | 0 | +0.0% | 1 | 0.26 |
| manuscript-illustrations | 96 | 0 | +0.0% | 4 | 0.46 |
| manuscript-text | 96 | 1 | +0.4% | 2 | 0.96 |
| mobile-screenshots | 96 | 4 | +3.1% | 7 | 1.80 |
| museum-aic | 48 | 0 | +0.0% | 3 | 0.86 |
| museum-met | 48 | 0 | +0.0% | 6 | 1.87 |
| noaa-documents | 96 | 4 | +7.4% | 7 | 0.42 |
| nps-brochures | 96 | 5 | +2.9% | 7 | 1.81 |
| patents | 96 | 4 | +4.7% | 9 | 2.99 |
| patents-gray-jpg | 48 | 0 | +0.0% | 1 | 0.17 |
| photos-food | 48 | 0 | +0.0% | 2 | 0.87 |
| photos-general | 96 | 0 | +0.0% | 5 | 1.47 |
| photos-interiors | 96 | 0 | +0.0% | 2 | 3.75 |
| photos-nature | 96 | 0 | +0.0% | 6 | 0.61 |
| photos-people | 48 | 0 | +0.0% | 4 | 0.73 |
| photos-png | 96 | 0 | +0.0% | 6 | 1.16 |
| plots | 144 | 4 | +52.9% | 13 | 3.05 |
| renders | 48 | 0 | +0.0% | 1 | 0.00 |
| textures | 48 | 0 | +0.0% | 2 | 0.12 |
| web-screenshots | 244 | 33 | +74.8% | 57 | 10.11 |
| **all** | 2068 | 66 (3%) | | 181 (9%) | |

## 3. Decision rules (leave-one-stratum-out cross-validation)

`coverage` = share of cells the rule sends to 2x. `mean ratio` = matched-quality bytes vs full-res on those cells (<1 is a win). `unreachable` = cells it selected where 2x cannot deliver the requested quality at any bitrate — a quality failure, not a byte trade.

| rule | coverage | selected | precision | mean ratio | worst ratio | unreachable selected |
|---|--:|--:|--:|--:|--:|--:|
| never resample (current zen default) | 0% | 0 | n/a | — | — | 0 |
| libjxl: d >= 10 | 46% | 1032 | 6% | 1.49 | 7.56 | 468 |
| floor gate, LOSO-fitted k | 1% | 20 | 40% | 1.08 | 1.54 | 0 |
| oracle (knows the answer) | 3% | 71 | 100% | 0.85 | 0.97 | 0 |

LOSO-chosen floor multipliers k: [2.75, 3.5, 4.0] (rule: take 2x iff `floor * k <= bfly_full(d)`)

### floor-gate sensitivity on all data (not cross-validated)

| k | coverage | precision | mean ratio | worst ratio | unreachable |
|--:|--:|--:|--:|--:|--:|
| 1 | 28% | 11% | 1.52 | 6.73 | 2 |
| 1.5 | 15% | 18% | 1.27 | 3.65 | 0 |
| 2 | 6% | 27% | 1.15 | 2.17 | 0 |
| 2.5 | 3% | 38% | 1.07 | 1.73 | 0 |
| 3 | 1% | 58% | 0.97 | 1.39 | 0 |
| 4 | 0% | 100% | 0.74 | 0.84 | 0 |
| 5 | 0% | 100% | 0.83 | 0.84 | 0 |

## 4. Does any image feature add signal over the floor?

Target: admissible (positives 71/2236). Predictors: log(floor/d), log d, effort, log MP (+ 101 zenanalyze features in the second model). Leave-one-stratum-out AUC.

- floor/d alone, as a ranking score (no fitting, so no folds): AUC **0.927**
- logistic on the 4 base predictors: AUC **0.913** (over 2236 scorable rows, 71 positive)
- logistic on base + 101 zenanalyze features: AUC **0.897** (over 2236 scorable rows, 71 positive)

A fitted model can only be compared with floor/d on the rows it could score; strata holding all the positives make their own fold unfittable.

