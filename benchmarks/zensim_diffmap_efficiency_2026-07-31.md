# zensim qf-targeting loop EFFICIENCY study (2026-07-31)

Follow-on to #69 (`benchmarks/zensim_attr_loop69_2026-07-29.md`): characterize
how efficient the zensim diffmap-driven qf-targeting loop is — iterations to
reach a quality target within tolerance, the byte cost of tighter tolerance,
and size (bytes) targeting. **Characterization instrument — no pass/fail
gates — but every endpoint below is FROZEN before the first measurement run
to prevent cherry-picking. Runs must match this protocol; deviations are
reported as such, never silently substituted.**

Substrate: jxl-encoder `d17cf7ce` (+ this study's additive instruments),
zensim read-only path dep at `e8cd105a`. Harness:
`jxl-encoder/examples/zensim_diffmap_rd.rs` target mode (`run_target_ab`) on
`jxl-encoder/src/vardct/zensim_loop.rs`. Committed runners:
`scripts/zensim-loop-eff/` (incl. the rescued #69 `run69.sh`).

## Loop semantics (read from code before registration; governs definitions)

- Budget: `--iters N` = `zensim_iters`; the loop runs compares at iteration
  indices `0..=N` (N+1 compares; index 0 = the seed state's compare; the
  last executed iteration is compare-only). Hard cap `ITER_MAX = 32`
  (`validation.rs:168`). **The loop has no intrinsic default budget** — the
  #69 study ran `--iters 6` (7 compares); this study's "default budget" =
  `--iters 6`, "extended" = `--iters 12` (13 compares).
- Early stop: with `JXL_ZENSIM_TARGET_SCORE` set, the loop breaks after the
  compare at iteration `i >= 1` when `|score_i − target| <= tol`
  (`JXL_ZENSIM_TARGET_TOL`, code default 0.25). Iteration 0 never stops the
  loop. `tol = -1` disables early stop (|err| ≤ −1 is never true).
- The emitted bitstream corresponds to the quant field measured by the LAST
  executed compare (break happens before redistribution/controller).
- Controller: damped global qf step `g = clamp((loss_ach/loss_tgt)^0.6,
  1/1.35, 1.35)` per iteration after sum-preserving redistribution.
- **The loop never entropy-codes**: per-iteration true bytes do NOT exist
  inside the loop, and there is no in-loop bytes estimate either. This is a
  recorded limitation. Per-iterate bytes are measured by budget-capped /
  early-stopped full encodes (deterministic trajectory ⇒ the capped run's
  iterates equal the traced run's; determinism is proven in R0 below).
- `JXL_ZENSIM_RD_STATS` emits ONE line per encode (compares_used,
  final_score, loop_ms, per-iter ms) — no per-iteration score, hence the
  new trace instrument below.

## Instruments (additive, env-gated, default-off; committed)

1. `JXL_ZENSIM_TRACE=<path>` (+ optional `JXL_ZENSIM_TRACE_ID=<id>`): the
   loop appends one TSV line per compare:
   `trace_id  iter  score  qf_mean  qf_min  qf_max  iter_ms`.
   **No bytes column** — see limitation above (registered honestly rather
   than emitting a proxy).
2. `JXL_ZENSIM_QF_GLOBAL_SCALE=<g>`: multiplies the float quant field once
   at loop entry (unset/1.0 = no-op). Actuator for E7's bytes-target outer
   loop — the same actuator (global multiplicative qf scale) the existing
   in-loop controller uses.
3. Harness: `run_target_ab` sets `JXL_ZENSIM_TRACE_ID=<label>|<name>|<class>|<target>|<arm>`
   per encode; `JXL_SAVE_BITSTREAM=1` saves the encoded `.jxl` next to the
   decoded PNG (identity gate); new `--bytes-targets-file` mode (E7).
4. **Default-unchanged gate (R0):** one cell encoded (a) with no instrument
   envs, (b) with TRACE+TRACE_ID set, (c) with `JXL_ZENSIM_QF_GLOBAL_SCALE=1.0`
   → all three `.jxl` outputs must be byte-identical (also proves run-to-run
   determinism, which E3/E6 alignment relies on). Gate failure = fix before
   any measurement run.

## Arms (in-run engagement controls per arm)

- v47A (`zensim/weights/v47_strict_qat_native_2026-05-27.bin`, MLP-class) ×
  {baseline, h3-mag (ZENSIM_H3_GAIN=10.0 default, not swept)}
- shippedB (`zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`,
  linear) × {baseline}
- Engagement controls: per arm-run, `JXL_ZENSIM_ATTR_PROBE=<per-run file>` —
  h3-mag runs MUST produce probe lines (attr-steered iterations); baseline
  runs MUST produce none (unknown `JXL_ZENSIM_MODEL_MAP` values silently
  fall through to baseline — the #69 fallthrough hazard). Additionally the
  per-cell traces must diverge between arms (identical trajectories = arm
  did not engage).

## Matrix and fixtures (#69 set, unchanged)

9 refs × targets {70, 80, 88} = 27 cells/arm: city/dog/girl 576²
(`/mnt/v/output/zensim/diffmap-coherence-2026-07-18/`); CID22-512 validation
1025469/1418519/1189261 (`~/work/codec-corpus/CID22/CID22-512/validation/`);
nonphoto 576² crops (`-crop 576x576+512+256`) of gb82-sc
codec_wiki/gui/imessage (rescued into `~/tmp/attrmap-69/fixtures/`, crops
regenerated if absent). Effort 8, seed distance per `seed_distance_for_target`
(2.5 / 1.5 / 0.9 for t70/t80/t88). Judged = decode the final bitstream and
score with the SAME bake (as #69).

## Runs

- R0: identity/determinism gate (above), 1 cell × 3 env conditions.
- R1 main matrix: tol=−1 (no early stop), iters=6, TRACE on. 3 arm×bake
  combos × 27 cells = 81 encodes. Feeds E1/E2/E4/E8 and E7's targets.
- R2 tolerance runs: tol ∈ {0.25, 0.5, 1.0, 2.0}, iters=6, TRACE on, same
  3 combos = 324 encodes. Feeds E3.
- R3 extended budget: iters=12, tol=−1, v47A × {baseline, h3-mag} = 54
  encodes. Feeds E5 (and E6's k=12 point).
- R4 budget-capped: k ∈ {1,2,4,8} (k=6 reused from R1, k=12 from R3),
  v47A × {baseline, h3-mag} × 3 refs × 3 targets = 72 encodes. Subsample
  refs FROZEN NOW: city (photo), 1025469 (CID photo), sc_wiki (nonphoto).
  Feeds E6.
- R5 bytes-target mode: v47A × baseline only, 27 cells × 8 outer
  iterations (fixed outer budget, no early acceptance — full curve) = 216
  encodes. Feeds E7.

## Endpoints (FROZEN)

- **E1 iterations-to-tolerance:** per cell, first iteration index where
  |score_i − target| ≤ τ, τ ∈ {0.25, 0.5, 1.0, 2.0}; median + p90 per
  arm × target; fraction of cells never reaching each τ within budget
  (iters=6 run). Reported both over all iterations (frozen definition) and
  over i ≥ 1 (the deployed stop rule's domain) when they differ.
- **E2 convergence curve:** median |score_i − target| vs iteration index
  per arm × target (R1 traces).
- **E3 byte cost of tolerance:** bytes at first-τ-hit (R2 early-stopped
  encodes) vs bytes at budget end (R1); overshoot ratio
  bytes_τ / bytes_budget_end; and does stopping earlier at looser τ save
  bytes or cost quality (judged scores of R2 vs R1 encodes).
- **E4 stability:** count of sign flips of (score_i − target) after
  iteration 2, per arm (R1 traces).
- **E5 tolerance floor:** min |score_i − target| over the extended run
  (iters=12) per cell; per-arm median/p90 (R3), vs the iters=6 floor (R1).
- **E6 judged calibration:** for the 3-ref × 3-target v47A subsample (both
  arms), decoded-judged score at k ∈ {1,2,4,8,6,12} — the TRUE convergence
  curve — and the internal-vs-judged delta (judged_k − score_k)
  distribution; alignment check: the capped run's final internal score must
  equal the R1/R3 trace's score at iteration k (determinism).
- **E7 SIZE targeting:** bytes-target mode (committed): controller error
  = (bytes − target_bytes)/target_bytes; the existing damped controller
  formula (ratio^0.6, per-step clamp [1/1.35, 1.35]) drives
  `JXL_ZENSIM_QF_GLOBAL_SCALE` across OUTER full encodes (the loop cannot
  measure bytes per iterate — recorded limitation; an E7 "iteration" is
  therefore one full encode with real bytes; inner loop runs iters=6
  redistribution-only, `JXL_ZENSIM_TARGET_SCORE` unset). target_bytes per
  cell = R1 v47A-baseline budget-end bytes for that (ref, target) —
  feasible by construction. g_0 = 1.0; update g_{j+1} = g_j ×
  clamp((target_bytes/bytes_j)^0.6, 1/1.35, 1.35) (qf up ⇒ more bytes).
  Endpoints: outer iterations to within {1%, 2%, 5%} of target bytes
  (median + p90; never-reached fraction at outer budget 8), and the
  achieved-quality spread at fixed size: judged score at the first
  within-2% iterate vs the R1 quality-run's judged score for the same cell.
- **E8 wall ms/iteration** per arm (median per-compare ms from traces;
  context for the single-pass perf follow-up).

## Honesty rules

Report every cell; nulls and ugly results stand. If an endpoint cannot be
measured as registered, the limitation is recorded explicitly (as already
done for in-loop per-iterate bytes) — never a silent proxy. No registered
endpoint is relaxed. Analysis medians must be re-derivable exactly from the
committed TSVs.

## Results

Data: `zensim_diffmap_eff_{traces,cells,bytes_target}_2026-07-31.tsv` (this
dir); analysis: `scripts/zensim-loop-eff/analyze_eff.py` (stat definitions
stated in its docstring: median = numpy.median, p90 = numpy.percentile
`method='linear'`; every table below re-derives exactly from the TSVs). Runs
2026-07-31, ~760 encodes, all phases + both in-run gates green:

- **R0 identity gate PASS** — instrument envs off / TRACE on /
  QF_GLOBAL_SCALE=1.0 produce byte-identical bitstreams (sha `12cf08e0…`).
- **Engagement gate PASS** — h3-mag probe = 162 lines (27 cells × 6 steered
  iterations exactly); both baseline probes = 0 lines.
- **Cross-binary reproduction**: the R2 τ=0.25 v47A runs reproduce the #69
  TSV **cell-for-cell, 54/54 exact** (bytes identical, scores to print
  precision) across a different build with the instruments compiled in —
  determinism + default-unchanged, independently of R0.

### Headline conclusions

1. **At the #69 budget (iters=6) the loop is budget-limited, not
   tolerance-limited.** The MEDIAN cell never early-stops even at τ=1.0
   (med compares used = 7 = budget); at τ=0.25 only 3-6 cells of 27 stop
   early. Doubling the budget cuts the achievable floor ~3.6-7×
   (E5: med min|err| 0.955→0.268 baseline, 0.649→0.090 h3-mag).
2. **Tight quality tolerance is not reachable in 6 iterations for a third
   of cells** — never-reach at τ=0.25 is 20/27 (baseline) / 17/27 (h3-mag)
   / 18/27 (B). The mechanism is seed miscalibration + controller damping:
   on nonphoto the seed distance for t70 lands ~20 points high (sc_wiki t70
   starts at 90.2), and the per-iteration loss-step clamp (≤1.35×) needs
   ~10+ iterations to close that — sc_wiki t70 reaches 87.0 by i6, 73.3 by
   i12. Iteration budget is spent compensating for a bad seed, not
   fine-tuning.
3. **h3-mag converges faster AND tighter than baseline on v47A at
   t70/t80** (E2 median |err| at i6: 0.88/0.13 vs 2.18/0.80), extending
   #69's accuracy verdict to the efficiency axis. At t88 the arms tie.
4. **Trajectories are one-sided on v47A** (E4: 2-6 sign flips across all
   27 cells) — the loop approaches the target from above (quality
   surplus). shippedB oscillates (18 flips, 10/27 cells ≥1, max 3).
5. **Stopping earlier at looser τ does NOT save bytes on v47A — it costs
   bytes.** Because the approach is from above, an early stop emits an
   above-target state: stopped cells at τ=2.0 emit +1.2% (baseline) /
   +3.9% (h3-mag) bytes and land +0.33/+0.97 above target. On shippedB the
   sign flips (−3.0% bytes at τ=2.0, med 3 compares) because its seed
   lands closer and oscillation crosses the target sooner. "Looser
   tolerance = cheaper" is FALSE for this loop's geometry on v47A.
6. **The internal score is a trustworthy stop signal on photos, softer on
   nonphoto** (E6): med (judged − internal) is within ±0.13 at every
   budget; the outliers are sc_wiki at ±1.2-1.5. E1/E2 read from the
   internal trace transfer to decoded reality at ~0.1-0.2 precision on
   photo content.
7. **The loop emits the LAST iterate, not the best** — with no early stop,
   h3-mag's judged error on the E6 subsample gets WORSE past its sweet
   spot (med |judged−target| 0.59 at k=6 → 0.90 at k=8 → 1.02 at k=12):
   after crossing the target its small net-downward drift does not
   self-correct. Extended budgets need the tolerance stop (or, cheaper, a
   best-iterate keeper — not built, noted as follow-up).
8. **Bytes targeting with the transplanted damped controller converges,
   but slowly** (E7): med 4 outer full encodes to within 5%, 6 to 2%,
   6.5 to 1% — and 17/27 cells never reach 1% within 8 encodes (10/27
   never reach 5%; large initial offsets + the 0.6/1.35 damping tuned for
   the score dial). **Quality at fixed size is path-independent**: judged
   score at the first within-2% iterate matches the quality-run's judged
   score at the same bytes to med +0.21 / max 0.57 — hitting the size
   from the size dial or the quality dial lands at the same quality.
9. **Per-compare cost** (E8): ~36 ms (baseline/B), ~47.5 ms (h3-mag
   fused compare) + a one-time ~196 ms model-gradient at iteration 0 —
   at 576², a 7-compare targeted encode is ~0.28 s (baseline) / ~0.5 s
   (h3-mag) of loop on top of ~0.12 s encode proper.

### E1 — iterations to |internal − target| ≤ τ (median / p90 over cells that
reach it; "never" = fraction not reaching within iters=6; med is over
reaching cells only, so read it WITH the never column. First-hit(i≥1) is
the deployed stop rule's domain — iteration 0 cannot stop the loop)

τ = 0.25:

| arm | target | first-hit med | p90 | never-frac | first-hit(i≥1) med |
|---|--:|--:|--:|--:|--:|
| v47A/baseline | 70 | 6.0 | 6.0 | 6/9 | 6.0 |
| v47A/baseline | 80 | 6.0 | 6.0 | 6/9 | 6.0 |
| v47A/baseline | 88 | 0.0 | 0.0 | 8/9 | 3.0 |
| v47A/h3-mag | 70 | 4.5 | 5.7 | 5/9 | 4.5 |
| v47A/h3-mag | 80 | 5.0 | 6.0 | 4/9 | 5.0 |
| v47A/h3-mag | 88 | 0.0 | 0.0 | 8/9 | 6.0 |
| B/baseline | 70 | 3.0 | 5.4 | 6/9 | 3.0 |
| B/baseline | 80 | 3.0 | 5.6 | 4/9 | 3.0 |
| B/baseline | 88 | 2.0 | 2.0 | 8/9 | 2.0 |

τ = 0.5:

| arm | target | first-hit med | p90 | never-frac | first-hit(i≥1) med |
|---|--:|--:|--:|--:|--:|
| v47A/baseline | 70 | 5.0 | 5.0 | 6/9 | 5.0 |
| v47A/baseline | 80 | 5.0 | 5.0 | 6/9 | 5.0 |
| v47A/baseline | 88 | 0.0 | 0.0 | 8/9 | 1.0 |
| v47A/h3-mag | 70 | 3.5 | 4.7 | 5/9 | 3.5 |
| v47A/h3-mag | 80 | 4.5 | 5.5 | 3/9 | 4.5 |
| v47A/h3-mag | 88 | 2.0 | 5.2 | 6/9 | 2.0 |
| B/baseline | 70 | 4.0 | 5.6 | 4/9 | 4.0 |
| B/baseline | 80 | 2.5 | 5.5 | 3/9 | 2.5 |
| B/baseline | 88 | 2.0 | 2.0 | 8/9 | 2.0 |

τ = 1.0:

| arm | target | first-hit med | p90 | never-frac | first-hit(i≥1) med |
|---|--:|--:|--:|--:|--:|
| v47A/baseline | 70 | 4.0 | 4.7 | 5/9 | 4.5 |
| v47A/baseline | 80 | 4.0 | 5.2 | 4/9 | 4.0 |
| v47A/baseline | 88 | 1.0 | 5.0 | 4/9 | 2.0 |
| v47A/h3-mag | 70 | 3.0 | 4.6 | 4/9 | 3.0 |
| v47A/h3-mag | 80 | 3.0 | 4.5 | 3/9 | 3.0 |
| v47A/h3-mag | 88 | 0.5 | 4.5 | 5/9 | 1.5 |
| B/baseline | 70 | 2.0 | 4.2 | 4/9 | 2.0 |
| B/baseline | 80 | 1.0 | 3.5 | 3/9 | 1.0 |
| B/baseline | 88 | 5.0 | 5.0 | 6/9 | 5.0 |

τ = 2.0:

| arm | target | first-hit med | p90 | never-frac | first-hit(i≥1) med |
|---|--:|--:|--:|--:|--:|
| v47A/baseline | 70 | 2.0 | 4.6 | 4/9 | 2.0 |
| v47A/baseline | 80 | 2.0 | 3.5 | 3/9 | 2.0 |
| v47A/baseline | 88 | 2.0 | 4.0 | 2/9 | 2.0 |
| v47A/h3-mag | 70 | 4.0 | 6.0 | 2/9 | 4.0 |
| v47A/h3-mag | 80 | 2.0 | 4.8 | 2/9 | 2.0 |
| v47A/h3-mag | 88 | 2.5 | 4.3 | 1/9 | 2.5 |
| B/baseline | 70 | 1.0 | 1.5 | 3/9 | 1.0 |
| B/baseline | 80 | 0.0 | 4.0 | 2/9 | 1.0 |
| B/baseline | 88 | 3.0 | 5.0 | 1/9 | 3.0 |

### E2 — median |err| vs iteration index (iters=6 run)

| arm | target | i0 | i1 | i2 | i3 | i4 | i5 | i6 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| v47A/baseline | 70 | 6.15 | 4.26 | 3.53 | 3.17 | 2.78 | 1.83 | 2.18 |
| v47A/baseline | 80 | 4.01 | 3.41 | 1.85 | 1.42 | 1.49 | 1.12 | 0.80 |
| v47A/baseline | 88 | 4.04 | 2.53 | 2.22 | 1.85 | 1.60 | 0.93 | 0.95 |
| v47A/h3-mag | 70 | 6.15 | 4.26 | 3.77 | 2.64 | 1.14 | 0.77 | 0.88 |
| v47A/h3-mag | 80 | 4.01 | 3.41 | 1.79 | 1.18 | 0.74 | 0.47 | 0.13 |
| v47A/h3-mag | 88 | 4.04 | 2.53 | 2.18 | 1.89 | 1.50 | 1.05 | 1.05 |
| B/baseline | 70 | 3.76 | 1.77 | 1.87 | 1.33 | 1.66 | 1.06 | 1.53 |
| B/baseline | 80 | 2.59 | 2.61 | 1.76 | 1.09 | 0.63 | 1.55 | 1.16 |
| B/baseline | 88 | 6.83 | 3.28 | 2.19 | 1.98 | 1.95 | 1.77 | 1.31 |

### E3 — byte cost of tolerance (R2 early-stop vs R1 budget-end)

All-cells medians AND the early-stopped subset (a cell that never hits τ
runs to budget ⇒ identical bitstream by determinism, diluting the all-cells
medians — the frozen endpoint is reported both ways).

| arm | τ | med bytes ratio | med Δjudged | med iters | n stopped<7 | stopped: med bytes ratio | stopped: med Δjudged |
|---|--:|--:|--:|--:|--:|--:|--:|
| v47A/baseline | 0.25 | 1.000 | +0.00 | 7 | 3/27 | 1.005 | -0.03 |
| v47A/baseline | 0.5 | 1.000 | +0.00 | 7 | 7/27 | 1.008 | +0.20 |
| v47A/baseline | 1.0 | 1.000 | +0.00 | 7 | 12/27 | 1.007 | +0.20 |
| v47A/baseline | 2.0 | 1.000 | +0.00 | 5 | 18/27 | 1.012 | +0.33 |
| v47A/h3-mag | 0.25 | 1.000 | +0.00 | 7 | 6/27 | 1.009 | +0.18 |
| v47A/h3-mag | 0.5 | 1.000 | +0.00 | 7 | 11/27 | 1.023 | +0.24 |
| v47A/h3-mag | 1.0 | 1.000 | +0.00 | 6 | 14/27 | 1.033 | +0.64 |
| v47A/h3-mag | 2.0 | 1.000 | +0.00 | 5 | 19/27 | 1.039 | +0.97 |
| B/baseline | 0.25 | 1.000 | +0.00 | 7 | 7/27 | 0.970 | +0.14 |
| B/baseline | 0.5 | 1.000 | +0.00 | 7 | 10/27 | 0.988 | +0.16 |
| B/baseline | 1.0 | 1.000 | +0.00 | 6 | 14/27 | 0.975 | +0.27 |
| B/baseline | 2.0 | 0.981 | +0.00 | 3 | 21/27 | 0.970 | -0.29 |

### E4 — sign flips of (score − target) after iteration 2 (iters=6 run)

| arm | flips total (27 cells) | cells with ≥1 flip | max flips/cell |
|---|--:|--:|--:|
| v47A/baseline | 2 | 2/27 | 1 |
| v47A/h3-mag | 6 | 6/27 | 1 |
| B/baseline | 18 | 10/27 | 3 |

### E5 — tolerance floor: min |internal − target| per cell

| arm | budget | med floor | p90 floor | max floor |
|---|--|--:|--:|--:|
| v47A/baseline | iters=6 | 0.955 | 5.609 | 16.980 |
| v47A/baseline | iters=12 | 0.268 | 1.284 | 3.275 |
| v47A/h3-mag | iters=6 | 0.649 | 3.883 | 13.478 |
| v47A/h3-mag | iters=12 | 0.090 | 0.636 | 3.517 |

(The 16.98/13.48 max floors are sc_wiki t70 — the seed-miscalibration cell;
see headline #2.)

### E6 — judged calibration (frozen 3-ref × 3-target v47A subsample)

| arm | k | med \|judged−target\| | med (judged−internal) | max \|judged−internal\| |
|---|--:|--:|--:|--:|
| v47A/baseline | 1 | 3.36 | -0.027 | 0.347 |
| v47A/baseline | 2 | 3.04 | -0.126 | 0.305 |
| v47A/baseline | 4 | 1.54 | +0.001 | 0.240 |
| v47A/baseline | 6 | 0.96 | -0.013 | 0.226 |
| v47A/baseline | 8 | 0.65 | -0.055 | 0.741 |
| v47A/baseline | 12 | 0.64 | -0.061 | 0.483 |
| v47A/h3-mag | 1 | 3.36 | -0.027 | 0.347 |
| v47A/h3-mag | 2 | 2.84 | -0.117 | 0.247 |
| v47A/h3-mag | 4 | 0.69 | +0.000 | 0.173 |
| v47A/h3-mag | 6 | 0.59 | -0.130 | 1.492 |
| v47A/h3-mag | 8 | 0.90 | -0.017 | 0.239 |
| v47A/h3-mag | 12 | 1.02 | -0.077 | 1.244 |

Both max-delta outliers are sc_wiki (nonphoto): −1.49 (h3 k=6 t70), +1.24
(h3 k=12 t80); photo cells stay ≤ 0.35. Determinism alignment (capped-run
internal@k vs R1-trace score@k, tol 1.1e-3 — above the 5.5e-4 print-rounding
bound of the 3dp/4dp TSVs, far below real divergence): **PASS**.

### E7 — bytes targeting (outer full encodes; v47A baseline)

| threshold | med outer-iters to \|rel_err\| ≤ x | p90 | never (of 27) |
|---|--:|--:|--:|
| 1% | 6.5 | 7.0 | 17 |
| 2% | 6.0 | 6.6 | 12 |
| 5% | 4.0 | 5.0 | 10 |

(med/p90 over reaching cells only — pair with the never column.) Quality at
fixed size (first within-2% iterate vs the R1 quality-run judged score, the
15 reaching cells): med +0.21, p90(|·|) 0.39, max |Δ| 0.57.

### E8 — wall ms per compare (median over all iters=6 compares)

| arm | med ms/compare | p90 | med i0 / i1 ms (h3 pays its one-time model gradient at i0) |
|---|--:|--:|--:|
| v47A/baseline | 36.0 | 39.7 | i0 38.1 / i1 35.1 |
| v47A/h3-mag | 47.5 | 193.1 | i0 196.0 / i1 47.0 |
| B/baseline | 36.5 | 40.0 | i0 39.2 / i1 36.5 |

### Limitations (registered + observed)

- Per-iterate bytes do not exist inside the loop (never entropy-codes; no
  estimate either) — E3/E6 bytes come from separate deterministic encodes,
  E7 iterations are full encodes. Registered before running; held.
- E1/E2/E4/E5 read the INTERNAL score; E6 prices the transfer to
  decoded-judged (±0.13 med, ±1.5 worst on nonphoto). Nonphoto conclusions
  at sub-point precision should use judged numbers.
- E7 ran baseline/v47A only (as registered); the controller transplant was
  deliberately NOT re-tuned for the bytes dial (0.6/1.35 as shipped) — the
  slow 1% convergence is a finding about the transplant, not the best
  achievable bytes controller.
- 576²-class fixtures only; no size sweep (this is a loop-dynamics study,
  not a perf calibration — E8 ms/compare scales with pixels).
