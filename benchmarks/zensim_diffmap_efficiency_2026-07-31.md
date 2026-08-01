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

_(filled in after the runs; protocol above is frozen as of the registration
commit)_
