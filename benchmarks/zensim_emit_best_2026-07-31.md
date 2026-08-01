# zensim loop BEST-SO-FAR emission A/B (2026-07-31)

#70 sub-item from the efficiency study's finding #7
(`benchmarks/zensim_diffmap_efficiency_2026-07-31.md`): the loop emits the
LAST iterate, and overshoot past the sweet spot does not self-correct (h3
judged err 0.59@k6 → 0.90@k8 → 1.02@k12). This change adds
`JXL_ZENSIM_EMIT_BEST=1` (default OFF): track the compare whose INTERNAL
fused score is closest to `JXL_ZENSIM_TARGET_SCORE`, snapshot the float
quant field it measured, and restore that field for the final SetQuantField
— the emitted bitstream corresponds to the BEST measured iterate, not the
last. Exact ties go to the LATEST iterate (so a tied last iterate emits the
default bitstream bit-for-bit). The internal score is the trusted proxy
(efficiency E6: judged−internal med ±0.13 on photos, ±1.5 worst nonphoto);
no decoded scoring is added inside the loop. With no target set the env is
a no-op ("best" is undefined without a target).

**Pre-registered: every arm, endpoint, gate, and expectation below is
FROZEN before the first measurement run. Runs must match this protocol;
deviations are reported as such, never silently substituted.**

## Substrate (recorded before running)

- jxl-encoder `main@origin` = `0eb31edc` + this change (workspace
  `emit-best`; committed shas in Results).
- zensim read-only path dep at `402f4e63` (clean tree — the SAME substrate
  as the metric-matrix study, so its k6 numbers are directly comparable).
- Loop semantics unchanged from the prior studies: `--iters N` runs
  compares at indices `0..=N`; without emit-best the emitted bitstream
  corresponds to the quant field measured by the LAST executed compare;
  the loop never entropy-codes (per-iterate bytes do not exist in-loop —
  registered limitation carried forward).
- Emit-best changes ONLY the post-loop emission state: the in-loop
  trajectory (redistribution, controller, traces) is identical by
  construction. Gate G-TRAJ below verifies this from the traces.

## Arms (v47A only — the #70-item-1 gate winner's model class)

`v47A` bake (`zensim/weights/v47_strict_qat_native_2026-05-27.bin`) ×
{baseline, h3-mag gain 20 clamp 1.35 (the #70-item-1 selected config;
clamp left at the 1.35 default)} × {emit-last, emit-best} ×
budgets {k=6 (primary — the directed matrix), k=12 (secondary — the
diagnosed overshoot regime)}. `JXL_ZENSIM_TARGET_TOL=-1` (no early stop;
full trajectories, as in the prior matrices' R1/inner runs).

8 run labels: `v47A_{base,h3g20c135}_k{6,12}_{last,best}`.

## Matrix and fixtures (#69 set, unchanged)

9 refs × targets {70, 80, 88} = 27 cells/run: city/dog/girl 576²
(`/mnt/v/output/zensim/diffmap-coherence-2026-07-18/`); CID22-512
validation 1025469/1418519/1189261; nonphoto 576² crops
(`-crop 576x576+512+256`) of gb82-sc codec_wiki/gui/imessage (regenerated
if absent). Effort 8, seed distance per `seed_distance_for_target`
(2.5/1.5/1.5 for t70/t80/t88). Judged = decode the emitted bitstream and
score with the SAME bake (as #69 / mm study). 216 encodes total + gates.

## Gates (in-run; failure = stop and fix, never measure through)

- **R0a default-unchanged:** city t80 baseline k6 encoded by (1) the
  MAIN-built binary (pre-change, built from `0eb31edc` in this workspace
  before the edit) and (2) the new binary with `JXL_ZENSIM_EMIT_BEST`
  unset → byte-identical (shas recorded).
- **R0b set-but-last-best:** on a cell whose emit-last trace argmin
  (latest-tie rule) IS the last compare, `JXL_ZENSIM_EMIT_BEST=1` must
  produce a byte-identical bitstream to the emit-last run of that cell.
  Cell selected from the emit-last traces after they exist; both R0b and
  R0c run BEFORE any emit-best measurement arm.
- **R0c engagement:** on a cell whose argmin is strictly before the last
  compare, `JXL_ZENSIM_EMIT_BEST=1` must produce a DIFFERENT bitstream.
- **G-TRAJ:** per cell, the emit-best run's trace must equal the
  emit-last run's trace (scores to print precision) — the emission rule
  must not perturb the trajectory.
- **G-EMIT:** per cell, emit-best's RD_STATS internal score must equal
  the trace's argmin score (latest-tie), and bitstream-changed cells must
  be exactly the cells whose argmin != last compare.

## Endpoints (FROZEN)

- **P1 (primary):** decoded-judged median |achieved − target| per run
  (all 27 cells + per target).
- **P2 (primary):** cells within ±2.0 decoded-judged per run.
- **P3:** bytes — per-cell ratio best/last (same arm × budget), median +
  count of changed bitstreams (a changed emission may cost or save bytes;
  the approach-from-above geometry predicts earlier iterates carry MORE
  bytes).
- **P4:** emitted-iterate index distribution per emit-best run (from
  RD_STATS + traces).
- **S1 (secondary, k12):** P1/P2/P3 at k=12 — the regime where finding #7
  diagnosed the emit-last penalty.

## Expectation (stated before running, from prior trace pricing)

- At k6 the mm study's best-of-≤6 pricing for THESE arms equalled
  as-emitted (18/18 base, 21/21 h3g20c135) → expected P2 delta ≈ 0 and a
  small (possibly null) P1 improvement. A null at k6 is a reportable
  outcome, not a failure.
- At k12 (secondary) the efficiency study measured h3 med min|internal
  err| 0.090 (E5) vs judged-as-emitted ~1.02 (E6, 3-ref subsample) →
  expect a LARGE h3 improvement (order 0.5+ on the median) if the
  internal-best transfers to judged at the E6 ±0.13 photo precision.
- Transfer caveat carried forward: best-of is selected on INTERNAL
  scores; judged deltas are bounded by the per-cell judged−internal
  transfer (±0.13 med photo, up to ~1.5-2.4 nonphoto/h3) — nonphoto
  cells may not realize the internal gain.

## Honesty rules

Every cell reported; nulls and regressions stand. No endpoint or gate is
relaxed after registration. All medians re-derive exactly from the
committed TSVs (`zensim_emitbest_{cells,traces}_2026-07-31.tsv`, this
dir); stat definitions in `scripts/zensim-loop-eff/analyze_emitbest.py`
(median = numpy.median, percentile method='linear').

## Results

_(filled after the runs; protocol above frozen first)_
