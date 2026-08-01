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

Data: `zensim_emitbest_{cells,traces}_2026-07-31.tsv` (this dir); analysis
`scripts/zensim-loop-eff/analyze_emitbest.py report` (every number below
re-derives exactly from the committed TSVs). Runs 2026-07-31, 216 encodes
+ gates, ~2 min nice'd. Substrate as registered: jxl-encoder `0eb31edc` +
instrument commit `b4b6fd95`, zensim `402f4e63` (clean), v47A bake.

**Gates — all green:**

- **R0a PASS** — env unset, new binary vs pre-change MAIN binary:
  byte-identical (sha `12cf08e0…` — the SAME gate-cell sha as the
  efficiency and metric-matrix studies' R0: cross-study determinism).
- **R0b PASS** — `JXL_ZENSIM_EMIT_BEST=1` on a last-is-best cell
  (cid1025469 t70 baseline k6): byte-identical to emit-last.
- **R0c PASS** — engagement on an overshoot cell (cid1025469 t70 h3 k12):
  bitstream differs.
- **G-TRAJ PASS** — all 4 run-pairs, 27/27 cells: emit-best traces equal
  emit-last traces exactly (max |Δscore| = 0) — the emission rule does
  not perturb the trajectory.
- **G-EMIT PASS** — all 4 emit-best runs: RD_STATS internal score equals
  the trace argmin score (0 mismatched cells), and the sha-changed cell
  set equals the argmin≠last set exactly (0 mismatches).

### Headline conclusions

1. **At k6 (primary) the within-2 census is UNCHANGED — exactly the
   pre-registered expectation from the mm-study trace pricing** (base
   18/27, h3g20c135 21/27, both emission rules). The medians still
   improve: base 1.174 → 1.052, h3 0.926 → 0.745 all-cells; h3 t70
   0.926 → 0.393 and t80 0.472 → 0.214 (the h3 sweet spot lands mid-run
   and emit-best keeps it).
2. **At k12 (secondary — the diagnosed overshoot regime) emit-best is a
   large win**: h3 med |err| 0.747 → **0.382** and census 22 → **25/27**;
   baseline 0.750 → **0.432**, census 23 → **25/27**. Finding #7's
   emit-last penalty (h3 judged 0.59@k6 → 1.02@k12 on the E6 subsample)
   is cured — extended budgets now HELP instead of hurting: h3 k12-best
   is the best arm measured in this series (t70 0.382 / t80 0.154 /
   t88 0.583 medians).
3. **Bytes are neutral-to-slightly-saving**: med best/last ratio 1.0000
   (both arms, k6), 0.9920/0.9928 (k12); per-cell range 0.914-1.059.
   Emit-best does not buy accuracy with size.
4. **Transfer caveat observed as registered**: base k6 t70 med worsened
   slightly (1.867 → 1.921) — the internal argmin can pick an iterate
   whose JUDGED error is marginally worse (E6 ±0.13 photo transfer);
   every other arm × target median improved or tied.
5. **Emitted-iterate distribution (P4)**: at k6 the argmin IS the last
   compare for 21/27 (base) / 15/27 (h3) cells; at k12 only 10/27 and
   6/27 — the longer the budget, the more emit-best engages (h3 k12 med
   emitted iterate 9 of 12).

### P1/P2 — decoded-judged med |err| (all + per target) and within-2 census

| run | med all | t70 | t80 | t88 | within2 |
|---|--:|--:|--:|--:|--:|
| v47A_base_k6_last | 1.174 | 1.867 | 0.633 | 1.174 | 18/27 |
| v47A_base_k6_best | 1.052 | 1.921 | 0.633 | 1.052 | 18/27 |
| v47A_base_k12_last | 0.750 | 1.049 | 0.641 | 0.633 | 23/27 |
| v47A_base_k12_best | 0.432 | 0.353 | 0.373 | 0.633 | 25/27 |
| v47A_h3g20c135_k6_last | 0.926 | 0.926 | 0.472 | 1.178 | 21/27 |
| v47A_h3g20c135_k6_best | 0.745 | 0.393 | 0.214 | 1.178 | 21/27 |
| v47A_h3g20c135_k12_last | 0.747 | 1.745 | 0.747 | 0.692 | 22/27 |
| v47A_h3g20c135_k12_best | 0.382 | 0.382 | 0.154 | 0.583 | 25/27 |

### P3 — bytes best/last (per-cell join)

| arm × budget | med ratio | min | max | bytes-differ cells |
|---|--:|--:|--:|--:|
| base k6 | 1.0000 | 0.9723 | 1.0422 | 6/27 |
| base k12 | 0.9920 | 0.9501 | 1.0000 | 17/27 |
| h3g20c135 k6 | 1.0000 | 0.9989 | 1.0587 | 12/27 |
| h3g20c135 k12 | 0.9928 | 0.9136 | 1.0550 | 21/27 |

### Limitations (registered + observed)

- Best-iterate selection reads the INTERNAL score (registered design —
  no in-loop decoded scoring); judged deltas are bounded by the
  judged−internal transfer, and one arm×target median (base k6 t70)
  regressed by 0.05 through that channel.
- 576²-class fixtures only; loop-dynamics study, not a perf calibration.
- k12 is a SECONDARY endpoint (the primary directed matrix is k6); its
  large wins are consistent with the efficiency study's E5/E6 but were
  measured here on the full 27-cell matrix, not E6's 9-cell subsample.
