# zensim loop METRIC-MATRIX study — model × mechanism at budget 3 (2026-07-31)

Third study in the loop series, extending
`benchmarks/zensim_attr_loop69_2026-07-29.md` (arms/env semantics) and
`benchmarks/zensim_diffmap_efficiency_2026-07-31.md` (trace instrument,
budget-capped runs, bytes-target outer controller, fixture set).
**Pre-registered: every arm, endpoint, and gate below is FROZEN before the
first measurement run. Runs must match this protocol; deviations are
reported as such, never silently substituted.**

**Question:** which (metric model × loop mechanism) reaches decoded-judged
|achieved − target| ≤ 2.0 (in the loop metric's own units) within an
iteration budget of 3, at what byte and wall-clock cost — zensim model
classes vs ssim2.

## Substrate (recorded before running)

- jxl-encoder `main@origin` = `475f50ad` + this study's additive
  instruments (committed on top; shas in Results).
- zensim read-only path dep at `402f4e63` (clean tree; **moved since the
  efficiency study's `e8cd105a`** — every arm re-runs on the current
  substrate; prior-study TSVs are context only, never merged into this
  study's tables).
- ssim2 = SSIMULACRA2, scored by the canonical owner: zenmetrics
  (`~/work/zen/zenmetrics/target/release/zenmetrics batch --metric ssim2`,
  binary of 2026-07-27, CPU `ssimulacra2` crate; ~70 ms per 576² pair
  measured). Decoded-PNG shelling per outer iteration.
- Loop semantics unchanged from the efficiency study: `--iters N` runs
  compares at indices `0..=N` (N+1 compares; index 0 = seed state; last
  executed iteration is compare-only); the emitted bitstream corresponds
  to the quant field measured by the LAST executed compare; the loop never
  entropy-codes (per-iterate bytes do not exist in-loop — registered
  limitation carried forward).
- Seed distance per `seed_distance_for_target` **as in code** (unchanged
  since `f195c8c0`): t ≥ 90 → 0.9, t ≥ 80 → 1.5, else 2.5 — i.e.
  **2.5 / 1.5 / 1.5 for t70/t80/t88**. (The efficiency-study doc's
  "0.9 for t88" was a doc error; 0.9 applies only at t ≥ 90. Same
  function shared by every arm and both mechanisms, so not load-bearing.)
- **924-class models are OUT OF SCOPE in-loop**: the extractor-side
  retention hooks for the folded-append streaming regime do not exist in
  the encoder's zensim loop (documented constraint). 372-contract models
  only.

## Model axis (inner loop scorer via `JXL_ZENSIM_RD_PROFILE`)

The supervisor-verified selector is `JXL_ZENSIM_RD_PROFILE=a|b|latest|
bake:<path>` (`zensim_loop.rs::rd_profile_from_env`). Named profile `a`
returns n_inputs=0 (model-map arms unsupported), so — as in both prior
studies — the A-class and B-class arms mount their bakes explicitly
(`bake:` → `ZensimProfile::Custom` via zensim's `custom-profiles` feature,
already enabled in the jxl dep; n_inputs probed = 372):

| arm id | scorer | class | file (sha-verified in Results) |
|---|---|---|---|
| v47A | `bake:` v47_strict_qat_native_2026-05-27.bin | MLP 372 | zensim/weights/ (27,316 B) |
| shippedB | `bake:` b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin | linear 372 | zensim/weights/ (7,325 B) |
| latest | named `JXL_ZENSIM_RD_PROFILE=latest` | = ZensimProfile::B | mount-equivalence CONTROL (see below) |
| v02bvls | `bake:` v02_bvls_NO_shaping_2026-05-28.bin | linear 372 (86 active w) | /mnt/v/output/zensim/bakes/ (8,622 B) |
| blend2L | `bake:` mlp_2L_diverse_H128_2026-07-15.bin | 2-layer MLP 372-128-128-1 | /mnt/v/output/zensim/reports/b_negatives/ (271,900 B, sha256 8898301955ac…) |

**Custom-mount verdict (pre-run investigation):** mounting works — the
`bake:` path has existed since 2026-07-18 and is how the two prior studies
ran v47A/shippedB. The two candidate bakes from the cookbook/memory chain
(v02-bvls 8.6 KB linear; r3-2L-H128 2-layer blend) are on disk and
sha-verified, so both are ADDED as inner-loop model arms as the task
directs.

**`latest` resolution (verified in zensim source at `402f4e63`):**
`ZensimProfile::latest_preview()` = `ZensimProfile::B`, and `PROFILE_B`'s
dispositions (skip_score_mapping=true, extrapolate_score=true,
extended_features=true, compute_iw_features=true, mlp = the same
b_sdr_linear bake bytes) match the `bake:` mount builder exactly. The
`latest` arm is therefore registered as a **mount-equivalence control**:
its bitstreams are EXPECTED byte-identical to shippedB's; any divergence
is a mount-path finding, reported, never ignored. It also exercises the
named-profile judge path end-to-end.

## Mechanism axis

- **INNER** — the native zensim loop: `JXL_ZENSIM_TARGET_SCORE=<t>`,
  `JXL_ZENSIM_TARGET_TOL=-1` (no early stop; full trajectories),
  budget-capped at **k=3** (primary; 4 compares, 3 redistribution +
  controller steps) plus the **k=6 reference run** (7 compares, the prior
  studies' budget).
- **OUTER** — full-encode-per-iteration controller on the env actuator
  `JXL_ZENSIM_QF_GLOBAL_SCALE` (transplanted from the efficiency study's
  E7, damping as-is): per cell, encodes j = 0..3 with g₀ = 1.0 and
  g_{j+1} = g_j × clamp(((100 − s_j)/(100 − t))^0.6, 1/1.35, 1.35),
  where **s_j is the DECODED-judged score of encode j in the arm's
  metric** (zensim-A judge in-process, or ssim2 via zenmetrics on the
  decoded PNG). Budget 3 = three controller steps after the seed encode
  (4 full encodes), the step-count parallel of inner k=3.
  - The actuator lives inside the zensim loop (gated `zensim_iters > 0`),
    so outer encodes run the inner loop at **zensim_iters=1,
    redistribution-only** (`JXL_ZENSIM_TARGET_SCORE` unset → no in-loop
    controller; one profile-independent Trained-diffmap sum-preserving
    redistribution + 2 compares per encode). This is the minimum
    engagement that applies the scale; the identical configuration runs
    in BOTH outer arms, so the comparison is fair; the in-encode zensim
    compare overhead is a recorded harness artifact of where the actuator
    lives, honestly labelled in F4.
  - Arms: **zensimA-outer** (metric = v47A bake judge — isolates
    mechanism-vs-metric: inner-A vs outer-A) and **ssim2-outer** (metric =
    SSIMULACRA2; targets {70, 80, 88} in ssim2's own units).
  - Both outer arms also record the OTHER metric per iterate (ssim2 via
    zenmetrics / zensim via the v47A judge) — feeds F5 symmetrically.

## Arms (10 inner × {k3, k6} + 2 outer; 27 cells each)

Inner, all on the 9-ref × {70, 80, 88} matrix:

1. v47A × baseline
2. v47A × baseline × **clamp 1.6** — attribution CONTROL for the co-sweep
   (separates "bigger controller steps" from "steering"); not a gate
   competitor
3. shippedB × baseline
4. latest × baseline (mount-equivalence control)
5. v02bvls × baseline
6. blend2L × baseline
7. v47A × h3-mag, gain 10, clamp 1.35 (registered default)
8. v47A × h3-mag, gain 10, clamp 1.6
9. v47A × h3-mag, gain 20, clamp 1.35
10. v47A × h3-mag, gain 20, clamp 1.6

7–10 are the pre-registered #70-item-1 co-sweep (`ZENSIM_H3_GAIN` ×
controller per-step clamp). The controller clamp (1.35 hardcoded at
`zensim_loop.rs` in the damped global step) is **not currently
env-tunable** → this study adds `JXL_ZENSIM_CTRL_CLAMP` (env-gated,
default 1.35; unset/1.35 = shipped behavior, R0 byte-identity gated).

Outer: 11. zensimA-outer; 12. ssim2-outer.

Encode budget: 10×2×27 = 540 inner encodes + 2×27×4 = 216 outer full
encodes (+R0), ~10-15 min nice'd at 576².

## Fixtures (#69 set, unchanged)

city/dog/girl 576² (`/mnt/v/output/zensim/diffmap-coherence-2026-07-18/`);
CID22-512 validation 1025469/1418519/1189261
(`~/work/codec-corpus/CID22/CID22-512/validation/`); nonphoto 576² crops
(`-crop 576x576+512+256`) of gb82-sc codec_wiki/gui/imessage. Effort 8.
Judged = decode the emitted bitstream, score with the SAME scorer that
drove the loop (bake judge for bake arms; named-B judge for latest;
zenmetrics ssim2 for ssim2-outer).

## New instruments (additive, env-gated, default-off; committed BEFORE runs)

1. `JXL_ZENSIM_CTRL_CLAMP=<c>` — in-loop controller per-step clamp
   override (default 1.35).
2. Harness `--bake profile:<a|b|latest>` — named-profile loop scorer +
   judge (judge via the named profile instead of Custom-from-bytes).
3. Harness outer mode `--score-targets-outer` (metric zensim|ssim2): the
   controller above, one TSV row per outer iterate (g, bytes, judged in
   the arm metric, cross-metric score, ms).
4. Runner `scripts/zensim-loop-eff/run_mm.sh` + analysis
   `scripts/zensim-loop-eff/analyze_mm.py` (extending the eff-study
   tooling conventions; run_eff.sh phases untouched).

## Gates (in-run; failure = stop and fix, never measure through)

- **R0 default-unchanged byte-identity** (city, t80, v47A baseline, k=6):
  (a) no instrument envs, (b) TRACE on + CTRL_CLAMP unset, (c)
  `JXL_ZENSIM_CTRL_CLAMP=1.35`, (d) `JXL_ZENSIM_QF_GLOBAL_SCALE=1.0` —
  all four bitstreams byte-identical.
- **Engagement** (the MODEL_MAP fallthrough hazard, per #69): per
  arm-run `JXL_ZENSIM_ATTR_PROBE` — h3 arms must emit exactly
  k×27 probe lines (3 steered compares/cell at k=3, 6 at k=6); every
  baseline arm must emit 0. Clamp-1.6 arms must diverge from their 1.35
  counterparts on ≥1 cell (else the knob didn't engage). latest-vs-
  shippedB per-cell bitstream sha equality is REPORTED either way
  (expected identical).
- ssim2-outer must call zenmetrics successfully on every iterate (a
  failed call is a recorded null, never a silent skip).

## Endpoints (FROZEN)

- **F1 (primary):** fraction of the 27 cells with decoded-judged
  |achieved − target| ≤ 2.0 by budget 3, per arm — for BOTH emission
  rules: **as-emitted** (inner: the k=3 run's decoded-judged score;
  outer: encode j=3's judged score) and **best-of-≤3** (inner: min
  |score_i − t| over i ∈ 0..3 from the k=3 run's internal trace — priced
  from traces per the efficiency study, which measured judged-vs-internal
  transfer at ±0.13 med / ±1.5 max-nonphoto (E6), stated with results;
  outer: min |judged_j − t| over j ∈ 0..3, decoded-judged by construction
  — the asymmetry is inherent to the mechanisms and stated).
- **F2:** median decoded |err| at budget 3, per arm × target (as-emitted
  primary; best-of context).
- **F3:** bytes ratio vs the same-model baseline at equal achieved
  (|Δachieved| ≤ 0.5 matching, as #69 G2), k=3 as-emitted: h3 co-sweep
  arms + clamp control vs v47A-baseline; zensimA-outer (j=3) vs
  v47A-baseline (k=3).
- **F4:** cost — ms per iteration and total ms to budget 3, honestly
  labelled: an inner iteration is one compare (~tens of ms; trace +
  loop_ms + encode_ms reported); an outer iteration is a FULL ENCODE
  (+judge/zenmetrics ms reported separately). Both per-iteration and
  wall-clock-to-done.
- **F5 (cross-metric consistency):** for zensim-targeted budget-3
  as-emitted emissions, the ssim2 spread at each target; for
  ssim2-targeted emissions, the zensim(v47A-judge) spread at each target.
  Primary comparator arms: v47A-baseline (inner k3), zensimA-outer (j3),
  ssim2-outer (j3); all other inner arms' ssim2 readings reported as
  secondary context. Spread stats frozen: IQR (p75−p25, numpy linear
  interpolation) primary, plus stdev, min/max, n=9 per (arm × target).
  This is the "which loop metric is tighter in the other's eyes" product
  read.
- **F6:** never-reached tail — cells not within 2.0 by budget 3 in ANY
  arm (per emission rule).

### #70-item-1 selection gate (FROZEN NOW, before any run)

Among the four h3 co-sweep configs {(10,1.35), (10,1.6), (20,1.35),
(20,1.6)}: **winner = highest fraction of the 27 cells within 2.0 by k=3
(as-emitted, decoded-judged); tie-break = lower median bytes ratio vs
v47A-baseline at equal achieved (±0.5); DISQUALIFIED if that median
bytes ratio > 1.02** (bytes at equal achieved > +2% vs baseline). The
baseline-clamp-1.6 control is reported alongside for attribution (a
winner whose F1 the control matches is flagged "controller-step effect,
not steering"), but does not compete in the gate.

## Honesty rules

Every cell reported; ties and nulls stand. If latest/a custom mount/an
outer call fails to run, the failure is recorded and the study continues.
The 2.0 / 3-iteration criterion is never relaxed. All medians/fractions
must re-derive exactly from the committed TSVs
(`zensim_mm_{cells,traces,outer,xmetric}_2026-07-31.tsv`, this dir);
stat definitions live in `analyze_mm.py`'s docstring (median =
numpy.median, percentile method='linear').

## Results

_(filled in after the runs; protocol above frozen first)_
