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

Data: `zensim_mm_{cells,traces,outer,xmetric}_2026-07-31.tsv` (this dir);
analysis `scripts/zensim-loop-eff/analyze_mm.py` (every table re-derives
exactly from the TSVs). Runs 2026-07-31, ~760 encodes (540 inner + 216
outer + gates), all phases + every in-run gate green. Substrate as
registered: jxl-encoder main `475f50ad` + instruments, zensim `402f4e63`
(clean), bakes sha256 v47A `d0ef7a30…` / shippedB `b6fe5233…` / v02bvls
`5ffb8a2c…` / blend2L `8898301955ac2d40…`.

**Gates:**
- **R0 identity PASS** — no-envs / TRACE on / `CTRL_CLAMP=1.35` /
  `QF_GLOBAL_SCALE=1.0` all byte-identical (sha `12cf08e0…` — the SAME sha
  as the efficiency study's R0 on the older binary + zensim `e8cd105a`:
  cross-study, cross-substrate determinism for the gate cell).
- **Engagement PASS** — h3 probes exactly k×27 (81 at k3, 162 at k6) in
  all 8 h3 runs; all 12 baseline-run probes 0; clamp-1.6 diverges from
  1.35 on ≥1 cell; ssim2-outer zenmetrics calls 108/108 (0 nulls).
- **Mount-equivalence CONFIRMED** — `latest` (named profile) vs shippedB
  (bake-mount): **27/27 bitstreams byte-identical**, decoded scores max
  |Δ| = 0.0000. The `bake:` mount path is exactly the named-B profile.
- **Determinism** — k3-vs-k6 trace prefixes identical (max |Δ| 0.0000
  over 108 compares × 5 baseline models).

### Headline conclusions

1. **Nothing reliably reaches ±2.0 in 3 iterations: the best arm lands
   16/27 cells.** At budget 3 (as-emitted): ssim2-outer 16/27 (its own
   units), shippedB/latest 14/27 and zensimA-outer 14/27, every v47A
   inner arm 13/27, blend2L 11/27, v02bvls 6/27. At k=6 the same arms
   reach 18-22/27 — the efficiency study's "budget-limited, not
   tolerance-limited" verdict extends to every model class and both
   mechanisms. (Cross-arm caveat: each arm's ±2.0 is in its OWN metric's
   units — F5 is the shared-scale read.)
2. **Model class matters more than mechanism at budget 3.** The linear
   B dial (14/27, med t70 |err| 1.24) beats the MLP A dial (13/27, 2.93)
   at k=3 — B's oscillating controller (eff-study E4) gets close fast,
   and best-of-≤3 lifts it to 16/27. Mechanism swap at fixed metric
   (inner-A 13/27 → outer-A 14/27) buys +1 cell for ~4× wall cost
   (214 ms → 849 ms median): the outer controller's advantage is reading
   decoded truth instead of reconstruction-domain estimates, and it is
   small.
3. **#70-item-1 gate verdict: WINNER = gain 20, clamp 1.35** (the
   registered-default clamp). All four co-sweep configs tie F1 at 13/27;
   tie-break median bytes ratio at equal achieved → g20c135 = 0.976
   (n=15) vs 0.989/0.989/0.985. None disqualified (all ratios < 1.0 —
   h3 keeps SAVING bytes at equal achieved, extending #69 G2). The
   baseline-clamp1.6 control ties baseline exactly (13/27, ratio 1.000,
   n=22) — the h3 arms' byte win is steering, not controller-step size.
   Gain 20 also improves k=3 accuracy medians (t70 2.34 vs 2.93, t80
   0.85 vs 1.03) without flipping the within-2 census.
4. **The clamp axis is dead at k=3 and mildly helpful at k=6.** At k=3,
   c1.6 arms are median-identical to their c1.35 counterparts (the clamp
   only binds on far-from-target cells, which stay unreachable-by-3
   anyway). At k=6: basec160 19/27 vs base 18/27; h3g10c16 22/27 vs 21,
   h3g20c16 22/27 vs 21 — consistent +1 cell. The frozen k3 gate is
   unaffected; a budget-6 deployment could prefer c1.6.
5. **F5 — the cross-metric product read: the zensim-targeted loop is
   ~2× tighter in ssim2's eyes than the ssim2-targeted loop is in
   zensim's eyes.** ssim2 IQR of zensim-A-targeted emissions:
   4.66/2.18/1.97 (inner k3, t70/80/88) and 4.25/2.40/1.73 (outer). \
   zensimA IQR of ssim2-targeted emissions: **8.76/5.03/2.69**. Targeting
   zensim yields more consistent ssim2 than the reverse at every target —
   the zensim dial is the safer loop metric when both metrics' opinions
   matter.
6. **Custom mounts work mechanically; dial SHAPE decides usefulness.**
   Both mounts ran (372-probe clean). blend2L ≈ v47A at t80/t88 but
   11/27 overall (soft bottom: t70 med 4.36). v02bvls is the census
   floor (6/27): its compressed top-end (dial p95 ≈ 87.4) makes t88
   structurally unreachable — photos saturate at 84-86 even at k=6
   (err 2.0-3.9). A bake's output-spline geometry, not its SROCC, decides
   whether a target is expressible — check the dial range BEFORE mounting
   a bake as a loop metric.
7. **F6 tail is nonphoto + one photo cell.** As-emitted, 7/27 cells are
   within-2 in NO arm: sc_gui/t70+t80, sc_imessage/t70+t80,
   sc_wiki/t70+t80, cid1418519/t70. Best-of-≤3 trims it to 5 (all
   nonphoto). Same seed-miscalibration mechanism the efficiency study
   identified: budget 3 is spent walking off a ~10-20-point seed offset.
8. **Emission rule matters for oscillating dials.** best-of-≤3 (min
   |err| over iterates 0..3) adds +2 cells for B/latest (14→16) and
   bvls (6→8), ~0 for the one-sided v47A arms. Transfer pricing caveat
   quantified in-study: med (judged − internal) within ±0.16 for every
   k3 arm; max |Δ| 0.35-0.62 for linear dials, up to 2.35 on h3/nonphoto
   (the arms where best-of is trace-priced least reliably).

### F1 — fraction of 27 cells within ±2.0 at budget 3 (primary)

| arm | as-emitted | best-of-≤3 |
|---|--:|--:|
| v47A_base_k3 | 13/27 | 13/27 |
| v47A_basec160_k3 | 13/27 | 13/27 |
| B_base_k3 | 14/27 | 16/27 |
| latest_base_k3 | 14/27 | 16/27 |
| bvls_base_k3 | 6/27 | 8/27 |
| blend2L_base_k3 | 11/27 | 11/27 |
| v47A_h3g10c135_k3 | 13/27 | 13/27 |
| v47A_h3g10c16_k3 | 13/27 | 13/27 |
| v47A_h3g20c135_k3 | 13/27 | 13/27 |
| v47A_h3g20c16_k3 | 13/27 | 13/27 |
| outer_zensimA | 14/27 | 14/27 |
| outer_ssim2 | 16/27 | 16/27 |

k=6 reference (as-emitted / best-of-≤6): v47A_base 18/18, basec160
19/19, B 20/21, latest 20/21, bvls 13/19, blend2L 18/18, h3g10c135
21/22, h3g10c16 22/23, h3g20c135 21/21, h3g20c16 22/23.

### F2 — median decoded |err| at budget 3 (as-emitted | best-of-≤3)

| arm | t70 | t80 | t88 | t70 best | t80 best | t88 best |
|---|--:|--:|--:|--:|--:|--:|
| v47A_base_k3 | 2.93 | 1.03 | 1.98 | 3.17 | 1.42 | 1.85 |
| v47A_basec160_k3 | 2.93 | 1.03 | 1.98 | 3.17 | 1.42 | 1.85 |
| B_base_k3 | 1.24 | 1.16 | 2.16 | 1.33 | 0.64 | 1.98 |
| latest_base_k3 | 1.24 | 1.16 | 2.16 | 1.33 | 0.64 | 1.98 |
| bvls_base_k3 | 4.27 | 2.66 | 3.99 | 3.29 | 2.43 | 3.66 |
| blend2L_base_k3 | 4.36 | 1.51 | 2.12 | 3.47 | 1.43 | 2.16 |
| v47A_h3g10c135_k3 | 2.54 | 0.97 | 1.91 | 2.64 | 1.18 | 1.89 |
| v47A_h3g10c16_k3 | 2.54 | 0.97 | 1.91 | 2.64 | 1.18 | 1.89 |
| v47A_h3g20c135_k3 | 2.34 | 0.85 | 1.99 | 2.98 | 0.95 | 1.89 |
| v47A_h3g20c16_k3 | 2.34 | 0.85 | 1.99 | 2.98 | 0.95 | 1.89 |
| outer_zensimA | 2.78 | 1.22 | 1.87 | 2.78 | 1.22 | 1.87 |
| outer_ssim2 | 2.75 | 1.57 | 1.37 | 2.75 | 1.57 | 1.37 |

(Inner best-of medians can exceed as-emitted medians — the rules read
DIFFERENT scorers (decoded-judged vs internal trace), so per-cell
domination doesn't hold across quantities and cell medians reorder;
e.g. v47A t70: judged 2.93 vs internal-min 3.17. Outer rows have no such
gap — both rules are decoded-judged there.)

### F3 — bytes at equal achieved (|Δachieved| ≤ 0.5) vs v47A_base_k3

| arm | n matched | med bytes ratio | med Δachieved |
|---|--:|--:|--:|
| v47A_basec160_k3 | 22 | 1.000 | +0.000 |
| v47A_h3g10c135_k3 | 17 | 0.989 | −0.177 |
| v47A_h3g10c16_k3 | 16 | 0.989 | −0.199 |
| v47A_h3g20c135_k3 | 15 | 0.976 | −0.091 |
| v47A_h3g20c16_k3 | 13 | 0.985 | −0.073 |
| outer_zensimA | 17 | 0.994 | +0.003 |

### F4 — cost (inner iteration = one COMPARE; outer iteration = one FULL ENCODE)

| arm | med ms/iter | med wall-to-budget-3 |
|---|--:|--:|
| inner baselines (all 6 models) | 36.5-37.4 | 209-216 ms |
| inner h3 arms | 79.9-83.8 (incl. i0 model-gradient amortized) | 386-406 ms |
| outer_zensimA | 150.7 encode + 68.8 scoring | 848.6 ms |
| outer_ssim2 | 142.7 encode + 68.7 scoring | 832.1 ms |

(576²-class; outer wall = 4 full encodes + per-iterate judge + zenmetrics
shell — the scoring ~69 ms/iterate splits ≈ judge + ssim2 shell.)

### F5 — cross-metric spread of budget-3 as-emitted emissions (n=9/target)

| emission arm | other metric | t70 IQR | t80 IQR | t88 IQR | worst stdev |
|---|---|--:|--:|--:|--:|
| v47A_base_k3 (inner) | ssim2 | 4.66 | 2.18 | 1.97 | 4.78 |
| outer_zensimA (j3) | ssim2 | 4.25 | 2.40 | 1.73 | 4.99 |
| outer_ssim2 (j3) | zensimA | **8.76** | **5.03** | **2.69** | 7.16 |

Secondary (ssim2 IQR of every inner k3 arm): v47A_base 4.66/2.18/1.97,
basec160 2.96/2.34/1.97, B=latest 4.90/2.56/1.28, bvls 3.27/1.30/1.66,
blend2L 5.06/2.17/1.78, h3g10c135 4.26/2.78/1.52, h3g10c16
2.18/1.65/1.52, h3g20c135 3.31/2.85/1.80, h3g20c16 2.35/1.78/1.80.

### F6 — never-reached tail (within 2.0 in NO arm at budget 3)

- as-emitted (7): cid1418519/t70, sc_gui/t70, sc_gui/t80,
  sc_imessage/t70, sc_imessage/t80, sc_wiki/t70, sc_wiki/t80
- best-of-≤3 (5): sc_gui/t70, sc_gui/t80, sc_imessage/t70, sc_wiki/t70,
  sc_wiki/t80

### Limitations (registered + observed)

- Inner best-of-≤3 is priced from INTERNAL traces (registered); this
  study's own per-arm transfer stats (headline 8) bound the error —
  worst max |judged−internal| 2.35 on an h3/nonphoto cell.
- F1 across arms compares each metric in its own units (registered);
  F5 carries the shared-scale comparison.
- Outer arms embed one dead-cost zensim iter-1 loop (the actuator lives
  inside the zensim loop) — identical in both outer arms, included in
  the reported encode ms.
- 924-class models out of scope in-loop (extractor-side retention hooks
  absent), as registered. 576²-class fixtures only; no size sweep (loop
  dynamics, not perf calibration).
