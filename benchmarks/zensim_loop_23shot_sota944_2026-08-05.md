# zensim loop 2/3-SHOT — SOTA-944 candidate panel (2026-08-05)

Fifth study in the loop series; the direct extension of
`zensim_loop_23shot_2026-08-01.md` (protocol inherited verbatim). **Purpose:** the
sota944 campaign's wave-11 candidate has never entered a loop-targeting study — the
last "uses" axis on its freeze surface. This study adds the candidate as one arm to
the existing 2/3-shot panel and regenerates the machine summary the zensim gauntlet
reads.

**Question:** does `W10L9_s4003_packed` (the 944-class pruned candidate) hit
decoded-judged targets within ±2.0 (its OWN units) at loop budget k=2/k=3 at least
as well as the incumbents (v47A, shippedB)?

## Candidate + controls

- **Candidate**: `/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin`
  (165,696 B; wave-11 battery-selected, k=8-confirmed recipe; PRUNED:
  `caller_input_width()` = 944, internal `n_inputs()` = 667 via
  `FeatureTransform::Drop`; dial mono 99.32 %, M3a 0.862). Board name
  `W10L9_s4003_packed`. Arm key **`W10L9_base`** (baseline mode — the same loop
  rule as every `*_base` arm; the score model is the only variable).
- **Controls**: the committed 2026-08-01 panel rows — v47A_base, B_base, bvls_base,
  blend2L_base, v47A_h3g20c135, outer_zensimA, outer_ssim2 — CARRIED, not re-run,
  gated by the substrate probe below.

## Registered outcomes (pre-committed before any run)

On k2/k3 emit-best within-±2 counts (and the medians as tie-breaks), candidate vs
v47A_base and B_base:

- **(a) beats both** ⇒ the candidate is the loop-recommended model; the last freeze
  axis fills in green.
- **(b) parity** (within ±1 cell of the better incumbent, medians comparable) ⇒
  fine — the candidate is no worse where B was the incumbent.
- **(c) worse** ⇒ the loop axis stays with the incumbents and the freeze surface
  says so. Honest nulls stand; nothing ships or swaps on this study alone.

Also recorded (cheap, registered): the candidate's per-compare loop cost vs v47A
from the SAME-session fresh runs (`loop_ms`, `ms_per_compare` medians — the probe
re-runs v47A k3 fresh, so both sides are measured on this box today). The zensim
perf work (forward ~25× faster; pruning −25 % forward) is NOT assumed to show up
end-to-end — the loop's 944 route pays a second walk (see mechanics), and the
number reported is the measured end-to-end one.

## Integration mechanics (jxl-encoder changes, registered before running)

The existing `bake:` mount extracts 372-class features and would fail the forward
for a 944-class bake — and the loop swallows compare errors
(`Err(_) => return Ok(current_params)`), silently emitting the seed. The changes:

1. **Width probe fixed + loud**: `rd_infer_n_inputs` probes
   `[156, 228, 300, 372, 720, 924, 944]` smallest-first over a 944-long zero
   vector (smallest-first preserves the ≤372 arms' resolved widths exactly;
   `bake:` mount PANICS if no width is accepted — no silent seed-emit).
2. **`JXL_ZENSIM_MODEL_MAP` loud guard**: unknown values PANIC (previously fell
   through to baseline silently — the loop69 hazard); map-steering arms with a
   folded-class (≥720) bake PANIC (the fused compare is 372-class only —
   registered limitation, not silently downgraded).
3. **Folded-class score route** (`rd_n_in ≥ 720`): per iterate, the score comes
   from the canonical folded extraction — 944 → `compute_folded720_append2_features`,
   924 → `compute_folded720_append_features`, 720 → `compute_folded720_features` —
   over (source, recon) as `LinearF32Rgba` StridedBytes, forwarded through
   `zensim::score_features_with_profile` (full bundle: transforms incl. Drop,
   spline, extrapolation; sizes by `caller_input_width()` — the metric.rs
   caller-width fix class, verified). The redistribution map stays the standard
   Trained diffmap computed from a no-mlp profile whose walk config (weights,
   num_scales=4, blur 5/1) is bit-identical to the mlp mounts' — the map and loop
   rule are IDENTICAL to every `*_base` arm; only the controlling/judging score
   model differs. Cost consequence (registered): the folded route runs the v1
   diffmap walk AND the 944 extraction per compare — there is no fused 944 entry
   (C3a fusion is v1/372-class) — so the candidate's per-compare cost carries a
   structural second pass, reported as measured.
4. **Judge**: `zensim_diffmap_rd`'s judge mounts the same bake; for folded-class
   bakes it judges via the same canonical folded extraction on (ref, decoded) u8
   sources + `score_features_with_profile`. Same-scorer judging contract
   unchanged.
5. `Cargo.toml`: zensim dep gains `feature-regime-v2` (compile-time only; no
   shipped-path behavior change — gated by the substrate probe).

## Cells

Standard 9-ref × {70, 80, 88} matrix (city/dog/girl 576², CID22-512 val
1025469/1418519/1189261, gb82-sc 576² crops sc_wiki/sc_gui/sc_imessage; effort 8;
seed distance 2.5/1.5/1.5; `JXL_ZENSIM_TARGET_TOL=-1`, full budget, no early stop).

**FRESH (this study):** `W10L9_base` × {k2_last, k2_best, k3_last, k3_best} = 4
runs = 108 encodes + the 27-encode probe. k3_last is run fresh (the candidate has
no mm-study rows to derive from).

**CARRIED:** every 2026-08-01 summary entry, provenance preserved.

## Gates (in-run, hard unless marked)

- **SUBSTRATE PROBE** (hard): re-run `v47A_base_k3` (emit-last, 27 cells) on the
  CURRENT substrate; `analyze_23shot.py verify` against the committed mm rows —
  27/27 cells equal (achieved_decoded/abs_err/bytes) and 108/108 trace compares
  equal. This simultaneously (i) validates carrying the 2026-08-01/mm rows across
  the substrate move, and (ii) is the R0-identity gate proving the integration
  changes above do NOT alter the 372-class loop. Any mismatch ⇒ STOP (fresh re-run
  of all control arms required; derivations invalid).
- **Engagement** (hard): candidate runs are baseline-mode — `JXL_ZENSIM_ATTR_PROBE`
  files must have 0 lines; trace files must carry exactly 27×(k+1) compare rows.
- **Emit-best k2/k3 A/B** (report): count of candidate bitstreams differing
  best-vs-last (0 = argmin==last everywhere, legal at low k).
- **Summarize anchors** (hard): the analyze owner re-derives every carried number
  and asserts the published mm F1@k3 + outer j3 counts; candidate rows must be
  27/27 per mode with `abs_err` consistent with |achieved−target| ≤ 5e-3.

## Outputs

- `benchmarks/zensim_loop_23shot_sota944_2026-08-05.tsv` — fresh per-cell rows
  (probe + candidate; target_ab schema with `run` prefix, as 2026-08-01).
- `benchmarks/zensim_loop_23shot_summary_2026-08-05.json` — the regenerated
  machine summary (schema `zensim_loop_23shot_summary.v1`): all 2026-08-01 models
  (provenance carried) + `W10L9_base` (4 fresh modes). The zensim gauntlet's
  `--loop-targeting` default moves to this file; `LOOP_BAKE_MAP` gains
  `W10L9_base → W10L9_s4003_packed`.
- Runner: `scripts/zensim-loop-eff/run_23shot_sota944.sh`; stats owner:
  `scripts/zensim-loop-eff/analyze_23shot.py` (extended with `--extra-arm`/
  `--extra-cells` — the owner is extended, never forked).

## Results

*(filled after the runs; gates first, then the panel table, then the cost read,
then which registered outcome fired)*

## Limitations (registered)

- Own-units caveat as 2026-08-01: within-±2 counts are in each arm's own metric
  units; candidate-vs-incumbent counts compare *target-hitting in own units*, not
  a shared perceptual scale.
- The candidate's loop route pays a structural second pass (no fused 944 entry);
  its cost number is an integration-surface fact, not a model-capability fact.
- The candidate cannot run map-steering arms (h3 etc.) — 372-class fused only;
  guarded loudly, listed as the map-steering axis's registered gap.
- 576²-class fixtures only; loop-dynamics study, not a perf calibration.
- In-loop features come from LinearF32 recon; judge features from decoded u8 —
  same relationship every arm has (the emit-best study's judged−internal transfer
  caveat applies to the emit-best selection channel).
