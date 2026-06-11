# MABSplit Phase-0 Variance Report (issue #64, CHUNK 2 Phase 0)

**Date**: 2026-06-11 · **Verdict: VARIANCE FAILURE — MAB-style property
pruning in `find_best_split` has little headroom; Phases 1–3 are NOT
sanctioned on this evidence.**

## Question

Can a multi-armed-bandit / successive-halving scheme prune losing
*properties* early in `find_best_split` (evaluate candidates on partial
sample data, drop hopeless ones, finish only the contenders), per the
MABSplit literature? Phase 0 asks the prerequisite: **how decisively does
the winning property separate from the runner-up?** If wins are mostly
near-ties, candidates must be evaluated to (near-)completion for the
ranking to stabilize, and a bandit saves nothing.

## Method

Instrumentation: `JXL_MABSPLIT_DUMP=<path>` (behind
`__env_var_diagnostics`, per the lossy-low hygiene rule) — both
`find_best_split` variants append one line per node: `weighted_total`,
`base_bits`, chosen property (−1 = no split beat base), `best_bits`, and
each evaluated property's best split total. Analysis:
`scripts/mabsplit_phase0_analyze.py`. Per decided node:

    gain_p     = base_bits − best_total_p
    margin     = gain(winner) − gain(runner-up)
    margin_rel = margin / gain(winner)

`margin_rel → 1` means the winner dominates (prunable field);
`margin_rel → 0` means a near-tie (pruning would mis-rank). "Decisive" =
`margin_rel > 0.5`.

Cells: lossless e7, current main (`acec55a8` lineage), 5 content classes.

## Results

| cell | nodes | no-split | med props | winner-gain p50 (bits) | margin_rel p25 | p50 | p75 | decisive (>0.5) |
|---|---|---|---|---|---|---|---|---|
| clic097 (photo 1 MP) | 1149 | 17 | 6 | 66 | 0.094 | 0.209 | 0.365 | 13.5 % |
| tokyo (photo 12 MP) | 8320 | 47 | 5 | 65 | 0.085 | 0.190 | 0.360 | 13.3 % |
| terminal (screen) | 280 | 10 | 7 | 75 | 0.100 | 0.260 | 0.412 | 15.9 % |
| noaa (document) | 1937 | 67 | 4 | 72 | 0.066 | 0.175 | 0.373 | 17.0 % |
| plot (graphic, 14-col) | 147 | 5 | 5 | 87 | 0.156 | 0.335 | 0.589 | 28.9 % |

Overall (n = 11,687 decided nodes): margin_rel p25 = **0.084**, p50 =
**0.193**, p75 = **0.366**.

## Interpretation

- The median node's winning property beats the runner-up by only ~19 % of
  its own gain, and a quarter of nodes sit under 8.4 % — the property
  ranking is dominated by near-ties on **every** content class (photos,
  screens, documents; graphics are mildly more separable at 29 % decisive
  but are also the cheapest class to begin with).
- These margins are measured on FULL sample data. A Hoeffding/empirical-
  Bernstein bound at partial samples adds estimation noise ON TOP of these
  thin margins: to rank winner vs runner-up with useful confidence, the
  bound width must shrink below the margin — for ~75 % of nodes that means
  evaluating close to all samples anyway. The bandit's savings are
  confined to the ~13–29 % decisive minority, and per the 2026-06-11
  step-0 profile, find_best_split's whole ceiling on the affected photo
  cells is 12.8–25.3 % of wall (and that share itself swings ±10 pp
  run-to-run — see `perf_gather_profile_2026-06-10.meta` ADDENDUM 4).
- Consistent with the issue #64 framing: Phase 1–3 (bandit implementation,
  strategy tier, content routing) are **not sanctioned** by Phase 0.
  The honest-stop condition ("documented variance failure + measurement")
  is met.

## What WOULD revive this

- A margin distribution measured at partial samples (Phase-1-style probe)
  showing early estimates are far LESS noisy than Hoeffding worst-case —
  empirical-Bernstein with tiny per-node variance could still prune the
  bottom half of properties at small n. Cheap probe, only worth running if
  find_best_split's wall share grows again.
- Pruning PREDICTORS rather than properties (num_pred is already
  count-pruned at 4/7/10 tiers; a finer scheme was not measured here).

## Reproduction

```
cargo build --release -p jxl-encoder-cli --features __env_var_diagnostics
JXL_MABSPLIT_DUMP=/tmp/cell.tsv ./target/release/cjxl-rs IMG out.jxl --lossless -e 7 --threads 8
python3 scripts/mabsplit_phase0_analyze.py name=/tmp/cell.tsv
```
