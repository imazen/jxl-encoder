# zensim loop BEATS-BUTTER consolidation — binned integration gate + clamp sweep (2026-08-07)

Sixth study in the loop series; direct extension of
`zensim_loop_ctrl_arms_2026-08-06` (appendix Y Part-2) and
`zensim_loop_23shot_sota944_2026-08-05` (protocol inherited verbatim).
**Registered BEFORE any run** (this section committed ahead of results).

## Purpose

The user directive: *"integrate fully with and iterate towards a jxl perfect
model better than the butter loop."* Three consolidation steps:

1. **Binned-attribution integration gate** — zensim `d0f624eb` (Level-2 binned
   accumulation) is now wired into the loop (`ZENSIM_ATTR_BIN`, default **8**;
   JXL var-DCT tiles are 8-px aligned, so every steering `query_rect` is
   bin-exact). Verify the frontier arm reproduces under it and price the
   per-compare delta.
2. **Clamp × exp100 sweep** — the 7-miss residue of the frontier arm is the
   nonphoto overshoot class (sc_gui/sc_wiki t70/t80 + cid1418519/t70: seeds
   land at achieved 86-91 for t70 and the 1.35-clamp controller cannot descend
   far enough in 2-3 steps). The clamp arms were swept ONLY at the shipped
   exp 0.6 (cl120/cl160, both ≤17/27); exp 1.0 × larger clamp is untested.
3. **Defaults adoption** — on passing gates, land `CTRL_EXP` default 1.0 (+
   winning clamp) in `zensim_loop.rs`; the Y study left it recommendation-only.

## The comparator ("the butter loop")

`outer_zensimA` (committed summary, provenance `zensim_mm_outer_2026-07-31`):
butteraugli-distance-driven encodes inside an outer zensim-judged distance
search — the butter-loop way to hit a zensim target. Its panel numbers:
**j2 12/27 (med 3.085) · j3 14/27 (med 1.942)**, at one FULL encode per outer
step. The frontier inner arm (committed ctrl-arms TSV, candidate bake
`W10L9_s4003_packed` + h3-mag + `CTRL_EXP=1.0`):
**k2 17/27 (med 1.395) · k3 20/27 (med 0.564)** at inner-compare cost.

## Cells (frozen; series-identical)

9 refs (city/dog/girl 576², CID22-512 val 1025469/1418519/1189261, gb82-sc
576² crops wiki/gui/imessage) × targets {70, 80, 88}; effort 8; seed distance
2.5/1.5/1.5; `JXL_ZENSIM_TARGET_TOL=-1`; `JXL_ZENSIM_EMIT_BEST=1`; stats owner
`analyze_23shot.cells_stats`. Arm recipe `exp100` = `--arms h3-mag`, candidate
bake, `JXL_ZENSIM_CTRL_EXP=1.00` (clamp default 1.35 unless swept).

## Registered runs

- **BINGATE**: `exp100` × {k2, k3} × {`ZENSIM_ATTR_BIN=1`, `ZENSIM_ATTR_BIN=8`}
  (4 × 27 cells).
- **CLAMPSWEEP**: `exp100` (bin 8) × `JXL_ZENSIM_CTRL_CLAMP` ∈ {1.6, 2.0, 2.5}
  × k3 (3 × 27); the winner (if any) re-run at k2.

## Registered gates + outcomes

- **G-BB1 (hard, substrate)**: the `bin=1` k3 run must reproduce the committed
  `exp100_k3` census/median (20/27 · 0.564) and k2 (17/27 · 1.395) on this
  substrate. Mismatch ⇒ STOP; diagnose before any claim (the integration or
  the substrate moved the 372-class-adjacent loop).
- **G-BB2 (adoption)**: `bin=8` within ±1 census cell of `bin=1` at BOTH
  budgets and med |err| within ±0.15. Pass ⇒ `ZENSIM_ATTR_BIN=8` stays the
  default (the integration claim). Fail ⇒ default reverts to 1 pending
  diagnosis; the binned maps are ≤1e-5-different so any census move is
  tile-decision flips at clamp boundaries — count them from traces.
  Also record ms/compare bin8-vs-bin1 (expectation: modest — trim+SAT
  elimination; the fused walk dominates).
- **G-BB3 (clamp)**: a clamp arm WINS iff k3 census ≥ 20/27 AND the nonphoto
  class census strictly improves (currently 2/9 at k3 for exp100... read from
  the fresh bin-8 exp100 run) AND photo census drops by at most 1. Winner
  confirmed at k2 before adoption. No winner ⇒ clamp stays 1.35; honest null.
- **G-BB4 (the beats-butter verdict)**: final best arm vs `outer_zensimA`:
  report census/med|err|/med bytes at k2-vs-j2 and k3-vs-j3 + cost basis
  (median ms/compare × compares vs outer full re-encodes). The claim
  "better than the butter loop" requires BOTH budgets to beat the outer arm's
  census AND median. (On committed data this already holds for exp100; this
  study's job is to hold it under the binned default on today's substrate,
  then extend the margin.)
- **Defaults (adoption, gated on G-BB1+G-BB2)**: `JXL_ZENSIM_CTRL_EXP`
  default 0.6 → **1.0**; clamp per G-BB3; both landed with the evidence
  chain in the commit message. h3-mag remains opt-in via
  `JXL_ZENSIM_MODEL_MAP` (flipping the silent product default from
  Trained-diffmap redistribution is NOT claimed by this study).

## Outputs

`benchmarks/zensim_loop_beatbutter_2026-08-07.tsv` (all fresh cells),
this doc's Results section, regenerated
`zensim_loop_23shot_summary_2026-08-07.json` via the analyze owner
(`--extra-arm`; the gauntlet `--loop-targeting` default moves on adoption),
runner `scripts/zensim-loop-eff/run_beatbutter.sh`.

## Results

Runs 2026-08-07 (~297 fresh encodes: bingate 4×27 + clampsweep 3×27 +
confirmations 2×27 + emit-last legs 2×27; nice'd, box CONTENDED — HDR encode
containers at load ~18/28, so ms columns are load-sensitive as registered in
the series; value columns deterministic). Substrate: jxl `646967a3`
(binned integration + this registration), zensim `d0f624eb` (path dep,
origin/main). Data: `zensim_loop_beatbutter_2026-08-07.tsv` (405 rows, 15
runs × 27 cells); regenerated summary
`zensim_loop_23shot_summary_2026-08-07.json`.

### Gates

- **G-BB1 PASS (exact)** — bin=1 reproduces the committed exp100 to the
  digit: k3 20/27 med 0.564, k2 17/27 med 1.395.
- **G-BB2 PASS (exact)** — bin=8 is IDENTICAL to bin=1 at both budgets
  (census, medians, med bytes all equal): the binned maps produced the same
  tile decisions everywhere. `ZENSIM_ATTR_BIN=8` stays the default.
  Per-compare ms delta is inside contended-box noise (±8% both directions at
  576²) — the binned time win is the large-image regime (zensim L2 measured
  −19% e2e at 12 MP); the 576² win is retained memory + decision-identity.
- **G-BB3 WINNER: clamp 2.00** — dose-response 1.35→20/27, 1.60→21/27,
  **2.00→24/27**, 2.50→23/27 (peak interior). Nonphoto class 3/9 → **7/9**
  with photo unchanged (17/18); med |err| 0.564 → 0.535. **k2 confirmation
  PASS**: 18/27 med 1.194 (vs 17/27 · 1.395 at clamp 1.35 — improves k2
  too). Controller-only confirmation: baseline arm at exp1.0+cl2.0 posts
  20/27 med 0.827 (vs 17/27 · 0.834 at cl1.35) — the default flip helps the
  UNSTEERED path as well (+3 census, median neutral).
- **G-BB4 — BEATS THE BUTTER LOOP at both budgets:**

| | census | med \|err\| | med bytes |
|---|--:|--:|--:|
| **W10L9_h3ctrl2 k2** | **18/27** | **1.19** | 21.6K |
| outer_zensimA j2 (butter) | 12/27 | 3.08 | 22.9K |
| **W10L9_h3ctrl2 k3** | **24/27** | **0.54** | 21.6K |
| outer_zensimA j3 (butter) | 14/27 | 1.94 | 21.9K |

  — at inner-compare cost vs one FULL butteraugli-driven encode per outer
  step. The frontier arm also beats every other arm in the series (best
  prior inner: W10L9_h3own 17/27 k3; outer_ssim2 16/27 j3). Emit-last
  census == emit-best at both budgets (the pure-proportional controller
  converges; the last iterate is the internal best nearly everywhere).

### Defaults ADOPTED (this commit)

`JXL_ZENSIM_CTRL_EXP` default 0.6 → **1.0** and `JXL_ZENSIM_CTRL_CLAMP`
default 1.35 → **2.00** in `zensim_loop.rs`, per the gates above (evidence
on the steered arm at both budgets + the controller-only confirmation).
`ZENSIM_ATTR_BIN=8` default confirmed. h3-mag steering remains opt-in via
`JXL_ZENSIM_MODEL_MAP` (the silent product default is unchanged — flipping
it was explicitly not claimed by this registration).

### Residue (honest; verified per-cell)

The 3 remaining k3 misses: **cid1418519/t70 (achieved 72.44, err 2.44),
sc_wiki/t80 (83.06, err 3.06), sc_gui/t80 (82.94, err 2.94)** — the clamp
CURED the worst screen-t70 cells at k3 (they now hit), and the residue is
the milder overshoot band. k2's 9 misses are the screen t70/t80 class at
2-step budget (sc_wiki/sc_gui t70 still land ≈ 80 — two steps cannot
descend from seed-quality ≈ 91 even at clamp 2) plus cid1418519 t70/t80
and sc_imessage t70/t80. A content-aware SEED distance (screen content
seeds far too conservative) is the registered next lever, out of this
study's scope.
