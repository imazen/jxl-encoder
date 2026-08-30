# zensim secant guard thresholds — FITTED, not guessed (2026-08-30)

T2 of the 2026-08-30 work program. The registered follow-up to
`benchmarks/zensim_secant_2026-08-25.md` was "the min-|Δln L| guard … to kill
the intermediate overshoot". Two premises in that note and in CLAUDE.md §T2
turned out to be **wrong on the current substrate**, and both are corrected
here.

## Premise check — what was already true, and what was not

**1. The guard was NOT missing. It has been in the code since `bbc2354c`.**
`zensim_loop.rs` already carried `&& (cur_log_l - prev_log_l).abs() > 1e-3`
inside the secant condition. CLAUDE.md §T2 said "Fix = a min-|Δln L| guard
falling back to the power law"; that work was already done. The genuinely
outstanding item was the one this brief names: the threshold was **guessed**.

**2. At 1e-3 the guard is INERT — it never fires.** Over **1053** secant-
eligible controller steps (all arms, both budgets, 9 refs × 3 targets) the
smallest observed |Δln L| is **3.11e-3**:

| stat | min | p01 | p05 | p10 | median | max |
|---|--:|--:|--:|--:|--:|--:|
| \|Δln L\| | 0.00311 | 0.00505 | 0.00866 | 0.01485 | 0.1751 | 0.5915 |

`powerlaw:dlnl` fired **0 times** at 1e-3. Every threshold ≤ 3e-3 is provably a
no-op on this grid.

**3. The 8-point overshoot does not reproduce.** CLAUDE.md §T2 cites "t=70 went
71.0 → 61.8 at iter 2" (8.2 points past target). Scored against the approach
direction — iterate 0's side of the target fixes which way the controller is
travelling, so an excursion only counts as overshoot if it lands on the FAR
side — the worst overshoot anywhere in this sweep is **3.07** points, and
**no cell anywhere exceeds 5**. The 08-25 figure was measured on the v47A bake
with `JXL_ZENSIM_CTRL_EXP=0.6`; the controller has since moved to exp 1.0 with
the S4 per-image elasticity prior, and the shipped bake is Profile C. The
overshoot is real but an order of magnitude smaller than recorded.

**4. |Δln L| is the wrong axis for the overshoot that DOES occur.** The step is
`ln S_target = ln S + (ln L_t − ln L)/ε̂`. What sets step size is the
**denominator ε̂**, and ε̂ can collapse two ways: a small numerator (Δln L —
what the registered guard watches) *or* a large denominator (Δln S, which the
±ln 2 clamp routinely permits). On this substrate it is always the second. The
worst cell, verbatim from the controller trace:

```
sc_win95 t=70, k3, guard off
  iterates: i0=79.30  i1=73.72  i2=72.74  i3=66.93
  step iter=2  |Δln L|=0.03671  Δln S=-0.20603  ε̂=-0.1782
               g_pow=0.9087  g_secant=0.5841        <- 41 % scale cut
```

|Δln L| = 0.0367 is 37× the guessed threshold — healthy by that test — while
ε̂ = −0.178 is a shallow local slope that extrapolates to a 41 % cut. The loss
then responds far more steeply than the slope predicted and the trajectory lands
3.07 below target. **A |Δln L| guard cannot catch this at any inert setting**,
and every setting large enough to catch it is large enough to start disabling
the secant on healthy steps.

## Setup

- Harness `examples/zensim_diffmap_rd`, arm `h3-mag`, Profile C bake
  `c_sdr_mlp944_corrmix_2026-08-05.bin` (sha256 `1a2c8d522fed8034…`, zensim
  `9c0635f1`) — the same recipe as the 08-25 A/B.
- **Corpus is NOT the 08-25 nine.** `city/dog/girl` and the
  `codec_wiki/gui/imessage` crops live under `/mnt/v`, which does not exist on
  this host; the local `CID22-512/validation` holds a different eight. The grid
  keeps the SHAPE (6 photo + 3 nonphoto × t{70,80,88} = 27 cells) with six
  local CID22-512 photos and three 480×480 crops of `gb82-sc`
  (`+64+0`): `benchmarks/zensim_secant_corpus9_2026-08-30.tsv`. **Absolute
  numbers here are therefore NOT comparable to the 08-25/08-26 tables** — only
  the within-sweep arm-to-arm deltas are.
- Judged on the instrument's own `achieved_decoded` / `abs_err` / `bytes`
  columns via `scripts/zensim-loop-eff/verdict_23shot_cells.py`; trajectory
  read via `scripts/zensim-loop-eff/analyze_secant_overshoot.py`.
- Engagement gated per arm (probe = 27·K, per-compare trace = 27·(K+1),
  controller trace = 27·K rows). Every arm passed; a silent fall-through would
  have made the sweep a null comparison.
- Runner: `scripts/zensim-loop-eff/run_secant_guard_fit.sh` (`AXIS=dlnl|eps`,
  `EMIT=best|last`).
- Host `mac`, Apple M4 Pro, 12 cpu, Darwin 25.5.0 arm64, rustc 1.98.0,
  base commit `9bb62a18`. No `target-cpu=native`.

## Sweep 1 — `JXL_ZENSIM_SECANT_MIN_DLNL` (the registered axis)

13 thresholds × K{2,3}, emit-best. Cells
`zensim_secant_dlnl_cells_2026-08-30.tsv`, controller traces
`zensim_secant_dlnl_ctrltrace_2026-08-30.tsv`.

| threshold | k2 census | k2 med \|err\| | k3 census | k3 med \|err\| | k3 bytes |
|---|--:|--:|--:|--:|--:|
| ctrl (secant off) | 19/27 | 0.700 | 23/27 | 0.407 | 793,676 |
| 0 (guard off) | 20/27 | 0.553 | 23/27 | 0.333 | 812,368 |
| 1e-4 | 20/27 | 0.553 | 23/27 | 0.333 | 812,368 |
| **1e-3 (shipped)** | 20/27 | 0.553 | 23/27 | 0.333 | 812,368 |
| 3e-3 | 20/27 | 0.553 | 23/27 | 0.333 | 812,368 |
| 6e-3 | 20/27 | 0.553 | 23/27 | 0.333 | 812,368 |
| 1e-2 | 20/27 | 0.553 | 23/27 | 0.333 | 812,032 |
| 3e-2 | 20/27 | 0.553 | 23/27 | 0.333 | 812,032 |
| 1e-1 | 20/27 | 0.553 | 23/27 | 0.331 | 809,896 |
| 2e-1 | 19/27 | 0.542 | 23/27 | 0.251 | 805,642 |
| 3e-1 | 18/27 | 0.700 | 23/27 | 0.314 | 793,710 |
| 6e-1 | 19/27 | 0.700 | 23/27 | 0.407 | 793,676 |

0 through 6e-3 are bit-identical to each other (the guard cannot fire below the
observed minimum). 6e-1 reproduces the control exactly — the guard has disabled
the secant, which is the mechanism working, just uselessly. In between there is
no consistent gain: the thresholds that look best at k3 (2e-1) cost census at
k2, and 3e-1 costs two.

**Fit: keep 1e-3, as a numerical floor and nothing more.** It is the largest
round value with real margin (3.1×) under the observed |Δln L| minimum, so it
protects against a genuine divide-by-zero without ever biting on live data.
Raising it to 3e-3 would sit exactly ON the observed minimum with zero margin;
every value that bites is a secant-disabler in disguise. The honest conclusion
is that this axis has no useful interior setting.

## Sweep 2 — `JXL_ZENSIM_SECANT_MIN_EPS` (the axis that governs step size)

A floor on |ε̂| was always present in the shape `eps_hat < -1e-6`. **1e-6 is a
sign test, not a trust region.** Same grid, sweeping that constant instead.
Cells `zensim_secant_eps_cells_2026-08-30.tsv` (emit-best) +
`zensim_secant_epslast_cells_2026-08-30.tsv` (emit-last).

The distribution says where to look. Over the same 779 eligible steps |ε̂| has a
low tail cleanly separated from its body:

| \|ε̂\| | min | p01 | p05 | p10 | p25 | median | max |
|---|--:|--:|--:|--:|--:|--:|--:|
| | 0.043 | 0.043 | 0.322 | 0.415 | 0.492 | 0.589 | 44.19 |

`<0.15`: 12/779 · `<0.20`: 25/779 · `<0.30`: 33/779 · `<0.40`: 57/779 ·
`<0.50`: 237/779. A floor at 0.20–0.30 clips 3–4 % of steps — the tail — and
leaves the body untouched; 0.50 is already inside the body.

Outcomes, emit-best (k2 is unchanged from 1e-6 at every threshold ≤ 0.40, so
only k3 varies):

| min\|ε̂\| | k3 census | k3 med \|err\| | k3 bytes | cells over>2 | worst over | med over | mean crossings | med final \|err\| |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ctrl (secant off) | 23/27 | 0.407 | 793,676 | 0 | 1.756 | 0.215 | 0.926 | 0.456 |
| 1e-6 (pre-fit) | 23/27 | 0.333 | 812,368 | 4 | 3.066 | 0.425 | 1.148 | 0.436 |
| 0.15 | 23/27 | 0.333 | 812,368 | 4 | 3.066 | 0.425 | 1.148 | 0.436 |
| **0.20** | 23/27 | 0.333 | 812,241 | 3 | 2.171 | 0.215 | 1.074 | 0.381 |
| **0.25 (adopted)** | 23/27 | 0.333 | 812,241 | 3 | 2.171 | 0.215 | 1.074 | 0.381 |
| **0.30** | 23/27 | 0.333 | 810,701 | 3 | 2.171 | 0.215 | 1.037 | 0.381 |
| 0.35 | 23/27 | 0.333 | 810,356 | 3 | 2.171 | 0.215 | 1.037 | 0.381 |
| 0.40 | 23/27 | **0.425** | 810,379 | 3 | 2.171 | 0.215 | 1.037 | 0.381 |
| 0.50 | 23/27 | **0.425** | 806,445 | 3 | 2.171 | 0.215 | 0.963 | 0.381 |
| 0.60 | 23/27 | **0.557** | 794,659 | 1 | 2.171 | 0.317 | 0.889 | 0.578 |

emit-last (the arm the 08-25 note said the overshoot actually hurt) narrows the
plateau's top edge:

| min\|ε̂\| | k2 census | k2 med \|err\| | k3 census | k3 med \|err\| |
|---|--:|--:|--:|--:|
| ctrl (secant off) | 18/27 | 0.766 | 23/27 | 0.598 |
| 1e-6 (pre-fit) | 20/27 | 0.616 | 23/27 | 0.477 |
| 0.15 / 0.20 / **0.25** | 20/27 | 0.616 | 23/27 | 0.477 |
| 0.30 | 20/27 | 0.616 | 23/27 | 0.482 |
| 0.35 / 0.40 | 20/27 | 0.616 | 23/27 | 0.505 |
| 0.50 | 21/27 | 0.616 | 23/27 | 0.505 |

**Fit: 0.25.** It sits in the interior of the [0.20, 0.30] plateau on which
every measured column is at its best or tied-best under BOTH emit modes; below
0.20 nothing happens, at 0.30 emit-last starts to slip and by 0.35–0.40 both
emit modes have lost median accuracy. The outcome boundary agrees with the
distribution boundary measured over 779 steps — the plateau ends exactly where
the floor stops clipping the tail and starts eating the body. Choosing the
interior rather than an edge leaves margin on both sides, so a substrate shift
does not immediately push the default off the plateau.

## What the adopted guard actually does

Against the pre-fit `1e-6`, at the shipped default and on this corpus:

| | k2 | k3 |
|---|--:|--:|
| trajectories changed | 0/27 | 2/27 |
| emit-best bitstreams changed | 0/27 | 1/27 |
| emit-last bitstreams changed | 0/27 | 2/27 |

It is a **rare, targeted intervention**: it fires on the |ε̂| tail only. On the
two k3 cells it touches it removes the worst overshoot (3.07 → 2.17), drops
overshooting cells 4 → 3, halves median overshoot (0.425 → 0.215), reduces
target crossings (1.148 → 1.074) and improves median final in-loop |err|
(0.436 → 0.381) — **with decoded census and decoded median |err| unchanged at
both budgets and both emit modes** (k3 23/27 @ 0.333 emit-best, 23/27 @ 0.477
emit-last; k2 identical throughout). Bytes move −127 (−0.016 %) emit-best and
+1,707 (+0.21 %) emit-last, both far inside the ±1 % bar.

The secant's advantage over the power law is preserved, not traded away —
on this corpus, secant vs control: k2 census 19→20 and median −21 % (0.700 →
0.553); k3 median −18 % (0.407 → 0.333); emit-last k2 census 18→20 and median
−20 %, k3 median −20 %. **These are NOT the 08-25 −55 %/−71 % figures and must
not be read as a reproduction of them** — different corpus (see Setup). What is
demonstrated is that the guard leaves the margin, whatever its size on a given
corpus, exactly where it found it.

Default-takes-effect gate: the shipped binary with no env override reproduces
the explicit `JXL_ZENSIM_SECANT_MIN_EPS=0.25` arm on **27/27** cells
byte-for-byte (`zensim_secant_ship_cells_2026-08-30.tsv`).

## Honest limits

1. **n = 27 cells per arm, and the adopted guard engages on 2 of them.** The
   effect size rests on those two cells; the direction is unambiguous (every
   column improves or ties, never degrades) but the magnitude is weakly
   evidenced. What carries the *threshold choice* is the plateau — five adjacent
   grid points agreeing across two emit modes — and the 779-step |ε̂|
   distribution, not the two cells.
2. **One corpus, one bake, one gain/clamp**, and a corpus that is not the
   registered nine (unavailable on this host). Re-running
   `run_secant_guard_fit.sh` on the `/mnt/v` corpus is the confirmation this
   note does not have.
3. **Loop-internal trajectory metrics** (overshoot, crossings, final in-loop
   |err|) come from the loop's own scorer. Census/median/bytes are decoded.
4. **No bytes column exists inside the loop** — it never entropy-codes. Every
   byte figure here is from the harness's separate full encodes, not an in-loop
   estimate.
5. The 08-25 note's claim that adding the 1e-3 guard moved k2-best median
   0.951 → 0.734 is **not reproducible on the current substrate**, where that
   threshold provably never fires. Either the substrate has changed enough to
   dissolve the effect (likely: bake, ctrl_exp and the S4 prior all moved) or
   the 08-25 comparison carried another difference. Recorded, not resolved.

## Reproduce

```bash
CARGO_TARGET_DIR=$HOME/tmp/jxlloop-target nice -n19 cargo build --release \
  -j 4 -p jxl-encoder --example zensim_diffmap_rd \
  --features "__expert butteraugli-loop zensim-loop ssim2-loop parallel"
AXIS=dlnl                ./scripts/zensim-loop-eff/run_secant_guard_fit.sh
AXIS=eps                 ./scripts/zensim-loop-eff/run_secant_guard_fit.sh
AXIS=eps EMIT=last TAG=epslast \
  THRESHOLDS="1e-6 0.15 0.20 0.25 0.30 0.35 0.40 0.50" \
                         ./scripts/zensim-loop-eff/run_secant_guard_fit.sh
python3 scripts/zensim-loop-eff/analyze_secant_overshoot.py \
  ~/tmp/t2sec/sweep-eps/trace_*_k3.tsv
```
