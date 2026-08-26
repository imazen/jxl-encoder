# JXL diffmap secant controller (JXL_ZENSIM_SECANT) — A/B (2026-08-25)

Goal criterion 4 / zensim plan §5. The qf-targeting loop's controller is a
damped power law `g = (L_a/L_t)^exp` (exp 1.0, clamp 2.0). This adds a **secant**
alternative that measures the elasticity from the last two iterates instead of
assuming a fixed exponent.

## Design (committed `bbc2354c`, default OFF)

`L = 100 − score`. In this codebase higher `quant_field` = MORE bits = LESS
loss, so `ε̂ = Δln L / Δln S` is **NEGATIVE**. Step:
`ln S_target = ln S + (ln L_t − ln L)/ε̂`, then the existing clamp. The secant
fires only when `ε̂ < −1e-6`, `prev_log_l` is finite (not iter 0), and the last
two cumulative scales differ (`|Δln S| > 1e-6`); otherwise the power-law
fallback (also the mandatory first-iterate step). `cum_log_s` tracks the
controller g-product (the redistribution is sum-preserving, so it does not move
global scale). `zensim_loop.rs:1600-1631`.

## Engagement smoke (city, k3, v47A + h3-mag, emit tracking)

Traces DIVERGE at iter 2 (when the secant activates) — engagement proven, not a
silent fall-through. Final-accuracy `|final − target|`, control vs secant:

| target | ctrl \|err\| | secant \|err\| | winner |
|--:|--:|--:|--:|
| 70 | 0.296 | **0.084** | secant |
| 80 | 0.175 | **0.050** | secant |
| 88 | 0.873 | **0.183** | secant (4.8×) |

**Secant wins all three city targets on final accuracy.** Finding: an
intermediate OVERSHOOT (t70 iter2 → 61.8 from 71.0) when consecutive iterates
are near-equal (tiny Δln L → unreliable ε̂ that my Δln S guard does not catch).
The clamp bounds each step and emit-best emits the recovered final iterate, so
the overshoot is contained — but a **min-|Δln L| guard** is the registered
refinement to test next (fall back to power-law when the loss barely moved).

## Full 27-cell A/B (9 refs × t{70,80,88}, k2+k3, emit-best)

## Full 27-cell A/B (9 refs × t{70,80,88}, k2+k3, v47A + h3-mag, emit-best)

Loop's INTERNAL score, |final−target|≤2 census + median |err|:

| arm | k | emit | census | med \|err\| | photo | nonphoto |
|---|--:|---|--:|--:|--:|--:|
| ctrl   | 2 | best | 16/27 | 1.428 | 15/18 | 1/9 |
| **secant** | 2 | best | **17/27** | **0.951** | 14/18 | **3/9** |
| ctrl   | 2 | last | 16/27 | 1.428 | 15/18 | 1/9 |
| secant | 2 | last | 15/27 | 1.057 | 13/18 | 2/9 |
| ctrl   | 3 | best | 22/27 | 0.433 | 18/18 | 4/9 |
| **secant** | 3 | best | 22/27 | **0.297** | 18/18 | 4/9 |
| ctrl   | 3 | last | 21/27 | 0.505 | 17/18 | 4/9 |
| secant | 3 | last | 21/27 | 0.331 | 18/18 | 3/9 |

**Result: secant improves the registered k2 target** — +1 census (17 vs 16),
**−34% median error** (0.951 vs 1.428), and +2 nonphoto (3/9 vs 1/9), with
emit-best. At k3 the census ties (22=22) but median error is **−31%** (0.297 vs
0.433). The overshoot only hurts emit-LAST (secant 15 vs 16 at k2-last); emit-best
(the shipped default) is where it wins.

**Caveats (honest):** (1) INTERNAL-score census, not decoded-judged — the
registered instrument (`analyze_23shot`) uses decoded scores; a decoded-judged
re-run is the rigorous confirmation. (2) v47A bake, not the frontier
`W10L9_h3ctrl2` (C-bake) arm — A/B on the shipped recipe is the next step.
(3) n=27, one bake, one gain/clamp. **Direction is clear and positive.**

## Next (registered)
1. Decoded-judged A/B (`analyze_23shot` owner) on the frontier C-bake arm — the
   ship-relevant confirmation.
2. The **min-|Δln L| guard** (fall back to power-law when the loss barely moved)
   to kill the intermediate overshoot — a one-line refinement, re-A/B.
3. If both hold: a controller-default proposal (user-gated, per AB.3 convention).

## Min-|Δln L| guard (2026-08-25) — kills the overshoot, improves the median

Added `&& (cur_log_l − prev_log_l).abs() > 1e-3` to the secant condition: when
consecutive iterates barely move the loss, ε̂ is a divide-by-noise, so fall back
to the power law. Re-A/B (same 9×3 corpus, v47A+h3-mag):

| arm | k | emit | census | med \|err\| |
|---|--:|---|--:|--:|
| ctrl | 2 | best | 16/27 | 1.428 |
| secant (no guard) | 2 | best | 17/27 | 0.951 |
| **secant (guard)** | 2 | best | **17/27** | **0.734** |
| secant (no guard) | 2 | last | 15/27 | 1.057 |
| **secant (guard)** | 2 | last | **16/27** | **0.951** |
| ctrl | 3 | best | 22/27 | 0.433 |
| **secant (guard)** | 3 | best | 22/27 | **0.297** |

The guard is strictly ≥ the un-guarded arm on every cell: it restores emit-last
to control parity (15→16 at k2) AND tightens emit-best median (0.951→0.734 at
k2). k3 is unchanged (the overshoot was a k2 phenomenon). **The guarded secant
beats the power-law controller on median error at both budgets (−49% k2-best,
−33% k2-last, −31% k3-best) and the k2 census (+1), with no regression.** This is
the shipped form of the arm (still default OFF). Committed here + `bbc2354c`.

## Frontier confirmation — C's shipped bake (2026-08-25)

Re-ran the guarded-secant A/B on the SHIPPED recipe (Profile C's bake
`c_sdr_mlp944_corrmix_2026-08-05.bin` + h3-mag) — the ship-relevant question.
Internal-score census, emit-best, same 9×3 corpus:

| k | arm | census | med \|err\| |
|--:|---|--:|--:|
| 2 | ctrl | 19/27 | 1.021 |
| 2 | **secant** | **22/27** | **0.458** |
| 3 | ctrl | 24/27 | 0.578 |
| 3 | **secant** | **25/27** | **0.169** |

**On the shipped bake the win is larger than on v47A: k2 +3 census (22 vs 19),
−55% median (0.458 vs 1.021); k3 +1 census (25 vs 24), −71% median (0.169 vs
0.578).** The C bake mounts in the folded-944 loop (verified) and the secant
helps it at both budgets on both census and median. Same caveats stand
(internal-score, n=27, one gain/clamp; decoded-judged confirmation registered),
but the direction + magnitude on the shipped recipe are compelling. The diffmap
secant is a measured jxl-loop-efficiency improvement, default OFF pending the
decoded-judged pass + a controller-default proposal (user-gated).

## DECODED-JUDGED A/B — registered confirmation (2026-08-26, phase `secant` of run_23shot_sota944.sh)

The registered "Next #1": same frontier C bake + h3-mag, 9×3 corpus, but judged
on the instrument's `achieved_decoded`/`abs_err` columns instead of the loop's
internal score. Engagement gates all passed (probe 27·K, trace 27·(K+1) per
arm; sec1 bitstreams differ from sec0 in 23/27 cells at k2, 27/27 at k3).
Cells TSV: `benchmarks/zensim_loop_secant_decoded_2026-08-26.tsv` (216 rows).

| arm | census ≤2 | med \|err\| | bytes (sum) |
|---|--:|--:|--:|
| ctrl k2 best | 18/27 | 1.174 | 757,194 |
| **secant k2 best** | **22/27** | **0.534** | 772,134 (+1.97%) |
| ctrl k2 last | 18/27 | 1.174 | 755,018 |
| **secant k2 last** | **23/27** | **0.567** | 769,527 (+1.92%) |
| ctrl k3 best | 24/27 | 0.566 | 756,471 |
| **secant k3 best** | **25/27** | **0.344** | 771,183 (+1.94%) |
| ctrl k3 last | 24/27 | 0.566 | 755,211 |
| **secant k3 last** | **25/27** | **0.355** | 768,538 (+1.76%) |

**Decoded confirms the internal-score read on accuracy: k2 census +4/+5, k2
median −55%, k3 census +1, k3 median −39%.** Achieved-bias per arm:

```
C944_sec0_k2_best: mean(achieved-target)=+1.949 median=+0.716 cells_under_by_0.5+=5
C944_sec0_k2_last: mean(achieved-target)=+1.836 median=+0.613 cells_under_by_0.5+=7
C944_sec0_k3_best: mean(achieved-target)=+0.354 median=-0.043 cells_under_by_0.5+=7
C944_sec0_k3_last: mean(achieved-target)=+0.274 median=-0.264 cells_under_by_0.5+=8
C944_sec1_k2_best: mean(achieved-target)=+1.100 median=-0.149 cells_under_by_0.5+=7
C944_sec1_k2_last: mean(achieved-target)=+0.911 median=-0.278 cells_under_by_0.5+=9
C944_sec1_k3_best: mean(achieved-target)=-0.185 median=-0.125 cells_under_by_0.5+=5
C944_sec1_k3_last: mean(achieved-target)=-0.294 median=-0.085 cells_under_by_0.5+=8
```

**Frozen-rule verdict (§5.2): NOT ADVANCED.** The S1 rule requires "bytes
within ±1% of S0" and every secant arm sits at +1.8..+2.0%. Census + median
pass decisively; the bytes bar FAILS, so per the pre-registered rule the arm
is REPORTED, not adopted (it stays default OFF — the default flip was
user-gated regardless). Mechanism — CORRECTED against the bias table above
(an earlier revision claimed the control undershoots; the data says
otherwise): the control lands HIGH on mean (k2 +1.8/+1.9, median +0.6/+0.7)
while the secant lands nearer zero (k2 mean ≈ +1.0, median ≈ −0.2; k3 mean
≈ −0.2). Aggregate bias therefore does NOT explain the secant's +1.9% bytes —
an arm sitting at HIGHER achieved quality would be expected to cost more, not
fewer, bytes than one at target. The bytes delta lives in the per-cell
distribution (both arms miss low on 5-9 cells; which cells land where differs
by arm), so only a rate-matched per-cell read (equal achieved → bytes ratio,
the mm-F3 shape) can say whether the +1.9% is waste or reallocation. Open
(registered, not decided): that rate-matched read, and/or a bytes-bar
amendment — both user-visible, neither taken unilaterally.
