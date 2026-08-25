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
