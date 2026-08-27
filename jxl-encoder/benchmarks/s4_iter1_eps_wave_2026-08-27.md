# S4 iteration-1 elasticity prior — pre-registered wave (2026-08-27)

REGISTERED BEFORE ANY INSTRUMENT RUN (census fires only after this commit).
The zensim-side S4+C2 wave (zensim plan doc, ruling + fit 2026-08-27)
re-scoped by the zq_seed census facts:

- **C2a (content-aware seed) CLOSES as MEASURED-REDUNDANT**: `zq_seed`
  (2026-08-26 wave, in-tree, env-gated) is the owner; its 27-cell census put
  the whole seed lever at 5.6% median |err| (bar ≥15%, FAIL) — a better seed
  regressor (the s4c2 190-feature ridge, MAE ≈ the head's p50) cannot
  plausibly 3× that. No second seed mechanism is built.
- **Flag on the zq record's registered "class-conditional seed" lever**: the
  census shows the head improved photo TOO (0.600→0.554), so
  head-for-nonphoto-only is arithmetically DOMINATED by all-head on the same
  evidence; a conditional wave should not be run as registered.
- **The live untested lever is S4**: with the secant ON by default
  (2026-08-25), iterations ≥2 use measured ε̂; ITERATION 1 still assumes
  ε̂ = −1/ctrl_exp = −1. The C2b slope prior carries rank signal at t80/t88
  (test SROCC 0.795/0.716; NOT t70, ratio 0.977 —
  zensim `benchmarks/s4c2_prior_fit_2026-08-27.md`).

## Arm (frozen)
Per-cell `JXL_ZENSIM_CTRL_EXP` = clamp(−1/ε̂_prior, 0.25, 2.0) at t∈{80,88};
t70 cells keep 1.0 (prior has no signal there — using it would inject noise).
Unit bridge, derived from the step law (`zensim_loop.rs` C3b block):
  power step ln g = exp·(ln L − ln L_t)  ⇔  ε̂ = −1/exp,  L = 100−score
  ε̂_prior,i(t) = (slope_i(t) / (100−t)) / DQ_i(t)
  slope_i(t) = s4c2 C2b ridge (dscore/dlogq, zensimA-proxy, gate-passed)
  DQ_i(t)    = dlog d/dlog q at the prior's own q_seed_i(t), numeric central
               diff of the public `jxl_encoder::api::quality_to_distance`
               (the zq wave's registered no-hand-rolled-mapping rule).
Features at encode time: the s4c2 identity-944 extractor on the corpus refs;
prediction offline into a frozen per-(image,t) table consumed by a per-cell
census driver. ZERO loop-code change (env is read per encode).

## Census + gates (frozen — G-J2 protocol, comparable to the zq census)
9-ref corpus9 × t{70,80,88} × k2 emit-best, v47 bake, baseline arm, TOL=-1,
secant ON (current defaults). Arm A control: all cells exp=1.0. Arm B: table.
- **PASS iff median decoded |achieved−target| improves ≥15% overall AND the
  ±2-hit count does not regress.** (Same bar as zq; expected effect is
  modest — a FAIL cleanly kills S4 and is a fully acceptable outcome.)
- Diagnostic rows (reported, not gated): per-class medians; t80/t88-only
  median (the cells the arm actually touches); engagement = distinct exp
  values used (must exceed 1, else the run is void, not a FAIL).
- Safety: any table miss ⇒ exp 1.0 (control behavior), loud stderr note.

## Endgame (frozen)
PASS ⇒ propose (user-gated, never flipped by me) a shipped per-image iter-1
ε̂ source; FAIL ⇒ numbers committed here, S4 closes, the constant stays.
Either way: census TSVs + driver committed, zensim plan + memory updated.
