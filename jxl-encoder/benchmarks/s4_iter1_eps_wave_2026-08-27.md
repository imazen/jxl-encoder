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

## RESULT — G-J2 **PASS** as registered (2026-08-27, same night)

Census ran per registration (54 cells; driver
`scripts/zensim-loop-eff/run_s4_iter1_census.sh`; table + both cell TSVs
committed alongside). Substrate validity came free twice: arm A's overall
median 0.832 EQUALS the zq census staircase control exactly, and the t70
cells (both arms exp 1.0 by registration) are identical.

- median decoded |err|: control **0.832** vs eps-prior **0.607** —
  improvement **27.0%**, over the frozen ≥15% bar ⇒ **PASS**.
- ±2 hits: 21/27 → 22/27 (no regression). Engagement: 18 touched cells,
  16 distinct exponents, 12W / 4L / 2T; touched-only median 0.672 → 0.408.
- Per t: t80 0.527→0.347 (−34%), t88 0.832→0.607 (−27%), t70 unchanged (by
  construction).
- **Honest asymmetry**: the win is PHOTO-concentrated (0.600→0.408, −32%);
  nonphoto median 1.836→1.943 (+5.8%, 3-ref class, includes untouched t70
  cells) — the nonphoto residual is controller/undershoot-class, not
  first-step-exponent, territory. Opposite concentration to the zq seed
  head's (−51% nonphoto). The two levers are complementary, not redundant.

## PROPOSAL (user-gated — nothing flips without an explicit yes)
Mechanism validated: per-image iteration-1 ε̂ beats the ε̂≡−1 assumption.
The census arm consumed the OFFLINE 190-feature table (instrument-only
form). Ship form would need the slope prior refit on cheap in-binary
features (zq_seed's 8 zenanalyze features are the natural basis — module
already in-tree) + its own census at the same bar, then a default wiring
decision. Registered as the follow-up; NOT run tonight.

## B2 (ship-form) — REGISTERED before fit/census (2026-08-27, same night)
Arm B2 = the same iter-1 ε̂ mechanism with SHIP-FEASIBLE inputs:
- slope_t{80,88} REFIT on the zq_seed 8-feature basis (in-binary-cheap
  zenanalyze features), same frozen ridge protocol as the 190-feature fit
  (λ on val, test once, constant baseline reported); features joined
  REF-ONLY from the 07-01 canonical root on (origin_id, width, height) —
  ref features are q- and root-vintage-independent.
- DQ evaluated at the SHIPPED `zq_seed::predict_q0_from_features` q0 (the
  seed owner's prediction), NOT a parallel qseed model — one seed owner.
- Census: same 27 cells, same control arm A (same substrate, same night,
  reuse registered), same bar: PASS iff ≥15% overall median |err|
  improvement AND ±2 hits not regressed. B2 exists to answer: does the
  27.0% table win survive the cheap-feature + owner-seed form?
