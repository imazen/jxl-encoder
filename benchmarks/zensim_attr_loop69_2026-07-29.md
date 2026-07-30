# #69 loop-steering study — controller × redistribution (2026-07-29)

Pre-registered at zensim `docs/PLAN_LOOP_STEERING_69.md` (frozen at zensim
`5f7d16a3`); run exactly as registered. Substrate: the C3b harness
(`f195c8c0`) + zensim fused compare (read-only path dep). Data:
`zensim_attr_loop69_{v47A,shippedB,v47A_g4stale}_2026-07-29.tsv`; run log
`~/tmp/attrmap-69/ab/run69.log`.

## Arms (env-gated, defaults unchanged; all share compare/controller/bounds/renorm)

H0: `baseline` (no model map), `abs` (ModelSensitivity fold), `attr` (C3b
fused-map ratio steering). New this study (`JXL_ZENSIM_MODEL_MAP=`):
- `h1-signed` — signed redistribution: per-tile signed mean density drives
  the qf factor directly (negative ⇒ coarser proportionally; no clamp-at-0,
  no anchor blend; CV-adaptive alpha over |field|, ±ratio_max, factor_max,
  shared sum-renorm).
- `h2-ctrl` — controller separation: as H1 but the field is CENTERED
  (tile − mean ⇒ zero-sum residual); the damped controller alone owns the
  level.
- `h3-mag` — magnitude steering: factor = 1 + `ZENSIM_H3_GAIN`(=10.0
  default, untuned) × per-tile `query_rect` (score units), capped by
  factor_max; NOT ratio-normalized.

## Fixtures (G3: n = 9 × 3 targets = 27 cells/arm/bake)

city/dog/girl 576² (`/mnt/v/output/zensim/diffmap-coherence-2026-07-18/`);
CID22-512 validation photos 1025469/1418519/1189261
(`~/work/codec-corpus/CID22/CID22-512/validation/` — encoder fixtures only,
no training use); nonphoto: pixel-exact 576² crops (`-crop 576x576+512+256`)
of gb82-sc `codec_wiki`/`gui`/`imessage`
(`~/work/codec-corpus/gb82-sc/`). Targets {70, 80, 88} per registration.

## Results — median |achieved − target| (decoded-judged, same-bake judge)

### v47A (A-class MLP bake)

| target | baseline | abs | attr | h1-signed | h2-ctrl | **h3-mag** |
|---|--:|--:|--:|--:|--:|--:|
| 70 | 1.867 | 2.488 | 2.899 | 1.652 | 3.288 | **0.306** |
| 80 | 0.633 | 1.470 | 1.429 | 0.938 | 1.451 | **0.262** |
| 88 | 1.174 | 0.865 | 0.887 | 1.028 | 0.972 | 1.211 |
| all | 1.174 | 1.628 | 1.530 | 1.053 | 1.470 | **0.593** |

### shippedB (B-class linear bake)

| target | baseline | abs | attr | h1-signed | h2-ctrl | h3-mag |
|---|--:|--:|--:|--:|--:|--:|
| 70 | **0.326** | 2.682 | 0.912 | 0.950 | 1.304 | 1.371 |
| 80 | **0.209** | 0.564 | 0.411 | 0.254 | 0.254 | 0.815 |
| 88 | 1.236 | 1.096 | 1.511 | 1.642 | 1.430 | 1.799 |
| all | **1.056** | 1.231 | 1.097 | 1.334 | 1.304 | 1.371 |

## Gate verdicts (frozen gates, stated per bake)

- **G1 (beat BASELINE on ≥2/3 targets):** v47A — **h3-mag PASSES (2/3)**
  (t70 6.1× and t80 2.4× better than baseline; t88 misses by 0.037) and
  h1-signed passes (2/3, modest); h2-ctrl fails (1/3). shippedB — **every
  arm FAILS (0/3)**; the plain controller with the linear bake is the best
  target-hitter. No arm passes G1 on both bakes.
- **G2 (bytes at equal achieved within +2%, v47A passers):** h3-mag
  **PASSES** (median bytes ratio 0.990 over 15 comparable cells,
  |Δachieved| ≤ 0.5); h1-signed marginal (median 1.016, max 1.069).
  Context: baseline's t70/t80 error is UPWARD overshoot (med achieved 71.9
  vs 70) — extra quality bought with extra bytes; h3-mag lands on-target
  (70.3) at commensurately smaller size. Accuracy is not bought with size.
- **G3 (breadth):** met by construction (27 cells/arm/bake; 3 nonphoto
  refs). Nonphoto texture: target-hitting is much harder for everyone
  (baseline med |err| 3.4 v47A / 3.1 shippedB) and **h3-mag is the only
  arm beating baseline on nonphoto on BOTH bakes** (2.44 vs 3.41; 1.20 vs
  3.12).
- **G4 (single-pass pricing for the passer):** `h3-mag-stale` (previous
  iteration's map) ≈ fresh on v47A — all-median 0.598 vs 0.593, t70
  0.224 vs 0.306. **Staleness remains free ⇒ the ≤1.1× single-pass
  endpoint stays viable for the one loop-rule that shows value.** (Run
  includes an in-run fresh-h3 control matching the main matrix exactly —
  it also caught and fixed a silent env-fallthrough in the first G4
  attempt, where an unwired arm value ran as baseline; unknown
  `JXL_ZENSIM_MODEL_MAP` values are a fallthrough hazard by design.)

## Honest conclusion

The registered all-arms-fail pivot clause does NOT fire — but the value is
narrower than hoped: **per-tile steering adds target-hitting value only as
MAGNITUDE steering (H3), only clearly with the MLP-class bake (v47A), and
mostly at low/mid targets and on nonphoto content.** Ratio-normalized
steering (C3b attr, H1, H2) never beats a plain damped controller; with the
shipped linear bake, nothing does. The C3b conclusion stands for the
ratio-rule family; H3's mechanism (score-unit steps skip the
normalization that erases magnitude information) is the survivor and the
candidate for any follow-up. `ZENSIM_H3_GAIN` was NOT swept (single
default 10.0, as registered) — a gain sweep is follow-up work, not part of
this study's claims.

## Wall time

Median ms/compare (576²-class): baseline 34-35 | abs 39-56 | attr/H-arms
47-65 (the H arms share the attr fused call; +~12-30 ms/compare over
baseline depending on bake). Budget: 351 encodes total, ~13 min nice'd.
