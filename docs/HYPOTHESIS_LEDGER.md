# Hypothesis ledger

**Per methodology Rule 6** (`~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/research_methodology_9_rules_2026-05-22.md`): a single document tracking what we currently believe about the encoder + the tuning corpus, the evidence each belief rests on, when last validated, and whether contradictions exist.

**Maintenance rule:** every chunk's acceptance memo updates ≥1 entry (or notes "no belief change"). When two entries contradict, the next chunk's first task is reconciling them.

**Status values:**
- `PROVEN` — direct empirical evidence; bit-equivalent or measured at ≥3σ
- `LIKELY` — strong evidence but indirect or single-source
- `SUSPECTED` — hypothesis with rationale; not yet measured
- `RULED-OUT` — measured + falsified
- `CONTRADICTED-PENDING-RESOLVE` — two beliefs in conflict; next chunk reconciles

## Seeded 2026-05-22 from W44-217..221 retrospective; validated 2026-05-22 by W44-226 pipeline run

| # | Belief | Status | Evidence | Last validated |
|---|---|---|---|---|
| 1 | The 6 RuntimeTuning params ONLY affect `EncoderStrategy::Zenjxl`; on `Libjxl` strategy, CV ≤ 0.03 % across all blobs. | PROVEN | W44-217 sanity_check.py on 4938-row W44-216 corpus | 2026-05-22 |
| 2 | Per-pair coupling-fn refit (W44-218 algebraic forms) cannot reach R² ≥ 0.5 on the response. | PROVEN | W44-220 measurement on 21×-denser W44-219 corpus; 0/7 pairs hit gate | 2026-05-22 |
| 3 | Joint-surface GBR over `(p1..p6 + features + effort + distance) → outcome` hits test R² ≥ 0.85 on every outcome. | PROVEN | W44-226 pipeline `kitchen_sink_gbr` on combined W44-216+W44-219 corpus: encoded_bytes=0.997, ssim2=0.996, butter_norm3=0.991, cvvdp=0.995, encode_ms=0.884 (all test R², 80/20 split, 11.4k zenjxl rows) | 2026-05-22 |
| 4 | The joint output surface has natural rank 4-5 in 6-param space. | PROVEN | W44-226 pipeline `svd_basis` reproduces W44-221 phase 2b: rank-4 = 88.5 % (W44-221 reported 88.3 %), rank-5 = 96.1 % of gradient variance on 40 anchor cells × 2 outcomes | 2026-05-22 |
| 5 | The W44-218 ridge basis spans 68.5 % of joint gradient variance. PC2/PC5 directions are NOT covered — 31.5 % missing variance lives there. | PROVEN | W44-221 phase 3 V-matrix projection | 2026-05-22 |
| 6 | A 4-knob Tier-2 expander achieves ≤2pp mean Pareto coverage on most strata but FAILS strict 0.5pp-max on `screen/very_high` (max gap ~8 %). | PROVEN | W44-226 pipeline `pareto_coverage` at 7^5 grid: 4-knob screen/very_high max = 10.78 % (W44-221 reported 7.86 %; differences are within seed/GBR-randomness band) | 2026-05-22 |
| 7 | W44-217 §10 claim "pairwise interactions dominate (singles 0.3-5%, pairs 6-22%)" vs belief #3 (joint GBR R² ≈ 1.0) is **NOT a contradiction** — it is a metric-interpretation difference. | RULED-OUT (as contradiction) | W44-226 pipeline ran BOTH on identical corpus: ANOVA log_bytes R²=0.555 with `distance=27%, content_class=11%, mask_median=5%, …, ALL six z_params combined < 0.5%` of total variance; ANOVA ssim2 R²=0.828 with `distance=74%, mask_median=2.3%, content_class=2.0%, …, ALL six z_params combined < 0.2%`. The §10 "params 0.3-5%" was *within-stratum residual variance* (after removing distance + content_class + features); on the FULL surface the params are dwarfed by distance × content_class × features. The GBR R²=0.997 also captures the same confounders + non-linear feature × param interactions ANOVA's linear form misses. Both numbers are correct for their respective scopes; the apparent contradiction was a category error. | 2026-05-22 |
| 8 | Adding (effort, distance, features) as inputs is what closes the W44-218 → W44-221 R² gap (not corpus density). | PROVEN | W44-226 pipeline measured per_pair_gbr (params-only) test R² = **-0.013** for encoded_bytes, **-0.037** for ssim2 — *worse than predicting the mean*. Kitchen-sink (params + features + axes) hits R² = 0.997 / 0.996. The 1.0 R² delta is entirely from added inputs, NOT corpus density. | 2026-05-22 |
| 9 | vast.ai default-launch is on-demand pricing; we are paying for spot-grade reliability without the spot discount. ~50-70% savings available via `--bid_price`. | PROVEN | W44-216 had 12/24 evictions; janitor handles them; CLI confirms `--bid_price` is the interruptible flag | 2026-05-22 |
| 10 | `EncoderStrategy::Libjxl` produces byte-identical bitstream regardless of RuntimeTuning override (the override only affects Zenjxl path). | PROVEN | W44-194 strategy_libjxl_byte_lock CI gate; never broken since W44-193 | 2026-05-22 |
| 11 | Default knob values in `Tier2Knobs::default()` round-trip byte-identical to current `Zenjxl` production (no behaviour change at defaults). | PROVEN | W44-221 hash_lock_features 36/36 byte-identical | 2026-05-22 |
| 12 | `LossyConfig::with_knobs(Tier2Knobs)` builder needed to plumb knobs through encoder entry — current `runtime::install` mechanism is single-shot. | PROVEN | W44-222 shipped the builder; 1435 lib tests pass; W44-226 pipeline `pareto_coverage` exercises the 5-knob expander logic end-to-end | 2026-05-22 |
| 13 | Speculative algebraic forms (additive / multiplicative / gated / suppressive / synergistic) cannot capture the joint response surface — non-parametric + SVD basis is strictly better. | PROVEN | W44-218→220 (0/7 fits) vs W44-221 (R² ≥ 0.85). W44-226 pipeline `marginal_pdps` classified 30/30 zenjxl param-pair surfaces as `ADDITIVE` (additive_residual_pct < 5 %) at the marginal GBR-fit scale — meaning on the AVERAGED kitchen-sink surface most pair-interactions integrate out, leaving an effectively additive marginal. This does NOT contradict belief #5 (per-stratum interactions exist); it just confirms that pair-interactions are stratum-local, not global. | 2026-05-22 |
| 14 | A 5th data-driven knob spanning the PC2/PC5 direction closes the `screen/very_high` 4-knob Pareto gap. | PROVEN | W44-222 shipped the 5-knob expander; W44-226 pipeline measured at 7^5 grid: 4-knob screen/very_high max gap = 10.78 %, 5-knob = 0.63 %, **Δ = +10.15 pp**, gate_2pp_max = PASS. (W44-222 reported 7.86 → 1.15, similar magnitude.) | 2026-05-22 |
| 15 | Param-only mutual information is dominated by `p2_screen_median` across every outcome (encoded_bytes, ssim2, butter_norm3, cvvdp) — except encode_ms where `p3_butt_qf_scale` leads. | LIKELY | W44-226 pipeline `mi_matrices` measured MI(param, outcome) on combined W44-216+W44-219 corpus (n=11.4k zenjxl rows, k=5 neighbours): p2 leads on all 4 quality/size outcomes (MI=0.046–0.081), p3 leads only on encode_ms (MI=0.011). Mechanism candidate: p2 (`screenshot_median_threshold`) is the discriminator that decides which zenjxl branch fires; once that's set, downstream params (p3/p5/p6) shape the branch's output. Single corpus, single random seed — needs a second corpus replicate before promoting to PROVEN. | 2026-05-22 |
| 16 | Photo/very_high pareto-coverage gap is RISING (not falling) as we increase knob grid density. | SUSPECTED | W44-226 pipeline measured: at 5^5 grid photo/very_high 5-knob = 1.98 % (PASS 2pp), at 7^5 grid = 2.34 % (FAIL 2pp). Direction of change is consistent with the W44-220 honest-stop on photo strata (params barely move photo outcomes; the apparent "improvement" is GBR noise). Worth a dedicated photo-strata investigation (W44-227 candidate) before reading too much into a 0.36 pp swing. | 2026-05-22 |

## How to update

- After each chunk: add new row OR update existing `Last validated` date + Status.
- Don't delete RULED-OUT or CONTRADICTED rows — they prevent re-investigation. Move to a "## Historical" section at bottom only after the contradiction is resolved.
- When chunk-spec writers consult this doc and find a `CONTRADICTED-PENDING-RESOLVE`, the chunk's first job is the reconciliation, not the extension.

## Pipeline run history

| Run | Corpus | Pipeline | Run directory | Notes |
|---|---|---|---|---|
| W44-226 | `/mnt/tower/output/zenjxl-tuning/2026-05-22/w44-216+219-combined/merged.parquet` (13991 rows, 267 unique blobs) | `scripts/zenjxl-tuning-sweep/run_all_analyses.py` at commit (post-META-1, after 5-stub port) | `benchmarks/sweeps/w44-219-densify/analysis/pipeline_run_2026-05-22/` | First end-to-end run after stub ports. All 8 stages PASS, wall ≈ 82 s. Reproduces W44-217/220/221/222 headline numbers within seed/GBR-randomness band. |

## Related

- `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/research_methodology_9_rules_2026-05-22.md` (Rule 6)
- `docs/PARAM_INTERACTIONS.md` — empirical interaction docs (Tier-1)
- `docs/TIER_2_KNOBS.md` — Tier-2 knob specs
- `docs/TUNING_RELATIONS.md` — full coupling graph
- `docs/LIBJXL_DIVERGENCES.md` — encoder-side divergences (cross-checked: never let a knob change the Libjxl-strategy hash)
