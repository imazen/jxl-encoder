# Hypothesis ledger

**Per methodology Rule 6** (`~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/research_methodology_9_rules_2026-05-22.md`): a single document tracking what we currently believe about the encoder + the tuning corpus, the evidence each belief rests on, when last validated, and whether contradictions exist.

**Maintenance rule:** every chunk's acceptance memo updates ≥1 entry (or notes "no belief change"). When two entries contradict, the next chunk's first task is reconciling them.

**Status values:**
- `PROVEN` — direct empirical evidence; bit-equivalent or measured at ≥3σ
- `LIKELY` — strong evidence but indirect or single-source
- `SUSPECTED` — hypothesis with rationale; not yet measured
- `RULED-OUT` — measured + falsified
- `CONTRADICTED-PENDING-RESOLVE` — two beliefs in conflict; next chunk reconciles

## Seeded 2026-05-22 from W44-217..221 retrospective

| # | Belief | Status | Evidence | Last validated |
|---|---|---|---|---|
| 1 | The 6 RuntimeTuning params ONLY affect `EncoderStrategy::Zenjxl`; on `Libjxl` strategy, CV ≤ 0.03 % across all blobs. | PROVEN | W44-217 sanity_check.py on 4938-row W44-216 corpus | 2026-05-22 |
| 2 | Per-pair coupling-fn refit (W44-218 algebraic forms) cannot reach R² ≥ 0.5 on the response. | PROVEN | W44-220 measurement on 21×-denser W44-219 corpus; 0/7 pairs hit gate | 2026-05-22 |
| 3 | Joint-surface GBR over `(p1..p6 + 25 features + effort + distance) → outcome` hits test R² ≥ 0.85 on every stratum × outcome. | PROVEN | W44-221 phase 1 on combined W44-216+W44-219 corpus | 2026-05-22 |
| 4 | The joint output surface has natural rank 4-5 in 6-param space (rank-4 = 88.3 %, rank-5 = 96.1 % of gradient variance). | PROVEN | W44-221 phase 2b gradient-SVD on 40 anchor cells × 2 outcomes | 2026-05-22 |
| 5 | The W44-218 ridge basis spans 68.5 % of joint gradient variance. PC2/PC5 directions are NOT covered — 31.5 % missing variance lives there. | PROVEN | W44-221 phase 3 V-matrix projection | 2026-05-22 |
| 6 | A 4-knob Tier-2 expander achieves ≤2pp mean Pareto coverage on every stratum but FAILS strict 0.5pp-max on `screen/very_high` (max gap 7.86 %). | PROVEN | W44-221 phase 4b coverage measurement | 2026-05-22 |
| 7 | W44-217 §10 claim "pairwise interactions dominate (singles 0.3-5%, pairs 6-22%)" CONTRADICTS belief #3 (joint dominates). | CONTRADICTED-PENDING-RESOLVE | W44-217 ANOVA tables vs W44-221 phase 1 GBR | 2026-05-22 |
| 8 | Adding (effort, distance, features) as inputs is what closes the W44-218 → W44-221 R² gap (not corpus density). | PROVEN | W44-221 phase 1 found this directly; W44-219 9K densification did NOT move per-pair R² | 2026-05-22 |
| 9 | vast.ai default-launch is on-demand pricing; we are paying for spot-grade reliability without the spot discount. ~50-70% savings available via `--bid_price`. | PROVEN | W44-216 had 12/24 evictions; janitor handles them; CLI confirms `--bid_price` is the interruptible flag | 2026-05-22 |
| 10 | `EncoderStrategy::Libjxl` produces byte-identical bitstream regardless of RuntimeTuning override (the override only affects Zenjxl path). | PROVEN | W44-194 strategy_libjxl_byte_lock CI gate; never broken since W44-193 | 2026-05-22 |
| 11 | Default knob values in `Tier2Knobs::default()` round-trip byte-identical to current `Zenjxl` production (no behaviour change at defaults). | PROVEN | W44-221 hash_lock_features 36/36 byte-identical | 2026-05-22 |
| 12 | `LossyConfig::with_knobs(Tier2Knobs)` builder needed to plumb knobs through encoder entry — current `runtime::install` mechanism is single-shot. | LIKELY | W44-221 memo + W44-222 currently in flight implementing this | 2026-05-22 |
| 13 | Speculative algebraic forms (additive / multiplicative / gated / suppressive / synergistic) cannot capture the joint response surface — non-parametric + SVD basis is strictly better. | PROVEN | W44-218→220 (0/7 fits) vs W44-221 (R² ≥ 0.85) | 2026-05-22 |
| 14 | A 5th data-driven knob spanning the PC2/PC5 direction can close the `screen/very_high` 7.86 % Pareto gap from belief #6. | SUSPECTED | W44-221 phase 3 identified the unspanned variance; W44-222 in flight testing | 2026-05-22 |

## How to update

- After each chunk: add new row OR update existing `Last validated` date + Status.
- Don't delete RULED-OUT or CONTRADICTED rows — they prevent re-investigation. Move to a "## Historical" section at bottom only after the contradiction is resolved.
- When chunk-spec writers consult this doc and find a `CONTRADICTED-PENDING-RESOLVE`, the chunk's first job is the reconciliation, not the extension.

## Related

- `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/research_methodology_9_rules_2026-05-22.md` (Rule 6)
- `docs/PARAM_INTERACTIONS.md` — empirical interaction docs (Tier-1)
- `docs/TIER_2_KNOBS.md` — Tier-2 knob specs
- `docs/TUNING_RELATIONS.md` — full coupling graph
- `docs/LIBJXL_DIVERGENCES.md` — encoder-side divergences (cross-checked: never let a knob change the Libjxl-strategy hash)
