# W44-PHASE4-S2-c2-validate — analysis artifacts

This directory mirrors the `analysis/` convention established for
`benchmarks/sweeps/w44-phase4-s1-recon-deep-revalidate/analysis/`.

## Source

These TSVs were generated during the S2-c2-validate finalize from the
canonical merged Parquet on Tower
(`/mnt/tower/output/zenjxl/sweeps/w44-phase4-s2-c2-validate-2026-05-24/merged.parquet`)
and the matching S1 canonical
(`/mnt/tower/output/zenjxl/sweeps/w44-phase4-s1-recon-deep-revalidate-2026-05-24/merged.parquet`).

## Files

| file | shape | purpose |
| --- | --- | --- |
| `s1_vs_s2_diff_per_stratum.tsv` | 8 strata + header | mean / max |Δbytes%| per content stratum, S2 minus S1 |
| `s1_vs_s2_clean_per_stratum.tsv` | 8 strata + header | same, filtered to cells present in both |
| `s1_vs_s2_diff_strategy_stratum.tsv` | 16 rows | split by `EncoderStrategy::{Zenjxl, Libjxl}` |
| `s1_vs_s2_stratum_strategy.tsv` | 16 rows | per-stratum × strategy mean / median / shift counts |
| `s1_vs_s2_stratum_strategy_FINAL.tsv` | 16 rows | FINAL = post-validation, the version used in the S2f finalize memo |
| `s1_vs_s2_zenjxl_only_FINAL.tsv` | 8 strata | Zenjxl-only view of the FINAL table |
| `s1_vs_s2_shifted_cells_FINAL.tsv` | 3 cells + header | individual cells where bytes shifted >0.5pp between S1 and S2 |

## Headline

S2-c2-validate on a 9-image OOD subset showed the c2 floor change
(`screen/very_high` + `screen/high` k1+k2 default → defaults) was a
no-op on the validation corpus (3 shifted cells of 5,160 → 0.058%).

See the CLAUDE.md "Investigation Notes" section for
`W44-PHASE4-S2-refit-c2` for the full mechanism + bench narrative + the
S2f finalize memo for the disposition.

## Methodology references

- `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/`
  has the `w44_phase4_s1_finalize_2026-05-24.md` (S1) and the S2
  finalize memo capturing the kitchen-sink GBR + per-stratum optima
  + Pareto coverage outputs that informed the c2 refit.
- `scripts/zenjxl-tuning-sweep/run_all_analyses.py` is the 8-stage
  pipeline that generated the per-stratum optima and SVD basis used
  to drive both refits.
