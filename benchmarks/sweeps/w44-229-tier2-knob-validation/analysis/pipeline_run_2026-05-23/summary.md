# Sweep analysis summary

Corpus: `/mnt/tower/output/zenjxl-tuning/2026-05-23/w44-229-tier2-knob-validation/merged.parquet` (21667 rows)

| Stage | Status | Duration | Headline |
|---|---|---|---|
| kitchen_sink_gbr | PASS | 2.7s | encoded_bytes=0.998 ssim2=0.992 butter_norm3=0.989 cvvdp=0.981 encode_ms=0.861 |
| per_pair_gbr | PASS | 0.1s | encoded_bytes=0.008 ssim2=0.015 |
| anova | PASS | 0.9s | encoded_bytes=0.732 ssim2=0.814 butter_norm3=0.830 cvvdp=0.678 encode_ms=0.757; top_log_bytes: z_p3_butt_qf_scale=0.11%, z_p6_aq_qf_e7=0.06%, z_p5_aq_qf_e56=0.06% |
| marginal_pdps | PASS | 6.0s | ADDITIVE=29 GATED=1 |
| stratum_pdps | PASS | 9.5s | plotted=8 skipped=0 |
| svd_basis | PASS | 10.5s | rank-4=90.2% rank-5=96.4% n_anchors=32 |
| mi_matrices | PASS | 12.2s | encoded_bytes: p1_mask_p25_min=0.000; ssim2: p1_mask_p25_min=0.000; butter_norm3: p1_mask_p25_min=0.000; cvvdp: p1_mask_p25_min=0.000; encode_ms: p1_mask_p25_min=0.000 |
| pareto_coverage | PASS | 45.3s | screen/very_high 5-knob max=0.00% (Δ from 4-knob: +1.16pp) |

**RULE 1 CHECK**: compare kitchen_sink_gbr vs per_pair_gbr. If kitchen_sink R² is materially higher, dropped axes (likely effort/distance/features) explain the per-pair shortfall — see `research_methodology_9_rules_2026-05-22.md` Rule 1.