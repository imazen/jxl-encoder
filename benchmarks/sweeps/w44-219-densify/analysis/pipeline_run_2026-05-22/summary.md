# Sweep analysis summary

Corpus: `/mnt/tower/output/zenjxl-tuning/2026-05-22/w44-216+219-combined/merged.parquet` (13991 rows)

| Stage | Status | Duration | Headline |
|---|---|---|---|
| kitchen_sink_gbr | PASS | 4.6s | encoded_bytes=0.997 ssim2=0.996 butter_norm3=0.991 cvvdp=0.995 encode_ms=0.884 |
| per_pair_gbr | PASS | 0.9s | encoded_bytes=-0.013 ssim2=-0.037 |
| anova | PASS | 1.0s | encoded_bytes=0.555 ssim2=0.828 butter_norm3=0.836 cvvdp=0.736 encode_ms=0.667; top_log_bytes: z_p3_butt_qf_scale=0.11%, z_p2_screen_median=0.10%, z_p6_aq_qf_e7=0.08% |
| marginal_pdps | PASS | 7.7s | ADDITIVE=30 |
| stratum_pdps | PASS | 9.1s | plotted=8 skipped=0 |
| svd_basis | PASS | 7.3s | rank-4=88.5% rank-5=96.1% n_anchors=40 |
| mi_matrices | PASS | 6.6s | encoded_bytes: p2_screen_median=0.046; ssim2: p2_screen_median=0.079; butter_norm3: p2_screen_median=0.061; cvvdp: p2_screen_median=0.081; encode_ms: p3_butt_qf_scale=0.011 |
| pareto_coverage | PASS | 45.2s | screen/very_high 5-knob max=0.63% (Δ from 4-knob: +10.15pp) |

**RULE 1 CHECK**: compare kitchen_sink_gbr vs per_pair_gbr. If kitchen_sink R² is materially higher, dropped axes (likely effort/distance/features) explain the per-pair shortfall — see `research_methodology_9_rules_2026-05-22.md` Rule 1.