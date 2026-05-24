# Sweep analysis summary

Corpus: `/mnt/v/zen/jxl-encoder/sweeps/w44-phase4-s1-recon-deep-revalidate/merged/merged.parquet` (22770 rows)

| Stage | Status | Duration | Headline |
|---|---|---|---|
| kitchen_sink_gbr | PASS | 2.8s | encoded_bytes=0.998 ssim2=0.975 butter_norm3=0.987 cvvdp=0.971 encode_ms=0.933 |
| per_pair_gbr | PASS | 0.3s | encoded_bytes=0.006 ssim2=0.013 |
| anova | PASS | 1.4s | encoded_bytes=0.732 ssim2=0.810 butter_norm3=0.817 cvvdp=0.674 encode_ms=0.821; top_log_bytes: z_p3_butt_qf_scale=0.07%, z_p5_aq_qf_e56=0.05%, z_p6_aq_qf_e7=0.04% |
| marginal_pdps | PASS | 6.2s | ADDITIVE=28 WEAKLY_COUPLED=2 |
| stratum_pdps | PASS | 10.0s | plotted=8 skipped=0 |
| svd_basis | PASS | 11.2s | rank-4=85.7% rank-5=98.2% n_anchors=32 |
| mi_matrices | PASS | 13.4s | encoded_bytes: p1_mask_p25_min=0.000; ssim2: p1_mask_p25_min=0.000; butter_norm3: p1_mask_p25_min=0.000; cvvdp: p1_mask_p25_min=0.000; encode_ms: p1_mask_p25_min=0.000 |
| pareto_coverage | PASS | 48.6s | screen/very_high 5-knob max=0.00% (Δ from 4-knob: +0.00pp) |

**RULE 1 CHECK**: compare kitchen_sink_gbr vs per_pair_gbr. If kitchen_sink R² is materially higher, dropped axes (likely effort/distance/features) explain the per-pair shortfall — see `research_methodology_9_rules_2026-05-22.md` Rule 1.