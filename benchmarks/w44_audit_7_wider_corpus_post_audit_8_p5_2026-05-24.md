# W44-AUDIT-8 P5b: 240-cell wider-corpus validation of Phase 5

**Compared**: pre = W44-AUDIT-7 baseline (`w44_audit_7_wider_corpus_2026-05-24.tsv`); post = post-W44-AUDIT-8-P5 (`w44_audit_7_wider_corpus_post_audit_8_p5_2026-05-24.tsv`).

**Joined**: 480 (image, effort, distance, strategy) cells.

## Per-class mean Δ (post-Phase-5 vs AUDIT-7 baseline)

| class | strategy | e5 dB% | e5 dSSIM2 | e7 dB% | e7 dSSIM2 | e9 dB% | e9 dSSIM2 |
|---|---|---|---|---|---|---|---|
| CLIC2025_WEB | zenjxl | +5.32% | +1.47 | +5.01% | +1.63 | +0.00% | +0.00 |
| CLIC2025_WEB | libjxl_strat | +4.77% | +1.62 | +4.57% | +1.49 | +0.00% | +0.00 |
| PHOTO_LANDSCAPE | zenjxl | +3.29% | +0.71 | +3.18% | +0.81 | +0.00% | +0.00 |
| PHOTO_LANDSCAPE | libjxl_strat | +3.14% | +0.76 | +3.04% | +0.77 | +0.00% | +0.00 |
| PHOTO_PORTRAIT | zenjxl | +5.79% | +1.28 | +5.54% | +1.42 | +0.00% | +0.00 |
| PHOTO_PORTRAIT | libjxl_strat | +5.37% | +1.37 | +5.00% | +1.21 | +0.00% | +0.00 |
| PHOTO_SMOOTH | zenjxl | +3.31% | +1.03 | +3.21% | +1.14 | +0.00% | +0.00 |
| PHOTO_SMOOTH | libjxl_strat | +3.10% | +1.12 | +2.87% | +0.76 | +0.00% | +0.00 |
| SCREEN_GRAPHICS | zenjxl | +1.63% | +0.66 | +1.52% | +0.73 | +0.00% | +0.00 |
| SCREEN_GRAPHICS | libjxl_strat | +1.75% | +0.54 | +1.60% | +0.35 | +0.00% | +0.00 |
| SCREEN_TEXT | zenjxl | +2.67% | +1.21 | +2.27% | +1.27 | +0.00% | +0.00 |
| SCREEN_TEXT | libjxl_strat | +1.78% | +1.46 | +2.14% | +1.00 | +0.00% | +0.00 |

## WIN/NEUTRAL/LOSS classification (zenjxl only)

- WIN: **70** (SSIM2 > +0.5)
- NEUTRAL: **170**
- LOSS: **0** (SSIM2 drops > 0.5 OR bytes > +5%)

## Aggregate verdict

- **zenjxl** (240 cells): mean dB% = +2.482%, mean dSSIM2 = +0.772
- **libjxl_strat** (240 cells): mean dB% = +2.267%, mean dSSIM2 = +0.726

## Per-effort aggregate (zenjxl)

| effort | n | mean dB% | mean dSSIM2 | notes |
|---|---|---|---|---|
| e5 | 80 | +3.834% | +1.103 | gate fires |
| e7 | 80 | +3.611% | +1.211 | gate fires |
| e9 | 80 | +0.000% | +0.000 | gate OFF (control) |
