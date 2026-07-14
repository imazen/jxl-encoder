# Scoreboard CJXL-DOMINATES — strict multi-metric Pareto re-classification (2026-07-14)

_git 

**Input**: `scoreboard_2026-06-12_post_wedges.tsv` (280 cells; 112 CJXL-DOMINATES).
**Script**: `reclassify_multimetric_2026-07-14.py` (re-analyses the committed TSV,
no re-encoding). q1 = butteraugli (smaller better), q2 = ssim2 (larger better).
Noise floors: butteraugli 2% relative, ssim2 0.25 abs, bytes 2%.

## Result — 112 "CJXL-DOMINATES" is 60% mislabel

| bucket | count | meaning |
|---|---:|---|
| REAL_LOSS (SDR) | 34 | cjxl not-worse on both quality axes AND meaningfully smaller — genuine gap |
| HDR_REAL_LOSS | 11 | HDR, ours butteraugli worse beyond 2% — likely real (needs HDR metric to confirm) |
| **CONFIRMED/LIKELY REAL** | **45** | **the actual "match always" work** |
| TRADEOFF_QUALITY | 5 | ours strictly BETTER on a quality axis, bigger bytes — we bought quality (MISLABEL) |
| NEAR_TIE (SDR) | 14 | \|bytes\| < 2% and no quality axis worse (MISLABEL) |
| HDR_NEAR_TIE | 48 | HDR butteraugli within 2% — noise-level (MISLABEL) |
| **MISLABEL (tie/tradeoff)** | **67** | not real losses |

**The scoreboard headline overstates the lossy+lossless cjxl gap by ~2.5x.**
The single biggest bucket (48 HDR near-ties) is butteraugli-within-2% noise; the
"hdr-lossy 59/96 cjxl-dominates" summary line is really ~11 real + 48 noise-ties
+ needs a real HDR metric (cvvdp/vdp2) to judge at all.

## The 45 real losses cluster into 4 addressable groups

| group | cells | character | route |
|---|---:|---|---|
| **lossless graphics** | 14 | documents / screenshots / clipart / textures; pure bytes (both lossless) | deep modular predictor/context RD on high-background-dominance content (HYPOTHESIS_LEDGER #24 — no clean global lever) |
| **sdr-lossy** | 15 | mostly marginal (<5% bytes near-ties); standout = **plots/line-art** (line-00081 e7 d0.5 +19% AND worse quality, 6 cells) + noaa-doc d4 (+9%) | investigate line-art/plot cluster; the rest are ~ties |
| **size-axis 64x64** | 5 | tiny crops, quality PERFECT both (ssim2 100), pure +29% bytes = fixed header/signaling overhead | small-image fixed-overhead (size-sweep artifact, low priority) |
| **HDR** | 11 | ours butteraugli worse; needs cvvdp/vdp2 to confirm | HDR calibration IF a real HDR metric confirms |

## Implications for "win or match always"
- Lossy **strategy-routing has no target** (Zenjxl already dominates Libjxl on
  50/50 cells — see `zenjxl_vs_libjxl_routing_pilot_2026-07-14`). The gap is NOT
  mis-routing between our strategies.
- The real, prioritised work is: (1) lossless-graphics modular RD (14),
  (2) the sdr-lossy plots/line-art cluster (~6), (3) an HDR quality metric to
  turn 59 HDR "losses" into a real count (11 candidates), (4) size-axis
  fixed-overhead (5, low priority).
- Scoreboard v2 MUST use strict multi-metric Pareto (this script's rule) and an
  HDR-appropriate metric, and MUST filter cjxl PNG-read failures (see
  `lossless_graphics_gap_probe_2026-07-14.meta`).
