# W44-219 — densify sweep

**Sweep ID:** `w44-219-densify`
**Date:** 2026-05-22
**Provenance:** Phase A chunk 3 of the 3-tier zenjxl design
(see `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/zenjxl_mode_design_goal_2026-05-22.md`).

## Why

W44-218 (`2ecc6f8f`) shipped 7 per-pair coupling ridges through
`tuning.rs::coupling`, but the per-pair response R² fits HONEST-STOPPED
below the 0.5 acceptance gate because the W44-216 corpus only had
**13 distinct LHS param blobs** — within-stratum noise dominated.

W44-219 produces a *denser* corpus W44-220 can refit on:

1. **Parameter blobs**: 150 LHS samples (scipy.stats.qmc.LatinHypercube,
   seed=44219, scrambled) + 5 pair-focused 2D grids (5×5 each over the
   top couplings from W44-217 interaction_ranking) = **255 unique
   blobs** (vs 13 in W44-216).
2. **Images**: 28 W44-216 corpus + 9 new (clic2025 validation, CID22
   training, gb82 lossless) = **37 images** (vs 27).
3. **Effort/distance**: same 5 efforts × 7 distances × 2 strategies
   grid as W44-216. Increased blob density alone takes e9 screen
   rows from 79 → ~870 (11×).
4. **Strategy axis pruned**: per W44-217 finding #1 ("6 RuntimeTuning
   params ONLY affect zenjxl strategy. libjxl CV ≤ 0.03 %"), libjxl
   is run only at the `defaults` blob (~1,295 libjxl cells × all
   blobs ≠ wasted compute).

## Cell budget

Theoretical max: 239,645 cells. Cost-cap $30 will drain a subset
proportional to fleet throughput (W44-216 produced 25K cells at $2.50
in 75 min; expect 30-150K cells here).

## Files

- `manifest.tsv` — top-level sweep metadata (cells, blobs, images, seeds)
- `blob_provenance.tsv` — per-blob source (defaults / lhs / grid_pNxpM)
  + decoded (p1..p6) values
- `lhs_design.json` — full LHS sample matrix (scipy `seed=44219` reproducible)
- `merged.parquet` (post-sweep) — canonical merged cells
- `merged.meta` (post-sweep) — provenance
- `combined_with_w44_216.parquet` (post-sweep) — W44-216 + W44-219 concat

## Reproducer

```sh
# Generate
python3 scripts/zenjxl-tuning-sweep/build_w44_219_chunks.py \
    --sweep-id w44-219-densify

# Upload
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
AWS_PROFILE=r2 aws s3 sync /tmp/w44-219/corpus s3://zen-tuning-ephemeral/corpus/ \
    --endpoint-url "$R2_ENDPOINT"
AWS_PROFILE=r2 aws s3 sync /tmp/w44-219/params \
    s3://zen-tuning-ephemeral/w44-219-densify/params/ --endpoint-url "$R2_ENDPOINT"
AWS_PROFILE=r2 aws s3 sync /tmp/w44-219/chunks \
    s3://zen-tuning-ephemeral/w44-219-densify/chunks/ --endpoint-url "$R2_ENDPOINT"

# Smoke (1 box, ~30min wall)
SWEEP_ID=w44-219-densify BOXES=1 LABEL_PREFIX=claude-w44-219-smoke \
    bash scripts/zenjxl-tuning-sweep/launch_w44_219_fleet.sh

# Fleet (15-25 pods, ~2-6h wall, $30 cap)
SWEEP_ID=w44-219-densify BOXES=20 LABEL_PREFIX=claude-w44-219-fullgrid \
    bash scripts/zenjxl-tuning-sweep/launch_w44_219_fleet.sh
bash scripts/zenjxl-tuning-sweep/janitor_w44_219.sh claude-w44-219- w44-219-densify

# Merge
aws s3 sync s3://zen-tuning-ephemeral/w44-219-densify/cells/ \
    /tmp/w44-219-cells/ --endpoint-url "$R2_ENDPOINT"
python3 scripts/zenjxl-tuning-sweep/merge_w44_219_cells.py \
    --in-dir /tmp/w44-219-cells \
    --out-dir /tmp/w44-219-merged \
    --w44-216-parquet /mnt/tower/output/zenjxl-tuning/2026-05-22/w44-216-stage-b/merged.parquet
```

## Follow-on

- **W44-220** (queued): refit the W44-218 ridge saturation strengths +
  endpoints from the W44-216 + W44-219 combined corpus. Per-pair
  response R² ≥ 0.5 acceptance gate now achievable.
