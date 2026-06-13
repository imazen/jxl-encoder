# Benchmarks — reproduction index

This directory holds committed benchmark results (TSV + `.meta` companion
files) and the scripts that produce them. Every comparison number in the
top-level `README.md` traces to a file here. The headline boards as of the
last measurement (2026-06-12, binary `3f025244`):

- **Bytes + quality scoreboard** —
  `scoreboard/scoreboard_2026-06-12_run4.tsv` (+ `_summary.md`, `_diff.md`,
  `.meta`). 280 cells across SDR lossy, SDR lossless, HDR lossy, and a
  fixed-overhead size axis.
- **Wall-time grid** — `scoreboard/wall_grid_2026-06-12.tsv` (+ `.meta`).
  40 cells, 5 strata × e{5,7} × {1,8} threads × {lossy, lossless}.

## A note on wall-time numbers (read before quoting any of them)

Wall numbers are only disposition-grade when measured on a quiet box.
`scripts/scoreboard/wall_grid.py` **refuses to run** when `loadavg(1m)`
> 1.0 and rebuilds the release CLI at the current checkout first (a
stale-binary guard that bit two earlier attempts). Anything tagged
"busy-box", "INDICATIVE ONLY", or `EXPLORATORY-` in a `.meta` is
exploration-only and must not feed decisions — see
`REVISIT_QUEUE_2026-06-11.md`. Bytes and quality are deterministic and
load-immune, so the scoreboard's bytes/quality verdicts hold regardless
of box state; the wall axis does not.

For new wall micro-benchmarks, prefer zenbench
(`~/work/zen/zenbench/`) over criterion — interleaved round-robin
execution kills the thermal/turbo A/B bias that back-to-back runs bake in.
The two scoreboard wall harnesses already use interleaved paired execution
in that spirit.

## Scripts under `scripts/scoreboard/`

| Script | What it does |
|---|---|
| `run_scoreboard.py` | Runs the bytes+quality scenario matrix, emits one verdict per cell (WE-DOMINATE / TIE / MIXED / CJXL-DOMINATES) with axis deltas. |
| `wall_grid.py` | Quiet-box paired wall-time grid; refuses under load. |
| `summarize_scoreboard.py` | Rolls a scoreboard TSV up into the `_summary.md` table. |
| `compare_scoreboards.py` | Diffs two scoreboard TSVs into a `_diff.md` (flip table). |

## How to reproduce each board

All commands run from the repo root. They assume a release CLI at
`target/release/cjxl-rs`, a built `cjxl`/`djxl` at
`~/work/jxl-efforts/libjxl/build/tools/`, and `zen-metrics` at
`~/work/zen/zenmetrics/target/release/` (paths are defined at the top of
`run_scoreboard.py`). cjxl here is libjxl v0.12.0.

### Bytes + quality scoreboard (`scoreboard_2026-06-12_run4.tsv`)

```bash
cargo build --release -p jxl-encoder-cli
python3 scripts/scoreboard/run_scoreboard.py benchmarks/scoreboard/out.tsv
python3 scripts/scoreboard/summarize_scoreboard.py benchmarks/scoreboard/out.tsv
```

- **Corpus**: 13 imazen-26 "core" picks from
  `benchmarks/lossless_bench_set_2026-06-10.tsv` (SDR lossy + lossless), 43
  picks at e5 (lossless), 12 HDR crops from
  `/mnt/v/input/jxl-encoder/hdr-crops-512` (HDR lossy), plus 64²/256² center
  crops of 4 picks (size axis).
- **What it measures**: encoded bytes and perceptual quality, per cell.
  Quality is a two-metric guard for SDR (butteraugli p-norm 3 AND SSIM2
  must agree in direction beyond tolerance, else the quality axis is a TIE
  and the cell is flagged `MIXED-METRICS`); lossless requires
  pixel-exact decode for both encoders (cv2 `IMREAD_UNCHANGED`); HDR uses
  PQ-EOTF butteraugli @ 1000 nits single-metric (`HDR-SINGLE-METRIC`).
- **Tolerances** (from the runner): bytes tie when |Δ| < 0.1 %; SSIM2 tie
  band 0.25; butteraugli p-norm-3 tie band 2 % rel (abs floor 0.005); PQ
  butteraugli tie band 2 % rel (abs floor 0.02).
- **Normalization**: SDR PNG sources are rewritten pixels-only (cv2) to
  strip ancillary chunks — both encoders get identical bytes, and it
  sidesteps the PNG-metadata butteraugli trap documented in `CLAUDE.md`.
- **Wall axis is UNMEASURED** in this board by design (the `.meta` and the
  runner docstring both say so) — see the wall grid below for that axis.

### Wall-time grid (`wall_grid_2026-06-12.tsv`)

```bash
# Must be a quiet box — the script refuses if loadavg(1m) > 1.0.
python3 scripts/scoreboard/wall_grid.py benchmarks/scoreboard/wall_grid.tsv
```

- **Corpus**: one core pick per stratum (`photos-png`, `web-screenshots`,
  `plots`, `noaa-documents`, `ai-illustrations`) from
  `benchmarks/lossless_bench_set_2026-06-10.tsv`, normalized pixels-only.
- **What it measures**: median + min wall over 5 measured iterations after
  1 warmup per side, interleaved paired (c,o,c,o,…), at matched thread
  counts (1T and 8T), lossy d1.0 and lossless, e5 and e7. Budget column
  marks `ratio <= 1.2` per `docs/GOAL_BEAT_CJXL.md`.
- Every row carries `commit`, `host`, and `loadavg_at_cell` for provenance.

### Lossless A/B walls + byte-equality (`scripts/bench_lossless_ab.py`)

```bash
python3 scripts/bench_lossless_ab.py \
    --base /path/to/cjxl --ours target/release/cjxl-rs \
    --iters 6 --out benchmarks/lossless_ab.tsv \
    --cell clic097:~/work/codec-corpus/clic2025-1024/097cb*.png:7:1 \
    --decode-verify ~/work/jxl-efforts/libjxl/build/tools/djxl
```

- **What it measures**: per-cell median lossless wall for two binaries,
  interleaved paired, with sha256 byte-equality asserted every iteration.
  `--decode-verify` additionally pixel-compares each output against the
  source (byte-equality alone passes when both sides emit the same broken
  bitstream — issue #68 hid behind exactly that).
- **Cell spec**: `name:image_glob:effort:threads`.

## `just` recipes (committed-reference comparisons)

These compare against committed cjxl reference CSVs, using in-process Rust
butteraugli + SSIM2 (metadata-immune, decode via jxl-oxide in linear RGB),
so they avoid the PNG-metadata butteraugli trap entirely.

| Recipe | What it measures | Corpus |
|---|---|---|
| `just quality-compare` | Lossy RD vs cjxl (Rust butteraugli + SSIM2) | CID22 (committed reference CSV), ~2 min |
| `just rd-regression` | Regression floor, 6 images × 2 distances | CLIC 2025, ~3 min debug |
| `just rd-regression-hd` | High-distance RD (d=2.0, d=3.0; exercises DCT32x32 / DCT64x64) | CLIC 2025 |
| `just lossless-compare` | Lossless bytes vs cjxl | CSV-backed, ~1 min |
| `just cid22-vs-webp` | cjxl-rs vs libwebp | CID22 validation (41 images × 4 quality points) |

Reference CSVs are regenerated with `just generate-reference` (~40 min,
once per cjxl version) and `just generate-lossless-reference` (~10 min).

## Selected committed result files

| File | Subject |
|---|---|
| `scoreboard/scoreboard_2026-06-12_run4.tsv` (+ `_summary.md`, `_diff.md`, `.meta`) | Canonical 280-cell bytes+quality board |
| `scoreboard/wall_grid_2026-06-12.tsv` (+ `.meta`) | 40-cell quiet-box wall grid |
| `hdr_lossy_parity_postdispatch_2026-06-12.tsv` (+ `.meta`) | HDR lossy bytes + PQ-butteraugli, 12 crops × e{5,7} × d{.5,1,2,4} |
| `lossless_8bit_tree_lift_2026-06-12.tsv` (+ `.meta`) | 8-bit e5/e6 tree-learn lift calibration, 43+13 imazen-26 picks, pixel-exact |
| `hdr_quantize_wp_ab_2026-06-12.tsv` | QuantizeWP DC-shaping A/B (HDR smooth-sky bytes) |
| `lossless_bench_set_2026-06-10.tsv` (+ `.meta`) | The k-means lossless bench set (43 picks / 13 core across 23 imazen-26 strata) |
| `cvvdp_vs_buttloop_tracking_2026-05-24.tsv` | 1,134-cell perceptual-backend tracking sweep (butteraugli / CVVDP / zensim) |

Many more dated `perf_*`, `dispatch_*`, and `hdr*` TSVs in this directory
back individual tuning chunks; each carries a `.meta` with its provenance,
git commit, and the command used. The boards above are the ones the
top-level `README.md` cites.
