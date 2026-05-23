# W44-218 analysis outputs

Per-pair coupling-ridge derivation for the W44-216 Stage B corpus.

## Files

| file | source script | contents |
|---|---|---|
| `fit_results.tsv` | `scripts/w44_218_fit_couplings.py` | top-5 pairs × 4-5 candidate models, train/test R² + MAE on 80/20 split with residualized fixed effects (effort + distance) |
| `fit_log.txt` | same | narrative log of all attempted fits |
| `ridge_params.json` | same | per-pair best-model parameters |
| `ridges.json` | `scripts/w44_218_derive_ridges.py` | calibrated ridge constants (P1_RIDGE_MAX, saturation strengths, etc.) + W44-216 LHS coverage diagnostics |
| `ridge_coverage.tsv` | same | per-knob-value (p1..p6) ridge sample for plotting |
| `derive_ridges.log` | same | narrative log of ridge derivation |

## Headline finding

**Per-pair response R² fits FAILED the 0.5 acceptance gate** on all 5 top
couplings (best ~0.08 on `p2_p5` ssim2 @ screen/very_high). Root cause:
the W44-216 corpus has only 13 distinct param blobs vs 27 images × 5
efforts × 7 distances of confounding variation — even after residualizing
(image, effort, distance) fixed effects, the within-stratum variance is
dominated by image-to-image noise the 2-param model cannot explain.

This was the **honest-stop trigger** in the W44-218 task spec. Per the
spec's honest-stop guidance, the W44-218 deliverable is therefore:

1. Ridge **geometry** calibrated from the W44-216 LHS empirical envelope
   (max bounds, saturation cap from top-N best-ssim2 blobs) — NOT from
   per-pair response fits.
2. Defaults round-trip byte-exact (hash-lock contract: 36/36 lossy +
   13/13 lossless fixtures unchanged).
3. W44-219 denser sweep (50+ LHS blobs queued) deferred for the per-pair
   response refit (W44-220 chunk).

## Reproducing

```bash
# 1) Per-pair response fit (run from repo root with /tmp/w44-217/corpus_prepped.parquet)
python3 benchmarks/sweeps/w44-216-stage-b/analysis/scripts/w44_218_fit_couplings.py

# 2) Ridge geometry derivation
python3 benchmarks/sweeps/w44-216-stage-b/analysis/scripts/w44_218_derive_ridges.py
```

Both scripts write to `/tmp/w44-218/`; the shipped artefacts here are a
copy of that dir at the time of the W44-218 commit.

## Provenance

- Corpus: `s3://zentrain/zenjxl-tuning/2026-05-22/w44-216-stage-b/merged.parquet`
  (Tower mirror: `/mnt/tower/output/zenjxl-tuning/2026-05-22/w44-216-stage-b/merged.parquet`)
- Prepped via W44-217 `scripts/prep_data.py` to add `p1..p6`, `z_p1..z_p6`,
  `content_class`, `is_default_params` columns.
- W44-217 analysis: `benchmarks/sweeps/w44-216-stage-b/analysis/`
  (ANOVA TSVs × 5, MI TSVs × 4, PDP PNGs × 38, classification TSV × 3).

## Cross-references

- Per-pair shipped formulas: `docs/PARAM_INTERACTIONS.md` Section 6.
- Tier-2 knob roadmap: `docs/PARAM_INTERACTIONS.md` "W44-218 status".
- Coupling fns: `jxl-encoder/src/tuning.rs::coupling`.
- Unit tests: `jxl-encoder/src/tuning.rs::coupling::tests` (15 tests).
