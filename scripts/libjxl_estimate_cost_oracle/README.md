# Global-stream palette cost oracle

`ref.cc` calls `jxl::EstimateCost` from the unmodified, pinned libjxl v0.12.0
(`a7a9c78`) static library. It does not transcribe or call our Rust estimator.
The 96 goldens were generated on macOS ARM64 on 2026-09-24. They cover eight
sizes, six sample distributions, original/palettized channels, metadata rows
and a second channel for whole-image fractional accumulation. Signed full-width
samples exercise the reference SIMD token conversion's large-integer shift.
These are component tests, not image-quality or performance measurements.

Regenerate from the repository root after building the pinned `.ci-libjxl`:

```sh
just libjxl-estimate-cost-oracle
nice -n 19 python3 scripts/libjxl_estimate_cost_oracle/generate.py \
  "$HOME/tmp/jxl-backlog/estimate-cost-ref" \
  > "$HOME/tmp/jxl-backlog/estimate-cost-goldens.tsv"
diff -u scripts/libjxl_estimate_cost_oracle/goldens.tsv \
  "$HOME/tmp/jxl-backlog/estimate-cost-goldens.tsv"
```

The Rust test `global_palette_cost_matches_libjxl_v012` requires exact integer
costs. The strict estimator uses the existing learner's canonical eight-lane
AVX2 reduction order; these 96 costs also agree with this host's C++ dispatch.
This is measured agreement on the grid, not a proof that native-width C++
reductions agree for every histogram on every architecture.
