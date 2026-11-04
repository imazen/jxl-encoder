# PQ failure attribution, 2026-11-04

The ten remaining issue-138 failures are not resolved. Seven pass the
unchanged signal-error gate when only the existing
`dc_encode_libjxl_parity` Custom field is enabled; three still fail.
The custom control reproduces all ten original errors and byte counts.
Every arm retains its source and encoded stream, and the driver checks
source identity against the original failing input.

[Measured pairs](issue138_dc_policy_2026-11-04.tsv) include input/output
SHA256s. [Run identity](issue138_dc_policy_2026-11-04.meta.json) identifies
the binary, base commit and retained artifacts. This is an attribution
experiment, not default adoption or broad HDR quality qualification.

The DC gate changes both extra DC precision and QuantizeWP scheduling.
This experiment does not separate those two effects. At effort 5 the
existing HDR dispatch already enables QuantizeWP, and the gate leaves
both reported failures unchanged (2965/49033 and 2966/14694). The third
remaining failure, 2966/66247, becomes slightly worse: mean signal error
0.038127 to 0.041433, against cjxl's 0.014093. It must remain visible in
any candidate acceptance grid. No production gate was changed here.

The wider option ablation keeps the original pixels and compares filters,
the full Libjxl strategy, and that strategy with resampling pinned to 1.
Disabling filters is not a general repair. Some variants pass because
cjxl gets worse, so a passing verdict alone is not an improvement claim.
Those controls are retained under
`~/tmp/jxl138-quality-ablations-matched-distance/`.

The experiment found another reference-harness mismatch: automatic
resampling rewrites our distance, while an explicit cjxl resampling
argument does not perform that rewrite. The reference now receives both
the effective distance and the effective factor. In particular, the
Libjxl d25 control uses distance 6.5 and factor 2 in both encoders.
The original Zen controls remain distance 25, factor 1. Earlier strict
controls under `~/tmp/jxl138-strict/` and
`~/tmp/jxl138-quality-ablations/` did not match that remapped distance;
they cannot qualify the strategy's relative quality.

Reproduce with `just issue138-ablate` or `just issue138-dc-ablate`, supplying
the original replay directory, exact corpus inputs, verified v0.12
reference tools and the resource wrapper. The sweep's diagnostic
`--options`, `--effort` and `--retain` flags require `--only`; they cannot
silently alter the ordinary random sweep. Retention keeps source and
encoded streams, and reference inputs/streams, for inspection.

DC-pair run: 16 seconds, peak-RSS 0.31 GiB, minimum available 25237 MiB,
peak load 0.99; `~/tmp/jxl138-dc-parity-ablation.log`. These observations
describe the diagnostic process and are not codec performance claims.

The final retention verification repeats all twenty encodes with the saved
binary `~/tmp/jxl138-reviewed-sweep`. Every input, encoded stream and study
row matches the first run exactly; reference rasters, streams and decoder
output are also retained in
`~/tmp/jxl138-dc-parity-final-retained/`. Wrapper: 16 seconds, peak-RSS
0.31 GiB, minimum available 25296 MiB, peak load 0.84. Log:
`~/tmp/jxl138-dc-parity-final-retained.log`.
