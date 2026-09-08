# Unlifted distance policy comparison, 2026-09-08

Build: `92d53a697a175bd20c83be0f93094662d07fcdb9`, preserved on
[`research/issue103-unlifted-probe-build`](https://github.com/imazen/jxl-encoder/tree/research/issue103-unlifted-probe-build).
Host: `mac`. Completed 4,536 cells: 9 source images × four crop caps
(64, 256, 1024, 4096) × efforts 5, 7, 8 × 21 distances × two policies.
Small sources are not upscaled; some caps therefore repeat the same geometry.
This is a policy ablation, not a fitted threshold or a representative percentile
estimate. There are only nine source images, including two photos.

`legacy` explicitly enables both seed lifts and both adaptive iteration flags.
`unlifted` disables those four choices and retains the other Zenjxl settings.
Every cell fully rendered with jxl-rs, djxl v0.12, and jxl-oxide. The comparison
scores normalized RGB8 input and linear decoded output with the same scorer.

Of 2,160 adjacent requested-distance steps per policy, byte increases above
10% fell from 60 to 17. For steps starting at distance 2 or above, they fell
from 24 to zero. The remaining largest increase is the line-art 7026 step
0.5→0.75: +31.84% bytes at effort 8 while Butteraugli worsens 0.5234→0.9226.
That separate filter-boundary case still needs the C++ reference comparison.
These counts include repeated geometries and are not population prevalence.

Removing the lifts is not a universal RD improvement. Among 576 cells where
the policies changed bytes and the unlifted grid bracketed legacy quality,
the legacy/unlifted matched-quality cost median was 1.00; legacy was over 2%
cheaper on 251 and over 2% dearer on 229. The comparison uses the cheapest
measured point reaching the required quality, without interpolation. Its
coarse quality grid can overstate the legacy wins. The rationale for removing
these discontinuities is preserving requested-distance semantics, not claiming
free compression. Production defaults have not changed in this report.

Raw data and artifacts:

- Result directory: `/Users/lilith/tmp/jxl103/unlifted-results-2026-09-08/`.
- Encodes, diffmaps, per-run metrics and metadata:
  `/Users/lilith/tmp/jxl103/unlifted-2026-09-08/`.
- Remote prefix: `s3://zen-tuning-ephemeral/jxl-encoder/issue103-unlifted-2026-09-08/`.
  Upload verification is recorded when the transfer completes.
- [Result file hashes](qfseed_unlifted_2026-09-08.manifest.tsv) and
  [source selection](qfseed_targeting_inputs_2026-09-08.tsv).
- `meta.json` records the full command grid and probe SHA256; each per-run
  metadata file records build commit and normalized source SHA256. Every cell
  row names content-addressed JXL and f32 diffmap files and all four p-norms.
- Tower mirror: not yet made. Local files have not been deleted.

Reproduce analysis with `just qfseed-lift-analyze <result-directory>`.
