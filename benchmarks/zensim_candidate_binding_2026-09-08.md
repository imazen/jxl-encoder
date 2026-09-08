# Complete zensim candidates in the native JXL loop — September 8, 2026

Custom bakes now score and spatialize each reconstruction through zensim’s
`BakeScorer` surface. The feature-width probe, custom-profile mount and
first-image gradient no longer define a custom bake’s semantics. Named-profile
defaults remain on their historical route. This is a serving repair and a
development RD screen; no model or spatial strategy is qualified for release.

## Binding and controls

`JXL_ZENSIM_RD_PROFILE=bake:<path>` loads per encode. Fresh attribution arms
call `compute_with_ref_and_attribution` on the current linear RGB reconstruction
and cached reference. Complete scoring also judges independently decoded
bitstreams. Explicit stale arms preserve map lag; split-role experiments use
a separately loaded complete map scorer. Active unsupported spatial terms
or corruption gating refuse. Historical signed/abs signal-fold arms refuse
for custom bakes; use an attribution-family arm. No new public encoder API
or controller/gain/seed constant was introduced. Negative scores are preserved.

The shared jxl-rs decoder owner now decodes the RD harness output. A bake
passed to its distance-grid mode is honored, with a separate full-precision
candidate ladder and saved bitstreams; low distances cannot overwrite each
other through two-decimal filenames. The existing trace column contracts
remain unchanged. Its staircase/old seed-head target mode is still historical;
it is not canonical train-family calibration.

## Evidence

- Existing attribution/H-arm tests pass. The candidate test proves nonconstant
  maps, active/neutral bitstream and jxl-rs pixel changes, one fresh map per
  reconstruction, per-encode model loading and repeat determinism.
- Reintroducing a first-bake process cache fails the changed-byte assertion.
  Restoring the implementation passes; the mutation is not retained.
- Both default routes remain byte-identical before/after: 16 cells each for
  the zensim loop and metric backend, over two real images and two odd-size
  procedural fixtures, two distances and two efforts.
- The 840 ladder bitstreams decode through jxl-rs. libjxl `djxl` v0.12.0
  additionally decodes 24 validation low/middle/high-distance bitstreams.
- Full default workspace tests: 2,320 passed, 228 ignored. Clippy passes for
  all workspace targets with default features and with `zensim-loop`.
  `just rd-regression` passes its two tests. Formatting/diff checks pass.

## Attained ranges and spatial RD screen

D baseline: `d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin`, SHA256
`cd1098b450ef6941b6925b24bcbd129715b6f07c4fe84838a92e13ab364ddea6`.
Formula revision 1. Sources reuse the recorded canonical imazen-26 split
intersection: 12 training and 8 validation families, long edge 256, with
photo/document/graphic/screen coverage. These are development families,
not a terminal holdout or broad product coverage.

Each source has 21 logarithmic distances from 0.01 to 25, effort 8, Zenjxl
strategy, default no resampling, two native quantization updates and no
target controller. H3 gain 0 is neutral; gain 10 is the existing active arm.
No gains or seeds were fitted in this screen. Each encode uses three internal
reconstructions on this configured path; complete-encode and map/score cost
instrumentation for target comparisons remains required.

All 840 ladder rows complete. Active maps change 242/252 training and 160/168
validation bitstreams. Neutral validation attained floors range from −40.69
to +17.45, with ceilings 99.20–100.00. These are witnessed extrema of the
sampled configuration, not proofs of the codec’s absolute mathematical bounds.
The controller is not given these per-image curves.

Independent CPU SSIMULACRA2 and Butteraugli pnorm3 score all 336 validation
reconstructions. The existing `rd_probe_analyze_2026-07-18.py` owner computes
log-byte interpolation within each image/judge’s neutral ladder, with no
extrapolation. Positive savings mean fewer bytes at matched judge quality.
Near-zero medians are reported as zero, not as positive wins.

| Content | Judge | Matched cells | Median saved % | Mean saved % |
|---|---|---:|---:|---:|
| document | butter | 41 | -0.217 | -0.352 |
| graphic | butter | 41 | 0.302 | 0.951 |
| photo | butter | 41 | 0.000 | -0.282 |
| screen | butter | 41 | 0.305 | 0.761 |
| document | ssim2 | 39 | 0.680 | -0.660 |
| graphic | ssim2 | 40 | 0.000 | -1.263 |
| photo | ssim2 | 42 | 0.000 | -0.311 |
| screen | ssim2 | 41 | 0.519 | 0.999 |

The spatial effect is real, but a broad RD benefit is not established. Photo
medians are approximately zero and means regress slightly on both judges;
documents regress on Butteraugli. Sparse near-identity interpolation and the
large worst-cell losses need direct matched-quality confirmation before a
gate decision. Good attribution coherence alone did not establish better
encoding. D remains a baseline, not the selected final model.

## Reproduction and remaining work

Encoder parent `690f09f5e3413f3df8c4428f9a6d365f78a60c15` includes the
latest decoder-reconstruction fixes; zensim surface `b33d6199f895`.
Frozen ladder binary SHA256
`a393989833cec863b914c17789ce0156dc3caa3e616411af69779f7159a8a2a1`;
independent judge binary SHA256
`6cb4744c0f1baa178c46c307db3f7a40c679dba2bf3d1ecb083a9d21f1e8cb6f`.
The judge uses explicit legacy CPU dispatch for stable backend selection.

Artifacts at `/mnt/v/output/zensim/jxl-candidate-binding-2026-09-08/` retain
input hashes/split manifests, exact commands and environment, source patch,
binaries, encoded/decoded ladders, independent scores, negative-control logs
and default parity TSVs. Timing here is small-image development work; no
production latency claim is made.

Next: fit native seeds only on training families, measure actual 1/2/3 full
encodes with separate internal-work accounting, expand/refine matched-quality
RD checks, and compare competitive complete models. JPEG/AVIF/WebP and
HDR/color/alpha qualification remain separate required work.
