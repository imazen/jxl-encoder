# Zensim delivered-pixel contract — September 8 registration

Before decoder-probe implementation or results. Reused honest JXL map outputs
show up to 2.425 D-score units of drift between this harness's upstream jxl
f32/plain-round output and zensim's canonical zenjxl-decoder U8 output. The
canonical decoder defaults to blue-noise dithering; F32 output does not dither.

Use the existing zensim_diffmap_rd instrument with a private --decode-probe
mode. Add a dev-only alias to the same zenjxl-decoder 0.4 implementation used
by canonical extraction; retain the older JBRD/primary reference dependencies.
For the 336 retained training-fit JXL map-arm bitstreams, independently compare
canonical U8 with dithering on/off and the unchanged legacy f32 helper.
Pin original bytes, dimensions, old/canonical decoded hashes, source revision
and binary. Report exact matches and per-channel max/count absolute differences.
No encodes, model fitting, validation or terminal content in this diagnostic.

Only after canonical output agrees with the independently extracted hashes,
route this harness's delivered RGB8 through that owner and version its target
calibration config. Test malformed/missing inputs, hash/geometry/coverage
failures, fresh output enforcement and old-config rejection. Preserve the
prior scores as their original decoder era; neither compiler success nor a
decoder parity pass establishes target accuracy or a spatial RD win.

## Controlled result and next registered execution

The probe matches all 336 canonical hashes and all 336 historical hashes.
Switching dithering alone changes 8,646,810 RGB samples, never by more than
one code. With dithering off, 218 images match the historical decoder exactly;
the remaining decoder/rounding differences total 178 samples, also at most one
code. Dithering therefore accounts for nearly all pixel differences; it is not
the sole arithmetic difference between the decoder packages.

Route the harness's delivered RGB8 through canonical U8 with dithering enabled.
The retained legacy helper is diagnostic only. Append the explicit decoder
identity to CONFIG so old target calibrations cannot be reused. Rebuild and
repeat the 336-row probe through final source, then regenerate training ladders
and codec seeds on the unchanged 12 canonical training families. Independently
decode every newly emitted bitstream through the pinned canonical extractor,
requiring exact decoded hashes and D scores within f32 reporting roundoff.
Record every full encode and native reconstruction/map evaluation. This is
decoder/seed calibration repair; the D model and allocation policy remain
unqualified. Do not advance this failed model/policy to validation merely
because the decoder discrepancy is fixed.

## Completed repair and verification

The final native instrument reproduces the complete 336-row diagnostic exactly.
All 756 regenerated train outputs match independently decoded RGB hashes and
D scores after the harness's f32 reporting conversion. Maximum f64-to-f32
reporting difference is 0.0000037952242. All 756 emitted bitstreams are also
byte-identical to the previous native ladder: the observed change belongs to
delivered decoding/scoring, not changed encoding. Every old decoded hash differs.

The twelve original training sources, 21 distances and scalar/neutral/active
arms are unchanged. Work is 756 full encodes, 1,512 internal reconstructions,
1,512 native comparisons and 1,512 maps, plus 756 delivered-pixel comparisons.
Independent verification adds, per row, two decodes, one canonical extraction,
one candidate pixel comparison and four cached comparisons. Diagnostic probes
and refusal controls perform no encodes. No new model fit or validation/terminal
scoring occurs; the prior honest-head fit already used canonical pixels and its
two honest calibration false positives remain valid failures.

Fifteen refusal controls cover bad schema/counts, duplicate paths, input hashes,
zero/wrong geometry, unknown fields, missing/malformed/truncated input, existing
output preservation, mixed operations, and obsolete decoder configuration.
The stale-config control retains the *current* driver/model hashes, isolating
that guard, and refuses before encoding or output creation.

Local checks pass: scoped formatting, locked workspace/all-targets Clippy,
locked exact-example Clippy with its full feature set, 605 zensim script checks,
and cargo test (2,374 passed, 228 ignored, zero failures). The initial test run
found a legacy W44 fixture's hard-coded djxl unable to load an old OpenEXR
library. The successful full rerun explicitly selects the verified libjxl
0.12.0 executable. That W44 test's optional jxl_cli subprocess is unavailable;
its five djxl/oxide cells pass, and the separate in-process upstream-jxl probe
covers all 336 diagnostic bitstreams. Do not claim missing CLI coverage passed.

The dev-only canonical dependency preserves the older JBRD decoder separately.
No public API or production encoder code changes. The current decoder commit
8e3edefa adds only dev/measurement files over CI's pinned 17dc3030: package
source is unchanged. Builds/checks above use the recorded local dependency
closure; no fresh CI-closure build or CI result is claimed.

Artifacts: `/mnt/v/output/zensim/jxl-decode-contract-2026-09-08/`, mirrored to
`~/work/zensim-validation-2026-09-08/jxl-decode-contract/`. `RECIPE.json` pins
commands and environments; `SOURCE_INPUTS.json`, Cargo metadata, source copies,
model/binary hashes, `VERIFICATION.json`, `CONTROLS.json`, `ENCODER_CONTINUITY.json`
and `CHECKS.json` bind results. `verify.py` rechecks the retained packet.

Disposition: delivered-pixel inconsistency repaired. D, its corruption head,
and its fixed spatial policy remain unqualified. Next improve honest protection
and base-model preferences before a new registered native steering screen.

### Delivery rebase

The guarded push found incoming commits through 6ad2bd51, including experimental
LZ77 switches. Both histories were preserved by rebasing. After a locked release
rebuild, all 756 train bounds (encoded bytes, decoded hashes and scores), work
counts, seed curves and the 336-row diagnostic reproduce exactly. Calibration
changes only its driver hash. The obsolete decoder-config refusal passes with
the rebuilt driver too. This replay adds 756 full encodes, 1,512 internal
reconstructions/maps/native comparisons and 756 delivered comparisons.

Rebased scoped formatting, workspace/all-targets Clippy, exact-feature example
Clippy and the full local tests pass (2,378 passed, 228 ignored). The final
binary is `native-rebased`, SHA-256
`f87260124908e072dc311676c102c811d0d80f21e7b9d8d7fce767e33c81630c`;
use its matching `train-rebased/calibration.json`. Original artifacts remain
immutable. `REBASE_VERIFICATION.json` and `SOURCE_INPUTS_REBASED.json` retain
the actual comparison and source binding.
