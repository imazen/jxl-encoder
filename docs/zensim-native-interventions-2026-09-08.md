# Native JXL finite-block interventions — 2026-09-08 registration

Registered before implementation/runs, after the completed JXL/AVIF native RD
screens and JPEG/WebP binding controls. Encoder base `8e2f0997fbe2` includes the
new high-bit-depth input work fetched before this packet. No model is qualified.

## Question and bounded experiment

D's pixel-restoration attribution coherence is strong, but native H3 allocation
has not produced a broad matched-RD win. Restoration score gain is not the same
quantity as score gain per extra encoded byte. Determine whether native local
quantizer interventions agree with map ranking and direction, and whether area
or rate cost explains poor allocation. Do not tune a gain or train a model from
an assumed diagnosis.

Reuse `EncoderPrecomputed::compute`, `VarDctEncoder::encode_from_precomputed`,
the production quant-field capture, the primary jxl-rs decoder, complete
`BakeScorer`, and the current source-manifest/PNG/scalar owners in the existing
`zensim_diffmap_rd` example. Add `--native-interventions <source-manifest>` mode
to that example, compiled with existing `__pre_quantized` and
`__internal_recon_hook` features. No new library API, feature, environment knob,
encoder controller or scoring implementation is proposed.

The exact D bake has SHA-256
`cd1098b450ef6941b6925b24bcbd129715b6f07c4fe84838a92e13ab364ddea6`.
Require explicit formula revision 1 and exact packed-model loading. Use the
first canonical training source of each content class in the existing 12-source
imazen-26 manifest, with original bytes/dimensions: four sources, distances 1
and 3. No validation or terminal source participates in this mechanism screen.

Freeze precomputed AC strategies, source XYB and initial quant field per cell.
Use effort 8 Reference profile, normal CfL/gaborish/pixel-domain loss, no noise,
denoise or patches, no iterative perceptual loop. The cached precomputed entry
is an existing native encoding path; it is a controlled actuator experiment,
not a claim of byte parity with the differently configured Zenjxl H3 loop.

For each cell, encode/decode/score the baseline and an exact neutral repeat.
Compute the complete baseline attribution at bin size 8 and require scalar
parity and explicit supported spatial terms. Enumerate actual transform regions;
select up to 16 uniformly spaced regions in raster order, independent of map
values or judge outcomes. For each selected region, multiply every covered raw
quantizer by 0.9 and 1.1, rounding to an integer and moving by at least one when
the limit allows, clamped to [1,255]. Keep every other raw quantizer fixed.
Each intervention is a separate full native encode, independently decoded and
scored. Do not silently discard unchanged/clipped interventions.

Save baseline and actual emitted quantizer fields, transform rectangles,
clipped source area, signed attribution integral/density, changed quantizer
counts, encoded bytes, exact decoded pixels and complete-model score. Report
actual score/byte changes and local quantizer changes, including negative or
inert responses. A zero-override repeat must match the full bitstream, pixels,
quantizer field and scalar score. Compare actual captured quantizers, not merely
requested factors, to establish actuator liveness.

## Interpretation and gates

Use independent CPU SSIMULACRA2 and Butteraugli on every emitted probe and verify
reference/distorted input identities. Decode a bounded low/high intervention
subset with libjxl v0.12 as a compatibility check. Retain all raw outcomes before
deriving rankings. Report sign agreement and rank associations for map mass,
map density, native score gain and extra-byte cost; distinguish derivative
direction from total-restoration magnitude and gain per byte. No correlation
threshold is retroactively a model release gate.

Count cached preparation, full encodes, independent decodes/scalar comparisons,
map evaluations and all judge comparisons. Record times, process RSS, model,
source, binary and source-code hashes. This tests native finite interventions;
it does not judge target attainment and cannot replace the per-image attained
bounds, train-only seed fits or held-out 1/2/3-shot/RD qualification already
required for the product. Preserve nulls and unsupported terms explicitly.


Prototype note before any encode: an additional, unregistered 1e-6 numeric
comparison between the encoder's approximate transfer and the decoder's exact
transfer failed (maximum 3.0994415e-6 over 256 values). It did not test a release
gate, and no encode/result was produced. The frozen linear inputs now explicitly
use the existing decoder transfer reference; the native approximation difference
is recorded in INPUTS rather than assumed away. No encoder formula is changed.
The failed prototype binary/command/log remain preserved.

Post-screen engineering coverage registration (before the additional run): the
repo requires a multi-group roundtrip alongside single-group coverage. Run the
unchanged final instrument on the existing 512-long-edge variants of the SAME
four training origins (410×512, 393×512, 512×512, 512×288), selected by geometry
without inspecting their outcomes. No new resize, family, encoder policy or
model fit. Keep this eight-cell/272-encode packet separate from the registered
256-long-edge mechanism screen. Apply the same hashes, neutral/region/actuator
checks, 24 libjxl 0.12 compatibility decodes and complete independent judges.
Report it as engineering coverage and a descriptive scale check; it cannot
increase the independent family count or qualify a model.
