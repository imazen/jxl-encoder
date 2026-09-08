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


# Native PNG and coarse JXL intervention registration

Registered 2026-09-08T20:09:41.967730+00:00, before implementation and new encodes. Extends the existing
`zensim_diffmap_rd --native-interventions` owner and its existing analyzer.
Previous per-transform +/-10% results remain immutable and mixed.

Freeze the same D artifact, formula 1, source manifests, four training origins
2010/6068/7066/8206, distances 1 and 3, effort 8 Reference, normal transform
strategies/CfL/gaborish, exact decoder transfer and cached precomputed fields.
No validation or terminal examples, no fit or policy selection.

First replace this private instrument's PNG IO with zenpng 0.1.4 and record
native decoded RGB hashes. Reproduce the old 272 transform probes using the
new IO; require all emitted JXL bytes, raw RGB, quantizers, scores and map
integrals to match the retained screen-final packet. PNG compression may
change. Historical PIL checks remain only in the v1 analyzer branch.
Source admission requires static, opaque, eight-bit sRGB-compatible PNG;
unsupported metadata/depth/alpha fail explicitly, not silent conversion.

Coarse mode: assign each complete transform to one of 4x4 grid cells by its
anchor block: gx=floor(4*x/xsize_blocks), gy=floor(4*y/ysize_blocks).
Keep all members whole; groups are unions of transforms, not assumed
rectangles. Enumerate nonempty cells in raster order. Every source pixel and
padded block belongs to exactly one group. Sum clipped areas and signed map
integrals, with density=mass/area. Large transforms may cross grid boundaries.
For each group change all covered raw quantizers by factors 0.8 and 1.2,
rounding with minimum step one and clamp [1,255]. Baseline and exact neutral
repeat plus two probes per nonempty group; at most 34 full encodes per cell.
No response-based group sampling, threshold tuning or omitted inert probes.

Measure actual captured quantizer changes, raw-pixel changes inside/outside
the union, complete D score, bytes, CPU SSIMULACRA2 and Butteraugli. Retain
nonpositive central byte differences; compute signed central derivatives
normalized by actual mean-log-q span and rank associations with mass/density
and quality gained per byte where the byte derivative is positive. Report
each image/distance, not only pooled correlation. Count all encodes, native
JXL decodes, comparisons, maps, source/PNG roundtrip decodes, compatibility
decodes, times and RSS. libjxl v0.12 remains port compatibility only.

Run the same coarse configuration on the existing 512-long-edge variants
of these four origins as separate multigroup/scale coverage, not new families.
Require whole-transform coverage, neutral byte/pixel/q/score equality, native
PNG readback parity, complete independent judges and content hashes. Add
negative controls for group membership, quantizer requests, PNG hashes and
coverage. Reproduce final-source output if implementation changes after run.

This bounded screen ends after these fixed experiments and integrity checks.
It cannot establish held-out RD gains, 1/2/3-shot target accuracy, or release
qualification. Do not turn descriptive correlation into a retrospective gate.
No runtime oracle probing or new allocation policy is authorized by these
measurements alone; preregister the next intervention separately.


# Coarse JXL allocation policy — registered 2026-09-08T20:35:32.658483+00:00

Question: can one complete-model coarse map improve an actual emitted JXL
against all locally reachable global quantizer fields, not just correlate with
single-region quality responses? No fitted model or seed, no new public API.

Reuse the current native precomputed e8 Reference/CfL/gaborish/pixel-loss path,
D bake cd1098b450ef6941b6925b24bcbd129715b6f07c4fe84838a92e13ab364ddea6,
formula 1, exact decoder transfer, native PNG IO and complete Rust surface.
First canonical train origin per class: 2010/6068/7066/8206, distances 1 and 3,
existing 256-long-edge variants. No validation or terminal sources in this
first bounded policy screen. No new feature/model training or source admission
exception. Frozen AC strategies are shared by all fields in each cell.

Build all scalar control fields BEFORE applying the policy. Domain: multiply
all initial integer raw q by one common factor in [2/3,3/2], half-up round,
clamp [1,255]. Enumerate exact rational breakpoints (2n+1)/(2q), n=1..254,
plus both endpoints; sort by integer cross-products and deduplicate resulting
full raw fields. Every distinct state of this declared scalar operation is
represented. Do not resample only convenient factors. Refuse more than 4096
states in a cell rather than silently truncate. Encode/decode/score each
unique field, retain counts/order and attained D/byte bounds. Baseline and an
independent exact neutral repeat remain explicit. This is an exhaustive local
raw-field comparator, not a claim to exhaust the codec's full distance range.

Policy inputs are ONLY baseline complete attribution, existing whole-transform
4x4 groups, and the original raw field. No control outcome, judge, bound or
intervention derivative is available to the policy. For group density d and
pixel area A, compute area-weighted center mu and mean absolute deviation m.
If m <= 1e-20, factors are one. Otherwise set f=1+0.2*clamp((d-mu)/m,-1,1).
Expand each factor to every block in its complete transforms. Preserve the
initial sum of requested q before rounding by multiplying all factors by
sum(raw)/sum(raw*f); then half-up round and clamp [1,255]. This preserves a
quantizer-sum proxy, not actual bytes. Record unnormalized/normalized factors,
center, dispersion, raw-sum normalization and actual requested/captured fields.
Apply the same function to zero densities and require raw-field/byte/pixel/
score identity with the neutral repeat. Active is one actual full encode after
one baseline full encode and one map: two encodes and one map, with all extra
control encodes separately labeled engineering cost. No runtime oracle probes.

Every emitted control and active output receives native JXL decoding, complete
D scoring and independent CPU SSIMULACRA2/Butteraugli. Bounded libjxl v0.12
compatibility checks remain port-only. Report exact measured scalar frontiers:
for each active byte budget, best scalar D/SSIM2/-BA within budget; for each
active quality, minimum measured scalar bytes meeting/exceeding that quality.
Keep no-match/coverage failures explicit. No interpolation or extrapolation.
Also report a single scalar output chosen by best D within the active budget,
including its independent judges, and any scalar output that dominates active
on bytes and all three qualities. Only compare in the attained scalar D/byte
ranges; do not count uncovered cells as wins.

Advance this fixed policy to a separately registered broader evaluation only
if every covered training cell is noninferior to best scalar at budget in D
(>= -0.05), SSIM2 (>= -0.1), and -BA (>= -0.005), every content class has positive
median D gain, and at least half the cells improve D by >= 0.05. These are
screening bars, not new release gates. Predefine floating comparison slack
1e-5 for D, 1e-6 for independent metrics, zero bytes. Report all counts regardless
of verdict. If it fails, preserve the failure and revise the hypothesis before
spending validation families. No opportunistic alternate arm/gain sweep.

Require complete native hash/region/scalar-state/policy coverage, exact neutral
identity, independently recomputed policy fields and rational state enumeration,
negative controls, final-source reproduction, scoped Rustfmt and local CI-exact
Clippy. Preserve any prototypes. Count all full encodes, JXL/PNG decodes, scalar
comparisons, maps, independent judges, preparation/IO time and RSS separately.
This screen does not establish general target attainment or shippability. The
full goal still requires train-calibrated 1/2/3-shot targeting and matched-RD
validation on separate families, model qualification and all four codecs.


## Engineering multigroup coverage, registered 2026-09-08T20:47:36.951101+00:00

The fixed 256 policy failed the independent-judge screen; that verdict is
unchanged and no separate validation family will be spent on it. JXL's repo
requires a multigroup roundtrip for changed encoding paths. Run the same final
policy/control instrument on the existing 512 variants of the SAME four
training origins from the previous multigroup manifest. This is software
coverage and a descriptive scale check, not promotion or a larger independent
validation set. Preserve all outcomes, direct scalar-state comparisons, native
PNG/JXL hashes, exact neutral controls and independent judges. Keep results
separate; they cannot overturn the registered 256 screening failure.
