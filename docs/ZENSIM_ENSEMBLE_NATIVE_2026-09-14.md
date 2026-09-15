# Zensim ensembles and neutral seed preservation — September 14, 2026

The optional existing Zensim loop accepts `ensemble:<manifest>` in its candidate
profile specification. The manifest declares a name, ordered member paths and
SHA-256 hashes, and convex weights. Every member is validated; inference uses
`zensim::BakeScorer::ensemble`. The existing `bake:` singleton route retains its
arithmetic. No new public Rust item is introduced.

The existing `zensim_diffmap_rd` native targeting example accepts that same
specification through `--bake`. It retains explicit `ZENSIM_FORMULA_REV=1` or `3`
(default 1 for historical callers) and binds TRAIN calibration to the full
manifest hash, driver executable and actual revision. Changing a nonprimary
member, weights, configuration or formula cannot reuse the calibration.

A TRAIN-only control revealed that gain0 changed 13 of 21 seed bitstreams.
Recomputing global scale/raw quantizers from an unchanged floating field changed
its discrete representation. Preserve the actual seed parameters and raw field
for unchanged-field comparisons and final emission, including emit-best seed
restoration. Skipping zero-gain clamping did not solve this and was removed.
The synthetic child-process regression compares zero versus two updates at six
distances; it does not mutate process environment across concurrent tests.

The frozen cohort admits seven compositions and refuses four with unavailable
spatial terms. On the admitted nine TRAIN/eight validation families, all 2,646
neutral ladder pairs match bitstream, decoded pixels and score; 2,642 active
pairs change bytes. Complete targeting has 4,914 cases and two independent judges
cover 8,442 output pairs. Independent sample score/pixel parity is 21/21; all 21
also decode with libjxl v0.12.0. Mixed RD diagnostics do not qualify new models.

Full scientific protocol, counts, controls and model-specific findings:
`../zensim/benchmarks/jxl_ensemble_native_2026-09-14.md` in the sibling checkout.
Served report path: `/zensim/reports/jxl-ensemble-native-2026-09-14/index.html`.
The replay binds original model bytes, inputs, calibration, binary and sources;
source images retain original manifest paths. All experiment timing includes
competing work and is not a production latency claim. No model was retrained.

Validation: six scoped loop tests, workspace/native-example Clippy, format,
default tests with explicit verified `DJXL_PATH`, and both `just rd-regression`
tests pass. The first default run hit the pre-existing stale decoder path and
failed to load its shared library; retained in the report. The compatibility
suite's published-reference exposure is recorded separately from model assessment.
