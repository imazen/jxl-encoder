# Candidate rectangle refinement — September 15, 2026

The native candidate adapter formerly checked additive-density coverage and
queried `AttributionResult::query_rect`. This rejected basic228 candidate max
features even though the Zensim scored owner now retains their exact frozen
signal maxima and exposes `ScoredAttribution::refinement_gain`.

The existing tile loops now use a private statically dispatched query adapter.
Named-profile density maps retain their original semantics. Complete candidate
maps query refinement once per actual transform footprint and check
`unsupported_refinement_feature_ids`. No maximum is spread into additive
density or summed from subtiles. Unsupported features and corruption gates
still refuse. Controller settings, scalar scores, model weights, native seed
construction and calibration contracts are unchanged.

Validation replays both frozen September 15 Zensim recovery ensembles on the
existing nine TRAIN/eight validation-family matrix: smoke, TRAIN seed fit,
validation; scalar/neutral/active arms; 1/2/3 shots. All 756 neutral ladder
pairs match encoded bytes, decoded pixels and scores. All 1,782 old/new
max-free fast-model records match exactly. The basic228 model now completes
the complete matrix instead of refusing. SSIM2 and Butteraugli judge every
one of the 2,304 decoded pairs across both models. Native loop timing includes
reconstructions and map work separately from full encodes.

This establishes integration coverage, not a shipping quality claim. Both
models show content-class RD regressions; active three-shot p95 score errors
are 1.969 and 1.902, above the registered 1-point bar. No EVAL-driven retuning
or profile promotion followed. Scalar/native plots, all failed and successful
attempts, input/model/binary hashes and commands are preserved under
`/var/tmp/zensim-validation-2026-09-15/native-jxl-refinement/` and the sibling
Zensim `benchmarks/recovery_completion_2026-09-15.md`.

The adapter correction was registered in the artifact packet before replay.
The example release build and scoped clippy passed. Zensim's retained-map,
max ownership, odd geometry and native HDR suites cover the shared numerical
owner; this adapter adds no second feature or scoring implementation.
