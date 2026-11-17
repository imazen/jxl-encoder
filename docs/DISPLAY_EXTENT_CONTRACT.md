# Display dimensions and codestream levels

Ordinary spatial resampling preserves the supplied display dimensions.
`already_downsampled` instead promises `input dimensions × effective factor`.
The request now computes that extent with checked multiplication before input
work and uses it for both the SizeHeader and container level selection.
Resource admission continues to budget the supplied pixel rows.

The file-header writer independently refuses zero dimensions, axes above 2^30
and area above 2^40 before changing the destination writer. Part 1 D.2/F.2
defines the geometry; Annex M supplies the level limits. This is shared by
all policies, with no strategy registry changes.

The optional RGB8 subsampled coefficient path does not implement spatial
resampling. `!Full444 && effective_resampling() > 1` now returns InvalidConfig
before encoding, including automatic resampling and already-downsampled input.
This refusal does not claim that the format cannot represent the combination.

## Actual failing controls

At 32769×1 and 1×32769, factor eight, the old public paths write actual
262152×8 / 8×262152 headers but declare Level 5. Their modular-buffer promise
is explicitly 16-bit sufficient, so it does not mask the dimension defect.
Adjacent 32768 controls fit Level 5. All offered policies, legacy/request,
writer, padded rows and seven-row streaming paths participate. Both primary
Rust and pinned djxl fully render every retained stream and recover black.

The unmodified libjxl v0.12 public encoder emits Level 5/10 for the adjacent
width controls. `scripts/clock_level_oracle.cc` records those native streams;
this reference control is separate from the Rust stream render checks.

An empty streaming finish previously reaches the incomplete-input error for
unrepresentable display sizes. The new controls require a display-geometry
error first, including both checked-multiply overflows and area overflow,
without allocating the image. Direct file-header controls previously panic
on zero width after increasing an existing writer prefix from five to 36 bits.

The optional-path before probe actually emits 33×17 for explicit factors
2/4/8 in all three subsampled modes, with either already-downsampled setting.
It records the stream dimensions rather than inferring them from a request.

## Paths and evidence

| Path | Selected checks |
|---|---|
| Full444 request, legacy, destination writer, padded rows, streaming | Both axis boundaries, actual levels/dimensions, complete Rust/native rendering, byte agreement |
| Full444 ordinary/already-downsampled | Factors 1/2/4/8, RGB8/Gray16/RGBA8, all offered policies, full Rust/native samples, streaming byte agreement |
| Invalid public extent | Both axes, u32 multiplication overflow, absolute area; refusal before missing-row error |
| File-header writer | Legal absolute boundaries without pixel allocation; invalid geometry preserves existing writer bits |
| Optional subsampled RGB8 | Three modes, explicit and automatic spatial factors, writer preservation and streaming admission refusals |
| Lossless and JPEG | No already-downsampled input API; their shared header writer receives absolute bounds checks |
| Animation | Config fields expose resampling, but both animation writers currently omit it. A separate failing/refusal control remains to be executed; no closure is claimed here. |
| Raw/prepared VarDCT helpers | They expose codestreams rather than container levels; callers retain level selection responsibility. The shared header writer protects serialized absolute dimensions. |
| CLI | No already-downsampled flag is offered |

Before logs: `resampled-level-expanded-before-corrected.log`,
`resampled-display-overflow-before.log`, `resampled-header-geometry-before.log`
and `resampled-optional-before.log`. The first candidate checks pass in
`resampled-level-candidate-corrected.log` (103 seconds; run-heavy peak RSS
1.86 GiB, minimum available 19,516 MiB).

The retained before/after comparison observes 68 identical codestream
payloads. The 34 boundary files remain entirely identical; the 34 oversized
files change their container wrapper. Those observations describe this
boundary grid, not every layout or encoder input.

Six separate faults disable level dimensions, header dimensions, early
admission, each checked axis multiplication and file-header preflight.
Every regression detects its fault; sources are restored afterward.
`resampled-continued-corrected.log` passes (168 seconds; peak RSS 1.72 GiB,
minimum available 19,637 MiB). The optional admission fault also detects,
with its own passing restoration run.

Reproduce with the CI-pinned source export and explicit v0.12 tools:

```
just resampled-level-candidate-evidence <manifest> <new-artifacts> <coverage-env>
just resampled-display-controls-and-faults <manifest> <new-artifacts> <coverage-env>
just resampled-optional-after-and-fault <manifest> <new-artifacts> <coverage-env>
just resampled-display-final-validation <manifest> <new-artifacts> <coverage-env> <lint-home> <lint-target>
just resampled-display-regions <llvm-bin> <new-report> <passing-archives>...
```

Canonicalized RGB8 input agrees byte-for-byte with explicit Gray8 input and retains Level 10 in default/minimal builds, with complete rendering in both decoders. The selected geometry grid does not render a 2^28-pixel image, test every
precision/layout combination, or execute the 32-bit working-buffer refusal
branch locally. Absolute header boundaries are header-only checks. Cumulative
zero-duration frame area and animation clock order are separate open findings.

Final validation preserves the full library baseline and all normal/strict byte
locks and drift assertions. `resampled-final.log` retains the failed strict-lint
verdict for two constant-chunk test loops after the functional checks passed.
The mechanical array-iteration correction first exposed a reference-comparison
compile error; that failed command is also retained. The corrected continuation
passes positive optional, canonicalized default/minimal and lossless controls,
strict focused/default/optional/workspace lint, no-default build and formatting
(`resampled-last-corrected.log`: 105 seconds; run-heavy peak RSS 1.08 GiB,
minimum available 19,784 MiB).

The checksum-bound ten-archive exact-function report is
`resampled-regions-final/regions.json` (6 seconds; peak RSS 0.54 GiB).
Observed regions: display extent helper 32/32, new admission window 11/11,
absolute header validator 10/10, and header preflight 4/4. The wrapper-selection
window (9/10) and the internal lossy assignment (2/3) each have one unexecuted error return:
the same immutable request already passed geometry admission before either
site. Their exact counts are retained in the report. These selected regions
do not establish whole-function, input-space or platform coverage.
