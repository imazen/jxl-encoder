# Animation clocks and automatic crop references

This records the issue-144 clock-order repair, the automatic-crop defect found
while validating it, and the refusal of unimplemented spatial resampling.
Cumulative zero-duration area remains separate work.

## Clock history

[ANIMATION_CLOCK_CONTRACT.md](ANIMATION_CLOCK_CONTRACT.md) owns clock order,
its independent native controls and the isolated clock-only verification.

## Source bytes are not always the saved canvas

The first clock-control fixture explicitly saved normal frames in slot two.
After a zero-duration white frame, the next identical 64×32 white frame became
a one-pixel crop reading default slot one. Primary Rust returned one white
pixel and black elsewhere; pinned djxl returned exactly three nonzero RGB
samples. The requested full canvas was white. The stream, executable, source,
native outputs and hashes are retained in `animation-clock-crop-attribution`
and `animation-clock-crop-native`.

Separate failing controls reproduce wrong pixels in both writers when:

- The previous normal frame saved in a nondefault slot or did not save.
- ReferenceOnly overwrote slot one after the cached source was recorded.
- A preceding Add frame produced a canvas different from its addend buffer.
- The next implicit replacement explicitly selected a different source slot.
- A hidden normal frame saved in a nondefault slot.

`advance_crop_reference` now retains source bytes only for an ordinary or
explicit Replace frame saved in slot one. A ReferenceOnly overwrite of slot
one clears that state; a write to another slot preserves it. Composed,
nondefault-slot and unsaved canvases clear it. The automatic-crop guards also
require an unset or explicit slot-one blend source. Other cases encode full
replacement frames. This keeps the existing full-canvas explicit-blend contract.

The selected regression matrix covers every offered policy, both legacy/request
entries, both writers, RGB8/RGBA16, 8×8, 64×32 and 259×17. Valid default-source,
matching explicit source, explicit Replace and unrelated-reference controls
still take partial crops. Every successful selected stream fully renders in
primary Rust and pinned djxl, with exact lossless endpoint/alpha samples and
the fixed lossy endpoint bound. It does not test every source format, partial
alpha value, profile or composition operation.

The secondary decoder rejects a valid 8×8 RGBA16 Add frame because it reads
the extra channel's source conditional using the main blend mode. Part 1 F.7
requires the mode of the same BlendingInfo instance. Primary Rust and native
v0.12 fully render it; an independent complete envelope/TOC walk accounts for
every byte. The regression uses that independent parser for structural checks.
[Decoder ledger](DECODER_FINDINGS.md). No dependency was changed for this gap.

The independent parser itself exposed two alignment omissions: timecodes and
the reference-eligibility condition on unsaved positive-duration frames. Both
actual streams fail its frozen regressions before `e933ad0c` and pass afterward.
That diagnostic correction changes no encoder payload.

## Animation resampling admission

Both writers previously omitted the spatial-resampling configuration. Actual
before streams from 33×17 buffers retain 33×17 for factors 2/4/8, including
already-downsampled requests. The shared animation entry now rejects effective
factors above one before frame work. Factor one remains accepted, including
already-downsampled no-op settings. Explicit, automatic and registered policy
selection use the same guard. This is an unsupported-combination refusal, not
a statement that the JPEG XL format cannot represent resampled animations.

## Evidence, faults and regions

`animation-crop-reference-before-corrected.log` retains each independent
both-writer pixel failure and the passing valid-reference controls (60 seconds;
run-heavy peak RSS 1.05 GiB, minimum available 20,350 MiB). The earlier own
fixture mistook a full 8×8 crop flag for a partial frame; its failed run is
retained, and the corrected test compares the actual frame extent.

`animation-clock-crop-after.log` passes clock/resampling controls but fails on
the separate secondary-decoder rejection. `animation-crop-reference-independent`
passes all pixel checks but retains a failing envelope verdict for the unsaved
frame parser omission. Neither encompassing command is labeled green.

`animation-state-final-selected.log` passes corrected default controls,
all minimal-build matrices and fourteen separately restored fault replays
(542 seconds; run-heavy peak RSS 1.72 GiB, minimum available 19,632 MiB).
Faults cover the animation resampling guard; equality, unset, hidden and
ReferenceOnly clock history; both writer admission calls; both crop-history
calls; slot validity, overwrite and composition; and both source guards.
Every fault must build, execute and fail its intended assertion. Failed
mutation profiles do not enter the passing coverage union.

`animation-state-verified.log` passes the selected default pixel matrix,
library baseline and unchanged byte-lock/drift checks, then fails a new-target
unused-helper lint (608 seconds; peak RSS 1.70 GiB, minimum available 19,710 MiB).
The helper was split without changing its frame-decoding body. The corrected
`animation-state-last.log` passes focused/workspace strict lint, minimal build,
formatting, registered resampling permissions and public clock controls
(106 seconds; peak RSS 1.06 GiB, minimum available 19,825 MiB).

`animation-crop-existing-corrected.log` retains the encompassing H1/H11
run's failed verdict: its earlier lane-authored patch fixture demanded a
partial crop with an explicitly different source (592 seconds; peak RSS
1.05 GiB, minimum available 19,842 MiB). All other selected tests pass;
primary/native rendering of that failed structural cell also passes.
The corrected fixture requires full replacement for unknown source canvases,
retains storage/dictionary/pixel checks, and adds a known-slot-one partial crop
with an actual winning dictionary. `animation-crop-patch-corrected.log` passes
that policy matrix, focused strict lint and formatting (155 seconds; peak RSS
1.04 GiB, minimum available 20,176 MiB). Existing assertions unrelated to the
new crop policy remain unchanged.

The exact-function report in `animation-state-verified/regions/regions.json`
observes: crop-history helper 14/14 regions; clock-order window 13/13; each
writer crop guard 7/7, history update 2/2 and admission call 6/6; animation
resampling guard 6/6. These are selected code windows, not complete function,
input-space, format or platform coverage. Other layout/feature/resource and
auxiliary-frame findings remain governed by their own contracts and queue.

Reproduce against the pinned source closure with explicit v0.12 tools:

```
just animation-clock-after-evidence <manifest> <new-artifacts> <coverage-env>
just animation-crop-reference-evidence <manifest> <new-artifacts> <coverage-env>
just animation-state-selected-and-faults <manifest> <new-artifacts> <coverage-env>
just animation-state-final-validation <manifest> <new-artifacts> <coverage-env> <llvm-bin> <lint-home> <lint-target> <prior-passing-archives>...
```
