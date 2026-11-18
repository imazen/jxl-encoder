# Extra-channel metadata contract

Issue [144](https://github.com/imazen/jxl-encoder/issues/144), H9/H10.
The metadata bounds and Black/profile requirements are on main. This record
covers the independent metadata bounds; [CMYK_SOURCE_CONTRACT.md](CMYK_SOURCE_CONTRACT.md)
covers source conversion, Black precision and animation forwarding, including
the separate primary-decoder float ICC blend discrepancy.

`ExtraChannelInfo::validate` checks dimension shift 0–3, name length at most
1071 bytes, CFA index 0–274, finite representable binary16 spot components,
and spot solidity within [0,1]. Only fields serialized for the selected type
participate. Negative or above-one RGB spot components remain supported.
Part 1 D.3.6 and L.4 distinguish wider integer syntax from semantic limits.

The check runs before public still-request allocation, at each direct,
prepared and prequantized VarDCT extras entry, and before channel/header bits
are written. Lossless channel construction retains its defensive check.
Streaming has no extra-channel attachment method; animation rejects caller
extras. Synthesized alpha/Black descriptors pass the header check too.
The existing shifted-VarDCT refusal is unchanged. No strategy gate is added.

## Actual before and reference controls

`extra-metadata-public-before.log` records accepted lossless shift 4 and accepted
solidity 2/negative in both modes. CFA 275 and u32::MAX panic in the checked
build. Existing nonfinite/overflow spot refusals happen after partial internal
header writes. The separate atomic-writer regression fails on those writes,
oversized names and shifts too. Public destination preservation is distinct
from internal writer atomicity.

Public before executable:
`7e4c04b33315841bb250f1dd2dfccd2f8ed43a767c0587598a51878223465fc4`.
Its wrapper exits 101 (19 seconds; run-heavy peak RSS 1.05 GiB).
The writer before wrapper exits 101 (34 seconds; peak RSS 1.87 GiB).

`scripts/extra_metadata_oracle.cc` calls unmodified v0.12.0 a7a9c78.
The public setter accepts the tested shift/CFA limits, while `Bundle::Write`
rejects shift 4 and CFA 275. Both accept solidity 2: the [0,1] requirement is
normative and cannot be established by copying that reference behavior.
Legal bundle bytes and bit lengths are frozen in the unit test; native
executable/log: `extra-metadata-native` / `.log`.

The first repaired public/writer controls pass in `extra-metadata-after.log`
(61 seconds; peak RSS 1.86 GiB). Legal 32×17 and 259×17 controls fully render
in primary Rust and pinned djxl across offered policies. Both recover every
zero auxiliary sample exactly. This establishes metadata handling/rendering,
not physical spot-color accuracy or general lossy color quality. Existing
shifted-edge tests supply separate nonconstant auxiliary-grid checks.

Prepared-entry admission has a separate failing before control: invalid
solidity with a missing working input reaches the input-length error rather
than metadata admission. The regression selects direct, prepared and
prequantized entries independently.

## Reproduction and final validation

Use the CI-pinned export and verified v0.12 tools:

```
just extra-metadata-native <v012-source> <v012-build> <new-executable>
just extra-metadata-final-evidence <manifest> <new-artifacts> <coverage-env>
```

The expanded default/minimal matrices pass in `extra-metadata-complete.log`
(115 seconds; run-heavy peak RSS 1.88 GiB, minimum available 19,550 MiB).
Packed and padded rows produce identical bytes. Positive direct, prepared and
prequantized outputs fully render in both decoders across the offered policies;
invalid metadata precedes bad extra-buffer length and working input.

`extra-metadata-finish-validation.log` passes the unchanged library baseline,
normal/strict byte locks and divergence checks, strict Rust 1.100 focused and
workspace all-target Clippy, no-default build and scoped formatting
(479 seconds; peak RSS 1.88 GiB, minimum available 19,597 MiB). Existing ignores,
byte expectations and scientific bounds are unchanged. The no-default check
retains its existing warnings; strict lint verdicts are separate.

The first fault batch remains failed overall: nine faults were detected, but
the prequantized entry mutant passed because the old fixture allowed a later
header check to catch the same metadata. A conflicting extra-buffer length
makes admission order distinguishable. The tenth-only continuation detects
that fault (20 seconds; peak RSS 1.03 GiB); all mutated sources are restored.
This is regression sensitivity, not a completeness proof.

Missing coverage is explicit: allocator/fallible-write failures are not injected.
The lossless construction error is excluded by earlier request admission.
No physical CMYK/spot-color qualification or whole-input/platform coverage
follows from the named code regions.

`extra-metadata-complete-regions.log` passes (38 seconds; peak RSS 1.87 GiB,
minimum available 19,620 MiB). Its seven passing, exact-body archives observe:

| Named code scope | Observed regions | Remaining regions |
|---|---:|---|
| Metadata validator | 27/27 | None in this compiled body |
| Channel writer | 51/60 | Nine fallible-write propagation returns |
| Name-length writer | 25/32 | Seven fallible-write propagation returns |
| CFA writer | 25/32 | Seven fallible-write propagation returns |
| File-header validation loop | 5/5 | None in this selected block |
| Public admission loop | 6/6 | None in this selected block |
| Lossless construction check | 3/4 | Defensive validation error |
| Direct/prepared/prequantized validation call | 3/3 each | None in each selected call |

Bodies, declaration spans, object/source/profile hashes and exact missing
regions are retained in `extra-metadata-regions-final/regions.json`. Test totals
are not used to establish these scopes or input completeness.

## Extra-channel count follow-up

A later public strict probe at 1x1 with 4097 supplied Depth channels panics in
both lossless/lossy byte-return and writer routes. The count reached a debug
assertion in `FileHeader::write_image_metadata`. Retained q78 runtime exits 101,
ELF `971d5d58b0504d9a573fc0c75c3b9e8b42248b59daa686683ff1f57d41f0f747`,
profile `e210e30294d393159272557a6f938cbe933a32cc9fa1a95145c17f8c73872925`.
Exact destination preservation is separate from avoiding that panic.

The pinned unmodified v0.12 `--counts` observer distinguishes two limits:

| Count | Metadata `Bundle::Write` | Level 10 `SetBasicInfo` |
|---|---|---|
|256|accept|accept|
|257|accept|refuse|
|4096|accept|refuse|
|4097|refuse before bits|refuse|

Zero, one, seventeen/eighteen and 255 are additional native controls. This
observes metadata serialization and public admission, not a complete native
encoded image. It agrees with the [v0.12 metadata visitor](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/image_metadata.cc)
and Part 1 D.3/Annex M's separate grammar/profile constraints.

The repair shares the existing 256 Level 10 cap between container
classification, request admission and all three public raw VarDCT extras
boundaries. Requests count supplied channels before preflight. Known implicit
alpha/Black participate then too, except potentially removable alpha under
canonicalization. The prepared layout is checked again after canonicalization.
This preserves an opaque-alpha reduction that leaves 256 caller extras.
The file-header writer independently refuses more than 4096 before writing
any bits, matching the native metadata grammar. Neither cap changes.
No public API, registry gate, preset or channel quantizer is added.

The boundary tests use distinct per-channel codes, fully decoded by primary
Rust and pinned djxl. Selected request entries include RGB/RGBA/CMYK, both
pixel coding modes, all seventeen audit policy bindings, immutable writer/into
refusals and declared-alpha/Black accounting. Cap-zero request control and
canonicalized opaque alpha have separate controls. Raw direct/precomputed/
prequantized positive codestreams receive their caller-owned Level 10 wrapper;
invalid counts must precede missing sample/working inputs. Header unit controls
separately retain the 4096 grammar boundary and an untouched seeded writer on
4097 rejection. These are selected interactions, not every layout/shape/type
combination or a whole-input-space proof.

The first new positive fixture assumed unpadded djxl EC filenames and an
implicit Level 10 declaration for raw codestreams. Its retained q80-v1 fails
those assumptions; it does not establish wrong boundary pixels. q80-v2 fails
compilation because the new fixture tried to call a private wrapper. The
corrected fixture uses the existing public wrapper and shared signature
constant, preserving every sample assertion.

The final q80-v3 controls pass in the standard and minimal builds. They check
the existing metadata contracts alongside the new count, caller and pixel
boundaries. The selected header unit test, normal and strict byte locks,
divergence checks, strict Clippy and formatting pass. The retained
instrumented object and profile are
`033e040d49bcdb0d103b40e35e6f15c735a9b44a997976a5a6f9a829c77df539` and
`fdd162ead1523f57ac45d2cc5c0498cfbeb74b89c1828031dbac66880fd7ae6e`.
The wrapper exits zero after 496 seconds with build and test peak RSS of
1.84 GiB and minimum available memory of 17103 MiB.
