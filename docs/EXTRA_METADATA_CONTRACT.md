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
