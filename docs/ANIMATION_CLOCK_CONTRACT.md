# Animation timecode order

Part 1 F.2 requires a nonzero timecode to exceed earlier positive-duration
frame clocks. Both animation writers now share input validation:

- Zero is unset and does not erase earlier known clocks.
- A zero-duration normal frame compares its nonzero clock with that history
  but does not advance the positive-duration history.
- ReferenceOnly has no serialized timecode or duration; those supplied fields
  do not participate in clock order.

The existing raw u32 clock domain is preserved. No public type or signature
changes, and valid encoded payloads are unchanged.

## Actual before and reference controls

The original public regression accepts descending/equal clocks through both
legacy/request entries, both writers and every offered policy. The independent
complete envelope/TOC walk confirms actual strict-mode fields:

| Stream SHA256 | Observed clocks |
|---|---|
| `bd9db190ed7083f902145c31753dd77411a1418c9c7d9db3dbd3748adfabab9f` | 2, 1 |
| `582808dd9c6bc5bebbe3cde362c4f4fb39900e3c5b9f7147d515cf5726db7baf` | 2, 2 |

The unmodified v0.12 public encoder and decoder also accept and fully render
descending/equal sequences. `scripts/clock_level_oracle.cc` records their
actual clocks, durations and every rendered endpoint sample. This shared
reference omission does not satisfy the normative order requirement.

## Repair and selected checks

Positive controls preserve actual clock/duration fields, fully render every
displayed frame in primary Rust and pinned djxl, require exact lossless
endpoint samples and compare legacy/request bytes. They cover unset clocks,
hidden normal frames, ReferenceOnly and u32 history boundaries. Inputs are
64×32 RGB8 at lossless effort one / lossy effort three, across all offered
policies. These controls do not qualify every animation layout or SMPTE
application convention.

The isolated clock-only source, without the subsequent crop repair, passes
its unit history and public matrices (`animation-clock-isolated.log`: 72 s;
run-heavy peak RSS 1.87 GiB, minimum available 19,564 MiB). Default and minimal
combined-candidate controls also pass. Separate faults weaken equality,
unset-history retention, hidden-frame handling, ReferenceOnly exclusion and
each writer's admission call; every fault builds and triggers its regression.

The checksum-bound report at `animation-state-verified/regions/regions.json`
observes 13/13 regions in the new clock-order window and 6/6 at each writer
admission call. Those are selected code windows, not complete validator or
input-space coverage. The combined candidate preserves the library baseline
and existing normal/strict/drift byte expectations. Its initial encompassing
command remains failed by the subsequently corrected unused test wrapper;
the strict lint continuation passes. No old expectation or ignore changed.

Reproduction uses the CI-pinned source closure and explicit v0.12 tools:

```
just animation-clock-after-evidence <manifest> <new-artifacts> <coverage-env>
just clock-level-native <v012-source> <v012-build> <new-executable> <new-output>
```

Automatic crop-reference validity, unsupported spatial resampling and
cumulative zero-duration area have separate contracts and queue entries.
