# Issue 138: strict-policy loop controls, 2026-11-04

The production policy still fails all ten selected PQ quality cells. A full
Libjxl strategy with resampling forced to one and an explicit Butteraugli loop
passes eight; raising the budget from two to eight passes the same eight.
All ten two/eight pairs have identical stream SHA256s. This does not establish
that the remaining cells meet their target, nor qualify a default change.

| Arm | Pass / 10 |
|---|---:|
| Original options | 0 / 10 |
| DC parity plus two Butteraugli iterations | 8 / 10 |
| Libjxl, no resampling, two Butteraugli iterations | 8 / 10 |
| Libjxl, no resampling, eight Butteraugli iterations | 8 / 10 |

Both controlled policies still fail 2965/49033 and 2966/66247. Their normalized
PQ signal mean errors are:

| Cell | Original | DC + loop | Full strict + loop | cjxl |
|---|---:|---:|---:|---:|
| 2965/49033 | 0.053535 | 0.041690 | 0.057987 | 0.017040 |
| 2966/66247 | 0.038127 | 0.046259 | 0.065053 | 0.014096 |

The full strict control worsens both residual cases. More iteration budget
alone is not an effective repair on these frozen inputs. The next diagnosis
checks loop execution and saturation using the existing reconstruction hooks;
no new policy or tolerance is selected from this matrix.

[All 40 records](issue138_strict_loop_2026-11-04.tsv) retain both encoders'
errors and sizes, verdicts and stream hashes. The driver checks every raw input
against the independently retained replay bytes. Original failures remain in
the table. All streams undergo the existing Rust/djxl decoding checks.

## Identity and reproduction

Production source is `29a0e23fdeb1068070c1b4b988db8ac52a8d98de`, including the
strict source-domain CfL correction. Driver snapshot:
`97006bba` (adds only the full strict loop arms and recipe). The source closure
is the unchanged CI-pinned sibling export in
`~/tmp/jxl-hdr-r7900x-20261104/SOURCES.json` on r7900x, Rust 1.99.1,
independent Cargo targets, no native-CPU flag. Binary SHA256:
`c570b9e214f63170eff24be6cd0b4d703ebb5e0dbaa24877d86be88e3bde8381`.
The reference remains the checksum-pinned cjxl/djxl v0.12.0 bundle described
in [the earlier loop study](issue138_loop_ablation_2026-11-04.md).

Run `just conformance-build`, then
`just issue138-strict-loop-ablate <binary> <replay> <new-out> <corpus> <icc>`.
The driver retains the source/streams/reference/decoder diagnostics under
`~/tmp/jxl-hdr-r7900x-20261104/strict-loop-ablation/` on r7900x.
Full log: `~/tmp/jxl-strict-loop-study.log` on that host. Resource receipt:
rc0, 68 s, run-heavy peak RSS 0.79 GiB, minimum available 18,200 MiB,
peak load 2.57. This includes a build and is not a codec timing measurement.
