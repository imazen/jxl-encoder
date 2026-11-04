# Issue 138: residual PQ loop execution, 2026-11-04

The two residual strict-loop cells execute normally and reach a valid fixed
point. They are not stalled above target at a quantizer clamp. Every measured
tile is already below the requested Butteraugli target at iteration zero and
remains below it through the final recorded iteration two.

| Cell | Target | Tiles | Iteration | Above target | Maximum tile distance |
|---|---:|---:|---:|---:|---:|
| 2965/49033 | 4.5 | 1056 | 0 | 0 | 3.282035 |
| 2965/49033 | 4.5 | 1056 | 1 | 0 | 2.058924 |
| 2965/49033 | 4.5 | 1056 | 2 | 0 | 1.932659 |
| 2966/66247 | 25 | 15594 | 0 | 0 | 9.248323 |
| 2966/66247 | 25 | 15594 | 1 | 0 | 12.175688 |
| 2966/66247 | 25 | 15594 | 2 | 0 | 15.036114 |

At iteration two the existing update uses `cur_pow = 0`: below-target tiles
cannot move, and there are no above-target tiles to bump. The repeatable-backend
fixed-point shortcut therefore stops correctly. This is not evidence to change
the shortcut, clamp or iteration budget. Both streams remain byte-identical to
the hook-disabled eight-iteration controls, and still fail the unchanged PQ
signal-error comparison. Butteraugli target satisfaction and that signal-error
gate are different observed results; neither is relabeled as the other.

## Reproduction and identity

`just conformance-recon-build` enables the existing internal reconstruction
hook. `just issue138-quant-trace <binary> <new-out> <seed> <index> <corpus>
<icc> <opts>` retains every iteration's quant field, DC/AC coefficients,
reconstruction, diffmap and tile distances. It refuses an existing run path.
Use the full strict/no-resampling/eight-iteration options from
[the 40-cell study](issue138_strict_loop_2026-11-04.md).

Production source is `29a0e23f`, with diagnostic snapshot `a31f98b8` adding only
recipes. Pinned siblings/reference tools are unchanged. The original binary is
retained as `strict-loop-binary-c570b9e2` under
`~/tmp/jxl-hdr-r7900x-20261104/` on r7900x. Hook artifacts are
`pq-quant-trace-{49033,66247}/quant-trace/`; complete stdout/stderr is
`~/tmp/jxl-pq-quant-trace.log` on that host. The driver explicitly expects these
two quality failures; each sweep exits 1 and no failure is converted to a pass.

Resource receipt including the diagnostic build: rc0, 20 s, run-heavy peak RSS
0.83 GiB, minimum available 19,179 MiB, peak load 0.89. This is not an encoder
performance comparison. No production policy or tolerance changed.
