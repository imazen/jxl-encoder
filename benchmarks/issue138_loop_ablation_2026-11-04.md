# Issue 138: explicit-loop and DC controls, 2026-11-04

Production still fails the ten PQ quality records. Enabling an existing metric
or DC option in this selected diagnostic does not qualify a default change.

| Diagnostic arm | Passing / 10 |
|---|---:|
| Original options | 0 / 10 |
| Butteraugli, two iterations | 3 / 10 |
| Butteraugli, eight iterations | 3 / 10 |
| DC parity + Butteraugli, two iterations | 8 / 10 |
| DC parity + Butteraugli, eight iterations | 8 / 10 |

The two combined-arm failures remain 2965/49033 and 2966/66247. The latter
**worsens**: mean PQ signal error 0.038127 originally, 0.038034 with the loop,
0.046259 with DC parity plus the loop, against cjxl 0.014096. On 49033 the
loop reduces 0.053535 to 0.041477, but the unchanged bound still fails
(cjxl 0.017040). More iterations change neither verdict. Nine of ten streams
are byte-identical between the two/eight-iteration budgets in each family;
14694 changes bytes in both families, so this does not prove universal
fixed-point convergence.

[All 50 rows](issue138_loop_ablation_2026-11-04.tsv) retain verdicts, both
encoders' errors/sizes and every output SHA256. Original failures remain in
the table even when an experimental arm passes. No threshold, allowlist,
expected byte or production strategy was changed.

## Inputs and identity

- Encoder main: `48ff52658e62e2bf366da68e43892b414b98aa8b`, including the
  separate PQ intensity repair `403263d5` and metric-domain repair `f3559269`.
  The intensity repair performs no rescale at these cells' default 10,000 nits.
- Exact sibling archives from `.github/sibling-revisions.tsv`, recorded in
  `~/tmp/jxl-hdr-r7900x-20261104/SOURCES.json` on r7900x. Rust 1.99.1;
  independent Cargo target directory, no native-CPU build flag.
- All 51 corpus PNG hashes match the original replay's 41 CID22/10 gb82-sc
  set. Each new raw input also matches its retained original byte for byte.
  These ten records embed no ICC; the control supplies one valid RGB profile
  to preserve the generator's nonempty-profile branch.
- Reference: checksum-pinned static cjxl/djxl 0.12.0 from the existing CI
  setup action. The archive SHA256 is
  `5318a1ea40adad76d023e0c17a03d4627f8282f83cb2b575e69be8e74f1ff456`.
- Diagnostic binary SHA256:
  `06fcd8f406cce8c8496139a31693bcb0df7fe0a54c0ce541a7140fb929c362e4`.
  Standalone Cargo.lock SHA256:
  `e8d87432a230cc361b09c14dbf8c494ee84ce16707c3363ed4ff3cc94f8eb988`.
  Diagnostic Rust source SHA256:
  `c619ac3a1d7e086e81ccb3b48348e3b37a40a50fe39a1c396a19ea002ed659b9`.

The driver normalizes streaming to one-shot and omits attached EXIF/XMP;
source pixels and quality-relevant options remain unchanged. Both reference
encodes retain the existing matched-option/PAM path. This is a selected
signal-error investigation, not matched-quality RD or performance qualification.

## Reproduction and retained evidence

`just issue138-loop-ablate <binary> <retained-replay> <new-output> <corpus> <icc>`
runs all five arms. The standalone tool exposes the existing `hdr_loss` and
`butteraugli_iters` builders only through diagnostic `--only --options` controls.
Its four reference/sample-contract tests and release all-target Clippy pass.

On r7900x, outputs are under `~/tmp/jxl-hdr-r7900x-20261104/` in
`loop-ablation/` and `combined-ablation/`; each variant retains its command,
source, encoded stream, reference study and decoder diagnostics. Full logs
on i270 are `~/tmp/jxl138-loop-remote-final.log` and
`~/tmp/jxl138-combined-remote.log`. Run with `flock ~/tmp/heavy.lock` and
`~/work/claudehints/scripts/run-heavy --mem 12G --jobs 4 -- ...`.

Resource receipts (build/test controls, not codec performance): initial paired
control rc0, 15 seconds, peak RSS 0.46 GiB, minimum available 19,093 MiB;
extended control rc0, 21 seconds, peak RSS 0.48 GiB, minimum available
18,948 MiB. Earlier setup/build failures and their logs are preserved too.
