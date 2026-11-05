# Correctness issue triage — 2026-11-04

This is an inventory, not a completion or release claim. Source baseline:
`554ad6966771bc2b6f79d6a6c569f55ce155644a`. Recheck GitHub and the relevant
tests before relying on a status below.

## Correctness first

| Work | Current evidence | Remaining gate |
|---|---|---|
| [#138](https://github.com/imazen/jxl-encoder/issues/138), conformance sweep | PQ DC repair `dfec8f05` passes all 21 original records and 76 additional paired PQ cases with unchanged decoder pins and error bounds. The owner-approved single fingerprint update `a73137ac` passes the complete validation recipe, including 69 hash checks, strict locks, drift checks and workspace/minimal Clippy. | Shared finer/nearest PQ DC quantization covers still, streaming and animation. The 512-square ramp costs 3189 to 5638 bytes; this selected fidelity repair is not broad matched-quality RD qualification. [Complete evidence](../benchmarks/issue138_pq_dc_fix_2026-11-05.md). [Decoder issue 73](https://github.com/imazen/zenjxl-decoder/issues/73) remains valid for the older 2965/28968 stream; the repaired encoder emits a different stream. Decoder findings remain in their own ledger. |
| [#128](https://github.com/imazen/jxl-encoder/issues/128), ICC rejection | Closed after current-main revalidation (`7197f8b5`): curve-predictor and compression-ratio repairs plus independent-decoder regressions (`icc_mab_curves`, `icc_ratio`). | Synthetic curve/repetition tests pass through 4 MB. All twelve named-profile streams render with exact ICC in djxl and the repaired native decoder; released upstream jxl 0.8.0 still rejects four CMYK cases. [Evidence](../benchmarks/icc_profiles_2026-11-04.md). |
| [#133](https://github.com/imazen/jxl-encoder/pull/133), native orientation | Merged `1989fb37` includes the MCU/HDR/ICC repairs and reuses `zenpixels::Orientation`. Local pinned-source checks pass 8,712 ordinary plus 6,534 subsampled native/EXIF/XMP combinations, 144 subsampled HDR/ICC cases and 288 HDR loop/input orientation cases with all-frame original-profile controls, 968 exact JPEG reconstructions, 64 gain-map pairs, 1,152 primary/secondary EXIF cases, 32 secondary-ICC cases and selected alpha/depth/ICC/intrinsic-size cases. | Merged in `1989fb37` after API review; all 15 platform CI jobs passed on the review head and the merge. The PR documents exact coverage and the independent original-ICC animation oracle. |
| [#136](https://github.com/imazen/jxl-encoder/issues/136), integration inventory | Animation request, clock, repeated-frame and pixel-compatibility work is integrated; see [paired contract evidence](PAIRED_CONTRACT_INTEGRATION.md). | Refresh the tracker against main; retain orientation and the separate research gates. |
| [#95](https://github.com/imazen/jxl-encoder/issues/95), wide integers | Float-output validation does not establish every low integer bit above f32 precision. | Independent raw-integer recovery and normative bit-depth audit. No new public input surface is implied. |
| [#111](https://github.com/imazen/jxl-encoder/issues/111), normative audit | Part 2 audit found repairs now on main. It did not audit Part 1. | Alternate-color bundle version/ICC-flag checks are repaired on main; embedded/nested gain-map image structure and ICC semantic compatibility remain separate. Complete the named Part 1 checks against authorized normative text; never equate decoder agreement with specification conformance. |
| [#112](https://github.com/imazen/jxl-encoder/issues/112), conformance tracking | Umbrella for these correctness gaps. | Link verified repairs and preserve unchecked acceptance criteria. |

## Related decoder work

- Native precision fixes are on decoder main `c1d457d9`, with all 19
  platform CI jobs green. This does not close the separate PQ reference
  disagreement; the encoder frozen closure still uses its recorded pin.

- [jxl-rs #1058](https://github.com/libjxl/jxl-rs/pull/1058), partial-group
  squeeze-border fix, merged 2026-10-29. The sweep manifest still selects
  `jxl = "0.8.0"`; findings from that version do not demonstrate a defect
  in the newer upstream source. Verify a released version before changing
  the sweep dependency.
- [lilith/jxl-rs #6](https://github.com/lilith/jxl-rs/pull/6), reordered
  multi-frame `jxlp` checkpoint repair, remains open for review.
- The imazen/jxl-oxide branches `fix/dqt-position-matching` (`dbfaa3a7`)
  and `fix/jxlp-minor-version` (`74b3a541`) exist remotely. Their existence
  does not mean either repair is merged into the decoder used here.
- HLG near-zero inverse-OOTF evidence is preserved in
  [the decoder ledger](DECODER_FINDINGS.md). No new HLG exclusion or
  tolerance change is authorized by this triage.

## Quality and performance after correctness

The [November 5 six-cell recheck](../benchmarks/rd_known_2026-11-05.md)
reproduces two hard known inversions and one below-threshold advisory. All
six source and encoder stream hashes match September 26 exactly. The fresh
[four-hour conformance run](../benchmarks/conformance_2026-11-05.md) is in
progress; its expanded ICC set is explicitly recorded. Four additional
[PQ agreement failures](../benchmarks/pq_agreement_2026-11-05.md) reproduce
with unchanged pins and bounds. Diagnostic linear controls localize the disagreement before PQ transfer;
its exact reconstruction cause and required precision remain open.
The nightly effort gate also reproduces a lossless scan growing from 180557
bytes at e3 to 182641 at e5; explicit tree learning produces 175842 bytes.
That isolated ablation does not qualify a new default.

- [#114](https://github.com/imazen/jxl-encoder/issues/114) and
  [#103](https://github.com/imazen/jxl-encoder/issues/103): retain the
  independently corroborated IQA inversions, screenshot matched-quality
  gate and optional resampling cost target. The open
  [#132](https://github.com/imazen/jxl-encoder/pull/132) changes RD policy;
  it is not a correctness repair or qualified default.
- [#99](https://github.com/imazen/jxl-encoder/issues/99),
  [#110](https://github.com/imazen/jxl-encoder/issues/110), and
  [#113](https://github.com/imazen/jxl-encoder/issues/113): keep the unmet
  wall targets and broad byte/time/memory qualification separate from
  selected regression evidence.
- [#140](https://github.com/imazen/jxl-encoder/pull/140), opened
  2026-11-04, removes metric-input copies and updates the sibling closure.
  GPU execution and complete encoder validation remain separate from its
  focused tests. It is not part of current main.
- [#24](https://github.com/imazen/jxl-encoder/issues/24) and
  [#105](https://github.com/imazen/jxl-encoder/issues/105): oracle benefit
  and held-out qualification precede picker training or effort-interface
  changes. The owner-deferred fleet qualification remains deferred.
- [#108](https://github.com/imazen/jxl-encoder/issues/108) needs the actual
  consuming application and deployment envelope. Library tests alone
  cannot supply that information or authorize a release.
