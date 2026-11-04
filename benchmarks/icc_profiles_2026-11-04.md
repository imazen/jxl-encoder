# ICC revalidation, 2026-11-04

The original encoder repairs are already on main: `0ef8515d` restores
mAB/mBA curve headers, and `fcb3565a` bounds repetitive-profile coding at
the decoder's compression-ratio checkpoints. The synthetic curve tests
pass. Expanded zero/repetition tests at 300 KB, 1 MB and 4 MB recover the
ICC bytes exactly through upstream jxl 0.8.0 and djxl v0.12.0.
The old 0.4.3 test oracle's 1 MiB encoded-stream cap is insufficient for
the expanded case, so this module uses a separate current-version dev
dependency; other tests retain their previous decoder dependency.

Named real profiles are checked on RGB or actual CMYK samples at 1x1
and 300x260, lossless and lossy. djxl renders all twelve streams and
returns every ICC byte exactly. The PQ and sRGB v4 profiles also pass
both Rust decoders. Probev1's four cases expose an independent shared
ICC read-ahead defect in the Rust decoders.

The decoder guessed `missing + remaining_predicted_bytes / 8` bytes of
additional input. LZ77 can generate more than eight predicted bytes per
coded byte, so that guess can exceed the remaining valid file. The
decoder then returns `UnexpectedCodestreamBoxEnd` before retrying the
ICC parser. Adding trailing bytes inside a diagnostic copy's jxlc box
makes upstream recover the exact original ICC, establishing the refill
mechanism; padding is not an encoder workaround.

Native decoder main `a997f0ed` requests the parser's actual deficit.
The 1x1 Probev1 header regression fails before that repair and passes
afterward with whole-file and 1/7/257-byte chunks. Its 24 ICC-filtered
library tests, strict library/example lint and formatting pass. The
feature corpus is provisioned explicitly rather than skipped.

The encoder's prior decoder pin was the unmerged paired API PR head,
not decoder main. Merge `515f2c8a` includes the repair while retaining
all existing review changes; the encoder pin advances to that merge.
With the repair the native decoder fully renders all twelve real-profile
streams and returns exact ICCs. Released upstream jxl 0.8.0 still rejects
the four Probev1 streams, so `--icc-regression` deliberately exits nonzero
and retains those failures. No upstream report has been posted for this
new read-ahead finding.

Reproduction: `just issue128-check <pinned-manifest>` and
`just issue128-real-profiles <sweep-binary> <new-output-dir> <icc-dir>`.
The required real profiles are `ITUR_2100_PQ_FULL.ICC`,
`Probev1_ICCv4.icc` and `sRGB_ICC_v4_Appearance.icc`; missing files fail.
Streams and complete diagnostics are retained under
`~/tmp/jxl128-read-ahead-controls-2026-11-04/` (before) and
`~/tmp/jxl128-native-fixed-real-2026-11-04/` (native repair).
The Probev1 profile SHA256 is
`47f4e71af3c03b5801f6555a57be645040c8a83011947a35bc50b599bfc8613a`.

Logs: `~/tmp/jxl138-icc128-current-regressions.log`,
`~/tmp/jxl128-native-header-{before,after}.log`,
`~/tmp/jxl128-native-final-check.log`, and
`~/tmp/jxl128-native-fixed-real.log`. The final frozen encoder closure
also passes the ICC regressions alongside the metric-domain, original
byte-lock, library, animation and lint checks; see
`~/tmp/jxl138-correctness-final-closure.log`.
