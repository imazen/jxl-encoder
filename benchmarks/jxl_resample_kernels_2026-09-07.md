# Resampling kernel batching, issue #103

The bit-preserving optimization reduces iterative round-trip time on the
measured 12 MP photo from 7197.3 ms to 1234.5 ms. The extra iterative cost
remains 82.5% of the accompanying e7 encode time. The issue's goal of making
that cost well below encode time is still open; these measurements do not
justify adding a sharper-versus-iterative keep-best policy.

## Method

`scripts/bench_resample_kernels.py` alternates the original and changed
`resample_kernel_cost` executables within each of three repetitions. Each
invocation warms both round trips and the encode, then measures each once.
The table reports the median of the three observations. Both round trips
include the same final upsample; their difference measures the extra
iterative downsampling work. The encode uses distance 2, effort 7, explicit
resampling 2, default features plus `__internals`, without `target-cpu=native`.

Host: macOS aarch64, as recorded in each `.meta.json`. Only this host was
measured; there are no x86 performance claims. Encodes and round trips run
sequentially under `nice -n 19`. The source images are centre-cropped, never
upscaled. The 4096 cap leaves the photo at its native 3000×4000 and terminal
at its native dimensions; it does not create 4096² images.

The original executable was built from `bc60076a`, before production edits.
Metadata records both executable SHA256s and the working-copy commit used
for each run. `*_tail_2026-09-07.tsv` measures the retained 16-lane plus
4-lane adjoint implementation. The earlier photo/terminal TSVs measure the
16-lane implementation with scalar tails. Full logs and content-addressed
JXL files live under `/Users/lilith/tmp/jxl103/artifacts/`; each metadata file
names its log, and each TSV names the encoded file's SHA256.

## Measured wall time

All values are milliseconds. “Original” and “changed” refer to the binaries,
not the sharper and iterative algorithm choices.

| Image and actual dimensions | Sharper original → changed | Iterative original → changed | Encode original → changed |
|---|---:|---:|---:|
| Terminal, 64×64 | 0.4 → 0.3 | 2.7 → 1.4 | 0.7 → 1.0 |
| Terminal, 256×256 | 5.5 → 2.2 | 39.8 → 8.3 | 8.9 → 7.3 |
| Terminal, 1024×1024 | 87.6 → 30.0 | 643.1 → 112.5 | 121.9 → 93.0 |
| Photo 1421, 64×64 | 0.3 → 0.2 | 2.3 → 0.8 | 0.6 → 0.5 |
| Photo 1421, 256×256 | 5.3 → 2.1 | 38.0 → 8.2 | 8.8 → 7.1 |
| Photo 1421, 1024×1024 | 85.3 → 29.2 | 619.9 → 111.0 | 132.6 → 103.8 |
| Photo 1421, 3000×4000 | 988.9 → 334.6 | 7197.3 → 1234.5 | 1414.8 → 1089.4 |
| Line art 7026, 64×64 | 0.3 → 0.2 | 2.2 → 0.8 | 0.6 → 0.5 |
| Line art 7026, 256×256 | 5.3 → 2.0 | 37.9 → 8.0 | 7.2 → 5.6 |
| Line art 7026, 1024×1024 | 84.8 → 29.0 | 620.1 → 110.9 | 105.0 → 75.9 |

The three-repeat tiny terminal encode showed a regression despite faster
resampling, and is retained in the table. A separate 16-repeat interleaved
check (`jxl_resample_kernels_terminal_tiny_2026-09-07.tsv`) did not reproduce
it: median encode 0.6 ms for both binaries, iterative 2.3→0.8 ms. The harness
rounds times to 0.1 ms, limiting conclusions on submillisecond cells. No
performance numbers are extrapolated to other sizes or machines.

## Implementation and correctness evidence

- The sharper kernel batches four interior outputs; each output retains its
  original 144-tap accumulation order. Border indexing and ringing limits
  retain the scalar code.
- The upsampler shares support loads and extrema across four 2×2 phases,
  preserving each phase's 25-tap accumulation order.
- The adjoint batches 16 outputs, then four-output tails. Its 100-tap
  coefficients are derived at compile time from the existing phase kernels.
  Each tap still performs f64 multiply/add followed by f32 rounding.
- The adjoint of the all-one normalization image is constant in the interior;
  only border supports need recalculation. This also removes that full-size
  all-one allocation.
- Frozen scalar differential tests compare individual output bits across
  widths 1–67, heights 1–19, and larger odd/tail dimensions through 513×257,
  using signed zeros, subnormals, large finite values, and pseudorandom data.
  They also compare the new normalization field against the original adjoint
  applied to ones. Existing libjxl golden tests remain unchanged.
- Every measured run requires identical sharper, iterative, and encoded
  SHA256s both across binaries and across repetitions. All measured cells
  passed. This is a performance change, with no kernel, iteration-count,
  ringing-bound, effort-gate, or default-resampling policy change.

Phase instrumentation is available with `just resample-kernel-profile` and
the benchmark's `IMGS`, `CROP`, `REPS`, and `ARTIFACT_DIR` environment variables.
The adjoint remains the largest measured iterative phase. An f64 `mul_add`
trial passed differential tests but did not improve the 1024² terminal timing;
the retained implementation uses the original separate operations.

## Validation

Scoped formatting and `cargo clippy --workspace --all-targets -- -D warnings`
passed. The default encoder test suite and doctests passed. The Libjxl strategy
byte-lock tests, divergence-table drift tests, explicit v0.12 djxl odd-dimension
resampling test, and both RD regression tests passed unchanged. The default
integration suite includes jxl-rs round trips; no existing expectations or hash
locks were regenerated. Logs: `/Users/lilith/tmp/jxl103-tests-final.log`,
`jxl103-clippy-tail.log`, and `jxl103-resampling-gates2.log` in the same directory.

## Adjoint row staging follow-up

Starting from `61dccb75`, the adjoint now deinterleaves each support row once
into even/odd arrays. Its five adjacent tap pairs load contiguous output
lanes from those arrays. Each output still accumulates the same 100 taps in
the same order, including f64 multiply/add and f32 rounding after every tap.
This changes load organization only.

Five interleaved repeats on the same Mac compare the original batched
adjoint with staging. Terminal 1024² iterative time is 106.7→99.5 ms; photo
1421 at 3000×4000 is 1215.4→1127.2 ms. Photo encode time is
1077.3→1078.2 ms. The tiny photo's medians remain 0.8 ms iterative and
0.5 ms encode in both arms. Line art 1024² improves from 109.6 to 102.2 ms iterative, with encode
76.6→76.3 ms. Tiny terminal measurements include startup drift
and are retained without a speed claim. The issue's extra-cost target remains
open. No kernel selection policy changes.

Data and binary hashes: `jxl_adjoint_staging_{terminal,photo,lineart}_2026-09-07`
TSVs and `.meta.json` companions. Full logs and encoded artifacts are under
`/Users/lilith/tmp/jxl103/adjoint-staging-{terminal,photo,lineart}/`.

Staging validation: scoped fmt, workspace all-target clippy, the full default
encoder suite and doctests, Libjxl byte locks, divergence drift, v0.12 djxl
odd-dimension decode, and both RD gates passed unchanged. Logs:
`/Users/lilith/tmp/jxl103-adjoint-staging-{parity,clippy,tests,gates}.log`.

A separate upsample phase-loop reordering was rejected: all 33 resampling
unit tests pass, but five interleaved terminal repeats show 105.7→106.6 ms
iterative at 1024², and no tiny-image benefit. No output hash moved.
`jxl_upsample_phases_terminal_2026-09-07.{tsv,meta.json}` preserves the
comparison; the source experiment remains in jj change `uoromknx`.
The production upsample loop retains its previous organization.
