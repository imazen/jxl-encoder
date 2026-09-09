# #109 F5 — lossless f32 parity vs cjxl v0.12

Data: [`lossless_float_parity_2026-09-08.tsv`](lossless_float_parity_2026-09-08.tsv)
(96 cells) + [`lossless_float_parity_large_2026-09-08.tsv`](lossless_float_parity_large_2026-09-08.tsv)
(the 4096² class, n=2 images × 4 efforts — see "The 4096² (large) class").
Harness: `examples/lossless_float_parity.rs`. Host: Mac aarch64.
Reference: cjxl/djxl **v0.12.0 `a7a9c78`** via the pinned `.ci-libjxl` tools.

## Method

Lossless has no quality axis, so "RD" here is bytes against wall time, and the
only honest comparison is at verified bit-exactness on **both** sides:

- **Our output** is decoded with djxl and asserted bit-identical to the input
  floats before its bytes enter the table. All 96 cells pass.
- **cjxl's output** was separately verified lossless on the 1024² e5/e7/e9
  cells: 0 of 3,145,728 samples differ. Without this control the byte
  comparison could have been lossless-vs-lossy and meaningless.

Source: imazen-26 `.hdr.png`, 16-bit RGB, converted to f32 by dividing by
65535 — exact, since every u16 is representable in binary32. Sizes are
**centre crops, never resamples**: resampling changes the local statistics
predictors key on, so a resampled 1024² row would not be the same content
class as its 4096² sibling. Arms are interleaved within each repetition
(block-ordered repeats inverted the sign of a result in the T3 work); wall is
the min of 3.

## Headline

Aggregate bytes (Σ ours / Σ cjxl), all sizes:

| effort | ratio |
|---|---|
| e3 | 0.958 |
| e5 | **1.010** |
| e7 | **0.508** |
| e9 | 0.518 |

At 1024² e7 the median is **ours 3,411,209 B vs cjxl 7,203,886 B** — we reach
27 % of the 12,582,912 B raw where cjxl reaches 57 %. We are smaller on 8/8
images at e3/e7/e9 and at every size, and **larger on 8/8 at 1024² e5**.

## The effort ladder is where the story is

Median bytes at 1024², each side relative to its own e3:

| | e3 | e5 | e7 | e9 |
|---|---|---|---|---|
| ours | 1.000 | 1.000 | **0.468** | 0.461 |
| cjxl | 1.000 | 0.925 | 0.924 | 0.867 |

**Our e3 and e5 are BYTE-IDENTICAL on the f32 path — 24/24 (image, size)
pairs. cjxl's differ in 24/24.** The effort dial does nothing for float below
e7. That is consistent with the #109 F3 budget: at `bit_depth = 32` the
`max_bitdepth` budget disables RCT and palette, which is most of what e5 would
otherwise add, so the profile change has nothing left to apply. Our whole gain
arrives in one step at e7, where `tree_learning` turns on
(`effort.rs`: `tree_learning: effort >= 7`).

This is worth knowing before anyone tunes the float path: **e5 is not a
meaningful operating point for lossless float today** — it costs e5 time for
e3 bytes, and it is the one cell where cjxl beats us.

## Cost model (α + β·pixels, both reported)

| effort | side | bytes α | bytes β (MB/MP) | wall α (ms) | wall β (ms/MP) |
|---|---|---|---|---|---|
| e3 | ours | −6,654 B | 7.08 | 1.2 | 229 |
| e3 | cjxl | 3,303 B | 7.36 | 5.9 | 16.5 |
| e5 | ours | −6,654 B | 7.08 | 1.4 | 221 |
| e5 | cjxl | 2,504 B | 6.98 | 11.5 | 33.3 |
| e7 | ours | **107,053 B** | **3.26** | 218.8 | 2,575 |
| e7 | cjxl | 2,687 B | 6.97 | 24.8 | 66.5 |
| e9 | ours | 105,742 B | 3.20 | 1,253.5 | 12,036 |
| e9 | cjxl | 4,883 B | 6.70 | 237.5 | 1,122 |

The e7 intercept is the learned tree: **~106 KB of fixed cost that halves the
per-pixel rate** (3.26 vs 6.97 MB/MP). Setting the two e7 fits equal gives a
crossover at **0.028 MP ≈ 168×168 px** — below that the tree does not pay for
itself, which is exactly why the 64² win is only ~4 % while the 1024² win is
2×. A "MB/MP" number alone would have hidden this.

## What it costs

Wall, median, ours/cjxl: 1024² e7 **33×** slower (2,891 ms vs 88 ms), e9
**10×** (14,135 ms vs 1,406 ms). So the honest summary is **~2× smaller files
for ~10–33× the encode time** at e7+. Whether that trade is right is a caller
policy question, not an encoder defect — but it does mean the current float
ladder offers only two useful points: e3 (fast, at cjxl parity) and e7 (2×
smaller, ~33× slower). e5 is dominated and e9 buys 1.5 % over e7 for 5× the
time.

## The 4096² (large) class

Run separately on the 13 float sources that are ≥4096 in both dimensions
(n = 2 images × 4 efforts, all bit-exact). It reproduces every finding above at
16 MP rather than softening them:

| effort | mean bytes ratio | ours ms | cjxl ms |
|---|---|---|---|
| e3 | 0.925 | 3,493 | 275 |
| e5 | **1.028** | 3,706 | 577 |
| e7 | **0.466** | 43,938 | 1,009 |
| e9 | 0.480 | 189,814 | 19,735 |

**e3 and e5 are byte-identical here too (2/2 images, 106,961,580 B and
105,103,771 B respectively)** — so the inert-ladder finding holds across the
whole size range, 64² to 4096², 26/26 (image, size) pairs in total.

The wall cost is where the large class adds something new: **e9 at 4096² takes
198 s** on our side (43,938 ms at e7 → 189,814 ms at e9, a 4.3× step for 1.4 %
fewer bytes). At 16 MP the e7→e9 trade is clearly not worth taking, which is
sharper than the 1024² picture suggested.

## Limitations — read before citing

- **Photographic content only.** The corpus's float set (`.hdr.png`) exists in
  four strata: nature, interiors, photos-general, food. This measurement says
  **nothing** about float line-art, plots, screenshots or synthetic/scientific
  data, where the predictor/entropy balance could differ entirely. Covering
  that needs a different corpus (e.g. EXR renders), and is not done here.
- **n = 8 images** in the primary arm, 2 in the 4096² arm. Enough to see a 2×
  effect and the e3/e5 identity (24/24 is not noise); not enough for percentile
  fits.
- **The 4096² class is a separate file** because only 13 of the 76 float
  sources are ≥4096 in both dimensions, and the stratified round-robin pick for
  the primary arm did not include any of them. The primary arm is therefore
  tiny/small/medium only; the large class is measured on the qualifying subset
  rather than by upscaling, which would fabricate statistics.
- **Single host (Mac aarch64), single thread count.** Wall ratios are not
  claimed to transfer to x86.
- Derived from 16-bit integer sources, so the mantissas are a 16-bit lattice.
  Genuinely continuous float (HDR capture, renders) may compress differently.
