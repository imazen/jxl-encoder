# Is our 2x resampling worse than libjxl's? — differential port validation

**Date** 2026-09-05 · **Scratch** `/home/lilith/tmp/libjxl_resample_check/`

## Verdict

**Our 2x resampling is modestly worse than libjxl's — ~9.5 % higher (worse) butteraugli
at the resampling floor, ~3.8 SSIM2 points — but the floor itself is real in BOTH
encoders, and swapping in libjxl's downsampler does not change the conclusion.**
Against our own full-resolution ladder, our 2x is worth choosing at 2/80 quality points
(2.5 %); libjxl's 2x is worth choosing at 3/80 (3.8 %). The "2x is essentially never
worth it" finding is a property of 2x resampling, not a defect in our implementation.

## Versions and commands

| | |
|---|---|
| our encoder | `jxl-encoder` @ `ac11a123` (repo read-only; another session holds `.workongoing`) |
| | *(that session advanced HEAD to `73867a58` mid-run; the pre-built probe binary was never rebuilt — mtime `2026-09-05 22:39:41` throughout — so every "ours" number is from the single `ac11a123`-era binary)* |
| our probe | `target/release/examples/distance_targeting_probe` (already built) |
| reference | `cjxl v0.11.1 [AVX3_DL,AVX2,SSE4,SSE2]` |
| decoder (both) | jxl-oxide 0.12.5, imazen fork rev `fd4e2c3` (the repo's patched rev) |
| metrics | `butteraugli` 0.9.4 path-dep `/home/lilith/work/butteraugli/butteraugli`; `fast-ssim2` 0.7.1 |

```bash
# ours
IMG=<crop.png> CROP=1024 EFFORTS=7 DISTANCES=<grid> RESAMPLING={1,2} \
  nice -n19 target/release/examples/distance_targeting_probe
# reference
nice -n19 cjxl --resampling={1,2} -d <D> -e 7 <crop.png> out.jxl
# then scored by the standalone scorer in this dir (src/main.rs)
./target/release/jxlscore <crop.png> out1.jxl out2.jxl ...
```

Grids, effort 7 throughout: full-res `d = 1,2,4,6,8,10,12,16,20,25`;
2x `d = 0.3,0.5,1,2,3,4,6,8`. 8 images × 2 regimes × 2 encoders = **288 cells, 0 failures.**

### Scoring is provably identical between the two encoders

`butteraugli_main` was never used. The standalone scorer (`src/main.rs`) is a line-for-line
copy of the probe's scoring: jxl-oxide → `EnumColourEncoding::srgb_linear(Relative)` →
`butteraugli_linear` + `fast_ssim2` at **full resolution** against the source PNG read as RGB8.

Cross-validated on a bitstream both paths can produce (`plot_lines`, d=4, e7, full-res),
our CLI file scored by the standalone scorer vs. the probe's in-process score:

| | bytes | bfly | ssim2 |
|---|---|---|---|
| probe (in-process) | 154151 | 3.4748 | 73.816 |
| standalone scorer | 154151 | 3.4748 | 73.816 |

Exact match — so cjxl numbers here are directly comparable to our probe's.
Decoder-side upsampling is **common-mode** (jxl-oxide decodes both encoders' 2x files to
full resolution), so any floor difference is purely encoder-side: downsample filter + quantisation.

### Corpus (centre-crops to ≤1024/axis, exact probe crop math, ICC/gamma-chunk free)

`photo_tokyo` (photos-general), `photo_coast` (photos-nature), `photospng_sunset` (photos-png),
`render_flower` (renders), `webshot_climate` (web-screenshots, 1024×900), `plot_lines` (plots/line art),
`doc_noaa` (noaa-documents), `aiclip_cupcake` (ai-clipart). Both strata flagged as most
favourable to resampling (`renders`, `photos-png`) are included.

## 1. Floor comparison (the decisive measurement)

Best butteraugli each encoder's 2x regime reaches at **any** bitrate on its ladder
(smaller = better):

| image | ours floor | @d | @bytes | cjxl floor | @d | @bytes | **ratio ours/cjxl** | SSIM2 gap (cjxl−ours) |
|---|---|---|---|---|---|---|---|---|
| aiclip_cupcake | 10.498 | 0.5 | 49,496 | 9.210 | 0.5 | 52,012 | **1.140** | +3.81 |
| doc_noaa | 13.568 | 2 | 18,103 | 13.867 | 0.5 | 32,220 | **0.978** | +0.38 |
| photo_coast | 7.159 | 0.3 | 145,503 | 7.081 | 0.5 | 109,597 | **1.011** | +5.19 |
| photo_tokyo | 9.623 | 0.5 | 111,979 | 8.446 | 1 | 80,700 | **1.139** | +3.74 |
| photospng_sunset | 7.218 | 0.5 | 73,144 | 7.004 | 0.5 | 69,683 | **1.031** | +0.06 |
| plot_lines | 23.849 | 0.3 | 186,936 | 21.310 | 0.5 | 150,123 | **1.119** | +4.70 |
| render_flower | 8.437 | 0.5 | 77,108 | 6.371 | 0.5 | 77,826 | **1.324** | +2.20 |
| webshot_climate | 24.090 | 0.5 | 64,793 | 22.476 | 0.3 | 78,927 | **1.072** | +8.10 |

**Median floor ratio ours/cjxl = 1.095** (mean 1.102). cjxl's floor is lower on 7/8 images;
ours is lower on 1 (`doc_noaa`). Median SSIM2 floor gap **+3.77 points in cjxl's favour**.
Worst case `render_flower` (+32 %), best case `doc_noaa` (−2 %).

**This is a real implementation deficit** — our downsample/2x path leaves roughly 10 %
butteraugli and ~4 SSIM2 points on the table versus libjxl. It is not, however, the
order-of-magnitude difference that would overturn the unreachability finding (see §3).

### The floor is a genuine saturation in BOTH encoders

Going from d=1 to d=0.3 inside the 2x regime buys almost nothing in either encoder:

| image | enc | bytes ×(d1→d0.3) | butteraugli improvement |
|---|---|---|---|
| photo_coast | ours | 1.88× | +4.3 % |
| photo_coast | **cjxl** | 1.88× | **+0.6 %** |
| photospng_sunset | ours | 2.91× | +3.0 % |
| photospng_sunset | **cjxl** | 2.86× | **+14.9 %** |
| render_flower | ours | 2.72× | +4.0 % |
| render_flower | **cjxl** | 2.77× | **+5.2 %** |
| plot_lines | ours | 1.40× | +0.0 % |
| plot_lines | **cjxl** | 1.48× | **+3.9 %** |
| webshot_climate | ours | 1.46× | +0.4 % |
| webshot_climate | **cjxl** | 1.52× | **+1.3 %** |
| doc_noaa | ours | 1.15× | -0.4 % |
| doc_noaa | **cjxl** | 1.20× | **+1.5 %** |
| photo_tokyo | ours | 1.74× | +3.6 % |
| photo_tokyo | **cjxl** | 1.76× | **-3.0 %** |
| aiclip_cupcake | ours | 1.68× | +3.1 % |
| aiclip_cupcake | **cjxl** | 1.75× | **+8.4 %** |

Nearly tripling the bitrate moves quality by low single-digit percent on 7 of 8 images in
both encoders; the single largest response anywhere is cjxl on `photospng_sunset`, where
2.86× the bytes buys +14.9 %. libjxl saturates essentially as ours does — the resampling floor is inherent to the regime.

## 2. Does 2x ever beat its OWN full-resolution encode at matched quality?

For every full-res point, the cheapest measured 2x point reaching the same-or-better
butteraugli (no interpolation; "unreachable" if no 2x setting got there):

| encoder | full-res pts | reachable by own 2x | 2x wins (<1.0) | win rate | best ratio |
|---|---|---|---|---|---|
| ours | 80 | 18 (22 %) | 2 | **2.5 %** | 0.775 |
| cjxl | 80 | 26 (32 %) | 6 | **7.5 %** | 0.755 |

Per-image: `plot_lines`, `webshot_climate` and `aiclip_cupcake` are 0-win for BOTH encoders
(line art, screenshots and flat clipart are destroyed by 2x — cjxl reaches 0/10 on the first two).
libjxl's higher rate comes from `doc_noaa` (2), `photospng_sunset` (2), `photo_coast` (1),
`render_flower` (1). Unreachability is 78 % (ours) / 68 % (cjxl) — the same regime as the
72 % reported in the original 2236-cell study.

## 3. The decisive test: can libjxl's 2x beat OUR full-resolution ladder?

This isolates the downsampler. Hold our full-res ladder fixed, and let each encoder's
2x regime try to undercut it at matched-or-better butteraugli:

| image | our full-res pts | ours-2x reach | ours-2x wins | **cjxl-2x reach** | **cjxl-2x wins** | best cjxl-2x ratio |
|---|---|---|---|---|---|---|
| aiclip_cupcake | 10 | 0 | 0 | 0 | 0 | – |
| doc_noaa | 10 | 1 | 1 | 1 | 0 | 1.082 |
| photo_coast | 10 | 5 | 0 | 5 | 0 | 1.170 |
| photo_tokyo | 10 | 3 | 0 | 4 | 0 | 1.480 |
| photospng_sunset | 10 | 6 | 1 | 6 | 2 | 0.746 |
| plot_lines | 10 | 0 | 0 | 0 | 0 | – |
| render_flower | 10 | 3 | 0 | 5 | 1 | 0.947 |
| webshot_climate | 10 | 0 | 0 | 0 | 0 | – |
| **total** | **80** | 18 (22 %) | **2 (2.5 %)** | 21 (26 %) | **3 (3.8 %)** | 0.746 |

Substituting libjxl's downsampler moves "2x is the cheaper choice" from **2.5 % → 3.8 %**
of quality points, and leaves **74 % unreachable**. The oracle gain is essentially unchanged.

## 4. Sanity anchor: full-resolution ladders are comparable

For each cjxl full-res point, the cheapest of our full-res points reaching the same-or-better
butteraugli:

| image | matched | median bytes_ours/bytes_cjxl |
|---|---|---|
| aiclip_cupcake | 9/10 | 1.133 |
| doc_noaa | 9/10 | 1.162 |
| photo_coast | 10/10 | 1.044 |
| photo_tokyo | 10/10 | 1.043 |
| photospng_sunset | 9/10 | 1.082 |
| plot_lines | 10/10 | 1.052 |
| render_flower | 9/10 | 1.082 |
| webshot_climate | 8/10 | 1.032 |

**Corpus median 1.061** (n=74) — our full-res is ~6 % larger at matched quality. Broad parity,
so the 2x differences above are not an artifact of an overall efficiency gap. (Note this
6 % full-res deficit is of the same order as the ~10 % floor deficit, i.e. our 2x path is
not uniquely bad relative to our own full-res path.)

## Caveats — things that would make me distrust these numbers

1. **An ICC contamination was found and fixed mid-run; it had inverted one image's result.**
   `photospng_sunset.png` (a phone-gallery screenshot) carried a 520-byte `iCCP` chunk that
   PIL preserved into my crop. cjxl honoured it; our probe (`image::open().to_rgb8()`) ignores
   ICC entirely. That mismatch pinned cjxl's 2x butteraugli at a constant ~12.6 regardless of
   bitrate and made cjxl's 2x look *far worse* than ours (floor ratio 0.575) while
   simultaneously inflating cjxl's own-2x "win" count to 11/80. Re-cropping with the ICC
   stripped (RGB8 payload verified byte-identical) moved the cjxl floor 12.561 → 7.004
   (ratio 0.575 → 1.031) and the cjxl own-win count 11 → 6. **All numbers in this report are
   post-fix**; the other 7 crops were audited and are free of `iCCP`/`gAMA`/`cHRM`/`sRGB`/`cICP`.
   This is the exact trap CLAUDE.md warns about, live in the corpus.
2. **Narrow grid vs the original study.** One effort (7), one crop size (1024), 8 images,
   288 cells — against the original 2236 decision cells. This validates the *mechanism*;
   it is not a re-run of the corpus study. Effort 8+ engages the butteraugli loop and was
   not tested at all, so a loop-specific 2x defect would be invisible here.
3. **Butteraugli is non-monotonic in requested distance on several cells** (it is a
   max-norm-flavoured metric): e.g. ours `photospng_sunset` 2x d=4 → 15.33 but d=6 → 12.78;
   cjxl `photo_tokyo` 2x d=0.3 is *worse* than d=0.5. Floors are taken as the min over the
   ladder, which is robust to this, but individual cell ratios carry that noise.
4. **Three floors sit at the d=0.3 grid edge** (ours `photo_coast`, ours `plot_lines`,
   cjxl `webshot_climate`) so a finer distance might nudge them lower. All three are already
   flat there (≤0.56 % better for 1.2–1.3× the bytes), so the floor is bracketed by
   saturation rather than by the grid. Not material.
5. **Distance semantics under `--resampling` may not be identical between the two encoders.**
   The floor comparison is distance-free (min over the whole ladder), so it is immune; the
   per-d rows in §1c are not, and should be read as ladder shape rather than a d-for-d parity claim.
6. **cjxl 0.11.1, not libjxl main.** A newer downsampler upstream would not have been measured.
7. `djxl` on PATH works (v0.11.1) but was **not** used — all decoding went through jxl-oxide so
   that both encoders share one decoder. The libjxl build-tree `djxl` is indeed broken
   (missing `libIlmImf-2_5.so.25`), as stated in the brief.

## Files

`ours.tsv`, `cjxl.tsv` (raw cells) · `ANALYSIS.md`, `ANALYSIS2.md` (generated tables) ·
`analyze.py`, `analyze2.py`, `mkcrop.py`, `run_ours.sh`, `run_cjxl.sh` · `src/main.rs` (scorer) ·
`logs/` (per-run stdout/stderr, cjxl logs, progress) · `png/` (crops) · `jxl/` (cjxl bitstreams).
