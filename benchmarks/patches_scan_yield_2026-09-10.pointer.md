# Does the patches scan earn its wall? — 2026-09-10

Harness: [`jxl-encoder/examples/patches_scan_yield_probe.rs`](../jxl-encoder/examples/patches_scan_yield_probe.rs).
Host: mac (Darwin 25.5.0, aarch64). Build: `c7decbb3` plus the BFS prefilter
under test.

Grid: 21 imazen-26 strata × 1 image each (first `.sdr.png` in sorted order —
deterministic, not random, so the modal class cannot dominate) × sizes
{64, 256, 1024, 2048} where the source spans them (never upscaled) ×
distances {0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12} at ≤1024 and {1, 4, 10} at 2048 ×
efforts {5, 7} × two arms (patches ON = shipped default, and
`with_patches(false)`). 1,330 cells, 5,320 encodes.

`used` is ground truth — the two arms' byte counts differ. `scanned` is whether
the W36-3 `PatchesDispatch::Auto` gate let the scan run, read from
`__internals::take_last_patches_detect_stats()`, which also supplies all
fifteen per-stage candidate counters.

## What it says

**542 scans ran; 109 produced any byte change. 433 scans (80 %) were pure
wasted wall** — 2,842 ms of 42,228 ms total encode time, 6.7 %. The scan as a
whole is 11.0 % of encode wall on this grid.

Per-stratum shares are in [the summary TSV](patches_scan_yield_2026-09-10.tsv).
The scan is ~0 % on six photo/illustration strata (it never fires), 20-29 % on
documents, screenshots, clipart, patents and textures — and on four of those
(`6000-patents` 23.3 %, `6800-manuscript-text` 28.7 %, `9000-ai-clipart`
26.2 %, `2400-textures` 22.0 %) **every single scan was wasted**. The strata
where it pays are the NPS brochures, NOAA documents, plots and both screenshot
classes, and even there most cells are wasted (e.g. mobile screenshots: 50
scans, 20 used).

**No cell was `used` without being `scanned`**, so the Auto gate had zero
false negatives here — its problem is precision, not recall, exactly as its
rustdoc says it was designed to prefer.

## The early-out that does NOT work

Phase timing puts steps 1-2 at 0.3 ms of an ~11 ms scan and the BFS at ~90 %,
so `num_seeds` is the only stat cheap enough to gate on. A strict
`num_seeds < 1329` cut looks like it kills 60 % of wasted scans with zero used
cells lost — but that is an artefact of size, not content: seeds scale with
pixel count, and all 90 of the 64² cells sit below any such bound while none of
them use patches. Normalised as seed density the classes overlap completely
(used cells go down to 0.023 seeds/block, wasted cells up to 0.71). **Do not
ship a `num_seeds` cut.**

The stats that DO separate are all post-BFS — `accepted_ccs < 39` kills 85 % of
wasted scans, `final_occurrences < 15` kills 94 %, both with zero used cells
lost — but they arrive after the money has been spent. They are only useful as
evidence that the scan's own late stages know the answer, which is why the
lever taken instead was making the BFS itself cheaper (see
[the A/B](patches_bfs_prefilter_ab_2026-09-10.tsv)).

## Caveats

One image per stratum is enough to rank strata and to prove the wasted-scan
mechanism; it is **not** enough to fit a threshold or claim a population
prevalence, and no threshold was fitted from it. Wall columns are min of 2 and
absolute values are not comparable across runs (the post-fix repeat of this
same grid drifted +7.2 % on total wall while the scan differential moved
−11.1 %); within-run differentials are what the conclusions rest on.

## Raw data

Two full 1,330-row TSVs (232 KB each, over the repo's 30 KB limit) are retained
outside git:

- pre-fix: `~/tmp/patches_yield_smoke.tsv`
  sha256 `6657c5fecb8ccb0b6dcdfd55ec5e25989e0276c7e0d1bdcd4acfa48d7bfa8e7b`
- post-fix: `~/tmp/patches_yield_after.tsv`
  sha256 `1212a740af63512c7dd2c5a6312e0476e09ff6559293c0c8c8a871c76b440ccb`

All 1,330 cells match between them on `bytes_on`, `bytes_off` and all fifteen
scan counters — that is the byte-identity evidence for the BFS prefilter.
Regenerate either with the harness command above.
