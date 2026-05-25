# W44-AUDIT-7 Phase 2 — Top-3 Investigation Brief

**Date**: 2026-05-24
**Parent**: W44-AUDIT-7 Phase 1 (wider-corpus bench, 240 cells × 20 images)
**Source data**:
- `benchmarks/w44_audit_7_wider_corpus_2026-05-24.tsv` (raw)
- `benchmarks/w44_audit_7_wider_corpus_2026-05-24.md` (tables A/B/C/D)
- `benchmarks/w44_audit_7_wider_corpus_2026-05-24.meta` (provenance)

Three follow-on chunks ranked by Pareto EV. Each is independently shippable;
do not block one on another.

---

## Chunk 1: AUDIT-6 sub-discriminator for CLIC2025 M3>=80 PHOTOS

**Status**: HIGHEST PRIORITY — measurement-driven, low-risk port.

**Wedge** (Phase 1 finding):

The AUDIT-6 M3-colourfulness gate (`m3_colourfulness >= 80` → W44-109/W44-105
screenshot qac seed scale lift) was calibrated on:
- codec_wiki (M3=145.73, mixed text+chart screenshot) — WIN target
- 5 text screens (M3 ∈ [10, 29]) — REJECT class

Phase 1 introduces 2 NEW image classes that fire the gate but were not in
calibration:
- clic_0c49a5 (M3=95.91, CLIC2025 web photo)
- clic_22ea12 (M3=105.30, CLIC2025 web photo)
- 1189261 (M3=98.84, CID22 landscape photo)

Results on M3>=80 cells (12 cells per image × 4 images = 48 cells):

| image       | M3     | mean dBytes% | mean dSsim2 | min dSsim2 |
|---          |---     |---           |---          |---         |
| codec_wiki  | 145.73 | -20.68%      | -1.23       | -4.33      |
| 1189261     | 98.84  |  -0.83%      | -0.03       | -0.61      |
| clic_0c49a5 | 95.91  |  -1.18%      | -0.54       | -1.11      |
| clic_22ea12 | 105.30 |  -0.88%      | **-1.54**   | **-3.84**  |

clic_22ea12 takes the biggest hit despite saving only -0.88% bytes — the
SSIM2-per-byte trade is structurally worse than codec_wiki's. Two CLIC
M3>=80 cells regress SSIM2 by 3+ points (clic_22ea12 e7 d=4 -3.84,
clic_22ea12 e9 d=4 -3.54). For a photo, that's visible quality loss.

**Hypothesis**: M3 alone is insufficient to distinguish "high-colour
screenshot" (codec_wiki) from "high-colour web photo" (clic_22ea12). The
W44-105 buttloop seed-scale lift wants the FORMER and hurts the LATTER.

**Mechanism candidates** (rank by EV):

A. **Mirror the W44-96 sub-discriminator pattern**: add an `edge_density >= X`
   AND `flat_color_block_ratio < Y` ANDed clause to the AUDIT-6 gate.
   codec_wiki has high edge density (chart lines + text); clic_22ea12 is
   smooth photography with low edge density.
   - Compute edge_density on the existing `ZenanalyzeProxies` struct
     (already wired in W44-91; just add the field).
   - Probe on the 4 M3>=80 images: confirm clic_0c49a5 + clic_22ea12 fall
     below the threshold while codec_wiki + 1189261 stay above.
   - Bisect threshold on (edge_density, fcbr) space.

B. **Cross-reference auto_classify_content_class** (W44-164): if AUDIT-6 only
   fires when `content_class == Screenshot`, the CLIC photos get rejected by
   the screenshot discriminator (`fcbr < 0.35` knocks out photo-class
   regardless of M3). Simpler — single AND with already-shipped state.

**Acceptance gates**:
- ≥ 2 of 4 wedge cells close (clic_22ea12 e7/e9 d=4 priority targets)
- codec_wiki SSIM2 wins preserved (mean dSsim2 stays ≥ -1.5)
- Zero NEW M3<80 wedges introduced (the gate stays disabled on those)
- 1189261 SSIM2 within ±0.5 vs Phase 1 baseline (don't break the landscape)

**Risk**: LOW. Single-line predicate extension; AUDIT-6 was already known to
be calibrated narrowly. The `ZenanalyzeProxies` plumbing exists from W44-91/96.

**Estimated wall**: 1-2 sessions (1 to bisect, 1 to validate + ship).

---

## Chunk 2: graph e7 d∈[0.5, 1.0, 2.0] BYTES regression cluster

**Wedge** (Phase 1 finding):

| image | class           | M3    | effort | dist | dBytes% | dSsim2 |
|---    |---              |---    |---     |---   |---      |---     |
| graph | SCREEN_GRAPHICS | 11.75 | e7     | 2.0  | **+56.50%** | +3.02 |
| graph | SCREEN_GRAPHICS | 11.75 | e7     | 1.0  | **+9.51%**  | -0.36 |
| graph | SCREEN_GRAPHICS | 11.75 | e7     | 0.5  | **+11.74%** | +0.19 |

3 cells × 1 image, all at e7. The +56.5% bytes regression at d=2 is the
LARGEST byte regression in the entire 240-cell bench. SSIM2 IMPROVES on 2 of
3 cells (+3.02 at d=2, +0.19 at d=0.5), so the encoder is correctly trading
bytes for quality — but the trade is way too generous.

graph.png is a small (796×481) high-contrast screenshot of a chart/graph
with thin lines + text labels + colored bars. Low M3 (11.75), low edge
density (chart lines but mostly flat), low fcbr.

**Mechanism candidates**:

A. **Identify which W44-* gate fires for graph e7**. Likely candidates:
   - W44-29 (HIGH_D_PHOTO_SMOOTH_THRESHOLD = 50) — mask1x1<50 path?
   - W44-91/164 auto_classify giving wrong content_class (Screenshot?)
   - W44-105 buttloop screen-seed lift at e7 — but AUDIT-6 should gate on M3
   - DC tree learning (W44-54/171) for small chart-class images
   - Custom coefficient orders (W44-77/201/205)

   The dispatch trace can use the `__internals` feature + the existing
   `JXL_W44_*_DISABLE` env hooks to bisect.

B. **Reproduce + bisect**: build an isolated `examples/w44_audit_7_graph_*.rs`
   that runs graph e7 d=2 with each major gate forced OFF via env, identify
   which one introduces the +56.5%.

C. The pattern (large e7 bytes regression, +3 SSIM2 win, low M3) suggests
   the encoder is engaging a "more bytes for better quality" path that cjxl
   skips. Likely AC strategy hierarchy decision or LZ77 + entropy clustering
   choice that favors small files in cjxl.

**Acceptance gates**:
- graph e7 d=2 bytes within ±15% of cjxl (current: +56.5%)
- SSIM2 within ±2.0 of cjxl (current: +3.02 — sacrificing some SSIM2 is OK)
- Other graph cells (e5, e9) byte-identical OR within ±5% bytes
- Other SCREEN_GRAPHICS images (gui, windows95) within ±5% of pre-fix bytes
- No PHOTO_* regression > 1% bytes

**Risk**: MEDIUM. The bisection is mechanical but the fix may require a
distance-narrowed gate (graph wedge is e7-only, not e5 or e9).

**Estimated wall**: 2-3 sessions (bisect, propose fix, validate).

---

## Chunk 3: PHOTO e7+ d=4 SSIM2 cluster

**Wedge** (Phase 1 finding):

8 of the 20 highest-severity wedges are PHOTO cells at e7+ d=4 with SSIM2
deficit ≥ 1.0:

| image       | class           | M3    | e  | dBytes% | dSsim2 |
|---          |---              |---    |--- |---      |---     |
| 1418519     | PHOTO_PORTRAIT  | 36.8  | e7 | -6.46%  | -1.65  |
| 1418519     | PHOTO_PORTRAIT  | 36.8  | e5 | -4.25%  | -1.49  |
| 1279330     | PHOTO_PORTRAIT  | 55.6  | e7 | -2.33%  | -1.25  |
| 1475938     | PHOTO_LANDSCAPE | 21.7  | e7 | -2.10%  | -1.61  |
| 1544947     | PHOTO_SMOOTH    | 10.8  | e7 | -1.04%  | -1.45  |
| 1544947     | PHOTO_SMOOTH    | 10.8  | e9 | -1.20%  | -1.38  |
| clic_100a02 | CLIC2025_WEB    | 48.4  | e5 | -2.58%  | -1.16  |
| clic_100a02 | CLIC2025_WEB    | 48.4  | e7 | -3.15%  | -2.22  |

Pattern: photo cells consistently saving bytes (good) but losing SSIM2
(bad) at high distance (d=4). The trade is consistent — bytes/SSIM2 ratio
roughly matches a "quality dial" being set 1-2 distance points higher than
cjxl on photo content at e7+ d=4.

**Hypothesis**: The W44-78 / W44-91 / W44-96 / W44-29 photo entropy_mul
suppression stack (which lifts large-DCT cost models on smooth photos) is
over-tuned at d=4. Most W44-9X chunks were calibrated against the
1420710/1531677/1189261 cells with mixed distance coverage. The wider corpus
exposes that PORTRAIT class images (faces, complex chroma) and CLIC
mid-content photos sit OUTSIDE the photo-smooth cluster the gates were
tuned on.

**Mechanism candidates**:

A. **Distance-band narrowing on the suppression gates**: most W44-29 and
   children fire at `distance >= 3.0`; the d=4 cells get the strongest
   suppression. Narrow the high-d gate to `distance >= 4.5` and see if the
   cluster softens.

B. **Per-image PORTRAIT discriminator**: photo-portrait content (high mask1x1
   variance on face regions, high chroma activity, low fcbr) is structurally
   different from photo-smooth. Add a sub-discriminator that rejects the
   suppression when the image is portrait-class (high chroma colourfulness +
   moderate edge density + mask1x1 std-dev > threshold).

C. **Re-bisect the W44-77 entropy_mul tunings on photo content at d=4**: the
   1420710/1531677 calibration was tight; the wider corpus may surface that
   the chosen `dct32x32 = 1.34` value sacrifices photo-portrait SSIM2 for
   photo-smooth bytes.

**Acceptance gates**:
- 5 of 8 PHOTO d=4 cluster cells close (dSsim2 within ±0.7 of cjxl)
- Bytes savings preserved within 2% (don't lose the byte wins entirely)
- 1418519/1531677/1420710 cells (Phase 1 baseline winners) stay within ±0.5
  SSIM2 of post-W44-78 (no regression of prior wins)
- gb82-sc screenshot wins preserved (the wedge is photo-only)

**Risk**: MEDIUM-HIGH. The entropy_mul calibration touches many w44_* gates
and could ripple. Best done after Chunk 2 to ensure the screen path stays
clean.

**Estimated wall**: 2-3 sessions (bisect, propose narrowing OR
sub-discriminator, validate).

---

## Combined verdict on AUDIT-6 generalization

**PARTIAL CONFIRMATION** (per acceptance gate (d) in the .meta file):

- The M3>=80 gate correctly partitions images (4 fire / 16 reject), zero
  false-fire on the 16 reject set
- The codec_wiki primary win is preserved (-20.7% bytes)
- BUT: 2 NEW M3>=80 images (clic_22ea12, clic_0c49a5) hit by the lift
  exhibit SSIM2 regression at d=4 that wasn't visible in the calibration
  corpus. The lift mechanism is over-aggressive on CLIC web photos.

**Recommended ship order**: Chunk 1 → Chunk 2 → Chunk 3. Chunk 1 closes the
clic_22ea12 SSIM2 wedge with a 1-line dispatch change; Chunk 2 is the
single largest byte regression in the corpus; Chunk 3 has the highest cell
count but is the most invasive fix.
