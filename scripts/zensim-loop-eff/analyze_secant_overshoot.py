#!/usr/bin/env python3
"""Per-arm intermediate-overshoot read over zensim-loop per-compare traces (T2).

The secant's guards exist to stop the controller stepping PAST the target when
the measured elasticity is a bad extrapolant. Census and median |err| only see
the EMITTED iterate, so under emit-best they cannot see an excursion the loop
later recovered from — this reads the per-compare trace directly.

Input: TSVs written by `JXL_ZENSIM_TRACE`, columns
`trace_id  iter  score  qf_mean  qf_min  qf_max  iter_ms`, trace_id
`label|image|class|target|arm`. One file per arm, or a concatenation.

**Overshoot is signed against the approach direction.** Iterate 0 is the
un-steered starting point, so its side of the target fixes which way the
controller is travelling; an excursion is only an overshoot if it lands on the
FAR side. (Scoring |score - target| instead measures the approach and reports
a large "overshoot" for a cell that merely started far away and converged
monotonically — the reading this file exists to avoid.)

    approach from above (score_0 > target):  over_i = max(0, target - score_i)
    approach from below (score_0 < target):  over_i = max(0, score_i - target)

Per arm: n cells, cells whose worst overshoot exceeds 2 / 5 / 8, median and max
worst-overshoot, mean crossings of the target over iterates 0..K (an
overshooting controller oscillates), and the median final |err| in-loop.
"""

import argparse
import statistics
from pathlib import Path


def med(xs):
    return statistics.median(xs) if xs else float("nan")


ap = argparse.ArgumentParser()
ap.add_argument("traces", nargs="+", help="trace TSV(s) written by JXL_ZENSIM_TRACE")
a = ap.parse_args()

# arm -> cell -> list[(iter, score, target)]
cells: dict[str, dict[str, list[tuple[int, float, float]]]] = {}
for f in a.traces:
    for line in Path(f).read_text().splitlines():
        if not line.strip():
            continue
        p = line.split("\t")
        if len(p) < 3:
            continue
        bits = p[0].split("|")
        if len(bits) < 4:
            continue
        arm, image, target = bits[0], bits[1], float(bits[3])
        cells.setdefault(arm, {}).setdefault(f"{image}|{target}", []).append(
            (int(p[1]), float(p[2]), target)
        )

hdr = ("arm", "n", "o>2", "o>5", "o>8", "med_over", "max_over", "mean_cross", "med_final")
print(f"{hdr[0]:16s} {hdr[1]:>3s} {hdr[2]:>4s} {hdr[3]:>4s} {hdr[4]:>4s} "
      f"{hdr[5]:>9s} {hdr[6]:>9s} {hdr[7]:>10s} {hdr[8]:>9s}")
for arm in sorted(cells):
    overs, crossings, finals = [], [], []
    for pts in cells[arm].values():
        pts.sort()
        target = pts[0][2]
        from_above = pts[0][1] > target
        rest = [s for i, s, _ in pts if i >= 1]
        if not rest:
            continue
        overs.append(
            max((target - s) if from_above else (s - target) for s in rest + [target])
        )
        seq = [s - target for _, s, _ in pts]
        crossings.append(
            sum(1 for x, y in zip(seq, seq[1:]) if (x > 0) != (y > 0))
        )
        finals.append(abs(rest[-1] - target))
    n = len(overs)
    print(
        f"{arm:16s} {n:3d} {sum(1 for o in overs if o > 2):4d} "
        f"{sum(1 for o in overs if o > 5):4d} {sum(1 for o in overs if o > 8):4d} "
        f"{med(overs):9.3f} {max(overs):9.3f} "
        f"{statistics.mean(crossings):10.3f} {med(finals):9.3f}"
    )
