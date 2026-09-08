# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0"]
# ///
"""Acceptance checks for the set produced by `gen_float_lineart_corpus.py`.

Reports, per PFM image:
  * distinct f32 sample values, and distinct/total (the pixel count is a hard
    cap: a 64x64 RGB image has only 12,288 samples, so "> 65536 distinct" is
    unachievable there by arithmetic, not by lack of continuity);
  * `off_lattice` -- the fraction of samples that are NOT exactly k/65535 for
    integer k. This is the decisive continuity test: a float arm that is a
    16-bit lattice in disguise would score ~0 here and would merely re-measure
    the integer path.
  * a non-blank check (distinct > 1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_float_lineart_corpus import read_pfm  # noqa: E402


def stats(p: Path) -> dict:
    a = read_pfm(p).reshape(-1)
    total = a.size
    distinct = int(np.unique(a).size)
    lat = a.astype(np.float64) * 65535.0
    on_lattice = np.isclose(lat, np.rint(lat), rtol=0.0, atol=1e-4)
    return {
        "path": str(p),
        "total": total,
        "distinct": distinct,
        "distinct_frac": distinct / total,
        "off_lattice": 1.0 - float(on_lattice.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main() -> int:
    paths = [Path(x) for x in sys.argv[1:]]
    print("distinct\tdistinct/total\toff_lattice\tmin\tmax\tpath")
    for p in paths:
        s = stats(p)
        print(f"{s['distinct']}\t{s['distinct_frac']:.4f}\t{s['off_lattice']:.4f}"
              f"\t{s['min']:.5f}\t{s['max']:.5f}\t{s['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
