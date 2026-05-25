#!/usr/bin/env python3
"""Diagnose cvvdp's position relative to butteraugli's bytes-vs-cvvdp Pareto front.

The Phase 6 verdict was OPT_IN_ONLY because cvvdp produced +155-351% larger
files at the same nominal `distance`. This is the bytes-axis cost the
verdict flagged. But the right question for the user is:

  Q1: At the same achieved cvvdp_jod, does butteraugli produce smaller files?
  Q2: At the same bytes, does cvvdp achieve higher cvvdp_jod?
  Q3: Is one backend Pareto-dominated by another on (bytes, cvvdp_jod)?

If C_GPU is Pareto-dominated by B (B has same/better cvvdp at same/fewer bytes
everywhere), the cvvdp loop is fundamentally over-encoding. The fix is to find
the JOD targets that yield Pareto-OPTIMAL bytes — i.e. the JODs butteraugli
HAPPENS to hit, mapped back through the distance axis.

If C_GPU sits on its OWN Pareto front but at a different point than B (e.g.
higher cvvdp + higher bytes everywhere), the user can pick their preferred
trade-off but cvvdp isn't strictly dominated.

This script does both diagnostics across all (image, corpus, effort) cells.

Output: scripts/cvvdp_pareto_diagnosis_<DATE>.tsv + console summary.
"""
from __future__ import annotations
import csv
import sys
import statistics
from collections import defaultdict
from typing import NamedTuple


class Row(NamedTuple):
    image: str
    corpus: str
    effort: int
    distance: float
    backend: str
    bytes: int
    wall_ms: float
    butter_cpu: float | None
    cvvdp_gpu: float | None
    ssim2: float | None


def load(path: str) -> list[Row]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            def _ofloat(s: str) -> float | None:
                if s in ("", "NA"):
                    return None
                try:
                    return float(s)
                except ValueError:
                    return None

            try:
                rows.append(
                    Row(
                        image=r["image"],
                        corpus=r["corpus"],
                        effort=int(r["effort"]),
                        distance=float(r["distance"]),
                        backend=r["backend"],
                        bytes=int(r["bytes"]) if r["bytes"] not in ("", "NA") else 0,
                        wall_ms=float(r["wall_ms"]) if r["wall_ms"] not in ("", "NA") else 0.0,
                        butter_cpu=_ofloat(r["score_butter_cpu"]),
                        cvvdp_gpu=_ofloat(r["score_cvvdp_gpu"]),
                        ssim2=_ofloat(r["score_ssim2"]),
                    )
                )
            except (ValueError, KeyError):
                continue
    return rows


def is_pareto_dominated(point: tuple[float, float], others: list[tuple[float, float]]) -> bool:
    """point = (bytes, -cvvdp) — smaller is better on both."""
    pb, pc = point
    for ob, oc in others:
        if (ob, oc) == (pb, pc):
            continue
        if ob <= pb and oc <= pc and (ob < pb or oc < pc):
            return True
    return False


def pareto_front(points: list[tuple[float, float]]) -> set[int]:
    """Return indices of points that are on the Pareto front
    (minimize bytes, minimize -cvvdp = maximize cvvdp)."""
    front = set()
    for i, p in enumerate(points):
        if not is_pareto_dominated(p, points):
            front.add(i)
    return front


def main(path: str) -> int:
    rows = load(path)
    print(f"Loaded {len(rows)} rows from {path}", file=sys.stderr)

    # Per (image, effort), compute Pareto front across all 4 backends × 7 distances
    by_cell: dict[tuple[str, int], list[Row]] = defaultdict(list)
    for r in rows:
        if r.cvvdp_gpu is None or r.bytes == 0:
            continue
        by_cell[(r.image, r.effort)].append(r)

    # We only care about cells where C_GPU diverges from B — i.e. effort=8 cells
    # where the buttloop fires. At effort<8 cvvdp opt-in is a no-op.
    e8_cells = {k: v for k, v in by_cell.items() if k[1] == 8}

    # Pareto win counts: across all (image, effort=8) cells, count how often each
    # backend lands on the Pareto front of (bytes, -cvvdp_gpu).
    backend_pareto_wins: dict[str, int] = defaultdict(int)
    backend_pareto_observations: dict[str, int] = defaultdict(int)
    backend_distance_pareto_wins: dict[tuple[str, float], int] = defaultdict(int)
    backend_distance_observations: dict[tuple[str, float], int] = defaultdict(int)

    # For each image+effort, see if there's a backend that strictly dominates
    # another at the same distance level — that's the "same-distance bytes
    # overhead" question.
    same_distance_comparison: dict[tuple[str, str, float], list[tuple[int, float]]] = defaultdict(list)

    out_rows = []
    for (image, effort), cells in sorted(e8_cells.items()):
        points = [(r.bytes, -r.cvvdp_gpu) for r in cells]
        front_indices = pareto_front(points)
        for i, r in enumerate(cells):
            backend_pareto_observations[r.backend] += 1
            backend_distance_observations[(r.backend, r.distance)] += 1
            if i in front_indices:
                backend_pareto_wins[r.backend] += 1
                backend_distance_pareto_wins[(r.backend, r.distance)] += 1
            same_distance_comparison[(image, r.backend, r.distance)].append((r.bytes, r.cvvdp_gpu))

    # Pareto win-pct per backend
    print("\n## Pareto-front position per backend (bytes vs cvvdp_gpu, e=8 cells)")
    print("# A point is Pareto-optimal if no other point in the same (image, e=8) cell")
    print("# has both ≤ bytes AND ≥ cvvdp_gpu.")
    print()
    print("backend\tobservations\tpareto_wins\tpareto_win_pct")
    for backend in sorted(backend_pareto_observations):
        n = backend_pareto_observations[backend]
        w = backend_pareto_wins[backend]
        pct = 100.0 * w / max(n, 1)
        print(f"{backend}\t{n}\t{w}\t{pct:.1f}")

    # Per-distance breakdown
    print("\n## Per-(backend, distance) Pareto win rate")
    print("# At each distance, what fraction of cells does this backend land on the front?")
    print()
    print("backend\tdistance\tobservations\twins\twin_pct")
    distances = sorted({d for _, d in backend_distance_observations})
    for backend in sorted({b for b, _ in backend_distance_observations}):
        for d in distances:
            n = backend_distance_observations.get((backend, d), 0)
            w = backend_distance_pareto_wins.get((backend, d), 0)
            if n == 0:
                continue
            pct = 100.0 * w / n
            print(f"{backend}\t{d:.2f}\t{n}\t{w}\t{pct:.1f}")

    # Same-distance bytes overhead: at SAME distance, how does C_GPU compare to B?
    print("\n## Same-distance comparison: C_GPU bytes overhead AND cvvdp gain vs B")
    print("# For each (image, distance) at e=8, compute:")
    print("#   bytes_overhead_pct = (C_GPU.bytes - B.bytes) / B.bytes * 100")
    print("#   cvvdp_gain         = C_GPU.cvvdp - B.cvvdp  (positive = C_GPU better on cvvdp metric)")
    print("# Median + p25/p75 per distance.")
    print()
    distance_stats: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for (image, b_backend, distance), pts in same_distance_comparison.items():
        if b_backend != "B":
            continue
        # Find matching C_GPU
        c_pts = same_distance_comparison.get((image, "C_GPU", distance))
        if not c_pts:
            continue
        b_bytes, b_cvvdp = pts[0]
        c_bytes, c_cvvdp = c_pts[0]
        if b_bytes == 0:
            continue
        bytes_overhead_pct = 100.0 * (c_bytes - b_bytes) / b_bytes
        cvvdp_gain = c_cvvdp - b_cvvdp
        distance_stats[distance].append((bytes_overhead_pct, cvvdp_gain))

    print("distance\tn\tmedian_bytes_overhead_pct\tp25\tp75\tmedian_cvvdp_gain\tp25\tp75")
    for d in sorted(distance_stats):
        vals = distance_stats[d]
        b_pcts = sorted(p[0] for p in vals)
        c_gains = sorted(p[1] for p in vals)
        n = len(vals)
        b_med = statistics.median(b_pcts)
        b_p25 = b_pcts[n // 4]
        b_p75 = b_pcts[3 * n // 4]
        c_med = statistics.median(c_gains)
        c_p25 = c_gains[n // 4]
        c_p75 = c_gains[3 * n // 4]
        print(
            f"{d:.2f}\t{n}\t{b_med:+.2f}\t{b_p25:+.2f}\t{b_p75:+.2f}"
            f"\t{c_med:+.4f}\t{c_p25:+.4f}\t{c_p75:+.4f}"
        )

    # Equal-cvvdp bytes comparison: at same target cvvdp value, which backend produces fewer bytes?
    # Interpolate per-image (bytes, cvvdp) curve for each backend; sample at common cvvdp grid.
    # Phase 8d (2026-05-25): extended to also report C_GPU_v2 (Phase 8c renorm-only) and
    # C_GPU_v3 (Phase 8c renorm + Phase 8d tighten). The Phase 8a baseline `C_GPU` is also
    # picked up automatically if rows exist in the input.
    print("\n## Equal-cvvdp comparison: bytes ratio (CVVDP_variant / B) at target cvvdp values")
    print("# For each (image, e=8), interpolate bytes-at-target-cvvdp for B and each")
    print("# cvvdp variant. Ratio = variant / B — < 1 means the variant uses fewer bytes")
    print("# for the same cvvdp score.")
    print()

    targets_to_test = [9.5, 9.7, 9.8, 9.9, 9.95, 9.98, 9.99]
    cvvdp_backends = ("C_GPU", "C_GPU_v2", "C_GPU_v3")
    by_image_backend: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for (image, _), cells in e8_cells.items():
        for r in cells:
            if r.backend == "B" or r.backend in cvvdp_backends:
                by_image_backend[(image, r.backend)].append((r.cvvdp_gpu, r.bytes))

    def bytes_at_cvvdp(curve: list[tuple[float, float]], target: float) -> float | None:
        """Interpolate bytes at target cvvdp.
        Curve is list of (cvvdp, bytes). Higher cvvdp = more bytes."""
        if not curve:
            return None
        sorted_curve = sorted(curve)  # by cvvdp ascending
        # If target outside range, can't extrapolate.
        if target < sorted_curve[0][0] or target > sorted_curve[-1][0]:
            return None
        # Linear interp.
        for i in range(len(sorted_curve) - 1):
            cv0, by0 = sorted_curve[i]
            cv1, by1 = sorted_curve[i + 1]
            if cv0 <= target <= cv1:
                if cv1 == cv0:
                    return by0
                t = (target - cv0) / (cv1 - cv0)
                return by0 + t * (by1 - by0)
        return None

    images = {img for (img, _), _ in e8_cells.items()}
    # Only report variants that actually have data in the TSV.
    active_variants = [
        v for v in cvvdp_backends
        if any(by_image_backend.get((img, v)) for img in images)
    ]
    if not active_variants:
        print("# (no cvvdp backend rows in the input — skipping equal-cvvdp comparison)")
    for variant in active_variants:
        print(f"\n### variant = {variant}")
        print(f"target_cvvdp\tn\tmedian_{variant}/B_bytes_ratio\tp25\tp75\tcvvdp_wins\tbutter_wins")
        for target in targets_to_test:
            ratios = []
            c_wins = 0
            b_wins = 0
            for image in images:
                b_curve = by_image_backend.get((image, "B"), [])
                c_curve = by_image_backend.get((image, variant), [])
                b_bytes = bytes_at_cvvdp(b_curve, target)
                c_bytes = bytes_at_cvvdp(c_curve, target)
                if b_bytes is None or c_bytes is None or b_bytes == 0:
                    continue
                ratio = c_bytes / b_bytes
                ratios.append(ratio)
                if ratio < 1.0:
                    c_wins += 1
                else:
                    b_wins += 1
            n = len(ratios)
            if n == 0:
                continue
            ratios.sort()
            med = statistics.median(ratios)
            p25 = ratios[n // 4]
            p75 = ratios[3 * n // 4]
            print(f"{target:.3f}\t{n}\t{med:.3f}\t{p25:.3f}\t{p75:.3f}\t{c_wins}\t{b_wins}")

    # Per-corpus split of the equal-cvvdp comparison
    print("\n## Per-corpus equal-cvvdp bytes ratio")
    print()
    by_image_corpus: dict[str, str] = {}
    for (img, _), cells in e8_cells.items():
        if cells:
            by_image_corpus[img] = cells[0].corpus

    print("corpus\ttarget_cvvdp\tn\tmedian_C/B\tcvvdp_wins\tbutter_wins")
    for corpus_name in sorted(set(by_image_corpus.values())):
        for target in [9.7, 9.8, 9.9, 9.95]:
            ratios = []
            c_wins = 0
            b_wins = 0
            for image in images:
                if by_image_corpus.get(image) != corpus_name:
                    continue
                b_curve = by_image_backend.get((image, "B"), [])
                c_curve = by_image_backend.get((image, "C_GPU"), [])
                b_bytes = bytes_at_cvvdp(b_curve, target)
                c_bytes = bytes_at_cvvdp(c_curve, target)
                if b_bytes is None or c_bytes is None or b_bytes == 0:
                    continue
                ratio = c_bytes / b_bytes
                ratios.append(ratio)
                if ratio < 1.0:
                    c_wins += 1
                else:
                    b_wins += 1
            n = len(ratios)
            if n == 0:
                print(f"{corpus_name}\t{target:.3f}\t0\tNA\t0\t0")
                continue
            med = statistics.median(ratios)
            print(f"{corpus_name}\t{target:.3f}\t{n}\t{med:.3f}\t{c_wins}\t{b_wins}")

    return 0


if __name__ == "__main__":
    p = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv"
    )
    sys.exit(main(p))
