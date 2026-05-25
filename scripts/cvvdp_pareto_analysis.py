#!/usr/bin/env python3
"""cvvdp_pareto_analysis.py — Phase 6 Pareto frontier analyzer.

Reads the master tracking TSV
(`benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`) produced by
`examples/cvvdp_track_baseline.rs` and computes:

1. Pareto frontier on (bytes, score) per (corpus, metric) tuple across
   the 4 backends {B, B_GPU, C_GPU, C_CPU}. A cell is a "Pareto win" for
   backend X if X is on the frontier AND no other backend strictly
   dominates X at that (cell, metric).
2. Per-backend wall_ms p50 / p95 (Phase 5 measured cvvdp-cpu's ~10×
   wall overhead per encode; confirm at scale).
3. Per-distance summary (cvvdp's tighter target table generates larger
   files at the same distance per Phase 4's finding).
4. Cells where rankings across metrics disagree (cvvdp says A wins,
   butteraugli says B wins — these are the "metric matters" cells).
5. Decision-rule application per RFC §5.4: default-flip if cvvdp
   Pareto-dominates ≥1 corpus + within 5% on others; opt-in only
   otherwise; revert if cvvdp produces broken bitstreams.

Output:
- `scripts/cvvdp_pareto_analysis_<DATE>.tsv` — long-form per-(corpus,
  metric, backend) summary with Pareto-win counts.
- `scripts/cvvdp_pareto_analysis_<DATE>.meta` — methodology + headline
  conclusions + decision verdict.

Note: metric DIRECTION matters.
- butter_cpu / butter_gpu: SMALLER is BETTER
- ssim2: LARGER is BETTER (0–100 scale, higher = closer to reference)
- cvvdp_gpu: LARGER is BETTER (JOD 0–10, 10 = imperceptible)

For Pareto analysis, we normalize every metric to "smaller is better"
by negating ssim2/cvvdp. So a Pareto frontier minimizes (bytes,
-ssim2) or (bytes, -cvvdp).
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


METRICS = [
    ("butter_cpu", "score_butter_cpu", False),  # smaller is better
    ("butter_gpu", "score_butter_gpu", False),
    ("cvvdp_gpu", "score_cvvdp_gpu", True),  # larger is better
    ("ssim2", "score_ssim2", True),
]


@dataclass(frozen=True)
class CellKey:
    """The (image, corpus, effort, distance) tuple shared across backends."""

    image: str
    corpus: str
    effort: int
    distance: str  # keep as string to avoid f32 reparse drift


@dataclass
class Row:
    backend: str
    bytes: int
    wall_ms: float
    score_butter_cpu: Optional[float]
    score_butter_gpu: Optional[float]
    score_cvvdp_gpu: Optional[float]
    score_ssim2: Optional[float]
    notes: str


def parse_f(s: str) -> Optional[float]:
    if not s or s == "NA":
        return None
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except ValueError:
        return None


def load_tsv(path: Path) -> dict[CellKey, dict[str, Row]]:
    """Returns nested dict: cell_key -> backend -> Row."""
    out: dict[CellKey, dict[str, Row]] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for raw in reader:
            try:
                key = CellKey(
                    image=raw["image"],
                    corpus=raw["corpus"],
                    effort=int(raw["effort"]),
                    distance=raw["distance"],
                )
            except (KeyError, ValueError):
                continue
            row = Row(
                backend=raw["backend"],
                bytes=int(raw["bytes"]) if raw["bytes"].isdigit() else 0,
                wall_ms=float(raw["wall_ms"]) if raw["wall_ms"] else float("nan"),
                score_butter_cpu=parse_f(raw.get("score_butter_cpu", "")),
                score_butter_gpu=parse_f(raw.get("score_butter_gpu", "")),
                score_cvvdp_gpu=parse_f(raw.get("score_cvvdp_gpu", "")),
                score_ssim2=parse_f(raw.get("score_ssim2", "")),
                notes=raw.get("notes", ""),
            )
            out.setdefault(key, {})[raw["backend"]] = row
    return out


def metric_value(row: Row, attr: str) -> Optional[float]:
    return getattr(row, attr)


def pareto_winners(
    candidates: list[tuple[str, float, float]], larger_is_better: bool
) -> set[str]:
    """Given [(backend_name, bytes, raw_score)] return the set of backends
    on the Pareto frontier.

    "Pareto-better" means: smaller bytes AND (larger if larger_is_better
    else smaller) score, with at least one strict inequality. A backend
    is on the frontier iff no other candidate strictly dominates it.
    Ties between backends on identical (bytes, score) count both as
    on-frontier.
    """
    n = len(candidates)
    on_frontier: set[str] = set()
    for i in range(n):
        bi_name, bi_bytes, bi_score = candidates[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            bj_name, bj_bytes, bj_score = candidates[j]
            if larger_is_better:
                bytes_le = bj_bytes <= bi_bytes
                score_ge = bj_score >= bi_score
                bytes_lt = bj_bytes < bi_bytes
                score_gt = bj_score > bi_score
            else:
                bytes_le = bj_bytes <= bi_bytes
                score_ge = bj_score <= bi_score  # smaller better
                bytes_lt = bj_bytes < bi_bytes
                score_gt = bj_score < bi_score
            if bytes_le and score_ge and (bytes_lt or score_gt):
                dominated = True
                break
        if not dominated:
            on_frontier.add(bi_name)
    return on_frontier


def main(argv: list[str]) -> int:
    # Args: <tsv> [date] [candidate_backend]
    # candidate_backend: explicit verdict backend. Default behaviour:
    #   - prefer C_GPU_v4 (Phase 8f shipped) if present
    #   - else prefer Z_GPU (zensim Phase 6) if present
    #   - else fall back to C_GPU (Phase 6 baseline)
    # Phase 6 zensim arc (2026-05-25): explicit `Z_GPU` candidate is
    # supplied to compute the zensim default-flip verdict; explicit
    # `C_GPU_v4` for the cvvdp closeout.
    tsv_path = Path(argv[1] if len(argv) > 1 else "benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv")
    date = argv[2] if len(argv) > 2 else "2026-05-24"
    candidate_override = argv[3] if len(argv) > 3 else None
    suffix = f"_{candidate_override}" if candidate_override else ""
    out_tsv = Path(f"scripts/cvvdp_pareto_analysis_{date}{suffix}.tsv")
    out_meta = Path(f"scripts/cvvdp_pareto_analysis_{date}{suffix}.meta")

    cells = load_tsv(tsv_path)
    print(f"loaded {len(cells)} cells from {tsv_path}", file=sys.stderr)

    # Phase 8f (2026-05-25): added C_GPU_v4 — Phase 8c renorm + Phase 8d
    # tighten + Phase 8g k_tile_norm=0.16, the cvvdp-fork's shipped stack.
    # zensim-fork Phase 6 (2026-05-25): added Z_CPU + Z_GPU — Phase 4
    # per-distance calibration + buttloop wiring. Both Z backends share
    # identical encoder behaviour (zensim::Zensim CPU and zensim-gpu
    # GPU score f32 → produce byte-identical encoded output at fixed
    # quant field).
    backends = ["B", "B_GPU", "C_GPU", "C_CPU", "C_GPU_v4", "Z_CPU", "Z_GPU"]

    # Per-backend coverage + wall_ms stats
    per_backend_count: dict[str, int] = {b: 0 for b in backends}
    per_backend_walls: dict[str, list[float]] = {b: [] for b in backends}
    per_backend_bytes_sum: dict[str, int] = {b: 0 for b in backends}
    decoder_failures: list[tuple[CellKey, str, str]] = []  # cell, backend, note

    for key, rows in cells.items():
        for b in backends:
            if b in rows:
                per_backend_count[b] += 1
                r = rows[b]
                if not math.isnan(r.wall_ms):
                    per_backend_walls[b].append(r.wall_ms)
                per_backend_bytes_sum[b] += r.bytes
                # detect decoder failures (notes contains decode error)
                if "decode" in r.notes.lower() or "panic_during_cell" in r.notes:
                    decoder_failures.append((key, b, r.notes))

    # Pareto win counts per (corpus, metric, backend)
    win_counts: dict[tuple[str, str, str], int] = {}
    cell_counts: dict[tuple[str, str], int] = {}  # (corpus, metric) -> n cells with ≥2 backends scored
    rank_disagreements: list[tuple[CellKey, str, str, str, str]] = []  # cell, m1, m1_winner, m2, m2_winner

    # Per-distance summary: per (corpus, distance, backend), mean bytes
    per_dist_bytes: dict[tuple[str, str, str], list[int]] = {}

    for key, rows in cells.items():
        # Per-(metric, cell) Pareto frontier
        winners_by_metric: dict[str, set[str]] = {}
        for metric_name, attr, larger_better in METRICS:
            candidates = []
            for b in backends:
                if b in rows:
                    score = metric_value(rows[b], attr)
                    if score is not None:
                        candidates.append((b, rows[b].bytes, score))
            if len(candidates) < 2:
                continue
            winners = pareto_winners(candidates, larger_better)
            winners_by_metric[metric_name] = winners
            cell_counts[(key.corpus, metric_name)] = cell_counts.get((key.corpus, metric_name), 0) + 1
            for w in winners:
                win_counts[(key.corpus, metric_name, w)] = (
                    win_counts.get((key.corpus, metric_name, w), 0) + 1
                )

        # Per-distance bytes
        for b in backends:
            if b in rows:
                per_dist_bytes.setdefault(
                    (key.corpus, key.distance, b), []
                ).append(rows[b].bytes)

        # Rank disagreements: butter_cpu winner vs cvvdp_gpu winner
        if "butter_cpu" in winners_by_metric and "cvvdp_gpu" in winners_by_metric:
            bw = winners_by_metric["butter_cpu"]
            cw = winners_by_metric["cvvdp_gpu"]
            # We care when the winner SETS are disjoint (true disagreement,
            # not just different sizes of overlapping frontiers).
            if bw and cw and not (bw & cw):
                rank_disagreements.append(
                    (key, "butter_cpu", ",".join(sorted(bw)), "cvvdp_gpu", ",".join(sorted(cw)))
                )

    # Write TSV summary
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w") as f:
        f.write(
            "corpus\tmetric\tbackend\tcells_scored\tpareto_wins\tpareto_win_pct\n"
        )
        for (corpus, metric), total in sorted(cell_counts.items()):
            for b in backends:
                wins = win_counts.get((corpus, metric, b), 0)
                pct = (100.0 * wins / total) if total > 0 else 0.0
                f.write(f"{corpus}\t{metric}\t{b}\t{total}\t{wins}\t{pct:.2f}\n")

        # Append per-backend wall_ms + bytes summary
        f.write("\n# Per-backend coverage + perf\n")
        f.write("backend\tcells_populated\twall_p50_ms\twall_p95_ms\twall_mean_ms\ttotal_bytes\tavg_bytes\n")
        for b in backends:
            ws = sorted(per_backend_walls[b])
            count = per_backend_count[b]
            if ws:
                p50 = ws[len(ws) // 2]
                p95_idx = max(0, int(0.95 * (len(ws) - 1)))
                p95 = ws[p95_idx]
                mean = sum(ws) / len(ws)
            else:
                p50 = p95 = mean = float("nan")
            total_b = per_backend_bytes_sum[b]
            avg_b = (total_b / count) if count else 0.0
            f.write(
                f"{b}\t{count}\t{p50:.2f}\t{p95:.2f}\t{mean:.2f}\t{total_b}\t{avg_b:.1f}\n"
            )

        # Per-distance bytes mean
        f.write("\n# Per-distance per-backend mean bytes (cvvdp tighter target => larger files)\n")
        f.write("corpus\tdistance\tbackend\tn\tmean_bytes\n")
        for (corpus, dist, b), arr in sorted(per_dist_bytes.items()):
            f.write(f"{corpus}\t{dist}\t{b}\t{len(arr)}\t{sum(arr) / max(1, len(arr)):.1f}\n")

        # Rank disagreements
        f.write("\n# Rank disagreements: cells where butter_cpu winner != cvvdp_gpu winner\n")
        f.write("image\tcorpus\teffort\tdistance\tbutter_cpu_winners\tcvvdp_gpu_winners\n")
        for key, _, bw, _, cw in rank_disagreements[:200]:  # cap at 200 for readability
            f.write(f"{key.image}\t{key.corpus}\t{key.effort}\t{key.distance}\t{bw}\t{cw}\n")

    print(f"wrote {out_tsv}", file=sys.stderr)

    # ---- Decision rule application ----
    # Per RFC §5.4: default-flip if cvvdp Pareto-dominates ≥1 corpus +
    # within 5% on others.
    #
    # Operational interpretation: per (corpus, metric), pick the
    # backend with the most Pareto wins. If the candidate cvvdp variant
    # dominates ≥1 corpus on ≥1 metric and is within 5% of leader on
    # every other (corpus, metric), recommend default-flip.
    #
    # Phase 8f (2026-05-25): the verdict candidate is auto-selected.
    # If `C_GPU_v4` rows exist in the input (Phase 8f's shipped stack),
    # the verdict applies to C_GPU_v4. Otherwise falls back to `C_GPU`
    # (Phase 6 baseline) for backwards-compatible output.

    corpora = sorted({c for (c, _) in cell_counts.keys()})

    # Verdict selector. Explicit override wins, otherwise auto-select.
    # zensim Phase 6 (2026-05-25): added Z_GPU as a third candidate.
    if candidate_override is not None:
        candidate = candidate_override
    elif per_backend_count.get("C_GPU_v4", 0) > 0:
        candidate = "C_GPU_v4"
    elif per_backend_count.get("Z_GPU", 0) > 0:
        candidate = "Z_GPU"
    else:
        candidate = "C_GPU"
    print(f"[verdict] candidate backend = {candidate}", file=sys.stderr)

    # Build per-(corpus, metric) winner-of-winners
    leaders: dict[tuple[str, str], tuple[str, float]] = {}
    cgpu_pct: dict[tuple[str, str], float] = {}
    for (corpus, metric), total in cell_counts.items():
        if total == 0:
            continue
        best_b = ""
        best_pct = -1.0
        for b in backends:
            w = win_counts.get((corpus, metric, b), 0)
            pct = 100.0 * w / total
            if pct > best_pct:
                best_pct = pct
                best_b = b
        leaders[(corpus, metric)] = (best_b, best_pct)
        cgpu_pct[(corpus, metric)] = 100.0 * win_counts.get((corpus, metric, candidate), 0) / total

    cgpu_dominates_any = False
    cgpu_within_5_everywhere = True
    cgpu_dominates_corpora: list[str] = []
    cgpu_weak_cells: list[tuple[str, str]] = []

    for (corpus, metric), (leader_b, leader_pct) in leaders.items():
        cgpu_p = cgpu_pct.get((corpus, metric), 0.0)
        if leader_b == candidate and leader_pct >= 50.0:
            cgpu_dominates_any = True
            if corpus not in cgpu_dominates_corpora:
                cgpu_dominates_corpora.append(corpus)
        # within 5%: cgpu_pct >= leader_pct - 5
        if cgpu_p < leader_pct - 5.0:
            cgpu_within_5_everywhere = False
            cgpu_weak_cells.append((corpus, metric))

    verdict: str
    rationale: list[str]
    if decoder_failures:
        verdict = "REVERT"
        rationale = [
            f"{len(decoder_failures)} decoder failure cells detected.",
            "RFC §5.4 hard rule: any cell that produces output failing decode under any decoder triggers a revert.",
            "See decoder_failures table in meta.",
        ]
    elif cgpu_dominates_any and cgpu_within_5_everywhere:
        verdict = "DEFAULT_FLIP"
        rationale = [
            f"{candidate} Pareto-dominates on corpora: {', '.join(cgpu_dominates_corpora)}",
            f"{candidate} is within 5% of leader on every other (corpus, metric).",
            "RFC §5.4 default-flip rule met.",
        ]
    else:
        verdict = "OPT_IN_ONLY"
        reasons: list[str] = []
        if not cgpu_dominates_any:
            reasons.append(
                f"{candidate} does not Pareto-dominate ≥50% of cells on any (corpus, metric)."
            )
        if not cgpu_within_5_everywhere:
            reasons.append(
                f"{candidate} is >5pp below leader on {len(cgpu_weak_cells)} (corpus, metric) cells."
            )
        rationale = reasons + [
            "RFC §5.4 default-flip rule NOT met; ship as opt-in.",
        ]

    # Write meta
    with out_meta.open("w") as f:
        f.write("Phase 6 Pareto analysis — cvvdp fork decision rule application\n")
        f.write(f"input: {tsv_path}\n")
        f.write(f"date: {date}\n\n")

        f.write("## Coverage summary\n\n")
        for b in backends:
            f.write(f"- {b}: {per_backend_count[b]} cells populated\n")
        f.write("\n")

        f.write(f"## Per-(corpus, metric) leader + {candidate} placement\n\n")
        f.write(f"corpus | metric | leader | leader_pct | {candidate}_pct | within_5pp?\n")
        f.write("--- | --- | --- | --- | --- | ---\n")
        for (corpus, metric), (leader_b, leader_pct) in sorted(leaders.items()):
            cgp = cgpu_pct.get((corpus, metric), 0.0)
            within = "Y" if cgp >= leader_pct - 5.0 else "N"
            f.write(f"{corpus} | {metric} | {leader_b} | {leader_pct:.1f}% | {cgp:.1f}% | {within}\n")
        f.write("\n")

        f.write("## Wall-time comparison\n\n")
        f.write("backend | n | p50 ms | p95 ms | mean ms | avg bytes\n")
        f.write("--- | --- | --- | --- | --- | ---\n")
        for b in backends:
            ws = sorted(per_backend_walls[b])
            count = per_backend_count[b]
            if ws:
                p50 = ws[len(ws) // 2]
                p95_idx = max(0, int(0.95 * (len(ws) - 1)))
                p95 = ws[p95_idx]
                mean = sum(ws) / len(ws)
            else:
                p50 = p95 = mean = float("nan")
            avg_b = per_backend_bytes_sum[b] / count if count else 0.0
            f.write(f"{b} | {count} | {p50:.1f} | {p95:.1f} | {mean:.1f} | {avg_b:.0f}\n")
        f.write("\n")

        if decoder_failures:
            f.write("## Decoder failures (HARD REVERT GATE)\n\n")
            f.write("image | corpus | effort | distance | backend | note\n")
            f.write("--- | --- | --- | --- | --- | ---\n")
            for key, b, note in decoder_failures[:50]:
                f.write(f"{key.image} | {key.corpus} | {key.effort} | {key.distance} | {b} | {note}\n")
            if len(decoder_failures) > 50:
                f.write(f"... ({len(decoder_failures) - 50} more)\n")
            f.write("\n")

        f.write(f"## VERDICT: {verdict}\n\n")
        for line in rationale:
            f.write(f"- {line}\n")
        f.write("\n")

        f.write("## Methodology notes\n\n")
        f.write("- Pareto frontier minimizes (bytes, score) for `smaller-is-better` metrics\n")
        f.write("  (butter_cpu, butter_gpu) and minimizes (bytes, -score) for larger-is-better\n")
        f.write("  metrics (cvvdp_gpu, ssim2).\n")
        f.write("- A backend is on the frontier iff no other backend strictly dominates it.\n")
        f.write("  Ties on identical (bytes, score) count both as on-frontier.\n")
        f.write("- 'Pareto wins' counts only cells with ≥2 backends scored on that metric.\n")
        f.write("- 'Within 5pp' means C_GPU's Pareto-win-percentage >= leader's - 5.\n")
        f.write("- Decoder failures are detected from the `notes` column substring match\n")
        f.write("  (`decode` or `panic_during_cell`).\n")

    print(f"wrote {out_meta}", file=sys.stderr)
    print(f"VERDICT: {verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
