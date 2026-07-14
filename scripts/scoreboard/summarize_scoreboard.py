#!/usr/bin/env python3
"""Roll a scoreboard TSV (run_scoreboard.py output) into the goal table,
with a strict multi-metric Pareto CALIBRATION of the cjxl-dominant cells.

The raw `verdict` column is an honest bytes+quality verdict, but "cjxl
dominates" lumps three very different situations together: a 50 %-bigger real
gap, a 0.3 %-bigger near-tie, and a cell where we deliberately spent bytes to
land strictly better quality. Treating all three as "losses owing a wedge"
inflated the June headline (112 cjxl-dominant was really 45 real + 67
mislabel — benchmarks/scoreboard/reclassify_multimetric_2026-07-14.md,
ledger #26). This summarizer applies that same lens automatically so every
run self-calibrates instead of needing a one-off reclassify script.

Per cjxl-dominant / mixed cell it computes, from the committed metric values,
one calibrated bucket:

  REAL_LOSS  cjxl is not-worse on EVERY quality metric AND meaningfully
             smaller (>=2 % bytes, or any lossless byte gap) — a genuine gap.
  TRADEOFF   ours is strictly better on >=1 quality metric while bigger —
             we bought quality; not a loss.
  NEAR_TIE   |bytes| < 2 % and no quality metric meaningfully worse — noise.

Quality direction is read per `quality_kind` so SDR (bfly lower-better +
ssim2 higher-better), the two-metric HDR guard (pq_bfly + vdp2, both
lower-better), legacy single-metric HDR (pq_bfly only), and lossless
(pixel-exact both sides) are all handled correctly.

Usage: summarize_scoreboard.py <scoreboard.tsv> [out.md]
"""

import csv
import sys
from collections import Counter, defaultdict

# Noise floors — MUST match run_scoreboard.py's tie bands.
BFLY_REL = 0.02      # butteraugli / pq_butteraugli / vdp2 relative tie band
SSIM_ABS = 0.25      # ssim2 absolute tie band
CVVDP_ABS = 0.1      # cvvdp JOD absolute tie band (higher = better)
BYTES_NEAR = 2.0     # |bytes delta %| below this (with no quality loss) = near-tie


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def rel_dir(ours, cjxl, lower_better, rel=BFLY_REL):
    """'OURS' / 'CJXL' / 'TIE' for a relative-tie-band metric."""
    if ours is None or cjxl is None:
        return "TIE"
    band = abs(cjxl) * rel
    if abs(ours - cjxl) <= band:
        return "TIE"
    better = ours < cjxl if lower_better else ours > cjxl
    return "OURS" if better else "CJXL"


def abs_dir(ours, cjxl, higher_better, band):
    """'OURS' / 'CJXL' / 'TIE' for an absolute-tie-band metric (ssim2, cvvdp)."""
    if ours is None or cjxl is None:
        return "TIE"
    if abs(ours - cjxl) <= band:
        return "TIE"
    better = ours > cjxl if higher_better else ours < cjxl
    return "OURS" if better else "CJXL"


def quality_dirs(r):
    """List of per-metric quality directions for a cell, per its quality_kind.
    Empty list => quality axis carries no signal (e.g. lossless: exact both
    sides, or an ERROR cell)."""
    kind = r.get("quality_kind", "")
    q1o, q1c = num(r["ours_q1"]), num(r["cjxl_q1"])
    q2o, q2c = num(r["ours_q2"]), num(r["cjxl_q2"])
    q3o, q3c = num(r.get("ours_q3", "")), num(r.get("cjxl_q3", ""))
    if kind == "bfly_pnorm3+ssim2":
        return [rel_dir(q1o, q1c, True), abs_dir(q2o, q2c, True, SSIM_ABS)]
    if kind == "pq_bfly+vdp2+cvvdp":  # 3-metric HDR: bfly+vdp2 lower, cvvdp higher
        return [rel_dir(q1o, q1c, True), rel_dir(q2o, q2c, True),
                abs_dir(q3o, q3c, True, CVVDP_ABS)]
    if kind == "pq_bfly+vdp2":  # two-metric HDR: both lower = better
        return [rel_dir(q1o, q1c, True), rel_dir(q2o, q2c, True)]
    if kind == "pq_bfly":       # legacy single-metric HDR
        return [rel_dir(q1o, q1c, True)]
    # pixel_exact (lossless) / unknown: no quality signal
    return []


def qstr(r):
    """Human-readable 'ours vs cjxl' quality string, including q2/q3 when set."""
    s = f"{r['ours_q1']} vs {r['cjxl_q1']}"
    if r.get("ours_q2"):
        s += f" / q2 {r['ours_q2']} vs {r['cjxl_q2']}"
    if r.get("ours_q3"):
        s += f" / q3 {r['ours_q3']} vs {r['cjxl_q3']}"
    return s


def calibrate(r):
    """REAL_LOSS / TRADEOFF / NEAR_TIE for a cjxl-dominant or mixed cell."""
    dirs = quality_dirs(r)
    bytes_delta = num(r["bytes_delta_pct"]) or 0.0
    ours_bigger = bytes_delta > 0
    any_quality_ours = any(d == "OURS" for d in dirs)
    any_quality_cjxl = any(d == "CJXL" for d in dirs)

    # We bought quality: strictly better on a quality metric while bigger.
    if any_quality_ours and ours_bigger and not any_quality_cjxl:
        return "TRADEOFF"
    # Lossless (no quality dirs, pixels exact both sides): pure byte gap.
    if not dirs:
        return "NEAR_TIE" if abs(bytes_delta) < BYTES_NEAR else "REAL_LOSS"
    # Genuine gap vs near-tie.
    if abs(bytes_delta) < BYTES_NEAR and not any_quality_cjxl:
        return "NEAR_TIE"
    return "REAL_LOSS"


def main():
    rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
    out = open(sys.argv[2], "w") if len(sys.argv) > 2 else sys.stdout
    w = out.write

    total = Counter(r["verdict"] for r in rows)
    fam = defaultdict(Counter)
    for r in rows:
        fam[r["family"].split("/")[0]][r["verdict"]] += 1

    w(f"# Scoreboard rollup — {sys.argv[1]}\n\n")
    w("Axes: BYTES + QUALITY only (wall axis UNMEASURED in v1 — quiet-box "
      "zenbench grid pending). Verdicts are bytes+quality verdicts.\n\n")
    n = len(rows)
    w(f"**{n} cells** — ")
    w(", ".join(f"{k}: {v} ({v / n:.0%})" for k, v in sorted(total.items(),
      key=lambda kv: -kv[1])) + "\n\n")

    w("| family | WE-DOMINATE | TIE | MIXED | CJXL-DOMINATES | ERROR |\n")
    w("|---|---|---|---|---|---|\n")
    for f_, c in sorted(fam.items()):
        w(f"| {f_} | {c.get('WE-DOMINATE', 0)} | {c.get('TIE', 0)} | "
          f"{c.get('MIXED', 0)} | {c.get('CJXL-DOMINATES', 0)} | "
          f"{c.get('ERROR', 0)} |\n")

    # ── Calibrated view: split the pure CJXL-DOMINATES cells (cjxl not-worse
    # on both axes, strictly better on >=1) into real gaps vs mislabels. MIXED
    # cells are tradeoffs by construction (we won one axis) — counted, not
    # calibrated. Philosophy: strict bytes+quality Pareto — a dominant cell is
    # a REAL_LOSS unless it's a byte near-tie (<2 %) or we are strictly better
    # on a quality metric (TRADEOFF). A quality-TIE with meaningfully fewer
    # cjxl bytes IS a real byte loss (same quality, fewer bytes). For legacy
    # single-metric HDR this is stricter than the one-off reclassify's HDR
    # leniency, because one HDR metric can't distinguish a true tie from its
    # own blind spot; the shipped two-metric HDR guard (pq_bfly+vdp2) makes
    # the tie trustworthy going forward.
    dominant = [r for r in rows if r["verdict"] == "CJXL-DOMINATES"]
    n_mixed = total.get("MIXED", 0)
    cal = defaultdict(list)
    fam_real = Counter()
    for r in dominant:
        b = calibrate(r)
        cal[b].append(r)
        if b == "REAL_LOSS":
            fam_real[r["family"].split("/")[0]] += 1
    real = cal["REAL_LOSS"]
    mislabel = cal["TRADEOFF"] + cal["NEAR_TIE"]

    w("\n## Calibrated (strict multi-metric Pareto)\n\n")
    w(f"Of {len(dominant)} CJXL-DOMINATES cells: **{len(real)} REAL_LOSS**, "
      f"{len(cal['TRADEOFF'])} TRADEOFF (bought quality), "
      f"{len(cal['NEAR_TIE'])} NEAR_TIE (noise) — "
      f"{len(mislabel)} mislabel. Plus {n_mixed} MIXED cells "
      f"(won one axis — tradeoffs, not gaps).\n\n")
    if fam_real:
        w("REAL_LOSS by family:\n\n")
        w("| family | real losses | of dominant |\n|---|---|---|\n")
        fam_dom = Counter(r["family"].split("/")[0] for r in dominant)
        for f_ in sorted(fam_real, key=lambda k: -fam_real[k]):
            w(f"| {f_} | {fam_real[f_]} | {fam_dom[f_]} |\n")

    if real:
        w(f"\n## REAL gaps owing a wedge ({len(real)})\n\n")
        w("| cell | verdict | bytes Δ% | quality (ours vs cjxl) | kind | flags |\n")
        w("|---|---|---|---|---|---|\n")
        for r in sorted(real, key=lambda r: -abs(num(r["bytes_delta_pct"]) or 0)):
            w(f"| {r['cell']} | {r['verdict']} | {r['bytes_delta_pct']} | {qstr(r)} | "
              f"{r.get('quality_kind', '')} | {r['flags']} |\n")
    else:
        w("\n**Zero REAL losses after calibration — goal floor holds on these axes.**\n")

    if mislabel:
        w(f"\n## Mislabeled (NOT wedges — tradeoff/near-tie) ({len(mislabel)})\n\n")
        w("| cell | bucket | bytes Δ% | quality (ours vs cjxl) | kind |\n")
        w("|---|---|---|---|---|\n")
        for r in cal["TRADEOFF"] + cal["NEAR_TIE"]:
            bucket = "TRADEOFF" if r in cal["TRADEOFF"] else "NEAR_TIE"
            w(f"| {r['cell']} | {bucket} | {r['bytes_delta_pct']} | {qstr(r)} | "
              f"{r.get('quality_kind', '')} |\n")


if __name__ == "__main__":
    main()
