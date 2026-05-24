#!/usr/bin/env python3
"""Build markdown summary tables from the W44-AUDIT-1 fresh
cjxl-parity bench TSV.

Reads:  benchmarks/cjxl_parity_2026-05-24_post_w44_205_s2_refit_c2.tsv
Writes: benchmarks/cjxl_parity_2026-05-24_post_w44_205_s2_refit_c2.md
        (Tables A / B / C / D + 200-word verdict appended by hand)

Table A: zenjxl-vs-cjxl bytes (% delta, negative = smaller / better)
Table B: zenjxl-vs-cjxl SSIM2 (absolute, positive = better)
Table C: zenjxl-vs-cjxl butteraugli (absolute, negative = better)
Table D: aggregate row with column means

Rows: image × distance, Cols: e5/e7/e9 for Zenjxl strategy and Libjxl strategy.
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict


def fmt(v, fmt_str):
    try:
        return fmt_str.format(float(v))
    except (ValueError, TypeError):
        return "n/a"


def main():
    tsv_path = Path(sys.argv[1] if len(sys.argv) > 1 else
                    "benchmarks/cjxl_parity_2026-05-24_post_w44_205_s2_refit_c2.tsv")
    out_path = tsv_path.with_suffix(".md")

    rows = []
    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)

    if not rows:
        print(f"ERROR: no rows in {tsv_path}", file=sys.stderr)
        sys.exit(1)

    # Sort canonically by class, image_id, distance
    class_order = {"SCREEN": 0, "PHOTO_SMOOTH": 1, "PHOTO": 2}
    rows.sort(key=lambda r: (class_order.get(r["class"], 9), r["image_id"], float(r["distance"])))

    # Build per-(image, distance) → per-effort metric dict
    cells = defaultdict(lambda: {})
    for r in rows:
        key = (r["class"], r["image_id"], float(r["distance"]))
        cells[key][int(r["effort"])] = r

    efforts = [5, 7, 9]

    def write_table(title, key_zen, key_libjxl, fmt_str, header_unit):
        lines = []
        lines.append(f"### {title}\n")
        lines.append(f"_{header_unit}_\n")
        hdr = ["class", "image", "distance"]
        for e in efforts:
            hdr.append(f"e{e}_zen")
        for e in efforts:
            hdr.append(f"e{e}_libjxlstrat")
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("| " + " | ".join(["---"] * len(hdr)) + " |")
        col_sums = [0.0] * (len(hdr) - 3)
        col_counts = [0] * (len(hdr) - 3)
        for (klass, img, d), per_effort in sorted(cells.items(),
                                                   key=lambda kv: (class_order.get(kv[0][0], 9), kv[0][1], kv[0][2])):
            row = [klass, img, f"d={d}"]
            ci = 0
            for e in efforts:
                if e in per_effort:
                    v = per_effort[e][key_zen]
                    row.append(fmt(v, fmt_str))
                    try:
                        col_sums[ci] += float(v); col_counts[ci] += 1
                    except Exception: pass
                else:
                    row.append("—")
                ci += 1
            for e in efforts:
                if e in per_effort:
                    v = per_effort[e][key_libjxl]
                    row.append(fmt(v, fmt_str))
                    try:
                        col_sums[ci] += float(v); col_counts[ci] += 1
                    except Exception: pass
                else:
                    row.append("—")
                ci += 1
            lines.append("| " + " | ".join(row) + " |")
        # Aggregate row
        agg_row = ["**MEAN**", "", ""]
        for s, c in zip(col_sums, col_counts):
            agg_row.append(fmt_str.format(s / c) if c > 0 else "—")
        lines.append("| " + " | ".join(agg_row) + " |")
        return "\n".join(lines)

    table_a = write_table(
        "Table A: bytes delta vs cjxl (% — negative = our file is SMALLER)",
        "zenjxl_dBytes_pct", "libjxl_dBytes_pct", "{:+.2f}%", "negative = smaller / better"
    )
    table_b = write_table(
        "Table B: SSIM2 delta vs cjxl (absolute — positive = better)",
        "zenjxl_dSsim2", "libjxl_dSsim2", "{:+.2f}", "positive = better quality"
    )
    table_c = write_table(
        "Table C: butteraugli delta vs cjxl (absolute — negative = better)",
        "zenjxl_dBfly", "libjxl_dBfly", "{:+.3f}", "negative = better quality"
    )

    # Aggregate Table D (winrates)
    n_cells = 0
    zen_bytes_wins = 0   # zenjxl bytes <= cjxl bytes
    zen_ssim2_wins = 0   # zenjxl ssim2 >= cjxl ssim2
    zen_double_wins = 0  # both
    zen_pareto_loses = 0 # bytes > cjxl by > 2% AND ssim2 < cjxl by > 0.5
    libjxl_byteid = 0    # libjxl_strat bytes == zenjxl bytes (would imply Libjxl strategy is byte-identical)
    sum_z_dbytes = 0.0
    sum_z_dssim2 = 0.0
    sum_z_dbfly = 0.0
    for r in rows:
        n_cells += 1
        zdb = float(r["zenjxl_dBytes_pct"])
        zdss = float(r["zenjxl_dSsim2"])
        zdbf = float(r["zenjxl_dBfly"])
        sum_z_dbytes += zdb; sum_z_dssim2 += zdss; sum_z_dbfly += zdbf
        if zdb <= 0:
            zen_bytes_wins += 1
        if zdss >= 0:
            zen_ssim2_wins += 1
        if zdb <= 0 and zdss >= 0:
            zen_double_wins += 1
        if zdb > 2.0 and zdss < -0.5:
            zen_pareto_loses += 1
    mean_z_dbytes = sum_z_dbytes / n_cells
    mean_z_dssim2 = sum_z_dssim2 / n_cells
    mean_z_dbfly = sum_z_dbfly / n_cells

    table_d = f"""### Table D: aggregate summary

| metric | value |
| --- | --- |
| cells benched | {n_cells} |
| mean zenjxl-vs-cjxl bytes delta | {mean_z_dbytes:+.2f}% |
| mean zenjxl-vs-cjxl SSIM2 delta | {mean_z_dssim2:+.2f} |
| mean zenjxl-vs-cjxl butteraugli delta | {mean_z_dbfly:+.3f} |
| cells with zenjxl bytes ≤ cjxl | {zen_bytes_wins} / {n_cells} ({100.0*zen_bytes_wins/n_cells:.0f}%) |
| cells with zenjxl SSIM2 ≥ cjxl | {zen_ssim2_wins} / {n_cells} ({100.0*zen_ssim2_wins/n_cells:.0f}%) |
| cells with zenjxl Pareto-dominant (both bytes ≤ AND SSIM2 ≥) | {zen_double_wins} / {n_cells} ({100.0*zen_double_wins/n_cells:.0f}%) |
| cells where zenjxl Pareto-loses (bytes > +2% AND SSIM2 < -0.5) | {zen_pareto_loses} / {n_cells} ({100.0*zen_pareto_loses/n_cells:.0f}%) |
"""

    # Write the .md
    out_lines = []
    out_lines.append("# cjxl-parity bench 2026-05-24 (post W44-205 + S2-refit-c2)\n")
    out_lines.append("Source TSV: `{}`\n".format(tsv_path.name))
    out_lines.append("Methodology: in-process Rust encode (jxl-encoder library), cjxl v0.12.0 reference, jxl-oxide srgb_linear decode, butteraugli + fast-ssim2 metrics. See CLAUDE.md \"CRITICAL: PNG Color Metadata\" — this harness is metadata-immune.\n")
    out_lines.append("Cell matrix: 4 images × 3 efforts {e5, e7, e9} × 3 distances {0.5, 2.0, 4.0} × {zenjxl, EncoderStrategy::Libjxl} = 72 zenjxl + 36 cjxl encodes.\n")
    out_lines.append("Images: codec_wiki (SCREEN), 1418519 + 1025469 (PHOTO), 1531677 (PHOTO_SMOOTH).\n")
    out_lines.append("---\n")
    out_lines.append(table_a + "\n")
    out_lines.append("---\n")
    out_lines.append(table_b + "\n")
    out_lines.append("---\n")
    out_lines.append(table_c + "\n")
    out_lines.append("---\n")
    out_lines.append(table_d + "\n")

    out_path.write_text("\n".join(out_lines))
    print(f"[md] wrote {out_path}")


if __name__ == "__main__":
    main()
