#!/usr/bin/env python3
"""Build the cvvdp_target_jod → distance calibration table.

Phase 8b of the cvvdp fork. Reads Phase 6's tracking TSV, extracts
butteraugli-driven (distance, achieved_cvvdp) curves per corpus, and
emits a Rust lookup table for `vardct/cvvdp_distance_lookup.rs`.

The table answers: "If the user wants achieved cvvdp_jod ≈ Y, what
encoder `distance` parameter should the butteraugli buttloop run at
to land there with Pareto-optimal bytes?"

Per Phase 8a diagnosis (scripts/cvvdp_pareto_diagnosis.py), butteraugli
sits on the (bytes, cvvdp) Pareto front 94% of cells. So driving the
encode with butteraugli at distance D(target_jod) gives ≈ Pareto-
optimal bytes for that achieved cvvdp.

Output:
  - scripts/cvvdp_calibration_table_<DATE>.tsv      (per-corpus medians)
  - jxl-encoder/src/vardct/cvvdp_distance_lookup.rs.snippet (Rust source
    that Phase 8c will fold into the actual lookup module)
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
    cvvdp_gpu: float | None


def load(path: str) -> list[Row]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                cv = r["score_cvvdp_gpu"]
                cv_v = float(cv) if cv not in ("", "NA") else None
            except (ValueError, KeyError):
                cv_v = None
            try:
                bs = int(r["bytes"]) if r["bytes"] not in ("", "NA") else 0
            except ValueError:
                bs = 0
            rows.append(
                Row(
                    image=r["image"],
                    corpus=r["corpus"],
                    effort=int(r["effort"]),
                    distance=float(r["distance"]),
                    backend=r["backend"],
                    bytes=bs,
                    cvvdp_gpu=cv_v,
                )
            )
    return rows


def main(path: str, date: str = "2026-05-24") -> int:
    rows = load(path)

    # For each (image, corpus), build butteraugli's (distance, cvvdp) curve at e=8.
    # Other efforts also work but e=8 is where the buttloop fires; lower efforts
    # produce identical output across backends so calibration is degenerate.
    per_image_curves: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if r.backend != "B" or r.effort != 8 or r.cvvdp_gpu is None:
            continue
        per_image_curves[(r.image, r.corpus)].append((r.distance, r.cvvdp_gpu))

    # Invert each curve: cvvdp → distance via linear interpolation.
    target_jods = [9.50, 9.60, 9.70, 9.80, 9.85, 9.90, 9.93, 9.95, 9.97, 9.98, 9.99]

    def cvvdp_to_distance(curve: list[tuple[float, float]], target_jod: float) -> float | None:
        """Curve = list of (distance, cvvdp). cvvdp DECREASES as distance INCREASES.
        Return the distance where cvvdp ≈ target_jod, via linear interp.
        Returns None if outside data range."""
        if not curve:
            return None
        # Sort by distance ascending → cvvdp generally descending.
        sc = sorted(curve)
        cvvdps = [cv for _, cv in sc]
        distances = [d for d, _ in sc]
        # If target is outside range, can't extrapolate.
        if target_jod > cvvdps[0] or target_jod < cvvdps[-1]:
            return None
        # Find the bracket where cvvdp[i] >= target >= cvvdp[i+1].
        for i in range(len(sc) - 1):
            cv_lo = cvvdps[i + 1]  # smaller cvvdp = larger distance
            cv_hi = cvvdps[i]
            d_lo = distances[i + 1]
            d_hi = distances[i]
            if cv_lo <= target_jod <= cv_hi:
                if cv_hi == cv_lo:
                    return d_hi
                # Linear interp on cvvdp scale.
                t = (cv_hi - target_jod) / (cv_hi - cv_lo)
                return d_hi + t * (d_lo - d_hi)
        return None

    # For each corpus + target_jod, collect distance estimates across images;
    # take median + p25 + p75.
    by_corpus: dict[str, set[str]] = defaultdict(set)
    for (image, corpus), _ in per_image_curves.items():
        by_corpus[corpus].add(image)
    corpora = sorted(by_corpus.keys())

    # Output TSV
    tsv_lines = ["corpus\ttarget_cvvdp\tn\tmedian_distance\tp25_distance\tp75_distance"]
    table: dict[str, dict[float, float]] = defaultdict(dict)
    for corpus in corpora:
        for target_jod in target_jods:
            estimates = []
            for image in by_corpus[corpus]:
                curve = per_image_curves.get((image, corpus), [])
                d = cvvdp_to_distance(curve, target_jod)
                if d is not None:
                    estimates.append(d)
            n = len(estimates)
            if n == 0:
                continue
            estimates.sort()
            median = statistics.median(estimates)
            p25 = estimates[n // 4]
            p75 = estimates[3 * n // 4]
            table[corpus][target_jod] = median
            tsv_lines.append(
                f"{corpus}\t{target_jod:.3f}\t{n}\t{median:.4f}\t{p25:.4f}\t{p75:.4f}"
            )

    # Also: blended-default table (median across corpora) for unclassified input.
    for target_jod in target_jods:
        per_corpus_medians = [table[c].get(target_jod) for c in corpora if target_jod in table[c]]
        per_corpus_medians = [m for m in per_corpus_medians if m is not None]
        if not per_corpus_medians:
            continue
        blend = statistics.median(per_corpus_medians)
        table["_DEFAULT"][target_jod] = blend
        tsv_lines.append(
            f"_DEFAULT\t{target_jod:.3f}\t{len(per_corpus_medians)}\t{blend:.4f}\tNA\tNA"
        )

    tsv_path = f"scripts/cvvdp_calibration_table_{date}.tsv"
    with open(tsv_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tsv_lines) + "\n")
    print(f"Wrote {tsv_path}", file=sys.stderr)

    # Output Rust snippet
    rs_lines = []
    rs_lines.append("// SPDX-License-Identifier: AGPL-3.0-or-later")
    rs_lines.append("// Algorithms and constants derived from libjxl (BSD-3-Clause).")
    rs_lines.append("// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing")
    rs_lines.append("")
    rs_lines.append("//! cvvdp target-JOD → effective-distance calibration table (Phase 8b).")
    rs_lines.append("//!")
    rs_lines.append("//! Built from Phase 6 tracking TSV:")
    rs_lines.append("//!   benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv")
    rs_lines.append("//!")
    rs_lines.append("//! For each (corpus, target_jod), the table records the MEDIAN distance")
    rs_lines.append("//! at which butteraugli's buttloop achieves cvvdp_gpu ≈ target_jod on that")
    rs_lines.append("//! corpus, across all images at effort=8.")
    rs_lines.append("//!")
    rs_lines.append("//! Use case: `LossyConfig::with_cvvdp_target_jod(jod)` looks up the")
    rs_lines.append("//! distance that achieves `jod`, then runs the butteraugli buttloop at")
    rs_lines.append("//! that distance. Output sits on the Pareto front per Phase 8a diagnosis.")
    rs_lines.append("//!")
    rs_lines.append("//! Linear interpolation between table points; clamp outside [min, max].")
    rs_lines.append("")
    rs_lines.append("/// Phase 6 commit SHA the table was extracted from.")
    rs_lines.append('pub(crate) const CALIBRATION_SOURCE: &str = "Phase 6 sweep, cvvdp-fork-rfc 8b5a13a7";')
    rs_lines.append("")

    def emit_table(name: str, label: str, entries: dict[float, float]):
        rs_lines.append(f"/// {label}")
        sorted_entries = sorted(entries.items())
        rs_lines.append(f"pub(crate) static {name}: &[(f32, f32)] = &[")
        for jod, dist in sorted_entries:
            rs_lines.append(f"    ({jod:.3f}, {dist:.4f}),")
        rs_lines.append("];")
        rs_lines.append("")

    emit_table(
        "CVVDP_DISTANCE_LOOKUP_DEFAULT",
        "Default (blended across all corpora). Lookup: target_jod → distance.",
        table.get("_DEFAULT", {}),
    )
    if "CID22" in table:
        emit_table(
            "CVVDP_DISTANCE_LOOKUP_CID22",
            "CID22 photos. Lookup: target_jod → distance.",
            table["CID22"],
        )
    if "GB82-SC" in table:
        emit_table(
            "CVVDP_DISTANCE_LOOKUP_GB82_SC",
            "GB82-SC screenshots. Lookup: target_jod → distance.",
            table["GB82-SC"],
        )
    if "W44-S1" in table:
        emit_table(
            "CVVDP_DISTANCE_LOOKUP_W44_S1",
            "W44-S1 extras (baby-lossless, bulb-lossless). Lookup: target_jod → distance.",
            table["W44-S1"],
        )

    rs_lines.append("")
    rs_lines.append("/// Linear interpolation: given a target JOD, return the distance from")
    rs_lines.append("/// the lookup table. Clamps outside the table range. Returns the table's")
    rs_lines.append("/// max-distance entry if `target_jod` is below the table's min cvvdp;")
    rs_lines.append("/// returns the table's min-distance entry if `target_jod` is above the max.")
    rs_lines.append("pub(crate) fn lookup_distance_for_target_jod(table: &[(f32, f32)], target_jod: f32) -> f32 {")
    rs_lines.append("    if table.is_empty() {")
    rs_lines.append("        return 1.0;  // safe default")
    rs_lines.append("    }")
    rs_lines.append("    let first = table[0];")
    rs_lines.append("    let last = table[table.len() - 1];")
    rs_lines.append("    if target_jod <= first.0 {")
    rs_lines.append("        return last.1;  // very lossy target → highest distance in table")
    rs_lines.append("    }")
    rs_lines.append("    if target_jod >= last.0 {")
    rs_lines.append("        return first.1;  // very strict target → lowest distance in table")
    rs_lines.append("    }")
    rs_lines.append("    for w in table.windows(2) {")
    rs_lines.append("        let (jod_lo, d_lo) = w[0];")
    rs_lines.append("        let (jod_hi, d_hi) = w[1];")
    rs_lines.append("        if jod_lo <= target_jod && target_jod <= jod_hi {")
    rs_lines.append("            let t = (target_jod - jod_lo) / (jod_hi - jod_lo);")
    rs_lines.append("            return d_lo + t * (d_hi - d_lo);")
    rs_lines.append("        }")
    rs_lines.append("    }")
    rs_lines.append("    1.0  // unreachable in practice")
    rs_lines.append("}")
    rs_lines.append("")
    rs_lines.append("#[cfg(test)]")
    rs_lines.append("mod tests {")
    rs_lines.append("    use super::*;")
    rs_lines.append("")
    rs_lines.append("    #[test]")
    rs_lines.append("    fn default_table_monotone_in_jod() {")
    rs_lines.append("        let t = CVVDP_DISTANCE_LOOKUP_DEFAULT;")
    rs_lines.append("        assert!(t.len() >= 2);")
    rs_lines.append("        for w in t.windows(2) {")
    rs_lines.append("            assert!(w[0].0 < w[1].0, \"jod must be strictly increasing\");")
    rs_lines.append("            assert!(w[0].1 >= w[1].1, \"distance must be monotone non-increasing as jod target tightens\");")
    rs_lines.append("        }")
    rs_lines.append("    }")
    rs_lines.append("")
    rs_lines.append("    #[test]")
    rs_lines.append("    fn lookup_clamps_outside_range() {")
    rs_lines.append("        let t = CVVDP_DISTANCE_LOOKUP_DEFAULT;")
    rs_lines.append("        let very_lossy = lookup_distance_for_target_jod(t, 0.0);")
    rs_lines.append("        let very_strict = lookup_distance_for_target_jod(t, 100.0);")
    rs_lines.append("        assert!(very_lossy >= very_strict);")
    rs_lines.append("    }")
    rs_lines.append("")
    rs_lines.append("    #[test]")
    rs_lines.append("    fn lookup_interpolates_inside_range() {")
    rs_lines.append("        let t = CVVDP_DISTANCE_LOOKUP_DEFAULT;")
    rs_lines.append("        if t.len() >= 2 {")
    rs_lines.append("            let mid_jod = (t[0].0 + t[t.len() - 1].0) / 2.0;")
    rs_lines.append("            let d = lookup_distance_for_target_jod(t, mid_jod);")
    rs_lines.append("            assert!(d > 0.0);")
    rs_lines.append("            assert!(d < 100.0);")
    rs_lines.append("        }")
    rs_lines.append("    }")
    rs_lines.append("}")
    rs_lines.append("")

    rs_path = "jxl-encoder/src/vardct/cvvdp_distance_lookup.rs.snippet"
    with open(rs_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rs_lines))
    print(f"Wrote {rs_path}", file=sys.stderr)

    print("\n## Calibration table preview (default-blend)", file=sys.stderr)
    for jod in target_jods:
        d = table["_DEFAULT"].get(jod)
        if d is None:
            continue
        print(f"  target cvvdp_jod {jod:.3f} → effective distance {d:.4f}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    p = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv"
    )
    sys.exit(main(p))
