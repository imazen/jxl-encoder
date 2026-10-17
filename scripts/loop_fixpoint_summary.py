#!/usr/bin/env python3
"""Summarize examples/loop_fixpoint_ab.rs output per effort, content class and distance.

usage: loop_fixpoint_summary.py cells.tsv > summary.tsv

Class: "screenshot" for gb82-sc paths, else "photo". Wall times are sums over the
group's cells; frames are LoopTrace frames (scored iterations) over all seeds.
"""
import csv
import sys
from collections import defaultdict


def main(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    groups = defaultdict(list)
    for r in rows:
        cls = "screenshot" if "gb82-sc" in r["image"] else "photo"
        for key in [(r["effort"], cls, r["distance"]), (r["effort"], cls, "all"), (r["effort"], "all", "all")]:
            groups[key].append(r)
    out = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    out.writerow(["effort", "class", "distance", "cells", "byte_identical", "stopped_early",
                  "stop_iter_min", "stop_iter_median", "frames_run", "frames_scheduled",
                  "s_stop_off", "s_stop_on", "wall_change_pct"])

    def order(k):
        e, c, d = k
        return (int(e), c == "all", c, d == "all", float(d) if d != "all" else 0.0)

    for key in sorted(groups, key=order):
        g = groups[key]
        stops = sorted(int(r["first_stop_iter"]) for r in g if r["first_stop_iter"] != "-")
        off = sum(float(r["ms_stop_off"]) for r in g) / 1e3
        on = sum(float(r["ms_stop_on"]) for r in g) / 1e3
        out.writerow([*key, len(g), sum(r["identical"] == "true" for r in g), len(stops),
                      stops[0] if stops else "-", stops[len(stops) // 2] if stops else "-",
                      sum(int(r["frames_run"]) for r in g), sum(int(r["frames_scheduled"]) for r in g),
                      f"{off:.2f}", f"{on:.2f}", f"{(on / off - 1) * 100:+.1f}"])


if __name__ == "__main__":
    main(sys.argv[1])
