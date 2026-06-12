#!/usr/bin/env python3
"""GOAL_BEAT_CJXL wall-axis grid (docs/GOAL_BEAT_CJXL.md axis 3, issue #74).

Zenbench-grade-shaped cross-encoder wall measurement: interleaved paired
execution (c,o,c,o,... per cell so thermal/cache/background drift hits
both sides equally), >= 5 measured iterations after 1 unmeasured warmup
per side, medians + mins reported, matched thread counts, every cell
tagged with commit + host + loadavg.

REFUSES to run when loadavg(1m) > 1.0 or when other heavy processes are
detected — per the standing quiet-box rule
(benchmarks/REVISIT_QUEUE_2026-06-11.md): ad-hoc walls under load are
exploration-only and must never feed dispositions. Override only via
--exploratory, which forces an EXPLORATORY- prefix on the output
filename and a load warning in every row.

Cells: 5 imazen-26 strata x e{5,7} x {1,8} threads, lossy d1.0 +
lossless — a compact representative slice of the scenario matrix's
wall axis (1T exposes algorithmic cost, 8T exposes parallel shape).
Budget check per docs/GOAL_BEAT_CJXL.md: ours <= 1.2x cjxl at e<=7.

Usage:
  python3 scripts/scoreboard/wall_grid.py benchmarks/scoreboard/wall_grid_<date>.tsv
"""

import argparse
import csv
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OURS = REPO / "target/release/cjxl-rs"
CJXL = Path.home() / "work/jxl-efforts/libjxl/build/tools/cjxl"
BENCH_SET = REPO / "benchmarks/lossless_bench_set_2026-06-10.tsv"
STRATA = ["photos-png", "web-screenshots", "plots", "noaa-documents", "ai-illustrations"]
ITERS = 5
LOAD_CEILING = 1.0


def commit():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def pick_sources():
    rows = list(csv.DictReader(open(BENCH_SET), delimiter="\t"))
    out = []
    for s in STRATA:
        r = next((r for r in rows if r["stratum"] == s and r["tier"] == "core"),
                 next((r for r in rows if r["stratum"] == s), None))
        assert r, s
        out.append((s, r["bench_input"]))
    return out


def run_once(binary, src, out, mode, effort, threads):
    cmd = ["nice", "-n19", str(binary), str(src), str(out), "-e", str(effort)]
    if mode == "lossless":
        cmd += (["--lossless"] if binary == OURS else ["-d", "0"])
    else:
        cmd += ["-d", "1.0"]
    cmd += ["--threads", str(threads)] if binary == OURS else ["--num_threads", str(threads)]
    t0 = time.monotonic()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_tsv")
    ap.add_argument("--exploratory", action="store_true",
                    help="run despite load (EXPLORATORY-prefixed output, never disposal-grade)")
    args = ap.parse_args()

    load1 = os.getloadavg()[0]
    if load1 > LOAD_CEILING and not args.exploratory:
        sys.exit(f"REFUSING: loadavg {load1:.2f} > {LOAD_CEILING} — quiet-box rule "
                 f"(REVISIT_QUEUE_2026-06-11.md). Re-run when idle, or pass "
                 f"--exploratory for non-disposal numbers.")
    out_path = Path(args.out_tsv)
    if args.exploratory and not out_path.name.startswith("EXPLORATORY-"):
        out_path = out_path.with_name("EXPLORATORY-" + out_path.name)

    srcs = pick_sources()
    f = open(out_path, "w")
    w = csv.writer(f, delimiter="\t")
    w.writerow(["stratum", "mode", "effort", "threads",
                "ours_med_s", "cjxl_med_s", "ratio", "budget_ok",
                "ours_min_s", "cjxl_min_s", "iters", "loadavg_at_cell",
                "commit", "host"])
    host = socket.gethostname()
    sha = commit()
    for stratum, src in srcs:
        for mode in ("lossy", "lossless"):
            for e in (5, 7):
                for threads in (1, 8):
                    tmp = f"/tmp/wg_{os.getpid()}.jxl"
                    run_once(CJXL, src, tmp, mode, e, threads)   # warmups
                    run_once(OURS, src, tmp, mode, e, threads)
                    tc, to = [], []
                    for _ in range(ITERS):
                        tc.append(run_once(CJXL, src, tmp, mode, e, threads))
                        to.append(run_once(OURS, src, tmp, mode, e, threads))
                    Path(tmp).unlink(missing_ok=True)
                    mo, mc = statistics.median(to), statistics.median(tc)
                    ratio = mo / mc
                    budget = ratio <= 1.2  # e<=7 budget per the goal doc
                    w.writerow([stratum, mode, e, threads,
                                f"{mo:.3f}", f"{mc:.3f}", f"{ratio:.3f}",
                                int(budget), f"{min(to):.3f}", f"{min(tc):.3f}",
                                ITERS, f"{os.getloadavg()[0]:.2f}", sha, host])
                    f.flush()
                    print(f"{stratum:18s} {mode:8s} e{e} {threads}T: "
                          f"ours {mo:.3f}s cjxl {mc:.3f}s = {ratio:.2f}x "
                          f"{'OK' if budget else 'OVER-BUDGET'}",
                          file=sys.stderr, flush=True)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
