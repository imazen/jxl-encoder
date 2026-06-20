#!/usr/bin/env python3
"""Turn a vcpu_resource_sweep TSV into the vCPU-axis findings the seed targets.

For each (path, effort, size) stratum reports:
  1. MEMORY vs threads — marginal working set (VmHWM delta) and whole-proc peak
     RSS per thread count, plus a linear fit delta_kb = a + b*threads (b = the
     per-thread scratch growth, the gamma term estimate_encode does not model).
  2. TIME speedup — wall(1)/wall(N) and parallel efficiency per thread count.
  3. EST vs MEASURED — the thread-independent estimate_encode prediction
     (working_pred = est_typ - fixed - input) vs measured marginal delta, and
     the typical/max band vs measured peak RSS.

Usage: vcpu_resource_fit.py <sweep.tsv>
"""
import sys
from collections import defaultdict

FIXED_KB = {"lossy": 16 * 1024, "lossless": 20 * 1024}  # heuristics.rs fixed overhead


def load(fn):
    rows = []
    with open(fn) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ix = {k: i for i, k in enumerate(hdr)}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < len(hdr):
                continue

            def g(k):
                v = c[ix[k]]
                return v

            def fnum(k):
                try:
                    return float(g(k))
                except ValueError:
                    return None

            rows.append(
                dict(
                    path=g("path"),
                    effort=int(g("effort")),
                    w=int(g("width")),
                    px=int(g("pixels")),
                    threads=int(g("threads")),
                    est_typ=fnum("est_typ_kb"),
                    est_max=fnum("est_max_kb"),
                    est_time=fnum("est_time_ms"),
                    ph=fnum("meas_peak_heap_kb"),
                    pr=fnum("meas_peak_rss_kb"),
                    vmhwm=fnum("meas_vmhwm_kb"),
                    delta=fnum("meas_delta_kb"),
                    wall=fnum("meas_wall_ms"),
                    user=fnum("meas_user_ms"),
                    sys=fnum("meas_sys_ms"),
                )
            )
    return rows


def linfit(xs, ys):
    n = len(xs)
    if n < 2:
        return None, None
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    d = n * sxx - sx * sx
    if d == 0:
        return None, None
    b = (n * sxy - sx * sy) / d
    a = (sy - b * sx) / n
    return a, b


def main():
    rows = load(sys.argv[1])
    by = defaultdict(list)
    for r in rows:
        by[(r["path"], r["effort"], r["w"])].append(r)

    print("=" * 78)
    print("vCPU RESOURCE SWEEP — jxl-encoder")
    print("=" * 78)
    for k in sorted(by):
        path, effort, w = k
        g = sorted(by[k], key=lambda r: r["threads"])
        px = g[0]["px"]
        fixed = FIXED_KB[path]
        input_kb = px * 3 // 1024  # rgb8
        print(f"\n### {path} e{effort}  {w}x{w} ({px/1e6:.2f} MP)  input={input_kb/1024:.1f} MB")
        print(f"  {'thr':>3} {'wall_ms':>8} {'speedup':>7} {'eff%':>5} "
              f"{'delta_MB':>8} {'peakRSS_MB':>10} {'peakHeap_MB':>11}")
        wall1 = next((r["wall"] for r in g if r["threads"] == 1 and r["wall"]), None)
        for r in g:
            sp = (wall1 / r["wall"]) if (wall1 and r["wall"]) else None
            eff = (sp / r["threads"] * 100) if sp else None
            ph = f"{r['ph']/1024:>11.1f}" if r["ph"] else f"{'—':>11}"
            print(f"  {r['threads']:>3} {r['wall'] or 0:>8.1f} "
                  f"{(f'{sp:.2f}x' if sp else '—'):>7} {(f'{eff:.0f}' if eff else '—'):>5} "
                  f"{(r['delta'] or 0)/1024:>8.1f} {(r['vmhwm'] or 0)/1024:>10.1f} {ph}")
        # gamma fit: delta_kb = a + b*threads
        ts = [r["threads"] for r in g if r["delta"]]
        ds = [r["delta"] for r in g if r["delta"]]
        a, b = linfit(ts, ds)
        if b is not None:
            print(f"  fit: marginal delta_kb ≈ {a:.0f} + {b:.0f}·threads "
                  f"(γ = {b/1024:.2f} MB/thread per-worker scratch)")
        # est vs measured (thread-independent model)
        et = g[0]["est_typ"]
        em = g[0]["est_max"]
        if et:
            working_pred = et - fixed - input_kb
            d1 = next((r["delta"] for r in g if r["threads"] == 1), None)
            dN = max((r["delta"] for r in g if r["delta"]), default=None)
            prN = max((r["vmhwm"] for r in g if r["vmhwm"]), default=None)
            print(f"  EST: typ={et/1024:.0f} MB (working_pred={working_pred/1024:.0f} MB), max={em/1024:.0f} MB")
            if d1:
                print(f"       measured marginal delta: t1={d1/1024:.0f} MB, "
                      f"max-over-threads={dN/1024:.0f} MB → working_pred/delta_t1 = {working_pred/max(d1,1):.2f}×")
            if prN:
                print(f"       measured peak RSS max-over-threads={prN/1024:.0f} MB "
                      f"vs est_max={em/1024:.0f} MB → {em/max(prN,1):.2f}× (cover={'OK' if em>=prN else 'UNDER'})")
    print()


if __name__ == "__main__":
    main()
