#!/usr/bin/env python3
"""Analyse benchmarks/auto_resample_monotonicity_<tag>.tsv (issue #101 follow-up).

Answers, per image x effort:
  1. self-check: every `auto` row byte-identical (sha16) to its `res2` twin at
     t = d*0.25+0.25 (the harness measures what it claims);
  2. default-ladder byte monotonicity across the d=10 switch (bytes must be
     non-increasing in d) and the size of the quality cliff (bfly / ssim2);
  3. matched-butteraugli regime comparison: for each full-res ladder point,
     the bytes the 2x regime needs to reach the SAME butteraugli (interpolated
     on its own t-curve); the crossing d* = smallest d where 2x is cheaper at
     matched quality; `inadmissible` where full-res quality is finer than the
     2x floor (bfly at t=0.5);
  4. how d* tracks the resampling floor (rank correlation), which is the
     hypothesis a floor-driven / matched-quality switch rule rests on.

Stdlib only. Usage: scripts/auto_resample_monotonicity_analyze.py <tsv> [out.md]
"""
import csv, math, sys
from collections import defaultdict

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            for k in ("w", "h", "cropped", "effort", "resampling", "bytes"):
                r[k] = int(r[k])
            for k in ("d_req", "d_internal", "bfly", "ssim2", "enc_ms"):
                r[k] = float(r[k])
            rows.append(r)
    return rows

def interp_bytes_at_bfly(curve, target):
    """curve: list of (bfly, bytes) sorted by bfly ascending (finer first).
    Linear interpolation of bytes at bfly == target. None if target < min bfly
    (regime cannot reach that quality) ; extrapolate flat beyond max."""
    if not curve:
        return None
    if target < curve[0][0]:
        return None
    if target >= curve[-1][0]:
        return curve[-1][1]
    for (b0, y0), (b1, y1) in zip(curve, curve[1:]):
        if b0 <= target <= b1:
            if b1 == b0:
                return min(y0, y1)
            t = (target - b0) / (b1 - b0)
            return y0 + t * (y1 - y0)
    return None

def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r

def spearman(xs, ys):
    if len(xs) < 3:
        return float("nan")
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")

def main():
    path = sys.argv[1]
    out_md = sys.argv[2] if len(sys.argv) > 2 else None
    rows = load(path)
    by = defaultdict(list)
    for r in rows:
        by[(r["image"], r["effort"])].append(r)
    lines = []
    P = lines.append
    P(f"# auto-resample monotonicity analysis — {path}")
    P("")
    # 1. self-check: what does the default (`auto`) path actually encode?
    #    libjxl rule on  -> byte-identical to the res2 row at t = d*0.25+0.25
    #    one regime      -> byte-identical to the full-res row at the same d
    n_res2 = n_full = n_neither = checked = 0
    for key, rs in by.items():
        res2 = {round(r["d_req"], 4): r for r in rs if r["mode"] == "res2"}
        full = {round(r["d_req"], 4): r for r in rs if r["mode"] == "full"}
        for a in (r for r in rs if r["mode"] == "auto"):
            t2 = res2.get(round(a["d_internal"], 4))
            tf = full.get(round(a["d_req"], 4))
            checked += 1
            if t2 is not None and t2["sha16"] == a["sha16"] and t2["bytes"] == a["bytes"]:
                n_res2 += 1
            elif tf is not None and tf["sha16"] == a["sha16"] and tf["bytes"] == a["bytes"]:
                n_full += 1
            else:
                n_neither += 1
                P(f"- SELF-CHECK MISMATCH {key} d={a['d_req']}: auto {a['bytes']}B/{a['sha16']} matches neither res2(t={a['d_internal']}) nor full(d)")
    P(f"## 1. self-check: {checked} auto rows — {n_res2} ≡ res2 twin (libjxl rule ON), {n_full} ≡ full-res row (one regime), {n_neither} match neither")
    P("")
    # 2. default ladder monotonicity + cliff
    P("## 2. default ladder (full below d=10, whatever `auto` encodes at d>=10): byte/quality monotonicity")
    P("")
    P("| image | class | e | bytes(9.9) | bytes(10) | Δbytes@switch | byte↑ steps | bfly(9.9) | bfly(10) | ssim2(9.9) | ssim2(10) | quality↑ steps (bfly/ssim2) |")
    P("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    agg = defaultdict(lambda: {"n": 0, "byte_up_switch": 0, "byte_up_any": 0, "cliff_bfly": [], "cliff_ssim2": [], "dbytes": []})
    for key in sorted(by):
        rs = by[key]
        cls = rs[0]["class"]
        full = {r["d_req"]: r for r in rs if r["mode"] == "full"}
        auto = {r["d_req"]: r for r in rs if r["mode"] == "auto"}
        ladder = sorted([(d, full[d]) for d in full if d < 10.0] + [(d, auto[d]) for d in auto])
        byte_up = sum(1 for (d0, a), (d1, b) in zip(ladder, ladder[1:]) if b["bytes"] > a["bytes"])
        q_up_b = sum(1 for (d0, a), (d1, b) in zip(ladder, ladder[1:]) if b["bfly"] < a["bfly"] - 1e-6)
        q_up_s = sum(1 for (d0, a), (d1, b) in zip(ladder, ladder[1:]) if b["ssim2"] > a["ssim2"] + 1e-6)
        f99, a10 = full.get(9.9), auto.get(10.0)
        if f99 is None or a10 is None:
            continue
        db = (a10["bytes"] - f99["bytes"]) / f99["bytes"] * 100
        A = agg[(cls, key[1])]
        A["n"] += 1
        A["byte_up_switch"] += int(a10["bytes"] > f99["bytes"])
        A["byte_up_any"] += int(byte_up > 0)
        A["cliff_bfly"].append(a10["bfly"] - f99["bfly"])
        A["cliff_ssim2"].append(a10["ssim2"] - f99["ssim2"])
        A["dbytes"].append(db)
        P(f"| {key[0]} | {cls} | {key[1]} | {f99['bytes']} | {a10['bytes']} | {db:+.1f}% | {byte_up} | {f99['bfly']:.2f} | {a10['bfly']:.2f} | {f99['ssim2']:.1f} | {a10['ssim2']:.1f} | {q_up_b}/{q_up_s} |")
    P("")
    P("### per class × effort")
    P("")
    P("| class | e | n | bytes↑ at switch | any bytes↑ on ladder | mean Δbytes@switch | mean Δbfly@switch | mean Δssim2@switch |")
    P("|---|--:|--:|--:|--:|--:|--:|--:|")
    for (cls, e), A in sorted(agg.items()):
        n = A["n"]
        P(f"| {cls} | {e} | {n} | {A['byte_up_switch']}/{n} | {A['byte_up_any']}/{n} | {sum(A['dbytes'])/n:+.1f}% | {sum(A['cliff_bfly'])/n:+.2f} | {sum(A['cliff_ssim2'])/n:+.1f} |")
    P("")
    # 2b. within-regime noise: the `full` ladder alone (no switch)
    P("## 2b. within-regime non-monotonicity on the full-res ladder alone (quantiser noise, no switch)")
    P("")
    P("| e | cells | bytes↑ steps / steps | max bytes↑ | bfly-improves steps / steps | max bfly improvement | ssim2-improves steps | max ssim2 improvement |")
    P("|--:|--:|--:|--:|--:|--:|--:|--:|")
    per_e = defaultdict(lambda: {"cells": 0, "steps": 0, "b_up": 0, "b_up_max": 0.0, "q_up": 0, "q_up_max": 0.0, "s_up": 0, "s_up_max": 0.0})
    for key, rs in by.items():
        full = sorted((r["d_req"], r) for r in rs if r["mode"] == "full")
        if len(full) < 2:
            continue
        E = per_e[key[1]]
        E["cells"] += 1
        for (d0, a), (d1, b) in zip(full, full[1:]):
            E["steps"] += 1
            if b["bytes"] > a["bytes"]:
                E["b_up"] += 1
                E["b_up_max"] = max(E["b_up_max"], (b["bytes"] - a["bytes"]) / a["bytes"] * 100)
            if b["bfly"] < a["bfly"] - 1e-6:
                E["q_up"] += 1
                E["q_up_max"] = max(E["q_up_max"], a["bfly"] - b["bfly"])
            if b["ssim2"] > a["ssim2"] + 1e-6:
                E["s_up"] += 1
                E["s_up_max"] = max(E["s_up_max"], b["ssim2"] - a["ssim2"])
    for e, E in sorted(per_e.items()):
        P(f"| {e} | {E['cells']} | {E['b_up']}/{E['steps']} | {E['b_up_max']:+.1f}% | {E['q_up']}/{E['steps']} | {E['q_up_max']:.2f} | {E['s_up']}/{E['steps']} | {E['s_up_max']:.1f} |")
    P("")
    # 3. matched-quality regime comparison
    P("## 3. matched-butteraugli comparison: bytes(2x at same bfly) / bytes(full) per ladder point")
    P("")
    P("`inadm` = full-res quality at that d is finer than the 2x floor (bfly at t=0.5): 2x cannot reach it. d* = first d where 2x is cheaper at matched bfly.")
    P("")
    dfull_all = sorted({r["d_req"] for r in rows if r["mode"] == "full"})
    hdr = "| image | class | e | floor bfly | " + " | ".join(f"d={d:g}" for d in dfull_all) + " | d* |"
    P(hdr)
    P("|---|---|--:|--:|" + "--:|" * len(dfull_all) + "--:|")
    floors, dstars, cls_of = [], [], []
    for key in sorted(by):
        rs = by[key]
        cls = rs[0]["class"]
        full = {r["d_req"]: r for r in rs if r["mode"] == "full"}
        res2 = sorted(((r["bfly"], r["bytes"], r["d_req"]) for r in rs if r["mode"] == "res2"), key=lambda x: x[2])
        # curve sorted by bfly ascending; enforce monotone envelope in bfly (t ascending should give bfly ascending)
        curve = []
        for b, y, t in res2:
            if not math.isnan(b):
                curve.append((b, y))
        curve.sort()
        floor = min((b for b, _ in curve), default=float("nan"))
        cells = []
        dstar = None
        for d in dfull_all:
            f = full.get(d)
            if f is None or math.isnan(f["bfly"]):
                cells.append("-")
                continue
            yb = interp_bytes_at_bfly(curve, f["bfly"])
            if yb is None:
                cells.append("inadm")
            else:
                ratio = yb / f["bytes"]
                cells.append(f"{ratio:.2f}")
                if dstar is None and ratio < 1.0:
                    dstar = d
        floors.append(floor); dstars.append(dstar if dstar is not None else 99.0); cls_of.append(cls)
        P(f"| {key[0]} | {cls} | {key[1]} | {floor:.2f} | " + " | ".join(cells) + f" | {dstar if dstar is not None else '>25'} |")
    P("")
    P(f"Spearman(floor bfly, d*) over all image×effort cells: {spearman(floors, dstars):.2f}  (d* = 99 where 2x never wins by d=25)")
    P("")
    # distribution of d*
    dist = defaultdict(int)
    for c, d in zip(cls_of, dstars):
        dist[(c, d)] += 1
    P("### d* distribution per class")
    P("")
    for c in sorted({c for c in cls_of}):
        items = sorted((d, n) for (cc, d), n in dist.items() if cc == c)
        P(f"- {c}: " + ", ".join(f"d*={'>25' if d == 99.0 else f'{d:g}'}: {n}" for d, n in items))
    # 4. cjxl cross-check rows (MODE=cjxl TSV)
    cj = [r for r in rows if r["mode"].startswith("cjxl_")]
    if cj:
        P("")
        P("## 4. cjxl cross-check: default flags (`cjxl_auto`) vs `--resampling=1` (`cjxl_full`), same decode+score path")
        P("")
        P("`same` = byte-identical (that cjxl did not switch at this d). Otherwise Δbytes / Δbfly / Δssim2 of auto relative to full.")
        P("")
        dl = sorted({r["d_req"] for r in cj})
        P("| image | class | " + " | ".join(f"d={d:g}" for d in dl) + " |")
        P("|---|---|" + "--:|" * len(dl))
        first_switch = defaultdict(list)
        for key in sorted({(r["image"], r["class"]) for r in cj}):
            auto = {r["d_req"]: r for r in cj if r["image"] == key[0] and r["mode"] == "cjxl_auto"}
            full = {r["d_req"]: r for r in cj if r["image"] == key[0] and r["mode"] == "cjxl_full"}
            cells = []
            fs_d = None
            for d in dl:
                a, f = auto.get(d), full.get(d)
                if a is None or f is None:
                    cells.append("-"); continue
                if a["sha16"] == f["sha16"]:
                    cells.append("same"); continue
                db = (a["bytes"] - f["bytes"]) / f["bytes"] * 100
                cells.append(f"{db:+.0f}%/{a['bfly']-f['bfly']:+.1f}/{a['ssim2']-f['ssim2']:+.0f}")
                if fs_d is None:
                    fs_d = d
                    first_switch[key[1]].append((d, db, a["bfly"] - f["bfly"], a["ssim2"] - f["ssim2"]))
            P(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
        P("")
        P("### first cjxl switch per class: d, mean Δbytes, mean Δbfly, mean Δssim2")
        P("")
        for c, items in sorted(first_switch.items()):
            n = len(items)
            P(f"- {c}: n={n}, first-switch d ∈ {sorted({i[0] for i in items})}, mean Δbytes {sum(i[1] for i in items)/n:+.1f}%, mean Δbfly {sum(i[2] for i in items)/n:+.2f}, mean Δssim2 {sum(i[3] for i in items)/n:+.1f}")
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if out_md:
        with open(out_md, "w") as f:
            f.write(text)

if __name__ == "__main__":
    main()
