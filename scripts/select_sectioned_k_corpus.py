#!/usr/bin/env python3
"""Pick representative origins + renditions for the sectioned prune-K gate corpus (#99 item 1).

Input: the TSVs produced by `sectioned_k_corpus features` (one row per source
image, content features in the axes predictor choice depends on).

Output: a `picks.tsv` for `sectioned_k_corpus render`
(`origin \t cluster \t split \t kind \t target`).

Why clustering and not random sampling: imazen-26 is dominated by one content
class (749 of 2233 images are `9226-lilith-ai-products`, another 370 are web
screenshots). A random draw reproduces that skew, so the modal class sets the
gate's threshold and the outliers — the images a content gate actually fails
on — are the ones most likely to be missing. k-means over the standardized
feature space and then the centroid-nearest member of each cluster gives one
origin per REGION of content space instead of one per unit of corpus mass.

Split policy: two representatives per cluster (the two members nearest the
centroid), one to `train`, one to `validate`. The split is therefore by
ORIGIN — every rendition of an origin lands on the same side, so no
derivative of a training image can appear in validation — while both sides
still span every content region.

Sizes: a log-spaced ladder (ratio sqrt(2)). Only ladder entries at or below
the source's long edge are legal: UPSCALING IS SKIPPED, because a synthetic
upscale has no high-frequency detail and would mislead every feature the gate
conditions on. Both `crop` (native detail statistics) and `resize`
(Mitchell-Netravali/CatmullRom) are emitted at each chosen size, because
production sees both.

Pure stdlib (no numpy on this host). Deterministic: fixed seed, fixed restart
count, ties broken by path.

Continuity origins: any path listed in the `JXL_K_CORPUS_CONTINUITY` env var
(colon-separated) is emitted with `split=continuity` in ADDITION to whatever
the clustering picked. These are the gb82-sc screens the 2026-08-28 prune-K
benchmark used; they are reported separately and are NOT part of the fit, so
they neither train the gate nor inflate the held-out number.

Usage:
  select_sectioned_k_corpus.py <picks.tsv> <clusters.tsv> <K> <feat.tsv>...
"""

import math
import random
import sys

# Feature columns consumed from the `features` TSV, with the transform applied
# before standardization. `log` entries are log10(x + eps).
FEATURES = [
    ("log_pixels", "log_px"),
    ("frac_left_equal", "raw"),
    ("frac_grad_zero", "raw"),
    ("uniq_per_px", "log"),
    ("frac_ctx_flat", "raw"),
    ("ctx_entropy", "raw"),
    ("best_bpp", "log"),
    ("root_spread", "raw"),
    ("ctx_winners", "raw"),
    ("frac_ctx_mismatch", "raw"),
]

# Log-spaced size ladder (ratio ~sqrt(2)). 362 is the smallest entry that
# still yields more than one 256-px modular group on a landscape source, which
# is the regime the sectioned per-group writer exists for.
LADDER = [362, 512, 724, 1024, 1448, 2048]
SIZES_PER_ORIGIN = 3
REPS_PER_CLUSTER = 2
RESTARTS = 8
ITERS = 120
SEED = 20260830


def read_rows(paths):
    rows = []
    for p in paths:
        header = None
        with open(p) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if header is None:
                    header = cols
                    continue
                rows.append(dict(zip(header, cols)))
    return rows


def vectorize(rows):
    vecs = []
    for r in rows:
        w, h = float(r["w"]), float(r["h"])
        raw = {
            "log_pixels": math.log10(max(w * h, 1.0)),
            "frac_left_equal": float(r["frac_left_equal"]),
            "frac_grad_zero": float(r["frac_grad_zero"]),
            "uniq_per_px": float(r["uniq_per_px"]),
            "frac_ctx_flat": float(r["frac_ctx_flat"]),
            "ctx_entropy": float(r["ctx_entropy"]),
            "best_bpp": float(r["best_bpp"]),
            "root_spread": float(r["root_spread"]),
            "ctx_winners": float(r["ctx_winners"]),
            "frac_ctx_mismatch": float(r["frac_ctx_mismatch"]),
        }
        v = []
        for name, kind in FEATURES:
            x = raw[name]
            if kind == "log":
                x = math.log10(x + 1e-6)
            v.append(x)
        vecs.append(v)
    return vecs


def standardize(vecs):
    n, d = len(vecs), len(vecs[0])
    mean = [sum(v[j] for v in vecs) / n for j in range(d)]
    var = [sum((v[j] - mean[j]) ** 2 for v in vecs) / max(n - 1, 1) for j in range(d)]
    sd = [math.sqrt(x) if x > 1e-12 else 1.0 for x in var]
    return [[(v[j] - mean[j]) / sd[j] for j in range(d)] for v in vecs], mean, sd


def d2(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def kmeans(pts, k, rng):
    # k-means++ seeding
    cents = [pts[rng.randrange(len(pts))]]
    while len(cents) < k:
        dists = [min(d2(p, c) for c in cents) for p in pts]
        tot = sum(dists)
        if tot <= 0:
            cents.append(pts[rng.randrange(len(pts))])
            continue
        r = rng.random() * tot
        acc = 0.0
        for p, dd in zip(pts, dists):
            acc += dd
            if acc >= r:
                cents.append(p)
                break
    assign = [0] * len(pts)
    for _ in range(ITERS):
        changed = False
        for i, p in enumerate(pts):
            best, bd = 0, float("inf")
            for ci, c in enumerate(cents):
                dd = d2(p, c)
                if dd < bd:
                    best, bd = ci, dd
            if assign[i] != best:
                assign[i] = best
                changed = True
        dim = len(pts[0])
        sums = [[0.0] * dim for _ in range(k)]
        cnt = [0] * k
        for i, p in enumerate(pts):
            a = assign[i]
            cnt[a] += 1
            for j in range(dim):
                sums[a][j] += p[j]
        for ci in range(k):
            if cnt[ci] == 0:
                cents[ci] = pts[rng.randrange(len(pts))]
            else:
                cents[ci] = [s / cnt[ci] for s in sums[ci]]
        if not changed:
            break
    inertia = sum(d2(p, cents[assign[i]]) for i, p in enumerate(pts))
    return assign, cents, inertia


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(2)
    picks_path, clusters_path, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rows = read_rows(sys.argv[4:])
    rows.sort(key=lambda r: r["path"])
    print(f"[select] {len(rows)} source images", file=sys.stderr)
    vecs = vectorize(rows)
    pts, mean, sd = standardize(vecs)

    best = None
    for s in range(RESTARTS):
        rng = random.Random(SEED + s)
        assign, cents, inertia = kmeans(pts, k, rng)
        print(f"[select] restart {s}: inertia {inertia:.1f}", file=sys.stderr)
        if best is None or inertia < best[2]:
            best = (assign, cents, inertia)
    assign, cents, inertia = best
    print(f"[select] best inertia {inertia:.1f}", file=sys.stderr)

    members = {ci: [] for ci in range(k)}
    for i, a in enumerate(assign):
        members[a].append((d2(pts[i], cents[a]), rows[i]["path"], i))
    for ci in members:
        members[ci].sort(key=lambda t: (t[0], t[1]))

    with open(clusters_path, "w") as cf:
        cf.write("# select_sectioned_k_corpus.py cluster assignment (#99 item 1)\n")
        cf.write(f"# K={k} restarts={RESTARTS} seed={SEED} inertia={inertia:.3f}\n")
        cf.write("cluster\tsize\trank\tdist\tsplit\tpath\n")
        picks = []
        for ci in sorted(members):
            mem = members[ci]
            for rank, (dist, path, _idx) in enumerate(mem):
                split = ""
                if rank < REPS_PER_CLUSTER:
                    split = "train" if rank % 2 == 0 else "validate"
                    picks.append((path, ci, split))
                cf.write(
                    f"{ci}\t{len(mem)}\t{rank}\t{dist:.4f}\t{split}\t{path}\n"
                )

    import os

    for cont in os.environ.get("JXL_K_CORPUS_CONTINUITY", "").split(":"):
        cont = cont.strip()
        if cont:
            picks.append((cont, -1, "continuity"))

    by_path = {r["path"]: r for r in rows}
    n_rend = 0
    with open(picks_path, "w") as pf:
        pf.write("# select_sectioned_k_corpus.py picks (#99 item 1)\n")
        pf.write(f"# K={k} ladder={LADDER} sizes_per_origin={SIZES_PER_ORIGIN}\n")
        pf.write("origin\tcluster\tsplit\tkind\ttarget\n")
        for path, ci, split in picks:
            r = by_path[path]
            long_edge = max(int(r["w"]), int(r["h"]))
            legal = [s for s in LADDER if s <= long_edge]
            if not legal:
                print(f"[select] SKIP (too small for ladder): {path}", file=sys.stderr)
                continue
            if len(legal) <= SIZES_PER_ORIGIN:
                chosen = legal
            else:
                # smallest, middle, largest legal — spans the ladder without
                # spending six encodes on one origin.
                chosen = sorted({legal[0], legal[len(legal) // 2], legal[-1]})
            for t in chosen:
                for kind in ("crop", "resize"):
                    pf.write(f"{path}\t{ci}\t{split}\t{kind}\t{t}\n")
                    n_rend += 1
    print(
        f"[select] {len(picks)} origins -> {n_rend} renditions "
        f"({picks_path}, {clusters_path})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
