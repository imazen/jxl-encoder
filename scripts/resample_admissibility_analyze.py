#!/usr/bin/env python3
"""Analyse benchmarks/resample_admissibility_<tag>.{cells,images}.tsv (#101).

Answers, over the imazen-26 stratified pick list:

  1. Is 2x resampling ever the cheaper regime, and where?  Per (image, size,
     effort, requested distance) we ask what the 2x regime must spend to reach
     the SAME butteraugli the full-res encode delivered.  The bytes come from
     the cheapest 2x setting that ACTUALLY met the target on the measured
     grid -- a realisable configuration, never an interpolation -- so the
     ratio cannot flatter 2x with a point no encoder can produce.

  2. Is the resampling FLOOR a sound gate?  The floor (down->up, no
     quantisation) bounds the whole regime: if floor bfly > the quality the
     full-res encode delivered, 2x provably cannot match it at any bitrate.
     We check the floor never admits an unreachable cell (soundness) and how
     tight it is (how many admissible cells it keeps).

  3. Can we beat a fixed distance threshold?  Heuristics are compared under
     leave-one-stratum-out cross-validation on the money metric: the realised
     matched-quality byte ratio on the cells each rule selects, plus its
     worst case and its rate of selecting UNREACHABLE cells.

Usage: scripts/resample_admissibility_analyze.py <tag-or-cells.tsv> [out.md]
"""
import csv
import math
import sys
from collections import defaultdict

# A 2x cell counts as admissible only if it beats full-res by more than this;
# below it the two regimes are a wash and the simpler one should win.
WIN_MARGIN = 0.02


def load(cells_path, images_path):
    cells = []
    with open(cells_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            for k in ("w", "h", "size_cap", "effort", "resampling", "bytes"):
                r[k] = int(r[k])
            for k in ("d_req", "d_internal", "bfly", "ssim2", "enc_ms"):
                r[k] = float(r[k])
            cells.append(r)
    floors, feats = {}, defaultdict(dict)
    with open(images_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = (r["image"], int(r["size_cap"]))
            if r["floor_kernel"]:
                floors[(key[0], key[1], r["floor_kernel"])] = (
                    float(r["floor_bfly"]),
                    float(r["floor_ssim2"]),
                )
            elif r["feature"]:
                try:
                    feats[key][r["feature"]] = float(r["value"])
                except ValueError:
                    pass
    return cells, floors, feats


def bytes_to_reach(curve, target_bfly):
    """Cheapest MEASURED 2x setting whose bfly <= target. None if none does."""
    ok = [b for (bf, b) in curve if bf <= target_bfly]
    return min(ok) if ok else None


def build_truth(cells, floors):
    """One record per (image, size, effort, full-res distance)."""
    by = defaultdict(lambda: {"full": [], "res2": []})
    for c in cells:
        if math.isnan(c["bfly"]):
            continue
        by[(c["image"], c["stratum"], c["size_cap"], c["w"], c["h"], c["effort"])][
            c["mode"]
        ].append(c)
    out = []
    for key, m in by.items():
        image, stratum, cap, w, h, effort = key
        curve = [(c["bfly"], c["bytes"]) for c in m["res2"]]
        if not curve or not m["full"]:
            continue
        best_2x_bfly = min(bf for bf, _ in curve)
        # The effort-matched kernel is what the encoder would actually use;
        # the iterative (decoder-adjoint) floor is the tighter bound on what
        # ANY coded frame could achieve, so soundness is judged against it.
        kernel = "iterative" if effort >= 10 else "sharper"
        floor = floors.get((image, cap, kernel), (float("nan"), float("nan")))[0]
        floor_bound = floors.get((image, cap, "iterative"), (float("nan"), float("nan")))[0]
        for c in m["full"]:
            need = bytes_to_reach(curve, c["bfly"])
            ratio = None if need is None else need / c["bytes"]
            out.append(
                {
                    "image": image,
                    "stratum": stratum,
                    "size_cap": cap,
                    "w": w,
                    "h": h,
                    "mp": w * h / 1e6,
                    "effort": effort,
                    "d": c["d_req"],
                    "bytes_full": c["bytes"],
                    "bfly_full": c["bfly"],
                    "ssim2_full": c["ssim2"],
                    "floor": floor,
                    "floor_bound": floor_bound,
                    "best_2x_bfly": best_2x_bfly,
                    "reachable": need is not None,
                    "ratio": ratio,
                    "admissible": ratio is not None and ratio < 1.0 - WIN_MARGIN,
                }
            )
    return out


def fmt_pct(x):
    return "n/a" if x is None else f"{100*x:.0f}%"


def evaluate(rule, rows):
    """Money metric for a decision rule over `rows`. `rule(row) -> bool`."""
    sel = [r for r in rows if rule(r)]
    n = len(sel)
    if n == 0:
        return {"n": 0, "cov": 0.0, "mean_r": None, "worst_r": None,
                "unreach": 0, "prec": None, "saved": 0.0}
    unreach = sum(1 for r in sel if not r["reachable"])
    reach = [r for r in sel if r["reachable"]]
    ratios = [r["ratio"] for r in reach]
    # Realised bytes if we follow the rule: 2x cost where reachable; where
    # UNREACHABLE the rule cannot deliver the requested quality at all, which
    # we charge as a quality failure (counted separately, never averaged away).
    saved = sum(1.0 - r for r in ratios)
    return {
        "n": n,
        "cov": n / len(rows),
        "mean_r": (sum(ratios) / len(ratios)) if ratios else None,
        "worst_r": max(ratios) if ratios else None,
        "unreach": unreach,
        "prec": sum(1 for r in sel if r["admissible"]) / n,
        "saved": saved,
    }


def main():
    arg = sys.argv[1]
    if arg.endswith(".tsv"):
        cells_path = arg
        images_path = arg.replace(".cells.tsv", ".images.tsv")
    else:
        cells_path = f"benchmarks/resample_admissibility_{arg}.cells.tsv"
        images_path = f"benchmarks/resample_admissibility_{arg}.images.tsv"
    out_md = sys.argv[2] if len(sys.argv) > 2 else None
    cells, floors, feats = load(cells_path, images_path)
    rows = build_truth(cells, floors)

    L = []
    P = L.append
    P(f"# resampling admissibility on imazen-26 — {cells_path}")
    P("")
    strata = sorted({r["stratum"] for r in rows})
    P(f"{len(cells)} encode cells, {len(rows)} decision cells, "
      f"{len({r['image'] for r in rows})} images, {len(strata)} strata, "
      f"efforts {sorted({r['effort'] for r in rows})}, "
      f"sizes {sorted({r['size_cap'] for r in rows})}.")
    P("")
    P(f"A cell counts as admissible only when 2x beats full-res by more than "
      f"{100*WIN_MARGIN:.0f}% at matched butteraugli.")
    P("")

    # ── 1. how often is 2x ever the cheaper regime ──
    adm = [r for r in rows if r["admissible"]]
    unreach = [r for r in rows if not r["reachable"]]
    P("## 1. Is 2x ever the cheaper regime?")
    P("")
    P(f"- admissible: **{len(adm)}/{len(rows)}** cells ({fmt_pct(len(adm)/len(rows))})")
    P(f"- unreachable at any bitrate (2x cannot match the full-res quality): "
      f"{len(unreach)}/{len(rows)} ({fmt_pct(len(unreach)/len(rows))})")
    if adm:
        g = sorted(1 - r["ratio"] for r in adm)
        P(f"- when admissible, byte saving: median {100*g[len(g)//2]:.0f}%, "
          f"max {100*g[-1]:.0f}%")
    P("")
    P("| stratum | cells | admissible | unreachable | first d where 2x wins | best saving |")
    P("|---|--:|--:|--:|--:|--:|")
    for s in strata:
        rs = [r for r in rows if r["stratum"] == s]
        a = [r for r in rs if r["admissible"]]
        u = [r for r in rs if not r["reachable"]]
        first_d = min((r["d"] for r in a), default=None)
        best = max((1 - r["ratio"] for r in a), default=None)
        P(f"| {s} | {len(rs)} | {len(a)} ({fmt_pct(len(a)/len(rs))}) | "
          f"{fmt_pct(len(u)/len(rs))} | {first_d if first_d else '—'} | "
          f"{'—' if best is None else f'{100*best:.0f}%'} |")
    P("")
    P("### by requested distance (all strata pooled)")
    P("")
    P("| d | cells | admissible | unreachable | mean ratio (reachable) |")
    P("|--:|--:|--:|--:|--:|")
    for d in sorted({r["d"] for r in rows}):
        rs = [r for r in rows if r["d"] == d]
        a = [r for r in rs if r["admissible"]]
        u = [r for r in rs if not r["reachable"]]
        rr = [r["ratio"] for r in rs if r["reachable"]]
        P(f"| {d:g} | {len(rs)} | {len(a)} ({fmt_pct(len(a)/len(rs))}) | "
          f"{fmt_pct(len(u)/len(rs))} | "
          f"{'—' if not rr else f'{sum(rr)/len(rr):.2f}'} |")
    P("")

    # ── 2. soundness of the floor gate ──
    P("## 2. Is the resampling floor a sound gate?")
    P("")
    P("The floor bounds the regime, so `floor > bfly_full(d)` should imply "
      "unreachable. Violations would mean the bound is not a bound.")
    P("")
    have_floor = [r for r in rows if not math.isnan(r["floor"])]
    P(f"- cells with a floor measured: {len(have_floor)}")
    for label, key in [("effort-matched kernel", "floor"),
                       ("iterative kernel (the tighter bound)", "floor_bound")]:
        hv = [r for r in have_floor if not math.isnan(r[key])]
        caught = [r for r in hv if r[key] > r["bfly_full"]]
        viol = [r for r in caught if r["reachable"]]
        P(f"- {label}: says unreachable on {len(caught)}/{len(hv)} cells; of "
          f"those actually reachable on the measured grid (bound violations): "
          f"**{len(viol)}**")
    viol = [r for r in have_floor if r["floor"] > r["bfly_full"] and r["reachable"]]
    caught = [r for r in have_floor if r["floor"] > r["bfly_full"]]
    miss = [r for r in have_floor if r["floor"] <= r["bfly_full"] and not r["reachable"]]
    P(f"- floor says reachable but the grid could not reach it: {len(miss)} "
      f"(expected — the floor needs infinite bitrate; the grid stops at "
      f"internal distance 0.3)")
    P("")
    # headroom distribution
    P("### headroom (bfly_full / floor) vs outcome")
    P("")
    P("| bfly_full/floor | cells | admissible | mean ratio |")
    P("|---|--:|--:|--:|")
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 1e9)]
    for lo, hi in bins:
        rs = [r for r in have_floor
              if r["floor"] > 0 and lo <= r["bfly_full"] / r["floor"] < hi]
        if not rs:
            continue
        a = [r for r in rs if r["admissible"]]
        rr = [r["ratio"] for r in rs if r["reachable"]]
        label = f"{lo:g}–{hi:g}" if hi < 1e9 else f"≥{lo:g}"
        P(f"| {label} | {len(rs)} | {len(a)} ({fmt_pct(len(a)/len(rs))}) | "
          f"{'—' if not rr else f'{sum(rr)/len(rr):.2f}'} |")
    P("")

    # ── 2b. how TIGHT is the bound (does the 2x curve saturate at the floor?) ──
    P("### is the bound tight? (best measured 2x quality vs the floor)")
    P("")
    P("If the 2x regime saturates at its floor, the floor is not merely an "
      "upper bound on quality but a good PREDICTION of it — which is what "
      "makes a floor gate decisive rather than merely safe.")
    P("")
    seen = {}
    for r in have_floor:
        seen.setdefault((r["image"], r["size_cap"], r["effort"]), r)
    rr = [(v["best_2x_bfly"] / v["floor"]) for v in seen.values() if v["floor"] > 0]
    if rr:
        rr.sort()
        P(f"- best measured 2x butteraugli / floor over {len(rr)} image×size×effort "
          f"cells: median **{rr[len(rr)//2]:.2f}**, p10 {rr[len(rr)//10]:.2f}, "
          f"p90 {rr[(9*len(rr))//10]:.2f}")
        P(f"- (1.00 = the grid reached the floor exactly; >1 = the grid's finest "
          f"2x setting is still short of it)")
    P("")

    # ── 2c. within-regime monotonicity on THIS corpus ──
    P("## 2c. Within-regime monotonicity on the full-res ladder (no switch involved)")
    P("")
    P("Bytes should fall as the requested distance rises. Violations here are "
      "the quantiser's own, not a regime switch — and they bound how monotone "
      "any single-regime encoder can be on this corpus.")
    P("")
    P("| stratum | steps | bytes rise | max rise | quality improves | max bfly improvement |")
    P("|---|--:|--:|--:|--:|--:|")
    ladders = defaultdict(list)
    for c in cells:
        if c["mode"] == "full" and not math.isnan(c["bfly"]):
            ladders[(c["image"], c["stratum"], c["size_cap"], c["effort"])].append(c)
    per_s = defaultdict(lambda: {"steps": 0, "up": 0, "up_max": 0.0, "q": 0, "q_max": 0.0})
    for (img, s, cap, e), cs in ladders.items():
        cs.sort(key=lambda c: c["d_req"])
        A = per_s[s]
        for a, b in zip(cs, cs[1:]):
            A["steps"] += 1
            if b["bytes"] > a["bytes"]:
                A["up"] += 1
                A["up_max"] = max(A["up_max"], (b["bytes"] - a["bytes"]) / a["bytes"] * 100)
            if b["bfly"] < a["bfly"] - 1e-6:
                A["q"] += 1
                A["q_max"] = max(A["q_max"], a["bfly"] - b["bfly"])
    tot = {"steps": 0, "up": 0, "q": 0}
    for s in sorted(per_s):
        A = per_s[s]
        tot["steps"] += A["steps"]; tot["up"] += A["up"]; tot["q"] += A["q"]
        P(f"| {s} | {A['steps']} | {A['up']} | {A['up_max']:+.1f}% | {A['q']} | {A['q_max']:.2f} |")
    P(f"| **all** | {tot['steps']} | {tot['up']} "
      f"({fmt_pct(tot['up']/max(1,tot['steps']))}) | | {tot['q']} "
      f"({fmt_pct(tot['q']/max(1,tot['steps']))}) | |")
    P("")

    # ── 3. heuristics, leave-one-stratum-out ──
    P("## 3. Decision rules (leave-one-stratum-out cross-validation)")
    P("")
    P("`coverage` = share of cells the rule sends to 2x. `mean ratio` = "
      "matched-quality bytes vs full-res on those cells (<1 is a win). "
      "`unreachable` = cells it selected where 2x cannot deliver the requested "
      "quality at any bitrate — a quality failure, not a byte trade.")
    P("")

    def rule_libjxl(r):
        return r["d"] >= 10.0

    def rule_never(_r):
        return False

    def rule_oracle(r):
        return r["admissible"]

    def make_floor_rule(k):
        return lambda r: (not math.isnan(r["floor"])) and r["floor"] * k <= r["bfly_full"]

    # Fit k per held-out fold on the training strata, by maximising total saving
    # while forbidding any unreachable selection.
    KS = [round(0.5 + 0.25 * i, 2) for i in range(20)]
    loso_sel = []
    chosen = {}
    for s in strata:
        train = [r for r in rows if r["stratum"] != s]
        test = [r for r in rows if r["stratum"] == s]
        best_k, best_score = None, -1e18
        for k in KS:
            m = evaluate(make_floor_rule(k), train)
            if m["n"] == 0:
                continue
            score = m["saved"] - 100.0 * m["unreach"]
            if score > best_score:
                best_k, best_score = k, score
        chosen[s] = best_k
        if best_k is not None:
            f = make_floor_rule(best_k)
            loso_sel.extend([(r, f(r)) for r in test])

    def eval_pairs(pairs):
        sel = [r for r, took in pairs if took]
        allr = [r for r, _ in pairs]
        if not sel:
            return {"n": 0, "cov": 0.0, "mean_r": None, "worst_r": None,
                    "unreach": 0, "prec": None, "saved": 0.0}
        return {
            "n": len(sel),
            "cov": len(sel) / len(allr),
            "mean_r": (sum(r["ratio"] for r in sel if r["reachable"]) /
                       max(1, sum(1 for r in sel if r["reachable"]))),
            "worst_r": max((r["ratio"] for r in sel if r["reachable"]), default=None),
            "unreach": sum(1 for r in sel if not r["reachable"]),
            "prec": sum(1 for r in sel if r["admissible"]) / len(sel),
            "saved": sum(1.0 - r["ratio"] for r in sel if r["reachable"]),
        }

    table = [
        ("never resample (current zen default)", evaluate(rule_never, rows)),
        ("libjxl: d >= 10", evaluate(rule_libjxl, rows)),
        ("floor gate, LOSO-fitted k", eval_pairs(loso_sel)),
        ("oracle (knows the answer)", evaluate(rule_oracle, rows)),
    ]
    def num(m, k):
        v = m[k]
        return "—" if v is None else f"{v:.2f}"

    P("| rule | coverage | selected | precision | mean ratio | worst ratio | unreachable selected |")
    P("|---|--:|--:|--:|--:|--:|--:|")
    for name, m in table:
        P(f"| {name} | {fmt_pct(m['cov'])} | {m['n']} | {fmt_pct(m['prec'])} | "
          f"{num(m, 'mean_r')} | {num(m, 'worst_r')} | {m['unreach']} |")
    P("")
    ks = sorted({v for v in chosen.values() if v is not None})
    P(f"LOSO-chosen floor multipliers k: {ks} (rule: take 2x iff "
      f"`floor * k <= bfly_full(d)`)")
    P("")
    P("### floor-gate sensitivity on all data (not cross-validated)")
    P("")
    P("| k | coverage | precision | mean ratio | worst ratio | unreachable |")
    P("|--:|--:|--:|--:|--:|--:|")
    for k in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        m = evaluate(make_floor_rule(k), rows)
        P(f"| {k:g} | {fmt_pct(m['cov'])} | {fmt_pct(m['prec'])} | "
          f"{num(m, 'mean_r')} | {num(m, 'worst_r')} | {m['unreach']} |")
    P("")

    # ── 4. does any zenanalyze feature add signal over the floor? ──
    P("## 4. Does any image feature add signal over the floor?")
    P("")
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except Exception as e:  # pragma: no cover
        P(f"(sklearn/numpy unavailable: {e})")
        text = "\n".join(L) + "\n"
        sys.stdout.write(text)
        if out_md:
            open(out_md, "w").write(text)
        return

    usable = [r for r in rows if not math.isnan(r["floor"]) and r["floor"] > 0]
    featnames = sorted(set().union(*[set(feats[(r["image"], r["size_cap"])].keys())
                                     for r in usable])) if usable else []
    featnames = [f for f in featnames
                 if all(f in feats[(r["image"], r["size_cap"])] for r in usable)]

    def design(rs, names):
        X = []
        for r in rs:
            base = [math.log(max(r["floor"], 1e-6) / max(r["d"], 1e-6)),
                    math.log(max(r["d"], 1e-6)),
                    float(r["effort"]),
                    math.log(max(r["mp"], 1e-6))]
            X.append(base + [feats[(r["image"], r["size_cap"])][n] for n in names])
        return np.asarray(X, dtype=float)

    y = np.asarray([1 if r["admissible"] else 0 for r in usable])
    P(f"Target: admissible (positives {int(y.sum())}/{len(y)}). "
      f"Predictors: log(floor/d), log d, effort, log MP "
      f"(+ {len(featnames)} zenanalyze features in the second model). "
      f"Leave-one-stratum-out AUC.")
    P("")

    def loso_auc(names):
        """LOSO AUC scored ONLY over rows that actually received a prediction.

        A fold whose training half contains no positives is unfittable and is
        skipped; scoring its rows as 0 would rank every positive below every
        negative and report an anti-predictive AUC that is a fold artifact,
        not a result. Returns (auc, n_scored, n_pos_scored).
        """
        preds = np.full(len(usable), np.nan)
        for s in strata:
            tr = [i for i, r in enumerate(usable) if r["stratum"] != s]
            te = [i for i, r in enumerate(usable) if r["stratum"] == s]
            if not te or len(set(y[tr])) < 2:
                continue
            Xtr = design([usable[i] for i in tr], names)
            Xte = design([usable[i] for i in te], names)
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd == 0] = 1.0
            clf = LogisticRegression(max_iter=2000, C=0.5)
            clf.fit((Xtr - mu) / sd, y[tr])
            preds[te] = clf.predict_proba((Xte - mu) / sd)[:, 1]
        m = ~np.isnan(preds)
        if m.sum() == 0 or len(set(y[m])) < 2:
            return (float("nan"), int(m.sum()), int(y[m].sum()))
        return (roc_auc_score(y[m], preds[m]), int(m.sum()), int(y[m].sum()))

    def fmt_auc(t):
        auc, n, npos = t
        a = "n/a" if (isinstance(auc, float) and math.isnan(auc)) else f"{auc:.3f}"
        return f"**{a}** (over {n} scorable rows, {npos} positive)"

    floor_only_auc = roc_auc_score(
        y, np.asarray([-math.log(max(r["floor"], 1e-6) / max(r["d"], 1e-6)) for r in usable])
    ) if len(set(y)) > 1 else float("nan")
    P(f"- floor/d alone, as a ranking score (no fitting, so no folds): "
      f"AUC **{floor_only_auc:.3f}**")
    P(f"- logistic on the 4 base predictors: AUC {fmt_auc(loso_auc([]))}")
    if featnames:
        P(f"- logistic on base + {len(featnames)} zenanalyze features: "
          f"AUC {fmt_auc(loso_auc(featnames))}")
    P("")
    P("A fitted model can only be compared with floor/d on the rows it could "
      "score; strata holding all the positives make their own fold unfittable.")
    P("")

    text = "\n".join(L) + "\n"
    sys.stdout.write(text)
    if out_md:
        open(out_md, "w").write(text)


if __name__ == "__main__":
    main()
