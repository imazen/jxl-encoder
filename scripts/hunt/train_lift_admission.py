#!/usr/bin/env python3
"""W44-231: train the learned qf-seed lift admission (train digits only).

Inputs:
  --raw-tsv     lift_rawband harness TSV (config A: raw-band lift, all
                content excludes disabled) — gives ours bytes/ssim2/bfly
                AND cjxl reference per (image, effort, distance) cell.
  --off-tsv     lift_off_bytes TSV (config B: lift fully off) —
                name, effort, distance, bytes.
  --features    zenanalyze features TSV (image stem -> 101 features).
  --val-raw/--val-off/--val-features  same trio for the held-out
                validation digits (1/3/5). NEVER used for fitting.

Label (per image): the lift FIRED on a cell iff bytes_A != bytes_B.
GOOD requires BOTH, on every fired cell/effort:
  (a) ssim2_A >= cjxl_ssim2 (W44-174/176 ship criterion — the buy must
      carry us at least to the reference's quality), AND
  (b) NO distance-monotonicity break: bytes_raw(d2.0) <=
      bytes_default(d1.5) per effort (--mono-tsv; the validated defect
      signature from the 2026-06-12 precedent and the W44-230
      adjudications — a file that GROWS when the caller asks for MORE
      distortion breaks the distance contract).
Images with no fired cell are excluded (admission is only consulted
when the band fires).

Model: small decision tree (depth<=4) + a random-forest cross-check,
class-weighted. Reports train CV + held-out precision/recall and the
tree rules. The tree is deliberately tiny so it can be audited and
embedded; if RF >> tree on held-out, says so.
"""
import argparse, csv, sys
import numpy as np

def load_cells(raw_tsv, off_tsv):
    off = {}
    for ln in open(off_tsv):
        p = ln.rstrip("\n").split("\t")
        if len(p) == 4 and p[3].isdigit():
            off[(p[0], p[1], f"{float(p[2]):.4f}")] = int(p[3])
    cells = {}
    for r in csv.DictReader(open(raw_tsv), delimiter="\t"):
        if r.get("status") != "OK":
            continue
        k = (r["image"], r["effort"], f'{float(r["distance"]):.4f}')
        b = off.get(k)
        if b is None or b == 0:
            continue
        cells[k] = {
            "bytes_a": int(r["ours_bytes"]),
            "bytes_off": b,
            "ssim2_a": float(r["ours_ssim2"]),
            "cjxl_ssim2": float(r["cjxl_ssim2"]),
            "cjxl_bytes": int(r["cjxl_bytes"]),
        }
    return cells

def load_mono(path):
    """name -> {effort: (bytes_default_d15, bytes_raw_d20)}"""
    out = {}
    for ln in open(path):
        p = ln.rstrip("\n").split("\t")
        if len(p) == 4 and p[2].isdigit() and p[3].isdigit():
            out.setdefault(p[0], {})[p[1]] = (int(p[2]), int(p[3]))
    return out

def image_labels(cells, mono):
    per_img = {}
    for (img, e, d), c in cells.items():
        fired = c["bytes_a"] != c["bytes_off"]
        if not fired:
            continue
        # STRICT-GOOD (web-aggressive product goal): with the lift on,
        # we must Pareto-dominate-or-tie cjxl — at least its ssim2 for
        # at most its bytes. Premium buys (quality above cjxl at +bytes)
        # are BAD here: in the d<3.5 sub-band this label's deployment
        # region, the premium class is RD-indistinguishable from the
        # adjudicated misfires (eff 0.03-0.13 ssim2/+1%bytes both). The
        # d>=3.5 main band (the W44-174 text-sharpness calibration) is
        # NOT governed by this model.
        good = c["ssim2_a"] >= c["cjxl_ssim2"] and c["bytes_a"] <= c["cjxl_bytes"]
        per_img.setdefault(img, []).append(good)
    return {img: all(goods) for img, goods in per_img.items()}

def load_features(path):
    rows = list(csv.reader(open(path), delimiter="\t"))
    header = rows[0][1:]
    out = {}
    for r in rows[1:]:
        stem = r[0].split("_")[0]
        vals = [float(v) if v not in ("", "nan") else np.nan for v in r[1:]]
        out[stem] = vals
    return header, out

def main():
    ap = argparse.ArgumentParser()
    for a in ("raw_tsv", "off_tsv", "features", "mono_tsv",
              "val_raw", "val_off", "val_features", "val_mono"):
        ap.add_argument("--" + a.replace("_", "-"), required=True)
    ap.add_argument("--tree-depth", type=int, default=4)
    args = ap.parse_args()

    header, feats = load_features(args.features)
    labels = image_labels(load_cells(args.raw_tsv, args.off_tsv), load_mono(args.mono_tsv))
    X, y, names = [], [], []
    for img, good in sorted(labels.items()):
        if img in feats:
            X.append(feats[img]); y.append(1 if good else 0); names.append(img)
    X = np.nan_to_num(np.array(X), nan=-1.0)
    y = np.array(y)
    print(f"train: {len(y)} fired images, GOOD={y.sum()} BAD={(1-y).sum()}")

    vh, vfeats = load_features(args.val_features)
    assert vh == header, "feature header mismatch train vs val"
    vlabels = image_labels(load_cells(args.val_raw, args.val_off), load_mono(args.val_mono))
    XV, yv, vnames = [], [], []
    for img, good in sorted(vlabels.items()):
        if img in vfeats:
            XV.append(vfeats[img]); yv.append(1 if good else 0); vnames.append(img)
    XV = np.nan_to_num(np.array(XV), nan=-1.0) if XV else np.zeros((0, len(header)))
    yv = np.array(yv)
    print(f"val:   {len(yv)} fired images, GOOD={yv.sum()} BAD={(1-yv).sum()}")

    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    tree = DecisionTreeClassifier(max_depth=args.tree_depth, class_weight="balanced",
                                  min_samples_leaf=5, random_state=42)
    cv = cross_val_score(tree, X, y, cv=5, scoring="balanced_accuracy")
    tree.fit(X, y)
    print(f"tree depth<={args.tree_depth}: train-CV balanced-acc {cv.mean():.3f} +/- {cv.std():.3f}")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                min_samples_leaf=3, random_state=42)
    rf.fit(X, y)

    def report(name, model):
        if len(yv) == 0:
            return
        p = model.predict(XV)
        tp = int(((p == 1) & (yv == 1)).sum()); fp = int(((p == 1) & (yv == 0)).sum())
        fn = int(((p == 0) & (yv == 1)).sum()); tn = int(((p == 0) & (yv == 0)).sum())
        print(f"{name} HELD-OUT: admit-precision {tp}/{tp+fp}, admit-recall {tp}/{tp+fn}, "
              f"block-precision {tn}/{tn+fn}, block-recall {tn}/{tn+fp}")
        wrong = [(vnames[i], int(yv[i]), int(p[i])) for i in range(len(yv)) if p[i] != yv[i]]
        if wrong:
            print(f"  misclassified: {wrong}")

    report("tree", tree)
    report("rf  ", rf)
    print("\ntree rules:")
    print(export_text(tree, feature_names=header, max_depth=args.tree_depth))
    imp = sorted(zip(rf.feature_importances_, header), reverse=True)[:12]
    print("rf top features:", [(n, round(v, 3)) for v, n in imp])

if __name__ == "__main__":
    main()
