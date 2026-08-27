#!/usr/bin/env python3
"""S4 arm B2: slope_t{80,88} refit on the zq_seed 8-feature basis.
Registered in benchmarks/s4_iter1_eps_wave_2026-08-27.md (##B2). Same frozen
ridge protocol as the 190-feature fit: lambda grid on VAL MAE, TEST once,
constant baseline. Targets from zensim's jxl_ladders_9pt (PRIMARY cell
vd-e7_zen_def, `ok` rows); features REF-ONLY from the 07-01 canonical root,
joined on (origin_id, width, height) — vintage-independent by construction.
`distinct_color_bins` enters ln_1p (matching the shipped head's convention).
"""
import json
import numpy as np
import pyarrow.parquet as pq

LAD = "/mnt/v/zen/zensim-training/s4c2-2026-08-27/jxl_ladders_9pt.parquet"
ROOT = "/mnt/v/output/canonical-picker-2026-07-01-zensimA/zenjxl_lossy"
OUT = "/mnt/v/output/jxl-encoder/s4-iter1-eps-2026-08-27"
NAMES = ["flat_color_block_ratio", "gradient_fraction", "distinct_color_bins",
         "high_freq_energy_ratio", "aq_map_std", "grayscale_score",
         "luma_histogram_entropy", "quant_survival_y"]
LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
PRIMARY = "vd-e7_zen_def"

feats = {}
for split in ("train", "validate", "test"):
    t = pq.read_table(f"{ROOT}/{split}.parquet",
                      columns=["origin_id", "width", "height"] + [f"feat_{n}" for n in NAMES])
    d = t.to_pydict()
    for j in range(t.num_rows):
        k = (d["origin_id"][j], float(d["width"][j]), float(d["height"][j]))
        if k not in feats:
            v = [float(d[f"feat_{n}"][j]) for n in NAMES]
            v[2] = np.log1p(v[2])
            feats[k] = v
print("renditions with features:", len(feats))

lad = pq.read_table(LAD).to_pydict()
report = {}
for t in (80, 88):
    rows = [j for j in range(len(lad["cell"]))
            if lad["cell"][j] == PRIMARY and lad[f"flag_t{t}"][j] == "ok"
            and (lad["origin_id"][j], float(lad["width"][j]), float(lad["height"][j])) in feats]
    X = np.array([feats[(lad["origin_id"][j], float(lad["width"][j]), float(lad["height"][j]))] for j in rows])
    y = np.array([lad[f"slope_dscore_dlogq_t{t}"][j] for j in rows])
    sp = [lad["split"][j] for j in rows]
    tr = [i for i, s in enumerate(sp) if s == "train"]
    va = [i for i, s in enumerate(sp) if s == "val"]
    te = [i for i, s in enumerate(sp) if s == "test"]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd < 1e-12] = 1.0
    def prep(idx):
        return np.hstack([(X[idx] - mu) / sd, np.ones((len(idx), 1))])
    Ztr, Zva, Zte = prep(tr), prep(va), prep(te)
    best = None
    for lam in LAMS:
        A = Ztr.T @ Ztr + lam * np.eye(9); A[-1, -1] -= lam
        w = np.linalg.solve(A, Ztr.T @ y[tr])
        mae = float(np.abs(Zva @ w - y[va]).mean())
        if best is None or mae < best[0]:
            best = (mae, lam, w)
    _, lam, w = best
    base = float(np.median(y[tr]))
    pt = Zte @ w
    report[f"t{t}"] = {
        "n": [len(tr), len(va), len(te)], "lambda": lam,
        "val_mae": round(best[0], 3),
        "test_mae": round(float(np.abs(pt - y[te]).mean()), 3),
        "const_test_mae": round(float(np.abs(base - y[te]).mean()), 3),
        "beat_ratio": round(float(np.abs(pt - y[te]).mean() / np.abs(base - y[te]).mean()), 4),
        "mu": [round(float(v), 6) for v in mu], "sd": [round(float(v), 6) for v in sd],
        "w": [round(float(v), 6) for v in w],
    }
    print(f"t{t}: test_mae {report[f't{t}']['test_mae']} vs const {report[f't{t}']['const_test_mae']} "
          f"(ratio {report[f't{t}']['beat_ratio']}, lam {lam}, n_te {len(te)})")
json.dump(report, open(f"{OUT}/b2_slope_fit.json", "w"), indent=1)
