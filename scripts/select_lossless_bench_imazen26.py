#!/usr/bin/env python3
"""Select the imazen-26 lossless benchmark set via per-stratum k-means.

Implements the sweep-discipline stratification rule for the lossless perf
bench refresh (2026-06-10): pick centroid-nearest representatives per
content-class stratum using k-means on zenanalyze `feat_*` embeddings,
instead of hand-picking or random-sampling (which over-represents the
modal class — 8100-web-screenshots alone is 370 of 1603 candidates).

Candidate pool is PNG-only: cjxl-rs auto-routes .jpg inputs to the
JPEG->JXL transcode path (a different code path than the modular
tree-learner the lossless bench measures), and HEIC/DNG don't decode in
the extractor. Megapixel cap 16 MP keeps e9 1T bench cells tractable.

Pipeline (three subcommands, run in order):

  1. prep    — read CORPUS-MANIFEST.tsv, filter (png, <=16 MP, known
               folder), write candidates.tsv + N extractor manifest
               shards for zenanalyze's extract_features_for_picker.
  2. (run the extractor on each shard — see --print-extract-cmds)
  3. select  — concat shard features, append feat_zz_log2_mp (native
               megapixels; gives k-means a size axis inside strata with
               mixed sizes, auto-dropped as zero-variance on uniform
               strata), run cluster_sources.py per stratum, merge picks,
               compute sha256 of the chosen files, emit the set TSV +
               clusters JSON.

Reuses (does not modify):
  ~/work/zen/zenanalyze/examples/extract_features_for_picker.rs
  ~/work/zen/zenanalyze/zenpicker-train/scripts/cluster_sources.py
"""

import argparse
import hashlib
import json
import subprocess
import sys
from math import log2
from pathlib import Path

import pandas as pd

CORPUS = Path("/home/lilith/work/codec-corpus/imazen-26")
CLUSTER_SOURCES = Path(
    "~/work/zen/zenanalyze/zenpicker-train/scripts/cluster_sources.py"
).expanduser()
MP_CAP_PX = 16_000_000
CORE_MP_CAP_PX = 8_500_000  # tier=core picks must fit (fast A/B gate set)
SEED = 42

# stratum -> (corpus folders, K picks). Folders merged into one stratum
# share one k-means pool (photos-png: the 10 stray PNGs in two photo
# classes — too few to cluster separately).
STRATA = {
    "photos-png": (["1200-lilith-interiors", "1400-lilith-nature"], 2),
    "nps-brochures": (["5000-national-park-service-brochures"], 2),
    "epa-report": (["5200-epa-climate-impact-2021-report"], 1),
    "noaa-documents": (["5300-noaa-hurricane-documents"], 2),
    "patents": (["6000-lilith-scans-public-patents"], 2),
    "manuscript-illustrations": (["6600-ia-scans-manuscript-illustrations"], 2),
    "manuscript-text": (["6800-ia-scans-manuscript-text"], 2),
    "plots": (["7000-lilith-plots"], 3),
    "mobile-screenshots": (["8000-lilith-mobile-screenshots"], 2),
    "web-screenshots": (["8100-lilith-web-screenshots"], 5),
    "ai-clipart": (["9000-lilith-ai-clipart"], 2),
    "ai-illustrations": (["9094-lilith-ai-illustrations"], 2),
    "ai-products": (["9226-lilith-ai-products"], 3),
}


def folder_to_stratum():
    m = {}
    for stratum, (folders, _k) in STRATA.items():
        for f in folders:
            m[f] = stratum
    return m


def load_candidates():
    df = pd.read_csv(CORPUS / "CORPUS-MANIFEST.tsv", sep="\t", dtype=str)
    df["width"] = df["width"].astype(int)
    df["height"] = df["height"].astype(int)
    df["bytes"] = df["bytes"].astype(int)
    df["px"] = df["width"] * df["height"]
    f2s = folder_to_stratum()
    df = df[
        (df["format"] == "png")
        & (df["px"] <= MP_CAP_PX)
        & (df["folder"].isin(f2s))
    ].copy()
    df["stratum"] = df["folder"].map(f2s)
    df["abs_path"] = df["path"].map(lambda p: str(CORPUS / p))
    return df


def cmd_prep(args):
    out = Path(args.workdir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_candidates()
    df.to_csv(out / "candidates.tsv", sep="\t", index=False)
    print(f"candidates: {len(df)} images, {df['bytes'].sum() / 1e6:.0f} MB", file=sys.stderr)
    print(df.groupby("stratum").size().to_string(), file=sys.stderr)

    # Extractor manifest shards (columns read by name in the extractor).
    n = args.shards
    for i in range(n):
        shard = df.iloc[i::n]
        m = pd.DataFrame(
            {
                "sha256": "",
                "split": "bench",
                "content_class": shard["stratum"],
                "source": "imazen-26",
                "path": shard["abs_path"],
            }
        )
        m.to_csv(out / f"extract_manifest_{i}.tsv", sep="\t", index=False)
    print(f"wrote {n} extractor manifest shards to {out}", file=sys.stderr)
    if args.print_extract_cmds:
        for i in range(n):
            print(
                f"<extractor-binary> --manifest {out}/extract_manifest_{i}.tsv "
                f"--output {out}/features_{i}.tsv --sizes 1024"
            )


def run_stratum_kmeans(stratum, k, sub, workdir):
    """Write the stratum's feature TSV and run cluster_sources.py on it."""
    feat_tsv = workdir / f"stratum_{stratum}.tsv"
    out_list = workdir / f"stratum_{stratum}.picks.txt"
    out_json = workdir / f"stratum_{stratum}.clusters.json"
    sub.to_csv(feat_tsv, sep="\t", index=False)
    k_eff = min(k, len(sub))
    subprocess.run(
        [
            sys.executable,
            str(CLUSTER_SOURCES),
            "--features", str(feat_tsv),
            "--k", str(k_eff),
            "--out-list", str(out_list),
            "--out-json", str(out_json),
            "--seed", str(SEED),
        ],
        check=True,
    )
    return json.loads(out_json.read_text())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def cmd_select(args):
    workdir = Path(args.workdir)
    cand = pd.read_csv(workdir / "candidates.tsv", sep="\t", dtype=str)
    cand["px"] = cand["width"].astype(int) * cand["height"].astype(int)

    shards = sorted(workdir.glob("features_*.tsv"))
    if not shards:
        sys.exit(f"no features_*.tsv in {workdir} — run the extractor first")
    feats = pd.concat([pd.read_csv(p, sep="\t", dtype=str) for p in shards])
    print(f"features: {len(feats)} rows from {len(shards)} shards", file=sys.stderr)

    # Join native dims onto features (extractor reports resized dims) and
    # append the explicit size axis for mixed-size strata.
    feats = feats.merge(
        cand[["abs_path", "stratum", "px"]],
        left_on="image_path",
        right_on="abs_path",
        how="inner",
        validate="one_to_one",
    )
    missing = len(cand) - len(feats)
    if missing:
        print(f"WARNING: {missing} candidates missing from features (decode fails?)", file=sys.stderr)
    feats["feat_zz_log2_mp"] = feats["px"].map(lambda p: f"{log2(int(p) / 1e6):.6f}")

    rows = []
    for stratum, (_folders, k) in STRATA.items():
        sub = feats[feats["stratum"] == stratum].drop(columns=["abs_path", "stratum", "px"])
        if sub.empty:
            print(f"WARNING: stratum {stratum} has no feature rows — skipped", file=sys.stderr)
            continue
        meta = run_stratum_kmeans(stratum, k, sub, workdir)
        # clusters are size-desc sorted by cluster_sources.py
        core_assigned = False
        for c in meta["clusters"]:
            row = cand[cand["abs_path"] == c["rep_path"]].iloc[0].to_dict()
            is_core = (not core_assigned) and int(row["px"]) <= CORE_MP_CAP_PX
            if is_core:
                core_assigned = True
            rows.append(
                {
                    "stratum": stratum,
                    "corpus_rel_path": row["path"],
                    "tier": "core" if is_core else "full",
                    "cluster_id": c["cluster"],
                    "cluster_size": c["size"],
                    "stratum_n": meta["n_samples"],
                    "rep_dist": f"{c['rep_dist']:.4f}",
                    "width": row["width"],
                    "height": row["height"],
                    "megapixels": f"{int(row['px']) / 1e6:.2f}",
                    "bytes": row["bytes"],
                    "sha256": sha256_file(CORPUS / row["path"]),
                    "descriptor": row.get("descriptor", ""),
                    "license": row.get("license", ""),
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["stratum", "cluster_size"], ascending=[True, False])
    out.to_csv(args.output, sep="\t", index=False)
    n_core = (out["tier"] == "core").sum()
    print(
        f"wrote {len(out)} picks ({n_core} core) to {args.output}",
        file=sys.stderr,
    )
    print(out[["stratum", "tier", "megapixels", "corpus_rel_path"]].to_string(index=False), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--workdir", required=True)
    p.add_argument("--shards", type=int, default=6)
    p.add_argument("--print-extract-cmds", action="store_true")
    p.set_defaults(fn=cmd_prep)
    p = sp.add_parser("select")
    p.add_argument("--workdir", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(fn=cmd_select)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
