#!/usr/bin/env python3
"""Select the imazen-26 lossless benchmark set via per-stratum k-means.

Implements the sweep-discipline stratification rule for the lossless perf
bench refresh (2026-06-10): pick centroid-nearest representatives per
content-class stratum using k-means on zenanalyze `feat_*` embeddings,
instead of hand-picking or random-sampling (which over-represents the
modal class — 8100-web-screenshots alone is 370 of 1603 candidates).

Candidate pool is per-stratum PNG or JPEG, <=16 MP (keeps e9 1T bench
cells tractable). HEIC/DNG classes are excluded — nothing in-pipeline
decodes them. JPEG strata exist because the photographic classes are
jpg-only; their *picks* are pre-decoded once to stripped PNGs (the
`materialize` step) because default-features cjxl-rs reads only PNG, and
jpeg-reencoding builds auto-route .jpg to the JPEG->JXL transcode path —
a different code path than the modular tree-learner this bench measures.
Caveat recorded in the .meta: decoded-JPEG pixels carry 8x8 block
structure + quantization noise, not camera-native statistics.

Pipeline (four subcommands, run in order):

  1. prep    — read CORPUS-MANIFEST.tsv, filter (per-stratum format,
               <=16 MP, known folder), write candidates.tsv + N extractor
               manifest shards for zenanalyze's
               extract_features_for_picker. Incremental: candidates whose
               features already exist in the workdir's features_*.tsv are
               skipped (re-runs only shard NEW candidates).
  2. (run the extractor on each shard — see --print-extract-cmds)
  3. select  — concat shard features, append feat_zz_log2_mp (native
               megapixels; gives k-means a size axis inside strata with
               mixed sizes, auto-dropped as zero-variance on uniform
               strata), run cluster_sources.py per stratum, merge picks,
               compute sha256 of the chosen files, emit the set TSV +
               clusters JSON.
  4. materialize — decode every jpg pick to a metadata-stripped 8-bit PNG
               under /mnt/v/input/jxl-encoder/lossless-bench-imazen26-png/
               (ImageMagick convert -strip, no auto-orient), fill the
               bench_input + bench_sha256 columns in the set TSV. PNG
               picks pass through with bench_input = corpus path.

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

# stratum -> (corpus folders, K picks, source format). Folders merged into
# one stratum share one k-means pool (photos-png: the 10 stray PNGs in two
# photo classes — too few to cluster separately). A folder may appear in
# one png stratum AND one jpg stratum (different candidate pools).
STRATA = {
    # v1 strata (2026-06-10, PNG): docs / screens / plots / AI classes.
    "photos-png": (["1200-lilith-interiors", "1400-lilith-nature"], 2, "png"),
    "nps-brochures": (["5000-national-park-service-brochures"], 2, "png"),
    "epa-report": (["5200-epa-climate-impact-2021-report"], 1, "png"),
    "noaa-documents": (["5300-noaa-hurricane-documents"], 2, "png"),
    "patents": (["6000-lilith-scans-public-patents"], 2, "png"),
    "manuscript-illustrations": (["6600-ia-scans-manuscript-illustrations"], 2, "png"),
    "manuscript-text": (["6800-ia-scans-manuscript-text"], 2, "png"),
    "plots": (["7000-lilith-plots"], 3, "png"),
    "mobile-screenshots": (["8000-lilith-mobile-screenshots"], 2, "png"),
    "web-screenshots": (["8100-lilith-web-screenshots"], 5, "png"),
    "ai-clipart": (["9000-lilith-ai-clipart"], 2, "png"),
    "ai-illustrations": (["9094-lilith-ai-illustrations"], 2, "png"),
    "ai-products": (["9226-lilith-ai-products"], 3, "png"),
    # v2 strata (2026-06-10, JPEG): the photographic classes — jpg-only in
    # the corpus; picks are PNG-materialized for the bench (see step 4).
    "photos-general": (["1000-lilith-photos-general"], 2, "jpg"),
    "photos-interiors": (["1200-lilith-interiors"], 2, "jpg"),
    "photos-nature": (["1400-lilith-nature"], 2, "jpg"),
    "photos-food": (["1600-lilith-food"], 1, "jpg"),
    "photos-people": (["2000-unsplash-people"], 1, "jpg"),
    "renders": (["2200-unsplash-renders"], 1, "jpg"),
    "textures": (["2400-unsplash-textures"], 1, "jpg"),
    "museum-aic": (["3000-art-institute-of-chicago-photos"], 1, "jpg"),
    "museum-met": (["3300-met-museum-photos"], 1, "jpg"),
    "patents-gray-jpg": (["6000-lilith-scans-public-patents"], 1, "jpg"),
}

MATERIALIZE_DIR = Path("/mnt/v/input/jxl-encoder/lossless-bench-imazen26-png")


def stratum_assignments():
    """(folder, format) -> stratum. A folder may appear in one png stratum
    AND one jpg stratum; the (folder, format) pair must be unique."""
    m = {}
    for stratum, (folders, _k, fmt) in STRATA.items():
        for f in folders:
            key = (f, fmt)
            assert key not in m, f"duplicate stratum assignment for {key}"
            m[key] = stratum
    return m


def load_candidates():
    df = pd.read_csv(CORPUS / "CORPUS-MANIFEST.tsv", sep="\t", dtype=str)
    df["width"] = df["width"].astype(int)
    df["height"] = df["height"].astype(int)
    df["bytes"] = df["bytes"].astype(int)
    df["px"] = df["width"] * df["height"]
    m = stratum_assignments()
    df["stratum"] = [m.get((f, fmt)) for f, fmt in zip(df["folder"], df["format"])]
    df = df[(df["px"] <= MP_CAP_PX) & df["stratum"].notna()].copy()
    df["abs_path"] = df["path"].map(lambda p: str(CORPUS / p))
    return df


def cmd_prep(args):
    out = Path(args.workdir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_candidates()
    df.to_csv(out / "candidates.tsv", sep="\t", index=False)
    print(f"candidates: {len(df)} images, {df['bytes'].sum() / 1e6:.0f} MB", file=sys.stderr)
    print(df.groupby("stratum").size().to_string(), file=sys.stderr)

    # Incremental: skip candidates whose features were already extracted in
    # a prior run (matched on absolute path in any features_*.tsv).
    done = set()
    for p in sorted(out.glob("features_*.tsv")):
        done.update(pd.read_csv(p, sep="\t", dtype=str, usecols=["image_path"])["image_path"])
    todo = df[~df["abs_path"].isin(done)]
    print(f"{len(done)} already extracted; {len(todo)} new to extract", file=sys.stderr)

    # Extractor manifest shards (columns read by name in the extractor).
    tag = f"{args.tag}_" if args.tag else ""
    n = args.shards
    for i in range(n):
        shard = todo.iloc[i::n]
        m = pd.DataFrame(
            {
                "sha256": "",
                "split": "bench",
                "content_class": shard["stratum"],
                "source": "imazen-26",
                "path": shard["abs_path"],
            }
        )
        m.to_csv(out / f"extract_manifest_{tag}{i}.tsv", sep="\t", index=False)
    print(f"wrote {n} extractor manifest shards (tag={tag!r}) to {out}", file=sys.stderr)
    if args.print_extract_cmds:
        for i in range(n):
            print(
                f"<extractor-binary> --manifest {out}/extract_manifest_{tag}{i}.tsv "
                f"--output {out}/features_{tag}{i}.tsv --sizes 1024"
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
    for stratum, (_folders, k, fmt) in STRATA.items():
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
                    "source_format": fmt,
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


def cmd_materialize(args):
    """Decode jpg picks to metadata-stripped PNGs; fill bench_input columns.

    ImageMagick `convert <src> -strip <dst>`: no auto-orient (pixels stay in
    sensor order), no ICC/EXIF carried into the PNG (the encoder's metadata
    paths stay out of the measured encode). PNG picks pass through with
    bench_input = the corpus file itself.
    """
    out = pd.read_csv(args.output, sep="\t", dtype=str)
    MATERIALIZE_DIR.mkdir(parents=True, exist_ok=True)
    bench_inputs, bench_shas = [], []
    for _, r in out.iterrows():
        src = CORPUS / r["corpus_rel_path"]
        if r["source_format"] == "png":
            bench_inputs.append(str(src))
            bench_shas.append(r["sha256"])
            continue
        dst = MATERIALIZE_DIR / (Path(r["corpus_rel_path"]).stem + ".png")
        if not dst.exists():
            subprocess.run(["convert", str(src), "-strip", str(dst)], check=True)
            print(f"materialized {dst.name}", file=sys.stderr)
        bench_inputs.append(str(dst))
        bench_shas.append(sha256_file(dst))
    out["bench_input"] = bench_inputs
    out["bench_sha256"] = bench_shas
    out.to_csv(args.output, sep="\t", index=False)
    n_jpg = (out["source_format"] != "png").sum()
    print(f"materialized/verified {n_jpg} jpg picks into {MATERIALIZE_DIR}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--workdir", required=True)
    p.add_argument("--shards", type=int, default=6)
    p.add_argument("--tag", default="", help="suffix tag for shard filenames (incremental runs)")
    p.add_argument("--print-extract-cmds", action="store_true")
    p.set_defaults(fn=cmd_prep)
    p = sp.add_parser("select")
    p.add_argument("--workdir", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(fn=cmd_select)
    p = sp.add_parser("materialize")
    p.add_argument("--output", required=True, help="set TSV to update in place")
    p.set_defaults(fn=cmd_materialize)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
