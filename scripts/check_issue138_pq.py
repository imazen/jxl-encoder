#!/usr/bin/env python3
"""Replay every lossy, unprofiled PQ cell from a retained sweep with two binaries.

Both arms retain full diagnostics. No option, source, decoder or threshold is
changed by this driver. Expected unsupported-input refusals must agree.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("records", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("corpus", type=Path)
    p.add_argument("icc", type=Path)
    args = p.parse_args()
    contents = args.records.read_text()
    seed = re.search(r"^# conformance sweep, seed (\d+)$", contents, re.M)
    if seed is None:
        p.error("records must contain their seed header")
    seed = seed[1]
    records = list(csv.DictReader(
        (line for line in contents.splitlines() if not line.startswith("#")), delimiter="\t"))
    cases = [r for r in records if re.search(r"\bRgba?PqF32\b", r["cell"])
             and "lossless" not in r["cell"] and "icc=" not in r["cell"]]
    if not cases or len({r["index"] for r in cases}) != len(cases):
        p.error("empty or duplicate PQ selection")
    binaries = {"before": args.baseline.resolve(strict=True),
                "after": args.candidate.resolve(strict=True)}
    photos = args.corpus / "CID22/CID22-512/validation"
    screens = args.corpus / "gb82-sc"
    for directory in [photos, screens, args.icc]:
        if not directory.is_dir():
            p.error(f"missing input directory: {directory}")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"seed": seed, "records_sha256": digest(args.records),
                "binary_sha256": {k: digest(v) for k, v in binaries.items()},
                "indices": [r["index"] for r in cases]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    failures = []
    with (args.out / "pairs.tsv").open("x") as table:
        columns = ["seed", "index", "arm", "status", "bytes", "ours_err",
                   "cjxl_err", "source_sha256", "encoded_sha256", "cell", "detail"]
        writer = csv.DictWriter(table, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for case in cases:
            index = case["index"]
            paired = {}
            for arm, binary in binaries.items():
                out = args.out / f"{seed}-{index}-{arm}"
                command = [str(binary), "--images", str(photos), "--images", str(screens),
                           "--icc", str(args.icc), "--out", str(out), "--seed", seed,
                           "--only", index, "--threads", "1", "--retain", "--reference-study",
                           "--study-max-pixels", "100000000"]
                with out.with_suffix(".log").open("x") as log:
                    log.write(f"command: {command!r}\n")
                    log.flush()
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                rows = list(csv.DictReader(
                    (line for line in (out / "results.tsv").open() if not line.startswith("#")),
                    delimiter="\t"))
                if len(rows) != 1 or rows[0]["index"] != index or rows[0]["cell"] != case["cell"]:
                    raise RuntimeError(f"changed or incomplete cell: {out}")
                row = rows[0]
                if result.returncode != (1 if row["status"] == "FAIL" else 0):
                    raise RuntimeError(f"unexpected process exit: {out}: {result.returncode}")
                source = out / f"cell{index}.input"
                encoded = out / f"cell{index}.jxl"
                study = out / "reference_study.tsv"
                scores = [] if not study.exists() else list(csv.DictReader(
                    (line for line in study.open() if not line.startswith("#")), delimiter="\t"))
                if row["status"] != "REJECTED" and len(scores) != 1:
                    raise RuntimeError(f"missing reference measurement: {out}")
                score = scores[0] if scores else {}
                record = {"seed": seed, "index": index, "arm": arm, "status": row["status"],
                          "bytes": row["bytes"], "ours_err": score.get("ours_err", ""),
                          "cjxl_err": score.get("cjxl_err", ""),
                          "source_sha256": digest(source) if source.exists() else "",
                          "encoded_sha256": digest(encoded) if encoded.exists() else "",
                          "cell": row["cell"], "detail": row["detail"]}
                writer.writerow(record)
                table.flush()
                paired[arm] = record
                print(f"{seed}/{index} {arm}: {row['status']} {row['detail']}", flush=True)
            before, after = paired["before"], paired["after"]
            if before["source_sha256"] != after["source_sha256"]:
                raise RuntimeError(f"source mismatch: {index}")
            if after["status"] == "FAIL" or ((before["status"] == "REJECTED") !=
                                              (after["status"] == "REJECTED")):
                failures.append(index)
    summary = {"cells": len(cases), "required": len(cases), "failed_indices": failures}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
