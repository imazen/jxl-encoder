#!/usr/bin/env python3

"""Replay four photo quality rungs cold and with caller-owned output."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from measure_lossy import Study, PINS, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("paired_output", type=Path)
    args = parser.parse_args()
    run, output, reuse = (p.resolve() for p in (args.run, args.output, args.paired_output))
    if not (run / "complete").is_file():
        raise ValueError(f"incomplete warm replay: {run}")
    output.mkdir()
    manifest = json.loads((run / "manifest.json").read_text())
    bins = {name: Path(data["path"]) for name, data in manifest["binaries"].items()}
    for name, binary in bins.items():
        if digest(binary) != manifest["binaries"][name]["sha256"]:
            raise ValueError(f"wrong binary: {binary}")
    rungs = {5, 55, 90, 99}
    cases = defaultdict(list)
    with (run / "selected.tsv").open() as table:
        for row in csv.DictReader(table, delimiter="\t"):
            if row["size"] == "1024" and int(row["q"]) in rungs:
                cases[row["id"], row["size"], float(row["distance"])].append(row)
    if len(cases) != 8 or any(len(rows) != 3 for rows in cases.values()):
        raise ValueError("incomplete photo endpoint choices")
    full = json.loads((run / "finalists.json").read_text())
    selected = {key: full["-".join(map(str, key))] for key in cases}
    study = Study(SimpleNamespace(**bins, output=output/"cold", stage="ladder-cold", blocks=3), cohort={})
    pairs = defaultdict(list)
    pins = {}
    for key, arms in selected.items():
        base = arms[0][1]
        for _, row in arms:
            filename = row["file"]
            path = run / "streams" / filename
            if filename in pins:
                if pins[filename] != row["stream_sha256"]:
                    raise ValueError(f"inconsistent stream hash: {filename}")
                continue
            if digest(path) != row["stream_sha256"]:
                raise ValueError(f"wrong stream: {path}")
            pins[filename] = row["stream_sha256"]
            (study.streams / filename).symlink_to(path.resolve())
        for row in cases[key]:
            candidate = dict(arms)[row["timed_as"]]
            if (row["base_sha256"] != base["stream_sha256"]
                    or row["candidate_sha256"] != candidate["stream_sha256"]):
                raise ValueError(f"selection hash mismatch: {key} {row['profile']}")
            pairs[row["profile"]].append((row["id"], row["q"], base["file"], candidate["file"]))
    study.manifest["stream_sha256"] = pins
    study.manifest["quality_rungs"] = sorted(rungs)
    (study.out / "manifest.json").write_text(json.dumps(study.manifest, indent=2) + "\n")
    (study.out / "finalists.json").write_text(json.dumps({"-".join(map(str, key)): arms for key, arms in selected.items()}, indent=2) + "\n")
    (output / "reuse-manifest.json").write_text(
        json.dumps({"binary": str(reuse), "sha256": digest(reuse), "stream_sha256": pins, "pairs": pairs}, indent=2) + "\n")
    study.measure_time(selected, warmup=0)
    (study.out / "complete").touch(exist_ok=False)
    with (output / "reuse-commands.log").open("x", buffering=1) as log:
        for profile, items in pairs.items():
            for workers, cpus in PINS.items():
                cmd = ["taskset", "-c", cpus, str(reuse), "--workers", str(workers),
                       *(str(study.streams / base) for _, _, base, _ in items),
                       "--candidate-inputs", *(str(study.streams / cand) for _, _, _, cand in items)]
                print(json.dumps(cmd), file=log)
                print("caller-owned output", profile, workers, flush=True)
                with (output / f"reused-{profile}-{workers}.tsv").open("x", buffering=1) as dest:
                    subprocess.run(cmd, stdout=dest, stderr=log, check=True)
    (output / "complete").touch(exist_ok=False)


if __name__ == "__main__":
    main()
