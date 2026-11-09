#!/usr/bin/env python3
"""Run the planar loop gate against the recorded CI sibling revisions.

Reuse zenmetrics' tracked-tree exporter, including its content-aware Cargo
mtime handling. Sibling working copies are read only at exact commits.
"""
import argparse
import csv
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path)
parser.add_argument("--exporter", type=Path, required=True)
parser.add_argument("--update-lock", action="store_true")
a = parser.parse_args()
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("zen_snapshot", a.exporter)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parent = a.destination.resolve() / "zen"
snapshot = parent / "jxl-encoder"
module.export(root, snapshot, "")
with (root / ".github/sibling-revisions.tsv").open() as f:
    for row in csv.DictReader(f, delimiter="\t"):
        name = row["repository"].split("/")[1]
        source = root.parent / name
        if not (source / ".git").exists():
            source = root.parent.parent / name
        target = a.destination.resolve() / name if name in ("butteraugli", "dssim") else parent / name
        module.export(source, target, row["revision"])
env = os.environ.copy()
env["CARGO_TARGET_DIR"] = str(a.destination.resolve() / "target")
if a.update_lock:
    baseline = subprocess.check_output(
        ["jj", "file", "show", "-r", "main@origin", "Cargo.lock"], cwd=root
    )
    (snapshot / "Cargo.lock").write_bytes(baseline)
    subprocess.run(
        ["cargo", "check", "-p", "jxl-encoder", "--no-default-features", "--features", "psy-loop"],
        cwd=snapshot, env=env, check=True,
    )
    # --locked qualification below establishes that this exact lock resolves.
subprocess.run(["just", "psy-loop-check"], cwd=snapshot, env=env, check=True)
if a.update_lock:
    shutil.copyfile(snapshot / "Cargo.lock", root / "Cargo.lock")
print(f"qualified snapshot: {snapshot}", flush=True)
