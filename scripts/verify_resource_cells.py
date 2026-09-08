#!/usr/bin/env python3
"""Repeat a saved resource grid with both decoders and exact bitstream checks.

Uses the existing resource-cell recipe. Each cell retains the bitstream and
complete encoder/time/decoder logs. Run under the workstation resource wrapper.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("baseline", type=Path)
parser.add_argument("inputs", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--build-commit", required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
rows = list(csv.DictReader(args.baseline.open(), delimiter="\t"))
assert rows, "baseline must contain cells"
assert re.fullmatch(r"[0-9a-f]{40}", args.build_commit), "full build commit required"
binary = Path("target/release/examples/mem_grid_probe")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(args.output / "provenance.json").write_text(json.dumps({
    "build_commit": args.build_commit, "hostname": platform.node(),
    "platform": platform.platform(), "binary_sha256": sha(binary),
    "baseline_sha256": sha(args.baseline),
    "measurement": "/usr/bin/time -l maximum resident set size",
    "inputs": {p.name: sha(p) for p in sorted(args.inputs.glob("*.ppm"))},
}, indent=2) + "\n")
with (args.output / "results.tsv").open("x") as table:
    writer = csv.DictWriter(table, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    table.flush()
    for index, row in enumerate(rows, 1):
        assert float(row["distance"]) == 4, "resource-cell currently fixes distance=4"
        source = args.inputs / (row["cell"].split("-")[0] + ".ppm")
        cell = args.output / row["cell"]
        print(f"{index}/{len(rows)} {row['cell']}", flush=True)
        with (args.output / (row["cell"] + ".command.log")).open("x") as log:
            subprocess.run(["just", "resource-cell", str(source), row["mode"],
                row["effort"], row["threads"], str(cell), row["budget"]],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        text = (cell / "encode.log").read_text()
        metrics = dict(re.findall(r"(\w+)=([^\s]+)", text.splitlines()[0]))
        measured = {name: metrics[name] for name in row if name in metrics}
        measured.update(cell=row["cell"],
            rss_bytes=re.search(r"(\d+)\s+maximum resident set size", text).group(1),
            encoded_sha256=sha(cell / "encoded.jxl"))
        writer.writerow(measured)
        table.flush()
        assert measured["ok"] == "1", measured
        assert measured["encoded_sha256"] == row["encoded_sha256"], measured
        assert int(measured["rss_bytes"]) <= int(measured["est_max_kb"]) * 1024, measured
        print(f"  byte-identical; both decoders passed; RSS {measured['rss_bytes']} bytes", flush=True)
