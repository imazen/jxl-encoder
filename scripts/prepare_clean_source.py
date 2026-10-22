#!/usr/bin/env python3
"""Export an encoder revision and fetch its exact CI sibling source closure.

The destination must not exist. Existing working checkouts are never modified.
The caller builds with Cargo under its normal resource controls and preserves
Cargo.lock plus the returned source revision with the resulting artifacts.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("revision")
parser.add_argument("destination", type=Path)
args = parser.parse_args()
root = args.destination.resolve()
root.mkdir(parents=True, exist_ok=False)
checkout = root / "siblings" / "jxl-encoder"
checkout.mkdir(parents=True)
revision = subprocess.check_output(["git", "rev-parse", args.revision], text=True).strip()
archive = root / "encoder-source.tar"
with archive.open("xb") as output:
    subprocess.run(["git", "archive", revision], stdout=output, check=True)
with tarfile.open(archive) as source:
    source.extractall(checkout, filter="data")
# `git archive` stamps every file with the commit time, and Cargo hashes path
# packages relative to their workspace root. In a target dir shared with any
# other tree (the live checkout, another export) these older files would look
# unchanged, and Cargo would reuse artifacts compiled from different sources.
# Stamping them now covers the first build; give each export its own
# CARGO_TARGET_DIR, or re-stamp before every build, when trees alternate.
now = time.time()
for path in checkout.rglob("*"):
    if path.is_file():
        os.utime(path, (now, now))
# Execute the same clone logic CI uses; do not maintain a second pin resolver.
lines = (checkout / ".github/actions/clone-siblings/action.yml").read_text().splitlines()
start = next(i for i, line in enumerate(lines) if line == "      run: |") + 1
body = []
for line in lines[start:]:
    if line and not line.startswith("        "):
        break
    body.append(line[8:])
script = root / "clone-siblings.sh"
script.write_text("\n".join(body) + "\n")
with (root / "source-closure.log").open("x") as log:
    print(f"fetching pinned siblings; progress: {root / 'source-closure.log'}", flush=True)
    subprocess.run(["bash", str(script)], cwd=checkout,
                   env=dict(os.environ, GITHUB_WORKSPACE=str(checkout)),
                   stdout=log, stderr=subprocess.STDOUT, check=True)
(root / "provenance.json").write_text(json.dumps(dict(encoder_commit=revision,
    checkout=str(checkout), sibling_pins="siblings/jxl-encoder/.github/sibling-revisions.tsv"), indent=2) + "\n")
print(checkout, flush=True)
