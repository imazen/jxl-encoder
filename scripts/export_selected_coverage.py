#!/usr/bin/env python3
"""Export LLVM coverage from explicitly retained current binaries only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('llvm_cov', type=Path)
p.add_argument('profile', type=Path)
p.add_argument('output', type=Path)
p.add_argument('objects', nargs='+', type=Path)
a = p.parse_args()
for path in [a.llvm_cov, a.profile, *a.objects]:
    if not path.is_file(): p.error('required file missing: ' + str(path))
command = [str(a.llvm_cov), 'export', '--instr-profile=' + str(a.profile)]
command += ['--object=' + str(path) for path in a.objects]
# Refuse accidental evidence replacement, including a failed preceding export.
with a.output.open('x') as stream:
    subprocess.run(command, stdout=stream, check=True)
rows = []
for path in [a.profile, *a.objects]:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024): digest.update(block)
    rows.append(dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size))
with Path(str(a.output) + '.inputs.json').open('x') as stream:
    json.dump(dict(command=command, inputs=rows), stream, indent=2)
