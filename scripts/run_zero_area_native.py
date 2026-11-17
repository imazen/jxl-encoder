#!/usr/bin/env python3
"""Retain exact native component sources, executable and observed verdict."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('oracle', type=Path)
p.add_argument('output', type=Path)
p.add_argument("--durations", action="store_true")
a = p.parse_args()
a.output.mkdir(exist_ok=False)
source = Path(__file__).with_name('zero_area_oracle.cc')
records = []
for original, name in [(source, 'source.cc'), (a.oracle, 'oracle')]:
    target = a.output / name
    shutil.copy2(original, target)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    records.append(dict(original=str(original), retained=str(target), sha256=digest))
identity = dict(files=records, command=[str(a.oracle), str(a.output / 'observed')])
if a.durations:
    identity['command'].append('duration')
with (a.output / 'native.log').open('x') as log:
    child = subprocess.Popen(identity['command'], stdout=log, stderr=subprocess.STDOUT)
    identity['pid'] = child.pid
    (a.output / 'identity.json').write_text(json.dumps(identity, indent=2))
    identity['returncode'] = child.wait()
(a.output / 'identity.json').write_text(json.dumps(identity, indent=2))
raise SystemExit(identity['returncode'])
