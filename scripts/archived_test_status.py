#!/usr/bin/env python3
"""Read a retained test's live PID and selected per-case journals."""
import argparse
import json
import os
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
p.add_argument('pixels', type=Path)
p.add_argument('--first-run', type=int, required=True)
a = p.parse_args()
identity = json.loads((a.output / 'identity.json').read_text())
pid = identity['pid']
try:
    os.kill(pid, 0)
    alive = True
except ProcessLookupError:
    alive = False
command_matches = False
if alive:
    command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
    command_matches = all(os.fsencode(word) in command for word in identity['command'])
rows = []
for directory in a.pixels.glob('run-*'):
    number = int(directory.name.removeprefix('run-'))
    if number < a.first_run:
        continue
    journal = directory / 'levels.tsv'
    policy = directory / 'policy.txt'
    if not journal.is_file() or not policy.is_file():
        raise RuntimeError(f'incomplete case initialization: {directory}')
    lines = journal.read_text().splitlines()
    rows.append(dict(run=number, policy=policy.read_text(),
                     completed_case_records=sum('stage=complete' in line for line in lines),
                     last_record=lines[-1] if lines else None))
print(json.dumps(dict(output=str(a.output), binary_sha256=identity['binary_sha256'],
                     pid=pid, pid_alive=alive, command_matches=command_matches,
                     exit_code=identity['exit_code'], cases=sorted(rows, key=lambda row: row['run'])), indent=2))
