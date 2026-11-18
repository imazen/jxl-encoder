#!/usr/bin/env python3
"""Run independent before probes serially; retain every failure as a failure."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest')
p.add_argument('output', type=Path)
p.add_argument('coverage_env')
a = p.parse_args()
a.output.mkdir(exist_ok=False)
commands = [
    ('lf-callers', ['just', 'lf-caller-before-check', a.manifest,
                    str(a.output / 'lf-callers'), a.coverage_env]),
    ('metric-admission', ['just', 'rate-control-metric-before-evidence', a.manifest,
                          str(a.output / 'metric-admission'), a.coverage_env]),
    ('minimal-rate-control', ['cargo', 'check', '--locked', '--manifest-path', a.manifest,
                             '-p', 'jxl-encoder', '--no-default-features',
                             '--features', 'std,rate-control,__expert', '--lib']),
]
rows = []
for name, command in commands:
    with (a.output / (name + '.log')).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    rows.append(dict(name=name, command=command, exit_code=result.returncode))
    (a.output / 'results.partial.json').write_text(json.dumps(rows, indent=2) + '\n')
(a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
# A before batch is never qualified by the fact that a failure was expected.
sys.exit(1 if any(row['exit_code'] for row in rows) else 0)
