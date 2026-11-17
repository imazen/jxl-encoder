#!/usr/bin/env python3
"""Verify retained passing objects and union matching source-color function bodies."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('llvm', type=Path)
p.add_argument('output', type=Path)
for scope in ('source-white', 'source-metric', 'rate-color'):
    p.add_argument('--' + scope, nargs='+', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
scripts = Path(__file__).resolve().parent
reports = {}
identities = []
for scope in ('source-white', 'source-metric', 'rate-color'):
    archives = getattr(a, scope.replace('-', '_'))
    pairs = []
    for archive in archives:
        if archive not in reports:
            identity = json.loads((archive / 'identity.json').read_text())
            assert identity['exit_code'] == 0, archive
            for row in identity['sources']:
                data = (archive / 'source' / row['path']).read_bytes()
                assert hashlib.sha256(data).hexdigest() == row['sha256'], (archive, row['path'])
            target = a.output / ('input-' + str(len(reports)))
            subprocess.run([sys.executable, str(scripts / 'coverage_from_archive.py'),
                            str(a.llvm), str(archive), str(target)], check=True)
            reports[archive] = target / 'selected.json'
            identities.append(dict(archive=str(archive), binary_sha256=identity['binary_sha256'],
                                   profile_sha256=identity['profile_sha256'], exit_code=0))
        pairs += ['--pair', str(archive), str(reports[archive])]
    subprocess.run([sys.executable, str(scripts / 'alpha_squeeze_coverage_summary.py'),
                    str(a.output / (scope + '.json')), '--scope', scope, *pairs], check=True)
(a.output / 'inputs.json').write_text(json.dumps(identities, indent=2))
