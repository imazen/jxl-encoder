#!/usr/bin/env python3
"""Export retained progressive objects and report exact bug windows/caller reach."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('llvm', type=Path)
p.add_argument('artifacts', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
scripts = Path(__file__).resolve().parent
names = [
    'progressive-ac-token-diagnostics',
    'progressive-ac-callers-first-fix',
    'progressive-ac-engines-typed',
    'progressive-dispatch-first-fix',
    'progressive-ac-reference-regular-fields',
    'progressive-ac-rate-control-export',
]
reports = {}
inputs = []
for name in names:
    archive = a.artifacts / name
    identity = json.loads((archive / 'identity.json').read_text())
    assert identity['exit_code'] == 0, (name, identity.get('exit_code'))
    for row in identity['sources']:
        source = archive / 'source' / row['path']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256'], source
    output = a.output / name
    subprocess.run([sys.executable, str(scripts / 'coverage_from_archive.py'),
                    str(a.llvm), str(archive), str(output)], check=True)
    reports[name] = output / 'selected.json'
    inputs.append(dict(archive=str(archive), exit_code=identity['exit_code'],
                       pid=identity['pid'], binary_sha256=identity['binary_sha256'],
                       profile_sha256=identity['profile_sha256']))
command = [sys.executable, str(scripts / 'alpha_squeeze_coverage_summary.py'),
           str(a.output / 'ac-functions.json'), '--scope', 'ac']
for name in names[:3]:
    command += ['--pair', str(a.artifacts / name), str(reports[name])]
subprocess.run(command, check=True)

rows = []
def observe(name, relative, function, first=None, last=None):
    source = a.artifacts / name / 'source' / relative
    lines = source.read_text().splitlines()
    command = [sys.executable, str(scripts / 'coverage_function_regions.py'),
               str(reports[name]), function, '--file-suffix', relative]
    bounds = None
    if first:
        starts = [i for i, line in enumerate(lines) if first in line]
        assert len(starts) == 1, (name, first, starts)
        start = starts[0]
        end = next(i for i in range(start, len(lines)) if last in lines[i])
        bounds = (start + 1, end + 1)
        command += ['--start-line', str(start + 1), '--end-line', str(end + 1)]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    assert not result.stderr, result.stderr
    observed = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(observed) == 1, observed
    row = dict(archive=name, source=relative,
               source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               bounds=bounds, **observed[0])
    rows.append(row)
    print(json.dumps(row))

observe('progressive-dispatch-first-fix', 'jxl-encoder/src/vardct/encoder.rs',
        'encode_inner', 'let single_pass_representable =', '&& !self.dc_tree_learning;')
observe('progressive-ac-reference-regular-fields', 'jxl-encoder/src/api/animate.rs',
        'encode_animation_lossy', 'enc.progressive = if frame.reference_only', '};')
for function in ['compute_with_budget_and_buffering',
                 'fill_dc_group_state_whole_image', 'fill_dc_group_state_per_region']:
    observe('progressive-ac-rate-control-export',
            'jxl-encoder/src/vardct/precomputed.rs', function)
observe('progressive-ac-rate-control-export', 'jxl-encoder/src/vardct/rate_control.rs',
        'encode_with_rate_control')
(a.output / 'windows-and-callers.json').write_text(json.dumps(rows, indent=2))
(a.output / 'inputs.json').write_text(json.dumps(inputs, indent=2))
