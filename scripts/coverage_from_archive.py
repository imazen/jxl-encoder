#!/usr/bin/env python3
"""Export coverage from a checksum-preserved completed-test archive."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('llvm_directory', type=Path)
p.add_argument('archive', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--source-root', type=Path)
p.add_argument('--failed-test', action='store_true',
               help='explicitly export a completed failed test, retaining its nonzero verdict')
p.add_argument('--source', action='append', default=[], type=Path)
a = p.parse_args()
if a.source and not a.source_root:
    p.error('--source-root is required for annotated source output')
for source in a.source:
    if source.is_absolute() or '..' in source.parts or not (a.archive / 'source' / source).is_file():
        p.error('required frozen source missing: ' + str(source))
if a.failed_test:
    identity = json.loads((a.archive / 'identity.json').read_text())
    if identity.get('exit_code') in (None, 0):
        p.error('--failed-test requires a recorded completed nonzero test verdict')
    rows = [dict(staged='test-binary', object=True, sha256=identity['binary_sha256']),
            dict(staged='test.profraw', object=False, sha256=identity['profile_sha256'])]
else:
    identity = json.loads((a.archive / 'identity.json').read_text())
    if identity.get('exit_code') != 0:
        p.error('positive coverage requires a recorded completed passing verdict')
    originals = a.archive / 'originals.json'
    if originals.is_file():
        rows = json.loads(originals.read_text())
    else:
        # Remote archived-ELF replays retain the same exact object/profile
        # identities, without the original build machine's filesystem paths.
        rows = [dict(staged='test-binary', object=True, sha256=identity['binary_sha256']),
                dict(staged='test.profraw', object=False, sha256=identity['profile_sha256'])]
objects = []
profiles = []
for row in rows:
    path = a.archive / Path(row['staged']).name
    if not path.is_file():
        p.error('missing archived input: ' + str(path))
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        p.error('archived input checksum mismatch: ' + str(path))
    if row['object']:
        objects.append(path)
    elif path.suffix == '.profraw':
        profiles.append(path)
if not objects or not profiles:
    p.error('archive must contain both completed profiles and exact objects')
a.output.mkdir(exist_ok=False)
if a.failed_test:
    (a.output / 'test-verdict.json').write_text(json.dumps(
        dict(exit_code=identity['exit_code'], pid=identity['pid'],
             binary_sha256=identity['binary_sha256'],
             note='Coverage of a failed test; not a passing validation verdict'), indent=2))
profile = a.output / 'merged.profdata'
subprocess.run([str(a.llvm_directory / 'llvm-profdata'), 'merge', '-sparse',
                *map(str, profiles), '-o', str(profile)], check=True)
subprocess.run([sys.executable, str(Path(__file__).with_name('export_selected_coverage.py')),
                str(a.llvm_directory / 'llvm-cov'), str(profile),
                str(a.output / 'selected.json'), *map(str, objects)], check=True)

if a.source:
    command = [str(a.llvm_directory / 'llvm-cov'), 'show', str(objects[0]),
               '--instr-profile=' + str(profile), '--show-line-counts',
               '--show-instantiations=false',
               '--path-equivalence=' + str(a.source_root) + ',' + str(a.archive / 'source')]
    command += ['--object=' + str(path) for path in objects[1:]]
    command += ['--sources', *[str(a.source_root / path) for path in a.source]]
    with (a.output / 'source-lines.txt').open('x') as stream:
        subprocess.run(command, stdout=stream, check=True)
