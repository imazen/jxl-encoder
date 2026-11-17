#!/usr/bin/env python3
"""Export coverage from a checksum-preserved completed-test archive."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('llvm_directory', type=Path)
p.add_argument('archive', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--source-root', type=Path)
p.add_argument('--source', action='append', default=[], type=Path)
a = p.parse_args()
if a.source and not a.source_root:
    p.error('--source-root is required for annotated source output')
for source in a.source:
    if source.is_absolute() or '..' in source.parts or not (a.archive / 'source' / source).is_file():
        p.error('required frozen source missing: ' + str(source))
rows = json.loads((a.archive / 'originals.json').read_text())
objects = []
profiles = []
for row in rows:
    path = a.archive / Path(row['staged']).name
    if not path.is_file():
        p.error('missing archived input: ' + str(path))
    if row['object']:
        objects.append(path)
    elif path.suffix == '.profraw':
        profiles.append(path)
if not objects or not profiles:
    p.error('archive must contain both completed profiles and exact objects')
a.output.mkdir(exist_ok=False)
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
