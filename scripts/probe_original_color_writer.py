#!/usr/bin/env python3
"""Retain public-path failures with a frozen original writer in an owned export."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("manifest", type=Path)
p.add_argument("original", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
root = a.manifest.resolve().parent
if (root / '.git').exists() or not (root.parent.parent / 'provenance.json').is_file():
    p.error('requires an owned prepare_clean_source.py export')
if hashlib.sha256(a.original.read_bytes()).hexdigest() != '9322a49db03bd523d9191cfc4ee1fd4438afd03d45f0396a091a1d8ef690ae61':
    p.error('requires the exact original 6e3a1f14 color writer')
a.output.mkdir(exist_ok=False)
source = root / 'jxl-encoder/src/headers/color_encoding.rs'
fixed = source.read_bytes()
binary = Path(os.environ['CARGO_TARGET_DIR']) / 'debug/examples/color_description_probe'
def build(log_name):
    with (a.output / log_name).open('x') as log:
        subprocess.run(['cargo', 'build', '--locked', '--manifest-path', str(a.manifest),
                        '-p', 'jxl-encoder', '--features', '__expert', '--example', 'color_description_probe'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)

build('fixed-build.log')
shutil.copy2(binary, a.output / 'fixed-probe')
try:
    source.write_bytes(a.original.read_bytes())
    build('original-build.log')
    shutil.copy2(binary, a.output / 'original-writer-probe')
    metadata = {'component_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'fixed_component_sha256': hashlib.sha256(fixed).hexdigest(),
                'fixed_binary_sha256': hashlib.sha256((a.output / 'fixed-probe').read_bytes()).hexdigest(),
                'scope': 'Original color writer inside the current pinned encoder closure; not the complete baseline encoder'}
    for space in ['xyb', 'unknown']:
        with (a.output / f'{space}.log').open('x') as log:
            result = subprocess.run([str(binary), space, str(a.output / space)], stdout=log, stderr=subprocess.STDOUT)
        metadata[f'{space}_returncode'] = result.returncode
        assert result.returncode == 101, (space, result.returncode)
    (a.output / 'identity.json').write_text(json.dumps(metadata, indent=2) + '\n')
finally:
    source.write_bytes(fixed)
    build('restored-build.log')
