#!/usr/bin/env python3
"""Stage only the SIMD crate for checks without unrelated sibling dependencies."""
import argparse
import re
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('destination', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
crate = root.name + '-simd'
dest = args.destination.resolve()
if dest.exists():
    raise SystemExit(f'Refusing to overwrite existing stage: {dest}')
dest.mkdir(parents=True)
shutil.copytree(root / crate, dest / crate, ignore=shutil.ignore_patterns('target'))
manifest = (root / 'Cargo.toml').read_text()
sections = re.findall(r'(?ms)^\[(workspace\.(?:package|lints\.[^\]]+))\]\n(.*?)(?=^\[|\Z)', manifest)
(dest / 'Cargo.toml').write_text('[workspace]\nmembers = ["' + crate + '"]\nresolver = "2"\n\n' + ''.join('[' + name + ']\n' + body + '\n' for name, body in sections))
# Preserve the resolved dependency versions; Cargo prunes unrelated packages.
shutil.copy2(root / 'Cargo.lock', dest / 'Cargo.lock')
for pattern in ['LICENSE*', 'README*.md']:
    for file in root.glob(pattern):
        shutil.copy2(file, dest / file.name)
shutil.copytree(root / 'apidoc', dest / 'apidoc', ignore=shutil.ignore_patterns('target', '.git'))
print(dest)
