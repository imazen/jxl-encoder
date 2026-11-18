#!/usr/bin/env python3
"""Stage a retained ELF with a checksum-bound existing Linux runtime closure."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('archive', type=Path)
p.add_argument('template', type=Path, help='manifest of an already verified runtime closure')
p.add_argument('output', type=Path)
a = p.parse_args()

def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

identity = json.loads((a.archive / 'identity.json').read_text())
binary = a.archive / 'test-binary'
assert digest(binary) == identity['binary_sha256']
rows = json.loads(a.template.read_text())
assert len({row['name'] for row in rows}) == len(rows)
assert sum(row['name'] == 'test-binary' for row in rows) == 1
sources = []
for row in rows:
    name = Path(row['name'])
    assert not name.is_absolute() and '..' not in name.parts
    source = binary if row['name'] == 'test-binary' else Path(row['source'])
    expected = identity['binary_sha256'] if row['name'] == 'test-binary' else row['sha256']
    assert digest(source) == expected, source
    sources.append((row['name'], source, expected))
# A new ELF must not acquire dependencies absent from the template.
linked = subprocess.run(['readelf', '-d', str(binary)], capture_output=True, text=True, check=True)
program = subprocess.run(['readelf', '-l', str(binary)], capture_output=True, text=True, check=True)
needed = re.findall(r'\(NEEDED\).*\[([^\]]+)\]', linked.stdout)
loader = re.search(r'Requesting program interpreter:\s*([^\]]+)\]', program.stdout)
assert needed and loader
for dependency in [*needed, Path(loader[1]).name]:
    assert any(name == 'lib/' + dependency for name, _, _ in sources), dependency
a.output.mkdir(exist_ok=False)
for name, source, expected in sources:
    target = a.output / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert digest(target) == expected
(a.output / 'manifest.json').write_text(json.dumps([
    dict(name=name, source=str(source), sha256=expected) for name, source, expected in sources
], indent=2))
shutil.copy2(Path(__file__), a.output / 'stager.py')
(a.output / 'elf-dependencies.txt').write_text(linked.stdout + program.stdout)
print('Staged exact ELF and verified its required runtime dependencies.')
