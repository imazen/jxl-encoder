#!/usr/bin/env python3
"""Copy completed objects, profiles and selected source into a new evidence directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source_root', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--object', action='append', default=[], type=Path)
p.add_argument('--profile', action='append', default=[], type=Path)
p.add_argument('--source', action='append', default=[], type=Path)
p.add_argument('--source-directory', action='append', default=[], type=Path)
a = p.parse_args()
if not a.object:
    p.error('at least one completed executable is required')
sources = set(a.source)
for relative in a.source_directory:
    if relative.is_absolute() or '..' in relative.parts:
        p.error('source directories must be relative to source_root')
    sources.update(path.relative_to(a.source_root)
                   for path in (a.source_root / relative).rglob('*.rs'))
for relative in sources:
    if relative.is_absolute() or '..' in relative.parts:
        p.error('source files must be relative to source_root')
for path in [*a.object, *a.profile, *[a.source_root / path for path in sources]]:
    if not path.is_file():
        p.error('required completed input missing: ' + str(path))
a.output.mkdir(exist_ok=False)


def preserve(source, destination):
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
    after = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return digest


rows = []
for index, (source, is_object) in enumerate(
        [(path, True) for path in a.object] + [(path, False) for path in a.profile]):
    target = a.output / (f'{index:04}-' + source.name)
    digest = preserve(source, target)
    rows.append(dict(original=str(source.resolve()), staged=str(target.resolve()),
                     object=is_object, sha256=digest))
(a.output / 'originals.json').write_text(json.dumps(rows, indent=2))
source_rows = []
for relative in sorted(sources):
    digest = preserve(a.source_root / relative, a.output / 'source' / relative)
    source_rows.append(dict(path=str(relative), sha256=digest))
(a.output / 'sources.json').write_text(json.dumps(source_rows, indent=2))
print('preserved exact objects/profiles/source with checksums: ' + str(a.output), flush=True)
