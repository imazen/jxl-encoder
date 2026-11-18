#!/usr/bin/env python3
"""Bind unchanged committed support sources after completed retained replays.

This supplement records its actual attachment time. It does not modify original
archive identities or claim that sources were captured when the tests started.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('base')
p.add_argument('output', type=Path)
p.add_argument('archives', nargs='+', type=Path)
p.add_argument('--source', action='append', required=True)
p.add_argument('--native-identity', type=Path)
a = p.parse_args()
root = a.manifest.resolve().parent
assert not (root / '.git').exists() and (root.parent.parent / 'provenance.json').is_file()
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
sources = []
for name in a.source:
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    expected = subprocess.check_output(['git', 'show', a.base + ':' + name])
    assert (root / name).read_bytes() == expected, ('owned export differs', name)
    assert Path(name).read_bytes() == expected, ('working source differs', name)
    sources.append(dict(path=name, sha256=hashlib.sha256(expected).hexdigest()))
archives = []
for archive in a.archives:
    identity = json.loads((archive / 'identity.json').read_text())
    assert identity['exit_code'] == 0, archive
    assert digest(archive / 'test-binary') == identity['binary_sha256'], archive
    assert digest(archive / 'test.profraw') == identity['profile_sha256'], archive
    archives.append(dict(path=str(archive), identity_sha256=digest(archive / 'identity.json'),
                         binary_sha256=identity['binary_sha256'],
                         profile_sha256=identity['profile_sha256']))
native = None
native_loader = None
if a.native_identity:
    native = json.loads(a.native_identity.read_text())
    for row in native['files']:
        assert digest(Path(row['path'])) == row['sha256'], row['path']
    binary = Path(native['files'][0]['path'])
    with binary.open('rb') as stream:
        assert stream.read(4) == b'\x7fELF', 'native identity must begin with its Linux ELF'
    native_loader = subprocess.check_output(['ldd', str(binary)], text=True)
    loaded = {Path(path).resolve() for path in re.findall(r'=>\s+(\S+)\s+\(', native_loader)}
    for row in native['files']:
        if Path(row['path']).name.startswith(('libjxl.so', 'libjxl_cms.so')):
            assert Path(row['path']).resolve() in loaded, ('loader differs', row['path'])
    header_identity = Path(native['header_identity'])
    headers = json.loads(header_identity.read_text())
    for row in headers['headers']:
        assert digest(header_identity.parent / row['path']) == row['sha256'], row['path']
    native = dict(identity=native, identity_sha256=digest(a.native_identity),
                  headers=headers, headers_identity_sha256=digest(header_identity),
                  loader_observation='Current ldd resolution at attachment time; not a historical environment capture.')
a.output.mkdir(exist_ok=False)
for row in sources:
    target = a.output / 'source' / row['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / row['path'], target)
    assert digest(target) == row['sha256']
shutil.copy2(Path(__file__), a.output / 'binder.py')
if native_loader is not None:
    (a.output / 'native-loader.txt').write_text(native_loader)
(a.output / 'binding.json').write_text(json.dumps(dict(
    attached_at=datetime.now(timezone.utc).isoformat(), base=a.base,
    timing='Supplement attached after completed replay, not a contemporaneous source capture.',
    assertion='Named support sources match both the committed base and current owned export.',
    sources=sources, archives=archives, native=native,
    binder_sha256=digest(a.output / 'binder.py')), indent=2) + '\n')
print('Bound unchanged committed support; original replay identities remain unchanged.')
