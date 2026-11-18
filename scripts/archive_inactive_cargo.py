#!/usr/bin/env python3
"""Verify a complete archive of inactive Cargo output before optional reclamation (Linux)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--reclaim', action='store_true')
a = p.parse_args()
source = a.source.resolve(); output = a.output.resolve()
assert (source / '.rustc_info.json').is_file()
assert (source / 'CACHEDIR.TAG').is_file() or (source / 'debug/.fingerprint').is_dir()
assert not output.is_relative_to(source)
assert Path('/proc').is_dir(), 'active-process verification currently requires Linux'
for proc in Path('/proc').iterdir():
    if not proc.name.isdecimal(): continue
    try:
        try:
            executable = (proc / 'exe').resolve(strict=True)
            assert not executable.is_relative_to(source), 'Cargo output contains a live executable'
        except (FileNotFoundError, PermissionError, ProcessLookupError): pass
        if (proc / 'comm').read_text().strip() not in ['cargo', 'rustc']: continue
        command = (proc / 'cmdline').read_bytes()
        environment = (proc / 'environ').read_bytes().split(b'\0')
        assert str(source).encode() not in command, 'Cargo output appears in a live build command'
        assert b'CARGO_TARGET_DIR=' + str(source).encode() not in environment, 'Cargo output is used by a live build'
    except (FileNotFoundError, PermissionError, ProcessLookupError): pass
output.mkdir(exist_ok=False)
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream, 'sha256').hexdigest()
files = {str(path.relative_to(source)):sha(path) for path in source.rglob('*') if path.is_file() and not path.is_symlink()}
(output / 'files.json').write_text(json.dumps(files, indent=2) + '\n')
with tarfile.open(output / 'target.tar', 'w') as archive: archive.add(source, arcname='target')
with tarfile.open(output / 'target.tar', 'r') as archive:
    seen = set()
    for member in archive:
        if not (member.isfile() or member.islnk()): continue
        name = member.name.removeprefix('target/')
        assert hashlib.file_digest(archive.extractfile(member), 'sha256').hexdigest() == files[name], name
        seen.add(name)
    assert seen == set(files)
assert all(sha(source / name) == digest for name, digest in files.items())
(output / 'verified.json').write_text(json.dumps({'source':str(source), 'files':len(files),
    'archive_sha256':sha(output / 'target.tar'), 'reclaimed':a.reclaim}, indent=2) + '\n')
if a.reclaim: shutil.rmtree(source)
print('Verified complete Cargo-output archive' + (' and reclaimed its inactive source.' if a.reclaim else '.'))
