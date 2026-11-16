#!/usr/bin/env python3
"""Relocate completed evidence, verifying every byte before removing its source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('destination', type=Path)
parser.add_argument('journal', type=Path)
args = parser.parse_args()
if args.source.is_symlink() or not args.source.is_dir():
    parser.error('source must be a real directory of completed, immutable evidence')
source = args.source.resolve()
destination = args.destination.resolve()
journal_path = args.journal.resolve()
if source == destination or source in destination.parents or destination in source.parents:
    parser.error('source and destination must not overlap')
if journal_path == source or source in journal_path.parents:
    parser.error('journal must be outside the source tree')
destination.mkdir(parents=True, exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


with journal_path.open('x') as journal:
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise RuntimeError('unexpected symlink: ' + str(path))
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError('unexpected special file: ' + str(path))
        before = path.stat()
        digest = sha256(path)
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != digest:
                raise RuntimeError('existing destination differs: ' + str(target))
        else:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.partial-', delete=False) as stream:
                temporary = Path(stream.name)
            shutil.copy2(path, temporary)
            if sha256(temporary) != digest:
                raise RuntimeError('copied bytes differ: ' + str(path))
            os.link(temporary, target)  # Refuse a destination created concurrently.
            temporary.unlink()
        with target.open('rb') as verified:
            os.fsync(verified.fileno())
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('source changed while copying: ' + str(path))
        journal.write(json.dumps(dict(source=str(path), archive=str(target), sha256=digest, size=before.st_size)) + '\n')
        journal.flush()
        os.fsync(journal.fileno())
        path.unlink()
print('all files relocated after destination checksum verification', flush=True)
