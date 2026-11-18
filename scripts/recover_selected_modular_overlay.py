#!/usr/bin/env python3
"""Recover a terminated owned-source qualification after verifying every byte."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('archive', type=Path)
a = p.parse_args()
root = a.manifest.resolve().parent
assert not (root / '.git').exists() and (root.parent.parent / 'provenance.json').is_file()
assert not (a.archive / 'restored.json').exists()
rows = json.loads((a.archive / 'overlay.json').read_text())
last = {row['path']: row for row in rows}
first = {}
for row in rows:
    first.setdefault(row['path'], row)
digest = lambda b: hashlib.sha256(b).hexdigest()
# Check the complete recovery set before changing any source. A mismatched
# interrupted mutation or unknown change requires its own explicit recovery.
for name, row in last.items():
    assert digest((root / name).read_bytes()) == row['overlay_sha256'], name
for name, row in first.items():
    if row['before_sha256'] is not None:
        assert digest((a.archive / 'original' / name).read_bytes()) == row['before_sha256'], name
for name, row in first.items():
    path = root / name
    if row['before_sha256'] is None:
        path.unlink()
    else:
        data = (a.archive / 'original' / name).read_bytes()
        path.write_bytes(data)
        assert path.read_bytes() == data
(a.archive / 'recovery-restored.json').write_text(json.dumps(dict(
    restored=True, files=len(first),
    reason='Terminated qualifier; complete original and overlay hashes checked before recovery.'), indent=2) + '\n')
print('Recovered exact owned overlays: ' + str(len(first)), flush=True)
