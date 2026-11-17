#!/usr/bin/env python3
"""Compare retained same-named streams, preserving file and codestream hashes."""
import argparse
import hashlib
import json
from pathlib import Path

from jxl_container import codestream

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('before', type=Path)
p.add_argument('after', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
before = {f.name: f for f in a.before.glob('*.jxl')}
after = {f.name: f for f in a.after.glob('*.jxl')}
assert before and before.keys() == after.keys(), 'empty or mismatched stream sets'
rows = []
for name in sorted(before):
    left, right = before[name].read_bytes(), after[name].read_bytes()
    left_code, _ = codestream(left)
    right_code, _ = codestream(right)
    digest = lambda value: hashlib.sha256(value).hexdigest()
    rows.append(dict(name=name, before=str(before[name]), after=str(after[name]),
                     before_sha256=digest(left), after_sha256=digest(right),
                     before_codestream_sha256=digest(left_code),
                     after_codestream_sha256=digest(right_code),
                     file_equal=left == right, codestream_equal=left_code == right_code))
with a.output.open('x') as out:
    json.dump(rows, out, indent=2)
assert all(row['codestream_equal'] for row in rows), 'codestream payload changed'
print(f'{len(rows)} codestreams identical; {sum(row["file_equal"] for row in rows)} whole files identical')
