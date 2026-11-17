#!/usr/bin/env python3
"""Select explicitly verified file contents in a jj diff editor's temporary tree."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('selection',type=Path)
p.add_argument('left',type=Path)
p.add_argument('right',type=Path)
a=p.parse_args()
if a.left.resolve()==a.right.resolve():p.error('diff trees must differ')
for name, row in json.loads(a.selection.read_text()).items():
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:p.error('unsafe relative selection path')
    left=a.left/relative
    right=a.right/relative
    assert hashlib.sha256(left.read_bytes()).hexdigest()==row['parent_sha256'],name
    content=Path(row['verified_source']).read_bytes()
    assert hashlib.sha256(content).hexdigest()==row['verified_sha256'],name
    right.parent.mkdir(parents=True,exist_ok=True)
    right.write_bytes(content)
print('selected verified file contents; remaining changes stay in the child change')
