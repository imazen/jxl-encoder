#!/usr/bin/env python3
"""Require identical named JXL streams and preserve both measured hash tables."""
import argparse
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();assert not a.output.exists()
def table(root):
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob('input.jxl'))}
before=table(a.before);after=table(a.after)
assert before and after,'both caller-selected stream trees must be nonempty'
keys=set(before)|set(after)
rows=[dict(path=k,before_sha256=before.get(k),after_sha256=after.get(k),equal=before.get(k)==after.get(k)) for k in sorted(keys)]
a.output.write_text(json.dumps(dict(before=str(a.before),after=str(a.after),streams=rows,note='Exact identity only for these named outputs; no general coverage or performance inference.'),indent=2)+'\n')
assert all(row['equal']for row in rows),'different or missing streams; full comparison retained'
print(f'All {len(rows)} named output pairs are byte-identical; comparison retained.')
