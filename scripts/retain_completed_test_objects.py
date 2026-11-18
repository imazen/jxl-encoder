#!/usr/bin/env python3
"""Retain Cargo-listed ELF test objects without treating their presence as a pass."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('qualification',type=Path);p.add_argument('target',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(exist_ok=True)
def sha(path):
 with path.open('rb') as file:
  return hashlib.file_digest(file,'sha256').hexdigest()
text=re.sub(r'\x1b\[[0-9;]*m','',(a.qualification/'checks.log').read_text())
paths=sorted(set(re.findall(r'Running [^\n]*\((/[^\n()]+)\)',text)))
rows=[]
for name in paths:
 source=Path(name);assert source.resolve().is_relative_to(a.target.resolve()),source
 with source.open('rb')as file:assert file.read(4)==b'\x7fELF',source
 digest=sha(source);dest=a.output/source.name
 if dest.exists():assert sha(dest)==digest,source
 else:shutil.copy2(source,dest)
 assert sha(source)==digest and sha(dest)==digest,source
 rows.append(dict(original=str(source),retained=dest.name,sha256=digest,bytes=dest.stat().st_size))
assert rows,'no actual Cargo-listed test objects'
record=dict(observed=datetime.now(timezone.utc).isoformat(),qualification=str(a.qualification),source_identity_sha256=sha(a.qualification/'production.json'),objects=rows,test_result_lines=[line for line in text.splitlines()if line.startswith('test result:')],qualification_verdict=json.loads((a.qualification/'verdict.json').read_text())if(a.qualification/'verdict.json').exists()else None,note='Object/source identities only. A listed or retained executable does not establish completed tests or coverage.')
(a.output/'identity.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(objects=len(rows),qualification_complete=record['qualification_verdict']is not None)))
