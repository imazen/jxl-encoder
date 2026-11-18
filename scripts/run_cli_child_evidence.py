#!/usr/bin/env python3
"""Select the exact CLI child of a retained Cargo build for a driver replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('archive',type=Path)
p.add_argument('runner_arguments',nargs=argparse.REMAINDER)
a=p.parse_args()
rows=json.loads((a.archive/'subprocess-objects.json').read_text())
rows=[row for row in rows if Path(row['original']).name in ('cjxl-rs','cjxl-rs.exe')]
assert len(rows)==1
row=rows[0]
for path in [Path(row['original']),Path(row['staged'])]:
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],path
args=a.runner_arguments
if args and args[0]=='--':args=args[1:]
assert args
result=subprocess.run([sys.executable,str(Path(__file__).with_name('run_selected_test_evidence.py')),*args],env=dict(os.environ,JXL_FRAME_LEVEL_CLI=row['original']))
assert hashlib.sha256(Path(row['original']).read_bytes()).hexdigest()==row['sha256'],'CLI changed during driver build/replay'
sys.exit(result.returncode)
