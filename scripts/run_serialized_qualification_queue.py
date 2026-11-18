#!/usr/bin/env python3
"""Run explicit independent qualifications serially, retaining every failing verdict."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('plan', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
rows = []
for item in json.loads(a.plan.read_text()):
    row = dict(name=item['name'], command=item['command'], started=datetime.now(timezone.utc).isoformat())
    if gate := item.get('require_runtime_failure'):
        archive = Path(gate['archive'])
        identity_path = archive / 'identity.json'
        valid = identity_path.is_file()
        if valid:
            identity = json.loads(identity_path.read_text())
            diagnostics = gate.get('diagnostics', [gate.get('diagnostic', '')])
            valid = (identity.get('exit_code') == 101
                     and all(message in (archive / 'test.log').read_text() for message in diagnostics))
        row['before_runtime_gate'] = dict(archive=str(archive), satisfied=valid)
        if not valid:
            row.update(exit_code=2, finished=datetime.now(timezone.utc).isoformat())
            rows.append(row)
            (a.output / 'results.partial.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(item['name'] + ': before runtime gate unsatisfied; candidate not applied', flush=True)
            continue
    if gate := item.get('require_qualification_success'):
        path = Path(gate)
        valid = path.is_file() and json.loads(path.read_text()).get('exit_code') == 0
        row['qualification_gate'] = dict(verdict=str(path), satisfied=valid)
        if not valid:
            row.update(exit_code=2, finished=datetime.now(timezone.utc).isoformat())
            rows.append(row)
            (a.output / 'results.partial.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(item['name'] + ': prerequisite qualification failed; candidate not applied', flush=True)
            continue
    with (a.output / (item['name'] + '.log')).open('x') as log:
        child = subprocess.Popen(item['command'], stdout=log, stderr=subprocess.STDOUT)
        row['pid'] = child.pid
        (a.output / 'active.json').write_text(json.dumps(row, indent=2) + '\n')
        row['exit_code'] = child.wait()
    row['finished'] = datetime.now(timezone.utc).isoformat()
    rows.append(row)
    (a.output / 'results.partial.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(item['name'] + ': rc=' + str(row['exit_code']), flush=True)
(a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
sys.exit(1 if any(row['exit_code'] for row in rows) else 0)
