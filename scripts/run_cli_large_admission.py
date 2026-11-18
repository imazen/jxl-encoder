#!/usr/bin/env python3
"""Measure the existing CLI's large-input admission without changing its limits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('runtime', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
closure = json.loads((a.runtime / 'manifest.json').read_text())
for row in closure:
    assert sha(a.runtime / row['name']) == row['sha256'], row['name']
source = a.output / 'source-16384-square.ppm'
header = b'P6\n16384 16384\n255\n'
with source.open('xb') as file:
    file.write(header)
    file.truncate(len(header) + 16384 * 16384 * 3)
# A sparse zero-filled raster is a complete valid black PPM, not a truncated input.
assert source.stat().st_size == len(header) + 16384 * 16384 * 3
(a.output / 'source.json').write_text(json.dumps(dict(width=16384, height=16384,
    raster='complete zero-filled RGB8', bytes=source.stat().st_size, host=socket.gethostname(),
    runtime_closure=closure), indent=2) + '\n')
rows = []
for strategy in ['libjxl', 'libjxl-exact', 'libjxl-strict', 'lean-faster', 'zenjxl', 'aggressive']:
    for experimental in [False, True]:
        for lf in [False, True]:
            cell = a.output / f'{strategy}-experimental{experimental}-lf{lf}'
            cell.mkdir()
            destination = cell / 'output.jxl'
            sentinel = b'existing destination'
            destination.write_bytes(sentinel)
            command = [str(a.runtime / 'lib/ld-linux-x86-64.so.2'), '--library-path',
                str(a.runtime / 'lib'), str(a.runtime / 'test-binary'), str(source), str(destination),
                '--rate-control', '--rc-iterations', '0', '--effort', '1', '--distance', '1',
                '--dct8-only', '--no-gaborish', '--epf', '0', '--no-pixel-domain-loss', '--quiet',
                '--strategy', strategy]
            if experimental: command.append('--experimental')
            if lf: command.append('--lf-frame')
            env = os.environ.copy()
            env['LLVM_PROFILE_FILE'] = str(cell / 'cli-%p-%m.profraw')
            (cell / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
            with (cell / 'stdout.log').open('xb') as stdout, (cell / 'stderr.log').open('xb') as stderr:
                child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
                (a.output / 'active.json').write_text(json.dumps(dict(command=command, pid=child.pid), indent=2))
                code = child.wait()
            row = dict(strategy=strategy, experimental=experimental, lf=lf, exit_code=code,
                stderr=(cell / 'stderr.log').read_text(), destination_preserved=destination.read_bytes()==sentinel)
            rows.append(row)
            (a.output / 'results.partial.json').write_text(json.dumps(rows, indent=2) + '\n')
            assert code != 0, 'Unexpected admission: validate any produced stream separately before drawing a pixel conclusion'
            assert row['destination_preserved'], cell
            assert 'memory' in row['stderr'].lower() or 'limit' in row['stderr'].lower(), row
            print(f'{strategy} experimental={experimental} lf={lf}: refused, existing destination preserved', flush=True)
(a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
