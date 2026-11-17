#!/usr/bin/env python3
"""Render retained unmodified v0.12 float alpha and exact visibility controls."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('log', type=Path)
p.add_argument('djxl', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
version = subprocess.run([str(a.djxl), '--version'], capture_output=True, text=True, check=True)
assert 'v0.12.0 a7a9c78' in version.stdout + version.stderr
rows = dict(line.split('\t') for line in a.log.read_text().splitlines()
            if line.startswith(('float32\t', 'visible-on\t', 'visible-off\t')))
assert 'float32' in rows
assert ('visible-on' in rows) == ('visible-off' in rows)
a.output.mkdir(exist_ok=False)
def read_pfm(path):
    with path.open('rb') as f:
        magic = f.readline().strip()
        assert magic in (b'Pf', b'PF')
        assert f.readline().strip() == b'3 1'
        scale = float(f.readline())
        raw = f.read()
        return struct.unpack(('<' if scale < 0 else '>') + str(len(raw)//4) + 'f', raw)
identities = []
colors = {}
for label, hex_data in rows.items():
    out = a.output / label
    out.mkdir()
    data = bytes.fromhex(hex_data)
    stream = out / 'native.jxl'
    stream.write_bytes(data)
    result = subprocess.run([str(a.djxl), str(stream), str(out / 'native.pfm'),
                             '--output_frames', '--output_extra_channels', '--num_threads=1'],
                            capture_output=True)
    (out / 'decode.log').write_bytes(result.stdout + result.stderr)
    assert result.returncode == 0
    paths = sorted(out.glob('*-ec1.pfm'))
    assert len(paths) == 2, paths
    expected = struct.pack('=3f', *([.0625, .9375, .1234567] if label == 'float32' else [1, 1/1024, 1]))
    for path in paths:
        assert struct.pack('=3f', *read_pfm(path)) == expected
    colors[label] = [read_pfm(path) for path in sorted(out.glob('*.pfm')) if '-ec' not in path.name]
    assert len(colors[label]) == 2
    identities.append(dict(label=label, stream_sha256=hashlib.sha256(data).hexdigest(),
                           expected_alpha_bits=[hex(v) for v in struct.unpack('=3I', expected)]))
if 'visible-on' in rows:
    assert colors['visible-on'] == colors['visible-off'], 'Nonzero alpha is visible in native preprocessing'
(a.output / 'identity.json').write_text(json.dumps(dict(controls=identities,
    source_log_sha256=hashlib.sha256(a.log.read_bytes()).hexdigest(),
    decoder_version=version.stdout + version.stderr), indent=2))
print('Native displayed frames preserve alpha; selected exact-visibility controls agree')
