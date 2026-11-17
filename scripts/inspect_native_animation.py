#!/usr/bin/env python3
"""Retain pinned native frame renders and observed integer sample summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('djxl', type=Path)
p.add_argument('input', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--expected-frames', type=int, required=True)
a = p.parse_args()
version = subprocess.run([str(a.djxl), '--version'], capture_output=True, text=True, check=True)
assert 'v0.12.0 a7a9c78' in version.stdout + version.stderr
a.output.mkdir(exist_ok=False)
result = subprocess.run([str(a.djxl), str(a.input), str(a.output / 'frame.pam'),
                         '--output_frames', '--bits_per_sample=16', '--num_threads=1'], capture_output=True)
(a.output / 'decode.log').write_bytes(result.stdout + result.stderr)
(a.output / 'verdict.json').write_text(json.dumps(dict(
    input=str(a.input), input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
    decoder=str(a.djxl), decoder_sha256=hashlib.sha256(a.djxl.read_bytes()).hexdigest(),
    version=version.stdout + version.stderr, returncode=result.returncode), indent=2))
assert result.returncode == 0, result.stderr.decode(errors='replace')
paths = sorted(a.output.glob('*.pam'))
assert len(paths) == a.expected_frames, (len(paths), a.expected_frames)
frames = []
for path in paths:
    data = path.read_bytes()
    start = data.index(b'ENDHDR\n') + len(b'ENDHDR\n')
    fields = dict(line.split() for line in data[:start].decode().splitlines()
                  if line.startswith(('WIDTH ', 'HEIGHT ', 'DEPTH ', 'MAXVAL ')))
    width, height, depth, maximum = [int(fields[key]) for key in ['WIDTH', 'HEIGHT', 'DEPTH', 'MAXVAL']]
    assert depth in (3, 4) and maximum in (255, 65535), fields
    payload = data[start:]
    count = width * height * depth
    assert len(payload) == count * (1 if maximum == 255 else 2)
    values = tuple(payload) if maximum == 255 else struct.unpack('>' + str(count) + 'H', payload)
    rgb = [v for index, v in enumerate(values) if index % depth < 3]
    frames.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                       width=width, height=height, depth=depth, maximum=maximum,
                       minimum_rgb=min(rgb), maximum_rgb=max(rgb),
                       nonzero_rgb_samples=sum(v != 0 for v in rgb),
                       fullscale_rgb_samples=sum(v == maximum for v in rgb), first_pixel=values[:depth]))
record = dict(input=str(a.input), input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
              decoder=str(a.djxl), decoder_sha256=hashlib.sha256(a.djxl.read_bytes()).hexdigest(),
              version=version.stdout + version.stderr, frames=frames)
(a.output / 'observed.json').write_text(json.dumps(record, indent=2))
for frame in frames:
    print(json.dumps(frame))
