#!/usr/bin/env python3
"""Retain small pinned-native delta-palette inputs and decoded pixel observations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from PIL import Image

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('cjxl', type=Path)
p.add_argument('djxl', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
def sha(data): return hashlib.sha256(data).hexdigest()
tools = {}
for name in ['cjxl', 'djxl']:
    path = getattr(a, name).resolve()
    version = subprocess.run([str(path), '--version'], capture_output=True, check=True)
    version = (version.stdout + version.stderr).decode()
    assert 'v0.12.0 a7a9c78' in version
    tools[name] = dict(path=str(path), sha256=sha(path.read_bytes()), version=version)
rows = []
for width, height in [(32, 32), (259, 17)]:
    case = a.output / f'{width}x{height}'
    case.mkdir()
    values = []
    for y in range(height):
        for x in range(width):
            xx, yy = x % 32, y % 32
            values.extend([128 + 4 * xx - 4 * yy,
                           120 + 4 * xx - 3 * yy,
                           136 + 3 * xx - 4 * yy] if xx else [128, 120, 136])
    source = bytes(values)
    image = Image.frombytes('RGB', (width, height), source)
    image.save(case / 'source.png')
    for palette in [False, True]:
        out = case / f'palette-{palette}'
        out.mkdir()
        command = [tools['cjxl']['path'], str(case / 'source.png'), str(out / 'output.jxl'),
                   '--num_threads=1', '-e', '7', '-d', '0', '-m', '1', '-C', '0',
                   '-P', '0', '-X', '0', '-Y', '0', '--modular_palette_colors=0']
        if palette: command.append('--modular_lossy_palette')
        encoded = subprocess.run(['nice', '-n', '19', 'ionice', '-c', '3', *command], capture_output=True)
        (out / 'encode.log').write_bytes(encoded.stdout + encoded.stderr)
        assert encoded.returncode == 0, (out, encoded.stderr)
        decoded = subprocess.run(['nice', '-n', '19', 'ionice', '-c', '3', tools['djxl']['path'],
                                  str(out / 'output.jxl'), str(out / 'decoded.png'),
                                  '--bits_per_sample=8', '--num_threads=1'], capture_output=True)
        (out / 'decode.log').write_bytes(decoded.stdout + decoded.stderr)
        assert decoded.returncode == 0
        result = Image.open(out / 'decoded.png').convert('RGB')
        assert result.size == image.size
        result = result.tobytes()
        rows.append(dict(size=[width, height], palette=palette, command=command,
                         png_sha256=sha((case / 'source.png').read_bytes()), source_rgb8_sha256=sha(source),
                         decoded_rgb8_sha256=sha(result), jxl_sha256=sha((out / 'output.jxl').read_bytes()),
                         changed_samples=sum(x != y for x, y in zip(source, result)),
                         maximum_code_difference=max(abs(x-y) for x, y in zip(source, result))))
        if not palette: assert result == source
(a.output / 'results.json').write_text(json.dumps(dict(tools=tools, runner_sha256=sha(Path(__file__).read_bytes()),
    observations=rows, limitation='Native public controls only; palette transform/delta selection is not observed here.'), indent=2))
print(json.dumps(rows, indent=2))
