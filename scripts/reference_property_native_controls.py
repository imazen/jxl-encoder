#!/usr/bin/env python3
"""Pin finite-f32 wide-reference inputs without claiming selected-tree reachability."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

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
width, height = 64, 32
state = 144
reference = []
for _ in range(width * height):
    state ^= (state << 13) & 0xffffffff
    state ^= state >> 17
    state ^= (state << 5) & 0xffffffff
    state &= 0xffffffff
    reference.append(0x60000000 if state & 1 else -0x60000000)
words, overflow_residuals = [], 0
for y in range(height):
    for x in range(width):
        value = reference[y * width + x]
        left = reference[y * width + x - 1] if x else 0
        top = reference[(y - 1) * width + x] if y else left
        northwest = reference[(y - 1) * width + x - 1] if x and y else left
        predicted = min(max(left + top - northwest, min(left, top)), max(left, top))
        residual = value - predicted
        overflow_residuals += abs(residual) > 0x7fffffff
        words.extend([value & 0xffffffff, 0x3f800000 if residual else 0, 0])
assert overflow_residuals
packed = struct.pack('<' + 'I' * len(words), *words)
rows = [packed[y * width * 12:(y + 1) * width * 12] for y in range(height)]
(a.output / 'source.pfm').write_bytes(f'PF\n{width} {height}\n-1.0\n'.encode() + b''.join(reversed(rows)))
command = [tools['cjxl']['path'], str(a.output / 'source.pfm'), str(a.output / 'output.jxl'),
           '--num_threads=1', '-e', '9', '-d', '0', '-m', '1', '-C', '0', '-P', '0', '-E', '1',
           '-X', '0', '-Y', '0', '--modular_palette_colors=0',
           '-x', 'color_space=RGB_D65_SRG_Rel_Lin']
encoded = subprocess.run(['nice', '-n', '19', 'ionice', '-c', '3', *command], capture_output=True)
(a.output / 'encode.log').write_bytes(encoded.stdout + encoded.stderr)
assert encoded.returncode == 0, encoded.stderr
command_decode = [tools['djxl']['path'], str(a.output / 'output.jxl'), str(a.output / 'decoded.pfm'),
                  '--num_threads=1', '--color_space=RGB_D65_SRG_Rel_Lin']
decoded = subprocess.run(['nice', '-n', '19', 'ionice', '-c', '3', *command_decode], capture_output=True)
(a.output / 'decode.log').write_bytes(decoded.stdout + decoded.stderr)
assert decoded.returncode == 0, decoded.stderr
with (a.output / 'decoded.pfm').open('rb') as stream:
    assert stream.readline().strip() == b'PF'
    dimensions = stream.readline()
    while dimensions.startswith(b'#'): dimensions = stream.readline()
    assert [int(x) for x in dimensions.split()] == [width, height]
    scale = float(stream.readline())
    assert abs(scale) == 1.0
    raw = stream.read()
assert len(raw) == len(packed)
actual = list(struct.unpack(('>' if scale > 0 else '<') + 'I' * len(words), raw))
actual_rows = [actual[y * width * 3:(y + 1) * width * 3] for y in range(height)]
actual = [value for row in reversed(actual_rows) for value in row]
changed = sum(x != y for x, y in zip(words, actual))
result = dict(tools=tools, command=command, command_decode=command_decode,
              runner_sha256=sha(Path(__file__).read_bytes()), source_sha256=sha(packed),
              source_pfm_sha256=sha((a.output / 'source.pfm').read_bytes()),
              jxl_sha256=sha((a.output / 'output.jxl').read_bytes()),
              mathematical_wide_residual_sites=overflow_residuals, changed_f32_words=changed,
              limitation='Native public pixel control only; no actual selected reference-property tree was observed.')
(a.output / 'results.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
assert changed == 0, changed
