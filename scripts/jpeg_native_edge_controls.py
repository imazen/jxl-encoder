#!/usr/bin/env python3
"""Retain pinned v0.12 observations for small JPEG metadata/sampling edges."""
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
def sha(data):
    return hashlib.sha256(data).hexdigest()
tools = {}
for name in ['cjxl', 'djxl']:
    path = getattr(a, name).resolve()
    result = subprocess.run([str(path), '--version'], capture_output=True, check=True)
    version = (result.stdout + result.stderr).decode()
    assert 'v0.12.0 a7a9c78' in version, (path, version)
    tools[name] = dict(path=str(path), sha256=sha(path.read_bytes()), version=version)
(a.output / 'identity.json').write_text(json.dumps(dict(tools=tools,
    runner_sha256=sha(Path(__file__).read_bytes()), pixel_precision='Pillow RGB8 of native PNG8; no 16-bit pixel claim'), indent=2))
fixture = Path(__file__).resolve().parent.parent / 'jxl-encoder/tests/fixtures/jbrd'
def segments(data):
    pos = 2
    while True:
        assert data[pos] == 255
        marker = data[pos + 1]
        length = 2 + int.from_bytes(data[pos + 2:pos + 4], 'big')
        yield pos, marker, length
        if marker == 0xda:
            return
        pos += length
def remove(data, marker):
    out = data[:2]
    for pos, kind, length in segments(data):
        if kind != marker:
            out += data[pos:pos + length]
        if kind == 0xda:
            out += data[pos + length:]
    return out
def insert(data, marker, payload):
    return data[:2] + bytes([255, marker]) + (len(payload) + 2).to_bytes(2, 'big') + payload + data[2:]
rows = []
def run(name, data):
    out = a.output / name
    out.mkdir()
    (out / 'input.jpg').write_bytes(data)
    def command(args, log):
        result = subprocess.run(['nice', '-n', '19', 'ionice', '-c', '3', *args], capture_output=True)
        (out / log).write_bytes(result.stdout + result.stderr)
        return result.returncode
    status = command([tools['cjxl']['path'], str(out / 'input.jpg'), str(out / 'output.jxl'),
                      '--lossless_jpeg=1', '--num_threads=1', '-e', '3'], 'encode.log')
    row = dict(case=name, input_sha256=sha(data), encode_exit=status)
    pixels = None
    if status == 0:
        encoded = (out / 'output.jxl').read_bytes()
        row['jxl_sha256'] = sha(encoded)
        render = command([tools['djxl']['path'], str(out / 'output.jxl'), str(out / 'render.png'),
                          '--bits_per_sample=8', '--num_threads=1'], 'render.log')
        rebuild = command([tools['djxl']['path'], str(out / 'output.jxl'), str(out / 'rebuilt.jpg'),
                           '--num_threads=1'], 'rebuild.log')
        row.update(render_exit=render, rebuild_exit=rebuild)
        if render == 0:
            image = Image.open(out / 'render.png').convert('RGB')
            pixels = image.tobytes()
            row.update(size=list(image.size), pixels_rgb8_sha256=sha(pixels))
        if rebuild == 0:
            rebuilt = (out / 'rebuilt.jpg').read_bytes()
            row.update(reconstructed_sha256=sha(rebuilt), exact_jpeg=rebuilt == data)
    rows.append(row)
    (a.output / 'results.partial.json').write_text(json.dumps(rows, indent=2))
    return row, pixels
rgb = (fixture / 'rgb_adobe.jpg').read_bytes()
for adobe in [False, True]:
    baseline = rgb if adobe else remove(rgb, 0xee)
    before, pixels = run(f'rgb-adobe-{adobe}', baseline)
    assert before['encode_exit'] == before['render_exit'] == before['rebuild_exit'] == 0 and before['exact_jpeg']
    changed, after = run(f'rgb-adobe-{adobe}-unknown-app0', insert(baseline, 0xe0, b'Private APP0 data'))
    if after is not None:
        assert len(after) == len(pixels)
        changed['changed_rgb8_samples'] = sum(x != y for x, y in zip(pixels, after))
        changed['maximum_rgb8_difference'] = max(abs(x - y) for x, y in zip(pixels, after))
base = (fixture / 'base_b_444.jpg').read_bytes()
for name, prefix in [('exif-prefix', b'Exif\0\0'), ('xmp-prefix', b'http://ns.adobe.com/xap/1.0/\0')]:
    run(name, insert(base, 0xe1, prefix))
for name, payload in [('empty-dht', b''), ('dc3-zero-symbols', bytes([3]) + bytes(16)),
                      ('ac3-zero-symbols', bytes([0x13]) + bytes(16))]:
    run(name, insert(base, 0xc4, payload))
subsampled = bytearray((fixture / 'base_a_420.jpg').read_bytes())
for offset, kind, _ in segments(subsampled):
    if kind == 0xc0:
        assert subsampled[offset + 9] == 3
        for i, value in enumerate(b'RGB'):
            subsampled[offset + 10 + 3 * i] = value
    if kind == 0xda:
        assert subsampled[offset + 4] == 3
        for i, value in enumerate(b'RGB'):
            subsampled[offset + 5 + 2 * i] = value
run('rgb-subsampled-420', remove(bytes(subsampled), 0xe0))
(a.output / 'results.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
