#!/usr/bin/env python3
"""Bind the proposed float-lock container migration to unchanged codestream bytes."""
import argparse
import hashlib
import json
from pathlib import Path
from jxl_bitstream_diff import Parser, BitReader
from jxl_container import codestream

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('pairs', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert not a.output.exists()
expected = {}
for line in Path('jxl-encoder/tests/hash_lock_expected.txt').read_text().splitlines():
    parts = line.split()
    if len(parts) == 4:
        expected[parts[0]] = (int(parts[1]), int(parts[2], 0), int(parts[3], 0))
def fnv(data):
    value = 0xcbf29ce484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    return value
rows = []
for pair in json.loads(a.pairs.read_text()):
    before = Path(pair['before']).read_bytes(); after = Path(pair['after']).read_bytes()
    original, _ = codestream(before); current, ranges = codestream(after)
    assert original == current, pair['name']
    assert len(ranges) == 1 and ranges[0][1] == len(current)
    parser = Parser(original); parser.br = BitReader(original, 2)
    dimensions = parser.size_header(); parser.image_metadata(); parser.custom_transform_data(); parser.br.jump_to_byte_boundary()
    header = parser.br.pos // 8
    fields = {key: value for _, key, value in parser.fields}
    assert fields['metadata.modular_16bit_buffers'] == 0
    assert (len(before), fnv(before[:header]), fnv(before[header:])) == expected[pair['name']], pair['name']
    offset = 0; level = 5; boxes = []
    while offset < len(after):
        size = int.from_bytes(after[offset:offset+4], 'big'); kind = after[offset+4:offset+8]
        assert size >= 8 and offset + size <= len(after)
        boxes.append({'type': kind.decode(), 'bytes': size})
        if kind == b'jxll':
            assert size == 9;level = after[offset+8]
        offset += size
    assert level == 10
    frame = ranges[0][0] + header
    assert after[frame:] == before[header:]
    rows.append({'name': pair['name'], 'dimensions': dimensions, 'before_bytes':len(before), 'after_bytes':len(after),
        'container_prefix_bytes':ranges[0][0], 'raw_header_bytes':header, 'after_frame_offset':frame,
        'before_header_fnv':f'{fnv(before[:header]):#018x}', 'after_header_fnv':f'{fnv(after[:frame]):#018x}',
        'frame_fnv':f'{fnv(before[header:]):#018x}', 'codestream_sha256':hashlib.sha256(original).hexdigest(),
        'codestream_exact':True, 'frame_exact':True, 'modular_16bit_sufficient':False, 'declared_level':level, 'boxes':boxes})
a.output.write_text(json.dumps(rows, indent=2) + '\n')
for row in rows:
    print(f'{row["name"]}: {row["before_bytes"]} -> {row["after_bytes"]}; header {row["before_header_fnv"]} -> {row["after_header_fnv"]}; frame {row["frame_fnv"]} unchanged')
