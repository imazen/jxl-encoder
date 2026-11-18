#!/usr/bin/env python3
"""Compare retained animation bytes and frame payloads across header changes.

This control requires matching relative output.jxl paths and parametric
profiles. It does not infer source/binary identity from directory names.
"""
import argparse
import hashlib
import json
from pathlib import Path
from jxl_bitstream_diff import Parser, BitReader
from jxl_container import codestream

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('before', type=Path)
p.add_argument('after', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--filename', default='output.jxl', choices=['output.jxl', 'input.jxl'])
a = p.parse_args()
assert not a.output.exists()
before = {x.relative_to(a.before): x for x in a.before.glob('*/' + a.filename)}
after = {x.relative_to(a.after): x for x in a.after.glob('*/' + a.filename)}
assert before and before.keys() == after.keys()

def inspect(stream):
    parser = Parser(stream)
    parser.br = BitReader(stream, 2)
    parser.size_header()
    parser.image_metadata()
    parser.custom_transform_data()
    assert not parser.want_icc, 'ICC boundary is outside this parser control'
    parser.br.jump_to_byte_boundary()
    offset = parser.br.pos // 8
    fields = {name: value for _, name, value in parser.fields}
    return stream[offset:], fields

rows = []
for relative in sorted(before):
    old = before[relative].read_bytes()
    new = after[relative].read_bytes()
    old_stream, _ = codestream(old)
    new_stream, _ = codestream(new)
    old_frames, old_fields = inspect(old_stream)
    new_frames, new_fields = inspect(new_stream)
    assert old_frames == new_frames, ('frame payload changed', relative)
    changes = {key: [old_fields.get(key), new_fields.get(key)]
               for key in old_fields.keys() | new_fields.keys()
               if old_fields.get(key) != new_fields.get(key)}
    # Only the finalized buffer promise may change a common-header field.
    assert all(key.endswith('.modular_16bit_buffers') for key in changes), (relative, changes)
    if not changes:
        assert old_stream == new_stream, ('unaccounted header bytes', relative)
    rows.append(dict(path=str(relative), before_sha256=hashlib.sha256(old).hexdigest(),
                     after_sha256=hashlib.sha256(new).hexdigest(),
                     exact_file=old == new, exact_codestream=old_stream == new_stream,
                     exact_frames=True, changed_fields=changes))
a.output.write_text(json.dumps(dict(rows=rows,
    limitation='Retained-byte comparison only; source/executable provenance is recorded separately'), indent=2))
print(json.dumps(dict(exact_files=sum(row['exact_file'] for row in rows),
    exact_codestreams=sum(row['exact_codestream'] for row in rows),
    exact_frame_payloads=len(rows), compared=len(rows))))
