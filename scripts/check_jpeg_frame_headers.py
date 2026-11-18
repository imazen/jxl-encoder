#!/usr/bin/env python3
"""Observe complete non-ICC JPEG-transcode envelopes independently of the writer."""
import argparse
import json
from pathlib import Path
from jxl_bitstream_diff import Parser, BitReader
from jxl_container import codestream

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert not a.output.exists()
rows = []
for row in json.loads(a.manifest.read_text()):
    data, _ = codestream(Path(row['path']).read_bytes())
    parser = Parser(data)
    parser.br = BitReader(data, 2)
    width, height = parser.size_header()
    parser.image_metadata()
    parser.custom_transform_data()
    assert not getattr(parser, 'want_icc', False), 'this selected fixture has no ICC'
    parser.br.jump_to_byte_boundary()
    parser.frame_header(width, height)
    parser.read_toc(width, height)
    fields = {key: value for _, key, value in parser.fields}
    assert parser.toc is not None
    assert parser.toc['payload_start'] + sum(parser.toc['entries']) == len(data)
    assert [fields.get(f'frame.chroma_subsampling[{i}]', 0) for i in range(3)] == row['sampling'], (row, fields)
    assert fields['frame.ycbcr'] == row['ycbcr'], (row, fields)
    rows.append(dict(path=row['path'], width=width, height=height, fields=fields))
a.output.write_text(json.dumps(rows, indent=2))
