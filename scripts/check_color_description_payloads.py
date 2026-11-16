#!/usr/bin/env python3
"""Check independent envelope parsing and lossless payload equality to RGB controls."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from jxl_bitstream_diff import Parser
from jxl_container import codestream

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("rgb", type=Path)
p.add_argument("xyb", type=Path)
p.add_argument("unknown", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
if a.output.exists():
    p.error("output must be new")
expected = {f"{name}-{mode}-{dims}-{wide}-{route}" for name, mode, dims, wide, route in itertools.product(
    ["libjxl", "lean", "zen", "aggressive", "custom-long", "custom-compact"],
    ["Reference", "Experimental"], ["64x32", "259x17"], ["false", "true"],
    ["still", "strided", "streaming", "animation"])}
tables = {}
for space, root in [("kRGB", a.rgb), ("kXYB", a.xyb), ("kUnknown", a.unknown)]:
    rows = list(csv.DictReader((root / "results.tsv").open(), delimiter="\t"))
    assert len(rows) == len(expected) and {r['cell'] for r in rows} == expected, space
    table = {}
    for row in rows:
        data = (root / row['cell'] / "encoded.jxl").read_bytes()
        assert len(data) == int(row['bytes'])
        assert hashlib.sha256(data).hexdigest() == row['sha256']
        raw, _ = codestream(data)
        parser = Parser(raw)
        parser.parse()
        fields = {name: value for _, name, value in parser.fields}
        assert fields['metadata.color_encoding.color_space'] == space, row['cell']
        assert fields['metadata.color_encoding.rendering_intent'] == 'kPerceptual'
        if space == 'kXYB':
            for key in ['white_point', 'primaries', 'have_gamma', 'transfer_function']:
                assert f'metadata.color_encoding.{key}' not in fields
        else:
            assert fields['metadata.color_encoding.primaries'] == 'kSRGB'
            assert fields['metadata.color_encoding.transfer_function'] == 'kSRGB'
        toc = parser.toc
        assert toc and toc['payload_start'] + sum(toc['entries']) == len(raw), row['cell']
        payload = raw[toc['payload_start']:]
        if space == 'kRGB':
            assert row['jxl_rs'] == row['djxl'] == 'true', row['cell']
        else:
            control = tables['kRGB'][row['cell']]
            assert payload == control['payload'], (space, row['cell'])
            assert toc['entries'] == control['entries']
        table[row['cell']] = {'payload': payload, 'entries': toc['entries'],
                              'jxl_rs': row['jxl_rs'], 'djxl': row['djxl']}
    tables[space] = table
summary = {space: {'cells': len(table),
                   'jxl_rs_rendered': sum(r['jxl_rs'] == 'true' for r in table.values()),
                   'djxl_rendered': sum(r['djxl'] == 'true' for r in table.values())}
           for space, table in tables.items()}
a.output.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
