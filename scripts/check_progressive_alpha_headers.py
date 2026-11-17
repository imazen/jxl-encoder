#!/usr/bin/env python3
"""Check actual reference/encoder progressive fields in the alpha controls."""
import argparse
import json
from pathlib import Path

from jxl_bitstream_diff import parse_file

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('controls', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
if a.output.exists():
    p.error('refusing existing result')
cells = sorted(a.controls.glob('strict-*'))
assert len(cells) == 9, ('requires the complete three-size/three-mode control', cells)
expected = {
    'Single': ((1, [], []), (1, [], [])),
    'QuantizedAcFullAc': ((2, [1], [(2, 0)]), (2, [1], [(2, 0)])),
    # Existing precision-split policy and native frequency-split CLI policy
    # intentionally have separate expectations; this gate does not equate them.
    'DcVlfLfAc': ((3, [2, 0], [(8, 0), (4, 1)]),
                 (3, [0, 0], [(4, 0), (2, 1)])),
}
rows = []
for cell in cells:
    modes = [name for name in expected if cell.name.endswith('-' + name)]
    assert len(modes) == 1, cell
    for filename, wanted in zip(['input.jxl', 'native.jxl'], expected[modes[0]]):
        parser, data = parse_file(cell / filename)
        assert parser.toc, ('missing TOC', cell, filename)
        assert parser.toc['payload_start'] + sum(parser.toc['entries']) == len(data)
        fields = {name: value for _, name, value in parser.fields}
        count = parser.frame['num_passes']
        shifts = [fields[f'frame.passes.shift[{i}]'] for i in range(count - 1)]
        brackets = [(fields[f'frame.passes.downsample[{i}]'],
                     fields[f'frame.passes.last_pass[{i}]'])
                    for i in range(fields.get('frame.passes.num_downsample', 0))]
        assert (count, shifts, brackets) == wanted, (cell, filename, count, shifts, brackets, wanted)
        rows.append(dict(cell=cell.name, file=filename, passes=count,
                         shifts=shifts, brackets=brackets))
with a.output.open('x') as sink:
    json.dump(rows, sink, indent=2)
print('Actual progressive fields and complete section accounting match their separate controls')
