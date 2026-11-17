#!/usr/bin/env python3
"""Summarize selected squeeze regions and actual default-writer caller sites.

Each --pair supplies an exact source/object archive and its LLVM export.
Source bodies must match before region counters from separate runs are united.
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
p.add_argument('--pair', action='append', nargs=2, required=True, metavar=('ARCHIVE', 'REPORT'))
a = p.parse_args()
if a.output.exists():
    p.error('refusing existing result')
modules = ['encode.rs', 'frame.rs', 'squeeze.rs', 'encode_transforms.rs']
for file in modules:
    sources = [(Path(archive) / 'source/jxl-encoder/src/modular' / file).read_text()
               for archive, _ in a.pair]
    if file == 'encode_transforms.rs':
        sources = [source.split('#[cfg(test)]')[0] for source in sources]
    assert all(source == sources[0] for source in sources), ('source bodies differ', file)
reports = [json.loads(Path(report).read_text()) for _, report in a.pair]
functions = ['write_squeeze_transform', 'write_default_squeeze_transform',
             '22default_squeeze_params', 'default_dimensions_fit',
             'build_default_squeeze_params', 'lossy_default_squeeze_params',
             'default_squeeze_params_for_groups', 'squeeze_geometry_step',
             'squeeze_group_geometry_fits']
rows = []
for name in functions:
    counts = {}
    for report in reports:
        for unit in report['data']:
            for function in unit['functions']:
                if name not in function['name']:
                    continue
                for region in function['regions']:
                    file = function['filenames'][region[5]]
                    if region[7] != 0 or not any(file.endswith('/modular/' + v) for v in modules):
                        continue
                    key = (file, *region[:4])
                    counts[key] = max(counts.get(key, 0), region[4])
    assert counts, ('no mapped function', name)
    row = dict(function=name, covered=sum(v > 0 for v in counts.values()), total=len(counts),
               missing=[key[1:] for key, value in counts.items() if not value])
    rows.append(row)
    print(json.dumps(row))
callers = []
root = Path(a.pair[0][0]) / 'source'
for part in ['encode.rs', 'frame.rs']:
    relative = Path('jxl-encoder/src/modular') / part
    for line, text in enumerate((root / relative).read_text().splitlines(), 1):
        token = 'write_default_squeeze_transform('
        if token not in text:
            continue
        column = text.index(token) + 1
        observed = []
        for report in reports:
            candidates = []
            for unit in report['data']:
                for function in unit['functions']:
                    for region in function['regions']:
                        file = function['filenames'][region[5]]
                        if (region[7] == 0 and file.endswith(str(relative))
                                and (region[0], region[1]) <= (line, column)
                                < (region[2], region[3])):
                            candidates.append((region[2] - region[0],
                                               region[3] - region[1], region[4]))
            if candidates:
                span = min(candidate[:2] for candidate in candidates)
                observed.append(max(candidate[2] for candidate in candidates if candidate[:2] == span))
        assert observed and max(observed) > 0, ('unexecuted caller', str(relative), line, observed)
        row = dict(file=str(relative), line=line, observed=observed)
        callers.append(row)
        print(json.dumps(row))
with a.output.open('x') as sink:
    json.dump(dict(functions=rows, callers=callers, inputs=a.pair,
                   source_comparison='Exact encode/frame/squeeze and pre-test transform writer bodies match'),
              sink, indent=2)
