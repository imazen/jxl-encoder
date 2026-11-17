#!/usr/bin/env python3
"""Union selected alpha-writer regions only after comparing exact function bodies.

Each --pair is a frozen source/object archive and that object's LLVM export.
Line offsets before a function may differ after documentation edits; counters
are normalized to its declaration. Code inside the compared body must match.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
p.add_argument('--pair', action='append', nargs=2, required=True)
p.add_argument('--scope', choices=['alpha', 'progressive', 'partition', 'admission'], default='alpha')
a = p.parse_args()
if a.output.exists():
    p.error('refusing existing result')
inputs = [(Path(archive), json.loads(Path(report).read_text()))
          for archive, report in a.pair]
functions = {
    'jxl-encoder/src/vardct/bitstream.rs': [
        'write_modular_extras_alpha_squeezed',
        'write_modular_extras_alpha_squeezed_global',
        'write_modular_extras_alpha_squeezed_lf_group',
        'write_modular_extras_alpha_squeezed_hf_group',
        'write_modular_extras_alpha_squeezed_section',
    ],
    'jxl-encoder/src/modular/encode_primitives.rs': [
        'write_hybrid_data_histogram_with_contexts',
    ],
}
if a.scope == 'progressive':
    functions = {'jxl-encoder/src/vardct/bitstream.rs': [
        'modular_shift_range', 'encode_ac_group_section', 'encode_dc_group',
        'write_modular_extras_alpha_squeezed_hf_group',
    ]}
elif a.scope == 'partition':
    functions = {'jxl-encoder/src/vardct/extras.rs': ['partition']}
if a.scope == 'admission':
    functions = {
        'jxl-encoder/src/vardct/extras.rs': ['build_alpha_squeeze_pipeline'],
        'jxl-encoder/src/modular/squeeze.rs': ['default_squeeze_params_for_groups'],
    }
rows = []
for relative, names in functions.items():
    for name in names:
        bodies, spans = [], []
        for archive, _ in inputs:
            lines = (archive / 'source' / relative).read_text().splitlines()
            starts = [i for i, line in enumerate(lines)
                      if re.search(r'\bfn ' + re.escape(name) + r'\(', line)]
            assert len(starts) == 1, (archive, name, starts)
            start = starts[0]
            indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
            end = next(i for i in range(start + 1, len(lines))
                       if lines[i] == indent + '}')
            bodies.append('\n'.join(lines[start:end + 1]))
            spans.append((start + 1, end + 1))
        assert all(body == bodies[0] for body in bodies), ('function bodies differ', name)
        counts, observed = {}, []
        for (_, report), (start, end) in zip(inputs, spans):
            local = {}
            for unit in report['data']:
                for function in unit['functions']:
                    # Both Rust manglings prefix identifiers with their byte
                    # length. v0 can end directly after the identifier; legacy
                    # mangling adds a hash. The prefix excludes sibling names.
                    if str(len(name)) + name not in function['name']:
                        continue
                    for region in function['regions']:
                        file = function['filenames'][region[5]]
                        if (region[7] != 0 or not file.endswith(relative)
                                or region[0] < start or region[2] > end):
                            continue
                        key = (region[0] - start, region[1], region[2] - start, region[3])
                        local[key] = max(local.get(key, 0), region[4])
            observed.append(sum(value > 0 for value in local.values()))
            for key, value in local.items():
                counts[key] = max(counts.get(key, 0), value)
        assert counts, ('no mapped code regions', name)
        row = dict(function=name, file=relative,
                   input_archives=[str(archive) for archive, _ in inputs],
                   input_test_exit_codes=[
                       json.loads((archive / 'identity.json').read_text()).get('exit_code')
                       if (archive / 'identity.json').is_file() else None
                       for archive, _ in inputs],
                   body_sha256=hashlib.sha256(bodies[0].encode()).hexdigest(),
                   source_spans=spans, observed_per_input=observed,
                   covered=sum(value > 0 for value in counts.values()), total=len(counts),
                   missing_relative_regions=[key for key, value in sorted(counts.items()) if not value])
        rows.append(row)
        print(json.dumps(row))
with a.output.open('x') as sink:
    json.dump(dict(functions=rows, inputs=a.pair,
                   source_comparison='Exact function bodies; relative declaration offsets'), sink, indent=2)
