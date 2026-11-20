#!/usr/bin/env python3
"""Compare completed request dictionary matrices without inferring quality coverage."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def inventory(root):
    completed = []
    started = []
    for line in (root / 'cases.tsv').read_text().splitlines():
        *key, stage = line.split('\t')
        assert len(key) == 4, line
        assert stage in ('encoding', 'complete'), line
        (started if stage == 'encoding' else completed).append(tuple(key))
    assert len(completed) == len(set(completed)), 'duplicate completed cells'
    assert started == completed, 'incomplete or reordered journal'
    policies = {row[0] for row in completed}
    assert len(policies) == 17, policies
    expected = set(itertools.product(policies, ['alphafalse', 'alphatrue'],
                                     ['i0', 'i2'], ['lffalse', 'lftrue']))
    assert set(completed) == expected, 'missing policy/layout/controller/LF combination'
    result = {}
    for key in completed:
        path = root / '-'.join(key) / 'input.jxl'
        result[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('baseline', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
baseline = inventory(args.baseline)
candidate = inventory(args.candidate)
assert baseline.keys() == candidate.keys(), 'policy identities differ'
rows = [dict(cell=list(key), baseline_sha256=baseline[key],
             candidate_sha256=candidate[key], equal=baseline[key] == candidate[key])
        for key in sorted(baseline)]
assert not args.output.exists(), 'preserve previous comparison'
args.output.write_text(json.dumps(dict(baseline=str(args.baseline),
    candidate=str(args.candidate), cells=rows,
    note='Exact encoded bytes on the completed named matrix; this comparison '
         'does not prove input-space or decoder coverage.'), indent=2) + '\n')
assert all(row['equal'] for row in rows), 'encoded bytes differ; see retained cells'
print('Every completed policy/layout/controller/LF cell retains exact encoded bytes.')
