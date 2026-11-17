#!/usr/bin/env python3
"""Report alpha preparation regions from an exact successful object/profile."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('llvm', type=Path)
p.add_argument('archive', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--support', type=Path, action='append', default=[])
a = p.parse_args()
identity = json.loads((a.archive / 'identity.json').read_text())
assert identity['exit_code'] == 0
for row in identity['sources']:
    assert hashlib.sha256((a.archive / 'source' / row['path']).read_bytes()).hexdigest() == row['sha256']
scripts = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(scripts/'coverage_from_archive.py'), str(a.llvm),
                str(a.archive), str(a.output)], check=True)
reports = [json.loads((a.output/'selected.json').read_text())]
support_identities = []
for index, archive in enumerate(a.support):
    supporting = json.loads((archive/'identity.json').read_text())
    assert supporting['exit_code'] == 0
    for row in supporting['sources']:
        assert hashlib.sha256((archive/'source'/row['path']).read_bytes()).hexdigest() == row['sha256']
    for relative in ['jxl-encoder/src/api.rs', 'jxl-encoder/src/api/ingest.rs', 'jxl-encoder/src/api/animate.rs']:
        assert (archive/'source'/relative).read_bytes() == (a.archive/'source'/relative).read_bytes(), relative
    target = a.output / ('support-' + str(index))
    subprocess.run([sys.executable, str(scripts/'coverage_from_archive.py'), str(a.llvm),
                    str(archive), str(target)], check=True)
    reports.append(json.loads((target/'selected.json').read_text()))
    support_identities.append(dict(archive=str(archive), object_sha256=supporting['binary_sha256'],
                                    profile_sha256=supporting['profile_sha256']))
combined = a.output/'selected-union.json'
combined.write_text(json.dumps(dict(data=[unit for report in reports for unit in report['data']])))
rows = []
def observe(relative, symbol, first, last):
    lines = (a.archive/'source'/relative).read_text().splitlines()
    starts = [i for i,line in enumerate(lines) if first in line]
    assert len(starts) == 1, (first, starts)
    start = starts[0]
    end = next(i for i in range(start+1,len(lines)) if last in lines[i])
    result = subprocess.run([sys.executable, str(scripts/'coverage_function_regions.py'),
                str(combined), symbol, '--file-suffix', relative,
                '--start-line', str(start+1), '--end-line', str(end+1)],
                capture_output=True, text=True, check=True)
    assert not result.stderr, result.stderr
    row, = [json.loads(line) for line in result.stdout.splitlines()]
    rows.append(dict(source=relative, bounds=[start+1,end+1], **row))
# Separate methods with same Rust name by their exact source ranges.
observe('jxl-encoder/src/api.rs','AlphaCoding','impl AlphaCoding {','impl AlphaPlane {')
observe('jxl-encoder/src/api.rs','AlphaPlane','impl AlphaPlane {',"impl<'a> ExtraChannel")
observe('jxl-encoder/src/api/ingest.rs','float_alpha_values','pub(super) fn float_alpha_values(', '/// Visibility mask')
observe('jxl-encoder/src/api/animate.rs','encode_animation_lossy','    let float_alpha_coding =','    let mut file_header =')
observe('jxl-encoder/src/api/animate.rs','encode_animation_lossy','        let alpha_u16: Option<Vec<u16>>','        let frame_options =')
observe('jxl-encoder/src/api/animate.rs','encode_animation_lossy','        let alpha_float =','        enc.encode_frame_to_writer(')
(a.output/'alpha-regions.json').write_text(json.dumps(dict(
    object_sha256=identity['binary_sha256'], profile_sha256=identity['profile_sha256'],
    source_checks='all captured sources verified; supporting source files byte-identical',
    supporting_inputs=support_identities, regions=rows), indent=2))
for row in rows:
    print(json.dumps(row))
