#!/usr/bin/env python3
"""Check unchanged roundtrip properties with raw and required-container outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('baseline_revision')
p.add_argument('output', type=Path)
a = p.parse_args()
root = a.manifest.resolve().parent
assert not (root / '.git').exists() and (root.parent.parent / 'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
relative = 'jxl-encoder/src/vardct/encoder.rs'
source = root / relative
candidate = source.read_bytes()
baseline = subprocess.run(['git', 'show', a.baseline_revision + ':' + relative],
    check=True, stdout=subprocess.PIPE).stdout
assert baseline != candidate
rows = []
try:
    for label, content in [('raw-baseline', baseline), ('container-candidate', candidate)]:
        source.write_bytes(content)
        (a.output / (label + '.encoder.rs')).write_bytes(content)
        command = [sys.executable, str(Path(__file__).with_name('run_selected_test_evidence.py')),
            str(a.manifest), str(a.output / label), 'f16_input_roundtrip::rgb_linear_f16_roundtrips',
            '--also-filter', 'hdr_vdp2_chunk3::vdp2_pq_pipeline_decodes',
            '--also-filter', 'non_finite_action::sanitize_mode',
            '--also-filter', 'sixteenbit_metadata::test_lossy_rgba16',
            '--test-target', 'it', '--features', '__expert,jpeg-reencoding',
            '--source', relative, '--source', 'jxl-encoder/src/api_tests.rs',
            '--source', 'jxl-encoder/src/container.rs',
            '--source', 'jxl-encoder/tests/support/jxl_signature.rs',
            '--source', 'jxl-encoder/tests/it/f16_input_roundtrip.rs',
            '--source', 'jxl-encoder/tests/it/hdr_vdp2_chunk3.rs',
            '--source', 'jxl-encoder/tests/it/non_finite_action.rs',
            '--source', 'jxl-encoder/tests/it/sixteenbit_metadata.rs']
        with (a.output / (label + '.log')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        identity = json.loads((a.output / label / 'identity.json').read_text())
        rows.append(dict(label=label, source_sha256=hashlib.sha256(content).hexdigest(),
                         binary_sha256=identity['binary_sha256'], exit_code=identity['exit_code']))
        print(label + ': unchanged roundtrip assertions pass', flush=True)
finally:
    source.write_bytes(candidate)
    assert source.read_bytes() == candidate
(a.output / 'completed.json').write_text(json.dumps(dict(
    baseline_revision=a.baseline_revision, controls=rows,
    scope='Format-helper compatibility only; raw baseline does not prove level conformance'), indent=2) + '\n')
