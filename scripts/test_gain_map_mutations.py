#!/usr/bin/env python3
"""Restore gain-map append faults in an owned export; require runtime failures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
root = a.manifest.resolve().parent
assert not (root / '.git').exists() and (root.parent.parent / 'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
cases = [
    ('direct', 'jxl-encoder/src/container.rs', 'b.box_type == *b"jhgm"',
     'b.box_type == *b"none"', 'duplicate_gain_map'),
    ('compressed', 'jxl-encoder/src/container.rs', 'b.box_type == *b"brob"',
     'b.box_type == *b"none"', 'duplicate_gain_map'),
    ('boundaries', 'jxl-encoder/src/container.rs',
     'top_level_boxes(jxl_data).ok_or_else(|| crate::EncodeError::InvalidInput {\n'
     '            message: "cannot append a gain map to malformed container boxes".into(),\n'
     '        })?', 'top_level_boxes(jxl_data).unwrap_or_default()', 'malformed_container_boundaries'),
    ('typed-propagation', 'jxl-encoder/src/hdr/bundle.rs',
     'container::append_gain_map_box(jxl_data, &payload)',
     'container::append_gain_map_box(jxl_data, &payload).or_else(|_| Ok(jxl_data.to_vec()))',
     'duplicate_gain_map'),
]
records = []
for name, relative, old, new, selection in cases:
    path = root / relative
    original = path.read_bytes()
    text = original.decode()
    assert text.count(old) == 1, (name, text.count(old))
    mutated = text.replace(old, new, 1).encode()
    path.write_bytes(mutated)
    try:
        out = a.output / name
        command = [sys.executable, str(Path(__file__).with_name('run_selected_test_evidence.py')),
                   str(a.manifest), str(out), selection, '--test-target', 'gain_map_duplicate_contract',
                   '--features', '__expert,parallel,hdr-gainmap,brotli-metadata',
                   '--source', 'jxl-encoder/src/container.rs', '--source', 'jxl-encoder/src/hdr/bundle.rs']
        with (a.output / (name + '.log')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        identity = json.loads((out / 'identity.json').read_text())
        failure = (out / 'test.log').read_text()
        assert result.returncode != 0 and identity['exit_code'] != 0
        assert 'test result: FAILED.' in failure and 'panicked at' in failure
        records.append(dict(case=name, exit_code=result.returncode,
                            original_sha256=hashlib.sha256(original).hexdigest(),
                            mutated_sha256=hashlib.sha256(mutated).hexdigest(),
                            binary_sha256=identity['binary_sha256']))
        print(name + ': runtime regression detects restored fault', flush=True)
    finally:
        assert path.read_bytes() == mutated, 'owned candidate unexpectedly changed'
        path.write_bytes(original)
        assert path.read_bytes() == original
    (a.output / 'faults.json').write_text(json.dumps(records, indent=2) + '\n')
