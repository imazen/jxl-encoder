#!/usr/bin/env python3
"""Require runtime detection of restored APP1 classification/admission faults."""
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
    ('exif-prefix', 'parse.rs', r'payload.len() > b"Exif\0\0".len() && ', '', 'strict_prefix_only'),
    ('xmp-prefix', 'parse.rs', r'payload.len() > b"http://ns.adobe.com/xap/1.0/\0".len()',
     'true', 'strict_prefix_only'),
    ('counts', 'jbrd.rs', 'if jpeg.app_data.len() != jpeg.app_marker_type.len() {',
     'if false {', 'parsed_metadata_classifications'),
    ('duplicate', 'jbrd.rs', 'if lifted[slot] {', 'if false {', 'nonempty_metadata'),
    ('early-entry', 'encode.rs', '    super::jbrd::validate_app_metadata(jpeg)?;\n',
     '', 'parsed_metadata_classifications'),
    ('direct-jbrd', 'jbrd.rs', '    validate_app_metadata(jpeg)?;\n',
     '', 'parsed_metadata_classifications'),
]
records = []
for name, filename, old, new, selection in cases:
    path = root / 'jxl-encoder/src/jpeg' / filename
    original = path.read_bytes()
    text = original.decode()
    assert text.count(old) == 1, (name, text.count(old))
    mutated = text.replace(old, new, 1).encode()
    path.write_bytes(mutated)
    try:
        out = a.output / name
        command = [sys.executable, str(Path(__file__).with_name('run_selected_test_evidence.py')),
                   str(a.manifest), str(out), selection, '--test-target', 'jpeg_sampling_contract',
                   '--features', '__expert,parallel,jpeg-reencoding',
                   '--source', 'jxl-encoder/src/jpeg/encode.rs',
                   '--source', 'jxl-encoder/src/jpeg/parse.rs',
                   '--source', 'jxl-encoder/src/jpeg/jbrd.rs',
                   '--source', 'jxl-encoder/tests/jpeg_sampling_contract/metadata.rs']
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
