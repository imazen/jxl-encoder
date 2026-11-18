#!/usr/bin/env python3
"""Replay a frozen original JPEG writer in the owned mutable source export."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('revision')
p.add_argument('output', type=Path)
a = p.parse_args()
assert re.fullmatch(r'[0-9a-f]{8,40}', a.revision)
root = a.manifest.resolve().parent
repository = Path(__file__).resolve().parent.parent
assert not (root / '.git').exists() and (root.parent.parent / 'provenance.json').is_file()
paths = [root / 'jxl-encoder/src/jpeg' / name for name in ['encode.rs', 'parse.rs', 'jbrd.rs']]
originals = {path: path.read_bytes() for path in paths}
replayed = {}
try:
    for path in paths:
        relative = str(path.relative_to(root))
        content = subprocess.check_output(['git', 'show', a.revision + ':' + relative], cwd=repository)
        path.write_bytes(content)
        replayed[path] = content
    command = [sys.executable, str(Path(__file__).with_name('run_selected_test_evidence.py')),
               str(a.manifest), str(a.output), 'parsed_distinct_metadata',
               '--test-target', 'jpeg_sampling_contract', '--features', '__expert,parallel,jpeg-reencoding',
               '--source', 'jxl-encoder/src/jpeg/encode.rs',
               '--source', 'jxl-encoder/src/jpeg/parse.rs',
               '--source', 'jxl-encoder/src/jpeg/jbrd.rs',
               '--source', 'jxl-encoder/tests/jpeg_sampling_contract/metadata.rs']
    result = subprocess.run(command)
    identity = json.loads((a.output / 'identity.json').read_text())
    failure = (a.output / 'test.log').read_text()
    assert result.returncode != 0 and identity['exit_code'] != 0
    assert 'test result: FAILED.' in failure and 'full original reconstruction' in failure
    (a.output / 'baseline-replay.json').write_text(json.dumps(dict(
        revision=a.revision, exit_code=result.returncode,
        restored_sha256={str(path.relative_to(root)): hashlib.sha256(content).hexdigest()
                         for path, content in originals.items()},
        binary_sha256=identity['binary_sha256']), indent=2) + '\n')
finally:
    for path, content in originals.items():
        if path in replayed:
            assert path.read_bytes() == replayed[path], 'owned replay source unexpectedly changed'
        path.write_bytes(content)
        assert path.read_bytes() == content
