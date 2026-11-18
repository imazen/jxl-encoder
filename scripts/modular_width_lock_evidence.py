#!/usr/bin/env python3
"""Retain float-lock before/after streams without changing any expectation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('manifest', type=Path)
parser.add_argument('before_revision')
parser.add_argument('output', type=Path)
args = parser.parse_args()
root = args.manifest.resolve().parent
assert not (root / '.git').exists() and (root.parent.parent / 'provenance.json').is_file()
args.output.mkdir(exist_ok=False)
source = root / 'jxl-encoder/src/vardct/encoder.rs'
fixed = source.read_bytes()
before = subprocess.check_output(['jj', 'file', 'show', '-r', args.before_revision, 'jxl-encoder/src/vardct/encoder.rs'])
assert before != fixed
(root / 'jxl-encoder/tests/it/hash_lock_features.rs').write_bytes(Path('jxl-encoder/tests/it/hash_lock_features.rs').read_bytes())
cases = ['lossy_mg_rgb_f16_512x512_noise_e7_d1', 'lossy_mg_rgb_f32_512x512_noise_e7_d1', 'lossy_sg_rgb_f32_64x64_noise_e7_d1']
rows = []
try:
    for phase, text in [('before', before), ('after', fixed)]:
        output = args.output / phase
        output.mkdir()
        source.write_bytes(text)
        command = ['cargo', 'test', '--locked', '--manifest-path', str(args.manifest), '-p', 'jxl-encoder',
                   '--features', '__expert,jpeg-reencoding', '--test', 'it', '--no-run', '--message-format=json']
        with (output / 'build.jsonl').open('x') as log:
            subprocess.run(command, stdout=log, check=True)
        records = [json.loads(line) for line in (output / 'build.jsonl').read_text().splitlines()]
        binaries = [row['executable'] for row in records if row.get('reason') == 'compiler-artifact'
                    and row.get('executable') and row['target']['name'] == 'it']
        assert len(binaries) == 1
        binary = Path(binaries[0]);shutil.copy2(binary, output / 'test-binary')
        (output / 'encoder.rs').write_bytes(text)
        (output / 'hash_lock_features.rs').write_bytes((root / 'jxl-encoder/tests/it/hash_lock_features.rs').read_bytes())
        identity = {'build_command': command, 'encoder_sha256': hashlib.sha256(text).hexdigest(),
                    'binary_sha256': hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest(), 'runs': []}
        for case in cases:
            run = [str(binary), 'hash_lock_features::' + case, '--exact', '--nocapture', '--test-threads=1']
            with (output / (case + '.log')).open('x') as log:
                result = subprocess.run(run, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, JXL_FLOAT_LOCK_ARTIFACT_DIR=str((output / 'streams').resolve())))
            log = (output / (case + '.log')).read_text()
            assert result.returncode == (0 if phase == 'before' else 101), (case, phase, log)
            assert ('1 passed;' in log if phase == 'before' else 'bytes moved.' in log), log
            path = output / 'streams' / (case + '.jxl')
            assert path.is_file()
            identity['runs'].append({'command': run, 'exit_code': result.returncode,
                'stream_sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(), 'bytes': path.stat().st_size})
            if phase == 'after':
                rows.append({'name': case, 'before': str((args.output / 'before/streams' / (case + '.jxl')).resolve()),
                             'after': str(path.resolve()), 'width': 64 if '_sg_' in case else 512,
                             'height': 64 if '_sg_' in case else 512})
        (output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
        print(phase + ': retained three actual streams with unchanged expectations', flush=True)
finally:
    source.write_bytes(fixed)
    assert source.read_bytes() == fixed
(args.output / 'pairs.json').write_text(json.dumps(rows, indent=2) + '\n')
