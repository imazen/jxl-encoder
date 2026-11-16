#!/usr/bin/env python3
"""Mutation-check cancellation boundaries in an owned source export."""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('log', type=Path)
p.add_argument('--only', choices=['still', 'animation', 'jpeg'], default='still')
a = p.parse_args()
root = a.manifest.resolve().parent
if (root / '.git').exists() or not (root.parent.parent / 'provenance.json').is_file():
    p.error('requires an owned prepare_clean_source.py export')
source = root / ('jxl-encoder/src/jpeg/parse.rs' if a.only == 'jpeg' else 'jxl-encoder/src/api.rs')
original = source.read_text()
if a.only == 'still':
    start = original.index('    fn encode_inner(self, pixels: &[u8]) -> Result<EncodeResult> {')
    end = original.index('    fn encode_inner_with_stop(', start)
    changed = original[:start] + '    fn encode_inner(self, pixels: &[u8]) -> Result<EncodeResult> {\n        self.encode_inner_with_stop(pixels)\n    }\n\n' + original[end:]
    selection = 'shifted_request_propagates'
elif a.only == 'animation':
    start = original.rindex('    pub fn encode_animation(')
    body = original.index('    ) -> Result<Vec<u8>> {', start) + len('    ) -> Result<Vec<u8>> {')
    end = original.index('    fn encode_animation_with_request_stop(', body)
    changed = original[:body] + '\n        self.encode_animation_with_request_stop(animation, frames)\n    }\n\n' + original[end:]
    selection = 'animation_requests_retain'
else:
    start = original.index('    let (decoded, jbrd_meta) = decode_result.map_err(|e| {')
    end = original.index('    })?;', start) + len('    })?;')
    changed = original[:start] + '    let (decoded, jbrd_meta) = decode_result.map_err(|e| JpegError::Decode(format!("{e}")))?;' + original[end:]
    selection = 'low_level_jpeg_operations'
try:
    source.write_text(changed)
    with a.log.open('x') as sink:
        result = subprocess.run(['cargo', 'llvm-cov', '--locked', '--manifest-path', str(a.manifest),
            '-p', 'jxl-encoder', '--features', '__expert,jpeg-reencoding', '--test', 'request_cancellation',
            '--no-report', '--', selection], stdout=sink, stderr=subprocess.STDOUT)
    text = a.log.read_text()
    assert result.returncode == 101 and 'test result: FAILED' in text
    expected = 'Decode(' if a.only == 'jpeg' else 'observed stop was lost'
    assert expected in text or 'Ok(EncodeResult' in text
    print(a.only + ': regression detected', flush=True)
finally:
    source.write_text(original)
    assert source.read_text() == original
