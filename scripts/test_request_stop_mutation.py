#!/usr/bin/env python3
"""Prove the public one-poll-stop regression detects a bypassed request latch."""
import argparse
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path)
p.add_argument('log',type=Path)
a=p.parse_args()
root=a.manifest.resolve().parent
if (root/'.git').exists() or not (root.parent.parent/'provenance.json').is_file():
    p.error('requires an owned prepare_clean_source.py export')
source=root/'jxl-encoder/src/api.rs'
original=source.read_text()
start=original.index('    fn encode_inner(self, pixels: &[u8]) -> Result<EncodeResult> {')
end=original.index('    fn encode_inner_with_stop(',start)
replacement='    fn encode_inner(self, pixels: &[u8]) -> Result<EncodeResult> {\n        self.encode_inner_with_stop(pixels)\n    }\n\n'
try:
    source.write_text(original[:start]+replacement+original[end:])
    with a.log.open('x') as sink:
        result=subprocess.run(['cargo','llvm-cov','--locked','--manifest-path',str(a.manifest),
            '-p','jxl-encoder','--features','__expert','--test','request_cancellation',
            '--no-report'],stdout=sink,stderr=subprocess.STDOUT)
    text=a.log.read_text()
    assert result.returncode==101 and 'test result: FAILED' in text
    assert 'observed stop was lost' in text or 'Ok(EncodeResult' in text
    print('bypassed request latch: regression detected')
finally:
    source.write_text(original)
    assert source.read_text()==original
