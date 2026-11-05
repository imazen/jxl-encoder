#!/usr/bin/env python3
"""Check the token-free surface, including bare token re-exports."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
crate = root.name + '-simd'
files = list((root / 'docs/public-api').rglob(crate + '*.txt'))
assert files and len(files) % 3 == 0, 'Regenerate each target’s three SIMD API snapshots first'
count = 0
for path in files:
    for line in path.read_text().splitlines():
        if not line.startswith('pub '):
            continue
        # Re-exports can be bare names, so checking only namespace text misses them.
        if line.startswith('pub use '):
            names = re.findall(r'\b\w*Token\b', line)
            allowed = {'SimdToken', 'X64V3Token', 'NeonToken', 'Wasm128Token'} if crate == 'jxl-encoder-simd' else {
                'ScalarToken', 'X64V2Token', 'X64V3Token', 'X64V4Token',
                'NeonToken', 'Wasm128Token', '__SimdToken',
            }
            assert set(names) <= allowed, (path, line)
        if 'target_features::' in line or '::from_context(' in line:
            assert not re.search(r'archmage|magetypes|\b\w*Token\b', line), (path, line)
            count += 1
assert count, 'No token-free API found in snapshots'
# Source gate also sees NEON/WASM functions absent from a native x86 snapshot.
source = (root / crate / 'src/target_features.rs').read_text()
for signature in re.findall(r'pub fn [^{]+', source):
    assert not re.search(r'archmage|magetypes|\b\w*Token\b', signature), signature
print(f'{crate}: checked token-free signatures and legacy token re-exports')
