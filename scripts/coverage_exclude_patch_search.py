#!/usr/bin/env python3
"""Exclude one unchanged patch-search function in an owned coverage export."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--restore', action='store_true', help='restore only the checksum-matching owned export')
a = p.parse_args()
root = a.manifest.resolve().parent
assert not (root / '.git').exists() and (root.parent.parent / 'provenance.json').is_file()
if a.restore:
    saved = json.loads((a.output / 'manifest.json').read_text())
    assert Path(saved['export_root']).resolve() == root
    # Check the complete set before modifying either file.
    for row in saved['files']:
        original = a.output / 'original' / row['path']
        assert hashlib.sha256(original.read_bytes()).hexdigest() == row['before_sha256']
        assert hashlib.sha256((root / row['path']).read_bytes()).hexdigest() == row['after_sha256']
    for row in saved['files']:
        (root / row['path']).write_bytes((a.output / 'original' / row['path']).read_bytes())
    print('Restored checksum-matching owned export; exclusion archive preserved.')
    raise SystemExit(0)
a.output.mkdir(exist_ok=False)
def sha(data): return hashlib.sha256(data).hexdigest()
name = 'find_text_like_patches_with_min_peak'
relative = Path('jxl-encoder/src/vardct/patches.rs')
source = (root / relative).read_text()
needle = f'pub(crate) fn {name}('
assert source.count(needle) == 1
attribute = '#[cfg_attr(coverage, coverage(off))]\n'
assert attribute + needle not in source
changed = source.replace(needle, attribute + needle, 1)
def body(text):
    lines = text.splitlines()
    start = next(i for i,line in enumerate(lines) if line.startswith(needle))
    end = next(i for i in range(start + 1,len(lines)) if lines[i] == '}')
    return '\n'.join(lines[start:end+1])
assert body(changed) == body(source)
lib_relative = Path('jxl-encoder/src/lib.rs')
lib = (root / lib_relative).read_text()
feature = '#![cfg_attr(coverage, feature(coverage_attribute))]\n'
assert feature not in lib
rows = []
for path, original, updated in [(relative, source, changed), (lib_relative, lib, feature+lib)]:
    target = a.output / 'original' / path
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(original)
    (root / path).write_text(updated)
    rows.append(dict(path=str(path), before_sha256=sha(original.encode()), after_sha256=sha(updated.encode())))
manifest = dict(export_root=str(root), files=rows,
                excluded_functions=['jxl_encoder::vardct::patches::'+name],
                function_body_sha256=sha(body(source).encode()),
                reason='Retained perf locates unrelated patch-template search overhead. Its body is unchanged; all affected level paths remain instrumented.',
                restore='Copy each original/<path> back only after verifying the corresponding after_sha256.')
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
