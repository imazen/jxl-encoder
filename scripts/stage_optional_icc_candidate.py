#!/usr/bin/env python3
"""Stage only the optional source-to-ICC candidate over an owned pinned export."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('baseline', type=Path)
p.add_argument('candidate', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
baseline = a.baseline.resolve()
candidate = a.candidate.resolve()
assert (baseline.parent.parent / 'provenance.json').is_file()
assert not (baseline / '.git').exists()
assert not a.output.exists()
a.output.mkdir()
siblings = a.output / 'siblings'
siblings.mkdir()
root = siblings / 'jxl-encoder'
shutil.copytree(baseline, root, symlinks=True,
                ignore=shutil.ignore_patterns('target', '.git', '.jj', '.workongoing'))
for sibling in baseline.parent.iterdir():
    if sibling.name != 'jxl-encoder':
        (siblings / sibling.name).symlink_to(sibling.resolve(), target_is_directory=True)

def function(text, name):
    lines = text.splitlines(keepends=True)
    start, = [i for i, line in enumerate(lines) if line.startswith('    fn ' + name + '(')]
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == '    }\n')
    return ''.join(lines[start:end + 1])

api = Path('jxl-encoder/src/api.rs')
original = (root / api).read_text()
proposed = (candidate / api).read_text()
name = 'encode_lossy_sub_via_jpeg_path'
original = original.replace(function(original, name), function(proposed, name), 1)
marker = '    /// Chunk-4 / chunk-5 entry point for any non-`Full444`'
assert original.count(marker) == 1
helper = '    #[cfg(all(feature = "std", feature = "chroma-subsampling", feature = "jpeg-reencoding"))]\n'
helper += function(proposed, 'subsampled_icc_pixels') + '\n\n'
original = original.replace(marker, helper + marker, 1)
module = '#[cfg(all(feature = "std", feature = "chroma-subsampling", feature = "jpeg-reencoding"))]\nmod cms;\n'
assert 'mod cms;' not in original
(root / api).write_text(original + '\n' + module)

manifest = root / 'jxl-encoder/Cargo.toml'
text = manifest.read_text()
before = 'chroma-subsampling = ["dep:zenyuv"]'
after = 'chroma-subsampling = ["dep:zenyuv", "zenpixels-convert/cms-moxcms"]'
assert text.count(before) == 1
manifest.write_text(text.replace(before, after, 1))
copies = ['jxl-encoder/src/api/cms.rs', 'jxl-encoder/tests/source_color_contract.rs',
          'jxl-encoder/tests/source_color_contract/optional_icc.rs']
for name in copies:
    destination = root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate / name, destination)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

changed = [str(api), 'jxl-encoder/Cargo.toml', *copies]
provenance = dict(base_export=str(baseline), candidate=str(candidate),
    source_provenance=json.loads((baseline.parent.parent / 'provenance.json').read_text()),
    overlays=[dict(path=name, sha256=sha(root / name)) for name in changed],
    sibling_pins_sha256=sha(root / '.github/sibling-revisions.tsv'),
    lock_sha256=sha(root / 'Cargo.lock'))
(a.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(root / 'Cargo.toml')
