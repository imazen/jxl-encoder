#!/usr/bin/env python3
"""Stage committed decoder pins in the lane-owned mutable source closure, preserving old files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('export', type=Path)
p.add_argument('base')
p.add_argument('encoder', type=Path)
p.add_argument('decoder', type=Path)
p.add_argument('decoder_revision')
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
assert (a.export / 'provenance.json').is_file()
root = a.export / 'siblings/jxl-encoder'
assert not (root / '.git').exists()
assert a.export.name == 'optional-icc-owned-closure', 'only the authorized mutable closure'
sha = lambda value: hashlib.sha256(value).hexdigest()
rows = []

def preserve(path, label):
    if path.exists() and path.is_file():
        backup = a.output / 'before' / label
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        rows.append(dict(path=str(path), before_sha256=sha(path.read_bytes()), backup=str(backup)))

def unpack(repo, revision, destination, label):
    archive = a.output / (label + '.tar')
    with archive.open('xb') as stream:
        subprocess.run(['git', '-C', str(repo), 'archive', revision], stdout=stream, check=True)
    with tarfile.open(archive) as stream:
        for member in stream:
            relative = Path(member.name)
            assert not relative.is_absolute() and '..' not in relative.parts
            path = destination / relative
            if member.isfile():
                assert not path.is_symlink(), str(path)
                preserve(path, Path(label) / relative)
        stream.extractall(destination, filter='data')
    now = time.time()
    # Cargo must see the replacement path packages even when archive mtimes are old.
    with tarfile.open(archive) as stream:
        for member in stream:
            if member.isfile(): os.utime(destination / member.name, (now, now))

unpack(a.encoder, a.base, root, 'encoder-base')
unpack(a.decoder, a.decoder_revision, a.export / 'siblings/zenjxl-decoder', 'decoder-pin')
for relative in ['jxl-encoder/Cargo.toml', 'jxl-encoder-cli/Cargo.toml', '.github/sibling-revisions.tsv']:
    path = root / relative
    preserve(path, Path('candidate') / relative)
    shutil.copy2(a.encoder / relative, path)
    os.utime(path, None)
path = root / 'jxl-encoder/src/vardct/rate_control.rs'
source = path.read_text()
assert source.count('    use zenjxl_decoder::api::{') == 1
source = source.replace('    use zenjxl_decoder::api::{', '    use zensim_decoder::api::{')
old = '    let decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());'
new = '''    // Encoded controller inputs can use Level 10 sample precision. This
    // selects the decoder's format limit; request resource caps stay separate.
    let mut options = JxlDecoderOptions::default();
    options.limits.max_codestream_level = zensim_decoder::api::JxlCodestreamLevel::Level10;
    let decoder = JxlDecoder::<states::Initialized>::new(options);'''
assert source.count(old) == 1
path.write_text(source.replace(old, new))
(a.output / 'staged.json').write_text(json.dumps(dict(base=a.base,
    decoder_revision=a.decoder_revision, prior_files=rows,
    note='Owned mutable export only; every overwritten prior file retained. Lock resolution remains a separate gate.'), indent=2) + '\n')
print('Staged exact committed sources and candidate dependency manifests; prior files retained.')
