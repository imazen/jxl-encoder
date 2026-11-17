#!/usr/bin/env python3
"""Restore dependency caches while retaining test executables in an archive."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('archive',type=Path)
p.add_argument('debug_directory',type=Path)
a=p.parse_args()
if a.debug_directory.name!='debug' or not (a.debug_directory.parent/'CACHEDIR.TAG').is_file():
    p.error('destination must be the debug tree of an owned Cargo target')
files=0
size=0
for source in a.archive.rglob('*'):
    if source.is_dir(): continue
    if source.is_symlink(): p.error('unexpected archive symlink')
    relative=source.relative_to(a.archive)
    parts=relative.parts
    if parts[0]=='examples': continue
    if parts[0]=='.fingerprint' and len(parts)>1 and re.fullmatch(r'jxl-encoder-[0-9a-f]+',parts[1]): continue
    if source.name.startswith(('libjxl_encoder-','jxl_encoder-','.jxl_encoder-')): continue
    if (len(parts)==1 or parts[0]=='deps') and source.suffix=='':
        with source.open('rb') as f:
            if f.read(4)==b'\x7fELF': continue
    target=a.debug_directory/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): p.error('refusing existing cache file: '+str(target))
    shutil.copy2(source,target)
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            while chunk:=stream.read(1024*1024): h.update(chunk)
        return h.digest()
    assert digest(source)==digest(target),target
    files+=1
    size+=source.stat().st_size
print('restored dependency cache files:',files,'verified bytes:',size)
