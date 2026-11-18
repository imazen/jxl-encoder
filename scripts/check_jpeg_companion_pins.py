#!/usr/bin/env python3
"""Verify owned JPEG companion production sources against the requested Git pins."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('producer',type=Path);p.add_argument('decoder',type=Path);p.add_argument('output',type=Path);a=p.parse_args();root=a.manifest.resolve().parent;a.output.mkdir(exist_ok=False)
assert not(root/'.git').exists() and(root.parent.parent/'provenance.json').is_file()
pins=dict(line.split('\t')for line in(root/'.github/sibling-revisions.tsv').read_text().splitlines()[1:]if line)
rows=[]
for name,repo in [('zenjpeg',a.producer),('zenjxl-decoder',a.decoder)]:
 revision=pins['imazen/'+name]
 names=subprocess.check_output(['git','-C',str(repo),'ls-tree','-r','--name-only',revision]).decode().splitlines()
 selected=[n for n in names if '/src/'in n or n.endswith('/Cargo.toml')]
 assert selected
 files=[]
 for relative in selected:
  expected=subprocess.check_output(['git','-C',str(repo),'show',revision+':'+relative]);path=root.parent/name/relative
  assert path.read_bytes()==expected,(revision,path)
  files.append({'path':relative,'sha256':hashlib.sha256(expected).hexdigest()})
 rows.append({'repository':name,'revision':revision,'files':files})
(a.output/'identity.json').write_text(json.dumps({'manifest':str(a.manifest),'companions':rows},indent=2)+'\n')
print('Both JPEG companion production source trees match their requested Git pins.')
