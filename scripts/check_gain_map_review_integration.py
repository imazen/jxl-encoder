#!/usr/bin/env python3
"""Validate the gain-map API review against committed main, restoring owned overlays."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path)
p.add_argument('review')
p.add_argument('output',type=Path)
p.add_argument('producer_repo',type=Path)
p.add_argument('decoder_repo',type=Path)
a=p.parse_args();a.output.mkdir(exist_ok=False)
root=a.manifest.resolve().parent
assert not (root/'.git').exists() and (root.parent.parent/'provenance.json').is_file()
files=['jxl-encoder/src/container.rs','jxl-encoder/src/hdr/bundle.rs','jxl-encoder/tests/it/compat_surface_76.rs','jxl-encoder/tests/gain_map_duplicate_contract.rs','jxl-encoder/tests/jpeg_sampling_contract.rs','jxl-encoder/src/jpeg/encode.rs']
original={}
rows=[]
try:
 for relative in files:
  path=root/relative
  original[path]=path.read_bytes() if path.exists() else None
  content=subprocess.check_output(['jj','file','show','-r',a.review,relative])
  backup=a.output/'original'/relative;backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(original[path] or b'')
  path.write_bytes(content)
  rows.append({'path':str(path),'before_sha256':hashlib.sha256(original[path]).hexdigest() if original[path] is not None else None,'overlay_sha256':hashlib.sha256(content).hexdigest()})
 pin_text=subprocess.check_output(['jj','file','show','-r',a.review,'.github/sibling-revisions.tsv'])
 assert (root/'.github/sibling-revisions.tsv').read_bytes()==pin_text
 pins=dict(line.split('\t')for line in pin_text.decode().splitlines()[1:]if line)
 dependencies=[]
 for name,repository in [('zenjpeg',a.producer_repo),('zenjxl-decoder',a.decoder_repo)]:
  revision=pins['imazen/'+name]
  paths=subprocess.check_output(['git','-C',str(repository),'ls-tree','-r','--name-only',revision,'--',name+'/src']).decode().splitlines()
  assert paths
  for relative in paths:
   expected=subprocess.check_output(['git','-C',str(repository),'show',revision+':'+relative])
   actual=root.parent/name/relative
   assert actual.read_bytes()==expected,actual
  dependencies.append({'repository':name,'revision':revision,'production_files_verified':len(paths)})
 (a.output/'dependencies.json').write_text(json.dumps(dependencies,indent=2)+'\n')
 (a.output/'overlay.json').write_text(json.dumps(rows,indent=2)+'\n')
 commands=[['cargo','test','--locked','--manifest-path',str(a.manifest),'-p','jxl-encoder','--features','__expert,parallel,hdr-gainmap,brotli-metadata,jpeg-reencoding','--test','gain_map_duplicate_contract','--test','jpeg_sampling_contract','--','--test-threads=1'],['cargo','test','--locked','--manifest-path',str(a.manifest),'-p','jxl-encoder','--features','__expert,hdr-gainmap,jpeg-reencoding','--test','it','compat_surface_76','--','--test-threads=1']]
 for index,command in enumerate(commands):
  with(a.output/f'check{index}.log').open('x')as out:subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True)
finally:
 for path,content in original.items():
  if content is None:
   path.unlink()
   assert not path.exists()
  else:
   path.write_bytes(content)
   assert path.read_bytes()==content
 (a.output/'restored.json').write_text(json.dumps({'restored':True,'files':len(original)},indent=2)+'\n')
print('Gain-map review and main integration passed; source overlays restored.')
