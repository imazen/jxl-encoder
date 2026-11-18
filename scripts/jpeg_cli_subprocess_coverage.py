#!/usr/bin/env python3
"""Bind CLI profiles to the exact retained executable, separately from its test driver."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('llvm',type=Path);p.add_argument('archive',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(exist_ok=False)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
identity=json.loads((a.archive/'identity.json').read_text());assert identity['exit_code']==0
for row in identity['sources']:assert digest(a.archive/'source'/row['path'])==row['sha256']
objects=json.loads((a.archive/'subprocess-objects.json').read_text())
objects=[r for r in objects if Path(r['original']).name in ('cjxl-rs','cjxl-rs.exe')]
assert len(objects)==1;row=objects[0];binary=a.archive/Path(row['staged']).name;assert digest(binary)==row['sha256']
profiles=[];commands=[]
for meta in sorted(a.archive.glob('grid-child-*.meta.txt')):
 lines=meta.read_text().splitlines();assert len(lines)==5
 pid,label,original,code,invocation=lines;assert original==row['original'] and code=='0'
 found=list(a.archive.glob(f'grid-child-{label}-{pid}-*.profraw'));assert len(found)==1,(meta,found)
 profiles.extend(found);commands.append({'pid':int(pid),'label':label,'invocation':invocation,'exit_code':0,'profile':found[0].name,'profile_sha256':digest(found[0])})
assert profiles and len(set(profiles))==len(profiles)
assert set(profiles)==set(a.archive.glob('grid-child-*.profraw'))
profile=a.output/'cli.profdata';subprocess.run([str(a.llvm/'llvm-profdata'),'merge','-sparse',*map(str,profiles),'-o',str(profile)],check=True)
with(a.output/'coverage.json').open('x')as out,(a.output/'coverage.stderr').open('x')as err:
 subprocess.run([str(a.llvm/'llvm-cov'),'export','--num-threads=1',str(binary),'--instr-profile='+str(profile)],stdout=out,stderr=err,check=True)
assert not(a.output/'coverage.stderr').read_text(),(a.output/'coverage.stderr').read_text()
report=json.loads((a.output/'coverage.json').read_text())
windows=[
 ('shared-grid','jxl-encoder/src/jpeg/encode.rs','let xsize_blocks = div_ceil(width, 8 << max_raw_hs)',2),
 ('transcode-selection','jxl-encoder-cli/src/main.rs','let want_jpeg_transcode =',1),
 ('transcode-call','jxl-encoder-cli/src/main.rs','let encoded = match cfg.encode_jpeg_transcode',1),
 ('coarsen-call','jxl-encoder-cli/src/main.rs','let encoded = match jxl_encoder::jpeg::encode_jpeg_recompress_auto_codestream',8),
]
summary=[]
for label,relative,marker,length in windows:
 lines=(a.archive/'source'/relative).read_text().splitlines();first=next(i+1 for i,line in enumerate(lines)if marker in line);last=first+length-1;regions={}
 for unit in report['data']:
  for fn in unit['functions']:
   for r in fn['regions']:
    if r[7] or not fn['filenames'][r[5]].endswith(relative)or r[0]<first or r[2]>last:continue
    key=tuple(r[:4]);regions[key]=max(regions.get(key,0),r[4])
 assert regions,label
 record={'label':label,'file':relative,'first':first,'last':last,'covered':sum(v>0 for v in regions.values()),'total':len(regions),'regions':[{'location':k,'count':v}for k,v in sorted(regions.items())]};summary.append(record)
 print(f"{label}: {record['covered']}/{record['total']}",flush=True)
(a.output/'identity.json').write_text(json.dumps({'parent_archive':str(a.archive),'binary':binary.name,'binary_sha256':digest(binary),'commands':commands,'profile_sha256':digest(profile)},indent=2)+'\n')
(a.output/'regions.json').write_text(json.dumps(summary,indent=2)+'\n')
