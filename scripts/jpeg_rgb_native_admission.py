#!/usr/bin/env python3
"""Observe pinned native RGB-sampling admission after independent source decoding."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('cjxl',type=Path);p.add_argument('djxl',type=Path);p.add_argument('djpeg',type=Path);p.add_argument('output',type=Path);p.add_argument('sources',nargs='+',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False)
def sha(data):return hashlib.sha256(data).hexdigest()
tools={}
for name in ['cjxl','djxl','djpeg']:
 path=getattr(a,name).resolve();r=subprocess.run([str(path),'--version'if name!='djpeg'else '-version'],capture_output=True,check=True);version=(r.stdout+r.stderr).decode()
 if name!='djpeg':assert 'v0.12.0 a7a9c78'in version,(name,version)
 tools[name]={'path':str(path),'version':version,'sha256':sha(path.read_bytes())}
rows=[]
for index,source in enumerate(a.sources):
 out=a.output/str(index);out.mkdir();data=source.read_bytes();(out/'source.jpg').write_bytes(data)
 def run(label,command):
  r=subprocess.run(command,capture_output=True);(out/(label+'.log')).write_bytes(r.stdout+r.stderr);return r.returncode
 assert run('source-decode',[tools['djpeg']['path'],'-rgb','-outfile',str(out/'source.ppm'),str(out/'source.jpg')])==0,source
 encode=run('native-encode',[tools['cjxl']['path'],str(out/'source.jpg'),str(out/'native.jxl'),'--lossless_jpeg=1','--num_threads=1','-e','3'])
 row={'source':str(source),'source_sha256':sha(data),'source_decode_exit':0,'native_encode_exit':encode}
 if encode==0:
  row['jxl_sha256']=sha((out/'native.jxl').read_bytes())
  row['native_render_exit']=run('native-render',[tools['djxl']['path'],str(out/'native.jxl'),str(out/'native.png'),'--bits_per_sample=16','--num_threads=1'])
  rebuild=run('native-rebuild',[tools['djxl']['path'],str(out/'native.jxl'),str(out/'rebuilt.jpg'),'--num_threads=1']);row['native_rebuild_exit']=rebuild
  row['native_jpeg_exact']=rebuild==0 and(out/'rebuilt.jpg').read_bytes()==data
 rows.append(row);(a.output/'observations.partial.json').write_text(json.dumps(rows,indent=2)+'\n')
(a.output/'identity.json').write_text(json.dumps({'tools':tools,'runner_sha256':sha(Path(__file__).read_bytes()),'observations':rows,'scope':'Observed reference admission and reconstruction; native success is not assumed.'},indent=2)+'\n')
for row in rows:print(json.dumps(row),flush=True)
