#!/usr/bin/env python3
"""Check explicit RGB APP0 source semantics independently with libjpeg-turbo."""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path
from PIL import Image, __version__, features

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('fixture',type=Path)
p.add_argument('djpeg',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(exist_ok=False)
version = subprocess.run([str(a.djpeg), '-version'], capture_output=True, check=True)
def decoded(data,name):
 source=a.output/(name+'.jpg');source.write_bytes(data)
 target=a.output/(name+'.ppm')
 result=subprocess.run([str(a.djpeg),'-rgb','-outfile',str(target),str(source)],capture_output=True)
 (a.output/(name+'.log')).write_bytes(result.stdout+result.stderr)
 assert result.returncode==0,(name,result.stderr)
 return Image.open(target).convert('RGB')
original=a.fixture.read_bytes()
def sha(data):return hashlib.sha256(data).hexdigest()
def strip_adobe(data):
 out=bytearray(data[:2]);at=2
 while at<len(data):
  assert data[at]==255
  marker=data[at+1];end=at+2+int.from_bytes(data[at+2:at+4],'big')
  if marker!=0xee:out.extend(data[at:end])
  if marker==0xda:out.extend(data[end:]);break
  at=end
 return bytes(out)
rows=[]
for adobe in [False,True]:
 baseline=original if adobe else strip_adobe(original)
 expected=decoded(baseline,f'baseline-{adobe}')
 for index,payload in enumerate([b'',b'JFIF',b'JFIFx',b'JFXX\0extension',b'Private APP0 data']):
  source=baseline[:2]+bytes([255,0xe0])+(len(payload)+2).to_bytes(2,'big')+payload+baseline[2:]
  observed=decoded(source,f'adobe-{adobe}-payload-{index}')
  assert observed.size==expected.size and observed.tobytes()==expected.tobytes(),(adobe,index)
  name=f'adobe-{adobe}-payload-{index}'
  (a.output/(name+'.jpg')).write_bytes(source)
  rows.append({'case':name,'size':observed.size,'baseline_sha256':sha(baseline),'input_sha256':sha(source),'rgb8_sha256':sha(observed.tobytes())})
(a.output/'identity.json').write_text(json.dumps({'Pillow_ppm_reader':__version__,'djpeg_version':(version.stdout+version.stderr).decode(),'djpeg_sha256':sha(a.djpeg.read_bytes()),'source_fixture_sha256':sha(original),'runner_sha256':sha(Path(__file__).read_bytes()),'rows':rows},indent=2)+'\n')
print('Every APP0 source preserves all independent RGB8 samples.')
