#!/usr/bin/env python3
"""Qualify a checksum-selected source repair, then restore owned overlays."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('selection',type=Path);p.add_argument('base');p.add_argument('output',type=Path)
mode=p.add_mutually_exclusive_group()
mode.add_argument('--reference-property',action='store_true',help='retain the selected E4 repair and remove only the separate model candidate')
mode.add_argument('--reconstruction-model',action='store_true',help='qualify only the model selection against the supplied committed base')
mode.add_argument('--spline-admission',action='store_true',help='qualify the manual spline selection against its committed base')
p.add_argument('--spline-lint-final',action='store_true',help='verify the final mechanical test lint repair against the same production source')
p.add_argument('--continue-spline-checks',action='store_true',help='after unchanged production library/locks passed, run final caller/minimal/doc/lint checks')
a=p.parse_args();root=a.manifest.resolve().parent
assert not(root/'.git').exists()and(root.parent.parent/'provenance.json').is_file()
assert not (a.continue_spline_checks or a.spline_lint_final) or a.spline_admission
a.output.mkdir(exist_ok=False)
sha=lambda data:hashlib.sha256(data).hexdigest()
selection=json.loads(a.selection.read_text());original={};rows=[]
get=lambda path:subprocess.check_output(['git','show',a.base+':'+path])
def overlay(relative,content):
 path=root/relative
 if path not in original:
  original[path]=path.read_bytes()if path.exists()else None
  if original[path]is not None:
   backup=a.output/'original'/relative;backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(original[path])
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
 rows.append(dict(path=relative,before_sha256=sha(original[path])if original[path]is not None else None,overlay_sha256=sha(content)))
try:
 for relative,row in selection.items():
  content=Path(row['verified_source']).read_bytes();assert sha(content)==row['verified_sha256']
  if row['parent_sha256']is not None:assert sha(get(relative))==row['parent_sha256']
  overlay(relative,content)
 reset_paths=[] if (a.reconstruction_model or a.spline_admission) else (['jxl-encoder/src/vardct/resampling.rs'] if a.reference_property else ['jxl-encoder/src/modular/encode_tree.rs','jxl-encoder/src/modular/tree_learn.rs','jxl-encoder/src/vardct/resampling.rs'])
 for relative in reset_paths:
  overlay(relative,get(relative))
 tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',a.base,'--','jxl-encoder/src']).decode().splitlines()
 verified=[]
 for relative in tracked:
  expected=(Path(selection[relative]['verified_source']).read_bytes()if relative in selection else get(relative))
  assert(root/relative).read_bytes()==expected,relative
  verified.append(dict(path=relative,sha256=sha(expected)))
 for relative in ['Cargo.lock','.github/sibling-revisions.tsv','jxl-encoder/tests/support/decoded_frames.rs']:
  assert(root/relative).read_bytes()==get(relative),relative
 (a.output/'overlay.json').write_text(json.dumps(rows,indent=2)+'\n')
 (a.output/'production.json').write_text(json.dumps(dict(base=a.base,files=verified),indent=2)+'\n')
 recipe=('spline-lint-final-check' if a.spline_lint_final else ('spline-continuation-check' if a.continue_spline_checks else 'spline-isolated-check')) if a.spline_admission else ('reconstruction-model-isolated-check' if a.reconstruction_model else ('reference-property-isolated-check' if a.reference_property else 'delta-palette-isolated-check'))
 command=['just',recipe,str(a.manifest)]
 with(a.output/'checks.log').open('x')as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 (a.output/'verdict.json').write_text(json.dumps(dict(command=command,exit_code=result.returncode),indent=2)+'\n')
 if result.returncode:raise subprocess.CalledProcessError(result.returncode,command)
finally:
 for path,content in original.items():
  if content is None:path.unlink();assert not path.exists()
  else:path.write_bytes(content);assert path.read_bytes()==content
 (a.output/'restored.json').write_text(json.dumps(dict(restored=True,files=len(original)),indent=2)+'\n')
print('Selected source passed independently; exact owned overlays restored.')
