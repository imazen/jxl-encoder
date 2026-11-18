#!/usr/bin/env python3
"""Qualify a checksum-selected source repair, then restore owned overlays."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('selection',type=Path);p.add_argument('base');p.add_argument('output',type=Path)
mode=p.add_mutually_exclusive_group()
mode.add_argument('--reference-property',action='store_true',help='retain the selected E4 repair and remove only the separate model candidate')
mode.add_argument('--reconstruction-model',action='store_true',help='qualify only the model selection against the supplied committed base')
mode.add_argument('--tc17-contract',action='store_true',help='qualify unchanged TC17 mapping and explicit physical normalization controls')
mode.add_argument('--gain-map-color',action='store_true',help='qualify only gain-map color selection against its committed base')
mode.add_argument('--progressive-cmyk',action='store_true',help='qualify fresh progressive CMYK caller tests on unchanged current production')
mode.add_argument('--still-frame-levels',action='store_true',help='qualify selected still/streaming frame-level transport against current committed production')
mode.add_argument('--delta-palette-float',action='store_true',help='qualify float palette target controls without unrelated production overlays')
mode.add_argument('--spline-admission',action='store_true',help='qualify the manual spline selection against its committed base')
p.add_argument('--cli-followup-before',action='store_true',help='run new before controls using the retained pure-main CLI child')
p.add_argument('--request-rate-prototype',action='store_true',help='qualify the private request rate-control adapter without exposing a public setter')
p.add_argument('--cli-rate-before',action='store_true',help='run actual-file CLI rate-control regressions against main source with expected before failures')
p.add_argument('--continue-progressive-checks',action='store_true',help='after retained complete runtime/library/lock passes, verify mechanical lint repair, direct std-feature-disabled callers and original fault')
p.add_argument('--check-recipe',choices=['lf-caller-before-check','request-rate-control-before-queue','gain-map-approved-contract-check','rate-control-feature-fix-check','lf-caller-animation-before-check','prepared-payload-fix-check','rate-control-metric-fix-check','animation-lf-refusal-check','rate-control-tile-lifetime-before-check','gain-map-final-continuation-check','gain-map-final-sensitivity-check','metric-map-final-check','rate-control-feature-continuation-check','lossless-config-limits-before-check','lossless-config-limits-after-check','request-limits-loop-before-check','request-limits-supplement-check','prepared-patch-alpha-supplement-check','raw-lf-upsampling-before-check','request-limits-existing-animation-check','prepared-metric-broad-check','request-limits-display-extents-check','raw-lf-upsampling-after-check','raw-lf-upsampling-continuation-check','gain-map-current-final-check','gain-map-import-format-check','prepared-legacy-lf-check','request-rate-control-current-private-check','request-rate-control-canonical-check','request-rate-control-typed-attribution-check','prepared-current-landing-check','request-rate-control-metadata-check','request-rate-control-independent-check','rate-control-direction-before-check','request-rate-control-unset-continuation-check','frame-level-current-continuation-check','rate-control-direction-after-check','rate-control-rounding-before-check','rate-control-direction-broad-check','frame-level-prepared-current-check','frame-level-core-test-closure-check','frame-level-core-format-check','decoder-pin-current-check','request-rate-control-public-library-check','request-rate-control-public-cli-check','request-rate-control-public-feature-check','request-rate-control-large-runtime-build','request-rate-control-public-mutations-check','container-policy-before-check','container-policy-final-check','container-policy-icc-final-check','container-policy-wrappers-lint-check','container-policy-boxes-supplement-check','request-rate-control-jpeg-route-check','request-rate-control-complete-cli-check','request-rate-control-lossless-refusal-check','container-policy-minimal-check','request-rate-control-cli-continuation-check','request-controller-patch-check','request-controller-patch-lint-check','request-decoder-border-pin-check','request-controller-dictionary-minimal-check','request-decoder-border-pin-continuation-check','animation-alpha-before-check','animation-alpha-fixed-check','alpha-large-tree-before-check','alpha-large-tree-independent-check','alpha-large-tree-continuation-check','animation-alpha-caller-check','alpha-coding-fault-check','animation-alpha-caller-continuation-check','animation-alpha-existing-precision-check','extra-count-before-check','extra-count-candidate-check','extra-count-sensitivity-check'],help='explicit additional caller qualification recipe')
p.add_argument('--check-arg',action='append',default=[],help='append an explicit argument to the selected check recipe')
p.add_argument('--spline-lint-final',action='store_true',help='verify the final mechanical test lint repair against the same production source')
p.add_argument('--continue-spline-checks',action='store_true',help='after unchanged production library/locks passed, run final caller/minimal/doc/lint checks')
a=p.parse_args();root=a.manifest.resolve().parent
assert not(root/'.git').exists()and(root.parent.parent/'provenance.json').is_file()
assert not (a.cli_rate_before or a.cli_followup_before or a.request_rate_prototype) or a.still_frame_levels
assert not a.continue_progressive_checks or a.progressive_cmyk
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
 reset_paths=[] if (a.reconstruction_model or a.spline_admission or a.gain_map_color or a.still_frame_levels) else (['jxl-encoder/src/vardct/resampling.rs'] if a.reference_property else ['jxl-encoder/src/modular/encode_tree.rs','jxl-encoder/src/modular/tree_learn.rs','jxl-encoder/src/vardct/resampling.rs'])
 if a.tc17_contract or a.delta_palette_float or a.progressive_cmyk:reset_paths=['jxl-encoder/src/api.rs','jxl-encoder/src/hdr/from_sdr.rs']
 if a.still_frame_levels:reset_paths=['jxl-encoder/src/hdr/from_sdr.rs','jxl-encoder/src/lib.rs','jxl-encoder/src/vardct/patches.rs']
 for relative in reset_paths:
  if relative not in selection:
   overlay(relative,get(relative))
 # A persistent owned export can retain older test overlays after other
 # isolated work. Reconcile tracked supporting callers to the selected base;
 # archive every differing original and restore it in finally.
 supporting=subprocess.check_output(['git','ls-tree','-r','--name-only',a.base,'--','jxl-encoder/tests','jxl-encoder/examples']).decode().splitlines()
 for relative in supporting:
  if relative not in selection:
   content=get(relative)
   if not (root/relative).exists() or (root/relative).read_bytes()!=content:
    overlay(relative,content)
 tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',a.base,'--','jxl-encoder/src']).decode().splitlines()
 manifests=[name for name in subprocess.check_output(['git','ls-tree','-r','--name-only',a.base]).decode().splitlines() if name.endswith('Cargo.toml')]
 for relative in tracked+manifests:
  if relative not in selection:
   content=get(relative)
   if not (root/relative).exists() or (root/relative).read_bytes()!=content:
    overlay(relative,content)
 verified=[]
 for relative in tracked:
  expected=(Path(selection[relative]['verified_source']).read_bytes()if relative in selection else get(relative))
  assert(root/relative).read_bytes()==expected,relative
  verified.append(dict(path=relative,sha256=sha(expected)))
 for relative in ['Cargo.lock','.github/sibling-revisions.tsv','jxl-encoder/tests/support/decoded_frames.rs']:
  expected=Path(selection[relative]['verified_source']).read_bytes()if relative in selection else get(relative)
  assert(root/relative).read_bytes()==expected,relative
 (a.output/'overlay.json').write_text(json.dumps(rows,indent=2)+'\n')
 (a.output/'production.json').write_text(json.dumps(dict(base=a.base,files=verified),indent=2)+'\n')
 recipe=('spline-lint-final-check' if a.spline_lint_final else ('spline-continuation-check' if a.continue_spline_checks else 'spline-isolated-check')) if a.spline_admission else ('reconstruction-model-isolated-check' if a.reconstruction_model else ('reference-property-isolated-check' if a.reference_property else 'delta-palette-isolated-check'))
 if a.progressive_cmyk:recipe='progressive-cmyk-final-check' if a.continue_progressive_checks else 'progressive-cmyk-current-source-check'
 if a.still_frame_levels:recipe='frame-level-cli-followup-before-check' if a.cli_followup_before else 'request-rate-control-prototype-check' if a.request_rate_prototype else ('frame-level-cli-before-check' if a.cli_rate_before else 'frame-level-current-source-check')
 if a.gain_map_color:recipe='gain-map-color-isolated-check'
 if a.tc17_contract:recipe='tc17-contract-isolated-check'
 if a.delta_palette_float:recipe='delta-palette-float-isolated-check'
 if a.check_recipe:
  assert a.still_frame_levels or a.gain_map_color
  assert (a.check_recipe in ['gain-map-approved-contract-check','gain-map-final-continuation-check','gain-map-final-sensitivity-check','gain-map-current-final-check','gain-map-import-format-check']) == a.gain_map_color
  recipe=a.check_recipe
 command=['just',recipe,str(a.manifest),*a.check_arg]
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
