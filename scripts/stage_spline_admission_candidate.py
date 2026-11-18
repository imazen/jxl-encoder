#!/usr/bin/env python3
"""Stage only the manual-spline candidate over a committed pre-fix owned-source base."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('base');p.add_argument('output',type=Path)
a=p.parse_args();root=a.manifest.resolve().parent
assert not(root/'.git').exists() and(root.parent.parent/'provenance.json').is_file()
a.output.mkdir(exist_ok=False)
sha=lambda content:hashlib.sha256(content).hexdigest()
def original(relative):return subprocess.check_output(['git','show',a.base+':'+relative]).decode()
def insert(text,needle,replacement):
 assert text.count(needle)==1,needle
 return text.replace(needle,replacement,1)
rows=[]
def write(relative,text):
 path=root/relative;before=path.read_bytes()if path.exists()else None
 if before is not None:
  backup=a.output/'before'/relative;backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(before)
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 rows.append(dict(path=relative,before_sha256=sha(before)if before is not None else None,after_sha256=sha(text.encode())))
relative='jxl-encoder/src/api/validate.rs';text=original(relative)
source=Path(relative).read_text();start=source.index('/// Validate manual spline data');end=source.index('/// JXL spec maximum dimension',start)
text=insert(text,'/// JXL spec maximum dimension',source[start:end]+'/// JXL spec maximum dimension');write(relative,text)
relative='jxl-encoder/src/api.rs';text=original(relative)
text=insert(text,'            ConfigRef::Lossy(cfg) => cfg.validate().map_err(at_from)?,','            ConfigRef::Lossy(cfg) => {\n                cfg.validate().map_err(at_from)?;\n                validate_lossy_manual_splines(cfg)?;\n            }')
text=insert(text,'    validate_lossy_source_encoding, validate_metadata_sizes, validate_source_gamma,','    validate_lossy_manual_splines, validate_lossy_source_encoding, validate_metadata_sizes, validate_source_gamma,')
start=text.index('impl LossyEncoder {');end=text.index('impl LosslessEncoder {',start);part=text[start:end]
part=insert(part,'            self.cfg.validate().map_err(at_from)?;\n            self.check_input_limits()?;','            self.cfg.validate().map_err(at_from)?;\n            validate_lossy_manual_splines(&self.cfg)?;\n            self.check_input_limits()?;');text=text[:start]+part+text[end:]
source=Path(relative).read_text();start=source.index('    /// Set manual splines');end=source.index('    pub fn with_splines',start)
old_start=text.index('    /// Set manual splines');old_end=text.index('    pub fn with_splines',old_start);text=text[:old_start]+source[start:end]+text[old_end:]
start=text.index('    fn encode_lossy_sub_via_jpeg_path(');at=text.index('        let mode = cfg.chroma_subsampling;',start)
text=text[:at]+'        if cfg.splines.as_ref().is_some_and(|splines| !splines.is_empty()) {\n            return Err(EncodeError::InvalidConfig {\n                message: "manual splines are not implemented with chroma subsampling".into(),\n            });\n        }\n'+text[at:];write(relative,text)
relative='jxl-encoder/src/api/animate.rs';text=original(relative);start=text.index('fn encode_animation_lossy(');at=text.index('    cfg.validate().map_err(at_from)?;',start)+len('    cfg.validate().map_err(at_from)?;')
source=Path(relative).read_text();ss=source.index('fn encode_animation_lossy(');lo=source.index('    validate_lossy_manual_splines(cfg)?;',ss);hi=source.index('    validate_animation_input(',lo)
text=text[:at]+'\n'+source[lo:hi]+text[at:];write(relative,text)
relative='jxl-encoder/src/vardct/encoder.rs';text=original(relative)
text=insert(text,'                    width,\n                    height,\n                );\n                {\n                    let mut xyb = [','                    width,\n                    height,\n                )?;\n                {\n                    let mut xyb = [')
source=Path(relative).read_text()
for function in ['encode_inner','encode_from_precomputed_with_extras','encode_from_pre_quantized_ac_with_extras']:
 ss=source.index('    '+('fn 'if function=='encode_inner'else'pub fn ')+function+'(')
 lo=source.index('        if let Some(splines) = &self.splines {',ss);hi=source.index('        if let Some(ce) = &self.color_encoding {',lo)
 ts=text.index('    '+('fn 'if function=='encode_inner'else'pub fn ')+function+'(');at=text.index('        if let Some(ce) = &self.color_encoding {',ts)
 text=text[:at]+source[lo:hi]+text[at:]
write(relative,text)
relative='jxl-encoder/src/vardct/bitstream.rs';text=original(relative);source=Path(relative).read_text();ss=source.index('    pub(crate) fn encode_frame_to_writer(');lo=source.index('        if let Some(splines) = &self.splines {',ss);hi=source.index('        // Reuse the full encode pipeline',lo)
ts=text.index('    pub(crate) fn encode_frame_to_writer(');at=text.index('        crate::error::check_stop(stop)?;',ts)+len('        crate::error::check_stop(stop)?;');text=text[:at]+'\n'+source[lo:hi]+text[at:];write(relative,text)
for relative in ['jxl-encoder/src/vardct/splines.rs','jxl-encoder/tests/spline_admission_contract.rs','jxl-encoder/src/vardct/spline_admission_tests.rs','jxl-encoder/src/vardct/spline_quantization_native.tsv']:
 write(relative,Path(relative).read_text())
(a.output/'stage.json').write_text(json.dumps(dict(base=a.base,files=rows),indent=2)+'\n')
print('Staged spline implementation and regression files; unrelated lane overlays are not imported.')
