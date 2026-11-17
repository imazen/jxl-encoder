// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Public v0.12 encoder control. A zero-duration regular layer supplies the
// hidden background; this does not claim native ReferenceOnly production.
#include <array>
#include <iomanip>
#include <iostream>
#include <vector>
#include "jxl/encode.h"
#include "jxl/color_encoding.h"
int Run(bool lossless, bool hidden) {
  JxlEncoder* e=JxlEncoderCreate(nullptr);
  if(!e) return 1;
  auto ok=[](JxlEncoderStatus s){return s==JXL_ENC_SUCCESS;};
  JxlBasicInfo info; JxlEncoderInitBasicInfo(&info);
  info.xsize=16; info.ysize=16; info.bits_per_sample=8;
  info.num_color_channels=3; info.have_animation=JXL_TRUE;
  info.uses_original_profile=lossless;
  if(!ok(JxlEncoderSetBasicInfo(e,&info))) return 2;
  JxlColorEncoding color; JxlColorEncodingSetToSRGB(&color,JXL_FALSE);
  if(!ok(JxlEncoderSetColorEncoding(e,&color))) return 3;
  auto* s=JxlEncoderFrameSettingsCreate(e,nullptr);
  if(!s || !ok(JxlEncoderFrameSettingsSetOption(s,JXL_ENC_FRAME_SETTING_EFFORT,3)) ||
     !ok(JxlEncoderSetFrameDistance(s,0.01)) ||
     !ok(JxlEncoderSetFrameLossless(s,lossless))) return 4;
  const JxlPixelFormat fmt={3,JXL_TYPE_UINT8,JXL_NATIVE_ENDIAN,0};
  for(int frame=0;frame<(hidden?3:2);++frame){
    JxlFrameHeader h; JxlEncoderInitFrameHeader(&h);
    h.duration=(hidden && frame==1)?0:10;
    h.layer_info.save_as_reference=(hidden && frame==1)?2:1;
    bool add=frame==(hidden?2:1);
    h.layer_info.blend_info.blendmode=add?JXL_BLEND_ADD:JXL_BLEND_REPLACE;
    h.layer_info.blend_info.source=hidden?2:1;
    std::array<uint8_t,3> rgb=hidden?(frame==0?std::array<uint8_t,3>{8,16,24}:
       frame==1?std::array<uint8_t,3>{32,48,64}:std::array<uint8_t,3>{64,48,32}):std::array<uint8_t,3>{64,64,64};
    std::vector<uint8_t> pixels;
    for(int p=0;p<256;++p)pixels.insert(pixels.end(),rgb.begin(),rgb.end());
    if(!ok(JxlEncoderSetFrameHeader(s,&h)) || !ok(JxlEncoderAddImageFrame(s,&fmt,pixels.data(),pixels.size())))return 5;
  }
  JxlEncoderCloseInput(e);
  std::vector<uint8_t> bytes(4096);size_t used=0;
  for(;;){uint8_t* next=bytes.data()+used;size_t avail=bytes.size()-used;
    auto result=JxlEncoderProcessOutput(e,&next,&avail);used=bytes.size()-avail;
    if(result==JXL_ENC_SUCCESS)break;if(result!=JXL_ENC_NEED_MORE_OUTPUT)return 6;bytes.resize(bytes.size()*2);}
  JxlEncoderDestroy(e);
  std::cout<<(lossless?"lossless":"lossy")<<'-'<<(hidden?"hidden":"canvas")<<'\t';
  for(size_t i=0;i<used;++i)std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<unsigned>(bytes[i]);
  std::cout<<'\n';return 0;
}
int main(){for(bool lossless:{true,false})for(bool hidden:{false,true})if(int rc=Run(lossless,hidden))return rc;}
