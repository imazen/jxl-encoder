// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Public unmodified v0.12 controls for clocks and already-downsampled geometry.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "jxl/encode.h"
#include "jxl/color_encoding.h"

bool Encode(const std::filesystem::path& output, uint32_t width,
            bool animation, uint32_t second_timecode, bool lossless) {
  JxlEncoder* encoder=JxlEncoderCreate(nullptr);
  auto ok=[](JxlEncoderStatus status){return status==JXL_ENC_SUCCESS;};
  JxlBasicInfo info;JxlEncoderInitBasicInfo(&info);
  info.xsize=animation?64:width*8;info.ysize=animation?32:8;
  info.bits_per_sample=8;info.num_color_channels=3;info.uses_original_profile=lossless;
  info.have_animation=animation;
  info.animation.have_timecodes=animation;
  info.animation.tps_numerator=1000;info.animation.tps_denominator=1;
  if(!ok(JxlEncoderSetBasicInfo(encoder,&info))) return false;
  JxlColorEncoding color;JxlColorEncodingSetToSRGB(&color,JXL_FALSE);
  if(!ok(JxlEncoderSetColorEncoding(encoder,&color))) return false;
  auto* settings=JxlEncoderFrameSettingsCreate(encoder,nullptr);
  if(!ok(JxlEncoderSetFrameLossless(settings,lossless)) ||
     !ok(JxlEncoderSetFrameDistance(settings,lossless?0.0f:1.0f)) ||
     !ok(JxlEncoderFrameSettingsSetOption(settings,JXL_ENC_FRAME_SETTING_EFFORT,3))) return false;
  if(!animation && (!ok(JxlEncoderFrameSettingsSetOption(settings,JXL_ENC_FRAME_SETTING_RESAMPLING,8)) ||
       !ok(JxlEncoderFrameSettingsSetOption(settings,JXL_ENC_FRAME_SETTING_ALREADY_DOWNSAMPLED,1)))) return false;
  JxlPixelFormat format{3,JXL_TYPE_UINT8,JXL_NATIVE_ENDIAN,0};
  for(size_t frame=0;frame<(animation?2u:1u);++frame) {
    JxlFrameHeader header;JxlEncoderInitFrameHeader(&header);
    header.duration=10;header.timecode=frame?second_timecode:2;
    if(!ok(JxlEncoderSetFrameHeader(settings,&header))) return false;
    std::vector<uint8_t> pixels(animation?64*32*3:width*3,frame?255:0);
    if(!ok(JxlEncoderAddImageFrame(settings,&format,pixels.data(),pixels.size()))) return false;
  }
  JxlEncoderCloseInput(encoder);
  std::vector<uint8_t> bytes(8192);size_t used=0;
  for(;;) {
    auto* next=bytes.data()+used;size_t available=bytes.size()-used;
    auto status=JxlEncoderProcessOutput(encoder,&next,&available);used=bytes.size()-available;
    if(status==JXL_ENC_SUCCESS) break;
    if(status!=JXL_ENC_NEED_MORE_OUTPUT) return false;
    bytes.resize(bytes.size()*2);
  }
  std::ofstream stream(output,std::ios::binary);
  stream.write(reinterpret_cast<const char*>(bytes.data()),used);
  JxlEncoderDestroy(encoder);
  return bool(stream);
}
int main(int argc,char** argv) {
  if(JxlEncoderVersion()!=12000) return 2;
  if(argc==1) return 0;
  if(argc!=2 || !std::filesystem::create_directory(argv[1])) return 3;
  const std::filesystem::path directory(argv[1]);
  for(auto width:{32768u,32769u}) {
    const auto label="width-"+std::to_string(width);
    std::cout<<label<<'\t'<<Encode(directory/(label+".jxl"),width,false,0,false)<<'\n';
  }
  for(auto second:{1u,2u,3u}) for(bool lossless:{false,true}) {
    const auto label="clock-2-"+std::to_string(second)+(lossless?"-lossless":"-lossy");
    std::cout<<label<<'\t'<<Encode(directory/(label+".jxl"),0,true,second,lossless)<<'\n';
  }
}
