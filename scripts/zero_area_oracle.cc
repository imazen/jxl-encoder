// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Public pinned encoder/decoder observations; level is reported, not assumed valid.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "jxl/color_encoding.h"
#include "jxl/decode.h"
#include "jxl/encode.h"

uint8_t Level(const std::vector<uint8_t>& bytes) {
  if(bytes.size()>2 && bytes[0]==255 && bytes[1]==10) return 5;
  for(size_t position=0;position+8<=bytes.size();) {
    size_t size=0;for(size_t i=0;i<4;++i) size=size*256+bytes[position+i];
    if(size<8 || position+size>bytes.size()) return 0;
    if(std::string(reinterpret_cast<const char*>(bytes.data()+position+4),4)=="jxll")
      return size==9?bytes[position+8]:0;
    position+=size;
  }
  return 5;
}

int Run(const std::filesystem::path& directory, unsigned zero_count, bool lossless) {
  const uint32_t width=4096,height=4096;
  const auto name=std::string(lossless?"lossless-":"lossy-")+std::to_string(zero_count);
  std::cerr<<name<<" stage=encoding\n";
  JxlEncoder* encoder=JxlEncoderCreate(nullptr);
  auto ok=[](JxlEncoderStatus status){return status==JXL_ENC_SUCCESS;};
  JxlBasicInfo info;JxlEncoderInitBasicInfo(&info);
  info.xsize=width;info.ysize=height;info.bits_per_sample=8;info.num_color_channels=3;
  info.have_animation=JXL_TRUE;info.animation.tps_numerator=100;
  info.animation.tps_denominator=1;info.uses_original_profile=lossless;
  if(!ok(JxlEncoderSetBasicInfo(encoder,&info))) return 10;
  JxlColorEncoding color;JxlColorEncodingSetToSRGB(&color,JXL_FALSE);
  if(!ok(JxlEncoderSetColorEncoding(encoder,&color))) return 11;
  auto* settings=JxlEncoderFrameSettingsCreate(encoder,nullptr);
  if(!ok(JxlEncoderSetFrameLossless(settings,lossless)) ||
     !ok(JxlEncoderSetFrameDistance(settings,lossless?0.0f:1.0f)) ||
     !ok(JxlEncoderFrameSettingsSetOption(settings,JXL_ENC_FRAME_SETTING_EFFORT,1))) return 12;
  JxlPixelFormat format{3,JXL_TYPE_UINT8,JXL_NATIVE_ENDIAN,0};
  std::vector<uint8_t> pixels(size_t(width)*height*3);
  for(unsigned index=0;index<=zero_count;++index) {
    JxlFrameHeader header;JxlEncoderInitFrameHeader(&header);
    header.duration=index==zero_count?10:0;
    header.layer_info.save_as_reference=1;
    header.layer_info.blend_info.blendmode=JXL_BLEND_REPLACE;
    std::fill(pixels.begin(),pixels.end(),index==zero_count?255:(index%2?128:0));
    if(!ok(JxlEncoderSetFrameHeader(settings,&header)) ||
       !ok(JxlEncoderAddImageFrame(settings,&format,pixels.data(),pixels.size()))) return 13;
  }
  JxlEncoderCloseInput(encoder);
  std::vector<uint8_t> bytes(8192);size_t used=0;
  for(;;) {
    auto* next=bytes.data()+used;size_t available=bytes.size()-used;
    auto status=JxlEncoderProcessOutput(encoder,&next,&available);used=bytes.size()-available;
    if(status==JXL_ENC_SUCCESS) break;
    if(status!=JXL_ENC_NEED_MORE_OUTPUT) return 14;
    bytes.resize(bytes.size()*2);
  }
  bytes.resize(used);JxlEncoderDestroy(encoder);
  std::ofstream stream(directory/(name+".jxl"),std::ios::binary);
  stream.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());stream.close();
  std::cerr<<name<<" stage=rendering bytes="<<bytes.size()<<"\n";
  JxlDecoder* decoder=JxlDecoderCreate(nullptr);
  JxlDecoderSetCoalescing(decoder,JXL_FALSE);
  JxlDecoderSubscribeEvents(decoder,JXL_DEC_BASIC_INFO|JXL_DEC_FRAME|JXL_DEC_FULL_IMAGE);
  JxlDecoderSetInput(decoder,bytes.data(),bytes.size());JxlDecoderCloseInput(decoder);
  std::ofstream fields(directory/(name+".tsv"));
  uint64_t consecutive=0,maximum=0;unsigned frames=0;
  for(;;) {
    auto status=JxlDecoderProcessInput(decoder);
    if(status==JXL_DEC_SUCCESS) break;
    if(status==JXL_DEC_BASIC_INFO) {
      JxlBasicInfo basic;
      if(JxlDecoderGetBasicInfo(decoder,&basic)!=JXL_DEC_SUCCESS ||
         basic.xsize!=width || basic.ysize!=height) return 15;
    } else if(status==JXL_DEC_FRAME) {
      JxlFrameHeader header;
      if(JxlDecoderGetFrameHeader(decoder,&header)!=JXL_DEC_SUCCESS) return 16;
      const uint64_t area=uint64_t(header.layer_info.xsize)*header.layer_info.ysize;
      consecutive=header.duration==0?consecutive+area:0;
      maximum=std::max(maximum,consecutive);
      fields<<frames<<'\t'<<header.layer_info.xsize<<'x'<<header.layer_info.ysize
            <<"\tduration="<<header.duration<<"\tconsecutive="<<consecutive<<'\n';
    } else if(status==JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t size=0;
      if(JxlDecoderImageOutBufferSize(decoder,&format,&size)!=JXL_DEC_SUCCESS ||
         size!=pixels.size() || JxlDecoderSetImageOutBuffer(decoder,&format,pixels.data(),size)!=JXL_DEC_SUCCESS) return 17;
    } else if(status==JXL_DEC_FULL_IMAGE) {
      const int expected=frames==zero_count?255:(frames%2?128:0);
      for(auto value:pixels) if(std::abs(int(value)-expected)>(lossless?0:2)) return 18;
      ++frames;
    } else return 19;
  }
  JxlDecoderDestroy(decoder);
  if(frames!=zero_count+1) return 20;
  std::cout<<name<<"\tlevel="<<int(Level(bytes))<<"\tframes="<<frames
           <<"\tmaximum_zero_area="<<maximum<<"\tall_samples_checked=1\n";
  return 0;
}

int main(int argc,char** argv) {
  if(JxlEncoderVersion()!=12000 || JxlDecoderVersion()!=12000) return 2;
  if(argc==1) return 0;
  if(argc!=2 || !std::filesystem::create_directory(argv[1])) return 3;
  for(bool lossless:{true,false}) for(unsigned count:{16u,17u})
    if(int result=Run(argv[1],count,lossless)) return result;
  return 0;
}
