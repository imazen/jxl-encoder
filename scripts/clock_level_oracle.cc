// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Public unmodified v0.12 controls for clocks and already-downsampled geometry.
#include <cstdlib>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "jxl/encode.h"
#include "jxl/decode.h"
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
bool Inspect(const std::filesystem::path& input, uint32_t width,
             bool animation, uint32_t second_timecode, bool lossless) {
  std::ifstream stream(input, std::ios::binary);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(stream)), {});
  JxlDecoder* decoder=JxlDecoderCreate(nullptr);
  JxlDecoderSubscribeEvents(decoder,JXL_DEC_BASIC_INFO|JXL_DEC_FRAME|JXL_DEC_FULL_IMAGE);
  JxlDecoderSetInput(decoder,bytes.data(),bytes.size());
  JxlDecoderCloseInput(decoder);
  JxlPixelFormat format{3,JXL_TYPE_UINT16,JXL_NATIVE_ENDIAN,0};
  std::vector<uint16_t> pixels;
  size_t frame=0;
  bool success=false;
  for(;;) {
    const auto status=JxlDecoderProcessInput(decoder);
    if(status==JXL_DEC_SUCCESS) { success=frame==(animation?2u:1u);break; }
    if(status==JXL_DEC_BASIC_INFO) {
      JxlBasicInfo info;
      if(JxlDecoderGetBasicInfo(decoder,&info)!=JXL_DEC_SUCCESS ||
         info.xsize!=(animation?64:width*8) || info.ysize!=(animation?32:8)) break;
    } else if(status==JXL_DEC_FRAME) {
      JxlFrameHeader header;
      if(JxlDecoderGetFrameHeader(decoder,&header)!=JXL_DEC_SUCCESS) break;
      std::cout<<input.filename().string()<<"\tframe="<<frame<<"\tclock="
               <<header.timecode<<"\tduration="<<header.duration<<'\n';
      if(animation && (header.timecode!=(frame?second_timecode:2) || header.duration!=10)) break;
    } else if(status==JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t size=0;
      if(JxlDecoderImageOutBufferSize(decoder,&format,&size)!=JXL_DEC_SUCCESS || size%2) break;
      pixels.resize(size/2);
      if(JxlDecoderSetImageOutBuffer(decoder,&format,pixels.data(),size)!=JXL_DEC_SUCCESS) break;
    } else if(status==JXL_DEC_FULL_IMAGE) {
      const int expected=frame?65535:0;
      bool valid=true;
      for(auto value:pixels) {
        const int error=std::abs(int(value)-expected);
        if(error>(lossless?0:328)) {valid=false;break;}
      }
      if(!valid) break;
      ++frame;
    } else break;
  }
  JxlDecoderDestroy(decoder);
  return success;
}
int main(int argc,char** argv) {
  if(JxlEncoderVersion()!=12000) return 2;
  if(argc==1) return 0;
  if(argc!=2 || !std::filesystem::create_directory(argv[1])) return 3;
  const std::filesystem::path directory(argv[1]);
  for(auto width:{32768u,32769u}) {
    const auto label="width-"+std::to_string(width);
    const auto path=directory/(label+".jxl");
    const bool encoded=Encode(path,width,false,0,false);
    const bool rendered=encoded && Inspect(path,width,false,0,false);
    std::cout<<label<<"\tencoded="<<encoded<<"\trendered="<<rendered<<'\n';
    if(!encoded || !rendered) return 4;
  }
  for(auto second:{1u,2u,3u}) for(bool lossless:{false,true}) {
    const auto label="clock-2-"+std::to_string(second)+(lossless?"-lossless":"-lossy");
    const auto path=directory/(label+".jxl");
    const bool encoded=Encode(path,0,true,second,lossless);
    const bool rendered=encoded && Inspect(path,0,true,second,lossless);
    std::cout<<label<<"\tencoded="<<encoded<<"\trendered="<<rendered<<'\n';
    if(!encoded || !rendered) return 5;
  }
}
