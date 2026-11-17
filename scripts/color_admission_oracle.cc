// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Separate public admission from unmodified v0.12 CMS matrix observations.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include "jxl/encode.h"
#include "jxl/color_encoding.h"
#include "lib/jxl/cms/jxl_cms_internal.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/memory_manager_internal.h"

struct Case {
  const char* name;
  bool gray=false;
  double wx=.3127, wy=.3290;
  double rx=.64,ry=.33,gx=.30,gy=.60,bx=.15,by=.06;
  double gamma=0;
  bool lossless=true;
  uint8_t sample=64;
};
jxl::Status GammaFields() {
  JxlMemoryManager memory;JXL_RETURN_IF_ERROR(jxl::MemoryManagerInit(&memory,nullptr));
  for(uint32_t gamma:{1221u,10000000u}) {
    jxl::ColorEncoding color;JXL_RETURN_IF_ERROR(color.SetSRGB(jxl::ColorSpace::kRGB));
    color.SetRenderingIntent(jxl::RenderingIntent::kRelative);
    color.Tf().have_gamma=true;color.Tf().gamma=gamma;
    jxl::BitWriter writer(&memory);
    JXL_RETURN_IF_ERROR(jxl::Bundle::Write(color,&writer,jxl::LayerType::Header,nullptr));
    const auto bits=writer.BitsWritten();
    JXL_RETURN_IF_ERROR(writer.WithMaxBits(8,jxl::LayerType::Header,nullptr,[&]()->jxl::Status {writer.ZeroPadToByte();return true;}));
    std::cout<<"gamma-field\t"<<gamma<<'\t'<<bits<<'\t';
    for(uint8_t byte:writer.GetSpan())std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(byte);
    std::cout<<std::dec<<'\n';
  }
  return true;
}
int main() {
  if (JxlEncoderVersion()!=12000) return 2;
  if(!GammaFields())return 5;
  const Case cases[] = {
    {"srgb"},
    {"white-y-zero",false,.3,0},
    {"white-x-negative",false,-.1,.3},
    {"white-x-over-one",false,1.1,.3},
    {"white-y-over-one",false,.3,1.1},
    {"white-wire-zero",false,.3,.0000004},
    {"white-chad-range",false,.3,.666961},
    {"gray-fixed-range",true,.3,.000001},
    {"primaries-singular",false,.3127,.3290,.3,.3,.3,.3,.3,.3},
    {"primaries-raw-small",false,.3127,.3290,.312699,.328999,.312701,.328999,.3127,.329001},
    {"primaries-wire-singular",false,.3127,.3290,.3126997,.3289998,.3127001,.3290002,.9,.1},
    {"primaries-fixed-range",false,.3127,.3290,.1,.1,.100001,.1,.1,.1002},
    {"xyz-basis-d65",false,.3127,.3290,1,0,0,1,0,0},
    {"xyz-basis-d50",false,.3457,.3585,1,0,0,1,0,0},
    {"aces-ap0-d65",false,.3127,.3290,.7347,.2653,0,1,.0001,-.077},
    {"gamma-zero",false,.3127,.3290,.64,.33,.30,.60,.15,.06,-1},
    {"gamma-too-small",false,.3127,.3290,.64,.33,.30,.60,.15,.06,.0001},
    {"gamma-min",false,.3127,.3290,.64,.33,.30,.60,.15,.06,.0001221},
    {"gamma-max",false,.3127,.3290,.64,.33,.30,.60,.15,.06,1},
    {"gamma-over-one",false,.3127,.3290,.64,.33,.30,.60,.15,.06,2},
    {"gamma-min-white-lossless",false,.3127,.3290,.64,.33,.30,.60,.15,.06,.0001221,true,255},
    {"gamma-min-white-lossy",false,.3127,.3290,.64,.33,.30,.60,.15,.06,.0001221,false,255},
    {"gamma-max-white-lossless",false,.3127,.3290,.64,.33,.30,.60,.15,.06,1,true,255},
    {"gamma-max-white-lossy",false,.3127,.3290,.64,.33,.30,.60,.15,.06,1,false,255},
  };
  jxl::Matrix3x3 target;
  if(!jxl::PrimariesToXYZD50(.64f,.33f,.30f,.60f,.15f,.06f,.3127f,.3290f,target) || !jxl::Inv3x3Matrix(target))return 3;
  std::cout<<"# v0.12 public admission and actual CMS results; these are distinct controls\n"<<std::setprecision(17);
  for (const auto& c:cases) {
    JxlEncoder* e=JxlEncoderCreate(nullptr);
    if(!e)return 3;
    JxlBasicInfo info; JxlEncoderInitBasicInfo(&info);
    info.xsize=16;info.ysize=17;info.bits_per_sample=8;
    info.num_color_channels=c.gray?1:3;info.uses_original_profile=c.lossless?JXL_TRUE:JXL_FALSE;
    if(JxlEncoderSetBasicInfo(e,&info)!=JXL_ENC_SUCCESS)return 3;
    JxlColorEncoding color;JxlColorEncodingSetToSRGB(&color,c.gray);
    color.white_point=JXL_WHITE_POINT_CUSTOM;
    color.white_point_xy[0]=c.wx;color.white_point_xy[1]=c.wy;
    if(!c.gray){
      color.primaries=JXL_PRIMARIES_CUSTOM;
      color.primaries_red_xy[0]=c.rx;color.primaries_red_xy[1]=c.ry;
      color.primaries_green_xy[0]=c.gx;color.primaries_green_xy[1]=c.gy;
      color.primaries_blue_xy[0]=c.bx;color.primaries_blue_xy[1]=c.by;
    }
    if(c.gamma){color.transfer_function=JXL_TRANSFER_FUNCTION_GAMMA;color.gamma=c.gamma<0?0:c.gamma;}
    const bool accepted=JxlEncoderSetColorEncoding(e,&color)==JXL_ENC_SUCCESS;
    std::cout<<"admission\t"<<c.name<<'\t'<<accepted<<'\n';
    bool added=false, finished=false;
    std::vector<uint8_t> encoded;
    if(accepted) {
      auto* settings=JxlEncoderFrameSettingsCreate(e,nullptr);
      if(!settings || JxlEncoderSetFrameLossless(settings,c.lossless?JXL_TRUE:JXL_FALSE)!=JXL_ENC_SUCCESS ||
         (!c.lossless && JxlEncoderSetFrameDistance(settings,.01f)!=JXL_ENC_SUCCESS) ||
         JxlEncoderFrameSettingsSetOption(settings,JXL_ENC_FRAME_SETTING_EFFORT,3)!=JXL_ENC_SUCCESS)return 4;
      const JxlPixelFormat format{c.gray?1u:3u,JXL_TYPE_UINT8,JXL_NATIVE_ENDIAN,0};
      const std::vector<uint8_t> pixels(16*17*format.num_channels,c.sample);
      added=JxlEncoderAddImageFrame(settings,&format,pixels.data(),pixels.size())==JXL_ENC_SUCCESS;
      if(added) {
        JxlEncoderCloseInput(e); encoded.resize(4096);size_t used=0;
        for(;;) {
          uint8_t* next=encoded.data()+used;size_t avail=encoded.size()-used;
          const auto status=JxlEncoderProcessOutput(e,&next,&avail);used=encoded.size()-avail;
          if(status==JXL_ENC_SUCCESS){encoded.resize(used);finished=true;break;}
          if(status!=JXL_ENC_NEED_MORE_OUTPUT){encoded.clear();break;}
          encoded.resize(encoded.size()*2);
        }
      }
    }
    std::cout<<"production\t"<<c.name<<'\t'<<added<<'\t'<<finished<<'\t';
    for(uint8_t value:encoded)std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(value);
    std::cout<<std::dec<<'\n';
    JxlEncoderDestroy(e);
    jxl::Matrix3x3 source,chad,matrix;
    const bool source_ok=!!jxl::PrimariesToXYZD50(c.rx,c.ry,c.gx,c.gy,c.bx,c.by,c.wx,c.wy,source);
    const bool chad_ok=!!jxl::AdaptToXYZD50(c.wx,c.wy,chad);
    std::cout<<"matrix\t"<<c.name<<'\t'<<source_ok<<'\t'<<chad_ok;
    if(source_ok){
      jxl::Mul3x3Matrix(target,source,matrix);
      for(size_t y=0;y<3;++y)for(size_t x=0;x<3;++x)std::cout<<'\t'<<matrix[y][x];
      bool fixed=true;for(size_t y=0;y<3;++y)for(size_t x=0;x<3;++x)
        fixed=fixed && std::isfinite(source[y][x]) && source[y][x]>=-32768. && source[y][x]<=32767.9999847412109375;
      std::cout<<'\t'<<fixed;
    }
    std::cout<<'\n';
    if(source_ok) {
      jxl::Vector3 input{.5f,.2f,.1f},output;
      jxl::Mul3x3Vector(matrix,input,output);
      std::cout<<"pixel\t"<<c.name<<'\t'<<output[0]<<'\t'<<output[1]<<'\t'<<output[2]<<'\n';
    }
    if(chad_ok){std::cout<<"chad\t"<<c.name;for(size_t y=0;y<3;++y)for(size_t x=0;x<3;++x)std::cout<<'\t'<<chad[y][x];std::cout<<'\n';}
    if(c.gray)std::cout<<"gray-white\t"<<c.name<<'\t'<<c.wx/c.wy<<"\t1\t"<<(1-c.wx-c.wy)/c.wy<<'\n';
  }
  return 0;
}
