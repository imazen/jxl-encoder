// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Calls unmodified v0.12 API admission and ICC predictor/inverse objects.
#include <iostream>
#include <vector>
#include "jxl/cms.h"
#include "jxl/encode.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/padded_bytes.h"
namespace jxl {
Status PredictICC(const uint8_t*, size_t, PaddedBytes*);
Status UnpredictICC(const uint8_t*, size_t, PaddedBytes*);
}

jxl::Status Run() {
  JxlMemoryManager memory;
  JXL_RETURN_IF_ERROR(jxl::MemoryManagerInit(&memory, nullptr));
  for (int gray : {0, 1}) {
    jxl::ColorEncoding color;
    JXL_RETURN_IF_ERROR(color.SetSRGB(gray ? jxl::ColorSpace::kGray : jxl::ColorSpace::kRGB));
    for (int channels : {1, 3}) {
      JxlEncoder* enc = JxlEncoderCreate(nullptr);
      JXL_RETURN_IF_ERROR(enc != nullptr);
      JxlEncoderSetCms(enc, *JxlGetDefaultCms());
      JxlBasicInfo info;
      JxlEncoderInitBasicInfo(&info);
      info.xsize = info.ysize = 8;
      info.num_color_channels = channels;
      info.uses_original_profile = JXL_TRUE;
      JXL_RETURN_IF_ERROR(JxlEncoderSetBasicInfo(enc, &info) == JXL_ENC_SUCCESS);
      auto status = JxlEncoderSetICCProfile(enc, color.ICC().data(), color.ICC().size());
      std::cout << "model:" << gray << ':' << channels << '\t' << (status == JXL_ENC_SUCCESS) << '\n';
      JxlEncoderDestroy(enc);
    }
  }
  for (size_t size : {size_t{132}, size_t{4194305}}) {
    std::vector<uint8_t> raw(size, size == 132 ? 0 : 0xAB);
    if (size == 132) { raw[3] = 132; raw[131] = 1; }
    jxl::ColorEncoding raw_color;
    raw_color.SetICCRaw(jxl::IccBytes(raw.begin(), raw.end()));
    JXL_RETURN_IF_ERROR(raw_color.WantICC() && !raw_color.HaveFields());
    JXL_RETURN_IF_ERROR(raw_color.ICC().size() == raw.size() &&
                        std::equal(raw.begin(), raw.end(), raw_color.ICC().data()));
    std::cout << "raw:" << size << ":internal_raw\t1\n";
    for (bool cms : {false, true}) {
      JxlEncoder* enc = JxlEncoderCreate(nullptr);
      JXL_RETURN_IF_ERROR(enc != nullptr);
      if (cms) JxlEncoderSetCms(enc, *JxlGetDefaultCms());
      JxlBasicInfo info;
      JxlEncoderInitBasicInfo(&info);
      info.xsize = info.ysize = 8;
      info.num_color_channels = 3;
      info.uses_original_profile = JXL_TRUE;
      JXL_RETURN_IF_ERROR(JxlEncoderSetBasicInfo(enc, &info) == JXL_ENC_SUCCESS);
      auto status = JxlEncoderSetICCProfile(enc, raw.data(), raw.size());
      std::cout << "raw:" << size << ":explicit_cms:" << cms << '\t' << (status == JXL_ENC_SUCCESS) << '\n';
      JxlEncoderDestroy(enc);
    }
  }
  for (size_t size : {132, 139, 140, 144}) {
    std::vector<uint8_t> input(size, 0);
    input[0] = static_cast<uint8_t>(size >> 24);
    input[1] = static_cast<uint8_t>(size >> 16);
    input[2] = static_cast<uint8_t>(size >> 8);
    input[3] = static_cast<uint8_t>(size);
    input[131] = 1;
    jxl::PaddedBytes predicted(&memory), decoded(&memory);
    JXL_RETURN_IF_ERROR(jxl::PredictICC(input.data(), input.size(), &predicted));
    bool ok = !!jxl::UnpredictICC(predicted.data(), predicted.size(), &decoded);
    std::cout << "tag:" << size << ":1\t" << ok << '\t'
              << (ok && decoded.size() == input.size() && std::equal(input.begin(), input.end(), decoded.data())) << '\n';
  }
  return true;
}
int main() { return Run() ? 0 : 1; }
