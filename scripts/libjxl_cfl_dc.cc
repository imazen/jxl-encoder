// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Reference objects retain their BSD license.
// The CFL expression below is from libjxl v0.12 AddVarDCTDC; quantizer
// construction and Gradient quantization call the unmodified reference objects.
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <jxl/version.h>
#include "lib/jxl/quantizer.h"
#include "lib/jxl/chroma_from_luma.h"

static_assert(JPEGXL_MAJOR_VERSION == 0 && JPEGXL_MINOR_VERSION == 12 &&
              JPEGXL_PATCH_VERSION == 0);
namespace jxl {
int QuantizeGradient(const int32_t*, size_t, size_t, size_t, size_t, size_t,
                     float, float);
}

int main() {
  int global_scale, quant_dc, precision;
  float y, b;
  while (std::cin >> global_scale >> quant_dc >> precision >> y >> b) {
    if (global_scale <= 0 || quant_dc <= 0 || precision < 0 || precision > 3)
      return 1;
    jxl::DequantMatrices matrices;
    jxl::Quantizer quantizer(matrices, quant_dc, global_scale);
    const float mul = static_cast<float>(1 << precision);
    int32_t qy = 0, qb = 0, legacy = 0;
    qy = jxl::QuantizeGradient(&qy, 1, 1, 0, 0, 1, y,
                              quantizer.GetInvDcStep(1) * mul);
    const float y_factor = quantizer.GetDcStep(1) / mul;
    jxl::ColorCorrelation correlation;
    correlation.SetYToBDC(0);
    const float cfl_factor = correlation.DCFactors()[2];
    qb = jxl::QuantizeGradient(&qb, 1, 2, 0, 0, 1,
                              b - qy * (y_factor * cfl_factor),
                              quantizer.GetInvDcStep(2) * mul);
    const float legacy_scaled = b * (quantizer.GetInvDcStep(2) * mul) - qy * 0.5f;
    legacy = jxl::QuantizeGradient(&legacy, 1, 2, 0, 0, 1, legacy_scaled, 1.0f);
    std::cout << qy << ',' << qb << ',' << legacy << ',' << std::setprecision(9)
              << quantizer.GetInvDcStep(1) * mul << ',' << y_factor << '\n';
  }
}
