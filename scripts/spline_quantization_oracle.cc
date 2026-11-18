// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Read the unmodified v0.12 Create result through its declared encoder friend.
#include <iostream>
#include "lib/jxl/splines.h"
namespace jxl {
class QuantizedSplineEncoder {
 public:
  static void Print(const QuantizedSpline& spline) {
    for (const auto& p : spline.control_points_)
      std::cout << '\t' << p.first << '\t' << p.second;
    for (const auto& channel : spline.color_dct_)
      for (int value : channel) std::cout << '\t' << value;
    for (int value : spline.sigma_dct_) std::cout << '\t' << value;
    std::cout << '\n';
  }
};
}
jxl::Status Run() {
  for (int shape : {0, 1}) {
    jxl::Spline source{};
    if (shape == 0) source.control_points = {{3.49f, 7.49f}, {3.51f, 7.49f}, {6.f, 8.f}};
    else source.control_points = {{20.f, 7.f}};
    source.color_dct[0][0] = .02f;
    source.color_dct[0][2] = -.04f;
    source.color_dct[1][0] = .03f;
    source.color_dct[1][1] = -.05f;
    source.color_dct[2][0] = -.06f;
    source.color_dct[2][3] = .08f;
    source.sigma_dct[0] = 1.5f;
    source.sigma_dct[2] = -.2f;
    for (int adjustment : {-3, 0, 5}) {
      JXL_ASSIGN_OR_RETURN(auto result, jxl::QuantizedSpline::Create(source, adjustment, .03125f, 1.125f));
      std::cout << shape << '\t' << adjustment;
      jxl::QuantizedSplineEncoder::Print(result);
    }
  }
  jxl::Spline empty{};
  JXL_ENSURE(!jxl::QuantizedSpline::Create(empty, 0, 0.f, 1.f).ok());
  return true;
}

int main() { return Run() ? 0 : 1; }
