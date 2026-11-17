// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Observe unmodified v0.12 natural orders and precision splitting.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <vector>
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/enc_progressive_split.h"

int main() {
  constexpr jxl::PassDefinition passes[] = {{8, 1, 2}, {8, 0, 0}};
  jxl::ProgressiveSplitter splitter;
  splitter.SetProgressiveMode(jxl::ProgressiveMode{passes});
  // Native bitstream codes corresponding to this encoder's nineteen types.
  for (uint8_t code : {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20}) {
    const auto strategy = jxl::AcStrategy::FromRawStrategy(code);
    size_t cx = strategy.covered_blocks_x(), cy = strategy.covered_blocks_y();
    jxl::CoefficientLayout(&cy, &cx);
    const size_t covered = cx * cy, size = covered * 64;
    std::vector<jxl::coeff_order_t> order(size);
    strategy.ComputeNaturalCoeffOrder(order.data());
    std::cout << "M\t" << unsigned(code) << '\t' << cx << '\t' << cy << '\t';
    for (size_t i = 0; i < covered; ++i) {
      if (i) std::cout << ',';
      std::cout << order[i];
    }
    std::cout << '\n';
    std::set<size_t> positions{0, 1, 2, 3, covered - 1, covered,
                              order[covered], order.back()};
    std::vector<int32_t> input(size), coarse(size), fine(size);
    int32_t* output[jxl::kMaxNumPasses] = {};
    output[0] = coarse.data();
    output[1] = fine.data();
    for (size_t position : positions) {
      const size_t rank = std::find(order.begin(), order.end(), position) - order.begin();
      if (rank == size) return 2;
      for (int32_t value : {2, 3, -2, -3}) {
        input[position] = value;
        splitter.SplitACCoefficients(input.data(), strategy, 0, 0, output);
        std::cout << "V\t" << unsigned(code) << '\t' << position << '\t'
                  << rank << '\t' << value << '\t' << coarse[position] << '\t'
                  << fine[position] << '\n';
        input[position] = 0;
      }
    }
  }
}
