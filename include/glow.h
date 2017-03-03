// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef INCLUDE_GLOW_H_
#define INCLUDE_GLOW_H_

#include <math.h>

#include <vector>
#include <algorithm>

#include "include/settings.h"
#include "include/cuda_helper.h"
#include "include/bounds.h"
#include "include/image.h"

namespace afx {

class GlowBase {
 public:
  void CreateKernel(float size, float softness, float falloff, float quality, unsigned int iterations = 200) {
    std::vector<Gaussian> gaussians;
    const float inner_size = (1.0f + 5.0f * softness) / 6.0f;
    unsigned int last_size = 0;
    for (int i = 0; i < iterations; ++i) {
      float lookup = static_cast<float>(i) / static_cast<float>(iterations - 1);
      float mapped_quality = cos(static_cast<float>(M_PI) * 0.5f * lookup) * (1.0f - quality) + quality;
      float sigma = powf(lookup, 1.0f / falloff) * (fmaxf(size, inner_size) - inner_size) + inner_size;
      unsigned int size  = std::max((static_cast<int>(ceilf(sigma_ptr_[i] * mapped_quality * (6.0f - 1.0f) + 1.0f)) ) | 1, 3) / 2;
      if (last_size > size) { size = last_size; }
      last_size = size;
      gaussians.push_back(sigma, size);
    }

    kernel_.clear();
    kernel_.resize(gaussians.back().size);

    float a_divisor = sqrtf(2.0f * static_cast<float>(M_PI));
    for(std::vector<Gaussian>::iterator it = gaussians.begin(); it < gaussians.end(); ++it) {
      unsigned int size = (*it).GetSize();
      float sigma = (*it).GetSigma();
      float two_sigma_squared = 2.0f * sigma * sigma;
      float a = 1.0f / (sigma * a_divisor);
      for (int i = 0; i < (*it).GetSize(); ++it) {
        kernel_[i] += a * expf(-static_cast<float>(i) * static_cast<float>(i) / two_sigma_squared);
      }
    }

  }
  std::vector<float> GetKernel() {
    return kernel_;
  }
  unsigned int GetKernelSize() {
    return kernel_.size();
  }

private:
  std::vector<float> kernel_;
};

class Glow : public GlowBase {


};

}  // namespace afx

#endif  // INCLUDE_GLOW_H_
