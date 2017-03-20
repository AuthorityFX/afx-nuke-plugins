// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2017, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef AFX_CONVOLUTION_H_
#define AFX_CONVOLUTION_H_

#include <vector>

#include "include/bounds.h"
#include "include/image.h"
#include "include/image_tools.h"
#include "include/threading.h"

namespace afx {

void NormalizeKernel(std::vector<float>* kernel, float scale = 1.0f) {
  float sum = 0.0f;
  for(auto it = kernel->begin(); it != kernel->end(); ++it) {
    sum += fabsf(*it);
  }
  float nomalize_scale_factor = scale / sum;
  for(auto it = kernel->begin(); it != kernel->end(); ++it) {
    *it = *it * nomalize_scale_factor;
  }
}

class Convolution {
 public:
  void Seperable(const afx::Image& in_image, afx::Image* out_image, const std::vector<float>& kernel) {
    if (!CheckPadding_(in_image.GetBounds(), out_image->GetBounds(), kernel.size() - 1)) {
      throw std::runtime_error("afx::Convolution - in_image must be padded by kernel size - 1");
    }
    if (kernel.size() < 2) {
      throw std::runtime_error("afx::Convolution - kernel size must be >= 2");
    }
    afx::Bounds in_region = in_image.GetBounds();
    afx::Bounds out_region = out_image->GetBounds();
    afx::Bounds temp_region = in_region;
    temp_region.SetX(out_region.x1(), out_region.x2());
    afx::Image temp_image(temp_region);

    ImageThreader threader;
    // Convolve rows - y valid for out_region, x valid for in_region
    threader.ThreadRowChunks(temp_region, boost::bind(&Convolution::ConvolveRows_, this, _1, boost::cref(in_image), &temp_image, boost::cref(kernel)));
    threader.Wait();
    // Convolve columns - x and y valid for out_region
    threader.ThreadColumnChunks(out_region, boost::bind(&Convolution::ConvolveColumns_, this, _1, boost::cref(temp_image), out_image, boost::cref(kernel)));
    threader.Wait();
  }

 private:
  bool CheckPadding_(const afx::Bounds& in_region, const afx::Bounds& out_region, unsigned int size) {
    if (out_region.x1() - in_region.x1() < size || in_region.x2() - out_region.x2() < size ||
        out_region.y1() - in_region.y1() < size || in_region.y2() - out_region.y2() < size
       ) {
      return false;
    } else {
      return true;
    }
  }

  void ConvolveRows_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, const std::vector<float>& kernel) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        float sum = 0.0f;
        const float* in_ptr = in_image.GetPtr(x - kernel.size() - 1, y);
        for (auto it = kernel.rbegin(); it != kernel.rend(); ++it) {
          sum += *it * *in_ptr++;
        }
        for (auto it = kernel.begin() + 1; it != kernel.end(); ++it) {
          sum += *it * *in_ptr++;;
        }
        *out_ptr++ = sum;
      }
    }
  }

  void ConvolveColumns_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, const std::vector<float>& kernel) {
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float* out_ptr = out_image->GetPtr(x, region.y1());
      for (int y = region.y1(); y <= region.y2(); ++y) {
        float sum = 0.0f;
        const float* in_ptr = in_image.GetPtr(x, y - kernel.size() - 1);
        for (auto it = kernel.rbegin(); it != kernel.rend(); ++it) {
          sum += *it * *in_ptr;
          in_ptr = in_image.GetNextRow(in_ptr);
        }
        for (auto it = kernel.begin() + 1; it != kernel.end(); ++it) {
          sum += *it * *in_ptr;
          in_ptr = in_image.GetNextRow(in_ptr);
        }
        *out_ptr = sum;
        out_ptr = out_image->GetNextRow(out_ptr);
      }
    }
  }
};

}  //  namespace afx

#endif  // AFX_CONVOLUTION_H_
