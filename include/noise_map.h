// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2017, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef INCLUDE_NOISE_MAP_H_
#define INCLUDE_NOISE_MAP_H_

#include <vector>
#include <cmath>

#include "include/image.h"
#include "include/bounds.h"
#include "include/threading.h"
#include "include/median.h"

#include "include/convolution.h"

static const float bior44[] = {
  0.0,
  -0.06453888262869706,
  0.04068941760916406,
  0.41809227322161724,
  -0.7884856164055829,
  0.41809227322161724,
  0.04068941760916406,
  -0.06453888262869706,
  0.0,
  0.0,
};

namespace afx {

class NoiseMap {
 public:
  void Calculate(const afx::Image& in_image, afx::Image* out_image, unsigned int window_size) {
    std::vector<float> wavelet(bior44, bior44 + 10);
    afx::Image temp(out_image->GetBounds());
    afx::Convolution cv;
    cv.Seperable(in_image, &temp, wavelet);
    MAD(temp, out_image, window_size);
  }

  void MAD(const afx::Image& in_image, afx::Image* out_image, unsigned int window_size) {
    ImageThreader threader;
    threader.ThreadImageChunks(in_image.GetBounds(), boost::bind(&NoiseMap::MAD_, this, _1, boost::cref(in_image), out_image, window_size));
    threader.Wait();
  }

 private:
  void MAD_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, unsigned int window_size) {
    unsigned int window_array_size = 4 * (window_size * window_size + window_size) + 1;
    float window_array[window_array_size];
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        unsigned int index = 0;
        for (int window_y = y - static_cast<int>(window_size); window_y <= y + static_cast<int>(window_size); ++window_y) {
          const float* in_ptr = in_image.GetPtrBnds(x - window_size, window_y);
          for (int window_x = x - static_cast<int>(window_size); window_x <= x + static_cast<int>(window_size); ++window_x) {
            window_array[index] = *in_ptr;
            index++;
            if (window_x >= in_image.GetBounds().x1() && window_x < in_image.GetBounds().x2()) { in_ptr++; }
          }
        }
        float median = MedianQuickSelect(window_array, window_array_size);
        for (int i = 0; i < window_array_size; ++i) {
          window_array[i] = fabsf(window_array[i] - median);
        }
        // Adapting to Unknown Smoothness via Wavelet Shrinkage David L. Donoho Iain M. Johnstone
        *out_image->GetPtr(x, y) = MedianQuickSelect(window_array, window_array_size) / 0.6745;
      }
    }
  }
};

}  // namespace afx

#endif  // INCLUDE_NOISE_MAP_H_

