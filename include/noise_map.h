// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2017, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef AFX_NOISE_MAP_H_
#define AFX_NOISE_MAP_H_

#include <vector>
#include <cmath>

#include "include/image.h"
#include "include/bounds.h"
#include "include/threading.h"
#include "include/median.h"

#include "include/convolution.h"

static const float bior44_array[] = {
  0.0f,
  -0.06453888262869706f,
  0.04068941760916406f,
  0.41809227322161724f,
  -0.7884856164055829f,
  0.41809227322161724f,
  0.04068941760916406f,
  -0.06453888262869706f,
  0.0f,
  0.0f,
};

// No go in VC10
// static const std::vector<float> bior44 = {
//   0.0f,
//   -0.06453888262869706f,
//   0.04068941760916406f,
//   0.41809227322161724f,
//   -0.7884856164055829f,
//   0.41809227322161724f,
//   0.04068941760916406f,
//   -0.06453888262869706f,
//   0.0f,
//   0.0f,
// };

namespace afx {

class NoiseMap {
 public:
  void Calculate(const afx::Image& in_image, afx::Image* out_image, unsigned int window_size) {
    ImageThreader threader;

    std::vector<float> bior44(bior44_array, bior44_array + sizeof(bior44_array) / sizeof(bior44_array[0]));

    afx::Image temp1(out_image->GetBounds());
    afx::Image temp2(out_image->GetBounds());
    afx::Convolution cv;
    cv.Seperable(in_image, &temp1, bior44);

    threader.ThreadRowChunks(out_image->GetBounds(), boost::bind(&NoiseMap::MAD_, this, _1, boost::cref(temp1), &temp2, window_size));
    threader.Wait();

    afx::Morphology morph;
    morph.Dilate(temp2, out_image, 3);

//     threader.ThreadImageChunks(out_image->GetBounds(), boost::bind(&NoiseMap::NormalizedVariance_, this, _1, boost::cref(in_image), &temp1, window_size));
//     threader.Wait();
//
//     morph.Dilate(temp1, &temp2, 3);
//
//     threader.ThreadImageChunks(out_image->GetBounds(), boost::bind(&NoiseMap::Combine_, this, _1, out_image, boost::cref(temp2)));
//     threader.Wait();

  }

 private:
  void MAD_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, unsigned int window_size) {
    std::vector<float> window_v;
    window_v.reserve(4 * (window_size * window_size + window_size) + 1);
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        for (int window_y = y - static_cast<int>(window_size); window_y <= y + static_cast<int>(window_size); ++window_y) {
          const float* in_ptr = in_image.GetPtrBnds(x - window_size, window_y);
          for (int window_x = x - static_cast<int>(window_size); window_x <= x + static_cast<int>(window_size); ++window_x) {
            window_v.push_back(*in_ptr);
            if (window_x >= in_image.GetBounds().x1() && window_x < in_image.GetBounds().x2()) { in_ptr++; }
          }
        }
        float median = MedianQuickSelect(window_v.data(), static_cast<int>(window_v.size()));
        for (auto it = window_v.begin(); it != window_v.end(); ++it) {
          *it = fabsf(*it - median);
        }
        // Adapting to Unknown Smoothness via Wavelet Shrinkage David L. Donoho Iain M. Johnstone
        *out_ptr++ = MedianQuickSelect(window_v.data(), static_cast<int>(window_v.size())) / 0.6745f;
      }
    }
  }

  void NormalizedVariance_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, unsigned int window_size) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        unsigned int n = 0;
        float mean = 0.0f;
        float s = 0.0f;
        for (int window_y = y - static_cast<int>(window_size); window_y <= y + static_cast<int>(window_size); ++window_y) {
          const float* in_ptr = in_image.GetPtrBnds(x - static_cast<int>(window_size), window_y);
          for (int window_x = x - static_cast<int>(window_size); window_x <= x + static_cast<int>(window_size); ++window_x) {
            // Welford variance algorithm for numerical stability
            n++;
            float delta = *in_ptr - mean;
            mean += delta / n;
            float delta2 = *in_ptr - mean;
            s += delta * delta2;
            if (window_x >= in_image.GetBounds().x1() && window_x < in_image.GetBounds().x2()) { in_ptr++; }
          }
        }
      *out_ptr++ = s / (n - 1) / mean;
      }
    }
  }

  void Combine_(const afx::Bounds& region, afx::Image* mad, const afx::Image& normalized_variance) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* mad_ptr = mad->GetPtr(region.x1(), y);
      const float* var_ptr = normalized_variance.GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        *mad_ptr++ *= powf(*var_ptr++, 0.25f);
      }
    }
  }
};

}  // namespace afx

#endif  // AFX_NOISE_MAP_H_

