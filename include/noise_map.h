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

#include "include/image.h"
#include "include/bounds.h"
#include "include/threading.h"
#include "include/median.h"

#include "include/settings.h"

static const float coif3[] = {0.003793512864491014,
                              0.007782596427325418,
                              -0.023452696141836267,
                              -0.0657719112818555,
                              0.06112339000267287,
                              0.4051769024096169,
                              -0.7937772226256206,
                              0.42848347637761874,
                              0.07179982161931202,
                              -0.08230192710688598,
                              -0.03455502757306163,
                              0.015880544863615904,
                              0.00900797613666158,
                              0.0025745176887502236,
                              -0.0011175187708906016,
                              0.0004662169601128863,
                              7.098330313814125e-05,
                              -3.459977283621256e-05
                              };

namespace afx {

class WaveletTransform {
public:
  void DiagonalDetail(const afx::Image& in_image, afx::Image* out_image) {
    std::vector<float> wavelet(coif3, coif3 + 18);
    afx::Image temp(in_image.GetBounds());
    ImageThreader threader;
    threader.ThreadImageChunksY(in_image.GetBounds(), boost::bind(&WaveletTransform::ConvolveY_, this, _1, boost::cref(in_image), &temp, boost::cref(wavelet)));
    threader.Wait();
    threader.ThreadImageChunks(in_image.GetBounds(), boost::bind(&WaveletTransform::ConvolveX_, this, _1, boost::cref(temp), out_image, boost::cref(wavelet)));
    threader.Wait();
  }
private:
  void ConvolveX_(afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, const std::vector<float>& wavelet) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        float sum = 0.0f;
        int window_x = x - (wavelet.size() - 1);
        const float* in_ptr = in_image.GetPtrBnds(window_x, y);
        for (std::vector<float>::const_reverse_iterator it = wavelet.rbegin(); it < wavelet.rend(); ++it) {
          sum += *it * *in_ptr;
          if (window_x >= in_image.GetBounds().x1()) { in_ptr++; }
          window_x++;
        }
        for (std::vector<float>::const_iterator it = wavelet.begin() + 1; it < wavelet.end(); ++it) {
          sum += *it * *in_ptr;
          if (window_x < in_image.GetBounds().x2()) { in_ptr++; }
          window_x++;
        }
        *out_ptr++ = sum;
      }
    }
  }
  void ConvolveY_(afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, const std::vector<float>& wavelet) {
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float* out_ptr = out_image->GetPtr(x, region.y1());
      for (int y = region.y1(); y <= region.y2(); ++y) {
        float sum = 0.0f;
        int window_y = y - (wavelet.size() - 1);
        const float* in_ptr = in_image.GetPtrBnds(x, window_y);
        for (std::vector<float>::const_reverse_iterator it = wavelet.rbegin(); it < wavelet.rend(); ++it) {
          sum += *it * *in_ptr;
          if (window_y > in_image.GetBounds().y1()) { in_ptr = in_image.GetNextRow(in_ptr); }
          window_y++;
        }
        for (std::vector<float>::const_iterator it = wavelet.begin() + 1; it < wavelet.end(); ++it) {
          sum += *it * *in_ptr;
          if (window_y < in_image.GetBounds().y2()) { in_ptr = in_image.GetNextRow(in_ptr); }
          window_y++;
        }
        *out_ptr = sum;
        out_ptr = out_image->GetNextRow(out_ptr);
      }
    }
  }
};

class NoiseMap {
public:
  void MAD(const afx::Image& in_image, afx::Image* out_image) {
    ImageThreader threader;
    threader.ThreadImageChunks(in_image.GetBounds(), boost::bind(&NoiseMap::MAD_, this, _1, boost::cref(in_image), out_image, 3));
    threader.Wait();
  }
private:
  void MAD_(afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, unsigned window_size) {
    std::vector<float> window_values;
    window_values.reserve(4 * (window_size * window_size + window_size) + 1);
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        window_values.clear();
        for (int window_y = y - window_size; window_y <= y + window_size; ++window_y) {
          const float* in_ptr = in_image.GetPtrBnds(x - window_size, window_y);
          for (int window_x = x - window_size; window_x <= x + window_size; ++window_x) {
            window_values.push_back(*in_ptr);
            if (window_x >= in_image.GetBounds().x1() && window_x < in_image.GetBounds().x2()) { in_ptr++; }
          }
        }
        float median = MedianQuickSelect(&(*window_values.begin()), window_values.size());
//         for (std::vector<float>::iterator it = window_values.begin(); it < window_values.end(); ++it) {
//           *it = fabsf(*it - median);
//         }
        *out_ptr++ = 1;//MedianQuickSelect(window_values.data(), window_values.size());
      }
    }
  }

public:



};


}  // namespace afx

#endif  // INCLUDE_NOISE_MAP_H_

