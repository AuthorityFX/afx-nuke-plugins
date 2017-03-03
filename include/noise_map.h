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

static const float bior44[] = {0.0
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

class WaveletTransform {
public:
  void StationaryWaveletTransformDiagonalOnly(const afx::Image& in_image, afx::Image* out_image, unsigned int level = 0) {
    std::vector<float> wavelet(bior44, bior44 + 10);

    if (in_image.GetBounds() != out_image->GetBounds()) {
      throw std::runtime_error("StationaryWaveletTransformDiagonalOnly - in_image and out_image bounds must be equal");
    }

    afx::Bounds region = in_image.GetBounds();

    unsigned int max_level = log2(static_cast<double>(std::min(region.GetWidth(), region.GetHeight())) / static_cast<double>(wavelet.size() - 1));
    level = std::min(level, max_level);
    float sum = 0.0f;
    for(std::vector<float>::iterator it = wavelet.begin(); it != wavelet.end(); ++it) {
      sum += fabsf(*it);
    }
    for(std::vector<float>::iterator it = wavelet.begin(); it != wavelet.end(); ++it) {
      *it = *it / sum;
    }

    afx::Image temp_image_1(region);
    afx::Image temp_image_2;
    if (level > 0) { // Don't need second temp image if level is 0
      temp_image_2.Allocate(region);
    }
    const afx::Image* in_ptr = &in_image;
    afx::Image* out_ptr = &temp_image_2;

    ImageThreader threader;
    for (unsigned int i = 0; i <= level; ++i) {
      if (i == level) {
        out_ptr = out_image;
      }
      threader.ThreadImageChunks(region, boost::bind(&WaveletTransform::StationaryConvolve_X_, this, _1, boost::cref(*in_ptr), &temp_image_1, boost::cref(wavelet), level));
      threader.Wait();
      out_ptr->MemCpyIn(temp_image_1.GetPtr(), temp_image_1.GetPitch());
      threader.ThreadImageChunksY(region, boost::bind(&WaveletTransform::StationaryConvolve_Y_, this, _1, boost::cref(temp_image_1), out_ptr, boost::cref(wavelet), level));
      threader.Wait();
      in_ptr = out_ptr;
    }
  }
private:
  void StationaryConvolve_X_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, const std::vector<float>& wavelet, unsigned int level) {
    int increment = std::pow(2.0f, static_cast<int>(level));
    int size = increment * static_cast<int>(wavelet.size() - 1);
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        float sum = 0.0f;
        int window_x = x - size;
        const float* in_ptr = in_image.GetPtrBnds(window_x, y);
        for (std::vector<float>::const_reverse_iterator it = wavelet.rbegin(); it < wavelet.rend(); ++it) {
          sum += *it * *in_image.GetPtrBnds(window_x, y);
          window_x += increment;
          // TODO(rpw): This is not working right.
//           if (window_x  >= in_image.GetBounds().x1()) {
//             in_ptr += increment;
//           }
        }
        for (std::vector<float>::const_iterator it = wavelet.begin() + 1; it < wavelet.end(); ++it) {
          sum += *it * *in_image.GetPtrBnds(window_x, y);
          window_x += increment;
          // TODO(rpw): This is not working right.
//           if (window_x >= in_image.GetBounds().x1() && window_x <= in_image.GetBounds().x2()) {
//             in_ptr += increment;
//           }
        }
        *out_ptr++ = sum;
      }
    }
  }
  void StationaryConvolve_Y_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, const std::vector<float>& wavelet, unsigned int level) {
    int increment = std::pow(2.0f, static_cast<int>(level));
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float* out_ptr = out_image->GetPtr(x, region.y1());
      for (int y = region.y1(); y <= region.y2(); ++y) {
        float sum = 0.0f;
        int window_y = y - increment * static_cast<int>(wavelet.size() - 1);
        const float* in_ptr = in_image.GetPtrBnds(x, window_y);
        for (std::vector<float>::const_reverse_iterator it = wavelet.rbegin(); it < wavelet.rend(); ++it) {
          sum += *it * *in_image.GetPtrBnds(x, window_y);
          window_y += increment;
          if (window_y >= in_image.GetBounds().y1()) {
//             for (int i = 0; i < increment; ++i) {
//               in_ptr = in_image.GetNextRow(in_ptr);
//             }
          }
        }
        for (std::vector<float>::const_iterator it = wavelet.begin() + 1; it < wavelet.end(); ++it) {
          sum += *it * *in_image.GetPtrBnds(x, window_y);
          window_y += increment;
          if (window_y >= in_image.GetBounds().y1() && window_y <= in_image.GetBounds().y2()) {
//             for (int i = 0; i < increment; ++i) {
//               in_ptr = in_image.GetNextRow(in_ptr);
//             }
          }
        }
        *out_ptr = sum;
        out_ptr = out_image->GetNextRow(out_ptr);
      }
    }
  }
};

class NoiseMap {
public:
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

public:



};


}  // namespace afx

#endif  // INCLUDE_NOISE_MAP_H_

