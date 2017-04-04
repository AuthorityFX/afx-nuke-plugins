// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef AFX_GLOW_H_
#define AFX_GLOW_H_

#include <math.h>

#include <ipp.h>
#include <ippcore.h>
#include <ipptypes.h>
#include <ippi.h>

#include <boost/atomic.hpp>
#include <boost/scoped_ptr.hpp>
#include <cuda.h>
#include <cufft.h>

#include <vector>
#include <algorithm>

#include "include/cuda_helper.h"
#include "include/bounds.h"
#include "include/image.h"

namespace afx {

class FindFFTSize {
public:
  FindFFTSize() { CalculateSizes_(); }
  unsigned int GetNextSize(unsigned int size, float pow2_threshold = 1.2) {
    unsigned int next_size = NextPowerOfTwo_(size);
    if (static_cast<float>(next_size) / static_cast<float>(size) > pow2_threshold) {
      auto it = std::lower_bound(sizes_.begin() + 1, sizes_.end(), size);
      if (it != sizes_.end()) {
        next_size = *it;
      }
    }
    return next_size;
  }

private:
  std::vector<unsigned int> sizes_;

  // https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  unsigned int NextPowerOfTwo_(unsigned int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
  }

  int IntPow_(int x, int p) {
    if (p == 0) { return 1; }
    if (p == 1) { return x; }
    int tmp = IntPow_(x, p / 2);
    if (p % 2 == 0) {
      return tmp * tmp;
    } else {
      return x * tmp * tmp;
    }
  }

  void CalculateSizes_() {
    const unsigned int max_size = IntPow_(2, 16);  // Max size that will be stored
    for (int i2 = 0; i2 <= 16; ++i2) {  // Loop powers of 2 from 2^0 to 2^16
      unsigned int pow2 = IntPow_(2, i2);
      for (int i3 = 0; i3 <= 10; ++i3) {  // Loop powers of 3 from 3^0 to 3^10 - 59,049
        unsigned int pow3 = IntPow_(3, i3);
        for (int i5 = 0; i5 <= 7; ++i5) {  // Loop powers of 5 from 5^0 to 5^7 - 78,125
          unsigned int pow5 = IntPow_(5, i5);
          for (int i7 = 0; i7 <= 6; ++i7) {  // Loop powers of 7 from 7^0 to 7^6 - 117,649
            unsigned int pow7 = IntPow_(7, i7);
            unsigned int multiple = pow2 * pow3 * pow5 * pow7;  // Multiple power of 2, 3, 5. and 7
            if (multiple > max_size) { break; }  // if multiple is greater than max size break loop
            sizes_.push_back(multiple);
          }
        }
      }
    }
    std::sort(sizes_.begin(), sizes_.end());
  }
};


class GlowBase {
 public:
   GlowBase() : gauss_ptr_ptr_(nullptr), sigma_ptr_(nullptr), gauss_size_ptr_(nullptr) {
    gauss_ptr_ptr_ = reinterpret_cast<float**>(ippMalloc(iterations_ * sizeof(float*)));
    for (int i = 0; i < iterations_; ++i) { gauss_ptr_ptr_[i] = nullptr; }
    gauss_size_ptr_ = reinterpret_cast<int*>(ippMalloc(iterations_ * sizeof(int)));
    sigma_ptr_ = reinterpret_cast<float*>(ippMalloc(iterations_ * sizeof(float)));
  }
  ~GlowBase() { DisposeGlowBase(); }
  void ComputeGaussSize(float size, float softness, float falloff, float quality) {
    const float inner_size = (1.0f + 5.0f * softness) / 6.0f;
    int last_size = 3;
    for (int i = 0; i < iterations_; ++i) {
      float lookup = static_cast<float>(i) / static_cast<float>(iterations_ - 1);
      float mapped_quality = cos(static_cast<float>(M_PI) * 0.5f * lookup) * (1.0f - quality) + quality;
      sigma_ptr_[i] = powf(lookup, 1.0f / falloff) * (fmaxf(size, inner_size) - inner_size) + inner_size;
      gauss_size_ptr_[i] = std::max((static_cast<int>(ceilf(sigma_ptr_[i] * mapped_quality * (5.0f) + 1.0f))) | 1, 3) / 2 + 1;
      // Make current glow larger than last glow size
      if (last_size > gauss_size_ptr_[i]) { gauss_size_ptr_[i] = last_size; }
      last_size = gauss_size_ptr_[i];
    }
    max_gauss_size_ = last_size;  // Set max gauss size
  }
  void DisposeGlowBase() {
    DisposeGaussians_();
    if (gauss_ptr_ptr_ != nullptr) {
      ippFree(gauss_ptr_ptr_);
      gauss_ptr_ptr_ = nullptr;
    }
    if (gauss_size_ptr_ != nullptr) {
      ippFree(gauss_size_ptr_);
      gauss_size_ptr_ = nullptr;
    }
    if (sigma_ptr_ != nullptr) {
      ippFree(sigma_ptr_);
      sigma_ptr_ = nullptr;
    }
  }
  int GetKernelPadding() const { return max_gauss_size_ - 1; }

 protected:
  static const int iterations_ = 200;
  float** gauss_ptr_ptr_;
  float* sigma_ptr_;  // Glow size
  int* gauss_size_ptr_;  // Size of half gauss + 1

  int max_gauss_size_;  // Max size of guass_size_

  void AllocateGaussians_() {
    DisposeGaussians_();
    for (int i = 0; i < iterations_; ++i) {
      gauss_ptr_ptr_[i] = reinterpret_cast<float*>(ippMalloc(gauss_size_ptr_[i] * sizeof(float)));
    }
  }
  void DisposeGaussians_() {
    if (gauss_ptr_ptr_ != nullptr) {
      for (int i = 0; i < iterations_; ++i) {
        if (gauss_ptr_ptr_[i] != nullptr) {
          ippFree(gauss_ptr_ptr_[i]);
          gauss_ptr_ptr_[i] = nullptr;
        }
      }
    }
  }
  void CreateGauss_(const Bounds& region) {
    float sqrt_of_two_pi = sqrtf(2.0f * static_cast<float>(M_PI));
    for (int size = region.y2(); size >= region.y1(); --size) {
      for (int i = region.x2(); i >= region.x1(); --i) {
        if (size > gauss_size_ptr_[i] - 1) { break; }
        float sigma = sigma_ptr_[i];
        float one_over_two_sigma_squared = 1.0f / (2.0f * sigma * sigma);
        float a = 1.0f / (sigma * sqrt_of_two_pi);
        gauss_ptr_ptr_[i][size] = a * expf(-powf(static_cast<float>(gauss_size_ptr_[i] - 1 - size), 2.0f) * one_over_two_sigma_squared);
      }
    }
  }

};


class Glow : public GlowBase {
 public:
  Glow() {}
  ~Glow() {}

  void InitKernel(float exposure, afx::ImageThreader* threader) {
    AllocateGaussians_();
    threader->ThreadRowChunks(Bounds(0, 0, iterations_ - 1, max_gauss_size_ - 1), boost::bind(&Glow::CreateGauss_, this, _1));
    kernel_.Allocate(max_gauss_size_ * 2 - 1, max_gauss_size_ * 2 - 1);
    threader->Wait();

    boost::atomic<double> kernel_sum(0.0);
    threader->ThreadRowChunks(Bounds(0, 0, max_gauss_size_ - 1, max_gauss_size_ - 1), boost::bind(&Glow::CreateKernel_, this, _1, &kernel_sum));
    threader->Wait();

    threader->ThreadRowChunks(kernel_.GetBounds(), boost::bind(&Glow::NormalizeKernel_, this, _1, 2.0f * powf(2.0f, exposure) / kernel_sum));
    DisposeGaussians_();
  }

  void Convolve(const Image& in_padded, Image* out) {
    int buffer_size;
    ippiConvGetBufferSize(in_padded.GetSize(), in_padded.GetSize(), ipp32f, 1, ippiROIValid, &buffer_size);
    boost::scoped_ptr<Ipp8u> buffer(new Ipp8u[buffer_size]);
    ippiConv_32f_C1R(in_padded.GetPtr(), in_padded.GetPitch(), in_padded.GetSize(),
                     kernel_.GetPtr(), kernel_.GetPitch(), kernel_.GetSize(),
                     out->GetPtr(), out->GetPitch(), ippiROIValid, buffer.get());

  }

private:
  Image kernel_;

  void CreateKernel_(const Bounds& region, boost::atomic<double>* kernel_sum) {
    double kernel_sum_l = 0.0;
    for (int y = region.y1(); y <= region.y2(); ++y) {
      int y0 = y - kernel_.GetBounds().y1();

      float* low_left_ptr = kernel_.GetPtr(region.x1(), y);
      float* low_right_ptr = kernel_.GetPtr(kernel_.GetBounds().x2() - region.x1(), y);
      float* up_left_ptr = kernel_.GetPtr(region.x1(), kernel_.GetBounds().y2() - y);
      float* up_right_ptr = kernel_.GetPtr(kernel_.GetBounds().x2() - region.x1(), kernel_.GetBounds().y2() - y);

      for (int x = region.x1(); x <= region.x2(); ++x) {
        int x0 = x - kernel_.GetBounds().x1();
        float sum = 0.0f;
        for (int i = iterations_ - 1; i >= 0; --i) {
          // Offset of largest gauss and current gauss
          int gauss_offset = max_gauss_size_ - gauss_size_ptr_[i];
          if (x0 >= gauss_offset && y0 >= gauss_offset) {
            sum += gauss_ptr_ptr_[i][x0 - gauss_offset] * gauss_ptr_ptr_[i][y0 - gauss_offset];
          } else {
            break;
          }
        }
        // Write to all quadrants of the kernel image
        *low_left_ptr = sum;
        kernel_sum_l += sum;
        if (x0 < max_gauss_size_ - 1) {
          *low_right_ptr = sum;
          kernel_sum_l += sum;
        }
        if (y0 < max_gauss_size_ - 1) {
          *up_left_ptr = sum;
          kernel_sum_l += sum;
        }
        if (x0 < max_gauss_size_ - 1 && y0 < max_gauss_size_ - 1) {
          *up_right_ptr = sum;
          kernel_sum_l += sum;
        }
        low_left_ptr++;
        low_right_ptr--;
        up_left_ptr++;
        up_right_ptr--;
      }
    }
    *kernel_sum = *kernel_sum + kernel_sum_l;
  }
  void NormalizeKernel_(const Bounds& region, float n_factor) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      boost::this_thread::interruption_point();
      float* ptr = kernel_.GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        *ptr = *ptr * n_factor;  // normalize
        ptr++;
      }
    }
  }
};

}  // namespace afx

#endif  // AFX_GLOW_H_
