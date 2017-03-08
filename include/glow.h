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

#include <ipp.h>
#include <ippcore.h>
#include <ipptypes.h>
#include <ippi.h>

#include <boost/atomic.hpp>
#include <cuda.h>
#include <cufft.h>

#include <vector>
#include <algorithm>

#include "include/settings.h"
#include "include/cuda_helper.h"
#include "include/bounds.h"
#include "include/image.h"

namespace afx {

class CudaFFT {
 private:
  cufftHandle plan_;

 public:
  CudaFFT() {}
  ~CudaFFT() { DisposePlan(); }
  void CreatPlanR2C(const CudaImage& src, const CudaComplex& dst) {
    DisposePlan();
    int rank = 2;
    int n[] = {src.GetWidth(), src.GetHeight()};
    int istride = 1, ostride = 1;
    int idist = 1, odist = 1;
    int inembed[] = {src.GetPitch() / sizeof(float), src.GetHeight()};
    int onembed[] = {dst.GetPitch() / sizeof(cufftComplex), dst.GetHeight()};
    int batch = 1;
    cufftPlanMany(&plan_, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
  }
  void CreatPlanC2R(const CudaComplex& src, const CudaImage& dst) {
    DisposePlan();
    int rank = 2;
    int n[] = {src.GetWidth(), src.GetHeight()};
    int istride = 1, ostride = 1;
    int idist = 1, odist = 1;
    int inembed[] = {src.GetPitch() / sizeof(cufftComplex), src.GetHeight()};
    int onembed[] = {dst.GetPitch() / sizeof(float), dst.GetHeight()};
    int batch = 1;
    cufftPlanMany(&plan_, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2R, batch);
  }
  void Forward(const CudaImage& src, CudaComplex* dst) {
    cufftExecR2C(plan_, src.GetPtr(), dst->GetPtr());
  }
  void Inverse(const CudaComplex& src, CudaImage* dst) {
    cufftExecC2R(plan_, src.GetPtr(), dst->GetPtr());
  }
  void DisposePlan() {
    cufftDestroy(plan_);
  }
};


class FindFFTSize {
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
    sizes_.erase(sizes_.begin());  // Erase pow 2^0
  }

 public:
  FindFFTSize() { CalculateSizes_(); }

  unsigned int GetNextSize(unsigned int size, float pow2_threshold = 1.2) {
    unsigned int next_size = NextPowerOfTwo_(size);
    if (static_cast<float>(next_size) / static_cast<float>(size) > pow2_threshold) {
      std::vector<unsigned int>::iterator it = std::lower_bound(sizes_.begin(), sizes_.end(), size);
      if (it != sizes_.end()) {
        next_size = *it;
      }
    }
    return next_size;
  }
};


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
      unsigned int size  = std::max((static_cast<int>(ceilf(sigma * mapped_quality * (6.0f - 1.0f) + 1.0f)) ) | 1, 3) / 2;
      if (last_size > size) { size = last_size; }
      last_size = size;
      gaussians.push_back(Gaussian(sigma, size));
    }

    kernel_.clear();
    kernel_.resize(gaussians.back().size);
    float sqrt_of_two_pi = sqrtf(2.0f * static_cast<float>(M_PI));
    for(std::vector<Gaussian>::iterator it = gaussians.begin(); it < gaussians.end(); ++it) {
      unsigned int size = (*it).size;
      float sigma = (*it).sigma;
      float two_sigma_squared = 2.0f * sigma * sigma;
      float a = 1.0f / (sigma * sqrt_of_two_pi);
      for (int x = 0; x < size; ++x) {
        kernel_[x] += a * expf(-static_cast<float>(x) * static_cast<float>(x) / two_sigma_squared);
      }
    }
    afx::NormalizeKernel(&kernel_);
  }
  std::vector<float> GetKernel() const {
    return kernel_;
  }
  unsigned int RequiredPadding() const {
    return kernel_.size() - 1;
  }
  unsigned int GetKernelSize() const {
    return kernel_.size();
  }

protected:
  std::vector<float> kernel_;

  struct Gaussian {
    float sigma;
    unsigned int size;
    Gaussian() : sigma(0), size(0) {}
    Gaussian(float sigma, unsigned int size) : sigma(sigma), size(size) {}
  };
};


class Glow : public GlowBase {
 private:
  Image kernel_image_;

  void CreateKernelTile_(const Bounds& region) {
    int kernel_size_minus_1 = GetKernelSize() - 1;
    for (int y = region.y1(); y <= region.y2(); ++y) {
      int kernel_y = kernel_size_minus_1 - y;
      float* low_left_ptr = kernel_image_.GetPtr(region.x1(), y);
      float* low_right_ptr = kernel_image_.GetPtr(kernel_image_.GetBounds().x2() - region.x1(), y);
      float* up_left_ptr = kernel_image_.GetPtr(region.x1(), kernel_image_.GetBounds().y2() - y);
      float* up_right_ptr = kernel_image_.GetPtr(kernel_image_.GetBounds().x2() - region.x1(), kernel_image_.GetBounds().y2() - y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        int kernel_x = kernel_size_minus_1 - x;
        float value = kernel_[kernel_x] * kernel_[kernel_y];
        // Write to all quadrants of the kernel image
        // Increment pointer for quads left of middle. Decrement for quads right of middle
        *low_left_ptr++ = value;
        *low_right_ptr-- = value;
        *up_left_ptr++ = value;
        *up_right_ptr-- = value;
      }
    }
  }

 public:
  Glow() : buffer_ptr_(nullptr) {}
  ~Glow() {
    if (buffer_ptr_ != nullptr) {
      ippFree(buffer_ptr_);
      buffer_ptr_ = nullptr;
    }
  }

  void CreateKerneImage(float exposure, afx::ImageThreader* threader) {
    afx::Bounds kernel_region(0, 0, 2 * GetKernelSize() - 1, 2 * GetKernelSize() - 1);
    afx::Bounds lower_left_quad_region(0, 0, GetKernelSize() - 1, GetKernelSize() - 1);
    kernel_image_.Allocate(kernel_region);
    threader->ThreadImageChunks(lower_left_quad_region, boost::bind(&Glow::CreateKernelTile_, this, _1));
    threader->Wait();
  }

  void Convolve(const Image& in_padded, Image* out) {
    int buffer_size;
    ippiConvGetBufferSize(in_padded.GetSize(), in_padded.GetSize(), ipp32f, 1, ippiROIValid, &buffer_size);
    if (buffer_ptr_ != nullptr) {
      ippFree(buffer_ptr_);
      buffer_ptr_ = nullptr;
    }
    buffer_ptr_ = static_cast<Ipp8u*>(ippMalloc(buffer_size));
    ippiConv_32f_C1R(in_padded.GetPtr(), in_padded.GetPitch(), in_padded.GetSize(),
                     kernel_image_.GetPtr(), kernel_image_.GetPitch(), kernel_image_.GetSize(), out->GetPtr(), out->GetPitch(), ippiROIValid, buffer_ptr_);
    if (buffer_ptr_ != nullptr) {
      ippFree(buffer_ptr_);
      buffer_ptr_ = nullptr;
    }
  }
  int GetKernelPadding() const { return max_gauss_size_ - 1; }

private:
  Ipp8u* buffer_ptr_;
};

}  // namespace afx

#endif  // INCLUDE_GLOW_H_
