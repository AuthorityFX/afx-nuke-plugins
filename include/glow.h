// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef GLOW_H_
#define GLOW_H_

#include <ipp.h>
#include <ippcore.h>
#include <ipptypes.h>
#include <ippi.h>

#include <boost/atomic.hpp>
#include <cuda.h>
#include <cufft.h>

#include <vector>
#include <math.h>

#include "cuda_helper.h"
#include "types.h"
#include "image.h"

#include "settings.h"

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
  std::vector<unsigned int> size_;

  //https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
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
    const unsigned int pow2_thresh = IntPow_(2, 10);
    const unsigned int max_s = IntPow_(2, 16);
    for (int i2 = 1; i2 <= 16; ++i2) {
      unsigned int pow2 = IntPow_(2, i2);
      if (pow2 < pow2_thresh) { size_.push_back(pow2); }
      for (int i3 = 0; i3 <= 10; ++i3) {
        unsigned int pow3 = IntPow_(3, i3);
        for (int i5 = 0; i5 <= 7; ++i5) {
          unsigned int pow5 = IntPow_(5, i5);
          for (int i7 = 0; i7 <= 6; ++i7) {
            unsigned int pow7 = IntPow_(7, i7);
            unsigned int s = pow2 * pow3 * pow5 * pow7;
            if (s > max_s) { break; }
            if (s > pow2_thresh) { size_.push_back(s); }
          }
        }
      }
    }
    std::sort(size_.begin(), size_.end());
  }

public:

  FindFFTSize() { CalculateSizes_(); }

  unsigned int GetNextSize(unsigned int size, float pow2_threshold = 1.2) {
    unsigned int next_size = NextPowerOfTwo_(size);
    if ((float)next_size / (float)size > pow2_threshold ) {
      std::vector<unsigned int>::iterator it = std::lower_bound(size_.begin(), size_.end(), size);
      if (it != size_.end()) {
        next_size = *it;
      }
    }
    return next_size;
  }
};


class GlowBase {
 protected:
  static const int iterations_ = 200;
  float** gauss_ptr_ptr_;
  float* sigma_ptr_; // Glow size
  int* gauss_size_ptr_; // Size of half gauss + 1

  int max_gauss_size_; // Max size of guass_size_

  void AllocateGaussians_() {
    DisposeGaussians_();
    for (int i = 0; i < iterations_; ++i) {
      gauss_ptr_ptr_[i] = (float*)ippMalloc(gauss_size_ptr_[i] * sizeof(float));
    }
  }
  void DisposeGaussians_() {
    if (gauss_ptr_ptr_ != nullptr) {
      for(int i = 0; i < iterations_; ++i) {
        if (gauss_ptr_ptr_[i] != nullptr) {
          ippFree(gauss_ptr_ptr_[i]);
          gauss_ptr_ptr_[i] = nullptr;
        }
      }
    }
  }
  void CreateGauss_(const Bounds& region) {
    for (int size = region.y2(); size >= region.y1(); --size) {
      boost::this_thread::interruption_point();
      for (int i = region.x2(); i >= region.x1(); --i) {
        if (size > gauss_size_ptr_[i] - 1) { break; }
        float a = 1.0f / (sigma_ptr_[i] * sqrtf(2.0f * (float)M_PI));
        gauss_ptr_ptr_[i][size] = a * expf(-powf((float)(gauss_size_ptr_[i] - 1 - size), 2.0f) / (2.0f * sigma_ptr_[i] * sigma_ptr_[i]));
      }
    }
  }
 public:
  GlowBase() : gauss_ptr_ptr_(nullptr), gauss_size_ptr_(nullptr), sigma_ptr_(nullptr) {
    gauss_ptr_ptr_ = (float**)ippMalloc(iterations_ * sizeof(float*));
    for(int i = 0; i < iterations_; ++i) { gauss_ptr_ptr_[i] = nullptr; }
    gauss_size_ptr_ = (int*)ippMalloc(iterations_ * sizeof(int));
    sigma_ptr_ = (float*)ippMalloc(iterations_ * sizeof(float));
  }
  ~GlowBase() { DisposeGlowBase(); }
  void ComputeGaussSize(float size, float softness, float falloff, float quality) {
    const float inner_size = (1.0f + 5.0f * softness) / 6.0f;
    int last_size = 0;
    for (int i = 0; i < iterations_; ++i) {
      float lookup = (float)i / (float)(iterations_ - 1);
      //float mapped_quality = cos((float)M_PI * 0.5f * powf(lookup, 10.0 * quality)) * (1.0f - (1.0f/ 6.0f)) + (1.0f/ 6.0f);
      float mapped_quality = cos((float)M_PI * 0.5f * lookup) * (1.0f - quality) + quality;
      sigma_ptr_[i] = powf(lookup, 1.0f / falloff) * (fmaxf(size, inner_size) - inner_size) + inner_size;
      gauss_size_ptr_[i] = std::max(((int)ceilf(sigma_ptr_[i] * mapped_quality * (6.0f - 1.0f) + 1.0f)) | 1, 3);
      gauss_size_ptr_[i] = (int)ceil((float)gauss_size_ptr_[i] / 2.0f);
      // Make current glow larger than last glow size
      if (last_size > gauss_size_ptr_[i]) { gauss_size_ptr_[i] = last_size; }
      last_size = gauss_size_ptr_[i];
    }
    max_gauss_size_ = last_size; // Set max gauss size
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
};


class GlowGPU {
private:

public:

};


class Glow : public GlowBase {
 private:
  Image kernel_;

  void CreateKernel_(const Bounds& region, boost::atomic<double>& kernel_sum) {
    double kernel_sum_l = 0.0;
    for (int y = region.y1(); y <= region.y2(); ++y) {
      boost::this_thread::interruption_point();
      int y0 = y - kernel_.GetBounds().y1();
      float* ptr = kernel_.GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        int x0 = x - kernel_.GetBounds().x1();
        float sum = 0.0f;
        for (int i = iterations_ - 1; i >= 0; --i) {
          //Offset of largest gauss and current gauss
          int gauss_offset = max_gauss_size_ - gauss_size_ptr_[i];
          if (x0 >= gauss_offset && y0 >= gauss_offset) {
            sum += gauss_ptr_ptr_[i][x0 - gauss_offset] * gauss_ptr_ptr_[i][y0 - gauss_offset];
          } else {
            break;
          }
        }
        //Write to all quadrants of the kernel image
        *ptr++ = sum; // Lower left quadrant
        kernel_sum_l += sum;
        if (x0 < max_gauss_size_ - 1) {
          *kernel_.GetPtr(kernel_.GetBounds().x2() - x0, y) = sum;
          kernel_sum_l += sum;
        }
        if (y0 < max_gauss_size_ - 1) {
          *kernel_.GetPtr(x, kernel_.GetBounds().y2() - y0) = sum;
          kernel_sum_l += sum;
        }
        if (x0 < max_gauss_size_ - 1 && y0 < max_gauss_size_ - 1) {
          *kernel_.GetPtr(kernel_.GetBounds().x2() - x0, kernel_.GetBounds().y2() - y0) = sum;
          kernel_sum_l += sum;
        }
      }
    }
    kernel_sum = kernel_sum + kernel_sum_l;
  }
  void NormalizeKernel_(const Bounds& region, float n_factor) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      boost::this_thread::interruption_point();
      float* ptr = kernel_.GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        *ptr = *ptr * n_factor; // normalize
        ptr++;
      }
    }
  }

 public:
  Glow() {}
  ~Glow() {}

  void InitKernel(float exposure, Threader& threader) {
    AllocateGaussians_();
    threader.ThreadImageChunks(Bounds(0, 0, iterations_ - 1, max_gauss_size_ - 1), boost::bind(&Glow::CreateGauss_, this, _1));
    kernel_.Create(max_gauss_size_ * 2 - 1, max_gauss_size_ * 2 - 1);
    threader.Synchonize();

    boost::atomic<double> kernel_sum(0.0);
    threader.ThreadImageChunks(Bounds(0, 0, max_gauss_size_ - 1, max_gauss_size_ - 1), boost::bind(&Glow::CreateKernel_, this, _1, boost::ref(kernel_sum)));
    threader.Synchonize();

    threader.ThreadImageChunks(kernel_.GetBounds(), boost::bind(&Glow::NormalizeKernel_, this, _1, 2.0f * powf(2.0f, exposure) / kernel_sum));
    DisposeGaussians_();
  }

  void Convolve(const Image& in_padded, Image* out) {
    int buffer_size;
    Ipp8u* buffer_ptr = nullptr;
    ippiConvGetBufferSize(in_padded.GetSize(), in_padded.GetSize(), ipp32f, 1, ippiROIValid, &buffer_size);
    buffer_ptr = (Ipp8u*)ippMalloc(buffer_size);
    ippiConv_32f_C1R(in_padded.GetPtr(), in_padded.GetPitch(), in_padded.GetSize(),
                     kernel_.GetPtr(), kernel_.GetPitch(), kernel_.GetSize(), out->GetPtr(), out->GetPitch(), ippiROIValid, buffer_ptr);
    if (buffer_ptr != nullptr) {
      ippFree(buffer_ptr);
      buffer_ptr = nullptr;
    }
  }
  int GetKernelPadding() const { return max_gauss_size_ - 1; }
};

} // namespace afx

#endif  // GLOW_H_
