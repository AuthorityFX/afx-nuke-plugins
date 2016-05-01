// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef GLOW_TEMP_H_
#define GLOW_TEMP_H_

#include <ipp.h>
#include <ippcore.h>
#include <ipptypes.h>
#include <ippi.h>

#include <vector>
#include <math.h>

#include <cuda_helper.h>
#include <cufft.h>

#include "types.h"
#include "image.h"

#include "settings.h"

namespace afx {

class FFT {
 private:
  int fft_spec_size_, mem_init_size_, buffer_size_;
  IppiFFTSpec_R_32f* fft_spec_ptr_;
  Ipp8u* mem_init_ptr_;
  Ipp8u* buffer_ptr_;
 public:
  FFT() : fft_spec_ptr_(nullptr), mem_init_ptr_(nullptr), buffer_ptr_(nullptr) {}
  FFT(const Bounds& region) : fft_spec_ptr_(nullptr), mem_init_ptr_(nullptr), buffer_ptr_(nullptr) { InitializeFFT(region); }
  ~FFT() { DisposeFFT(); }
  void DisposeFFT() {
    if (fft_spec_ptr_ != nullptr) {
      ippFree(fft_spec_ptr_);
      fft_spec_ptr_ = nullptr;
    }
    if (mem_init_ptr_ != nullptr) {
      ippFree(mem_init_ptr_);
      mem_init_ptr_ = nullptr;
    }
    if (buffer_ptr_ != nullptr) {
      ippFree(buffer_ptr_);
      buffer_ptr_ = nullptr;
    }
  }
  void InitializeFFT(const Bounds& region) {
    DisposeFFT();
    ippiFFTGetSize_R_32f(region.GetWidth(), region.GetHeight(), IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, &fft_spec_size_, &mem_init_size_, &buffer_size_);
    if (fft_spec_size_ > 0) { fft_spec_ptr_ = (IppiFFTSpec_R_32f*)ippMalloc(fft_spec_size_); }
    if (mem_init_size_ > 0) { mem_init_ptr_ = (Ipp8u*)ippMalloc(mem_init_size_); }
    if (buffer_size_ > 0) { buffer_ptr_ = (Ipp8u*)ippMalloc(buffer_size_); }
    ippiFFTInit_R_32f(region.GetWidth(), region.GetHeight(), IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, fft_spec_ptr_, mem_init_ptr_);
  }
  void ForwardFFT(Image* image, bool unique_buffer = false) {
    Ipp8u* u_buffer_ptr = nullptr;
    if (unique_buffer && buffer_size_ > 0) { u_buffer_ptr = (Ipp8u*)ippMalloc(buffer_size_); }
    ippiFFTFwd_RToPack_32f_C1IR(image->GetPtr(), image->GetPitch(), fft_spec_ptr_, unique_buffer ? u_buffer_ptr : buffer_ptr_);
    if (u_buffer_ptr != nullptr) { ippFree(u_buffer_ptr); }
  }
  void ForwardFFT(const Image& input, Image output, bool unique_buffer = false) {
    Ipp8u* u_buffer_ptr = nullptr;
    if (unique_buffer && buffer_size_ > 0) { u_buffer_ptr = (Ipp8u*)ippMalloc(buffer_size_); }
    ippiFFTFwd_RToPack_32f_C1R(input.GetPtr(), input.GetPitch(), output.GetPtr(), output.GetPitch(), fft_spec_ptr_, unique_buffer ? u_buffer_ptr : buffer_ptr_);
    if (u_buffer_ptr != nullptr) { ippFree(u_buffer_ptr); }
  }
  void InverseFFT(Image* image, bool unique_buffer = false) {
    Ipp8u* u_buffer_ptr = nullptr;
    if (unique_buffer && buffer_size_ > 0) { u_buffer_ptr = (Ipp8u*)ippMalloc(buffer_size_); }
    ippiFFTInv_PackToR_32f_C1IR(image->GetPtr(), image->GetPitch(), fft_spec_ptr_, unique_buffer ? u_buffer_ptr : buffer_ptr_);
    if (u_buffer_ptr != nullptr) { ippFree(u_buffer_ptr); }
  }
  void InverseFFT(const Image& input, Image output, bool unique_buffer = false) {
    Ipp8u* u_buffer_ptr = nullptr;
    if (unique_buffer && buffer_size_ > 0) { u_buffer_ptr = (Ipp8u*)ippMalloc(buffer_size_); }
    ippiFFTInv_PackToR_32f_C1R(input.GetPtr(), input.GetPitch(), output.GetPtr(), output.GetPitch(), fft_spec_ptr_, unique_buffer ? u_buffer_ptr : buffer_ptr_);
    if (u_buffer_ptr != nullptr) { ippFree(u_buffer_ptr); }
  }
};

class Glow : private FFT, private FindFFTSize {
 private:
  static const int iterations_ = 200;
  Image kernel_fft_;
  Bounds kernel_bnds_;

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
    for (int size = region.y2; size >= region.y1; --size) {
      for (int i = region.x2; i >= region.x1; --i) {
        if (size > gauss_size_ptr_[i] - 1) { break; }
        float a = 1.0f / (sigma_ptr_[i] * sqrtf(2.0f * (float)M_PI));
        gauss_ptr_ptr_[i][size] = a * expf(-powf((float)(gauss_size_ptr_[i] - 1 - size), 2.0f) / (2.0f * sigma_ptr_[i] * sigma_ptr_[i]));
      }
    }
  }

  void CreateKernel_(const Bounds& region, boost::atomic<double>& kernel_sum) {
    double kernel_sum_l = 0.0;
    for (int y = region.y1; y <= region.y2; ++y) {
      int y0 = y - kernel_bnds_.y1;
      float* ptr = kernel_fft_.GetPtr(region.x1, y);
      for (int x = region.x1; x <= region.x2; ++x) {
        int x0 = x - kernel_bnds_.x1;
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
          *kernel_fft_.GetPtr(kernel_bnds_.x2 - x0, y) = sum;
          kernel_sum_l += sum;
        }
        if (y0 < max_gauss_size_ - 1) {
          *kernel_fft_.GetPtr(x, kernel_bnds_.y2 - y0) = sum;
          kernel_sum_l += sum;
        }
        if (x0 < max_gauss_size_ - 1 && y0 < max_gauss_size_ - 1) {
          *kernel_fft_.GetPtr(kernel_bnds_.x2 - x0, kernel_bnds_.y2 - y0) = sum;
          kernel_sum_l += sum;
        }
      }
    }

    kernel_sum = kernel_sum + kernel_sum_l;
  }

  void NormalizeKernel_(const Bounds& region, float n_factor) {
    for (int y = region.y1; y <= region.y2; ++y) {
      float* ptr = kernel_.GetPtr(region.x1, y);
      for (int x = region.x1; x <= region.x2; ++x) {
        *ptr = *ptr * n_factor; // normalize
        ptr++;
      }
    }
  }

  void NormalizeKernel_(const Bounds& region, float n_factor) {
    for (int y = region.y1; y <= region.y2; ++y) {
      float* ptr = kernel_fft_.GetPtr(region.x1, y);
      for (int x = region.x1; x <= region.x2; ++x) {
        *ptr = *ptr * n_factor; // normalize
        ptr++;
      }
    }
  }

  void ComplexMultiply_(const Bounds& region) {}


 public:
  Glow() : gauss_ptr_ptr_(nullptr), gauss_size_ptr_(nullptr), sigma_ptr_(nullptr) {
    gauss_ptr_ptr_ = (float**)ippMalloc(iterations_ * sizeof(float*));
    for(int i = 0; i < iterations_; ++i) { gauss_ptr_ptr_[i] = nullptr; }
    gauss_size_ptr_ = (int*)ippMalloc(iterations_ * sizeof(int));
    sigma_ptr_ = (float*)ippMalloc(iterations_ * sizeof(float));
  }
  ~Glow() {
    DisposeFFT();
    DisposeGlow();
  }

  void InitKernel(const Bounds& input_bnds, float exposure = 0.0f) {

    AllocateGaussians_();

    unsigned long new_width = GetNextSize(input_bnds.GetWidth() + (2 * (max_gauss_size_ - 1)));
    unsigned long new_height = GetNextSize(input_bnds.GetHeight() + (2 * (max_gauss_size_ - 1)));
    Bounds fft_bnds = input_bnds;
    if (input_bnds.GetWidth() % 2 != 0) { fft_bnds.x2 += 1;}
    if (input_bnds.GetHeight() % 2 != 0) { fft_bnds.y2 += 1;}
    fft_bnds.PadBounds((new_width - input_bnds.GetWidth()) / 2, (new_height - input_bnds.GetHeight()) / 2);
    kernel_fft_.Create(fft_bnds);
    kernel_fft_.MemSet(0, kernel_fft_.GetBounds()); // Memset kernel as we won't be writting to the whole thing.

    Bounds kernel_create_bnds; // Calculate region within kernel_ to run CreateKernel.
    kernel_create_bnds.x1 = (int)fft_bnds.GetCenterX() - max_gauss_size_ - 1;
    kernel_create_bnds.y1 = (int)fft_bnds.GetCenterY() - max_gauss_size_ - 1;
    kernel_create_bnds.x2 = kernel_create_bnds.x1 + max_gauss_size_ - 1; // Only go form start to half gauss. Due to symmetry only need to compute quarter of image.
    kernel_create_bnds.y2 = kernel_create_bnds.y1 + max_gauss_size_ - 1; // Only go form start to half gauss. Due to symmetry only need to compute quarter of image.
    kernel_bnds_ = kernel_create_bnds;
    kernel_bnds_.x2 = kernel_bnds_.x1 + 2 * (max_gauss_size_ - 1);
    kernel_bnds_.y2 = kernel_bnds_.y1 + 2 * (max_gauss_size_ - 1);


    { // Create gaussians
      ThreaderOld threader(Bounds(0, 0, iterations_ - 1, max_gauss_size_ - 1));
      threader.RunLines(boost::bind(&Glow::CreateGauss_, this, _1));  //Creat gaussians
    }

    boost::atomic<double> kernel_sum(0.0);
    { // Create kernel
      ThreaderOld threader(Bounds(0, 0, max_gauss_size_ - 1, max_gauss_size_ - 1));
      threader.RunChunks(boost::bind(&Glow::CreateKernel_, this, _1, boost::ref(kernel_sum))); // Create kernel
    }

    { // Normalize kernel
      ThreaderOld threader(kernel_bnds_);
      threader.RunChunks(boost::bind(&Glow::NormalizeKernel_, this, _1, 2.0f * powf(2.0f, exposure) / kernel_sum));
    }


    InitializeFFT(kernel_fft_.GetBounds());
    ForwardFFT(&kernel_fft_);
  }

  void ComputeGaussSize(float outer_size, float falloff, float quality) {
    const float inner_size = 1.0f / 6.0f;
    int last_size = 0;
    for (int i = 0; i < iterations_; ++i) {
      float lookup = (float)i / (float)(iterations_ - 1);
      //float mapped_quality = cos((float)M_PI * 0.5f * powf(lookup, 10.0 * quality)) * (1.0f - (1.0f/ 6.0f)) + (1.0f/ 6.0f);
      float mapped_quality = cos((float)M_PI * 0.5f * lookup) * (1.0f - quality) + quality;
      sigma_ptr_[i] = powf(lookup, 1.0f / falloff) * (fmaxf(outer_size, inner_size) - inner_size) + inner_size;
      gauss_size_ptr_[i] = std::max(((int)ceilf(sigma_ptr_[i] * mapped_quality * (6.0f - 1.0f) + 1.0f)) | 1, 3);
      gauss_size_ptr_[i] = (int)ceil((float)gauss_size_ptr_[i] / 2.0f);
      // Make current glow larger than last glow size
      if (last_size > gauss_size_ptr_[i]) { gauss_size_ptr_[i] = last_size; }
      last_size = gauss_size_ptr_[i];
    }
    max_gauss_size_ = last_size; // Set max gauss size
  }

  void DisposeGlow() {
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

  void Convolve(Image* image, bool unique_buffer = false, bool replicate_borders = true, float border_value = 0.0f) {

    Image image_fft(kernel_fft_.GetBounds());

    image_fft.MemSet(0.5, image_fft.GetBounds());
//     if (replicate_borders) {
//       ippiCopyReplicateBorder_32f_C1R(image->GetPtr(), image->GetPitch(), image->GetSize(), image_fft.GetPtr(), image_fft.GetPitch(), image_fft.GetSize(),
//                                       image_fft.GetBounds().GetWidth() - image->GetBounds().GetWidth(),
//                                       image_fft.GetBounds().GetHeight() - image->GetBounds().GetHeight());
//     } else {
//       ippiCopyConstBorder_32f_C1R(image->GetPtr(), image->GetPitch(), image->GetSize(), image_fft.GetPtr(), image_fft.GetPitch(), image_fft.GetSize(),
//                                   image_fft.GetBounds().GetWidth() - image->GetBounds().GetWidth(),
//                                   image_fft.GetBounds().GetHeight() - image->GetBounds().GetHeight(), border_value);
//     }

    //ippiCopy_32f_C1R(image->GetPtr(), image->GetPitch(), image_fft.GetPtr(image->GetBounds().x1, image->GetBounds().y1), image_fft.GetPitch(), image->GetSize());

    for (int y = image->GetBounds().y1; y <= image->GetBounds().y2; ++y) {
      float* in = image_fft.GetPtr(image->GetBounds().x1, y);
      float* out = image->GetPtr(image->GetBounds().x1, y);
      for (int x = image->GetBounds().x1; x <= image->GetBounds().x2; ++x) {
        *in++ = *out++;
      }
    }

    ForwardFFT(&image_fft, true);
    ippiMul_32f_C1IR(kernel_fft_.GetPtr(), kernel_fft_.GetPitch(), image_fft.GetPtr(), image_fft.GetPitch(), image_fft.GetSize());
    InverseFFT(&image_fft, true);

    for (int y = image->GetBounds().y1; y <= image->GetBounds().y2; ++y) {
      float* in = image_fft.GetPtr(image->GetBounds().x1, y);
      float* out = image->GetPtr(image->GetBounds().x1, y);
      for (int x = image->GetBounds().x1; x <= image->GetBounds().x2; ++x) {
        *out++ = *in++;
      }
    }
  }

  int GetKernelPadding() const { return max_gauss_size_ - 1; }

};

} // namespace afx


#endif  // GLOW_TEMP_H_
