// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef IMAGE_H_
#define IMAGE_H_

#include <algorithm>
#include <list>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/functional/hash.hpp>
#include <boost/any.hpp>

#include <ipp.h>
#include <ippcore.h>
#include <ipptypes.h>
#include <ippi.h>

#include "types.h"
#include "threading.h"
#include "attribute.h"

#include "settings.h"

namespace afx {

// Cuda error handling functions
inline void IppSafeCall(IppStatus status) {
  if (status != ippStsNoErr) { throw status; }
}

class ImageComplex {
 private:
  Ipp32fc* ptr_;
  int pitch_;
  IppiSize image_size_;
  Bounds region_;
 public:
  ImageComplex() : ptr_(nullptr), pitch_(0), region_(Bounds()) {}
  ImageComplex(const Bounds& region) : ptr_(nullptr) { Create(region); }
  ~ImageComplex() { Dispose(); }
  void Create(const Bounds& region) {
    Dispose();
    region_ = region;
    image_size_.width = region.GetWidth();
    image_size_.height = region.GetHeight();
    ptr_ = ippiMalloc_32fc_C1(region.GetWidth(), region.GetHeight(), &pitch_);
  }
  void Dispose() {
    if (ptr_ != nullptr) {
      ippiFree(ptr_);
      ptr_ = nullptr;
    }
  }
  Ipp32fc* GetPtr() const { return ptr_; }
  Ipp32fc* GetPtr(int x, int y) const {
    return (Ipp32fc*)((char*)(ptr_) + (y - region_.y1()) * pitch_ + (x - region_.x1()) * sizeof(Ipp32fc));
  }
  size_t GetPitch() const { return pitch_; }
  IppiSize GetSize() const { return image_size_; }
  Bounds GetBounds() const { return region_; }


};

class Image : public AttributeBase {
 private:
  Ipp32f* ptr_;
  int pitch_;
  IppiSize image_size_;
  Bounds region_;
 public:
  Image();
  Image(unsigned int width, unsigned int height);
  Image(const Bounds& region);
  Image(const Image& other);
  Image& operator=(const Image& other);
  ~Image();
  void Create(unsigned int width, unsigned int height);
  void Create(const Bounds& region);
  void Dispose();
  void MemSet(float value);
  void MemSet(float value, const Bounds& region);
  void MemCpyIn(const float* src_ptr, size_t src_pitch, const Bounds& region);
  void MemCpyOut(float* dst_ptr, size_t dst_pitch, const Bounds& region);
  Ipp32f* GetPtr() const;
  Ipp32f* GetPtr(int x, int y) const;
  Ipp32f GetVal(int x, int y) const;
  Ipp32f GetValBnds(int x, int y) const;
  size_t GetPitch() const;
  IppiSize GetSize() const;
  Bounds GetBounds() const;
};

class ImageArray : public Array<Image> {
 public:
  void AddImage(const Bounds& region);
};


class Resize {
 private:
  IppiResizeSpec_32f* spec_ptr_;
  int spec_size_, init_size_, buf_size_ ;
  Ipp8u* buffer_ptr_;
  Ipp32f* init_buf_ptr_;
  IppiPoint dest_offset_;
  float* border_value_;

  void ResizeTile_(Bounds region, Image* in, Image* out) {
    IppiBorderType border_style = ippBorderRepl;

    if (in->GetBounds().x1() > region.x1()) { border_style = (IppiBorderType)((int)border_style | (int)ippBorderInMemRight); }
    if (in->GetBounds().x2() < region.x2()) { border_style = (IppiBorderType)((int)border_style | (int)ippBorderInMemLeft); }
    if (in->GetBounds().y1() > region.y1()) { border_style = (IppiBorderType)((int)border_style | (int)ippBorderInMemBottom); }
    if (in->GetBounds().y2() < region.y2()) { border_style = (IppiBorderType)((int)border_style | (int)ippBorderInMemTop); }

    // select algorithm based on image size change. TODO dest tile size not full image size
    ippiResizeAntialiasing_32f_C1R(in->GetPtr(region.x1(), region.y1()), in->GetPitch(), out->GetPtr(region.x1(), region.y1()), out->GetPitch(),
                                   dest_offset_, out->GetSize(), border_style, border_value_, spec_ptr_, buffer_ptr_);

  };

  void Dispose() {
    if (spec_ptr_ != nullptr) {
      ippsFree(spec_ptr_);
      spec_ptr_ = nullptr;
    }
    if (buffer_ptr_ != nullptr) {
      ippsFree(buffer_ptr_);
      buffer_ptr_ = nullptr;
    }
  }

 public:
  Resize(Image* in, Image* out) : init_buf_ptr_(nullptr), spec_ptr_(nullptr), buffer_ptr_(nullptr) {
    int x_offset = (in->GetBounds().GetWidth() - out->GetBounds().GetWidth()) >> 1;
    int y_offset = (in->GetBounds().GetHeight() - out->GetBounds().GetHeight() ) >> 1;

    dest_offset_ = (IppiPoint){x_offset, y_offset};

    // Spec and ippBBuf sizes TODO tile sizes not full image size
    ippiResizeGetSize_32f(in->GetSize(), out->GetSize(), ippLanczos, 1, &spec_size_, &init_size_);
    if (init_buf_ptr_ == nullptr) {
      init_buf_ptr_ = (Ipp32f*)ippsMalloc_32f(init_size_);
    }
    if (spec_ptr_ == nullptr) {
      spec_ptr_ = (IppiResizeSpec_32f*)ippsMalloc_32f(spec_size_);
    }
    // Filter initialization TODO tile sizes not full image size
    ippiResizeLanczosInit_32f(in->GetSize(), out->GetSize(), 2, spec_ptr_, buffer_ptr_);
    ippiResizeGetBufferSize_32f(spec_ptr_, out->GetSize(), 1, &buf_size_);
    if (buffer_ptr_ == nullptr) {
      buffer_ptr_ = (Ipp8u*)ippsMalloc_8u(buf_size_);
    }

    *border_value_ = 0.0f;

//     Threader threader(in->GetBounds());
//     threader.RunLines(boost::bind(&Resize::ResizeTile_, this, _1, in, out));
  }
};

}  //  namespace afx

#endif  // IMAGE_H_
