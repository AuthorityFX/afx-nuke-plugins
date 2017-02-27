// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef INCLUDE_IMAGE_H_
#define INCLUDE_IMAGE_H_

#include <ipp.h>
#include <ippcore.h>
#include <ipptypes.h>
#include <ippi.h>

#include <boost/ptr_container/ptr_list.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/functional/hash.hpp>
#include <boost/any.hpp>

#include <algorithm>
#include <list>

#include "include/settings.h"
#include "include/bounds.h"
#include "include/memory.h"
#include "include/threading.h"
#include "include/attribute.h"

namespace afx {

// Ipp error handling function
inline void IppSafeCall(IppStatus status) {
  if (status != ippStsNoErr) { throw status; }
}

template <class T> class ImageBase : public AttributeBase {
 private:
  T* ptr_;
  size_t pitch_;
  afx::Bounds region_;

 public:
  ImageBase<T>() : ptr_(nullptr), pitch_(0), region_(afx::Bounds()) {}
  explicit ImageBase<T>(const afx::Bounds& region) : ptr_(nullptr), pitch_(0) {
    Allocate(region);
  }
  ImageBase<T>(unsigned int width, unsigned int height) : ptr_(nullptr), pitch_(0) {
    afx::Bounds region(0, 0, width, height);
    Allocate(region);
  }
  ImageBase<T>(const ImageBase<T>& other) {
    Copy(other);
  }
  ImageBase<T>& operator=(const ImageBase<T>& other) {
    Copy(other);
  }
  virtual ~ImageBase<T>() {
    Dispose();
  }
  virtual void Allocate(const afx::Bounds& region) {
    Dispose();
    region_ = region;
    ptr_ = afx::ImageMalloc<T>(region_.GetWidth(), region_.GetHeight(), &pitch_);
  }
  virtual void Copy(const ImageBase<T>& other) {
    region_ = other.region_;
    Allocate(region_);
    for (int y = region_.y1(); y <= region_.y2(); ++y) {
      memcpy(GetPtr(region_.x1(), y), other.GetPtr(other.region_.x1(), y), region_.GetWidth() * sizeof(T));
    }
  }
  virtual void Dispose() {
    afx::ImageFree(ptr_);
  }
  T* GetPtr() const {
    return ptr_;
  }
  T* GetPtr(int x, int y) const {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + (y - region_.y1()) * pitch_ + (x - region_.x1()) * sizeof(T));
  }
  T* GetPtrBnds(int x, int y) const {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + (region_.ClampY(y) - region_.y1()) * pitch_ + (region_.ClampX(x) - region_.x1()) * sizeof(T));
  }
  T* NextRow(T* ptr) {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + pitch_);
  }
  T* PreviousRow(T* ptr) {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) - pitch_);
  }
  size_t GetPitch() const {
    return pitch_;
  }
  afx::Bounds GetBounds() const {
    return region_;
  }
};


class Image : public AttributeBase {
 protected:
  Ipp32f* ptr_;
  int pitch_;
  IppiSize image_size_;
  Bounds region_;

 public:
  Image();
  Image(unsigned int width, unsigned int height);
  explicit Image(const Bounds& region);
  Image(const Image& other);
  Image& operator=(const Image& other);
  ~Image();
  void Create(unsigned int width, unsigned int height);
  virtual void Create(const Bounds& region);
  virtual void Copy(const Image& other);
  void Dispose();
  virtual void MemSet(float value);
  virtual void MemSet(float value, const Bounds& region);
  virtual void MemCpyIn(const float* src_ptr, size_t src_pitch, const Bounds& region);
  virtual void MemCpyOut(float* dst_ptr, size_t dst_pitch, const Bounds& region);
  Ipp32f* GetPtr() const;
  Ipp32f* GetPtr(int x, int y) const;
  Ipp32f* GetPtrBnds(int x, int y) const;
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
  void ResizeTile_(Bounds region, const Image& in, Image* out) {
    IppiBorderType border_style = ippBorderRepl;
    if (in.GetBounds().x1() > region.x1()) { border_style = static_cast<IppiBorderType>(border_style | ippBorderInMemRight); }
    if (in.GetBounds().x2() < region.x2()) { border_style = static_cast<IppiBorderType>(border_style | ippBorderInMemLeft); }
    if (in.GetBounds().y1() > region.y1()) { border_style = static_cast<IppiBorderType>(border_style | ippBorderInMemBottom); }
    if (in.GetBounds().y2() < region.y2()) { border_style = static_cast<IppiBorderType>(border_style | ippBorderInMemTop); }

    int spec_structure_size;
    int init_buffer_size;

    ippiResizeGetSize_32f(in.GetSize(), out->GetSize(), ippLanczos, 1, &spec_structure_size, &init_buffer_size);
    boost::scoped_ptr<Ipp8u> spec_structure_ptr(new Ipp8u(spec_structure_size));
    boost::scoped_ptr<Ipp8u> init_buffer_ptr(new Ipp8u(init_buffer_size));

    ippiResizeLanczosInit_32f(in.GetSize(), out->GetSize(), 2, reinterpret_cast<IppiResizeSpec_32f*>(spec_structure_ptr.get()), init_buffer_ptr.get());

    IppiSize dst_row_size = {out->GetBounds().GetWidth(), 1};
    int buffer_size;
    ippiResizeGetBufferSize_32f(reinterpret_cast<IppiResizeSpec_32f*>(spec_structure_ptr.get()), dst_row_size, 1, &buffer_size);
    boost::scoped_ptr<Ipp8u> buffer_ptr(new Ipp8u(buffer_size));

    IppiPoint dst_offset = {0, region.y1() - in.GetBounds().y1()};
    Ipp32f border_value = 0.0f;
    ippiResizeAntialiasing_32f_C1R(in.GetPtr(region.x1(), region.y1()), in.GetPitch(), out->GetPtr(region.x1(), region.y1()), out->GetPitch(),
                                   dst_offset, dst_row_size, border_style, &border_value, reinterpret_cast<IppiResizeSpec_32f*>(spec_structure_ptr.get()), buffer_ptr.get());
  }

 public:
  Resize(const Image& in, Image* out) {
    afx::Threader threader;
    threader.ThreadImageRows(in.GetBounds(), boost::bind(&Resize::ResizeTile_, this, _1, boost::cref(in), out));
  }
};

}  //  namespace afx

#endif  // INCLUDE_IMAGE_H_
