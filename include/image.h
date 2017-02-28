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

#include "include/settings.h"
#include "include/bounds.h"
#include "include/memory.h"
#include "include/threading.h"
#include "include/attribute.h"

namespace afx {

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
    Allocate(afx::Bounds(0, 0, width - 1, height - 1));
  }
  ImageBase<T>(const ImageBase<T>& other) {
    Copy(other);
  }
  ImageBase<T>& operator=(const ImageBase<T>& other) {
    Copy(other);
  }
  ~ImageBase<T>() {
    Dispose();
  }
  void Allocate(unsigned int width, unsigned int height) {
    Allocate(afx::Bounds(0, 0, width - 1, height - 1));
  }
  void Allocate(const afx::Bounds& region) {
    Dispose();
    region_ = region;
    ptr_ = afx::ImageMalloc<T>(region_.GetWidth(), region_.GetHeight(), &pitch_);
  }
  void Copy(const ImageBase<T>& other) {
    region_ = other.region_;
    Allocate(region_);
    for (int y = region_.y1(); y <= region_.y2(); ++y) {
      memcpy(GetPtr(region_.x1(), y), other.GetPtr(other.region_.x1(), y), region_.GetWidth() * sizeof(T));
    }
  }
  void Dispose() {
    afx::ImageFree(ptr_);
  }
  void MemCpyIn(const T* ptr, size_t pitch) {
    const T* source_ptr = ptr;
    T* dest_ptr = GetPtr(region_.x1(), region_.y1());
    size_t size = region_.GetWidth() * sizeof(T);
    for (int y = region_.y1(); y <= region_.y2(); ++y) {
      memcpy(dest_ptr, source_ptr, size);
      source_ptr = reinterpret_cast<const T*>((reinterpret_cast<const char*>(source_ptr) + pitch));
      dest_ptr = GetNextRow(dest_ptr);
    }
  }
  void MemCpyIn(const T* ptr, size_t pitch, afx::Bounds region) {
    const T* source_ptr = ptr;
    T* dest_ptr = GetPtr(region.x1(), region.y1());
    size_t size = region.GetWidth() * sizeof(T);
    for (int y = region.y1(); y <= region.y2(); ++y) {
      memcpy(dest_ptr, source_ptr, size);
      source_ptr = reinterpret_cast<const T*>((reinterpret_cast<const char*>(source_ptr) + pitch));
      dest_ptr = GetNextRow(dest_ptr);
    }
  }
  void MemCpyOut(T* ptr, size_t pitch) {
    T* source_ptr = GetPtr(region_.x1(), region_.y1());
    T* dest_ptr = ptr;
    size_t size = region_.GetWidth() * sizeof(T);
    for (int y = region_.y1(); y <= region_.y2(); ++y) {
      memcpy(dest_ptr, source_ptr, size);
      source_ptr = GetNextRow(dest_ptr);
      dest_ptr = reinterpret_cast<T*>((reinterpret_cast<char*>(dest_ptr) + pitch));
    }
  }
  void MemCpyOut(T* ptr, size_t pitch, afx::Bounds region) {
    T* source_ptr = GetPtr(region.x1(), region.y1());
    T* dest_ptr = ptr;
    size_t size = region.GetWidth() * sizeof(T);
    for (int y = region.y1(); y <= region.y2(); ++y) {
      memcpy(dest_ptr, source_ptr, size);
      source_ptr = GetNextRow(dest_ptr);
      dest_ptr = reinterpret_cast<T*>((reinterpret_cast<char*>(dest_ptr) + pitch));
    }
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
  T* GetNextRow(T* ptr) const {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + pitch_);
  }
  T* GetPreviousRow(T* ptr) const {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) - pitch_);
  }
  size_t GetPitch() const {
    return pitch_;
  }
  afx::Bounds GetBounds() const {
    return region_;
  }
  IppiSize GetSize() const {
    IppiSize size = {region_.GetWidth(), region_.GetHeight()};
    return size;
  }
};

typedef ImageBase<float> Image;

class ImageArray : public Array<ImageBase<float> > {
 public:
  void Add(const Bounds& region) {
    this->array_.push_back(new Image(region));
  }
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
    afx::ImageThreader threader;
    threader.ThreadImageRows(in.GetBounds(), boost::bind(&Resize::ResizeTile_, this, _1, boost::cref(in), out));
  }
};

}  //  namespace afx

#endif  // INCLUDE_IMAGE_H_
