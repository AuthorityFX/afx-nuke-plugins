// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#include "image.h"

#include <ipp.h>
#include <ippcore.h>
#include <ippi.h>

#include "settings.h"

namespace afx {

Image::Image() : ptr_(nullptr), pitch_(0), image_size_((IppiSize){0, 0}), region_(Bounds()) {}
Image::Image(unsigned int width, unsigned int height) : ptr_(nullptr) { Create(Bounds(0, 0, width - 1, height - 1)); }
Image::Image(const Bounds& region) : ptr_(nullptr) { Create(region); }
Image::Image(const Image& other) {
  Copy(other);
}
Image& Image::operator=(const Image& other) {
  Copy(other);
  return *this;
}
Image::~Image() {
  if (ptr_ != nullptr) {
    ippiFree(ptr_);
    ptr_ = nullptr;
  }
}
void Image::Create(unsigned int width, unsigned int height) {
  Create(Bounds(0, 0, width - 1, height - 1));
}
void Image::Create(const Bounds& region) {
  Dispose();
  region_ = region;
  image_size_.width = region_.x2() - region_.x1() + 1;
  image_size_.height = region_.y2() - region_.y1() + 1;
  ptr_ = ippiMalloc_32f_C1(image_size_.width, image_size_.height, &pitch_);
}
void Image::Copy(const Image& other) {
  Create(other.region_);
  IppSafeCall(ippiCopy_32f_C1R(other.GetPtr(), other.GetPitch(), ptr_, pitch_, image_size_));
}
void Image::Dispose() {
  if (ptr_ != nullptr) {
    ippiFree(ptr_);
    ptr_ = nullptr;
  }
}
void Image::MemSet(float value) {
  IppSafeCall(ippiSet_32f_C1R(value, GetPtr(), pitch_, image_size_));
}
void Image::MemSet(float value, const Bounds& region) {
    IppiSize size;
    size.width = region.GetWidth();
    size.height = region.GetHeight();
    IppSafeCall(ippiSet_32f_C1R(value, GetPtr(region.x1(), region.y1()), pitch_, size));
}
void Image::MemCpyIn(const float* src_ptr, size_t src_pitch, const Bounds& region) {
  IppiSize size;
  size.width = region.GetWidth();
  size.height = region.GetHeight();
  IppSafeCall(ippiCopy_32f_C1R(src_ptr, src_pitch, GetPtr(region.x1(), region.y1()), pitch_, size));
}
void Image::MemCpyOut(float* dst_ptr, size_t dst_pitch, const Bounds& region) {
  IppiSize size;
  size.width = region.GetWidth();
  size.height = region.GetHeight();
  IppSafeCall(ippiCopy_32f_C1R(GetPtr(region.x1(), region.y1()), pitch_, dst_ptr, dst_pitch, size));
}
Ipp32f* Image::GetPtr() const { return ptr_; }
Ipp32f* Image::GetPtr(int x, int y) const {
  return (Ipp32f*)((char*)(ptr_) + (y - region_.y1()) * pitch_ + (x - region_.x1()) * sizeof(Ipp32f));
}
Ipp32f* Image::GetPtrBnds(int x, int y) const {
  return (Ipp32f*)((char*)(ptr_) + (region_.ClampY(y) - region_.y1()) * pitch_ + (region_.ClampX(x) - region_.x1()) * sizeof(Ipp32f));
}
Ipp32f Image::GetVal(int x, int y) const {
  return *(Ipp32f*)((char*)(ptr_) + (y - region_.y1()) * pitch_ + (x - region_.x1()) * sizeof(Ipp32f));
}
Ipp32f Image::GetValBnds(int x, int y) const {
  return *(Ipp32f*)((char*)(ptr_) + (region_.ClampY(y) - region_.y1()) * pitch_ + (region_.ClampX(x) - region_.x1()) * sizeof(Ipp32f));
}

size_t Image::GetPitch() const { return pitch_; }
IppiSize Image::GetSize() const { return image_size_; }
Bounds Image::GetBounds() const { return region_; }



void ImageArray::AddImage(const Bounds& region) {
    this->array_.push_back(new Image(region));
}

} // namespace afx
