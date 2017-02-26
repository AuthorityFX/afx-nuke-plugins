// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef INCLUDE_PIXEL_H_
#define INCLUDE_PIXEL_H_

#include "include/settings.h"

namespace afx {

template <class T> class PixelPtr {
private:
  T* ptr_;
  size_t stride_;
public:
  explicit PixelPtr(T* ptr) : ptr_(ptr), stride_(0) {}
  PixelPtr(T* ptr, size_t stride) : ptr_(ptr), stride_(stride) {}
  PixelPtr(const PixelPtr& other) {
    ptr_ = other.ptr_;
  }
  PixelPtr& operator=(const PixelPtr& other) {
    ptr_ = other.ptr_;
    return *this;
  }
  T& operator*() { return *ptr_; }
  void operator=(T* ptr) {
    ptr_ = ptr;
  }
  T* operator->() { return ptr_; }
  T* operator++() {
    ptr_ += stride_;
    return ptr_;
  }
  T* operator++(int) {
    T* old = ptr_;
    ptr_ += stride_;
    return old;
  }
  T* operator+=(const size_t& stride) {
    T* old = ptr_;
    ptr_ += stride;
    return old;
  }
  T* operator--() {
    ptr_ -= stride_;
    return ptr_;
  }
  T* operator--(int) {
    T* old = ptr_;
    ptr_ -= stride_;
    return old;
  }
};

class ReadOnlyPixel {
private:
  const float** pixel_;
  unsigned int size_;
  void Init(unsigned int size) {
    Dispose();
    size_ = size;
    pixel_ = new const float*[size_];
    for (unsigned int x = 0; x < size_; ++x) { pixel_[x] = nullptr; }
  }
  void CopyPixel(const ReadOnlyPixel& other) {
    Init(other.size_);
    for (unsigned int i = 0; i < size_; ++i) {
      pixel_[i] = other.pixel_[i];
    }
  }
  void Dispose() {
    if (pixel_ != nullptr) {
      delete[] pixel_;
      pixel_ = nullptr;
    }
  }

public:
  ReadOnlyPixel() : pixel_(nullptr) { Init(3); }
  explicit ReadOnlyPixel(unsigned int size) : pixel_(nullptr) { Init(size); }
  ReadOnlyPixel(const ReadOnlyPixel& other) { CopyPixel(other); }
  ReadOnlyPixel& operator=(const ReadOnlyPixel& other) {
    CopyPixel(other);
    return *this;
  }
  ~ReadOnlyPixel() { Dispose(); }

  void NextPixel() {
    for (unsigned int i = 0; i < size_; ++i) {
      if (pixel_ != nullptr) { pixel_[i]++; }
    }
  }
  void operator++(int) {
    NextPixel();
  }
  const float& operator[](unsigned int index) {
    return *pixel_[index];
  }
  void SetPtr(const float* ptr, unsigned int index) { pixel_[index] = ptr; }
  const float* GetPtr(unsigned int index) const { return pixel_[index]; }
  float GetVal(unsigned int index) const { return *pixel_[index]; }
  unsigned int GetSize() const {return size_; }
};

class Pixel {
private:
  float** pixel_;
  unsigned int size_;
  void Init(unsigned int size) {
    Dispose();
    size_ = size;
    pixel_ = new float*[size_];
    for (unsigned int x = 0; x < size_; ++x) { pixel_[x] = nullptr; }
  }
  void CopyPixel(const Pixel& other) {
    Init(other.size_);
    for (unsigned int i = 0; i < size_; ++i) {
      pixel_[i] = other.pixel_[i];
    }
  }
  void Dispose() {
    if (pixel_ != nullptr) {
      delete[] pixel_;
      pixel_ = nullptr;
    }
  }

public:
  Pixel() : pixel_(nullptr) { Init(3); }
  explicit Pixel(unsigned int size) : pixel_(nullptr) { Init(size); }
  Pixel(const Pixel& other) { CopyPixel(other); }
  Pixel& operator=(const Pixel& other) {
    CopyPixel(other);
    return *this;
  }
  ~Pixel() { Dispose(); }

  void NextPixel() {
    for (unsigned int i = 0; i < size_; ++i) {
      if (pixel_ != nullptr) { pixel_[i]++; }
    }
  }
  void operator++(int) {
    NextPixel();
  }
  float& operator[](unsigned int index) {
    return *pixel_[index];
  }
  void SetPtr(float* ptr, unsigned int index) { pixel_[index] = ptr; }
  void SetVal(float val, unsigned int index) { *pixel_[index] = val; }
  float* GetPtr(unsigned int index) const { return pixel_[index]; }
  float GetVal(unsigned int index) const { return *pixel_[index]; }
  unsigned int GetSize() const {return size_; }
};

}  //  namespace afx

#endif  // INCLUDE_PIXEL_H_