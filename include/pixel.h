// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef AFX_PIXEL_H_
#define AFX_PIXEL_H_

namespace afx {

template <class T> class PackedPixel {
 public:
  explicit PackedPixel(T* ptr) : ptr_(ptr), stride_(0) {}
  PackedPixel(T* ptr, size_t stride) : ptr_(ptr), stride_(stride) {}
  PackedPixel(const PackedPixel& other) {
    ptr_ = other.ptr_;
  }
  PackedPixel& operator=(const PackedPixel& other) {
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

private:
  T* ptr_;
  size_t stride_;
};

template <class T> class Pixel {
 public:
  Pixel() : pointers_(nullptr) { Allocate_(3); }
  explicit Pixel(unsigned int size) : pointers_(nullptr) { Allocate_(size); }
  Pixel(const Pixel& other) { CopyPixel_(other); }
  Pixel& operator=(const Pixel& other) {
    CopyPixel_(other);
    return *this;
  }
  ~Pixel() { Dispose_(); }

  void NextPixel() {
    for (unsigned int i = 0; i < size_; ++i) {
      if (pointers_ != nullptr) { pointers_[i]++; }
    }
  }
  void operator++(int) {
    NextPixel();
  }
  T& operator[](unsigned int index) {
    return *pointers_[index];
  }
  void SetPtr(T* ptr, unsigned int index) { pointers_[index] = ptr; }
  void SetVal(T val, unsigned int index) { *pointers_[index] = val; }
  T* GetPtr(unsigned int index) const { return pointers_[index]; }
  T GetVal(unsigned int index) const { return *pointers_[index]; }
  unsigned int GetSize() const {return size_; }

 private:
  T** pointers_;
  unsigned int size_;
  void Allocate_(unsigned int size) {
    Dispose_();
    size_ = size;
    pointers_ = new T*[size_];
    for (unsigned int x = 0; x < size_; ++x) { pointers_[x] = nullptr; }
  }
  void CopyPixel_(const Pixel& other) {
    Allocate_(other.size_);
    for (unsigned int i = 0; i < size_; ++i) {
      pointers_[i] = other.pointers_[i];
    }
  }
  void Dispose_() {
    if (pointers_ != nullptr) {
      delete[] pointers_;
      pointers_ = nullptr;
    }
  }
};

}  //  namespace afx

#endif  // AFX_PIXEL_H_
