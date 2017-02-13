// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef TYPES_H_
#define TYPES_H_

#include <cuda_runtime.h>

#include "settings.h"

namespace afx {

template<typename T>
__host__ __device__
T Clamp(T value, T min_v, T max_v) {
  return value >= min_v ? value <= max_v ? value : max_v : min_v;
}

class Bounds {
private:
  int x1_, x2_, y1_, y2_;
public:
  __host__ __device__
  Bounds() : x1_(0), y1_(0), x2_(0), y2_(0) {}
  __host__ __device__
  Bounds(int x1, int y1, int x2, int y2) :  x1_(x1), y1_(y1), x2_(x2), y2_(y2) {}

  __host__ __device__
  int x1() const { return x1_; }
  __host__ __device__
  int x2() const { return x2_; }
  __host__ __device__
  int y1() const { return y1_; }
  __host__ __device__
  int y2() const { return y2_; }

  __host__ __device__
  int& x1Ref() { return x1_; }
  __host__ __device__
  int& x2Ref() { return x2_; }
  __host__ __device__
  int& y1Ref() { return y1_; }
  __host__ __device__
  int& y2Ref() { return y2_; }

  __host__ __device__
  void SetX(int x1, int x2) {
    x1_ = x1;
    x2_ = x2;
  }
  __host__ __device__
  void SetY(int y1, int y2) {
    y1_ = y1;
    y2_ = y2;
  }

  __host__ __device__
  void SetX1(int x) { x1_ = x; }
  __host__ __device__
  void SetX2(int x) { x2_ = x; }
  __host__ __device__
  void SetY1(int y) { y1_ = y; }
  __host__ __device__
  void SetY2(int y) { y2_ = y; }

  __host__ __device__
  void SetBounds(int x1, int y1, int x2, int y2) {
    x1_ = x1;
    x2_ = x2;
    y1_ = y1;
    y2_ = y2;
  }
  __host__ __device__
  void PadBounds(unsigned int size) {
    x1_ -= size;
    y1_ -= size;
    x2_ += size;
    y2_ += size;
  }
  __host__ __device__
  void PadBounds(unsigned int x, unsigned int y) {
    x1_ -= x;
    y1_ -= y;
    x2_ += x;
    y2_ += y;
  }
  __host__ __device__
  void Intersect(const Bounds& other) {
    x1_ = x1_ < other.x1_ ? other.x1_ : x1_;
    y1_ = y1_ < other.y1_ ? other.y1_ : y1_;
    x2_ = x2_ > other.x2_ ? other.x2_ : x2_;
    y2_ = y2_ > other.y2_ ? other.y2_ : y2_;
  }
  __host__ __device__
  bool Intersects(const Bounds& other) {
    if (other.x2_ < x1_) { return false; }
    if (other.y2_ < y1_) { return false; }
    if (other.x1_ > x2_) { return false; }
    if (other.y1_ > y2_) { return false; }
    return true;
  }
  __host__ __device__
  Bounds GetPadBounds(unsigned int size) const {
    Bounds padded = *this;
    padded.PadBounds(size);
    return padded;
  }
  __host__ __device__
  Bounds GetPadBounds(unsigned int x, unsigned int y) const {
    Bounds padded = *this;
    padded.PadBounds(x, y);
    return padded;
  }
  __host__ __device__
  unsigned int GetWidth() const { return x2_ - x1_ + 1; }
  __host__ __device__
  unsigned int GetHeight() const { return y2_ - y1_ + 1; }
  __host__ __device__
  bool CheckBounds(int x, int y) const {
    if (x < x1_ || x > x2_ || y < y1_ || y > y2_) {
      return false;
    } else {
      return true;
    }
  }
  __host__ __device__
  bool CheckBoundsX(int x) const {
    if (x < x1_ || x > x2_) {
      return false;
    } else {
      return true;
    }
  }
  __host__ __device__
  bool CheckBoundsY(int y) const {
    if (y < y1_ || y > y2_) {
      return false;
    } else {
      return true;
    }
  }
  __host__ __device__
  float GetCenterX() const { return (float)(x2_ - x1_) / 2.0f + x1_; }
  __host__ __device__
  float GetCenterY() const { return (float)(y2_ - y1_) / 2.0f + y1_; }
  __host__ __device__
  int ClampX(int x) const {
    return x >= x1_ ? (x <= x2_ ? x : x2_) : x1_;
  }
  __host__ __device__
  int ClampY(int y) const {
    return y >= y1_ ? (y <= y2_ ? y : y2_) : y1_;
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
  ReadOnlyPixel(unsigned int size) : pixel_(nullptr) { Init(size); }
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
  Pixel(unsigned int size) : pixel_(nullptr) { Init(size); }
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

} // namespace afx

#endif  // TYPES_H_
