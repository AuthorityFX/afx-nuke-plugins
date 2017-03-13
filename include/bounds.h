// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef INCLUDE_BOUNDS_H_
#define INCLUDE_BOUNDS_H_

#include <cuda_runtime.h>

#include <algorithm>

#include "include/settings.h"

namespace afx {

namespace math {

template<typename T>
__host__ __device__
T Clamp(T value, T min_v, T max_v) {
  return value >= min_v ? value <= max_v ? value : max_v : min_v;
}

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
  Bounds(unsigned int width, unsigned int height) {
    x1_ = 0;
    y1_ = 0;
    x2_ = width > 0 ? width - 1 : 0;
    y2_ = height > 0 ? height - 1 : 0;
  }

  __host__ __device__
  bool operator==(const Bounds& other) {
    if (other.x1_ == x1_ && other.x2_ == x2_ &&
      other.y1_ == y1_ && other.y2_ == y2_) {
      return true;
      } else {
        return false;
      }
  }
  __host__ __device__
  bool operator!=(const Bounds& other) {
    if (other.x1_ != x1_ || other.x2_ != x2_ ||
        other.y1_ != y1_ || other.y2_ != y2_) {
      return true;
    } else {
      return false;
    }
  }
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
  void ErodeBounds(unsigned int size) {
    unsigned int max_x_size = (GetWidth() + 1) / 2 - 1;
    unsigned int max_y_size = (GetHeight() + 1) / 2 - 1;
    x1_ += std::min(size, max_x_size);
    y1_ += std::min(size, max_x_size);
    x2_ -= std::min(size, max_y_size);
    y2_ -= std::min(size, max_y_size);
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
  Bounds GetIntersection(const Bounds& other) const {
    afx::Bounds new_bounds = *this;
    new_bounds.Intersect(other);
    return new_bounds;
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
  bool WithinBounds(int x, int y) const {
    if (x < x1_ || x > x2_ || y < y1_ || y > y2_) {
      return false;
    } else {
      return true;
    }
  }
  __host__ __device__
  bool WithinBounds(const Bounds& other) const {
    if (other.x1_ < x1_ || other.x2_ > x2_ || other.y1_ < y1_ || other.y2_ > y2_) {
      return false;
    } else {
      return true;
    }
  }
  __host__ __device__
  bool WithinBoundsX(int x) const {
    if (x < x1_ || x > x2_) {
      return false;
    } else {
      return true;
    }
  }
  __host__ __device__
  bool WithinBoundsY(int y) const {
    if (y < y1_ || y > y2_) {
      return false;
    } else {
      return true;
    }
  }
  __host__ __device__
  float GetCenterX() const { return static_cast<float>(x2_ - x1_) / 2.0f + x1_; }
  __host__ __device__
  float GetCenterY() const { return static_cast<float>(y2_ - y1_) / 2.0f + y1_; }
  __host__ __device__
  int ClampX(int x) const {
    return x > x1_ ? (x < x2_ ? x : x2_) : x1_;
  }
  __host__ __device__
  int ClampY(int y) const {
    return y > y1_ ? (y < y2_ ? y : y2_) : y1_;
  }
};

}  // namespace afx

#endif  // INCLUDE_BOUNDS_H_
