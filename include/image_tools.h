// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef INCLUDE_IMAGE_TOOLS_H_
#define INCLUDE_IMAGE_TOOLS_H_

#include "math.h"

#include <stdexcept>

#include "include/image.h"
#include "include/threading.h"

namespace afx {

enum Borders {
  LeftBorder    = 0x01,
  RightBorder   = 0x02,
  BottomBorder  = 0x04,
  TopBorder     = 0x08,
};

class Point {
public:
  Point() : x_(0), y_(0) {}
  Point(int x, int y) : x_(x), y_(y) {}
  void SetPoint(int x, int y) {
    x_ = x;
    y_ = y;
  }
  float Distance(const Point& other) {
    return sqrtf(powf(x_ - other.x_, 2.0f) + powf(y_ - other.y_, 2.0f));
  }
  float Distance(int x, int y) {
    return sqrtf(powf(x_ - x, 2.0f) + powf(y_ - y, 2.0f));
  }
  int x() const {
    return x_;
  }
  int y() const {
    return y_;
  }
private:
  int x_;
  int y_;
};

class BorderExtender {
public:
  void Constant(const afx::Image& in_image, afx::Image* out_image, float value) {
    afx::Bounds in_region = in_image.GetBounds();
    afx::Bounds out_region = out_image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);

    afx::ImageThreader threader;
    if (out_region.x1() < in_region.x1()) {
      threader.ThreadImageChunks(left_region, boost::bind(&BorderExtender::Constant_, this, _1, out_image, value));
    }
    if (out_region.x2() > in_region.x2()) {
      threader.ThreadImageChunks(right_region, boost::bind(&BorderExtender::Constant_, this, _1, out_image, value));
    }
    if (out_region.y1() < in_region.y1()) {
      threader.ThreadImageChunks(bottom_region, boost::bind(&BorderExtender::Constant_, this, _1, out_image, value));
    }
    if (out_region.y2() > in_region.y2()) {
      threader.ThreadImageChunks(top_region, boost::bind(&BorderExtender::Constant_, this, _1, out_image, value));
    }
    out_image->MemCpyIn(in_image);
    threader.Wait();
  }
  void Repeat(const afx::Image& in_image, afx::Image* out_image) {
    afx::Bounds in_region = in_image.GetBounds();
    afx::Bounds out_region = out_image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);
    afx::ImageThreader threader;
    if (out_region.x1() < in_region.x1()) {
      threader.ThreadImageChunks(left_region, boost::bind(&BorderExtender::RepeatX_, this, _1, boost::cref(in_image), out_image, LeftBorder));
    }
    if (out_region.x2() > in_region.x2()) {
      threader.ThreadImageChunks(right_region, boost::bind(&BorderExtender::RepeatX_, this, _1, boost::cref(in_image), out_image, RightBorder));
    }
    if (out_region.y1() < in_region.y1()) {
      threader.ThreadImageChunksY(bottom_region, boost::bind(&BorderExtender::RepeatY_, this, _1, boost::cref(in_image), out_image, BottomBorder));
    }
    if (out_region.y2() > in_region.y2()) {
      threader.ThreadImageChunksY(top_region, boost::bind(&BorderExtender::RepeatY_, this, _1, boost::cref(in_image), out_image, TopBorder));
    }
    out_image->MemCpyIn(in_image);
    threader.Wait();
  }
  void RepeatFalloff(const afx::Image& in_image, afx::Image* out_image, float falloff) {
    afx::Bounds in_region = in_image.GetBounds();
    afx::Bounds out_region = out_image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);

    falloff *= 0.1f; // Falloff between 0 - 1. Scale by 0.1 for distance function

    afx::ImageThreader threader;
    if (out_region.x1() < in_region.x1()) {
      threader.ThreadImageChunks(left_region, boost::bind(&BorderExtender::RepeatFalloffX_, this, _1, boost::cref(in_image), out_image, falloff, LeftBorder));
    }
    if (out_region.x2() > in_region.x2()) {
      threader.ThreadImageChunks(right_region, boost::bind(&BorderExtender::RepeatFalloffX_, this, _1, boost::cref(in_image), out_image, falloff, RightBorder));
    }
    if (out_region.y1() < in_region.y1()) {
      threader.ThreadImageChunksY(bottom_region, boost::bind(&BorderExtender::RepeatFalloffY_, this, _1, boost::cref(in_image), out_image, falloff, BottomBorder));
    }
    if (out_region.y2() > in_region.y2()) {
      threader.ThreadImageChunksY(top_region, boost::bind(&BorderExtender::RepeatFalloffY_, this, _1, boost::cref(in_image), out_image, falloff, TopBorder));
    }
    out_image->MemCpyIn(in_image.GetPtr(), in_image.GetPitch(), in_image.GetBounds());
    threader.Wait();
  }
  void Mirror(const afx::Image& in_image, afx::Image* out_image) {
    afx::Bounds in_region = in_image.GetBounds();
    afx::Bounds out_region = out_image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);

    afx::ImageThreader threader;
    if (out_region.x1() < in_region.x1()) {
      threader.ThreadImageChunks(left_region, boost::bind(&BorderExtender::MirrorX_, this, _1, boost::cref(in_image), out_image, LeftBorder));
    }
    if (out_region.x2() > in_region.x2()) {
      threader.ThreadImageChunks(right_region, boost::bind(&BorderExtender::MirrorX_, this, _1, boost::cref(in_image), out_image, RightBorder));
    }
    if (out_region.y1() < in_region.y1()) {
      threader.ThreadImageChunksY(bottom_region, boost::bind(&BorderExtender::MirrorY_, this, _1, boost::cref(in_image), out_image, BottomBorder));
    }
    if (out_region.y2() > in_region.y2()) {
      threader.ThreadImageChunksY(top_region, boost::bind(&BorderExtender::MirrorY_, this, _1, boost::cref(in_image), out_image, TopBorder));
    }
    out_image->MemCpyIn(in_image.GetPtr(), in_image.GetPitch(), in_image.GetBounds());
    threader.Wait();
  }
private:
  void CalculateBounds_(const afx::Bounds& in_region, const afx::Bounds& out_region, afx::Bounds* left, afx::Bounds* right, afx::Bounds* bottom, afx::Bounds* top) {
    if (!out_region.WithinBounds(in_region)) {
      throw std::runtime_error("in_image bounds must be within out_image bounds");
    }
    // _______________
    // |x|_________|x|
    // |x|         |x|
    // |x|         |x|
    // |x|_________|x|
    // |x|_________|x|
    // Left and right regions y bounds go from out_image bounds y1 to y2
    *left = out_region;
    left->SetX2(out_region.ClampX(in_region.x1() - 1));
    *right = out_region;
    right->SetX1(out_region.ClampX(in_region.x2() + 1));
    // _______________
    // |_|_x_x_x_x_|_|
    // | |         | |
    // | |         | |
    // |_|_________|_|
    // |_|_x_x_x_x_|_|
    // Top and bottom regions x bounds go from in_image bounds x1 to x2
    *bottom = out_region;
    bottom->SetY2(out_region.ClampY(in_region.y1() - 1));
    bottom->SetX(in_region.x1(), in_region.x2());
    *top = out_region;
    top->SetY1(out_region.ClampY(in_region.y2() + 1));
    top->SetX(in_region.x1(), in_region.x2());
  }

  void Constant_(const afx::Bounds& region, afx::Image* out_image, float value) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        *out_ptr++ = value;
      }
    }
  }

  void RepeatX_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, unsigned char side) {
    afx::Bounds in_region = in_image.GetBounds();
    int border_pixel_x;
    if (side & LeftBorder) {
      border_pixel_x = in_region.x1();
    } else {
      border_pixel_x = in_region.x2();
    }
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      const float* border_ptr = in_image.GetPtr(border_pixel_x, in_region.ClampY(y));
      for (int x = region.x1(); x <= region.x2(); ++x) {
        *out_ptr++ = *border_ptr;
      }
    }
  }
  void RepeatY_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, unsigned char side) {
    afx::Bounds in_region = in_image.GetBounds();
    int border_pixel_y;
    if (side & BottomBorder) {
      border_pixel_y = in_region.y1();
    } else {
      border_pixel_y = in_region.y2();
    }
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float* out_ptr = out_image->GetPtr(x, region.y1());
      const float* border_ptr = in_image.GetPtr(in_region.ClampX(x), border_pixel_y);
      for (int y = region.y1(); y <= region.y2(); ++y) {
        *out_ptr = *border_ptr;
        out_ptr = out_image->GetNextRow(out_ptr);
      }
    }
  }

  void RepeatFalloffX_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, float falloff, unsigned char side) {
    afx::Bounds in_region = in_image.GetBounds();
    int border_pixel_x;
    if (side & LeftBorder) {
      border_pixel_x = in_region.x1();
    } else {
      border_pixel_x = in_region.x2();
    }
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      afx::Point border_point(border_pixel_x, in_region.ClampY(y));
      const float* border_ptr = in_image.GetPtr(border_pixel_x, in_region.ClampY(y));
      for (int x = region.x1(); x <= region.x2(); ++x) {
        float distance_falloff;
        if(y > in_region.y2()) {
          distance_falloff = expf(-border_point.Distance(x, y) * falloff);
        } else if (y < in_region.y1()) {
          distance_falloff = expf(-border_point.Distance(x, y) * falloff);
        } else {
          float distance = (side & LeftBorder) ? (in_region.x1() - x) : (x - in_region.x2());
          distance_falloff = expf(-distance * falloff);
        }
        *out_ptr++ = *border_ptr * distance_falloff;
      }
    }
  }
  void RepeatFalloffY_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, float falloff, unsigned char side) {
    afx::Bounds in_region = in_image.GetBounds();
    int border_pixel_y;
    if (side & BottomBorder) {
      border_pixel_y = in_region.y1();
    } else {
      border_pixel_y = in_region.y2();
    }
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float* out_ptr = out_image->GetPtr(x, region.y1());
      const float* border_ptr = in_image.GetPtr(in_region.ClampX(x), border_pixel_y);
      for (int y = region.y1(); y <= region.y2(); ++y) {
        float distance = (side & BottomBorder) ? (in_region.y1() - y) : (y - in_region.y2());
        *out_ptr = *border_ptr * expf(-distance * falloff);
        out_ptr = out_image->GetNextRow(out_ptr);
      }
    }
  }

  void MirrorX_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, unsigned char side) {
    afx::Bounds in_region = in_image.GetBounds();
    int x_offset;
    if (side & LeftBorder) {
      x_offset = 2 * in_region.x1() - 1;
    } else {
      x_offset = 2 * in_region.x2() + 1;
    }
    int y_top_offset = 2 * in_region.y2() + 1;
    int y_bottom_offset = 2 * in_region.y1() - 1;
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      if(y > in_region.y2()) {
        for (int x = region.x1(); x <= region.x2(); ++x) {
          *out_ptr++ = *in_image.GetPtrBnds(x_offset - x, y_top_offset - y);
        }
      } else if (y < in_region.y1()) {
        for (int x = region.x1(); x <= region.x2(); ++x) {
          *out_ptr++ = *in_image.GetPtrBnds(x_offset - x, y_bottom_offset - y);
        }
      } else {
        for (int x = region.x1(); x <= region.x2(); ++x) {
          *out_ptr++ = *in_image.GetPtrBnds(x_offset - x, y);
        }
      }
    }
  }
  void MirrorY_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, unsigned char side) {
    afx::Bounds in_region = in_image.GetBounds();
    int y_offset;
    if (side & BottomBorder) {
      y_offset = 2 * in_region.y1() - 1;
    } else {
      y_offset = 2 * in_region.y2() + 1;
    }
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float* out_ptr = out_image->GetPtr(x, region.y1());
      const float* border_ptr = in_image.GetPtr(x, in_region.y1());
      for (int y = region.y1(); y <= region.y2(); ++y) {
        *out_ptr = *in_image.GetPtrBnds(x, y_offset - y);
        out_ptr = out_image->GetNextRow(out_ptr);
      }
    }
  }
};

class Compliment {
 public:
  Compliment(afx::Image* image) {
    Compute(image, image->GetBounds());
  }
  Compliment(afx::Image* image, const afx::Bounds& region) {
    Compute(image, region);
  }
  void Compute(afx::Image* image) {
    Compute(image, image->GetBounds());
  }
  void Compute(afx::Image* image, const afx::Bounds& region) {
    afx::ImageThreader threader;
    threader.ThreadImageChunks(region, boost::bind(&Compliment::ComplimentTile_, this, _1, image));
    threader.Wait();
  }
 private:
  void ComplimentTile_(const afx::Bounds& region, afx::Image* image) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* ptr = image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        *ptr = 1.0f - *ptr;
        ptr++;
      }
    }
  }
};

}  //  namespace afx

#endif  // INCLUDE_IMAGE_TOOLS_H_
