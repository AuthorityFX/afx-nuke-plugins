// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef AFX_IMAGE_TOOLS_H_
#define AFX_IMAGE_TOOLS_H_

#include "math.h"

#include <stdexcept>
#include <cstdint>

#include "include/image.h"
#include "include/pixel.h"
#include "include/threading.h"
#include "include/color_op.h"

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


class Morphology {
 public:
  void Dilate(const afx::Image& in_image, afx::Image* out_image, unsigned int window_size) {
    ImageThreader threader;
    threader.ThreadRowChunks(in_image.GetBounds(), boost::bind(&Morphology::Dilate_, this, _1, boost::cref(in_image), out_image, window_size));
    threader.Wait();
  }
   void Erode(const afx::Image& in_image, afx::Image* out_image, unsigned int window_size) {
    ImageThreader threader;
    threader.ThreadRowChunks(in_image.GetBounds(), boost::bind(&Morphology::Erode_, this, _1, boost::cref(in_image), out_image, window_size));
    threader.Wait();
  }

 private:
  void Dilate_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, unsigned int window_size) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* in_ptr = in_image.GetPtr(region.x1(), y);
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        float max_value = *in_ptr++;
        for (int window_y = y - static_cast<int>(window_size); window_y <= y + static_cast<int>(window_size); ++window_y) {
          const float* window_ptr = in_image.GetPtrBnds(x - window_size, window_y);
          for (int window_x = x - static_cast<int>(window_size); window_x <= x + static_cast<int>(window_size); ++window_x) {
            if (*window_ptr > max_value) { max_value = *window_ptr; }
            if (window_x >= in_image.GetBounds().x1() && window_x < in_image.GetBounds().x2()) { window_ptr++; }
          }
        }
      *out_ptr++ = max_value;
      }
    }
  }

  void Erode_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, unsigned int window_size) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* in_ptr = in_image.GetPtr(region.x1(), y);
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        float min_value = *in_ptr++;
        for (int window_y = y - static_cast<int>(window_size); window_y <= y + static_cast<int>(window_size); ++window_y) {
          const float* window_ptr = in_image.GetPtrBnds(x - window_size, window_y);
          for (int window_x = x - static_cast<int>(window_size); window_x <= x + static_cast<int>(window_size); ++window_x) {
            if (*window_ptr < min_value) { min_value = *window_ptr; }
            if (window_x >= in_image.GetBounds().x1() && window_x < in_image.GetBounds().x2()) { window_ptr++; }
          }
        }
        *out_ptr++ = min_value;
      }
    }
  }
};

class BorderExtender {
public:
  // out_image must be larger than in_image. The difference between the bounding
  // regions will be extended with constant value
  void Constant(const afx::Image& in_image, afx::Image* out_image, float value) {
    afx::Bounds in_region = in_image.GetBounds();
    afx::Bounds out_region = out_image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);

    afx::ImageThreader threader;
    if (out_region.x1() < in_region.x1()) {
      threader.ThreadRowChunks(left_region, boost::bind(&BorderExtender::Constant_, this, _1, out_image, value));
    }
    if (out_region.x2() > in_region.x2()) {
      threader.ThreadRowChunks(right_region, boost::bind(&BorderExtender::Constant_, this, _1, out_image, value));
    }
    if (out_region.y1() < in_region.y1()) {
      threader.ThreadRowChunks(bottom_region, boost::bind(&BorderExtender::Constant_, this, _1, out_image, value));
    }
    if (out_region.y2() > in_region.y2()) {
      threader.ThreadRowChunks(top_region, boost::bind(&BorderExtender::Constant_, this, _1, out_image, value));
    }
    out_image->MemCpyIn(in_image);
    threader.Wait();
  }
  // in_place borded extend. in_region will remain unchanged.
  void Constant(afx::Image* image, float value, afx::Bounds in_region) {
    afx::Bounds out_region = image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);

    afx::ImageThreader threader;
    if (out_region.x1() < in_region.x1()) {
      threader.ThreadRowChunks(left_region, boost::bind(&BorderExtender::Constant_, this, _1, image, value));
    }
    if (out_region.x2() > in_region.x2()) {
      threader.ThreadRowChunks(right_region, boost::bind(&BorderExtender::Constant_, this, _1, image, value));
    }
    if (out_region.y1() < in_region.y1()) {
      threader.ThreadRowChunks(bottom_region, boost::bind(&BorderExtender::Constant_, this, _1, image, value));
    }
    if (out_region.y2() > in_region.y2()) {
      threader.ThreadRowChunks(top_region, boost::bind(&BorderExtender::Constant_, this, _1, image, value));
    }
    threader.Wait();
  }
  void Repeat(const afx::Image& in_image, afx::Image* out_image) {
    afx::Bounds in_region = in_image.GetBounds();
    afx::Bounds out_region = out_image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);
    afx::ImageThreader threader;
    if (out_region.x1() < in_region.x1()) {
      threader.ThreadRowChunks(left_region, boost::bind(&BorderExtender::RepeatX_, this, _1, boost::cref(in_image), out_image, LeftBorder));
    }
    if (out_region.x2() > in_region.x2()) {
      threader.ThreadRowChunks(right_region, boost::bind(&BorderExtender::RepeatX_, this, _1, boost::cref(in_image), out_image, RightBorder));
    }
    if (out_region.y1() < in_region.y1()) {
      threader.ThreadColumnChunks(bottom_region, boost::bind(&BorderExtender::RepeatY_, this, _1, boost::cref(in_image), out_image, BottomBorder));
    }
    if (out_region.y2() > in_region.y2()) {
      threader.ThreadColumnChunks(top_region, boost::bind(&BorderExtender::RepeatY_, this, _1, boost::cref(in_image), out_image, TopBorder));
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
      threader.ThreadRowChunks(left_region, boost::bind(&BorderExtender::RepeatFalloffX_, this, _1, boost::cref(in_image), out_image, falloff, LeftBorder));
    }
    if (out_region.x2() > in_region.x2()) {
      threader.ThreadRowChunks(right_region, boost::bind(&BorderExtender::RepeatFalloffX_, this, _1, boost::cref(in_image), out_image, falloff, RightBorder));
    }
    if (out_region.y1() < in_region.y1()) {
      threader.ThreadColumnChunks(bottom_region, boost::bind(&BorderExtender::RepeatFalloffY_, this, _1, boost::cref(in_image), out_image, falloff, BottomBorder));
    }
    if (out_region.y2() > in_region.y2()) {
      threader.ThreadColumnChunks(top_region, boost::bind(&BorderExtender::RepeatFalloffY_, this, _1, boost::cref(in_image), out_image, falloff, TopBorder));
    }
    out_image->MemCpyIn(in_image.GetPtr(), in_image.GetPitch(), in_image.GetBounds());
    threader.Wait();
  }
  void RepeatFalloff(afx::Image* image, float falloff, afx::Bounds in_region) {
    afx::Bounds out_region = image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);

    falloff *= 0.1f; // Falloff between 0 - 1. Scale by 0.1 for distance function

    afx::ImageThreader threader;
    if (out_region.x1() < in_region.x1()) {
      threader.ThreadRowChunks(left_region, boost::bind(&BorderExtender::RepeatFalloffX_, this, _1, boost::cref(*image), image, falloff, LeftBorder));
    }
    if (out_region.x2() > in_region.x2()) {
      threader.ThreadRowChunks(right_region, boost::bind(&BorderExtender::RepeatFalloffX_, this, _1, boost::cref(*image), image, falloff, RightBorder));
    }
    if (out_region.y1() < in_region.y1()) {
      threader.ThreadColumnChunks(bottom_region, boost::bind(&BorderExtender::RepeatFalloffY_, this, _1, boost::cref(*image), image, falloff, BottomBorder));
    }
    if (out_region.y2() > in_region.y2()) {
      threader.ThreadColumnChunks(top_region, boost::bind(&BorderExtender::RepeatFalloffY_, this, _1, boost::cref(*image), image, falloff, TopBorder));
    }
    threader.Wait();
  }
  void Mirror(const afx::Image& in_image, afx::Image* out_image) {
    afx::Bounds in_region = in_image.GetBounds();
    afx::Bounds out_region = out_image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);

    afx::ImageThreader threader;
    if (out_region.x1() < in_region.x1()) {
      threader.ThreadRowChunks(left_region, boost::bind(&BorderExtender::MirrorX_, this, _1, boost::cref(in_image), out_image, LeftBorder));
    }
    if (out_region.x2() > in_region.x2()) {
      threader.ThreadRowChunks(right_region, boost::bind(&BorderExtender::MirrorX_, this, _1, boost::cref(in_image), out_image, RightBorder));
    }
    if (out_region.y1() < in_region.y1()) {
      threader.ThreadColumnChunks(bottom_region, boost::bind(&BorderExtender::MirrorY_, this, _1, boost::cref(in_image), out_image, BottomBorder));
    }
    if (out_region.y2() > in_region.y2()) {
      threader.ThreadColumnChunks(top_region, boost::bind(&BorderExtender::MirrorY_, this, _1, boost::cref(in_image), out_image, TopBorder));
    }
    out_image->MemCpyIn(in_image.GetPtr(), in_image.GetPitch(), in_image.GetBounds());
    threader.Wait();
  }
  void Mirror(afx::Image* image, const afx::Bounds& in_region) {
    afx::Bounds out_region = image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);

    afx::ImageThreader threader;
    if (out_region.x1() < in_region.x1()) {
      threader.ThreadRowChunks(left_region, boost::bind(&BorderExtender::MirrorX_, this, _1, boost::cref(*image), image, LeftBorder));
    }
    if (out_region.x2() > in_region.x2()) {
      threader.ThreadRowChunks(right_region, boost::bind(&BorderExtender::MirrorX_, this, _1, boost::cref(*image), image, RightBorder));
    }
    if (out_region.y1() < in_region.y1()) {
      threader.ThreadColumnChunks(bottom_region, boost::bind(&BorderExtender::MirrorY_, this, _1, boost::cref(*image), image, BottomBorder));
    }
    if (out_region.y2() > in_region.y2()) {
      threader.ThreadColumnChunks(top_region, boost::bind(&BorderExtender::MirrorY_, this, _1, boost::cref(*image), image, TopBorder));
    }
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

  void RepeatX_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, std::uint8_t side) {
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
  void RepeatY_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, std::uint8_t side) {
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

  void RepeatFalloffX_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, float falloff, std::uint8_t side) {
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
  void RepeatFalloffY_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, float falloff, std::uint8_t side) {
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

  void MirrorX_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, std::uint8_t side) {
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
  void MirrorY_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, std::uint8_t side) {
    afx::Bounds in_region = in_image.GetBounds();
    int y_offset;
    if (side & BottomBorder) {
      y_offset = 2 * in_region.y1() - 1;
    } else {
      y_offset = 2 * in_region.y2() + 1;
    }
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float* out_ptr = out_image->GetPtr(x, region.y1());
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
    threader.ThreadRowChunks(region, boost::bind(&Compliment::ComplimentTile_, this, _1, image));
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

class Threshold {
public:
  void ThresholdLayer(afx::ImageLayer* layer, const afx::Image& mask, float threshold, float falloff) {
    afx::Bounds region = layer->GetChannel(0)->GetBounds();
    afx::ImageThreader threader;
    threader.ThreadRowChunks(region, boost::bind(&Threshold::ThresholdLayerTile_, this, _1, layer, boost::cref(mask), threshold, falloff));
    threader.Wait();
  }
  void ThresholdImage(afx::Image* image, float threshold, float falloff) {
    afx::Bounds region = image->GetBounds();
    afx::ImageThreader threader;
    threader.ThreadRowChunks(region, boost::bind(&Threshold::ThresholdImageTile_, this, _1, image, threshold, falloff));
    threader.Wait();
  }
  void ThresholdImage(afx::Image* image, const afx::Image& mask, float threshold, float falloff) {
    afx::Bounds region = image->GetBounds();
    afx::ImageThreader threader;
    threader.ThreadRowChunks(region, boost::bind(&Threshold::ThresholdImageTile_, this, _1, image, boost::cref(mask), threshold, falloff));
    threader.Wait();
  }
  void ThresholdImage(afx::Image* image, float threshold, float falloff, afx::Bounds region) {
    afx::ImageThreader threader;
    threader.ThreadRowChunks(region, boost::bind(&Threshold::ThresholdImageTile_, this, _1, image, threshold, falloff));
    threader.Wait();
  }
  void ThresholdImage(afx::Image* image, const afx::Image& mask, float threshold, float falloff, afx::Bounds region) {
    afx::ImageThreader threader;
    threader.ThreadRowChunks(region, boost::bind(&Threshold::ThresholdImageTile_, this, _1, image, boost::cref(mask), threshold, falloff));
    threader.Wait();
  }
private:
  void ThresholdLayerTile_(const afx::Bounds& region, afx::ImageLayer* layer, const afx::Image& mask, float threshold, float falloff) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      afx::Pixel<float> pixel;
      pixel = layer->GetWritePixel(region.x1(), y);
      const float* mask_ptr = mask.GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        float rgb[3];
        for (int i = 0; i < 3; ++i) {
          rgb[i] = pixel[i];
        }
        afx::ThresholdColor(rgb, threshold, falloff);
        for (int i = 0; i < 3; ++i) {
          pixel[i] = rgb[i] * *mask_ptr;
        }
      }
    }
  }
  void ThresholdLayerTile_(const afx::Bounds& region, afx::ImageLayer* layer, float threshold, float falloff) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      afx::Pixel<float> pixel;
      pixel = layer->GetWritePixel(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        float rgb[3];
        for (int i = 0; i < 3; ++i) {
          rgb[i] = pixel[i];
        }
        afx::ThresholdColor(rgb, threshold, falloff);
        for (int i = 0; i < 3; ++i) {
          pixel[i] = rgb[i];
        }
      }
    }
  }
  void ThresholdImageTile_(const afx::Bounds& region, afx::Image* image, const afx::Image& mask, float threshold, float falloff) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* in_ptr = image->GetPtr(region.x1(), y);
      const float* mask_ptr = mask.GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        afx::ThresholdValue(in_ptr, threshold, falloff);
        *in_ptr++ *= *mask_ptr++;
      }
    }
  }
  void ThresholdImageTile_(const afx::Bounds& region, afx::Image* image, float threshold, float falloff) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* in_ptr = image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        afx::ThresholdValue(in_ptr, threshold, falloff);
        in_ptr++;
      }
    }
  }

};

}  //  namespace afx

#endif  // AFX_IMAGE_TOOLS_H_
