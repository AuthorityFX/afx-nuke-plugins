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

#include <stdexcept>

#include "include/image.h"
#include "include/threading.h"

namespace afx {

class ExtendBorder {
public:
  void ExtendConstant(const afx::Image& in_image, afx::Image* out_image, float value) {
    afx::Bounds in_region = in_image.GetBounds();
    afx::Bounds out_region = out_image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);

    afx::ImageThreader threader;
    threader.ThreadImageChunks(left_region, boost::bind(&ExtendBorder::Constant_, this, _1, in_image, out_image, value));
    threader.ThreadImageChunks(right_region, boost::bind(&ExtendBorder::Constant_, this, _1, in_image, out_image, value));
    threader.ThreadImageChunks(bottom_region, boost::bind(&ExtendBorder::Constant_, this, _1, in_image, out_image, value));
    threader.ThreadImageChunks(top_region, boost::bind(&ExtendBorder::Constant_, this, _1, in_image, out_image, value));
    out_image->MemCpyIn(in_image.GetPtr(), in_image.GetPitch(), in_image.GetBounds());
    threader.Wait();
  }
  void ExtendRepeat(const afx::Image& in_image, afx::Image* out_image) {
    afx::Bounds in_region = in_image.GetBounds();
    afx::Bounds out_region = out_image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);

    afx::ImageThreader threader;
    threader.ThreadImageChunks(left_region, boost::bind(&ExtendBorder::RepeatLeft_, this, _1, in_image, out_image));
    threader.ThreadImageChunks(right_region, boost::bind(&ExtendBorder::RepeatRight_, this, _1, in_image, out_image));
    threader.ThreadImageChunksY(bottom_region, boost::bind(&ExtendBorder::RepeatBottom_, this, _1, in_image, out_image));
    threader.ThreadImageChunksY(top_region, boost::bind(&ExtendBorder::RepeatTop_, this, _1, in_image, out_image));
    out_image->MemCpyIn(in_image.GetPtr(), in_image.GetPitch(), in_image.GetBounds());
    threader.Wait();
  }
  void ExtendRepeatFalloff(const afx::Image& in_image, afx::Image* out_image, float falloff) {
    afx::Bounds in_region = in_image.GetBounds();
    afx::Bounds out_region = out_image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);

    falloff *= 0.1f; // Falloff between 0 - 1. Scale by 0.1 for distance function

    afx::ImageThreader threader;
    threader.ThreadImageChunks(left_region, boost::bind(&ExtendBorder::RepeatFalloffLeft_, this, _1, in_image, out_image, falloff));
    threader.ThreadImageChunks(right_region, boost::bind(&ExtendBorder::RepeatFalloffRight_, this, _1, in_image, out_image, falloff));
    threader.ThreadImageChunksY(bottom_region, boost::bind(&ExtendBorder::RepeatFalloffBottom_, this, _1, in_image, out_image, falloff));
    threader.ThreadImageChunksY(top_region, boost::bind(&ExtendBorder::RepeatFalloffTop_, this, _1, in_image, out_image, falloff));
    out_image->MemCpyIn(in_image.GetPtr(), in_image.GetPitch(), in_image.GetBounds());
    threader.Wait();
  }
  void ExtendMirror(const afx::Image& in_image, afx::Image* out_image) {
    afx::Bounds in_region = in_image.GetBounds();
    afx::Bounds out_region = out_image->GetBounds();

    afx::Bounds left_region, right_region, bottom_region, top_region;
    CalculateBounds_(in_region, out_region, &left_region, &right_region, &bottom_region, &top_region);

    afx::ImageThreader threader;
    threader.ThreadImageChunks(left_region, boost::bind(&ExtendBorder::MirrorLeft_, this, _1, in_image, out_image));
    threader.ThreadImageChunks(right_region, boost::bind(&ExtendBorder::MirrorRight_, this, _1, in_image, out_image));
    threader.ThreadImageChunksY(bottom_region, boost::bind(&ExtendBorder::MirrorBottom_, this, _1, in_image, out_image));
    threader.ThreadImageChunksY(top_region, boost::bind(&ExtendBorder::MirrorTop_, this, _1, in_image, out_image));
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
    afx::Bounds left_region = out_region;
    left_region.SetX2(out_region.ClampX(in_region.x1() - 1));
    afx::Bounds right_region = out_region;
    right_region.SetX1(out_region.ClampX(in_region.x2() + 1));
    // _______________
    // |_|_x_x_x_x_|_|
    // | |         | |
    // | |         | |
    // |_|_________|_|
    // |_|_x_x_x_x_|_|
    // Top and bottom regions x bounds go from in_image bounds x1 to x2
    afx::Bounds bottom_region = out_region;
    bottom_region.SetY2(out_region.ClampY(in_region.y1() - 1));
    bottom_region.SetX(in_region.x1(), in_region.x2());
    afx::Bounds top_region = out_region;
    top_region.SetY1(out_region.ClampY(in_region.y2() + 1));
    top_region.SetX(in_region.x1(), in_region.x2());
  }

  void Constant_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, float value) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        *out_ptr++ = value;
      }
    }
  }

  void RepeatLeft_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      const float* border_ptr = in_image.GetPtr(in_image.GetBounds().x1, y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        *out_ptr++ = *border_ptr;
      }
    }
  }
  void RepeatRight_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image) {
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      const float* border_ptr = in_image.GetPtr(in_image.GetBounds().x2, y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        *out_ptr++ = *border_ptr;
      }
    }
  }
  void RepeatBottom_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image) {
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float* out_ptr = out_image->GetPtr(x, region.y1());
      const float* border_ptr = in_image.GetPtr(x, in_image.GetBounds().y1);
      for (int y = region.y1(); y <= region.y2(); ++y) {
        *out_ptr = *border_ptr;
        out_ptr = out_image->GetNextRow(out_ptr);
      }
    }
  }
  void RepeatTop_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image) {
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float* out_ptr = out_image->GetPtr(x, region.y1());
      const float* border_ptr = in_image.GetPtr(x, in_image.GetBounds().y2);
      for (int y = region.y1(); y <= region.y2(); ++y) {
        *out_ptr = *border_ptr;
        out_ptr = out_image->GetNextRow(out_ptr);
      }
    }
  }

  void RepeatFalloffLeft_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, float falloff) {
    afx::Bounds in_region = in_image.GetBounds();
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      const float* border_ptr = in_image.GetPtr(in_region.x1, y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        float distance = in_region.x1 - x;
        float distance_falloff = expf(-distance * falloff);
        if(y > in_region.y2()) {
          float y_distance = y - in_region.y2();
          distance_falloff *= expf(-y_distance * falloff);
        }
        if(y < in_region.y1()) {
          float y_distance = in_region.y1() - y;
          distance_falloff *= expf(-y_distance * falloff);
        }
        *out_ptr++ = *border_ptr * distance_falloff;
      }
    }
  }
  void RepeatFalloffRight_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, float falloff) {
    afx::Bounds in_region = in_image.GetBounds();
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      const float* border_ptr = in_image.GetPtr(in_image.GetBounds().x2, y);
      for (int x = region.x1(); x <= region.x2(); ++x) {
        float distance = x - in_region.x2();
        float distance_falloff = expf(-distance * falloff);
        if(y > in_region.y2()) {
          float y_distance = y - in_region.y2();
          distance_falloff *= expf(-y_distance * falloff);
        }
        if(y < in_region.y1()) {
          float y_distance = in_region.y1() - y;
          distance_falloff *= expf(-y_distance * falloff);
        }
        *out_ptr++ = *border_ptr * distance_falloff;
      }
    }
  }
  void RepeatFalloffBottom_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, float falloff) {
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float* out_ptr = out_image->GetPtr(x, region.y1());
      const float* border_ptr = in_image.GetPtr(x, in_image.GetBounds().y1);
      for (int y = region.y1(); y <= region.y2(); ++y) {
        float distance = in_image.GetBounds().y1 - y;
        *out_ptr++ = *border_ptr * expf(-distance * falloff);
        out_ptr = out_image->GetNextRow(out_ptr);
      }
    }
  }
  void RepeatFalloffTop_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image, float falloff) {
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float* out_ptr = out_image->GetPtr(x, region.y1());
      const float* border_ptr = in_image.GetPtr(x, in_image.GetBounds().y2);
      for (int y = region.y1(); y <= region.y2(); ++y) {
        float distance = y - in_image.GetBounds().y2;
        *out_ptr++ = *border_ptr * expf(-distance * falloff);
        out_ptr = out_image->GetNextRow(out_ptr);
      }
    }
  }

  void MirrorLeft_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image) {
    afx::Bounds in_region = in_image.GetBounds();
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      if(y > in_region.y2()) {
        for (int x = region.x1(); x <= region.x2(); ++x) {
          *out_ptr++ = in_image.GetPtrBnds(2 * in_region.x1() - x - 1, 2 * in_region.y2() - y + 1);
        }
      } else if (y < in_region.y1()) {
        for (int x = region.x1(); x <= region.x2(); ++x) {
          *out_ptr++ = in_image.GetPtrBnds(2 * in_region.x1() - x - 1, 2 * in_region.y1() - y - 1);
        }
      } else {
        for (int x = region.x1(); x <= region.x2(); ++x) {
          *out_ptr++ = in_image.GetPtrBnds(2 * in_region.x1() - x - 1, y);
        }
      }
    }
  }
  void MirrorRight_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image) {
    afx::Bounds in_region = in_image.GetBounds();
    for (int y = region.y1(); y <= region.y2(); ++y) {
      float* out_ptr = out_image->GetPtr(region.x1(), y);
      if(y > in_region.y2()) {
        for (int x = region.x1(); x <= region.x2(); ++x) {
          *out_ptr++ = in_image.GetPtrBnds(2 * in_region.x1() - x + 1, 2 * in_region.y2() - y + 1);
        }
      } else if (y < in_region.y1()) {
        for (int x = region.x1(); x <= region.x2(); ++x) {
          *out_ptr++ = in_image.GetPtrBnds(2 * in_region.x1() - x + 1, 2 * in_region.y1() - y - 1);
        }
      } else {
        for (int x = region.x1(); x <= region.x2(); ++x) {
          *out_ptr++ = in_image.GetPtrBnds(2 * in_region.x1() - x + 1, y);
        }
      }
    }
  }
  void MirrorBottom_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image) {
    afx::Bounds in_region = in_image.GetBounds();
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float* out_ptr = out_image->GetPtr(x, region.y1());
      const float* border_ptr = in_image.GetPtr(x, in_image.GetBounds().y1);
      for (int y = region.y1(); y <= region.y2(); ++y) {
        *out_ptr = in_image.GetPtrBnds(y, 2 * in_region.y1() - x - 1);
        out_ptr = out_image->GetNextRow(out_ptr);
      }
    }
  }
  void MirrorTop_(const afx::Bounds& region, const afx::Image& in_image, afx::Image* out_image) {
    afx::Bounds in_region = in_image.GetBounds();
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float* out_ptr = out_image->GetPtr(x, region.y1());
      const float* border_ptr = in_image.GetPtr(x, in_image.GetBounds().y2);
      for (int y = region.y1(); y <= region.y2(); ++y) {
        *out_ptr = in_image.GetPtrBnds(y, 2 * in_region.y2() - x + 1);
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
  void Compliment(afx::Image* image) {
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
