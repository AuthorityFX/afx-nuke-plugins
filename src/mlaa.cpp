// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#include "mlaa.h"

#include <boost/bind.hpp>
#include <math.h>

#include "settings.h"
#include "image.h"
#include "threading.h"
#include "memory.h"

namespace afx {

ImageInfo::ImageInfo() : ptr_(nullptr), pitch_(0), region_(Bounds()) {}
ImageInfo::ImageInfo(const Bounds& region) : ptr_(nullptr) { Create(region); }
ImageInfo::ImageInfo(unsigned int width, unsigned int height) : ptr_(nullptr) { Create(Bounds(0, 0, width - 1, height - 1)); }
ImageInfo::~ImageInfo() { Dispose(); }
void ImageInfo::Create(const Bounds& region) {
  Dispose();
  region_ = region;
  ptr_ = afx::ImageMalloc<PixelInfo>(region_.GetWidth(), region.GetHeight(), &pitch_);
}
void ImageInfo::Dispose() {
  if (ptr_ != nullptr) {
    afx::ImageFree(ptr_);
    ptr_ = nullptr;
  }
}
PixelInfo& ImageInfo::GetVal(unsigned int x, unsigned int y) const {
  return *(PixelInfo*)((char*)ptr_ + (y - region_.y1()) * pitch_ + (x - region_.x1()) * sizeof(PixelInfo));
}
PixelInfo& ImageInfo::GetValBnds(unsigned int x, unsigned int y) const {
  return *(PixelInfo*)((char*)ptr_ + (region_.ClampY(y) - region_.y1()) * pitch_ + (region_.ClampX(x) - region_.x1()) * sizeof(PixelInfo));
}
PixelInfo* ImageInfo::GetPtr(unsigned int x, unsigned int y) const {
  return (PixelInfo*)((char*)ptr_ + (y - region_.y1()) * pitch_ + (x - region_.x1()) * sizeof(PixelInfo));
}
PixelInfo* ImageInfo::GetPtrBnds(unsigned int x, unsigned int y) const {
  return (PixelInfo*)((char*)ptr_ + (region_.ClampY(y) - region_.y1()) * pitch_ + (region_.ClampX(x) - region_.x1()) * sizeof(PixelInfo));
}
PixelInfo* ImageInfo::NextRow(PixelInfo* ptr) {
  return (PixelInfo*)((char*)ptr + pitch_);
}
PixelInfo* ImageInfo::PreviousRow(PixelInfo* ptr) {
  return (PixelInfo*)((char*)ptr - pitch_);
}
size_t ImageInfo::GetPitch() const { return pitch_; }
Bounds ImageInfo::GetBounds() const { return region_; }

float MorphAA::CalcTrapArea_(int pos, float length) {
  //Will never be called when length is 0
  return (float)pos / length + (0.5f / length);
}
float MorphAA::Diff_(float a, float b) {
  return (a - b) / (a + b);
}
void MorphAA::MarkDisc_(const Bounds& region, const Image& input) {
  for (int y = region.y1(); y <= region.y2(); ++y) {
    const float* in_ptr = input.GetPtr(region.x1(), y);
    const float* in_bot_ptr = input.GetPtrBnds(region.x1(), y - 1);
    const float* in_right_ptr = input.GetPtrBnds(region.x1() + 1, y);
    PixelInfo* info_ptr = info_.GetPtr(region.x1(), y);
    for (int x = region.x1(); x <= region.x2(); ++x) {
      *info_ptr = PixelInfo(); // Initialize PixelInfo
      if (y > info_.GetBounds().y1()) {
        float dif = Diff_(*in_ptr, *in_bot_ptr); // Compare current pixel to bottom neighbour
        if (fabsf(dif) >= threshold_) {
          if (dif > 0) {
            info_ptr->flags |= afx::kDisPosDown;
          } else {
            info_ptr->flags |= afx::kDisNegDown;
          }
        }
        in_bot_ptr++;
      }
      if (x < info_.GetBounds().x2()) {
        float dif = Diff_(*in_ptr, *in_right_ptr); // Compare current pixel to right neighbour
        if (fabsf(dif) >= threshold_) {
          if (dif > 0) {
            info_ptr->flags |= afx::kDisPosRight;
          } else {
            info_ptr->flags |= afx::kDisNegRight;
          }
        }
        in_right_ptr++;
      }
      info_ptr++;
      in_ptr++;
    }
  }
}
// Find X lines
void MorphAA::FindXLines_(const Bounds& region, const Image& input) {
  PixelInfo* info_ptr = nullptr;
  for (int y = region.y1(); y <= region.y2(); ++y) {
    info_ptr = info_.GetPtr(region.x1(), y);
    int length = 0;  // Length of line, also used to determine if looking for new line or end of line.
    unsigned char start_flag_x;
    for (int x = region.x1(); x <= region.x2(); ++x) {
      switch (length) { // if new line found
        case 0: {
          start_flag_x = info_ptr->flags & (afx::kDisPosDown | afx::kDisNegDown);
          if(start_flag_x) { length++; } // New line found
          break;
        }
        default: { // looking for end of line
          if((info_ptr->flags & start_flag_x) and x < info_.GetBounds().x2()) {
            length++;
          } else { // End of line, loop back through and set info for all pixels in line
            SetXLine_(info_ptr, length, x, y);
            x--; // Search for new line form this pixel. Decrement all counters.
            info_ptr--;
            length = 0;
          }
          break;
        }
      }
      info_ptr++;
    }
  }
}
// Find Y lines
void MorphAA::FindYLines_(const Bounds& region, const Image& input) {
  PixelInfo* info_ptr = nullptr;
  for (int x = region.x1(); x <= region.x2(); ++x) {
    info_ptr = info_.GetPtr(x, region.y1());
    int length = 0; // Length of line, also used to determine if looking for new line or end of line.
    unsigned char start_flag_y;
    for (int y = region.y1(); y <= region.y2(); ++y) {
      switch (length) { // if new line found
        case 0: {
          start_flag_y = info_ptr->flags & (afx::kDisPosRight | afx::kDisNegRight);
          if(start_flag_y) { length++; } // New line found
          break;
        }
        default: { // Looking for end of line
          if((info_ptr->flags & start_flag_y) and y < info_.GetBounds().y2()) { // Line continues
            length++;
          } else { // End of line
            SetYLine_(info_ptr, length, x, y);
            y--; // Search for new line form this pixel. Decrement all counters.
            info_ptr = (PixelInfo*)((char*)info_ptr - info_.GetPitch());
            length = 0;
          }
          break;
        }
      }
      info_ptr = (PixelInfo*)((char*)info_ptr + info_.GetPitch());
    }
  }
}
// Set X Line
void MorphAA::SetXLine_(PixelInfo* info_ptr, int length, int x, int y) {
  bool start_blend, end_blend;
  // [] is a pixel in the line
  // {} is the current pixel x, y    y
  // _____    1.0f    ______         |
  //      |[][][][][]|{}             |
  //      ¯¯¯¯¯¯¯¯¯¯¯¯               |______x
  //          0.0f
  if ((info_ptr - 1)->flags & afx::kDisPosDown) { // last pixel in line is greater than bottom pixel
    if (info_.GetPtrBnds(x - 1, y - 1)->flags & afx::kDisNegRight or not (info_ptr - 1)->flags & (afx::kDisPosRight | afx::kDisNegRight)) {
      // if Diff(a, b) > thresh
      // _____
      //      |[][][][][]|{}
      //      ¯¯¯¯¯¯¯¯¯¯a|b_____
      //
      end_blend = false;
    } else { // Since we know the pixel above "b" has no dis_h, end is by default orientation false
      // _____           ______
      //      |[][][][][]|{}
      //      ¯¯¯¯¯¯¯¯¯¯¯¯
      end_blend = true;
    }
    if (info_.GetPtrBnds(x - length - 1, y)->flags & afx::kDisNegRight) {
      // if Diff(a, b) > thresh
      // _____            ______
      //     a|[b][][][][]|{}
      //      ¯¯¯¯¯¯¯¯¯¯¯¯
      start_blend = true;
    } else { // Since we know "a" has no dis_h, start is by default orientation false
      //                 ______
      //       [][][][][]|{}
      // _____|¯¯¯¯¯¯¯¯¯¯
      start_blend = false;
    }
    //          0.0f
    //       [][][][][]{}
    // _____|¯¯¯¯¯¯¯¯¯¯|______
    //          1.0f
  } else { // last pixel in line is less than bottom pixel
    if ((info_ptr - 1)->flags & afx::kDisNegRight or info_.GetPtrBnds(x, y + 1)->flags & afx::kDisNegDown) {
      // if Diff(a, b) > thresh or Diff(c, b) > thresh
      //                  __c_____
      //       [][][][][a]|{b}
      // _____|¯¯¯¯¯¯¯¯¯¯¯
      end_blend = false;
    } else {
      //       [][][][][]{}
      // _____|¯¯¯¯¯¯¯¯¯¯|_____
      end_blend = true;
    }
    if (info_.GetPtrBnds(x - length - 1, y - 1)->flags & afx::kDisNegRight) {
      // if Diff(a, b) > thresh
      //
      //       [][][][][]{}
      // ____a|b¯¯¯¯¯¯¯¯¯|_____
      start_blend = true;
    } else {
      // _____
      //      |[][][][][]{}
      //       ¯¯¯¯¯¯¯¯¯¯|_____
      start_blend = false;
    }
  }
  // Loop back through line and pos and length based on end orientations
  PixelInfo* l_i = info_ptr;
  if (start_blend and end_blend) { // Both ends blend
    float half_length = length / 2.0f + 0.5;
    int max_pos = (int)floor((length - 1) / 2.0f);
    int pos = max_pos;
    for ( ; pos >= 0 ; --pos) {
      l_i--;
      l_i->blend_x = CalcTrapArea_(pos, half_length);
    }
    if (length & 1) { // Is odd
      pos = 1;
    } else { // Is even
      pos = 0;
    }
    // TODO wasting CPU time by recalculating the same trap area as above
    for ( ; pos <= max_pos; ++pos) {
      l_i--;
      l_i->blend_x = CalcTrapArea_(pos, half_length);
    }
  } else if (start_blend and not end_blend) { // Start blends, end does NOT blend
    for (int pos = 0; pos <= length - 1; ++pos) {
      l_i--;
      l_i->blend_x = CalcTrapArea_(pos, length);
    }
  } else if (not start_blend and end_blend) { // Start does NOT blend, end does blend
    for (int pos = length - 1; pos >= 0; --pos) {
      l_i--;
      l_i->blend_x = CalcTrapArea_(pos, length);
    }
  }
}
// Set Y Line
void MorphAA::SetYLine_(PixelInfo* info_ptr, int length, int x, int y) {
  bool start_blend, end_blend;
  // [] is a pixel in the line
  // {} is the current pixel x, y     _____y
  // _____    1.0f    ______         |
  //      |[][][][][]|{}             |
  //      ¯¯¯¯¯¯¯¯¯¯¯¯               x
  //          0.0f
  if (info_.PreviousRow(info_ptr)->flags & afx::kDisPosRight) { // last pixel in line is greater than right pixel
    if (info_.GetPtrBnds(x + 1, y)->flags & afx::kDisPosDown or not info_ptr->flags & (afx::kDisPosDown | afx::kDisNegDown)) {
      // if Diff(a, b) > thresh
      // _____
      //      |[][][][][d]|{c}
      //      ¯¯¯¯¯¯¯¯¯¯b|_a_____
      end_blend = false;
    } else { // Since we know the pixel above "b" has no dis_h, end is by default orientation false
      // _____           ______
      //      |[][][][][]|{}
      //      ¯¯¯¯¯¯¯¯¯¯¯¯
      end_blend = true;
    }
    if (info_.GetPtr(x, y - length)->flags & afx::kDisPosDown) {
      // if Diff(a, b) > thresh
      // _____            ______
      //     b|[a][][][][]|{}
      //      ¯¯¯¯¯¯¯¯¯¯¯¯
      start_blend = true;
    } else { // Since we know "a" has no dis_h, start is by default orientation false
      //                 ______
      //       [][][][][]|{}
      // _____|¯¯¯¯¯¯¯¯¯¯
      start_blend = false;
    }
    //          0.0f
    //       [][][][][]{}
    // _____|¯¯¯¯¯¯¯¯¯¯|______
    //          1.0f
  } else { // last pixel in line is less than right pixel
    if (info_ptr->flags & afx::kDisPosDown) {
      // if Diff(a, b) > thresh
      //                  _______
      //       [][][][][b]|{a}
      // _____|¯¯¯¯¯¯¯¯¯¯¯
      end_blend = false;
    } else {
      //       [][][][][]{}
      // _____|¯¯¯¯¯¯¯¯¯¯|_____
      end_blend = true;
    }
    if (info_.GetPtrBnds(x + 1, y - length)->flags & afx::kDisPosDown) {
      // if Diff(a, b) > thresh
      //
      //       [][][][][]{}
      // ____b|a¯¯¯¯¯¯¯¯¯|_____
      start_blend = true;
    } else {
      // _____
      //      |[][][][][]{}
      //       ¯¯¯¯¯¯¯¯¯¯|_____
      start_blend = false;
    }
  }
  // Loop back through line and pos and length based on end orientations
  PixelInfo* l_i = info_ptr;
  if (start_blend and end_blend) { // Both ends blend
    float half_length = length / 2.0f + 0.5;
    int max_pos = (int)floor((length - 1) / 2.0f);
    int pos = max_pos;
    for ( ; pos >= 0 ; --pos) {
      l_i = info_.PreviousRow(l_i);
      l_i->blend_y = CalcTrapArea_(pos, half_length);
    }
    if (length & 1) { // Is odd
      pos = 1;
    } else { // Is even
      pos = 0;
    }
    for ( ; pos <= max_pos; ++pos) {
      l_i = info_.PreviousRow(l_i);
      l_i->blend_y = CalcTrapArea_(pos, half_length);
    }
  } else if (start_blend and not end_blend) { // Start blends, end does NOT blend
    for (int pos = 0; pos <= length - 1; ++pos) {
      l_i = info_.PreviousRow(l_i);
      l_i->blend_y = CalcTrapArea_(pos, length);
    }
  } else if (not start_blend and end_blend) { // Start does NOT blend, end does blend
    for (int pos = length - 1; pos >= 0; --pos) {
      l_i = info_.PreviousRow(l_i);
      l_i->blend_y = CalcTrapArea_(pos, length);
    }
  }
}
// Blend pixels
void MorphAA::BlendPixels_(const Bounds& region, const Image& input, Image& output) {
  const float* in_ptr = nullptr;
  float* out_ptr = nullptr;
  PixelInfo* info_ptr = nullptr;
  const PixelInfo* t_info_ptr = nullptr;
  const PixelInfo* l_info_ptr = nullptr;
  for (int y = region.y1(); y <= region.y2(); ++y) {
    in_ptr = input.GetPtr(region.x1(), y);
    out_ptr = output.GetPtr(region.x1(), y);
    info_ptr = info_.GetPtr(region.x1(), y);
    t_info_ptr = info_.GetPtrBnds(region.x1(), y + 1);
    l_info_ptr = info_.GetPtrBnds(region.x1() - 1, y);
    for (int x = region.x1(); x <= region.x2(); ++x) {
      *out_ptr = *in_ptr;

      if (info_ptr->flags & afx::kDisPosDown) {
        const float* b_ptr = y > info_.GetBounds().y1() ? in_ptr - input.GetPitch() : in_ptr;
        *out_ptr = info_ptr->blend_x * *b_ptr +  (1.0f - info_ptr->blend_x) * *out_ptr;
      } else if (t_info_ptr->flags & afx::kDisNegDown) {
        const float* t_ptr = y < info_.GetBounds().y2() ? in_ptr + input.GetPitch() : in_ptr;
        *out_ptr = t_info_ptr->blend_x * *t_ptr + (1.0f - t_info_ptr->blend_x) * *out_ptr;
      }
      if (info_ptr->flags & afx::kDisPosRight) {
        const float* r_ptr = x < info_.GetBounds().x2() ? in_ptr + 1 : in_ptr;
        *out_ptr = info_ptr->blend_y * *r_ptr + (1.0f - info_ptr->blend_y) * *out_ptr;
      } else if(l_info_ptr->flags & afx::kDisNegRight) {
        const float* l_ptr = x > info_.GetBounds().x1() ? in_ptr - 1 : in_ptr;
        *out_ptr = l_info_ptr->blend_y * *l_ptr + (1.0f - l_info_ptr->blend_y) * *out_ptr;
      }

      in_ptr++;
      out_ptr++;
      info_ptr++;
      t_info_ptr++;
      if (x > info_.GetBounds().x1()) { l_info_ptr++; }
    }
  }
}

void MorphAA::Process(const Image& input, Image& output, float threshold, afx::Threader& threader) {
  threshold_ = threshold;
  info_.Create(input.GetBounds());

  threader.ThreadImageChunks(info_.GetBounds(), boost::bind(&MorphAA::MarkDisc_, this, _1, boost::cref(input)));
  threader.Synchonize();
  threader.ThreadImageChunks(info_.GetBounds(), boost::bind(&MorphAA::FindXLines_, this, _1, boost::cref(input)));
  threader.Synchonize();
  threader.ThreadImageChunksY(info_.GetBounds(), boost::bind(&MorphAA::FindYLines_, this, _1, boost::cref(input)));
  threader.Synchonize();
  threader.ThreadImageChunks(info_.GetBounds(), boost::bind(&MorphAA::BlendPixels_, this, _1, boost::cref(input), boost::ref(output)));
  threader.Synchonize();
}

void MorphAA::Process(const Image& input, Image& output, float threshold) {
  threshold_ = threshold;
  info_.Create(input.GetBounds());

  MarkDisc_(info_.GetBounds(), input);
  FindXLines_(info_.GetBounds(), input);
  FindYLines_(info_.GetBounds(), input);
  BlendPixels_(info_.GetBounds(), input, output);
}

} // namespace afx
