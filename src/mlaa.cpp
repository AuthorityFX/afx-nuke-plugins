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

namespace afx {

ImageInfo::ImageInfo() : ptr_(nullptr), pitch_(0), region_(Bounds()) {}
ImageInfo::ImageInfo(const Bounds& region) : ptr_(nullptr) { Create(region); }
ImageInfo::ImageInfo(unsigned int width, unsigned int height) : ptr_(nullptr) { Create(Bounds(0, 0, width - 1, height - 1)); }
ImageInfo::~ImageInfo() { Dispose(); }
void ImageInfo::Create(const Bounds& region) {
  Dispose();
  region_ = region;
  pitch_ = sizeof(PixelInfo) * region_.GetWidth();
  ptr_ = (PixelInfo*)malloc(region_.GetWidth() * region_.GetHeight() * sizeof(PixelInfo));
}
void ImageInfo::Dispose() {
  if (ptr_ != nullptr) {
    free(ptr_);
    ptr_ = nullptr;
  }
}
PixelInfo& ImageInfo::GetPixel(unsigned int x, unsigned int y) const {
  return *(PixelInfo*)((char*)ptr_ + (y - region_.y1()) * pitch_ + (x - region_.x1()) * sizeof(PixelInfo));
}
PixelInfo* ImageInfo::GetPtr(unsigned int x, unsigned int y) const {
  return (PixelInfo*)((char*)ptr_ + (y - region_.y1()) * pitch_ + (x - region_.x1()) * sizeof(PixelInfo));
}
size_t ImageInfo::GetPitch() const { return pitch_; }
Bounds ImageInfo::GetBounds() const { return region_; }


void MorphAA::MarkDisc_(const Bounds& region, const Image& input) {
  for (int y = region.y1(); y <= region.y2(); ++y) {
    const float* in_ptr = input.GetPtr(region.x1(), y);
    const float* in_bot_ptr = input.GetPtr(region.x1(), input.GetBounds().ClampY(y - 1));
    PixelInfo* info_ptr = info_.GetPtr(region.x1(), y);
    for (int x = region.x1(); x <= region.x2(); ++x) {
      float dif = Diff_(*in_ptr, *in_bot_ptr); // Compare current pixel to bottom neighbour
      if (fabsf(dif) >= threshold_) {
        info_ptr->dis_h = true;
        if (dif >= 0) {
          info_ptr->dis_dir_h = true;
        } else {
          info_ptr->dis_dir_h = false;
        }
      }
      if (x < input.GetBounds().x2()) {
        float dif = Diff_(*in_ptr, *(in_ptr + 1)); // Compare current pixel to right neighbour
        if (fabsf(dif) >= threshold_) {
          info_ptr->dis_v = true;
          if (dif >= 0) {
            info_ptr->dis_dir_v = true;
          } else {
            info_ptr->dis_dir_v = false;
          }
        }
      }
      info_ptr++;
      in_ptr++;
      in_bot_ptr++;
    }
  }
}
// Find horizontal lines
void MorphAA::FindHorLines_(const Bounds& region, const Image& input) {
  PixelInfo* info_ptr = nullptr;
  for (int y = region.y1(); y <= region.y2(); ++y) {
    info_ptr = info_.GetPtr(region.x1(), y);
    int length = 0;  // Length of line, also used to determine if looking for new line or end of line.
    bool direction;  // Direction of discontinuity
    for (int x = region.x1(); x <= region.x2(); ++x) {
      if (length == 0) { // if new line found
        if(info_ptr->dis_h == true) { // if new line found
          length++;
          direction = info_ptr->dis_dir_h;
        }
      } else { // looking for end of line
        if(info_ptr->dis_h == true && x < info_.GetBounds().x2() && direction == info_ptr->dis_dir_h) {
          length++;
        } else { // End of line, loop back through and set info for all pixels in line

          bool start_orientation, end_orientation;
          // 'end_orientation = true' means the end of the line is no blend, ie. CalcBlendingWeights returns 0 for pos = length - 1
          // [] is a pixel in the line
          // {} is the current pixel x, y
          // _____    1.0f    ______
          //      |[][][][][]|{}
          //      ¯¯¯¯¯¯¯¯¯¯¯¯
          //          0.0f
          if ((info_ptr - 1)->dis_dir_h == true) { // last pixel in line is greater than bottom pixel
            if (fabs(Diff_(input.GetValBnds(x - 1, y - 1), input.GetValBnds(x, y - 1))) > threshold_) {
              // if Diff(a, b) > 0.5
              // _____
              //      |[][][][][]|{}
              //      ¯¯¯¯¯¯¯¯¯¯a|_b_____
              end_orientation = true;
            } else { // Since we know the pixel above "b" has no dis_h, end is by default orientation false
              // _____           ______
              //      |[][][][][]|{}
              //      ¯¯¯¯¯¯¯¯¯¯¯¯
              end_orientation = false;
            }
          //          0.0f
          //       [][][][][]{}
          // _____|¯¯¯¯¯¯¯¯¯¯|______
          //          1.0f
          } else { // last pixel in line is less than bottom pixel
            if (fabs(Diff_(input.GetVal(x - 1, y), input.GetVal(x, y))) > threshold_) {
              // if Diff(a, b) > 0.5
              //                  _______
              //       [][][][][a]|{b}
              // _____|¯¯¯¯¯¯¯¯¯¯¯
              end_orientation = true;
            } else {
              //       [][][][][]{}
              // _____|¯¯¯¯¯¯¯¯¯¯|_____
              end_orientation = false;
            }
          }

          // 'start_orientation = true' means the start of the line is full blend, ie. CalcBlendingWeights returns 1 for pos = 0;
          // [] is a pixel in the line
          // {} is the current pixel x, y
          // _____    1.0f    ______
          //      |[][][][][]|{}
          //      ¯¯¯¯¯¯¯¯¯¯¯¯
          //          0.0f
          if ((info_ptr - 1)->dis_dir_h == true) { // last pixel in line is greater than bottom pixel
            if (fabs(Diff_(input.GetValBnds(x - length - 1, y), input.GetVal(x - length, y))) > threshold_) {
              // if Diff(a, b) > 0.5
              // _____            ______
              //     a|[b][][][][]|{}
              //      ¯¯¯¯¯¯¯¯¯¯¯¯
              start_orientation = true;
            } else { // Since we know "a" has no dis_h, start is by default orientation false
              //                 ______
              //       [][][][][]|{}
              // _____|¯¯¯¯¯¯¯¯¯¯
              start_orientation = false;
            }
            //          0.0f
            //       [][][][][]{}
            // _____|¯¯¯¯¯¯¯¯¯¯|______
            //          1.0f
          } else { // last pixel in line is less than bottom pixel
            if (fabs(Diff_(input.GetValBnds(x - length - 1, y - 1), input.GetValBnds(x - length, y - 1))) > threshold_) {
              // if Diff(a, b) > 0.5
              //
              //       [][][][][]{}
              // ____a|b¯¯¯¯¯¯¯¯¯|_____
              start_orientation = true;
            } else {
              // _____
              //      |[][][][][]{}
              //       ¯¯¯¯¯¯¯¯¯¯|_____
              start_orientation = false;
            }
          }
          // Loop back through line and pos and length based on end orientations
          PixelInfo* l_i = info_ptr;
          if (start_orientation == true && end_orientation == false) { // Both ends blend
            float half_length = length / 2.0f + 0.5;
            int max_pos = (int)floor((length - 1) / 2.0f);
            int pos = max_pos;
            for ( ; pos >= 0 ; --pos) {
              l_i--;
              l_i->pos_h = pos;
              l_i->length_h = half_length;
            }
            if (length % 2) { // Is odd
              pos = 1;
            } else { // Is even
              pos = 0;
            }
            for ( ; pos <= max_pos; ++pos) {
              l_i--;
              l_i->pos_h = pos;
              l_i->length_h = half_length;
            }
          } else if (start_orientation == true && end_orientation == true) { // Start blends, end does NOT blend
            for (int pos = 0; pos <= length - 1; ++pos) {
              l_i--;
              l_i->pos_h = pos;
              l_i->length_h = length;
            }
          } else if (start_orientation == false && end_orientation == false) { // Start does NOT blend, end does blend
            for (int pos = length - 1; pos >= 0; --pos) {
              l_i--;
              l_i->pos_h = pos;
              l_i->length_h = length;
            }
          }
          // Search for new line form this pixel. Decrement all counters.
          x--;
          info_ptr--;
          length = 0;
        }
      }
      info_ptr++;
    }
  }
}
// Find vertical lines
void MorphAA::FindVertLines_(const Bounds& region, const Image& input) {
  PixelInfo* info_ptr = nullptr;
  for (int x = region.x1(); x <= region.x2(); ++x) {
    info_ptr = info_.GetPtr(x, region.y1());
    int length = 0; // Length of line, also used to determine if looking for new line or end of line.
    bool direction; // Directoin of discontinuity
    for (int y = region.y1(); y <= region.y2(); ++y) {
      if (length == 0) { // if loooking for new line
        if(info_ptr->dis_v == true) { // if new line found
          length++;
          direction = info_ptr->dis_dir_v;
        }
      }
      else { // Looking for end of line
        if(info_ptr->dis_v == true && y < info_.GetBounds().y2() && direction == info_ptr->dis_dir_v) { // Line continues
          length++;
        } else { // End of line

          bool start_orientation, end_orientation;

          // 'end_orientation = true' means the end of the line is no blend, ie. CalcBlendingWeights returns 0 for pos = length - 1
          // [] is a pixel in the line
          // {} is the current pixel x, y
          // _____    1.0f    ______
          //      |[][][][][]|{}
          //      ¯¯¯¯¯¯¯¯¯¯¯¯
          //          0.0f
          if (((PixelInfo*)((char*)info_ptr - info_.GetPitch()))->dis_dir_v == true) { // last pixel in line is greater than right pixel
            if (fabs(Diff_(input.GetValBnds(x + 1, y - 1), input.GetValBnds(x + 1, y))) > threshold_) {
              // if Diff(a, b) > 0.5
              // _____
              //      |[][][][][]|{}
              //      ¯¯¯¯¯¯¯¯¯¯a|_b_____
              end_orientation = true;
            } else { // Since we know the pixel above "b" has no dis_h, end is by default orientation false
              // _____           ______
              //      |[][][][][]|{}
              //      ¯¯¯¯¯¯¯¯¯¯¯¯
              end_orientation = false;
            }
            //          0.0f
            //       [][][][][]{}
            // _____|¯¯¯¯¯¯¯¯¯¯|______
            //          1.0f
          } else { // last pixel in line is less than bottom pixel
            if (fabs(Diff_(input.GetVal(x, y - 1), input.GetVal(x, y))) > threshold_) {
              // if Diff(a, b) > 0.5
              //                  _______
              //       [][][][][a]|{b}
              // _____|¯¯¯¯¯¯¯¯¯¯¯
              end_orientation = true;
            } else {
              //       [][][][][]{}
              // _____|¯¯¯¯¯¯¯¯¯¯|_____
              end_orientation = false;
            }
          }

          // 'start_orientation = true' means the start of the line is full blend, ie. CalcBlendingWeights returns 1 for pos = 0;
          // [] is a pixel in the line
          // {} is the current pixel x, y
          // _____    1.0f    ______
          //      |[][][][][]|{}
          //      ¯¯¯¯¯¯¯¯¯¯¯¯
          //          0.0f
          if (((PixelInfo*)((char*)info_ptr - info_.GetPitch()))->dis_dir_v == true) { // last pixel in line is greater than bottom pixel
            if (fabs(Diff_(input.GetValBnds(x, y - length - 1), input.GetVal(x, y - length))) > threshold_) {
              // if Diff(a, b) > 0.5
              // _____            ______
              //     a|[b][][][][]|{}
              //      ¯¯¯¯¯¯¯¯¯¯¯¯
              start_orientation = true;
            } else { // Since we know "a" has no dis_h, start is by default orientation false
              //                 ______
              //       [][][][][]|{}
              // _____|¯¯¯¯¯¯¯¯¯¯
              start_orientation = false;
            }
            //          0.0f
            //       [][][][][]{}
            // _____|¯¯¯¯¯¯¯¯¯¯|______
            //          1.0f
          } else { // last pixel in line is less than bottom pixel
            if (fabs(Diff_(input.GetValBnds(x + 1, y - length - 1), input.GetValBnds(x + 1, y - length))) > threshold_) {
              // if Diff(a, b) > 0.5
              //
              //       [][][][][]{}
              // ____a|b¯¯¯¯¯¯¯¯¯|_____
              start_orientation = true;
            } else {
              // _____
              //      |[][][][][]{}
              //       ¯¯¯¯¯¯¯¯¯¯|_____
              start_orientation = false;
            }
          }
          // Loop back through line and pos and length based on end orientations
          PixelInfo* l_i = info_ptr;
          if (start_orientation == true && end_orientation == false) { // Both ends blend
            float half_length = length / 2.0f + 0.5;
            int max_pos = (int)floor((length - 1) / 2.0f);
            int pos = max_pos;
            for ( ; pos >= 0 ; --pos) {
              l_i = (PixelInfo*)((char*)l_i - info_.GetPitch());
              l_i->pos_v = pos;
              l_i->length_v = half_length;
            }
            if (length % 2) { // Is odd
              pos = 1;
            } else { // Is even
              pos = 0;
            }
            for ( ; pos <= max_pos; ++pos) {
              l_i = (PixelInfo*)((char*)l_i - info_.GetPitch());
              l_i->pos_v = pos;
              l_i->length_v = half_length;
            }
          } else if (start_orientation == true && end_orientation == true) { // Start blends, end does NOT blend
            for (int pos = 0; pos <= length - 1; ++pos) {
              l_i = (PixelInfo*)((char*)l_i - info_.GetPitch());
              l_i->pos_v = pos;
              l_i->length_v = length;
            }
          } else if (start_orientation == false && end_orientation == false) { // Start does NOT blend, end does blend
            for (int pos = length - 1; pos >= 0; --pos) {
              l_i = (PixelInfo*)((char*)l_i - info_.GetPitch());
              l_i->pos_v = pos;
              l_i->length_v = length;
            }
          }
          // Search for new line form this pixel. Decrement all counters.
          y--;
          info_ptr = (PixelInfo*)((char*)info_ptr - info_.GetPitch());
          length = 0;
        }
      }
      info_ptr = (PixelInfo*)((char*)info_ptr + info_.GetPitch());
    }
  }
}
// Blend pixels
void MorphAA::BlendPixels_(const Bounds& region, const Image& input, Image& output) {
  const float* in_ptr = nullptr;
  float* out_ptr = nullptr;
  PixelInfo* info_ptr = nullptr;
  for (int y = region.y1(); y <= region.y2(); ++y) {
    in_ptr = input.GetPtr(region.x1(), y);
    out_ptr = output.GetPtr(region.x1(), y);
    info_ptr = info_.GetPtr(region.x1(), y);
    for (int x = region.x1(); x <= region.x2(); ++x) {
      PixelInfo zero_pixel;
      const PixelInfo* t_info_ptr = &zero_pixel;
      const PixelInfo* l_info_ptr = &zero_pixel;
      if (y < info_.GetBounds().y2()) { t_info_ptr = (PixelInfo*)((char*)info_ptr + info_.GetPitch()); }
      if (x > info_.GetBounds().x1()) { l_info_ptr = (info_ptr - 1); }

      const float* l_ptr = input.GetPtr(input.GetBounds().ClampX(x - 1), y);
      const float* r_ptr = input.GetPtr(input.GetBounds().ClampX(x + 1), y);
      const float* b_ptr = input.GetPtr(x, input.GetBounds().ClampY(y - 1));
      const float* t_ptr = input.GetPtr(x, input.GetBounds().ClampY(y + 1));

      *out_ptr = *in_ptr;

      if (info_ptr->dis_dir_h && info_ptr->length_h >= 1) {
        float blend = CalcTrapArea_(info_ptr->pos_h, info_ptr->length_h);
        *out_ptr = blend * *b_ptr + (1.0f - blend) * *out_ptr;
      } else if(!t_info_ptr->dis_dir_h && t_info_ptr->length_h >= 1) {
        float blend = CalcTrapArea_(t_info_ptr->pos_h, t_info_ptr->length_h);
        *out_ptr = blend * *t_ptr + (1.0f - blend) * *out_ptr;
      }

      if (info_ptr->dis_dir_v && info_ptr->length_v >= 1) {
        float blend = CalcTrapArea_(info_ptr->pos_v, info_ptr->length_v);
        *out_ptr = blend * *r_ptr + (1.0f - blend) * *out_ptr;
      } else if(!l_info_ptr->dis_dir_v && l_info_ptr->length_v >= 1) {
        float blend = CalcTrapArea_(l_info_ptr->pos_v, l_info_ptr->length_v);
        *out_ptr = blend * *l_ptr + (1.0f - blend) * *out_ptr;
      }

      in_ptr++;
      out_ptr++;
      info_ptr++;
    }
  }
}

void MorphAA::Process(const Image& input, Image& output, float threshold, afx::Threader& threader) {
  threshold_ = threshold;
  info_.Create(input.GetBounds());

  threader.ThreadImageChunks(info_.GetBounds(), boost::bind(&MorphAA::MarkDisc_, this, _1, boost::cref(input)));
  threader.Synchonize();
  threader.ThreadImageChunks(info_.GetBounds(), boost::bind(&MorphAA::FindHorLines_, this, _1, boost::cref(input)));
  threader.Synchonize();
  threader.ThreadImageChunks(info_.GetBounds(), boost::bind(&MorphAA::FindVertLines_, this, _1, boost::cref(input)));
  threader.Synchonize();
  threader.ThreadImageChunks(info_.GetBounds(), boost::bind(&MorphAA::BlendPixels_, this, _1, boost::cref(input), boost::ref(output)));
  threader.Synchonize();
}

void MorphAA::Process(const Image& input, Image& output, float threshold) {
  threshold_ = threshold;
  info_.Create(input.GetBounds());

  MarkDisc_(info_.GetBounds(), input);
  FindHorLines_(info_.GetBounds(), input);
  FindVertLines_(info_.GetBounds(), input);
  BlendPixels_(info_.GetBounds(), input, output);
}


} // namespace afx
