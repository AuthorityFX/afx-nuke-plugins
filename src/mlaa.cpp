// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2017, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#include "include/mlaa.h"

#include <math.h>
#include <boost/bind.hpp>

#include <cstdint>

#include "include/image.h"
#include "include/threading.h"

namespace afx {

float MorphAA::BlendWeight_(unsigned int pos, float length) {
  // Will never be called when length is 0
  return static_cast<float>(pos) / length + (0.5f / length);
}

float MorphAA::RelativeDifference_(float a, float b) {
  return (a - b) / ((a + b) / 2.0f);
}

void MorphAA::MarkDisc_(const Bounds& region, const Image& input) {
  for (int y = region.y1(); y <= region.y2(); ++y) {
    const float* in_ptr = input.GetPtr(region.x1(), y);
    const float* in_bot_ptr = input.GetPtrBnds(region.x1(), y - 1);
    const float* in_right_ptr = input.GetPtrBnds(region.x1() + 1, y);
    Discontinuity* disc_ptr = disc_image_.GetPtr(region.x1(), y);
    for (int x = region.x1(); x <= region.x2(); ++x) {
      *disc_ptr = Discontinuity();  // Initialize PixelInfo
      if (y > disc_image_.GetBounds().y1()) {
        float dif = RelativeDifference_(*in_ptr, *in_bot_ptr);
        if (dif > threshold_) {
          disc_ptr->direction |= afx::kDownPositive;
        } else if (dif < -threshold_) {
          disc_ptr->direction |= afx::kDownNegative;
        }
        in_bot_ptr++;
      }
      if (x < disc_image_.GetBounds().x2()) {
        float dif = RelativeDifference_(*in_ptr, *in_right_ptr);
        if (dif > threshold_) {
          disc_ptr->direction |= afx::kRightPositive;
        } else if (dif < -threshold_) {
          disc_ptr->direction |= afx::kRightNegative;
        }
        in_right_ptr++;
      }
      disc_ptr++;
      in_ptr++;
    }
  }
}

void MorphAA::FindRowLines_(const Bounds& region, const Image& input) {
  Discontinuity* disc_ptr = nullptr;
  for (int y = region.y1(); y <= region.y2(); ++y) {
    disc_ptr = disc_image_.GetPtr(region.x1(), y);
    unsigned int length = 0;  // Length of line, also used to determine if looking for new line or end of line.
    std::uint8_t start_disc = 0;
    for (int x = region.x1(); x <= region.x2(); ++x) {
      switch (length) {  // if new line found
        case 0: {
          start_disc = disc_ptr->direction & afx::kDown;
          if (start_disc) { length++; }  // New line found
          break;
        }
        default: {  // looking for end of line
          if ((disc_ptr->direction & start_disc) && x < disc_image_.GetBounds().x2() && length <= max_line_length_) {
            length++;
          } else {  // End of line, loop back through and set blend weights for all pixels in line
            RowBlendWeights_(disc_ptr, length, x, y);
            x--;  // Search for new line form this pixel. Decrement all counters.
            disc_ptr--;
            length = 0;
          }
          break;
        }
      }
      disc_ptr++;
    }
  }
}

void MorphAA::FindColumnLines_(const Bounds& region, const Image& input) {
  Discontinuity* disc_ptr = nullptr;
  for (int x = region.x1(); x <= region.x2(); ++x) {
    disc_ptr = disc_image_.GetPtr(x, region.y1());
    unsigned int length = 0;  // Length of line, also used to determine if looking for new line or end of line.
    std::uint8_t start_disc = 0;
    for (int y = region.y1(); y <= region.y2(); ++y) {
      switch (length) {  // if new line found
        case 0: {
          start_disc = disc_ptr->direction & afx::kRight;
          if (start_disc) { length++; }  // New line found
          break;
        }
        default: {  // Looking for end of line
          if ((disc_ptr->direction & start_disc) && y < disc_image_.GetBounds().y2() && length <= max_line_length_) {  // Line continues
            length++;
          } else {  // End of line, loop back through and set blend weights for all pixels in line
            ColumnBlendWeights_(disc_ptr, length, x, y);
            y--;  // Search for new line form this pixel. Decrement all counters.
            disc_ptr = disc_image_.GetPreviousRow(disc_ptr);
            length = 0;
          }
          break;
        }
      }
      disc_ptr = disc_image_.GetNextRow(disc_ptr);
    }
  }
}

void MorphAA::RowBlendWeights_(Discontinuity* disc_ptr, int length, int x, int y) {
  bool start_blend = false;
  bool end_blend = false;
  // [] is a pixel in the line
  // {} is the current pixel x, y    y
  // _____    1.0f    ______         |
  //      |[][][][][]|{}             |
  //      ¯¯¯¯¯¯¯¯¯¯¯¯               |______x
  //          0.0f
  if ((disc_ptr - 1)->direction & afx::kDownPositive) {  // last pixel in line is greater than bottom pixel
    if ((disc_ptr - 1)->direction & afx::kRightPositive && !(disc_image_.GetPtrBnds(x - 1, y - 1)->direction & kRightNegative)) {
      // _____            ______
      //      |[][][][][a]|{}
      //      ¯¯¯¯¯¯¯¯¯¯b¯¯
      end_blend = true;
    }
    if (disc_image_.GetPtrBnds(x - length - 1, y)->direction & afx::kRightNegative && !(disc_image_.GetPtrBnds(x - length - 1, y - 1)->direction & afx::kRightPositive)) {
      // _____            ______
      //     a|[][][][][]|{}
      //     b¯¯¯¯¯¯¯¯¯¯¯¯
      start_blend = true;
    }
    //          0.0f
    //       [][][][][]{}
    // _____|¯¯¯¯¯¯¯¯¯¯|______
    //          1.0f
  } else {  // last pixel in line is less than bottom pixel
    if (disc_image_.GetPtrBnds(x - 1, y - 1)->direction & afx::kRightPositive && !((disc_ptr - 1)->direction & afx::kRightNegative)) {
      //       [][][][][b]{}
      // _____|¯¯¯¯¯¯¯¯¯a|_____
      end_blend = true;
    }
    if (disc_image_.GetPtrBnds(x - length - 1, y - 1)->direction & afx::kRightNegative && !(disc_image_.GetPtrBnds(x - length - 1, y)->direction & afx::kRightPositive)) {
      // if Diff(a, b) > thresh
      //
      //     b [][][][][]{}
      // ____a|¯¯¯¯¯¯¯¯¯|_____
      start_blend = true;
    }
  }
  // Loop back through line and pos and length based on end orientations
  Discontinuity* line_disc_ptr = disc_ptr;
  if (start_blend && end_blend) {  // Both ends blend
    float half_length = length / 2.0f + 0.5;
    int max_pos = static_cast<int>(half_length) - 1;
    std::vector<float> weights(static_cast<int>(half_length));
    for (int pos = max_pos; pos >= 0; --pos) {
      line_disc_ptr--;
      line_disc_ptr->vert_weight = BlendWeight_(pos, half_length);
      weights[pos] = line_disc_ptr->vert_weight;
    }
    for (auto it = (length & 1) ? (weights.begin() + 1) : weights.begin(); it != weights.end(); ++it) {
      line_disc_ptr--;
      line_disc_ptr->vert_weight = *it;
    }
  } else if (start_blend && !end_blend) {  // Start blends, end does NOT blend
    for (int pos = 0; pos <= length - 1; ++pos) {
      line_disc_ptr--;
      line_disc_ptr->vert_weight = BlendWeight_(pos, length);
    }
  } else if (!start_blend && end_blend) {  // Start does NOT blend, end does blend
    for (int pos = length - 1; pos >= 0; --pos) {
      line_disc_ptr--;
      line_disc_ptr->vert_weight = BlendWeight_(pos, length);
    }
  }
}

void MorphAA::ColumnBlendWeights_(Discontinuity* disc_ptr, int length, int x, int y) {
  bool start_blend = false;
  bool end_blend = false;
  // [] is a pixel in the line
  // {} is the current pixel x, y     _____y
  // _____    1.0f    ______         |
  //      |[][][][][]|{}             |
  //      ¯¯¯¯¯¯¯¯¯¯¯¯               x
  //          0.0f
  if (disc_image_.GetPreviousRow(disc_ptr)->direction & afx::kRightPositive) {  // last pixel in line is greater than right pixel
    if ((disc_ptr->direction & afx::kDownNegative) && !(disc_image_.GetPtrBnds(x + 1, y)->direction & afx::kDownPositive)) {
      // _____           ______
      //      |[][][][][]|{a}
      //      ¯¯¯¯¯¯¯¯¯¯¯¯ b
      end_blend = true;
    }
    if (disc_image_.GetPtrBnds(x, y - length)->direction & afx::kDownPositive && !(disc_image_.GetPtrBnds(x + 1, y - length)->direction & afx::kDownNegative)) {
      // if Diff(a, b) > thresh
      // _____            ______
      //      |[a][][][][]|{}
      //       ¯b¯¯¯¯¯¯¯¯¯¯
      start_blend = true;
    }
    //          0.0f
    //       [][][][][]{}
    // _____|¯¯¯¯¯¯¯¯¯¯|______
    //          1.0f
  } else {  // last pixel in line is less than right pixel
    if ((disc_image_.GetPtrBnds(x + 1, y)->direction & afx::kDownNegative) && !(disc_ptr->direction & afx::kDownPositive)) {
      //       [][][][][]{b}
      // _____|¯¯¯¯¯¯¯¯¯¯|a_____
      end_blend = true;
    }
    if (disc_image_.GetPtrBnds(x + 1, y - length)->direction & afx::kDownPositive && !(disc_image_.GetPtrBnds(x, y - length)->direction & afx::kDownNegative)) {
      //
      //      [b][][][][]{}
      // _____|a¯¯¯¯¯¯¯¯¯|_____
      start_blend = true;
    }
  }
  // Loop back through line and pos and length based on end orientations
  Discontinuity* line_disc_ptr = disc_ptr;
  if (start_blend && end_blend) {  // Both ends blend
    float half_length = length / 2.0f + 0.5;
    int max_pos = static_cast<int>(half_length) - 1;
    std::vector<float> weights(static_cast<int>(half_length));
    for (int pos = max_pos; pos >= 0; --pos) {
      line_disc_ptr = disc_image_.GetPreviousRow(line_disc_ptr);
      line_disc_ptr->horiz_weight = BlendWeight_(pos, half_length);
      weights[pos] = line_disc_ptr->horiz_weight;
    }
    for (auto it = (length & 1) ? (weights.begin() + 1) : weights.begin(); it != weights.end(); ++it) {
      line_disc_ptr = disc_image_.GetPreviousRow(line_disc_ptr);
      line_disc_ptr->horiz_weight = *it;
    }
  } else if (start_blend && !end_blend) {  // Start blends, end does NOT blend
    for (int pos = 0; pos <= length - 1; ++pos) {
      line_disc_ptr = disc_image_.GetPreviousRow(line_disc_ptr);
      line_disc_ptr->horiz_weight = BlendWeight_(pos, length);
    }
  } else if (!start_blend && end_blend) {  // Start does NOT blend, end does blend
    for (int pos = length - 1; pos >= 0; --pos) {
      line_disc_ptr = disc_image_.GetPreviousRow(line_disc_ptr);
      line_disc_ptr->horiz_weight = BlendWeight_(pos, length);
    }
  }
}

void MorphAA::BlendPixels_(const Bounds& region, const Image& input, Image* output) {
  const float* in_ptr = nullptr;
  float* out_ptr = nullptr;
  Discontinuity* disc_ptr = nullptr;
  const Discontinuity* t_disc_ptr = nullptr;
  const Discontinuity* l_disc_ptr = nullptr;
  for (int y = region.y1(); y <= region.y2(); ++y) {
    in_ptr = input.GetPtr(region.x1(), y);
    out_ptr = output->GetPtr(region.x1(), y);
    disc_ptr = disc_image_.GetPtr(region.x1(), y);
    t_disc_ptr = disc_image_.GetPtrBnds(region.x1(), y + 1);
    l_disc_ptr = disc_image_.GetPtrBnds(region.x1() - 1, y);
    for (int x = region.x1(); x <= region.x2(); ++x) {
      *out_ptr = *in_ptr;

      if (disc_ptr->direction & afx::kDownPositive) {
        const float* b_ptr = y > disc_image_.GetBounds().y1() ? in_ptr - input.GetPitch() : in_ptr;
        *out_ptr = disc_ptr->vert_weight * *b_ptr +  (1.0f - disc_ptr->vert_weight) * *out_ptr;
      } else if (t_disc_ptr->direction & afx::kDownNegative) {
        const float* t_ptr = y < disc_image_.GetBounds().y2() ? in_ptr + input.GetPitch() : in_ptr;
        *out_ptr = t_disc_ptr->vert_weight * *t_ptr + (1.0f - t_disc_ptr->vert_weight) * *out_ptr;
      }
      if (disc_ptr->direction & afx::kRightPositive) {
        const float* r_ptr = x < disc_image_.GetBounds().x2() ? in_ptr + 1 : in_ptr;
        *out_ptr = disc_ptr->horiz_weight * *r_ptr + (1.0f - disc_ptr->horiz_weight) * *out_ptr;
      } else if (l_disc_ptr->direction & afx::kRightNegative) {
        const float* l_ptr = x > disc_image_.GetBounds().x1() ? in_ptr - 1 : in_ptr;
        *out_ptr = l_disc_ptr->horiz_weight * *l_ptr + (1.0f - l_disc_ptr->horiz_weight) * *out_ptr;
      }

      in_ptr++;
      out_ptr++;
      disc_ptr++;
      t_disc_ptr++;
      // Pixel to left, so only increment when x is greater than x1
      if (x > disc_image_.GetBounds().x1()) { l_disc_ptr++; }
    }
  }
}

void MorphAA::Process(const Image& input, Image* output, float threshold, unsigned int max_line_length, afx::ImageThreader* threader) {
  threshold_ = threshold;
  disc_image_.Allocate(input.GetBounds());
  max_line_length_ = max_line_length;

  threader->ThreadRowChunks(disc_image_.GetBounds(), boost::bind(&MorphAA::MarkDisc_, this, _1, boost::cref(input)));
  threader->Wait();
  threader->ThreadRowChunks(disc_image_.GetBounds(), boost::bind(&MorphAA::FindRowLines_, this, _1, boost::cref(input)));
  threader->Wait();
  threader->ThreadColumnChunks(disc_image_.GetBounds(), boost::bind(&MorphAA::FindColumnLines_, this, _1, boost::cref(input)));
  threader->Wait();
  threader->ThreadRowChunks(disc_image_.GetBounds(), boost::bind(&MorphAA::BlendPixels_, this, _1, boost::cref(input), output));
  threader->Wait();
}

void MorphAA::Process(const Image& input, Image* output, float threshold, unsigned int max_line_length) {
  threshold_ = threshold;
  disc_image_.Allocate(input.GetBounds());
  max_line_length_ = max_line_length;

  MarkDisc_(disc_image_.GetBounds(), input);
  FindRowLines_(disc_image_.GetBounds(), input);
  FindColumnLines_(disc_image_.GetBounds(), input);
  BlendPixels_(disc_image_.GetBounds(), input, output);
}

}  // namespace afx
