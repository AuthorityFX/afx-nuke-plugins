// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef AFX_MLAA_H_
#define AFX_MLAA_H_

#include <OpenEXR/half.h>

#include <cstdint>

#include "include/bounds.h"
#include "include/image.h"
#include "include/threading.h"

namespace afx {

enum DiscontinuityFlags {
  kNone           = 0,
  kDownPositive   = 0x01,  // (pixel - bottom pixel) < 0
  kDownNegative   = 0x02,  // (pixel - bottom pixel) > 0
  kRightPositive  = 0x04,  // (pixel - right pixel) < 0
  kRightNegative  = 0x08,  // (pixel - right pixel) > 0
  kDown           = kDownPositive | kDownNegative,
  kRight          = kRightPositive | kRightNegative,
};

struct Discontinuity {
  half vert_weight;
  half horiz_weight;
  std::uint8_t disc_dir;
  Discontinuity() : vert_weight(0.0f), horiz_weight(0.0f), disc_dir(kNone) {}
};

class ImageInfo : public ImageBase<Discontinuity> {};

class MorphAA {
 public:
  void Process(const Image& input, Image* output, float threshold, unsigned int max_line_length, afx::ImageThreader* threader);
  void Process(const Image& input, Image* output, float threshold, unsigned int max_line_length);

 private:
  ImageInfo disc_image_;
  float threshold_;

  unsigned int max_line_length_;
  // Relative difference between two pixels
  float RelativeDifference_(float a, float b);
  // Trapezoid area of pixel dissected by jaggie line
  float BlendWeight_(unsigned int pos, float length);
  // Set bitflags for vertical and horizontal discontinuities
  void MarkDisc_(const Bounds& region, const Image& input);
  // March row continuous discontinuities to determine line legnth
  void FindRowLines_(const Bounds& region, const Image& input);
  // March columm continuous discontinuities to determine line legnth
  void FindColumnLines_(const Bounds& region, const Image& input);
  // Loop back through row line and calculate trapezoid weights
  void RowBlendWeights_(Discontinuity* disc_ptr, int length, int x, int y);
  // Loop back through column line and calculate trapezoid weights
  void ColumnBlendWeights_(Discontinuity* disc_ptr, int length, int x, int y);
  // Perform anti-aliasing by blending neighbor pixels with trapezoid weights
  void BlendPixels_(const Bounds& region, const Image& input, Image* output);
};

}  // namespace afx

#endif  // AFX_MLAA_H_



