// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef INCLUDE_MLAA_H_
#define INCLUDE_MLAA_H_

#include <OpenEXR/half.h>

#include "include/bounds.h"
#include "include/image.h"
#include "include/threading.h"

namespace afx {

enum DisFlags {
  kDisPosDown   = 0x01,  // pixel - bottom pixel is less than 0
  kDisNegDown   = 0x02,  // pixel - bottom pixel is greater than 0
  kDisPosRight  = 0x04,  // pixel - right pixel less than 0
  kDisNegRight  = 0x08,  // pixel - right pixel greater than 0
};

struct PixelInfo {
  half blend_x;
  half blend_y;
  unsigned char flags;
  PixelInfo() : blend_x(0.0f), blend_y(0.0f), flags(0) {}
};

class ImageInfo : public ImageBase<PixelInfo> {};

class MorphAA {
 private:
  ImageInfo info_;
  float threshold_;
  unsigned int max_line_length_;

  float Diff_(float a, float b);
  float CalcTrapArea_(int pos, float length);
  void MarkDisc_(const Bounds& region, const Image& input);
  void FindXLines_(const Bounds& region, const Image& input);
  void FindYLines_(const Bounds& region, const Image& input);
  void SetXLine_(PixelInfo* info_ptr, int length, int x, int y);
  void SetYLine_(PixelInfo* info_ptr, int length, int x, int y);
  void BlendPixels_(const Bounds& region, const Image& input, Image* output);

 public:
  void Process(const Image& input, Image* output, float threshold, unsigned int max_line_length, afx::ImageThreader* threader);
  void Process(const Image& input, Image* output, float threshold, unsigned int max_line_length);
};

}  // namespace afx

#endif  // INCLUDE_MLAA_H_



