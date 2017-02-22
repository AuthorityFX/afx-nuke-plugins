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

#include "settings.h"
#include "types.h"
#include "image.h"
#include "threading.h"

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

class ImageInfo {
 private:
  PixelInfo* ptr_;
  size_t pitch_;
  Bounds region_;
 public:
  ImageInfo();
  explicit ImageInfo(const Bounds& region);
  ImageInfo(unsigned int width, unsigned int height);
  ~ImageInfo();
  void Create(const Bounds& region);
  void Dispose();
  PixelInfo& GetVal(unsigned int x, unsigned int y) const;
  PixelInfo& GetValBnds(unsigned int x, unsigned int y) const;
  PixelInfo* GetPtr(unsigned int x, unsigned int y) const;
  PixelInfo* GetPtrBnds(unsigned int x, unsigned int y) const;
  PixelInfo* NextRow(PixelInfo* ptr);
  PixelInfo* PreviousRow(PixelInfo* ptr);
  size_t GetPitch() const;
  Bounds GetBounds() const;
};

class MorphAA {
 private:
  ImageInfo info_;
  float threshold_;

  float Diff_(float a, float b);
  float CalcTrapArea_(int pos, float length);
  void MarkDisc_(const Bounds& region, const Image& input);
  void FindXLines_(const Bounds& region, const Image& input);
  void FindYLines_(const Bounds& region, const Image& input);
  void SetXLine_(PixelInfo* info_ptr, int length, int x, int y);
  void SetYLine_(PixelInfo* info_ptr, int length, int x, int y);
  void BlendPixels_(const Bounds& region, const Image& input, Image* output);

 public:
  void Process(const Image& input, Image* output, float threshold, afx::Threader* threader);
  void Process(const Image& input, Image* output, float threshold);
};

}  // namespace afx

#endif  // INCLUDE_MLAA_H_



