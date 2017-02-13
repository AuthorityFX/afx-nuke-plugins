// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef MLAA_H_
#define MLAA_H_

#include "settings.h"
#include "types.h"
#include "image.h"
#include "threading.h"

namespace afx {

enum Direction {
  kBlendUp = true,
  kBlendDown = false,
  kBlendLeft = true,
  kBlendRight = false
};

struct PixelInfo {
  bool dis_x, dis_y, dis_dir_x, dis_dir_y;
  unsigned int pos_h, pos_v;
  float length_x, length_y;
  PixelInfo() : dis_x(false), dis_y(false), dis_dir_x(false), dis_dir_y(false),
                pos_h(0), pos_v(0), length_x(0.0f), length_y(0.0f) {}
};

struct BlendWeight {
  float top, bottom, left, right;
   BlendWeight() : top(0.0f), bottom(0.0f), left(0.0f), right(0.0f) {}
};

class ImageInfo {
 private:
  PixelInfo* ptr_;
  size_t pitch_;
  Bounds region_;
 public:
  ImageInfo ();
  ImageInfo(const Bounds& region);
  ImageInfo(unsigned int width, unsigned int height);
  ~ImageInfo();
  void Create(const Bounds& region);
  void Dispose();
  PixelInfo& GetPixel(unsigned int x, unsigned int y) const;
  PixelInfo* GetPtr(unsigned int x, unsigned int y) const;
  PixelInfo* GetPtrBnds(unsigned int x, unsigned int y) const;
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
  void SetXLine_(PixelInfo* info_ptr, const Image& input, int length, int x, int y);
  void SetYLine_(PixelInfo* info_ptr, const Image& input, int length, int x, int y);
  void BlendPixels_(const Bounds& region, const Image& input, Image& output);

 public:
  void Process(const Image& input, Image& output, float threshold, afx::Threader& threader);
  void Process(const Image& input, Image& output, float threshold);
};

} // namespace afx

#endif  // MLAA_H_
