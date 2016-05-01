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
  kUp = false,
  kDown = true,
  kRight = false,
  kLeft = true
};

struct PixelInfo {
  bool dis_h, dis_v, dis_dir_h, dis_dir_v;
  unsigned int pos_h, pos_v;
  float length_h, length_v;
  PixelInfo() : dis_h(false), dis_v(false), dis_dir_h(false), dis_dir_v(false),
                pos_h(0), pos_v(0), length_h(0.0f), length_v(0.0f) {}
};

struct BlendWeight {
  float top, bottom, left, right;
   BlendWeight() : top(0), bottom(0), left(0), right(0) {}
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
  size_t GetPitch() const;
  Bounds GetBounds() const;
};

class MorphAA {
 private:

  ImageInfo info_;
  float threshold_;

  inline float Diff_(float a, float b) {
//     return (a - b) / (a + b);
    return a - b;
  }
  inline float CalcTrapArea_(int pos, float length) {
    return (float)pos / length + (0.5f / length);
  }

  void MarkDisc_(const Bounds& region, const Image& input);
  void FindHorLines_(const Bounds& region, const Image& input);
  void FindVertLines_(const Bounds& region, const Image& input);
  void BlendPixels_(const Bounds& region, const Image& input, Image& output);

 public:
  void Process(const Image& input, Image& output, float threshold, afx::Threader& threader);
  void Process(const Image& input, Image& output, float threshold);
};

} // namespace afx

#endif  // MLAA_H_
