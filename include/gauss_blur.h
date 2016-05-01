// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Jordan Melo
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef GAUSS_BLUR_H_
#define GAUSS_BLUR_H_

#include "image.h"
#include "threading.h"

namespace {

class GaussianBlur : private Threader {
 private:
  inline void HorizontalGaussianBlur(Bounds region, AfxImageBase* in, AfxImageBase* out, float* kernel, int kernel_size);
  inline void VerticalGaussianBlur(Bounds region, AfxImageBase* in, AfxImageBase* out, float* kernel, int kernel_size);

 public:
  GaussianBlur(Bounds region) : Threader(region) {}

  void Process(AfxImageBase* in, float blurRadius) { Process(in, in, blurRadius); }
  inline void Process(AfxImageBase* in, AfxImageBase* out, float blurRadius);
};

} // namespace afx

#endif  // GAUSS_BLUR_H_
