// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef INCLUDE_NUKE_HELPER_H_
#define INCLUDE_NUKE_HELPER_H_

#include <DDImage/Box.h>
#include <DDImage/Iop.h>
#include <DDImage/ImagePlane.h>

#include "include/bounds.h"

namespace nuke = DD::Image;

namespace afx {

afx::Bounds BoxToBounds(nuke::Box box) {
  return afx::Bounds(box.x(), box.y(), box.r() - 1, box.t() - 1);
}
nuke::Box BoundsToBox(afx::Bounds bnds) {
  return nuke::Box(bnds.x1(), bnds.y1(), bnds.x2() + 1, bnds.y2() + 1);
}

afx::Bounds InputBounds(nuke::Iop* input) {
  return afx::BoxToBounds(input->info().box());
}

const float* GetPlanePtr(const nuke::ImagePlane& plane, int x, int y, nuke::Channel channel = nuke::Chan_Black) {
  int channel_offset = plane.chanNo(channel);
  return plane.readable() + (plane.bounds().clampy(y) - plane.bounds().y()) * plane.rowStride() + (plane.bounds().clampx(x) - plane.bounds().x()) * plane.colStride() + channel_offset;
}
std::size_t GetPlanePitch(const nuke::ImagePlane& plane) {
  return static_cast<std::size_t>(plane.rowStride() * sizeof(float));
}

}  // namespace afx

#endif  // INCLUDE_NUKE_HELPER_H_
