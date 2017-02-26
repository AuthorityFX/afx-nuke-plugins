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

#include "include/bounds.h"

namespace afx {

  afx::Bounds BoxToBounds(DD::Image::Box box) {
    return afx::Bounds(box.x(), box.y(), box.r() - 1, box.t() - 1);
  }
  DD::Image::Box BoundsToBox(afx::Bounds bnds) {
    return DD::Image::Box(bnds.x1(), bnds.y1(), bnds.x2() + 1, bnds.y2() + 1);
  }

  afx::Bounds InputBounds(DD::Image::Iop* input) {
    return afx::BoxToBounds(input->info().box());
  }

}  // namespace afx

#endif  // INCLUDE_NUKE_HELPER_H_
