// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef NUKE_HELPER_H_
#define NUKE_HELPER_H_

#include <DDImage/Box.h>

#include "types.h"

using namespace DD::Image;

namespace afx {

  afx::Bounds BoxToBounds(Box box) {
    return afx::Bounds(box.x(), box.y(), box.r() - 1, box.t() - 1);
  }
  Box BoundsToBox(afx::Bounds bnds) {
    return Box(bnds.x1(), bnds.y1(), bnds.x2() + 1, bnds.y2() + 1);
  }

}

#endif  // NUKE_HELPER_H_