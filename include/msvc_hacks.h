// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2017, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef AFX_MSVC_HACKS_H_
#define AFX_MSVC_HACKS_H_

// MSVC10 does not have C99 support. fmin and fmax functions are not available
#ifdef _WIN32
  float fminf(float a, float b) {
    return std::min(a, b);
  }
  float fmaxf(float a, float b) {
    return std::max(a, b);
  }
#endif

#endif  // AFX_MSVC_HACKS_H_
