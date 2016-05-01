// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef SETTINGS_H_
#define SETTINGS_H_

// Use Google C++ Style
// https://google.github.io/styleguide/cppguide.html

// not using c++11, so we'll set nullptr = __null
#define nullptr __null

// Boost enable dynamic linking
#define BOOST_ALL_DYN_LINK
#define BOOST_DYN_LINK

// Import config settings from cmake
// VERSION VARIABLES:
// afx_nuke_plugins_VERSION_MAJOR
// afx_nuke_plugins_VERSION_MINOR
//#include "cmake_config.h"

#endif  // AFX_SETTINGS_H_
