# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (C) 2017, Ryan P. Wilson
#
#      Authority FX, Inc.
#      www.authorityfx.com


if(NOT DEFINED NUKE_ROOT)
  file(GLOB _nuke_ROOT_DIRS "/usr/local/Nuke*")
  list(SORT _nuke_ROOT_DIRS)
  list(GET _nuke_ROOT_DIRS 0 NUKE_ROOT)
endif()

string(REGEX MATCHALL "[0-9]+\\.[0-9]+v[0-9]" Nuke_VERSION ${NUKE_ROOT})

find_path(
  Nuke_INCLUDE_DIR
  NAME DDImage/Iop.h
  PATHS ${NUKE_ROOT}/include/
  NO_DEFAULT_PATH
)

if(Nuke_INCLUDE_DIR)
  include_directories(${Nuke_INCLUDE_DIR})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Nuke
  REQUIRED_VARS
    NUKE_ROOT
    Nuke_INCLUDE_DIR
  VERSION_VAR
    Nuke_VERSION
 )
