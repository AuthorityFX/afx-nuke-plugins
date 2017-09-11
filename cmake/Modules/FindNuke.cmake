# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (C) 2017, Ryan P. Wilson
#
#      Authority FX, Inc.
#      www.authorityfx.com

if(NOT DEFINED NUKE_ROOT)
  if(WIN32)
    file(GLOB _nuke_ROOT_DIRS "C:/Program Files/Nuke*")
  elseif(APPLE)
    file(GLOB _nuke_ROOT_DIRS "/Applications/Nuke*")
  else()
    file(GLOB _nuke_ROOT_DIRS "/usr/local/Nuke*")
  endif()
  list(SORT _nuke_ROOT_DIRS)
  list(LENGTH _nuke_ROOT_DIRS _nuke_ROOT_DIRS_LENGTH)
  if(${_nuke_ROOT_DIRS_LENGTH} GREATER 0)
    list(GET _nuke_ROOT_DIRS 0 NUKE_ROOT)
  endif()
endif()

string(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\v([0-9]+)" "\\0;\\1;\\2;\\3" _nuke_VERSION_LIST ${NUKE_ROOT})

list(LENGTH _nuke_VERSION_LIST _list_LENGTH)
if(_list_LENGTH GREATER 2)
  list(GET _nuke_VERSION_LIST 1 Nuke_VERSION_MAJOR)
  list(GET _nuke_VERSION_LIST 2 Nuke_VERSION_MINOR)
  list(GET _nuke_VERSION_LIST 3 Nuke_VERSION_PATCH)
  set(Nuke_VERSION "${Nuke_VERSION_MAJOR}.${Nuke_VERSION_MINOR}v${Nuke_VERSION_PATCH}")
endif()

find_path(
  Nuke_INCLUDE_DIR
  NAMES DDImage/Iop.h
  PATHS ${NUKE_ROOT}/include/
  NO_DEFAULT_PATH
)

if(Nuke_INCLUDE_DIR)
  include_directories(${Nuke_INCLUDE_DIR})
endif()

find_library(
  Nuke_LIBRARIES
  NAMES DDImage
  PATHS ${NUKE_ROOT}
  NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Nuke
  FOUND_VAR
    Nuke_FOUND
  REQUIRED_VARS
    NUKE_ROOT
    Nuke_INCLUDE_DIR
    Nuke_LIBRARIES
  VERSION_VAR
    Nuke_VERSION
 )
