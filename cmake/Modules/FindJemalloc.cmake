# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (C) 2017, Ryan P. Wilson
#
#      Authority FX, Inc.
#      www.authorityfx.com


if(NOT DEFINED JEMALLOC_ROOT)
  set(JEMALLOC_ROOT "/usr/local/jemalloc")
endif()

find_path(
  Jemalloc_INCLUDE_DIR
  NAME jemalloc/jemalloc.h
  PATHS ${JEMALLOC_ROOT}/include/
  NO_DEFAULT_PATH
)

if(Jemalloc_INCLUDE_DIR)
  include_directories(${Jemalloc_INCLUDE_DIR})
endif()

find_library(
  Jemalloc_LIBRARIES
  NAMES jemalloc
  PATHS ${JEMALLOC_ROOT}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)

if(Jemalloc_LIBRARIES)
  get_filename_component(_dir "${Jemalloc_LIBRARIES}" PATH)
  set(Jemalloc_LIBRARY_DIR "${_dir}" CACHE PATH "Jemalloc library directory" FORCE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Jemalloc DEFAULT_MSG
    JEMALLOC_ROOT
    Jemalloc_INCLUDE_DIR
    Jemalloc_LIBRARIES
    Jemalloc_LIBRARY_DIR
 )
