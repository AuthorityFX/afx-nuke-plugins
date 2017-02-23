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

find_file(
  _jemalloc_CONFIG
  NAME jemalloc-config
  PATHS ${JEMALLOC_ROOT}
  PATH_SUFFIXES bin
  NO_DEFAULT_PATH
  )

if(_jemalloc_CONFIG)
  execute_process (COMMAND ${_jemalloc_CONFIG} "--version" OUTPUT_VARIABLE _version_STRING)
  string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+).*" "\\1" Jemalloc_VERSION_MAJOR ${_version_STRING})
  string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+).*" "\\2" Jemalloc_VERSION_MINOR ${_version_STRING})
  string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+).*" "\\3" Jemalloc_VERSION_PATCH ${_version_STRING})
  set(Jemalloc_VERSION "${Jemalloc_VERSION_MAJOR}.${Jemalloc_VERSION_MINOR}.${Jemalloc_VERSION_PATCH}" CACHE STRING "Version of Jemalloc computed from jemallo-config.")
endif()

find_library(
  Jemalloc_LIBRARIES
  NAMES jemalloc
  PATHS ${JEMALLOC_ROOT}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH
)

if(Jemalloc_LIBRARIES)
  get_filename_component(_dir "${Jemalloc_LIBRARIES}" PATH)
  set(Jemalloc_LIBRARY_DIR "${_dir}" CACHE PATH "Jemalloc library directory" FORCE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Jemalloc
  REQUIRED_VARS
    JEMALLOC_ROOT
    Jemalloc_INCLUDE_DIR
    Jemalloc_LIBRARIES
    Jemalloc_LIBRARY_DIR
  VERSION_VAR
    Jemalloc_VERSION
 )
