# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (C) 2017, Ryan P. Wilson
#
#      Authority FX, Inc.
#      www.authorityfx.com

if(NOT DEFINED HOARD_ROOT)
  set(HOARD_ROOT "/usr/local/Hoard")
endif()

# Find Hoard library
find_library(
  Hoard_LIBRARY
  NAME hoard
  HINTS ${HOARD_ROOT}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)
# Set library directory
get_filename_component(_dir ${Hoard_LIBRARY} PATH)
set(Hoard_LIBRARY_DIR "${_dir}" CACHE PATH "Hoard library directory")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Hoard
  REQUIRED_VARS
    Hoard_LIBRARY
    Hoard_LIBRARY_DIR
 )
