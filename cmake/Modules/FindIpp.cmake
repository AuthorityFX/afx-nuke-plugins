# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (C) 2017, Ryan P. Wilson
#
#      Authority FX, Inc.
#      www.authorityfx.com


if(NOT DEFINED IPP_ROOT)
  set(IPP_ROOT "/opt/intel/ipp")
endif()

find_path(
  Ipp_INCLUDE_DIR
  NAME ipp.h
  PATHS ${IPP_ROOT}/include/
  NO_DEFAULT_PATH
)

if(Ipp_INCLUDE_DIR)
  include_directories(${Ipp_INCLUDE_DIR})
endif()

foreach(COMPONENT ${Ipp_FIND_COMPONENTS})
  string(TOUPPER ${COMPONENT} UPPERCOMPONENT)
  find_library(
    Ipp_${UPPERCOMPONENT}_LIBRARY
    NAMES ${COMPONENT}
    PATHS ${IPP_ROOT}/lib/intel64
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
  )

  if(Ipp_${UPPERCOMPONENT}_LIBRARY)
    if(NOT DEFINED Ipp_LIBRARY_DIR)
      get_filename_component(_dir "${Ipp_${UPPERCOMPONENT}_LIBRARY}" PATH)
      set(Ipp_LIBRARY_DIR "${_dir}" CACHE PATH "Ipp library directory" FORCE)
    endif()

    list(APPEND Ipp_LIBRARIES ${Ipp_${UPPERCOMPONENT}_LIBRARY})
  endif()

endforeach()

foreach(COMPONENT ${Ipp_FIND_COMPONENTS})
  string(TOUPPER ${COMPONENT} UPPERCOMPONENT)

  if(NOT Ipp_${UPPERCOMPONENT}_LIBRARY)
    set(Ipp_LIBRARIES "")
    set(Ipp_MISSING_LIBS "${Ipp_MISSING_LIBS}\n  -- ${COMPONENT}")
  endif()
endforeach()

if(Ipp_MISSING_LIBS)
  message("**********Ipp Missing Libraries************")
  message("Unable to find the following Ipp libraries: ${Ipp_MISSING_LIBS}")
  message("Use cmake -D IPP_ROOT=")
  message("***********************************************")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Ipp DEFAULT_MSG
    IPP_ROOT
    Ipp_INCLUDE_DIR
    Ipp_LIBRARIES
    Ipp_LIBRARY_DIR
 )
