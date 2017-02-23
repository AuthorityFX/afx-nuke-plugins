# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (C) 2017, Ryan P. Wilson
#
#      Authority FX, Inc.
#      www.authorityfx.com


if(NOT DEFINED ILMBASE_ROOT)
  set(ILMBASE_ROOT "/usr/local/IlmBase")
endif()

find_path(
  IlmBase_INCLUDE_DIR
  NAME OpenEXR/half.h
  PATHS ${ILMBASE_ROOT}/include/
  NO_DEFAULT_PATH
)

if(IlmBase_INCLUDE_DIR)
  include_directories(${IlmBase_INCLUDE_DIR})

  find_file(
    _IlmBase_CONFIG
    NAME IlmBaseConfig.h
    PATHS ${IlmBase_INCLUDE_DIR}/OpenEXR
    NO_DEFAULT_PATH
    )

  if(_IlmBase_CONFIG)
    file(
      STRINGS ${_IlmBase_CONFIG} _IlmBase_VERSION_MAJOR
      REGEX "#define[ ]+ILMBASE_VERSION_MAJOR[ ]+([0-9]+)"
      )
    if(_IlmBase_VERSION_MAJOR)
      string(REGEX MATCH "[0-9]+" IlmBase_VERSION_MAJOR ${_IlmBase_VERSION_MAJOR})
    endif()

    file(
      STRINGS ${_IlmBase_CONFIG} _IlmBase_VERSION_MINOR
      REGEX "#define[ ]+ILMBASE_VERSION_MINOR[ ]+([0-9]+)"
      )
    if(_IlmBase_VERSION_MINOR)
      string(REGEX MATCH "[0-9]+" IlmBase_VERSION_MINOR ${_IlmBase_VERSION_MINOR})
    endif()

    file(
      STRINGS ${_IlmBase_CONFIG} _IlmBase_VERSION_PATCH
      REGEX "#define[ ]+ILMBASE_VERSION_PATCH[ ]+([0-9]+)"
      )
    if(_IlmBase_VERSION_PATCH)
      string(REGEX MATCH "[0-9]+" IlmBase_VERSION_PATCH ${_IlmBase_VERSION_PATCH})
    endif()

    set(IlmBase_VERSION "${IlmBase_VERSION_MAJOR}.${IlmBase_VERSION_MINOR}.${IlmBase_VERSION_PATCH}" CACHE STRING "Version of IlmBase computed from IlmBaseConfig.h")
  endif()
endif()

foreach(COMPONENT ${IlmBase_FIND_COMPONENTS})
  string(TOUPPER ${COMPONENT} UPPERCOMPONENT)
  find_library(
    IlmBase_${UPPERCOMPONENT}_LIBRARY
    NAMES ${COMPONENT}
    PATHS ${ILMBASE_ROOT}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
  )

  if(IlmBase_${UPPERCOMPONENT}_LIBRARY)
    if(NOT DEFINED IlmBase_LIBRARY_DIR)
      get_filename_component(_dir "${IlmBase_${UPPERCOMPONENT}_LIBRARY}" PATH)
      set(IlmBase_LIBRARY_DIR "${_dir}" CACHE PATH "IlmBase library directory" FORCE)
    endif()

    list(APPEND IlmBase_LIBRARIES ${IlmBase_${UPPERCOMPONENT}_LIBRARY})
  endif()
endforeach()

foreach(COMPONENT ${IlmBase_FIND_COMPONENTS})
  string(TOUPPER ${COMPONENT} UPPERCOMPONENT)

  if(NOT IlmBase_${UPPERCOMPONENT}_LIBRARY)
    set(IlmBase_LIBRARIES "")
    set(IlmBase_MISSING_LIBS "${IlmBase_MISSING_LIBS}\n  -- ${COMPONENT}")
  endif()
endforeach()

if(IlmBase_MISSING_LIBS)
  message("**********IlmBase Missing Libraries************")
  message("Unable to find the following IlmBase libraries: ${IlmBase_MISSING_LIBS}")
  message("Use cmake -D ILMBASE_ROOT=")
  message("***********************************************")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(IlmBase
  REQUIRED_VARS
    ILMBASE_ROOT
    IlmBase_INCLUDE_DIR
    IlmBase_LIBRARIES
    IlmBase_LIBRARY_DIR
  VERSION_VAR
    IlmBase_VERSION
 )
