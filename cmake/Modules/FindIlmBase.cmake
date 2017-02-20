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
  include_directories(${Jemalloc_INCLUDE_DIR})
endif()

foreach(COMPONENT ${IlmBase_FIND_COMPONENTS})
  string(TOUPPER ${COMPONENT} UPPERCOMPONENT)
  find_library(
    IlmBase_${UPPERCOMPONENT}_LIBRARY
    NAMES ${COMPONENT}
    PATHS ${ILMBASE_ROOT}
    PATH_SUFFIXES lib
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
find_package_handle_standard_args(IlmBase DEFAULT_MSG
    ILMBASE_ROOT
    IlmBase_INCLUDE_DIR
    IlmBase_LIBRARIES
    IlmBase_LIBRARY_DIR
 )
