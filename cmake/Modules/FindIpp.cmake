# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (C) 2017, Ryan P. Wilson
#
#      Authority FX, Inc.
#      www.authorityfx.com

if(NOT DEFINED IPP_ROOT)
  if(WIN32)
    file(GLOB _ipp_WIN_ROOT_DIRS "C:/Program Files (x86)/IntelSWTools/parallel_studio_xe_2017*/compilers_and_libraries_2017/windows/ipp")
    list(SORT _ipp_WIN_ROOT_DIRS)
    list(GET _ipp_WIN_ROOT_DIRS 0 IPP_ROOT)
  else()
    set(IPP_ROOT "/opt/intel/ipp")
  endif()
endif()

find_path(
  Ipp_INCLUDE_DIR
  NAMES ipp.h
  PATHS ${IPP_ROOT}/include/
  NO_DEFAULT_PATH
)

if(Ipp_INCLUDE_DIR)
  include_directories(${Ipp_INCLUDE_DIR})

  find_file(
    _Ipp_VERSION_H
    NAMES ippversion.h
    PATHS ${Ipp_INCLUDE_DIR}
    NO_DEFAULT_PATH
    )

  if(_Ipp_VERSION_H)
    file(
      STRINGS ${_Ipp_VERSION_H} _Ipp_VERSION_MAJOR
      REGEX "#define[ ]+IPP_VERSION_MAJOR[ ]+[0-9]+"
      )
    if(_Ipp_VERSION_MAJOR)
      string(REGEX MATCH "[0-9]+" Ipp_VERSION_MAJOR ${_Ipp_VERSION_MAJOR})
    endif()

    file(
      STRINGS ${_Ipp_VERSION_H} _Ipp_VERSION_MINOR
      REGEX "#define[ ]+IPP_VERSION_MINOR[ ]+[0-9]+"
      )
    if(_Ipp_VERSION_MINOR)
      string(REGEX MATCH "[0-9]+" Ipp_VERSION_MINOR ${_Ipp_VERSION_MINOR})
    endif()

    file(
      STRINGS ${_Ipp_VERSION_H} _Ipp_VERSION_UPDATE
      REGEX "#define[ ]+IPP_VERSION_UPDATE[ ]+[0-9]+"
      )
    if(_Ipp_VERSION_UPDATE)
      string(REGEX MATCH "[0-9]+" Ipp_VERSION_UPDATE ${_Ipp_VERSION_UPDATE})
    endif()

    set(Ipp_VERSION "${Ipp_VERSION_MAJOR}.${Ipp_VERSION_MINOR}.${Ipp_VERSION_UPDATE}" CACHE STRING "Version of Ipp computed from ippversion.h")
  endif()
endif()

# Find ippcore library
find_library(
  _Ipp_CORE_LIBRARY
  NAMES ippcore
  PATHS ${IPP_ROOT}
  PATH_SUFFIXES
    lib
    lib/intel64
  NO_DEFAULT_PATH
)
# Set library directory
get_filename_component(_dir ${_Ipp_CORE_LIBRARY} PATH)
set(Ipp_LIBRARY_DIR "${_dir}" CACHE PATH "Ipp library directory")

foreach(COMPONENT ${Ipp_FIND_COMPONENTS})
  find_library(
    Ipp_${COMPONENT}_FOUND
    NAMES ${COMPONENT}
    PATHS ${Ipp_LIBRARY_DIR}
    NO_DEFAULT_PATH
  )
  if(Ipp_${COMPONENT}_FOUND)
    list(APPEND Ipp_LIBRARIES ${Ipp_${COMPONENT}_FOUND})
  endif()
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Ipp
  FOUND_VAR
    Ipp_FOUND
  REQUIRED_VARS
    IPP_ROOT
    Ipp_INCLUDE_DIR
    Ipp_LIBRARIES
    Ipp_LIBRARY_DIR
  VERSION_VAR
    Ipp_VERSION
  HANDLE_COMPONENTS
 )
