# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (C) 2017, Ryan P. Wilson
#
#      Authority FX, Inc.
#      www.authorityfx.com

cmake_minimum_required(VERSION 2.8)
project(afx-nuke-plugins)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8.1 OR CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8.5)
    message(FATAL_ERROR "Need GCC 4.8.1 - 4.8.5")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  if(NOT MSVC_VERSION MATCHES "1600")
    message(FATAL_ERROR "Need MSVC++ 10.0 - Visual C++ 2010")
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  if(WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O2 -DNDEBUG")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -DNDEBUG -Wall")
  endif()
endif()

# Enable C++11
if(UNIX AND NOT CYGWIN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# Set install RPATH
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")
# Build with with install RPATH
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
# Add RPATHS outside of the build tree to install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# Append project Modules directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# afx include files
include_directories(${CMAKE_SOURCE_DIR})

find_package(Nuke REQUIRED)
find_package(Ipp REQUIRED COMPONENTS ippcore ippi)
if(NOT DEFINED BOOST_ROOT)
  set(BOOST_ROOT "/usr/local/boost")
endif()
find_package(Boost REQUIRED COMPONENTS system thread)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()
find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2")
find_package(IlmBase COMPONENTS Half)
if(IlmBase_FOUND)
  # USE_HALF checked by preprocessor in mlaa.h
  add_definitions(-DUSE_HALF)
endif()
find_package(Hoard)
if(Hoard_FOUND)
  set(Hoard_LIBRARIES ${Hoard_LIBRARY})
endif()

add_library(core SHARED src/threading.cpp)
set_target_properties(core PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
target_link_libraries(core ${Ipp_LIBRARIES} ${Boost_LIBRARIES} ${IlmBase_LIBRARIES} ${Hoard_LIBRARIES})
list(APPEND afx_LIBRARIES core)

cuda_add_library(cuda src/cuda_helper.cpp SHARED)
set_target_properties(cuda PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
target_link_libraries(cuda core)
add_dependencies(cuda core)
list(APPEND afx_LIBRARIES cuda)

add_library(afx_soft_clip SHARED src/afx_soft_clip.cpp)
set_target_properties(afx_soft_clip PROPERTIES PREFIX "")
target_link_libraries(afx_soft_clip ${Nuke_LIBRARIES})
list(APPEND afx_PLUGINS afx_soft_clip)

add_library(afx_tone_map SHARED src/afx_tone_map.cpp)
set_target_properties(afx_tone_map PROPERTIES PREFIX "")
target_link_libraries(afx_tone_map ${Nuke_LIBRARIES})
list(APPEND afx_PLUGINS afx_tone_map)

add_library(afx_noise_map SHARED src/afx_noise_map.cpp)
set_target_properties(afx_noise_map PROPERTIES PREFIX "")
target_link_libraries(afx_noise_map ${Nuke_LIBRARIES} core)
add_dependencies(afx_noise_map core)
list(APPEND afx_PLUGINS afx_noise_map)

add_library(afx_chroma_key SHARED src/afx_chroma_key.cpp)
set_target_properties(afx_chroma_key PROPERTIES PREFIX "")
target_link_libraries(afx_chroma_key ${Nuke_LIBRARIES} core)
add_dependencies(afx_chroma_key core)
list(APPEND afx_PLUGINS afx_chroma_key)

add_library(afx_anti_alias SHARED src/afx_anti_alias.cpp src/mlaa.cpp)
set_target_properties(afx_anti_alias PROPERTIES PREFIX "")
target_link_libraries(afx_anti_alias ${Nuke_LIBRARIES} core)
add_dependencies(afx_anti_alias core)
list(APPEND afx_PLUGINS afx_anti_alias)

add_library(afx_despill SHARED src/afx_despill.cpp)
set_target_properties(afx_despill PROPERTIES PREFIX "")
target_link_libraries(afx_despill ${Nuke_LIBRARIES} core)
add_dependencies(afx_despill core)
list(APPEND afx_PLUGINS afx_despill)

cuda_add_library(afx_median src/afx_median.cpp src/afx_median.cu SHARED)
set_target_properties(afx_median PROPERTIES PREFIX "")
target_link_libraries(afx_median ${Nuke_LIBRARIES} core cuda)
add_dependencies(afx_median core cuda)
list(APPEND afx_PLUGINS afx_median)

add_library(afx_glow SHARED src/afx_glow.cpp)
set_target_properties(afx_glow PROPERTIES PREFIX "")
target_link_libraries(afx_glow ${Nuke_LIBRARIES} core cuda)
add_dependencies(afx_glow core cuda)
list(APPEND afx_PLUGINS afx_glow)

# Nuke home directory
if(WIN32 AND NOT CYGWIN)
  get_filename_component(Nuke_HOME_DIR "$ENV{HOMEPATH}/.nuke/" ABSOLUTE)
else()
  get_filename_component(Nuke_HOME_DIR "$ENV{HOME}/.nuke/" ABSOLUTE)
endif()

# Set default install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${Nuke_HOME_DIR}/afx-nuke-plugins" CACHE PATH "Install path prefix" FORCE)
  message("CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
endif()

# Create init.py for $HOME/.nuke
if(EXISTS ${Nuke_HOME_DIR}/init.py)
  set(_pluginAddPath_FOUND "")
  file(READ ${Nuke_HOME_DIR}/init.py _nuke_INIT)
  if(_nuke_INIT)
    string(REGEX REPLACE "nuke.pluginAddPath\\(['\"]?(.+afx-nuke-plugins\\/?)['\"]?\\)" "\\0;\\1;" _afx_PLUGIN_PATH_LIST ${_nuke_INIT})
    list(LENGTH _afx_PLUGIN_PATH_LIST _afx_PLUGIN_PATH_LIST_LENGTH)
    if(${_afx_PLUGIN_PATH_LIST_LENGTH} GREATER 1)
      list(GET _afx_PLUGIN_PATH_LIST 1 _afx_PLUGIN_PATH)
      string(REGEX MATCH ${CMAKE_INSTALL_PREFIX} _afx_PLUGIN_PATH_MATCH ${_afx_PLUGIN_PATH})
      if(_afx_PLUGIN_PATH_MATCH)
        set(_pluginAddPath_FOUND "FOUND")
      endif()
    endif()
  endif()
  if(NOT _pluginAddPath_FOUND)
    set(_nuke_INIT "${_nuke_INIT}nuke.pluginAddPath('${CMAKE_INSTALL_PREFIX}')\n\n")
  endif()
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/init.py ${_nuke_INIT})
else()
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/init.py "import nuke\n\nnuke.pluginAddPath('${CMAKE_INSTALL_PREFIX}')\n\n")
endif()


install(TARGETS ${afx_LIBRARIES} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(TARGETS ${afx_PLUGINS} DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY icons/ DESTINATION ${CMAKE_INSTALL_PREFIX}/icons)
install(FILES menu.py DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/init.py DESTINATION ${Nuke_HOME_DIR})

if(UNIX)
  foreach(_lib ${Boost_LIBRARIES})
    get_filename_component(_name ${_lib} NAME_WE)
    file(GLOB _files "${Boost_LIBRARY_DIR}/${_name}*.so*")
    list(APPEND Redist_LIBRARIES ${_files})
  endforeach()
  foreach(_lib ${Ipp_LIBRARIES})
    get_filename_component(_name ${_lib} NAME_WE)
    file(GLOB _files FOLLOW_SYMLINKS "${Ipp_LIBRARY_DIR}/${_name}*.so*")
    list(APPEND Redist_LIBRARIES ${_files})
  endforeach()
  foreach(_lib ${CUDA_LIBRARIES})
    get_filename_component(_name ${_lib} NAME_WE)
    get_filename_component(_cuda_lib_dir ${_lib} DIRECTORY)
    file(GLOB _files "${_cuda_lib_dir}/${_name}*.so*")
    list(APPEND Redist_LIBRARIES ${_files})
  endforeach()
  foreach(_lib ${IlmBase_LIBRARIES})
    get_filename_component(_name ${_lib} NAME_WE)
    file(GLOB _files "${IlmBase_LIBRARY_DIR}/${_name}*.so*")
    list(APPEND Redist_LIBRARIES ${_files})
  endforeach()
  foreach(_lib ${Hoard_LIBRARIES})
    get_filename_component(_name ${_lib} NAME_WE)
    file(GLOB _files "${Hoard_LIBRARY_DIR}/${_name}*.so*")
    list(APPEND Redist_LIBRARIES ${_files})
  endforeach()

  install(
    CODE "
      EXECUTE_PROCESS(COMMAND cp -Rp ${Redist_LIBRARIES} ${CMAKE_INSTALL_PREFIX}/lib)
  ")
endif()
