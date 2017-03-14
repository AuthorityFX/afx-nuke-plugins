# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (C) 2017, Ryan P. Wilson
#
#      Authority FX, Inc.
#      www.authorityfx.com

cmake_minimum_required(VERSION 2.6)
project(afx-nuke-plugins)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -DNDEBUG")
endif()

# Set install RPATH
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")
# Build with with install RPATH
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
# Add RPATHS outside of the build tree to install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

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
list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O3")
SET(CUDA_PROPAGATE_HOST_FLAGS ON)
find_package(IlmBase REQUIRED COMPONENTS Half)
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
list(APPEND afx_PLUGINS afx_soft_clip)

add_library(afx_tone_map SHARED src/afx_tone_map.cpp)
set_target_properties(afx_tone_map PROPERTIES PREFIX "")
list(APPEND afx_PLUGINS afx_tone_map)

add_library(afx_noise_map SHARED src/afx_noise_map.cpp)
set_target_properties(afx_noise_map PROPERTIES PREFIX "")
target_link_libraries(afx_noise_map core)
add_dependencies(afx_noise_map core)
list(APPEND afx_PLUGINS afx_noise_map)

add_library(afx_chroma_key SHARED src/afx_chroma_key.cpp)
set_target_properties(afx_chroma_key PROPERTIES PREFIX "")
target_link_libraries(afx_chroma_key core)
add_dependencies(afx_chroma_key core)
list(APPEND afx_PLUGINS afx_chroma_key)

add_library(afx_anti_alias SHARED src/afx_anti_alias.cpp src/mlaa.cpp)
set_target_properties(afx_anti_alias PROPERTIES PREFIX "")
target_link_libraries(afx_anti_alias core)
add_dependencies(afx_anti_alias core)
list(APPEND afx_PLUGINS afx_anti_alias)

add_library(afx_despill SHARED src/afx_despill.cpp)
set_target_properties(afx_despill PROPERTIES PREFIX "")
target_link_libraries(afx_despill core)
add_dependencies(afx_despill core)
list(APPEND afx_PLUGINS afx_despill)

cuda_add_library(afx_median src/afx_median.cpp src/afx_median.cu SHARED)
set_target_properties(afx_median PROPERTIES PREFIX "")
target_link_libraries(afx_median core cuda)
add_dependencies(afx_median core cuda)
list(APPEND afx_PLUGINS afx_median)

add_library(afx_glow SHARED src/afx_glow.cpp)
set_target_properties(afx_glow PROPERTIES PREFIX "")
target_link_libraries(afx_glow core cuda)
add_dependencies(afx_glow core cuda)
list(APPEND afx_PLUGINS afx_glow)


install(
  TARGETS ${afx_LIBRARIES}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  )
install(
  TARGETS ${afx_PLUGINS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  )
install(
  DIRECTORY icons/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/icons
  )
install(
  FILES menu.py
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  )
