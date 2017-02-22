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

# Set CMAKE_SKIP_RPATH TRUE for redistributable build, place all
# redistributable library dependencies in lib folder
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'$ORIGIN/lib'")
# Disable rpath stripping at install
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -DNDEBUG")
endif()

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
find_package(IlmBase REQUIRED COMPONENTS Half)
find_package(Jemalloc REQUIRED)

# AFX CORE LIBRARY
add_library(lib_afx_core SHARED src/image.cpp src/threading.cpp src/mlaa.cpp)
set_target_properties(lib_afx_core PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "./lib")
target_link_libraries(lib_afx_core ${Ipp_LIBRARIES} ${Boost_LIBRARIES} ${IlmBase_LIBRARIES} ${Jemalloc_LIBRARIES})

# Cuda Helper Library
add_library(lib_afx_cuda SHARED src/cuda_helper.cpp)
set_target_properties(lib_afx_cuda PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "./lib")
target_link_libraries(lib_afx_cuda lib_afx_core)

add_library(afx_soft_clip SHARED src/afx_soft_clip.cpp)
set_target_properties(afx_soft_clip PROPERTIES PREFIX "")
target_link_libraries(afx_soft_clip)

add_library(afx_tone_map SHARED src/afx_tone_map.cpp)
set_target_properties(afx_tone_map PROPERTIES PREFIX "")
target_link_libraries(afx_tone_map)

add_library(afx_chroma_key SHARED src/afx_chroma_key.cpp)
set_target_properties(afx_chroma_key PROPERTIES PREFIX "")
target_link_libraries(afx_chroma_key lib_afx_core)

add_library(afx_anti_alias SHARED src/afx_anti_alias.cpp)
set_target_properties(afx_anti_alias PROPERTIES PREFIX "")
target_link_libraries(afx_anti_alias lib_afx_core)

add_library(afx_despill SHARED src/afx_despill.cpp)
set_target_properties(afx_despill PROPERTIES PREFIX "")
target_link_libraries(afx_despill lib_afx_core)

cuda_add_library(afx_median src/afx_median.cpp src/afx_median.cu SHARED)
set_target_properties(afx_median PROPERTIES PREFIX "")
target_link_libraries(afx_median lib_afx_core lib_afx_cuda)

cuda_add_library(afx_glow src/afx_glow.cpp SHARED)
set_target_properties(afx_glow PROPERTIES PREFIX "")
target_link_libraries(afx_glow lib_afx_core lib_afx_cuda)


install(TARGETS lib_afx_core lib_afx_cuda DESTINATION lib)
install(TARGETS afx_soft_clip afx_tone_map afx_median afx_chroma_key afx_despill afx_anti_alias afx_glow DESTINATION .)
install(DIRECTORY icons/ DESTINATION icons)
install(FILES menu.py DESTINATION .)
