cmake_minimum_required(VERSION 2.6)
project(afx-nuke-plugins)

# Set CMAKE_SKIP_RPATH TRUE for redistributable build, place all
# redistributable library dependencies in lib folder
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'$ORIGIN/lib'")
#Disable rpath stripping at install
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -DNDEBUG")
set(CMAKE_BUILD_TYPE Release)

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
endif()

# afx include files
include_directories(${PROJECT_SOURCE_DIR}/include)

# NUKE NDK INCLUDES AND LIBRARIES
set(NUKE_ROOT /usr/local/Nuke10.5v1)
include_directories(${NUKE_ROOT}/include)
set(NUKE_LIB_PATH ${NUKE_ROOT})

# IPP INCLUDES AND LIBRARIES
set(IPP_ROOT /opt/intel/compilers_and_libraries/linux/ipp)
include_directories(${IPP_ROOT}/include)
set(IPP_LIB_PATH ${IPP_ROOT}/lib/intel64)
set(IPP_COMP_PATH ${IPP_ROOT}/../compiler/lib/intel64)
set(IPP_LIBS ${IPP_LIB_PATH}/libippi.so ${IPP_LIB_PATH}/libippcore.so)

# BOOST INCLUDES AND LIBRARIES
set(BOOST_ROOT /usr/local/boost)
include_directories(${BOOST_ROOT}/include)
set(BOOST_LIB_PATH ${BOOST_ROOT}/lib)
set(BOOST_LIBS ${BOOST_LIB_PATH}/libboost_system.so ${BOOST_LIB_PATH}/libboost_thread.so)

# CUDA INCLUDES AND link_libraries
find_package(CUDA REQUIRED)
set(CUDA_ROOT /usr/local/cuda)
include_directories(${CUDA_ROOT}/include)
set(CUDA_LIB_PATH ${CUDA_ROOT}/lib64)
set(CUDA_LIBS ${CUDA_LIB_PATH}/libcudart.so)
set(NPP_LIBS ${CUDA_LIB_PATH}/libnppi.so)
set(NPP_LIBS ${CUDA_LIB_PATH}/libcufft.so)
set(CUDA_VERBOSE_BUILD OFF)

# IlmBase
set(ILMBASE_ROOT /usr/local/IlmBase/)
include_directories(${ILMBASE_ROOT}/include)
set(ILMBASE_LIB_PATH ${ILMBASE_ROOT}/lib)
set(ILMBASE_LIBS ${ILMBASE_LIB_PATH}/libHalf.so)

# OpenEXR
#set(OPENEXR_ROOT /usr/local/IlmBase/)
#include_directories(${OPENEXR_ROOT}/include)
#set(OPENEXR_LIB_PATH ${OPENEXR_ROOT}/lib)
#set(OPENEXR_LIBS ${OPENEXR_LIB_PATH}/)

# JEMALLOC INCLUDES AND LIBRARIES
set(JEMALLOC_ROOT /usr/local/jemalloc/)
include_directories(${JEMALLOC_ROOT}/include)
set(JEMALLOC_LIB_PATH ${JEMALLOC_ROOT}/lib)
set(JEMALLOC_LIBS ${JEMALLOC_LIB_PATH}/libjemalloc.so)

# AFX CORE LIBRARY
add_library(lib_afx_core SHARED src/image.cpp src/threading.cpp src/mlaa.cpp)
set_target_properties(lib_afx_core PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "./lib")
target_link_libraries(lib_afx_core ${BOOST_LIBS} ${IPP_LIBS} ${JEMALLOC_LIBS})

# Cuda Helper Library
add_library(lib_afx_cuda SHARED src/cuda_helper.cpp)
set_target_properties(lib_afx_cuda PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "./lib")
target_link_libraries(lib_afx_cuda ${BOOST_LIBS} ${CUDA_LIBS} lib_afx_core)

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
target_link_libraries(afx_median ${CUDA_LIBS} lib_afx_core lib_afx_cuda)

cuda_add_library(afx_glow src/afx_glow.cpp SHARED)
set_target_properties(afx_glow PROPERTIES PREFIX "")
target_link_libraries(afx_glow ${CUDA_LIBS} lib_afx_core lib_afx_cuda)


install(TARGETS lib_afx_core lib_afx_cuda DESTINATION lib)
install(TARGETS afx_soft_clip afx_tone_map afx_median afx_chroma_key afx_despill afx_anti_alias afx_glow DESTINATION .)
install(DIRECTORY icons DESTINATION  icons)
install(FILES menu.py DESTINATION .)


