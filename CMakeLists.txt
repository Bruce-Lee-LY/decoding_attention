# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: cmake for cpp decoding attention

cmake_minimum_required (VERSION 3.16)

project (decoding_attention LANGUAGES C CXX CUDA)

if (POLICY CMP0146)
    cmake_policy (SET CMP0146 OLD) 
endif ()

find_program (CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property (GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property (GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

set (CMAKE_VERBOSE_MAKEFILE ${DA_VERBOSE_MAKEFILE})

set (CMAKE_C_FLAGS "-std=c17")
set (CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O0 -g2 -ggdb")
set (CMAKE_C_FLAGS_RELEASE "$ENV{CFLAGS} -O3")

set (CMAKE_CXX_FLAGS "-std=c++17")
set (CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb")
set (CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

set (CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")

add_compile_options (
    -Wall
    -Werror
    -Wextra
    -Wswitch-default
    # -Wfloat-equal
    -Wshadow
    -Wcast-qual
)

# Nvidia GPU
find_package (CUDA REQUIRED)
# unset (CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
# option (CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set (CUDA_VERBOSE_BUILD ${DA_VERBOSE_MAKEFILE})
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -Xcompiler -fopenmp --expt-relaxed-constexpr")
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES} -g -lineinfo -Xptxas=-v -O0")
else ()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES} --use_fast_math -O3")
endif ()

set (SYS_CUDART_PATH "/usr/local/cuda")
set (SYS_CUDA_DRIVER_PATH "/usr/lib/x86_64-linux-gnu")

find_package(gflags REQUIRED)
find_package(OpenMP REQUIRED)

include_directories (
    ${PROJECT_SOURCE_DIR}/csrc/common
    ${PROJECT_SOURCE_DIR}/csrc/kernel
    ${PROJECT_SOURCE_DIR}/csrc/ops
    ${SYS_CUDART_PATH}/include
    ${GFLAGS_INCLUDE_DIR}
)

link_directories (
    ${SYS_CUDART_PATH}/lib64
    ${SYS_CUDA_DRIVER_PATH}
)

file (GLOB DA_SRCS 
    ${PROJECT_SOURCE_DIR}/csrc/kernel/decoding_attn/*.cu
    ${PROJECT_SOURCE_DIR}/csrc/ops/*.cpp
    ${PROJECT_SOURCE_DIR}/benchmarks/cpp/*.cpp
)

cuda_add_executable (benchmark_decoding_attn ${DA_SRCS})
target_link_libraries (benchmark_decoding_attn OpenMP::OpenMP_CXX ${GFLAGS_LIBRARIES})

install (TARGETS benchmark_decoding_attn RUNTIME DESTINATION bin)
