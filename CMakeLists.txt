cmake_minimum_required(VERSION 3.22)
project(gpulidar LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# RTX 2060 Max-Q = Turing, SM 7.5. CUDA 13 dropped everything below this.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(GPULIDAR_BUILD_TESTS "Build unit tests" ON)

add_library(gpulidar_core
  src/kitti_loader.cpp
  src/oxts.cpp
  src/reference/cpu_filters.cpp
)
target_include_directories(gpulidar_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(gpulidar_core PRIVATE -Wall -Wextra -O3)

add_library(gpulidar_cuda
  src/cuda/undistort.cu
  src/cuda/crop_box.cu
)
target_include_directories(gpulidar_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gpulidar_cuda PUBLIC gpulidar_core)
# -lineinfo so Nsight can map stalls back to source lines. No fast-math:
# correctness first, then measure whether the intrinsics buy anything.
target_compile_options(gpulidar_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(device_info src/cuda/device_info.cu)
target_include_directories(device_info PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(kitti_probe tools/kitti_probe.cpp)
target_link_libraries(kitti_probe PRIVATE gpulidar_core)

add_executable(bench_bandwidth tools/bench_bandwidth.cu)
target_link_libraries(bench_bandwidth PRIVATE gpulidar_cuda)

add_executable(bench_crop tools/bench_crop.cu)
target_link_libraries(bench_crop PRIVATE gpulidar_cuda)

add_executable(bench_undistort tools/bench_undistort.cu)
target_link_libraries(bench_undistort PRIVATE gpulidar_cuda)

if(GPULIDAR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
