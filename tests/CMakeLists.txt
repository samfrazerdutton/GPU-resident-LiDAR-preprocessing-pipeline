include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_cpu_filters test_cpu_filters.cpp)
target_link_libraries(test_cpu_filters PRIVATE gpulidar_core GTest::gtest_main)
add_test(NAME cpu_filters COMMAND test_cpu_filters)

add_executable(test_undistort test_undistort.cpp)
target_link_libraries(test_undistort PRIVATE gpulidar_core GTest::gtest_main)
add_test(NAME undistort COMMAND test_undistort)

add_executable(test_gpu_undistort test_gpu_undistort.cu)
target_link_libraries(test_gpu_undistort PRIVATE gpulidar_cuda GTest::gtest_main)
add_test(NAME gpu_undistort COMMAND test_gpu_undistort)

add_executable(test_gpu_crop test_gpu_crop.cu)
target_link_libraries(test_gpu_crop PRIVATE gpulidar_cuda GTest::gtest_main)
add_test(NAME gpu_crop COMMAND test_gpu_crop)

add_executable(test_gpu_voxel test_gpu_voxel.cu)
target_link_libraries(test_gpu_voxel PRIVATE gpulidar_cuda GTest::gtest_main)
add_test(NAME gpu_voxel COMMAND test_gpu_voxel)

add_executable(test_gpu_pillar test_gpu_pillar.cu)
target_link_libraries(test_gpu_pillar PRIVATE gpulidar_cuda GTest::gtest_main)
add_test(NAME gpu_pillar COMMAND test_gpu_pillar)
