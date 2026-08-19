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
