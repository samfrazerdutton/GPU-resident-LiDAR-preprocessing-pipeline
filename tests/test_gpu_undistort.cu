#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gpulidar/cuda/device_buffer.hpp"
#include "gpulidar/cuda/kernels.hpp"
#include "gpulidar/kitti_loader.hpp"
#include "gpulidar/reference/cpu_filters.hpp"

using namespace gpulidar;

namespace {
bool haveGpu() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
}  // namespace

TEST(GpuUndistort, MatchesCpuReference) {
  if (!haveGpu()) GTEST_SKIP() << "no CUDA device";

  const auto host_in = makeSyntheticSweep(200000);
  EgoTwist tw;
  tw.vf = 12.5f;
  tw.vl = 0.3f;
  tw.wu = 0.15f;

  const auto expected = reference::undistort(host_in, tw);

  cuda::DeviceBuffer<PointXYZIRT> d_in(host_in.size());
  cuda::DeviceBuffer<PointXYZIRT> d_out(host_in.size());
  d_in.upload(host_in.data(), host_in.size());
  d_out.resize(host_in.size());

  cuda::launchUndistort(d_in.data(), d_out.data(),
                        static_cast<int>(host_in.size()), tw);

  std::vector<PointXYZIRT> got(host_in.size());
  d_out.download(got.data(), got.size());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  double max_err = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    max_err = std::max({max_err,
                        (double)std::fabs(got[i].x - expected[i].x),
                        (double)std::fabs(got[i].y - expected[i].y),
                        (double)std::fabs(got[i].z - expected[i].z)});
    EXPECT_EQ(got[i].ring, expected[i].ring);
  }
  // 1e-4 m = 0.1 mm. Two orders of magnitude below LiDAR range noise.
  EXPECT_LT(max_err, 1e-4) << "max GPU/CPU divergence: " << max_err << " m";
  std::printf("max GPU/CPU divergence: %.3e m\n", max_err);
}

TEST(DeviceBuffer, ResizePastCapacityThrows) {
  if (!haveGpu()) GTEST_SKIP() << "no CUDA device";
  cuda::DeviceBuffer<PointXYZIRT> buf(1000);
  EXPECT_NO_THROW(buf.resize(1000));
  EXPECT_THROW(buf.resize(1001), std::runtime_error);
}
