#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include "gpulidar/cuda/device_buffer.hpp"
#include "gpulidar/cuda/voxel_grid.hpp"
#include "gpulidar/kitti_loader.hpp"
#include "gpulidar/reference/cpu_filters.hpp"

using namespace gpulidar;

namespace {

bool haveGpu() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void sortPoints(std::vector<PointXYZIRT>& v) {
  std::sort(v.begin(), v.end(), [](const PointXYZIRT& a, const PointXYZIRT& b) {
    return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z);
  });
}

}  // namespace

TEST(GpuVoxel, MatchesCpuReferenceAsSet) {
  if (!haveGpu()) GTEST_SKIP() << "no CUDA device";

  const auto host_in = makeSyntheticSweep(200000);
  VoxelGridConfig cfg;
  auto expected = reference::voxelDownsample(host_in, cfg);

  cuda::DeviceBuffer<PointXYZIRT> d_in(host_in.size()), d_out(host_in.size());
  d_in.upload(host_in.data(), host_in.size());
  d_out.resize(host_in.size());

  cuda::VoxelGrid grid(static_cast<int>(host_in.size()));
  const int count = grid.process(d_in.data(), static_cast<int>(host_in.size()),
                                 d_out.data(), cfg);
  ASSERT_EQ(count, static_cast<int>(expected.size()));

  std::vector<PointXYZIRT> got(count);
  cudaMemcpy(got.data(), d_out.data(), count * sizeof(PointXYZIRT), cudaMemcpyDeviceToHost);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  sortPoints(expected);
  sortPoints(got);

  // fp32 atomic accumulation vs fp64 CPU sums: 1e-3 m = 1 mm, far below
  // LiDAR range noise. Bitwise equality is not achievable here by design.
  double max_err = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    max_err = std::max({max_err,
                        (double)std::fabs(got[i].x - expected[i].x),
                        (double)std::fabs(got[i].y - expected[i].y),
                        (double)std::fabs(got[i].z - expected[i].z)});
    EXPECT_EQ(got[i].ring, expected[i].ring) << "at " << i;
  }
  EXPECT_LT(max_err, 1e-3) << "max centroid divergence: " << max_err << " m";
  std::printf("max centroid divergence: %.3e m over %d cells\n", max_err, count);
}

TEST(GpuVoxel, TableIsReusableAcrossFrames) {
  if (!haveGpu()) GTEST_SKIP() << "no CUDA device";

  const auto host_in = makeSyntheticSweep(120000);
  VoxelGridConfig cfg;
  const auto expected = reference::voxelDownsample(host_in, cfg);

  cuda::DeviceBuffer<PointXYZIRT> d_in(host_in.size()), d_out(host_in.size());
  d_in.upload(host_in.data(), host_in.size());
  d_out.resize(host_in.size());

  cuda::VoxelGrid grid(static_cast<int>(host_in.size()));
  // Same input five times: if the per-frame reset is wrong, counts drift.
  for (int frame = 0; frame < 5; ++frame) {
    const int count = grid.process(d_in.data(), static_cast<int>(host_in.size()),
                                   d_out.data(), cfg);
    EXPECT_EQ(count, static_cast<int>(expected.size())) << "drift at frame " << frame;
  }
}
