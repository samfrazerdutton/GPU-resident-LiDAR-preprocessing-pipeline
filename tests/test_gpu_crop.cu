#include <gtest/gtest.h>

#include <algorithm>
#include <tuple>
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

void sortPoints(std::vector<PointXYZIRT>& v) {
  std::sort(v.begin(), v.end(), [](const PointXYZIRT& a, const PointXYZIRT& b) {
    return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z);
  });
}

std::vector<PointXYZIRT> runGpuCrop(const std::vector<PointXYZIRT>& host_in,
                                    bool aggregated, unsigned int* out_count) {
  cuda::DeviceBuffer<PointXYZIRT> d_in(host_in.size());
  cuda::DeviceBuffer<PointXYZIRT> d_out(host_in.size());
  cuda::DeviceBuffer<unsigned int> d_count(1);
  d_in.upload(host_in.data(), host_in.size());
  d_out.resize(host_in.size());
  d_count.resize(1);
  cudaMemset(d_count.data(), 0, sizeof(unsigned int));

  const int n = static_cast<int>(host_in.size());
  if (aggregated) {
    cuda::launchCropBoxWarpAggregated(d_in.data(), d_out.data(), n, CropBoxConfig{},
                                      d_count.data());
  } else {
    cuda::launchCropBoxNaive(d_in.data(), d_out.data(), n, CropBoxConfig{},
                             d_count.data());
  }

  unsigned int count = 0;
  cudaMemcpy(&count, d_count.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  std::vector<PointXYZIRT> got(count);
  if (count > 0) {
    cudaMemcpy(got.data(), d_out.data(), count * sizeof(PointXYZIRT),
               cudaMemcpyDeviceToHost);
  }
  cudaDeviceSynchronize();
  *out_count = count;
  return got;
}

}  // namespace

TEST(GpuCrop, NaiveMatchesCpuReferenceAsSet) {
  if (!haveGpu()) GTEST_SKIP() << "no CUDA device";

  const auto host_in = makeSyntheticSweep(200000);
  auto expected = reference::cropBox(host_in, CropBoxConfig{});

  unsigned int count = 0;
  auto got = runGpuCrop(host_in, false, &count);
  ASSERT_EQ(count, expected.size());

  sortPoints(expected);
  sortPoints(got);
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_FLOAT_EQ(got[i].x, expected[i].x) << "at " << i;
    EXPECT_FLOAT_EQ(got[i].y, expected[i].y) << "at " << i;
    EXPECT_FLOAT_EQ(got[i].z, expected[i].z) << "at " << i;
  }
}

TEST(GpuCrop, WarpAggregatedMatchesCpuReferenceAsSet) {
  if (!haveGpu()) GTEST_SKIP() << "no CUDA device";

  const auto host_in = makeSyntheticSweep(200000);
  auto expected = reference::cropBox(host_in, CropBoxConfig{});

  unsigned int count = 0;
  auto got = runGpuCrop(host_in, true, &count);
  ASSERT_EQ(count, expected.size());

  sortPoints(expected);
  sortPoints(got);
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_FLOAT_EQ(got[i].x, expected[i].x) << "at " << i;
    EXPECT_FLOAT_EQ(got[i].y, expected[i].y) << "at " << i;
    EXPECT_FLOAT_EQ(got[i].z, expected[i].z) << "at " << i;
  }
}

TEST(GpuCrop, BothVariantsAgreeOnCountForAwkwardSizes) {
  if (!haveGpu()) GTEST_SKIP() << "no CUDA device";
  // Sizes that are not multiples of the warp or block size exercise the
  // tail handling in the ballot loop.
  for (int n : {1, 31, 33, 255, 257, 1023, 100001}) {
    const auto host_in = makeSyntheticSweep(n);
    unsigned int c_naive = 0, c_agg = 0;
    runGpuCrop(host_in, false, &c_naive);
    runGpuCrop(host_in, true, &c_agg);
    EXPECT_EQ(c_naive, c_agg) << "disagreement at n=" << n;
    EXPECT_EQ(c_naive, reference::cropBox(host_in, CropBoxConfig{}).size())
        << "wrong count at n=" << n;
  }
}
