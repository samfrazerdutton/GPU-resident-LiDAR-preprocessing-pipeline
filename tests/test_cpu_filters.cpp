#include <gtest/gtest.h>

#include <cmath>
#include "gpulidar/kitti_loader.hpp"
#include "gpulidar/reference/cpu_filters.hpp"

using namespace gpulidar;

TEST(CropBox, DropsOutOfRangeAndEgoPoints) {
  std::vector<PointXYZIRT> in(3);
  in[0] = {10.0f, 0.0f, 0.0f, 0.5f, 0, 0, 0.0f, 0.0f};   // keep
  in[1] = {500.0f, 0.0f, 0.0f, 0.5f, 0, 0, 0.0f, 0.0f};  // out of ROI
  in[2] = {0.5f, 0.0f, -1.0f, 0.5f, 0, 0, 0.0f, 0.0f};   // inside ego box

  const auto out = reference::cropBox(in, CropBoxConfig{});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FLOAT_EQ(out[0].x, 10.0f);
}

TEST(VoxelDownsample, MergesPointsInSameCellAndAveragesCentroid) {
  VoxelGridConfig cfg;
  cfg.leaf_x = cfg.leaf_y = cfg.leaf_z = 1.0f;

  std::vector<PointXYZIRT> in(2);
  in[0] = {10.2f, 0.2f, 0.2f, 0.0f, 3, 0, 0.01f, 0.0f};
  in[1] = {10.4f, 0.4f, 0.4f, 1.0f, 7, 0, 0.03f, 0.0f};

  const auto out = reference::voxelDownsample(in, cfg);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NEAR(out[0].x, 10.3f, 1e-5f);
  EXPECT_NEAR(out[0].intensity, 0.5f, 1e-5f);
  EXPECT_NEAR(out[0].time, 0.01f, 1e-6f);  // earliest point in the cell
}

TEST(VoxelDownsample, IsDeterministicAndIdempotentInSize) {
  const auto cloud = makeSyntheticSweep(50000);
  const auto a = reference::voxelDownsample(cloud, VoxelGridConfig{});
  const auto b = reference::voxelDownsample(cloud, VoxelGridConfig{});

  ASSERT_EQ(a.size(), b.size());
  EXPECT_LT(a.size(), cloud.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_FLOAT_EQ(a[i].x, b[i].x) << "ordering not deterministic at " << i;
  }
}

TEST(SyntheticSweep, RingsStayInHdl64eRange) {
  const auto cloud = makeSyntheticSweep(10000);
  for (const auto& p : cloud) EXPECT_LE(p.ring, 63);
}
