#include <gtest/gtest.h>

#include <cmath>
#include "gpulidar/kitti_loader.hpp"
#include "gpulidar/reference/cpu_filters.hpp"

using namespace gpulidar;

namespace {
PointXYZIRT mk(float x, float y, float z, float t) {
  PointXYZIRT p{};
  p.x = x; p.y = y; p.z = z;
  p.intensity = 0.5f;
  p.ring = 10;
  p.time = t;
  return p;
}
}  // namespace

TEST(Undistort, ZeroTwistIsIdentity) {
  const auto in = makeSyntheticSweep(1000);
  const auto out = reference::undistort(in, EgoTwist{});
  ASSERT_EQ(in.size(), out.size());
  for (size_t i = 0; i < in.size(); ++i) {
    EXPECT_NEAR(in[i].x, out[i].x, 1e-6f);
    EXPECT_NEAR(in[i].y, out[i].y, 1e-6f);
  }
}

TEST(Undistort, PureForwardMotionShiftsByVelocityTimesTime) {
  EgoTwist tw;
  tw.vf = 10.0f;

  std::vector<PointXYZIRT> in{mk(20.0f, 0.0f, 0.0f, 0.0f),
                              mk(20.0f, 0.0f, 0.0f, 0.05f)};
  const auto out = reference::undistort(in, tw);

  EXPECT_NEAR(out[0].x, 20.0f, 1e-5f);
  EXPECT_NEAR(out[1].x, 20.5f, 1e-4f);
  EXPECT_NEAR(out[1].y, 0.0f, 1e-5f);
}

TEST(Undistort, PureYawRotatesAboutOrigin) {
  EgoTwist tw;
  tw.wu = 0.5f;

  const float t = 0.08f;
  const float theta = tw.wu * t;
  std::vector<PointXYZIRT> in{mk(10.0f, 0.0f, 1.0f, t)};
  const auto out = reference::undistort(in, tw);

  EXPECT_NEAR(out[0].x, 10.0f * std::cos(theta), 1e-4f);
  EXPECT_NEAR(out[0].y, 10.0f * std::sin(theta), 1e-4f);
  EXPECT_NEAR(out[0].z, 1.0f, 1e-6f);
}

TEST(Undistort, PreservesPointCountAndAttributes) {
  EgoTwist tw;
  tw.vf = 15.0f;
  tw.wu = 0.1f;
  const auto in = makeSyntheticSweep(5000);
  const auto out = reference::undistort(in, tw);
  ASSERT_EQ(in.size(), out.size());
  for (size_t i = 0; i < in.size(); ++i) {
    EXPECT_EQ(in[i].ring, out[i].ring);
    EXPECT_FLOAT_EQ(in[i].intensity, out[i].intensity);
    EXPECT_FLOAT_EQ(in[i].time, out[i].time);
  }
}
