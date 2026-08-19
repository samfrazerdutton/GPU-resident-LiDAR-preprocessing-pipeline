#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "gpulidar/cuda/device_buffer.hpp"
#include "gpulidar/cuda/pillar_scatter.hpp"
#include "gpulidar/kitti_loader.hpp"

using namespace gpulidar;

namespace {

bool haveGpu() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// CPU reference: which cells are occupied, and how many points each holds.
std::map<std::pair<int, int>, int> cpuCells(const std::vector<PointXYZIRT>& in,
                                            const cuda::PillarConfig& c) {
  std::map<std::pair<int, int>, int> m;
  for (const auto& p : in) {
    if (p.x < c.x_min || p.x >= c.x_max) continue;
    if (p.y < c.y_min || p.y >= c.y_max) continue;
    if (p.z < c.z_min || p.z >= c.z_max) continue;
    const int gx = (int)((p.x - c.x_min) / c.pillar_x);
    const int gy = (int)((p.y - c.y_min) / c.pillar_y);
    if (gx < 0 || gx >= c.gridW() || gy < 0 || gy >= c.gridH()) continue;
    ++m[{gx, gy}];
  }
  return m;
}

}  // namespace

TEST(GpuPillar, PillarSetMatchesCpuReference) {
  if (!haveGpu()) GTEST_SKIP() << "no CUDA device";

  auto cloud = makeSyntheticSweep(150000);
  for (auto& p : cloud) p.x = std::fabs(p.x);  // push points into the +x ROI

  cuda::PillarConfig cfg;
  cfg.max_pillars = 120000;  // synthetic clouds occupy far more pillars than real sweeps
  const auto expected = cpuCells(cloud, cfg);
  ASSERT_GT(expected.size(), 100u) << "test cloud does not populate the ROI";

  cuda::DeviceBuffer<PointXYZIRT> d_in(cloud.size());
  d_in.upload(cloud.data(), cloud.size());
  cuda::PillarScatter ps(static_cast<int>(cloud.size()), cfg);
  const int np = ps.process(d_in.data(), static_cast<int>(cloud.size()));
  ASSERT_LT(np, cfg.max_pillars) << "pillar cap bound; count would be clamped";
  ASSERT_EQ(np, static_cast<int>(expected.size()));

  std::vector<int> coords(np * 2);
  cudaMemcpy(coords.data(), ps.coords(), np * 2 * sizeof(int), cudaMemcpyDeviceToHost);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::set<std::pair<int, int>> got;
  for (int i = 0; i < np; ++i) got.insert({coords[i * 2], coords[i * 2 + 1]});
  ASSERT_EQ(got.size(), (size_t)np) << "duplicate pillar coordinates";

  for (const auto& kv : expected) {
    EXPECT_TRUE(got.count(kv.first)) << "missing pillar " << kv.first.first << ","
                                     << kv.first.second;
  }
}

TEST(GpuPillar, FeaturesAreConsistentWithStoredPoints) {
  if (!haveGpu()) GTEST_SKIP() << "no CUDA device";

  auto cloud = makeSyntheticSweep(150000);
  for (auto& p : cloud) p.x = std::fabs(p.x);

  cuda::PillarConfig cfg;
  cfg.max_pillars = 120000;
  cuda::DeviceBuffer<PointXYZIRT> d_in(cloud.size());
  d_in.upload(cloud.data(), cloud.size());
  cuda::PillarScatter ps(static_cast<int>(cloud.size()), cfg);
  const int np = ps.process(d_in.data(), static_cast<int>(cloud.size()));
  ASSERT_GT(np, 0);
  ASSERT_LT(np, cfg.max_pillars);

  const int N = cfg.max_points_per_pillar;
  const int check = std::min(np, 512);

  std::vector<PointXYZIRT> pts((size_t)check * N);
  std::vector<float> feats((size_t)check * N * 9);
  std::vector<unsigned int> counts(check);
  std::vector<int> coords(check * 2);
  cudaMemcpy(pts.data(), ps.pillarPoints(), pts.size() * sizeof(PointXYZIRT),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(feats.data(), ps.features(), feats.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(counts.data(), ps.pointCounts(), counts.size() * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(coords.data(), ps.coords(), coords.size() * sizeof(int),
             cudaMemcpyDeviceToHost);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Which points landed in a full pillar is nondeterministic, so recompute the
  // expected features from the points the GPU actually kept.
  double max_err = 0.0;
  int padded = 0;
  for (int p = 0; p < check; ++p) {
    const int kept = std::min<int>(counts[p], N);
    double sx = 0, sy = 0, sz = 0;
    for (int j = 0; j < kept; ++j) {
      const auto& q = pts[(size_t)p * N + j];
      sx += q.x; sy += q.y; sz += q.z;
    }
    const double cx = sx / kept, cy = sy / kept, cz = sz / kept;
    const double px = cfg.x_min + (coords[p * 2] + 0.5) * cfg.pillar_x;
    const double py = cfg.y_min + (coords[p * 2 + 1] + 0.5) * cfg.pillar_y;

    for (int j = 0; j < N; ++j) {
      const float* f = &feats[((size_t)p * N + j) * 9];
      if (j >= kept) {
        for (int k = 0; k < 9; ++k) EXPECT_FLOAT_EQ(f[k], 0.0f) << "padding not zeroed";
        ++padded;
        continue;
      }
      const auto& q = pts[(size_t)p * N + j];
      // Everything promoted to double explicitly: mixing float and double in
      // an initializer list breaks std::max's template deduction.
      auto note = [&](double a, double b) { max_err = std::max(max_err, std::fabs(a - b)); };
      note(f[0], q.x);
      note(f[3], q.intensity);
      note(f[4], (double)q.x - cx);
      note(f[6], (double)q.z - cz);
      note(f[7], (double)q.x - px);
      note(f[8], (double)q.y - py);
    }
  }
  EXPECT_LT(max_err, 1e-3) << "feature encoding divergence: " << max_err;
  std::printf("checked %d pillars, %d padded slots, max feature err %.3e\n", check, padded,
              max_err);
}

TEST(GpuPillar, ReusableAcrossFrames) {
  if (!haveGpu()) GTEST_SKIP() << "no CUDA device";

  auto cloud = makeSyntheticSweep(120000);
  for (auto& p : cloud) p.x = std::fabs(p.x);

  cuda::PillarConfig cfg;
  cfg.max_pillars = 120000;
  cuda::DeviceBuffer<PointXYZIRT> d_in(cloud.size());
  d_in.upload(cloud.data(), cloud.size());
  cuda::PillarScatter ps(static_cast<int>(cloud.size()), cfg);

  const int first = ps.process(d_in.data(), static_cast<int>(cloud.size()));
  for (int frame = 1; frame < 5; ++frame) {
    EXPECT_EQ(ps.process(d_in.data(), static_cast<int>(cloud.size())), first)
        << "state leaked at frame " << frame;
  }
}
