#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "gpulidar/cuda/check.hpp"
#include "gpulidar/cuda/device_buffer.hpp"
#include "gpulidar/cuda/pillar_scatter.hpp"
#include "gpulidar/kitti_loader.hpp"
#include "gpulidar/oxts.hpp"  // listFiles

using namespace gpulidar;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_pillar <drive_sync_dir> [repeats]\n");
    return 2;
  }
  const std::string drive = argv[1];
  const int repeats = (argc >= 3) ? std::stoi(argv[2]) : 8;

  const auto bins = listFiles(drive + "/velodyne_points/data", ".bin");
  if (bins.empty()) { std::fprintf(stderr, "no frames found\n"); return 1; }

  std::vector<std::vector<PointXYZIRT>> clouds;
  std::size_t max_points = 0;
  for (const auto& b : bins) {
    clouds.push_back(loadVelodyneBin(b));
    max_points = std::max(max_points, clouds.back().size());
  }
  std::printf("loaded %zu frames, max %zu points\n", clouds.size(), max_points);

  cuda::PillarConfig cfg;
  std::printf("grid %d x %d = %d cells, cap %d pillars x %d points\n", cfg.gridW(),
              cfg.gridH(), cfg.gridW() * cfg.gridH(), cfg.max_pillars,
              cfg.max_points_per_pillar);

  cuda::DeviceBuffer<PointXYZIRT> d_in(static_cast<int>(max_points));
  cuda::PillarScatter ps(static_cast<int>(max_points), cfg);

  cudaEvent_t start, stop;
  GPULIDAR_CUDA_CHECK(cudaEventCreate(&start));
  GPULIDAR_CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < 20; ++i) {
    d_in.upload(clouds[0].data(), clouds[0].size());
    ps.process(d_in.data(), static_cast<int>(clouds[0].size()));
  }

  std::vector<float> ms;
  std::vector<int> pillars;
  int clamped = 0;

  for (int r = 0; r < repeats; ++r) {
    for (const auto& c : clouds) {
      d_in.upload(c.data(), c.size());
      GPULIDAR_CUDA_CHECK(cudaDeviceSynchronize());

      GPULIDAR_CUDA_CHECK(cudaEventRecord(start));
      const int np = ps.process(d_in.data(), static_cast<int>(c.size()));
      GPULIDAR_CUDA_CHECK(cudaEventRecord(stop));
      GPULIDAR_CUDA_CHECK(cudaEventSynchronize(stop));
      float t = 0.0f;
      GPULIDAR_CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
      ms.push_back(t);
      pillars.push_back(ps.lastRawPillars());
      if (ps.lastOverflowed()) ++clamped;
    }
  }

  std::sort(ms.begin(), ms.end());
  auto pct = [&](double p) {
    return ms[std::min<size_t>(ms.size() - 1, (size_t)(p * ms.size()))] * 1000.0;
  };
  auto minmax = std::minmax_element(pillars.begin(), pillars.end());
  double mean = 0.0;
  for (int p : pillars) mean += p;
  mean /= pillars.size();

  std::printf("\nframes measured : %zu\n", ms.size());
  std::printf("pillars/frame   : min %d  mean %.0f  max %d\n", *minmax.first, mean,
              *minmax.second);
  std::printf("frames clamped  : %d (%.1f%%)\n", clamped, 100.0 * clamped / pillars.size());
  std::printf("\n  p50   %8.1f us\n", pct(0.50));
  std::printf("  p99   %8.1f us\n", pct(0.99));
  std::printf("  p99.9 %8.1f us\n", pct(0.999));
  // Same work without the count readback, to isolate the sync cost.
  std::vector<float> nosync;
  for (int r = 0; r < 2; ++r) {
    for (const auto& c : clouds) {
      d_in.upload(c.data(), c.size());
      GPULIDAR_CUDA_CHECK(cudaDeviceSynchronize());
      GPULIDAR_CUDA_CHECK(cudaEventRecord(start));
      ps.process(d_in.data(), static_cast<int>(c.size()), nullptr, false);
      GPULIDAR_CUDA_CHECK(cudaEventRecord(stop));
      GPULIDAR_CUDA_CHECK(cudaEventSynchronize(stop));
      float t = 0.0f;
      GPULIDAR_CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
      nosync.push_back(t);
    }
  }
  std::sort(nosync.begin(), nosync.end());
  std::printf("\nwithout count readback: p50 %8.1f us (sync costs %.1f us)\n",
              nosync[nosync.size() / 2] * 1000.0,
              pct(0.50) - nosync[nosync.size() / 2] * 1000.0);
  return 0;
}
