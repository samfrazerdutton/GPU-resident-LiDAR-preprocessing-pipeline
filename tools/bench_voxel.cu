#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "gpulidar/cuda/check.hpp"
#include "gpulidar/cuda/device_buffer.hpp"
#include "gpulidar/cuda/voxel_grid.hpp"
#include "gpulidar/kitti_loader.hpp"

using namespace gpulidar;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_voxel <velodyne.bin> [iters] [leaf_m]\n");
    return 2;
  }
  const int iters = (argc >= 3) ? std::stoi(argv[2]) : 2000;
  VoxelGridConfig cfg;
  if (argc >= 4) cfg.leaf_x = cfg.leaf_y = cfg.leaf_z = std::stof(argv[3]);

  const auto cloud = loadVelodyneBin(argv[1]);
  const int n = static_cast<int>(cloud.size());

  cuda::DeviceBuffer<PointXYZIRT> d_in(n), d_out(n);
  d_in.upload(cloud.data(), n);
  d_out.resize(n);
  cuda::VoxelGrid grid(n);
  GPULIDAR_CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  GPULIDAR_CUDA_CHECK(cudaEventCreate(&start));
  GPULIDAR_CUDA_CHECK(cudaEventCreate(&stop));

  int cells = 0;
  for (int i = 0; i < 50; ++i) cells = grid.process(d_in.data(), n, d_out.data(), cfg);

  // --- Mode A: sync every frame (host reads the count) ---
  std::vector<float> ms;
  ms.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    GPULIDAR_CUDA_CHECK(cudaEventRecord(start));
    grid.process(d_in.data(), n, d_out.data(), cfg, nullptr, true);
    GPULIDAR_CUDA_CHECK(cudaEventRecord(stop));
    GPULIDAR_CUDA_CHECK(cudaEventSynchronize(stop));
    float t = 0.0f;
    GPULIDAR_CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  auto pct = [&](double p) {
    return ms[std::min<size_t>(ms.size() - 1, (size_t)(p * ms.size()))] * 1000.0;
  };

  // --- Mode B: no per-frame sync, amortized over a batch ---
  const int batch = 100;
  const int rounds = std::max(1, iters / batch);
  std::vector<float> amort;
  amort.reserve(rounds);
  for (int r = 0; r < rounds; ++r) {
    GPULIDAR_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < batch; ++i) {
      grid.process(d_in.data(), n, d_out.data(), cfg, nullptr, false);
    }
    GPULIDAR_CUDA_CHECK(cudaEventRecord(stop));
    GPULIDAR_CUDA_CHECK(cudaEventSynchronize(stop));
    float t = 0.0f;
    GPULIDAR_CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
    amort.push_back(t / batch);
  }
  std::sort(amort.begin(), amort.end());
  const double amort_p50 = amort[amort.size() / 2] * 1000.0;

  std::printf("points in     : %d\n", n);
  std::printf("leaf size     : %.2f m\n", cfg.leaf_x);
  std::printf("cells out     : %d (%.1fx reduction)\n", cells, (double)n / cells);
  std::printf("\nA) sync per frame   p50 %8.1f us   p99 %8.1f   p99.9 %8.1f\n",
              pct(0.50), pct(0.99), pct(0.999));
  std::printf("B) no sync, amortized over %d   p50 %8.1f us\n", batch, amort_p50);
  std::printf("\nper-frame sync cost : %8.1f us (%.0f%% of mode A)\n",
              pct(0.50) - amort_p50, 100.0 * (pct(0.50) - amort_p50) / pct(0.50));
  return 0;
}
