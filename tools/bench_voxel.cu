#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "gpulidar/cuda/check.hpp"
#include "gpulidar/cuda/device_buffer.hpp"
#include "gpulidar/cuda/voxel_grid.hpp"
#include "gpulidar/kitti_loader.hpp"
#include "gpulidar/reference/cpu_filters.hpp"

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

  std::vector<float> ms;
  ms.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    GPULIDAR_CUDA_CHECK(cudaEventRecord(start));
    grid.process(d_in.data(), n, d_out.data(), cfg);
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

  std::printf("points in     : %d\n", n);
  std::printf("leaf size     : %.2f m\n", cfg.leaf_x);
  std::printf("cells out     : %d (%.1fx reduction)\n", cells, (double)n / cells);
  std::printf("table capacity: %d slots\n", grid.tableCapacity());
  std::printf("p50           : %8.1f us\n", pct(0.50));
  std::printf("p99           : %8.1f us\n", pct(0.99));
  std::printf("p99.9         : %8.1f us\n", pct(0.999));
  std::printf("max           : %8.1f us\n", ms.back() * 1000.0);
  std::printf("\nnote: p50 includes 4 launches + a device-to-host count read.\n");
  return 0;
}
