#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "gpulidar/cuda/check.hpp"
#include "gpulidar/cuda/device_buffer.hpp"
#include "gpulidar/cuda/kernels.hpp"
#include "gpulidar/kitti_loader.hpp"
#include "gpulidar/oxts.hpp"

using namespace gpulidar;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: bench_undistort <velodyne.bin> <oxts.txt> [iters]\n");
    return 2;
  }
  const int iters = (argc >= 4) ? std::stoi(argv[3]) : 1000;

  const auto cloud = loadVelodyneBin(argv[1]);
  const EgoTwist tw = loadOxts(argv[2]);
  const int n = static_cast<int>(cloud.size());

  cuda::DeviceBuffer<PointXYZIRT> d_in(n), d_out(n);
  d_in.upload(cloud.data(), n);
  d_out.resize(n);
  GPULIDAR_CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  GPULIDAR_CUDA_CHECK(cudaEventCreate(&start));
  GPULIDAR_CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < 50; ++i) cuda::launchUndistort(d_in.data(), d_out.data(), n, tw);
  GPULIDAR_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> ms;
  ms.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    GPULIDAR_CUDA_CHECK(cudaEventRecord(start));
    cuda::launchUndistort(d_in.data(), d_out.data(), n, tw);
    GPULIDAR_CUDA_CHECK(cudaEventRecord(stop));
    GPULIDAR_CUDA_CHECK(cudaEventSynchronize(stop));
    float t = 0.0f;
    GPULIDAR_CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());

  auto pct = [&](double p) { return ms[std::min<size_t>(ms.size() - 1, (size_t)(p * ms.size()))]; };
  const double bytes = static_cast<double>(n) * sizeof(PointXYZIRT) * 2.0;
  const double gbs = bytes / (pct(0.50) * 1e-3) / 1e9;

  std::printf("points        : %d\n", n);
  std::printf("iterations    : %d\n", iters);
  std::printf("traffic/frame : %.2f MB (32B read + 32B write per point)\n", bytes / 1e6);
  std::printf("p50           : %8.1f us\n", pct(0.50) * 1000.0);
  std::printf("p99           : %8.1f us\n", pct(0.99) * 1000.0);
  std::printf("p99.9         : %8.1f us\n", pct(0.999) * 1000.0);
  std::printf("max           : %8.1f us\n", ms.back() * 1000.0);
  std::printf("achieved BW   : %8.1f GB/s  (%.0f%% of 264 GB/s peak)\n", gbs, 100.0 * gbs / 264.0);
  return 0;
}
