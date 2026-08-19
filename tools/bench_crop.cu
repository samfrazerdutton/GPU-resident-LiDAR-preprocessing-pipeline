#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "gpulidar/cuda/check.hpp"
#include "gpulidar/cuda/device_buffer.hpp"
#include "gpulidar/cuda/kernels.hpp"
#include "gpulidar/kitti_loader.hpp"

using namespace gpulidar;

namespace {

struct Stats { double p50, p99, p999, max; };

Stats summarize(std::vector<float> ms) {
  std::sort(ms.begin(), ms.end());
  auto pct = [&](double p) {
    return ms[std::min<size_t>(ms.size() - 1, (size_t)(p * ms.size()))] * 1000.0;
  };
  return {pct(0.50), pct(0.99), pct(0.999), ms.back() * 1000.0};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_crop <velodyne.bin> [iters]\n");
    return 2;
  }
  const int iters = (argc >= 3) ? std::stoi(argv[2]) : 1000;

  const auto cloud = loadVelodyneBin(argv[1]);
  const int n = static_cast<int>(cloud.size());

  cuda::DeviceBuffer<PointXYZIRT> d_in(n), d_out(n);
  cuda::DeviceBuffer<unsigned int> d_count(1);
  d_in.upload(cloud.data(), n);
  d_out.resize(n);
  d_count.resize(1);
  GPULIDAR_CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  GPULIDAR_CUDA_CHECK(cudaEventCreate(&start));
  GPULIDAR_CUDA_CHECK(cudaEventCreate(&stop));

  const CropBoxConfig cfg;
  unsigned int kept = 0;

  auto run = [&](bool aggregated) {
    // Counter reset is inside the timed region: a real pipeline pays it too.
    for (int i = 0; i < 50; ++i) {
      GPULIDAR_CUDA_CHECK(cudaMemsetAsync(d_count.data(), 0, sizeof(unsigned int)));
      if (aggregated)
        cuda::launchCropBoxWarpAggregated(d_in.data(), d_out.data(), n, cfg, d_count.data());
      else
        cuda::launchCropBoxNaive(d_in.data(), d_out.data(), n, cfg, d_count.data());
    }
    GPULIDAR_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> ms;
    ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      GPULIDAR_CUDA_CHECK(cudaEventRecord(start));
      GPULIDAR_CUDA_CHECK(cudaMemsetAsync(d_count.data(), 0, sizeof(unsigned int)));
      if (aggregated)
        cuda::launchCropBoxWarpAggregated(d_in.data(), d_out.data(), n, cfg, d_count.data());
      else
        cuda::launchCropBoxNaive(d_in.data(), d_out.data(), n, cfg, d_count.data());
      GPULIDAR_CUDA_CHECK(cudaEventRecord(stop));
      GPULIDAR_CUDA_CHECK(cudaEventSynchronize(stop));
      float t = 0.0f;
      GPULIDAR_CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
      ms.push_back(t);
    }
    GPULIDAR_CUDA_CHECK(cudaMemcpy(&kept, d_count.data(), sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));
    return summarize(std::move(ms));
  };

  const Stats naive = run(false);
  const Stats agg = run(true);

  const double bytes = static_cast<double>(n) * sizeof(PointXYZIRT) +
                       static_cast<double>(kept) * sizeof(PointXYZIRT);

  std::printf("points in     : %d\n", n);
  std::printf("points kept   : %u (%.1f%%)\n", kept, 100.0 * kept / n);
  std::printf("traffic/frame : %.2f MB\n", bytes / 1e6);
  std::printf("floor (236GB/s + 7.2us launch) : %.1f us\n",
              bytes / 236e9 * 1e6 + 7.2);
  std::printf("\n                  p50      p99    p99.9      max     GB/s\n");
  std::printf("naive atomics %8.1f %8.1f %8.1f %8.1f %8.1f\n", naive.p50, naive.p99,
              naive.p999, naive.max, bytes / (naive.p50 * 1e-6) / 1e9);
  std::printf("warp-aggreg.  %8.1f %8.1f %8.1f %8.1f %8.1f\n", agg.p50, agg.p99,
              agg.p999, agg.max, bytes / (agg.p50 * 1e-6) / 1e9);
  std::printf("\nspeedup (p50) : %.2fx\n", naive.p50 / agg.p50);
  return 0;
}
