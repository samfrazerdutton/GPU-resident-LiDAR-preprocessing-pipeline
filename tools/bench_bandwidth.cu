#include <algorithm>
#include <cstdio>
#include <vector>

#include "gpulidar/cuda/check.hpp"
#include "gpulidar/cuda/device_buffer.hpp"
#include "gpulidar/point_types.hpp"

using namespace gpulidar;

// Pure streaming copy: the practical ceiling for any bandwidth-bound kernel
// on this device. Every later kernel is scored against THIS, not the
// datasheet figure -- no real kernel reaches theoretical peak.
__global__ void copyKernel(const float4* __restrict__ in, float4* __restrict__ out, int n4) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    out[i] = in[i];
  }
}

int main(int argc, char** argv) {
  const int n_points = (argc >= 2) ? std::atoi(argv[1]) : 115964;
  const int iters = (argc >= 3) ? std::atoi(argv[2]) : 1000;

  const std::size_t bytes_per_side = static_cast<std::size_t>(n_points) * sizeof(PointXYZIRT);
  const int n4 = static_cast<int>(bytes_per_side / sizeof(float4));

  cuda::DeviceBuffer<float4> d_in(n4), d_out(n4);
  d_in.resize(n4);
  d_out.resize(n4);
  GPULIDAR_CUDA_CHECK(cudaMemset(d_in.data(), 0, bytes_per_side));

  cudaEvent_t start, stop;
  GPULIDAR_CUDA_CHECK(cudaEventCreate(&start));
  GPULIDAR_CUDA_CHECK(cudaEventCreate(&stop));

  const int block = 256;
  int grid = (n4 + block - 1) / block;
  if (grid > 4096) grid = 4096;

  for (int i = 0; i < 50; ++i) copyKernel<<<grid, block>>>(d_in.data(), d_out.data(), n4);
  GPULIDAR_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> ms;
  ms.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    GPULIDAR_CUDA_CHECK(cudaEventRecord(start));
    copyKernel<<<grid, block>>>(d_in.data(), d_out.data(), n4);
    GPULIDAR_CUDA_CHECK(cudaEventRecord(stop));
    GPULIDAR_CUDA_CHECK(cudaEventSynchronize(stop));
    float t = 0.0f;
    GPULIDAR_CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());

  auto pct = [&](double p) { return ms[std::min<size_t>(ms.size() - 1, (size_t)(p * ms.size()))]; };
  const double traffic = 2.0 * bytes_per_side;
  const double gbs = traffic / (pct(0.50) * 1e-3) / 1e9;

  std::printf("copy size     : %.2f MB each way (%d points equivalent)\n",
              bytes_per_side / 1e6, n_points);
  std::printf("p50           : %8.1f us\n", pct(0.50) * 1000.0);
  std::printf("p99.9         : %8.1f us\n", pct(0.999) * 1000.0);
  std::printf("ACHIEVABLE BW : %8.1f GB/s  (%.0f%% of 264 GB/s theoretical)\n",
              gbs, 100.0 * gbs / 264.0);
  return 0;
}
