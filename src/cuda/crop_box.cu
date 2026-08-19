#include "gpulidar/cuda/check.hpp"
#include "gpulidar/cuda/kernels.hpp"

namespace gpulidar::cuda {
namespace {

constexpr int kBlock = 256;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ bool inBox(float x, float y, float z,
                                      float x0, float x1, float y0, float y1,
                                      float z0, float z1) {
  return x >= x0 && x <= x1 && y >= y0 && y <= y1 && z >= z0 && z <= z1;
}

__device__ __forceinline__ bool keepPoint(const PointXYZIRT& p, const CropBoxConfig& c) {
  const bool roi = inBox(p.x, p.y, p.z, c.min_x, c.max_x, c.min_y, c.max_y,
                         c.min_z, c.max_z);
  const bool ego = inBox(p.x, p.y, p.z, c.ego_min_x, c.ego_max_x, c.ego_min_y,
                         c.ego_max_y, c.ego_min_z, c.ego_max_z);
  return roi && !ego;
}

__global__ void cropNaiveKernel(const PointXYZIRT* __restrict__ in,
                                PointXYZIRT* __restrict__ out, int n,
                                CropBoxConfig cfg, unsigned int* __restrict__ counter) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const PointXYZIRT p = in[i];
    if (keepPoint(p, cfg)) {
      const unsigned int slot = atomicAdd(counter, 1u);
      out[slot] = p;
    }
  }
}

__global__ void cropWarpAggKernel(const PointXYZIRT* __restrict__ in,
                                  PointXYZIRT* __restrict__ out, int n,
                                  CropBoxConfig cfg, unsigned int* __restrict__ counter) {
  const int stride = blockDim.x * gridDim.x;
  // Round the trip count up so every lane in a warp exits the loop on the
  // same iteration -- __ballot_sync with divergent exits is undefined.
  const int n_up = ((n + stride - 1) / stride) * stride;
  const int lane = threadIdx.x & 31;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_up; i += stride) {
    const bool valid = (i < n);
    PointXYZIRT p;
    bool keep = false;
    if (valid) {
      p = in[i];
      keep = keepPoint(p, cfg);
    }

    const unsigned int mask = __ballot_sync(kFullMask, keep);
    const int count = __popc(mask);
    if (count == 0) continue;

    const int leader = __ffs(mask) - 1;
    unsigned int base = 0;
    if (lane == leader) base = atomicAdd(counter, static_cast<unsigned int>(count));
    base = __shfl_sync(kFullMask, base, leader);

    if (keep) {
      const int rank = __popc(mask & ((1u << lane) - 1u));
      out[base + rank] = p;
    }
  }
}

int gridFor(int n) {
  int grid = (n + kBlock - 1) / kBlock;
  return grid > 4096 ? 4096 : grid;
}

}  // namespace

void launchCropBoxNaive(const PointXYZIRT* d_in, PointXYZIRT* d_out, int n,
                        CropBoxConfig cfg, unsigned int* d_count, cudaStream_t stream) {
  if (n <= 0) return;
  cropNaiveKernel<<<gridFor(n), kBlock, 0, stream>>>(d_in, d_out, n, cfg, d_count);
  GPULIDAR_CUDA_CHECK(cudaGetLastError());
}

void launchCropBoxWarpAggregated(const PointXYZIRT* d_in, PointXYZIRT* d_out, int n,
                                 CropBoxConfig cfg, unsigned int* d_count,
                                 cudaStream_t stream) {
  if (n <= 0) return;
  cropWarpAggKernel<<<gridFor(n), kBlock, 0, stream>>>(d_in, d_out, n, cfg, d_count);
  GPULIDAR_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpulidar::cuda
