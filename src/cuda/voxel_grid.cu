#include <stdexcept>

#include "gpulidar/cuda/check.hpp"
#include "gpulidar/cuda/voxel_grid.hpp"

namespace gpulidar::cuda {
namespace {

constexpr int kBlock = 256;
constexpr unsigned long long kEmpty = 0xFFFFFFFFFFFFFFFFULL;
constexpr unsigned int kNoTime = 0xFFFFFFFFu;
constexpr unsigned int kNoRing = 0xFFFFFFFFu;

int nextPow2(int v) {
  int p = 1;
  while (p < v) p <<= 1;
  return p;
}

__device__ __forceinline__ unsigned long long voxelKeyOf(float x, float y, float z,
                                                         const VoxelGridConfig& cfg) {
  const long long kOff = 1 << 20;
  const long long ix = static_cast<long long>(floorf(x / cfg.leaf_x)) + kOff;
  const long long iy = static_cast<long long>(floorf(y / cfg.leaf_y)) + kOff;
  const long long iz = static_cast<long long>(floorf(z / cfg.leaf_z)) + kOff;
  return static_cast<unsigned long long>((ix << 42) | (iy << 21) | iz);
}

__device__ __forceinline__ unsigned int hashSlot(unsigned long long k, unsigned int mask) {
  k *= 0x9E3779B97F4A7C15ULL;
  k ^= k >> 29;
  return static_cast<unsigned int>(k) & mask;
}

__device__ __forceinline__ void clearCell(VoxelCell& c) {
  c.key = kEmpty;
  c.sum_x = 0.0f; c.sum_y = 0.0f; c.sum_z = 0.0f; c.sum_i = 0.0f;
  c.count = 0u;
  c.min_time_bits = kNoTime;
  c.min_ring = kNoRing;
}

__global__ void initTableKernel(VoxelCell* table, int m) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x) {
    clearCell(table[i]);
  }
}

__global__ void insertKernel(const PointXYZIRT* __restrict__ in, int n, VoxelGridConfig cfg,
                             VoxelCell* __restrict__ table, unsigned int mask,
                             unsigned int* __restrict__ occupied,
                             unsigned int* __restrict__ occ_count) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const PointXYZIRT p = in[i];
    const unsigned long long key = voxelKeyOf(p.x, p.y, p.z, cfg);

    unsigned int slot = hashSlot(key, mask);
    const unsigned int probe_limit = mask + 1u;

    for (unsigned int probe = 0; probe < probe_limit; ++probe) {
      VoxelCell& cell = table[slot];
      const unsigned long long prev = atomicCAS(&cell.key, kEmpty, key);
      const bool claimed = (prev == kEmpty);

      if (claimed) {
        const unsigned int idx = atomicAdd(occ_count, 1u);
        occupied[idx] = slot;
      }

      if (claimed || prev == key) {
        atomicAdd(&cell.sum_x, p.x);
        atomicAdd(&cell.sum_y, p.y);
        atomicAdd(&cell.sum_z, p.z);
        atomicAdd(&cell.sum_i, p.intensity);
        atomicAdd(&cell.count, 1u);
        // time >= 0, so IEEE bit order matches value order
        atomicMin(&cell.min_time_bits, __float_as_uint(p.time));
        atomicMin(&cell.min_ring, static_cast<unsigned int>(p.ring));
        break;
      }
      slot = (slot + 1u) & mask;
    }
  }
}

__global__ void insertKernelDyn(const PointXYZIRT* __restrict__ in,
                                const unsigned int* __restrict__ d_n, VoxelGridConfig cfg,
                                VoxelCell* __restrict__ table, unsigned int mask,
                                unsigned int* __restrict__ occupied,
                                unsigned int* __restrict__ occ_count) {
  const int n = static_cast<int>(*d_n);
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const PointXYZIRT p = in[i];
    const unsigned long long key = voxelKeyOf(p.x, p.y, p.z, cfg);

    unsigned int slot = hashSlot(key, mask);
    const unsigned int probe_limit = mask + 1u;

    for (unsigned int probe = 0; probe < probe_limit; ++probe) {
      VoxelCell& cell = table[slot];
      const unsigned long long prev = atomicCAS(&cell.key, kEmpty, key);
      const bool claimed = (prev == kEmpty);

      if (claimed) {
        const unsigned int idx = atomicAdd(occ_count, 1u);
        occupied[idx] = slot;
      }

      if (claimed || prev == key) {
        atomicAdd(&cell.sum_x, p.x);
        atomicAdd(&cell.sum_y, p.y);
        atomicAdd(&cell.sum_z, p.z);
        atomicAdd(&cell.sum_i, p.intensity);
        atomicAdd(&cell.count, 1u);
        atomicMin(&cell.min_time_bits, __float_as_uint(p.time));
        atomicMin(&cell.min_ring, static_cast<unsigned int>(p.ring));
        break;
      }
      slot = (slot + 1u) & mask;
    }
  }
}

__global__ void extractKernel(const unsigned int* __restrict__ occupied,
                              const unsigned int* __restrict__ occ_count,
                              const VoxelCell* __restrict__ table,
                              PointXYZIRT* __restrict__ out) {
  const unsigned int total = *occ_count;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += blockDim.x * gridDim.x) {
    const VoxelCell c = table[occupied[i]];
    const float inv = 1.0f / static_cast<float>(c.count);
    PointXYZIRT p{};
    p.x = c.sum_x * inv;
    p.y = c.sum_y * inv;
    p.z = c.sum_z * inv;
    p.intensity = c.sum_i * inv;
    p.ring = static_cast<unsigned short>(c.min_ring);
    p._pad = 0;
    p.time = __uint_as_float(c.min_time_bits);
    p._pad2 = 0.0f;
    out[i] = p;
  }
}

__global__ void resetKernel(const unsigned int* __restrict__ occupied,
                            const unsigned int* __restrict__ occ_count,
                            VoxelCell* __restrict__ table) {
  const unsigned int total = *occ_count;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += blockDim.x * gridDim.x) {
    clearCell(table[occupied[i]]);
  }
}

int gridFor(int n) {
  int g = (n + kBlock - 1) / kBlock;
  if (g > 4096) g = 4096;
  return g < 1 ? 1 : g;
}

}  // namespace

VoxelGrid::VoxelGrid(int max_points) : max_points_(max_points) {
  table_capacity_ = nextPow2(max_points * 2);

  table_.reserve(table_capacity_);
  occupied_.reserve(max_points);
  occ_count_.reserve(1);
  host_count_.reserve(1);

  table_.resize(table_capacity_);
  occupied_.resize(max_points);
  occ_count_.resize(1);

  initTableKernel<<<gridFor(table_capacity_), kBlock>>>(table_.data(), table_capacity_);
  GPULIDAR_CUDA_CHECK(cudaGetLastError());
  GPULIDAR_CUDA_CHECK(cudaDeviceSynchronize());
}

int VoxelGrid::process(const PointXYZIRT* d_in, int n, PointXYZIRT* d_out,
                       VoxelGridConfig cfg, cudaStream_t stream, bool sync) {
  if (n <= 0) return 0;
  if (n > max_points_) throw std::runtime_error("VoxelGrid: n exceeds reserved capacity");

  const unsigned int mask = static_cast<unsigned int>(table_capacity_) - 1u;

  GPULIDAR_CUDA_CHECK(cudaMemsetAsync(occ_count_.data(), 0, sizeof(unsigned int), stream));

  insertKernel<<<gridFor(n), kBlock, 0, stream>>>(d_in, n, cfg, table_.data(), mask,
                                                  occupied_.data(), occ_count_.data());
  GPULIDAR_CUDA_CHECK(cudaGetLastError());

  extractKernel<<<gridFor(n), kBlock, 0, stream>>>(occupied_.data(), occ_count_.data(),
                                                   table_.data(), d_out);
  GPULIDAR_CUDA_CHECK(cudaGetLastError());

  GPULIDAR_CUDA_CHECK(cudaMemcpyAsync(host_count_.data(), occ_count_.data(),
                                      sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

  resetKernel<<<gridFor(n), kBlock, 0, stream>>>(occupied_.data(), occ_count_.data(),
                                                 table_.data());
  GPULIDAR_CUDA_CHECK(cudaGetLastError());

  if (!sync) return -1;
  GPULIDAR_CUDA_CHECK(cudaStreamSynchronize(stream));
  return static_cast<int>(host_count_.data()[0]);
}

void VoxelGrid::enqueue(const PointXYZIRT* d_in, const unsigned int* d_n, int max_n,
                        PointXYZIRT* d_out, VoxelGridConfig cfg, cudaStream_t stream) {
  const unsigned int mask = static_cast<unsigned int>(table_capacity_) - 1u;

  GPULIDAR_CUDA_CHECK(cudaMemsetAsync(occ_count_.data(), 0, sizeof(unsigned int), stream));

  insertKernelDyn<<<gridFor(max_n), kBlock, 0, stream>>>(
      d_in, d_n, cfg, table_.data(), mask, occupied_.data(), occ_count_.data());
  GPULIDAR_CUDA_CHECK(cudaGetLastError());

  extractKernel<<<gridFor(max_n), kBlock, 0, stream>>>(
      occupied_.data(), occ_count_.data(), table_.data(), d_out);
  GPULIDAR_CUDA_CHECK(cudaGetLastError());

  resetKernel<<<gridFor(max_n), kBlock, 0, stream>>>(
      occupied_.data(), occ_count_.data(), table_.data());
  GPULIDAR_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpulidar::cuda
