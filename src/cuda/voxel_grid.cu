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

// Fibonacci hashing -- voxel keys are highly structured, so the low bits of
// the raw key are a terrible slot index.
__device__ __forceinline__ unsigned int hashSlot(unsigned long long k, unsigned int mask) {
  k *= 0x9E3779B97F4A7C15ULL;
  k ^= k >> 29;
  return static_cast<unsigned int>(k) & mask;
}

__global__ void initTableKernel(unsigned long long* keys, float* sx, float* sy, float* sz,
                                float* si, unsigned int* cnt, unsigned int* mt,
                                unsigned int* mr, int m) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x) {
    keys[i] = kEmpty;
    sx[i] = 0.0f; sy[i] = 0.0f; sz[i] = 0.0f; si[i] = 0.0f;
    cnt[i] = 0u;
    mt[i] = kNoTime;
    mr[i] = kNoRing;
  }
}

__global__ void insertKernel(const PointXYZIRT* __restrict__ in, int n, VoxelGridConfig cfg,
                             unsigned long long* __restrict__ keys,
                             float* __restrict__ sx, float* __restrict__ sy,
                             float* __restrict__ sz, float* __restrict__ si,
                             unsigned int* __restrict__ cnt, unsigned int* __restrict__ mt,
                             unsigned int* __restrict__ mr, unsigned int mask,
                             unsigned int* __restrict__ occupied,
                             unsigned int* __restrict__ occ_count) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const PointXYZIRT p = in[i];
    const unsigned long long key = voxelKeyOf(p.x, p.y, p.z, cfg);

    unsigned int slot = hashSlot(key, mask);
    const unsigned int probe_limit = mask + 1u;
    bool placed = false;

    for (unsigned int probe = 0; probe < probe_limit; ++probe) {
      const unsigned long long prev = atomicCAS(&keys[slot], kEmpty, key);
      if (prev == kEmpty) {
        // We claimed this slot -- record it so the reset pass can find it.
        const unsigned int idx = atomicAdd(occ_count, 1u);
        occupied[idx] = slot;
        placed = true;
      } else if (prev == key) {
        placed = true;
      }

      if (placed) {
        atomicAdd(&sx[slot], p.x);
        atomicAdd(&sy[slot], p.y);
        atomicAdd(&sz[slot], p.z);
        atomicAdd(&si[slot], p.intensity);
        atomicAdd(&cnt[slot], 1u);
        atomicMin(&mt[slot], __float_as_uint(p.time));   // time >= 0, so bit order == value order
        atomicMin(&mr[slot], static_cast<unsigned int>(p.ring));
        break;
      }
      slot = (slot + 1u) & mask;
    }
  }
}

__global__ void extractKernel(const unsigned int* __restrict__ occupied,
                              const unsigned int* __restrict__ occ_count,
                              const float* __restrict__ sx, const float* __restrict__ sy,
                              const float* __restrict__ sz, const float* __restrict__ si,
                              const unsigned int* __restrict__ cnt,
                              const unsigned int* __restrict__ mt,
                              const unsigned int* __restrict__ mr,
                              PointXYZIRT* __restrict__ out) {
  const unsigned int total = *occ_count;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += blockDim.x * gridDim.x) {
    const unsigned int slot = occupied[i];
    const float inv = 1.0f / static_cast<float>(cnt[slot]);
    PointXYZIRT p{};
    p.x = sx[slot] * inv;
    p.y = sy[slot] * inv;
    p.z = sz[slot] * inv;
    p.intensity = si[slot] * inv;
    p.ring = static_cast<unsigned short>(mr[slot]);
    p._pad = 0;
    p.time = __uint_as_float(mt[slot]);
    p._pad2 = 0.0f;
    out[i] = p;
  }
}

__global__ void resetKernel(const unsigned int* __restrict__ occupied,
                            const unsigned int* __restrict__ occ_count,
                            unsigned long long* keys, float* sx, float* sy, float* sz,
                            float* si, unsigned int* cnt, unsigned int* mt,
                            unsigned int* mr) {
  const unsigned int total = *occ_count;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += blockDim.x * gridDim.x) {
    const unsigned int slot = occupied[i];
    keys[slot] = kEmpty;
    sx[slot] = 0.0f; sy[slot] = 0.0f; sz[slot] = 0.0f; si[slot] = 0.0f;
    cnt[slot] = 0u;
    mt[slot] = kNoTime;
    mr[slot] = kNoRing;
  }
}

int gridFor(int n) {
  int g = (n + kBlock - 1) / kBlock;
  return g > 4096 ? 4096 : (g < 1 ? 1 : g);
}

}  // namespace

VoxelGrid::VoxelGrid(int max_points) : max_points_(max_points) {
  // Load factor <= 0.5 keeps linear probing short.
  table_capacity_ = nextPow2(max_points * 2);

  keys_.reserve(table_capacity_);
  sum_x_.reserve(table_capacity_);
  sum_y_.reserve(table_capacity_);
  sum_z_.reserve(table_capacity_);
  sum_i_.reserve(table_capacity_);
  count_.reserve(table_capacity_);
  min_time_bits_.reserve(table_capacity_);
  min_ring_.reserve(table_capacity_);
  occupied_.reserve(max_points);
  occ_count_.reserve(1);
  host_count_.reserve(1);

  keys_.resize(table_capacity_);
  sum_x_.resize(table_capacity_);
  sum_y_.resize(table_capacity_);
  sum_z_.resize(table_capacity_);
  sum_i_.resize(table_capacity_);
  count_.resize(table_capacity_);
  min_time_bits_.resize(table_capacity_);
  min_ring_.resize(table_capacity_);
  occupied_.resize(max_points);
  occ_count_.resize(1);

  initTableKernel<<<gridFor(table_capacity_), kBlock>>>(
      keys_.data(), sum_x_.data(), sum_y_.data(), sum_z_.data(), sum_i_.data(),
      count_.data(), min_time_bits_.data(), min_ring_.data(), table_capacity_);
  GPULIDAR_CUDA_CHECK(cudaGetLastError());
  GPULIDAR_CUDA_CHECK(cudaDeviceSynchronize());
}

int VoxelGrid::process(const PointXYZIRT* d_in, int n, PointXYZIRT* d_out,
                       VoxelGridConfig cfg, cudaStream_t stream) {
  if (n <= 0) return 0;
  if (n > max_points_) throw std::runtime_error("VoxelGrid: n exceeds reserved capacity");

  const unsigned int mask = static_cast<unsigned int>(table_capacity_) - 1u;

  GPULIDAR_CUDA_CHECK(cudaMemsetAsync(occ_count_.data(), 0, sizeof(unsigned int), stream));

  insertKernel<<<gridFor(n), kBlock, 0, stream>>>(
      d_in, n, cfg, keys_.data(), sum_x_.data(), sum_y_.data(), sum_z_.data(),
      sum_i_.data(), count_.data(), min_time_bits_.data(), min_ring_.data(), mask,
      occupied_.data(), occ_count_.data());
  GPULIDAR_CUDA_CHECK(cudaGetLastError());

  extractKernel<<<gridFor(n), kBlock, 0, stream>>>(
      occupied_.data(), occ_count_.data(), sum_x_.data(), sum_y_.data(), sum_z_.data(),
      sum_i_.data(), count_.data(), min_time_bits_.data(), min_ring_.data(), d_out);
  GPULIDAR_CUDA_CHECK(cudaGetLastError());

  GPULIDAR_CUDA_CHECK(cudaMemcpyAsync(host_count_.data(), occ_count_.data(),
                                      sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

  // Reset runs after extract on the same stream, so it cannot race the read.
  resetKernel<<<gridFor(n), kBlock, 0, stream>>>(
      occupied_.data(), occ_count_.data(), keys_.data(), sum_x_.data(), sum_y_.data(),
      sum_z_.data(), sum_i_.data(), count_.data(), min_time_bits_.data(), min_ring_.data());
  GPULIDAR_CUDA_CHECK(cudaGetLastError());

  GPULIDAR_CUDA_CHECK(cudaStreamSynchronize(stream));
  return static_cast<int>(host_count_.data()[0]);
}

}  // namespace gpulidar::cuda
