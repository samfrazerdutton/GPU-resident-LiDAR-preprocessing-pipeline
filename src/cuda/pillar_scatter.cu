#include <stdexcept>

#include "gpulidar/cuda/check.hpp"
#include "gpulidar/cuda/pillar_scatter.hpp"

namespace gpulidar::cuda {
namespace {

constexpr int kBlock = 256;
constexpr unsigned int kNoPillar = 0xFFFFFFFFu;

__device__ __forceinline__ bool cellOf(const PointXYZIRT& p, const PillarConfig& c,
                                       int gw, int gh, int* gx, int* gy) {
  if (p.x < c.x_min || p.x >= c.x_max) return false;
  if (p.y < c.y_min || p.y >= c.y_max) return false;
  if (p.z < c.z_min || p.z >= c.z_max) return false;
  *gx = (int)((p.x - c.x_min) / c.pillar_x);
  *gy = (int)((p.y - c.y_min) / c.pillar_y);
  return (*gx >= 0 && *gx < gw && *gy >= 0 && *gy < gh);
}

// Pass 1: how many points land in each grid cell.
__global__ void countKernel(const PointXYZIRT* __restrict__ in, int n, PillarConfig cfg,
                            int gw, int gh, unsigned int* __restrict__ cell_count) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    int gx, gy;
    if (cellOf(in[i], cfg, gw, gh, &gx, &gy)) {
      atomicAdd(&cell_count[gx * gh + gy], 1u);
    }
  }
}

// Pass 2: assign a pillar id to every non-empty cell.
__global__ void compactKernel(unsigned int* __restrict__ cell_count, int cells, int gh,
                              int max_pillars, unsigned int* __restrict__ pillar_index,
                              unsigned int* __restrict__ pillar_count,
                              int* __restrict__ coords,
                              unsigned int* __restrict__ occupied_cells) {
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cells;
       c += blockDim.x * gridDim.x) {
    if (cell_count[c] == 0u) continue;
    const unsigned int pid = atomicAdd(pillar_count, 1u);
    if (pid >= (unsigned int)max_pillars) continue;  // overflow: cell stays unmapped
    pillar_index[c] = pid;
    coords[pid * 2 + 0] = c / gh;
    coords[pid * 2 + 1] = c % gh;
    occupied_cells[pid] = (unsigned int)c;
  }
}

// Pass 3: place points into their pillar, up to N.
__global__ void scatterKernel(const PointXYZIRT* __restrict__ in, int n, PillarConfig cfg,
                              int gw, int gh, const unsigned int* __restrict__ pillar_index,
                              unsigned int* __restrict__ counts,
                              PointXYZIRT* __restrict__ points, float* __restrict__ sums,
                              int max_pts) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    const PointXYZIRT p = in[i];
    int gx, gy;
    if (!cellOf(p, cfg, gw, gh, &gx, &gy)) continue;
    const unsigned int pid = pillar_index[gx * gh + gy];
    if (pid == kNoPillar) continue;

    const unsigned int slot = atomicAdd(&counts[pid], 1u);
    if (slot >= (unsigned int)max_pts) continue;  // pillar full; point dropped
    points[pid * max_pts + slot] = p;
    atomicAdd(&sums[pid * 3 + 0], p.x);
    atomicAdd(&sums[pid * 3 + 1], p.y);
    atomicAdd(&sums[pid * 3 + 2], p.z);
  }
}

// Pass 4: encode features, zero-padding unused slots.
__global__ void featureKernel(const unsigned int* __restrict__ pillar_count,
                              const unsigned int* __restrict__ counts,
                              const int* __restrict__ coords,
                              const PointXYZIRT* __restrict__ points,
                              const float* __restrict__ sums, PillarConfig cfg,
                              int max_pts, int max_pillars, float* __restrict__ features) {
  const unsigned int np = min(*pillar_count, (unsigned int)max_pillars);
  const int total = (int)np * max_pts;
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
       t += blockDim.x * gridDim.x) {
    const int pid = t / max_pts;
    const int j = t % max_pts;
    float* f = features + (size_t)t * 9;

    const unsigned int kept = min(counts[pid], (unsigned int)max_pts);
    if ((unsigned int)j >= kept) {
      for (int k = 0; k < 9; ++k) f[k] = 0.0f;
      continue;
    }

    const PointXYZIRT p = points[pid * max_pts + j];
    const float inv = 1.0f / (float)kept;
    const float cx = sums[pid * 3 + 0] * inv;
    const float cy = sums[pid * 3 + 1] * inv;
    const float cz = sums[pid * 3 + 2] * inv;
    const float px = cfg.x_min + ((float)coords[pid * 2 + 0] + 0.5f) * cfg.pillar_x;
    const float py = cfg.y_min + ((float)coords[pid * 2 + 1] + 0.5f) * cfg.pillar_y;

    f[0] = p.x; f[1] = p.y; f[2] = p.z; f[3] = p.intensity;
    f[4] = p.x - cx; f[5] = p.y - cy; f[6] = p.z - cz;
    f[7] = p.x - px; f[8] = p.y - py;
  }
}

// Pass 5: sparse reset -- only the cells this frame claimed.
__global__ void resetKernel(const unsigned int* __restrict__ pillar_count,
                            const unsigned int* __restrict__ occupied_cells,
                            int max_pillars, unsigned int* __restrict__ cell_count,
                            unsigned int* __restrict__ pillar_index,
                            unsigned int* __restrict__ counts, float* __restrict__ sums) {
  const unsigned int np = min(*pillar_count, (unsigned int)max_pillars);
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < np;
       i += blockDim.x * gridDim.x) {
    const unsigned int cell = occupied_cells[i];
    cell_count[cell] = 0u;
    pillar_index[cell] = kNoPillar;
    counts[i] = 0u;
    sums[i * 3 + 0] = 0.0f;
    sums[i * 3 + 1] = 0.0f;
    sums[i * 3 + 2] = 0.0f;
  }
}

__global__ void initIndexKernel(unsigned int* idx, unsigned int* cell_count, int cells) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cells;
       i += blockDim.x * gridDim.x) {
    idx[i] = kNoPillar;
    cell_count[i] = 0u;
  }
}

int gridFor(int n) {
  int g = (n + kBlock - 1) / kBlock;
  if (g > 4096) g = 4096;
  return g < 1 ? 1 : g;
}

}  // namespace

PillarScatter::PillarScatter(int max_points, PillarConfig cfg)
    : cfg_(cfg), max_points_(max_points) {
  cells_ = cfg_.gridW() * cfg_.gridH();
  const int P = cfg_.max_pillars;
  const int N = cfg_.max_points_per_pillar;

  cell_count_.reserve(cells_);       cell_count_.resize(cells_);
  pillar_index_.reserve(cells_);     pillar_index_.resize(cells_);
  occupied_cells_.reserve(P);        occupied_cells_.resize(P);
  pillar_count_.reserve(1);          pillar_count_.resize(1);
  counts_.reserve(P);                counts_.resize(P);
  coords_.reserve(P * 2);            coords_.resize(P * 2);
  points_.reserve((size_t)P * N);    points_.resize((size_t)P * N);
  sums_.reserve(P * 3);              sums_.resize(P * 3);
  features_.reserve((size_t)P * N * 9); features_.resize((size_t)P * N * 9);
  host_count_.reserve(1);

  initIndexKernel<<<gridFor(cells_), kBlock>>>(pillar_index_.data(), cell_count_.data(),
                                               cells_);
  GPULIDAR_CUDA_CHECK(cudaGetLastError());
  GPULIDAR_CUDA_CHECK(cudaMemset(counts_.data(), 0, P * sizeof(unsigned int)));
  GPULIDAR_CUDA_CHECK(cudaMemset(sums_.data(), 0, P * 3 * sizeof(float)));
  // Zero so the first frame's leading reset is a harmless no-op.
  GPULIDAR_CUDA_CHECK(cudaMemset(pillar_count_.data(), 0, sizeof(unsigned int)));
  GPULIDAR_CUDA_CHECK(cudaDeviceSynchronize());
}

int PillarScatter::process(const PointXYZIRT* d_in, int n, cudaStream_t stream,
                           bool sync) {
  if (n <= 0) return 0;
  if (n > max_points_) throw std::runtime_error("PillarScatter: n exceeds capacity");

  const int gw = cfg_.gridW(), gh = cfg_.gridH();
  const int N = cfg_.max_points_per_pillar;

  // Reset the PREVIOUS frame's cells first. Doing this at the end of a frame
  // would clobber features/counts/coords before the caller has read them --
  // outputs stay valid until the next process() call.
  resetKernel<<<gridFor(cfg_.max_pillars), kBlock, 0, stream>>>(
      pillar_count_.data(), occupied_cells_.data(), cfg_.max_pillars, cell_count_.data(),
      pillar_index_.data(), counts_.data(), sums_.data());
  GPULIDAR_CUDA_CHECK(cudaGetLastError());

  GPULIDAR_CUDA_CHECK(cudaMemsetAsync(pillar_count_.data(), 0, sizeof(unsigned int), stream));

  countKernel<<<gridFor(n), kBlock, 0, stream>>>(d_in, n, cfg_, gw, gh, cell_count_.data());
  compactKernel<<<gridFor(cells_), kBlock, 0, stream>>>(
      cell_count_.data(), cells_, gh, cfg_.max_pillars, pillar_index_.data(),
      pillar_count_.data(), coords_.data(), occupied_cells_.data());
  scatterKernel<<<gridFor(n), kBlock, 0, stream>>>(
      d_in, n, cfg_, gw, gh, pillar_index_.data(), counts_.data(), points_.data(),
      sums_.data(), N);
  featureKernel<<<gridFor(cfg_.max_pillars * N), kBlock, 0, stream>>>(
      pillar_count_.data(), counts_.data(), coords_.data(), points_.data(), sums_.data(),
      cfg_, N, cfg_.max_pillars, features_.data());
  GPULIDAR_CUDA_CHECK(cudaGetLastError());

  if (!sync) return -1;

  GPULIDAR_CUDA_CHECK(cudaMemcpyAsync(host_count_.data(), pillar_count_.data(),
                                      sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  GPULIDAR_CUDA_CHECK(cudaStreamSynchronize(stream));
  const unsigned int raw = host_count_.data()[0];

  last_raw_ = (int)raw;
  last_overflowed_ = raw > (unsigned int)cfg_.max_pillars;
  return (int)(raw < (unsigned int)cfg_.max_pillars ? raw : (unsigned int)cfg_.max_pillars);
}

}  // namespace gpulidar::cuda
