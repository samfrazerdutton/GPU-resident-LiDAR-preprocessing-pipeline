#pragma once
#include <cuda_runtime.h>

#include "gpulidar/config.hpp"
#include "gpulidar/cuda/device_buffer.hpp"
#include "gpulidar/point_types.hpp"

namespace gpulidar::cuda {

// One cell, contiguous. The SoA version of this table spread each cell's
// eight fields across eight separate arrays -- every cell touch cost eight
// scattered memory transactions, at ~17.5 ns per occupied cell. Packed at
// 64 B with 32 B alignment, a cell touch is two aligned sectors: ~4.35 ns.
struct alignas(32) VoxelCell {
  unsigned long long key;
  float sum_x, sum_y, sum_z, sum_i;
  unsigned int count;
  unsigned int min_time_bits;
  unsigned int min_ring;
  unsigned int _pad[7];
};
static_assert(sizeof(VoxelCell) == 64, "VoxelCell must be exactly 64 B");

// Voxel-grid downsample backed by an open-addressing hash table.
//
// All storage is reserved in the constructor; process() never allocates.
// The table is NOT cleared per frame -- each frame records the slots it
// claimed and clears only those afterwards, so reset cost scales with
// occupied cells rather than table capacity.
//
// Output order is undefined and centroids are not bitwise reproducible
// (float atomicAdd accumulates in nondeterministic order). Compare as a
// set, with tolerance.
class VoxelGrid {
 public:
  explicit VoxelGrid(int max_points);

  // sync = true blocks and returns the cell count. sync = false enqueues the
  // work and returns -1 -- the count stays on the device, which is what a
  // real pipeline wants when the next stage consumes it there.
  int process(const PointXYZIRT* d_in, int n, PointXYZIRT* d_out,
              VoxelGridConfig cfg, cudaStream_t stream = nullptr, bool sync = true);

  // Fully async: point count comes from device memory (d_n), so this can be
  // chained after a compaction stage without a host round-trip. The output
  // count lands in countPtr() on the device.
  void enqueue(const PointXYZIRT* d_in, const unsigned int* d_n, int max_n,
               PointXYZIRT* d_out, VoxelGridConfig cfg, cudaStream_t stream);

  const unsigned int* countPtr() const { return occ_count_.data(); }

  int tableCapacity() const { return table_capacity_; }

 private:
  int max_points_ = 0;
  int table_capacity_ = 0;

  DeviceBuffer<VoxelCell> table_;
  DeviceBuffer<unsigned int> occupied_;
  DeviceBuffer<unsigned int> occ_count_;
  PinnedBuffer<unsigned int> host_count_;
};

}  // namespace gpulidar::cuda
