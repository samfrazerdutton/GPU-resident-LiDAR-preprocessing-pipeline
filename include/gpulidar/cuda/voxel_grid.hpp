#pragma once
#include <cuda_runtime.h>

#include "gpulidar/config.hpp"
#include "gpulidar/cuda/device_buffer.hpp"
#include "gpulidar/point_types.hpp"

namespace gpulidar::cuda {

// Voxel-grid downsample backed by an open-addressing hash table.
//
// All storage is reserved in the constructor; process() never allocates.
// The table is NOT cleared per frame -- each frame records the slots it
// claimed and clears only those afterwards, so reset cost scales with
// occupied cells rather than table capacity.
//
// Output order is undefined (slots are claimed by whichever block gets
// there first) and centroids are not bitwise reproducible (float atomicAdd
// accumulates in nondeterministic order). Compare as a set, with tolerance.
class VoxelGrid {
 public:
  explicit VoxelGrid(int max_points);

  // Returns the number of occupied cells written to d_out.
  // d_out must have capacity for at least max_points entries.
  int process(const PointXYZIRT* d_in, int n, PointXYZIRT* d_out,
              VoxelGridConfig cfg, cudaStream_t stream = nullptr);

  int tableCapacity() const { return table_capacity_; }

 private:
  int max_points_ = 0;
  int table_capacity_ = 0;

  DeviceBuffer<unsigned long long> keys_;
  DeviceBuffer<float> sum_x_, sum_y_, sum_z_, sum_i_;
  DeviceBuffer<unsigned int> count_, min_time_bits_, min_ring_;
  DeviceBuffer<unsigned int> occupied_;
  DeviceBuffer<unsigned int> occ_count_;
  PinnedBuffer<unsigned int> host_count_;
};

}  // namespace gpulidar::cuda
