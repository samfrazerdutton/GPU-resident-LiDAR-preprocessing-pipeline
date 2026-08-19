#pragma once
#include <cuda_runtime.h>

#include "gpulidar/cuda/device_buffer.hpp"
#include "gpulidar/point_types.hpp"

namespace gpulidar::cuda {

// PointPillars-style grid over the KITTI ROI. Defaults give 432 x 496 pillars.
struct PillarConfig {
  float x_min = 0.0f,   x_max = 69.12f;
  float y_min = -39.68f, y_max = 39.68f;
  float z_min = -3.0f,  z_max = 1.0f;
  float pillar_x = 0.16f, pillar_y = 0.16f;
  // The PointPillars paper's P = 12000 assumes points are pre-filtered to the
  // front camera frustum. This pipeline keeps the full ROI, which occupies
  // ~11k pillars on average and peaks higher, so 12000 clamps ~two thirds of
  // KITTI frames. OpenPCDet uses 16000 (train) / 40000 (inference) similarly.
  int max_pillars = 30000;
  int max_points_per_pillar = 32;

  int gridW() const { return (int)((x_max - x_min) / pillar_x); }
  int gridH() const { return (int)((y_max - y_min) / pillar_y); }
};

// Builds the (P, N, 9) feature tensor a PointPillars encoder consumes.
// Feature layout per point, matching the original paper's D = 9:
//   [x, y, z, intensity, xc, yc, zc, xp, yp]
// where c = offset from the pillar's point centroid and p = offset from the
// pillar's geometric centre.
//
// The pillar grid is dense and bounded, so it uses a flat index array rather
// than the hash table the voxel stage needs. Storage is preallocated;
// process() never allocates.
class PillarScatter {
 public:
  PillarScatter(int max_points, PillarConfig cfg);

  // Returns the number of occupied pillars, clamped to cfg.max_pillars.
  // sync = false skips the count readback and returns -1; the count remains
  // available on the device via countPtr().
  int process(const PointXYZIRT* d_in, int n, cudaStream_t stream = nullptr,
              bool sync = true);

  // True if the last synced frame hit the cap and dropped pillars. Overflow
  // is silent in the kernels by necessity, so it must be observable here.
  bool lastOverflowed() const { return last_overflowed_; }
  int lastRawPillars() const { return last_raw_; }

  const float* features() const { return features_.data(); }        // P*N*9
  const int* coords() const { return coords_.data(); }              // P*2 (gx, gy)
  const unsigned int* pointCounts() const { return counts_.data(); }// P
  const PointXYZIRT* pillarPoints() const { return points_.data(); }// P*N
  const unsigned int* countPtr() const { return pillar_count_.data(); }

  const PillarConfig& config() const { return cfg_; }
  int featureStride() const { return cfg_.max_points_per_pillar * 9; }

 private:
  PillarConfig cfg_;
  int max_points_ = 0;
  int cells_ = 0;

  DeviceBuffer<unsigned int> cell_count_, pillar_index_, occupied_cells_;
  DeviceBuffer<unsigned int> pillar_count_, counts_;
  DeviceBuffer<int> coords_;
  DeviceBuffer<PointXYZIRT> points_;
  DeviceBuffer<float> sums_, features_;
  PinnedBuffer<unsigned int> host_count_;
  bool last_overflowed_ = false;
  int last_raw_ = 0;
};

}  // namespace gpulidar::cuda
