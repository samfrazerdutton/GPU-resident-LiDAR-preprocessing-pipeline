#pragma once
#include <cuda_runtime.h>

#include <vector>

#include "gpulidar/config.hpp"
#include "gpulidar/cuda/device_buffer.hpp"
#include "gpulidar/cuda/voxel_grid.hpp"
#include "gpulidar/oxts.hpp"
#include "gpulidar/point_types.hpp"

namespace gpulidar::cuda {

// undistort -> crop -> voxel, resident on the GPU.
//
// The crop stage's output size is only known on the device, so it is passed
// to the voxel stage as a device pointer rather than read back to the host.
// enqueueDevice() therefore issues the whole chain with zero synchronization.
class Pipeline {
 public:
  explicit Pipeline(int max_points);
  ~Pipeline();

  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  struct Config {
    CropBoxConfig crop;
    VoxelGridConfig voxel;
  };

  // Enqueues the full chain on the internal stream. No sync, no host reads.
  // Output lands in d_out (capacity >= max_points); the count is on device.
  void enqueueDevice(const PointXYZIRT* d_in, int n, const EgoTwist& twist,
                     const Config& cfg, PointXYZIRT* d_out);

  // Full frame: H2D from pinned staging, chain, D2H of results. Blocks.
  // Returns the number of output points, appended into out.
  int processFrame(const std::vector<PointXYZIRT>& in, const EgoTwist& twist,
                   const Config& cfg, std::vector<PointXYZIRT>* out);

  PointXYZIRT* deviceInput() { return d_in_.data(); }
  PointXYZIRT* deviceOutput() { return d_out_.data(); }
  cudaStream_t stream() const { return stream_; }
  const unsigned int* outCountPtr() const { return voxel_.countPtr(); }

  void sync();

 private:
  int max_points_ = 0;
  cudaStream_t stream_ = nullptr;

  DeviceBuffer<PointXYZIRT> d_in_, d_undistorted_, d_cropped_, d_out_;
  DeviceBuffer<unsigned int> d_crop_count_;
  PinnedBuffer<PointXYZIRT> h_stage_, h_result_;
  PinnedBuffer<unsigned int> h_count_;
  VoxelGrid voxel_;
};

}  // namespace gpulidar::cuda
