#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "gpulidar/cuda/check.hpp"
#include "gpulidar/cuda/kernels.hpp"
#include "gpulidar/cuda/pipeline.hpp"

namespace gpulidar::cuda {

Pipeline::Pipeline(int max_points) : max_points_(max_points), voxel_(max_points) {
  GPULIDAR_CUDA_CHECK(cudaStreamCreate(&stream_));

  d_in_.reserve(max_points);
  d_undistorted_.reserve(max_points);
  d_cropped_.reserve(max_points);
  d_out_.reserve(max_points);
  d_crop_count_.reserve(1);
  h_stage_.reserve(max_points);
  h_result_.reserve(max_points);
  h_count_.reserve(1);

  d_in_.resize(max_points);
  d_undistorted_.resize(max_points);
  d_cropped_.resize(max_points);
  d_out_.resize(max_points);
  d_crop_count_.resize(1);
}

Pipeline::~Pipeline() {
  if (stream_) cudaStreamDestroy(stream_);
}

void Pipeline::enqueueDevice(const PointXYZIRT* d_in, int n, const EgoTwist& twist,
                             const Config& cfg, PointXYZIRT* d_out) {
  if (n <= 0) return;
  if (n > max_points_) throw std::runtime_error("Pipeline: n exceeds reserved capacity");

  launchUndistort(d_in, d_undistorted_.data(), n, twist, stream_);

  GPULIDAR_CUDA_CHECK(cudaMemsetAsync(d_crop_count_.data(), 0, sizeof(unsigned int), stream_));
  launchCropBoxWarpAggregated(d_undistorted_.data(), d_cropped_.data(), n, cfg.crop,
                              d_crop_count_.data(), stream_);

  // Crop's output count never leaves the device -- the voxel stage reads it
  // from d_crop_count_ inside the kernel.
  voxel_.enqueue(d_cropped_.data(), d_crop_count_.data(), n, d_out, cfg.voxel, stream_);
}

int Pipeline::processFrame(const std::vector<PointXYZIRT>& in, const EgoTwist& twist,
                           const Config& cfg, std::vector<PointXYZIRT>* out) {
  const int n = static_cast<int>(in.size());
  if (n <= 0) return 0;
  if (n > max_points_) throw std::runtime_error("Pipeline: n exceeds reserved capacity");

  std::memcpy(h_stage_.data(), in.data(), n * sizeof(PointXYZIRT));
  GPULIDAR_CUDA_CHECK(cudaMemcpyAsync(d_in_.data(), h_stage_.data(),
                                      n * sizeof(PointXYZIRT), cudaMemcpyHostToDevice,
                                      stream_));

  enqueueDevice(d_in_.data(), n, twist, cfg, d_out_.data());

  GPULIDAR_CUDA_CHECK(cudaMemcpyAsync(h_count_.data(), voxel_.countPtr(),
                                      sizeof(unsigned int), cudaMemcpyDeviceToHost, stream_));
  GPULIDAR_CUDA_CHECK(cudaStreamSynchronize(stream_));

  const int count = static_cast<int>(h_count_.data()[0]);
  if (out) {
    GPULIDAR_CUDA_CHECK(cudaMemcpyAsync(h_result_.data(), d_out_.data(),
                                        count * sizeof(PointXYZIRT),
                                        cudaMemcpyDeviceToHost, stream_));
    GPULIDAR_CUDA_CHECK(cudaStreamSynchronize(stream_));
    out->assign(h_result_.data(), h_result_.data() + count);
  }
  return count;
}

void Pipeline::sync() { GPULIDAR_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

}  // namespace gpulidar::cuda
