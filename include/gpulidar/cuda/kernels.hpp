#pragma once
#include <cuda_runtime.h>

#include "gpulidar/oxts.hpp"
#include "gpulidar/point_types.hpp"

namespace gpulidar::cuda {

// Motion compensation. Elementwise, in-place safe (d_in may equal d_out).
// Bandwidth-bound: 32 B read + 32 B written per point.
void launchUndistort(const PointXYZIRT* d_in, PointXYZIRT* d_out, int n,
                     const EgoTwist& twist, cudaStream_t stream = nullptr);

}  // namespace gpulidar::cuda
