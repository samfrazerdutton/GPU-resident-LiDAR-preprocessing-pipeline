#pragma once
#include <cuda_runtime.h>

#include "gpulidar/config.hpp"
#include "gpulidar/oxts.hpp"
#include "gpulidar/point_types.hpp"

namespace gpulidar::cuda {

// Motion compensation. Elementwise, in-place safe (d_in may equal d_out).
// Bandwidth-bound: 32 B read + 32 B written per point.
void launchUndistort(const PointXYZIRT* d_in, PointXYZIRT* d_out, int n,
                     const EgoTwist& twist, cudaStream_t stream = nullptr);

// Stream compaction. d_count must be zeroed by the caller before launch;
// after the kernel it holds the surviving point count. Output ORDER IS NOT
// DEFINED -- blocks race for slots, so compare against the CPU reference as
// a set, not a sequence.
//
// Naive: one global atomicAdd per surviving point.
void launchCropBoxNaive(const PointXYZIRT* d_in, PointXYZIRT* d_out, int n,
                        CropBoxConfig cfg, unsigned int* d_count,
                        cudaStream_t stream = nullptr);

// Warp-aggregated: one atomicAdd per warp, lane slots from the ballot mask.
void launchCropBoxWarpAggregated(const PointXYZIRT* d_in, PointXYZIRT* d_out, int n,
                                 CropBoxConfig cfg, unsigned int* d_count,
                                 cudaStream_t stream = nullptr);

}  // namespace gpulidar::cuda
