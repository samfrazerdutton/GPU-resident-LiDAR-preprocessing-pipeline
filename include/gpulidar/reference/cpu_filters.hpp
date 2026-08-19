#pragma once
#include <vector>
#include "gpulidar/config.hpp"
#include "gpulidar/oxts.hpp"
#include "gpulidar/point_types.hpp"

namespace gpulidar::reference {

// Ground-truth CPU implementations. Intentionally simple and obviously
// correct -- every CUDA kernel is diffed against these in CI.
std::vector<PointXYZIRT> cropBox(const std::vector<PointXYZIRT>& in,
                                 const CropBoxConfig& cfg);

// Centroid-averaging voxel grid. Output is sorted by voxel key so results
// are order-deterministic and directly comparable to the GPU version,
// whose output order will differ.
std::vector<PointXYZIRT> voxelDownsample(const std::vector<PointXYZIRT>& in,
                                         const VoxelGridConfig& cfg);

// Rigid-body motion compensation. Every point is re-expressed in the sensor
// frame as it was at sweep start (t = 0), assuming constant twist across the
// ~100 ms sweep. At 15 m/s the far end of the sweep is displaced ~1.5 m --
// that is the error this removes.
std::vector<PointXYZIRT> undistort(const std::vector<PointXYZIRT>& in,
                                   const EgoTwist& twist);

}  // namespace gpulidar::reference
