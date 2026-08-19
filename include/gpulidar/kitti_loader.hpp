#pragma once
#include <string>
#include <vector>
#include "gpulidar/point_types.hpp"

namespace gpulidar {

// Reads a KITTI velodyne_points/data/*.bin (float32 x,y,z,intensity).
// ring and time are DERIVED, not stored in the file:
//   ring -- elevation-angle binning across the HDL-64E vertical FOV
//   time -- azimuth sweep fraction * sweep_period_s
// Both are approximations; see README "Limitations".
std::vector<PointXYZIRT> loadVelodyneBin(const std::string& path,
                                         float sweep_period_s = 0.1f);

// Deterministic synthetic cloud so tests run without the dataset present.
std::vector<PointXYZIRT> makeSyntheticSweep(std::size_t n, unsigned seed = 42);

}  // namespace gpulidar
