#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "gpulidar/kitti_loader.hpp"
#include "gpulidar/oxts.hpp"
#include "gpulidar/reference/cpu_filters.hpp"

using namespace gpulidar;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: kitti_probe <velodyne.bin> [oxts.txt] [sweep_period_s]\n");
    return 2;
  }
  const std::string bin = argv[1];
  const float period = (argc >= 4) ? std::stof(argv[3]) : 0.1f;

  const auto cloud = loadVelodyneBin(bin, period);
  std::printf("points            : %zu\n", cloud.size());
  std::printf("sweep period used : %.4f s\n", period);

  float tmin = 1e9f, tmax = -1e9f, rmax = 0.0f;
  std::vector<int> ring_hist(64, 0);
  for (const auto& p : cloud) {
    tmin = std::min(tmin, p.time);
    tmax = std::max(tmax, p.time);
    rmax = std::max(rmax, std::sqrt(p.x * p.x + p.y * p.y));
    if (p.ring < 64) ++ring_hist[p.ring];
  }
  const int occupied = static_cast<int>(
      std::count_if(ring_hist.begin(), ring_hist.end(), [](int n) { return n > 0; }));
  std::printf("derived time range: %.4f .. %.4f s\n", tmin, tmax);
  std::printf("occupied rings    : %d / 64\n", occupied);
  std::printf("max planar range  : %.1f m\n", rmax);

  const auto cropped = reference::cropBox(cloud, CropBoxConfig{});
  std::printf("after crop        : %zu (%.1f%% kept)\n", cropped.size(),
              100.0 * cropped.size() / std::max<std::size_t>(cloud.size(), 1));

  const auto voxels = reference::voxelDownsample(cropped, VoxelGridConfig{});
  std::printf("after voxel (0.2m): %zu (%.1fx reduction)\n", voxels.size(),
              static_cast<double>(cropped.size()) / std::max<std::size_t>(voxels.size(), 1));

  if (argc >= 3) {
    const EgoTwist tw = loadOxts(argv[2]);
    std::printf("ego twist         : vf=%.2f m/s  vl=%.2f m/s  wu=%.4f rad/s\n",
                tw.vf, tw.vl, tw.wu);

    const auto fixed = reference::undistort(cloud, tw);
    double max_shift = 0.0;
    for (std::size_t i = 0; i < cloud.size(); ++i) {
      const double dx = fixed[i].x - cloud[i].x;
      const double dy = fixed[i].y - cloud[i].y;
      const double dz = fixed[i].z - cloud[i].z;
      max_shift = std::max(max_shift, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    std::printf("max undistortion  : %.3f m\n", max_shift);
  }
  return 0;
}
