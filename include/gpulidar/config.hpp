#pragma once

namespace gpulidar {

// Keep points inside the ROI box AND outside the ego-vehicle box.
struct CropBoxConfig {
  float min_x = -60.0f, max_x = 60.0f;
  float min_y = -40.0f, max_y = 40.0f;
  float min_z =  -3.0f, max_z =  3.0f;

  float ego_min_x = -2.0f, ego_max_x = 2.0f;
  float ego_min_y = -1.2f, ego_max_y = 1.2f;
  float ego_min_z = -2.0f, ego_max_z = 0.5f;
};

struct VoxelGridConfig {
  float leaf_x = 0.2f;
  float leaf_y = 0.2f;
  float leaf_z = 0.2f;
};

struct SweepConfig {
  float period_s = 0.1f;   // 10 Hz HDL-64E
};

}  // namespace gpulidar
