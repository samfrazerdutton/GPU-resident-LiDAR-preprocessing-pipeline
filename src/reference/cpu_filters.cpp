#include "gpulidar/reference/cpu_filters.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace gpulidar::reference {
namespace {

inline bool inBox(float x, float y, float z,
                  float x0, float x1, float y0, float y1, float z0, float z1) {
  return x >= x0 && x <= x1 && y >= y0 && y <= y1 && z >= z0 && z <= z1;
}

// 21 bits per axis, offset to keep indices non-negative.
inline std::int64_t voxelKey(int ix, int iy, int iz) {
  constexpr std::int64_t kOff = 1 << 20;
  return ((static_cast<std::int64_t>(ix) + kOff) << 42) |
         ((static_cast<std::int64_t>(iy) + kOff) << 21) |
          (static_cast<std::int64_t>(iz) + kOff);
}

struct Accum {
  double x = 0, y = 0, z = 0, intensity = 0;
  float min_time = 0;
  std::uint16_t ring = 0;
  std::uint32_t count = 0;
};

}  // namespace

std::vector<PointXYZIRT> cropBox(const std::vector<PointXYZIRT>& in,
                                 const CropBoxConfig& c) {
  std::vector<PointXYZIRT> out;
  out.reserve(in.size());
  for (const auto& p : in) {
    const bool keep_roi =
        inBox(p.x, p.y, p.z, c.min_x, c.max_x, c.min_y, c.max_y, c.min_z, c.max_z);
    const bool is_ego =
        inBox(p.x, p.y, p.z, c.ego_min_x, c.ego_max_x, c.ego_min_y, c.ego_max_y,
              c.ego_min_z, c.ego_max_z);
    if (keep_roi && !is_ego) out.push_back(p);
  }
  return out;
}

std::vector<PointXYZIRT> voxelDownsample(const std::vector<PointXYZIRT>& in,
                                         const VoxelGridConfig& cfg) {
  std::unordered_map<std::int64_t, Accum> cells;
  cells.reserve(in.size() / 4 + 16);

  for (const auto& p : in) {
    const int ix = static_cast<int>(std::floor(p.x / cfg.leaf_x));
    const int iy = static_cast<int>(std::floor(p.y / cfg.leaf_y));
    const int iz = static_cast<int>(std::floor(p.z / cfg.leaf_z));
    Accum& a = cells[voxelKey(ix, iy, iz)];
    if (a.count == 0) {
      a.min_time = p.time;
      a.ring = p.ring;
    } else {
      a.min_time = std::min(a.min_time, p.time);
      a.ring = std::min(a.ring, p.ring);
    }
    a.x += p.x; a.y += p.y; a.z += p.z;
    a.intensity += p.intensity;
    ++a.count;
  }

  std::vector<std::pair<std::int64_t, Accum>> ordered(cells.begin(), cells.end());
  std::sort(ordered.begin(), ordered.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  std::vector<PointXYZIRT> out;
  out.reserve(ordered.size());
  for (const auto& [key, a] : ordered) {
    (void)key;
    const double n = static_cast<double>(a.count);
    PointXYZIRT p{};
    p.x = static_cast<float>(a.x / n);
    p.y = static_cast<float>(a.y / n);
    p.z = static_cast<float>(a.z / n);
    p.intensity = static_cast<float>(a.intensity / n);
    p.ring = a.ring;
    p._pad = 0;
    p.time = a.min_time;
    p._pad2 = 0.0f;
    out.push_back(p);
  }
  return out;
}

std::vector<PointXYZIRT> undistort(const std::vector<PointXYZIRT>& in,
                                   const EgoTwist& twist) {
  std::vector<PointXYZIRT> out(in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    const PointXYZIRT& p = in[i];
    const float t = p.time;

    // Pose of the frame at time t, expressed in the frame at t = 0.
    const float theta = twist.wu * t;
    const float c = std::cos(theta);
    const float s = std::sin(theta);

    PointXYZIRT q = p;
    q.x = c * p.x - s * p.y + twist.vf * t;
    q.y = s * p.x + c * p.y + twist.vl * t;
    q.z = p.z;
    out[i] = q;
  }
  return out;
}

}  // namespace gpulidar::reference
