#pragma once
#include <cstdint>
#include <cstddef>

namespace gpulidar {

// AoS layout, 32 B/point, 16 B aligned for coalesced 128-bit loads.
// SoA variant is a planned optimization -- benchmark both, report both.
struct alignas(16) PointXYZIRT {
  float x;
  float y;
  float z;
  float intensity;
  std::uint16_t ring;   // derived on load for KITTI (HDL-64E)
  std::uint16_t _pad;
  float time;           // seconds since sweep start; derived from azimuth
  float _pad2;
};

static_assert(sizeof(PointXYZIRT) == 32, "unexpected point size");
static_assert(alignof(PointXYZIRT) == 16, "unexpected point alignment");

}  // namespace gpulidar
