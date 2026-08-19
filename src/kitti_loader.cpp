#include "gpulidar/kitti_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>

namespace gpulidar {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

// HDL-64E: 64 lasers spanning roughly +2.0 deg to -24.8 deg.
std::uint16_t deriveRing(float x, float y, float z) {
  const float r = std::sqrt(x * x + y * y);
  if (r < 1e-6f) return 0;
  const float elev_deg = std::atan2(z, r) * 180.0f / kPi;
  int ring = static_cast<int>(std::lround((elev_deg + 24.8f) / 26.8f * 63.0f));
  ring = std::clamp(ring, 0, 63);
  return static_cast<std::uint16_t>(ring);
}

}  // namespace

std::vector<PointXYZIRT> loadVelodyneBin(const std::string& path,
                                         float sweep_period_s) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);

  const std::streamsize bytes = f.tellg();
  const std::size_t stride = 4 * sizeof(float);
  if (bytes <= 0 || static_cast<std::size_t>(bytes) % stride != 0) {
    throw std::runtime_error("not a KITTI velodyne bin: " + path);
  }
  const std::size_t n = static_cast<std::size_t>(bytes) / stride;

  f.seekg(0);
  std::vector<float> raw(n * 4);
  f.read(reinterpret_cast<char*>(raw.data()), bytes);
  if (!f) throw std::runtime_error("short read: " + path);

  std::vector<PointXYZIRT> out(n);
  // Sensor spins clockwise, so atan2(y,x) decreases through the sweep.
  const float az0 = (n > 0) ? std::atan2(raw[1], raw[0]) : 0.0f;

  for (std::size_t i = 0; i < n; ++i) {
    const float x = raw[i * 4 + 0];
    const float y = raw[i * 4 + 1];
    const float z = raw[i * 4 + 2];
    const float in = raw[i * 4 + 3];

    float rel = az0 - std::atan2(y, x);
    if (rel < 0.0f) rel += kTwoPi;
    if (rel >= kTwoPi) rel -= kTwoPi;

    PointXYZIRT& p = out[i];
    p.x = x; p.y = y; p.z = z;
    p.intensity = in;
    p.ring = deriveRing(x, y, z);
    p._pad = 0;
    p.time = rel / kTwoPi * sweep_period_s;
    p._pad2 = 0.0f;
  }
  return out;
}

std::vector<PointXYZIRT> makeSyntheticSweep(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> range(1.0f, 70.0f);
  std::uniform_real_distribution<float> az(-kPi, kPi);
  std::uniform_real_distribution<float> elev(-24.8f, 2.0f);
  std::uniform_real_distribution<float> inten(0.0f, 1.0f);

  std::vector<PointXYZIRT> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float r = range(rng);
    const float a = az(rng);
    const float e = elev(rng) * kPi / 180.0f;
    PointXYZIRT& p = out[i];
    p.x = r * std::cos(e) * std::cos(a);
    p.y = r * std::cos(e) * std::sin(a);
    p.z = r * std::sin(e);
    p.intensity = inten(rng);
    p.ring = deriveRing(p.x, p.y, p.z);
    p._pad = 0;
    p.time = static_cast<float>(i) / static_cast<float>(n) * 0.1f;
    p._pad2 = 0.0f;
  }
  return out;
}

}  // namespace gpulidar
