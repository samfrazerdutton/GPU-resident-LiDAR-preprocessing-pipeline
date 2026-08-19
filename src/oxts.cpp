#include "gpulidar/oxts.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace gpulidar {

EgoTwist loadOxts(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open oxts file " + path);

  std::string line;
  if (!std::getline(f, line)) throw std::runtime_error("empty oxts file " + path);

  std::istringstream ss(line);
  std::vector<double> v;
  double x = 0.0;
  while (ss >> x) v.push_back(x);
  if (v.size() < 23) {
    throw std::runtime_error("oxts frame has " + std::to_string(v.size()) +
                             " fields, expected >= 23: " + path);
  }

  EgoTwist t;
  t.vf = static_cast<float>(v[8]);
  t.vl = static_cast<float>(v[9]);
  t.wu = static_cast<float>(v[22]);
  return t;
}

std::vector<double> loadTimestampsSecondsOfDay(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open timestamps file " + path);

  std::vector<double> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const std::size_t sp = line.find(' ');
    if (sp == std::string::npos) continue;
    const std::string clock = line.substr(sp + 1);

    int hh = 0, mm = 0;
    double sec = 0.0;
    char c1 = 0, c2 = 0;
    std::istringstream ts(clock);
    ts >> hh >> c1 >> mm >> c2 >> sec;
    if (!ts || c1 != ':' || c2 != ':') {
      throw std::runtime_error("bad timestamp line: " + line);
    }
    out.push_back(hh * 3600.0 + mm * 60.0 + sec);
  }
  return out;
}

std::vector<std::string> listFiles(const std::string& dir, const std::string& ext) {
  namespace fs = std::filesystem;
  if (!fs::is_directory(dir)) throw std::runtime_error("not a directory: " + dir);

  std::vector<std::string> out;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.is_regular_file() && e.path().extension() == ext) {
      out.push_back(e.path().string());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace gpulidar
