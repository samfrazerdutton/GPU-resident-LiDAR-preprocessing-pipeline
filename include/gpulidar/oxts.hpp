#pragma once
#include <string>
#include <vector>

namespace gpulidar {

// Planar ego-motion, expressed in the KITTI vehicle frame
// (x forward, y left, z up -- same axis convention as the velodyne frame).
struct EgoTwist {
  float vf = 0.0f;   // forward velocity, m/s   (oxts field 8)
  float vl = 0.0f;   // leftward velocity, m/s  (oxts field 9)
  float wu = 0.0f;   // yaw rate about up axis, rad/s (oxts field 22)
};

// Parses one KITTI oxts/data/NNNNNNNNNN.txt frame.
EgoTwist loadOxts(const std::string& path);

// KITTI timestamp files: "YYYY-MM-DD HH:MM:SS.fffffffff" per line.
// Returns seconds-of-day (drives never cross midnight); we only ever use
// differences, so the absolute epoch is irrelevant.
std::vector<double> loadTimestampsSecondsOfDay(const std::string& path);

// Sorted list of files in dir with the given extension (e.g. ".bin").
std::vector<std::string> listFiles(const std::string& dir, const std::string& ext);

}  // namespace gpulidar
