#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "gpulidar/cuda/check.hpp"
#include "gpulidar/cuda/pipeline.hpp"
#include "gpulidar/kitti_loader.hpp"
#include "gpulidar/oxts.hpp"

using namespace gpulidar;

namespace {

struct Stats { double p50, p90, p99, p999, max; };

Stats summarize(std::vector<float> ms) {
  std::sort(ms.begin(), ms.end());
  auto pct = [&](double p) {
    return ms[std::min<size_t>(ms.size() - 1, (size_t)(p * ms.size()))] * 1000.0;
  };
  return {pct(0.50), pct(0.90), pct(0.99), pct(0.999), ms.back() * 1000.0};
}

void printRow(const char* label, const Stats& s) {
  std::printf("%-22s %8.1f %8.1f %8.1f %8.1f %9.1f\n", label, s.p50, s.p90, s.p99,
              s.p999, s.max);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_pipeline <drive_sync_dir> [repeats]\n");
    return 2;
  }
  const std::string drive = argv[1];
  const int repeats = (argc >= 3) ? std::stoi(argv[2]) : 12;

  namespace fs = std::filesystem;
  const auto all_bins = listFiles(drive + "/velodyne_points/data", ".bin");
  std::vector<std::string> bins, oxts;
  for (const auto& b : all_bins) {
    const std::string stem = fs::path(b).stem().string();
    const std::string o = drive + "/oxts/data/" + stem + ".txt";
    if (fs::exists(o)) { bins.push_back(b); oxts.push_back(o); }
  }
  if (bins.empty()) {
    std::fprintf(stderr, "no paired frames under %s\n", drive.c_str());
    return 1;
  }
  std::printf("paired %zu of %zu velodyne frames with oxts\n", bins.size(), all_bins.size());

  std::vector<std::vector<PointXYZIRT>> clouds;
  std::vector<EgoTwist> twists;
  std::size_t max_points = 0, total_points = 0;
  for (std::size_t i = 0; i < bins.size(); ++i) {
    clouds.push_back(loadVelodyneBin(bins[i]));
    twists.push_back(loadOxts(oxts[i]));
    max_points = std::max(max_points, clouds.back().size());
    total_points += clouds.back().size();
  }
  const std::size_t mean_points = total_points / clouds.size();
  std::printf("max points/frame: %zu   mean: %zu   (%.2f MB/frame)\n", max_points,
              mean_points, mean_points * sizeof(PointXYZIRT) / 1e6);

  cuda::Pipeline pipe(static_cast<int>(max_points));
  cuda::Pipeline::Config cfg;
  cuda::PinnedBuffer<PointXYZIRT> staging(static_cast<int>(max_points));

  cudaEvent_t start, stop;
  GPULIDAR_CUDA_CHECK(cudaEventCreate(&start));
  GPULIDAR_CUDA_CHECK(cudaEventCreate(&stop));

  auto timed = [&](auto&& body) {
    GPULIDAR_CUDA_CHECK(cudaEventRecord(start, pipe.stream()));
    body();
    GPULIDAR_CUDA_CHECK(cudaEventRecord(stop, pipe.stream()));
    GPULIDAR_CUDA_CHECK(cudaEventSynchronize(stop));
    float t = 0.0f;
    GPULIDAR_CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
    return t;
  };

  // Warm up
  for (int i = 0; i < 20; ++i) {
    pipe.enqueueDevice(pipe.deviceInput(), static_cast<int>(clouds[0].size()), twists[0],
                       cfg, pipe.deviceOutput());
  }
  pipe.sync();

  std::vector<float> t_h2d, t_gpu, t_e2e;
  const std::size_t samples = clouds.size() * repeats;
  t_h2d.reserve(samples); t_gpu.reserve(samples); t_e2e.reserve(samples);

  std::size_t total_cells = 0;
  int frames = 0;

  for (int r = 0; r < repeats; ++r) {
    for (std::size_t f = 0; f < clouds.size(); ++f) {
      const int n = static_cast<int>(clouds[f].size());
      const std::size_t bytes = n * sizeof(PointXYZIRT);

      // Host staging copy happens OUTSIDE every timed region -- a real sensor
      // driver writes into pinned memory directly, so counting it would
      // measure an artifact of reading KITTI from a std::vector.
      std::memcpy(staging.data(), clouds[f].data(), bytes);

      // A) PCIe host-to-device only
      t_h2d.push_back(timed([&] {
        GPULIDAR_CUDA_CHECK(cudaMemcpyAsync(pipe.deviceInput(), staging.data(), bytes,
                                            cudaMemcpyHostToDevice, pipe.stream()));
      }));

      // B) GPU-resident compute only: data already on device
      t_gpu.push_back(timed([&] {
        pipe.enqueueDevice(pipe.deviceInput(), n, twists[f], cfg, pipe.deviceOutput());
      }));

      // C) End to end: transfer + compute + count readback
      t_e2e.push_back(timed([&] {
        GPULIDAR_CUDA_CHECK(cudaMemcpyAsync(pipe.deviceInput(), staging.data(), bytes,
                                            cudaMemcpyHostToDevice, pipe.stream()));
        pipe.enqueueDevice(pipe.deviceInput(), n, twists[f], cfg, pipe.deviceOutput());
      }));

      ++frames;
    }
  }

  // One correctness-ish sanity pass for the reported cell count
  {
    std::vector<PointXYZIRT> out;
    for (std::size_t f = 0; f < clouds.size(); ++f) {
      total_cells += pipe.processFrame(clouds[f], twists[f], cfg, nullptr);
    }
  }

  const Stats h2d = summarize(t_h2d);
  const Stats gpu = summarize(t_gpu);
  const Stats e2e = summarize(t_e2e);

  std::printf("\nframes measured : %d (%zu unique x %d repeats)\n", frames, clouds.size(),
              repeats);
  std::printf("mean cells out  : %zu\n", total_cells / clouds.size());
  std::printf("\n%-22s %8s %8s %8s %8s %9s\n", "microseconds", "p50", "p90", "p99",
              "p99.9", "max");
  printRow("A) PCIe H2D only", h2d);
  printRow("B) GPU compute only", gpu);
  printRow("C) end to end", e2e);

  const double mb = mean_points * sizeof(PointXYZIRT) / 1e6;
  std::printf("\nH2D effective   : %.1f GB/s at p50 (%.2f MB/frame)\n",
              (mean_points * sizeof(PointXYZIRT)) / (h2d.p50 * 1e-6) / 1e9, mb);
  std::printf("transfer share  : %.0f%% of end-to-end p50\n", 100.0 * h2d.p50 / e2e.p50);
  std::printf("\nvs 100 ms sensor period: GPU-resident p50 = %.2f%%, end-to-end p50 = %.2f%%\n",
              gpu.p50 / 1000.0, e2e.p50 / 1000.0);
  return 0;
}
