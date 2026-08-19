#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                    \
  do {                                                                      \
    cudaError_t err__ = (expr);                                             \
    if (err__ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                   cudaGetErrorString(err__), __FILE__, __LINE__);          \
      return 1;                                                             \
    }                                                                       \
  } while (0)

int main() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count == 0) {
    std::fprintf(stderr, "no CUDA devices visible\n");
    return 1;
  }

  for (int i = 0; i < count; ++i) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, i));

    // memoryClockRate / memoryBusWidth were removed from cudaDeviceProp in
    // CUDA 13. cudaDeviceGetAttribute is the supported path on 12.x and 13.x.
    int mem_clock_khz = 0;   // kHz
    int bus_width_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, i));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, i));

    // DDR -> 2 transfers per clock. This is the ceiling every kernel is scored against.
    const double bw_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8.0) / 1e9;

    std::printf("device %d: %s\n", i, p.name);
    std::printf("  compute capability : %d.%d\n", p.major, p.minor);
    std::printf("  SMs                : %d\n", p.multiProcessorCount);
    std::printf("  global memory      : %.2f GiB\n", p.totalGlobalMem / 1073741824.0);
    std::printf("  memory clock       : %.0f MHz\n", mem_clock_khz / 1000.0);
    std::printf("  memory bus width   : %d bits\n", bus_width_bits);
    std::printf("  peak bandwidth     : %.1f GB/s\n", bw_gbs);
    std::printf("  shared mem per blk : %zu B\n", p.sharedMemPerBlock);
    std::printf("  max threads per SM : %d\n", p.maxThreadsPerMultiProcessor);
  }
  return 0;
}
