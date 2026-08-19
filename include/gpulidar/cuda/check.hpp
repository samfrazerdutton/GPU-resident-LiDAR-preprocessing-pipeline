#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace gpulidar::cuda {

inline void checkImpl(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) +
                             " at " + file + ":" + std::to_string(line));
  }
}

}  // namespace gpulidar::cuda

#define GPULIDAR_CUDA_CHECK(expr) ::gpulidar::cuda::checkImpl((expr), __FILE__, __LINE__)
