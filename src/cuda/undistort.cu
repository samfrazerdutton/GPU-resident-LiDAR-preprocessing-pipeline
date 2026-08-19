#include "gpulidar/cuda/check.hpp"
#include "gpulidar/cuda/kernels.hpp"

namespace gpulidar::cuda {
namespace {

constexpr int kBlock = 256;

__global__ void undistortKernel(const PointXYZIRT* __restrict__ in,
                                PointXYZIRT* __restrict__ out, int n,
                                float vf, float vl, float wu) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const PointXYZIRT p = in[i];
    const float t = p.time;

    // sincosf, not __sincosf -- accuracy first, then measure whether the
    // fast intrinsic buys anything on a bandwidth-bound kernel.
    float s, c;
    sincosf(wu * t, &s, &c);

    PointXYZIRT q = p;
    q.x = c * p.x - s * p.y + vf * t;
    q.y = s * p.x + c * p.y + vl * t;
    q.z = p.z;
    out[i] = q;
  }
}

}  // namespace

void launchUndistort(const PointXYZIRT* d_in, PointXYZIRT* d_out, int n,
                     const EgoTwist& twist, cudaStream_t stream) {
  if (n <= 0) return;
  int grid = (n + kBlock - 1) / kBlock;
  if (grid > 4096) grid = 4096;  // grid-stride loop handles the remainder
  undistortKernel<<<grid, kBlock, 0, stream>>>(d_in, d_out, n,
                                               twist.vf, twist.vl, twist.wu);
  GPULIDAR_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpulidar::cuda
