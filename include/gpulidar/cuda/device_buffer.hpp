#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "gpulidar/cuda/check.hpp"

namespace gpulidar::cuda {

// Capacity is reserved once, up front. resize() only moves the logical size
// and never allocates -- so "zero mid-frame cudaMalloc" is enforced by the
// type rather than by discipline. Exceeding capacity is a hard error.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t capacity) { reserve(capacity); }

  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& o) noexcept
      : ptr_(o.ptr_), size_(o.size_), capacity_(o.capacity_) {
    o.ptr_ = nullptr; o.size_ = 0; o.capacity_ = 0;
  }

  void reserve(std::size_t capacity) {
    if (capacity <= capacity_) return;
    if (ptr_) GPULIDAR_CUDA_CHECK(cudaFree(ptr_));
    void* p = nullptr;
    GPULIDAR_CUDA_CHECK(cudaMalloc(&p, capacity * sizeof(T)));
    ptr_ = static_cast<T*>(p);
    capacity_ = capacity;
    size_ = 0;
  }

  void resize(std::size_t n) {
    if (n > capacity_) {
      throw std::runtime_error("DeviceBuffer::resize would allocate mid-frame");
    }
    size_ = n;
  }

  void upload(const T* host, std::size_t n, cudaStream_t stream = nullptr) {
    resize(n);
    GPULIDAR_CUDA_CHECK(
        cudaMemcpyAsync(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice, stream));
  }

  void download(T* host, std::size_t n, cudaStream_t stream = nullptr) const {
    if (n > size_) throw std::runtime_error("DeviceBuffer::download past size");
    GPULIDAR_CUDA_CHECK(
        cudaMemcpyAsync(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost, stream));
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  std::size_t size() const { return size_; }
  std::size_t capacity() const { return capacity_; }

 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

// Pinned host memory -- required for genuinely async H2D/D2H later.
template <typename T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(std::size_t capacity) { reserve(capacity); }

  ~PinnedBuffer() {
    if (ptr_) cudaFreeHost(ptr_);
  }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  void reserve(std::size_t capacity) {
    if (capacity <= capacity_) return;
    if (ptr_) GPULIDAR_CUDA_CHECK(cudaFreeHost(ptr_));
    void* p = nullptr;
    GPULIDAR_CUDA_CHECK(cudaHostAlloc(&p, capacity * sizeof(T), cudaHostAllocDefault));
    ptr_ = static_cast<T*>(p);
    capacity_ = capacity;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  std::size_t capacity() const { return capacity_; }

 private:
  T* ptr_ = nullptr;
  std::size_t capacity_ = 0;
};

}  // namespace gpulidar::cuda
