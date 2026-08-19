# GPU-Resident LiDAR Preprocessing Pipeline

CUDA preprocessing for automotive LiDAR: motion compensation, ROI cropping, and
voxel downsampling, chained on the GPU with no host round-trip between stages.
Validated against CPU reference implementations and benchmarked for tail latency
on 5,316 real KITTI frames.

    raw sweep -> undistort -> crop box -+-> voxel downsample -> downsampled cloud
                (ego-motion)  (ROI+ego)   |    (hash grid)
                                          +-> pillar scatter -> (P, 32, 9) tensor
                                               (dense grid)

Voxel downsampling and pillar scatter are parallel consumers of the cropped
cloud, not a chain: a PointPillars encoder takes raw points, not voxel
centroids.

## Headline result

Per-frame latency, 443 KITTI frames x 12 repeats = 5,316 samples, mean 116,021
points/frame (3.71 MB):

| microseconds        |   p50 |   p90 |   p99 | p99.9 |    max |
|---------------------|------:|------:|------:|------:|-------:|
| PCIe host-to-device | 601.8 | 660.8 | 725.8 | 934.0 | 5201.0 |
| GPU compute only    | 241.0 | 368.5 | 517.2 |2478.3 | 9546.0 |
| End to end          | 848.1 | 917.9 |1018.2 |1825.6 |12394.3 |

GPU-resident compute is **0.24% of a 100 ms sensor period**. Getting the data
onto the device costs 2.5x more than processing it.

## Findings

**1. Transfer dominates; compute does not.** PCIe host-to-device is 71% of
end-to-end latency at 6.17 GB/s over what appears to be an x8 link. This is the
entire argument for GPU residency: in a production vehicle the sensor driver
DMAs into device memory and this cost disappears, but any design that shuttles
point clouds across PCIe between stages pays it repeatedly.

**2. Data layout beat kernel tuning by 4x.** The voxel hash table originally
stored each cell's eight fields in eight separate arrays (SoA). Every cell touch
cost eight scattered memory transactions, and frame time scaled at 17.5 ns per
occupied cell. Packing one cell into a single 64 B, 32 B-aligned struct made a
cell touch two aligned sectors: 17.5 ns -> 4.35 ns per cell, a 4.0x reduction
against a predicted 8-transactions-to-2. SoA is right for streaming access and
wrong for random access; this table is random access.

**3. Warp aggregation bought tail stability, not throughput.** Replacing
per-thread global atomics with one atomicAdd per warp (ballot mask + popc for
lane slots) in the compaction kernel moved p50 by 1.05x -- but p99 from 155.5 to
125.4 us, p99.9 from 352.3 to 211.8 us, and max from 1514.5 to 595.8 us. The
kernel was bandwidth-bound, not atomic-bound, so there was no median headroom to
recover; reducing 115k atomics to 3.6k removed contention variance instead.
Measured over 20,000 iterations, with the naive variant running first (cooler
GPU), so thermal bias works against the reported result.

**4. Measure the achievable ceiling before claiming an efficiency number.**
The undistort kernel first appeared to hit 72% of the 264 GB/s datasheet peak.
A streaming-copy sweep across three working-set sizes fit
t = 7.2 us + bytes / 236 GB/s -- an asymptotic ceiling of 236 GB/s plus 7.2 us
of fixed launch overhead. Against that model the kernel's predicted floor is
38.6 us and it measures 38.9 us: ~100% of achievable, not 72%. Quoting
theoretical peak understated the kernel and would have sent optimization effort
somewhere with nothing to gain.

**5. A four-byte readback cost 18% of the frame.** Synchronizing each frame to
read the output cell count to the host cost 52.3 us (291.1 us with sync vs
238.8 us amortized without). The pipeline now passes the compaction count to the
voxel stage as a device pointer, so the whole chain enqueues with zero
synchronization.

**6. The latency tail is the platform, not the code.** A pure streaming-copy
kernel -- no branches, no atomics, no math -- has a worse p99.9 than the
undistort kernel that does strictly more work. On a Max-Q part under WSL2's WDDM
scheduler, p99.9 runs roughly 10x p50 and run-to-run p50 variance is ~12%. Any
optimization claim smaller than 12% is noise here. Timing validation belongs on
a dedicated Linux host with locked clocks; these numbers characterize this
platform honestly rather than pretending otherwise.

**7. A published hyperparameter carried a hidden assumption.** The PointPillars
paper's P = 12,000 pillar cap presumes points are pre-filtered to the front
camera frustum. This pipeline keeps the full ROI, where real KITTI frames occupy
min 6,710 / mean 11,632 / max 14,630 pillars -- so the paper's cap silently
dropped pillars on 66.6% of frames. Nothing crashed and every unit test passed;
only an overflow counter over 3,544 real frames surfaced it. The cap is now
30,000 and overflow is observable through the API rather than silent.

## Stages

**Motion compensation.** A spinning LiDAR takes ~100 ms per revolution, during
which the vehicle moves. Each point is re-expressed in the sensor frame as it
was at sweep start, using a constant-twist model from the KITTI oxts stream. On
frame 100 of drive 0009 at 11.11 m/s, this displaces the far end of the sweep by
1.111 m -- the error the stage removes. Elementwise and bandwidth-bound: 38.9 us
at 190.9 GB/s.

**Crop box.** Stream compaction removing out-of-ROI returns and ego-vehicle
self-hits. Output size is unknown until the kernel runs, so threads claim output
slots atomically. Both naive and warp-aggregated variants are kept and
benchmarked (see finding 3). At the default ROI this keeps 99.4% of points --
its job is ego-point removal, not decimation.

**Pillar scatter.** Builds the (P, 32, 9) feature tensor a PointPillars encoder
consumes, with per-point offsets from both the pillar's point centroid and its
geometric centre. Unlike the voxel grid, the pillar grid is dense and bounded
(432 x 496 = 214,272 cells), so a flat index array replaces the hash table --
use the simpler structure when the domain allows it. 230.1 us p50 over 3,544
frames; the count readback costs a further 85.6 us, so the pipeline exposes an
async path that leaves the count on the device.

**Voxel downsample.** Open-addressing hash grid with atomicCAS insertion,
Fibonacci hashing, and linear probing at a load factor of 0.5. The table is
never cleared wholesale: each frame records the slots it claimed and clears only
those, so reset cost scales with occupied cells rather than the 262,144-slot
table. Reduces 116,021 points to 28,636 cells (4.05x) at a 0.2 m leaf.

## Correctness

Every CUDA kernel is diffed against a CPU reference implementation in unit tests
(GoogleTest, ctest):

- Undistort: max GPU/CPU divergence below 1e-4 m, plus closed-form checks (zero
  twist is identity; pure translation shifts by v*t; pure yaw rotates about the
  origin).
- Crop box: compared as a set, since block scheduling makes output order
  undefined. Exercised at sizes that are not multiples of the warp or block size
  (1, 31, 33, 255, 257, 1023, 100001) to catch ballot-loop tail bugs.
- Voxel: compared as a set within 1e-3 m. Bitwise equality is not achievable by
  design -- float atomicAdd accumulates in nondeterministic order, so centroids
  vary in the last bits between runs. A separate test runs five identical frames
  through one grid instance to prove the sparse per-frame reset leaves no state
  behind.

## Limitations

Stated plainly, because a benchmark without its caveats is not a measurement.

- **Platform.** RTX 2060 Max-Q (TU106, SM 7.5, 30 SMs, 6 GiB, 192-bit, 264 GB/s
  theoretical / 236 GB/s measured) under WSL2. Mobile power management and WDDM
  scheduling produce the tail behaviour in finding 6. Desktop 2060 figures are
  higher; do not compare directly.
- **Derived point fields.** KITTI .bin files carry only x/y/z/intensity. Ring is
  derived by elevation-angle binning across the HDL-64E vertical FOV (recovers
  62 of 64 rings; the two empties are edge bins of the linear approximation) and
  per-point time from azimuth sweep fraction. Both are documented
  approximations, not sensor ground truth.
- **Motion model.** Planar constant twist over the sweep -- forward, lateral,
  and yaw rate. No pitch, roll, or intra-sweep acceleration.
- **Coverage.** One drive, one sensor geometry, one leaf size for the headline
  numbers. A leaf sweep from 0.05 m to 1.00 m is included and is what produced
  the per-cell cost model.

## Build

Requires CUDA 12+ (developed on 13.2; note CUDA 13 removed several deprecated
cudaDeviceProp fields), CMake 3.22+, and a C++17 compiler.

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)
    ctest --test-dir build --output-on-failure

Fetch a KITTI raw drive, then reproduce the headline table:

    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0009/2011_09_26_drive_0009_sync.zip
    unzip -q 2011_09_26_drive_0009_sync.zip
    ./build/bench_pipeline path/to/2011_09_26_drive_0009_sync 12

Other tools: device_info (device properties and peak bandwidth), bench_bandwidth
(achievable-bandwidth ceiling), kitti_probe (per-frame stats and undistortion
magnitude), bench_undistort, bench_crop, bench_voxel (accepts a leaf size for
the cost-model sweep).

## Roadmap

- CUDA Graphs to amortize the measured 7.2 us per-launch overhead across the chain
- SoA point layout for the streaming stages, benchmarked against the current AoS
- Validation on a dedicated Linux host with locked clocks, to separate kernel
  behaviour from platform jitter
