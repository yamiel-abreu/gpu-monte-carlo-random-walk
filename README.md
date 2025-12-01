# gpu-monte-carlo-random-walk


# GPU-Accelerated Monte Carlo Random Walk (CUDA + C++)

## Overview

This project implements a 3D random walk Monte Carlo simulation
and compares a CPU-only baseline against a CUDA-accelerated GPU version.

Each "walker" performs `num_steps` random steps in 3D.
We compute the mean squared displacement (MSD) over all walkers.

This is a simple but representative pattern for HPC Monte Carlo:
embarrassingly parallel, RNG-heavy, and bandwidth-light.

## Directory structure

- `cpu_version/` – C++17 single-thread baseline
- `gpu_cuda_version/` – CUDA C++ implementation
- `profiling/` – Nsight / nvprof output and notes
- `analysis/` – Python script to generate performance plots

## Architecture

### CPU version

- Uses `std::mt19937` + `std::uniform_int_distribution<int>` for random steps
- Single-thread loop over walkers, each with its own trajectory
- Measures wall clock time with `std::chrono::high_resolution_clock`

### GPU version

- Assigns **one CUDA thread per walker**
- RNG is a simple per-thread `xorshift32` seeded from a global seed + thread ID
- Each thread performs `num_steps` updates of its `(x, y, z)` position
- Final squared displacement `r^2` is stored in device array
- Host copies back `r^2` array and reduces to MSD

Timing:
- Uses CUDA events to measure:
  - Kernel-only time
  - Total GPU time (including allocation and transfers)

## How to build

### CPU

```bash
mkdir -p build_cpu
cd build_cpu
cmake ../cpu_version
cmake --build .
./cpu_random_walk 1000000 1000
`

### GPU

```bash
mkdir -p build_gpu
cd build_gpu
cmake ../gpu_cuda_version
cmake --build .
./gpu_random_walk 1000000 1000


It can be that you will need that you will need to add specific options on the cmake like:

cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe" ../gpu_cuda_version

cmake --build . --config Release
cmake --build .
./gpu_random_walk 1000000 1000



### Profiling

Example with nsys:

cd build_gpu
nsys profile -o ../profiling/nsight_random_walk ./gpu_random_walk 1000000 1000



Example with nvprof:
cd build_gpu
nvprof --print-gpu-trace ./gpu_random_walk 1000000 1000 \
  2> ../profiling/nvprof_random_walk.txt
  
  
### Performance analysis

1. Run CPU and GPU codes for a range of num_walkers (and optionally num_steps).
2. Save timings into analysis/results.csv.
3. Generate plots:

cd analysis
python plot_speedup.py results.csv

This produces:
runtime_comparison.png
speedup.png
