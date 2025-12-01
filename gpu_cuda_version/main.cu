// gpu_cuda_version/main.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <string>

// Simple xorshift RNG for device
__device__ unsigned int xorshift32(unsigned int& state) {
    // xorshift32
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__global__ void random_walk_kernel(int num_steps,
                                   unsigned int seed,
                                   double* d_r2,
                                   int num_walkers)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_walkers) return;

    unsigned int state = seed ^ (unsigned int)tid;

    int x = 0, y = 0, z = 0;

    for (int step = 0; step < num_steps; ++step) {
        unsigned int rnd = xorshift32(state);
        int dir = rnd % 6;

        switch (dir) {
            case 0: x += 1; break;
            case 1: x -= 1; break;
            case 2: y += 1; break;
            case 3: y -= 1; break;
            case 4: z += 1; break;
            case 5: z -= 1; break;
        }
    }

    double r2 = static_cast<double>(x*x + y*y + z*z);
    d_r2[tid] = r2;
}

double run_random_walk_gpu(int num_walkers, int num_steps, unsigned int seed,
                           float& kernel_ms, float& total_ms)
{
    double* d_r2 = nullptr;
    size_t bytes = static_cast<size_t>(num_walkers) * sizeof(double);

    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    cudaEventRecord(start_total);

    cudaMalloc(&d_r2, bytes);

    int blockSize = 256;
    int gridSize = (num_walkers + blockSize - 1) / blockSize;

    cudaEventRecord(start_kernel);
    random_walk_kernel<<<gridSize, blockSize>>>(num_steps, seed, d_r2, num_walkers);
    cudaEventRecord(stop_kernel);

    cudaEventSynchronize(stop_kernel);

    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
    cudaEventElapsedTime(&total_ms, start_total, stop_total);

    // Copy back and reduce on host
    double* h_r2 = new double[num_walkers];
    cudaMemcpy(h_r2, d_r2, bytes, cudaMemcpyDeviceToHost);

    double sum_r2 = 0.0;
    for (int i = 0; i < num_walkers; ++i) {
        sum_r2 += h_r2[i];
    }
    double msd = sum_r2 / static_cast<double>(num_walkers);

    delete[] h_r2;
    cudaFree(d_r2);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    return msd;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <num_walkers> <num_steps> [seed]\n";
        return 1;
    }

    int num_walkers = std::stoi(argv[1]);
    int num_steps   = std::stoi(argv[2]);
    unsigned int seed = (argc > 3) ? static_cast<unsigned int>(std::stoul(argv[3])) : 42u;

    std::cout << "GPU Random Walk Simulation (CUDA)\n";
    std::cout << "  num_walkers = " << num_walkers << "\n";
    std::cout << "  num_steps   = " << num_steps << "\n";
    std::cout << "  seed        = " << seed << "\n";

    float kernel_ms = 0.0f, total_ms = 0.0f;
    double msd = run_random_walk_gpu(num_walkers, num_steps, seed,
                                     kernel_ms, total_ms);

    std::cout << "Result:\n";
    std::cout << "  Mean squared displacement = " << msd << "\n";
    std::cout << "Timing (on GPU):\n";
    std::cout << "  Kernel time = " << kernel_ms << " ms\n";
    std::cout << "  Total time  = " << total_ms << " ms\n";

    return 0;
}
