# gpu_cuda_version/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(gpu_random_walk LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(gpu_random_walk main.cu)

set_target_properties(gpu_random_walk PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 61         # GTX 1050 Ti = Pascal, CC 6.1
)
