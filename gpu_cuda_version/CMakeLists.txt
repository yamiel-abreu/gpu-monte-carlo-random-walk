# gpu_cuda_version/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(gpu_random_walk CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gpu_random_walk main.cu)
set_target_properties(gpu_random_walk PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
