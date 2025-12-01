# cpu_version/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(cpu_random_walk CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cpu_random_walk main.cpp)
