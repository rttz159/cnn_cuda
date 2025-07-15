cmake_minimum_required(VERSION 3.18)
project(MyCudaProject LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(app test.cpp cuda_tensor.cu kernel.cpp mlp.cpp)

target_include_directories(app PRIVATE ${CMAKE_SOURCE_DIR}/include)