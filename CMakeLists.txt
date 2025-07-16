cmake_minimum_required(VERSION 3.18)
project(MyCudaProject LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(app test_1.cu mlp.cu activation_utils.cu cuda_tensor.cu)

target_include_directories(app PRIVATE ${CMAKE_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})