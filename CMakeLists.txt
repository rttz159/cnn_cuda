cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(MyCudaProject LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package( OpenCV REQUIRED )

add_executable(app main.cu mlp.cu activation_utils.cu cuda_tensor.cu convblock.cu Datasets.cu image_loader.cpp)

target_include_directories(app PRIVATE ${CMAKE_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})

set_target_properties(app PROPERTIES
    CUDA_RUNTIME_LIBRARY SHARED
)

target_link_libraries(app 
    ${OpenCV_LIBS} 
    CUDA::cudart
)
