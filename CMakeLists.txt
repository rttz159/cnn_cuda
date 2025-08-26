cmake_minimum_required(VERSION 3.10)

project(CNN LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 87)

set(CPP_SOURCES
    main.cpp
    CNN_Cuda.cpp
    Datasets.cpp
)

set(CUDA_SOURCES
    utilities_kernel.cu
    MLP_Cuda.cu
    Conv_Cuda.cu
)

set(ALL_SOURCES ${CPP_SOURCES} ${CUDA_SOURCES})

add_executable(CNN ${ALL_SOURCES})

set_target_properties(CNN PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(CNN PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(CNN
    PUBLIC
        OpenMP::OpenMP_CXX
        MPI::MPI_CXX
        CUDA::cudart
)