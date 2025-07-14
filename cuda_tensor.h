#pragma once
#include <cuda_runtime.h>
#include <array>
#include <stdexcept>
#include <iostream>
#include <assert.h>

template <typename T, size_t N>
class CudaTensor {
public:
    CudaTensor() : device_data(nullptr), total_size(0) {
        shape.fill(0);
        strides.fill(0);
    }

    CudaTensor(const std::array<size_t, N>& shape) : shape(shape) {
        compute_strides();
        total_size = 1;
        for (auto s : shape) total_size *= s;

        cudaMalloc(&device_data, total_size * sizeof(T));
        if (!device_data)
            throw std::runtime_error("CUDA malloc failed.");
    }

    ~CudaTensor() {
        if (device_data) cudaFree(device_data);
    }

    // Copy data from host tensor
    void copy_from_host(const T* host_data) {
        cudaMemcpy(device_data, host_data, total_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Copy data to host
    void copy_to_host(T* host_data) const {
        cudaMemcpy(host_data, device_data, total_size * sizeof(T), cudaMemcpyDeviceToHost);
    }

    const std::array<size_t, N>& get_shape() const { 
        return shape; 
    }

    size_t size() const { 
        return total_size; 
    }

    T* data() { 
        return device_data; 
    }
    
    const T* data() const { 
        return device_data; 
    }

    std::array<size_t, N> shape_as_array() const { 
        return shape; 
    }

    // Element-wise addition: this = A + B
    void elementwise_add(const CudaTensor<T, N>& A, const CudaTensor<T, N>& B);

    // Scalar multiplication: this = A * scalar
    void scalar_multiply(const CudaTensor<T, N>& A, T scalar);

    // Matrix multiplication: C = A * B
    void matmul_device(const CudaTensor<T, 2>& A, const CudaTensor<T, 2>& B, CudaTensor<T, 2>& C, bool transpose_A = false, bool transpose_B = false);

private:
    std::array<size_t, N> shape;
    std::array<size_t, N> strides;
    size_t total_size;
    T* device_data;

    void compute_strides() {
        strides[N - 1] = 1;
        for (int i = N - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
};